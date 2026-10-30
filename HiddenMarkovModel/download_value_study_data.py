"""Freeze market inputs for the October 2026 value study; never place orders.

Run with the repository's .venv Python. Re-running uses saved CSVs and retains
the original three-stock cache. Yahoo-adjusted OHLC is a total-return proxy,
not an executable historical price tape. See docs/benchmark_data_notes.md.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data"
BASELINE = ROOT / "data"
START = "2018-01-01"
END = "2026-10-08"
DOWNLOAD_END = "2026-10-09"
FRED_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv?"
    "id=DGS3MO&cosd=2017-12-01&coed=2026-10-08"
)
ISSUER_URL = "https://www.blackrock.com/us/individual/products/239714/ishares-russell-3000-etf"
OBSERVED_VENDOR_ALIASES = {"BK": "BNY", "MMC": "MRSH"}
ALIAS_EVIDENCE = {
    "BK": {"effective_date": "2026-05-21", "source": "https://www.bny.com/corporate/global/en/about-us/newsroom/press-release/bny-announces-planned-change-of-stock-ticker-symbol-to-bny-130465.html"},
    "MMC": {"effective_date": "2026-01-14", "source": "https://www.marsh.com/en/corp/about/news/marsh-mclennan-to-change-nyse-symbol-to-mrsh.html"},
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(path: Path) -> dict:
    frame = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    required = ["Open", "High", "Low", "Close"]
    clean = frame[required].apply(pd.to_numeric, errors="coerce")
    complete = clean.notna().all(axis=1) & clean.gt(0).all(axis=1)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha(path),
        "rows": len(frame),
        "first_observation": frame.index.min().strftime("%Y-%m-%d"),
        "last_observation": frame.index.max().strftime("%Y-%m-%d"),
        "invalid_ohlc_rows": int((~complete).sum()),
        "duplicate_dates": int(frame.index.duplicated().sum()),
    }


def normalize(block: pd.DataFrame) -> pd.DataFrame:
    frame = block.copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame = frame.loc[(frame.index >= START) & (frame.index <= END)]
    fields = ["Open", "High", "Low", "Close"]
    frame = frame.dropna(subset=fields)
    frame = frame.loc[frame[fields].gt(0).all(axis=1), fields]
    if frame.empty:
        raise ValueError("No positive complete OHLC observations")
    if frame.index.duplicated().any():
        raise ValueError("Duplicate vendor dates")
    frame.index.name = "Date"
    return frame.sort_index()


def fetch_group(
    symbols: list[str], *, start: str = START, end: str = DOWNLOAD_END,
    auto_adjust: bool = True,
) -> dict[str, pd.DataFrame]:
    raw = yf.download(
        tickers=symbols, start=start, end=end, interval="1d",
        auto_adjust=auto_adjust, repair=True, actions=True, progress=False,
        threads=False, group_by="ticker", multi_level_index=True, timeout=20,
    )
    result = {}
    if raw is None or raw.empty:
        return result
    for symbol in symbols:
        if isinstance(raw.columns, pd.MultiIndex):
            if symbol not in raw.columns.get_level_values(0):
                continue
            block = raw[symbol].dropna(how="all")
        else:
            block = raw.dropna(how="all")
        if not block.empty and all(f in block for f in ["Open", "High", "Low", "Close"]):
            result[symbol] = block
    return result


def audit_panel(universe: dict[str, str]) -> dict:
    reference = pd.DatetimeIndex(pd.read_csv(BASELINE / "NVDA.csv", parse_dates=["Date"])["Date"])
    out, repairs, moves, bad = {}, {}, {}, {}
    manifest = json.loads((OUT / "data_manifest.json").read_text())
    for ticker in universe:
        path = OUT / "breadth" / f"{ticker}.csv"
        if not path.exists():
            out[ticker] = {"status": "missing"}
            continue
        frame = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
        out[ticker] = {
            "rows": len(frame), "first": str(frame.index.min().date()), "last": str(frame.index.max().date()),
            "missing_sessions": [str(d.date()) for d in reference.difference(frame.index)],
            "extra_sessions": [str(d.date()) for d in frame.index.difference(reference)],
        }
        repaired = manifest["files"][ticker].get("repair_marked_rows")
        if repaired:
            repairs[ticker] = repaired
        invalid = (frame.High + 1e-9 < frame[["Open", "Close"]].max(axis=1)) | (frame.Low - 1e-9 > frame[["Open", "Close"]].min(axis=1))
        if invalid.any():
            bad[ticker] = [str(d.date()) for d in frame.index[invalid]]
        returns = frame.Close.pct_change()
        large = returns.abs() > 0.5
        if large.any():
            moves[ticker] = {str(d.date()): float(returns.loc[d]) for d in returns.index[large]}
    audit = {
        "benchmark_sessions": len(reference), "stock_count": len(out),
        "session_audit": out, "repair_marked_rows": repairs,
        "ohlc_bounds_violations": bad, "close_to_close_moves_over_50pct": moves,
    }
    path = OUT / "panel_audit.json"
    path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def get_cash(manifest: dict) -> None:
    cash_dir = OUT / "cash"
    cash_dir.mkdir(exist_ok=True)
    raw_path = cash_dir / "DGS3MO.csv"
    if not raw_path.exists():
        with urllib.request.urlopen(FRED_URL, timeout=35) as response:
            payload = response.read()
        check = pd.read_csv(io.BytesIO(payload))
        if "DGS3MO" not in check or "observation_date" not in check:
            raise ValueError("Unexpected FRED CSV schema")
        raw_path.write_bytes(payload)
    raw = pd.read_csv(raw_path, parse_dates=["observation_date"])
    raw["DGS3MO"] = pd.to_numeric(raw["DGS3MO"], errors="coerce")
    normalized = raw.rename(columns={"observation_date": "Date", "DGS3MO": "annual_rate_decimal"})
    normalized["annual_rate_decimal"] = normalized["annual_rate_decimal"] / 100.0
    normalized = normalized.dropna(subset=["annual_rate_decimal"])
    rates_path = OUT / "cash_rates.csv"
    normalized.to_csv(rates_path, index=False, date_format="%Y-%m-%d")
    manifest["cash"] = {
        "status": "available",
        "source": "Board of Governors H.15, via Federal Reserve Bank of St. Louis FRED",
        "series": "DGS3MO",
        "citation": "https://fred.stlouisfed.org/series/DGS3MO",
        "download_url": FRED_URL,
        "raw_path": str(raw_path.relative_to(ROOT)),
        "raw_sha256": sha(raw_path),
        "normalized_path": str(rates_path.relative_to(ROOT)),
        "normalized_sha256": sha(rates_path),
        "normalized_lag_status": "Unlagged observations. Study caller must lag two trading sessions before using annual_rate_decimal.",
        "first_observation": raw.observation_date.min().strftime("%Y-%m-%d"),
        "last_observation": raw.observation_date.max().strftime("%Y-%m-%d"),
        "lag_rule": "Forward-fill published observation then lag two IWV trading sessions",
        "units": "Annual percent investment-basis constant maturity yield, not a realized cash fund return",
        "vintage_caveat": "Retrieved historical series may include revisions; two-session lag is conservative but is not an archival real-time vintage.",
    }
    benchmark = OUT / "IWV.csv"
    if benchmark.exists():
        sessions = pd.DatetimeIndex(pd.read_csv(benchmark, parse_dates=["Date"])["Date"])
        known = raw.dropna(subset=["DGS3MO"]).set_index("observation_date")
        known["SourceObservationDate"] = known.index
        daily = known.reindex(sessions.union(known.index)).sort_index().ffill()
        daily = daily.reindex(sessions).shift(2)
        daily.columns = ["AnnualYieldPercent", "SourceObservationDate"]
        daily.index.name = "Date"
        daily_path = cash_dir / "cash_yield_lag2sessions.csv"
        daily.to_csv(daily_path, date_format="%Y-%m-%d")
        manifest["cash"]["lagged_yield_path"] = str(daily_path.relative_to(ROOT))
        manifest["cash"]["lagged_yield_sha256"] = sha(daily_path)
        manifest["cash"]["initial_missing_sessions"] = int(daily.AnnualYieldPercent.isna().sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()
    for directory in [OUT, OUT / "benchmark", OUT / "breadth", OUT / "vendor_raw", OUT / "yfinance_cache"]:
        directory.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(OUT / "yfinance_cache"))
    config = json.loads((ROOT / "future_stock_universe.json").read_text(encoding="utf-8"))
    manifest_path = OUT / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update({
        "requested_start": START, "requested_end_inclusive": END,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "yfinance_version": yf.__version__,
        "yfinance_settings": {"auto_adjust": True, "repair": True, "actions": True, "threads": False},
        "universe": config["stocks"], "vendor_aliases": config["vendor_aliases"],
        "additional_vendor_aliases": OBSERVED_VENDOR_ALIASES,
        "additional_alias_evidence": ALIAS_EVIDENCE,
        "universe_caveat": "Today's user-selected list, not historical Russell 3000 membership; selection and survivor bias apply.",
        "benchmark": {"ticker": "IWV", "name": "iShares Russell 3000 ETF", "type": "Investable Russell 3000 ETF proxy, not the index", "issuer_url": ISSUER_URL, "expense_ratio_current": 0.002, "fee_policy": "Expenses already affect fund prices/returns; never subtract the expense ratio again"},
    })
    files = manifest.setdefault("files", {})
    for ticker, details in files.items():
        path = ROOT / details["path"]
        if path.exists() and sha(path) != details["sha256"]:
            raise ValueError(f"Previously frozen file hash mismatch for {ticker}")
    before = {t: sha(BASELINE / f"{t}.csv") for t in ["NVDA", "GOOG", "INTC"]}
    manifest["baseline_cache_sha256_before"] = before
    requested = {"IWV": OUT / "IWV.csv"}
    if not args.benchmark_only:
        for ticker in config["stocks"]:
            requested[ticker] = OUT / "breadth" / f"{ticker}.csv"
            if ticker in ["NVDA", "INTC"]:
                path = requested[ticker]
                if not path.exists():
                    shutil.copyfile(BASELINE / f"{ticker}.csv", path)
                if sha(path) != before[ticker]:
                    raise ValueError(f"Frozen baseline copy mismatch for {ticker}")
                files[ticker] = {**describe(path), "source": "Existing research cache copied byte-for-byte; never redownloaded", "source_cache_path": str((BASELINE / f"{ticker}.csv").relative_to(ROOT))}
    missing = [ticker for ticker, path in requested.items() if not path.exists()]
    errors = []
    aliases = {**config["vendor_aliases"], **OBSERVED_VENDOR_ALIASES}
    for offset in range(0, len(missing), 15):
        chunk = missing[offset:offset + 15]
        vendors = [aliases.get(t, t) for t in chunk]
        print(f"Fetching {offset + 1}-{offset + len(chunk)} of {len(missing)}: {', '.join(vendors)}", flush=True)
        try:
            fetched = fetch_group(vendors)
        except Exception as exc:
            fetched = {}
            errors.append({"symbols": chunk, "attempt": 1, "error": repr(exc)})
        failed = [v for v in vendors if v not in fetched]
        if failed:
            time.sleep(1)
            try:
                fetched.update(fetch_group(failed))
            except Exception as exc:
                errors.append({"symbols": failed, "attempt": 2, "error": repr(exc)})
        for ticker, vendor in zip(chunk, vendors):
            if vendor not in fetched:
                errors.append({"symbol": ticker, "vendor_symbol": vendor, "error": "No data after two bounded attempts"})
                continue
            try:
                block = fetched[vendor]
                frame = normalize(block)
                frame.to_csv(requested[ticker], date_format="%Y-%m-%d", float_format="%.15g")
                raw_path = OUT / "vendor_raw" / f"{ticker}.csv"
                block.index.name = "Date"
                block.to_csv(raw_path, date_format="%Y-%m-%d")
                files[ticker] = {
                    **describe(requested[ticker]), "source": "Yahoo Finance via yfinance",
                    "vendor_symbol": vendor, "vendor_raw_path": str(raw_path.relative_to(ROOT)),
                    "vendor_raw_sha256": sha(raw_path),
                    "repair_marked_rows": int(block["Repaired?"].fillna(False).astype(bool).sum()) if "Repaired?" in block else None,
                    "source_url": f"https://finance.yahoo.com/quote/{vendor}/history/",
                }
            except Exception as exc:
                errors.append({"symbol": ticker, "error": repr(exc)})
        manifest["files"] = files
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    try:
        get_cash(manifest)
    except Exception as exc:
        manifest["cash"] = {"status": "unavailable", "error": repr(exc), "download_url": FRED_URL}
        errors.append({"source": "FRED DGS3MO", "error": repr(exc)})
    manifest["baseline_cache_sha256_after"] = {t: sha(BASELINE / f"{t}.csv") for t in before}
    if manifest["baseline_cache_sha256_after"] != before:
        raise RuntimeError("Baseline cache changed during retrieval")
    manifest["missing_symbols"] = [ticker for ticker, path in requested.items() if not path.exists()]
    manifest["download_errors"] = errors
    manifest["requested_symbols_this_run"] = list(requested)
    manifest["available_symbols_this_run"] = [t for t, path in requested.items() if path.exists()]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not args.benchmark_only:
        manifest["panel_audit"] = audit_panel(config["stocks"])
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"available": len(manifest["available_symbols_this_run"]), "missing": manifest["missing_symbols"], "cash": manifest["cash"]["status"]}, indent=2), flush=True)
    if manifest["missing_symbols"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
