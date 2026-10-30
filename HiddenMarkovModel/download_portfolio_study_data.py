"""Freeze an extended, audited panel without touching previous study caches."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
import yfinance as yf

from DecisionValueStudy import save_json, sha256
from download_value_study_data import fetch_group, OBSERVED_VENDOR_ALIASES, ALIAS_EVIDENCE

ROOT = Path(__file__).resolve().parent


def market_frames(raw: pd.DataFrame, end: str) -> tuple[pd.DataFrame, pd.Series]:
    """Yahoo OHLC is split-normalized even with auto_adjust=False."""
    raw = raw.copy()
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    if raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
        raise ValueError("Vendor dates must be unique and increasing.")
    required = ["Open", "High", "Low", "Close", "Adj Close", "Stock Splits"]
    if not set(required).issubset(raw.columns):
        raise ValueError(f"Vendor response lacks {set(required).difference(raw.columns)}")
    raw = raw.dropna(subset=["Open", "High", "Low", "Close", "Adj Close"])
    if raw.empty or not np.isfinite(raw[required].to_numpy()).all():
        raise ValueError("Empty/nonfinite vendor observations.")
    if (raw[["Open", "High", "Low", "Close", "Adj Close"]] <= 0).any().any() or (raw["Stock Splits"] < 0).any():
        raise ValueError("Invalid vendor prices or split ratios.")
    splits = raw["Stock Splits"].replace(0.0, 1.0)
    # shortcut: quoted prices rely on vendor action accuracy; reconcile issuer events before execution-grade use.
    # Exclude today's split: the split-day opening quote already uses new shares.
    future_splits = splits.iloc[::-1].cumprod().iloc[::-1] / splits
    quoted_open = (raw["Open"] * future_splits).rename("QuotedOpen")
    adjustment = raw["Adj Close"] / raw["Close"]
    adjusted = raw[["Open", "High", "Low", "Close"]].mul(adjustment, axis=0)
    adjusted.index.name = quoted_open.index.name = "Date"
    return adjusted.loc[:end], quoted_open.loc[:end]


def download(plan_path: Path) -> None:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    directory = ROOT / plan["data_dir"]
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("prices", "quoted", "vendor_raw", "rates", "yfinance_cache"):
        (directory / name).mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(directory / "yfinance_cache"))
    universe = json.loads((ROOT / plan["universe_file"]).read_text(encoding="utf-8"))
    aliases = {**universe["vendor_aliases"], **OBSERVED_VENDOR_ALIASES}
    tickers = [*universe["stocks"], plan["benchmark_ticker"]]
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "files": {},
        "plan_sha256": sha256(plan_path), "universe_sha256": sha256(ROOT / plan["universe_file"]),
        "yfinance_version": yf.__version__, "vendor_aliases": aliases,
        "alias_evidence": ALIAS_EVIDENCE,
        "settings": {"start": plan["download_start"], "end_exclusive": plan["download_end_exclusive"],
                     "auto_adjust": False, "repair": True, "actions": True},
    }
    if manifest["plan_sha256"] != sha256(plan_path) or manifest["universe_sha256"] != sha256(ROOT / plan["universe_file"]):
        raise ValueError("Data manifest belongs to a different plan/universe; use a new data directory.")
    for relative, digest in manifest["files"].items():
        if sha256(directory / relative) != digest:
            raise ValueError(f"Frozen data changed: {relative}")
    missing = [t for t in tickers if f"prices/{t}.csv" not in manifest["files"]]
    for offset in range(0, len(missing), 15):
        chunk = missing[offset:offset + 15]
        vendors = [aliases.get(t, t) for t in chunk]
        print(f"Fetching {offset + 1}-{offset + len(chunk)} / {len(missing)}: {vendors}", flush=True)
        fetched = fetch_group(vendors, start=plan["download_start"],
                              end=plan["download_end_exclusive"], auto_adjust=False)
        failed = [v for v in vendors if v not in fetched]
        if failed:
            fetched.update(fetch_group(failed, start=plan["download_start"],
                                       end=plan["download_end_exclusive"], auto_adjust=False))
        for ticker, vendor in zip(chunk, vendors):
            if vendor not in fetched:
                continue
            raw = fetched[vendor]
            prices, quoted = market_frames(raw, plan["end"])
            raw.index.name = "Date"
            for name, frame in (("vendor_raw", raw), ("prices", prices), ("quoted", quoted)):
                path = directory / name / f"{ticker}.csv"
                frame.to_csv(path, date_format="%Y-%m-%d")
                manifest["files"][f"{name}/{ticker}.csv"] = sha256(path)
        save_json(manifest_path, manifest)
    unavailable = [t for t in tickers if f"prices/{t}.csv" not in manifest["files"]]
    if unavailable:
        save_json(directory / "audit.json", {"status": "missing_symbols", "missing": unavailable})
        raise ValueError(f"No complete panel; unavailable symbols: {unavailable}")
    for series in ("DGS10", "DGS3MO"):
        raw_path = directory / "rates" / f"{series}_raw.csv"
        if not raw_path.exists():
            url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
                   f"&cosd={plan['download_start']}&coed={plan['end']}")
            with urllib.request.urlopen(url, timeout=45) as response:
                payload = response.read()
            check = pd.read_csv(io.BytesIO(payload))
            if not {"observation_date", series}.issubset(check.columns):
                raise ValueError(f"Unexpected {series} schema")
            raw_path.write_bytes(payload)
        raw = pd.read_csv(raw_path)
        normalized = pd.DataFrame({"Date": raw["observation_date"],
                                   "annual_rate_decimal": pd.to_numeric(raw[series], errors="coerce") / 100.0}).dropna()
        path = directory / "rates" / f"{series}.csv"
        normalized.to_csv(path, index=False)
        for source in (raw_path, path):
            manifest["files"][source.relative_to(directory).as_posix()] = sha256(source)
        save_json(manifest_path, manifest)
    reference = pd.read_csv(directory / "prices" / f"{plan['benchmark_ticker']}.csv", index_col="Date", parse_dates=True)
    errors, bounds = {}, {}
    for ticker in tickers:
        prices = pd.read_csv(directory / "prices" / f"{ticker}.csv", index_col="Date", parse_dates=True)
        missing_dates = reference.index.difference(prices.index)
        invalid = (prices["High"] + 1e-8 < prices[["Open", "Close"]].max(axis=1)) | (prices["Low"] - 1e-8 > prices[["Open", "Close"]].min(axis=1))
        if len(missing_dates) or invalid.any():
            errors[ticker] = {"missing_dates": [str(d.date()) for d in missing_dates],
                              "invalid_ohlc_dates": [str(d.date()) for d in prices.index[invalid]]}
        bounds[ticker] = {"first": str(prices.index[0].date()), "last": str(prices.index[-1].date()), "rows": len(prices)}
    prior = ROOT / "output/value_study_20261023/data/IWV.csv"
    if prior.exists():
        prior_dates = pd.DatetimeIndex(pd.read_csv(prior, parse_dates=["Date"])["Date"])
        lost = prior_dates.difference(reference.index)
        if len(lost):
            errors["benchmark_calendar"] = [str(d.date()) for d in lost]
    audit = {"status": "complete" if not errors else "incomplete", "stock_count": len(universe["stocks"]),
             "sessions": len(reference), "bounds": bounds, "errors": errors}
    save_json(directory / "audit.json", audit)
    manifest["audit_sha256"] = sha256(directory / "audit.json")
    save_json(manifest_path, manifest)
    if errors:
        raise ValueError("Incomplete data panel; see audit.json. No symbols or sessions silently removed.")
    print(json.dumps({"data": str(directory), "status": audit["status"], "sessions": len(reference)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="portfolio_study_plan.json")
    download(ROOT / parser.parse_args().plan)
