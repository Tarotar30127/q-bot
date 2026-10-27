"""Generate the readable study report and flat audit tables from frozen results.

This command reads completed results only; it does not download data, fit a
model, select a strategy, or run another evaluation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
from typing import Any

ROOT = Path(__file__).resolve().parent
LABELS = {
    "hmm": "HMM selector",
    "ungated_momentum": "Momentum without HMM",
    "confidence_sized_ungated": "Momentum with posterior sizing, no gate",
    "passive_stocks": "Passive stock basket",
    "iwv": "IWV Russell 3000 ETF proxy",
    "train_exposure_stocks": "Passive stocks at training allocation",
    "train_exposure_iwv": "IWV at training allocation",
    "same_gate_iwv": "IWV using HMM target exposure",
    "cash": "Cash yield proxy",
    "hmm_zero_cash_yield": "HMM, zero cash interest",
    "ungated_zero_cash_yield": "Momentum, zero cash interest",
    "momentum_only_hmm": "HMM restricted to momentum experts",
}
SPLITS = {
    "validation_2023_2025": "2023–2025 research validation",
    "evaluation_2026": "2026 through October 8 research check",
}


def percent(value: Any, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.2%}" if signed else f"{float(value):.2%}"


def pp(value: float) -> str:
    return f"{100 * value:+.2f} pp"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in row) + " |" for row in rows]
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def cases(report: dict) -> list[tuple[str, str, dict]]:
    result = [("reference", key, value) for key, value in report["reference"].items()]
    for row in report["rolling"]:
        result.append(("rolling", str(row["year"]), row["evaluation"]))
    breadth = report["breadth"]
    if "splits" in breadth:
        result += [("breadth", key, value) for key, value in breadth["splits"].items()]
    return result


def title(scope: str, split: str) -> str:
    if scope == "rolling":
        return f"Rolling {split}"
    short = "2023–2025" if split == "validation_2023_2025" else "2026 through Oct 8"
    return f"{'Reference' if scope == 'reference' else '100 stocks'} {short}"


def trading_fraction(metrics: dict, initial: float) -> float:
    return float(metrics["position_pnl"]) / initial


def metric_table(record: dict, initial: float, names: list[str] | None = None) -> str:
    metrics = record["metrics"]
    names = names or [name for name in LABELS if name in metrics]
    rows = []
    for name in names:
        m = metrics[name]
        rows.append([LABELS[name], percent(m["net_cumulative_return"], True),
                     pp(trading_fraction(m, initial)), pp(m["cash_interest_fraction_initial"]),
                     percent(m["max_drawdown"]), percent(m["average_gross_exposure"]), m["completed_positions"]])
    return table(["Strategy", "Net return", "Trade contribution", "Cash contribution", "Max drawdown", "Avg capital invested", "Completed positions"], rows)


def main(inputdir: Path, destination: Path) -> None:
    source = inputdir / "research_report.json"
    report = json.loads(source.read_text(encoding="utf-8"))
    registration_link = Path(os.path.relpath(ROOT / "paper_evaluation_registration.json", destination.parent)).as_posix()
    notes_link = Path(os.path.relpath(ROOT / "docs" / "benchmark_data_notes.md", destination.parent)).as_posix()
    plan = report["plan"]
    initial = float(plan["configuration"]["initial_equity"])
    all_cases = cases(report)
    out = inputdir / "report_tables"
    out.mkdir(exist_ok=True)
    comparisons, offsets, offset_summaries, bootstraps, decisions = [], [], [], [], []
    for scope, split, record in all_cases:
        for name, m in record["metrics"].items():
            if abs(m["net_cumulative_return"] - trading_fraction(m, initial) - m["cash_interest_fraction_initial"]) > 1e-8:
                raise ValueError(f"Trading/cash attribution does not reconcile for {scope}/{split}/{name}")
            comparisons.append({"scope": scope, "split": split, "strategy": name,
                                "trade_contribution_initial": trading_fraction(m, initial), **m})
        for row in record["offsets"]:
            item = {"scope": scope, "split": split, "offset": row["offset"],
                    "hmm_minus_ungated_return": row["hmm_minus_ungated_return"]}
            for name in ["hmm", "ungated_momentum", "same_gate_iwv"]:
                for key in ["net_cumulative_return", "max_drawdown", "average_gross_exposure", "completed_positions", "position_pnl", "cash_interest_fraction_initial"]:
                    item[name + "_" + key] = row[name][key]
            offsets.append(item)
        if record["offsets"]:
            entries = record["offsets"]
            returns = [row["hmm"]["net_cumulative_return"] for row in entries]
            differences = [row["hmm_minus_ungated_return"] for row in entries]
            count = len(entries)
            offset_summaries.append({
                "scope": scope, "split": split, "offsets": count,
                "hmm_return_min": min(returns), "hmm_return_median": statistics.median(returns), "hmm_return_max": max(returns),
                "hmm_minus_ungated_min": min(differences), "hmm_minus_ungated_median": statistics.median(differences), "hmm_minus_ungated_max": max(differences),
                "positive_total_return_fraction": sum(value > 0 for value in returns) / count,
                "positive_trading_pnl_fraction": sum(row["hmm"]["position_pnl"] > 0 for row in entries) / count,
                "hmm_beats_ungated_fraction": sum(value > 0 for value in differences) / count,
                "cash_only_offsets": sum(row["hmm"]["completed_positions"] == 0 for row in entries),
            })
        for control, b in record["paired_bootstrap"].items():
            control_name = control.removeprefix("hmm_minus_")
            expected = record["metrics"]["hmm"]["net_cumulative_return"] - record["metrics"][control_name]["net_cumulative_return"]
            if abs(expected - b["observed_cumulative_excess"]) > 1e-8:
                raise ValueError(f"Paired bootstrap observation mismatch for {scope}/{split}/{control}")
            bootstraps.append({"scope": scope, "split": split, "comparison": control, **b})
        for d in record["decision_audit"]:
            decisions.append({"scope": scope, "split": split, **d})
    for name, rows in [("comparison_metrics", comparisons), ("offset_details", offsets),
                       ("offset_summary", offset_summaries), ("bootstrap_summary", bootstraps),
                       ("decision_audit_summary", decisions)]:
        write_csv(out / f"{name}.csv", rows)

    rolling = report["rolling"]
    n = len(rolling)
    positive_trading = sum(r["evaluation"]["metrics"]["hmm"]["position_pnl"] > 0 for r in rolling)
    positive_total = sum(r["evaluation"]["metrics"]["hmm"]["net_cumulative_return"] > 0 for r in rolling)
    cash_years = sum(r["evaluation"]["metrics"]["hmm"]["completed_positions"] == 0 for r in rolling)
    count_positions = sum(r["evaluation"]["metrics"]["hmm"]["completed_positions"] for r in rolling)
    beat_ungated = sum(r["evaluation"]["metrics"]["hmm"]["net_cumulative_return"] > r["evaluation"]["metrics"]["ungated_momentum"]["net_cumulative_return"] for r in rolling)
    beat_iwv = sum(r["evaluation"]["metrics"]["hmm"]["net_cumulative_return"] > r["evaluation"]["metrics"]["iwv"]["net_cumulative_return"] for r in rolling)
    rolling_offsets = [r for r in offsets if r["scope"] == "rolling"]
    beat_offset_fraction = sum(r["hmm_minus_ungated_return"] > 0 for r in rolling_offsets) / len(rolling_offsets)
    positive_total_offsets = sum(r["hmm_net_cumulative_return"] > 0 for r in rolling_offsets) / len(rolling_offsets)
    positive_trading_offsets = sum(r["hmm_position_pnl"] > 0 for r in rolling_offsets) / len(rolling_offsets)
    lower_positive = sum(r["evaluation"]["paired_bootstrap"]["hmm_minus_ungated_momentum"]["cumulative_excess_ci"][0] > 0 for r in rolling)
    dd_protection = sum(r["evaluation"]["metrics"]["hmm"]["max_drawdown"] > r["evaluation"]["metrics"]["ungated_momentum"]["max_drawdown"] for r in rolling)
    thresholds = plan["success_evidence_reference"]
    criteria = [
        {"criterion": "Positive total-return evaluation years, including interest", "observed": positive_total / n, "minimum": thresholds["minimum_positive_evaluation_year_fraction"]},
        {"criterion": "HMM beats ungated evaluation years", "observed": beat_ungated / n, "minimum": thresholds["minimum_positive_hmm_minus_momentum_year_fraction"]},
        {"criterion": "Positive total-return rolling offsets, including interest", "observed": positive_total_offsets, "minimum": thresholds["minimum_positive_offset_fraction"]},
        {"criterion": "Completed rolling HMM positions", "observed": count_positions, "minimum": thresholds["minimum_completed_positions_total"]},
    ]
    for item in criteria:
        item["passes"] = item["observed"] >= item["minimum"]
    write_csv(out / "rolling_criteria.csv", criteria)
    reference = report["reference"]
    current = reference["evaluation_2026"]
    c = current["metrics"]
    failed = sum(not item["passes"] for item in criteria)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    lines = ["# Do the HMM's trade decisions add value?", "", "Study declared October 23, 2026; prices stop October 8, 2026.", ""]
    if failed or lower_positive < n:
        lines += [
            "**The study does not establish a repeatable HMM trading advantage.** "
            f"The rolling evaluations produced positive trading P&L in {positive_trading}/{n} years and {count_positions} completed positions. "
            f"The HMM beat ungated momentum in {beat_ungated}/{n} years and {percent(beat_offset_fraction)} of the fixed rebalance schedules. "
            f"{failed} of the four declared activity and comparative evidence references were missed. "
            "The evidence supports keeping this as a research control and testing the next strategy hypothesis under a new protocol.", ""]
    else:
        lines += ["The historical study passes the declared comparative evidence references. These retrospective results justify a prospective paper evaluation; they do not establish future profitability.", ""]
    lines += [
        "## Frozen model and comparison", "",
        "The reference is the existing 20-session HMM selector with a 20-session momentum signal, 50 bps entry threshold, "
        "and 100-session trend check. A dominant-state probability of at least 0.55 is required; posterior confidence sizes "
        "the position. Its state/expert policy and HMM are fitted only before the evaluation window. "
        "Training calibration retains zero cash interest so positive interest cannot qualify an unprofitable expert.", "",
        "Momentum without HMM removes both the eligibility gate and confidence sizing. The posterior-sized ungated control "
        "keeps confidence sizing and removes the state/minimum-confidence gates. The same-gate IWV control applies the "
        "HMM's available target exposure to IWV on the same schedule. Passive allocation controls use only average training "
        "exposure; subsequent weights drift, so they do not match realized exposure or volatility exactly.", "",
        f"Each split starts with ${initial:,.0f}; entry and exit use adjusted opens. Slippage is "
        f"{plan['configuration']['slippage_bps_per_side']:g} bps per side with "
        f"{plan['configuration']['commission_bps_per_side']:g} bps commission; turnover and terminal liquidation incur costs. "
        "Cash uses a Treasury-yield approximation available with a two-session lag. Each annual book liquidates at its "
        "last open and restarts next year, omitting cross-year overnight gaps.", "",
        "All returns below are cumulative net returns for the indicated period. Trade and cash contributions are "
        "percentage points of starting equity and reconcile to net return. Trade contribution includes execution and "
        "borrow costs and excludes interest. Completed positions count contiguous holding episodes in a stock, "
        "including additions/trims; they are not order counts or independent statistical observations.", "",
        "## Reference model versus Russell 3000 proxy", "",
        "**IWV is an investable Russell 3000 ETF proxy, not the official Russell 3000 total-return index.** "
        "Yahoo-adjusted OHLC incorporates dividend/split adjustments; dividends are not added again. Fund expenses "
        "already affect returns and are not charged again. All passive books pay the same modeled entry/exit costs. "
        "[The issuer identifies the Russell 3000 Index as IWV's benchmark](https://www.blackrock.com/us/individual/products/239714/ishares-russell-3000-etf).", "",
    ]
    main_names = ["hmm", "ungated_momentum", "confidence_sized_ungated", "passive_stocks", "iwv", "cash", "train_exposure_stocks", "train_exposure_iwv", "same_gate_iwv"]
    for split, record in reference.items():
        lines += ["### " + SPLITS[split], "", metric_table(record, initial, main_names), ""]
    lines += [
        f"For 2026 the reference HMM earned {percent(c['hmm']['net_cumulative_return'], True)}, versus "
        f"{percent(c['iwv']['net_cumulative_return'], True)} for IWV: a {pp(c['hmm']['net_cumulative_return'] - c['iwv']['net_cumulative_return'])} gap. "
        f"Cash alone earned {percent(c['cash']['net_cumulative_return'], True)}. The HMM's trade contribution was "
        f"{pp(trading_fraction(c['hmm'], initial))} and interest contributed {pp(c['hmm']['cash_interest_fraction_initial'])}. "
        f"It held a position on {percent(c['hmm']['portfolio_active_day_fraction'])} of holding intervals, with "
        f"{percent(c['hmm']['average_gross_exposure'])} average capital invested and "
        f"{c['hmm']['completed_positions']} completed positions across {c['hmm']['participating_stocks']} stock(s).", "",
        f"A passive IWV allocation fixed from training earned {percent(c['train_exposure_iwv']['net_cumulative_return'], True)} "
        f"in 2026 with a {percent(c['train_exposure_iwv']['max_drawdown'])} drawdown, versus "
        f"{percent(c['hmm']['max_drawdown'])} for the HMM. That similar drawdown and higher return weaken "
        "the case that the HMM adds useful risk control beyond choosing a smaller passive allocation. "
        "The allocation was fixed before scoring and is not an exact realized-risk match.", "",
        "Cash interest changes portfolio equity and later order sizes, so an attribution contribution is different "
        "from rerunning the entire strategy without interest. The zero-interest reruns reproduce the earlier report:", "",
        table(["Period", "HMM zero-interest return", "Ungated zero-interest return"], [
            ["2023–2025", percent(reference["validation_2023_2025"]["metrics"]["hmm_zero_cash_yield"]["net_cumulative_return"], True), percent(reference["validation_2023_2025"]["metrics"]["ungated_zero_cash_yield"]["net_cumulative_return"], True)],
            ["2026 through October 8", percent(c["hmm_zero_cash_yield"]["net_cumulative_return"], True), percent(c["ungated_zero_cash_yield"]["net_cumulative_return"], True)],
        ]), "",
        "## Chronological evaluations: does trading profitability repeat?", "",
        "For evaluation year Y, training uses Y−4 through Y−2, the following year records validation without selecting "
        "a new configuration, and Y is evaluated. Training windows overlap and data were previously studied. "
        "These are chronological robustness checks on reused history, not pristine independent holdouts. "
        "The 2026 row ends October 8 and is not a full calendar year.", "",
    ]
    yearly_rows = []
    for row in rolling:
        m = row["evaluation"]["metrics"]
        yearly_rows.append([row["year"], percent(m["hmm"]["net_cumulative_return"], True), pp(trading_fraction(m["hmm"], initial)),
                            m["hmm"]["completed_positions"], percent(m["ungated_momentum"]["net_cumulative_return"], True),
                            percent(m["iwv"]["net_cumulative_return"], True), percent(m["hmm"]["max_drawdown"]),
                            ", ".join(f"{state}: {expert}" for state, expert in row["policy"]["policies"].items())])
    lines += [table(["Year", "HMM net", "HMM trade contribution", "HMM positions", "Ungated net", "IWV net", "HMM drawdown", "Fitted state policies"], yearly_rows), "",
              f"Total portfolio return was positive in {positive_total}/{n} years, but trading P&L was positive in "
              f"{positive_trading}/{n}; {cash_years}/{n} years held only cash. The HMM beat IWV in {beat_iwv}/{n} years. "
              f"Drawdown was smaller than ungated momentum in {dd_protection}/{n} years, with substantially lower exposure. "
              "Cash-only positive years demonstrate interest income, not successful trade selection.", "",
              "## Rebalance date sensitivity", "",
              "All 20 starting offsets were evaluated using the same previously fitted offset-zero policy. No best "
              "offset is selected. These schedules share most observations and are correlated.", ""]
    lines += [table(["Case", "HMM min / median / max net return", "Positive trade P&L", "HMM beats ungated", "Cash-only offsets"], [
        [title(r["scope"], r["split"]), " / ".join(percent(r[key], True) for key in ["hmm_return_min", "hmm_return_median", "hmm_return_max"]),
         percent(r["positive_trading_pnl_fraction"]), percent(r["hmm_beats_ungated_fraction"]), f"{r['cash_only_offsets']}/{r['offsets']}"]
        for r in offset_summaries]), "",
        f"Across the {len(rolling_offsets)} rolling schedules, positive trading P&L occurred in "
        f"{percent(positive_trading_offsets)} and HMM return exceeded same-offset ungated momentum in "
        f"{percent(beat_offset_fraction)}. Return-difference min/median/max are retained in `report_tables/offset_summary.csv`; "
        "positive total returns from cash are not counted as positive trading P&L.", "",
        "## Uncertainty around comparative returns", "",
        "Paired daily returns are resampled jointly in 20-session circular blocks, 2,000 times with seed 42. "
        "The table shows the observed cumulative return difference and descriptive 95% interval, in percentage points. "
        "An interval spanning zero does not establish a stable advantage. These intervals do not adjust for previous "
        "research, multiple comparisons, fitting uncertainty, or the choice of calendar; the bootstrap fraction above "
        "zero is not a probability of future profitability.", ""]
    ci_rows = []
    for scope, split, record in all_cases:
        cells = []
        for key in ["hmm_minus_ungated_momentum", "hmm_minus_confidence_sized_ungated", "hmm_minus_iwv"]:
            b = record["paired_bootstrap"][key]
            cells.append(f"{pp(b['observed_cumulative_excess'])} [{pp(b['cumulative_excess_ci'][0])}, {pp(b['cumulative_excess_ci'][1])}]")
        ci_rows.append([title(scope, split), *cells])
    lines += [table(["Case", "HMM − ungated", "HMM − posterior-sized ungated", "HMM − IWV"], ci_rows), "",
              f"Only {lower_positive}/{n} rolling evaluation intervals have an HMM-minus-ungated lower bound above zero. "
              "Exposure-control and cash-control intervals are also retained in the bootstrap CSV.", "",
              "## Allowed and blocked momentum decisions", "",
              "For every momentum-eligible stock on a scheduled rebalance, the audit records whether the HMM allowed "
              "or blocked it, then measures that stock's next same-schedule holding return, less hypothetical entry "
              "and exit costs. These are equally weighted stock-interval counterfactuals. They do not reproduce "
              "actual portfolio sizing, cash compounding, or turnover from carrying an existing position; they "
              "must not be read as causal proof or independent trade trials.", ""]
    lines += [table(["Case", "Decision", "Stock intervals", "Distinct dates", "Mean hypothetical net holding return", "Profitable intervals"], [
        [title(r["scope"], r["split"]), "Allowed" if r["hmm_allowed"] else "Blocked", r["stock_intervals"], r["distinct_decision_dates"],
         percent(r["mean_hypothetical_net_return"], True), percent(r["profitable_fraction"])] for r in decisions]), "",
        "Decision-level dates and symbols are in each split's `decision_audit.csv`. The posterior-sized ungated "
        "control helps distinguish state exclusion from confidence-based sizing. If a full-library rolling policy "
        "chooses mean reversion, a separately trained momentum-only HMM control is also saved so expert changes "
        "are not attributed solely to gating.", "",
        "## Separate 100-stock breadth hypothesis", ""]
    breadth = report["breadth"]
    if "splits" in breadth:
        lines += [
            f"All {breadth['symbols']} requested stocks were available on the full common session calendar. "
            "This separate hypothesis uses momentum-only experts and fits its HMM on the broad stock universe; "
            "it is not the three-stock trained policy transferred unchanged. No 100-variable, 60-row "
            "cointegration expert is fitted. The original stock mapping is preserved, with documented vendor "
            "aliases for BRK.B, BK, and MMC.", "",
        ]
        for split, record in breadth["splits"].items():
            lines += ["### " + SPLITS[split], "", metric_table(record, initial, ["hmm", "ungated_momentum", "confidence_sized_ungated", "passive_stocks", "iwv", "cash"]), ""]
        if all(r["metrics"]["hmm"]["completed_positions"] == 0 for r in breadth["splits"].values()):
            lines += ["**The breadth HMM selected cash in both states and made no trades in either split.** "
                      "Its positive portfolio returns are entirely cash interest. Expanding the stock list did "
                      "not establish a profitable trading model under the frozen eligibility rules.", ""]
        if all(r["metrics"]["ungated_momentum"]["position_pnl"] > 0 for r in breadth["splits"].values()):
            lines += ["The broad ungated momentum strategy produced positive trading P&L in both observed "
                      "splits with many more completed positions. Its portfolio return nevertheless trailed "
                      "IWV in both splits, with lower exposure and drawdown. This is a useful simpler "
                      "research candidate; it does not establish Russell 3000 outperformance.", ""]
        lines += ["The user's current selected list is not point-in-time Russell 3000 membership. Survivor, "
                  "selection, and reused-period biases apply. Multiple stocks entering on the same date are "
                  "correlated exposures, not independent confirmations. See the separate breadth policy lock "
                  "and training diagnostics for state eligibility.", ""]
    else:
        lines += [f"Breadth evaluation status: `{breadth['status']}`. No breadth profitability claim is made. "
                  f"Recorded missing input details: {json.dumps(breadth, sort_keys=True)}", ""]
    lines += ["## Continue or pivot", "",
              table(["Declared evidence reference", "Observed", "Reference minimum", "Outcome"], [
                  [r["criterion"], str(r["observed"]) if r["criterion"].startswith("Completed") else percent(r["observed"]),
                   str(r["minimum"]) if r["criterion"].startswith("Completed") else percent(r["minimum"]),
                   "Pass" if r["passes"] else "Insufficient"] for r in criteria]), "",
              "The frozen positivity references use total portfolio returns, including cash interest. "
              f"The observed values are {positive_total}/{n} evaluation years and {percent(positive_total_offsets)} "
              f"of rolling offsets; cash interest accounts for all returns in {cash_years} evaluation years. "
              "Positive portfolio returns alone do not demonstrate profitable trade selection. "
              f"Trading-only positivity ({positive_trading}/{n} years) and paired HMM-minus-momentum offset "
              f"advantage ({percent(beat_offset_fraction)}) "
              "are additional diagnostics, without retroactively assigning the plan's positivity thresholds "
              "to those different metrics.", "",
              "The next useful work is a separately registered strategy hypothesis with an explicit opportunity "
              "cost of cash and enough expected activity to evaluate trade selection. Keep ungated momentum, "
              "IWV, and cash as controls. Freeze the model and decision rules before a prospective paper "
              "evaluation using sessions arriving after registration. None of this study's historical periods "
              "can become unseen data by relabeling them.", "",
              f"[The prospective paper evaluation registration]({registration_link}) was saved "
              "October 23, 2026 at 22:39:06 UTC, with the first eligible execution session October 26, 2026. "
              "It tracks the frozen reference HMM, three-stock and 100-stock ungated momentum, IWV, and cash. "
              "Its status is awaiting future observations: no prospective observations or results have been scored. "
              "A rule change requires a new registered version. The 12-month and 30-completed-position evidence "
              "targets do not establish statistical independence or prove an advantage. This registration "
              "does not schedule polling or submit brokerage orders.", "",
              "## Data, reproduction, and audit files", "",
              f"[Benchmark and data notes]({notes_link}) document issuer/FRED citations, dividend "
              "adjustments, ticker changes, complete calendars, hashes, and current-vintage limitations. "
              "The cash source is [Federal Reserve DGS3MO via FRED](https://fred.stlouisfed.org/series/DGS3MO), "
              "a 3-month yield proxy with lagged ACT/365 accrual; it is not a realized cash fund return or a "
              "guaranteed broker interest rate. Original three-stock cache bytes are unchanged.", "",
              "The full JSON, source/data protocol hashes, policy locks, daily books, order records, completed "
              "positions, decision audits, and offset results are under `output/value_study_reproduction_20261027/`. "
              "Flat CSV tables generated with this report are in its `report_tables/` subdirectory. "
              "Machine tables preserve unrounded values; report figures are rounded only for display.", "",
              "Generate the report from completed results without rerunning research:", "", "```powershell",
              ".\\.venv\\Scripts\\python.exe write_value_study_report.py --inputdir output/value_study_reproduction_20261027", "```", "",
              "Run a deliberate fresh reproduction with a new output directory because an existing study "
              "protocol is locked:", "", "```powershell",
              ".\\.venv\\Scripts\\python.exe DecisionValueStudy.py --plan value_study_plan.json --output-dir output/value_study_repeat", "```", "",
              f"Source result SHA-256: `{digest}`.", ""]
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(destination), "tables": str(out), "source_sha256": digest, "rolling_positive_trading_years": positive_trading, "rolling_completed_positions": count_positions, "evidence_references_missed": failed}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputdir", type=Path, default=Path("output/value_study_reproduction_20261027"))
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "VALUE_STUDY_RESULTS.md")
    args = parser.parse_args()
    main(args.inputdir.resolve(), args.output.resolve())
