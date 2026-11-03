"""Rebuild this fixed cash-cap comparison from the three saved studies."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import DecisionValueStudy as study
from run_portfolio_study import equity_chart
from sharpe_statistics import add_sharpe_statistics

OUTPUT = Path(__file__).resolve().parent
RUNS = {
    "80/20": ROOT / "output/portfolio_cash_limit_20261030/results",
    "90/10": ROOT / "output/portfolio_cash_limit_10_20261102/results",
    "95/5": ROOT / "output/portfolio_cash_limit_5_20261102/results",
}
VERSIONS = {"zero_slippage": 0.0, "half_cent_slippage": 0.005}
PERIOD = "continuous_2023_2026"
base_protocol = json.loads((RUNS["80/20"] / "protocol.json").read_text())
base_metrics = pd.read_csv(RUNS["80/20"] / "metrics.csv")
pd.testing.assert_frame_equal(base_metrics, pd.read_csv(ROOT / "results/portfolio_cash_limit_20261030/metrics.csv"))
base_config = base_protocol["plan"]["configuration"]
records, policies, hashes = [], {}, {}
books_checked = 0
max_residual = 0.0

for ratio, folder in RUNS.items():
    protocol = json.loads((folder / "protocol.json").read_text())
    plan = protocol["plan"]
    config = plan["configuration"]
    cap = int(ratio.split("/")[1]) / 100
    assert config["max_cash_weight"] == cap
    assert {k: v for k, v in config.items() if k != "max_cash_weight"} == {k: v for k, v in base_config.items() if k != "max_cash_weight"}
    for key in ("training_start", "validation_start", "evaluation_start", "end", "data_dir", "universe_file", "slippage_versions", "rolling_evaluation_years"):
        assert plan[key] == base_protocol["plan"][key], key
    assert protocol["data_manifest_sha256"] == base_protocol["data_manifest_sha256"]
    for source in ("SwingRegimeSelector.py", "DecisionValueStudy.py", "HyperplaneStock.py"):
        assert protocol["sources"][source] == base_protocol["sources"][source], source
    metrics = pd.read_csv(folder / "metrics.csv")
    assert len(metrics) == len(base_metrics) == 96
    assert metrics["attribution_residual"].abs().max() < 1e-6
    constrained = metrics.loc[metrics.strategy.isin(["hmm", "ungated_momentum", "equal_weight_floor"])]
    assert constrained.maximum_cash_weight_before_liquidation.max() <= cap + 1e-9
    for benchmark in ("iwv", "passive_stocks", "cash"):
        columns = ["version", "period", "net_cumulative_return", "final_equity", "sharpe", "max_drawdown"]
        pd.testing.assert_frame_equal(metrics.loc[metrics.strategy.eq(benchmark), columns].reset_index(drop=True), base_metrics.loc[base_metrics.strategy.eq(benchmark), columns].reset_index(drop=True))
    hashes[ratio] = {name: study.sha256(folder / name) for name in ("protocol.json", "metrics.csv", "research_report.json")}
    policies[ratio] = {}
    for version in VERSIONS:
        lock = json.loads((folder / version / "training/policy_lock.json").read_text())
        policies[ratio][version] = lock["policies"]
        curves = pd.read_csv(folder / version / PERIOD / "equity_curves.csv", index_col="Date", parse_dates=True)
        assert len(curves) == 945 and not curves.isna().any().any()
        for strategy in ("hmm", "equal_weight_floor"):
            daily = pd.read_csv(folder / version / PERIOD / strategy / "daily.csv", index_col="date", parse_dates=True)
            assert daily.index.equals(curves.index)
            assert daily.cash_weight.iloc[:-1].max() <= cap + 1e-9
            assert daily.gross_exposure.iloc[:-1].min() >= 1 - cap - 1e-9
            expected = config["initial_equity"] + (daily.market_pnl + daily.cash_interest - daily.transaction_cost - daily.borrow_cost).cumsum()
            residual = float(np.max(np.abs(expected - daily.equity)))
            assert residual < 1e-6
            max_residual = max(max_residual, residual)
            np.testing.assert_allclose(curves[strategy], daily.equity, rtol=0, atol=1e-8)
            books_checked += 1
        row = metrics.loc[metrics.version.eq(version) & metrics.period.eq(PERIOD) & metrics.strategy.eq("hmm")].iloc[0].to_dict()
        assert abs(row["final_equity"] - curves.hmm.iloc[-1]) < 1e-6
        row.update(allocation=ratio, matches_stock_floor=bool(np.allclose(curves.hmm, curves.equal_weight_floor, rtol=0, atol=1e-8)))
        records.append(row)

for version in VERSIONS:
    combined = {}
    for ratio, folder in RUNS.items():
        curves = pd.read_csv(folder / version / PERIOD / "equity_curves.csv", index_col="Date", parse_dates=True)
        if combined:
            assert next(iter(combined.values())).index.equals(curves.index)
        combined[f"{ratio} stocks/cash"] = curves.hmm
    combined["IWV (Russell 3000 ETF proxy)"] = curves.iwv
    combined["100-stock buy and hold"] = curves.passive_stocks
    frame = pd.DataFrame(combined)
    frame.to_csv(OUTPUT / f"equity_{version}.csv", index_label="Date")
    equity_chart(frame, OUTPUT / f"comparison_{version}.svg", base_config["initial_equity"])
    equity_chart(frame.iloc[:, :3], OUTPUT / f"allocations_{version}.svg", base_config["initial_equity"])
    for strategy, label in (("iwv", "IWV (Russell 3000 ETF proxy)"), ("passive_stocks", "100-stock buy and hold")):
        row = base_metrics.loc[base_metrics.version.eq(version) & base_metrics.period.eq(PERIOD) & base_metrics.strategy.eq(strategy)].iloc[0].to_dict()
        row.update(allocation=label, matches_stock_floor=False)
        records.append(row)

comparison = pd.DataFrame(records)
comparison["return_vs_80_20_pp"] = comparison.apply(lambda row: 100 * (row.net_cumulative_return - comparison.loc[comparison.version.eq(row.version) & comparison.allocation.eq("80/20"), "net_cumulative_return"].iloc[0]), axis=1)
comparison = add_sharpe_statistics(comparison, RUNS, PERIOD, OUTPUT)
comparison.to_csv(OUTPUT / "comparison.csv", index=False)
study.save_json(OUTPUT / "verification.json", {
    "continuous_books_reconciled": books_checked,
    "maximum_equity_reconciliation_residual_dollars": max_residual,
    "all_288_saved_metric_rows_reconciled": True,
    "common_frozen_data_and_model_sources": True,
    "benchmark_metrics_unchanged_across_cash_caps": True,
    "cash_caps_and_stock_floors_passed": True,
    "original_80_20_committed_metrics_match": True,
    "policies": policies,
    "input_sha256": hashes,
})

lines = ["# Cash-cap comparison: 80/20, 90/10, and 95/5", "",
    "Declared November 2, 2026. Same frozen 100-stock data, 2015–2022 training, seed 42, 20-session rebalance schedule, costs, and Treasury accounting as the existing 80/20 study. Only max_cash_weight changes; every cap/slippage version is independently fitted. Continuous evaluation: January 3, 2023 through October 8, 2026. All returns are cumulative after costs and modeled cash interest.", "",
    "Each book starts with $1 million fully invested in the same equal-weight basket. The cash cap applies after the first scheduled rebalance, at daily opens; it is a maximum cash allocation, not an exact daily stock/cash ratio. Paid terminal liquidation is exempt.", "",
    "The 95/5 run has the highest cumulative return among the three constrained versions; 80/20 has the smallest drawdown and highest Sharpe. Both 90/10 and 95/5 beat IWV in this historical window, while all three trail full-stock buy and hold. All six cap/slippage fits select the mechanical allocation fallback, so the comparison establishes allocation sensitivity rather than additional HMM trading value.", "",
    "![After-cost comparison and return differences](comparison.png)", ""]
for version, slippage in VERSIONS.items():
    lines += [f"## Slippage: ${slippage:g} per actual share per execution", "",
        "| Allocation / benchmark | Final equity | Net return | Annualized return | Sharpe | Mean yearly Sharpe | SD yearly Sharpe | Max drawdown | Avg stock exposure | Return vs 80/20 | Execution costs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in comparison.loc[comparison.version.eq(version)].to_dict("records"):
        lines.append(f"| {row['allocation']} | ${row['final_equity']:,.2f} | {row['net_cumulative_return']:+.2%} | {row['annualized_return']:+.2%} | {row['sharpe']:.3f} | {row['annual_sharpe_mean']:.3f} | {row['annual_sharpe_std']:.3f} | {row['max_drawdown']:.2%} | {row['average_gross_exposure']:.2%} | {row['return_vs_80_20_pp']:+.2f} pp | ${row['transaction_cost']:,.2f} |")
    lines += ["", f"![Allocation and benchmark comparison](comparison_{version}.svg)", ""]
lines += ["## Trained policy checks", ""]
for ratio, versions in policies.items():
    for version, policy in versions.items():
        matches = comparison.loc[comparison.allocation.eq(ratio) & comparison.version.eq(version), "matches_stock_floor"].iloc[0]
        lines.append(f"- {ratio}, {version}: {json.dumps(policy, sort_keys=True)}; daily equity matches its mechanical stock-floor control: {matches}.")
lines += ["", "Policy labels of cash refer to the mandatory equal-weight stock-floor fallback in these constrained experiments. When a run matches that control, increased returns reflect allocation exposure rather than demonstrated HMM selection value.", "",
    "## Verification and reproduction", "",
    "The portfolio acceptance suite passed 29 tests, including cap enforcement at 20%, 10%, and 5% after execution costs. All 288 saved metric rows reconcile their P&L attribution. Twelve continuous model/control daily books reconcile equity, meet their stock floors and cash caps, and match their chart data. Model sources and frozen-data manifest match across runs; benchmark metrics are unchanged. See verification.json for hashes and residuals.", "",
    "Run the new studies with `.\\.venv\\Scripts\\python.exe run_portfolio_study.py --plan portfolio_cash_limit_10_plan.json` and the corresponding `portfolio_cash_limit_5_plan.json`. Existing protocols are locked: a deliberate repeat requires a new `--output-dir`. Rebuild these tables and SVGs with `.\\.venv\\Scripts\\python.exe results/cash_cap_comparison_20261102/build_comparison.py` after the studies complete.", "",
    "Full daily books, policy locks, costs, standalone windows, and rolling 2022–2026 checks remain in each study's output folder. The original 80/20 run is preserved. PNG previews use the saved CSVs; SVGs need no plotting dependency. Rebuild the PNG preview with `python results/cash_cap_comparison_20261102/plot_preview.py` using a Python installation with pandas and matplotlib; no project dependencies were added.", "",
    "Historical periods have already been examined. The selected stock universe has survivor and selection bias. These are retrospective sensitivity tests, not independent prospective evidence. IWV is an ETF proxy, not the official Russell 3000 index. Daily data cannot enforce an intraday cash cap.", ""]
lines += ["## Yearly Sharpe summary", "", f"Mean yearly Sharpe and SD yearly Sharpe summarize {int(comparison.annual_sharpe_count.iloc[0])} calendar-year observations per strategy, including partial 2026. SD is the sample standard deviation (ddof=1). These are separate from the full-period Sharpe. See [methodology and summary](SHARPE_STATISTICS.md), [yearly observations](yearly_sharpe.csv), and [summary CSV](sharpe_statistics.csv).", ""]
(OUTPUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
print(comparison[["allocation", "version", "net_cumulative_return", "sharpe", "max_drawdown", "matches_stock_floor"]].to_string(index=False))
print(f"Verified {books_checked} continuous books; maximum equity residual ${max_residual:.9f}")
