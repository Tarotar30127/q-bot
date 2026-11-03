"""Rebuild the two current cash-cap comparisons from completed portfolio studies."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_utils import save_json, sha256
from run_portfolio_study import equity_chart
from sharpe_statistics import add_sharpe_statistics

ROOT = Path(__file__).resolve().parent
STUDIES = {
    "2023": (["portfolio_cash_limit_plan.json", "portfolio_cash_limit_10_plan.json", "portfolio_cash_limit_5_plan.json"], "cash_cap_comparison_20261102"),
    "2020": ([f"portfolio_train2000_cash{cap}_plan.json" for cap in (20, 10, 5)], "train2000_comparison_20261102"),
}


def build(window: str) -> pd.DataFrame:
    names, result_name = STUDIES[window]
    plans = [json.loads((ROOT / name).read_text(encoding="utf-8")) for name in names]
    base = plans[0]
    period = f"continuous_{pd.Timestamp(base['validation_start']).year}_{pd.Timestamp(base['end']).year}"
    output = ROOT / "results" / result_name
    output.mkdir(parents=True, exist_ok=True)
    records, runs, protocols, policies = [], {}, {}, {}
    books_checked, max_residual = 0, 0.0
    base_metrics = pd.read_csv(ROOT / base["output_dir"] / "metrics.csv")
    for plan in plans:
        cap = plan["configuration"]["max_cash_weight"]
        label = f"{1 - cap:.0%}/{cap:.0%}".replace("%", "")
        folder = ROOT / plan["output_dir"]
        runs[label] = folder
        protocol = json.loads((folder / "protocol.json").read_text())
        assert protocol["plan"] == plan
        protocols[label] = protocol
        policies[label] = {}
        for key in ("training_start", "validation_start", "evaluation_start", "end", "data_dir", "universe_file", "slippage_versions", "rolling_evaluation_years"):
            assert plan[key] == base[key], key
        assert {k: v for k, v in plan["configuration"].items() if k != "max_cash_weight"} == {k: v for k, v in base["configuration"].items() if k != "max_cash_weight"}
        for name, digest in protocol["sources"].items():
            assert sha256(folder / "sources" / name) == digest, name
        metrics = pd.read_csv(folder / "metrics.csv")
        assert len(metrics) == 6 * len(plan["slippage_versions"]) * (3 + len(plan["rolling_evaluation_years"]))
        assert metrics.attribution_residual.abs().max() < 1e-6
        constrained = metrics.loc[metrics.strategy.isin(["hmm", "ungated_momentum", "equal_weight_floor"])]
        assert constrained.maximum_cash_weight_before_liquidation.max() <= cap + 1e-9
        for benchmark in ("iwv", "passive_stocks", "cash"):
            columns = ["version", "period", "net_cumulative_return", "final_equity", "sharpe", "max_drawdown"]
            pd.testing.assert_frame_equal(metrics.loc[metrics.strategy.eq(benchmark), columns].reset_index(drop=True), base_metrics.loc[base_metrics.strategy.eq(benchmark), columns].reset_index(drop=True))
        for version in plan["slippage_versions"]:
            lock = json.loads((folder / version / "training/policy_lock.json").read_text())
            assert lock["selection_used_evaluation_outcomes"] is False
            policies[label][version] = lock["policies"]
            curves = pd.read_csv(folder / version / period / "equity_curves.csv", index_col="Date", parse_dates=True)
            assert not curves.isna().any().any()
            for strategy in ("hmm", "equal_weight_floor"):
                daily = pd.read_csv(folder / version / period / strategy / "daily.csv", index_col="date", parse_dates=True)
                assert daily.index.equals(curves.index)
                assert daily.cash_weight.iloc[:-1].max() <= cap + 1e-9
                assert daily.gross_exposure.iloc[:-1].min() >= 1 - cap - 1e-9
                expected = plan["configuration"]["initial_equity"] + (daily.market_pnl + daily.cash_interest - daily.transaction_cost - daily.borrow_cost).cumsum()
                residual = float(np.max(np.abs(expected - daily.equity)))
                assert residual < 1e-6
                max_residual = max(max_residual, residual)
                np.testing.assert_allclose(curves[strategy], daily.equity, rtol=0, atol=1e-8)
                books_checked += 1
            row = metrics.loc[metrics.version.eq(version) & metrics.period.eq(period) & metrics.strategy.eq("hmm")].iloc[0].to_dict()
            assert abs(row["final_equity"] - curves.hmm.iloc[-1]) < 1e-6
            row.update(allocation=label, matches_stock_floor=bool(np.allclose(curves.hmm, curves.equal_weight_floor, rtol=0, atol=1e-8)))
            records.append(row)
    for protocol in protocols.values():
        assert protocol["data_manifest_sha256"] == next(iter(protocols.values()))["data_manifest_sha256"]
    for version in base["slippage_versions"]:
        combined = {}
        for label, folder in runs.items():
            curves = pd.read_csv(folder / version / period / "equity_curves.csv", index_col="Date", parse_dates=True)
            if combined:
                assert next(iter(combined.values())).index.equals(curves.index)
            combined[f"{label} stocks/cash"] = curves.hmm
        for strategy, label in (("iwv", "IWV (Russell 3000 ETF proxy)"), ("passive_stocks", "100-stock buy and hold")):
            combined[label] = curves[strategy]
            row = base_metrics.loc[base_metrics.version.eq(version) & base_metrics.period.eq(period) & base_metrics.strategy.eq(strategy)].iloc[0].to_dict()
            row.update(allocation=label, matches_stock_floor=False)
            records.append(row)
        pd.DataFrame(combined).to_csv(output / f"equity_{version}.csv", index_label="Date")
    comparison = pd.DataFrame(records)
    comparison["return_vs_80_20_pp"] = comparison.apply(lambda row: 100 * (row.net_cumulative_return - comparison.loc[comparison.version.eq(row.version) & comparison.allocation.eq("80/20"), "net_cumulative_return"].iloc[0]), axis=1)
    comparison = add_sharpe_statistics(comparison, runs, period, output)
    comparison.to_csv(output / "comparison.csv", index=False)
    chart_data = pd.read_csv(output / "equity_half_cent_slippage.csv", index_col="Date", parse_dates=True)
    equity_chart(chart_data, output / "comparison.svg", base["configuration"]["initial_equity"])
    save_json(output / "verification.json", {
        "studies_rerun_in_qbot": True, "continuous_books_reconciled": books_checked,
        "maximum_equity_reconciliation_residual_dollars": max_residual,
        "benchmark_metrics_unchanged_across_cash_caps": True,
        "cash_caps_and_stock_floors_passed": True, "policies": policies,
        "input_sha256": {label: {name: sha256(folder / name) for name in ("protocol.json", "metrics.csv", "research_report.json")} for label, folder in runs.items()},
        "summary_sources": {name: sha256(ROOT / name) for name in ("compare_portfolios.py", "sharpe_statistics.py")},
    })
    lines = [f"# Cash-cap comparison: {base['validation_start']} through {base['end']}", "",
             f"Training: {base['training_start']} through {pd.Timestamp(base['validation_start']).year - 1}. Each cash-cap/slippage version is fitted independently using the same frozen 100-stock data and seed 42. Continuous books retain holdings across year boundaries. All returns are cumulative, after execution costs and modeled cash interest.", "",
             "Each book starts with $1 million equally invested. The cash cap applies at daily opens after the initial 20-session holding; terminal liquidation is exempt. Cash policy labels mean the mandatory equal-weight stock-floor fallback. A fitted momentum policy must show incremental value above that matched control.", "",
             "![After-cost allocation and benchmark comparison](comparison.svg)", ""]
    for version, slippage in base["slippage_versions"].items():
        lines += [f"## Slippage: ${slippage:g} per actual share per execution", "",
                  "| Allocation / benchmark | Final equity | Net return | Annualized return | Full-period Sharpe | Mean yearly Sharpe | SD yearly Sharpe | Max drawdown | Avg exposure |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in comparison.loc[comparison.version.eq(version)].to_dict("records"):
            lines.append(f"| {row['allocation']} | ${row['final_equity']:,.2f} | {row['net_cumulative_return']:+.2%} | {row['annualized_return']:+.2%} | {row['sharpe']:.3f} | {row['annual_sharpe_mean']:.3f} | {row['annual_sharpe_std']:.3f} | {row['max_drawdown']:.2%} | {row['average_gross_exposure']:.2%} |")
        lines.append("")
    lines += ["## Trained policies", ""]
    for label, versions in policies.items():
        for version, policy in versions.items():
            matched = comparison.loc[comparison.allocation.eq(label) & comparison.version.eq(version), "matches_stock_floor"].iloc[0]
            lines.append(f"- {label}, {version}: {json.dumps(policy, sort_keys=True)}; matches the mechanical stock-floor control: {matched}.")
    lines += ["", "Matching the control means the HMM added no incremental value in that run. More compulsory stock exposure can increase both return and drawdown.", "",
              "## Method and verification", "", "[Methodology](../../docs/METHODOLOGY.md) · [Yearly Sharpe details](SHARPE_STATISTICS.md) · [Comparison CSV](comparison.csv) · [Yearly observations](yearly_sharpe.csv) · [Verification](verification.json)", "",
              f"The yearly Sharpe summaries use {int(comparison.annual_sharpe_count.iloc[0])} calendar-year observations, including partial 2026. Standard deviation is the sample SD across yearly Sharpes (ddof=1), not a standard error or confidence interval.", "",
              "Rebuild this report after running its three plans: `python compare_portfolios.py --window " + window + "`. Optional PNG preview: `python plot_comparison.py --window " + window + "` using a Python installation with matplotlib. Full study books and source/policy locks remain in ignored output/.", "",
              base["limitations"], ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Verified {books_checked} continuous books; report: {output / 'REPORT.md'}", flush=True)
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", choices=STUDIES, required=True)
    build(parser.parse_args().window)
