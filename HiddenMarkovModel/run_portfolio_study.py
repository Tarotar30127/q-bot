"""Two independently calibrated 100-stock backtests, with audited frozen inputs."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

import DecisionValueStudy as study
import HyperplaneStock as hs
import SwingRegimeSelector as swing

ROOT = Path(__file__).resolve().parent


def fit_cash_limited_policy(data, quoted, inputs, train_dates, config, folder):
    """Require evidence above the compulsory stock allocation, after actual costs."""
    policies = {state: "cash" for state in range(config.hmm_states)}
    baseline_targets = swing.build_targets(inputs, policies, config)
    baseline = swing.simulate_portfolio(data, baseline_targets, train_dates, config,
                                         execution_prices=quoted)
    study.write_simulation(folder / "equal_weight_floor", baseline, data)
    diagnostics = []
    lags = max(config.rebalance_sessions, config.hac_lags or 5)
    scheduled = pd.Series(np.arange(len(train_dates)) % config.rebalance_sessions == 0,
                          index=train_dates)
    scheduled.iloc[-1] = False
    for state in range(config.hmm_states):
        candidate = {other: "momentum" if other == state else "cash" for other in policies}
        targets = swing.build_targets(inputs, candidate, config)
        result = swing.simulate_portfolio(data, targets, train_dates, config,
                                           execution_prices=quoted)
        incremental = result.daily["net_return"] - baseline.daily["net_return"]
        mean, error = swing.hac_mean_standard_error(incremental, lags)
        bound = mean - config.confidence_z * error
        changed = (targets.loc[train_dates] - baseline_targets.loc[train_dates]).abs().max(axis=1).gt(1e-12)
        active = changed.where(scheduled).ffill().fillna(False).astype(bool)
        active_days = int(active.iloc[:-1].sum())
        active_rebalances = int((changed & scheduled).sum())
        eligible = active_days >= config.min_train_days and active_rebalances >= config.min_train_rebalances
        if eligible and bound > 0.0:
            policies[state] = "momentum"
        diagnostics.append({"state": state, "expert": "momentum", "selected": policies[state] == "momentum",
                            "selection_reference": "equal_weight_stock_floor_with_matching_costs",
                            "mean_incremental_net_daily_return": mean, "standard_error": error,
                            "lower_bound": bound, "eligible": eligible, "model_active_days": active_days,
                            "model_active_rebalances": active_rebalances, "training_metrics": result.metrics,
                            "baseline_training_metrics": baseline.metrics})
    return policies, diagnostics


def fit_version(data, quoted, plan, config, train_end, folder):
    """Fit every version anew; only training observations decide state eligibility."""
    inputs = swing.prepare_inputs(data, study._empty_predictions(), train_end, config,
                                  training_start=plan["training_start"])
    train_dates = data.index[(data.index >= plan["training_start"]) & (data.index < train_end)]
    if config.max_cash_weight < 1.0:
        policy, diagnostics = fit_cash_limited_policy(data, quoted, inputs, train_dates, config, folder)
    else:
        policy, diagnostics = study._fit_library(data, inputs, train_dates, config, True,
                                                 execution_prices=quoted)
    targets = swing.build_targets(inputs, policy, config)
    study.save_json(folder / "policy_lock.json", {
        "config": asdict(config), "policies": policy, "training_diagnostics": diagnostics,
        "training_start": str(train_dates[0].date()), "training_end": str(train_dates[-1].date()),
        "hmm_fit_independently_for_this_version": True, "hmm_seed": config.random_seed,
        "calibration_cash_interest": 0.0, "calibration_initial_equal_weight": False,
        "cash_policy_fallback": "equal_weight_stock_floor" if config.max_cash_weight < 1.0 else "cash",
        "selection_used_evaluation_outcomes": False,
    })
    inputs.regimes.to_csv(folder / "regime_probabilities.csv", index_label="Date")
    targets.to_csv(folder / "target_weights.csv", index_label="Date")
    return inputs, targets, policy


def score_period(data, benchmark, quoted, benchmark_quoted, cash_rates, risk_free_rates,
                 inputs, target, dates, config, folder):
    passive = replace(config, rebalance_sessions=len(dates), hac_lags=None, max_cash_weight=1.0)
    specs = {
        "hmm": (data, quoted, target, config, True),
        "ungated_momentum": (data, quoted, inputs.momentum / len(data.tickers), config, True),
        "passive_stocks": (data, quoted, study.benchmark_targets(data.index, data.tickers, 1.0), passive, True),
        "iwv": (benchmark, benchmark_quoted, study.benchmark_targets(benchmark.index, benchmark.tickers, 1.0), passive, True),
        "cash": (benchmark, benchmark_quoted, study.benchmark_targets(benchmark.index, benchmark.tickers, 0.0), passive, False),
    }
    if config.max_cash_weight < 1.0:
        specs["equal_weight_floor"] = (data, quoted, study.benchmark_targets(
            data.index, data.tickers, 1.0 - config.max_cash_weight), config, True)
    metrics, curves = {}, {}
    for name, (prices, execution, weights, settings, initial_basket) in specs.items():
        result = swing.simulate_portfolio(
            prices, weights, dates, settings, cash_annual_rates=cash_rates,
            execution_prices=execution, risk_free_annual_rates=risk_free_rates,
            initial_equal_weight=initial_basket,
        )
        metrics[name] = study.write_simulation(folder / name, result, prices)
        curves[name] = result.daily["equity"]
        if name in ("hmm", "ungated_momentum"):
            metrics[name]["first_model_rebalance"] = str(dates[config.rebalance_sessions].date()) if len(dates) > config.rebalance_sessions + 1 else None
            metrics[name]["imposed_initial_holding_market_pnl"] = float(result.daily["market_pnl"].iloc[1:config.rebalance_sessions + 1].sum())
            later = result.daily["gross_exposure"].iloc[config.rebalance_sessions:-1]
            metrics[name]["average_model_exposure_after_initial_holding"] = float(later.mean()) if len(later) else None
            study.save_json(folder / name / "metrics.json", metrics[name])
    curve_frame = pd.DataFrame(curves)
    curve_frame.to_csv(folder / "equity_curves.csv", index_label="Date")
    stock_returns = data.open.loc[dates[-1]] / data.open.loc[dates[0]] - 1.0
    stock_returns.rename("adjusted_open_to_open_return").to_csv(folder / "individual_stock_returns.csv", index_label="ticker")
    study.save_json(folder / "summary.json", metrics)
    return metrics, curve_frame


def equity_chart(curves: pd.DataFrame, path: Path, initial_equity: float) -> None:
    """Standalone vector chart using saved observations; no remote chart dependency."""
    width, height, left, top = 1080, 550, 95, 65
    plot_width, plot_height = 930, 360
    low = min(initial_equity, float(curves.min().min())) * 0.95
    high = max(initial_equity, float(curves.max().max())) * 1.05
    span = max(high - low, 1.0)
    dates = pd.DatetimeIndex(curves.index)
    seconds = (dates - dates[0]).total_seconds().to_numpy()
    duration = max(1.0, float(seconds[-1]))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="Arial, sans-serif" font-size="14" fill="#202020">',
             '<text x="95" y="28" font-size="20">100-stock portfolio: continuous equity after costs</text>']
    for index in range(5):
        value = low + span * index / 4
        y = top + plot_height * (1 - index / 4)
        parts += [f'<line x1="{left}" y1="{y}" x2="{left + plot_width}" y2="{y}" stroke="#dddddd"/>',
                  f'<text x="{left - 10}" y="{y + 5}" text-anchor="end">${value / 1e6:.2f}m</text>']
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#333333", "#8C6D31"]
    for index, column in enumerate(curves):
        color = colors[index % len(colors)]
        values = curves[column].to_numpy(dtype=float)
        coordinates = [f"{left:.1f},{top + plot_height * (high - initial_equity) / span:.1f}"]
        coordinates += [f"{left + plot_width * x / duration:.1f},{top + plot_height * (high - y) / span:.1f}" for x, y in zip(seconds, values)]
        dash = ' stroke-dasharray="7 4"' if "$0.005" in column else ""
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2"{dash} points="{" ".join(coordinates)}"/>')
        legend_x, legend_y = left + (index % 2) * 465, 470 + (index // 2) * 24
        parts += [f'<line x1="{legend_x}" y1="{legend_y - 5}" x2="{legend_x + 28}" y2="{legend_y - 5}" stroke="{color}" stroke-width="3"{dash}/>',
                  f'<text x="{legend_x + 36}" y="{legend_y}">{escape(column.replace("_", " "))}</text>']
    parts += [f'<text x="{left}" y="450">{dates[0].date()}</text>',
              f'<text x="{left + plot_width}" y="450" text-anchor="end">{dates[-1].date()}</text>', '</g></svg>']
    path.write_text("\n".join(parts), encoding="utf-8")


def write_report(output: Path, plan: dict, records: list[dict], policies: dict, curves: dict):
    def percentage(value):
        return f"{value:+.2%}"

    lines = ["# 100-stock portfolio: 2015 training and per-share slippage", "",
             "Each slippage version fits independently on the same inputs and seed. The primary model uses the existing 100-stock momentum-only HMM hypothesis.", "",
             "Main training: 2015–2022. Warm-up prices begin June 2014. Evaluation history was previously observed and is research data.", "",
             "Every independent book starts with $1,000,000. All stock strategies buy equal dollar amounts at the first open; costs come from that budget. Shares remain fixed until the first model rebalance 20 trading sessions later. Subsequent model decisions may hold cash. The continuous book carries positions across year boundaries; standalone windows and yearly checks restart separately.", "",
             "Slippage is fixed at $0 or $0.005 per actual share, on each side, including liquidation. Commission remains zero. Adjusted-price total returns approximate dividend reinvestment without adding dividends again; quoted opens are reconstructed from Yahoo split history for share-based fees.", "",
             "Sharpe uses [DGS10](https://fred.stlouisfed.org/series/DGS10), lagged two stock sessions and accrued at the prior-open yield using elapsed calendar days / 365. It is sqrt(252) times the mean interval excess return divided by its sample standard deviation. Entry costs belong to the first holding interval. Idle cash separately earns the existing [DGS3MO](https://fred.stlouisfed.org/series/DGS3MO) approximation. The 10-year yield is a reference yield, not a realized bond total return.", "",
             "## Continuous portfolio: 2023 through October 8, 2026", "",
             "| Slippage | Strategy | Final equity | Net return | Annualized return | Sharpe | Max drawdown | Execution costs | Avg exposure |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in records:
        if row["period"] != "continuous_2023_2026":
            continue
        sharpe = "n/a" if row["sharpe"] is None else f"{row['sharpe']:.3f}"
        lines.append(f"| ${row['slippage_dollars_per_share']:g}/share | {row['strategy']} | ${row['final_equity']:,.2f} | {percentage(row['net_cumulative_return'])} | {percentage(row['annualized_return'])} | {sharpe} | {percentage(row['max_drawdown'])} | ${row['transaction_cost']:,.2f} | {row['average_gross_exposure']:.2%} |")
    lines += ["", "![Continuous equity comparison](equity_curves.svg)", "", "![Model slippage comparison](slippage_comparison.svg)", "", "## Independently trained policies", ""]
    for version, policy in policies.items():
        lines.append(f"- {version}: {json.dumps(policy, sort_keys=True)}")
    max_cash = plan["configuration"].get("max_cash_weight", 1.0)
    if max_cash < 1.0:
        lines += ["", f"Cash is capped at {max_cash:.0%} at each observed daily open, except terminal liquidation. After the initial 100% stock holding, target shortfalls are filled equally across all 100 stocks. Price declines or interest can trigger equal-dollar top-up purchases between scheduled rebalances; intraday breaches cannot be observed with daily data.", "",
                  "State policies labeled cash mean the equal-weight stock-floor fallback in this constrained study. A momentum policy must clear the conservative training bound on incremental daily returns above that fallback, with matching execution costs and zero calibration interest. Required stock exposure alone cannot qualify a model.", "",
                  "The equal_weight_floor control follows the same first-month holding, scheduled rebalances, daily floor enforcement, cash interest, and costs, with no HMM or stock-selection signal. Positive model-minus-control returns may also reflect different stock exposure within the allowed 80–100% range; they do not isolate stock-selection skill."]
    if len({json.dumps(p, sort_keys=True) for p in policies.values()}) == 1:
        lines += ["", "The separate training runs selected the same state policy. This is a valid result; no different policy was forced."]
    if max_cash == 1.0 and all(all(expert == "cash" for expert in policy.values()) for policy in policies.values()):
        first_rebalance = next(row["first_model_rebalance"] for row in records
                               if row["period"] == "continuous_2023_2026" and row["strategy"] == "hmm")
        lines += ["", f"Both main HMM policies choose cash in every state. Each continuous model book therefore sells the imposed basket on {first_rebalance} and remains in cash. Its gains come from the initial holding period and subsequent modeled cash interest, not later HMM-selected stock trades."]
    lines += ["", "The initial basket is imposed, so its 100 completed positions cannot be interpreted as 100 model-selected entries. Training eligibility excludes that imposed purchase. See imposed_initial_holding_market_pnl and average_model_exposure_after_initial_holding in the machine tables.", "",
              "## Standalone windows and expanding-training checks", "",
              "| Period | Slippage | Strategy | Net return | Sharpe | Max drawdown |",
              "|---|---|---|---:|---:|---:|"]
    for row in records:
        if row["period"] == "continuous_2023_2026" or row["strategy"] not in ("hmm", "passive_stocks"):
            continue
        sharpe = "n/a" if row["sharpe"] is None else f"{row['sharpe']:.3f}"
        lines.append(f"| {row['period']} | ${row['slippage_dollars_per_share']:g}/share | {row['strategy']} | {percentage(row['net_cumulative_return'])} | {sharpe} | {percentage(row['max_drawdown'])} |")
    lines += ["", "Rolling year Y trains from 2015 through Y−2, leaves Y−1 as a validation gap, and evaluates Y. No configuration is selected using those evaluations. These independently restarted books are not compounded into the primary continuous result.", "",
              "## Files and limitations", "",
              "metrics.csv contains every strategy and window. Each version has policy locks, training diagnostics, regime probabilities, target weights, daily books, actual-share order costs, completed-position attribution, and per-stock passive returns. Original studies and the prospective paper registration are unchanged.", "",
              plan["limitations"], "",
              "IWV is an investable Russell 3000 ETF proxy, not the official index. The zero-slippage passive_stocks row represents the same 100-stock basket's adjusted open-to-open performance over the matching dates. Neither ETF fund fees nor dividends are deducted/added a second time.", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    selected = {
        "Model (zero slippage)": curves["zero_slippage"]["hmm"],
        "Model ($0.005/share slippage)": curves["half_cent_slippage"]["hmm"],
        "100-stock portfolio (buy and hold)": curves["zero_slippage"]["passive_stocks"],
    }
    equity_chart(pd.DataFrame(selected), output / "equity_curves.svg", plan["configuration"]["initial_equity"])
    equity_chart(pd.DataFrame(selected).iloc[:, :2], output / "slippage_comparison.svg", plan["configuration"]["initial_equity"])


def run(plan_path: Path, output_override: str | None = None):
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    output = ROOT / (output_override or plan["output_dir"])
    if (output / "protocol.json").exists():
        raise ValueError("Existing protocol is locked; use --output-dir for a deliberate repeat.")
    directory = ROOT / plan["data_dir"]
    manifest = json.loads((directory / "manifest.json").read_text())
    data_plan_path = ROOT / plan.get("data_plan_file", str(plan_path))
    data_plan = json.loads(data_plan_path.read_text(encoding="utf-8"))
    for field in ("download_start", "download_end_exclusive", "end", "universe_file", "benchmark_ticker"):
        if plan[field] != data_plan[field]:
            raise ValueError(f"Reused data plan has different {field}.")
    if manifest["plan_sha256"] != study.sha256(data_plan_path) or manifest["universe_sha256"] != study.sha256(ROOT / plan["universe_file"]):
        raise ValueError("Plan/universe does not match frozen inputs.")
    for relative, digest in manifest["files"].items():
        if study.sha256(directory / relative) != digest:
            raise ValueError(f"Frozen input changed: {relative}")
    audit = json.loads((directory / "audit.json").read_text())
    if audit["status"] != "complete" or study.sha256(directory / "audit.json") != manifest["audit_sha256"]:
        raise ValueError("A complete, unchanged panel audit is required.")
    universe = json.loads((ROOT / plan["universe_file"]).read_text(encoding="utf-8"))
    tickers = list(universe["stocks"])
    if len(tickers) != 100:
        raise ValueError("This protocol requires the approved 100-stock universe.")
    data = hs.load_csv_market_data(tickers, directory / "prices", plan["download_start"], plan["end"])
    benchmark = hs.load_csv_market_data([plan["benchmark_ticker"]], directory / "prices", plan["download_start"], plan["end"])
    if not data.index.equals(benchmark.index):
        raise ValueError("Every stock must match the benchmark calendar without price filling.")
    quoted_all = pd.concat({ticker: pd.read_csv(directory / "quoted" / f"{ticker}.csv", index_col="Date", parse_dates=True)["QuotedOpen"]
                            for ticker in [*tickers, plan["benchmark_ticker"]]}, axis=1)
    quoted = quoted_all[tickers]
    benchmark_quoted = quoted_all[[plan["benchmark_ticker"]]]
    rate_dates = data.index[data.index >= plan["training_start"]]
    cash_rates = study._load_cash_rates(directory / "rates/DGS3MO.csv", rate_dates)
    risk_free_rates = study._load_cash_rates(directory / "rates/DGS10.csv", rate_dates)
    sources = [plan_path, Path(__file__), ROOT / "SwingRegimeSelector.py", ROOT / "DecisionValueStudy.py",
               ROOT / "HyperplaneStock.py", ROOT / "download_portfolio_study_data.py", ROOT / "download_value_study_data.py"]
    if data_plan_path.resolve() != plan_path.resolve():
        sources.append(data_plan_path)
    study.save_json(output / "protocol.json", {"plan": plan, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                                               "created_before_training_and_evaluation": True,
                                               "sources": {p.name: study.sha256(p) for p in sources},
                                               "data_manifest_sha256": study.sha256(directory / "manifest.json"),
                                               "python": sys.version})
    source_folder = output / "sources"
    source_folder.mkdir()
    for source in sources:
        shutil.copyfile(source, source_folder / source.name)
    pd.DataFrame({"DGS10_lagged_annual_decimal": risk_free_rates, "DGS3MO_lagged_annual_decimal": cash_rates}).to_csv(output / "lagged_rates.csv", index_label="Date")
    records, policies, curves, fitted = [], {}, {}, {}
    for version, slippage in plan["slippage_versions"].items():
        config = swing.SwingConfig(**plan["configuration"], slippage_dollars_per_share=slippage)
        fitted[version] = (config, *fit_version(data, quoted, plan, config, pd.Timestamp(plan["validation_start"]), output / version / "training"))
        policies[version] = fitted[version][-1]
    for version, (config, inputs, target, _) in fitted.items():
        windows = {
            "continuous_2023_2026": data.index[data.index >= plan["validation_start"]],
            "standalone_2023_2025": data.index[(data.index >= plan["validation_start"]) & (data.index < plan["evaluation_start"])],
            "standalone_2026": data.index[data.index >= plan["evaluation_start"]],
        }
        for year in plan["rolling_evaluation_years"]:
            windows[f"rolling_{year}"] = data.index[(data.index.year == year)]
        for period, dates in windows.items():
            active_inputs, active_targets = inputs, target
            if period.startswith("rolling_"):
                year = int(period.split("_")[1])
                active_inputs, active_targets, _ = fit_version(data, quoted, plan, config,
                                                              pd.Timestamp(year=year - 1, month=1, day=1),
                                                              output / version / period / "training")
            metrics, curve = score_period(data, benchmark, quoted, benchmark_quoted, cash_rates, risk_free_rates,
                                           active_inputs, active_targets, dates, config, output / version / period)
            for strategy, values in metrics.items():
                records.append({"version": version, "slippage_dollars_per_share": config.slippage_dollars_per_share,
                                "period": period, "strategy": strategy, **values})
            if period == "continuous_2023_2026":
                curves[version] = curve
            print(json.dumps(study.clean_json({"version": version, "period": period,
                                              "hmm_return": metrics["hmm"]["net_cumulative_return"],
                                              "hmm_sharpe": metrics["hmm"]["sharpe"],
                                              "passive_return": metrics["passive_stocks"]["net_cumulative_return"]})), flush=True)
    pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, dict)} for row in records]).to_csv(output / "metrics.csv", index=False)
    study.save_json(output / "research_report.json", {"plan": plan, "main_policies": policies, "records": records})
    write_report(output, plan, records, policies, curves)
    print(f"Completed: {output / 'REPORT.md'}", flush=True)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="portfolio_study_plan.json")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    run(ROOT / args.plan, args.output_dir)
