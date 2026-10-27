"""Frozen decision-value study: causal HMM ablations and an IWV market proxy.

No brokerage access or model tuning is performed. Run this module with
``.venv/Scripts/python.exe DecisionValueStudy.py --plan value_study_plan.json``.
Every historical evaluation in this study is disclosed as reused research data.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import HyperplaneStock as hs
import SwingRegimeSelector as swing


ROOT = Path(__file__).resolve().parent
EPSILON = 1e-9


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def benchmark_targets(index: pd.DatetimeIndex, tickers: Sequence[str], weight: float) -> pd.DataFrame:
    """Allocate a fixed, previously determined total weight across cash slots."""

    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("Passive allocation must be finite and in [0, 1].")
    if not tickers:
        raise ValueError("At least one ticker is required.")
    return pd.DataFrame(weight / len(tickers), index=index, columns=list(tickers))


def training_exposure(result: swing.SimulationResult) -> float:
    """Training holding-interval average, excluding the liquidated last row."""

    values = result.daily["gross_exposure"].iloc[:-1]
    if values.empty or not np.isfinite(values).all():
        raise ValueError("Training exposure needs finite holding intervals.")
    return min(1.0, max(0.0, float(values.mean())))


def completed_positions(result: swing.SimulationResult, data: hs.MarketData) -> pd.DataFrame:
    """Reconcile contiguous same-direction positions including adds and trims.

    An episode closes only when shares reach zero or reverse direction. Reversal
    fees are split by closing/opening share notional. Borrow is allocated from
    the book's actual daily borrow by each stock's previous-open short notional.
    Cash interest belongs to cash and is excluded from position P&L.
    """

    columns = ["ticker", "entry_date", "exit_date", "direction", "market_pnl",
               "transaction_cost", "borrow_cost", "net_pnl", "entry_notional",
               "total_additions_notional", "holding_sessions", "net_return_on_additions"]
    if result.daily.empty:
        return pd.DataFrame(columns=columns)
    episodes: dict[str, dict[str, Any]] = {}
    completed: list[dict[str, Any]] = []
    prior_shares = {ticker: 0.0 for ticker in data.tickers}
    prior_prices: dict[str, float] | None = None
    trades = result.trades.copy()
    if not trades.empty:
        trades["date"] = pd.to_datetime(trades["date"])
    grouped = {pd.Timestamp(date): frame for date, frame in trades.groupby("date")} if not trades.empty else {}
    for position, (date, day) in enumerate(result.daily.iterrows()):
        date = pd.Timestamp(date)
        opening = data.open.loc[date]
        if prior_prices is not None:
            short_values = {ticker: max(-prior_shares[ticker], 0.0) * prior_prices[ticker]
                            for ticker in data.tickers}
            short_total = sum(short_values.values())
            for ticker, episode in episodes.items():
                episode["market_pnl"] += prior_shares[ticker] * (float(opening[ticker]) - prior_prices[ticker])
                if short_total > EPSILON:
                    episode["borrow_cost"] += float(day.get("borrow_cost", 0.0)) * short_values[ticker] / short_total
        orders = grouped.get(date)
        if orders is not None:
            for order in orders.itertuples(index=False):
                ticker = str(order.ticker)
                before, after = float(order.shares_before), float(order.shares_after)
                fee = float(order.transaction_cost)
                price = float(order.execution_open)
                old_sign = int(np.sign(before)) if abs(before) > EPSILON else 0
                new_sign = int(np.sign(after)) if abs(after) > EPSILON else 0
                close_old = old_sign != 0 and new_sign != old_sign
                if close_old:
                    episode = episodes.pop(ticker)
                    old_fee = fee * abs(before) / (abs(before) + abs(after))
                    episode["transaction_cost"] += old_fee
                    episode["exit_date"] = date
                    episode["holding_sessions"] = position - episode.pop("entry_position")
                    episode["net_pnl"] = episode["market_pnl"] - episode["transaction_cost"] - episode["borrow_cost"]
                    episode["net_return_on_additions"] = episode["net_pnl"] / episode["total_additions_notional"]
                    completed.append(episode)
                    fee -= old_fee
                if new_sign and (old_sign == 0 or close_old):
                    episodes[ticker] = {
                        "ticker": ticker, "entry_date": date, "exit_date": None,
                        "direction": "long" if new_sign > 0 else "short",
                        "market_pnl": 0.0, "transaction_cost": fee, "borrow_cost": 0.0,
                        "entry_notional": abs(after) * price,
                        "total_additions_notional": abs(after) * price,
                        "entry_position": position,
                    }
                elif new_sign:
                    episodes[ticker]["transaction_cost"] += fee
                    episodes[ticker]["total_additions_notional"] += max(abs(after) - abs(before), 0.0) * price
                prior_shares[ticker] = after
        prior_prices = {ticker: float(opening[ticker]) for ticker in data.tickers}
    if episodes:
        raise ValueError("Completed-position attribution requires terminal liquidation.")
    return pd.DataFrame(completed, columns=columns)


def paired_block_bootstrap(
    left: pd.Series, right: pd.Series, block_sessions: int = 20,
    replications: int = 2000, seed: int = 42,
) -> dict[str, Any]:
    """Resample aligned joint dates in circular blocks, retaining paired risk."""

    if left.index.has_duplicates or right.index.has_duplicates:
        raise ValueError("Bootstrap dates must be unique.")
    paired = pd.concat([left.rename("left"), right.rename("right")], axis=1, join="inner").sort_index().dropna()
    if len(paired) < 2 or not np.isfinite(paired.to_numpy()).all():
        raise ValueError("Bootstrap requires at least two common finite returns.")
    if block_sessions < 1 or replications < 2:
        raise ValueError("Bootstrap blocks and replication count must be positive.")
    values = paired.to_numpy(dtype=float)
    if (values <= -1.0).any():
        raise ValueError("Returns must exceed minus one for compounding.")
    count = len(values)
    block = min(int(block_sessions), count)
    random = np.random.default_rng(seed)
    starts = random.integers(0, count, size=(replications, math.ceil(count / block)))
    indices = ((starts[:, :, None] + np.arange(block)) % count).reshape(replications, -1)[:, :count]
    sampled = values[indices]
    cumulative = np.expm1(np.log1p(sampled).sum(axis=1))
    difference = cumulative[:, 0] - cumulative[:, 1]
    daily_difference = (sampled[:, :, 0] - sampled[:, :, 1]).mean(axis=1)
    observed = np.expm1(np.log1p(values).sum(axis=0))
    return {
        "n_common_sessions": count, "block_sessions": block, "replications": replications,
        "seed": seed, "observed_cumulative_excess": float(observed[0] - observed[1]),
        "cumulative_excess_ci": np.quantile(difference, [0.025, 0.975]).tolist(),
        "observed_mean_daily_excess": float((values[:, 0] - values[:, 1]).mean()),
        "mean_daily_excess_ci": np.quantile(daily_difference, [0.025, 0.975]).tolist(),
        "probability_positive_cumulative_excess": float(np.mean(difference > 0.0)),
        "interpretation": "Descriptive paired resampling. probability_positive_cumulative_excess is the fraction of bootstrap replications above zero, not a posterior probability of future profitability. Not independent trials or a research-adjusted significance test.",
    }


def interval_decisions(
    data: hs.MarketData, inputs: swing.SwingInputs, hmm_targets: pd.DataFrame,
    dates: pd.DatetimeIndex, config: swing.SwingConfig, offset: int = 0,
) -> pd.DataFrame:
    """Audit available entry decisions and later same-schedule counterfactuals."""

    rows: list[dict[str, Any]] = []
    fee = (config.commission_bps_per_side + config.slippage_bps_per_side) / 10_000.0
    for position in range(offset, len(dates) - 1, config.rebalance_sessions):
        start, end = dates[position], dates[min(position + config.rebalance_sessions, len(dates) - 1)]
        for ticker in data.tickers:
            if float(inputs.momentum.loc[start, ticker]) <= 0.0:
                continue
            price_ratio = float(data.open.loc[end, ticker] / data.open.loc[start, ticker])
            regime = inputs.regimes.reindex([start]).iloc[0]
            weight = float(hmm_targets.loc[start, ticker])
            rows.append({
                "decision_date": start, "exit_date": end, "ticker": ticker,
                "regime_state": regime.get("regime_state"),
                "regime_probability": regime.get("regime_probability"),
                "momentum_eligible": True, "hmm_allowed": weight > EPSILON,
                "hmm_target_weight": weight,
                "holding_sessions": min(config.rebalance_sessions, len(dates) - 1 - position),
                "hypothetical_gross_return": price_ratio - 1.0,
                "hypothetical_roundtrip_net_return": price_ratio * (1.0 - fee) / (1.0 + fee) - 1.0,
            })
    return pd.DataFrame(rows, columns=[
        "decision_date", "exit_date", "ticker", "regime_state", "regime_probability",
        "momentum_eligible", "hmm_allowed", "hmm_target_weight", "holding_sessions",
        "hypothetical_gross_return", "hypothetical_roundtrip_net_return",
    ])


def write_simulation(folder: Path, result: swing.SimulationResult, data: hs.MarketData) -> dict[str, Any]:
    folder.mkdir(parents=True, exist_ok=True)
    result.daily.to_csv(folder / "daily.csv", index=True)
    result.trades.to_csv(folder / "trades.csv", index=False)
    positions = completed_positions(result, data)
    positions.to_csv(folder / "completed_positions.csv", index=False)
    metrics = dict(result.metrics)
    metrics.update({"completed_positions": len(positions), "distinct_entry_dates": int(positions["entry_date"].nunique()),
                    "participating_stocks": int(positions["ticker"].nunique()),
                    "profitable_positions": int(positions["net_pnl"].gt(0.0).sum()) if len(positions) else 0})
    attributed = float(positions["net_pnl"].sum()) if len(positions) else 0.0
    cash_interest = float(result.daily.get("cash_interest", pd.Series(0.0, index=result.daily.index)).sum())
    first = result.daily.iloc[0]
    initial_equity = float(first["equity"] / (1.0 + first["net_return"]))
    equity_change = float(result.daily["equity"].iloc[-1]) - initial_equity
    metrics["position_pnl"] = attributed
    metrics["attribution_residual"] = equity_change - cash_interest - attributed
    if abs(float(metrics["attribution_residual"])) > 1e-5:
        raise ValueError(f"Unreconciled position attribution at {folder}: {metrics['attribution_residual']}")
    if len(positions):
        grouped = positions.groupby("ticker")["net_pnl"].sum().sort_values(ascending=False)
        metrics["position_pnl_by_ticker"] = grouped.to_dict()
        metrics["largest_absolute_ticker_pnl_share"] = float(grouped.abs().max() / grouped.abs().sum()) if grouped.abs().sum() else 0.0
    else:
        metrics["position_pnl_by_ticker"] = {}
        metrics["largest_absolute_ticker_pnl_share"] = None
    save_json(folder / "metrics.json", metrics)
    return metrics


def _exact_calendar(data: hs.MarketData, dates: pd.DatetimeIndex) -> hs.MarketData:
    missing = dates.difference(data.index)
    if len(missing):
        raise ValueError(f"Benchmark lacks {len(missing)} stock sessions; first missing date {missing[0]}. No price filling allowed.")
    return hs.MarketData(**{name: getattr(data, name).loc[dates] for name in ("open", "high", "low", "close")})


def _load_cash_rates(path: Path, dates: pd.DatetimeIndex) -> pd.Series:
    frame = pd.read_csv(path)
    normalized = {str(column).lower(): column for column in frame.columns}
    date_column = normalized.get("date", frame.columns[0])
    if "annual_rate_decimal" not in normalized:
        raise ValueError("Cash proxy CSV requires annual_rate_decimal, available at each dated open.")
    values = pd.Series(pd.to_numeric(frame[normalized["annual_rate_decimal"]], errors="coerce").to_numpy(),
                       index=pd.to_datetime(frame[date_column])).sort_index()
    if values.index.has_duplicates:
        raise ValueError("Cash proxy dates must be unique.")
    # FRED yields are end-of-day observations. Fill from past observations only,
    # then lag two actual stock sessions to avoid same-day publication/revision
    # assumptions. Prefix known pre-start observations for initial coverage.
    prefix = values.index[values.index < dates[0]][-3:]
    execution_calendar = prefix.union(dates)
    values = values.reindex(values.index.union(execution_calendar)).sort_index().ffill()
    values = values.reindex(execution_calendar).shift(2).reindex(dates)
    if values.isna().any() or not np.isfinite(values).all():
        raise ValueError("Cash proxy does not cover every evaluation date.")
    return values


def _fit_library(
    data: hs.MarketData, inputs: swing.SwingInputs, train_dates: pd.DatetimeIndex,
    config: swing.SwingConfig, momentum_only: bool,
) -> tuple[dict[int, str], list[dict[str, Any]]]:
    if not momentum_only:
        return swing.fit_policy_library(data, inputs, train_dates, config)
    policy = {state: "cash" for state in range(config.hmm_states)}
    diagnostics: list[dict[str, Any]] = []
    lags = max(config.rebalance_sessions, config.hac_lags or 5)
    for state in range(config.hmm_states):
        candidate = dict(policy)
        candidate = {other: "momentum" if other == state else "cash" for other in policy}
        targets = swing.build_targets(inputs, candidate, config)
        result = swing.simulate_portfolio(data, targets, train_dates, config)
        mean, error = swing.hac_mean_standard_error(result.daily["net_return"], lags)
        bound = mean - config.confidence_z * error
        eligible = result.metrics["active_days"] >= config.min_train_days and result.metrics["active_rebalances"] >= config.min_train_rebalances
        if eligible and bound > 0.0:
            policy[state] = "momentum"
        diagnostics.append({"state": state, "expert": "momentum", "mean_net_daily_return": mean,
                            "standard_error": error, "lower_bound": bound, "eligible": eligible,
                            "selected": policy[state] == "momentum", "training_metrics": result.metrics})
    return policy, diagnostics


def _forecast_inputs(data: hs.MarketData) -> pd.DataFrame:
    config = hs.BacktestConfig(tickers=list(data.tickers), start=str(data.index[0].date()),
                               end=str(data.index[-1].date()), lookback=60, model="cointegration",
                               require_stationarity=True, signal_threshold_bps=150.0, use_hmm=False)
    return hs.walk_forward_predictions(data, config)


def _empty_predictions() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "ticker", "predicted_return", "model_valid"])


def _prepare_case(
    name: str, data: hs.MarketData, predictions: pd.DataFrame,
    train_end: pd.Timestamp, config: swing.SwingConfig, folder: Path,
    momentum_only: bool = False,
) -> dict[str, Any]:
    inputs = swing.prepare_inputs(data, predictions, train_end, config)
    first = data.index[100]
    train_dates = data.index[(data.index >= first) & (data.index < train_end)]
    policies, diagnostics = _fit_library(data, inputs, train_dates, config, momentum_only)
    targets = swing.build_targets(inputs, policies, config)
    train = swing.simulate_portfolio(data, targets, train_dates, config)
    allocation = training_exposure(train)
    ablation_policy, ablation_diagnostics = _fit_library(data, inputs, train_dates, config, True)
    ablation_targets = swing.build_targets(inputs, ablation_policy, config)
    frozen = {"name": name, "config": asdict(config), "policies": policies,
              "momentum_only_policies": ablation_policy, "training_start": str(train_dates[0].date()),
              "training_end": str(train_dates[-1].date()), "hmm_training_before": str(train_end.date()),
              "training_exposure_allocation": allocation, "training_diagnostics": diagnostics,
              "momentum_only_training_diagnostics": ablation_diagnostics,
              "selection_used_validation_or_evaluation": False,
              "calibration_cash_yield": 0.0}
    save_json(folder / "policy_lock.json", frozen)
    train_metrics = write_simulation(folder / "train" / "hmm", train, data)
    return {"inputs": inputs, "targets": targets, "ablation_targets": ablation_targets,
            "policy": frozen, "train_metrics": train_metrics, "allocation": allocation,
            "data": data, "momentum_only": momentum_only}


def _score_split(
    case: dict[str, Any], benchmark: hs.MarketData, dates: pd.DatetimeIndex,
    config: swing.SwingConfig, cash_rates: pd.Series, folder: Path, plan: dict[str, Any],
    all_offsets: bool = True,
) -> dict[str, Any]:
    data: hs.MarketData = case["data"]
    inputs: swing.SwingInputs = case["inputs"]
    target: pd.DataFrame = case["targets"]
    ungated = inputs.momentum / len(data.tickers)
    max_probability = inputs.regimes.reindex(data.index)["regime_probability"].fillna(0.0)
    confidence_sized = ungated.mul(max_probability, axis=0)
    passive_config = replace(config, rebalance_sessions=len(dates), hac_lags=None)
    same_gate_iwv = pd.DataFrame({benchmark.tickers[0]: target.abs().sum(axis=1)}, index=data.index)
    strategy_specs = {
        "hmm": (data, target, config, cash_rates),
        "ungated_momentum": (data, ungated, config, cash_rates),
        "confidence_sized_ungated": (data, confidence_sized, config, cash_rates),
        "passive_stocks": (data, benchmark_targets(data.index, data.tickers, 1.0), passive_config, cash_rates),
        "iwv": (benchmark, benchmark_targets(benchmark.index, benchmark.tickers, 1.0), passive_config, cash_rates),
        "train_exposure_stocks": (data, benchmark_targets(data.index, data.tickers, case["allocation"]), passive_config, cash_rates),
        "train_exposure_iwv": (benchmark, benchmark_targets(benchmark.index, benchmark.tickers, case["allocation"]), passive_config, cash_rates),
        "same_gate_iwv": (benchmark, same_gate_iwv, config, cash_rates),
        "cash": (benchmark, benchmark_targets(benchmark.index, benchmark.tickers, 0.0), passive_config, cash_rates),
        "hmm_zero_cash_yield": (data, target, config, None),
        "ungated_zero_cash_yield": (data, ungated, config, None),
    }
    if case["policy"]["policies"] != case["policy"]["momentum_only_policies"]:
        strategy_specs["momentum_only_hmm"] = (data, case["ablation_targets"], config, cash_rates)
    results: dict[str, swing.SimulationResult] = {}
    metrics: dict[str, Any] = {}
    for name, (prices, weights, settings, rates) in strategy_specs.items():
        result = swing.simulate_portfolio(prices, weights, dates, settings, cash_annual_rates=rates)
        results[name] = result
        metrics[name] = write_simulation(folder / name, result, prices)
    paired = {}
    for control in ("ungated_momentum", "confidence_sized_ungated", "iwv", "train_exposure_stocks", "train_exposure_iwv", "same_gate_iwv", "cash"):
        paired["hmm_minus_" + control] = paired_block_bootstrap(
            results["hmm"].daily["net_return"], results[control].daily["net_return"],
            plan["bootstrap_block_sessions"], plan["bootstrap_replications"], plan["bootstrap_seed"])
    decisions = interval_decisions(data, inputs, target, dates, config)
    decisions.to_csv(folder / "decision_audit.csv", index=False)
    decision_summary = []
    if len(decisions):
        for allowed, group in decisions.groupby("hmm_allowed"):
            decision_summary.append({"hmm_allowed": bool(allowed), "stock_intervals": len(group),
                                     "distinct_decision_dates": int(group["decision_date"].nunique()),
                                     "mean_hypothetical_net_return": float(group["hypothetical_roundtrip_net_return"].mean()),
                                     "profitable_fraction": float(group["hypothetical_roundtrip_net_return"].gt(0).mean())})
    offset_metrics = []
    if all_offsets:
        for offset in plan["rebalance_offsets"]:
            row: dict[str, Any] = {"offset": offset}
            for name, prices, weights in (("hmm", data, target), ("ungated_momentum", data, ungated),
                                          ("same_gate_iwv", benchmark, same_gate_iwv)):
                if offset == 0:
                    row[name] = metrics[name]
                else:
                    result = swing.simulate_portfolio(prices, weights, dates, config, rebalance_offset=offset,
                                                       cash_annual_rates=cash_rates)
                    row[name] = write_simulation(folder / "offsets" / f"offset_{offset:02d}" / name, result, prices)
            row["hmm_minus_ungated_return"] = row["hmm"]["net_cumulative_return"] - row["ungated_momentum"]["net_cumulative_return"]
            offset_metrics.append(row)
    record = {"metrics": metrics, "paired_bootstrap": paired, "decision_audit": decision_summary,
              "offsets": offset_metrics, "training_exposure_allocation": case["allocation"]}
    save_json(folder / "summary.json", record)
    print(json.dumps(clean_json({"split": str(folder), "hmm": metrics["hmm"],
                                 "ungated": metrics["ungated_momentum"]["net_cumulative_return"],
                                 "iwv": metrics["iwv"]["net_cumulative_return"]})), flush=True)
    return record


def _roll_summary(records: list[dict[str, Any]], plan: dict[str, Any]) -> dict[str, Any]:
    evaluations = [record["evaluation"] for record in records]
    positive = [record["metrics"]["hmm"]["net_cumulative_return"] > 0 for record in evaluations]
    differences = [record["metrics"]["hmm"]["net_cumulative_return"] - record["metrics"]["ungated_momentum"]["net_cumulative_return"]
                   for record in evaluations]
    offsets = [item for record in evaluations for item in record["offsets"]]
    return {
        "evaluation_years": len(evaluations), "positive_hmm_years": int(sum(positive)),
        "positive_hmm_years_includes_cash_interest": True,
        "positive_trading_pnl_years": int(sum(record["metrics"]["hmm"]["position_pnl"] > 0 for record in evaluations)),
        "hmm_beats_cash_years": int(sum(record["metrics"]["hmm"]["net_cumulative_return"] > record["metrics"]["cash"]["net_cumulative_return"] for record in evaluations)),
        "cash_only_years": int(sum(record["metrics"]["hmm"]["completed_positions"] == 0 for record in evaluations)),
        "hmm_minus_momentum_bootstrap_lower_above_zero_years": int(sum(record["paired_bootstrap"]["hmm_minus_ungated_momentum"]["cumulative_excess_ci"][0] > 0 for record in evaluations)),
        "hmm_beats_momentum_years": int(sum(value > 0 for value in differences)),
        "hmm_beats_iwv_years": int(sum(record["metrics"]["hmm"]["net_cumulative_return"] > record["metrics"]["iwv"]["net_cumulative_return"] for record in evaluations)),
        "completed_positions": int(sum(record["metrics"]["hmm"]["completed_positions"] for record in evaluations)),
        "positive_offset_fraction": float(np.mean([row["hmm"]["net_cumulative_return"] > 0 for row in offsets])),
        "hmm_beats_momentum_offset_fraction": float(np.mean([row["hmm_minus_ungated_return"] > 0 for row in offsets])),
        "yearly_return_difference_hmm_minus_momentum": differences,
        "criteria": plan["success_evidence_reference"],
        "interpretation": "Chronological disjoint evaluation windows; overlapping training windows and offset runs are correlated. No offset is selected.",
    }


def run(plan_path: Path, output_override: str | None = None) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    output = ROOT / (output_override or plan["output_dir"])
    if (output / "protocol.json").exists():
        raise ValueError("Existing study protocol is locked. Use a new output directory for a deliberate repeat.")
    output.mkdir(parents=True, exist_ok=True)
    config = swing.SwingConfig(**plan["configuration"])
    sources = [Path(__file__), ROOT / "SwingRegimeSelector.py", ROOT / "HyperplaneStock.py", plan_path,
               ROOT / plan["breadth_universe_file"]]
    csv_paths = [ROOT / plan["data_dir"] / f"{ticker}.csv" for ticker in plan["tickers"]]
    csv_paths += [ROOT / plan["benchmark_data_dir"] / f"{plan['benchmark_ticker']}.csv", ROOT / plan["cash_rate_file"]]
    universe = json.loads((ROOT / plan["breadth_universe_file"]).read_text(encoding="utf-8"))
    breadth_paths = [ROOT / plan["breadth_data_dir"] / f"{ticker}.csv" for ticker in universe["stocks"]]
    csv_paths += [path for path in breadth_paths if path.exists()]
    missing = [str(path) for path in csv_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Study inputs missing: {missing}")
    save_json(output / "protocol.json", {"plan": plan, "sources": {path.name: sha256(path) for path in sources},
                                        "data": {str(path.relative_to(ROOT)): sha256(path) for path in csv_paths},
                                        "runtime": {"python": sys.version, "dependencies": {
                                            package: version(package) for package in ["numpy", "pandas", "scipy", "hmmlearn", "statsmodels"]}},
                                        "created_before_any_evaluation": True})
    data = hs.load_csv_market_data(plan["tickers"], ROOT / plan["data_dir"], plan["start"], plan["end"])
    benchmark = hs.load_csv_market_data([plan["benchmark_ticker"]], ROOT / plan["benchmark_data_dir"], plan["start"], plan["end"])
    benchmark = _exact_calendar(benchmark, data.index)
    cash_rates = _load_cash_rates(ROOT / plan["cash_rate_file"], data.index)
    predictions = _forecast_inputs(data)
    predictions.to_csv(output / "cached_predictions.csv", index=False)
    validation_start = pd.Timestamp(plan["reference_validation_start"])
    evaluation_start = pd.Timestamp(plan["reference_evaluation_start"])
    reference = _prepare_case("reference_2018_2022", data, predictions, validation_start, config, output / "reference")
    reference_results = {}
    for split, dates in (("validation_2023_2025", data.index[(data.index >= validation_start) & (data.index < evaluation_start)]),
                         ("evaluation_2026", data.index[data.index >= evaluation_start])):
        reference_results[split] = _score_split(reference, benchmark, dates, config, cash_rates, output / "reference" / split, plan)
    rolling = []
    for year in plan["rolling_evaluation_years"]:
        train_start = pd.Timestamp(year=year - 4, month=1, day=1)
        train_end = pd.Timestamp(year=year - 1, month=1, day=1)
        evaluation_start = pd.Timestamp(year=year, month=1, day=1)
        evaluation_end = pd.Timestamp(year=year + 1, month=1, day=1) - pd.Timedelta(days=1)
        local = data.subset(str(train_start.date()), str(evaluation_end.date()))
        local_predictions = predictions[pd.to_datetime(predictions["date"]).between(local.index[0], local.index[-1])]
        folder = output / "rolling" / str(year)
        case = _prepare_case(f"rolling_{year}", local, local_predictions, train_end, config, folder)
        validation_dates = local.index[(local.index >= train_end) & (local.index < evaluation_start)]
        evaluation_dates = local.index[local.index >= evaluation_start]
        validation = _score_split(case, benchmark, validation_dates, config, cash_rates, folder / "validation", plan, all_offsets=False)
        evaluation = _score_split(case, benchmark, evaluation_dates, config, cash_rates, folder / "evaluation", plan)
        rolling.append({"year": year, "policy": case["policy"], "validation": validation, "evaluation": evaluation})
    breadth = _run_breadth(plan, config, benchmark, cash_rates, output)
    report = {"plan": plan, "reference_policy": reference["policy"], "reference": reference_results,
              "rolling": rolling, "rolling_summary": _roll_summary(rolling, plan), "breadth": breadth,
              "limitations": ["IWV is an ETF proxy, not official Russell 3000 total return.",
                              "Previously observed historical data; no pristine holdout claim.",
                              "No offset or period result chooses a model.",
                              "Training exposure controls use a fixed allocation, not matched realized volatility.",
                              "Annual books liquidate before year end and restart at the next available open, omitting cross-year gaps.",
                              "Cash accrual is a Treasury-rate approximation; zero-interest ablations are retained."]}
    save_json(output / "research_report.json", report)
    print(json.dumps(clean_json({"complete": str(output / "research_report.json"), "rolling": report["rolling_summary"],
                                 "breadth_status": breadth.get("status")})), flush=True)
    return report


def _run_breadth(plan: dict[str, Any], config: swing.SwingConfig, benchmark: hs.MarketData,
                cash_rates: pd.Series, output: Path) -> dict[str, Any]:
    universe_path = ROOT / plan["breadth_universe_file"]
    universe = json.loads(universe_path.read_text(encoding="utf-8"))
    tickers = list(universe["stocks"])
    directory = ROOT / plan["breadth_data_dir"]
    missing = [ticker for ticker in tickers if not (directory / f"{ticker}.csv").exists()]
    if missing:
        return {"status": "blocked_missing_input_symbols", "missing": missing, "expected_symbols": len(tickers),
                "hypothesis": plan["breadth"]}
    paths = [directory / f"{ticker}.csv" for ticker in tickers]
    save_json(output / "breadth" / "protocol.json", {"universe_sha256": sha256(universe_path),
                                                   "csv_sha256": {path.name: sha256(path) for path in paths},
                                                   "hypothesis": plan["breadth"], "created_before_breadth_evaluation": True})
    data = hs.load_csv_market_data(tickers, directory, plan["start"], plan["end"])
    # A complete panel is required: otherwise the common join could silently
    # compress execution intervals or drop a newly listed/renamed constituent.
    expected = benchmark.index
    missing_dates = expected.difference(data.index)
    if len(missing_dates):
        return {"status": "blocked_incomplete_common_calendar", "missing_sessions": len(missing_dates),
                "missing_dates": [str(date.date()) for date in missing_dates], "expected_symbols": len(tickers),
                "hypothesis": plan["breadth"]}
    data = _exact_calendar(data, expected)
    validation_start = pd.Timestamp(plan["reference_validation_start"])
    evaluation_start = pd.Timestamp(plan["reference_evaluation_start"])
    case = _prepare_case("breadth_momentum_only_hmm", data, _empty_predictions(), validation_start,
                         config, output / "breadth", momentum_only=True)
    results = {}
    for split, dates in (("validation_2023_2025", data.index[(data.index >= validation_start) & (data.index < evaluation_start)]),
                         ("evaluation_2026", data.index[data.index >= evaluation_start])):
        results[split] = _score_split(case, benchmark, dates, config, cash_rates, output / "breadth" / split, plan)
    return {"status": "evaluated_fixed_distinct_momentum_only_hypothesis", "symbols": len(tickers),
            "hypothesis": plan["breadth"], "policy": case["policy"], "splits": results,
            "selection_bias": "Current user-selected list, not point-in-time Russell 3000 membership; survivor and selection bias."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="value_study_plan.json")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    run(ROOT / args.plan, args.output_dir)
