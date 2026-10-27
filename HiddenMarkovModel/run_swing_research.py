"""Run the declared six-design comparison, then score its frozen winner once.

All data come from the existing local research cache. No orders are submitted.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields, replace
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import HyperplaneStock as hs
import SwingRegimeSelector as swing


ROOT = Path(__file__).resolve().parent


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_simulation(folder: Path, result: swing.SimulationResult) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    result.daily.to_csv(folder / "daily.csv", index=True)
    result.trades.to_csv(folder / "trades.csv", index=False)
    save_json(folder / "metrics.json", result.metrics)


def build_config(plan: dict[str, Any], rebalance: int, trend: int) -> swing.SwingConfig:
    supported = {field.name for field in fields(swing.SwingConfig)}
    parameters = {key: value for key, value in plan.items() if key in supported}
    parameters.update(rebalance_sessions=rebalance, trend_lookback=trend)
    if "short_borrow_bps_per_calendar_day" in supported:
        parameters["short_borrow_bps_per_calendar_day"] = plan["short_borrow_bps_per_day"]
    return swing.SwingConfig(**parameters)


def forecast_inputs(data: hs.MarketData, plan: dict[str, Any]) -> pd.DataFrame:
    config = hs.BacktestConfig(
        tickers=plan["tickers"], start=plan["start"], end=plan["end"],
        lookback=60, model="cointegration", require_stationarity=True,
        signal_threshold_bps=plan["mean_reversion_threshold_bps"], use_hmm=False,
    )
    return hs.walk_forward_predictions(data, config)


def run(plan_path: Path, output_override: str | None = None) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    output = ROOT / (output_override or plan["output_dir"])
    if (output / "selection_lock.json").exists():
        raise ValueError("This output already has a selection lock. Use a new --output-dir for a repeat.")
    output.mkdir(parents=True, exist_ok=True)
    data_dir = ROOT / plan["data_dir"]
    sources = [ROOT / "SwingRegimeSelector.py", Path(__file__), plan_path]
    manifest = {
        "source_sha256": {p.name: sha256(p) for p in sources},
        "csv_sha256": {f"{t}.csv": sha256(data_dir / f"{t}.csv") for t in plan["tickers"]},
        "plan": plan,
    }
    save_json(output / "protocol.json", manifest)
    reserved_start = pd.Timestamp(plan["reserved_start"])
    validation_start = pd.Timestamp(plan["validation_start"])
    research_end = str((reserved_start - pd.Timedelta(days=1)).date())
    research_data = hs.load_csv_market_data(plan["tickers"], data_dir, plan["start"], research_end)
    predictions = forecast_inputs(research_data, plan)
    # A shared warm-up prevents different configurations from seeing different periods.
    first_date = research_data.index[max(100, 61)]
    training_dates = research_data.index[(research_data.index >= first_date) & (research_data.index < validation_start)]
    validation_dates = research_data.index[(research_data.index >= validation_start) & (research_data.index < reserved_start)]
    candidates: list[dict[str, Any]] = []
    candidate_objects: dict[str, tuple[swing.SwingConfig, dict[int, str]]] = {}
    for rebalance, trend in itertools.product(plan["rebalance_sessions_grid"], plan["trend_lookback_grid"]):
        name = f"rebalance_{rebalance}_trend_{trend}"
        config = build_config(plan, rebalance, trend)
        inputs = swing.prepare_inputs(research_data, predictions, validation_start, config)
        policies, diagnostics = swing.fit_policy_library(research_data, inputs, training_dates, config)
        train = swing.evaluate_selector(research_data, inputs, policies, training_dates, config)
        validation = swing.evaluate_selector(research_data, inputs, policies, validation_dates, config)
        folder = output / name
        write_simulation(folder / "train", train)
        write_simulation(folder / "validation", validation)
        record = {
            "name": name, "config": asdict(config), "policies": policies,
            "training_diagnostics": diagnostics, "train": train.metrics,
            "validation": validation.metrics,
        }
        record["eligible"] = (
            float(validation.metrics["net_cumulative_return"]) > 0.0
            and float(validation.metrics["coverage"]) >= plan["minimum_validation_coverage"]
            and int(validation.metrics["trade_count"]) > 0
        )
        save_json(folder / "summary.json", record)
        candidates.append(record)
        candidate_objects[name] = config, policies
        print(json.dumps(clean_json({"candidate": name, "policies": policies,
                                    "validation": validation.metrics,
                                    "eligible": record["eligible"]})), flush=True)
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    winner = max(eligible, key=lambda c: (
        c["validation"]["net_cumulative_return"],
        c["validation"].get("sharpe") if c["validation"].get("sharpe") is not None else -math.inf,
        -c["validation"].get("total_turnover", math.inf),
    )) if eligible else None
    lock = {"selected": winner["name"] if winner else "no_trade",
            "selection_used_reserved_returns": False,
            "selection_rule": plan["selection_rule"],
            "source_sha256": manifest["source_sha256"],
            "candidate_validation": [{"name": c["name"], "eligible": c["eligible"],
                                      "metrics": c["validation"]} for c in candidates]}
    save_json(output / "selection_lock.json", lock)
    report: dict[str, Any] = {
        "plan": plan, "candidates": candidates, "selection": lock,
        "reserved_evaluation": {"status": "skipped_no_eligible_candidate"},
    }
    # The benchmark is context and never participates in model selection.
    benchmark_config = build_config(plan, len(validation_dates), 0)
    benchmark = swing.evaluate_buy_and_hold(research_data, validation_dates, benchmark_config)
    write_simulation(output / "validation_buy_and_hold", benchmark)
    report["validation_buy_and_hold"] = benchmark.metrics
    if winner:
        config, policies = candidate_objects[winner["name"]]
        all_data = hs.load_csv_market_data(plan["tickers"], data_dir, plan["start"], plan["end"])
        all_predictions = forecast_inputs(all_data, plan)
        all_inputs = swing.prepare_inputs(all_data, all_predictions, validation_start, config)
        test_dates = all_data.index[all_data.index >= reserved_start]
        reserved = swing.evaluate_selector(all_data, all_inputs, policies, test_dates, config)
        write_simulation(output / "reserved_selected", reserved)
        report["reserved_evaluation"] = {
            "status": "evaluated_once_after_selection_lock", "model": winner["name"],
            "metrics": reserved.metrics,
            "positive_with_trades": reserved.metrics["net_cumulative_return"] > 0 and reserved.metrics["trade_count"] > 0,
            "period_caveat": plan["period_caveat"],
        }
        # Cost scenarios describe the frozen design; they cannot choose a new winner.
        report["reserved_cost_sensitivity"] = []
        for slippage in [0.0, 1.0, 2.0, 5.0, 10.0]:
            scenario_config = replace(config, slippage_bps_per_side=slippage)
            scenario = swing.evaluate_selector(all_data, all_inputs, policies, test_dates, scenario_config)
            report["reserved_cost_sensitivity"].append({"slippage_bps_per_side": slippage, **scenario.metrics})
        benchmark = swing.evaluate_buy_and_hold(all_data, test_dates, config)
        write_simulation(output / "reserved_buy_and_hold", benchmark)
        report["reserved_buy_and_hold"] = benchmark.metrics
    save_json(output / "research_report.json", report)
    print(json.dumps(clean_json({"selected": lock["selected"], "reserved": report["reserved_evaluation"],
                                "report": str(output / "research_report.json")})), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="swing_research_plan.json")
    parser.add_argument("--output-dir", default=None)
    arguments = parser.parse_args()
    run(ROOT / arguments.plan, arguments.output_dir)
