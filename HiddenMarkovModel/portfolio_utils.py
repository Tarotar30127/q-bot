"""Shared portfolio input/output, position attribution, and Treasury-rate handling."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

import market_data as hs
import SwingRegimeSelector as swing

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
    *, execution_prices: pd.DataFrame | None = None,
) -> tuple[dict[int, str], list[dict[str, Any]]]:
    if not momentum_only:
        return swing.fit_policy_library(data, inputs, train_dates, config, execution_prices=execution_prices)
    policy = {state: "cash" for state in range(config.hmm_states)}
    diagnostics: list[dict[str, Any]] = []
    lags = max(config.rebalance_sessions, config.hac_lags or 5)
    for state in range(config.hmm_states):
        candidate = dict(policy)
        candidate = {other: "momentum" if other == state else "cash" for other in policy}
        targets = swing.build_targets(inputs, candidate, config)
        result = swing.simulate_portfolio(data, targets, train_dates, config, execution_prices=execution_prices)
        mean, error = swing.hac_mean_standard_error(result.daily["net_return"], lags)
        bound = mean - config.confidence_z * error
        eligible = result.metrics["active_days"] >= config.min_train_days and result.metrics["active_rebalances"] >= config.min_train_rebalances
        if eligible and bound > 0.0:
            policy[state] = "momentum"
        diagnostics.append({"state": state, "expert": "momentum", "mean_net_daily_return": mean,
                            "standard_error": error, "lower_bound": bound, "eligible": eligible,
                            "selected": policy[state] == "momentum", "training_metrics": result.metrics})
    return policy, diagnostics


def _empty_predictions() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "ticker", "predicted_return", "model_valid"])
