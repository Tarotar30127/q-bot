"""Causal HMM expert selection with persistent, self-financing portfolios.

This research module leaves the original intraday pipeline unchanged. Signals
known after the previous close are submitted for the next session's open. HMM
parameters and state/expert assignments are fitted before validation and remain
frozen. Each supplied date split starts in cash and liquidates at its last open,
so no training or validation return uses a later split's prices.

The simulator uses fixed shares between scheduled rebalances, adjusted open
prices, explicit cash, traded-notional execution fees, calendar-day short borrow
and paid terminal liquidation. Adjusted prices approximate reinvested corporate
actions; this is not an execution or corporate-action accounting system.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

import HyperplaneStock as hs


EPSILON = 1e-12


@dataclass(frozen=True)
class SwingConfig:
    """Small, explicit research configuration; no return target or forced trade."""

    rebalance_sessions: int = 5
    momentum_lookback: int = 20
    momentum_threshold_bps: float = 50.0
    trend_lookback: int = 0
    mean_reversion_threshold_bps: float = 150.0
    commission_bps_per_side: float = 0.0
    slippage_bps_per_side: float = 5.0
    short_borrow_bps_per_calendar_day: float = 0.5
    hmm_states: int = 2
    hmm_min_probability: float = 0.55
    confidence_z: float = 1.96
    min_train_days: int = 30
    min_train_rebalances: int = 3
    hac_lags: int | None = None
    initial_equity: float = 100_000.0
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.rebalance_sessions < 1:
            raise ValueError("rebalance_sessions must be positive.")
        if self.momentum_lookback < 2:
            raise ValueError("momentum_lookback must be at least two sessions.")
        if self.trend_lookback != 0 and self.trend_lookback < 2:
            raise ValueError("trend_lookback must be zero or at least two sessions.")
        if self.hmm_states < 2:
            raise ValueError("hmm_states must be at least two.")
        if not 0 <= self.hmm_min_probability <= 1:
            raise ValueError("hmm_min_probability must be in [0, 1].")
        if self.min_train_days < 2 or self.min_train_rebalances < 1:
            raise ValueError("Training minima must include two dates and one rebalance.")
        if self.hac_lags is not None and self.hac_lags < self.rebalance_sessions:
            raise ValueError("hac_lags must cover at least one holding interval.")
        for name in (
            "momentum_threshold_bps", "mean_reversion_threshold_bps",
            "commission_bps_per_side", "slippage_bps_per_side",
            "short_borrow_bps_per_calendar_day", "confidence_z",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.commission_bps_per_side + self.slippage_bps_per_side >= 1_000:
            raise ValueError("Combined per-side execution costs must be below 1,000 bps.")
        if not math.isfinite(self.initial_equity) or self.initial_equity <= 0:
            raise ValueError("initial_equity must be finite and positive.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SwingInputs:
    """Daily next-open signals and forward-filtered, frozen-parameter regimes."""

    mean_reversion: pd.DataFrame
    momentum: pd.DataFrame
    regimes: pd.DataFrame


@dataclass
class SimulationResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, object]


def _dates(values: Sequence[pd.Timestamp] | pd.DatetimeIndex) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(values))
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Dates must be unique and sorted in increasing order.")
    if len(dates) < 2:
        raise ValueError("A simulation needs an entry date and a later liquidation date.")
    return dates


def build_expert_signals(
    data: hs.MarketData, baseline_predictions: pd.DataFrame, config: SwingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use only information present before each execution session's open.

    Mean reversion intentionally reuses the previous session's cached forecast
    and validity flag. It is a lagged version of the existing intraday expert,
    not a newly fitted multi-session price forecast. Momentum is long-only and
    uses completed close-to-close returns; optional slow-trend confirmation also
    uses completed closes. An excluded symbol retains its equal-weight cash slot.
    """

    required = {"date", "ticker", "predicted_return", "model_valid"}
    missing = required.difference(baseline_predictions.columns)
    if missing:
        raise ValueError(f"Missing cached prediction columns: {sorted(missing)}")
    predictions = baseline_predictions.copy()
    predictions["date"] = pd.to_datetime(predictions["date"])
    if predictions.duplicated(["date", "ticker"]).any():
        raise ValueError("Cached predictions must be unique by date and ticker.")
    returns = predictions.pivot(index="date", columns="ticker", values="predicted_return")
    validity = predictions.pivot(index="date", columns="ticker", values="model_valid")
    # Reindex before shifting: a missing forecast cannot jump over a market date.
    returns = returns.reindex(index=data.index, columns=data.tickers).apply(
        pd.to_numeric, errors="coerce"
    ).shift(1)
    validity = validity.reindex(index=data.index, columns=data.tickers)
    validity = validity.apply(lambda column: column.map(
        lambda value: value is True or isinstance(value, (bool, np.bool_)) and bool(value)
        or isinstance(value, str) and value.strip().lower() == "true"
        or isinstance(value, (int, float, np.integer, np.floating)) and value == 1
    )).shift(1).fillna(False).astype(bool)
    mean_reversion = np.sign(returns).where(
        validity & returns.abs().ge(config.mean_reversion_threshold_bps / 10_000.0), 0.0
    ).fillna(0.0)
    prior_close = data.close.shift(1)
    momentum_return = prior_close / data.close.shift(config.momentum_lookback + 1) - 1.0
    momentum_active = momentum_return.gt(config.momentum_threshold_bps / 10_000.0)
    if config.trend_lookback:
        slow_average = data.close.rolling(
            config.trend_lookback, min_periods=config.trend_lookback
        ).mean().shift(1)
        momentum_active &= prior_close.gt(slow_average)
    momentum = momentum_active.astype(float).reindex(columns=data.tickers).fillna(0.0)
    return mean_reversion.astype(float), momentum.astype(float)


def prepare_inputs(
    data: hs.MarketData, baseline_predictions: pd.DataFrame,
    validation_start: str | pd.Timestamp, config: SwingConfig,
) -> SwingInputs:
    """Fit on pre-validation features and forward-filter each completed session.

    Training HMM features use only prices strictly before validation_start.
    In-sample training posteriors are model calibration diagnostics, not claimed
    as historical out-of-sample performance. All later probabilities are causal
    forward filters with frozen training normalization and HMM parameters.
    """

    mean_reversion, momentum = build_expert_signals(data, baseline_predictions, config)
    features = hs.build_regime_features(data).drop(columns="opening_gap")
    regimes = hs.fit_asof_hmm_regimes(
        features, training_end=pd.Timestamp(validation_start),
        n_states=config.hmm_states, random_seed=config.random_seed,
        include_probabilities=True,
    )
    return SwingInputs(mean_reversion=mean_reversion, momentum=momentum, regimes=regimes)


def build_targets(
    inputs: SwingInputs, policies: Mapping[int, str], config: SwingConfig,
) -> pd.DataFrame:
    """Create desired daily weights; only scheduled rows execute in simulation.

    At a rebalance, the dominant confident state supplies its frozen expert.
    Posterior confidence sizes exposure and the remainder stays in cash. Between
    rebalances the book keeps its existing shares through state or signal changes.
    """

    columns = inputs.mean_reversion.columns
    dates = inputs.mean_reversion.index
    output = pd.DataFrame(0.0, index=dates, columns=columns)
    regimes = inputs.regimes.reindex(dates)
    for state in range(config.hmm_states):
        expert = policies.get(state, "cash")
        if expert not in {"cash", "mean_reversion", "momentum"}:
            raise ValueError(f"Unsupported expert: {expert}")
        if expert == "cash":
            continue
        probability_column = f"regime_probability_{state}"
        if probability_column not in regimes:
            raise ValueError(f"Missing {probability_column}")
        probability = pd.to_numeric(regimes[probability_column], errors="coerce")
        confident = regimes["regime_state"].eq(state) & probability.ge(
            config.hmm_min_probability
        ) & probability.le(1.0)
        sizing = probability.where(confident, 0.0).fillna(0.0)
        signal = getattr(inputs, expert).reindex(index=dates, columns=columns).fillna(0.0)
        output += signal.mul(sizing, axis=0) / len(columns)
    return output


def _post_cost_target_equity(
    equity_before: float, existing_values: np.ndarray,
    weights: np.ndarray, cost_rate: float,
) -> float:
    """Solve E_after = E_before - fee * sum(abs(w * E_after - old_book))."""

    if cost_rate <= EPSILON:
        return equity_before
    lower, upper = 0.0, equity_before
    # Gross target weight <= 1 and fee < .1 make this equation strictly monotone.
    if equity_before - cost_rate * float(np.abs(existing_values).sum()) <= 0:
        raise ValueError("Portfolio cannot afford liquidation; equity is exhausted.")
    for _ in range(50):
        middle = (lower + upper) / 2.0
        residual = middle + cost_rate * float(
            np.abs(weights * middle - existing_values).sum()
        ) - equity_before
        if residual > 0:
            upper = middle
        else:
            lower = middle
    return (lower + upper) / 2.0


def simulate_portfolio(
    data: hs.MarketData, target_weights: pd.DataFrame,
    dates: Sequence[pd.Timestamp] | pd.DatetimeIndex, config: SwingConfig,
    *, rebalance_offset: int = 0, cash_annual_rates: pd.Series | None = None,
) -> SimulationResult:
    """Execute a single split using fixed shares, cash, and complete accounting.

    There are len(dates)-1 holding intervals. The first open pays entry costs,
    every later open marks the existing shares and accrues calendar-day borrow,
    scheduled rebalances pay costs only on changed notional, and the final open
    liquidates. Desired weight is measured against equity *after* current costs.
    No target or state change moves positions between scheduled rebalances.

    Short borrow uses the prior open's short market value for the elapsed number
    of calendar days. Optional annual cash rates must already be known at each
    open: the prior open's quote accrues simple ACT/365 interest. Only capital
    above a full gross-position collateral reserve earns interest; short-sale
    proceeds cannot manufacture yield. Defaults reproduce the zero-yield model.
    ``rebalance_offset`` shifts the first scheduled entry without dropping dates.
    """

    dates = _dates(dates)
    if isinstance(rebalance_offset, bool) or not isinstance(rebalance_offset, (int, np.integer)):
        raise ValueError("rebalance_offset must be an integer.")
    if not 0 <= rebalance_offset < config.rebalance_sessions:
        raise ValueError("rebalance_offset must be within one rebalance interval.")
    if not dates.isin(data.index).all():
        raise ValueError("Every simulation date must exist in market data.")
    locations = data.index.get_indexer(dates)
    if not np.all(np.diff(locations) == 1):
        raise ValueError("A split must include every intervening market session.")
    tickers = list(data.tickers)
    prices = data.open.reindex(index=dates, columns=tickers).to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Adjusted open prices must be complete, finite, and positive.")
    targets = target_weights.reindex(index=dates, columns=tickers)
    if targets.isna().any().any():
        raise ValueError("Every simulation date and ticker needs an explicit target weight.")
    target_values = targets.to_numpy(dtype=float)
    if not np.isfinite(target_values).all():
        raise ValueError("Target weights must be finite.")
    if (np.abs(target_values).sum(axis=1) > 1.0 + 1e-10).any():
        raise ValueError("Target gross exposure must not exceed one.")
    if cash_annual_rates is None:
        cash_rates = np.zeros(len(dates), dtype=float)
    else:
        if cash_annual_rates.index.has_duplicates:
            raise ValueError("Cash rate dates must be unique.")
        cash_rates = pd.to_numeric(cash_annual_rates.reindex(dates), errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(cash_rates).all() or (cash_rates <= -1.0).any():
            raise ValueError("Every simulation open needs a finite cash rate greater than -100%.")
    cost_rate = (
        config.commission_bps_per_side + config.slippage_bps_per_side
    ) / 10_000.0
    shares = np.zeros(len(tickers), dtype=float)
    cash = float(config.initial_equity)
    previous_equity = float(config.initial_equity)
    previous_prices: np.ndarray | None = None
    previous_short_notional = 0.0
    previous_gross_notional = 0.0
    daily_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    for position, (date, opening_prices) in enumerate(zip(dates, prices)):
        existing_values = shares * opening_prices
        calendar_days = int((date - dates[position - 1]).days) if position else 0
        interestable_capital = max(0.0, min(cash, previous_equity - previous_gross_notional))
        cash_rate = float(cash_rates[position - 1]) if position else 0.0
        cash_interest = interestable_capital * cash_rate * calendar_days / 365.0
        cash += cash_interest
        marked_equity = float(cash + existing_values.sum())
        market_pnl = (
            float(shares @ (opening_prices - previous_prices))
            if previous_prices is not None else 0.0
        )
        borrow_cost = (
            previous_short_notional * calendar_days
            * config.short_borrow_bps_per_calendar_day / 10_000.0
        )
        cash -= borrow_cost
        equity_before = marked_equity - borrow_cost
        if equity_before <= EPSILON:
            raise ValueError(f"Portfolio equity exhausted on {date.date()}.")
        terminal = position == len(dates) - 1
        rebalance = (position >= rebalance_offset
                     and (position - rebalance_offset) % config.rebalance_sessions == 0
                     and not terminal)
        transaction_cost = 0.0
        traded_notional = 0.0
        order_count = 0
        if rebalance or terminal:
            weights = np.zeros(len(tickers)) if terminal else target_values[position]
            target_equity = _post_cost_target_equity(
                equity_before, existing_values, weights, cost_rate
            )
            desired_shares = weights * target_equity / opening_prices
            change = desired_shares - shares
            traded_values = change * opening_prices
            traded_notional = float(np.abs(traded_values).sum())
            transaction_cost = traded_notional * cost_rate
            cash -= float(traded_values.sum()) + transaction_cost
            for ticker_position, ticker in enumerate(tickers):
                if abs(traded_values[ticker_position]) > 1e-8:
                    order_count += 1
                    trade_rows.append({
                        "date": date, "ticker": ticker,
                        "shares_before": float(shares[ticker_position]),
                        "shares_after": float(desired_shares[ticker_position]),
                        "shares_traded": float(change[ticker_position]),
                        "execution_open": float(opening_prices[ticker_position]),
                        "traded_notional": float(abs(traded_values[ticker_position])),
                        "transaction_cost": float(abs(traded_values[ticker_position]) * cost_rate),
                        "terminal_liquidation": terminal,
                    })
            shares = desired_shares
        values_after = shares * opening_prices
        equity_after = float(cash + values_after.sum())
        gross_notional = float(np.abs(values_after).sum())
        short_notional = float((-values_after).clip(min=0.0).sum())
        net_return = equity_after / previous_equity - 1.0
        row: dict[str, object] = {
            "date": date, "equity": equity_after, "cash": float(cash),
            "equity_before_costs": marked_equity,
            "net_return": net_return,
            "gross_return": market_pnl / previous_equity,
            "market_pnl": market_pnl,
            "cash_interest": cash_interest,
            "cash_interest_return": cash_interest / previous_equity,
            "cash_annual_rate": cash_rate,
            "transaction_cost": transaction_cost, "borrow_cost": borrow_cost,
            "transaction_cost_return": transaction_cost / previous_equity,
            "borrow_cost_return": borrow_cost / previous_equity,
            "traded_notional": traded_notional,
            "turnover": traded_notional / equity_before,
            "gross_exposure": gross_notional / equity_after,
            "net_exposure": float(values_after.sum()) / equity_after,
            "short_exposure": short_notional / equity_after,
            "cash_weight": float(cash / equity_after),
            "is_rebalance": rebalance, "terminal_liquidation": terminal,
            "trade_count": order_count,
            "calendar_days_borrowed": calendar_days,
        }
        for ticker_position, ticker in enumerate(tickers):
            row[f"shares_{ticker}"] = float(shares[ticker_position])
            row[f"weight_{ticker}"] = float(values_after[ticker_position] / equity_after)
        daily_rows.append(row)
        previous_equity = equity_after
        previous_prices = opening_prices
        previous_short_notional = short_notional
        previous_gross_notional = gross_notional
    daily = pd.DataFrame(daily_rows).set_index("date")
    trades = pd.DataFrame(trade_rows, columns=[
        "date", "ticker", "shares_before", "shares_after", "shares_traded",
        "execution_open", "traded_notional", "transaction_cost", "terminal_liquidation",
    ])
    holding = daily.iloc[:-1]
    active = holding["gross_exposure"].gt(EPSILON)
    ticker_active = holding[[f"weight_{ticker}" for ticker in tickers]].abs().gt(EPSILON)
    active_rebalances = holding["is_rebalance"] & active
    holding_episodes = active & ~active.shift(1, fill_value=False)
    # Include the initial NAV when assessing entry-day losses and drawdown.
    equity_path = np.r_[config.initial_equity, daily["equity"].to_numpy(dtype=float)]
    drawdown = equity_path / np.maximum.accumulate(equity_path) - 1.0
    net_returns = daily["net_return"].to_numpy(dtype=float)
    volatility = float(np.std(net_returns, ddof=1)) if len(net_returns) > 1 else 0.0
    cumulative = float(daily["equity"].iloc[-1] / config.initial_equity - 1.0)
    calendar_span = max(1, int((dates[-1] - dates[0]).days))
    metrics: dict[str, object] = {
        "start": str(dates[0].date()), "end": str(dates[-1].date()),
        "sessions": len(dates), "holding_intervals": len(dates) - 1,
        "net_cumulative_return": cumulative,
        "annualized_return": float((1.0 + cumulative) ** (365.25 / calendar_span) - 1.0),
        "annualized_volatility": volatility * math.sqrt(252.0),
        "sharpe": float(np.mean(net_returns) / volatility * math.sqrt(252.0))
        if volatility > EPSILON else None,
        "max_drawdown": float(drawdown.min()),
        # Preserve the existing pipeline's symbol/session coverage definition.
        "coverage": float(ticker_active.to_numpy(dtype=float).mean()),
        "portfolio_active_day_fraction": float(active.mean()),
        "active_days": int(active.sum()),
        "active_rebalances": int(active_rebalances.sum()),
        "holding_episodes": int(holding_episodes.sum()),
        "average_gross_exposure": float(holding["gross_exposure"].mean()),
        "average_short_exposure": float(holding["short_exposure"].mean()),
        "total_turnover": float(daily["turnover"].sum()),
        "annualized_turnover": float(daily["turnover"].sum() * 365.25 / calendar_span),
        "trade_count": len(trades),
        "transaction_cost": float(daily["transaction_cost"].sum()),
        "borrow_cost": float(daily["borrow_cost"].sum()),
        "cash_interest": float(daily["cash_interest"].sum()),
        "cash_interest_fraction_initial": float(daily["cash_interest"].sum() / config.initial_equity),
        "transaction_cost_fraction_initial": float(daily["transaction_cost"].sum() / config.initial_equity),
        "borrow_cost_fraction_initial": float(daily["borrow_cost"].sum() / config.initial_equity),
        "market_pnl_fraction_initial": float(daily["market_pnl"].sum() / config.initial_equity),
        "final_equity": float(daily["equity"].iloc[-1]),
        "final_cash": float(daily["cash"].iloc[-1]),
        "terminal_gross_exposure": float(daily["gross_exposure"].iloc[-1]),
    }
    return SimulationResult(daily=daily, trades=trades, metrics=metrics)


def hac_mean_standard_error(values: Sequence[float], lags: int) -> tuple[float, float]:
    """Newey-West standard error of the mean with Bartlett calendar-session lags."""

    observations = np.asarray(values, dtype=float)
    if len(observations) < 2 or not np.isfinite(observations).all():
        raise ValueError("HAC estimation needs at least two finite daily returns.")
    mean = float(observations.mean())
    residual = observations - mean
    lag_count = min(max(0, int(lags)), len(observations) - 1)
    variance_sum = float(residual @ residual)
    for lag in range(1, lag_count + 1):
        variance_sum += 2.0 * (1.0 - lag / (lag_count + 1.0)) * float(
            residual[lag:] @ residual[:-lag]
        )
    return mean, math.sqrt(max(variance_sum, 0.0)) / len(observations)


def fit_policy_library(
    data: hs.MarketData, inputs: SwingInputs,
    train_dates: Sequence[pd.Timestamp] | pd.DatetimeIndex, config: SwingConfig,
) -> tuple[dict[int, str], list[dict[str, object]]]:
    """Select frozen state experts from real, gated training portfolio returns.

    Each state/expert candidate enters only on scheduled dates when that state is
    dominant and confident; holdings persist until the next rebalance. Daily
    returns retain cash days, mark all intervening prices, and pay actual turnover
    and liquidation costs. This avoids attributing an ungated strategy's return
    to whichever regime happens to be visible on its exit date.

    A state keeps cash unless an expert clears the positive HAC lower bound and
    both distinct active calendar dates and active rebalance decisions minima.
    The caller supplies training dates ending strictly before validation.
    """

    dates = _dates(train_dates)
    policies = {state: "cash" for state in range(config.hmm_states)}
    diagnostics: list[dict[str, object]] = []
    lags = max(config.rebalance_sessions, config.hac_lags or 5)
    regimes = inputs.regimes.reindex(dates)
    for state in range(config.hmm_states):
        best_bound = 0.0
        state_diagnostics: list[dict[str, object]] = []
        for expert in ("mean_reversion", "momentum"):
            candidate = {other: "cash" for other in range(config.hmm_states)}
            candidate[state] = expert
            targets = build_targets(inputs, candidate, config)
            result = simulate_portfolio(data, targets, dates, config)
            mean, standard_error = hac_mean_standard_error(result.daily["net_return"], lags)
            bound = mean - config.confidence_z * standard_error
            eligible = (
                int(result.metrics["active_days"]) >= config.min_train_days
                and int(result.metrics["active_rebalances"]) >= config.min_train_rebalances
            )
            if eligible and bound > best_bound:
                policies[state] = expert
                best_bound = bound
            state_diagnostics.append({
                "state": state, "expert": expert,
                "regime_days": int((regimes["regime_state"].eq(state)
                    & regimes["regime_probability"].ge(config.hmm_min_probability)).sum()),
                "active_days": result.metrics["active_days"],
                "active_rebalances": result.metrics["active_rebalances"],
                "holding_episodes": result.metrics["holding_episodes"],
                "training_net_cumulative_return": result.metrics["net_cumulative_return"],
                "mean_net_daily_return": mean, "standard_error": standard_error,
                "lower_bound": bound, "eligible": eligible, "hac_lags": lags,
                "training_transaction_cost_fraction": result.metrics["transaction_cost_fraction_initial"],
                "training_borrow_cost_fraction": result.metrics["borrow_cost_fraction_initial"],
                "training_market_pnl_fraction": result.metrics["market_pnl_fraction_initial"],
                "training_coverage": result.metrics["coverage"],
                "training_total_turnover": result.metrics["total_turnover"],
            })
        for diagnostic in state_diagnostics:
            diagnostic["selected"] = policies[state] == diagnostic["expert"]
        diagnostics.extend(state_diagnostics)
    return policies, diagnostics


def evaluate_selector(
    data: hs.MarketData, inputs: SwingInputs, policies: Mapping[int, str],
    dates: Sequence[pd.Timestamp] | pd.DatetimeIndex, config: SwingConfig,
) -> SimulationResult:
    """Apply an already selected library to one independently liquidated split."""

    return simulate_portfolio(data, build_targets(inputs, policies, config), dates, config)


def evaluate_buy_and_hold(
    data: hs.MarketData, dates: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    config: SwingConfig,
) -> SimulationResult:
    """Equal initial allocation, fixed shares until paid final liquidation."""

    from dataclasses import replace

    dates = _dates(dates)
    targets = pd.DataFrame(1.0 / len(data.tickers), index=dates, columns=data.tickers)
    # HAC controls policy estimation only; they do not constrain a passive book.
    benchmark_config = replace(config, rebalance_sessions=len(dates), hac_lags=None)
    return simulate_portfolio(data, targets, dates, benchmark_config)
