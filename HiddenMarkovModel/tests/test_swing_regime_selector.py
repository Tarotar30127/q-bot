"""Persistent-position accounting and information-boundary acceptance tests."""

import json

import numpy as np
import pandas as pd
import pytest

import HyperplaneStock as hs
import SwingRegimeSelector as swing
from test_hyperplane_stock import synthetic_market_data


def market_data(prices, dates=None):
    """Use explicit opening prices so expected fixed-share P&L is transparent."""
    if dates is None:
        dates = pd.bdate_range("2020-01-02", periods=len(next(iter(prices.values()))))
    frame = pd.DataFrame(prices, index=pd.DatetimeIndex(dates), dtype=float)
    return hs.MarketData(open=frame.copy(), high=frame.copy(),
                         low=frame.copy(), close=frame.copy())


def config(**overrides):
    values = dict(initial_equity=1.0, rebalance_sessions=5,
                  slippage_bps_per_side=0.0, commission_bps_per_side=0.0,
                  short_borrow_bps_per_calendar_day=0.0)
    values.update(overrides)
    return swing.SwingConfig(**values)


def targets(data, weights):
    return pd.DataFrame([weights] * len(data.index), index=data.index,
                        columns=data.tickers, dtype=float)


def test_held_positions_drift_with_fixed_shares_between_rebalances():
    data = market_data({"AAA": [100, 110, 110], "BBB": [100, 100, 110]})
    result = swing.simulate_portfolio(
        data, targets(data, [0.5, -0.5]), data.index, config()
    )
    # Buy 0.005 AAA shares and short 0.005 BBB shares.  The second day's
    # gain of $0.05 is lost on day three, without a daily weight reset.
    assert result.metrics["net_cumulative_return"] == pytest.approx(0.0)
    np.testing.assert_allclose(result.daily["equity"], [1.0, 1.05, 1.0])
    assert result.daily["net_return"].iloc[2] == pytest.approx(-0.05 / 1.05)


@pytest.mark.parametrize("commission,slippage", [(0.0, 10.0), (3.0, 7.0)])
def test_entry_and_terminal_exit_are_charged_on_actual_traded_notional(
    commission, slippage
):
    data = market_data({"AAA": [100, 110, 120]})
    result = swing.simulate_portfolio(
        data, targets(data, [1.0]), data.index,
        config(commission_bps_per_side=commission, slippage_bps_per_side=slippage),
    )
    rate = (commission + slippage) / 10000.0
    # Entry notional equals equity after paying its own transaction cost.
    # Closing at 120 costs 1.2 times the entry notional times the fee rate.
    expected = 1.2 * (1.0 - rate) / (1.0 + rate)
    assert result.metrics["net_cumulative_return"] == pytest.approx(expected - 1.0)
    assert result.daily["equity"].iloc[0] == pytest.approx(1.0 / (1.0 + rate))
    assert result.daily["equity"].iloc[-1] == pytest.approx(expected)
    assert result.metrics["trade_count"] == 2
    # Turnover divides each trade by the NAV immediately before that trade.
    assert result.metrics["total_turnover"] == pytest.approx(1.0 + 1.0 / (1.0 + rate))


def test_flat_price_position_held_unchanged_does_not_pay_daily_round_trips():
    data = market_data({"AAA": [100] * 9})
    result = swing.simulate_portfolio(
        data, targets(data, [1.0]), data.index,
        config(rebalance_sessions=1, slippage_bps_per_side=10),
    )
    rate = 0.001
    assert result.metrics["net_cumulative_return"] == pytest.approx(
        (1.0 - rate) / (1.0 + rate) - 1.0
    )
    assert result.metrics["trade_count"] == 2


def test_signal_changes_between_scheduled_dates_keep_existing_shares():
    data = market_data({"AAA": [100, 110, 120, 130]})
    weights = pd.DataFrame({"AAA": [1.0, -1.0, -1.0, -1.0]}, index=data.index)
    result = swing.simulate_portfolio(data, weights, data.index, config())
    assert result.metrics["net_cumulative_return"] == pytest.approx(0.3)
    np.testing.assert_allclose(result.daily["shares_AAA"].iloc[:-1], [0.01] * 3)
    assert result.daily["shares_AAA"].iloc[-1] == 0.0
    assert list(result.trades["date"]) == [data.index[0], data.index[-1]]


def test_reversal_pays_for_both_closed_long_and_new_short():
    data = market_data({"AAA": [100, 100, 100]})
    weights = pd.DataFrame({"AAA": [1.0, -1.0, 0.0]}, index=data.index)
    result = swing.simulate_portfolio(
        data, weights, data.index,
        config(rebalance_sessions=1, slippage_bps_per_side=10),
    )
    rate = 0.001
    expected = ((1.0 - rate) / (1.0 + rate)) ** 2
    assert result.metrics["net_cumulative_return"] == pytest.approx(expected - 1.0)
    assert result.metrics["trade_count"] == 3


def test_short_borrow_uses_prior_short_value_and_actual_calendar_days():
    dates = pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"])
    data = market_data({"AAA": [100, 110, 90]}, dates)
    result = swing.simulate_portfolio(
        data, targets(data, [-1.0]), data.index,
        config(short_borrow_bps_per_calendar_day=1.0),
    )
    # Fixed short shares earn $0.10.  Borrow costs $1 * 3 days * 1 bp,
    # then $1.10 * 1 day * 1 bp.  Weekend accrual is material here.
    assert result.metrics["net_cumulative_return"] == pytest.approx(
        0.10 - 0.0003 - 0.00011
    )
    np.testing.assert_allclose(result.daily["equity"], [1.0, 0.8997, 1.09959])
    np.testing.assert_allclose(
        result.daily["cash"] + result.daily["shares_AAA"] * data.open["AAA"],
        result.daily["equity"],
    )


def test_each_split_liquidates_at_its_last_open_without_next_split_price():
    data = market_data({"AAA": [100, 110, 120, 1000, 2000, 3000]})
    train_dates = data.index[:3]
    first = swing.simulate_portfolio(data, targets(data, [1.0]), train_dates, config())
    changed = market_data({"AAA": [100, 110, 120, 1, 0.5, 0.25]}, data.index)
    repeated = swing.simulate_portfolio(
        changed, targets(changed, [1.0]), train_dates, config()
    )
    assert first.metrics["net_cumulative_return"] == pytest.approx(0.2)
    pd.testing.assert_frame_equal(first.daily, repeated.daily)
    pd.testing.assert_frame_equal(first.trades, repeated.trades)
    assert first.daily.index.max() <= train_dates[-1]
    assert first.trades["date"].max() <= train_dates[-1]


def test_missing_sessions_cannot_be_silently_skipped_by_holding_interval():
    data = market_data({"AAA": [100, 110, 120, 130]})
    with pytest.raises(ValueError, match="intervening market session"):
        swing.simulate_portfolio(data, targets(data, [1.0]), data.index[[0, 2, 3]], config())


def cached_forecasts(data):
    return pd.DataFrame([
        {"date": date, "ticker": ticker, "predicted_return": 0.02,
         "model_valid": True, "actual_return": 0.01}
        for date in data.index for ticker in data.tickers
    ])


def test_signals_use_prior_session_and_missing_forecast_does_not_skip_a_day():
    data = market_data({"AAA": [100, 101, 102, 103, 104, 105]})
    forecasts = cached_forecasts(data)
    forecasts.loc[forecasts["date"].eq(data.index[2]), "predicted_return"] = -0.02
    forecasts = forecasts.loc[forecasts["date"].ne(data.index[3])]
    mr, momentum = swing.build_expert_signals(
        data, forecasts, config(momentum_lookback=2, momentum_threshold_bps=0.0)
    )
    np.testing.assert_allclose(mr["AAA"], [0, 1, 1, -1, 0, 1])
    np.testing.assert_allclose(momentum["AAA"], [0, 0, 0, 1, 1, 1])


def test_current_and_future_prices_do_not_change_earlier_targets_or_regimes():
    data = synthetic_market_data(185, seed=81)
    forecasts = cached_forecasts(data)
    settings = config(momentum_lookback=10)
    validation_start, cutoff = data.index[120], data.index[155]
    original = swing.prepare_inputs(data, forecasts, validation_start, settings)
    altered_prices = []
    for frame in (data.open, data.high, data.low, data.close):
        changed = frame.copy()
        changed.loc[changed.index >= cutoff] *= np.array([1.5, 0.6, 1.1])
        altered_prices.append(changed)
    changed = hs.MarketData(*altered_prices)
    changed_forecasts = forecasts.copy()
    future = changed_forecasts["date"].ge(cutoff)
    changed_forecasts.loc[future, "predicted_return"] = -0.02
    changed_forecasts.loc[future, "actual_return"] = -100.0
    repeated = swing.prepare_inputs(changed, changed_forecasts, validation_start, settings)
    pd.testing.assert_frame_equal(
        original.mean_reversion.loc[:cutoff], repeated.mean_reversion.loc[:cutoff]
    )
    pd.testing.assert_frame_equal(original.momentum.loc[:cutoff], repeated.momentum.loc[:cutoff])
    pd.testing.assert_frame_equal(original.regimes.loc[:cutoff], repeated.regimes.loc[:cutoff])
    policies = {0: "mean_reversion", 1: "momentum"}
    pd.testing.assert_frame_equal(
        swing.build_targets(original, policies, settings).loc[:cutoff],
        swing.build_targets(repeated, policies, settings).loc[:cutoff],
    )


def training_fixture(days=75):
    prices = 100.0 * 1.01 ** np.arange(days)
    data = market_data({"AAA": prices, "BBB": prices, "CCC": prices})
    positive = pd.DataFrame(1.0, index=data.index, columns=data.tickers)
    negative = -positive
    regimes = pd.DataFrame({
        "regime_state": 0, "regime_probability": 1.0,
        "regime_probability_0": 1.0, "regime_probability_1": 0.0,
    }, index=data.index)
    return data, swing.SwingInputs(positive, negative, regimes)


def test_policy_library_ignores_outcomes_and_signals_beyond_training_dates():
    data, inputs = training_fixture()
    train_dates = data.index[:50]
    settings = config(min_train_days=20, min_train_rebalances=3)
    policies, diagnostics = swing.fit_policy_library(data, inputs, train_dates, settings)
    assert policies[0] == "mean_reversion"
    assert policies[1] == "cash"

    changed_prices = data.open.copy()
    changed_prices.loc[data.index[50:]] = np.linspace(10.0, 1.0, 25)[:, None]
    changed = hs.MarketData(open=changed_prices, high=changed_prices,
                           low=changed_prices, close=changed_prices)
    changed_mr = inputs.mean_reversion.copy()
    changed_momentum = inputs.momentum.copy()
    changed_regimes = inputs.regimes.copy()
    changed_mr.loc[data.index[50:]] *= -1.0
    changed_momentum.loc[data.index[50:]] *= -1.0
    changed_regimes.loc[data.index[50:], "regime_state"] = 1
    altered_inputs = swing.SwingInputs(changed_mr, changed_momentum, changed_regimes)
    repeated_policies, repeated_diagnostics = swing.fit_policy_library(
        changed, altered_inputs, train_dates, settings
    )
    assert repeated_policies == policies
    assert json.dumps(repeated_diagnostics, sort_keys=True) == json.dumps(
        diagnostics, sort_keys=True
    )


def test_positive_average_can_still_fail_conservative_serial_correlation_bound():
    returns = np.r_[np.repeat(0.02, 20), np.repeat(-0.005, 20)]
    prices = 100.0 * np.r_[1.0, np.cumprod(1.0 + returns)]
    data = market_data({"AAA": prices})
    signals = pd.DataFrame(1.0, index=data.index, columns=data.tickers)
    regimes = pd.DataFrame({
        "regime_state": 0, "regime_probability": 1.0,
        "regime_probability_0": 1.0, "regime_probability_1": 0.0,
    }, index=data.index)
    inputs = swing.SwingInputs(signals, signals, regimes)
    policies, diagnostics = swing.fit_policy_library(
        data, inputs, data.index, config(min_train_days=30)
    )
    state_zero = [row for row in diagnostics if row["state"] == 0]
    assert all(row["eligible"] for row in state_zero)
    assert all(row["mean_net_daily_return"] > 0 for row in state_zero)
    assert all(row["lower_bound"] < 0 for row in state_zero)
    assert policies[0] == "cash"


@pytest.mark.parametrize("reason", ["negative_returns", "insufficient_days", "uncertain_state"])
def test_policy_library_can_legitimately_hold_cash(reason):
    data, inputs = training_fixture()
    settings = config(min_train_days=20, min_train_rebalances=3)
    if reason == "negative_returns":
        inputs = swing.SwingInputs(-inputs.mean_reversion, inputs.momentum, inputs.regimes)
    elif reason == "insufficient_days":
        settings = config(min_train_days=100, min_train_rebalances=3)
    else:
        uncertain = inputs.regimes.copy()
        uncertain["regime_probability"] = 0.5
        uncertain[["regime_probability_0", "regime_probability_1"]] = 0.5
        inputs = swing.SwingInputs(inputs.mean_reversion, inputs.momentum, uncertain)
    policies, _ = swing.fit_policy_library(data, inputs, data.index, settings)
    assert set(policies.values()) == {"cash"}
    result = swing.evaluate_selector(data, inputs, policies, data.index, settings)
    assert result.metrics["trade_count"] == 0
    assert result.metrics["net_cumulative_return"] == pytest.approx(0.0)
    assert result.metrics["coverage"] == 0.0
