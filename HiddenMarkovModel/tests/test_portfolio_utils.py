"""Cash, execution, and completed-position accounting checks for current portfolios."""
import numpy as np
import pandas as pd
import pytest

import market_data as hs
import SwingRegimeSelector as swing
import portfolio_utils as study


def market(prices, dates=None):
    if dates is None:
        dates = pd.bdate_range("2024-01-02", periods=len(next(iter(prices.values()))))
    frame = pd.DataFrame(prices, index=pd.DatetimeIndex(dates), dtype=float)
    return hs.MarketData(open=frame.copy(), high=frame.copy(),
                         low=frame.copy(), close=frame.copy())


def config(**changes):
    values = dict(initial_equity=10_000.0, rebalance_sessions=3,
                  commission_bps_per_side=0.0, slippage_bps_per_side=0.0,
                  short_borrow_bps_per_calendar_day=0.0)
    values.update(changes)
    return swing.SwingConfig(**values)


def weights(data, desired):
    return pd.DataFrame([desired] * len(data.index), index=data.index,
                        columns=data.tickers, dtype=float)


def test_rebalance_offset_starts_in_cash_and_respects_declared_calendar():
    data = market({"AAA": [100.0] * 9})
    settings = config(slippage_bps_per_side=10.0)
    result = swing.simulate_portfolio(data, weights(data, [1.0]), data.index,
                                      settings, rebalance_offset=2)
    np.testing.assert_allclose(result.daily["shares_AAA"].iloc[:2], 0.0)
    assert list(result.daily.index[result.daily["is_rebalance"]]) == list(data.index[[2, 5]])
    assert list(result.trades["date"]) == list(data.index[[2, 8]])
    expected_final = settings.initial_equity * (1.0 - 0.001) / (1.0 + 0.001)
    assert result.daily["equity"].iloc[-1] == pytest.approx(expected_final)
    assert result.daily["transaction_cost"].sum() == pytest.approx(
        result.trades["transaction_cost"].sum())


def test_offset_does_not_open_a_new_position_on_terminal_liquidation_date():
    data = market({"AAA": [100.0] * 3})
    result = swing.simulate_portfolio(data, weights(data, [1.0]), data.index,
                                      config(), rebalance_offset=2)
    assert result.trades.empty
    assert result.daily["shares_AAA"].eq(0.0).all()
    assert result.metrics["net_cumulative_return"] == 0.0


@pytest.mark.parametrize("offset", [-1, 3, 1.5])
def test_rebalance_offset_requires_a_valid_integer_schedule(offset):
    data = market({"AAA": [100.0] * 5})
    with pytest.raises(ValueError, match="offset"):
        swing.simulate_portfolio(data, weights(data, [1.0]), data.index,
                                 config(), rebalance_offset=offset)


def test_default_cash_and_rebalance_options_preserve_existing_simulations():
    data = market({"AAA": [100.0, 102.0, 101.0, 105.0]})
    desired = weights(data, [0.5])
    implicit = swing.simulate_portfolio(data, desired, data.index, config())
    explicit = swing.simulate_portfolio(data, desired, data.index, config(),
                                        rebalance_offset=0, cash_annual_rates=None)
    pd.testing.assert_frame_equal(implicit.daily, explicit.daily)
    pd.testing.assert_frame_equal(implicit.trades, explicit.trades)
    assert implicit.metrics == explicit.metrics


def test_cash_rate_at_interval_exit_cannot_earn_interest_retroactively():
    data = market({"AAA": [100.0] * 5})
    desired = weights(data, [0.0])
    zero = pd.Series(0.0, index=data.index)
    surprise = zero.copy()
    surprise.iloc[-1] = 0.9
    original = swing.simulate_portfolio(data, desired, data.index, config(),
                                        cash_annual_rates=zero)
    repeated = swing.simulate_portfolio(data, desired, data.index, config(),
                                        cash_annual_rates=surprise)
    pd.testing.assert_frame_equal(original.daily, repeated.daily)


def test_all_cash_compounds_calendar_day_interest_consistently_with_net_returns():
    dates = pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"])
    data = market({"AAA": [100.0] * 3}, dates)
    rates = pd.Series(0.365, index=data.index)
    settings = config()
    result = swing.simulate_portfolio(data, weights(data, [0.0]), data.index,
                                      settings, cash_annual_rates=rates)
    expected = settings.initial_equity * (1.0 + 0.003) * (1.0 + 0.001)
    assert result.metrics["final_equity"] == pytest.approx(expected)
    assert result.metrics["cash_interest"] == pytest.approx(expected - settings.initial_equity)
    assert np.prod(1.0 + result.daily["net_return"]) == pytest.approx(expected / settings.initial_equity)
    assert result.trades.empty


@pytest.mark.parametrize("desired", [[0.5], [-0.5]])
def test_cash_interest_uses_only_uncommitted_capital_and_excludes_short_proceeds(desired):
    dates = pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"])
    data = market({"AAA": [100.0] * 3}, dates)
    rates = pd.Series(0.365, index=data.index)
    result = swing.simulate_portfolio(data, weights(data, desired), data.index,
                                      config(), cash_annual_rates=rates)
    # $5,000 idle equity earns $15 over the weekend, then $5.015 Tuesday.
    # Short sale receipts cannot make the credited interest larger.
    assert result.daily["cash_interest"].sum() == pytest.approx(20.015)
    assert result.metrics["final_equity"] == pytest.approx(10_020.015)


def test_fully_committed_long_short_book_does_not_earn_interest_on_collateral():
    data = market({"AAA": [100.0] * 5, "BBB": [100.0] * 5})
    rates = pd.Series(0.365, index=data.index)
    result = swing.simulate_portfolio(data, weights(data, [0.5, -0.5]), data.index,
                                      config(), cash_annual_rates=rates)
    assert result.daily["cash_interest"].eq(0.0).all()
    assert result.metrics["final_equity"] == pytest.approx(10_000.0)


def test_cash_interest_and_market_profit_reconcile_to_final_equity_after_costs():
    data = market({"AAA": [100.0, 110.0, 90.0, 95.0, 97.0],
                   "BBB": [100.0, 90.0, 105.0, 95.0, 93.0]})
    rates = pd.Series(0.05, index=data.index)
    desired = pd.DataFrame({"AAA": [0.2, 0.2, -0.3, -0.3, 0.0],
                            "BBB": [-0.2, -0.2, 0.2, 0.2, 0.0]}, index=data.index)
    settings = config(rebalance_sessions=2, slippage_bps_per_side=7.0,
                      commission_bps_per_side=2.0,
                      short_borrow_bps_per_calendar_day=0.5)
    result = swing.simulate_portfolio(data, desired, data.index, settings,
                                      cash_annual_rates=rates)
    expected_profit = (result.daily["market_pnl"] + result.daily["cash_interest"]
                       - result.daily["transaction_cost"] - result.daily["borrow_cost"]).sum()
    assert result.metrics["final_equity"] - settings.initial_equity == pytest.approx(expected_profit)
    assert result.trades["transaction_cost"].sum() == pytest.approx(result.daily["transaction_cost"].sum())
    positions = study.completed_positions(result, data)
    assert positions["net_pnl"].sum() + result.daily["cash_interest"].sum() == pytest.approx(expected_profit)


def test_completed_positions_reconstruct_partial_rebalances_without_counting_orders():
    data = market({"AAA": [100.0] * 5})
    desired = pd.DataFrame({"AAA": [0.4, 0.8, 0.2, 0.6, 0.0]}, index=data.index)
    settings = config(rebalance_sessions=1, slippage_bps_per_side=10.0)
    result = swing.simulate_portfolio(data, desired, data.index, settings)
    positions = study.completed_positions(result, data)
    assert len(result.trades) == 5
    assert len(positions) == 1
    position = positions.iloc[0]
    assert position["entry_date"] == data.index[0]
    assert position["exit_date"] == data.index[-1]
    assert position["market_pnl"] == pytest.approx(0.0)
    assert position["transaction_cost"] == pytest.approx(result.trades["transaction_cost"].sum())
    assert position["net_pnl"] == pytest.approx(result.metrics["final_equity"] - settings.initial_equity)
    assert position["holding_sessions"] == len(data.index) - 1


def test_completed_position_reversals_split_costs_and_reconcile_short_borrow():
    dates = pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"])
    data = market({"AAA": [100.0, 110.0, 90.0, 95.0]}, dates)
    desired = pd.DataFrame({"AAA": [0.7, -0.7, -0.4, 0.0]}, index=data.index)
    settings = config(rebalance_sessions=1, slippage_bps_per_side=10.0,
                      short_borrow_bps_per_calendar_day=2.0)
    result = swing.simulate_portfolio(data, desired, data.index, settings)
    positions = study.completed_positions(result, data)
    assert len(positions) == 2
    assert list(positions["direction"]) == ["long", "short"]
    assert list(positions["entry_date"]) == list(data.index[:2])
    assert list(positions["exit_date"]) == list(data.index[[1, 3]])
    assert positions["transaction_cost"].sum() == pytest.approx(result.daily["transaction_cost"].sum())
    assert positions["borrow_cost"].sum() == pytest.approx(result.daily["borrow_cost"].sum())
    assert positions["market_pnl"].sum() == pytest.approx(result.daily["market_pnl"].sum())
    assert positions["net_pnl"].sum() == pytest.approx(result.metrics["final_equity"] - settings.initial_equity)
    assert positions.iloc[0]["borrow_cost"] == pytest.approx(0.0)


def test_completed_positions_keep_overlapping_symbols_as_distinct_episodes():
    data = market({"AAA": [100.0, 105.0, 110.0, 120.0],
                   "BBB": [100.0, 90.0, 80.0, 70.0]})
    desired = weights(data, [0.5, 0.5])
    settings = config(rebalance_sessions=10, slippage_bps_per_side=5.0)
    result = swing.simulate_portfolio(data, desired, data.index, settings)
    positions = study.completed_positions(result, data)
    assert len(positions) == 2
    assert set(positions["ticker"]) == {"AAA", "BBB"}
    assert (positions["net_pnl"] > 0).sum() == 1
    assert (positions["net_pnl"] < 0).sum() == 1
    assert positions["entry_date"].nunique() == 1
    assert positions["net_pnl"].sum() == pytest.approx(result.metrics["final_equity"] - settings.initial_equity)


def test_passive_benchmark_targets_do_not_scale_with_number_of_symbols():
    dates = pd.bdate_range("2024-01-02", periods=3)
    target = study.benchmark_targets(dates, ["AAA", "BBB", "CCC"], 0.18)
    np.testing.assert_allclose(target.sum(axis=1), 0.18)
    np.testing.assert_allclose(target.to_numpy(), 0.06)


def test_simulation_writer_attributes_profit_for_configured_initial_capital(tmp_path):
    data = market({"AAA": [100.0, 105.0, 110.0]})
    settings = config(slippage_bps_per_side=5.0)
    result = swing.simulate_portfolio(data, weights(data, [0.5]), data.index, settings)
    metrics = study.write_simulation(tmp_path / "small_portfolio", result, data)
    assert metrics["attribution_residual"] == pytest.approx(0.0, abs=1e-9)


def test_cash_rate_loader_lags_observation_quotes_two_actual_sessions(tmp_path):
    dates = pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09",
                            "2024-01-10", "2024-01-11"])
    raw_dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04",
                                "2024-01-05", "2024-01-08", "2024-01-09",
                                "2024-01-10", "2024-01-11"])
    raw = pd.DataFrame({"date": raw_dates, "annual_rate_decimal": np.arange(1, 9) / 100.0})
    source = tmp_path / "rates.csv"
    raw.to_csv(source, index=False)
    result = study._load_cash_rates(source, dates)
    np.testing.assert_allclose(result, [0.02, 0.03, 0.04, 0.05, 0.06])
    raw.loc[raw["date"].ge(dates[-2]), "annual_rate_decimal"] = 0.95
    raw.to_csv(source, index=False)
    repeated = study._load_cash_rates(source, dates)
    pd.testing.assert_series_equal(result, repeated)
