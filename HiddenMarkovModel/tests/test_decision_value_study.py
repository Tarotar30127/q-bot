"""Economic accounting and information-boundary checks for decision value research."""

import numpy as np
import pandas as pd
import pytest

import HyperplaneStock as hs
import SwingRegimeSelector as swing
import DecisionValueStudy as study


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


def test_training_exposure_excludes_terminal_cash_and_uses_only_training_book():
    data = market({"AAA": [100.0] * 5})
    train = swing.simulate_portfolio(data, weights(data, [0.4]), data.index,
                                     config(rebalance_sessions=10))
    assert study.training_exposure(train) == pytest.approx(0.4)
    altered = swing.SimulationResult(train.daily.copy(), train.trades.copy(),
                                     dict(train.metrics))
    altered.daily.iloc[-1, altered.daily.columns.get_loc("gross_exposure")] = 123.0
    assert study.training_exposure(altered) == pytest.approx(0.4)


def test_passive_benchmark_targets_do_not_scale_with_number_of_symbols():
    dates = pd.bdate_range("2024-01-02", periods=3)
    target = study.benchmark_targets(dates, ["AAA", "BBB", "CCC"], 0.18)
    np.testing.assert_allclose(target.sum(axis=1), 0.18)
    np.testing.assert_allclose(target.to_numpy(), 0.06)


def test_paired_bootstrap_aligns_returns_by_date_before_resampling():
    dates = pd.bdate_range("2020-01-02", periods=60)
    left = pd.Series(np.tile([0.03, -0.025, 0.01, -0.01, 0.005], 12), index=dates)
    right = left * 0.7 + np.linspace(-0.001, 0.001, len(left))
    original = study.paired_block_bootstrap(left, right, block_sessions=5,
                                           replications=200, seed=14)
    shuffled = study.paired_block_bootstrap(left, right.sample(frac=1.0, random_state=5),
                                           block_sessions=5, replications=200, seed=14)
    assert original == shuffled


def test_paired_bootstrap_is_reproducible_and_uses_common_dates():
    dates = pd.bdate_range("2020-01-02", periods=70)
    left = pd.Series(np.linspace(-0.01, 0.02, 70), index=dates)
    right = pd.Series(np.linspace(0.01, -0.02, 70), index=dates)
    left = left.iloc[:-5]
    right = right.iloc[5:]
    first = study.paired_block_bootstrap(left, right, block_sessions=5,
                                        replications=200, seed=42)
    second = study.paired_block_bootstrap(left.iloc[5:], right.iloc[:-5],
                                         block_sessions=5, replications=200, seed=42)
    assert first == second


def test_paired_bootstrap_preserves_shared_shocks_and_exact_constant_daily_difference():
    dates = pd.bdate_range("2020-01-02", periods=80)
    shared_shocks = np.tile([0.04, -0.03, 0.02, -0.01], 20)
    left = pd.Series(shared_shocks + 0.001, index=dates)
    right = pd.Series(shared_shocks, index=dates)
    result = study.paired_block_bootstrap(left, right, block_sessions=8,
                                         replications=200, seed=8)
    assert result["n_common_sessions"] == len(dates)
    np.testing.assert_allclose(result["mean_daily_excess_ci"], [0.001, 0.001], atol=1e-15)
    expected = np.prod(1.0 + left) - np.prod(1.0 + right)
    assert result["observed_cumulative_excess"] == pytest.approx(expected)


def test_identical_paired_strategies_have_zero_excess_in_every_resampled_block():
    dates = pd.bdate_range("2020-01-02", periods=65)
    returns = pd.Series(np.sin(np.arange(65)) * 0.02, index=dates)
    result = study.paired_block_bootstrap(returns, returns, block_sessions=7,
                                         replications=200, seed=19)
    assert result["observed_cumulative_excess"] == 0.0
    assert result["observed_mean_daily_excess"] == 0.0
    assert result["cumulative_excess_ci"] == [0.0, 0.0]
    assert result["mean_daily_excess_ci"] == [0.0, 0.0]


def test_decision_audit_labels_are_fixed_before_future_counterfactual_return():
    data = market({"AAA": [100.0, 101.0, 110.0, 99.0, 105.0, 120.0],
                   "BBB": [100.0, 99.0, 90.0, 101.0, 110.0, 90.0]})
    signals = weights(data, [1.0, 1.0])
    regime = pd.DataFrame({"regime_state": 0, "regime_probability": 0.8,
                           "regime_probability_0": 0.8,
                           "regime_probability_1": 0.2}, index=data.index)
    inputs = swing.SwingInputs(signals, signals, regime)
    desired = weights(data, [0.4, 0.0])
    settings = config(rebalance_sessions=2, slippage_bps_per_side=10.0)
    original = study.interval_decisions(data, inputs, desired, data.index, settings)
    changed_prices = data.open.copy()
    changed_prices.loc[data.index[2]:, "AAA"] *= 2.0
    changed_prices.loc[data.index[2]:, "BBB"] *= 0.5
    changed = hs.MarketData(open=changed_prices, high=changed_prices.copy(),
                            low=changed_prices.copy(), close=changed_prices.copy())
    repeated = study.interval_decisions(changed, inputs, desired, data.index, settings)
    known = ["decision_date", "exit_date", "ticker", "regime_state",
             "regime_probability", "momentum_eligible", "hmm_allowed", "hmm_target_weight"]
    pd.testing.assert_frame_equal(original[known], repeated[known])
    entry = original.loc[original["decision_date"].eq(data.index[0])]
    assert list(entry["hmm_allowed"]) == [True, False]
    np.testing.assert_allclose(entry["hypothetical_gross_return"], [0.1, -0.1])
    np.testing.assert_allclose(entry["hypothetical_roundtrip_net_return"],
                               np.array([1.1, 0.9]) * 0.999 / 1.001 - 1.0)
    assert not original["hypothetical_gross_return"].equals(repeated["hypothetical_gross_return"])


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


def test_case_policy_and_exposure_are_frozen_before_future_prices_or_signals(monkeypatch, tmp_path):
    dates = pd.bdate_range("2018-01-02", periods=180)
    train_end = dates[155]
    rising = 100.0 * 1.004 ** np.arange(len(dates))
    data = market({"AAA": rising}, dates)
    positive = weights(data, [1.0])
    regimes = pd.DataFrame({"regime_state": 0, "regime_probability": 1.0,
                            "regime_probability_0": 1.0,
                            "regime_probability_1": 0.0}, index=dates)
    original_inputs = swing.SwingInputs(positive, positive, regimes)
    call_log = []
    current_inputs = [original_inputs]

    def causal_preparation(prices, predictions, validation_start, settings):
        call_log.append(pd.Timestamp(validation_start))
        return current_inputs[0]

    monkeypatch.setattr(swing, "prepare_inputs", causal_preparation)
    settings = config(min_train_days=20, min_train_rebalances=3)
    original = study._prepare_case("isolated", data, study._empty_predictions(), train_end,
                                   settings, tmp_path / "original")
    future_prices = data.open.copy()
    future_prices.loc[train_end:] *= np.linspace(0.5, 0.01, len(dates[dates >= train_end]))[:, None]
    future_data = hs.MarketData(open=future_prices, high=future_prices.copy(),
                                low=future_prices.copy(), close=future_prices.copy())
    future_signals = positive.copy()
    future_signals.loc[train_end:] = -1.0
    future_regimes = regimes.copy()
    future_regimes.loc[train_end:, "regime_state"] = 1
    current_inputs[0] = swing.SwingInputs(future_signals, future_signals, future_regimes)
    repeated = study._prepare_case("isolated", future_data, study._empty_predictions(), train_end,
                                   settings, tmp_path / "repeated")
    assert call_log == [train_end, train_end]
    assert original["policy"] == repeated["policy"]
    assert original["allocation"] == repeated["allocation"]
    assert original["policy"]["selection_used_validation_or_evaluation"] is False
    assert pd.Timestamp(original["policy"]["training_end"]) < train_end
    assert original["policy"]["policies"][0] == "mean_reversion"
