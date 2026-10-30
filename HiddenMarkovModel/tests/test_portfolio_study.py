"""Acceptance checks for the approved two-version portfolio protocol."""
import json
import math
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

import DecisionValueStudy as study
import SwingRegimeSelector as swing
import run_portfolio_study as portfolio
from download_portfolio_study_data import market_frames
from test_swing_regime_selector import market_data, config, targets


def test_equal_initial_weights_hold_until_first_model_rebalance_and_allow_cash():
    data = market_data({"AAA": np.arange(100, 124), "BBB": [100] * 24})
    settings = config(initial_equity=1_000_000, rebalance_sessions=20,
                      slippage_dollars_per_share=0.005)
    result = swing.simulate_portfolio(data, targets(data, [0, 0]), data.index, settings,
                                       execution_prices=data.open, initial_equal_weight=True)
    first = result.daily.iloc[0]
    assert first["equity"] + first["transaction_cost"] == pytest.approx(1_000_000)
    assert first["cash"] == pytest.approx(0, abs=1e-8)
    assert first["weight_AAA"] == pytest.approx(0.5)
    assert first["weight_BBB"] == pytest.approx(0.5)
    assert result.daily["shares_AAA"].iloc[:20].nunique() == 1
    assert not result.daily["is_rebalance"].iloc[1:20].any()
    assert result.daily["gross_exposure"].iloc[20] == 0
    assert result.daily["cash_weight"].iloc[20] == pytest.approx(1)
    assert set(result.trades["date"]) == {data.index[0], data.index[20]}
    assert result.daily["initial_equal_weight_entry"].sum() == 1


def test_per_share_costs_use_actual_share_units_across_a_split():
    data = market_data({"AAA": [10, 10, 10]})
    actual = pd.DataFrame({"AAA": [20, 10, 10]}, index=data.index)
    settings = config(initial_equity=1000, slippage_dollars_per_share=0.005)
    result = swing.simulate_portfolio(data, targets(data, [1]), data.index, settings,
                                       execution_prices=actual)
    after_entry = 1000 / (1 + 0.005 / 20)
    assert result.metrics["final_equity"] == pytest.approx(after_entry * (1 - 0.005 / 10))
    np.testing.assert_allclose(result.trades["transaction_cost"], result.trades["actual_shares_traded"].abs() * 0.005)
    assert abs(result.trades["actual_shares_traded"].iloc[-1]) == pytest.approx(2 * result.trades["actual_shares_traded"].iloc[0])
    assert result.metrics["transaction_cost"] == pytest.approx(1000 - result.metrics["final_equity"])


def test_each_stock_has_its_own_per_share_cost_and_reversal_pays_both_sides():
    data = market_data({"AAA": [10] * 3, "BBB": [100] * 3})
    settings = config(initial_equity=1000, rebalance_sessions=1, slippage_dollars_per_share=0.005)
    weights = targets(data, [0.5, 0.5])
    weights.iloc[1] = [-0.5, -0.5]
    result = swing.simulate_portfolio(data, weights, data.index, settings, execution_prices=data.open)
    entry_equity = 1000 / (1 + 0.5 * 0.005 / 10 + 0.5 * 0.005 / 100)
    assert result.daily["equity"].iloc[0] == pytest.approx(entry_equity)
    np.testing.assert_allclose(result.trades["transaction_cost"], result.trades["actual_shares_traded"].abs() * 0.005)
    reversals = result.trades.loc[result.trades["date"].eq(data.index[1])]
    assert (reversals["shares_before"] > 0).all() and (reversals["shares_after"] < 0).all()
    assert result.metrics["final_equity"] == pytest.approx(1000 - result.metrics["transaction_cost"])


def test_treasury_sharpe_aligns_calendar_intervals_and_includes_entry_cost():
    dates = pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"])
    data = market_data({"AAA": [100, 101, 103, 102]}, dates)
    settings = config(initial_equity=1000, slippage_dollars_per_share=0.005)
    rates = pd.Series([0.0365, 0.073, 0.1095, 0.9], index=dates)
    result = swing.simulate_portfolio(data, targets(data, [1]), dates, settings,
                                       execution_prices=data.open, risk_free_annual_rates=rates)
    expected_returns = result.daily["net_return"].iloc[1:].to_numpy().copy()
    expected_returns[0] = result.daily["equity"].iloc[1] / 1000 - 1
    risk_free = np.array([0.0003, 0.0002, 0.0003])
    excess = expected_returns - risk_free
    np.testing.assert_allclose(result.daily["risk_free_return"].iloc[1:], risk_free)
    assert result.metrics["sharpe"] == pytest.approx(math.sqrt(252) * excess.mean() / excess.std(ddof=1))
    assert result.metrics["cash_interest"] == 0
    rates.iloc[-1] = 0.01
    repeated = swing.simulate_portfolio(data, targets(data, [1]), dates, settings,
                                         execution_prices=data.open, risk_free_annual_rates=rates)
    assert repeated.metrics["sharpe"] == result.metrics["sharpe"]


def test_undefined_excess_return_volatility_produces_null_sharpe():
    data = market_data({"AAA": [100] * 4})
    result = swing.simulate_portfolio(data, targets(data, [1]), data.index, config(),
                                       risk_free_annual_rates=pd.Series(0.0, index=data.index))
    assert result.metrics["sharpe"] is None


@pytest.mark.parametrize("fault", ["missing_execution", "bad_execution", "bad_rate", "duplicate_rate", "offset"])
def test_new_inputs_validate_trust_boundaries(fault):
    data = market_data({"AAA": [100] * 3})
    settings = config(slippage_dollars_per_share=0.005)
    kwargs = {"execution_prices": data.open.copy()}
    if fault == "missing_execution":
        kwargs = {}
    elif fault == "bad_execution":
        kwargs["execution_prices"].iloc[1] = np.nan
    elif fault == "bad_rate":
        kwargs["risk_free_annual_rates"] = pd.Series([0.04, np.nan, 0.04], index=data.index)
    elif fault == "duplicate_rate":
        kwargs["risk_free_annual_rates"] = pd.Series(0.04, index=[data.index[0]] * 3)
    else:
        kwargs.update(initial_equal_weight=True, rebalance_offset=1)
    with pytest.raises(ValueError):
        swing.simulate_portfolio(data, targets(data, [1]), data.index, settings, **kwargs)


@pytest.mark.parametrize("amount", [-0.005, float("nan"), float("inf")])
def test_invalid_slippage_is_rejected(amount):
    with pytest.raises(ValueError):
        config(slippage_dollars_per_share=amount)


def test_basis_point_and_per_share_slippage_cannot_be_charged_together():
    with pytest.raises(ValueError):
        config(slippage_bps_per_side=5, slippage_dollars_per_share=0.005)


def test_vendor_conversion_reverses_splits_including_actions_after_evaluation_end():
    dates = pd.bdate_range("2024-01-02", periods=4)
    raw = pd.DataFrame({"Open": [10.0] * 4, "High": [11.0] * 4,
                        "Low": [9.0] * 4, "Close": [10.0] * 4,
                        "Adj Close": [9.0] * 4, "Stock Splits": [0, 2, 0, 3]}, index=dates)
    adjusted, quoted = market_frames(raw, str(dates[2].date()))
    np.testing.assert_allclose(adjusted.Open, [9, 9, 9])
    np.testing.assert_allclose(quoted, [60, 30, 30])
    assert len(quoted) == 3
    raw.index = [dates[0]] * 4
    with pytest.raises(ValueError):
        market_frames(raw, "2024-12-31")


def test_training_start_excludes_warmup_from_hmm_calibration(monkeypatch):
    dates = pd.bdate_range("2014-06-02", "2015-05-01")
    data = market_data({"AAA": np.linspace(100, 130, len(dates)),
                        "BBB": np.linspace(100, 120, len(dates))}, dates)
    seen = {}

    def capture(features, **kwargs):
        seen["features"] = features
        seen.update(kwargs)
        return pd.DataFrame(index=features.index)

    monkeypatch.setattr(swing.hs, "fit_asof_hmm_regimes", capture)
    swing.prepare_inputs(data, study._empty_predictions(), "2015-04-01", config(), training_start="2015-01-01")
    assert seen["features"].index[0] == pd.Timestamp("2015-01-01")
    assert seen["features"].iloc[0].notna().all()
    assert seen["training_end"] == pd.Timestamp("2015-04-01")


def test_versions_refit_separately_and_pass_their_own_cost_to_calibration(monkeypatch, tmp_path):
    dates = pd.bdate_range("2014-12-01", "2015-05-01")
    data = market_data({"AAA": [100] * len(dates)}, dates)
    calls = []
    prepare_calls = []

    def prepare(*args, **kwargs):
        prepare_calls.append(kwargs)
        return swing.SwingInputs(targets(data, [0]), targets(data, [1]), pd.DataFrame({
            "regime_state": 0, "regime_probability": 1.0,
            "regime_probability_0": 1.0, "regime_probability_1": 0.0}, index=dates))

    def fit(prices, inputs, train_dates, settings, momentum_only, **kwargs):
        calls.append((settings.slippage_dollars_per_share, train_dates, kwargs["execution_prices"]))
        return {0: "momentum" if settings.slippage_dollars_per_share == 0 else "cash", 1: "cash"}, []

    monkeypatch.setattr(portfolio.swing, "prepare_inputs", prepare)
    monkeypatch.setattr(portfolio.study, "_fit_library", fit)
    policies = []
    for name, amount in (("zero", 0), ("half_cent", 0.005)):
        _, _, policy = portfolio.fit_version(data, data.open, {"training_start": "2015-01-01"},
                                             config(slippage_dollars_per_share=amount), pd.Timestamp("2015-04-01"), tmp_path / name)
        policies.append(policy)
        saved = json.loads((tmp_path / name / "policy_lock.json").read_text())
        assert saved["config"]["slippage_dollars_per_share"] == amount
        assert saved["calibration_initial_equal_weight"] is False
    assert len(prepare_calls) == 2
    assert [call[0] for call in calls] == [0, 0.005]
    assert all(call[1].min() >= pd.Timestamp("2015-01-01") and call[1].max() < pd.Timestamp("2015-04-01") for call in calls)
    assert policies[0] != policies[1]


def test_period_output_reconciles_passive_returns_and_marks_imposed_initial_entries(tmp_path):
    data = market_data({"AAA": np.linspace(100, 130, 24), "BBB": [100] * 24})
    benchmark = market_data({"IWV": np.linspace(100, 110, 24)}, data.index)
    regimes = pd.DataFrame(index=data.index)
    inputs = swing.SwingInputs(targets(data, [0, 0]), targets(data, [1, 1]), regimes)
    rates = pd.Series(0.04, index=data.index)
    settings = config(initial_equity=1_000_000, rebalance_sessions=20)
    metrics, curves = portfolio.score_period(data, benchmark, data.open, benchmark.open, rates, rates,
                                              inputs, targets(data, [0, 0]), data.index, settings, tmp_path)
    assert metrics["passive_stocks"]["net_cumulative_return"] == pytest.approx(0.15)
    assert metrics["hmm"]["first_model_rebalance"] == str(data.index[20].date())
    assert metrics["hmm"]["average_model_exposure_after_initial_holding"] == 0
    assert metrics["hmm"]["completed_positions"] == 2
    assert metrics["hmm"]["attribution_residual"] == pytest.approx(0, abs=1e-6)
    returns = pd.read_csv(tmp_path / "individual_stock_returns.csv")
    np.testing.assert_allclose(returns.adjusted_open_to_open_return, [0.3, 0])
    path = tmp_path / "curves.svg"
    portfolio.equity_chart(curves[["hmm", "passive_stocks"]], path, settings.initial_equity)
    assert ET.parse(path).getroot().tag.endswith("svg")


def test_cash_shortfall_is_added_equally_without_removing_model_tilts():
    weights = pd.DataFrame([[0.6, 0.0], [0.7, 0.2], [0.0, 0.0]], columns=["AAA", "BBB"])
    limited = swing.apply_cash_limit(weights, 0.2)
    np.testing.assert_allclose(limited, [[0.7, 0.1], [0.7, 0.2], [0.4, 0.4]])
    with pytest.raises(ValueError):
        swing.apply_cash_limit(pd.DataFrame([[-0.2, 0.3]]), 0.2)


@pytest.mark.parametrize("amount", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_cash_caps_are_rejected(amount):
    with pytest.raises(ValueError):
        config(max_cash_weight=amount)


def test_cash_cap_waits_for_first_rebalance_then_tops_up_drift_after_costs():
    data = market_data({"AAA": [100] * 6 + [70, 90, 60, 65],
                        "BBB": [100] * 6 + [80, 100, 70, 75]})
    settings = config(initial_equity=1_000_000, rebalance_sessions=5,
                      slippage_dollars_per_share=0.005, max_cash_weight=0.2)
    rates = pd.Series(0.04, index=data.index)
    result = swing.simulate_portfolio(data, targets(data, [0, 0]), data.index, settings,
                                       execution_prices=data.open, cash_annual_rates=rates,
                                       initial_equal_weight=True)
    assert result.daily["shares_AAA"].iloc[:5].nunique() == 1
    assert not result.daily["is_cash_limit_topup"].iloc[:6].any()
    assert result.daily["cash_weight"].iloc[5] == pytest.approx(0.2)
    assert result.daily["is_cash_limit_topup"].iloc[6]
    assert result.daily["cash_weight"].iloc[:-1].max() <= 0.2 + 1e-9
    assert result.daily["gross_exposure"].iloc[:-1].min() >= 0.8 - 1e-9
    topup = result.trades.loc[result.trades.date.eq(data.index[6])]
    assert (topup.shares_traded > 0).all()
    assert topup.traded_notional.iloc[0] == pytest.approx(topup.traded_notional.iloc[1])
    np.testing.assert_allclose(result.trades.transaction_cost,
                               result.trades.actual_shares_traded.abs() * 0.005)
    assert result.daily["cash_weight"].iloc[-1] == pytest.approx(1)
    assert result.metrics["cash_limit_topup_days"] >= 1
    assert result.metrics["maximum_cash_weight_before_liquidation"] <= 0.2 + 1e-9


def test_mandatory_stock_gains_cannot_qualify_an_inactive_trained_policy(tmp_path):
    data = market_data({"AAA": 100 * 1.003 ** np.arange(80), "BBB": [100] * 80})
    regimes = pd.DataFrame({"regime_state": 0, "regime_probability": 1.0,
                            "regime_probability_0": 1.0, "regime_probability_1": 0.0}, index=data.index)
    inputs = swing.SwingInputs(targets(data, [0, 0]), targets(data, [0, 0]), regimes)
    settings = config(initial_equity=10000, max_cash_weight=0.2)
    policies, diagnostics = portfolio.fit_cash_limited_policy(data, data.open, inputs, data.index, settings, tmp_path)
    assert all(expert == "cash" for expert in policies.values())
    assert all(d["baseline_training_metrics"]["net_cumulative_return"] > 0 for d in diagnostics)
    assert all(d["mean_incremental_net_daily_return"] == 0 for d in diagnostics)
    assert all(d["model_active_days"] == 0 for d in diagnostics)


def test_training_can_select_a_tilt_that_improves_on_the_stock_floor(tmp_path):
    data = market_data({"AAA": 100 * 1.003 ** np.arange(100), "BBB": [100] * 100})
    regimes = pd.DataFrame({"regime_state": 0, "regime_probability": 1.0,
                            "regime_probability_0": 1.0, "regime_probability_1": 0.0}, index=data.index)
    inputs = swing.SwingInputs(targets(data, [0, 0]), targets(data, [1, 0]), regimes)
    settings = config(initial_equity=10000, max_cash_weight=0.2, slippage_dollars_per_share=0.005)
    policies, diagnostics = portfolio.fit_cash_limited_policy(data, data.open, inputs, data.index, settings, tmp_path)
    assert policies == {0: "momentum", 1: "cash"}
    assert diagnostics[0]["lower_bound"] > 0
    assert diagnostics[0]["training_metrics"]["net_cumulative_return"] > diagnostics[0]["baseline_training_metrics"]["net_cumulative_return"]
    assert diagnostics[0]["training_metrics"]["transaction_cost"] > 0
    assert diagnostics[0]["model_active_rebalances"] >= settings.min_train_rebalances


def test_cash_cap_control_keeps_cash_benchmark_and_full_passive_book_distinct(tmp_path):
    data = market_data({"AAA": np.linspace(100, 130, 24), "BBB": [100] * 24})
    benchmark = market_data({"IWV": np.linspace(100, 110, 24)}, data.index)
    inputs = swing.SwingInputs(targets(data, [0, 0]), targets(data, [0, 0]), pd.DataFrame(index=data.index))
    rates = pd.Series(0.04, index=data.index)
    settings = config(initial_equity=1_000_000, rebalance_sessions=5, max_cash_weight=0.2)
    metrics, _ = portfolio.score_period(data, benchmark, data.open, benchmark.open, rates, rates,
                                         inputs, targets(data, [0, 0]), data.index, settings, tmp_path)
    assert metrics["cash"]["average_gross_exposure"] == 0
    assert metrics["passive_stocks"]["average_gross_exposure"] == pytest.approx(1)
    assert metrics["hmm"]["final_equity"] == pytest.approx(metrics["equal_weight_floor"]["final_equity"])
    assert metrics["hmm"]["maximum_cash_weight_before_liquidation"] <= 0.2 + 1e-9
