"""Causal strategy selection and fixed-capital accounting acceptance tests."""

import numpy as np
import pandas as pd
import pytest

import HyperplaneStock as hs
from test_hyperplane_stock import synthetic_market_data


def selector_config(**overrides):
    parameters = {
        "tickers": ["AAA", "BBB", "CCC"],
        "start": "2020-01-01",
        "end": "2022-01-01",
        "lookback": 60,
        "require_stationarity": False,
        "use_hmm": True,
        "use_regime_selector": True,
        "hmm_states": 3,
        "commission_bps_per_side": 0.0,
        "slippage_grid_bps_per_side": [0.0],
        "short_borrow_bps_per_day": 0.0,
        "selector_min_train_days": 30,
    }
    parameters.update(overrides)
    return hs.BacktestConfig(**parameters)


def policy_training_fixture(days_per_state=40):
    dates = pd.bdate_range("2020-01-02", periods=3 * days_per_state)
    rows = []
    for day, date in enumerate(dates):
        state = day // days_per_state
        for ticker in ("AAA", "BBB", "CCC"):
            rows.append({
                "date": date,
                "ticker": ticker,
                "split": "train",
                "actual_return": 0.01 if state != 2 else -0.01,
                "selector_mean_reversion_signal": -1 if state == 1 else 1,
                "momentum_signal": -1 if state == 0 else 1,
                "regime_state": state,
                "regime_probability": 1.0,
                **{f"regime_probability_{k}": float(k == state) for k in range(3)},
            })
    return pd.DataFrame(rows)


@pytest.mark.parametrize("overrides", [
    {"use_hmm": False},
    {"momentum_lookback": 1},
    {"momentum_threshold_bps": -1.0},
    {"momentum_threshold_bps": np.nan},
    {"selector_min_train_days": 1},
    {"selector_confidence_z": -1.0},
    {"selector_confidence_z": np.inf},
    {"selector_min_exposure": 0.0},
    {"selector_min_exposure": 1.01},
])
def test_selector_configuration_rejects_invalid_parameters(overrides):
    with pytest.raises(ValueError):
        selector_config(**overrides)


def test_momentum_uses_prior_closes_and_ignores_current_intraday_data():
    dates = pd.bdate_range("2020-01-02", periods=150)
    close = pd.DataFrame(
        np.exp(np.arange(150)[:, None] * np.array([0.01, -0.01, 0.0]))
        * np.array([100.0, 150.0, 50.0]),
        index=dates,
        columns=["AAA", "BBB", "CCC"],
    )
    data = hs.MarketData(open=close.copy(), high=close * 1.02,
                         low=close * 0.98, close=close)
    config = selector_config(momentum_lookback=10, momentum_threshold_bps=50.0)
    original = hs.walk_forward_predictions(data, config)
    first = original[original["date"].eq(dates[60])].set_index("ticker")
    assert first.loc["AAA", "momentum_signal"] == 1
    assert first.loc["BBB", "momentum_signal"] == -1
    assert first.loc["CCC", "momentum_signal"] == 0

    cutoff = dates[100]
    changed_close = close.copy()
    changed_close.loc[changed_close.index >= cutoff] *= 1.5
    changed_high = data.high.copy()
    changed_high.loc[changed_high.index >= cutoff] *= 2.0
    changed_low = data.low.copy()
    changed_low.loc[changed_low.index >= cutoff] *= 0.5
    changed = hs.MarketData(open=data.open, high=changed_high,
                           low=changed_low, close=changed_close)
    altered = hs.walk_forward_predictions(changed, config)
    columns = ["date", "ticker", "momentum_return", "momentum_signal"]
    pd.testing.assert_frame_equal(
        original.loc[original["date"] <= cutoff, columns],
        altered.loc[altered["date"] <= cutoff, columns],
    )


def test_policy_library_uses_training_only_and_finds_distinct_experts():
    training = policy_training_fixture()
    config = selector_config()
    policies, diagnostics = hs.fit_regime_policy_library(training, config)
    assert policies == {0: "mean_reversion", 1: "momentum", 2: "cash"}
    assert diagnostics

    future = training.copy()
    future["date"] += pd.Timedelta(days=365)
    future["split"] = "validation"
    future["actual_return"] *= -100.0
    test = future.copy()
    test["split"] = "test"
    test["date"] += pd.Timedelta(days=365)
    contaminated = pd.concat([training, future, test], ignore_index=True)
    repeated_policies, repeated_diagnostics = hs.fit_regime_policy_library(
        contaminated, config
    )
    assert repeated_policies == policies
    assert repeated_diagnostics == diagnostics


def test_policy_library_counts_calendar_days_and_requires_confident_training():
    training = policy_training_fixture(days_per_state=3)
    policies, _ = hs.fit_regime_policy_library(
        training, selector_config(selector_min_train_days=5)
    )
    assert set(policies.values()) == {"cash"}

    training = policy_training_fixture()
    training["regime_probability"] = 1.0 / 3.0
    for state in range(3):
        training[f"regime_probability_{state}"] = 1.0 / 3.0
    policies, _ = hs.fit_regime_policy_library(training, selector_config())
    assert set(policies.values()) == {"cash"}


def test_policy_library_uses_conservative_costs_and_deterministic_tie_break():
    training = policy_training_fixture()
    training["actual_return"] = 0.0001
    training["selector_mean_reversion_signal"] = 1
    training["momentum_signal"] = 1
    policies, _ = hs.fit_regime_policy_library(
        training, selector_config(slippage_grid_bps_per_side=[0.0, 10.0])
    )
    assert set(policies.values()) == {"cash"}

    training["actual_return"] = 0.01
    policies, _ = hs.fit_regime_policy_library(training, selector_config())
    assert set(policies.values()) == {"mean_reversion"}


def test_policy_library_rejects_serially_correlated_uncertain_positive_mean():
    training = policy_training_fixture().query("regime_state == 0").copy()
    dates = sorted(training["date"].unique())
    returns = {date: (0.02 if index < 20 else -0.005)
               for index, date in enumerate(dates)}
    training["actual_return"] = training["date"].map(returns)
    policies, diagnostics = hs.fit_regime_policy_library(training, selector_config())
    mean_reversion = next(item for item in diagnostics
                          if item["state"] == 0 and item["expert"] == "mean_reversion")
    iid_standard_error = np.std(list(returns.values()), ddof=0) / np.sqrt(len(dates))
    assert mean_reversion["mean_net_daily_return"] > 0.0
    assert mean_reversion["standard_error"] > iid_standard_error
    assert mean_reversion["lower_bound"] < 0.0
    assert policies[0] == "cash"


def mixture_fixture(probability_zero=0.7, probability_one=0.3):
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"]),
        "ticker": ["AAA"],
        "selector_mean_reversion_signal": [1],
        "momentum_signal": [-1],
        "regime_state": [int(probability_one > probability_zero)],
        "regime_probability": [max(probability_zero, probability_one)],
        "regime_probability_0": [probability_zero],
        "regime_probability_1": [probability_one],
        "regime_probability_2": [0.0],
    })


def test_selector_soft_mixture_preserves_probability_cash_and_net_exposure():
    predictions = mixture_fixture()
    original = predictions.copy(deep=True)
    result = hs.apply_regime_selector(
        predictions, {0: "mean_reversion", 1: "momentum", 2: "cash"},
        selector_config(),
    )
    pd.testing.assert_frame_equal(predictions, original)
    assert result.loc[0, "regime_selector_signal"] == pytest.approx(0.4)
    assert result.loc[0, "selector_mean_reversion_weight"] == pytest.approx(0.7)
    assert result.loc[0, "selector_momentum_weight"] == pytest.approx(0.3)
    assert result.loc[0, "selector_cash_weight"] == pytest.approx(0.0)

    unknown = hs.apply_regime_selector(
        predictions, {0: "mean_reversion"}, selector_config()
    )
    assert unknown.loc[0, "regime_selector_signal"] == pytest.approx(0.7)
    assert unknown.loc[0, "selector_cash_weight"] == pytest.approx(0.3)


@pytest.mark.parametrize("predictions,minimum_exposure", [
    (mixture_fixture(0.5, 0.5), 0.1),
    (mixture_fixture(0.56, 0.44), 0.2),
    (mixture_fixture(np.nan, np.nan), 0.1),
])
def test_uncertain_missing_or_near_cancelled_mixtures_hold_cash(
    predictions, minimum_exposure
):
    result = hs.apply_regime_selector(
        predictions, {0: "mean_reversion", 1: "momentum", 2: "cash"},
        selector_config(selector_min_exposure=minimum_exposure),
    )
    assert result.loc[0, "regime_selector_signal"] == 0.0
    assert result.loc[0, "selector_mean_reversion_weight"] == 0.0
    assert result.loc[0, "selector_momentum_weight"] == 0.0
    assert result.loc[0, "selector_cash_weight"] == 1.0


def test_fractional_positions_scale_both_costs_and_keep_cash_capital():
    predictions = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-02",
                                "2024-01-03", "2024-01-03"]),
        "ticker": ["AAA", "BBB", "AAA", "BBB"],
        "split": ["test"] * 4,
        "actual_return": [0.02, -0.04, -0.08, 0.5],
        "position": [1.0, 0.5, -0.25, 0.0],
    })
    score = hs.score_signals(
        predictions, "position", split="test",
        commission_bps_per_side=1.0, slippage_bps_per_side=4.0,
        short_borrow_bps_per_day=2.0,
    )
    assert score["predictions"] == 3
    assert score["coverage"] == 0.75
    assert score["dpa"] == pytest.approx(2.0 / 3.0)
    assert score["gross_cumulative_return"] == pytest.approx(0.01)
    assert score["net_cumulative_return"] == pytest.approx(
        (1.0 - 0.00075) * (1.0 + 0.00985) - 1.0
    )
    assert score["max_drawdown"] == pytest.approx(-0.00075)
    assert hs._maximum_drawdown(pd.Series([-0.1])) == pytest.approx(-0.1)


def test_adaptive_hmm_full_posteriors_are_causal_and_normalized():
    data = synthetic_market_data(180)
    cutoff = data.index[132]
    dates = list(data.index[130:136])
    arguments = {
        "training_end": data.index[130],
        "prediction_dates": dates,
        "n_states": 2,
        "random_seed": 19,
        "finetune_window": 60,
        "finetune_iterations": 2,
        "include_probabilities": True,
        "align_states": True,
    }
    original = hs.fit_adaptive_hmm_regimes(hs.build_regime_features(data), **arguments)
    probabilities = original[["regime_probability_0", "regime_probability_1"]]
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_allclose(probabilities.max(axis=1), original["regime_probability"])
    np.testing.assert_array_equal(probabilities.to_numpy().argmax(axis=1),
                                  original["regime_state"])

    changed_close = data.close.copy()
    changed_close.loc[changed_close.index >= cutoff] *= 1.2
    changed = hs.MarketData(open=data.open, high=data.high,
                           low=data.low, close=changed_close)
    altered = hs.fit_adaptive_hmm_regimes(hs.build_regime_features(changed), **arguments)
    # Features on cutoff use the previous close, so this day's action is unchanged.
    pd.testing.assert_frame_equal(original.loc[:cutoff], altered.loc[:cutoff])


def test_adaptive_hmm_alignment_restores_permuted_state_labels(monkeypatch):
    def fit_permuted_model(model, observations):
        dimensions = observations.shape[1]
        model.n_features = dimensions
        model.means_ = np.array([[-2.0] * dimensions, [2.0] * dimensions])
        model.covars_ = np.ones((2, dimensions))
        model.startprob_ = np.array([0.4, 0.6])
        model.transmat_ = np.array([[0.9, 0.1], [0.2, 0.8]])
        if model.n_iter != 300:
            model.means_ = model.means_[::-1].copy()
            model.startprob_ = model.startprob_[::-1].copy()
            model.transmat_ = model.transmat_[::-1, ::-1].copy()

    monkeypatch.setattr(hs, "_fit_hmm_silently", fit_permuted_model)
    data = synthetic_market_data(160)
    features = hs.build_regime_features(data)
    arguments = {
        "training_end": data.index[130],
        "prediction_dates": list(data.index[130:134]),
        "n_states": 2,
        "random_seed": 19,
        "finetune_window": 60,
        "finetune_iterations": 2,
        "include_probabilities": True,
    }
    unaligned = hs.fit_adaptive_hmm_regimes(features, align_states=False, **arguments)
    aligned = hs.fit_adaptive_hmm_regimes(features, align_states=True, **arguments)
    columns = ["regime_probability_0", "regime_probability_1"]
    np.testing.assert_allclose(aligned[columns], unaligned[columns].to_numpy()[:, ::-1])
    np.testing.assert_array_equal(aligned["regime_state"], 1 - unaligned["regime_state"])


def test_pipeline_selector_uses_training_policies_and_validation_choice_only():
    data = synthetic_market_data(210, seed=31)
    config = selector_config(
        hmm_states=2, selector_min_train_days=5,
        signal_threshold_bps=0.0, momentum_threshold_bps=0.0,
        minimum_validation_coverage=0.0, random_seed=17,
    )
    first = hs.run_backtest(data, config)
    assert {"regime_strategy_selector", "momentum_baseline"}.issubset(
        set(first.metrics["model"])
    )
    assert first.summary["test_used_for_selection"] is False
    assert pd.Timestamp(first.summary["selector_training_end"]) < pd.Timestamp(
        first.summary["validation_start"]
    )
    test_models = set(first.metrics.loc[first.metrics["split"].eq("test"), "model"])
    assert test_models == {first.summary["selected_model"]}
    weights = first.predictions[["selector_mean_reversion_weight",
                                 "selector_momentum_weight", "selector_cash_weight"]]
    evaluated = first.predictions["regime_selector_evaluation_status"].ne("not_evaluated_on_test")
    np.testing.assert_allclose(weights.loc[evaluated].sum(axis=1), 1.0)
    assert first.predictions.loc[evaluated, "regime_selector_signal"].abs().le(1.0).all()

    test_start = pd.Timestamp(first.summary["test_start"])
    changed_close = data.close.copy()
    changed_close.loc[changed_close.index >= test_start] *= 1.1
    changed = hs.MarketData(open=data.open, high=data.high,
                           low=data.low, close=changed_close)
    second = hs.run_backtest(changed, config)
    for key in ("selected_model", "selected_threshold_bps",
                "selector_mean_reversion_threshold_bps",
                "regime_selector_policies", "regime_selector_policy_diagnostics"):
        assert first.summary[key] == second.summary[key]
    validation_columns = ["date", "ticker", "regime_selector_signal", "selected_signal"]
    pd.testing.assert_frame_equal(
        first.predictions.loc[first.predictions["date"] < test_start, validation_columns],
        second.predictions.loc[second.predictions["date"] < test_start, validation_columns],
    )
    pd.testing.assert_frame_equal(
        first.metrics.loc[first.metrics["split"].eq("validation")].reset_index(drop=True),
        second.metrics.loc[second.metrics["split"].eq("validation")].reset_index(drop=True),
    )


@pytest.mark.parametrize("cash_only", [False, True])
def test_adaptive_selector_reports_selected_or_skipped_test_evaluation(monkeypatch, cash_only):
    data = synthetic_market_data(100)
    state_for_date = {date: index % 2 for index, date in enumerate(data.index)}
    predictions = pd.DataFrame([
        {
            "date": date,
            "ticker": ticker,
            "train_end": date - pd.Timedelta(days=1),
            "actual_return": -0.01 if cash_only else 0.01,
            "predicted_return": 0.01 if cash_only or state == 0 else -0.01,
            "model_valid": True,
            "momentum_return": -0.01 if state == 0 else 0.01,
            "momentum_signal": 1 if cash_only else (-1 if state == 0 else 1),
        }
        for date, state in state_for_date.items()
        for ticker in data.tickers
    ])
    monkeypatch.setattr(hs, "walk_forward_predictions", lambda *_: predictions.copy())

    def regimes_for_dates(dates):
        return pd.DataFrame([
            {
                "date": date,
                "regime_state": state_for_date[date],
                "regime_probability": 1.0,
                "regime_probability_0": float(state_for_date[date] == 0),
                "regime_probability_1": float(state_for_date[date] == 1),
            }
            for date in dates
        ]).set_index("date")

    inference_calls = []

    def adaptive_regimes(features, **arguments):
        inference_calls.append(arguments)
        return regimes_for_dates(arguments["prediction_dates"])

    monkeypatch.setattr(hs, "fit_adaptive_hmm_regimes", adaptive_regimes)
    monkeypatch.setattr(
        hs, "fit_asof_hmm_regimes",
        lambda features, **arguments: regimes_for_dates(features.index),
    )
    validation_start = data.index[60]
    test_start = data.index[80]
    config = selector_config(
        hmm_states=2, hmm_adaptive=True, selector_min_train_days=5,
        signal_threshold_bps=0.0, minimum_validation_coverage=0.0,
        validation_start=str(validation_start.date()), test_start=str(test_start.date()),
    )
    result = hs.run_backtest(data, config)
    evaluation = result.summary["regime_selector_evaluation"]
    assert evaluation["validation"]["observations"] == 60
    assert evaluation["validation"]["forecasts_computed"] == 60
    assert evaluation["validation"]["regime_probability_rows"] == 60
    test_rows = result.predictions.loc[result.predictions["split"].eq("test")]
    if cash_only:
        assert result.summary["selected_model"] == "no_trade"
        assert result.summary["regime_selector_policies"] == {"0": "cash", "1": "cash"}
        assert "no training expert" in result.summary["regime_selector_status"]
        assert evaluation["validation"]["active_positions"] == 0
        assert evaluation["validation"]["coverage"] == 0.0
        assert evaluation["validation"]["decision_counts"] == {"cash_no_trading_policy": 60}
        assert evaluation["test"]["status"] == "not_evaluated"
        assert evaluation["test"]["active_positions"] is None
        assert evaluation["test"]["coverage"] is None
        assert evaluation["test"]["forecasts_computed"] == 60
        assert evaluation["test"]["regime_probability_rows"] == 0
        assert len(inference_calls) == 1
        assert test_rows["regime_selector_signal"].isna().all()
        assert test_rows["selector_cash_allocation"].isna().all()
        assert test_rows["regime_selector_evaluation_status"].eq("not_evaluated_on_test").all()
        assert set(result.metrics.loc[result.metrics["split"].eq("test"), "model"]) == {"no_trade"}
        cash_score = result.metrics.loc[
            result.metrics["model"].eq("regime_strategy_selector")
            & result.metrics["split"].eq("validation")
            & result.metrics["scope"].eq("portfolio")
        ].iloc[0]
        assert cash_score["observations"] == 60
        assert cash_score["predictions"] == cash_score["active_positions"] == 0
        return
    assert result.summary["selected_model"] == "regime_strategy_selector"
    assert result.summary["regime_selector_policies"] == {
        "0": "mean_reversion", "1": "momentum"
    }
    assert result.summary["selector_policy_updates_on_validation_or_test"] is False
    assert len(inference_calls) == 2
    assert all(call["include_probabilities"] and call["align_states"]
               for call in inference_calls)
    assert list(inference_calls[0]["prediction_dates"]) == list(data.index[60:80])
    assert list(inference_calls[1]["prediction_dates"]) == list(data.index[80:])
    assert all(call["training_end"] == validation_start for call in inference_calls)
    assert evaluation["test"]["status"] == "evaluated"
    assert evaluation["test"]["active_positions"] == 60
    assert evaluation["test"]["coverage"] == 1.0
    assert evaluation["test"]["decision_counts"] == {"active_position": 60}
    np.testing.assert_allclose(test_rows["selected_signal"], 1.0)
    assert set(result.metrics.loc[result.metrics["split"].eq("test"), "model"]) == {
        "regime_strategy_selector"
    }


def test_disabled_selector_preserves_existing_candidates():
    data = synthetic_market_data(160)
    config = selector_config(use_regime_selector=False, use_hmm=False)
    result = hs.run_backtest(data, config)
    assert "regime_strategy_selector" not in set(result.metrics["model"])
    assert result.summary["selected_model"] in {"hyperplane_baseline", "no_trade"}
    assert result.summary["test_used_for_selection"] is False
