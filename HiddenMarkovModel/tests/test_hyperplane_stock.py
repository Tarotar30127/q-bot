import numpy as np
import pandas as pd
from pathlib import Path

import HyperplaneStock as hs


def synthetic_market_data(rows: int = 360, seed: int = 7) -> hs.MarketData:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=rows)
    common = np.cumsum(rng.normal(0.0003, 0.009, rows))
    stationary = np.zeros((rows, 3))
    shocks = rng.normal(0.0, 0.012, size=(rows, 3))
    for index in range(1, rows):
        stationary[index] = 0.82 * stationary[index - 1] + shocks[index]
    bases = np.log(np.array([100.0, 150.0, 50.0]))
    log_close = bases + common[:, None] + stationary
    close = pd.DataFrame(
        np.exp(log_close), index=dates, columns=["AAA", "BBB", "CCC"]
    )
    gap = rng.normal(0.0, 0.003, size=close.shape)
    open_price = close * np.exp(-gap)
    high = pd.DataFrame(
        np.maximum(open_price, close) * (1.0 + np.abs(rng.normal(0.002, 0.001, close.shape))),
        index=dates,
        columns=close.columns,
    )
    low = pd.DataFrame(
        np.minimum(open_price, close) * (1.0 - np.abs(rng.normal(0.002, 0.001, close.shape))),
        index=dates,
        columns=close.columns,
    )
    return hs.MarketData(open=open_price, high=high, low=low, close=close)


def test_importable_forecast_is_finite_and_deterministic():
    data = synthetic_market_data(180)
    current = data.open.iloc[-1]
    first = hs.forecast_prices(
        data.close.iloc[-120:], current, require_stationarity=False
    )
    second = hs.forecast_prices(
        data.close.iloc[-120:], current, require_stationarity=False
    )
    assert np.isfinite(first.predicted_prices).all()
    assert (first.predicted_prices > 0).all()
    np.testing.assert_allclose(first.predicted_prices, second.predicted_prices)


def test_walk_forward_predictions_do_not_use_future_rows():
    data = synthetic_market_data(180)
    config = hs.BacktestConfig(
        tickers=["AAA", "BBB", "CCC"],
        start="2020-01-01",
        end="2021-01-01",
        lookback=60,
        require_stationarity=False,
        use_hmm=False,
    )
    original = hs.walk_forward_predictions(data, config)
    cutoff = data.index[100]
    changed_close = data.close.copy()
    changed_close.loc[changed_close.index > cutoff] *= 10.0
    changed = hs.MarketData(
        open=data.open, high=data.high, low=data.low, close=changed_close
    )
    altered = hs.walk_forward_predictions(changed, config)
    left = original[original["date"] <= cutoff]["predicted_close"].to_numpy()
    right = altered[altered["date"] <= cutoff]["predicted_close"].to_numpy()
    np.testing.assert_allclose(left, right)


def test_dpa_coverage_confidence_and_costed_pnl():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    predictions = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAA"] * 3,
            "split": ["test"] * 3,
            "actual_return": [0.01, -0.02, 0.03],
            "signal": [1, -1, 0],
        }
    )
    score = hs.score_signals(
        predictions,
        "signal",
        split="test",
        commission_bps_per_side=0.0,
        slippage_bps_per_side=5.0,
        short_borrow_bps_per_day=0.5,
    )
    assert score["commission_bps_per_side"] == 0.0
    assert score["slippage_bps_per_side"] == 5.0
    assert score["predictions"] == 2
    assert score["coverage"] == 2 / 3
    assert score["dpa"] == 1.0
    assert score["dpa_ci_low"] < score["dpa_ci_high"] <= 1.0
    assert score["net_cumulative_return"] < score["gross_cumulative_return"]


def test_end_to_end_pipeline_includes_hmm_and_untouched_splits():
    data = synthetic_market_data(360)
    config = hs.BacktestConfig(
        tickers=["AAA", "BBB", "CCC"],
        start="2020-01-01",
        end="2022-01-01",
        lookback=80,
        require_stationarity=False,
        use_hmm=True,
        hmm_adaptive=True,
        hmm_finetune_window=60,
        hmm_finetune_iterations=2,
        random_seed=11,
    )
    result = hs.run_backtest(data, config)
    repeated = hs.run_backtest(data, config)
    assert set(result.predictions["split"]) == {"train", "validation", "test"}
    assert result.summary["validation_start"] < result.summary["test_start"]
    assert "hmm_regime_filter" in set(result.metrics["model"])
    assert result.summary["selected_model"] in {
        "hyperplane_baseline",
        "hmm_regime_filter",
        "no_trade",
    }
    test_models = set(result.metrics.loc[result.metrics["split"].eq("test"), "model"])
    assert test_models == {result.summary["selected_model"]}
    test_slippage = set(
        result.metrics.loc[result.metrics["split"].eq("test"), "slippage_bps_per_side"]
    )
    assert test_slippage == {0.0, 1.0, 2.0, 5.0}
    assert result.summary["selection_slippage_bps_per_side"] == 5.0
    assert result.summary["test_used_for_selection"] is False
    assert result.predictions["train_end"].lt(result.predictions["date"]).all()
    pd.testing.assert_series_equal(
        result.predictions["regime_state"], repeated.predictions["regime_state"]
    )
    pd.testing.assert_series_equal(
        result.predictions["hmm_signal"], repeated.predictions["hmm_signal"]
    )
    assert result.summary == repeated.summary


def test_adaptive_hmm_does_not_change_past_regimes_when_future_data_changes():
    data = synthetic_market_data(240)
    features = hs.build_regime_features(data)
    training_end = data.index[160]
    prediction_dates = list(data.index[160:201])
    original = hs.fit_adaptive_hmm_regimes(
        features,
        training_end=training_end,
        prediction_dates=prediction_dates,
        n_states=2,
        random_seed=13,
        finetune_window=60,
        finetune_iterations=2,
    )

    cutoff = data.index[180]
    changed_close = data.close.copy()
    changed_close.loc[changed_close.index > cutoff] *= 5.0
    changed = hs.MarketData(
        open=data.open, high=data.high, low=data.low, close=changed_close
    )
    altered = hs.fit_adaptive_hmm_regimes(
        hs.build_regime_features(changed),
        training_end=training_end,
        prediction_dates=prediction_dates,
        n_states=2,
        random_seed=13,
        finetune_window=60,
        finetune_iterations=2,
    )
    pd.testing.assert_frame_equal(
        original.loc[original.index <= cutoff],
        altered.loc[altered.index <= cutoff],
    )
