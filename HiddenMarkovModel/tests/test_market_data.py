import numpy as np
import pandas as pd
import pytest

import market_data as hs


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


def test_csv_market_data_round_trip_and_date_bounds(tmp_path):
    data = synthetic_market_data(25)
    for ticker in data.tickers:
        pd.DataFrame({name.title(): getattr(data, name)[ticker] for name in ("open", "high", "low", "close")}).to_csv(tmp_path / f"{ticker}.csv", index_label="Date")
    loaded = hs.load_csv_market_data(data.tickers, tmp_path, str(data.index[2].date()), str(data.index[-2].date()))
    for name in ("open", "high", "low", "close"):
        pd.testing.assert_frame_equal(getattr(loaded, name), getattr(data, name).iloc[2:-1], check_freq=False, check_names=False)


def test_frozen_hmm_probabilities_ignore_later_prices():
    data = synthetic_market_data(200)
    features = hs.build_regime_features(data).drop(columns="opening_gap")
    settings = dict(training_end=data.index[120], n_states=2, random_seed=42, include_probabilities=True)
    original = hs.fit_asof_hmm_regimes(features, **settings)
    changed = features.copy()
    changed.loc[data.index[160]:] *= 10
    repeated = hs.fit_asof_hmm_regimes(changed, **settings)
    pd.testing.assert_frame_equal(original.loc[:data.index[159]], repeated.loc[:data.index[159]])
    probabilities = original[["regime_probability_0", "regime_probability_1"]]
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert probabilities.ge(0).all().all() and probabilities.le(1).all().all()


def test_hmm_rejects_insufficient_training_history():
    features = hs.build_regime_features(synthetic_market_data(100))
    with pytest.raises(ValueError, match="Insufficient"):
        hs.fit_asof_hmm_regimes(features, training_end=features.index[30], n_states=2, random_seed=42)
