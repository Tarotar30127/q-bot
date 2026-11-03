"""Adjusted market data and causal Gaussian HMM filtering for the portfolio studies."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Mapping, Sequence
import warnings

import numpy as np
import pandas as pd

EPSILON = 1e-12


@dataclass(frozen=True)
class MarketData:
    """Aligned adjusted OHLC data; columns are ticker symbols."""

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(str(column) for column in self.close.columns)

    @property
    def index(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.close.index)

    def subset(self, start: str | None = None, end: str | None = None) -> "MarketData":
        mask = pd.Series(True, index=self.index)
        if start is not None:
            mask &= self.index >= pd.Timestamp(start)
        if end is not None:
            mask &= self.index <= pd.Timestamp(end)
        dates = self.index[mask.to_numpy()]
        return MarketData(
            open=self.open.loc[dates], high=self.high.loc[dates],
            low=self.low.loc[dates], close=self.close.loc[dates]
        )


def _numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(r"[$,]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def _assemble_market_data(
    fields: Mapping[str, Mapping[str, pd.Series]], tickers: Sequence[str]
) -> MarketData:
    frames: dict[str, pd.DataFrame] = {}
    for field_name in ("open", "high", "low", "close"):
        frame = pd.concat(
            {ticker: fields[field_name][ticker] for ticker in tickers}, axis=1, join="inner"
        )
        frame.columns = list(tickers)
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        frames[field_name] = frame.astype(float)

    combined = pd.concat(frames.values(), axis=1, join="inner").dropna()
    combined = combined[(combined > 0).all(axis=1)]
    dates = combined.index
    if len(dates) == 0:
        raise ValueError("No common positive OHLC observations were found.")
    return MarketData(
        open=frames["open"].loc[dates], high=frames["high"].loc[dates],
        low=frames["low"].loc[dates], close=frames["close"].loc[dates]
    )


def load_csv_market_data(
    tickers: Sequence[str], data_dir: str | Path,
    start: str | None = None, end: str | None = None
) -> MarketData:
    """Load one ``TICKER.csv`` file per symbol and align their OHLC rows."""

    directory = Path(data_dir)
    normalized = [ticker.strip().upper() for ticker in tickers]
    fields: dict[str, dict[str, pd.Series]] = {
        name: {} for name in ("open", "high", "low", "close")
    }
    for ticker in normalized:
        path = directory / f"{ticker}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing CSV for {ticker}: {path}")
        raw = pd.read_csv(path, sep=None, engine="python")
        columns = {str(column).strip().lower(): column for column in raw.columns}
        required = {"date", "open", "high", "low"}
        if not required.issubset(columns):
            missing = sorted(required - set(columns))
            raise ValueError(f"{path} is missing columns: {missing}")
        close_column = columns.get("close", columns.get("close/last"))
        if close_column is None:
            raise ValueError(f"{path} requires a Close or Close/Last column.")
        date_text = raw[columns["date"]].astype(str).str.strip()
        slash_dates = date_text[date_text.str.contains("/", regex=False)]
        day_first = False
        if not slash_dates.empty:
            first_number = pd.to_numeric(
                slash_dates.str.split("/").str[0], errors="coerce"
            )
            day_first = bool(first_number.gt(12).any())
        index = pd.to_datetime(
            date_text, format="mixed", dayfirst=day_first, errors="coerce"
        )
        mapping = {
            "open": columns["open"], "high": columns["high"],
            "low": columns["low"], "close": close_column
        }
        for field_name, column in mapping.items():
            values = _numeric(raw[column])
            fields[field_name][ticker] = pd.Series(values.to_numpy(), index=index, name=ticker)
    return _assemble_market_data(fields, normalized).subset(start, end)


def build_regime_features(data: MarketData) -> pd.DataFrame:
    """Build features known no later than each prediction day's market open."""

    log_close_return = np.log(data.close).diff()
    market_return = log_close_return.mean(axis=1)
    range_fraction = np.log(data.high / data.low).mean(axis=1)
    opening_gap = np.log(data.open / data.close.shift(1)).mean(axis=1)
    return pd.DataFrame({
        "lag_market_return": market_return.shift(1),
        "lag_20d_volatility": market_return.rolling(20, min_periods=10).std().shift(1),
        "lag_intraday_range": range_fraction.shift(1),
        "lag_cross_sectional_dispersion": log_close_return.std(axis=1).shift(1),
        "opening_gap": opening_gap,
    }, index=data.index).replace([np.inf, -np.inf], np.nan)


def _filtered_hmm_probabilities(model: object, observations: np.ndarray) -> np.ndarray:
    means = np.asarray(model.means_, dtype=float)
    covariance = np.asarray(model.covars_, dtype=float)
    if covariance.ndim == 3:
        covariance = np.diagonal(covariance, axis1=1, axis2=2)
    if covariance.ndim == 1:
        covariance = np.tile(covariance, (means.shape[0], 1))
    covariance = np.maximum(covariance, 1e-8)
    difference = observations[:, None, :] - means[None, :, :]
    log_emission = -0.5 * np.sum(
        np.log(2.0 * np.pi * covariance)[None, :, :]
        + difference**2 / covariance[None, :, :], axis=2
    )
    probabilities = np.zeros((len(observations), means.shape[0]), dtype=float)
    previous: np.ndarray | None = None
    for index, log_likelihood in enumerate(log_emission):
        prior = (
            np.asarray(model.startprob_, dtype=float)
            if previous is None else previous @ np.asarray(model.transmat_, dtype=float)
        )
        log_weight = np.log(np.maximum(prior, EPSILON)) + log_likelihood
        log_weight -= np.max(log_weight)
        current = np.exp(log_weight)
        current /= current.sum()
        probabilities[index] = current
        previous = current
    return probabilities


def _fit_hmm_silently(model: object, observations: np.ndarray) -> None:
    """Fit while suppressing negligible EM monotonicity warnings from hmmlearn."""

    logger = logging.getLogger("hmmlearn.base")
    previous_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(observations)
    finally:
        logger.setLevel(previous_level)


def fit_asof_hmm_regimes(
    features: pd.DataFrame, *, training_end: pd.Timestamp,
    n_states: int, random_seed: int, include_probabilities: bool = False
) -> pd.DataFrame:
    """Fit on pre-validation data and forward-filter all later observations."""

    from hmmlearn.hmm import GaussianHMM

    usable = features.dropna().copy()
    training = usable[usable.index < training_end]
    if len(training) < max(60, n_states * 20):
        raise ValueError("Insufficient pre-validation observations to fit the HMM.")
    location = training.mean(axis=0)
    scale = training.std(axis=0, ddof=0).replace(0.0, 1.0)
    standardized = ((usable - location) / scale).to_numpy(dtype=float)
    training_rows = int((usable.index < training_end).sum())
    model = GaussianHMM(
        n_components=n_states, covariance_type="diag", n_iter=300, tol=1e-4,
        min_covar=1e-6, random_state=random_seed, implementation="log"
    )
    _fit_hmm_silently(model, standardized[:training_rows])
    probabilities = _filtered_hmm_probabilities(model, standardized)
    output = pd.DataFrame({
        "regime_state": probabilities.argmax(axis=1).astype(int),
        "regime_probability": probabilities.max(axis=1),
    }, index=usable.index)
    if include_probabilities:
        for state in range(n_states):
            output[f"regime_probability_{state}"] = probabilities[:, state]
    return output
