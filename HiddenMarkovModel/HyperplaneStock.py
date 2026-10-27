# -*- coding: utf-8 -*-
"""Deterministic cross-stock mean-reversion research pipeline.

``forecast_current`` estimates fair values for supplied current prices.
``run_backtest`` produces strictly as-of walk-forward predictions, selects
parameters on validation data, and evaluates once on an untouched test set.

Importing this module performs no downloads, plotting, or computation.
This software is for research and education, not investment advice.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252
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


@dataclass(frozen=True)
class ModelForecast:
    predicted_prices: pd.Series
    valid: pd.Series
    stationarity_pvalues: pd.Series
    planarity: float


@dataclass
class BacktestConfig:
    tickers: list[str]
    start: str
    end: str
    lookback: int = 252
    model: str = "normalized"
    validation_start: str | None = None
    test_start: str | None = None
    require_stationarity: bool = True
    stationarity_alpha: float = 0.10
    signal_threshold_bps: float | None = None
    threshold_grid_bps: list[float] = field(
        default_factory=lambda: [
            100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 750.0, 1_000.0
        ]
    )
    commission_bps_per_side: float = 0.0
    slippage_grid_bps_per_side: list[float] = field(
        default_factory=lambda: [0.0, 1.0, 2.0, 5.0]
    )
    short_borrow_bps_per_day: float = 0.5
    minimum_validation_coverage: float = 0.20
    use_hmm: bool = True
    hmm_states: int = 2
    hmm_min_probability: float = 0.55
    hmm_adaptive: bool = False
    hmm_finetune_window: int = 126
    hmm_finetune_iterations: int = 10
    use_regime_selector: bool = False
    momentum_lookback: int = 20
    momentum_threshold_bps: float = 50.0
    selector_min_train_days: int = 30
    selector_confidence_z: float = 1.96
    selector_min_exposure: float = 0.10
    random_seed: int = 42
    data_dir: str | None = None
    output_dir: str = "output/hyperplane_backtest"

    def __post_init__(self) -> None:
        self.tickers = [ticker.strip().upper() for ticker in self.tickers]
        if len(self.tickers) < 3:
            raise ValueError("At least three tickers are required.")
        if len(set(self.tickers)) != len(self.tickers):
            raise ValueError("Ticker symbols must be unique.")
        if self.lookback < 60:
            raise ValueError("lookback must be at least 60 trading days.")
        if self.model not in {"normalized", "cointegration"}:
            raise ValueError("model must be 'normalized' or 'cointegration'.")
        if not 0 < self.stationarity_alpha <= 1:
            raise ValueError("stationarity_alpha must be in (0, 1].")
        if not self.threshold_grid_bps:
            raise ValueError("threshold_grid_bps must not be empty.")
        if any(threshold < 0 for threshold in self.threshold_grid_bps):
            raise ValueError("threshold_grid_bps values must be non-negative.")
        if self.commission_bps_per_side < 0:
            raise ValueError("commission_bps_per_side must be non-negative.")
        if not self.slippage_grid_bps_per_side:
            raise ValueError("slippage_grid_bps_per_side must not be empty.")
        if any(value < 0 for value in self.slippage_grid_bps_per_side):
            raise ValueError("slippage_grid_bps_per_side values must be non-negative.")
        if self.short_borrow_bps_per_day < 0:
            raise ValueError("short_borrow_bps_per_day must be non-negative.")
        if not 0 <= self.minimum_validation_coverage <= 1:
            raise ValueError("minimum_validation_coverage must be in [0, 1].")
        if self.hmm_states < 2:
            raise ValueError("hmm_states must be at least 2.")
        if not 0 <= self.hmm_min_probability <= 1:
            raise ValueError("hmm_min_probability must be in [0, 1].")
        if self.hmm_finetune_window < 60:
            raise ValueError("hmm_finetune_window must be at least 60 trading days.")
        if self.hmm_finetune_iterations < 1:
            raise ValueError("hmm_finetune_iterations must be positive.")
        if self.use_regime_selector and not self.use_hmm:
            raise ValueError("use_regime_selector requires use_hmm=True.")
        if self.momentum_lookback < 2:
            raise ValueError("momentum_lookback must be at least 2 trading days.")
        if not math.isfinite(self.momentum_threshold_bps) or self.momentum_threshold_bps < 0:
            raise ValueError("momentum_threshold_bps must be finite and non-negative.")
        if self.selector_min_train_days < 2:
            raise ValueError("selector_min_train_days must be at least 2.")
        if not math.isfinite(self.selector_confidence_z) or self.selector_confidence_z < 0:
            raise ValueError("selector_confidence_z must be finite and non-negative.")
        if not math.isfinite(self.selector_min_exposure) or not 0 < self.selector_min_exposure <= 1:
            raise ValueError("selector_min_exposure must be in (0, 1].")


@dataclass
class PipelineResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    summary: dict[str, object]
    config: BacktestConfig


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


def download_market_data(
    tickers: Sequence[str], start: str, end: str, *, repair: bool = True
) -> MarketData:
    """Download adjusted daily OHLC data; ``end`` is treated as inclusive."""

    import yfinance as yf

    normalized = [ticker.strip().upper() for ticker in tickers]
    download_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(
        tickers=normalized, start=start, end=download_end, interval="1d",
        auto_adjust=True, repair=repair, actions=False, progress=False,
        threads=False, group_by="ticker", multi_level_index=True
    )
    if raw is None or raw.empty:
        raise ValueError("Yahoo Finance returned no data.")

    fields: dict[str, dict[str, pd.Series]] = {
        name: {} for name in ("open", "high", "low", "close")
    }
    for ticker in normalized:
        for field_name in fields:
            display_name = field_name.title()
            series: pd.Series | None = None
            if isinstance(raw.columns, pd.MultiIndex):
                if ticker in raw.columns.get_level_values(0):
                    block = raw[ticker]
                    if display_name in block.columns:
                        series = block[display_name]
                if series is None and display_name in raw.columns.get_level_values(0):
                    block = raw[display_name]
                    if ticker in block.columns:
                        series = block[ticker]
            elif len(normalized) == 1 and display_name in raw.columns:
                series = raw[display_name]
            if series is None:
                raise ValueError(f"Downloaded data is missing {display_name} for {ticker}.")
            fields[field_name][ticker] = series.rename(ticker)
    return _assemble_market_data(fields, normalized).subset(start, end)


def load_market_data(config: BacktestConfig) -> MarketData:
    if config.data_dir:
        return load_csv_market_data(config.tickers, config.data_dir, config.start, config.end)
    return download_market_data(config.tickers, config.start, config.end)


def _adf_pvalue(residual: np.ndarray) -> float:
    residual = np.asarray(residual, dtype=float)
    residual = residual[np.isfinite(residual)]
    if len(residual) < 30 or np.std(residual) < EPSILON:
        return 1.0
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(
            residual, maxlag=1, regression="c", autolag=None, result_object=True
        )
        return float(result.pvalue)
    except (ValueError, np.linalg.LinAlgError):
        return 1.0


def _normalized_plane_forecast(
    train_close: pd.DataFrame, current_prices: pd.Series, *,
    require_stationarity: bool, stationarity_alpha: float
) -> ModelForecast:
    if (train_close <= 0).any().any() or (current_prices <= 0).any():
        raise ValueError("Prices must be positive for log normalization.")
    log_prices = np.log(train_close.astype(float))
    location = log_prices.mean(axis=0)
    scale = log_prices.std(axis=0, ddof=0).clip(lower=EPSILON)
    normalized = (log_prices - location) / scale
    centroid = normalized.mean(axis=0).to_numpy()
    centered = normalized.to_numpy() - centroid
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1] / np.linalg.norm(vt[-1])
    current_normalized = (np.log(current_prices.astype(float)) - location) / scale
    signed_distance = float((current_normalized.to_numpy() - centroid) @ normal)
    projected = current_normalized.to_numpy() - signed_distance * normal
    predicted = np.exp(projected * scale.to_numpy() + location.to_numpy())
    residual = centered @ normal
    pvalue = _adf_pvalue(residual)
    valid_value = (not require_stationarity) or pvalue <= stationarity_alpha
    variance = singular_values**2
    planarity = float(1.0 - variance[-1] / variance.sum()) if variance.sum() else 0.0
    return ModelForecast(
        predicted_prices=pd.Series(predicted, index=train_close.columns, dtype=float),
        valid=pd.Series(valid_value, index=train_close.columns, dtype=bool),
        stationarity_pvalues=pd.Series(pvalue, index=train_close.columns, dtype=float),
        planarity=planarity,
    )


def _cointegration_forecast(
    train_close: pd.DataFrame, current_prices: pd.Series, *,
    require_stationarity: bool, stationarity_alpha: float
) -> ModelForecast:
    if (train_close <= 0).any().any() or (current_prices <= 0).any():
        raise ValueError("Prices must be positive for log cointegration regressions.")
    log_prices = np.log(train_close.astype(float))
    current_log = np.log(current_prices.astype(float))
    predicted: dict[str, float] = {}
    pvalues: dict[str, float] = {}
    validity: dict[str, bool] = {}
    r_squared: list[float] = []
    for ticker in train_close.columns:
        others = [column for column in train_close.columns if column != ticker]
        x = np.column_stack([np.ones(len(log_prices)), log_prices[others].to_numpy()])
        y = log_prices[ticker].to_numpy()
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        fitted = x @ beta
        residual = y - fitted
        total_ss = float(np.sum((y - y.mean()) ** 2))
        residual_ss = float(np.sum(residual**2))
        r_squared.append(1.0 - residual_ss / total_ss if total_ss else 0.0)
        pvalue = _adf_pvalue(residual)
        fair_log_price = float(np.r_[1.0, current_log[others].to_numpy(dtype=float)] @ beta)
        predicted[ticker] = math.exp(fair_log_price)
        pvalues[ticker] = pvalue
        validity[ticker] = (not require_stationarity) or pvalue <= stationarity_alpha
    return ModelForecast(
        predicted_prices=pd.Series(predicted).reindex(train_close.columns),
        valid=pd.Series(validity, dtype=bool).reindex(train_close.columns),
        stationarity_pvalues=pd.Series(pvalues).reindex(train_close.columns),
        planarity=float(np.mean(r_squared)),
    )


def forecast_prices(
    train_close: pd.DataFrame, current_prices: pd.Series, *,
    model: str = "normalized", require_stationarity: bool = True,
    stationarity_alpha: float = 0.10
) -> ModelForecast:
    """Fit on historical closes and forecast from an as-of price vector."""

    current = current_prices.reindex(train_close.columns)
    if current.isna().any():
        missing = current.index[current.isna()].tolist()
        raise ValueError(f"Current prices are missing for: {missing}")
    if model == "normalized":
        return _normalized_plane_forecast(
            train_close, current, require_stationarity=require_stationarity,
            stationarity_alpha=stationarity_alpha
        )
    if model == "cointegration":
        return _cointegration_forecast(
            train_close, current, require_stationarity=require_stationarity,
            stationarity_alpha=stationarity_alpha
        )
    raise ValueError("model must be 'normalized' or 'cointegration'.")


def forecast_current(
    data: MarketData, current_prices: Mapping[str, float] | Sequence[float], *,
    lookback: int = 252, model: str = "normalized",
    require_stationarity: bool = True, stationarity_alpha: float = 0.10
) -> pd.DataFrame:
    """Forecast fair values for a supplied current price vector."""

    if len(data.close) < lookback:
        raise ValueError(f"Need {lookback} rows; only {len(data.close)} are available.")
    if isinstance(current_prices, Mapping):
        current = pd.Series({ticker: float(current_prices[ticker]) for ticker in data.tickers})
    else:
        values = list(current_prices)
        if len(values) != len(data.tickers):
            raise ValueError("The number of prices must match the number of tickers.")
        current = pd.Series(values, index=data.tickers, dtype=float)
    result = forecast_prices(
        data.close.iloc[-lookback:], current, model=model,
        require_stationarity=require_stationarity,
        stationarity_alpha=stationarity_alpha
    )
    frame = pd.DataFrame({
        "current_price": current,
        "predicted_fair_price": result.predicted_prices,
        "predicted_change": result.predicted_prices - current,
        "predicted_return": result.predicted_prices / current - 1.0,
        "signal": np.sign(result.predicted_prices - current).astype(int),
        "model_valid": result.valid,
        "stationarity_pvalue": result.stationarity_pvalues,
        "planarity": result.planarity,
    })
    frame.index.name = "ticker"
    return frame.reset_index()


def walk_forward_predictions(data: MarketData, config: BacktestConfig) -> pd.DataFrame:
    """Generate one-day open-to-close predictions without future information."""

    if len(data.close) <= config.lookback + 20:
        raise ValueError(f"Need more than lookback + 20 rows; got {len(data.close)} rows.")
    rows: list[dict[str, object]] = []
    momentum = data.close.shift(1) / data.close.shift(config.momentum_lookback + 1) - 1.0
    for position in range(config.lookback, len(data.close)):
        prediction_date = data.index[position]
        training = data.close.iloc[position - config.lookback:position]
        current_open = data.open.iloc[position]
        actual_close = data.close.iloc[position]
        current_momentum = momentum.iloc[position]
        try:
            forecast = forecast_prices(
                training, current_open, model=config.model,
                require_stationarity=config.require_stationarity,
                stationarity_alpha=config.stationarity_alpha
            )
        except (ValueError, np.linalg.LinAlgError):
            forecast = ModelForecast(
                predicted_prices=pd.Series(np.nan, index=data.tickers),
                valid=pd.Series(False, index=data.tickers),
                stationarity_pvalues=pd.Series(np.nan, index=data.tickers),
                planarity=np.nan,
            )
        for ticker in data.tickers:
            opening_price = float(current_open[ticker])
            closing_price = float(actual_close[ticker])
            predicted_price = float(forecast.predicted_prices[ticker])
            predicted_return = (
                predicted_price / opening_price - 1.0
                if np.isfinite(predicted_price) and opening_price > 0 else np.nan
            )
            actual_return = closing_price / opening_price - 1.0
            momentum_return = float(current_momentum[ticker])
            rows.append({
                "date": prediction_date, "ticker": ticker,
                "train_start": training.index[0], "train_end": training.index[-1],
                "open": opening_price, "actual_close": closing_price,
                "predicted_close": predicted_price, "predicted_return": predicted_return,
                "actual_return": actual_return,
                "raw_signal": int(np.sign(predicted_return))
                if np.isfinite(predicted_return) and bool(forecast.valid[ticker]) else 0,
                "model_valid": bool(forecast.valid[ticker]),
                "stationarity_pvalue": float(forecast.stationarity_pvalues[ticker]),
                "planarity": float(forecast.planarity),
                "momentum_return": momentum_return,
                "momentum_signal": int(np.sign(momentum_return))
                if np.isfinite(momentum_return)
                and abs(momentum_return) >= config.momentum_threshold_bps / 10_000.0
                else 0,
            })
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


def _assign_splits(
    predictions: pd.DataFrame, config: BacktestConfig
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(predictions["date"]).unique()))
    if len(dates) < 60:
        raise ValueError("At least 60 walk-forward dates are required for splitting.")
    if config.validation_start is None and config.test_start is None:
        validation_start = dates[max(1, int(len(dates) * 0.60))]
        test_start = dates[max(2, int(len(dates) * 0.80))]
    elif config.validation_start and config.test_start:
        validation_start = pd.Timestamp(config.validation_start)
        test_start = pd.Timestamp(config.test_start)
    else:
        raise ValueError("Specify both validation_start and test_start, or neither.")
    if not dates.min() < validation_start < test_start <= dates.max():
        raise ValueError(
            "Split dates must satisfy first_prediction < validation_start < "
            "test_start <= last_prediction."
        )
    output = predictions.copy()
    output["split"] = np.select(
        [output["date"] < validation_start, output["date"] < test_start],
        ["train", "validation"], default="test"
    )
    counts = output.groupby("split")["date"].nunique()
    for split_name in ("train", "validation", "test"):
        if counts.get(split_name, 0) < 10:
            raise ValueError(f"The {split_name} split must contain at least 10 dates.")
    return output, validation_start, test_start


def _wilson_interval(correct: int, total: int, z: float = 1.95996398454) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
    proportion = correct / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z**2 / (4.0 * total**2)
    ) / denominator
    return center - radius, center + radius


def _maximum_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    return float((wealth / wealth.cummax().clip(lower=1.0) - 1.0).min())


def score_signals(
    predictions: pd.DataFrame, signal_column: str, *, split: str,
    commission_bps_per_side: float, slippage_bps_per_side: float,
    short_borrow_bps_per_day: float,
    ticker: str | None = None
) -> dict[str, float | int | str]:
    subset = predictions[predictions["split"] == split].copy()
    if ticker is not None:
        subset = subset[subset["ticker"] == ticker].copy()
    if subset.empty:
        raise ValueError(f"No rows for split={split!r}, ticker={ticker!r}.")
    signal = pd.to_numeric(subset[signal_column], errors="coerce").fillna(0).astype(float)
    if not np.isfinite(signal).all():
        raise ValueError("Positions must be finite.")
    direction = np.sign(signal)
    actual = pd.to_numeric(subset["actual_return"], errors="coerce")
    active = signal.ne(0) & actual.notna()
    correct_mask = active & direction.eq(np.sign(actual))
    prediction_count = int(active.sum())
    correct = int(correct_mask.sum())
    total = len(subset)
    dpa = correct / prediction_count if prediction_count else np.nan
    ci_low, ci_high = _wilson_interval(correct, prediction_count)
    up = active & actual.gt(0)
    down = active & actual.lt(0)
    up_recall = float(direction[up].eq(1).mean()) if up.any() else np.nan
    down_recall = float(direction[down].eq(-1).mean()) if down.any() else np.nan
    valid_recalls = [value for value in (up_recall, down_recall) if np.isfinite(value)]
    balanced_accuracy = float(np.mean(valid_recalls)) if valid_recalls else np.nan
    subset["_gross_return"] = signal * actual.fillna(0.0)
    round_trip_cost = 2.0 * (
        commission_bps_per_side + slippage_bps_per_side
    ) / 10_000.0
    subset["_cost"] = signal.abs() * round_trip_cost
    subset["_cost"] += (-signal).clip(lower=0.0) * short_borrow_bps_per_day / 10_000.0
    subset["_net_return"] = subset["_gross_return"] - subset["_cost"]
    daily = subset.groupby("date")[["_gross_return", "_net_return"]].mean()
    gross_cumulative = float((1.0 + daily["_gross_return"]).prod() - 1.0)
    net_cumulative = float((1.0 + daily["_net_return"]).prod() - 1.0)
    net_std = float(daily["_net_return"].std(ddof=1))
    sharpe = (
        float(math.sqrt(TRADING_DAYS_PER_YEAR) * daily["_net_return"].mean() / net_std)
        if net_std > EPSILON else np.nan
    )
    return {
        "split": split, "scope": ticker or "portfolio", "observations": int(total),
        "commission_bps_per_side": commission_bps_per_side,
        "slippage_bps_per_side": slippage_bps_per_side,
        "short_borrow_bps_per_day": short_borrow_bps_per_day,
        # Keep the legacy field, but make clear that it counts trades, not
        # forecasts generated by the underlying price model.
        "predictions": prediction_count, "active_positions": prediction_count,
        "coverage": prediction_count / total,
        "correct": correct, "dpa": dpa, "dpa_ci_low": ci_low,
        "dpa_ci_high": ci_high, "balanced_accuracy": balanced_accuracy,
        "up_recall": up_recall, "down_recall": down_recall,
        "gross_cumulative_return": gross_cumulative,
        "net_cumulative_return": net_cumulative, "annualized_sharpe": sharpe,
        "max_drawdown": _maximum_drawdown(daily["_net_return"]),
        "average_gross_exposure": float(signal.abs().mean()),
    }


def _signal_for_threshold(predictions: pd.DataFrame, threshold_bps: float) -> pd.Series:
    threshold = threshold_bps / 10_000.0
    valid = predictions["model_valid"].astype(bool) & predictions["predicted_return"].notna()
    active = valid & predictions["predicted_return"].abs().ge(threshold)
    return pd.Series(
        np.where(active, np.sign(predictions["predicted_return"]), 0),
        index=predictions.index, dtype=int
    )


def _select_threshold(
    predictions: pd.DataFrame, config: BacktestConfig, *, split: str = "validation"
) -> tuple[
    float, dict[str, float | int | str],
    list[dict[str, object]],
]:
    thresholds = (
        [float(config.signal_threshold_bps)]
        if config.signal_threshold_bps is not None
        else [float(value) for value in config.threshold_grid_bps]
    )
    selection_slippage = max(config.slippage_grid_bps_per_side)
    candidates: list[
        tuple[
            float,
            dict[str, float | int | str],
            dict[float, dict[str, float | int | str]],
        ]
    ] = []
    for threshold in thresholds:
        candidate = predictions.copy()
        candidate["_candidate_signal"] = _signal_for_threshold(candidate, threshold)
        scenario_scores = {
            float(slippage): score_signals(
                candidate, "_candidate_signal", split=split,
                commission_bps_per_side=config.commission_bps_per_side,
                slippage_bps_per_side=float(slippage),
                short_borrow_bps_per_day=config.short_borrow_bps_per_day,
            )
            for slippage in config.slippage_grid_bps_per_side
        }
        candidates.append(
            (threshold, scenario_scores[selection_slippage], scenario_scores)
        )

    covered = [
        item for item in candidates
        if float(item[1]["coverage"]) >= config.minimum_validation_coverage
    ]
    pool = covered or candidates

    def selection_key(
        item: tuple[
            float,
            dict[str, float | int | str],
            dict[float, dict[str, float | int | str]],
        ]
    ) -> tuple[float, float, float]:
        score = item[1]
        dpa = float(score["dpa"]) if np.isfinite(score["dpa"]) else -np.inf
        return (
            float(score["net_cumulative_return"]), dpa, float(score["coverage"])
        )

    selected_threshold, selected_score, _ = max(pool, key=selection_key)
    diagnostics = [
        {
            "threshold_bps": threshold,
            "selection_slippage_bps_per_side": selection_slippage,
            "worst_case_net_cumulative_return": float(
                selection_score["net_cumulative_return"]
            ),
            "dpa": float(selection_score["dpa"]),
            "coverage": float(selection_score["coverage"]),
            "meets_minimum_coverage": (
                float(selection_score["coverage"])
                >= config.minimum_validation_coverage
            ),
            "slippage_scenarios": [
                {
                    "slippage_bps_per_side": slippage,
                    "net_cumulative_return": float(score["net_cumulative_return"]),
                }
                for slippage, score in sorted(scenario_scores.items())
            ],
        }
        for threshold, selection_score, scenario_scores in candidates
    ]
    return selected_threshold, selected_score, diagnostics


def _add_baseline_signals(predictions: pd.DataFrame) -> pd.DataFrame:
    output = predictions.copy()
    output["always_up_signal"] = 1
    output["always_down_signal"] = -1
    previous = output.groupby("ticker", sort=False)["actual_return"].shift(1)
    output["previous_direction_signal"] = np.sign(previous.fillna(0.0)).astype(int)
    return output


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


def fit_adaptive_hmm_regimes(
    features: pd.DataFrame, *, training_end: pd.Timestamp,
    prediction_dates: Sequence[pd.Timestamp], n_states: int,
    random_seed: int, finetune_window: int, finetune_iterations: int,
    include_probabilities: bool = False, align_states: bool = False
) -> pd.DataFrame:
    """Warm-start and fine-tune HMM parameters using only prior observations."""

    from hmmlearn.hmm import GaussianHMM

    usable = features.dropna().copy()
    training = usable[usable.index < training_end]
    minimum_rows = max(60, n_states * 20)
    if len(training) < minimum_rows:
        raise ValueError("Insufficient pre-validation observations to fit the HMM.")

    location = training.mean(axis=0)
    scale = training.std(axis=0, ddof=0).replace(0.0, 1.0)
    standardized = (usable - location) / scale
    base_model = GaussianHMM(
        n_components=n_states, covariance_type="diag", n_iter=300, tol=1e-4,
        min_covar=1e-6, random_state=random_seed, implementation="log"
    )
    _fit_hmm_silently(
        base_model, standardized.loc[training.index].to_numpy(dtype=float)
    )

    rows: list[dict[str, float | int | pd.Timestamp]] = []
    for prediction_date in sorted(pd.Timestamp(date) for date in prediction_dates):
        if prediction_date not in standardized.index:
            continue
        recent = standardized[standardized.index < prediction_date].tail(finetune_window)
        if len(recent) < minimum_rows:
            continue

        adaptive_model = copy.deepcopy(base_model)
        adaptive_model.n_iter = finetune_iterations
        adaptive_model.init_params = ""
        # Keep the initial-state distribution fixed while updating transition
        # probabilities and Gaussian emission means/covariances.
        adaptive_model.params = "tmc"
        recent_values = recent.to_numpy(dtype=float)
        _fit_hmm_silently(adaptive_model, recent_values)

        current = standardized.loc[[prediction_date]].to_numpy(dtype=float)
        filtered = _filtered_hmm_probabilities(
            adaptive_model, np.vstack([recent_values, current])
        )[-1]
        if align_states:
            # Adaptive EM may permute state labels. Map probabilities back to
            # the base model's labels before looking up stored expert policies.
            from scipy.optimize import linear_sum_assignment
            distance = np.sum(
                (base_model.means_[:, None, :] - adaptive_model.means_[None, :, :]) ** 2,
                axis=2,
            )
            base_indices, adaptive_indices = linear_sum_assignment(distance)
            aligned = np.zeros_like(filtered)
            aligned[base_indices] = filtered[adaptive_indices]
            filtered = aligned
        row = {
            "date": prediction_date,
            "regime_state": int(filtered.argmax()),
            "regime_probability": float(filtered.max()),
        }
        if include_probabilities:
            row.update({f"regime_probability_{state}": float(filtered[state])
                        for state in range(n_states)})
        rows.append(row)

    if not rows:
        raise ValueError("No prediction dates had enough prior data for HMM fine-tuning.")
    return pd.DataFrame(rows).set_index("date")


def _selector_probabilities(predictions: pd.DataFrame, n_states: int) -> pd.DataFrame:
    columns = [f"regime_probability_{state}" for state in range(n_states)]
    probabilities = predictions.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(probabilities).all(axis=1) & probabilities.ge(0.0).all(axis=1)
    total = probabilities.sum(axis=1)
    valid &= total.gt(EPSILON)
    return probabilities.div(total.where(valid), axis=0).fillna(0.0)


def fit_regime_policy_library(
    predictions: pd.DataFrame, config: BacktestConfig
) -> tuple[dict[int, str], list[dict[str, object]]]:
    """Calibrate a frozen expert library on pre-validation outcomes only.

    Each state retains the expert with the highest positive approximate
    lower confidence bound on daily net portfolio return. HAC uncertainty
    uses calendar-day lags rather than treating correlated tickers as
    independent observations. Unproven states retain cash.
    """

    training = predictions.loc[predictions["split"].eq("train")].copy()
    policies = {state: "cash" for state in range(config.hmm_states)}
    diagnostics: list[dict[str, object]] = []
    if training.empty:
        return policies, diagnostics
    probability_columns = [f"regime_probability_{state}" for state in range(config.hmm_states)]
    training[probability_columns] = _selector_probabilities(training, config.hmm_states)
    daily = training.groupby("date", sort=True)[
        ["regime_state", "regime_probability", *probability_columns]
    ].first()
    experts = {"mean_reversion": "selector_mean_reversion_signal", "momentum": "momentum_signal"}
    round_trip_cost = 2.0 * (
        config.commission_bps_per_side + max(config.slippage_grid_bps_per_side)
    ) / 10_000.0
    actual = pd.to_numeric(training["actual_return"], errors="coerce").fillna(0.0)
    expert_returns: dict[str, pd.Series] = {}
    expert_active: dict[str, pd.Series] = {}
    for expert, column in experts.items():
        position = pd.to_numeric(training[column], errors="coerce").fillna(0.0)
        net = position * actual - position.abs() * round_trip_cost
        net -= (-position).clip(lower=0.0) * config.short_borrow_bps_per_day / 10_000.0
        expert_returns[expert] = net.groupby(training["date"]).mean().reindex(daily.index)
        expert_active[expert] = position.abs().gt(EPSILON).groupby(training["date"]).any().reindex(daily.index)
    for state in range(config.hmm_states):
        confident = daily["regime_state"].eq(state) & daily["regime_probability"].ge(config.hmm_min_probability)
        weights = daily[f"regime_probability_{state}"].where(confident, 0.0).to_numpy(dtype=float)
        total_weight = float(weights.sum())
        best_bound = 0.0
        state_diagnostics: list[dict[str, object]] = []
        for expert in experts:
            returns = expert_returns[expert].to_numpy(dtype=float)
            active_days = int((confident & expert_active[expert]).sum())
            mean = float(weights @ returns / total_weight) if total_weight > EPSILON else 0.0
            influence = weights * (returns - mean)
            variance = float(influence @ influence)
            lag_count = min(5, len(influence) - 1)
            for lag in range(1, lag_count + 1):
                variance += 2.0 * (1.0 - lag / (lag_count + 1.0)) * float(
                    influence[lag:] @ influence[:-lag]
                )
            standard_error = math.sqrt(max(variance, 0.0)) / total_weight if total_weight > EPSILON else 0.0
            bound = mean - config.selector_confidence_z * standard_error
            eligible = total_weight > EPSILON and active_days >= config.selector_min_train_days
            if eligible and bound > best_bound:
                policies[state] = expert
                best_bound = bound
            state_diagnostics.append({
                "state": state, "expert": expert, "active_days": active_days,
                "regime_days": int(confident.sum()), "mean_net_daily_return": mean,
                "standard_error": standard_error, "lower_bound": bound,
                "eligible": eligible,
            })
        for diagnostic in state_diagnostics:
            diagnostic["selected"] = diagnostic["expert"] == policies[state]
        diagnostics.extend(state_diagnostics)
    return policies, diagnostics


def apply_regime_selector(
    predictions: pd.DataFrame, policies: Mapping[int, str], config: BacktestConfig
) -> pd.DataFrame:
    """Mix training-selected experts using today's causal HMM posterior."""

    output = predictions.copy()
    probabilities = _selector_probabilities(output, config.hmm_states)
    mean_reversion_weight = pd.Series(0.0, index=output.index)
    momentum_weight = pd.Series(0.0, index=output.index)
    for state in range(config.hmm_states):
        probability = probabilities[f"regime_probability_{state}"]
        expert = policies.get(state, "cash")
        if expert == "mean_reversion":
            mean_reversion_weight += probability
        elif expert == "momentum":
            momentum_weight += probability
    position = (
        mean_reversion_weight * output["selector_mean_reversion_signal"].fillna(0.0)
        + momentum_weight * output["momentum_signal"].fillna(0.0)
    ).clip(-1.0, 1.0)
    valid_probabilities = probabilities.sum(axis=1).gt(EPSILON)
    trade = output["regime_probability"].ge(config.hmm_min_probability)
    trade &= valid_probabilities & position.abs().ge(config.selector_min_exposure)
    has_trading_policy = any(
        policies.get(state) in {"mean_reversion", "momentum"}
        for state in range(config.hmm_states)
    )
    output["regime_selector_evaluation_status"] = np.select(
        [~valid_probabilities,
         pd.Series(not has_trading_policy, index=output.index),
         output["regime_probability"].lt(config.hmm_min_probability),
         ~trade],
        ["cash_missing_regime_probabilities", "cash_no_trading_policy",
         "cash_low_regime_confidence", "cash_below_min_exposure"],
        default="active_position",
    )
    mean_reversion_weight = mean_reversion_weight.where(trade, 0.0)
    momentum_weight = momentum_weight.where(trade, 0.0)
    output["regime_selector_signal"] = position.where(trade, 0.0)
    output["selector_mean_reversion_weight"] = mean_reversion_weight
    output["selector_momentum_weight"] = momentum_weight
    output["selector_cash_weight"] = (1.0 - mean_reversion_weight - momentum_weight).clip(0.0, 1.0)
    # Opposing experts and inactive signals leave additional actual cash.
    output["selector_cash_allocation"] = 1.0 - output["regime_selector_signal"].abs()
    return output


def _regime_selector_evaluation_report(predictions: pd.DataFrame, config: BacktestConfig) -> dict[str, object]:
    """Distinguish computed forecasts, cash decisions, and skipped holdout work."""

    if "regime_selector_evaluation_status" not in predictions:
        return {}
    report: dict[str, object] = {}
    for split, subset in predictions.groupby("split", sort=False):
        skipped = subset["regime_selector_evaluation_status"].eq("not_evaluated_on_test")
        active = subset["regime_selector_signal"].fillna(0.0).ne(0.0)
        probabilities = _selector_probabilities(subset, config.hmm_states)
        report[str(split)] = {
            "status": "not_evaluated" if skipped.all() else "evaluated",
            "observations": len(subset),
            "forecasts_computed": int(np.isfinite(subset["predicted_return"]).sum()),
            "regime_probability_rows": int(probabilities.sum(axis=1).gt(EPSILON).sum()),
            "mean_reversion_signal_rows": int(subset["selector_mean_reversion_signal"].ne(0.0).sum()),
            "momentum_signal_rows": int(subset["momentum_signal"].ne(0.0).sum()),
            "active_positions": None if skipped.all() else int(active.sum()),
            "coverage": None if skipped.all() else float(active.mean()),
            "decision_counts": {
                str(reason): int(count)
                for reason, count in subset["regime_selector_evaluation_status"].value_counts().items()
            },
        }
    return report


def _choose_hmm_states(predictions: pd.DataFrame, config: BacktestConfig) -> list[int]:
    selection_slippage = max(config.slippage_grid_bps_per_side)
    states = [
        int(state)
        for state in sorted(predictions["regime_state"].dropna().astype(int).unique())
    ]
    qualified: list[int] = []
    scored: list[tuple[float, float, int, int]] = []
    for state in states:
        candidate = predictions.copy()
        gate = candidate["regime_state"].eq(state) & candidate["regime_probability"].ge(
            config.hmm_min_probability
        )
        candidate["_state_signal"] = np.where(gate, candidate["base_signal"], 0)
        score = score_signals(
            candidate, "_state_signal", split="validation",
            commission_bps_per_side=config.commission_bps_per_side,
            slippage_bps_per_side=selection_slippage,
            short_borrow_bps_per_day=config.short_borrow_bps_per_day,
        )
        dpa = float(score["dpa"]) if np.isfinite(score["dpa"]) else -np.inf
        net_return = float(score["net_cumulative_return"])
        prediction_count = int(score["predictions"])
        scored.append((net_return, dpa, prediction_count, state))
        if net_return > 0 and prediction_count >= 5:
            qualified.append(state)
    if qualified:
        return qualified
    return [max(scored)[-1]] if scored else []


def _metric_rows(
    predictions: pd.DataFrame, models: Mapping[str, str], config: BacktestConfig,
    *, selected_model: str
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, signal_column in models.items():
        # Every candidate is evaluated on validation. Only the strategy chosen
        # on validation is allowed to touch the final test split.
        splits = ["validation"]
        if model_name == selected_model:
            splits.append("test")
        for split in splits:
            for slippage in config.slippage_grid_bps_per_side:
                portfolio = score_signals(
                    predictions, signal_column, split=split,
                    commission_bps_per_side=config.commission_bps_per_side,
                    slippage_bps_per_side=float(slippage),
                    short_borrow_bps_per_day=config.short_borrow_bps_per_day,
                )
                portfolio["model"] = model_name
                rows.append(portfolio)
                for ticker in config.tickers:
                    ticker_score = score_signals(
                        predictions, signal_column, split=split,
                        commission_bps_per_side=config.commission_bps_per_side,
                        slippage_bps_per_side=float(slippage),
                        short_borrow_bps_per_day=config.short_borrow_bps_per_day,
                        ticker=ticker,
                    )
                    ticker_score["model"] = model_name
                    rows.append(ticker_score)
    columns = ["model", "split", "scope"] + [
        column for column in rows[0] if column not in {"model", "split", "scope"}
    ]
    return pd.DataFrame(rows)[columns]


def run_backtest(data: MarketData, config: BacktestConfig) -> PipelineResult:
    """Run the leakage-controlled research pipeline."""

    if tuple(config.tickers) != data.tickers:
        raise ValueError(f"Config tickers {config.tickers} do not match data {data.tickers}.")
    predictions = walk_forward_predictions(data, config)
    predictions, validation_start, test_start = _assign_splits(predictions, config)
    (
        selected_threshold,
        threshold_validation_score,
        threshold_validation_candidates,
    ) = _select_threshold(predictions, config)
    predictions["base_signal"] = _signal_for_threshold(predictions, selected_threshold)
    predictions = _add_baseline_signals(predictions)

    selector_status = "disabled"
    selector_policies: dict[int, str] = {}
    selector_diagnostics: list[dict[str, object]] = []
    selector_threshold: float | None = None
    if config.use_regime_selector:
        # This expert's threshold and policy assignments must not inherit a
        # threshold selected using validation outcomes.
        selector_threshold, _, _ = _select_threshold(predictions, config, split="train")
        predictions["selector_mean_reversion_signal"] = _signal_for_threshold(
            predictions, selector_threshold
        )
        selector_status = "waiting for HMM"

    hmm_status = "disabled"
    hmm_allowed_states: list[int] = []
    hmm_available = False
    predictions["regime_state"] = np.nan
    predictions["regime_probability"] = np.nan
    predictions["hmm_signal"] = 0
    if config.use_hmm:
        try:
            features = build_regime_features(data)
            if config.hmm_adaptive:
                prediction_dates = predictions.loc[
                    predictions["split"].eq("validation"), "date"
                ].drop_duplicates()
                regimes = fit_adaptive_hmm_regimes(
                    features, training_end=validation_start,
                    prediction_dates=list(pd.to_datetime(prediction_dates)),
                    n_states=config.hmm_states, random_seed=config.random_seed,
                    finetune_window=config.hmm_finetune_window,
                    finetune_iterations=config.hmm_finetune_iterations,
                    include_probabilities=config.use_regime_selector,
                    align_states=config.use_regime_selector,
                )
                if config.use_regime_selector:
                    training_regimes = fit_asof_hmm_regimes(
                        features.loc[features.index < validation_start],
                        training_end=validation_start,
                        n_states=config.hmm_states, random_seed=config.random_seed,
                        include_probabilities=True,
                    )
                    regimes = pd.concat([training_regimes, regimes]).sort_index()
            else:
                regimes = fit_asof_hmm_regimes(
                    features, training_end=validation_start,
                    n_states=config.hmm_states, random_seed=config.random_seed,
                    include_probabilities=config.use_regime_selector,
                )
            predictions = predictions.drop(
                columns=["regime_state", "regime_probability"]
            ).merge(
                regimes, left_on="date", right_index=True, how="left",
                validate="many_to_one"
            )
            hmm_allowed_states = _choose_hmm_states(predictions, config)
            gate = predictions["regime_state"].isin(hmm_allowed_states) & predictions[
                "regime_probability"
            ].ge(config.hmm_min_probability)
            predictions["hmm_signal"] = np.where(gate, predictions["base_signal"], 0)
            hmm_available = True
            hmm_status = "candidate evaluated on validation"
            if config.use_regime_selector:
                selector_policies, selector_diagnostics = fit_regime_policy_library(predictions, config)
                predictions = apply_regime_selector(predictions, selector_policies, config)
                selector_status = "training-only policies evaluated on validation"
        except (ValueError, np.linalg.LinAlgError) as error:
            hmm_status = f"unavailable: {error}"
            if config.use_regime_selector:
                selector_status = f"unavailable: {error}"

    models = {
        "hyperplane_baseline": "base_signal", "always_up": "always_up_signal",
        "always_down": "always_down_signal",
        "previous_direction": "previous_direction_signal",
    }
    if hmm_available:
        models["hmm_regime_filter"] = "hmm_signal"
    if config.use_regime_selector:
        models["momentum_baseline"] = "momentum_signal"
        if "regime_selector_signal" in predictions:
            models["regime_strategy_selector"] = "regime_selector_signal"
    predictions["no_trade_signal"] = 0
    models["no_trade"] = "no_trade_signal"

    candidate_models = {"hyperplane_baseline": "base_signal"}
    if hmm_available:
        candidate_models["hmm_regime_filter"] = "hmm_signal"
    if config.use_regime_selector:
        candidate_models["momentum_baseline"] = "momentum_signal"
        if "regime_selector_signal" in predictions:
            candidate_models["regime_strategy_selector"] = "regime_selector_signal"
    selection_slippage = max(config.slippage_grid_bps_per_side)
    validation_sensitivity = {
        model_name: {
            float(slippage): score_signals(
                predictions, signal_column, split="validation",
                commission_bps_per_side=config.commission_bps_per_side,
                slippage_bps_per_side=float(slippage),
                short_borrow_bps_per_day=config.short_borrow_bps_per_day,
            )
            for slippage in config.slippage_grid_bps_per_side
        }
        for model_name, signal_column in candidate_models.items()
    }
    validation_scores = {
        model_name: scenarios[selection_slippage]
        for model_name, scenarios in validation_sensitivity.items()
    }

    qualified = {
        model_name: score for model_name, score in validation_scores.items()
        if float(score["net_cumulative_return"]) > 0.0
        and float(score["coverage"]) >= config.minimum_validation_coverage
    }
    if qualified:
        def candidate_key(item: tuple[str, dict[str, float | int | str]]) -> tuple[float, float, float]:
            score = item[1]
            dpa = float(score["dpa"]) if np.isfinite(score["dpa"]) else -np.inf
            return (
                float(score["net_cumulative_return"]), dpa, float(score["coverage"])
            )

        selected_model = max(qualified.items(), key=candidate_key)[0]
        selection_reason = (
            "selected on validation at the maximum slippage assumption by net "
            "return, then DPA, then coverage"
        )
    else:
        selected_model = "no_trade"
        selection_reason = (
            "no candidate achieved positive validation net return and minimum coverage"
        )

    if selected_model in {"hmm_regime_filter", "regime_strategy_selector"} and config.hmm_adaptive:
        test_dates = predictions.loc[
            predictions["split"].eq("test"), "date"
        ].drop_duplicates()
        test_regimes = fit_adaptive_hmm_regimes(
            features, training_end=validation_start,
            prediction_dates=list(pd.to_datetime(test_dates)),
            n_states=config.hmm_states, random_seed=config.random_seed,
            finetune_window=config.hmm_finetune_window,
            finetune_iterations=config.hmm_finetune_iterations,
            include_probabilities=config.use_regime_selector,
            align_states=config.use_regime_selector,
        )
        test_mask = predictions["split"].eq("test")
        for column in test_regimes.columns:
            predictions.loc[test_mask, column] = predictions.loc[
                test_mask, "date"
            ].map(test_regimes[column])
        gate = predictions["regime_state"].isin(hmm_allowed_states) & predictions[
            "regime_probability"
        ].ge(config.hmm_min_probability)
        predictions["hmm_signal"] = np.where(gate, predictions["base_signal"], 0)
        if config.use_regime_selector:
            predictions = apply_regime_selector(predictions, selector_policies, config)

    if "regime_selector_signal" in predictions and selected_model != "regime_strategy_selector":
        # Only the validation winner is evaluated on the held-out test period.
        # Blanks prevent uncomputed selector decisions from looking like cash.
        test_mask = predictions["split"].eq("test")
        selector_columns = [
            "regime_selector_signal", "selector_mean_reversion_weight",
            "selector_momentum_weight", "selector_cash_weight", "selector_cash_allocation",
        ]
        predictions.loc[test_mask, selector_columns] = np.nan
        predictions.loc[test_mask, "regime_selector_evaluation_status"] = "not_evaluated_on_test"

    predictions["selected_signal"] = predictions[models[selected_model]]
    hmm_kept = selected_model == "hmm_regime_filter"
    if "regime_strategy_selector" in validation_scores:
        selector_score = validation_scores["regime_strategy_selector"]
        if selected_model == "regime_strategy_selector":
            selector_status = "selected using validation data only"
        elif selector_policies and all(expert == "cash" for expert in selector_policies.values()):
            selector_status = (
                "evaluated on validation: cash in every regime; no training expert "
                "had a positive eligible lower bound"
            )
        elif (float(selector_score["net_cumulative_return"]) <= 0.0
              or float(selector_score["coverage"]) < config.minimum_validation_coverage):
            selector_status = "rejected on validation: requires positive net return and minimum coverage"
        else:
            selector_status = "not selected: another validation candidate ranked higher"
    if hmm_available:
        hmm_score = validation_scores["hmm_regime_filter"]
        if hmm_kept:
            hmm_status = "selected using validation data only"
        elif selected_model == "regime_strategy_selector":
            hmm_status = "used by selected regime strategy selector"
        elif (
            float(hmm_score["net_cumulative_return"]) <= 0.0
            or float(hmm_score["coverage"]) < config.minimum_validation_coverage
        ):
            hmm_status = (
                "rejected on validation: requires positive net return and minimum coverage"
            )
        else:
            hmm_status = "not selected: another validation candidate ranked higher"

    metrics = _metric_rows(
        predictions, models, config, selected_model=selected_model
    )
    summary: dict[str, object] = {
        "model": config.model, "tickers": config.tickers,
        "walk_forward_start": str(pd.Timestamp(predictions["date"].min()).date()),
        "validation_start": str(validation_start.date()),
        "test_start": str(test_start.date()),
        "selected_threshold_bps": selected_threshold,
        "threshold_validation_net_return": float(
            threshold_validation_score["net_cumulative_return"]
        ),
        "threshold_validation_coverage": float(threshold_validation_score["coverage"]),
        "threshold_validation_candidates": threshold_validation_candidates,
        "minimum_validation_coverage": config.minimum_validation_coverage,
        "commission_bps_per_side": config.commission_bps_per_side,
        "slippage_grid_bps_per_side": config.slippage_grid_bps_per_side,
        "selection_slippage_bps_per_side": selection_slippage,
        "short_borrow_bps_per_day": config.short_borrow_bps_per_day,
        "hmm_adaptive": config.hmm_adaptive,
        "hmm_status": hmm_status,
        "hmm_allowed_states_selected_on_validation": hmm_allowed_states,
        "hmm_kept": hmm_kept,
        "hmm_used_by_selected_strategy": selected_model in {"hmm_regime_filter", "regime_strategy_selector"},
        "regime_selector_status": selector_status,
        "regime_selector_evaluation": _regime_selector_evaluation_report(predictions, config),
        "regime_selector_policies": {str(state): expert for state, expert in selector_policies.items()},
        "regime_selector_policy_diagnostics": selector_diagnostics,
        "selector_mean_reversion_threshold_bps": selector_threshold,
        "selector_training_end": str(pd.Timestamp(
            predictions.loc[predictions["split"].eq("train"), "date"].max()
        ).date())
        if config.use_regime_selector else None,
        "selector_policy_updates_on_validation_or_test": False,
        "regime_selector_implementation": "HMM-gated expert mixture; inspired by ReCAP, not its PPO implementation",
        "selected_model": selected_model,
        "selection_reason": selection_reason,
        "test_used_for_selection": False,
        "validation_candidates": {
            name: {
                "observations": int(validation_scores[name]["observations"]),
                "active_positions": int(validation_scores[name]["active_positions"]),
                "passes_positive_net_return": float(validation_scores[name]["net_cumulative_return"]) > 0.0,
                "passes_minimum_coverage": float(validation_scores[name]["coverage"]) >= config.minimum_validation_coverage,
                "eligible_for_selection": name in qualified,
                "selection_net_cumulative_return": float(
                    validation_scores[name]["net_cumulative_return"]
                ),
                "dpa": float(validation_scores[name]["dpa"]),
                "coverage": float(validation_scores[name]["coverage"]),
                "slippage_scenarios": [
                    {
                        "slippage_bps_per_side": slippage,
                        "net_cumulative_return": float(
                            score["net_cumulative_return"]
                        ),
                    }
                    for slippage, score in sorted(
                        validation_sensitivity[name].items()
                    )
                ],
            }
            for name in validation_scores
        },
    }
    return PipelineResult(predictions=predictions, metrics=metrics, summary=summary, config=config)


def write_pipeline_result(result: PipelineResult, output_dir: str | Path | None = None) -> Path:
    directory = Path(output_dir or result.config.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(directory / "walk_forward_predictions.csv", index=False)
    result.metrics.to_csv(directory / "metrics.csv", index=False)
    (directory / "summary.json").write_text(
        json.dumps(result.summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (directory / "config.json").write_text(
        json.dumps(asdict(result.config), indent=2, sort_keys=True), encoding="utf-8"
    )
    return directory.resolve()


def PlanePlot(
    t1: str, t2: str, t3: str, s: str, e: str,
    p1: float, p2: float, p3: float
) -> pd.DataFrame:
    """Compatibility wrapper for the original eight-argument function."""

    tickers = [t1.upper(), t2.upper(), t3.upper()]
    data = download_market_data(tickers, s, e)
    return forecast_current(
        data, [p1, p2, p3], lookback=min(252, len(data.close)),
        model="normalized", require_stationarity=True
    )


def _load_config(path: str | Path) -> BacktestConfig:
    return BacktestConfig(**json.loads(Path(path).read_text(encoding="utf-8")))


def _print_key_metrics(result: PipelineResult) -> None:
    summary = result.summary
    selected_model = str(summary["selected_model"])
    selected_label = "cash (no_trade fallback)" if selected_model == "no_trade" else selected_model
    selection_slippage = float(summary["selection_slippage_bps_per_side"])
    minimum_coverage = float(summary["minimum_validation_coverage"])
    print(f"Selected strategy: {selected_label}")
    print(f"Underlying price model: {summary['model']}")
    print(
        f"Selection requires positive validation net return at {selection_slippage:g} bps "
        f"slippage per side and at least {minimum_coverage:.2%} trading coverage."
    )
    print(f"Selection reason: {summary['selection_reason']}")
    candidates = summary["validation_candidates"]
    if candidates:
        best_name, best_score = max(
            candidates.items(), key=lambda item: float(item[1]["selection_net_cumulative_return"])
        )
        best_return = float(best_score["selection_net_cumulative_return"])
        best_coverage = float(best_score["coverage"])
        print(
            f"Highest validation return: {best_name}, {best_return:+.2%}, "
            f"coverage {best_coverage:.2%}."
        )
        if best_coverage < minimum_coverage:
            print(
                f"Its coverage is below the configured {minimum_coverage:.2%} minimum; "
                "the highest return alone does not make it eligible."
            )
    selector_evaluation = summary.get("regime_selector_evaluation", {})
    if selector_evaluation:
        validation = selector_evaluation["validation"]
        print(
            f"Regime selector: {summary['regime_selector_status']}. "
            f"Computed {validation['forecasts_computed']} forecasts and "
            f"{validation['regime_probability_rows']} regime probability vectors; "
            f"opened {validation['active_positions']} validation positions."
        )
        if selector_evaluation["test"]["status"] == "not_evaluated":
            print("Regime selector test: not evaluated because it was not selected on validation.")
    print("\nPortfolio metrics (n/a means the statistic is undefined, such as accuracy with no trades):")
    columns = [
        "model", "split", "scope", "observations", "active_positions",
        "slippage_bps_per_side", "dpa", "coverage",
        "net_cumulative_return", "annualized_sharpe", "max_drawdown"
    ]
    rows = result.metrics[
        result.metrics["scope"].eq("portfolio")
        & result.metrics["model"].ne("no_trade")
        & (
            result.metrics["model"].isin([
                "hyperplane_baseline", "hmm_regime_filter", "momentum_baseline", "regime_strategy_selector"
            ])
            | result.metrics["model"].eq(selected_model)
        )
    ]
    print(rows[columns].drop_duplicates().to_string(
        index=False, na_rep="n/a", float_format=lambda x: f"{x:.4f}"
    ))
    if selected_model == "no_trade":
        cash_test = result.metrics.loc[
            result.metrics["model"].eq("no_trade")
            & result.metrics["scope"].eq("portfolio")
            & result.metrics["split"].eq("test")
            & result.metrics["slippage_bps_per_side"].eq(selection_slippage)
        ].iloc[0]
        print(
            f"\nHeld-out test: cash fallback across {int(cash_test['observations'])} ticker-days; "
            f"{int(cash_test['active_positions'])} positions, "
            f"{float(cash_test['net_cumulative_return']):.2%} net return. "
            "Cash earns no interest in this simulation."
        )
    print("Full candidate diagnostics and cash benchmark metrics are saved in summary.json and metrics.csv.")


def _backtest_from_args(args: argparse.Namespace) -> int:
    if args.config:
        config = _load_config(args.config)
    else:
        missing = [name for name, value in (
            ("--tickers", args.tickers), ("--start", args.start), ("--end", args.end)
        ) if not value]
        if missing:
            raise SystemExit(f"Missing without --config: {', '.join(missing)}")
        config = BacktestConfig(
            tickers=args.tickers, start=args.start, end=args.end,
            lookback=args.lookback, model=args.model,
            validation_start=args.validation_start, test_start=args.test_start,
            require_stationarity=not args.no_stationarity_gate,
            signal_threshold_bps=args.signal_threshold_bps,
            commission_bps_per_side=args.commission_bps_per_side,
            slippage_grid_bps_per_side=args.slippage_grid_bps_per_side,
            short_borrow_bps_per_day=args.short_borrow_bps_per_day,
            minimum_validation_coverage=args.minimum_validation_coverage,
            use_hmm=not args.no_hmm, hmm_adaptive=args.adaptive_hmm,
            hmm_finetune_window=args.hmm_finetune_window,
            hmm_finetune_iterations=args.hmm_finetune_iterations,
            use_regime_selector=args.regime_selector,
            momentum_lookback=args.momentum_lookback,
            momentum_threshold_bps=args.momentum_threshold_bps,
            selector_min_train_days=args.selector_min_train_days,
            selector_confidence_z=args.selector_confidence_z,
            selector_min_exposure=args.selector_min_exposure,
            random_seed=args.seed,
            data_dir=args.data_dir, output_dir=args.output_dir,
        )
    result = run_backtest(load_market_data(config), config)
    output = write_pipeline_result(result)
    _print_key_metrics(result)
    print(f"Results written to: {output}")
    return 0


def _forecast_from_args(args: argparse.Namespace) -> int:
    config = BacktestConfig(
        tickers=args.tickers, start=args.start, end=args.end,
        lookback=args.lookback, model=args.model,
        require_stationarity=not args.no_stationarity_gate,
        use_hmm=False, data_dir=args.data_dir,
    )
    result = forecast_current(
        load_market_data(config), args.prices, lookback=args.lookback,
        model=args.model, require_stationarity=not args.no_stationarity_gate,
        stationarity_alpha=config.stationarity_alpha
    )
    print(result.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="As-of cross-stock forecasting and validation pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    backtest = subparsers.add_parser("backtest", help="Run a walk-forward backtest.")
    backtest.add_argument("--config", help="JSON configuration file.")
    backtest.add_argument("--tickers", nargs="+", help="Three or more ticker symbols.")
    backtest.add_argument("--start", help="First historical date (YYYY-MM-DD).")
    backtest.add_argument("--end", help="Last date, inclusive (YYYY-MM-DD).")
    backtest.add_argument("--lookback", type=int, default=252)
    backtest.add_argument("--model", choices=["normalized", "cointegration"], default="normalized")
    backtest.add_argument("--validation-start")
    backtest.add_argument("--test-start")
    backtest.add_argument("--signal-threshold-bps", type=float)
    backtest.add_argument("--commission-bps-per-side", type=float, default=0.0)
    backtest.add_argument(
        "--slippage-grid-bps-per-side", nargs="+", type=float,
        default=[0.0, 1.0, 2.0, 5.0],
    )
    backtest.add_argument("--short-borrow-bps-per-day", type=float, default=0.5)
    backtest.add_argument("--minimum-validation-coverage", type=float, default=0.20)
    backtest.add_argument("--adaptive-hmm", action="store_true")
    backtest.add_argument("--hmm-finetune-window", type=int, default=126)
    backtest.add_argument("--hmm-finetune-iterations", type=int, default=10)
    backtest.add_argument("--regime-selector", action="store_true", help="Compare training-calibrated regime experts.")
    backtest.add_argument("--momentum-lookback", type=int, default=20)
    backtest.add_argument("--momentum-threshold-bps", type=float, default=50.0)
    backtest.add_argument("--selector-min-train-days", type=int, default=30)
    backtest.add_argument("--selector-confidence-z", type=float, default=1.96)
    backtest.add_argument("--selector-min-exposure", type=float, default=0.10)
    backtest.add_argument("--seed", type=int, default=42)
    backtest.add_argument("--data-dir", help="Directory containing TICKER.csv files.")
    backtest.add_argument("--output-dir", default="output/hyperplane_backtest")
    backtest.add_argument("--no-hmm", action="store_true")
    backtest.add_argument("--no-stationarity-gate", action="store_true")
    backtest.set_defaults(handler=_backtest_from_args)
    forecast = subparsers.add_parser("forecast", help="Forecast supplied current prices.")
    forecast.add_argument("--tickers", nargs="+", required=True)
    forecast.add_argument("--prices", nargs="+", type=float, required=True)
    forecast.add_argument("--start", required=True)
    forecast.add_argument("--end", required=True)
    forecast.add_argument("--lookback", type=int, default=252)
    forecast.add_argument("--model", choices=["normalized", "cointegration"], default="normalized")
    forecast.add_argument("--data-dir")
    forecast.add_argument("--no-stationarity-gate", action="store_true")
    forecast.set_defaults(handler=_forecast_from_args)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
