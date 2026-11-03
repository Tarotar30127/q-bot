"""Yearly summaries must retain the continuous book's costs and return timing."""
import numpy as np
import pandas as pd
import pytest

from sharpe_statistics import yearly_sharpe_statistics


def test_yearly_sharpe_preserves_entry_fee_and_cross_year_interval():
    dates = pd.to_datetime(["2023-01-03", "2023-01-04", "2023-12-29", "2024-01-02", "2024-12-31"])
    daily = pd.DataFrame({"equity": [99, 101, 103.02, 101.9898, 105.049494],
                          "net_return": [-0.01, 101 / 99 - 1, 0.02, -0.01, 0.03],
                          "risk_free_return": [0, 0.001, 0.002, 0.004, 0.001]}, index=dates)
    annual, summary = yearly_sharpe_statistics(daily, 100)
    first = np.array([0.01 - 0.001, 0.02 - 0.002])
    second = np.array([-0.01 - 0.004, 0.03 - 0.001])
    expected = np.array([np.sqrt(252) * sample.mean() / sample.std(ddof=1) for sample in (first, second)])
    np.testing.assert_allclose(annual.sharpe, expected)
    assert annual.holding_intervals.tolist() == [2, 2]
    assert not annual.partial_year.any()
    assert summary["annual_sharpe_mean"] == pytest.approx(expected.mean())
    assert summary["annual_sharpe_std"] == pytest.approx(expected.std(ddof=1))
    all_excess = np.r_[first, second]
    assert summary["full_period_sharpe_recomputed"] == pytest.approx(np.sqrt(252) * all_excess.mean() / all_excess.std(ddof=1))


def test_partial_year_is_included_and_undefined_sharpe_is_not_zero():
    dates = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-12-31", "2026-01-02", "2026-10-08"])
    daily = pd.DataFrame({"equity": 100.0, "net_return": [0, 0, 0, 0.01, -0.02],
                          "risk_free_return": 0.0}, index=dates)
    annual, summary = yearly_sharpe_statistics(daily, 100)
    assert np.isnan(annual.sharpe.iloc[0])
    assert annual.partial_year.tolist() == [False, True]
    assert summary["annual_sharpe_count"] == 1
    assert summary["annual_sharpe_undefined_years"] == 1
    assert summary["annual_sharpe_includes_partial_year"]
    assert summary["annual_sharpe_mean"] == annual.sharpe.iloc[1]
    assert np.isnan(summary["annual_sharpe_std"])


@pytest.mark.parametrize("fault", ["duplicate", "unordered", "nonfinite", "capital"])
def test_invalid_saved_book_is_rejected(fault):
    dates = pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05"])
    daily = pd.DataFrame({"equity": 100.0, "net_return": 0.0, "risk_free_return": 0.0}, index=dates)
    capital = 100
    if fault == "duplicate":
        daily.index = [dates[0]] * 3
    elif fault == "unordered":
        daily = daily.iloc[::-1]
    elif fault == "nonfinite":
        daily.loc[dates[-1], "net_return"] = np.nan
    else:
        capital = 0
    with pytest.raises(ValueError):
        yearly_sharpe_statistics(daily, capital)
