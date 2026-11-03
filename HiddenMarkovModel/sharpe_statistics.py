"""Calendar-year Sharpe summaries from saved continuous portfolio books."""
from pathlib import Path

import numpy as np
import pandas as pd


def sharpe_ratio(excess: pd.Series) -> float:
    deviation = excess.std(ddof=1)
    return float(np.sqrt(252) * excess.mean() / deviation) if deviation > 1e-12 else np.nan


def yearly_sharpe_statistics(daily: pd.DataFrame, initial_equity: float) -> tuple[pd.DataFrame, dict]:
    required = ["equity", "net_return", "risk_free_return"]
    if (len(daily) < 2 or daily.index.has_duplicates or not daily.index.is_monotonic_increasing
            or not isinstance(daily.index, pd.DatetimeIndex)
            or not np.isfinite(initial_equity) or initial_equity <= 0
            or not np.isfinite(daily[required].to_numpy()).all()):
        raise ValueError("Require an ordered, finite continuous book and positive starting capital.")
    returns = daily.net_return.iloc[1:].copy()
    # Match the simulator: entry fees belong to the first actual holding interval.
    returns.iloc[0] = daily.equity.iloc[1] / initial_equity - 1
    excess = returns - daily.risk_free_return.iloc[1:]
    records = []
    for year, sample in excess.groupby(excess.index.year):
        records.append({"year": year, "first_interval_end": sample.index[0].date(),
                        "last_interval_end": sample.index[-1].date(), "holding_intervals": len(sample),
                        "mean_daily_excess_return": sample.mean(),
                        "std_daily_excess_return": sample.std(ddof=1), "sharpe": sharpe_ratio(sample),
                        "partial_year": sample.index[0].month != 1 or sample.index[-1].month != 12})
    annual = pd.DataFrame(records)
    summary = {"full_period_sharpe_recomputed": sharpe_ratio(excess),
               "annual_sharpe_mean": annual.sharpe.mean(),
               "annual_sharpe_std": annual.sharpe.std(ddof=1),
               "annual_sharpe_count": int(annual.sharpe.count()),
               "annual_sharpe_undefined_years": int(annual.sharpe.isna().sum()),
               "annual_sharpe_includes_partial_year": bool(annual.partial_year.any())}
    return annual, summary


def add_sharpe_statistics(comparison: pd.DataFrame, runs: dict, period: str, output: Path) -> pd.DataFrame:
    """Extend comparison rows and export the annual observations behind each mean/SD."""
    annual_tables, summaries = [], []
    base = next(iter(runs.values()))
    for row in comparison.to_dict("records"):
        folder = runs.get(row["allocation"], base)
        daily = pd.read_csv(folder / row["version"] / period / row["strategy"] / "daily.csv",
                            index_col="date", parse_dates=True,
                            usecols=["date", "equity", "net_return", "risk_free_return"])
        initial_equity = row["final_equity"] / (1 + row["net_cumulative_return"])
        annual, summary = yearly_sharpe_statistics(daily, initial_equity)
        if not np.isclose(summary["full_period_sharpe_recomputed"], row["sharpe"], rtol=1e-10, atol=1e-10, equal_nan=True):
            raise ValueError(f"Saved Sharpe does not reconcile: {row['allocation']}, {row['version']}")
        annual.insert(0, "allocation", row["allocation"])
        annual.insert(1, "version", row["version"])
        annual_tables.append(annual)
        summaries.append(summary)
    enriched = pd.concat([comparison.reset_index(drop=True), pd.DataFrame(summaries)], axis=1)
    annual = pd.concat(annual_tables, ignore_index=True)
    annual.to_csv(output / "yearly_sharpe.csv", index=False)
    columns = ["allocation", "version", "start", "end", "sharpe", *summaries[0]]
    enriched[columns].to_csv(output / "sharpe_statistics.csv", index=False)
    lines = ["# Yearly Sharpe mean and standard deviation", "",
             f"Continuous evaluation: {comparison.start.iloc[0]} through {comparison.end.iloc[0]}.", "",
             "For each calendar year, annualized Sharpe = sqrt(252) × mean(daily excess return) / sample SD(daily excess return). Excess returns retain the original lagged DGS10 ACT/365 reference, execution costs, cash income, and borrow costs. Initial entry fees are folded into the first holding interval, matching the simulator. Intervals belong to the year of their ending date; cross-year holdings are preserved without extra liquidation or fees.", "",
             "The reported average is the equally weighted arithmetic mean of yearly Sharpes. Their standard deviation is the sample SD (ddof=1). Neither statistic replaces the full-period Sharpe. This SD describes variation across years, not a standard error or confidence interval. The partial 2026 year is included and annualized using the same sqrt(252) convention. Undefined yearly Sharpes are excluded and counted explicitly; they are not replaced with zero.", ""]
    for version, group in enriched.groupby("version", sort=False):
        lines += [f"## {version}", "", "| Allocation / benchmark | Full-period Sharpe | Mean yearly Sharpe | SD of yearly Sharpe | Valid years | Undefined years |",
                  "|---|---:|---:|---:|---:|---:|"]
        for row in group.to_dict("records"):
            lines.append(f"| {row['allocation']} | {row['sharpe']:.3f} | {row['annual_sharpe_mean']:.3f} | {row['annual_sharpe_std']:.3f} | {row['annual_sharpe_count']} | {row['annual_sharpe_undefined_years']} |")
        lines.append("")
    lines += ["[Year-by-year observations](yearly_sharpe.csv) · [Summary CSV](sharpe_statistics.csv)", ""]
    (output / "SHARPE_STATISTICS.md").write_text("\n".join(lines), encoding="utf-8")
    return enriched
