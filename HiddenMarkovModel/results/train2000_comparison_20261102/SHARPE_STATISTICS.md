# Yearly Sharpe mean and standard deviation

Continuous evaluation: 2020-01-02 through 2026-10-30.

For each calendar year, annualized Sharpe = sqrt(252) × mean(daily excess return) / sample SD(daily excess return). Excess returns retain the original lagged DGS10 ACT/365 reference, execution costs, cash income, and borrow costs. Initial entry fees are folded into the first holding interval, matching the simulator. Intervals belong to the year of their ending date; cross-year holdings are preserved without extra liquidation or fees.

The reported average is the equally weighted arithmetic mean of yearly Sharpes. Their standard deviation is the sample SD (ddof=1). Neither statistic replaces the full-period Sharpe. This SD describes variation across years, not a standard error or confidence interval. The partial 2026 year is included and annualized using the same sqrt(252) convention. Undefined yearly Sharpes are excluded and counted explicitly; they are not replaced with zero.

## zero_slippage

| Allocation / benchmark | Full-period Sharpe | Mean yearly Sharpe | SD of yearly Sharpe | Valid years | Undefined years |
|---|---:|---:|---:|---:|---:|
| 80/20 | 0.875 | 1.183 | 0.853 | 7 | 0 |
| 90/10 | 0.877 | 1.184 | 0.855 | 7 | 0 |
| 95/5 | 0.877 | 1.184 | 0.856 | 7 | 0 |
| IWV (Russell 3000 ETF proxy) | 0.620 | 0.830 | 0.877 | 7 | 0 |
| 100-stock buy and hold | 0.863 | 1.073 | 0.835 | 7 | 0 |

## half_cent_slippage

| Allocation / benchmark | Full-period Sharpe | Mean yearly Sharpe | SD of yearly Sharpe | Valid years | Undefined years |
|---|---:|---:|---:|---:|---:|
| 80/20 | 0.874 | 1.183 | 0.853 | 7 | 0 |
| 90/10 | 0.876 | 1.184 | 0.855 | 7 | 0 |
| 95/5 | 0.877 | 1.184 | 0.856 | 7 | 0 |
| IWV (Russell 3000 ETF proxy) | 0.619 | 0.830 | 0.877 | 7 | 0 |
| 100-stock buy and hold | 0.863 | 1.073 | 0.835 | 7 | 0 |

[Year-by-year observations](yearly_sharpe.csv) · [Summary CSV](sharpe_statistics.csv)
