# Yearly Sharpe mean and standard deviation

Continuous evaluation: 2023-01-03 through 2026-10-08.

For each calendar year, annualized Sharpe = sqrt(252) × mean(daily excess return) / sample SD(daily excess return). Excess returns retain the original lagged DGS10 ACT/365 reference, execution costs, cash income, and borrow costs. Initial entry fees are folded into the first holding interval, matching the simulator. Intervals belong to the year of their ending date; cross-year holdings are preserved without extra liquidation or fees.

The reported average is the equally weighted arithmetic mean of yearly Sharpes. Their standard deviation is the sample SD (ddof=1). Neither statistic replaces the full-period Sharpe. This SD describes variation across years, not a standard error or confidence interval. The partial 2026 year is included and annualized using the same sqrt(252) convention. Undefined yearly Sharpes are excluded and counted explicitly; they are not replaced with zero.

## zero_slippage

| Allocation / benchmark | Full-period Sharpe | Mean yearly Sharpe | SD of yearly Sharpe | Valid years | Undefined years |
|---|---:|---:|---:|---:|---:|
| 80/20 | 1.349 | 1.400 | 0.162 | 4 | 0 |
| 90/10 | 1.339 | 1.390 | 0.174 | 4 | 0 |
| 95/5 | 1.335 | 1.386 | 0.179 | 4 | 0 |
| IWV (Russell 3000 ETF proxy) | 1.040 | 1.109 | 0.358 | 4 | 0 |
| 100-stock buy and hold | 1.323 | 1.347 | 0.076 | 4 | 0 |

## half_cent_slippage

| Allocation / benchmark | Full-period Sharpe | Mean yearly Sharpe | SD of yearly Sharpe | Valid years | Undefined years |
|---|---:|---:|---:|---:|---:|
| 80/20 | 1.349 | 1.399 | 0.162 | 4 | 0 |
| 90/10 | 1.339 | 1.390 | 0.174 | 4 | 0 |
| 95/5 | 1.334 | 1.386 | 0.179 | 4 | 0 |
| IWV (Russell 3000 ETF proxy) | 1.040 | 1.109 | 0.358 | 4 | 0 |
| 100-stock buy and hold | 1.323 | 1.347 | 0.076 | 4 | 0 |

[Year-by-year observations](yearly_sharpe.csv) · [Summary CSV](sharpe_statistics.csv)
