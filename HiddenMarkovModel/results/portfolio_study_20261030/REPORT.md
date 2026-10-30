# 100-stock portfolio: 2015 training and per-share slippage

Each slippage version fits independently on the same inputs and seed. The primary model uses the existing 100-stock momentum-only HMM hypothesis.

Main training: 2015–2022. Warm-up prices begin June 2014. Evaluation history was previously observed and is research data.

Every independent book starts with $1,000,000. All stock strategies buy equal dollar amounts at the first open; costs come from that budget. Shares remain fixed until the first model rebalance 20 trading sessions later. Subsequent model decisions may hold cash. The continuous book carries positions across year boundaries; standalone windows and yearly checks restart separately.

Slippage is fixed at $0 or $0.005 per actual share, on each side, including liquidation. Commission remains zero. Adjusted-price total returns approximate dividend reinvestment without adding dividends again; quoted opens are reconstructed from Yahoo split history for share-based fees.

Sharpe uses [DGS10](https://fred.stlouisfed.org/series/DGS10), lagged two stock sessions and accrued at the prior-open yield using elapsed calendar days / 365. It is sqrt(252) times the mean interval excess return divided by its sample standard deviation. Entry costs belong to the first holding interval. Idle cash separately earns the existing [DGS3MO](https://fred.stlouisfed.org/series/DGS3MO) approximation. The 10-year yield is a reference yield, not a realized bond total return.

## Continuous portfolio: 2023 through October 8, 2026

| Slippage | Strategy | Final equity | Net return | Annualized return | Sharpe | Max drawdown | Execution costs | Avg exposure |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| $0/share | hmm | $1,248,456.29 | +24.85% | +6.08% | 0.709 | -2.84% | $0.00 | 2.12% |
| $0/share | ungated_momentum | $1,551,188.53 | +55.12% | +12.38% | 1.142 | -6.46% | $0.00 | 49.05% |
| $0/share | passive_stocks | $2,613,423.62 | +161.34% | +29.09% | 1.323 | -18.46% | $0.00 | 100.00% |
| $0/share | iwv | $2,059,354.19 | +105.94% | +21.17% | 1.040 | -20.41% | $0.00 | 100.00% |
| $0/share | cash | $1,192,020.58 | +19.20% | +4.78% | 6.992 | +0.00% | $0.00 | 0.00% |
| $0.005/share | hmm | $1,248,328.55 | +24.83% | +6.07% | 0.708 | -2.84% | $104.88 | 2.12% |
| $0.005/share | ungated_momentum | $1,549,718.58 | +54.97% | +12.35% | 1.138 | -6.47% | $1,153.23 | 49.05% |
| $0.005/share | passive_stocks | $2,613,217.91 | +161.32% | +29.09% | 1.323 | -18.46% | $121.33 | 100.00% |
| $0.005/share | iwv | $2,059,284.30 | +105.93% | +21.17% | 1.040 | -20.41% | $46.02 | 100.00% |
| $0.005/share | cash | $1,192,020.58 | +19.20% | +4.78% | 6.992 | +0.00% | $0.00 | 0.00% |

![Continuous equity comparison](equity_curves.svg)

## Independently trained policies

- zero_slippage: {"0": "cash", "1": "cash"}
- half_cent_slippage: {"0": "cash", "1": "cash"}

The separate training runs selected the same state policy. This is a valid result; no different policy was forced.

Both main HMM policies choose cash in every state. Each continuous model book therefore sells the imposed basket on 2023-02-01 and remains in cash. Its gains come from the initial holding period and subsequent modeled cash interest, not later HMM-selected stock trades.

The initial basket is imposed, so its 100 completed positions cannot be interpreted as 100 model-selected entries. Training eligibility excludes that imposed purchase. See imposed_initial_holding_market_pnl and average_model_exposure_after_initial_holding in the machine tables.

## Standalone windows and expanding-training checks

| Period | Slippage | Strategy | Net return | Sharpe | Max drawdown |
|---|---|---|---:|---:|---:|
| standalone_2023_2025 | $0/share | hmm | +21.25% | 0.859 | -2.84% |
| standalone_2023_2025 | $0/share | passive_stocks | +101.84% | 1.305 | -18.46% |
| standalone_2026 | $0/share | hmm | +7.31% | 1.246 | -1.23% |
| standalone_2026 | $0/share | passive_stocks | +19.33% | 1.503 | -6.99% |
| rolling_2022 | $0/share | hmm | -1.13% | -0.758 | -7.44% |
| rolling_2022 | $0/share | passive_stocks | -6.78% | -0.366 | -20.33% |
| rolling_2023 | $0/share | hmm | +9.18% | 0.745 | -4.49% |
| rolling_2023 | $0/share | passive_stocks | +22.66% | 1.317 | -8.83% |
| rolling_2024 | $0/share | hmm | +8.03% | 1.763 | -1.40% |
| rolling_2024 | $0/share | passive_stocks | +20.00% | 1.272 | -7.54% |
| rolling_2025 | $0/share | hmm | +6.77% | 0.492 | -3.89% |
| rolling_2025 | $0/share | passive_stocks | +31.45% | 1.386 | -15.93% |
| rolling_2026 | $0/share | hmm | +11.08% | 1.345 | -4.37% |
| rolling_2026 | $0/share | passive_stocks | +19.33% | 1.503 | -6.99% |
| standalone_2023_2025 | $0.005/share | hmm | +21.24% | 0.858 | -2.84% |
| standalone_2023_2025 | $0.005/share | passive_stocks | +101.82% | 1.305 | -18.46% |
| standalone_2026 | $0.005/share | hmm | +7.30% | 1.243 | -1.23% |
| standalone_2026 | $0.005/share | passive_stocks | +19.32% | 1.503 | -6.99% |
| rolling_2022 | $0.005/share | hmm | -1.14% | -0.760 | -7.44% |
| rolling_2022 | $0.005/share | passive_stocks | -6.78% | -0.367 | -20.33% |
| rolling_2023 | $0.005/share | hmm | +9.14% | 0.741 | -4.50% |
| rolling_2023 | $0.005/share | passive_stocks | +22.65% | 1.316 | -8.83% |
| rolling_2024 | $0.005/share | hmm | +8.02% | 1.759 | -1.40% |
| rolling_2024 | $0.005/share | passive_stocks | +19.99% | 1.271 | -7.54% |
| rolling_2025 | $0.005/share | hmm | +6.76% | 0.489 | -3.89% |
| rolling_2025 | $0.005/share | passive_stocks | +31.44% | 1.385 | -15.93% |
| rolling_2026 | $0.005/share | hmm | +11.06% | 1.342 | -4.37% |
| rolling_2026 | $0.005/share | passive_stocks | +19.32% | 1.503 | -6.99% |

Rolling year Y trains from 2015 through Y−2, leaves Y−1 as a validation gap, and evaluates Y. No configuration is selected using those evaluations. These independently restarted books are not compounded into the primary continuous result.

## Files and limitations

metrics.csv contains every strategy and window. Each version has policy locks, training diagnostics, regime probabilities, target weights, daily books, actual-share order costs, completed-position attribution, and per-stock passive returns. Original studies and the prospective paper registration are unchanged.

Current selected 100-stock list has survivor and selection bias. Historical evaluation periods were already observed. Adjusted-price total returns and reconstructed quoted prices depend on vendor corporate-action accuracy; they are research approximations. Neither independent fitting nor lower execution costs guarantees a different selected policy or profitable results.

IWV is an investable Russell 3000 ETF proxy, not the official index. The zero-slippage passive_stocks row represents the same 100-stock basket's adjusted open-to-open performance over the matching dates. Neither ETF fund fees nor dividends are deducted/added a second time.
