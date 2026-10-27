# Do the HMM's trade decisions add value?

Study declared October 23, 2026; prices stop October 8, 2026.

**The study does not establish a repeatable HMM trading advantage.** The rolling evaluations produced positive trading P&L in 1/5 years and 7 completed positions. The HMM beat ungated momentum in 2/5 years and 26.00% of the fixed rebalance schedules. 2 of the four declared activity and comparative evidence references were missed. The evidence supports keeping this as a research control and testing the next strategy hypothesis under a new protocol.

## Frozen model and comparison

The reference is the existing 20-session HMM selector with a 20-session momentum signal, 50 bps entry threshold, and 100-session trend check. A dominant-state probability of at least 0.55 is required; posterior confidence sizes the position. Its state/expert policy and HMM are fitted only before the evaluation window. Training calibration retains zero cash interest so positive interest cannot qualify an unprofitable expert.

Momentum without HMM removes both the eligibility gate and confidence sizing. The posterior-sized ungated control keeps confidence sizing and removes the state/minimum-confidence gates. The same-gate IWV control applies the HMM's available target exposure to IWV on the same schedule. Passive allocation controls use only average training exposure; subsequent weights drift, so they do not match realized exposure or volatility exactly.

Each split starts with $100,000; entry and exit use adjusted opens. Slippage is 5 bps per side with 0 bps commission; turnover and terminal liquidation incur costs. Cash uses a Treasury-yield approximation available with a two-session lag. Each annual book liquidates at its last open and restarts next year, omitting cross-year overnight gaps.

All returns below are cumulative net returns for the indicated period. Trade and cash contributions are percentage points of starting equity and reconcile to net return. Trade contribution includes execution and borrow costs and excludes interest. Completed positions count contiguous holding episodes in a stock, including additions/trims; they are not order counts or independent statistical observations.

## Reference model versus Russell 3000 proxy

**IWV is an investable Russell 3000 ETF proxy, not the official Russell 3000 total-return index.** Yahoo-adjusted OHLC incorporates dividend/split adjustments; dividends are not added again. Fund expenses already affect returns and are not charged again. All passive books pay the same modeled entry/exit costs. [The issuer identifies the Russell 3000 Index as IWV's benchmark](https://www.blackrock.com/us/individual/products/239714/ishares-russell-3000-etf).

### 2023–2025 research validation

| Strategy | Net return | Trade contribution | Cash contribution | Max drawdown | Avg capital invested | Completed positions |
| --- | --- | --- | --- | --- | --- | --- |
| HMM selector | +58.19% | +43.42 pp | +14.77 pp | -8.10% | 21.05% | 19 |
| Momentum without HMM | +123.35% | +111.94 pp | +11.41 pp | -18.71% | 52.83% | 25 |
| Momentum with posterior sizing, no gate | +120.48% | +108.28 pp | +12.20 pp | -17.07% | 48.13% | 25 |
| Passive stock basket | +490.28% | +490.28 pp | +0.00 pp | -38.08% | 100.00% | 3 |
| IWV Russell 3000 ETF proxy | +81.98% | +81.98 pp | +0.00 pp | -20.41% | 100.00% | 1 |
| Cash yield proxy | +15.77% | +0.00 pp | +15.77 pp | 0.00% | 0.00% | 0 |
| Passive stocks at training allocation | +158.50% | +147.47 pp | +11.03 pp | -23.66% | 53.69% | 3 |
| IWV at training allocation | +35.69% | +24.66 pp | +11.03 pp | -7.47% | 35.32% | 1 |
| IWV using HMM target exposure | +31.39% | +18.22 pp | +13.17 pp | -5.99% | 20.90% | 8 |

### 2026 through October 8 research check

| Strategy | Net return | Trade contribution | Cash contribution | Max drawdown | Avg capital invested | Completed positions |
| --- | --- | --- | --- | --- | --- | --- |
| HMM selector | +3.29% | +0.53 pp | +2.76 pp | -2.12% | 5.46% | 2 |
| Momentum without HMM | -0.67% | -2.22 pp | +1.55 pp | -16.17% | 45.50% | 8 |
| Momentum with posterior sizing, no gate | -1.62% | -3.23 pp | +1.60 pp | -16.12% | 43.83% | 8 |
| Passive stock basket | +75.35% | +75.35 pp | +0.00 pp | -26.49% | 100.00% | 3 |
| IWV Russell 3000 ETF proxy | +13.30% | +13.30 pp | +0.00 pp | -8.02% | 100.00% | 1 |
| Cash yield proxy | +2.94% | +0.00 pp | +2.94 pp | 0.00% | 0.00% | 0 |
| Passive stocks at training allocation | +24.72% | +22.67 pp | +2.06 pp | -11.84% | 37.67% | 3 |
| IWV at training allocation | +6.06% | +4.00 pp | +2.06 pp | -2.09% | 30.99% | 1 |
| IWV using HMM target exposure | +3.15% | +0.38 pp | +2.78 pp | -0.65% | 5.48% | 2 |

For 2026 the reference HMM earned +3.29%, versus +13.30% for IWV: a -10.01 pp gap. Cash alone earned +2.94%. The HMM's trade contribution was +0.53 pp and interest contributed +2.76 pp. It held a position on 20.83% of holding intervals, with 5.46% average capital invested and 2 completed positions across 1 stock(s).

A passive IWV allocation fixed from training earned +6.06% in 2026 with a -2.09% drawdown, versus -2.12% for the HMM. That similar drawdown and higher return weaken the case that the HMM adds useful risk control beyond choosing a smaller passive allocation. The allocation was fixed before scoring and is not an exact realized-risk match.

Cash interest changes portfolio equity and later order sizes, so an attribution contribution is different from rerunning the entire strategy without interest. The zero-interest reruns reproduce the earlier report:

| Period | HMM zero-interest return | Ungated zero-interest return |
| --- | --- | --- |
| 2023–2025 | +41.11% | +108.53% |
| 2026 through October 8 | +0.51% | -2.24% |

## Chronological evaluations: does trading profitability repeat?

For evaluation year Y, training uses Y−4 through Y−2, the following year records validation without selecting a new configuration, and Y is evaluated. Training windows overlap and data were previously studied. These are chronological robustness checks on reused history, not pristine independent holdouts. The 2026 row ends October 8 and is not a full calendar year.

| Year | HMM net | HMM trade contribution | HMM positions | Ungated net | IWV net | HMM drawdown | Fitted state policies |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2022 | +2.05% | +0.00 pp | 0 | -23.85% | -20.06% | 0.00% | 0: cash, 1: cash |
| 2023 | +5.34% | +0.00 pp | 0 | +56.70% | +25.40% | 0.00% | 0: cash, 1: cash |
| 2024 | +5.31% | +0.00 pp | 0 | +23.29% | +25.23% | 0.00% | 0: cash, 1: cash |
| 2025 | +14.78% | +11.15 pp | 6 | +39.52% | +17.23% | -6.97% | 0: momentum, 1: cash |
| 2026 | +2.33% | -0.50 pp | 1 | -0.67% | +13.30% | -2.12% | 0: momentum, 1: cash |

Total portfolio return was positive in 5/5 years, but trading P&L was positive in 1/5; 3/5 years held only cash. The HMM beat IWV in 1/5 years. Drawdown was smaller than ungated momentum in 5/5 years, with substantially lower exposure. Cash-only positive years demonstrate interest income, not successful trade selection.

## Rebalance date sensitivity

All 20 starting offsets were evaluated using the same previously fitted offset-zero policy. No best offset is selected. These schedules share most observations and are correlated.

| Case | HMM min / median / max net return | Positive trade P&L | HMM beats ungated | Cash-only offsets |
| --- | --- | --- | --- | --- |
| Reference 2023–2025 | +58.19% / +100.01% / +138.28% | 100.00% | 5.00% | 0/20 |
| Reference 2026 through Oct 8 | -4.33% / +2.94% / +12.64% | 15.00% | 5.00% | 10/20 |
| Rolling 2022 | +2.05% / +2.05% / +2.05% | 0.00% | 100.00% | 20/20 |
| Rolling 2023 | +5.34% / +5.34% / +5.34% | 0.00% | 0.00% | 20/20 |
| Rolling 2024 | +5.31% / +5.31% / +5.31% | 0.00% | 0.00% | 20/20 |
| Rolling 2025 | +13.63% / +33.36% / +45.27% | 100.00% | 25.00% | 0/20 |
| Rolling 2026 | -6.31% / +2.33% / +18.33% | 20.00% | 5.00% | 4/20 |
| 100 stocks 2023–2025 | +15.77% / +15.77% / +15.77% | 0.00% | 0.00% | 20/20 |
| 100 stocks 2026 through Oct 8 | +2.94% / +2.94% / +2.94% | 0.00% | 10.00% | 20/20 |

Across the 100 rolling schedules, positive trading P&L occurred in 24.00% and HMM return exceeded same-offset ungated momentum in 26.00%. Return-difference min/median/max are retained in `report_tables/offset_summary.csv`; positive total returns from cash are not counted as positive trading P&L.

## Uncertainty around comparative returns

Paired daily returns are resampled jointly in 20-session circular blocks, 2,000 times with seed 42. The table shows the observed cumulative return difference and descriptive 95% interval, in percentage points. An interval spanning zero does not establish a stable advantage. These intervals do not adjust for previous research, multiple comparisons, fitting uncertainty, or the choice of calendar; the bootstrap fraction above zero is not a probability of future profitability.

| Case | HMM − ungated | HMM − posterior-sized ungated | HMM − IWV |
| --- | --- | --- | --- |
| Reference 2023–2025 | -65.16 pp [-211.61 pp, +27.46 pp] | -62.29 pp [-173.88 pp, +16.63 pp] | -23.79 pp [-116.18 pp, +47.23 pp] |
| Reference 2026 through Oct 8 | +3.96 pp [-44.55 pp, +34.09 pp] | +4.92 pp [-42.78 pp, +34.47 pp] | -10.01 pp [-37.24 pp, +10.23 pp] |
| Rolling 2022 | +25.90 pp [-0.03 pp, +52.16 pp] | +24.79 pp [-0.13 pp, +50.39 pp] | +22.11 pp [-18.82 pp, +48.82 pp] |
| Rolling 2023 | -51.36 pp [-122.06 pp, -8.69 pp] | -40.03 pp [-90.02 pp, -5.37 pp] | -20.06 pp [-54.83 pp, +9.13 pp] |
| Rolling 2024 | -17.98 pp [-65.00 pp, +18.73 pp] | -19.20 pp [-63.60 pp, +17.06 pp] | -19.92 pp [-45.28 pp, +4.07 pp] |
| Rolling 2025 | -24.73 pp [-90.03 pp, +10.80 pp] | -23.49 pp [-87.37 pp, +11.50 pp] | -2.45 pp [-41.61 pp, +31.51 pp] |
| Rolling 2026 | +3.00 pp [-45.77 pp, +33.38 pp] | +3.09 pp [-45.72 pp, +33.45 pp] | -10.98 pp [-37.91 pp, +8.85 pp] |
| 100 stocks 2023–2025 | -15.88 pp [-44.16 pp, +7.89 pp] | -14.23 pp [-41.00 pp, +9.13 pp] | -66.20 pp [-163.88 pp, -0.22 pp] |
| 100 stocks 2026 through Oct 8 | -4.48 pp [-18.45 pp, +7.84 pp] | -4.94 pp [-18.05 pp, +6.05 pp] | -10.36 pp [-37.00 pp, +9.62 pp] |

Only 0/5 rolling evaluation intervals have an HMM-minus-ungated lower bound above zero. Exposure-control and cash-control intervals are also retained in the bootstrap CSV.

## Allowed and blocked momentum decisions

For every momentum-eligible stock on a scheduled rebalance, the audit records whether the HMM allowed or blocked it, then measures that stock's next same-schedule holding return, less hypothetical entry and exit costs. These are equally weighted stock-interval counterfactuals. They do not reproduce actual portfolio sizing, cash compounding, or turnover from carrying an existing position; they must not be read as causal proof or independent trade trials.

| Case | Decision | Stock intervals | Distinct dates | Mean hypothetical net holding return | Profitable intervals |
| --- | --- | --- | --- | --- | --- |
| Reference 2023–2025 | Blocked | 34 | 19 | +3.33% | 61.76% |
| Reference 2023–2025 | Allowed | 26 | 14 | +4.54% | 69.23% |
| Reference 2026 through Oct 8 | Blocked | 12 | 6 | -0.41% | 41.67% |
| Reference 2026 through Oct 8 | Allowed | 2 | 2 | +1.89% | 50.00% |
| Rolling 2022 | Blocked | 6 | 4 | -13.13% | 16.67% |
| Rolling 2023 | Blocked | 24 | 12 | +5.58% | 66.67% |
| Rolling 2024 | Blocked | 21 | 12 | +2.74% | 52.38% |
| Rolling 2025 | Blocked | 14 | 7 | +4.70% | 71.43% |
| Rolling 2025 | Allowed | 7 | 4 | +4.56% | 71.43% |
| Rolling 2026 | Blocked | 13 | 7 | +0.03% | 46.15% |
| Rolling 2026 | Allowed | 1 | 1 | -1.49% | 0.00% |
| 100 stocks 2023–2025 | Blocked | 1795 | 38 | +1.09% | 54.15% |
| 100 stocks 2026 through Oct 8 | Blocked | 433 | 10 | +1.28% | 50.12% |

Decision-level dates and symbols are in each split's `decision_audit.csv`. The posterior-sized ungated control helps distinguish state exclusion from confidence-based sizing. If a full-library rolling policy chooses mean reversion, a separately trained momentum-only HMM control is also saved so expert changes are not attributed solely to gating.

## Separate 100-stock breadth hypothesis

All 100 requested stocks were available on the full common session calendar. This separate hypothesis uses momentum-only experts and fits its HMM on the broad stock universe; it is not the three-stock trained policy transferred unchanged. No 100-variable, 60-row cointegration expert is fitted. The original stock mapping is preserved, with documented vendor aliases for BRK.B, BK, and MMC.

### 2023–2025 research validation

| Strategy | Net return | Trade contribution | Cash contribution | Max drawdown | Avg capital invested | Completed positions |
| --- | --- | --- | --- | --- | --- | --- |
| HMM selector | +15.77% | +0.00 pp | +15.77 pp | 0.00% | 0.00% | 0 |
| Momentum without HMM | +31.65% | +23.13 pp | +8.52 pp | -6.55% | 47.26% | 861 |
| Momentum with posterior sizing, no gate | +30.00% | +21.29 pp | +8.70 pp | -6.53% | 45.88% | 861 |
| Passive stock basket | +101.64% | +101.64 pp | +0.00 pp | -18.46% | 100.00% | 100 |
| IWV Russell 3000 ETF proxy | +81.98% | +81.98 pp | +0.00 pp | -20.41% | 100.00% | 1 |
| Cash yield proxy | +15.77% | +0.00 pp | +15.77 pp | 0.00% | 0.00% | 0 |

### 2026 through October 8 research check

| Strategy | Net return | Trade contribution | Cash contribution | Max drawdown | Avg capital invested | Completed positions |
| --- | --- | --- | --- | --- | --- | --- |
| HMM selector | +2.94% | +0.00 pp | +2.94 pp | 0.00% | 0.00% | 0 |
| Momentum without HMM | +7.42% | +5.73 pp | +1.69 pp | -4.42% | 44.42% | 241 |
| Momentum with posterior sizing, no gate | +7.88% | +6.08 pp | +1.80 pp | -3.52% | 41.17% | 241 |
| Passive stock basket | +19.21% | +19.21 pp | +0.00 pp | -6.99% | 100.00% | 100 |
| IWV Russell 3000 ETF proxy | +13.30% | +13.30 pp | +0.00 pp | -8.02% | 100.00% | 1 |
| Cash yield proxy | +2.94% | +0.00 pp | +2.94 pp | 0.00% | 0.00% | 0 |

**The breadth HMM selected cash in both states and made no trades in either split.** Its positive portfolio returns are entirely cash interest. Expanding the stock list did not establish a profitable trading model under the frozen eligibility rules.

The broad ungated momentum strategy produced positive trading P&L in both observed splits with many more completed positions. Its portfolio return nevertheless trailed IWV in both splits, with lower exposure and drawdown. This is a useful simpler research candidate; it does not establish Russell 3000 outperformance.

The user's current selected list is not point-in-time Russell 3000 membership. Survivor, selection, and reused-period biases apply. Multiple stocks entering on the same date are correlated exposures, not independent confirmations. See the separate breadth policy lock and training diagnostics for state eligibility.

## Continue or pivot

| Declared evidence reference | Observed | Reference minimum | Outcome |
| --- | --- | --- | --- |
| Positive total-return evaluation years, including interest | 100.00% | 60.00% | Pass |
| HMM beats ungated evaluation years | 40.00% | 60.00% | Insufficient |
| Positive total-return rolling offsets, including interest | 97.00% | 70.00% | Pass |
| Completed rolling HMM positions | 7 | 30 | Insufficient |

The frozen positivity references use total portfolio returns, including cash interest. The observed values are 5/5 evaluation years and 97.00% of rolling offsets; cash interest accounts for all returns in 3 evaluation years. Positive portfolio returns alone do not demonstrate profitable trade selection. Trading-only positivity (1/5 years) and paired HMM-minus-momentum offset advantage (26.00%) are additional diagnostics, without retroactively assigning the plan's positivity thresholds to those different metrics.

The next useful work is a separately registered strategy hypothesis with an explicit opportunity cost of cash and enough expected activity to evaluate trade selection. Keep ungated momentum, IWV, and cash as controls. Freeze the model and decision rules before a prospective paper evaluation using sessions arriving after registration. None of this study's historical periods can become unseen data by relabeling them.

[The prospective paper evaluation registration](../paper_evaluation_registration.json) was saved October 23, 2026 at 22:39:06 UTC, with the first eligible execution session October 26, 2026. It tracks the frozen reference HMM, three-stock and 100-stock ungated momentum, IWV, and cash. Its status is awaiting future observations: no prospective observations or results have been scored. A rule change requires a new registered version. The 12-month and 30-completed-position evidence targets do not establish statistical independence or prove an advantage. This registration does not schedule polling or submit brokerage orders.

## Data, reproduction, and audit files

[Benchmark and data notes](benchmark_data_notes.md) document issuer/FRED citations, dividend adjustments, ticker changes, complete calendars, hashes, and current-vintage limitations. The cash source is [Federal Reserve DGS3MO via FRED](https://fred.stlouisfed.org/series/DGS3MO), a 3-month yield proxy with lagged ACT/365 accrual; it is not a realized cash fund return or a guaranteed broker interest rate. Original three-stock cache bytes are unchanged.

The full JSON, source/data protocol hashes, policy locks, daily books, order records, completed positions, decision audits, and offset results are under `output/value_study_reproduction_20261027/`. Flat CSV tables generated with this report are in its `report_tables/` subdirectory. Machine tables preserve unrounded values; report figures are rounded only for display.

Generate the report from completed results without rerunning research:

```powershell
.\.venv\Scripts\python.exe write_value_study_report.py --inputdir output/value_study_reproduction_20261027
```

Run a deliberate fresh reproduction with a new output directory because an existing study protocol is locked:

```powershell
.\.venv\Scripts\python.exe DecisionValueStudy.py --plan value_study_plan.json --output-dir output/value_study_repeat
```

Source result SHA-256: `d05d7fce4de39f2c17687fabb6a396cac42743a761acd796c25f8b223bbc4e24`.
