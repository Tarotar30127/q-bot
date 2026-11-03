# Current backtest methodology

Each independently fitted version uses two Gaussian HMM states, seed 42,
20-session momentum, a 50-basis-point momentum threshold, and a 100-session
trend check. Training normalization and HMM parameters are frozen before each
main evaluation window; later states are causal forward-filtered probabilities.
Only training outcomes select state policies. The 2023 plans also retain their
original rolling 2022–2026 checks; these independently refit and restart and are
not compounded into the continuous book. The 2020 plans have no rolling refits.

A state can select momentum only with sufficient active training observations
and a positive conservative HAC lower bound on daily net return improvement
over the identical mechanical stock floor. The current results select the floor
in both states. Required stock exposure is not evidence of HMM selection skill.

Each evaluation book starts with $1 million equally invested in all 100 stocks.
Shares are fixed until the first scheduled rebalance 20 sessions later. Thereafter,
stock-allocation shortfalls are spread equally across the universe. Cash caps
are enforced after costs at daily opens, with top-up purchases between rebalances.
Daily bars cannot enforce intraday caps. Final paid liquidation is exempt.
Continuous books carry holdings across year boundaries. Standalone windows
restart independently and are not compounded into the continuous result.

Slippage is zero or $0.005 per actual share on each execution, including entry
and liquidation. Commission is zero. Historical quoted opens are reconstructed
from vendor split history for per-share fees. Adjusted OHLC prices approximate
reinvested dividends and splits; dividends are not added again. Corporate-action
reconstruction remains a vendor-based research approximation.

Cash interest uses FRED DGS3MO; Sharpe uses FRED DGS10. Both retain the original
two-stock-session publication lag and prior-open ACT/365 calendar-day accrual.
The 10-year yield is a reference yield, not a bond total return. Sharpe equals
sqrt(252) times mean holding-interval excess return divided by its sample SD.
Initial entry fees are included in the first holding interval.

Yearly Sharpe ratios use the same continuous portfolio returns, grouped by the
year of each interval's ending date. Holdings are not restarted or liquidated
at year end. The yearly average is an equally weighted arithmetic mean; SD is
the sample standard deviation across yearly ratios (ddof=1). These do not replace
the full-period Sharpe or estimate its confidence interval. Partial 2026 is
included: October 8 for the 2023 study and October 30 for the 2020 study.
Undefined yearly ratios are excluded and counted, not replaced with zero.

IWV is an investable Russell 3000 ETF proxy, not the official index. Its history
is required for every evaluation session, not for early stock-model training.
It starts in May 2000 in the frozen vendor panel. [The issuer documents the
fund's benchmark and launch](https://www.blackrock.com/us/individual/products/239714/ishares-russell-3000-etf).
Treasury sources: [DGS10](https://fred.stlouisfed.org/series/DGS10) and
[DGS3MO](https://fred.stlouisfed.org/series/DGS3MO). Historical data are current
vendor vintages, not archived real-time vintages. All 100 stocks use the user's
selected list, rather than historical Russell 3000 membership. Survivor,
selection, and reused-history biases remain.
