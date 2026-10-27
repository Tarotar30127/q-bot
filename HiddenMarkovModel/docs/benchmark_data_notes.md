# Frozen data for the trading decision value study

The frozen panel contains **100 user-supplied stocks and IWV**, covering **2,204 matching trading sessions from January 2, 2018 through October 8, 2026**. Every requested stock has the full benchmark session calendar. There are no omitted stocks, duplicated dates, invalid OHLC rows, or missing historical sessions in the downloaded panel. This checks data completeness, not economic correctness of every vendor observation.

## Russell 3000 comparison

The benchmark is **iShares Russell 3000 ETF (IWV), an investable proxy for the Russell 3000**, not the index itself. Its issuer identifies the Russell 3000 Index as the benchmark. IWV prices can diverge from index returns because of fund expenses, tracking, and market price versus NAV. The issuer currently reports a 0.20% expense ratio; fund expenses already affect prices and fund returns, so the simulator must not subtract this expense ratio again. Entry and exit transaction costs should still use the study's stated execution assumption.

Sources checked October 23, 2026:

- [iShares issuer page: benchmark, expenses, and return methodology](https://www.blackrock.com/us/individual/products/239714/ishares-russell-3000-etf).
- [Yahoo Finance IWV historical data](https://finance.yahoo.com/quote/IWV/history/).

`data/IWV.csv` contains adjusted daily Open, High, Low, and Close. The return basis uses Yahoo's dividend and split adjustments through yfinance. Adjusted OHLC provides a research total return approximation; it is not a tape of historical executable dollar prices. Do not add dividends again to these returns.

## Universe and ticker identity

The original 100-stock mapping in `future_stock_universe.json` is unchanged. Frozen breadth files use the user's canonical symbols in `data/breadth/`, including `BRK.B.csv`, `BK.csv`, and `MMC.csv`. Vendor aliases are recorded separately:

| User symbol | Yahoo symbol | Reason and evidence |
|---|---|---|
| BRK.B | BRK-B | Vendor punctuation convention |
| BK | BNY | [Issuer announced common stock ticker change effective May 21, 2026](https://www.bny.com/corporate/global/en/about-us/newsroom/press-release/bny-announces-planned-change-of-stock-ticker-symbol-to-bny-130465.html) |
| MMC | MRSH | [Issuer announced ticker change effective January 14, 2026, with legal name and CUSIP unchanged](https://www.marsh.com/en/corp/about/news/marsh-mclennan-to-change-nyse-symbol-to-mrsh.html) |

BK and MMC failed under their old vendor symbols; issuer-confirmed current symbols then returned their complete vendor-linked histories. These are ticker identity resolutions, not substitutions with other companies. File names retain the original user symbols.

NVDA and INTC were copied **byte for byte** from the original 2018–2026 research cache. GOOG remains in that original cache for the three-stock study and is not in the user-supplied breadth universe. No original three-stock CSV was downloaded again or changed. The new manifest records all three original hashes before and after retrieval, and the copies' source paths and hashes.

This is today's user-selected list. It is not a historical list of Russell 3000 constituents, and the study must describe historical breadth results as a retrospective selection subject to survivor and selection bias. Vendor-linked histories may span mergers, name changes, and reorganizations. Adjustments and corporate action treatment are vendor estimates and were not independently reconciled against every issuer event.

## Cash yield

The cash input is **DGS3MO**, the Federal Reserve Board's H.15 daily 3-month constant maturity Treasury yield quoted on an investment basis, retrieved from FRED. It is a yield-based cash opportunity cost proxy, not the realized total return of a Treasury bill portfolio or money market fund.

- [Official FRED series and source citation](https://fred.stlouisfed.org/series/DGS3MO).
- [Exact FRED CSV request](https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO&cosd=2017-12-01&coed=2026-10-08).

The exact response is saved as `data/cash/DGS3MO.csv`. It begins December 1, 2017 to permit a conservative lag before the first research period. `data/cash_rates.csv` normalizes nonmissing observations to `Date,annual_rate_decimal`; for example 4.23% becomes 0.0423. **This normalized file is unlagged.** The study caller must forward-fill from earlier observations only and then lag two stock trading sessions before passing the annual decimal rate to the simulator. The simulator accrues the prior-open rate over elapsed calendar days using its documented ACT/365 convention. Holidays are forward-filled from past observations, never filled from future quotes. A separate convenience `cash/cash_yield_lag2sessions.csv` contains an already-lagged IWV-calendar series in annual percent units; the study uses the unlagged normalized file to avoid applying lag twice.

The downloaded historical yield series is the current vendor vintage and may include revisions. The two-session publication lag reduces timing leakage; it does not turn the observations into a separately archived real-time vintage. The comparison should include the zero-interest cash sensitivity and should not imply that a broker would have paid the full Treasury yield.

## Retrieval, audit, and reproducibility

The source project retrieved the panel with the repository's yfinance 1.7.0 environment, daily interval, inclusive research end October 8 (`end='2026-10-09'` in Yahoo's exclusive-end API), `auto_adjust=True`, `repair=True`, and corporate actions retained. Downloads use groups of 15 symbols, sequential vendor calls, and at most one retry for a failed group. Existing frozen files are reused; NVDA and INTC are always sourced from the unchanged cache.

- [yfinance download parameter documentation](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html).
- [yfinance repair behavior and limitations](https://ranaroussi.github.io/yfinance/advanced/price_repair.html).

Each newly downloaded symbol has a normalized OHLC file and a vendor response table under `data/vendor_raw/`, which includes reported dividends, splits, volume, and repair flags. `data/data_manifest.json` records per-file SHA-256, source symbol, source URL, row counts, date bounds, and settings. The original cache manifest documents the existing NVDA/INTC history as historical rows followed by adjusted and repaired 2026 data; those mixed-vintage source observations are preserved for comparability.

`data/panel_audit.json` records the full stock session audit. Results:

- All 100 stocks and IWV match the original three-stock calendar exactly: 2,204 rows each.
- No missing sessions, extra sessions, duplicate dates, nonpositive or incomplete OHLC rows, or OHLC high/low bound violations.
- No adjusted daily close-to-close move larger than 50% in absolute value.
- No newly downloaded vendor rows are marked `Repaired?=True`. This does not prove no adjustment repair occurred: yfinance documents that some dividend adjustment corrections do not set that flag.
- All original NVDA, GOOG, and INTC cache hashes are unchanged.

The initial restricted-network request failed. The authorized network execution then succeeded. Old BK and MMC vendor tickers failed and were resolved using the issuer references above. No synthetic market history was inserted and no unavailable symbol was silently dropped.
