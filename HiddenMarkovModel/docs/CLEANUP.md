# Cleanup scope

The retained workflows are the six cash-capped plans for the 2023–2026 and
2020–2026 windows, their controls, and both slippage settings. Portfolio
parameters, calendars, cost assumptions, frozen data, and training boundaries
are unchanged. Earlier files can be recovered from q-bot commit `b881710`;
no Git history was rewritten.

| Removed or replaced | Reason |
|---|---|
| HyperplaneStock intraday/forecast CLI | Unused by the current momentum-only portfolio studies. The required market-data and causal HMM functions moved unchanged into `market_data.py`. |
| DecisionValueStudy CLI, bootstrap/audit workflow, and report writer | Superseded research. Required JSON, position accounting, benchmark, and Treasury helpers moved unchanged into `portfolio_utils.py`. |
| Legacy swing-design runner and old study plans | Not part of either current experiment. |
| Unexecuted paper registration | No prospective workflow is being run or maintained here. |
| Old `data/` panel and vendor/authentication caches | The current studies use separately verified, ignored `output/` caches. Local old files are preserved but untracked. Earlier commits still contain them. |
| Value-study reproduction and unconstrained/standalone 80/20 result snapshots | Superseded by the two current cash-cap comparison folders. |
| Two copies each of result-building and plotting scripts | Replaced by `compare_portfolios.py` and `plot_comparison.py`. |
| Repeated allocation/slippage SVG variants | The primary benchmark comparison and all underlying per-cost CSVs are retained. |
| Tests solely for removed workflows | Current accounting, calibration, causal HMM, data, and Sharpe checks remain; extracted helper implementations are checked against the pre-cleanup source. |
| statsmodels requirement | Only the removed cointegration/forecast path used it. SciPy remains pinned for HMM reproducibility. |

`portfolio_study_plan.json` remains necessary as the data-manifest identity for
the 2014-start cache. Changing it would invalidate frozen inputs, so it is retained
with its original bytes and documented separately from the active six plans.
`.gitattributes` preserves the exact JSON bytes across checkouts so Git does
not silently alter frozen plan/universe hashes. Existing LF and CRLF inputs
retain their original manifest identities.

Regression verification reruns all six plans, including their existing
standalone and rolling windows, from the identical frozen input files. All
strategy metrics and yearly Sharpe summaries are compared with the pre-cleanup
values. Numerical comparison results are saved in `results/cleanup_verification.json`.
The DMD scripts and Transformer notebook outside this folder are unchanged.

## Verification result

The 76 current-workflow tests pass. All six studies were fully rerun: all 396
strategy/window/slippage metric records, 396 daily books, 396 trade logs, and
396 completed-position files match their earlier outputs exactly. Comparison,
equity, and yearly Sharpe CSVs also match exactly. All 614 frozen input files
retain their recorded hashes. All 42 selected policies and target-weight files
are unchanged. The largest HMM probability difference is approximately
3.21e-11, below the recorded 1e-9 numerical tolerance. See the
[complete verification record](../results/cleanup_verification.json).
