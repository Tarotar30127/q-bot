# HMM portfolio backtests

This folder contains the two current 100-stock portfolio studies. Each compares
80/20, 90/10, and 95/5 stock/cash floors at zero and $0.005/share execution
slippage. Controls are ungated momentum, the matched mechanical stock floor,
passive ownership of the same stocks, IWV, and cash.

| Testing window | Training observations | Active plans |
|---|---|---|
| January 3, 2023–October 8, 2026 | 2015–2022 | `portfolio_cash_limit_plan.json`, `portfolio_cash_limit_10_plan.json`, `portfolio_cash_limit_5_plan.json` |
| January 2, 2020–October 30, 2026 | 2000–2019 | `portfolio_train2000_cash20_plan.json`, `portfolio_train2000_cash10_plan.json`, `portfolio_train2000_cash5_plan.json` |

## Setup

Run commands from `HiddenMarkovModel/` using Python 3.13:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Download inputs

```powershell
.\.venv\Scripts\python.exe download_portfolio_study_data.py --plan portfolio_study_plan.json
.\.venv\Scripts\python.exe download_portfolio_study_data.py --plan portfolio_train2000_cash20_plan.json
```

`portfolio_study_plan.json` is retained as the frozen data-plan identity for the
2014-start cache; it is not one of the six active backtests. The other plans
reuse these two verified caches. Raw prices, Treasury yields, authentication
caches, and full run books belong in ignored `output/`, not Git.

Downloads are checked for complete stock histories, OHLC validity, and matching
benchmark sessions during evaluation. Existing file hashes must match their
manifest. No missing prices or stocks are silently substituted. The 2000 study
uses October 1999 warm-up prices; unavailable early trend signals remain inactive.

## Run the backtests

```powershell
.\.venv\Scripts\python.exe run_portfolio_study.py --plan portfolio_cash_limit_plan.json
```

The downloader freezes a separate 100-stock-plus-IWV panel with June 2014 warm-up
data, corporate actions, reconstructed historical quoted opens for per-share
costs, Treasury yields, hashes, and a complete-calendar audit. Existing `data/`
files are preserved. Each version fits independently from 2015 through 2022:
zero slippage versus $0.005 per actual share per execution. Both start each book
with $1 million equally invested in all 100 stocks and first apply model targets
20 trading sessions later. Signals and model selection use prior information.

Sharpe uses historical FRED DGS10 with a two-stock-session publication lag and
matching ACT/365 holding intervals. Cash income remains a separate DGS3MO
approximation. Execution costs include entry, changes in holdings, and final
liquidation. Dividend/split-adjusted prices approximate reinvested returns;
dividends are not credited twice. Quoted-price reconstruction depends on vendor
corporate-action accuracy.

`portfolio_study_plan.json` allows subsequent cash allocations.
`portfolio_cash_limit_plan.json` caps cash at 20% of post-cost equity at daily
opens after the initial holding. Any stock-allocation shortfall is spread
equally across all 100 stocks. Price changes or interest can trigger equal-dollar
top-ups between scheduled rebalances. Intraday breaches cannot be enforced with
daily bars, and final research liquidation is exempt.

Constrained training compares each state candidate's daily net returns against
the same mechanical equal-weight 80%-stock fallback, with matching costs and
zero calibration interest. A trained policy needs a positive conservative
incremental return bound and sufficient model-active observations. Compulsory
stock exposure alone cannot qualify a state. The policy label `cash` means the
equal-weight stock-floor fallback in the constrained experiment.

The continuous book covers January 3, 2023 through October 8, 2026 without annual
resets. Standalone 2023-2025 and 2026 books, plus yearly expanding-training checks,
restart independently. Controls include the same 100-stock passive portfolio,
ungated momentum, IWV, cash, and the matching constrained allocation rule. The
chart includes only both model versions and the full-stock passive portfolio.

## Results and limitations

Small, reviewable result snapshots are included here:

- [Unconstrained study](results/portfolio_study_20261030/REPORT.md): the two main
  fits chose cash after the first rebalance; gains largely reflect cash interest.
- [20% cash-cap study](results/portfolio_cash_limit_20261030/REPORT.md): returns
  match the mechanical stock-floor control. Required exposure makes the portfolio
  trade but does not establish added value from training.

| Continuous study | Zero-slippage model | $0.005/share model | Full-stock passive, zero slippage |
|---|---:|---:|---:|
| Cash allowed | +24.85% | +24.83% | +161.34% |
| Cash capped at 20% | +100.93% | +100.89% | +161.34% |

These historical periods were previously studied, and the current stock list has
selection/survivorship bias. Returns are research results, not an unseen test or
evidence of a trading advantage. IWV is an ETF proxy for the Russell 3000, not the
official index. No brokerage orders are placed.

Full daily books, trade logs, policy locks, source snapshots, data, and generated
reports are written under ignored `output/`. Summary snapshots under `results/`
exclude raw market data and authentication caches. An existing protocol is locked;
use `--output-dir output/a_new_run` for a deliberate reproduction. Frozen downloads
are reused only when their recorded hashes match.

## Existing studies

The earlier q-bot configurations continue to use the existing `data/` directory:

```powershell
python plot_comparison.py --window 2023
python plot_comparison.py --window 2020
```

## Results

- [2023–2026 comparison](results/cash_cap_comparison_20261102/REPORT.md)
- [2020–2026 comparison](results/train2000_comparison_20261102/REPORT.md)
- [2023–2026 yearly Sharpe mean and SD](results/cash_cap_comparison_20261102/SHARPE_STATISTICS.md)
- [2020–2026 yearly Sharpe mean and SD](results/train2000_comparison_20261102/SHARPE_STATISTICS.md)

Each result folder contains the report, primary graph, comparison and equity
CSVs, yearly Sharpe observations, and verification. [Methodology](docs/METHODOLOGY.md)
explains portfolio timing, costs, Treasury rates, and statistical interpretation.
[Cleanup notes](docs/CLEANUP.md) identify what was removed and how to recover
older studies from Git history.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests
```

The 76 current-workflow tests pass. All six reruns match the earlier 396
strategy/window/slippage result records exactly; see
[regression verification](results/cleanup_verification.json).

Tests cover causal HMM probabilities, fixed-share accounting, execution costs,
Treasury timing, cap enforcement, policy calibration, and yearly Sharpe summaries.
The cleanup also reruns all six studies on the same frozen inputs and compares
all strategies/windows and yearly Sharpe results with their earlier values.
Historical reuse and the selected stock universe retain survivorship and
selection bias; these are retrospective research results, not independent
prospective evidence. No broker orders are placed.
