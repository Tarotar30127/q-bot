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
.\.venv\Scripts\python.exe run_portfolio_study.py --plan portfolio_cash_limit_10_plan.json
.\.venv\Scripts\python.exe run_portfolio_study.py --plan portfolio_cash_limit_5_plan.json
.\.venv\Scripts\python.exe run_portfolio_study.py --plan portfolio_train2000_cash20_plan.json
.\.venv\Scripts\python.exe run_portfolio_study.py --plan portfolio_train2000_cash10_plan.json
.\.venv\Scripts\python.exe run_portfolio_study.py --plan portfolio_train2000_cash5_plan.json
.\.venv\Scripts\python.exe compare_portfolios.py --window 2023
.\.venv\Scripts\python.exe compare_portfolios.py --window 2020
```

The default runner plan is 80/20 for the 2023 window. Existing protocols are
locked: a deliberate repeat requires a new `--output-dir`. Comparisons read the
output paths declared by their three plans; use separately versioned plans
when preserving multiple full reproductions. Reports and SVG graphs are
rebuilt from completed runs, without fitting models again.

Optional PNG previews use an installation of Python with pandas and matplotlib:

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
