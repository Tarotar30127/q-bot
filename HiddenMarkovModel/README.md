# HMM portfolio backtests

This folder contains the two current 100-stock portfolio studies. Each compares
80/20, 90/10, and 95/5 stock/cash floors at zero and $0.005/share execution
slippage. Controls are ungated momentum, the matched mechanical stock floor,
passive ownership of the same stocks, IWV, and cash.

| Testing window | Training observations | Active plans |
|---|---|---|
| January 3, 2023–October 8, 2026 | 2015–2022 | `portfolio_cash_limit_plan.json`, `portfolio_cash_limit_10_plan.json`, `portfolio_cash_limit_5_plan.json` |
| January 2, 2020–October 30, 2026 | 2000–2019 | `portfolio_train2000_cash20_plan.json`, `portfolio_train2000_cash10_plan.json`, `portfolio_train2000_cash5_plan.json` |

## Setup (Windows / PowerShell)

Use Python 3.13 and Git. These instructions create `.venv` at the **repo root**
and run the scripts from **`HiddenMarkovModel`**. Virtual environment activation
is optional; every command below calls its Python executable directly.

For a new checkout, run this from the directory where you want the repo:

```powershell
git clone https://github.com/Tarotar30127/q-bot.git
cd .\q-bot
```

For an existing checkout, open PowerShell at the repo root and update to the
current `main` branch before continuing:

```powershell
# Replace this example with the location of your own checkout.
cd "C:\Users\tseba\VSCode_Projects\q-bot"
git switch main
git pull --ff-only
```

From the repo root, create the environment once and install its dependencies:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\HiddenMarkovModel\requirements.txt
cd .\HiddenMarkovModel
```

If the root `.venv` already exists, reuse it and run the dependency-install
command. Use `py -3.13 -m venv .venv` instead of `python -m venv .venv` if your
system's `python` points to a different version.

**Keep PowerShell in `HiddenMarkovModel` for every command below.** The prompt
should end in `q-bot\HiddenMarkovModel>`. From here, the root environment is
`..\.venv\Scripts\python.exe`. A plain `pip install` can install into a different
Python and leave the backtest dependencies missing.

## Download inputs

Run the downloader for each testing window you want to use. Both commands are
needed if you intend to run both windows:

```powershell
# Inputs for all three 2023-2026 allocation plans.
..\.venv\Scripts\python.exe download_portfolio_study_data.py --plan portfolio_study_plan.json

# Inputs for all three 2020-2026 allocation plans.
..\.venv\Scripts\python.exe download_portfolio_study_data.py --plan portfolio_train2000_cash20_plan.json
```

`portfolio_study_plan.json` is the frozen data-plan identity for the 2014-start
cache; it is not one of the six active backtests. The six plans reuse these two
verified caches. Inputs and full run books live under ignored `output/` and are
not included when you clone the repo. A run needs its corresponding download to
finish successfully first.

Downloads are checked for complete stock histories, OHLC validity, and matching
benchmark sessions during evaluation. Existing file hashes must match their
manifest. No missing prices or stocks are silently substituted. The 2000 study
uses October 1999 warm-up prices; unavailable early trend signals remain inactive.

## Run your own backtests

These commands reproduce the two 80/20 studies in new folders, preserving any
previous runs. Generate a fresh timestamp each time you repeat them:

```powershell
$runStamp = Get-Date -Format "yyyyMMdd_HHmmss"

# Train on 2015-2022; test January 2023-October 8, 2026.
..\.venv\Scripts\python.exe run_portfolio_study.py --plan portfolio_cash_limit_plan.json --output-dir "output/manual_2023_80_20_$runStamp"

# Train on 2000-2019; test January 2020-October 30, 2026.
..\.venv\Scripts\python.exe run_portfolio_study.py --plan portfolio_train2000_cash20_plan.json --output-dir "output/manual_2020_80_20_$runStamp"
```

Each command trains and evaluates both zero-slippage and $0.005/share versions,
including the controls and the evaluation windows declared in its plan. Wait
for `Completed: .../REPORT.md` before opening the finished report. The 2023 plan
also includes yearly expanding-training checks; the 2020 plan keeps its main
training window fixed.

To run a different stock/cash allocation, choose the matching plan:

| Stock/cash allocation | 2023-2026 testing plan | 2020-2026 testing plan |
|---|---|---|
| 80/20 | `portfolio_cash_limit_plan.json` | `portfolio_train2000_cash20_plan.json` |
| 90/10 | `portfolio_cash_limit_10_plan.json` | `portfolio_train2000_cash10_plan.json` |
| 95/5 | `portfolio_cash_limit_5_plan.json` | `portfolio_train2000_cash5_plan.json` |

For example, run the 2020-2026, 95/5 version in its own folder:

```powershell
$runStamp = Get-Date -Format "yyyyMMdd_HHmmss"
..\.venv\Scripts\python.exe run_portfolio_study.py --plan portfolio_train2000_cash5_plan.json --output-dir "output/manual_2020_95_5_$runStamp"
```

Use a distinct output folder for each plan and repeat. An existing
`protocol.json` locks a run against overwriting, even if that run was interrupted.
Keep the old folder and pass a fresh `--output-dir` when restarting.

## Find your results

From `HiddenMarkovModel`, open the output directory:

```powershell
explorer .\output
```

For the manual commands above, open `manual_2023_80_20_<timestamp>` or
`manual_2020_80_20_<timestamp>`. The files are **directly inside the folder passed
to `--output-dir`**, with no extra `results` subfolder:

| File or folder | Contents |
|---|---|
| `REPORT.md` | Summary of returns, Sharpe, drawdown, and selected policies |
| `metrics.csv` | Metrics for every strategy, evaluation window, and cost version |
| `equity_curves.svg` | Both model cost versions and the same-stock passive portfolio |
| `slippage_comparison.svg` | Comparison of the two model cost versions |
| `zero_slippage/`, `half_cent_slippage/` | Training outputs and per-window daily books, trades, and position records |
| `protocol.json`, `sources/` | Locked settings, input/source hashes, and source snapshots |

To read a Markdown report with formatting in VS Code, open it and press
`Ctrl+Shift+V`. Results committed under `results/` are saved reference summaries;
the manual commands write your new results under `output/`.

## Rebuild allocation comparisons (optional)

The comparison builder uses the `output_dir` recorded in each of the three
checked-in allocation plans. **It does not discover timestamped manual runs or
read their `--output-dir` overrides.** The manual reports above are complete
individual studies; rebuilding a combined allocation report is a separate step.

For a first reproduction in the plans' default output locations, run the plans
without `--output-dir`, then build the comparisons:

```powershell
..\.venv\Scripts\python.exe run_portfolio_study.py --plan portfolio_cash_limit_plan.json
..\.venv\Scripts\python.exe run_portfolio_study.py --plan portfolio_cash_limit_10_plan.json
..\.venv\Scripts\python.exe run_portfolio_study.py --plan portfolio_cash_limit_5_plan.json
..\.venv\Scripts\python.exe compare_portfolios.py --window 2023

..\.venv\Scripts\python.exe run_portfolio_study.py --plan portfolio_train2000_cash20_plan.json
..\.venv\Scripts\python.exe run_portfolio_study.py --plan portfolio_train2000_cash10_plan.json
..\.venv\Scripts\python.exe run_portfolio_study.py --plan portfolio_train2000_cash5_plan.json
..\.venv\Scripts\python.exe compare_portfolios.py --window 2020
```

Run only the group for your desired window. If the default runs are already
complete, skip the runner commands and rebuild just the comparison. If a default
run is locked but incomplete, it must be resolved before using this workflow;
a timestamped manual rerun will not replace the builder's input.

These comparison commands overwrite the summaries in
`results/cash_cap_comparison_20261102/` and
`results/train2000_comparison_20261102/`, respectively. They generate the combined
allocation/benchmark SVG, CSVs, and the mean and sample standard deviation across
yearly Sharpe ratios (including partial 2026). They read completed daily books
without training models again.

Optional PNG previews require matplotlib in the same environment:

```powershell
..\.venv\Scripts\python.exe -m pip install matplotlib
..\.venv\Scripts\python.exe plot_comparison.py --window 2023
..\.venv\Scripts\python.exe plot_comparison.py --window 2020
```

## Run and visualize in Jupyter

Open [current_experiments.ipynb](current_experiments.ipynb) to run the same
experiments and generate allocation/benchmark equity curves, return differences,
drawdowns, yearly Sharpe charts, and summary tables.

Install the notebook tools from the **repo root** using the same environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\HiddenMarkovModel\requirements-notebook.txt
.\.venv\Scripts\python.exe -m jupyter lab .\HiddenMarkovModel\current_experiments.ipynb
```

In Jupyter, select the kernel backed by the root `.venv`; the first code cell
prints its executable. If that kernel is not listed, register it from the repo
root and select **q-bot (.venv)**:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name q-bot --display-name "q-bot (.venv)"
```

Run the notebook from top to bottom. The configuration cell offers three modes:

- `reference` (default): visualize the committed comparison snapshots immediately.
- `existing`: rebuild verified comparisons from completed run folders. The
  `EXISTING_RUN_DIRS` setting can point to your timestamped manual runs.
- `fresh`: run all three allocations for each selected testing window in new
  output folders. Set `DOWNLOAD_INPUTS = True` if inputs have not been downloaded.

Choose one or both `WINDOWS` and the cost version to plot. Every comparison keeps
both cost versions in its tables. Notebook exports go to the printed
`SESSION_DIR` under ignored `output/`; reference reports stay intact. Clear cell
outputs before committing the notebook. The backtest implementation and checks
are shared with the command-line workflow.

## Common setup errors

| Error | Fix |
|---|---|
| `can't open file ...\\q-bot\\download_portfolio_study_data.py` | You are at the repo root. Run `cd .\HiddenMarkovModel` first. |
| `..\.venv\Scripts\python.exe` is not recognized | Check your folder. At the repo root use `.\.venv\Scripts\python.exe`; inside `HiddenMarkovModel` use `..\.venv\Scripts\python.exe`. If the root environment is missing, create it using the setup steps. |
| `ModuleNotFoundError: No module named 'numpy'` (or another dependency) | From `HiddenMarkovModel`, run `..\.venv\Scripts\python.exe -m pip install -r requirements.txt`. Plain `pip` may target your system Python. |
| A `portfolio_train2000_..._plan.json` file is missing | From the repo root, run `git switch main` and `git pull --ff-only`, then return to `HiddenMarkovModel`. |
| `manifest.json` is missing | Finish the corresponding input-download command before running that backtest. |
| `Existing protocol is locked` | Generate a new timestamp and use a fresh `--output-dir`. |

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
..\.venv\Scripts\python.exe -m pytest -q tests
```

The 80 current-workflow tests pass. All six reruns match the earlier 396
strategy/window/slippage result records exactly; see
[regression verification](results/cleanup_verification.json).

Tests cover causal HMM probabilities, fixed-share accounting, execution costs,
Treasury timing, cap enforcement, policy calibration, and yearly Sharpe summaries.
The cleanup also reruns all six studies on the same frozen inputs and compares
all strategies/windows and yearly Sharpe results with their earlier values.
Historical reuse and the selected stock universe retain survivorship and
selection bias; these are retrospective research results, not independent
prospective evidence. No broker orders are placed.
