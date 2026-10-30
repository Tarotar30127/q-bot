# Hidden Markov Model portfolio research

Run the commands below from `HiddenMarkovModel/`. This port brings the portfolio
study from HedgeFundProject commit `5ca98e9` into q-bot's existing HMM module.
The existing DMD scripts, Transformer notebook, legacy study configurations,
data directory, and earlier research results retain their current structure.

## Setup

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the new portfolio studies

```powershell
.\.venv\Scripts\python.exe download_portfolio_study_data.py
.\.venv\Scripts\python.exe run_portfolio_study.py
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
.\.venv\Scripts\python.exe run_swing_research.py --plan swing_research_plan.json
.\.venv\Scripts\python.exe DecisionValueStudy.py --plan value_study_plan.json
```

Those outputs may already be locked. Use a new `--output-dir` for a deliberate
repeat. `write_value_study_report.py` retains q-bot's October 27 reproduction
defaults. Existing configurations retain the old simulator behavior unless the
new cash-cap, per-share execution, or Treasury-reference inputs are supplied.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests
```

The 98 tests cover existing HMM/regime logic plus independent calibration,
initial holdings, stock-split share units, execution fees, Treasury timing and
Sharpe, cash-cap top-ups, and qualification against the mechanical control.
