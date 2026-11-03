"""Render the saved comparison CSV with the workstation's installed matplotlib."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

folder = Path(__file__).resolve().parent
curves = pd.read_csv(folder / "equity_half_cent_slippage.csv", index_col="Date", parse_dates=True)
metrics = pd.read_csv(folder / "comparison.csv")
models = metrics.loc[metrics.strategy.eq("hmm")]
assert models.matches_stock_floor.all()
assert curves.shape == (1717, 5) and not curves.isna().any().any()
colors = ["#0072B2", "#D55E00", "#009E73", "#9B59B6", "#52525B"]
labels = ["80/20", "90/10", "95/5", "IWV (Russell 3000 ETF proxy)", "100-stock buy and hold"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#D4D4D8"})
fig, (equity, difference) = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1.25], "hspace": 0.20})
fig.subplots_adjust(left=0.09, right=0.97, top=0.85, bottom=0.14)
fig.suptitle("2000–2019 training: 80/20 vs 90/10 vs 95/5", x=0.09, y=0.975,
             ha="left", fontsize=21, fontweight="bold")
fig.text(0.09, 0.93, "January 2, 2020 – October 30, 2026  |  $1 million start  |  $0.005/share per execution", color="#52525B", parse_math=False)
fig.text(0.09, 0.90, "All 100 stocks; frozen 2000–2019 training; cash caps apply after the first 20-session holding period.", color="#52525B")
for index, (column, color, label) in enumerate(zip(curves, colors, labels)):
    total_return = curves[column].iloc[-1] / 1_000_000 - 1
    equity.plot(curves.index, curves[column], color=color, linewidth=2.1,
                linestyle="--" if index >= 3 else "-", label=f"{label}: {total_return:+.2%}")
equity.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value / 1e6:.2f}m"))
equity.set_ylabel("Portfolio equity after costs")
equity.grid(axis="y", color="#E4E4E7", linewidth=0.7)
equity.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#E4E4E7", fontsize=10)
equity.margins(x=0)
for column, color, label in zip(curves.columns[:3], colors, labels):
    gap = (curves[column] - curves.iloc[:, 0]) / 1_000_000 * 100
    difference.plot(curves.index, gap, color=color, linewidth=2, label=label)
difference.set_title("Additional cumulative return over 80/20", loc="left", fontsize=12, pad=10)
difference.set_ylabel("Percentage points")
difference.grid(axis="y", color="#E4E4E7", linewidth=0.7)
difference.xaxis.set_major_locator(mdates.YearLocator())
difference.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
difference.margins(x=0)
fig.text(0.09, 0.065, "All three fitted models match their mechanical stock-floor controls; these gains do not establish HMM selection value.", fontsize=10, color="#52525B")
fig.text(0.09, 0.035, "Retrospective sensitivity test on a selected stock universe; historical reuse and survivorship bias apply.", fontsize=10, color="#71717A")
fig.savefig(folder / "comparison.png", dpi=180, facecolor="white")
fig.savefig(folder / "comparison.svg", facecolor="white")
plt.close(fig)
print(folder / "comparison.png")
