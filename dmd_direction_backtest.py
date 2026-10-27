"""
DMD Direction Backtest
======================
Backtest-only trading simulation built on the attached single-stock DMD
direction predictor (Hankel/time-delay Dynamic Mode Decomposition).

STRATEGY
--------
Every Rebalance_Portfolio trading days (1 = daily, 2 = every 2 days, ...), for each
stock:
  1. DMD "heat map" step: for every sampling-window size n in WINDOW_SIZES
     (3..24 days), look back over the last SELECTION_LOOKBACK days and measure
     how often "DMD forecast HORIZON_DAYS ahead is above the DMD fit of the
     last day" matched "price actually rose HORIZON_DAYS later". Pick the
     window size with the best hit rate (the best cell of the heat map).
  2. Use that window on today's most recent n closes -> predicted UP or
     NOT-UP (the attached `prediction` logic: forecast above/below the fit of
     the last sampled day).
Then trade:
  * SELL everything predicted to decline (long-only: "sell" means exit, no shorting).
  * Split the whole portfolio value EQUALLY across the stocks predicted to rise
    (whole shares). If nothing is predicted to rise, the portfolio sits in cash.

COSTS AND METRICS
-----------------
  * Commission of COMMISSION_PER_SHARE ($0.005) on every share bought or sold.
  * Reports gross (before costs) vs net (after costs) results, total costs,
    shares traded, annualized Sharpe ratio, volatility, drawdown, and an
    equal-weight buy-and-hold benchmark.

ASSUMPTIONS (change in the config if you disagree)
--------------------------------------------------
  * Signals use closes through day t and trades execute at day t's close.
  * Cash earns nothing. The risk-free rate is only used in the Sharpe ratio: by default
    the daily 10-year Treasury yield (Yahoo ticker ^TNX), or the fixed RISK_FREE_RATE
    if USE_10Y_TREASURY_FOR_RISK_FREE is False or the download fails.
  * No lookahead: the window selection at day t only uses slides whose outcome
    was already known by day t. NOTE: the attached heat-map code compares every
    sliding window against the *final* HORIZON days of the data set, which would
    leak the future in a backtest, so here each slide is compared with the days
    that actually followed it.
"""

import os

# One math-library thread per process. The matrices here are tiny, so extra
# BLAS threads only add overhead (and memory) -- especially with many workers.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import time
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Universe ──────────────────────────────────────────────────────────────────
STOCKS = {
    "NVDA": "NVIDIA",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "BRK.B": "Berkshire Hathaway",
    "LLY": "Eli Lilly",
    "MU": "Micron Technology",
    "WMT": "Walmart",
    "JPM": "JPMorgan Chase",
    "AMD": "Advanced Micro Devices",
    "XOM": "Exxon Mobil",
    "JNJ": "Johnson & Johnson",
    "INTC": "Intel",
    "COST": "Costco",
    "CVX": "Chevron",
    "CSCO": "Cisco",
    "KO": "Coca-Cola",
    "PG": "Procter & Gamble",
    "ORCL": "Oracle",
    "UNH": "UnitedHealth Group",
    "MRK": "Merck",
    "BAC": "Bank of America",
    "HD": "Home Depot",
    "CAT": "Caterpillar",
    "AMAT": "Applied Materials",
    "GE": "GE Aerospace",
    "LRCX": "Lam Research",
    "MS": "Morgan Stanley",
    "GS": "Goldman Sachs",
    "RTX": "RTX",
    "TXN": "Texas Instruments",
    "TMO": "Thermo Fisher Scientific",
    "WFC": "Wells Fargo",
    "AMGN": "Amgen",
    "LIN": "Linde",
    "IBM": "IBM",
    "AXP": "American Express",
    "C": "Citigroup",
    "KLAC": "KLA",
    "DIS": "Walt Disney",
    "GILD": "Gilead Sciences",
    "ADI": "Analog Devices",
    "PEP": "PepsiCo",
    "QCOM": "Qualcomm",
    "VZ": "Verizon",
    "APH": "Amphenol",
    "MCD": "McDonald's",
    "ABT": "Abbott Laboratories",
    "BLK": "BlackRock",
    "UNP": "Union Pacific",
    "DE": "Deere & Company",
    "PFE": "Pfizer",
    "WELL": "Welltower",
    "NEE": "NextEra Energy",
    "DHR": "Danaher",
    "TJX": "TJX Companies",
    "SCHW": "Charles Schwab",
    "T": "AT&T",
    "BA": "Boeing",
    "ETN": "Eaton",
    "CB": "Chubb",
    "VRTX": "Vertex Pharmaceuticals",
    "LMT": "Lockheed Martin",
    "SPGI": "S&P Global",
    "WDC": "Western Digital",
    "PGR": "Progressive",
    "BMY": "Bristol Myers Squibb",
    "MO": "Altria",
    "COF": "Capital One",
    "MDT": "Medtronic",
    "BKNG": "Booking Holdings",
    "ADP": "Automatic Data Processing",
    "CVS": "CVS Health",
    "VLO": "Valero Energy",
    "PLD": "Prologis",
    "SYK": "Stryker",
    "NEM": "Newmont",
    "MCK": "McKesson",
    "SBUX": "Starbucks",
    "LOW": "Lowe's",
    "PH": "Parker-Hannifin",
    "GLW": "Corning",
    "SO": "Southern Company",
    "ADBE": "Adobe",
    "MAR": "Marriott International",
    "BNY": "Bank of New York Mellon",
    "DUK": "Duke Energy",
    "CSX": "CSX",
    "WM": "Waste Management",
    "INTU": "Intuit",
    "COP": "Conoco Phillips",
    "FCX": "Freeport-McMoRan",
    "GD": "General Dynamics",
    "USB": "U.S. Bancorp",
    "MNST": "Monster Beverage",
    "MMM": "3M",
    "PNC": "PNC Financial Services",
    "CTAS": "Cintas",
    "EMR": "Emerson Electric",
    "JCI": "Johnson Controls",
}
TICKERS = list(STOCKS)

# ── Strategy / data config ────────────────────────────────────────────────────
Rebalance_Portfolio = 1                  # trade every N trading days (integer >= 1):
                                         # 1 = every day, 2 = every 2 days, ...
HORIZON_DAYS        = 1                  # how many days ahead the DMD predicts direction
                                         # (1..10). Independent of Rebalance_Portfolio; you
                                         # may want them to match (e.g. both 5).
WINDOW_SIZES        = list(range(3, 25)) # sampling-window sizes tried (heat map x-axis)
SELECTION_LOOKBACK  = 60                 # past slides used to score each window size
HISTORY_DAYS        = 2000                # trading days of history to download
                                         # (the first ~85 are warm-up)

# ── Trading / reporting config ────────────────────────────────────────────────
STARTING_CASH        = 1_000_000.0
COMMISSION_PER_SHARE = 0.005             # $ per share, charged on buys AND sells
USE_10Y_TREASURY_FOR_RISK_FREE = True    # Sharpe risk-free rate = daily 10-yr Treasury yield
RISK_FREE_RATE       = 0.0               # annual fixed rate (e.g. 0.04 = 4%); used when the
                                         # flag above is False or the download fails
TRADING_DAYS_PER_YEAR = 252

# ── Runtime config ────────────────────────────────────────────────────────────
USE_PARALLEL = True                      # falls back to serial automatically if the
                                         # process pool breaks (e.g. inside Spyder)
MAX_WORKERS  = os.cpu_count() or 1
SHOW_PLOT    = True


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

def fetch_prices(tickers, history_days):
    """Daily closes (split-adjusted 'Close', as in the attached code)."""
    import yfinance as yf

    end   = datetime(2015, 7, 10)          #.today()
    
    start = end - timedelta(days=int(history_days * 365 / 252) + 10)
    # Yahoo writes share-class tickers with a dash (BRK.B -> BRK-B); results are
    # renamed back to the tickers as you wrote them.
    yahoo = {t: t.replace(".", "-") for t in tickers}
    print(f"Downloading {len(tickers)} tickers from {start.date()} to {end.date()} ...")
    raw = yf.download(list(yahoo.values()), start=start, end=end, progress=False, auto_adjust=False)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].rename(columns={"Close": yahoo[tickers[0]]})
    close = close.rename(columns={v: k for k, v in yahoo.items()})

    close = close.dropna(axis=1, thresh=int(len(close) * 0.8))
    close = close.ffill().dropna()
    missing = [t for t in tickers if t not in close.columns]
    if missing:
        print(f"  Dropped (insufficient data): {missing}")
    return close[[t for t in tickers if t in close.columns]]


# ══════════════════════════════════════════════════════════════════════════════
# DMD DIRECTION PREDICTOR  (the attached `prediction` logic, no plots/prints)
# ══════════════════════════════════════════════════════════════════════════════

def dmd_predicts_up(segment, horizon):
    """
    Fit Hankel DMD to `segment` (the last n closes). Returns True if the DMD
    forecast `horizon` days past the end is above the DMD fit of the last
    sampled day; False if below (or if the fit is unusable).
    """
    n  = len(segment)
    ws = int(n / 2 + 1)                       # snapshot (Hankel row) length
    X  = np.column_stack([segment[a:a + ws] for a in range(n - ws + 1)])
    X1, X2 = X[:, :-1], X[:, 1:]

    try:
        with np.errstate(all="ignore"):
            U, S, Vh = np.linalg.svd(X1, full_matrices=False)
            r      = min(len(S), 25)
            Ur     = U[:, :r]
            Vr     = Vh[:r, :]
            Sr_inv = np.linalg.inv(np.diag(S[:r]))

            A_tilde = Ur.T @ X2 @ Vr.T @ Sr_inv
            eigs, W = np.linalg.eig(A_tilde)

            valid = np.where(np.real(eigs) > 0)[0]   # drop zero/negative-real eigenvalues
            if len(valid) == 0:
                return False
            eigs_v = eigs[valid]
            Phi    = X2 @ Vr.T @ Sr_inv @ W[:, valid]

            b     = np.linalg.lstsq(Phi, X, rcond=None)[0][:, 0]
            omega = np.log(eigs_v)

            row0 = Phi[0, :]                          # first row = the price series itself
            ref  = (row0 @ (b * np.exp(omega * (n - 1)))).real
            fut  = (row0 @ (b * np.exp(omega * (n - 1 + horizon)))).real
        return bool(fut > ref)                        # NaN/inf compare False
    except np.linalg.LinAlgError:
        return False


def precompute_stock(job):
    """
    For ONE stock, precompute for every window size and every end-day e:
      pred_up[w, e] : DMD predicts a rise `horizon` days after day e
      hit[w, e]     : that prediction matched what actually happened
                      (only meaningful for e <= T-1-horizon)
    Module-level so it can be sent to worker processes.
    """
    prices, windows, horizon = job
    T = len(prices)
    pred_up = np.zeros((len(windows), T), dtype=bool)
    for wi, n in enumerate(windows):
        for e in range(n - 1, T):
            pred_up[wi, e] = dmd_predicts_up(prices[e - n + 1:e + 1], horizon)

    realized_up = np.zeros(T, dtype=bool)
    realized_up[:T - horizon] = prices[horizon:] > prices[:T - horizon]
    hit = (pred_up == realized_up[None, :])
    return pred_up, hit


def first_signal_index(windows, lookback, horizon):
    """Earliest day with a full set of scored slides for every window size."""
    return (max(windows) - 1) + (lookback - 1) + horizon


def select_signals(pred_up, hit, windows, lookback, horizon):
    """
    Heat-map window selection, walk-forward. On day t, window w is scored by its
    hit rate over the `lookback` most recent slides whose outcome is already known
    (end-day e <= t - horizon). The best window's prediction for day t is the signal.
    Returns (signal[T] bool, chosen_window[T] int, chosen_hit_rate[T] float).
    """
    n_w, T = pred_up.shape
    t0 = first_signal_index(windows, lookback, horizon)

    csum = np.concatenate([np.zeros((n_w, 1)), np.cumsum(hit.astype(float), axis=1)], axis=1)
    ts   = np.arange(t0, T)
    hi   = ts - horizon + 1                    # exclusive end index -> e <= t - horizon
    lo   = hi - lookback
    acc  = (csum[:, hi] - csum[:, lo]) / lookback          # (n_w, len(ts))
    best = np.argmax(acc, axis=0)

    signal = np.zeros(T, dtype=bool)
    chosen = np.zeros(T, dtype=int)
    rate   = np.full(T, np.nan)
    signal[t0:] = pred_up[best, ts]
    chosen[t0:] = np.array(windows)[best]
    rate[t0:]   = acc[best, np.arange(len(ts))]
    return signal, chosen, rate


def compute_signals(prices_df):
    """Signals for every stock (parallel across stocks, serial fallback)."""
    values = prices_df.to_numpy(dtype=float)
    jobs   = [(values[:, j].copy(), WINDOW_SIZES, HORIZON_DAYS) for j in range(values.shape[1])]
    t_start = time.perf_counter()
    results = None

    if USE_PARALLEL and MAX_WORKERS > 1 and len(jobs) > 1:
        try:
            print(f"Computing DMD signals for {len(jobs)} stocks on up to {MAX_WORKERS} processes ...")
            with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(jobs))) as ex:
                results = list(ex.map(precompute_stock, jobs))
        except (BrokenProcessPool, OSError) as exc:
            print(f"  Process pool failed ({type(exc).__name__}); falling back to serial. "
                  f"(Running from a terminal instead of Spyder/Jupyter usually fixes this.)")
            results = None

    if results is None:
        print(f"Computing DMD signals for {len(jobs)} stocks serially ...")
        results = []
        for k, job in enumerate(jobs, 1):
            results.append(precompute_stock(job))
            print(f"  {k}/{len(jobs)} stocks done", end="\r")
        print()

    print(f"  Signals ready in {time.perf_counter() - t_start:.1f}s")

    signals, windows_chosen, rates = [], [], []
    for pred_up, hit in results:
        s, w, r = select_signals(pred_up, hit, WINDOW_SIZES, SELECTION_LOOKBACK, HORIZON_DAYS)
        signals.append(s); windows_chosen.append(w); rates.append(r)

    return (np.column_stack(signals), np.column_stack(windows_chosen), np.column_stack(rates),
            first_signal_index(WINDOW_SIZES, SELECTION_LOOKBACK, HORIZON_DAYS))


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def simulate(prices_df, signals, t_first):
    """
    Walk forward from t_first. On each rebalance day: sell all predicted-down
    holdings, then rebalance the whole portfolio to equal dollar weights across
    predicted-up stocks. Every share traded pays COMMISSION_PER_SHARE.
    """
    px      = prices_df.to_numpy(dtype=float)
    dates   = prices_df.index
    tickers = list(prices_df.columns)
    T, N    = px.shape
    fee     = COMMISSION_PER_SHARE

    cash      = STARTING_CASH
    shares    = np.zeros(N, dtype=np.int64)
    cum_cost  = 0.0
    trades    = []

    equity, cum_costs, idx = [STARTING_CASH], [0.0], [dates[t_first - 1]]
    n_rebalances = 0

    def sell(j, qty, t):
        nonlocal cash, cum_cost
        commission = qty * fee
        cash      += qty * px[t, j] - commission
        cum_cost  += commission
        shares[j] -= qty
        trades.append((dates[t], tickers[j], "SELL", int(qty), px[t, j], commission))

    def buy(j, qty, t):
        nonlocal cash, cum_cost
        commission = qty * fee
        cash      -= qty * px[t, j] + commission
        cum_cost  += commission
        shares[j] += qty
        trades.append((dates[t], tickers[j], "BUY", int(qty), px[t, j], commission))

    for t in range(t_first, T):
        if (t - t_first) % Rebalance_Portfolio == 0:
            n_rebalances += 1
            up = signals[t]

            for j in np.where(~up & (shares > 0))[0]:           # exit predicted decliners
                sell(j, shares[j], t)

            if up.any():
                equity_now = cash + float(shares @ px[t])
                alloc      = equity_now / up.sum()
                target     = np.where(up, np.floor(alloc / (px[t] + fee)), 0).astype(np.int64)

                for j in np.where(shares > target)[0]:           # trim over-weight winners
                    sell(j, shares[j] - target[j], t)
                for j in np.where(shares < target)[0]:           # top up / open positions
                    qty = int(min(target[j] - shares[j], cash // (px[t, j] + fee)))
                    if qty > 0:
                        buy(j, qty, t)

        equity.append(cash + float(shares @ px[t]))
        cum_costs.append(cum_cost)
        idx.append(dates[t])

    equity    = pd.Series(equity, index=idx, name="net")
    cum_costs = pd.Series(cum_costs, index=idx, name="cum_costs")
    trade_log = pd.DataFrame(trades, columns=["date", "ticker", "action", "shares", "price", "commission"])
    return equity, cum_costs, trade_log, n_rebalances


def benchmark_equity(prices_df, t_first):
    """Equal-weight buy-and-hold from the same start day, paying commission once."""
    px    = prices_df.to_numpy(dtype=float)
    N     = px.shape[1]
    sh    = (STARTING_CASH / N) / (px[t_first] + COMMISSION_PER_SHARE)   # fractional shares
    vals  = [STARTING_CASH] + [float(sh @ px[t]) for t in range(t_first, len(px))]
    return pd.Series(vals, index=[prices_df.index[t_first - 1]] + list(prices_df.index[t_first:]),
                     name="benchmark"), float(sh.sum() * COMMISSION_PER_SHARE)


# ══════════════════════════════════════════════════════════════════════════════
# METRICS / REPORT
# ══════════════════════════════════════════════════════════════════════════════

def fetch_risk_free(dates):
    """
    Annual risk-free rate for each backtest date, as a decimal (0.043 = 4.3%).
    Uses the 10-year Treasury yield (Yahoo ticker ^TNX, quoted in percent),
    forward-filled across non-trading days. Falls back to the fixed
    RISK_FREE_RATE if disabled or the download fails.
    Returns (rate, description) where rate is a pd.Series indexed by `dates`
    or a float.
    """
    if not USE_10Y_TREASURY_FOR_RISK_FREE:
        return RISK_FREE_RATE, f"fixed {RISK_FREE_RATE * 100:.2f}%"
    try:
        import yfinance as yf
        raw = yf.download("^TNX", start=dates[0] - timedelta(days=10),
                          end=dates[-1] + timedelta(days=3), progress=False, auto_adjust=False)
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        if close.empty:
            raise ValueError("no ^TNX data returned")
        if close.median() > 20:          # some feeds quote yield x10; normalise to percent
            close = close / 10
        close.index = pd.to_datetime(close.index).tz_localize(None)
        rate = (close / 100.0).reindex(dates.union(close.index)).ffill().bfill().reindex(dates)
        return rate, f"10-yr Treasury (^TNX), avg {rate.mean() * 100:.2f}% over the period"
    except Exception as exc:
        print(f"  Could not download the 10-yr Treasury yield ({type(exc).__name__}: {exc}); "
              f"using fixed RISK_FREE_RATE = {RISK_FREE_RATE * 100:.2f}%.")
        return RISK_FREE_RATE, f"fixed {RISK_FREE_RATE * 100:.2f}% (10-yr Treasury download failed)"


def perf_stats(equity, risk_free):
    """`risk_free` is an annual rate: a float, or a pd.Series indexed by date."""
    rets = equity.pct_change().dropna()
    rf_annual = risk_free.reindex(rets.index) if isinstance(risk_free, pd.Series) else risk_free
    excess = rets - rf_annual / TRADING_DAYS_PER_YEAR
    vol    = rets.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
              if excess.std(ddof=1) > 0 else float("nan"))
    total  = equity.iloc[-1] / equity.iloc[0] - 1
    years  = len(rets) / TRADING_DAYS_PER_YEAR
    cagr   = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else float("nan")
    dd     = (equity / equity.cummax() - 1).min()
    return {"Total return": total, "Annualized return": cagr, "Annualized volatility": vol,
            "Sharpe ratio": sharpe, "Max drawdown": dd, "Ending value": equity.iloc[-1]}


def run_backtest(prices_df, show_plot=SHOW_PLOT, save_files=True):
    if not isinstance(Rebalance_Portfolio, (int, np.integer)) or isinstance(Rebalance_Portfolio, bool) \
            or Rebalance_Portfolio < 1:
        raise ValueError(f"Rebalance_Portfolio must be an integer >= 1 (got {Rebalance_Portfolio!r}).")

    print("=" * 78)
    print("  DMD DIRECTION BACKTEST")
    print(f"  {prices_df.shape[1]} stocks | trades every {Rebalance_Portfolio} day(s) | predicts {HORIZON_DAYS}d ahead "
          f"| windows {WINDOW_SIZES[0]}-{WINDOW_SIZES[-1]}")
    print(f"  selection lookback {SELECTION_LOOKBACK} | ${COMMISSION_PER_SHARE}/share")
    print("=" * 78)

    signals, chosen_w, chosen_rate, t_first = compute_signals(prices_df)
    T = len(prices_df)
    if T - t_first < 30:
        raise ValueError(f"Only {T - t_first} tradable days after warm-up ({t_first} days). "
                         f"Increase HISTORY_DAYS.")

    equity_net, cum_costs, trade_log, n_reb = simulate(prices_df, signals, t_first)
    equity_gross = equity_net + cum_costs
    bench, bench_cost = benchmark_equity(prices_df, t_first)

    risk_free, rf_desc = fetch_risk_free(equity_net.index)
    stats = {
        "Strategy (net of costs)":    perf_stats(equity_net, risk_free),
        "Strategy (before costs)":    perf_stats(equity_gross, risk_free),
        "Equal-weight buy & hold":    perf_stats(bench, risk_free),
    }

    # direction accuracy of the signals actually traded
    px = prices_df.to_numpy(dtype=float)
    ts = np.arange(t_first, T - HORIZON_DAYS)
    realized_up = px[ts + HORIZON_DAYS] > px[ts]
    acc = float((signals[ts] == realized_up).mean())
    up_frac = float(signals[t_first:].mean())

    total_cost   = float(cum_costs.iloc[-1])
    shares_traded = int(trade_log["shares"].sum()) if len(trade_log) else 0

    print(f"\n  Period: {equity_net.index[0].date()} -> {equity_net.index[-1].date()}  "
          f"({len(equity_net) - 1} trading days, {n_reb} rebalances)")
    print(f"  Starting value: ${STARTING_CASH:,.2f}")
    print(f"  Risk-free rate (Sharpe): {rf_desc}\n")

    rows = ["Ending value", "Total return", "Annualized return", "Annualized volatility",
            "Sharpe ratio", "Max drawdown"]
    print(f"  {'':<24}" + "".join(f"{name:>26}" for name in stats))
    for r in rows:
        line = f"  {r:<24}"
        for s in stats.values():
            v = s[r]
            if r == "Ending value":      line += f"{'$' + format(v, ',.2f'):>26}"
            elif r == "Sharpe ratio":    line += f"{v:>26.2f}"
            else:                        line += f"{v * 100:>25.2f}%"
        print(line)

    print("\n  TRADING COSTS  (${:.3f} per share, buys and sells)".format(COMMISSION_PER_SHARE))
    print(f"    Total commissions paid:      ${total_cost:,.2f}")
    print(f"    As % of starting capital:    {total_cost / STARTING_CASH * 100:.2f}%")
    print(f"    Return lost to costs:        {(stats['Strategy (before costs)']['Total return'] - stats['Strategy (net of costs)']['Total return']) * 100:.2f} percentage points")
    print(f"    Shares traded / orders:      {shares_traded:,} / {len(trade_log):,}")
    print(f"    Avg commission/rebalance:    ${total_cost / max(n_reb, 1):,.2f}")
    print(f"    (Benchmark one-time cost:    ${bench_cost:,.2f})")

    print("\n  SIGNAL QUALITY")
    print(f"    Direction accuracy ({HORIZON_DAYS}d, all stock-days traded): {acc * 100:.2f}%")
    print(f"    Share of stock-days predicted UP:                  {up_frac * 100:.1f}%")
    print("=" * 78)

    if save_files:
        trade_log.to_csv("dmd_backtest_trades.csv", index=False)
        pd.DataFrame({"net": equity_net, "gross": equity_gross, "cum_costs": cum_costs,
                      "benchmark": bench}).to_csv("dmd_backtest_equity.csv")
        print("  Saved dmd_backtest_trades.csv and dmd_backtest_equity.csv")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(equity_net.index, equity_net, label=f"Strategy (net)  Sharpe {stats['Strategy (net of costs)']['Sharpe ratio']:.2f}",
             color="#2563eb", linewidth=2)
    ax1.plot(equity_gross.index, equity_gross, label=f"Strategy (before costs)  Sharpe {stats['Strategy (before costs)']['Sharpe ratio']:.2f}",
             color="#16a34a", linewidth=1.5, linestyle="-.")
    ax1.plot(bench.index, bench, label=f"Equal-weight buy & hold  Sharpe {stats['Equal-weight buy & hold']['Sharpe ratio']:.2f}",
             color="#9ca3af", linewidth=2, linestyle="--")
    ax1.set_ylabel("Portfolio value ($)")
    ax1.set_title("DMD direction strategy: portfolio value")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(cum_costs.index, cum_costs, color="#dc2626", linewidth=2)
    ax2.set_ylabel("Cumulative\ncommissions ($)"); ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    fig.autofmt_xdate(); plt.tight_layout()
    if save_files:
        plt.savefig("dmd_backtest_results.png", dpi=150)
        print("  Saved dmd_backtest_results.png")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return {"equity_net": equity_net, "equity_gross": equity_gross, "benchmark": bench,
            "stats": stats, "trades": trade_log, "total_cost": total_cost, "accuracy": acc}


def main():
    prices = fetch_prices(TICKERS, HISTORY_DAYS)
    run_backtest(prices)


if __name__ == "__main__":
    main()
