"""
DMD Direction Backtest -- 10 random-period trials, 2000-01-01 to 2026-09-01
===========================================================================
Backtest-only trading simulation built on the single-stock Hankel/time-delay
DMD direction predictor.

HOW A SIGNAL IS MADE (per stock, every day)
-------------------------------------------
For every window size n in WINDOW_SIZES (the heat-map columns) the model fits DMD
to the last n closes and predicts whether the price will be higher `h` trading
days later. Each window size is scored by its hit rate over the last
SELECTION_LOOKBACK predictions whose outcome was already known (no lookahead).
Then ONE window is chosen:

  FURTHEST_FROM_50 = True  (default)  pick the window whose hit rate is furthest
        from 50%. If that hit rate is BELOW 50% (e.g. "right 30% of the time"),
        the model does the OPPOSITE of that window's prediction.
  FURTHEST_FROM_50 = False            pick the window with the highest hit rate
        and follow its prediction (no flipping).

  MIN_EDGE_FROM_50 = x  (percentage points): if the chosen hit rate is not at
        least x points away from 50%, NO TRADE is placed for that stock that day
        (an existing position is left alone).

  Forward days `h`:  USE_REBALANCE_HORIZON = False -> h = HORIZON_DAYS (1)
                     USE_REBALANCE_HORIZON = True  -> h = Rebalance_Portfolio
                     (rebalancing every x days uses the x-day forward prediction)

TRADING
-------
Every Rebalance_Portfolio trading days: SELL stocks signalled down, leave
no-signal stocks untouched, and split the rest of the portfolio EQUALLY (whole
shares) across the stocks signalled up. Long-only. INVERT_SIGNALS = True swaps
buy and sell signals. Commission: COMMISSION_PER_SHARE on every share bought/sold.

TRIALS
------
NUM_TRIALS runs, each a TRIAL_YEARS-long window starting on a random date
(reproducible via TRIAL_SEED). Signals are computed using only data on or after
DATA_START, so no training ever starts before Jan 1, 2000. For every trial it
reports Strategy (net of costs), Strategy (before costs) and equal-weight
buy & hold, then the mean and standard deviation across the trials.

KNOWN LIMITS
------------
  * The stock list is today's large companies, so a 2000-2026 test has survivorship
    bias (it flatters all three methods). Stocks that listed after 2000 are
    simply not traded until they have enough history.
  * 'Close' prices are split-adjusted but exclude dividends; dividends are added
    separately (INCLUDE_DIVIDENDS) as cash on each ex-dividend date and are held as
    cash until the next rebalance (not reinvested immediately).
  * Signals use the close of day t and trades execute at that same close.
  * With 22 window sizes to choose from, the "furthest from 50%" hit rate is the
    maximum of 22 noisy numbers, so it looks like an edge even for random data.
"""

import os

# One math-library thread per process. The matrices here are tiny, so extra
# BLAS threads only add overhead (and memory) -- especially with many workers.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import time
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
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

# ── Strategy config ───────────────────────────────────────────────────────────
Rebalance_Portfolio   = 4       # trade every N trading days (integer >= 1):
                                # 1 = every day, 2 = every 2 days, ...
USE_REBALANCE_HORIZON = True   # True: predict Rebalance_Portfolio days ahead
                                # False: predict HORIZON_DAYS ahead (1 by default)
HORIZON_DAYS          = 4       # forward days used when USE_REBALANCE_HORIZON is False
FURTHEST_FROM_50      = False    # True (default): window furthest from 50%, flipped if <50%
                                # False: window with the highest hit rate, no flipping
MIN_EDGE_FROM_50      = 3    # percentage points: no trade unless the chosen hit rate is
                                # at least this far from 50 (10 -> need >=60% or <=40%).
                                # 0 = always trade
INVERT_SIGNALS        = True   # True = do the OPPOSITE of the model (buys become sells)
WINDOW_SIZES          = list(range(3, 25))  # sampling-window sizes tried (heat-map columns)
SELECTION_LOOKBACK    = 60      # past predictions used to score each window size default=60

# ── Data / trial config ───────────────────────────────────────────────────────
DATA_START       = "2000-01-01"  # no data, and so no training, before this date
DATA_END         = "2026-09-01"  # last day included
MIN_HISTORY_DAYS = 300           # stocks with less history than this are dropped
NUM_TRIALS       = 10
TRIAL_YEARS      = 3.0           # length of each trial window
TRIAL_SEED       = 42            # change for a different set of random start dates

# ── Trading / reporting config ────────────────────────────────────────────────
STARTING_CASH        = 1_000_000.0
COMMISSION_PER_SHARE = 0.005             # $ per share, charged on buys AND sells
INCLUDE_DIVIDENDS    = True              # credit cash dividends on the ex-dividend date for shares
                                         # held at the previous close (strategy AND buy & hold)
USE_10Y_TREASURY_FOR_RISK_FREE = True    # Sharpe risk-free rate = daily 10-yr Treasury yield
RISK_FREE_RATE       = 0.0               # annual fixed rate (e.g. 0.04 = 4%); used when the
                                         # flag above is False or the download fails
TRADING_DAYS_PER_YEAR = 252

# ── Runtime config ────────────────────────────────────────────────────────────
USE_PARALLEL = True                      # falls back to serial automatically if the
                                         # process pool breaks (e.g. inside Spyder)
MAX_WORKERS  = 18 #os.cpu_count() or 1
SHOW_PLOT    = True

METHODS = [("net",   "Strategy (net of costs)"),
           ("gross", "Strategy (before costs)"),
           ("bh",    "Equal-weight buy & hold")]


def forward_days():
    """Forward prediction horizon h (trading days)."""
    return int(Rebalance_Portfolio) if USE_REBALANCE_HORIZON else int(HORIZON_DAYS)


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

def fetch_prices(tickers, start, end):
    """
    Returns (close, dividends): daily closes (split-adjusted 'Close') from `start` to
    `end` inclusive, and a same-shaped table of cash dividends per share on each
    ex-dividend date (0 otherwise).
    Stocks that listed after `start` keep NaN before their first trading day;
    they are only dropped if they have fewer than MIN_HISTORY_DAYS of data.
    """
    import yfinance as yf

    # Yahoo writes share-class tickers with a dash (BRK.B -> BRK-B); results are
    # renamed back to the tickers as you wrote them.
    yahoo   = {t: t.replace(".", "-") for t in tickers}
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")   # Yahoo's end is exclusive
    print(f"Downloading {len(tickers)} tickers from {start} to {end} ...")
    raw = yf.download(list(yahoo.values()), start=start, end=end_excl,
                      progress=False, auto_adjust=False, actions=True)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
        divs  = raw["Dividends"] if "Dividends" in raw.columns.get_level_values(0) else None
    else:
        close = raw[["Close"]].rename(columns={"Close": yahoo[tickers[0]]})
        divs  = raw[["Dividends"]].rename(columns={"Dividends": yahoo[tickers[0]]}) if "Dividends" in raw.columns else None
    close = close.rename(columns={v: k for k, v in yahoo.items()})
    if divs is None:
        print("  WARNING: Yahoo returned no dividend data; dividends will be zero.")
        divs = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    divs = divs.rename(columns={v: k for k, v in yahoo.items()})

    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    close = close.loc[pd.Timestamp(start):pd.Timestamp(end)]       # never before DATA_START
    close = close.dropna(how="all")
    keep  = close.notna().sum() >= MIN_HISTORY_DAYS
    close = close.loc[:, keep]
    close = close.ffill()                                           # fill gaps; pre-listing NaN stays NaN
    divs.index = pd.DatetimeIndex(divs.index).tz_localize(None)
    divs  = divs.reindex(index=close.index, columns=close.columns).fillna(0.0).clip(lower=0.0)

    missing = [t for t in tickers if t not in close.columns]
    if missing:
        print(f"  Dropped (insufficient data): {missing}")
    late = [c for c in close.columns if close[c].first_valid_index() > close.index[0] + pd.Timedelta(days=30)]
    if late:
        print(f"  Listed after {close.index[0].date()} (traded only once they have history): {len(late)} stocks")
    cols = [t for t in tickers if t in close.columns]
    if INCLUDE_DIVIDENDS:
        n_pay = int((divs[cols].sum() > 0).sum())
        print(f"  Dividends: {n_pay} of {len(cols)} stocks paid at least one dividend in the data")
    return close[cols], divs[cols]


# ══════════════════════════════════════════════════════════════════════════════
# DMD DIRECTION PREDICTOR
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


def precompute_stock_signals(job):
    """
    Signals for ONE stock: +1 buy, -1 sell, 0 no trade, for every day flagged in
    `sig_days`. Module-level so it can be sent to worker processes.

    For each window size and each needed end-day e it fits DMD once. A window's
    score on day t is its hit rate over the SELECTION_LOOKBACK most recent
    predictions whose outcome was known by t (end-day e <= t - h). A window is
    only eligible if all of those predictions exist (the stock had enough history).
    """
    prices, windows, h, L, need_e, sig_days, furthest, min_edge = job
    T, n_w = len(prices), len(windows)
    sig = np.zeros(T, dtype=np.int8)

    ok_px = ~np.isnan(prices)
    if not ok_px.any() or T <= h:
        return sig
    first = int(np.argmax(ok_px))                    # first listed day

    e_all    = np.nonzero(need_e)[0]
    pred_up  = np.zeros((n_w, T), dtype=bool)
    cell_ok  = np.zeros((n_w, T), dtype=bool)        # segment fully inside the listed history
    for wi, n in enumerate(windows):
        for e in e_all[e_all >= first + n - 1]:
            pred_up[wi, e] = dmd_predicts_up(prices[e - n + 1:e + 1], h)
            cell_ok[wi, e] = True

    known = np.zeros(T, dtype=bool)
    known[:T - h] = True                              # outcome h days later exists
    realized_up = np.zeros(T, dtype=bool)
    with np.errstate(invalid="ignore"):
        realized_up[:T - h] = prices[h:] > prices[:T - h]
    val = cell_ok & known[None, :]
    hit = val & (pred_up == realized_up[None, :])

    zero   = np.zeros((n_w, 1))
    cs_val = np.concatenate([zero, np.cumsum(val, axis=1)], axis=1)
    cs_hit = np.concatenate([zero, np.cumsum(hit, axis=1)], axis=1)

    ts = np.nonzero(sig_days)[0]
    ts = ts[(ts - h + 1 - L) >= 0]
    if len(ts) == 0:
        return sig
    hi, lo = ts - h + 1, ts - h + 1 - L               # slides with end-day lo <= e < hi
    cnt    = cs_val[:, hi] - cs_val[:, lo]
    hits   = cs_hit[:, hi] - cs_hit[:, lo]
    full   = (cnt == L) & cell_ok[:, ts]
    acc    = np.where(full, hits / L, np.nan)

    # Compare whole hit counts (not floats) so exact ties are decided the same way every
    # time: the smallest window size wins.
    score  = np.where(full, np.abs(2 * hits - L) if furthest else hits, -1.0)
    best   = np.argmax(score, axis=0)
    cols   = np.arange(len(ts))
    acc_b  = acc[best, cols]
    up     = pred_up[best, ts]
    if furthest:
        up = np.where(acc_b < 0.5, ~up, up)           # right <50% of the time -> do the opposite

    with np.errstate(invalid="ignore"):
        tradable = full.any(axis=0) & (np.abs(acc_b - 0.5) >= min_edge - 1e-12)
    sig[ts] = np.where(tradable, np.where(up, 1, -1), 0)
    return sig


def compute_signals(prices_df, h, need_e, sig_days):
    """Signals (T x N, int8) for every stock; parallel across stocks with a serial fallback."""
    values = prices_df.to_numpy(dtype=float)
    N      = values.shape[1]
    jobs   = [(values[:, j].copy(), WINDOW_SIZES, h, SELECTION_LOOKBACK, need_e, sig_days,
               FURTHEST_FROM_50, MIN_EDGE_FROM_50 / 100.0) for j in range(N)]
    out = [None] * N
    t_start = time.perf_counter()

    if USE_PARALLEL and MAX_WORKERS > 1 and N > 1:
        try:
            print(f"Computing signals for {N} stocks on up to {MAX_WORKERS} processes ...")
            with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, N)) as ex:
                futs = {ex.submit(precompute_stock_signals, job): j for j, job in enumerate(jobs)}
                for k, f in enumerate(as_completed(futs), 1):
                    out[futs[f]] = f.result()
                    print(f"  {k}/{N} stocks done", end="\r")
            print()
        except (BrokenProcessPool, OSError) as exc:
            print(f"\n  Process pool failed ({type(exc).__name__}); falling back to serial. "
                  f"(Running from a terminal instead of Spyder/Jupyter usually fixes this.)")
            out = [None] * N

    if any(o is None for o in out):
        print(f"Computing signals for {N} stocks serially ...")
        for j, job in enumerate(jobs):
            out[j] = precompute_stock_signals(job)
            print(f"  {j + 1}/{N} stocks done", end="\r")
        print()

    print(f"  Signals ready in {time.perf_counter() - t_start:.1f}s")
    return np.column_stack(out)


# ══════════════════════════════════════════════════════════════════════════════
# TRIAL WINDOWS
# ══════════════════════════════════════════════════════════════════════════════

def pick_trial_windows(T, h):
    """NUM_TRIALS random (start_index, end_index) windows of TRIAL_YEARS each."""
    L      = SELECTION_LOOKBACK
    t_min  = (max(WINDOW_SIZES) - 1) + (L - 1) + h    # first day with a full training history
    length = int(round(TRIAL_YEARS * TRADING_DAYS_PER_YEAR))
    last_start = T - length
    if last_start - t_min + 1 < NUM_TRIALS:
        raise ValueError(f"Not enough data for {NUM_TRIALS} trial windows of {length} days "
                         f"(have {T} days, earliest start {t_min}). Lower TRIAL_YEARS or NUM_TRIALS.")
    rng    = np.random.default_rng(TRIAL_SEED)
    starts = np.sort(rng.choice(np.arange(t_min, last_start + 1), size=NUM_TRIALS, replace=False))
    return [(int(s), int(s + length - 1)) for s in starts]


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def simulate_window(px, dates, tickers, signals, t_start, t_end, div=None):
    """
    Walk forward from t_start to t_end. On each rebalance day: sell stocks signalled
    down, leave no-signal stocks alone, and split the rest of the portfolio equally
    (whole shares) across stocks signalled up. Every share traded pays the commission.
    Dividends (div[t, j] per share) go to cash on the ex-date for shares held at the
    previous close; the cash is redeployed at the next rebalance.
    Returns (equity, cum_costs, trades, n_reb, cum_div).
    """
    fee = COMMISSION_PER_SHARE
    pxv = np.nan_to_num(px, nan=0.0)
    N   = px.shape[1]

    cash, cum_cost, tot_div = STARTING_CASH, 0.0, 0.0
    shares = np.zeros(N, dtype=np.int64)
    trades = []
    equity, cum_costs, idx = [STARTING_CASH], [0.0], [dates[t_start - 1]]
    cum_div = [0.0]
    n_reb = 0
    dv = np.nan_to_num(div, nan=0.0) if (div is not None and INCLUDE_DIVIDENDS) else None

    for t in range(t_start, t_end + 1):
        if dv is not None:                                              # ex-date: paid on shares held at t-1 close
            d = float(shares @ dv[t])
            cash += d; tot_div += d
        if (t - t_start) % Rebalance_Portfolio == 0:
            n_reb += 1
            s, p = signals[t], pxv[t]

            for j in np.where((s == -1) & (shares > 0))[0]:            # exit predicted decliners
                qty = int(shares[j]); c = qty * fee
                cash += qty * p[j] - c; cum_cost += c; shares[j] = 0
                trades.append((dates[t], tickers[j], "SELL", qty, p[j], c))

            up = np.where(s == 1)[0]
            if len(up):
                hold    = (s == 0)                                      # no-signal stocks stay as they are
                frozen  = float(shares[hold] @ p[hold])
                alloc   = (cash + float(shares @ p) - frozen) / len(up)
                target  = np.floor(alloc / (p[up] + fee)).astype(np.int64)

                for k, j in enumerate(up):                              # trim over-weight winners
                    if shares[j] > target[k]:
                        qty = int(shares[j] - target[k]); c = qty * fee
                        cash += qty * p[j] - c; cum_cost += c; shares[j] -= qty
                        trades.append((dates[t], tickers[j], "SELL", qty, p[j], c))
                for k, j in enumerate(up):                              # top up / open positions
                    if shares[j] < target[k]:
                        qty = int(min(target[k] - shares[j], cash // (p[j] + fee)))
                        if qty > 0:
                            c = qty * fee
                            cash -= qty * p[j] + c; cum_cost += c; shares[j] += qty
                            trades.append((dates[t], tickers[j], "BUY", qty, p[j], c))

        equity.append(cash + float(shares @ pxv[t]))
        cum_costs.append(cum_cost)
        cum_div.append(tot_div)
        idx.append(dates[t])

    return (pd.Series(equity, index=idx), pd.Series(cum_costs, index=idx), trades, n_reb,
            pd.Series(cum_div, index=idx))


def benchmark_window(px, dates, t_start, t_end, div=None):
    """Equal-weight buy-and-hold over the stocks listed on t_start; commission paid once.
    Dividends on its (fractional) shares are kept as cash. Returns (equity, commission, dividends)."""
    pxv = np.nan_to_num(px, nan=0.0)
    ok  = ~np.isnan(px[t_start])
    sh  = np.zeros(px.shape[1])
    sh[ok] = (STARTING_CASH / ok.sum()) / (px[t_start, ok] + COMMISSION_PER_SHARE)   # fractional shares
    dv   = np.nan_to_num(div, nan=0.0) if (div is not None and INCLUDE_DIVIDENDS) else None
    vals, tot_div = [STARTING_CASH], 0.0
    for t in range(t_start, t_end + 1):
        if dv is not None:
            tot_div += float(sh @ dv[t])
        vals.append(float(sh @ pxv[t]) + tot_div)
    idx  = [dates[t_start - 1]] + list(dates[t_start:t_end + 1])
    return pd.Series(vals, index=idx), float(sh.sum() * COMMISSION_PER_SHARE), tot_div


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


def model_accuracy(model_sig, px, a, b, h):
    """How often the MODEL's (pre-inversion) up/down calls matched the realised move."""
    ts = np.arange(a, min(b, len(px) - 1 - h) + 1)
    s  = model_sig[ts]
    with np.errstate(invalid="ignore"):
        fut = px[ts + h] > px[ts]
    mask = s != 0
    if not mask.any():
        return float("nan"), 0
    return float(((s == 1) == fut)[mask].mean()), int(mask.sum())


def fmt_num(v):
    return f"{v:.2f}" if np.isfinite(v) else "n/a"


def fmt_pct(v):
    return f"{v * 100:7.2f}%" if np.isfinite(v) else "    n/a "


def run_trials(prices_df, dividends_df=None, show_plot=SHOW_PLOT, save_files=True):
    if not isinstance(Rebalance_Portfolio, (int, np.integer)) or isinstance(Rebalance_Portfolio, bool) \
            or Rebalance_Portfolio < 1:
        raise ValueError(f"Rebalance_Portfolio must be an integer >= 1 (got {Rebalance_Portfolio!r}).")

    h       = forward_days()
    dates   = prices_df.index
    px      = prices_df.to_numpy(dtype=float)
    div     = (dividends_df.reindex(index=prices_df.index, columns=prices_df.columns)
               .fillna(0.0).to_numpy(dtype=float)) if dividends_df is not None else None
    if INCLUDE_DIVIDENDS and div is None:
        print("  NOTE: no dividend table supplied; dividends are zero.")
    tickers = list(prices_df.columns)
    T, N    = px.shape
    L, nmax = SELECTION_LOOKBACK, max(WINDOW_SIZES)

    windows = pick_trial_windows(T, h)

    # Which days actually need signals / DMD fits (the union of the trial windows plus lookback).
    need_e   = np.zeros(T, dtype=bool)
    sig_days = np.zeros(T, dtype=bool)
    for a, b in windows:
        need_e[max(0, a - h - L + 1): b + 1] = True
        sig_days[a: b + 1] = True
    earliest_idx = min(a - h - L + 1 for a, _ in windows) - (nmax - 1)
    assert earliest_idx >= 0 and dates[0] >= pd.Timestamp(DATA_START), "training would start before DATA_START"

    print("=" * 96)
    print("  DMD DIRECTION BACKTEST -- RANDOM-PERIOD TRIALS")
    print(f"  {N} stocks | data {dates[0].date()} to {dates[-1].date()} | {NUM_TRIALS} trials x {TRIAL_YEARS:g} years (seed {TRIAL_SEED})")
    print(f"  trades every {Rebalance_Portfolio} day(s) | forward days h = {h}"
          f"{' (= Rebalance_Portfolio)' if USE_REBALANCE_HORIZON else ''} | ${COMMISSION_PER_SHARE}/share | start ${STARTING_CASH:,.0f}")
    print(f"  window rule: {'furthest from 50% (flip if <50%)' if FURTHEST_FROM_50 else 'highest hit rate (no flip)'}"
          f" | min edge {MIN_EDGE_FROM_50:g} pts | lookback {SELECTION_LOOKBACK} | windows {WINDOW_SIZES[0]}-{WINDOW_SIZES[-1]}")
    print(f"  dividends: {'ON (cash on ex-date, redeployed at next rebalance)' if INCLUDE_DIVIDENDS else 'OFF'}")
    if INVERT_SIGNALS:
        print("  *** INVERT_SIGNALS is ON: trading the opposite of the model's predictions ***")
    print(f"  earliest training date used by any trial: {dates[earliest_idx].date()} (never before {DATA_START})")
    n_pairs = NUM_TRIALS * (NUM_TRIALS - 1) // 2
    n_over  = sum(1 for i in range(len(windows)) for j in range(i + 1, len(windows))
                  if windows[j][0] <= windows[i][1])                     # windows are sorted by start
    if n_over:
        print(f"  note: {n_over} of {n_pairs} pairs of trials overlap in time, so the trials are not fully "
              f"independent and the std dev understates the true spread")
    print("=" * 96)

    model_sig = compute_signals(prices_df, h, need_e, sig_days)
    signals   = (-model_sig) if INVERT_SIGNALS else model_sig

    risk_free, rf_desc = fetch_risk_free(dates)
    print(f"  Risk-free rate (Sharpe): {rf_desc}\n")

    rows, all_trades = [], []
    for k, (a, b) in enumerate(windows, 1):
        eq_net, cum_costs, trades, n_reb, cum_div = simulate_window(px, dates, tickers, signals, a, b, div)
        eq_gross = eq_net + cum_costs
        bench, _, bh_div = benchmark_window(px, dates, a, b, div)
        acc, n_sig = model_accuracy(model_sig, px, a, b, h)

        row = {"trial": k, "start": dates[a].date(), "end": dates[b].date(),
               "commissions": float(cum_costs.iloc[-1]), "dividends": float(cum_div.iloc[-1]),
               "bh_dividends": float(bh_div), "orders": len(trades),
               "rebalances": n_reb, "model_accuracy": acc, "n_signals": n_sig}
        for key, eq in (("net", eq_net), ("gross", eq_gross), ("bh", bench)):
            for metric, val in perf_stats(eq, risk_free).items():
                row[f"{key}|{metric}"] = val
        rows.append(row)
        all_trades += [(k,) + tr for tr in trades]

    res = pd.DataFrame(rows)

    # ── Per-trial report ──────────────────────────────────────────────────────
    print("  EACH TRIAL")
    print(f"  {'':<6}{'':<24}{'Total return':^27}{'Sharpe ratio':^24}{'Commis-':>9}{'Strat':>10}{'B&H':>10}{'Model':>8}")
    print(f"  {'Trial':<6}{'Period':<24}{'Net':>9}{'Gross':>9}{'B&H':>9}{'Net':>8}{'Gross':>8}{'B&H':>8}{'sions $':>9}{'divs $':>10}{'divs $':>10}{'acc.':>8}")
    print("  " + "-" * 114)
    for r in rows:
        period = f"{r['start']} - {r['end']}"
        print(f"  {r['trial']:<6}{period:<24}"
              f"{fmt_pct(r['net|Total return'])}{fmt_pct(r['gross|Total return'])}{fmt_pct(r['bh|Total return'])}"
              f"{fmt_num(r['net|Sharpe ratio']):>8}{fmt_num(r['gross|Sharpe ratio']):>8}{fmt_num(r['bh|Sharpe ratio']):>8}"
              f"{r['commissions']:>9,.0f}{r['dividends']:>10,.0f}{r['bh_dividends']:>10,.0f}"
              f"{fmt_pct(r['model_accuracy']).strip():>8}")

    # ── Mean / standard deviation across trials ───────────────────────────────
    print(f"\n  MEAN AND STANDARD DEVIATION ACROSS {NUM_TRIALS} TRIALS  (sample std, n-1)")
    summary_rows = []
    for metric, kind in (("Total return", "pct"), ("Annualized return", "pct"),
                         ("Annualized volatility", "pct"), ("Sharpe ratio", "num"),
                         ("Max drawdown", "pct")):
        print(f"\n  {metric}")
        print(f"    {'Method':<28}{'Mean':>12}{'Std dev':>12}")
        for key, name in METHODS:
            v    = res[f"{key}|{metric}"].to_numpy(dtype=float)
            v    = v[np.isfinite(v)]                    # a run with no trades has no Sharpe ratio
            mean = float(v.mean()) if len(v) else float("nan")
            std  = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
            summary_rows.append({"metric": metric, "method": name, "mean": mean, "std": std,
                                 "trials_used": int(len(v))})
            scale, unit = (100.0, "%") if kind == "pct" else (1.0, "")
            m_txt = f"{mean * scale:.2f}{unit}" if np.isfinite(mean) else "n/a"
            s_txt = f"{std * scale:.2f}{unit}" if np.isfinite(std) else "n/a"
            print(f"    {name:<28}{m_txt:>12}{s_txt:>12}")
    if INCLUDE_DIVIDENDS:
        print("\n  DIVIDENDS COLLECTED (cash received, $)")
        print(f"    {'Method':<28}{'Mean/trial':>14}{'Std dev':>14}{'Sum of trials':>16}")
        for name, col in (("Strategy (net & before costs)", "dividends"), ("Equal-weight buy & hold", "bh_dividends")):
            v = res[col].to_numpy(dtype=float)
            sd = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
            print(f"    {name:<28}{v.mean():>14,.0f}{sd:>14,.0f}{v.sum():>16,.0f}")
            summary_rows.append({"metric": "Dividends collected ($)", "method": name, "mean": float(v.mean()),
                                 "std": sd, "trials_used": int(len(v))})
    print(f"\n  Average commissions per trial: ${res['commissions'].mean():,.0f}"
          f"  | average model direction accuracy: {res['model_accuracy'].mean(skipna=True) * 100:.2f}%")
    print("=" * 96)

    if save_files:
        res.to_csv("dmd_trials_results.csv", index=False)
        pd.DataFrame(summary_rows).to_csv("dmd_trials_summary.csv", index=False)
        pd.DataFrame(all_trades, columns=["trial", "date", "ticker", "action", "shares", "price", "commission"]
                     ).to_csv("dmd_trials_trades.csv", index=False)
        print("  Saved dmd_trials_results.csv, dmd_trials_summary.csv, dmd_trials_trades.csv")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    x, w = np.arange(len(res)), 0.27
    colors = {"net": "#2563eb", "gross": "#16a34a", "bh": "#9ca3af"}
    for i, (key, name) in enumerate(METHODS):
        ax1.bar(x + (i - 1) * w, res[f"{key}|Total return"] * 100, w, label=name, color=colors[key])
        ax2.bar(x + (i - 1) * w, res[f"{key}|Sharpe ratio"], w, label=name, color=colors[key])
    ax1.set_ylabel("Total return (%)"); ax2.set_ylabel("Sharpe ratio")
    ax1.set_title(f"{NUM_TRIALS} random {TRIAL_YEARS:g}-year trials")
    ax2.set_xticks(x); ax2.set_xticklabels([f"{r['trial']}\n{r['start']}" for r in rows], fontsize=8)
    ax2.set_xlabel("Trial / start date")
    for ax in (ax1, ax2):
        ax.axhline(0, color="black", linewidth=0.8); ax.grid(True, alpha=0.3, axis="y")
    ax1.legend()
    plt.tight_layout()
    if save_files:
        plt.savefig("dmd_trials_results.png", dpi=150)
        print("  Saved dmd_trials_results.png")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return {"results": res, "summary": pd.DataFrame(summary_rows), "windows": windows}


def main():
    prices, dividends = fetch_prices(TICKERS, DATA_START, DATA_END)
    run_trials(prices, dividends)


if __name__ == "__main__":
    main()
