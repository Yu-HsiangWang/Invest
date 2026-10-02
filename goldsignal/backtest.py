"""Event-driven trade simulator for signal research.

Design choices (all deliberately conservative):
* Signals are evaluated on bar CLOSE; the trade is entered at the NEXT bar's open
  (a human needs time to react).
* Prices are BID. Longs pay the spread on entry (buy at ask), shorts pay it on exit.
  Stops for shorts trigger on the ASK (= bid high + spread).
* If a bar touches both the stop and the target we assume the stop was hit first.
* A gap through the stop fills at the bar open (worse than the stop).
* Extra slippage `slip` (price units) is charged on every market fill.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:  # numba is optional; the pure-python path is ~50x slower but identical
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):
        def wrap(f):
            return f
        return wrap if not (args and callable(args[0])) else args[0]

# exit reason codes
EXIT_SL, EXIT_TP, EXIT_TRAIL, EXIT_TIME, EXIT_SESSION, EXIT_OPPOSITE, EXIT_END = 0, 1, 2, 3, 4, 5, 6
EXIT_NAMES = {0: "stop", 1: "target", 2: "trail", 3: "time", 4: "session", 5: "opposite", 6: "end"}


@njit(cache=True)
def _simulate(o, h, l, c, spread, sig, stop_long, stop_short, atr, force_exit,
              tp_r, be_r, be_lock_r, trail_mult, max_bars, cooldown, slip, opp_exit, min_hold):
    n = len(o)
    out = np.zeros((n // 2 + 1, 11))  # entry_i, exit_i, dir, entry, exit, risk, R, reason, mfeR, stop_last, init_stop
    k = 0
    i = 0
    next_ok = 0
    while i < n - 1:
        d = sig[i]
        if d == 0 or i < next_ok or force_exit[i]:
            i += 1
            continue
        j0 = i + 1
        if d > 0:
            entry = o[j0] + spread[j0] + slip[j0]
            stop = stop_long[i]
            risk = entry - stop
        else:
            entry = o[j0] - slip[j0]
            stop = stop_short[i]
            risk = stop - entry
        if not (risk > 0) or np.isnan(risk):
            i += 1
            continue
        init_stop = stop
        tp = entry + d * tp_r * risk if tp_r > 0 else np.nan
        best = entry
        exit_px = np.nan
        reason = EXIT_END
        j = j0
        be_done = False
        while j < n:
            sp = spread[j]
            if d > 0:
                if j > j0 and o[j] <= stop:  # gap through stop
                    exit_px = o[j] - slip[j]
                    reason = EXIT_SL if not be_done else EXIT_TRAIL
                    break
                if l[j] <= stop:
                    exit_px = stop - slip[j]
                    reason = EXIT_SL if not be_done else EXIT_TRAIL
                    break
                if tp_r > 0 and h[j] >= tp:
                    exit_px = tp
                    reason = EXIT_TP
                    break
                if h[j] > best:
                    best = h[j]
            else:
                if j > j0 and o[j] + sp >= stop:
                    exit_px = o[j] + sp + slip[j]
                    reason = EXIT_SL if not be_done else EXIT_TRAIL
                    break
                if h[j] + sp >= stop:
                    exit_px = stop + slip[j]
                    reason = EXIT_SL if not be_done else EXIT_TRAIL
                    break
                if tp_r > 0 and l[j] + sp <= tp:
                    exit_px = tp
                    reason = EXIT_TP
                    break
                if l[j] + sp < best:
                    best = l[j] + sp
            # ---- bar close management ----
            held = j - j0 + 1
            if force_exit[j]:
                exit_px = c[j] - slip[j] if d > 0 else c[j] + sp + slip[j]
                reason = EXIT_SESSION
                break
            if max_bars > 0 and held >= max_bars:
                exit_px = c[j] - slip[j] if d > 0 else c[j] + sp + slip[j]
                reason = EXIT_TIME
                break
            if opp_exit and held >= min_hold and sig[j] == -d:
                exit_px = c[j] - slip[j] if d > 0 else c[j] + sp + slip[j]
                reason = EXIT_OPPOSITE
                break
            cur_r = (c[j] - entry) / risk if d > 0 else (entry - (c[j] + sp)) / risk
            if be_r > 0 and (not be_done) and cur_r >= be_r:
                lock = entry + d * be_lock_r * risk
                if d > 0 and lock > stop:
                    stop = lock
                elif d < 0 and lock < stop:
                    stop = lock
                be_done = True
            if trail_mult > 0 and not np.isnan(atr[j]):
                if d > 0:
                    ts = best - trail_mult * atr[j]
                    if ts > stop:
                        stop = ts
                        be_done = True
                else:
                    ts = best + trail_mult * atr[j]
                    if ts < stop:
                        stop = ts
                        be_done = True
            j += 1
        if j >= n:
            j = n - 1
            exit_px = c[j] - slip[j] if d > 0 else c[j] + spread[j] + slip[j]
            reason = EXIT_END
        r_mult = (exit_px - entry) / risk if d > 0 else (entry - exit_px) / risk
        mfe = (best - entry) / risk if d > 0 else (entry - best) / risk
        out[k, 0] = i
        out[k, 1] = j
        out[k, 2] = d
        out[k, 3] = entry
        out[k, 4] = exit_px
        out[k, 5] = risk
        out[k, 6] = r_mult
        out[k, 7] = reason
        out[k, 8] = mfe
        out[k, 9] = stop
        out[k, 10] = init_stop
        k += 1
        next_ok = j + cooldown + 1
        i = j + 1 if cooldown == 0 else j + 1
        if k >= out.shape[0]:
            break
    return out[:k]


@dataclass
class ExitRules:
    tp_r: float = 2.0          # take profit in R (0 = none)
    be_r: float = 0.0          # move stop once trade is +be_r R at a bar close (0 = off)
    be_lock_r: float = 0.0     # where to move it: entry + be_lock_r * R
    trail_mult: float = 0.0    # chandelier trail in ATRs from best price (0 = off)
    max_bars: int = 0          # time stop in bars (0 = off)
    cooldown: int = 4          # bars to wait after an exit before a new entry
    opp_exit: bool = False     # exit on opposite signal
    min_hold: int = 4          # bars before an opposite signal may close the trade


def simulate(df: pd.DataFrame, sig: np.ndarray, stop_long: np.ndarray, stop_short: np.ndarray,
             rules: ExitRules, spread: np.ndarray, slip: float | np.ndarray = 0.0,
             atr: np.ndarray | None = None, force_exit: np.ndarray | None = None) -> pd.DataFrame:
    n = len(df)
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    if atr is None:
        atr = np.full(n, np.nan)
    if force_exit is None:
        force_exit = np.zeros(n, dtype=np.bool_)
    slip_v = np.asarray(slip, float) if isinstance(slip, np.ndarray) else np.full(n, float(slip))
    res = _simulate(o, h, l, c, np.asarray(spread, float), np.asarray(sig, np.int64),
                    np.asarray(stop_long, float), np.asarray(stop_short, float), np.asarray(atr, float),
                    np.asarray(force_exit, np.bool_), float(rules.tp_r), float(rules.be_r), float(rules.be_lock_r),
                    float(rules.trail_mult), int(rules.max_bars), int(rules.cooldown), slip_v,
                    bool(rules.opp_exit), int(rules.min_hold))
    cols = ["sig_i", "exit_i", "dir", "entry", "exit", "risk", "R", "reason", "mfeR", "stop_last", "init_stop"]
    t = pd.DataFrame(res, columns=cols)
    if t.empty:
        return t
    t["sig_i"] = t["sig_i"].astype(int)
    t["exit_i"] = t["exit_i"].astype(int)
    t["dir"] = t["dir"].astype(int)
    t["reason"] = t["reason"].astype(int).map(EXIT_NAMES)
    t["signal_time"] = df.index[t["sig_i"].to_numpy()]
    t["entry_time"] = df.index[np.minimum(t["sig_i"].to_numpy() + 1, n - 1)]
    t["exit_time"] = df.index[t["exit_i"].to_numpy()]
    t["bars"] = t["exit_i"] - t["sig_i"]
    return t


def metrics(trades: pd.DataFrame, years: float | None = None) -> dict:
    if trades is None or trades.empty:
        return {"n": 0}
    r = trades["R"].to_numpy()
    wins = r[r > 0]
    losses = r[r <= 0]
    eq = np.cumsum(r)
    dd = eq - np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]
    pf = wins.sum() / -losses.sum() if losses.sum() < 0 else np.inf
    out = {
        "n": int(len(r)),
        "win%": round(100 * len(wins) / len(r), 1),
        "avgR": round(float(r.mean()), 3),
        "totR": round(float(r.sum()), 1),
        "PF": round(float(pf), 2),
        "maxDD_R": round(float(-dd.min()), 1),
        "avg_bars": round(float(trades["bars"].mean()), 1),
    }
    if years:
        out["per_week"] = round(len(r) / (years * 52.0), 2)
    # longest losing streak
    streak = best = 0
    for x in r:
        streak = streak + 1 if x <= 0 else 0
        best = max(best, streak)
    out["max_losing_streak"] = int(best)
    return out


def by_year(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    g = trades.groupby(trades["entry_time"].dt.year)["R"]
    return pd.DataFrame({"n": g.size(), "win%": (g.apply(lambda x: (x > 0).mean() * 100)).round(1),
                         "totR": g.sum().round(1), "avgR": g.mean().round(3)})
