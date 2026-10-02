"""Trade-management plans for a manual trader: same-day exits, scale-outs, breakeven.

This is a slower but more flexible simulator than ``backtest._simulate``. It uses the
exact same signals, entry and cost conventions, and adds:

* several equal-size *legs* per trade, each with its own take-profit in R
  (``0`` = runner, managed by the trailing stop only) - e.g. 3 x 0.01 lot;
* a breakeven move once price has traded at +x R (applied from the next bar,
  as a person would do after seeing the first take-profit fill);
* an end-of-day decision at a fixed New York time (default 16:45, just before the
  17:00 NY daily break when brokers charge the overnight fee):
      hold        - ignore the clock (the original strategy)
      close       - flatten everything every day
      keep_locked - keep only if the stop is already at/through the entry price
                    (so the position can no longer lose, barring a price gap)
      keep_if_r   - as keep_locked, and also keep when the trade is >= k R in profit
                    after moving the stop to entry + lock R; otherwise flatten
* overnight financing charged per rollover crossed (weekends count 3 nights).

R is always measured against the initial risk of the WHOLE position, so a trade's
result is the mean of its legs' R.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

NY = "America/New_York"

# Mitrade overnight financing (from its fee schedule / public reviews; both sides pay)
SWAP_LONG_PCT = 0.000168   # 0.0168 % of notional per night
SWAP_SHORT_PCT = 0.000140  # 0.0140 % of notional per night


@dataclass
class Plan:
    name: str = "base"
    legs: tuple = (0.0,)          # take-profit in R for each equal-size leg (0 = runner)
    be_trigger_r: float = 0.0     # once price has traded at +x R, stop -> entry + be_lock_r R (0 = off)
    be_lock_r: float = 0.0
    trail_mult: float = 5.0       # chandelier trail (ATR14 1h) from the best price
    stop_mult: float = 2.0        # initial stop (ATR14 1h)
    eod: str = "hold"             # hold | close | keep_locked | keep_if_r | close_losers | friday
    eod_k: float = 1.0            # keep_if_r threshold (R, at the decision bar close)
    eod_lock_r: float = 0.0       # where the stop goes when keep_if_r keeps a trade
    eod_tighten_r: float | None = None  # close_losers: kept trades get stop >= entry + x R (e.g. -0.5)
    eod_time: float = 16.75       # New York clock time of the decision (bar close)
    friday_flat: bool = False     # with eod != hold: never hold over the weekend
    entry_cutoff_h: float = 0.0   # skip signals within this many hours before the decision
    entry_hours: tuple | None = None  # only take signals whose bar closes within [a, b) NY hours
    cooldown: int = 4
    swap: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["legs"] = list(self.legs)
        return d


@dataclass
class Clock:
    """Per-bar calendar helpers (all derived from the bar OPEN time in UTC)."""
    session: np.ndarray       # trading-day number (days since epoch of the NY+7h date)
    sess_h: np.ndarray        # hours since the 17:00 NY session start at the bar CLOSE (0.25 .. 24)
    sess_dow: np.ndarray      # weekday of the trading day (0=Mon .. 4=Fri)
    ny_close_h: np.ndarray    # NY clock hour at bar close
    eod: dict = field(default_factory=dict)

    @classmethod
    def build(cls, index: pd.DatetimeIndex) -> "Clock":
        ny = index.tz_convert(NY)
        shifted = (ny.tz_localize(None) + pd.Timedelta(hours=7))
        days = (shifted.normalize() - pd.Timestamp("1970-01-01")) // pd.Timedelta(days=1)
        session = np.asarray(days, dtype=np.int64)
        open_h = ny.hour + ny.minute / 60.0
        close_h = (open_h + 0.25) % 24
        sess_h = (np.asarray(close_h) - 17.0) % 24
        sess_h[sess_h == 0] = 24.0
        return cls(session=session, sess_h=sess_h, sess_dow=np.asarray(shifted.dayofweek),
                   ny_close_h=np.asarray(close_h, float))

    def eod_mask(self, eod_time: float) -> np.ndarray:
        """True on the bar whose close is the first at/after `eod_time` NY in its trading day
        (or the last bar of the day when the market closes early)."""
        key = round(float(eod_time), 4)
        if key in self.eod:
            return self.eod[key]
        target = (eod_time - 17.0) % 24 or 24.0
        n = len(self.session)
        mask = np.zeros(n, dtype=bool)
        # boundaries of each trading day
        starts = np.flatnonzero(np.r_[True, self.session[1:] != self.session[:-1]])
        ends = np.r_[starts[1:], n]
        for a, b in zip(starts, ends):
            seg = self.sess_h[a:b]
            hit = np.flatnonzero(seg >= target - 1e-9)
            if hit.size:
                mask[a + hit[0]] = True
            elif b < n:  # the day ended early (holiday) - never guess for the still-running last day
                mask[b - 1] = True
        self.eod[key] = mask
        return mask


def simulate_plan(df: pd.DataFrame, sig: np.ndarray, atr: np.ndarray, plan: Plan,
                  spread: np.ndarray, slip: np.ndarray, clock: Clock | None = None,
                  exit_sig: np.ndarray | None = None, target_px: np.ndarray | None = None) -> pd.DataFrame:
    """Simulate `plan` on the signal array. Returns one row per trade with per-leg results.

    Research hooks (both optional, used by research/tool_backtest.py):
    exit_sig   per bar: -1 = close longs at this bar's close, +1 = close shorts (a chart-tool exit)
    target_px  per signal bar: a price target for the whole position (NaN = none)"""
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(df)
    clock = clock or Clock.build(df.index)
    is_eod = clock.eod_mask(plan.eod_time) if plan.eod != "hold" else np.zeros(n, dtype=bool)
    target = (plan.eod_time - 17.0) % 24 or 24.0
    legs = np.asarray(plan.legs, float)
    m = len(legs)
    rows = []
    next_ok = 0
    for i in np.flatnonzero(sig):
        if i < next_ok or i >= n - 1:
            continue
        if plan.entry_cutoff_h > 0 and plan.eod != "hold":
            if target - plan.entry_cutoff_h < clock.sess_h[i] <= target:
                continue
        if plan.entry_hours is not None:
            a, b = plan.entry_hours
            x = clock.ny_close_h[i]
            if not ((a <= x < b) if a <= b else (x >= a or x < b)):
                continue
        d = int(sig[i])
        j0 = i + 1
        if d > 0:
            entry = o[j0] + spread[j0] + slip[j0]
            stop = c[i] - plan.stop_mult * atr[i]
            risk = entry - stop
        else:
            entry = o[j0] - slip[j0]
            stop = c[i] + plan.stop_mult * atr[i]
            risk = stop - entry
        if not (risk > 0):
            continue
        init_stop = stop
        tp_px = np.where(legs > 0, entry + d * legs * risk, np.nan)
        if target_px is not None and np.isfinite(target_px[i]) and d * (target_px[i] - entry) > 0:
            tp_px = np.full(m, float(target_px[i]))
        leg_open = np.ones(m, dtype=bool)
        leg_exit = np.full(m, np.nan)
        leg_j = np.full(m, -1, dtype=np.int64)
        leg_reason = [""] * m
        best = entry
        mae = 0.0
        be_done = False
        reason = "end"
        j = j0
        while j < n:
            sp = spread[j]
            px = np.nan
            if d > 0:
                if j > j0 and o[j] <= stop:
                    px = o[j] - slip[j]
                elif lo[j] <= stop:
                    px = stop - slip[j]
            else:
                if j > j0 and o[j] + sp >= stop:
                    px = o[j] + sp + slip[j]
                elif h[j] + sp >= stop:
                    px = stop + slip[j]
            if not np.isnan(px):
                locked = (d > 0 and stop >= entry) or (d < 0 and stop <= entry)
                reason = "trail" if (locked or stop != init_stop) else "stop"
                for k in np.flatnonzero(leg_open):
                    leg_exit[k], leg_j[k], leg_reason[k] = px, j, reason
                leg_open[:] = False
                break
            for k in np.flatnonzero(leg_open & np.isfinite(tp_px)):
                if (d > 0 and h[j] >= tp_px[k]) or (d < 0 and lo[j] + sp <= tp_px[k]):
                    leg_exit[k], leg_j[k], leg_reason[k] = tp_px[k], j, "target"
                    leg_open[k] = False
            if not leg_open.any():
                reason = "target"
                break
            if d > 0:
                best = max(best, h[j])
                mae = max(mae, (entry - lo[j]) / risk)
            else:
                best = min(best, lo[j] + sp)
                mae = max(mae, (h[j] + sp - entry) / risk)
            if plan.be_trigger_r > 0 and not be_done:
                if (d > 0 and h[j] >= entry + plan.be_trigger_r * risk) or \
                   (d < 0 and lo[j] + sp <= entry - plan.be_trigger_r * risk):
                    ns = entry + d * plan.be_lock_r * risk
                    if (d > 0 and ns > stop) or (d < 0 and ns < stop):
                        stop = ns
                    be_done = True
            if plan.trail_mult > 0 and not np.isnan(atr[j]):
                ts = best - d * plan.trail_mult * atr[j]
                if (d > 0 and ts > stop) or (d < 0 and ts < stop):
                    stop = ts
            if exit_sig is not None and exit_sig[j] == -d:
                px = c[j] - slip[j] if d > 0 else c[j] + sp + slip[j]
                for k in np.flatnonzero(leg_open):
                    leg_exit[k], leg_j[k], leg_reason[k] = px, j, "tool"
                leg_open[:] = False
                reason = "tool"
                break
            if is_eod[j]:
                locked = (d > 0 and stop >= entry) or (d < 0 and stop <= entry)
                friday = clock.sess_dow[j] == 4
                flat = plan.eod == "close" or (friday and (plan.friday_flat or plan.eod == "friday"))
                if not flat and not locked:
                    cur = (c[j] - entry) / risk if d > 0 else (entry - (c[j] + sp)) / risk
                    if plan.eod == "keep_locked":
                        flat = True
                    elif plan.eod == "close_losers":
                        flat = cur < plan.eod_k
                        if not flat and plan.eod_tighten_r is not None:
                            ns = entry + d * plan.eod_tighten_r * risk
                            if (d > 0 and ns > stop) or (d < 0 and ns < stop):
                                stop = ns
                    elif plan.eod == "keep_if_r":
                        if cur >= plan.eod_k and cur > plan.eod_lock_r:
                            stop = entry + d * plan.eod_lock_r * risk
                        else:
                            flat = True
                if flat:
                    px = c[j] - slip[j] if d > 0 else c[j] + sp + slip[j]
                    for k in np.flatnonzero(leg_open):
                        leg_exit[k], leg_j[k], leg_reason[k] = px, j, "eod"
                    leg_open[:] = False
                    reason = "eod"
                    break
            j += 1
        if leg_open.any():  # ran out of data
            jj = n - 1
            px = c[jj] - slip[jj] if d > 0 else c[jj] + spread[jj] + slip[jj]
            for k in np.flatnonzero(leg_open):
                leg_exit[k], leg_j[k], leg_reason[k] = px, jj, "end"
            reason = "end"
        leg_r = d * (leg_exit - entry) / risk
        nights = clock.session[leg_j] - clock.session[j0]
        if plan.swap:
            rate = SWAP_LONG_PCT if d > 0 else SWAP_SHORT_PCT
            leg_r = leg_r - nights * rate * entry / risk
        jx = int(leg_j.max())
        rows.append({
            "sig_i": int(i), "exit_i": jx, "dir": d, "entry": entry, "exit": float(np.mean(leg_exit)),
            "init_stop": init_stop, "stop_last": stop, "risk": risk,
            "R": float(leg_r.mean()), "legR": leg_r.tolist(), "leg_reason": leg_reason, "reason": reason,
            "mfeR": d * (best - entry) / risk, "mae": mae, "nights": int(nights.max()), "risk_pct": risk / entry,
        })
        next_ok = jx + plan.cooldown + 1
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["signal_time"] = df.index[t["sig_i"].to_numpy()]
    t["entry_time"] = df.index[np.minimum(t["sig_i"].to_numpy() + 1, n - 1)]
    t["exit_time"] = df.index[t["exit_i"].to_numpy()]
    t["bars"] = t["exit_i"] - t["sig_i"]
    return t


# ----------------------------------------------------------------------------------------
# Position sizing helpers (shared by the research script and the live dashboard)
# ----------------------------------------------------------------------------------------

def leg_targets(units: int) -> list[float]:
    """Scale-out plan for a position of `units` minimum lots (R multiples, 0 = runner).
    Research (docs/RESEARCH.md section 8): taking profit early (+1R) or moving the stop to
    breakeven costs 40-60 % of the edge, so small positions exit all together. From 3 units
    on, one third may be banked at +2R (the cheapest way to lock in something)."""
    if units < 3:
        return [0.0] * max(units, 1)
    k = units // 3
    return [2.0] * k + [0.0] * (units - k)


def size_position(balance: float, risk_pct: float, stop_dist: float, price: float,
                  oz_per_lot: float, min_lot: float = 0.01, max_lots: float = 0.3,
                  leverage: float = 100.0, spread: float = 0.0, stop_out_level: float = 0.5) -> dict:
    """Lots for one signal.

    units = floor(balance x risk% / loss per minimum lot), in `min_lot` steps. If even one
    minimum lot is over budget but at most twice the budget, one minimum lot is suggested with
    a warning; above that the trade is too big for the account (skip). Capped by `max_lots`
    and by the margin the broker requires (price x ounces / leverage)."""
    out = {"lots": 0.0, "units": 0, "skip": True}
    if not (balance > 0 and stop_dist > 0 and price > 0 and oz_per_lot > 0 and min_lot > 0):
        return out
    oz_unit = oz_per_lot * min_lot
    loss_unit = (stop_dist + spread) * oz_unit                 # $ lost per minimum lot at the stop
    margin_unit = price * oz_unit / leverage
    budget = balance * risk_pct / 100.0
    units = int(np.floor(budget / loss_unit + 1e-9))
    over = False
    if units == 0 and loss_unit <= 2.0 * budget:
        units, over = 1, True
    units = min(units, int(np.floor(max_lots / min_lot + 1e-9)), int(np.floor(balance / margin_unit + 1e-9)))
    out.update({
        "oz_per_unit": oz_unit, "loss_per_unit": round(loss_unit, 2), "margin_per_unit": round(margin_unit, 2),
        "min_lot_risk_pct": round(100.0 * loss_unit / balance, 2), "budget_usd": round(budget, 2),
        "over_budget": over, "skip": units <= 0,
    })
    if units <= 0:
        return out
    oz = units * oz_unit
    margin = units * margin_unit
    out.update({
        "lots": round(units * min_lot, 2), "units": units,
        "risk_usd": round(units * loss_unit, 2), "risk_pct_actual": round(100.0 * units * loss_unit / balance, 2),
        "margin": round(margin, 2), "margin_pct": round(100.0 * margin / balance, 1),
        "liq_move": round(max(0.0, (balance - stop_out_level * margin) / oz), 4),
        "legs": leg_targets(units),
    })
    return out


def lots_table(balance: float, stop_dist: float, price: float, oz_per_lot: float, spread: float = 0.0,
               leverage: float = 100.0, lots=(0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3),
               stop_out_level: float = 0.5) -> list[dict]:
    """What a stop-out and a margin call would cost for a few position sizes."""
    rows = []
    for lt in lots:
        oz = lt * oz_per_lot
        loss = (stop_dist + spread) * oz
        margin = price * oz / leverage
        liq = (balance - stop_out_level * margin) / oz if oz > 0 else float("inf")
        rows.append({"lots": lt, "loss_usd": round(loss, 2), "loss_pct": round(100.0 * loss / balance, 1) if balance > 0 else None,
                     "margin": round(margin, 2), "margin_pct": round(100.0 * margin / balance, 1) if balance > 0 else None,
                     "liq_move": round(max(liq, 0.0), 4), "can_open": margin <= balance,
                     "per_dollar": round(oz, 4)})
    return rows


def eod_decision(direction: int, entry: float, risk: float, mark: float, keep_r: float) -> dict:
    """The 16:45 New York check of day mode for a position (prices in the same units)."""
    if not (risk and risk > 0) or mark is None:
        return {}
    r_now = direction * (mark - entry) / risk
    keep = r_now >= keep_r
    return {"R_now": round(r_now, 2), "keep_r": keep_r, "keep": bool(keep),
            "target_price": entry + direction * keep_r * risk}
