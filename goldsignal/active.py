"""積極模式 (active mode): the same trend-breakout rules, traded more often.

Research (docs/RESEARCH.md section 10, research/active_strategies.py and research/multi_market.py):
nothing that trades gold 5-10 times a week kept a positive expectancy after costs (grids,
channel fades, dip buying, opening-range and session trades all lost or broke even). What
does raise the number of trades while staying positive is to loosen the system itself:

* a shorter breakout channel (12 hours instead of 24),
* no Bollinger-squeeze filter,
* every new signal opens its own position, even if earlier ones are still open
  (each with its own stop, trailing stop and 16:45 New York check), up to MAX_OPEN at a time.

Positions are independent, so there is no cooldown after an exit. Gold and silver together give
roughly 1.2-2 trades a week this way (two positions at most per instrument), against ~0.6-0.9
for the normal settings, with roughly 1.5-3 times the drawdown (positions in the same direction
add up). docs/RESEARCH.md section 10 has the numbers.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .plans import Clock, simulate_plan
from .strategy import StrategyParams, rule_table, signals_from_rules

ACTIVE_BREAKOUT_HOURS = 12
MAX_OPEN = 2


def active_params(base: StrategyParams, breakout_hours: int = ACTIVE_BREAKOUT_HOURS) -> StrategyParams:
    """The system's rules with a shorter breakout channel and no squeeze filter."""
    return replace(base, breakout_hours=int(breakout_hours), squeeze_bars=0)


def run_multi(m15: pd.DataFrame, f: pd.DataFrame, p: StrategyParams, spread, slip, blackout=None,
              swap: bool = True, max_open: int = MAX_OPEN, clock: Clock | None = None,
              rules: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every signal opens its own position (managed exactly like a normal system trade) unless
    `max_open` positions are already open. Returns one row per position, sorted by entry; the
    still-open ones have reason 'end' and exit_i == last bar (as in strategy.run)."""
    n = len(m15)
    t = rules if rules is not None else rule_table(m15, f, p, blackout)
    sig = signals_from_rules(t)
    atr1 = f["h1_atr"].to_numpy(float)
    sp = np.full(n, float(spread)) if np.isscalar(spread) else np.asarray(spread, float)
    sl = np.full(n, float(slip)) if np.isscalar(slip) else np.asarray(slip, float)
    plan = p.plan(swap=swap)
    clock = clock or Clock.build(m15.index)
    rows = []
    exits: list[int] = []
    for i in np.flatnonzero(sig):
        if i >= n - 1:
            continue
        if max_open and sum(1 for x in exits if x >= i + 1) >= max_open:
            continue
        one = np.zeros(n, dtype=np.int64)
        one[i] = sig[i]
        tr = simulate_plan(m15, one, atr1, plan, sp, sl, clock)
        if tr.empty:
            continue
        rows.append(tr)
        exits.append(int(tr["exit_i"].iloc[0]))
    if not rows:
        return pd.DataFrame(columns=["sig_i", "exit_i", "dir", "entry", "exit", "R", "reason", "entry_time", "exit_time"])
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values("sig_i").reset_index(drop=True)
