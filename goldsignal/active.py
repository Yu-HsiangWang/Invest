"""Trading modes: 一般設定, 高效積極模式 and 積極模式 (docs/RESEARCH.md sections 10-11).

一般設定 (normal) - one position at a time, 24-hour breakouts; gold also needs the
Bollinger-squeeze filter (布林收窄濾網), silver does not.

積極模式 (active, docs/RESEARCH.md section 10) - nothing that trades gold 5-10 times a week kept a
positive expectancy after costs (grids, channel fades, dip buying, opening-range and session
trades all lost or broke even), so this mode loosens the system itself: a 12-hour channel, no
squeeze filter, every new signal its own position (no cooldown), at most MAX_OPEN per
instrument. About twice the trades of the normal settings, but a much lower profit per trade
and roughly 2-3 times the drawdown; on 2013-2026 data the silver part barely made money.

高效積極模式 (efficient, docs/RESEARCH.md section 11) - more trades WITHOUT diluting the edge:
gold keeps the squeeze filter but uses an 18-hour channel and may hold a second position when a
new signal in the same direction appears; silver keeps its normal rules (faster silver
variants lost efficiency on every data set). Gold and silver together: about 1 trade a week
(normal ~0.75, active ~1.7), about the same yearly profit as the active mode with half its
drawdown, and about the same profit per trade as the normal settings.

Every position is managed exactly like a normal system trade (its own stop, trailing stop and
16:45 New York check); only the decision to open it differs between the modes.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .plans import Clock, simulate_plan
from .strategy import SQUEEZE_BARS, StrategyParams, rule_table, signals_from_rules

ACTIVE_BREAKOUT_HOURS = 12
MAX_OPEN = 2

MODES = ("normal", "efficient", "active")
MODE_LABEL = {"normal": "一般設定", "efficient": "高效積極模式", "active": "積極模式"}
# 高效積極模式, per instrument; an instrument that is not listed keeps its normal rules
EFFICIENT = {"XAUUSD": {"breakout_hours": 18, "max_open": 2}}


@dataclass(frozen=True)
class ModeRules:
    """How one instrument is traded in one mode."""
    mode: str
    params: StrategyParams
    max_open: int = 1               # positions that may be open at the same time
    cooldown_when_flat: bool = True  # wait cooldown_bars after an exit before a new first position
    same_dir: bool = True           # an extra position only in the direction of the open ones

    @property
    def multi(self) -> bool:
        return self.max_open != 1

    @property
    def label(self) -> str:
        return MODE_LABEL.get(self.mode, self.mode)


def norm_mode(mode) -> str:
    m = str(mode or "normal").strip().lower()
    return m if m in MODES else "normal"


def active_params(base: StrategyParams, breakout_hours: int = ACTIVE_BREAKOUT_HOURS) -> StrategyParams:
    """積極模式: the system's rules with a shorter breakout channel and no squeeze filter."""
    return replace(base, breakout_hours=int(breakout_hours), squeeze_bars=0)


def mode_rules(inst, mode: str, day_mode: bool = True, squeeze: bool | None = None) -> ModeRules:
    """The rules for instrument `inst` (goldsignal.instruments.Instrument) in `mode`. `squeeze`
    is the user's 布林收窄濾網 setting for this instrument, used by the normal rules."""
    mode = norm_mode(mode)
    if mode == "active":
        return ModeRules("active", active_params(inst.strategy_params(day_mode, squeeze=False)), MAX_OPEN,
                         cooldown_when_flat=False, same_dir=False)
    cfg = EFFICIENT.get(inst.key) if mode == "efficient" else None
    if cfg:
        p = replace(inst.strategy_params(day_mode, squeeze=False), breakout_hours=int(cfg["breakout_hours"]),
                    squeeze_bars=SQUEEZE_BARS)
        return ModeRules("efficient", p, int(cfg["max_open"]), cooldown_when_flat=True, same_dir=True)
    return ModeRules(mode, inst.strategy_params(day_mode, squeeze=squeeze), 1)


def run_multi(m15: pd.DataFrame, f: pd.DataFrame, p: StrategyParams, spread, slip, blackout=None,
              swap: bool = True, max_open: int = MAX_OPEN, clock: Clock | None = None,
              rules: pd.DataFrame | None = None, cooldown_when_flat: bool = False,
              same_dir: bool = False) -> pd.DataFrame:
    """Every signal opens its own position (managed exactly like a normal system trade) unless
    `max_open` positions are already open (0 = no limit). With `cooldown_when_flat`, a signal that
    would open the first position must wait p.cooldown_bars after the last exit (as in the normal
    rules); with `same_dir`, an extra position is only opened in the direction of the open ones.
    Returns one row per position, sorted by entry; the still-open ones have reason 'end' and
    exit_i == last bar (as in strategy.run)."""
    n = len(m15)
    t = rules if rules is not None else rule_table(m15, f, p, blackout)
    sig = signals_from_rules(t)
    atr1 = f["h1_atr"].to_numpy(float)
    sp = np.full(n, float(spread)) if np.isscalar(spread) else np.asarray(spread, float)
    sl = np.full(n, float(slip)) if np.isscalar(slip) else np.asarray(slip, float)
    plan = p.plan(swap=swap)
    clock = clock or Clock.build(m15.index)
    rows = []
    opened: list[tuple[int, int]] = []      # (exit_i, dir) of every position taken
    last_exit = -10 ** 9
    for i in np.flatnonzero(sig):
        if i >= n - 1:
            continue
        d = int(sig[i])
        live = [(x, dd) for x, dd in opened if x >= i + 1]
        if not live:
            if cooldown_when_flat and i < last_exit + plan.cooldown + 1:
                continue
        else:
            if max_open and len(live) >= max_open:
                continue
            if same_dir and any(dd != d for _, dd in live):
                continue
        one = np.zeros(n, dtype=np.int64)
        one[i] = d
        tr = simulate_plan(m15, one, atr1, plan, sp, sl, clock)
        if tr.empty:
            continue
        rows.append(tr)
        x = int(tr["exit_i"].iloc[0])
        opened.append((x, d))
        last_exit = max(last_exit, x)
    if not rows:
        return pd.DataFrame(columns=["sig_i", "exit_i", "dir", "entry", "exit", "R", "reason", "entry_time", "exit_time"])
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values("sig_i").reset_index(drop=True)


def run_mode(m15: pd.DataFrame, f: pd.DataFrame, r: ModeRules, spread, slip, blackout=None, swap: bool = True,
             rules: pd.DataFrame | None = None, clock: Clock | None = None) -> pd.DataFrame:
    """Back-test / live trades of one instrument in one mode (same output as strategy.run's trades)."""
    from . import strategy as S
    if not r.multi:
        tr, _, _ = S.run(m15, f, r.params, spread, slip, blackout, swap=swap)
        return tr
    return run_multi(m15, f, r.params, spread, slip, blackout, swap=swap, max_open=r.max_open, clock=clock,
                     rules=rules, cooldown_when_flat=r.cooldown_when_flat, same_dir=r.same_dir)
