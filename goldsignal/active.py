"""Trading modes: 一般設定, 高效積極模式 and 積極模式 (docs/RESEARCH.md sections 10-11).

一般設定 (normal) - one position at a time, 24-hour breakouts; gold also needs the
Bollinger-squeeze filter (布林收窄濾網), silver does not.

積極模式 (active, docs/RESEARCH.md section 12.2) - the 高效積極 rules for gold, but entered with a
STOP ORDER resting at the channel edge instead of waiting for a 15-minute close beyond it: while
the trend / ADX / extension / timing / squeeze rules hold at a bar's close and that close is still
inside the channel, a buy-stop (sell-stop) sits at the 18-hour high (low), stop-loss at that
close minus (plus) 2 x ATR(1h); it is filled the moment the next bar touches the level (at the
level, or at the open after a gap). Same trades per week as the first 積極模式 (12-hour channel, no
squeeze filter, 2005-2025 and 2013-2026) but a higher profit per trade and a third less
drawdown in every period on both data sets. Silver keeps its normal rules (stop entries did not
help silver). The first 積極模式's rules remain available to research scripts (active_params).

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

ACTIVE_BREAKOUT_HOURS = 12      # the first 積極模式 (research only)
MAX_OPEN = 2

MODES = ("normal", "efficient", "active")
MODE_LABEL = {"normal": "一般設定", "efficient": "高效積極模式", "active": "積極模式"}
# 高效積極模式 / 積極模式, per instrument; an instrument that is not listed keeps its normal rules
EFFICIENT = {"XAUUSD": {"breakout_hours": 18, "max_open": 2, "entry": "close"}}
ACTIVE = {"XAUUSD": {"breakout_hours": 18, "max_open": 2, "entry": "stop"}}


@dataclass(frozen=True)
class ModeRules:
    """How one instrument is traded in one mode."""
    mode: str
    params: StrategyParams
    max_open: int = 1               # positions that may be open at the same time
    cooldown_when_flat: bool = True  # wait cooldown_bars after an exit before a new first position
    same_dir: bool = True           # an extra position only in the direction of the open ones
    entry: str = "close"            # "close": 15m close beyond the channel; "stop": stop order at the channel edge

    @property
    def multi(self) -> bool:
        return self.max_open != 1

    @property
    def stop_entry(self) -> bool:
        return self.entry == "stop"

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
    cfg = {"efficient": EFFICIENT, "active": ACTIVE}.get(mode, {}).get(inst.key)
    if cfg:
        p = replace(inst.strategy_params(day_mode, squeeze=False), breakout_hours=int(cfg["breakout_hours"]),
                    squeeze_bars=SQUEEZE_BARS)
        return ModeRules(mode, p, int(cfg["max_open"]), cooldown_when_flat=True, same_dir=True, entry=cfg.get("entry", "close"))
    return ModeRules(mode, inst.strategy_params(day_mode, squeeze=squeeze), 1)


def stop_entry_candidates(m15: pd.DataFrame, f: pd.DataFrame, t: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Stop-order entries (積極模式). Returns (sig, fills): sig[i] = +1/-1 at the bar BEFORE the fill
    bar (so the simulator, which fills at the next bar, keeps its stop = close[i] -/+ 2 ATR), and
    fills[i+1] = the price paid at the fill bar: the channel edge as of that bar (= the high/low of
    the previous `breakout_hours`), or the open if the bar opened beyond it. A bar fills a long when
    its high reaches the level, the previous close was at or inside the level, and the trend / ADX /
    extension / timing rules and the squeeze filter held at the previous close."""
    n = len(m15)
    hi, lo = f["chan_hi"].to_numpy(float), f["chan_lo"].to_numpy(float)
    h, l, o, c = (m15[k].to_numpy(float) for k in ("high", "low", "open", "close"))
    c_prev = np.r_[np.nan, c[:-1]]
    setup = np.where(t["long_setup"].to_numpy(bool), 1, np.where(t["short_setup"].to_numpy(bool), -1, 0))
    setup_prev = np.r_[0, setup[:-1]]
    sq_prev = np.r_[False, t["squeeze_ok"].to_numpy(bool)[:-1]]
    timing = t["timing_ok"].to_numpy(bool)
    with np.errstate(invalid="ignore"):
        long_touch = (h >= hi) & (c_prev <= np.r_[np.nan, hi[:-1]]) & (setup_prev == 1) & sq_prev & timing
        short_touch = (l <= lo) & (c_prev >= np.r_[np.nan, lo[:-1]]) & (setup_prev == -1) & sq_prev & timing
    sig = np.zeros(n, dtype=np.int64)
    fills = np.full(n, np.nan)
    idx = np.flatnonzero(np.nan_to_num(long_touch) | np.nan_to_num(short_touch))
    idx = idx[(idx >= 1) & (idx < n)]
    d = np.where(np.nan_to_num(long_touch)[idx], 1, -1)
    sig[idx - 1] = d
    fills[idx] = np.where(d > 0, np.maximum(o[idx], hi[idx]), np.minimum(o[idx], lo[idx]))
    return sig, fills


def pending_order(m15: pd.DataFrame, f: pd.DataFrame, t: pd.DataFrame, p: StrategyParams) -> dict | None:
    """The stop order to rest for the NEXT bar, given the last closed bar: direction, level (the
    high/low of the last `breakout_hours`), stop-loss (last close -/+ stop_atr x ATR(1h)). None when
    the rules do not hold at the last close or the close is already beyond the level."""
    tl, fl, last = t.iloc[-1], f.iloc[-1], m15.iloc[-1]
    nb = int(p.breakout_hours * 4)
    if len(m15) < nb + 1 or not bool(tl["squeeze_ok"]):
        return None
    # "still inside": the close must not have broken the channel as of THIS bar (chan_hi/lo exclude
    # the bar itself) - the same test stop_entry_candidates applies to the previous close
    if bool(tl["long_setup"]):
        d, level = 1, float(m15["high"].iloc[-nb:].max())
        if not (float(last["close"]) <= float(fl["chan_hi"])):
            return None
    elif bool(tl["short_setup"]):
        d, level = -1, float(m15["low"].iloc[-nb:].min())
        if not (float(last["close"]) >= float(fl["chan_lo"])):
            return None
    else:
        return None
    stop = float(last["close"]) - d * p.stop_atr_h1 * float(fl["h1_atr"])
    return {"dir": d, "level": level, "stop": stop, "risk": abs(level - stop)}


def pending_reason(m15: pd.DataFrame, f: pd.DataFrame, t: pd.DataFrame, p: StrategyParams) -> str | None:
    """Why there is no stop order although the trend rules hold (None when they do not hold)."""
    tl, fl, last = t.iloc[-1], f.iloc[-1], m15.iloc[-1]
    if bool(tl["long_setup"]):
        if float(last["close"]) > float(fl["chan_hi"]):
            return "收盤已經在通道外"
    elif bool(tl["short_setup"]):
        if float(last["close"]) < float(fl["chan_lo"]):
            return "收盤已經在通道外"
    else:
        return None
    if not bool(tl["squeeze_ok"]):
        return "波動還沒收斂"
    return None


def run_multi(m15: pd.DataFrame, f: pd.DataFrame, p: StrategyParams, spread, slip, blackout=None,
              swap: bool = True, max_open: int = MAX_OPEN, clock: Clock | None = None,
              rules: pd.DataFrame | None = None, cooldown_when_flat: bool = False,
              same_dir: bool = False, sig: np.ndarray | None = None) -> pd.DataFrame:
    """Every signal opens its own position (managed exactly like a normal system trade) unless
    `max_open` positions are already open (0 = no limit). With `cooldown_when_flat`, a signal that
    would open the first position must wait p.cooldown_bars after the last exit (as in the normal
    rules); with `same_dir`, an extra position is only opened in the direction of the open ones.
    Returns one row per position, sorted by entry; the still-open ones have reason 'end' and
    exit_i == last bar (as in strategy.run)."""
    n = len(m15)
    if sig is None:
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
    if r.stop_entry:
        return run_stop(m15, f, r.params, spread, slip, blackout, swap=swap, max_open=r.max_open, clock=clock,
                        rules=rules, cooldown_when_flat=r.cooldown_when_flat, same_dir=r.same_dir)
    if not r.multi:
        tr, _, _ = S.run(m15, f, r.params, spread, slip, blackout, swap=swap)
        return tr
    return run_multi(m15, f, r.params, spread, slip, blackout, swap=swap, max_open=r.max_open, clock=clock,
                     rules=rules, cooldown_when_flat=r.cooldown_when_flat, same_dir=r.same_dir)


def run_stop(m15: pd.DataFrame, f: pd.DataFrame, p: StrategyParams, spread, slip, blackout=None, swap: bool = True,
             max_open: int = MAX_OPEN, clock: Clock | None = None, rules: pd.DataFrame | None = None,
             cooldown_when_flat: bool = True, same_dir: bool = True) -> pd.DataFrame:
    """積極模式: positions opened by stop orders at the channel edge (stop_entry_candidates), managed
    like every other position. The simulator takes the entry from the fill bar's open, so a copy
    of the bars with those opens replaced by the fill prices is simulated (only entry bars change)."""
    t = rules if rules is not None else rule_table(m15, f, p, blackout)
    sig, fills = stop_entry_candidates(m15, f, t)
    df = m15.copy()
    o = df["open"].to_numpy(float).copy()
    m = ~np.isnan(fills)
    o[m] = fills[m]
    df["open"] = o
    return run_multi(df, f, p, spread, slip, blackout, swap=swap, max_open=max_open, clock=clock, rules=t,
                     cooldown_when_flat=cooldown_when_flat, same_dir=same_dir, sig=sig)
