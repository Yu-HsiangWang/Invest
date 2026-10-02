"""Gold trend-breakout signal engine ("黃金趨勢突破").

Evaluated on every completed 15-minute bar. A LONG signal needs ALL of:

  1. Daily trend up      : last completed daily close > EMA20(d) > EMA50(d)
  2. 4-hour trend up     : last completed 4h close  > EMA20(4h) > EMA50(4h)
  3. 1-hour trend up     : last completed 1h close  > EMA20(1h) > EMA50(1h)
  4. Not exhausted       : ADX14(1h) <= 30  and  (daily close - EMA20(d)) / ATR14(d) <= 1.5
  5. Breakout            : 15m close is the FIRST close above the highest high of the previous 24 hours
  6. Strong close        : the breakout bar closes in the top 30 % of its own range
  7. Timing              : Mon-Fri, not after 12:00 New York on Friday, no high-impact news blackout
                           (day mode: also no new entries from `day_cutoff_ny` until the 17:00 break)

SHORT is the mirror image. Trade management (identical in live mode and backtest):

  * entry      : next bar open (in practice: as soon as you see the signal)
  * stop       : entry -/+ 2.0 x ATR14(1h)
  * trailing   : best price since entry -/+ 5.0 x ATR14(1h), only ever tightened, updated every 15m close
  * no fixed target - trends are allowed to run; the trailing stop takes you out
  * cooldown   : 1 hour after an exit before a new signal may be taken
  * day mode   : at 16:45 New York (before the 17:00 rollover) the trade is closed unless it is
                 at least `eod_keep_r` R in profit; winners keep their trailing stop overnight
                 (see goldsignal/plans.py and docs/RESEARCH.md section 8)

Only one position at a time and direction is gated by the daily trend, so the
engine cannot flip long/short from one bar to the next.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import indicators as ind
from .backtest import ExitRules, simulate
from .bars import align_htf, resample
from .plans import Clock, Plan, simulate_plan

M15 = pd.Timedelta(minutes=15)
H1 = pd.Timedelta(hours=1)
H4 = pd.Timedelta(hours=4)
D1 = pd.Timedelta(days=1)


@dataclass
class StrategyParams:
    breakout_hours: int = 24
    stop_atr_h1: float = 2.0
    trail_atr_h1: float = 5.0
    adx_max: float = 30.0
    ext_max: float = 1.5
    close_pos_min: float = 0.7
    cooldown_bars: int = 4
    friday_cutoff_ny: float = 12.0
    use_h4: bool = True
    use_h1: bool = True
    # 當日平倉模式 (None = off): entry cutoff (NY hour) and the 16:45 NY keep-overnight threshold in R
    day_cutoff_ny: float | None = None
    eod_keep_r: float | None = None
    eod_time_ny: float = 16.75

    @property
    def day_mode(self) -> bool:
        return self.eod_keep_r is not None

    def plan(self, swap: bool = True) -> Plan:
        """The trade-management plan the live engine and the backtests share."""
        return Plan(name="day" if self.day_mode else "hold", eod="close_losers" if self.day_mode else "hold",
                    eod_k=float(self.eod_keep_r or 0.0), eod_time=self.eod_time_ny, trail_mult=self.trail_atr_h1,
                    stop_mult=self.stop_atr_h1, cooldown=self.cooldown_bars, swap=swap)

    def exit_rules(self) -> ExitRules:
        return ExitRules(tp_r=0.0, trail_mult=self.trail_atr_h1, cooldown=self.cooldown_bars)

    def to_dict(self) -> dict:
        return asdict(self)


def _trend(close, e20, e50) -> np.ndarray:
    up = (close > e20) & (e20 > e50)
    dn = (close < e20) & (e20 < e50)
    return np.where(up, 1, np.where(dn, -1, 0))


def compute_features(m15: pd.DataFrame, h1_hist: pd.DataFrame | None = None,
                     params: StrategyParams | None = None) -> pd.DataFrame:
    """All inputs the rules need, aligned to the 15m bars with no look-ahead.

    m15     : 15-minute OHLC bars (UTC index = bar open time).
    h1_hist : optional longer 1h history (for warming up daily/4h EMAs in live mode).
              Bars overlapping m15 are rebuilt from m15 so both modes see identical data.
    """
    p = params or StrategyParams()
    f = pd.DataFrame(index=m15.index)
    h1 = resample(m15, "1h")
    if h1_hist is not None and not h1_hist.empty:
        older = h1_hist[h1_hist.index < h1.index[0]]
        h1 = pd.concat([older[h1.columns.intersection(older.columns)], h1]).sort_index()
        h1 = h1[~h1.index.duplicated(keep="last")]
    h4 = resample(h1, "4h")
    d1 = resample(h1, "1D")

    hf = pd.DataFrame(index=h1.index)
    hf["close"] = h1["close"]
    hf["ema20"] = ind.ema(h1["close"], 20)
    hf["ema50"] = ind.ema(h1["close"], 50)
    hf["atr"] = ind.atr(h1, 14)
    hf["adx"] = ind.adx(h1, 14)["adx"]
    hf["rsi"] = ind.rsi(h1["close"], 14)
    a1 = align_htf(m15.index, M15, hf, H1)

    f4 = pd.DataFrame(index=h4.index)
    f4["close"] = h4["close"]
    f4["ema20"] = ind.ema(h4["close"], 20)
    f4["ema50"] = ind.ema(h4["close"], 50)
    f4["rsi"] = ind.rsi(h4["close"], 14)
    f4["adx"] = ind.adx(h4, 14)["adx"]
    f4["atr"] = ind.atr(h4, 14)
    a4 = align_htf(m15.index, M15, f4, H4)

    fd = pd.DataFrame(index=d1.index)
    fd["close"] = d1["close"]
    fd["ema20"] = ind.ema(d1["close"], 20)
    fd["ema50"] = ind.ema(d1["close"], 50)
    fd["atr"] = ind.atr(d1, 14)
    fd["rsi"] = ind.rsi(d1["close"], 14)
    fd["adx"] = ind.adx(d1, 14)["adx"]
    ad = align_htf(m15.index, M15, fd, D1)

    for prefix, src in (("h1", a1), ("h4", a4), ("d1", ad)):
        for col in src.columns:
            f[f"{prefix}_{col}"] = src[col]

    f["trend_d1"] = _trend(f["d1_close"], f["d1_ema20"], f["d1_ema50"])
    f["trend_h4"] = _trend(f["h4_close"], f["h4_ema20"], f["h4_ema50"])
    f["trend_h1"] = _trend(f["h1_close"], f["h1_ema20"], f["h1_ema50"])
    f["d1_ext"] = (f["d1_close"] - f["d1_ema20"]) / f["d1_atr"]

    n = int(p.breakout_hours * 4)
    f["chan_hi"] = m15["high"].rolling(n, min_periods=n).max().shift(1)
    f["chan_lo"] = m15["low"].rolling(n, min_periods=n).min().shift(1)
    rng = (m15["high"] - m15["low"]).replace(0.0, np.nan)
    f["cpos_long"] = (m15["close"] - m15["low"]) / rng
    f["cpos_short"] = (m15["high"] - m15["close"]) / rng
    f["m15_atr"] = ind.atr(m15, 14)
    f["m15_rsi"] = ind.rsi(m15["close"], 14)
    f["m15_ema20"] = ind.ema(m15["close"], 20)
    f["m15_ema50"] = ind.ema(m15["close"], 50)

    ny = m15.index.tz_convert("America/New_York")
    f["ny_close_h"] = ((ny.hour + ny.minute / 60.0 + 0.25) % 24).astype(float)
    f["ny_dow"] = ny.dayofweek
    return f


def rule_table(m15: pd.DataFrame, f: pd.DataFrame, params: StrategyParams | None = None,
               blackout: np.ndarray | None = None) -> pd.DataFrame:
    """Boolean table of every rule for both directions (used for signals AND for the UI checklist)."""
    p = params or StrategyParams()
    c = m15["close"]
    prev_c = c.shift(1)
    t = pd.DataFrame(index=m15.index)
    t["timing_ok"] = (f["ny_dow"] <= 4) & ~((f["ny_dow"] == 4) & (f["ny_close_h"] > p.friday_cutoff_ny))
    if p.day_cutoff_ny is not None:  # day mode: no new trades late in the NY afternoon
        t["timing_ok"] &= ~((f["ny_close_h"] >= p.day_cutoff_ny) & (f["ny_close_h"] < 17.0))
    t["news_ok"] = True if blackout is None else ~pd.Series(blackout, index=m15.index)
    t["adx_ok"] = ~(f["h1_adx"] > p.adx_max)
    for side, sgn in (("long", 1), ("short", -1)):
        t[f"{side}_d1"] = f["trend_d1"] == sgn
        t[f"{side}_h4"] = f["trend_h4"] == sgn
        t[f"{side}_h1"] = f["trend_h1"] == sgn
        t[f"{side}_ext"] = (sgn * f["d1_ext"]) <= p.ext_max
        if side == "long":
            t["long_break"] = (c > f["chan_hi"]) & (prev_c <= f["chan_hi"].shift(1))
            t["long_close"] = f["cpos_long"] >= p.close_pos_min
        else:
            t["short_break"] = (c < f["chan_lo"]) & (prev_c >= f["chan_lo"].shift(1))
            t["short_close"] = f["cpos_short"] >= p.close_pos_min
    common = t["timing_ok"] & t["news_ok"] & t["adx_ok"]
    lh4 = t["long_h4"] if p.use_h4 else True
    sh4 = t["short_h4"] if p.use_h4 else True
    lh1 = t["long_h1"] if p.use_h1 else True
    sh1 = t["short_h1"] if p.use_h1 else True
    t["long_setup"] = common & t["long_d1"] & lh4 & lh1 & t["long_ext"]
    t["short_setup"] = common & t["short_d1"] & sh4 & sh1 & t["short_ext"]
    t["long_signal"] = t["long_setup"] & t["long_break"] & t["long_close"]
    t["short_signal"] = t["short_setup"] & t["short_break"] & t["short_close"]
    return t.fillna(False)


def signals_from_rules(t: pd.DataFrame) -> np.ndarray:
    return np.where(t["long_signal"], 1, np.where(t["short_signal"], -1, 0)).astype(np.int64)


def run(m15: pd.DataFrame, f: pd.DataFrame | None = None, params: StrategyParams | None = None,
        spread: np.ndarray | float = 0.0, slip: np.ndarray | float = 0.0,
        blackout: np.ndarray | None = None, swap: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the engine over the bars. Returns (trades, features, rule_table).
    The last trade may be still open (reason == 'end' and exit_i == last bar).
    `swap=True` charges the broker's overnight financing; day mode and swaps use the
    plan simulator (identical results to the fast simulator for the plain strategy)."""
    p = params or StrategyParams()
    if f is None:
        f = compute_features(m15, params=p)
    t = rule_table(m15, f, p, blackout)
    sig = signals_from_rules(t)
    c = m15["close"].to_numpy(float)
    atr1 = f["h1_atr"].to_numpy(float)
    stop_l = c - p.stop_atr_h1 * atr1
    stop_s = c + p.stop_atr_h1 * atr1
    n = len(m15)
    sp = np.full(n, float(spread)) if np.isscalar(spread) else np.asarray(spread, float)
    sl = np.full(n, float(slip)) if np.isscalar(slip) else np.asarray(slip, float)
    if p.day_mode or swap:
        trades = simulate_plan(m15, sig, atr1, p.plan(swap=swap), sp, sl, Clock.build(m15.index))
        if trades.empty:
            trades = simulate(m15, sig, stop_l, stop_s, p.exit_rules(), sp, sl, atr=atr1)
    else:
        trades = simulate(m15, sig, stop_l, stop_s, p.exit_rules(), sp, sl, atr=atr1)
    return trades, f, t
