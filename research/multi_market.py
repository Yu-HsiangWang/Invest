"""Same trend-breakout rules on more markets (docs/RESEARCH.md section 10).

Gold alone gives the system about 0.3-0.5 trades a week, and nothing faster held up
(research/active_strategies.py). The classic way to trade more often WITHOUT lowering the
edge per trade is to run the same rules on more markets. This script runs an hourly version
of the system (identical rules, signals on 1h closes) on Dukascopy hourly data for a basket of
liquid CFD markets that Mitrade lists. The basket and the costs were fixed before running it;
no market gets its own parameters.

Hourly vs 15m: the live app evaluates the rules on 15m closes. The hourly version is the same
logic one step coarser; on gold it is checked against the 15m results (calibrate()).

In-browser (Pyodide) usage, after loading hourly bars into DataFrames:
    import multi_market as M
    res = M.run_basket({"EUR-USD": df_eurusd, ...})
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from goldsignal import indicators as ind  # noqa: E402
from goldsignal.bars import resample  # noqa: E402
from goldsignal.plans import Clock, Plan, simulate_plan  # noqa: E402

NY = "America/New_York"
H1, H4, D1 = pd.Timedelta(hours=1), pd.Timedelta(hours=4), pd.Timedelta(days=1)
PERIODS = {"2013/7-2016": ("2013-07-01", "2016-12-31"), "2017-2020": ("2017-01-01", "2020-12-31"),
           "2021-2025/3": ("2021-01-01", "2025-03-21"), "2025/3-2026/10": ("2025-03-22", "2026-10-02")}


@dataclass(frozen=True)
class Market:
    code: str            # Dukascopy code
    name: str
    group: str
    spread: float        # assumed broker spread, fraction of price (Mitrade-like, conservative)
    slip: float          # per market fill
    swap: float          # overnight financing per night, fraction of notional


BASKET = [
    Market("XAU-USD", "黃金", "金屬", 0.00012, 0.00005, 0.00016),
    Market("XAG-USD", "白銀", "金屬", 0.0006, 0.0001, 0.00016),
    Market("XPT.CMD-USD", "鉑金", "金屬", 0.001, 0.0001, 0.0002),
    Market("COPPER.CMD-USD", "銅", "原物料", 0.0006, 0.0001, 0.0002),
    Market("LIGHT.CMD-USD", "WTI 原油", "原物料", 0.0005, 0.0001, 0.0002),
    Market("BRENT.CMD-USD", "布蘭特原油", "原物料", 0.0005, 0.0001, 0.0002),
    Market("EUR-USD", "歐元/美元", "外匯", 0.00012, 0.00005, 0.0001),
    Market("GBP-USD", "英鎊/美元", "外匯", 0.00012, 0.00005, 0.0001),
    Market("USD-JPY", "美元/日圓", "外匯", 0.00012, 0.00005, 0.0001),
    Market("AUD-USD", "澳幣/美元", "外匯", 0.00015, 0.00005, 0.0001),
    Market("USD-CAD", "美元/加幣", "外匯", 0.00015, 0.00005, 0.0001),
    Market("USD-CHF", "美元/瑞郎", "外匯", 0.00015, 0.00005, 0.0001),
    Market("NZD-USD", "紐幣/美元", "外匯", 0.0002, 0.00005, 0.0001),
    Market("EUR-JPY", "歐元/日圓", "外匯", 0.00015, 0.00005, 0.0001),
    Market("GBP-JPY", "英鎊/日圓", "外匯", 0.0002, 0.00005, 0.0001),
    Market("USA500.IDX-USD", "美國標普 500", "指數", 0.00012, 0.00005, 0.0002),
    Market("USATECH.IDX-USD", "那斯達克 100", "指數", 0.00012, 0.00005, 0.0002),
    Market("USA30.IDX-USD", "道瓊 30", "指數", 0.0001, 0.00005, 0.0002),
    Market("DEU.IDX-EUR", "德國 DAX 40", "指數", 0.00012, 0.00005, 0.0002),
    Market("GBR.IDX-GBP", "英國富時 100", "指數", 0.00015, 0.00005, 0.0002),
    Market("JPN.IDX-JPY", "日經 225", "指數", 0.0002, 0.00005, 0.0002),
    Market("BTC-USD", "比特幣", "加密", 0.001, 0.0002, 0.0005),
    Market("ETH-USD", "以太幣", "加密", 0.0015, 0.0002, 0.0005),
]


def clock_h1(index: pd.DatetimeIndex) -> Clock:
    """plans.Clock for 1-hour bars (Clock.build assumes 15-minute bars)."""
    ny = index.tz_convert(NY)
    shifted = ny.tz_localize(None) + pd.Timedelta(hours=7)
    session = np.asarray((shifted.normalize() - pd.Timestamp("1970-01-01")) // pd.Timedelta(days=1), dtype=np.int64)
    close_h = np.asarray((ny.hour + ny.minute / 60.0 + 1.0) % 24, float)
    sess_h = (close_h - 17.0) % 24
    sess_h[sess_h == 0] = 24.0
    return Clock(session=session, sess_h=sess_h, sess_dow=np.asarray(shifted.dayofweek), ny_close_h=close_h)


def _aligned(base: pd.DatetimeIndex, htf: pd.DataFrame, htf_len: pd.Timedelta) -> pd.DataFrame:
    """Completed higher-timeframe rows as known at each 1h bar CLOSE."""
    k = htf.index.searchsorted(base + H1 - htf_len, side="right") - 1
    out = htf.iloc[np.maximum(k, 0)].copy()
    out[k < 0] = np.nan
    out.index = base
    return out


def _trend(close, e20, e50) -> np.ndarray:
    return np.where((close > e20) & (e20 > e50), 1, np.where((close < e20) & (e20 < e50), -1, 0))


def signals(h1: pd.DataFrame, squeeze: bool = False, breakout_bars: int = 24, adx_max: float = 30.0, ext_max: float = 1.5,
            close_pos: float = 0.7, day_cutoff: float = 12.0, friday_cutoff: float = 12.0) -> tuple[np.ndarray, np.ndarray]:
    """The system's rules on 1h bars. Returns (signal array, ATR14 of the 1h bars)."""
    c = h1["close"]
    h4 = resample(h1, "4h")
    d1 = resample(h1, "1D")
    f4 = pd.DataFrame({"c": h4["close"], "e20": ind.ema(h4["close"], 20), "e50": ind.ema(h4["close"], 50)})
    fd = pd.DataFrame({"c": d1["close"], "e20": ind.ema(d1["close"], 20), "e50": ind.ema(d1["close"], 50), "atr": ind.atr(d1, 14)})
    a4, ad = _aligned(h1.index, f4, H4), _aligned(h1.index, fd, D1)
    t4 = _trend(a4["c"], a4["e20"], a4["e50"])
    td = _trend(ad["c"], ad["e20"], ad["e50"])
    t1 = _trend(c, ind.ema(c, 20), ind.ema(c, 50))
    adx = ind.adx(h1, 14)["adx"].to_numpy()
    ext = ((ad["c"] - ad["e20"]) / ad["atr"]).to_numpy()
    hi = h1["high"].rolling(breakout_bars, min_periods=breakout_bars).max().shift(1)
    lo = h1["low"].rolling(breakout_bars, min_periods=breakout_bars).min().shift(1)
    pc = c.shift(1)
    rng = (h1["high"] - h1["low"]).replace(0.0, np.nan)
    up = ((c > hi) & (pc <= hi.shift(1)) & ((c - h1["low"]) / rng >= close_pos)).to_numpy()
    dn = ((c < lo) & (pc >= lo.shift(1)) & ((h1["high"] - c) / rng >= close_pos)).to_numpy()
    clk = clock_h1(h1.index)
    hcl, dow = clk.ny_close_h, clk.sess_dow
    timing = (dow <= 4) & ~((dow == 4) & (hcl > friday_cutoff)) & ~((hcl >= day_cutoff) & (hcl < 17.0))
    ok = timing & ~(adx > adx_max)
    if squeeze:
        bb = ind.bollinger(c, 20, 2.0)
        bwp = ((bb["upper"] - bb["lower"]) / bb["mid"]).rolling(200, min_periods=50).rank(pct=True)
        ok &= (bwp.shift(1).rolling(4, min_periods=1).min() <= 0.3).fillna(False).to_numpy()
    long_ = ok & up & (td == 1) & (t4 == 1) & (t1 == 1) & (ext <= ext_max)
    short = ok & dn & (td == -1) & (t4 == -1) & (t1 == -1) & (-ext <= ext_max)
    sig = np.where(long_, 1, np.where(short, -1, 0)).astype(np.int64)
    return sig, ind.atr(h1, 14).to_numpy()


def run_market(h1: pd.DataFrame, m: Market, squeeze: bool = False, start: str = "2013-07-01", keep_r: float = 0.25) -> pd.DataFrame:
    """Day-mode trades (16:00 NY check on 1h bars: close unless >= keep_r R; winners trail overnight)."""
    sig, atr = signals(h1, squeeze=squeeze)
    px = h1["open"].to_numpy(float)
    plan = Plan(name="day", eod="close_losers", eod_k=keep_r, eod_time=16.0, trail_mult=5.0, stop_mult=2.0, cooldown=1, swap=False)
    tr = simulate_plan(h1, sig, atr, plan, px * m.spread, px * m.slip, clock_h1(h1.index))
    if tr.empty:
        return tr
    tr = tr[(tr["reason"] != "end") & (tr["entry_time"] >= pd.Timestamp(start, tz="UTC"))].copy()
    tr["R"] = tr["R"] - tr["nights"] * m.swap * tr["entry"] / tr["risk"]
    tr["market"] = m.code
    return tr.reset_index(drop=True)


def stats(r: np.ndarray, weeks: float) -> dict:
    if len(r) == 0:
        return {"n": 0}
    eq = np.cumsum(r)
    dd = float(np.max(np.maximum.accumulate(np.r_[0, eq])[1:] - eq))
    sd = r.std(ddof=1) if len(r) > 1 else np.nan
    return {"n": int(len(r)), "per_week": round(len(r) / weeks, 2), "win%": round(100 * float((r > 0).mean()), 1),
            "avgR": round(float(r.mean()), 3), "t": round(float(r.mean() / (sd / np.sqrt(len(r)))), 2) if sd and sd > 0 else 0.0,
            "totR": round(float(r.sum()), 1), "maxDD": round(dd, 1)}


def by_period(tr: pd.DataFrame) -> dict:
    out = {}
    for name, (a, b) in PERIODS.items():
        m = (tr["entry_time"] >= pd.Timestamp(a, tz="UTC")) & (tr["entry_time"] < pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1))
        out[name] = stats(tr.loc[m, "R"].to_numpy(), (pd.Timestamp(b) - pd.Timestamp(a)).days / 7)
    a, b = PERIODS["2013/7-2016"][0], PERIODS["2025/3-2026/10"][1]
    out["all"] = stats(tr["R"].to_numpy(), (pd.Timestamp(b) - pd.Timestamp(a)).days / 7)
    return out


def max_open(tr: pd.DataFrame) -> int:
    if tr.empty:
        return 0
    t = np.r_[tr["entry_time"].to_numpy(), (tr["exit_time"] + H1).to_numpy()]
    d = np.r_[np.ones(len(tr)), -np.ones(len(tr))]
    o = np.lexsort((d, t))
    return int(np.max(np.cumsum(d[o])))


def run_basket(bars: dict, squeeze: bool = False) -> dict:
    """bars: {code: hourly DataFrame}. Returns per-market and portfolio statistics plus all trades."""
    per, trades = {}, []
    for m in BASKET:
        if m.code not in bars or len(bars[m.code]) < 2000:
            continue
        tr = run_market(bars[m.code], m, squeeze=squeeze)
        per[m.code] = {"name": m.name, "group": m.group, **{k: v for k, v in by_period(tr).items()}} if len(tr) else {"name": m.name, "n": 0}
        if len(tr):
            trades.append(tr)
    allt = pd.concat(trades, ignore_index=True).sort_values("exit_time") if trades else pd.DataFrame()
    port = by_period(allt) if len(allt) else {}
    return {"markets": per, "portfolio": port, "max_open": max_open(allt), "trades": allt}
