"""Open-source / published trading algorithms applied to gold 15m (docs/RESEARCH.md section 12).

The user asked whether an open-source quant algorithm could raise BOTH the return and the number
of trades. Every candidate below is a published, widely used algorithm (TradingView scripts,
freqtrade / QuantConnect examples, textbook systems, de Prado's meta-labeling). Each one is run
through exactly the same harness as the live system:

  * signals decided at the close of a 15m bar, filled at the next open;
  * the system's exits (initial stop 2 x ATR(1h), chandelier trail 5 x ATR(1h), day mode with the
    16:45 New York check, Friday cutoff), costs (spread 0.012 %, slippage 0.005 % per fill,
    overnight fees) - so only the ENTRY decision differs;
  * periods: dev 2005-2016 / val 2017-2020 / test 2021-2025/3, and the same efficiency
    yardstick as section 11 (R per year / max drawdown in R);
  * three ways of using each trigger: raw (only the timing rules), "filtered" (plus the system's
    trend / ADX / extension filters), and as the SECOND position next to 高效積極模式's own
    signals (max 2 positions, same direction) - the direct "does it add good trades" test.

Decision rule, fixed before running: a candidate is worth adopting only if it beats 高效積極模式's
efficiency in all three periods while trading more, or adds trades to it without lowering the
efficiency in any period.

    python research/open_source_algos.py classic     # rule-based triggers and regime filters
    python research/open_source_algos.py knn         # Lorentzian classification (jdehorty)
    python research/open_source_algos.py ml          # gradient boosting direction model
    python research/open_source_algos.py meta        # triple-barrier style meta-labeling
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

import efficient_active as E  # noqa: E402
from goldsignal import indicators as ind  # noqa: E402
from goldsignal import strategy as S  # noqa: E402
from goldsignal.active import mode_rules  # noqa: E402
from goldsignal.bars import align_htf, resample  # noqa: E402
from goldsignal.instruments import INSTRUMENTS  # noqa: E402
from goldsignal.plans import Clock, simulate_plan  # noqa: E402

OUT = ROOT / "results"
PERIODS = E.PERIODS
H15 = pd.Timedelta(minutes=15)
H60 = pd.Timedelta(hours=1)


# ----------------------------------------------------------------------------- harness

class Harness:
    """Data, features, costs and the three baselines; `run(sig, ...)` back-tests any signal array."""

    def __init__(self, df: pd.DataFrame, sym: str = "XAUUSD"):
        self.df = df
        self.inst = inst = INSTRUMENTS[sym]
        self.n = len(df)
        px = df["open"].to_numpy(float)
        self.sp, self.sl = px * inst.spread_pct, px * inst.slip_pct
        self.clock = Clock.build(df.index)
        self.rules = {m: mode_rules(inst, m, True, inst.squeeze_default) for m in ("normal", "efficient", "active")}
        self.p = self.rules["efficient"].params                      # 18h channel, squeeze
        self.f = S.compute_features(df, params=self.p)
        self.t = S.rule_table(df, self.f, self.p, None)
        self.atr1 = self.f["h1_atr"].to_numpy(float)
        self.plan = self.p.plan(swap=True)
        self.h1 = resample(df, "1h")
        self.h1 = self.h1[self.h1.index >= df.index[0]]
        self.c = df["close"].to_numpy(float)
        self.h = df["high"].to_numpy(float)
        self.l = df["low"].to_numpy(float)
        self.o = df["open"].to_numpy(float)
        self.v = df["volume"].fillna(0).to_numpy(float)
        self._base = {}

    # --- system signals for any channel length / squeeze setting (cached)
    def sys_sig(self, hours: int, squeeze: bool) -> np.ndarray:
        key = (hours, squeeze)
        if key not in self._base:
            p = replace(self.p, breakout_hours=int(hours), squeeze_bars=S.SQUEEZE_BARS if squeeze else 0)
            f = self.f if hours == self.p.breakout_hours else S.compute_features(self.df, params=p)
            self._base[key] = S.signals_from_rules(S.rule_table(self.df, f, p, None))
        return self._base[key]

    def timing(self) -> np.ndarray:
        return self.t["timing_ok"].to_numpy(bool)

    def setup(self) -> np.ndarray:
        """+1 / -1 where the system's trend, ADX, extension and timing rules allow a long / short."""
        return np.where(self.t["long_setup"], 1, np.where(self.t["short_setup"], -1, 0))

    def filtered(self, sig: np.ndarray) -> np.ndarray:
        s = self.setup()
        return np.where((sig != 0) & (sig == s), sig, 0).astype(np.int64)

    def timed(self, sig: np.ndarray) -> np.ndarray:
        return np.where(self.timing(), sig, 0).astype(np.int64)

    # --- back-tests
    def run1(self, sig: np.ndarray) -> pd.DataFrame:
        """One position at a time, cooldown after an exit (= strategy.run)."""
        tr = simulate_plan(self.df, sig.astype(np.int64), self.atr1, self.plan, self.sp, self.sl, self.clock)
        return self._trim(tr)

    def run2(self, base: np.ndarray, add: np.ndarray | None = None, max_open: int = 2) -> pd.DataFrame:
        """Up to `max_open` positions, same direction, cooldown when flat (= 高效積極模式's engine;
        same rules as research/efficient_active.run_variant with add_rule 'any', written for many signals)."""
        add = base if add is None else add
        bk = E.Book(self.df, self.atr1, self.plan, self.sp, self.sl, self.clock)
        n = self.n
        last_exit = -10 ** 9
        live: list[dict] = []
        for i in np.flatnonzero((base != 0) | (add != 0)):
            if i >= n - 1:
                continue
            live = [p for p in live if p["exit_i"] >= i + 1]
            if not live:
                d = int(base[i])
                if d == 0 or i < last_exit + self.plan.cooldown + 1:
                    continue
            else:
                if len(live) >= max_open:
                    continue
                d = int(add[i]) or int(base[i])
                if d == 0 or any(p["dir"] != d for p in live):
                    continue
            r = bk.open(int(i), d)
            if r is not None:
                r["add"] = bool(live)
                live.append(r)
                last_exit = max(last_exit, int(r["exit_i"]))
        if not bk.rows:
            return pd.DataFrame()
        return self._trim(pd.DataFrame(bk.rows).sort_values("sig_i").reset_index(drop=True))

    def _trim(self, tr: pd.DataFrame) -> pd.DataFrame:
        if tr.empty:
            return tr
        return tr[(tr["reason"] != "end") & (tr["entry_time"] >= pd.Timestamp("2005-01-01", tz="UTC"))].reset_index(drop=True)

    def baselines(self) -> dict[str, pd.DataFrame]:
        return {"normal": self.run1(self.sys_sig(24, True)),
                "efficient": self.run2(self.sys_sig(18, True)),
                "active": self._trim(E.run_variant(self.df, self.sys_sig(12, False), self.sys_sig(12, False), self.atr1,
                                                   self.plan, self.sp, self.sl, self.clock, "any", 2, False))}

    def h1_to_m15(self, s: pd.Series) -> np.ndarray:
        """Value of a 1h series at the 15m bar that closes the hour (and carried until the next close)."""
        return np.nan_to_num(align_htf(self.df.index, H15, s.to_frame("x"), H60)["x"].to_numpy(float), nan=0.0)


def row(name: str, tr: pd.DataFrame, extra: dict | None = None) -> dict:
    ev = E.evaluate(tr) if not tr.empty else {k: {"n": 0} for k in list(PERIODS) + ["all"]}
    r = {"name": name}
    for per, d in ev.items():
        for k in ("n", "per_week", "avgR", "R_yr", "maxDD", "eff", "win%"):
            r[f"{per}_{k}"] = d.get(k)
    if extra:
        r.update(extra)
    a = ev["all"]
    print(f"  {name:48s} {a.get('per_week', 0) or 0:5.2f}/wk avg {a.get('avgR') or 0:+.3f} R/yr {a.get('R_yr') or 0:5.1f} "
          f"DD {a.get('maxDD') or 0:5.1f} eff {a.get('eff')} | dev {ev['dev'].get('eff')} val {ev['val'].get('eff')} "
          f"test {ev['test'].get('eff')}", flush=True)
    return r


def first_of_run(cond: np.ndarray) -> np.ndarray:
    """True on the first bar of each run of True."""
    c = np.asarray(cond, bool)
    return c & ~np.r_[False, c[:-1]]


# ----------------------------------------------------------------------------- open-source triggers

def trig_supertrend(H: Harness, n=10, mult=3.0, tf="15m") -> np.ndarray:
    """SuperTrend (Olivier Seban; TradingView built-in): signal on the direction flip."""
    if tf == "15m":
        d = ind.supertrend(H.df, n, mult)["direction"].to_numpy()
    else:
        d = H.h1_to_m15(ind.supertrend(H.h1, n, mult)["direction"])
    flip = np.r_[0, np.diff(np.nan_to_num(d))]
    return np.where(flip > 0, 1, np.where(flip < 0, -1, 0))


def trig_utbot(H: Harness, key=1.0, n=10, tf="15m") -> np.ndarray:
    """UT Bot Alerts (QuantNomad, TradingView): ATR trailing stop on the close, buy when the close
    crosses above the stop, sell when it crosses below."""
    df = H.df if tf == "15m" else H.h1
    c = df["close"].to_numpy(float)
    a = ind.atr(df, n).to_numpy(float) * key
    stop = np.full(len(c), np.nan)
    pos = np.zeros(len(c))
    for i in range(1, len(c)):
        if np.isnan(a[i]):
            continue
        prev = stop[i - 1] if not np.isnan(stop[i - 1]) else c[i - 1]
        if c[i] > prev and c[i - 1] > prev:
            stop[i] = max(prev, c[i] - a[i])
        elif c[i] < prev and c[i - 1] < prev:
            stop[i] = min(prev, c[i] + a[i])
        elif c[i] > prev:
            stop[i] = c[i] - a[i]
        else:
            stop[i] = c[i] + a[i]
        pos[i] = 1 if c[i] > stop[i] else -1
    flip = np.r_[0, np.diff(pos)]
    s = np.where(flip > 0, 1, np.where(flip < 0, -1, 0))
    if tf != "15m":
        s = H.h1_to_m15(pd.Series(s, index=df.index))
        s = np.where(first_of_run(s != 0), s, 0)      # the 1h flip is a one-bar event on 15m
    return s.astype(np.int64)


def trig_lazybear(H: Harness, n=20, bb_k=2.0, kc_k=1.5, tf="15m") -> np.ndarray:
    """LazyBear's Squeeze Momentum: Bollinger(20,2) inside Keltner(20, 1.5 ATR) = squeeze; signal on
    the bar the squeeze releases, in the direction of the momentum histogram (linear regression of
    close minus the midpoint of the Donchian / SMA midlines)."""
    df = H.df if tf == "15m" else H.h1
    c = df["close"]
    bb = ind.bollinger(c, n, bb_k)
    mid = ind.sma(c, n)
    rng = ind.atr(df, n) * kc_k
    sq_on = (bb["lower"] > mid - rng) & (bb["upper"] < mid + rng)
    base = (df["high"].rolling(n).max() + df["low"].rolling(n).min()) / 2
    dev = c - (base + mid) / 2
    # linear-regression value (endpoint) of dev over n bars
    x = np.arange(n)
    xm = x.mean()
    def lr(a):
        return ((x - xm) * (a - a.mean())).sum() / ((x - xm) ** 2).sum() * (n - 1 - xm) + a.mean()
    val = dev.rolling(n).apply(lr, raw=True)
    rel = sq_on.shift(1, fill_value=False) & ~sq_on
    s = np.where(rel & (val > 0), 1, np.where(rel & (val < 0), -1, 0))
    if tf != "15m":
        s = H.h1_to_m15(pd.Series(s, index=df.index))
        s = np.where(first_of_run(s != 0), s, 0)
    return s.astype(np.int64)


def trig_keltner(H: Harness, n=20, k=1.5, tf="15m") -> np.ndarray:
    """Keltner-channel breakout (Chester Keltner / Linda Raschke): first close outside EMA20 +/- 1.5 ATR."""
    df = H.df if tf == "15m" else H.h1
    c = df["close"]
    e = ind.ema(c, n)
    a = ind.atr(df, n)
    up, dn = (c > e + k * a).to_numpy(), (c < e - k * a).to_numpy()
    s = np.where(first_of_run(up), 1, np.where(first_of_run(dn), -1, 0))
    if tf != "15m":
        s = H.h1_to_m15(pd.Series(s, index=df.index))
        s = np.where(first_of_run(s != 0), s, 0)
    return s.astype(np.int64)


def trig_lw(H: Harness, k=0.5) -> np.ndarray:
    """Larry Williams volatility breakout: today's open +/- k x yesterday's range (17:00 NY days);
    signal on the first 15m close beyond the level, once per day and direction."""
    sess = H.clock.session
    starts = np.flatnonzero(np.r_[True, sess[1:] != sess[:-1]])
    ends = np.r_[starts[1:], len(sess)]
    s = np.zeros(H.n, dtype=np.int64)
    prev_range = np.nan
    for a, b in zip(starts, ends):
        if not np.isnan(prev_range) and b - a > 8:
            op = H.o[a]
            up, dn = op + k * prev_range, op - k * prev_range
            cl = H.c[a:b]
            i_up = np.flatnonzero(cl > up)
            i_dn = np.flatnonzero(cl < dn)
            if len(i_up):
                s[a + i_up[0]] = 1
            if len(i_dn):
                s[a + i_dn[0]] = -1
        prev_range = H.h[a:b].max() - H.l[a:b].min()
    return s


def trig_bos(H: Harness, n=10) -> np.ndarray:
    """Market-structure break (Smart Money Concepts, LuxAlgo): a swing high = highest high with n
    bars on each side (confirmed n bars later); signal on the first close above the last confirmed
    swing high (BOS up) / below the last confirmed swing low."""
    h, l, c = H.h, H.l, H.c
    N = H.n
    last_hi = np.full(N, np.nan)
    last_lo = np.full(N, np.nan)
    cur_hi = cur_lo = np.nan
    for i in range(n, N - n):
        j = i + n                        # the swing at i is known at bar j
        if h[i] == h[i - n:i + n + 1].max():
            cur_hi = h[i]
        if l[i] == l[i - n:i + n + 1].min():
            cur_lo = l[i]
        last_hi[j] = cur_hi
        last_lo[j] = cur_lo
    up = c > last_hi
    dn = c < last_lo
    # a level is broken once; the next signal needs a new swing
    s = np.zeros(N, dtype=np.int64)
    used_hi = used_lo = np.nan
    for i in range(N):
        if up[i] and last_hi[i] != used_hi:
            s[i] = 1
            used_hi = last_hi[i]
        elif dn[i] and last_lo[i] != used_lo:
            s[i] = -1
            used_lo = last_lo[i]
    return s


def kama(c: np.ndarray, n=10, fast=2, slow=30) -> tuple[np.ndarray, np.ndarray]:
    """Kaufman adaptive moving average and its efficiency ratio."""
    change = np.abs(c - np.r_[np.full(n, np.nan), c[:-n]])
    vol = pd.Series(np.abs(np.diff(c, prepend=c[0]))).rolling(n).sum().to_numpy()
    er = np.where(vol > 0, change / vol, 0.0)
    sc = (er * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
    k = np.full(len(c), np.nan)
    k[n] = c[n]
    for i in range(n + 1, len(c)):
        k[i] = k[i - 1] + sc[i] * (c[i] - k[i - 1])
    return k, er


def trig_kama(H: Harness, n=10, er_min=0.3) -> np.ndarray:
    """Kaufman's adaptive moving average: close crosses KAMA while the efficiency ratio says 'trending'."""
    k, er = kama(H.c, n)
    above = H.c > k
    s = np.where(first_of_run(above) & (er >= er_min), 1, np.where(first_of_run(~above) & (er >= er_min), -1, 0))
    return s.astype(np.int64)


def trig_donchian_h1(H: Harness, n=55) -> np.ndarray:
    """Turtle system 2 entry on 1h bars: close above the 55-bar high / below the 55-bar low."""
    h1 = H.h1
    hi = h1["high"].rolling(n).max().shift(1)
    lo = h1["low"].rolling(n).min().shift(1)
    up, dn = (h1["close"] > hi).to_numpy(), (h1["close"] < lo).to_numpy()
    s = np.where(first_of_run(up), 1, np.where(first_of_run(dn), -1, 0))
    s = H.h1_to_m15(pd.Series(s, index=h1.index))
    return np.where(first_of_run(s != 0), s, 0).astype(np.int64)


# ----------------------------------------------------------------------------- regime filters

def hurst_rs(x: np.ndarray, min_k=8) -> float:
    """Rescaled-range Hurst exponent of a return series (classic, no corrections)."""
    n = len(x)
    ks, rs = [], []
    k = min_k
    while k <= n // 2:
        m = n // k
        seg = x[: m * k].reshape(m, k)
        dev = np.cumsum(seg - seg.mean(axis=1, keepdims=True), axis=1)
        r = dev.max(axis=1) - dev.min(axis=1)
        sd = seg.std(axis=1)
        ok = sd > 0
        if ok.sum():
            ks.append(k)
            rs.append((r[ok] / sd[ok]).mean())
        k *= 2
    if len(ks) < 3:
        return np.nan
    return float(np.polyfit(np.log(ks), np.log(rs), 1)[0])


def hurst_series(H: Harness, window=256) -> np.ndarray:
    """Hurst of 1h log returns over the last `window` hours, at each 15m bar (closed 1h bars only)."""
    r = np.log(H.h1["close"]).diff().to_numpy()
    out = np.full(len(r), np.nan)
    for i in range(window, len(r), 4):          # every 4 hours is plenty for a regime measure
        out[i] = hurst_rs(r[i - window + 1: i + 1])
    s = pd.Series(out, index=H.h1.index).ffill()
    return H.h1_to_m15(s)


def efficiency_ratio(H: Harness, n=48) -> np.ndarray:
    c = H.c
    change = np.abs(c - np.r_[np.full(n, np.nan), c[:-n]])
    vol = pd.Series(np.abs(np.diff(c, prepend=c[0]))).rolling(n).sum().to_numpy()
    return np.where(vol > 0, change / vol, np.nan)


def volume_ratio(H: Harness, n=20) -> np.ndarray:
    v = pd.Series(H.v)
    return (v / v.rolling(n).mean().shift(1)).to_numpy()


# ----------------------------------------------------------------------------- part 1

TRIGGERS = [
    ("SuperTrend(10,3) 15m 翻轉", lambda H: trig_supertrend(H, 10, 3.0, "15m")),
    ("SuperTrend(10,3) 1h 翻轉", lambda H: trig_supertrend(H, 10, 3.0, "1h")),
    ("UT Bot (key 1, ATR10) 15m", lambda H: trig_utbot(H, 1.0, 10, "15m")),
    ("UT Bot (key 2, ATR10) 1h", lambda H: trig_utbot(H, 2.0, 10, "1h")),
    ("LazyBear 擠壓動能 釋放 15m", lambda H: trig_lazybear(H, tf="15m")),
    ("LazyBear 擠壓動能 釋放 1h", lambda H: trig_lazybear(H, tf="1h")),
    ("Keltner 突破 (EMA20±1.5ATR) 15m", lambda H: trig_keltner(H, tf="15m")),
    ("Keltner 突破 (EMA20±1.5ATR) 1h", lambda H: trig_keltner(H, tf="1h")),
    ("Larry Williams 波動突破 (k=0.5)", lambda H: trig_lw(H, 0.5)),
    ("Larry Williams 波動突破 (k=0.8)", lambda H: trig_lw(H, 0.8)),
    ("結構突破 BOS (擺動 10 根)", lambda H: trig_bos(H, 10)),
    ("結構突破 BOS (擺動 20 根)", lambda H: trig_bos(H, 20)),
    ("KAMA(10,2,30) 穿越 + ER≥0.3", lambda H: trig_kama(H, 10, 0.3)),
    ("海龜 55 根 1h 突破", lambda H: trig_donchian_h1(H, 55)),
]


def part_classic(H: Harness) -> pd.DataFrame:
    rows = []
    print("== baselines")
    base = H.baselines()
    for k, tr in base.items():
        rows.append(row(f"baseline {k}", tr, {"kind": "baseline"}))
    eff_sig = H.sys_sig(18, True)
    print("== open-source triggers: raw / with the system's trend filters / as the 2nd position of 高效積極模式")
    for name, fn in TRIGGERS:
        t0 = time.time()
        raw = H.timed(fn(H))
        flt = H.filtered(raw)
        rows.append(row(f"{name} | 原始", H.run1(raw), {"kind": "raw", "signals": int((raw != 0).sum())}))
        rows.append(row(f"{name} | 加系統過濾", H.run1(flt), {"kind": "filtered", "signals": int((flt != 0).sum())}))
        rows.append(row(f"{name} | 加系統過濾, 最多 2 張", H.run2(flt), {"kind": "filtered2"}))
        rows.append(row(f"{name} | 當高效積極的第 2 張", H.run2(eff_sig, flt), {"kind": "addon", "secs": round(time.time() - t0)}))
    print("== regime filters on the active mode's extra signals (12h, no squeeze)")
    s12 = H.sys_sig(12, False)
    hu = hurst_series(H)
    er = efficiency_ratio(H, 48)
    vr = volume_ratio(H, 20)
    for name, mask in [("Hurst(256h) > 0.5", hu > 0.5), ("Hurst(256h) > 0.55", hu > 0.55),
                       ("Kaufman ER(48) ≥ 0.3", er >= 0.3), ("Kaufman ER(48) ≥ 0.4", er >= 0.4),
                       ("訊號K成交量 ≥ 1.5×", vr >= 1.5), ("訊號K成交量 ≥ 2×", vr >= 2.0)]:
        sig = np.where(np.nan_to_num(mask, nan=False), s12, 0)
        rows.append(row(f"12h 不收窄 + {name} | 最多 2 張", H.run2(sig), {"kind": "regime", "signals": int((sig != 0).sum())}))
        rows.append(row(f"12h 不收窄 + {name} | 當高效積極的第 2 張", H.run2(eff_sig, sig), {"kind": "regime_addon"}))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "XAUUSD_open_source_classic.csv", index=False)
    return res


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    which = sys.argv[1] if len(sys.argv) > 1 else "classic"
    df = E.load()
    H = Harness(df)
    if which == "classic":
        part_classic(H)
    else:
        from open_source_ml import main as ml_main
        ml_main(H, which)
