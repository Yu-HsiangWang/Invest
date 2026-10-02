"""Refine the 1h breakout + daily-trend family; robustness across neighbours."""
import sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goldsignal import indicators as ind
from goldsignal.bars import resample, align_htf
from goldsignal.backtest import ExitRules, simulate, metrics, by_year
from research.common import PERIODS, SPREAD_PCT, SLIP_PCT
from research.run_grid import load_cached, fmt

df15, _ = load_cached()
h = resample(df15, "1h"); d = resample(df15, "1D"); h4 = resample(df15, "4h")
H1 = pd.Timedelta(hours=1); D1 = pd.Timedelta(days=1); H4 = pd.Timedelta(hours=4)
c = h["close"]; atr = ind.atr(h, 14)
ny = h.index.tz_convert("America/New_York")
close_ny = ((ny.hour + 1) % 24).to_numpy(); dow = ny.dayofweek.to_numpy()

def costs(mult=1.0):
    p = h["open"].to_numpy()
    return p * SPREAD_PCT * mult, p * SLIP_PCT * mult

dc = d["close"]
dfe = pd.DataFrame({"close": dc, "ema10": ind.ema(dc, 10), "ema20": ind.ema(dc, 20), "ema50": ind.ema(dc, 50),
                    "mom20": dc - dc.shift(20), "st": ind.supertrend(d, 10, 3.0)["direction"]}, index=d.index)
D = align_htf(h.index, H1, dfe, D1)
h4c = h4["close"]
h4e = pd.DataFrame({"close": h4c, "ema20": ind.ema(h4c, 20), "ema50": ind.ema(h4c, 50), "st": ind.supertrend(h4, 10, 3.0)["direction"],
                    "adx": ind.adx(h4, 14)["adx"]}, index=h4.index)
H4F = align_htf(h.index, H1, h4e, H4)
adx1 = ind.adx(h, 14)["adx"].to_numpy()
er1 = ind.efficiency_ratio(c, 20).to_numpy()

def dtrend(kind):
    if kind == "ema20/50":
        up = (D["close"] > D["ema20"]) & (D["ema20"] > D["ema50"]); dn = (D["close"] < D["ema20"]) & (D["ema20"] < D["ema50"])
    elif kind == "close>ema20":
        up = D["close"] > D["ema20"]; dn = D["close"] < D["ema20"]
    elif kind == "mom20":
        up = D["mom20"] > 0; dn = D["mom20"] < 0
    elif kind == "d-supertrend":
        up = D["st"] > 0; dn = D["st"] < 0
    elif kind == "none":
        return np.zeros(len(h)) + 9
    return np.where(up, 1, np.where(dn, -1, 0))

def signals(n, kind, win=None, extra=None):
    up = h["high"].rolling(n).max().shift(1); dn = h["low"].rolling(n).min().shift(1)
    lt = ((c > up) & (c.shift(1) <= up.shift(1))).to_numpy(); st = ((c < dn) & (c.shift(1) >= dn.shift(1))).to_numpy()
    tdir = dtrend(kind)
    ok = (dow <= 4) & ~((dow == 4) & (close_ny >= 12))
    if win is not None:
        ok &= (close_ny > win[0]) & (close_ny <= win[1])
    if extra is not None:
        ok &= extra
    L = ok & lt & ((tdir > 0) | (tdir == 9)); S = ok & st & ((tdir < 0) | (tdir == 9))
    return np.where(L, 1, np.where(S, -1, 0))

def run(sig, stop_k, rules, label, cost_mult=1.0, force=None, show=True):
    sl = (c - stop_k * atr).to_numpy(); ss = (c + stop_k * atr).to_numpy()
    sp, sl_ = costs(cost_mult)
    tr = simulate(h, sig, sl, ss, rules, sp, sl_, atr=atr.to_numpy(), force_exit=force)
    rows = {}
    for name, (a, b) in PERIODS.items():
        m = (tr["entry_time"] >= pd.Timestamp(a, tz="UTC")) & (tr["entry_time"] < pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1))
        rows[name] = metrics(tr[m], (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25)
    if show:
        print(f"{label}\n    {fmt(rows)}  avg_bars={tr['bars'].mean():.1f}")
    return tr, rows

if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "1"
    if part == "1":
        print("### daily trend definition (n=40, stop 3, trail 4.5)")
        for kind in ["none", "close>ema20", "ema20/50", "mom20", "d-supertrend"]:
            run(signals(40, kind), 3.0, ExitRules(tp_r=0, trail_mult=4.5, cooldown=2), f"dtrend={kind}")
        print("### neighbourhood (dtrend=ema20/50)")
        for n, sk, tm in itertools.product([24, 32, 40, 48, 60], [2.5, 3.0, 3.5], [3.5, 4.5, 5.5]):
            run(signals(n, "ema20/50"), sk, ExitRules(tp_r=0, trail_mult=tm, cooldown=2), f"n={n} stop={sk} trail={tm}")

if __name__ == "__main__" and part == "2":
    print("### holding-time constraints (n=32, dtrend ema20/50, stop 3, trail 4.5)")
    sig = signals(32, "ema20/50")
    sess_exit = close_ny == 16  # bar closing 16:00 NY -> flat before rollover
    fri_exit = (dow == 4) & (close_ny == 16)
    run(sig, 3.0, ExitRules(tp_r=0, trail_mult=4.5, cooldown=2), "no limit")
    run(sig, 3.0, ExitRules(tp_r=0, trail_mult=4.5, cooldown=2), "flat before weekend", force=fri_exit)
    for mb in [6, 12, 24, 48]:
        run(sig, 3.0, ExitRules(tp_r=0, trail_mult=4.5, cooldown=2, max_bars=mb), f"max_bars={mb}", force=fri_exit)
    print("### intraday mode: entries 03-12 NY, flat 16:00 NY")
    for n, sk, tm, tp in itertools.product([12, 16, 24, 32], [2.0, 3.0], [3.0, 4.5], [0, 2.0]):
        sig_w = signals(n, "ema20/50", win=(3, 12))
        run(sig_w, sk, ExitRules(tp_r=tp, trail_mult=tm, cooldown=2), f"intraday n={n} stop={sk} trail={tm} tp={tp}", force=sess_exit)
