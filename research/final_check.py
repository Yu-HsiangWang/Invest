"""Final configuration checks: exit variants, alt data feed, costs, by-year."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goldsignal.backtest import ExitRules, simulate, metrics, by_year
from research.common import PERIODS, costs, htf_features, ROOT
from research.run_grid import fmt
from goldsignal.bars import load_ejtrader_15m, resample, align_htf
from goldsignal import indicators as ind

def build(df, f):
    c, h, l = df["close"], df["high"], df["low"]
    d = resample(df, "1D")
    D = align_htf(df.index, pd.Timedelta(minutes=15), pd.DataFrame({"close": d["close"], "ema20": ind.ema(d["close"], 20), "ema50": ind.ema(d["close"], 50), "atr": ind.atr(d, 14)}), pd.Timedelta(days=1))
    dtr = np.where((D.close > D.ema20) & (D.ema20 > D.ema50), 1, np.where((D.close < D.ema20) & (D.ema20 < D.ema50), -1, 0))
    h4t = np.where((f.h4_close > f.h4_ema20) & (f.h4_ema20 > f.h4_ema50), 1, np.where((f.h4_close < f.h4_ema20) & (f.h4_ema20 < f.h4_ema50), -1, 0))
    h1t = np.where((f.h1_close > f.h1_ema20) & (f.h1_ema20 > f.h1_ema50), 1, np.where((f.h1_close < f.h1_ema20) & (f.h1_ema20 < f.h1_ema50), -1, 0))
    ext = ((D.close - D.ema20) / D.atr).to_numpy()
    rng = (h - l).to_numpy(); rng = np.where(rng > 0, rng, np.nan)
    cpl = (c - l).to_numpy() / rng; cps = (h - c).to_numpy() / rng
    return dict(dtr=dtr, h4t=h4t, h1t=h1t, ext=ext, cpl=cpl, cps=cps, adx1=f["h1_adx"].to_numpy(), atr1=f["h1_atr"].to_numpy())

def signals(df, f, X, n=96, cpos=0.7, adx_max=30.0, ext_max=1.5):
    c, h, l = df["close"], df["high"], df["low"]
    up = h.rolling(n).max().shift(1); dn = l.rolling(n).min().shift(1)
    lt = ((c > up) & (c.shift(1) <= up.shift(1))).to_numpy(); st = ((c < dn) & (c.shift(1) >= dn.shift(1))).to_numpy()
    dow = f["dow"].to_numpy(); cny = (f["ny_h"].to_numpy() + .25) % 24
    ok = (dow <= 4) & ~((dow == 4) & (cny > 12)) & ~(X["adx1"] > adx_max)
    L = ok & lt & (X["dtr"] > 0) & (X["h4t"] > 0) & (X["h1t"] > 0) & (X["ext"] <= ext_max) & (X["cpl"] >= cpos)
    S = ok & st & (X["dtr"] < 0) & (X["h4t"] < 0) & (X["h1t"] < 0) & (-X["ext"] <= ext_max) & (X["cps"] >= cpos)
    return np.where(L, 1, np.where(S, -1, 0))

def run(df, f, X, rules, stop_k=2.0, cost_mult=1.0, label="", periods=PERIODS, **kw):
    sig = signals(df, f, X, **kw)
    c = df["close"].to_numpy()
    sp, sl_ = costs(df, cost_mult)
    tr = simulate(df, sig, c - stop_k * X["atr1"], c + stop_k * X["atr1"], rules, sp, sl_, atr=X["atr1"])
    rows = {}
    for name, (a, b) in periods.items():
        m = (tr["entry_time"] >= pd.Timestamp(a, tz="UTC")) & (tr["entry_time"] < pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1))
        rows[name] = metrics(tr[m], (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25)
    print(f"{label}  hrs={tr['bars'].mean()/4:.1f}\n    {fmt(rows)}")
    return tr

if __name__ == "__main__":
    df = pd.read_parquet(ROOT / "data/cache_xau15.parquet"); f = pd.read_parquet(ROOT / "data/cache_feat15.parquet")
    X = build(df, f)
    base = ExitRules(tp_r=0, trail_mult=5.0, cooldown=4)
    tr = run(df, f, X, base, label="FINAL n=96 stop2.0 trail5.0")
    run(df, f, X, ExitRules(tp_r=0, trail_mult=5.0, cooldown=4, be_r=1.0, be_lock_r=0.1), label="+ breakeven after +1R")
    run(df, f, X, ExitRules(tp_r=0, trail_mult=5.0, cooldown=4, be_r=1.5, be_lock_r=0.5), label="+ lock 0.5R after +1.5R")
    run(df, f, X, ExitRules(tp_r=3.0, trail_mult=5.0, cooldown=4), label="+ TP at 3R")
    run(df, f, X, base, cost_mult=2.0, label="costs x2")
    run(df, f, X, base, cost_mult=3.0, label="costs x3")
    run(df, f, X, base, label="no close-pos filter", cpos=0.0)
    run(df, f, X, base, label="no adx cap", adx_max=999)
    run(df, f, X, base, label="no extension cap", ext_max=99)
    print(by_year(tr).to_string())
    print("long:", metrics(tr[tr.dir > 0]), "\nshort:", metrics(tr[tr.dir < 0]))
    print("hold hours median/p75/p90:", tr.bars.median() / 4, tr.bars.quantile(.75) / 4, tr.bars.quantile(.9) / 4)
    # ---------- second data feed
    print("\n##### ALT DATA FEED (ejtraderLabs, 2012-05..2022-03)")
    alt = load_ejtrader_15m(ROOT / "data/raw/ejt_XAUUSD_m15.csv")
    fa = htf_features(alt)
    Xa = build(alt, fa)
    alt_periods = {"ALT 2012-2016": ("2012-06-01", "2016-12-31"), "ALT 2017-2020": ("2017-01-01", "2020-12-31"), "ALT 2021-2022Q1": ("2021-01-01", "2022-03-04")}
    run(alt, fa, Xa, base, label="FINAL on alt feed", periods=alt_periods)
    run(df, f, X, base, label="FINAL on main feed, same windows", periods=alt_periods)
