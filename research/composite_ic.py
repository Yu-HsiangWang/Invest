"""Do agreeing weak signals combine into a stronger directional edge?"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.run_grid import load_cached
from research.common import PERIODS
from research.strategies import trend_dir

df, f = load_cached()
o = df["open"]; c = df["close"]
def fwd(k): return (np.log(o.shift(-1 - k)) - np.log(o.shift(-1))) * 1e4
F = {k: fwd(k) for k in (8, 16, 32)}
close_ny = (f["ny_h"] + 0.25) % 24
dow = f["dow"]
ldn = df.index.tz_convert("Europe/London"); day = ldn.normalize()
tmp = pd.DataFrame({"d": day, "h": f["ldn_h"].values, "o": o.values, "c": c.values}, index=df.index)
asia_ret = np.sign(tmp[tmp.h == 6.75].groupby("d")["c"].last() - tmp[tmp.h == 0.0].groupby("d")["o"].first())
asia = pd.Series(day, index=df.index).map(asia_ret).where(f["ldn_h"].values >= 7.0).fillna(0)

comp = {
  "m15ema50": np.sign(c - f["ema50"]),
  "m15rsi": np.sign(f["rsi"] - 50),
  "h1st": f["h1_st_dir"],
  "h4st": f["h4_st_dir"],
  "h1ema": pd.Series(trend_dir(f, "h1", "ema"), index=f.index),
  "asia": asia,
}
def table(score, mask, label):
    rows=[]
    for pname,(a,b) in PERIODS.items():
        m = mask & (df.index >= pd.Timestamp(a, tz="UTC")) & (df.index < pd.Timestamp(b, tz="UTC"))
        for lvl in sorted(score[m].dropna().unique()):
            mm = m & (score == lvl)
            r = {"period": pname.split()[0], "score": lvl, "n": int(mm.sum())}
            for k, fr in F.items():
                r[f"{k*15//60}h"] = round(fr[mm].mean(), 2)
            rows.append(r)
    t = pd.DataFrame(rows)
    print(f"\n== {label}")
    print(t.pivot(index="score", columns="period", values=["n", "2h", "4h", "8h"]).to_string())

win = (close_ny > 3) & (close_ny <= 12) & (dow <= 4)
s1 = comp["m15ema50"] + comp["h1st"] + comp["h4st"] + comp["asia"]
table(s1, win, "S1 = m15ema50 + h1st + h4st + asia   (entries 03-12 NY)")
s2 = comp["m15ema50"] + comp["m15rsi"] + comp["h1st"] + comp["h1ema"] + comp["h4st"]
table(s2, win, "S2 = m15ema50 + m15rsi + h1st + h1ema + h4st")
