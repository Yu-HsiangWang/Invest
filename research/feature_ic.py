"""Which features carry directional information at intraday horizons?
Mean forward return (bp) after conditioning on feature sign, per period."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.run_grid import load_cached
from research.common import PERIODS
from research.strategies import trend_dir

df, f = load_cached()
o = df["open"]; c = df["close"]
# forward return from next open to the open k bars later (log, bp)
def fwd(k):
    return (np.log(o.shift(-1 - k)) - np.log(o.shift(-1))) * 1e4
F = {k: fwd(k) for k in (4, 16, 32)}
close_ny = (f["ny_h"] + 0.25) % 24
active = (close_ny > 3) & (close_ny <= 12) & (f["dow"] <= 4)

d1_mom20 = np.sign(f["d1_close"] - f["d1_close"].shift(20 * 92))  # rough: 20 trading days of 15m bars
feats = {
    "d1: close>ema20": np.sign(f["d1_close"] - f["d1_ema20"]),
    "h4: ema trend": pd.Series(trend_dir(f, "h4", "ema"), index=f.index),
    "h4: supertrend": f["h4_st_dir"],
    "h1: ema trend": pd.Series(trend_dir(f, "h1", "ema"), index=f.index),
    "h1: supertrend": f["h1_st_dir"],
    "m15: close>ema50": np.sign(c - f["ema50"]),
    "m15: rsi>50": np.sign(f["rsi"] - 50),
    "m15: macd_h>0": np.sign(f["macd_h"]),
    "h1: ema200": pd.Series(trend_dir(f, "h1", "ema200"), index=f.index),
}
# Asian session (London 00-07) return sign, known after 07:00 London
ldn = df.index.tz_convert("Europe/London")
day = ldn.normalize()
tmp = pd.DataFrame({"d": day, "h": f["ldn_h"].values, "o": o.values, "c": c.values}, index=df.index)
asia_open = tmp[tmp.h == 0.0].groupby("d")["o"].first()
asia_close = tmp[tmp.h == 6.75].groupby("d")["c"].last()
asia_ret = np.sign(asia_close - asia_open)
ar = pd.Series(day, index=df.index).map(asia_ret)
ar = ar.where(f["ldn_h"].values >= 7.0)  # known only after 07:00 London
feats["asia session dir (after 07 LDN)"] = ar
# previous trading day direction
tday = f["tday"]
dclose = c.groupby(tday).last()
dopen = o.groupby(tday).first()
prev_dir = np.sign(dclose - dopen).shift(1)
feats["prev day dir"] = tday.map(prev_dir)

rows = []
for name, s in feats.items():
    for pname, (a, b) in PERIODS.items():
        m = active & (df.index >= pd.Timestamp(a, tz="UTC")) & (df.index < pd.Timestamp(b, tz="UTC"))
        r = {}
        for k, fr in F.items():
            x = fr[m]; sg = s[m]
            up = x[sg > 0].mean(); dn = x[sg < 0].mean()
            r[f"{k*15//60}h spread(bp)"] = round(up - dn, 2)
        rows.append({"feature": name, "period": pname.split()[0], **r})
out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(out.pivot(index="feature", columns="period").round(2).to_string())
