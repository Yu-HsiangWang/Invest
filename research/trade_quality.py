"""Which signal-time features separate good from bad trades of the core 1h breakout system?
Selection is judged on IS; VAL+TEST are shown only to confirm."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.h1_refine import *  # noqa

sig = signals(32, "ema20/50")
tr, rows = run(sig, 3.0, ExitRules(tp_r=0, trail_mult=4.5, cooldown=2), "core n=32 stop3 trail4.5")
i = tr["sig_i"].to_numpy(); dirn = tr["dir"].to_numpy()
hh = h.iloc[i]
feat = pd.DataFrame(index=tr.index)
feat["h4_st_agree"] = (H4F["st"].to_numpy()[i] == dirn)
h4trend = np.where((H4F["close"] > H4F["ema20"]) & (H4F["ema20"] > H4F["ema50"]), 1, np.where((H4F["close"] < H4F["ema20"]) & (H4F["ema20"] < H4F["ema50"]), -1, 0))
feat["h4_ema_agree"] = (h4trend[i] == dirn)
feat["adx1"] = adx1[i]
feat["adx4"] = H4F["adx"].to_numpy()[i]
feat["er1"] = er1[i]
datr = align_htf(h.index, H1, pd.DataFrame({"atr": ind.atr(d, 14)}), D1)["atr"].to_numpy()
feat["d_ext"] = dirn * (D["close"].to_numpy()[i] - D["ema20"].to_numpy()[i]) / datr[i]
bar_body = (hh["close"].to_numpy() - hh["open"].to_numpy()) * dirn / atr.to_numpy()[i]
feat["body_atr"] = bar_body
rng = (hh["high"] - hh["low"]).to_numpy()
pos = np.where(dirn > 0, (hh["close"] - hh["low"]).to_numpy() / rng, (hh["high"] - hh["close"]).to_numpy() / rng)
feat["close_pos"] = pos
feat["vol_regime"] = (atr / atr.rolling(500).median()).to_numpy()[i]
feat["ny_close_h"] = close_ny[i]
feat["session"] = pd.cut(close_ny[i], [-1, 2, 7, 12, 17, 24], labels=["NYlate/Asia(18-02)", "Asia/LDNopen(02-07)", "LDN/NY am(07-12)", "NY pm(12-17)", "Asia open(17-24)"])
feat["dow"] = dow[i]
feat["R"] = tr["R"].to_numpy()
feat["period"] = np.where(tr["entry_time"] < pd.Timestamp("2017-01-01", tz="UTC"), "IS", "OOS")

def bucket_report(col, q=None, bins=None):
    x = feat[col]
    if q:
        edges = np.unique(np.nanquantile(feat.loc[feat.period == "IS", col], np.linspace(0, 1, q + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        b = pd.cut(x, edges)
    elif bins is not None:
        b = pd.cut(x, bins)
    else:
        b = x
    t = feat.groupby([b, "period"], observed=True)["R"].agg(["size", "mean"]).unstack("period")
    print(f"\n-- {col}\n{t.round(3).to_string()}")

for col in ["h4_st_agree", "h4_ema_agree", "session", "dow"]:
    bucket_report(col)
for col in ["adx1", "adx4", "er1", "d_ext", "body_atr", "close_pos", "vol_regime"]:
    bucket_report(col, q=4)
