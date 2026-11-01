"""Intermarket filters for the gold / silver signals (docs/RESEARCH.md section 12.5): the US dollar
index, the S&P 500, the 10-year T-note future and the other metal, as daily-trend conditions on
the 15m signals. Gold is usually said to move against the dollar and with silver; the question
is whether those relations improve the system's trades after costs.

Data: Dukascopy hourly BID candles 2013-2026 for DOLLAR.IDX-USD, USA500.IDX-USD, USTBOND.TR-USD
(downloaded in the browser like everything else from Dukascopy) and the 15m gold / silver bars.

Each market gives a trend state at every gold bar, computed from hourly closes with no
look-ahead (the hourly bar must have closed before the 15m bar): +1 when close > EMA(20 days)
> EMA(50 days), -1 for the opposite, 0 otherwise; plus the 5-day direction (sign of the close
change over 120 hourly bars).

Decision rule, fixed before running: a filter is adopted only if it raises the efficiency
(R per year / max drawdown) in every Dukascopy period and overall while keeping at least 80 %
of the trades; otherwise it is reported and not used.

    res = run_all(frames15, hourly)      # in Pyodide: frames15 = {"XAUUSD": df, "XAGUSD": df}, hourly = {"DXY": df, ...}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

import efficient_active as E  # noqa: E402
import efficient_dukascopy as D  # noqa: E402
import open_source_algos as A  # noqa: E402
from goldsignal.bars import align_htf, resample  # noqa: E402

H15, H60 = pd.Timedelta(minutes=15), pd.Timedelta(hours=1)


def trend_state(h1: pd.DataFrame, base_index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """(+1/0/-1 trend, +1/-1 5-day direction) of an hourly series at every 15m bar of `base_index`."""
    c = h1["close"]
    f = pd.DataFrame({"e20": c.ewm(span=20 * 24, adjust=False).mean(), "e50": c.ewm(span=50 * 24, adjust=False).mean(),
                      "c": c, "c5": c.shift(120)})
    al = align_htf(base_index, H15, f, H60)
    tr = np.where((al["c"] > al["e20"]) & (al["e20"] > al["e50"]), 1, np.where((al["c"] < al["e20"]) & (al["e20"] < al["e50"]), -1, 0))
    mom = np.sign((al["c"] - al["c5"]).to_numpy(float))
    return np.nan_to_num(tr).astype(int), np.nan_to_num(mom).astype(int)


def period_of(ts: pd.Series) -> pd.Series:
    out = pd.Series("2025/4–2026/10", index=ts.index)
    out[ts < pd.Timestamp("2025-04-01", tz="UTC")] = "2021–2025/3"
    out[ts < pd.Timestamp("2021-01-01", tz="UTC")] = "2017–2020"
    out[ts < pd.Timestamp("2017-01-01", tz="UTC")] = "2013/7–2016"
    return out


def split(tr: pd.DataFrame, col: str, label: str) -> list[dict]:
    rows = []
    tr = tr.copy(); tr["period"] = period_of(tr["entry_time"])
    for per in list(dict.fromkeys(tr["period"])) + ["all"]:
        X = tr if per == "all" else tr[tr["period"] == per]
        for g, Y in X.groupby(col):
            if len(Y) >= 5:
                r = Y["R"].to_numpy()
                rows.append({"feature": label, "period": per, "group": str(g), "n": int(len(Y)), "avgR": round(float(r.mean()), 3),
                             "totR": round(float(r.sum()), 1), "win%": round(100 * float((r > 0).mean()), 1)})
    return rows


def run_all(frames: dict[str, pd.DataFrame], hourly: dict[str, pd.DataFrame]) -> dict:
    res = {"how": ("Dukascopy 15m gold / silver 2013/7-2026/10 with hourly DXY / S&P 500 / 10y T-note; 高效積極 rules for gold "
                   "(18h + squeeze, 2 same-direction positions) and the normal rules for silver; a filter removes the signals "
                   "whose condition fails. eff = R per year / max drawdown."), "splits": [], "backtests": {}}
    states = {}
    for sym, df in frames.items():
        H = A.Harness(df, sym)
        base = H.sys_sig(18, True) if sym == "XAUUSD" else H.sys_sig(24, False)
        run = (lambda s: H.run2(s)) if sym == "XAUUSD" else (lambda s: H.run1(s))
        feats = {}
        for name, h1 in hourly.items():
            feats[name + "_trend"], feats[name + "_5d"] = trend_state(h1, df.index)
        other = "XAGUSD" if sym == "XAUUSD" else "XAUUSD"
        if other in frames:
            oh1 = resample(frames[other], "1h")
            feats["other_trend"], feats["other_5d"] = trend_state(oh1, df.index)
        states[sym] = feats
        tr0 = run(base)
        tr0 = tr0[tr0["entry_time"] >= pd.Timestamp(D.START, tz="UTC")]
        bt = {"baseline": D.by_period(tr0)}
        splits = []
        for name, st in feats.items():
            rel = np.sign(st[tr0["sig_i"].to_numpy()] * tr0["dir"].to_numpy())      # +1 same direction as the trade, -1 opposite
            tr = tr0.copy(); tr["rel"] = np.where(rel > 0, "同向", np.where(rel < 0, "反向", "中性"))
            splits += split(tr, "rel", f"{sym} {name}")
            sgn = np.sign(st)
            for cond_name, cond in [("同向", sgn == np.sign(base)), ("非反向", sgn != -np.sign(base)), ("反向", sgn == -np.sign(base))]:
                tr = run(np.where(cond, base, 0))
                if tr.empty:
                    continue
                tr = tr[tr["entry_time"] >= pd.Timestamp(D.START, tz="UTC")]
                bt[f"{name} {cond_name}"] = D.by_period(tr)
        res["backtests"][sym] = bt
        res["splits"] += splits
    return res
