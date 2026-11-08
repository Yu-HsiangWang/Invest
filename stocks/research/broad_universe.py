"""Study 2: survivorship check. Do the state -> forward-return patterns hold on EVERY stock that existed,
not just today's winners?

* Taiwan: all TWSE common stocks in the official daily files (tsec mirror, 1994-2019/5; raw prices,
  ex-dividend days' returns set to 0 so stock dividends do not create fake drops; cash dividends are
  therefore ignored for every state alike).
* US: all historical S&P 500 constituents (plain_backtest, adjusted), using only the days each stock
  was actually a member (membership_intervals.csv) so delisted / dropped names are included.

    python research/broad_universe.py --datasets /path/to/datasets
Outputs results/broad_forward_TW.csv, results/broad_forward_US.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.common import RESULTS, period_of  # noqa: E402
from stocksignal import zones as Z  # noqa: E402
from stocksignal.indicators import momentum  # noqa: E402

HORIZONS = (126, 252)


def states(close: pd.Series) -> pd.DataFrame:
    st = pd.DataFrame(index=close.index)
    st["channel"] = Z.channel_zone(Z.channel_bands(close)["zscore"])
    st["deviation"] = Z.deviation_zone(Z.deviation_bands(close)["dev_rank"])
    st["trend"] = Z.trend_state(close)
    m = momentum(close)
    st["mom"] = np.sign(m).where(m.notna())
    return st


def pooled(frames: dict[str, pd.Series], mask: dict[str, pd.Series] | None, cap_ret: float = 3.0) -> pd.DataFrame:
    rows = []
    recs = []
    for sym, c in frames.items():
        if len(c) < 1000:
            continue
        st = states(c)
        for h in HORIZONS:
            fwd = (c.shift(-h) / c - 1.0).clip(upper=cap_ret)
            d = pd.DataFrame({"fwd": fwd, "period": period_of(c.index), "h": h, "sym": sym}).join(st)
            if mask is not None and sym in mask:
                d = d[mask[sym].reindex(d.index).fillna(False).astype(bool)]
            recs.append(d.dropna(subset=["fwd"]))
    d = pd.concat(recs)
    for col in ("channel", "deviation", "trend", "mom"):
        for h in HORIZONS:
            dh = d[d.h == h]
            for per in ("IS", "VAL", "TEST", "ALL"):
                dd = dh if per == "ALL" else dh[dh.period == per]
                if len(dd) < 500:
                    continue
                base = dd["fwd"]
                for stv, g in dd.dropna(subset=[col]).groupby(col):
                    if len(g) < 300:
                        continue
                    rows.append({"method": col, "horizon": h, "period": per, "state": stv, "n": len(g), "n_sym": g["sym"].nunique(),
                                 "mean": g["fwd"].mean(), "median": g["fwd"].median(), "hit": (g["fwd"] > 0).mean(),
                                 "base_mean": base.mean(), "base_hit": (base > 0).mean(), "excess_mean": g["fwd"].mean() - base.mean(),
                                 "excess_median": g["fwd"].median() - base.median(), "excess_hit": (g["fwd"] > 0).mean() - (base > 0).mean()})
    return pd.DataFrame(rows)


def load_tw(ds: Path, max_files: int | None = None) -> dict[str, pd.Series]:
    out = {}
    files = sorted(glob.glob(str(ds / "tsec" / "repo" / "data" / "*.csv")))
    for f in files:
        code = os.path.basename(f)[:-4]
        if not (len(code) == 4 and code.isdigit()):          # common stocks only (ETFs 00xx, warrants 6 digits excluded)
            continue
        if code.startswith("00"):
            continue
        try:
            t = pd.read_csv(f, header=None, names=["d", "vol", "val", "o", "h", "l", "c", "chg", "n"], dtype=str)
        except Exception:  # noqa: BLE001
            continue
        if len(t) < 1000:
            continue
        def roc(x):
            y, m, dd = str(x).split("/")
            return pd.Timestamp(int(y) + 1911, int(m), int(dd))
        try:
            idx = t["d"].map(roc)
        except Exception:  # noqa: BLE001
            continue
        c = pd.to_numeric(t["c"].str.replace(",", ""), errors="coerce")
        ex = t["chg"].astype(str).str.contains("X")
        c.index = idx
        c = c[c > 0].sort_index()
        c = c[~c.index.duplicated(keep="last")]
        c = c[c.index >= "2000-01-01"]
        if len(c) < 1000:
            continue
        r = c.pct_change()
        exd = ex.to_numpy()[: len(t)]
        ex_s = pd.Series(ex.to_numpy(), index=idx).reindex(c.index).fillna(False).astype(bool)
        r[ex_s] = 0.0                                    # ignore the ex-dividend/ex-rights drop
        r[r.abs() > 0.12] = 0.0                          # impossible under the 7%/10% daily limit -> glitch
        adj = (1 + r.fillna(0)).cumprod() * float(c.iloc[0])
        out[code] = adj
        if max_files and len(out) >= max_files:
            break
    return out


def load_us(ds: Path) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    base = ds / "plain_backtest" / "repo" / "data" / "processed"
    mem = pd.read_csv(base / "universes" / "sp500" / "membership_intervals.csv")
    mem.columns = [c.lower() for c in mem.columns]
    sym_col = next(c for c in mem.columns if "symbol" in c or "ticker" in c)
    start_col = next(c for c in mem.columns if "start" in c)
    end_col = next(c for c in mem.columns if "end" in c)
    frames, masks = {}, {}
    for sym, g in mem.groupby(sym_col):
        p = base / "daily" / "equities" / f"{sym}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, usecols=["date", "close"])
        c = pd.Series(pd.to_numeric(df["close"], errors="coerce").to_numpy(), index=pd.to_datetime(df["date"])).dropna()
        c = c[c > 0]
        if len(c) < 1000:
            continue
        m = pd.Series(False, index=c.index)
        for _, row in g.iterrows():
            s = pd.to_datetime(row[start_col], errors="coerce")
            e = pd.to_datetime(row[end_col], errors="coerce")
            s = s if pd.notna(s) else c.index[0]
            e = e if pd.notna(e) else c.index[-1]
            m[(c.index >= s) & (c.index <= e)] = True
        frames[sym] = c
        masks[sym] = m
    return frames, masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--market", default="both")
    a = ap.parse_args()
    ds = Path(a.datasets)
    pd.set_option("display.width", 250)
    if a.market in ("both", "TW"):
        fr = load_tw(ds)
        print(f"TW broad universe: {len(fr)} stocks")
        t = pooled(fr, None)
        t.to_csv(RESULTS / "broad_forward_TW.csv", index=False)
        print(t[t.horizon == 252].pivot_table(index=["method", "state"], columns="period", values=["excess_mean", "hit", "base_hit"]).round(3).to_string())
    if a.market in ("both", "US"):
        fr, mk = load_us(ds)
        print(f"US S&P 500 historical constituents: {len(fr)} stocks")
        t = pooled(fr, mk)
        t.to_csv(RESULTS / "broad_forward_US.csv", index=False)
        print(t[t.horizon == 252].pivot_table(index=["method", "state"], columns="period", values=["excess_mean", "hit", "base_hit"]).round(3).to_string())


if __name__ == "__main__":
    main()
