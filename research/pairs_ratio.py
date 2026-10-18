"""Gold / silver ratio pairs trading (statistical arbitrage), docs/RESEARCH.md section 13.

The classic open-source "pairs" algorithm: z-score of the log gold/silver ratio against its rolling
mean; when the ratio is stretched, sell the expensive metal and buy the cheap one in equal dollar
amounts, close when the ratio comes back. Market-neutral, so it would add trades that do not
depend on the trend system - if it pays after costs.

Data: Dukascopy hourly BID candles 2013-2026 for XAU-USD and XAG-USD (downloaded in the browser,
see research/efficient_dukascopy.py for why). Costs per leg: spread (gold 0.012 %, silver 0.06 %)
plus slippage on both fills (gold 0.005 %, silver 0.01 % per fill); overnight fees on both legs
(Mitrade: long 0.0168 %, short 0.014 % per night). P&L is in % of the notional of one leg
(the legs are equal in dollars) and in dollars for 0.01 lot of each (gold 1 oz, silver 50 oz).

    python research/pairs_ratio.py data/dukascopy_h1_XAUUSD.csv data/dukascopy_h1_XAGUSD.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

COST_RT = (0.012 + 2 * 0.005 + 0.06 + 2 * 0.01) / 100       # round trip, both legs, fraction of notional
SWAP_NIGHT = (0.0168 + 0.014) / 100                          # both legs, per night
PERIODS = [("2013/7–2016", "2013-07-01", "2016-12-31"), ("2017–2020", "2017-01-01", "2020-12-31"),
           ("2021–2025/3", "2021-01-01", "2025-03-31"), ("2025/4–2026/10", "2025-04-01", "2026-10-09")]


def ny_hour(idx: pd.DatetimeIndex) -> np.ndarray:
    return (idx.tz_convert("America/New_York").hour + idx.tz_convert("America/New_York").minute / 60).to_numpy()


def ny_day(idx: pd.DatetimeIndex) -> np.ndarray:
    """Trading-day number on the 17:00 New York clock."""
    t = idx.tz_convert("America/New_York") + pd.Timedelta(hours=7)
    return (t.normalize().astype("int64") // 86_400_000_000_000).to_numpy()


def run(g: pd.Series, s: pd.Series, window: int = 240, entry: float = 2.0, exit_z: float = 0.0, stop_z: float = 4.0,
        max_hours: int | None = None, day_mode: bool = False, eod_hour: float = 16.0) -> pd.DataFrame:
    """g, s: aligned hourly closes (UTC index). Signals at the bar close, fills at the next bar's
    close price (conservative: no intra-bar fills), costs as above."""
    x = np.log(g.to_numpy(float)) - np.log(s.to_numpy(float))
    xs = pd.Series(x)
    z = ((xs - xs.rolling(window).mean()) / xs.rolling(window).std(ddof=0)).to_numpy()
    idx = g.index
    hour = ny_hour(idx)
    day = ny_day(idx)
    gv, sv = g.to_numpy(float), s.to_numpy(float)
    n = len(x)
    rows = []
    pos = 0          # +1 = long ratio (long gold / short silver), -1 = short ratio
    i0 = 0
    for i in range(window + 1, n - 1):
        if pos == 0:
            if np.isnan(z[i]):
                continue
            if day_mode and hour[i] >= eod_hour - 1:          # no new pairs into the close
                continue
            if z[i] >= entry and z[i - 1] < entry:
                pos, i0 = -1, i + 1
            elif z[i] <= -entry and z[i - 1] > -entry:
                pos, i0 = 1, i + 1
            continue
        # open: exit rules checked at bar close, filled at next bar close
        held = i - i0
        pnl_now = pos * (x[i] - x[i0])
        why = None
        if pos * z[i] >= -exit_z:            # the ratio came back (to the mean, or within exit_z of it)
            why = "target"
        elif -pos * z[i] >= stop_z:          # it kept diverging
            why = "stop"
        elif max_hours and held >= max_hours:
            why = "time"
        elif day_mode and hour[i] >= eod_hour and day[i] == day[i0] and pnl_now < 0:
            why = "eod"
        elif day_mode and hour[i] >= eod_hour and day[i] != day[i0]:
            why = "eod"
        if why is None:
            continue
        j = i + 1
        ret = pos * (x[j] - x[i0]) - COST_RT
        nights = int(day[j] - day[i0])
        ret -= nights * SWAP_NIGHT
        usd = pos * ((gv[j] - gv[i0]) * 1.0 - (sv[j] - sv[i0]) * 50.0) - COST_RT / 2 * (gv[i0] + 50 * sv[i0]) \
            - nights * SWAP_NIGHT / 2 * (gv[i0] + 50 * sv[i0])
        rows.append({"entry_time": idx[i0], "exit_time": idx[j], "dir": pos, "hours": j - i0, "nights": nights,
                     "ret_pct": ret * 100, "usd": usd, "reason": why, "z_in": z[i0 - 1]})
        pos = 0
    return pd.DataFrame(rows)


def stats(tr: pd.DataFrame, a: str, b: str) -> dict:
    m = (tr["entry_time"] >= pd.Timestamp(a, tz="UTC")) & (tr["entry_time"] < pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1))
    x = tr[m]
    yrs = (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25
    if x.empty:
        return {"n": 0}
    r = x["ret_pct"].to_numpy()
    eq = np.cumsum(r)
    dd = float(np.max(np.maximum.accumulate(np.r_[0.0, eq])[1:] - eq))
    u = x["usd"].to_numpy()
    equ = np.cumsum(u)
    ddu = float(np.max(np.maximum.accumulate(np.r_[0.0, equ])[1:] - equ))
    return {"n": int(len(r)), "per_week": round(len(r) / (yrs * 52.18), 2), "win%": round(100 * float((r > 0).mean()), 1),
            "avg_pct": round(float(r.mean()), 3), "tot_pct": round(float(r.sum()), 1), "maxDD_pct": round(dd, 1),
            "sharpe": round(float(r.mean() / r.std(ddof=0) * np.sqrt(len(r) / yrs)), 2) if r.std() > 0 else None,
            "avg_usd": round(float(u.mean()), 1), "tot_usd": round(float(u.sum())), "maxDD_usd": round(ddu),
            "avg_hours": round(float(x["hours"].mean()), 1)}


def evaluate(tr: pd.DataFrame) -> dict:
    out = {lab: stats(tr, a, b) for lab, a, b in PERIODS}
    out["all"] = stats(tr, PERIODS[0][1], PERIODS[-1][2])
    return out


GRID = [  # window, entry, exit, stop, max_hours, day_mode
    (120, 2.0, 0.0, 4.0, None, False), (240, 2.0, 0.0, 4.0, None, False), (480, 2.0, 0.0, 4.0, None, False),
    (1000, 2.0, 0.0, 4.0, None, False),
    (240, 1.5, 0.0, 3.5, None, False), (240, 2.5, 0.0, 4.5, None, False),
    (240, 2.0, 0.5, 4.0, None, False), (240, 2.0, 0.0, 4.0, 48, False), (240, 2.0, 0.0, 4.0, 120, False),
    (120, 2.0, 0.0, 4.0, None, True), (240, 2.0, 0.0, 4.0, None, True), (480, 2.0, 0.0, 4.0, None, True),
    (24, 2.0, 0.0, 4.0, None, True), (48, 2.0, 0.0, 4.0, None, True), (48, 1.5, 0.0, 3.5, None, True),
]


def run_all(g: pd.Series, s: pd.Series) -> dict:
    res = {}
    for w, e, ex, st, mh, dm in GRID:
        tr = run(g, s, w, e, ex, st, mh, dm)
        name = f"window {w}h, entry |z|>{e}, exit z→{ex}, stop {st}" + (f", max {mh}h" if mh else "") + (", 當日平倉" if dm else "")
        res[name] = evaluate(tr) if not tr.empty else {"all": {"n": 0}}
        a = res[name]["all"]
        print(f"  {name:60s} {a.get('per_week', 0)}/wk avg {a.get('avg_pct', 0):+.3f}% win {a.get('win%')}% "
              f"tot {a.get('tot_pct')}% DD {a.get('maxDD_pct')}% sharpe {a.get('sharpe')} | ${a.get('tot_usd')} DD ${a.get('maxDD_usd')}", flush=True)
    return res


def load_h1(path: str) -> pd.Series:
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    return df["close"].astype(float)


def main():
    g, s = load_h1(sys.argv[1]), load_h1(sys.argv[2])
    both = pd.concat([g.rename("g"), s.rename("s")], axis=1).dropna()
    res = run_all(both["g"], both["s"])
    (ROOT / "results" / "pairs_ratio_h1.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
