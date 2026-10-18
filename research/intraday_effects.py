"""Published intraday effects applied to gold 15m (docs/RESEARCH.md section 12):

  noise   "Beat the Market" intraday momentum (Zarattini, Barbon & Aziz 2024; open-source
          TradingView ports "Out of the Noise", "Concretum Bands"): for every time-of-day slot the
          noise level sigma_t = average over the last 14 sessions of |close_t / open - 1|; the day's
          noise area is [min(open, prev close) x (1 - sigma_t), max(open, prev close) x (1 + sigma_t)];
          go long on a close above the area, short below, exit when the close comes back inside
          (the band is the trailing stop), reverse on the opposite band, flat at the session end.
          Tested with the session anchored at 17:00 New York (gold's trading day), 03:00 NY (London
          open) and 08:30 NY (the US open), checked every 15 or 30 minutes.
  rod     Last-half-hour momentum (Baltussen, Da, Lammers & Martens 2021, 62 futures incl. COMEX gold):
          30 minutes before the close, trade the sign of the return since the previous close and hold
          to the close. Tested with the 13:30 NY COMEX pit close and the 17:00 NY electronic close.

Both are simulated natively (their own exits, fills at the next bar's open, the usual costs) and
the P&L is expressed in the system's risk unit R = 2 x ATR(1h) at entry so the numbers are
comparable with everything else; the breakouts are also fed through the system's exit plan.

    python research/intraday_effects.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

import efficient_active as E  # noqa: E402
import open_source_algos as A  # noqa: E402

OUT = ROOT / "results"


def ny_hours(idx: pd.DatetimeIndex) -> np.ndarray:
    t = idx.tz_convert("America/New_York")
    return (t.hour + t.minute / 60).to_numpy()


def sessions(H: A.Harness, anchor_h: float) -> tuple[np.ndarray, np.ndarray]:
    """Session id per bar for a day that starts at `anchor_h` NY (bar open time) and the slot index."""
    hrs_open = ny_hours(H.df.index)
    # a new session starts at the first bar whose open hour >= anchor on each calendar day (NY)
    days = H.df.index.tz_convert("America/New_York").normalize()
    daynum = days.tz_localize(None).to_numpy().astype("datetime64[D]").astype(np.int64)   # index unit may be us or ns
    sess = daynum - (hrs_open < anchor_h)          # bars before the anchor belong to the previous day's session
    # make ids contiguous
    _, ids = np.unique(sess, return_inverse=True)
    starts = np.r_[True, ids[1:] != ids[:-1]]
    slot = np.zeros(len(ids), dtype=np.int64)
    k = 0
    for i in range(len(ids)):
        k = 0 if starts[i] else k + 1
        slot[i] = k
    return ids, slot


def trades_to_frame(rows: list[dict], H: A.Harness) -> pd.DataFrame:
    tr = pd.DataFrame(rows)
    if tr.empty:
        return tr
    tr["entry_time"] = H.df.index[tr["j0"].to_numpy()]
    tr["exit_time"] = H.df.index[tr["j1"].to_numpy()]
    return tr[tr["entry_time"] >= pd.Timestamp("2005-01-01", tz="UTC")].reset_index(drop=True)


def sim_noise(H: A.Harness, anchor_h: float, lookback: int = 14, step: int = 1, flat_h: float = 16.75,
              last_entry_h: float = 15.0) -> tuple[pd.DataFrame, np.ndarray]:
    """Native simulation of the noise-area strategy. Returns trades and the raw breakout signal array
    (first close outside the area, for the system-exit variant)."""
    ids, slot = sessions(H, anchor_h)
    c, o = H.c, H.o
    n = H.n
    hc = H.clock.ny_close_h
    # per session: open and previous close
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], n]
    nses = len(starts)
    max_slot = int(slot.max()) + 1
    # move table: |close_t / open - 1| per session x slot
    mv = np.full((nses, max_slot), np.nan)
    sopen = np.full(nses, np.nan)
    pclose = np.full(nses, np.nan)
    for s, (a, b) in enumerate(zip(starts, ends)):
        sopen[s] = o[a]
        pclose[s] = c[a - 1] if a > 0 else o[a]
        mv[s, : b - a] = np.abs(c[a:b] / o[a] - 1)
    # sigma per session x slot = mean of the previous `lookback` sessions (nan-aware)
    sig = np.full_like(mv, np.nan)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for s in range(lookback, nses):
                sig[s] = np.nanmean(mv[s - lookback: s], axis=0)
    sess_of = np.searchsorted(starts, np.arange(n), side="right") - 1
    rows = []
    raw = np.zeros(n, dtype=np.int64)
    pos = 0
    j0 = 0
    entry_px = 0.0
    risk = 1.0
    for i in range(1, n - 1):
        s = sess_of[i]
        k = slot[i]
        if np.isnan(sig[s, k]) or np.isnan(H.atr1[i]):
            continue
        ub = max(sopen[s], pclose[s]) * (1 + sig[s, k])
        lb = min(sopen[s], pclose[s]) * (1 - sig[s, k])
        above, below = c[i] > ub, c[i] < lb
        if k % step != 0 and pos == 0:
            continue
        flat_now = hc[i] >= flat_h and hc[i] < 17.0 or (i + 1 < n and ids[i + 1] != ids[i])
        if pos != 0:
            exit_now = flat_now or (pos > 0 and c[i] < ub) or (pos < 0 and c[i] > lb)
            if exit_now:
                j1 = i + 1
                x_px = o[j1] - pos * (H.sp[j1] / 2 + H.sl[j1])
                rows.append({"j0": j0, "j1": j1, "dir": pos, "R": pos * (x_px - entry_px) / risk, "entry": entry_px,
                             "exit": x_px, "reason": "flat" if flat_now else "band", "sig_i": j0 - 1, "exit_i": j1})
                pos = 0
                if flat_now:
                    continue
        if pos == 0 and not flat_now and hc[i] < last_entry_h and (above or below):
            d = 1 if above else -1
            # the paper's reversal: an exit on the opposite band is immediately a new position; here a
            # fresh entry on the next bar is the same thing because the exit was filled at this open
            raw[i] = d if raw[i - 1] != d else 0
            pos, j0 = d, i + 1
            entry_px = o[j0] + d * (H.sp[j0] / 2 + H.sl[j0])
            risk = 2.0 * H.atr1[i]
    # the raw array for the system plan: first close outside the area each day & direction
    raw2 = np.zeros(n, dtype=np.int64)
    for s, (a, b) in enumerate(zip(starts, ends)):
        if s < lookback:
            continue
        done = set()
        for i in range(a, b):
            k = slot[i]
            if np.isnan(sig[s, k]):
                continue
            ub = max(sopen[s], pclose[s]) * (1 + sig[s, k])
            lb = min(sopen[s], pclose[s]) * (1 - sig[s, k])
            if c[i] > ub and 1 not in done:
                raw2[i] = 1
                done.add(1)
            elif c[i] < lb and -1 not in done:
                raw2[i] = -1
                done.add(-1)
    return trades_to_frame(rows, H), raw2


def sim_rod(H: A.Harness, close_h: float, minutes: int = 30, min_move: float = 0.0) -> pd.DataFrame:
    """Native simulation of the last-half-hour momentum: at close_h - minutes, go with the sign of
    the return since the previous session's close, exit at close_h."""
    hc = H.clock.ny_close_h
    c, o = H.c, H.o
    n = H.n
    bars = minutes // 15
    rows = []
    prev_close = np.nan
    prev_day = -1
    day = H.clock.session
    i_sig = None
    # previous close = close of the bar that closes at close_h on the previous trading day
    last_close_by_day = {}
    for i in range(n):
        if abs(hc[i] - close_h) < 1e-9:
            last_close_by_day[day[i]] = c[i]
    for i in range(1, n - bars - 1):
        if abs(hc[i] - (close_h - minutes / 60)) > 1e-9:
            continue
        d_prev = day[i] - 1
        # the last trading day with a close (skip weekends / holidays)
        pc = None
        for back in range(1, 5):
            if day[i] - back in last_close_by_day:
                pc = last_close_by_day[day[i] - back]
                break
        if pc is None:
            continue
        rod = c[i] / pc - 1
        if abs(rod) < min_move:
            continue
        d = 1 if rod > 0 else -1
        j0, j1 = i + 1, i + bars
        if abs(hc[j1] - close_h) > 1e-9:
            continue
        e_px = o[j0] + d * (H.sp[j0] / 2 + H.sl[j0])
        x_px = c[j1] - d * (H.sp[j1] / 2 + H.sl[j1])
        rows.append({"j0": j0, "j1": j1, "dir": d, "R": d * (x_px - e_px) / (2.0 * H.atr1[i]), "pct": d * (x_px / e_px - 1) * 100,
                     "entry": e_px, "exit": x_px, "reason": "time", "sig_i": i, "exit_i": j1})
    return trades_to_frame(rows, H)


def main():
    H = A.Harness(E.load())
    rows = [A.row("baseline efficient", H.baselines()["efficient"], {"kind": "baseline"})]
    eff = H.sys_sig(18, True)
    print("== noise-area intraday momentum (Zarattini et al.)")
    for anchor, label in ((17.0, "17:00 NY 整個交易日"), (3.0, "03:00 NY 倫敦開盤"), (8.5, "08:30 NY 美盤開盤")):
        for step, sl in ((1, "每 15 分檢查"), (2, "每 30 分檢查")):
            tr, raw = sim_noise(H, anchor, 14, step)
            rows.append(A.row(f"噪音區間動能 {label}, {sl} | 原論文出場", tr, {"kind": "noise_native"}))
            if step == 1:
                flt = H.filtered(H.timed(raw))
                rows.append(A.row(f"噪音區間動能 {label} | 系統出場", H.run1(H.timed(raw)), {"kind": "noise_sys"}))
                rows.append(A.row(f"噪音區間動能 {label} | 加系統過濾", H.run1(flt), {"kind": "noise_filtered"}))
                rows.append(A.row(f"噪音區間動能 {label} | 當高效積極的第 2 張", H.run2(eff, flt), {"kind": "noise_addon"}))
    print("== last-half-hour momentum (Baltussen et al.)")
    for close_h, label in ((13.5, "13:30 NY COMEX 場內收盤"), (17.0, "17:00 NY 電子盤收盤")):
        for minutes in (30, 60):
            for mm in (0.0, 0.003, 0.006):
                tr = sim_rod(H, close_h, minutes, mm)
                tag = f"收盤前 {minutes} 分動能 {label}" + (f", |當日報酬| ≥ {mm * 100:.1f}%" if mm else "")
                extra = {"kind": "rod", "avg_pct": round(float(tr["pct"].mean()), 4) if not tr.empty else None}
                rows.append(A.row(tag, tr, extra))
    pd.DataFrame(rows).to_csv(OUT / "XAUUSD_intraday_effects.csv", index=False)


if __name__ == "__main__":
    main()
