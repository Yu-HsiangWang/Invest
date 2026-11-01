"""Things not yet tested that could raise the profit per trade or the trade count of 高效積極模式
(docs/RESEARCH.md section 12.4): direction asymmetry, the day's range at the signal, and the
entry mechanics themselves (stop order at the level instead of waiting for the 15m close;
a limit order on the retest of the level).

Everything runs through research/open_source_algos.Harness: the 18h + squeeze signals with up
to 2 same-direction positions, the usual exits, costs and day mode; dev 2005-2016 / val
2017-2020 / test 2021-2025/3 with the efficiency yardstick (R per year / max drawdown).

    python research/entry_mechanics.py
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


def period_of(ts: pd.Series) -> pd.Series:
    out = pd.Series("test", index=ts.index)
    out[ts < pd.Timestamp("2021-01-01", tz="UTC")] = "val"
    out[ts < pd.Timestamp("2017-01-01", tz="UTC")] = "dev"
    return out


def split_table(name: str, tr: pd.DataFrame, by: str) -> list[dict]:
    rows = []
    tr = tr.copy()
    tr["period"] = period_of(tr["entry_time"])
    for per in ("dev", "val", "test", "all"):
        X = tr if per == "all" else tr[tr["period"] == per]
        for g, Y in X.groupby(by):
            if len(Y) >= 5:
                r = Y["R"].to_numpy()
                rows.append({"what": name, "period": per, "group": str(g), "n": len(Y), "avgR": round(float(r.mean()), 3),
                             "win%": round(100 * float((r > 0).mean()), 1), "totR": round(float(r.sum()), 1)})
    return rows


def run_with_fills(H: A.Harness, sig_i: np.ndarray, dirs: np.ndarray, fills: np.ndarray, max_open: int = 2) -> pd.DataFrame:
    """Positions opened at bar sig_i+1 at price `fills` (instead of that bar's open): the simulator
    reads the entry from the open, so a copy of the data with those opens replaced is used.
    Only entry bars change, so exits of other positions are untouched except in the rare case
    where an exit falls on another position's entry bar."""
    df = H.df.copy()
    o = df["open"].to_numpy(float).copy()
    o[sig_i + 1] = fills
    df["open"] = o
    sig = np.zeros(H.n, dtype=np.int64)
    sig[sig_i] = dirs
    tr = E.run_variant(df, sig, sig, H.atr1, H.plan, H.sp, H.sl, H.clock, "any", max_open, True)
    return H._trim(tr)


def stop_entry_signals(H: A.Harness, hours: int = 18, squeeze: bool = True, buf: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Buy-stop / sell-stop resting `buf` ATR(1h) beyond the `hours` channel edge: filled on the
    first bar whose high (low) reaches the level while the trend / ADX / extension / timing rules
    (and the squeeze filter) held at the previous bar's close, and the previous close was still
    inside. Fill = max(open, level) for longs (the simulator adds spread and slippage)."""
    from dataclasses import replace
    from goldsignal import strategy as S
    p = replace(H.p, breakout_hours=int(hours), squeeze_bars=S.SQUEEZE_BARS if squeeze else 0)
    f = H.f if hours == H.p.breakout_hours else S.compute_features(H.df, params=p)
    t = S.rule_table(H.df, f, p, None)
    hi = f["chan_hi"].to_numpy(float) + buf * H.atr1
    lo = f["chan_lo"].to_numpy(float) - buf * H.atr1
    c_prev = np.r_[np.nan, H.c[:-1]]
    setup = np.where(t["long_setup"], 1, np.where(t["short_setup"], -1, 0))
    setup_prev = np.r_[0, setup[:-1]]
    sq_prev = np.r_[False, t["squeeze_ok"].to_numpy(bool)[:-1]]
    timing = t["timing_ok"].to_numpy(bool)
    long_touch = (H.h >= hi) & (c_prev <= np.r_[np.nan, hi[:-1]]) & (setup_prev == 1) & sq_prev & timing
    short_touch = (H.l <= lo) & (c_prev >= np.r_[np.nan, lo[:-1]]) & (setup_prev == -1) & sq_prev & timing
    long_touch, short_touch = np.nan_to_num(long_touch), np.nan_to_num(short_touch)
    idx = np.flatnonzero(long_touch | short_touch)
    idx = idx[(idx >= 1) & (idx < H.n - 1)]
    dirs = np.where(long_touch[idx], 1, -1)
    fills = np.where(dirs > 0, np.maximum(H.o[idx], hi[idx]), np.minimum(H.o[idx], lo[idx]))
    return idx - 1, dirs, fills      # signal bar = the bar before the fill bar


def retest_signals(H: A.Harness, base: np.ndarray, k_bars: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """After a normal close-breakout signal at bar t, a limit order at the breakout level (the
    channel edge as of bar t) for the next k bars; filled on the first bar whose low (high)
    reaches it, at min(open, level) for longs. Signals without a retest are skipped."""
    hi, lo = H.f["chan_hi"].to_numpy(float), H.f["chan_lo"].to_numpy(float)
    sig_i, dirs, fills = [], [], []
    for t in np.flatnonzero(base):
        d = int(base[t])
        lvl = hi[t] if d > 0 else lo[t]
        for u in range(t + 1, min(t + 1 + k_bars, H.n - 1)):
            if (d > 0 and H.l[u] <= lvl) or (d < 0 and H.h[u] >= lvl):
                sig_i.append(u - 1); dirs.append(d)
                fills.append(min(H.o[u], lvl) if d > 0 else max(H.o[u], lvl))
                break
    return np.asarray(sig_i), np.asarray(dirs), np.asarray(fills)


def main():
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    H = A.Harness(E.load())
    e = H.sys_sig(18, True)
    base_tr = H.run2(e)
    rows_bt = [A.row("baseline 高效積極 (18h+收窄, 同向最多 2 張)", base_tr)]
    splits = []

    # 1. direction and the 200-day trend
    h1 = H.h1["close"]
    ema200d = H.h1_to_m15(h1.ewm(span=24 * 200, adjust=False).mean())
    above = H.c > ema200d
    tr = base_tr.copy(); tr["dir_word"] = np.where(tr["dir"] > 0, "多", "空")
    tr["regime"] = np.where(above[tr["sig_i"]], "200日線上", "200日線下")
    tr["dir_regime"] = tr["dir_word"] + "・" + tr["regime"]
    splits += split_table("方向", tr, "dir_word") + split_table("方向×200日線", tr, "dir_regime")
    rows_bt.append(A.row("只做多", H.run2(np.where(e > 0, e, 0))))
    rows_bt.append(A.row("多單只在 200 日線上、空單只在 200 日線下", H.run2(np.where(((e > 0) & above) | ((e < 0) & ~above), e, 0))))
    rows_bt.append(A.row("多單不限、空單只在 200 日線下", H.run2(np.where((e > 0) | ((e < 0) & ~above), e, 0))))

    # 2. the day's range so far at the signal (session from 17:00 NY), in daily ATR units
    sess = H.clock.session
    g = pd.Series(sess)
    sh = pd.Series(H.h).groupby(g).cummax().to_numpy()
    sl_ = pd.Series(H.l).groupby(g).cummin().to_numpy()
    rng_atr = (sh - sl_) / H.f["d1_atr"].to_numpy(float)
    hrs = H.clock.sess_h
    tr = base_tr.copy()
    tr["range"] = pd.cut(rng_atr[tr["sig_i"]], [0, 0.5, 1.0, 1.5, 9], labels=["<0.5", "0.5–1", "1–1.5", ">1.5"])
    tr["hours"] = pd.cut(hrs[tr["sig_i"]], [0, 6, 12, 16, 24], labels=["0–6h", "6–12h", "12–16h", "16–24h"])
    splits += split_table("訊號時當日已走幅度 (日ATR)", tr, "range") + split_table("訊號時距開盤幾小時", tr, "hours")
    for k in (1.0, 1.5):
        rows_bt.append(A.row(f"當日已走幅度 ≤ {k} 日ATR 才進場", H.run2(np.where(np.nan_to_num(rng_atr, nan=9) <= k, e, 0))))

    # 3. chase guard: skip if the next open is already > 0.5 ATR past the signal close
    nxt = np.r_[H.o[1:], H.o[-1]]
    chase = np.abs(nxt - H.c) > 0.5 * H.atr1
    rows_bt.append(A.row("下一根開盤已超過 0.5 ATR 就不追", H.run2(np.where(chase, 0, e))))

    # 4. entry mechanics
    si, d, fl = stop_entry_signals(H)
    rows_bt.append(A.row(f"碰到 18h 高低點就進場 (停損單, {len(si)} 次)", run_with_fills(H, si, d, fl)))
    for k in (8, 16, 32):
        si, d, fl = retest_signals(H, e, k)
        rows_bt.append(A.row(f"收盤突破後 {k // 4}h 內回測到突破價才進 (限價, {len(si)} 次)", run_with_fills(H, si, d, fl)))

    bt = pd.DataFrame(rows_bt)
    sp = pd.DataFrame(splits)
    bt.to_csv(OUT / "XAUUSD_entry_mechanics.csv", index=False)
    sp.to_csv(OUT / "XAUUSD_entry_splits.csv", index=False)
    print(sp.to_string(index=False))


if __name__ == "__main__" and "--followup" not in sys.argv:
    main()


def followup():
    """The stop-order entry in more detail: channel length, squeeze, positions, level buffer,
    extra slippage on stop fills; per-period numbers."""
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    H = A.Harness(E.load())
    rows = [A.row("baseline 高效積極 (收盤突破, 18h+收窄, 最多 2 張)", H.run2(H.sys_sig(18, True))),
            A.row("baseline 一般設定 (收盤突破, 24h+收窄, 1 張)", H.run1(H.sys_sig(24, True))),
            A.row("baseline 積極模式 (12h, 不收窄, 2 張)", H.baselines()["active"])]
    for hours, sq, mo, buf in [(18, True, 2, 0.0), (18, True, 1, 0.0), (24, True, 1, 0.0), (24, True, 2, 0.0), (18, False, 2, 0.0),
                               (12, True, 2, 0.0), (18, True, 2, 0.1), (18, True, 2, 0.25), (24, True, 2, 0.1)]:
        si, d, fl = stop_entry_signals(H, hours, sq, buf)
        rows.append(A.row(f"停損單進場 {hours}h{'+收窄' if sq else ''}，最多 {mo} 張，緩衝 {buf} ATR ({len(si)} 次)",
                          run_with_fills(H, si, d, fl, mo)))
    si, d, fl = stop_entry_signals(H, 18, True, 0.0)
    for slip_x in (2, 3):
        rows.append(A.row(f"停損單進場 18h+收窄，最多 2 張，進場滑價 ×{slip_x}", run_with_fills(H, si, d, fl + d * (slip_x - 1) * H.sl[si + 1], 2)))
    bt = pd.DataFrame(rows)
    cols = ["name", "all_per_week", "all_avgR", "all_R_yr", "all_maxDD", "all_eff"] + [f"{p}_{k}" for p in ("dev", "val", "test") for k in ("R_yr", "maxDD", "eff")]
    print(bt[cols].to_string(index=False))
    bt.to_csv(OUT / "XAUUSD_entry_stop_followup.csv", index=False)


if __name__ == "__main__" and "--followup" in sys.argv:
    followup()
