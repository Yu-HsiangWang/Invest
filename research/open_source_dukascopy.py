"""Second-vendor check of the open-source triggers that passed on gold 2005-2025 (docs/RESEARCH.md
section 13): fed into the SECOND position of 高效積極模式, on Dukascopy 15m gold AND silver
2013/7-2026/10 (run in the browser, like research/efficient_dukascopy.py).

Finalists (chosen on the broker gold data before this check): the Turtle 55-bar 1h breakout and the
noise-area momentum breakout (Zarattini et al., 17:00 NY session) - the only two second-position
feeds that beat 高效積極模式 in all three gold periods - plus, for reference, the union of the four
best triggers, the two finalists together, and BOS / LazyBear / Keltner on their own.

For silver, whose 高效積極模式 rules are the normal ones (one position), the same feeds are tested
as a second position next to silver's normal signals (the normal signals themselves never add).

Decision rule, fixed before running: adopt a feed only if, for gold, it does not lower the
efficiency of 高效積極模式 on Dukascopy 2013-2026 nor in the newest period 2025/4-2026/10, and adds
trades; for silver the same against the normal rules.

    res = run_all({"XAUUSD": df_gold, "XAGUSD": df_silver})     # in Pyodide
    python research/open_source_dukascopy.py --kaggle            # same table on the broker gold data
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

import efficient_active as E  # noqa: E402
import efficient_dukascopy as D  # noqa: E402
import intraday_effects as I  # noqa: E402
import open_source_algos as A  # noqa: E402


def union(sigs) -> np.ndarray:
    pos = sum((s > 0).astype(int) for s in sigs)
    neg = sum((s < 0).astype(int) for s in sigs)
    return np.where((pos > 0) & (neg == 0), 1, np.where((neg > 0) & (pos == 0), -1, 0)).astype(np.int64)


def run_add(H: A.Harness, base: np.ndarray, add: np.ndarray, max_open: int = 2, base_adds: bool = True) -> pd.DataFrame:
    """A.Harness.run2, but `base_adds=False` lets only `add` open the second position."""
    if base_adds:
        return H.run2(base, add, max_open)
    bk = E.Book(H.df, H.atr1, H.plan, H.sp, H.sl, H.clock)
    last_exit, live = -10 ** 9, []
    for i in np.flatnonzero((base != 0) | (add != 0)):
        if i >= H.n - 1:
            continue
        live = [p for p in live if p["exit_i"] >= i + 1]
        if not live:
            d = int(base[i])
            if d == 0 or i < last_exit + H.plan.cooldown + 1:
                continue
        else:
            d = int(add[i])
            if len(live) >= max_open or d == 0 or any(p["dir"] != d for p in live):
                continue
        r = bk.open(int(i), d)
        if r is not None:
            r["add"] = bool(live)
            live.append(r)
            last_exit = max(last_exit, int(r["exit_i"]))
    return H._trim(pd.DataFrame(bk.rows).sort_values("sig_i").reset_index(drop=True)) if bk.rows else pd.DataFrame()


def feeds(H: A.Harness) -> dict[str, np.ndarray]:
    trig = {name: H.filtered(H.timed(fn(H))) for name, fn in A.TRIGGERS}
    _, noise = I.sim_noise(H, 17.0, 14, 1)
    noise = H.filtered(H.timed(noise))
    turtle = trig["海龜 55 根 1h 突破"]
    best4 = union([trig[k] for k in trig if k.startswith(("海龜", "結構突破 BOS (擺動 10", "LazyBear 擠壓動能 釋放 15m",
                                                            "Keltner 突破 (EMA20±1.5ATR) 15m"))])
    return {"turtle55_1h": turtle, "noise17": noise, "turtle+noise": union([turtle, noise]), "best4": best4,
            "bos10": trig["結構突破 BOS (擺動 10 根)"], "lazybear15": trig["LazyBear 擠壓動能 釋放 15m"],
            "keltner15": trig["Keltner 突破 (EMA20±1.5ATR) 15m"]}


def run_symbol(df: pd.DataFrame, sym: str) -> dict[str, pd.DataFrame]:
    H = A.Harness(df, sym)
    if sym == "XAUUSD":
        base, base_adds, out = H.sys_sig(18, True), True, {}
        out["base"] = H.run2(base)                              # 高效積極模式
    else:
        base, base_adds, out = H.sys_sig(24, False), False, {}
        out["base"] = H.run1(base)                              # silver normal = its 高效積極模式
    for name, add in feeds(H).items():
        out[name] = run_add(H, base, add, 2, base_adds)
    for k, tr in out.items():
        if not tr.empty:
            out[k] = tr[tr["entry_time"] >= pd.Timestamp(D.START, tz="UTC")].copy()
            out[k]["sym"] = sym
    return out


def run_all(frames: dict[str, pd.DataFrame]) -> tuple[dict, dict]:
    trades = {}
    for sym, df in frames.items():
        D.PRICE[sym] = float(df["close"].iloc[-1])
        trades[sym] = run_symbol(df, sym)
    res = {"how": ("Dukascopy 1-minute BID candles aggregated to 15m, 2013/7-2026/10; 高效積極模式 (gold) / normal rules "
                   "(silver) with each feed allowed to open a second same-direction position; exits, costs and day mode as "
                   "everywhere else. eff = R per year / max drawdown (R)."),
           "periods": [p[0] for p in D.PERIODS] + ["all"],
           "instruments": {s: {k: D.by_period(t) for k, t in v.items()} for s, v in trades.items()}}
    if len(trades) == 2:
        both = {}
        for g in trades["XAUUSD"]:
            for s in ("base", g):
                if s in trades["XAGUSD"]:
                    both[f"gold {g} + silver {s}"] = D.by_period(pd.concat([trades["XAUUSD"][g], trades["XAGUSD"][s]]))
        res["both"] = both
        res["money"] = {f"both gold {g} + silver base": D.money([trades["XAUUSD"][g], trades["XAGUSD"]["base"]],
                                                                {"XAUUSD": 0.01, "XAGUSD": 0.01})
                        for g in ("base", "turtle55_1h", "noise17", "turtle+noise")}
        res["money_how"] = D.MONEY_HOW
    return res, trades


def main():
    if "--kaggle" in sys.argv:          # the same feeds on the broker gold data, 2005-2025/3, for the record
        H = A.Harness(E.load())
        base = H.sys_sig(18, True)
        rows = [A.row("baseline efficient", H.run2(base))]
        for name, add in feeds(H).items():
            rows.append(A.row(f"第 2 張 = {name}", H.run2(base, add)))
        pd.DataFrame(rows).to_csv(ROOT / "results" / "XAUUSD_open_source_feeds.csv", index=False)
        return
    frames = {("XAGUSD" if "XAG" in Path(p).name.upper() else "XAUUSD"): D.load_csv(p) for p in sys.argv[1:]}
    res, _ = run_all(frames)
    (ROOT / "results" / "open_source_dukascopy.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, sort_keys=True),
                                                                 encoding="utf-8")


if __name__ == "__main__":
    main()
