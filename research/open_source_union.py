"""Follow-up to research/open_source_algos.py: is the SECOND-position slot of 高效積極模式 worth
feeding with more triggers (the union of the open-source triggers, or the system's own faster
channels), and does a 3rd slot help? (docs/RESEARCH.md section 12)

    python research/open_source_union.py
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


def main():
    H = A.Harness(E.load())
    eff = H.sys_sig(18, True)
    rows = [A.row("baseline efficient", H.baselines()["efficient"], {"kind": "baseline"})]
    trig = {name: H.filtered(H.timed(fn(H))) for name, fn in A.TRIGGERS}
    def union(sigs):
        pos = sum((s > 0).astype(int) for s in sigs)
        neg = sum((s < 0).astype(int) for s in sigs)
        return np.where((pos > 0) & (neg == 0), 1, np.where((neg > 0) & (pos == 0), -1, 0)).astype(np.int64)
    u_all = union(list(trig.values()))
    u_sys = union([H.sys_sig(h, sq) for h, sq in ((8, False), (12, False), (16, False), (24, False), (12, True), (16, True), (24, True))])
    u_best = union([trig[k] for k in trig if k.startswith(("海龜", "結構突破 BOS (擺動 10", "LazyBear 擠壓動能 釋放 15m", "Keltner 突破 (EMA20±1.5ATR) 15m"))])
    tests = [
        ("第 2 張 = 12h 不收窄訊號", H.sys_sig(12, False), 2),
        ("第 2 張 = 系統各種通道的聯集 (8–24h)", u_sys, 2),
        ("第 2 張 = 14 種開源觸發的聯集", u_all, 2),
        ("第 2 張 = 4 種最好的開源觸發的聯集", u_best, 2),
        ("第 2、3 張 = 14 種開源觸發的聯集 (最多 3 張)", u_all, 3),
        ("第 2、3 張 = 18h 自己的訊號 (最多 3 張)", eff, 3),
        ("第 2 張 = 開源聯集, 第一張也可用聯集 (最多 2 張)", None, 2),
    ]
    for name, add, mo in tests:
        if add is None:
            tr = H.run2(union([eff, u_all]), u_all, mo)
        else:
            tr = H.run2(eff, add, mo)
        rows.append(A.row(name, tr, {"kind": "union", "adds": int(tr.get("add", pd.Series(dtype=bool)).sum()) if not tr.empty else 0}))
    pd.DataFrame(rows).to_csv(ROOT / "results" / "XAUUSD_open_source_union.csv", index=False)


if __name__ == "__main__":
    main()
