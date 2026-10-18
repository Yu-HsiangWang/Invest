"""Two ideas from the literature survey not covered by research/open_source_algos.py
(docs/RESEARCH.md section 13):

  rvol     time-of-day relative tick volume: volume of the signal bar / median volume of the same
           15m slot over the last 20 days (removes the London/NY-open profile that a plain
           20-bar volume ratio picks up). Used as a filter on 12h/18h breakouts and as the
           second-position feed of 高效積極模式.
  bigmove  large-move-day continuation (Caporale & Plastun 2021, gold): the first 15m close of the
           day (from 17:00 NY) where |close / session open - 1| > k x the 20-day mean of
           |open-to-close|, traded with the move; system filters / exits.

    python research/volume_bigmove.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import efficient_active as E  # noqa: E402
import intraday_effects as I  # noqa: E402
import open_source_algos as A  # noqa: E402


def rvol(H: A.Harness, days: int = 20) -> np.ndarray:
    idx = H.df.index
    slot = (idx.hour * 4 + idx.minute // 15).to_numpy()
    v = pd.Series(H.v, index=idx)
    out = np.full(H.n, np.nan)
    for s in np.unique(slot):
        m = slot == s
        vs = v[m]
        med = vs.rolling(days, min_periods=10).median().shift(1)
        out[m] = (vs / med).to_numpy()
    return out


def bigmove(H: A.Harness, k: float) -> np.ndarray:
    ids, _ = I.sessions(H, 17.0)
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    sess_of = np.searchsorted(starts, np.arange(H.n), side="right") - 1
    sopen = H.o[starts][sess_of]
    ends = np.r_[starts[1:] - 1, H.n - 1]
    oc = np.abs(H.c[ends] / H.o[starts] - 1)
    sig20 = pd.Series(oc).rolling(20).mean().shift(1).to_numpy()[sess_of]
    mv = H.c / sopen - 1
    cond_up, cond_dn = mv > k * sig20, mv < -k * sig20
    out = np.zeros(H.n, dtype=np.int64)
    seen = set()
    for i in np.flatnonzero(cond_up | cond_dn):
        s = sess_of[i]
        if s in seen:
            continue
        seen.add(s)
        out[i] = 1 if cond_up[i] else -1
    return out


def main():
    H = A.Harness(E.load())
    eff = H.sys_sig(18, True)
    rows = [A.row("baseline efficient", H.run2(eff))]
    rv = rvol(H)
    print("rvol median", np.nanmedian(rv), "corr with plain ratio", np.corrcoef(np.nan_to_num(rv), np.nan_to_num(A.volume_ratio(H, 20)))[0, 1])
    s12 = H.sys_sig(12, False)
    for thr in (1.2, 1.5, 2.0):
        m = np.nan_to_num(rv) >= thr
        rows.append(A.row(f"12h 不收窄 + 同時段相對量 ≥ {thr} | 最多 2 張", H.run2(np.where(m, s12, 0))))
        rows.append(A.row(f"12h 不收窄 + 同時段相對量 ≥ {thr} | 當高效積極的第 2 張", H.run2(eff, np.where(m, s12, 0))))
        rows.append(A.row(f"18h 收窄 + 同時段相對量 ≥ {thr} | 最多 2 張", H.run2(np.where(m, eff, 0))))
    for k in (1.0, 1.5):
        raw = H.timed(bigmove(H, k))
        rows.append(A.row(f"大波動日順勢 k={k} | 原始", H.run1(raw)))
        rows.append(A.row(f"大波動日順勢 k={k} | 加系統過濾", H.run1(H.filtered(raw))))
        rows.append(A.row(f"大波動日順勢 k={k} | 當高效積極的第 2 張", H.run2(eff, H.filtered(raw))))
    pd.DataFrame(rows).to_csv(ROOT / "results" / "XAUUSD_volume_bigmove.csv", index=False)


if __name__ == "__main__":
    main()
