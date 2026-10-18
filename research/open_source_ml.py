"""Machine-learning candidates for research/open_source_algos.py (docs/RESEARCH.md section 13):

  knn   Lorentzian Classification (jdehorty, TradingView's most-used ML script): kNN over the last
        2000 bars with a Lorentzian distance on RSI / WaveTrend / CCI / ADX features, predicting the
        4-bar direction; entries on the signal flip, exits after 4 bars (the script's default) or
        with the system's trailing stop. Both the script's quirky neighbour selection and a true kNN.
  ml    A gradient-boosting direction model (what most "XAUUSD bot" repositories do, done properly):
        ~30 backward-looking features, target = the next 4-hour move in ATR units, retrained every
        year on all earlier data (walk-forward), trade when the forecast is in the top / bottom
        decile of the training forecasts.
  meta  Meta-labeling (Lopez de Prado, Advances in Financial ML / mlfinlab): a pool of candidate
        entries from the system's channels (8-24 h, with and without the squeeze filter) and the
        open-source triggers; every candidate is simulated alone to get its R; a walk-forward
        gradient-boosting model learns which candidates pay, and only candidates with a high enough
        probability are traded (up to 2 positions, same direction). This is the principled way to
        "take more signals but keep the edge".

All models use only information available at the signal bar's close; walk-forward windows are
purged so no label overlaps the test year. Costs, exits and periods as in open_source_algos.py.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

import efficient_active as E  # noqa: E402
import open_source_algos as A  # noqa: E402
from goldsignal import indicators as ind  # noqa: E402
from goldsignal import strategy as S  # noqa: E402

OUT = ROOT / "results"
YEARS = list(range(2009, 2026))          # walk-forward test years (training starts in 2004)


# ----------------------------------------------------------------------------- features

def feature_table(H: A.Harness) -> pd.DataFrame:
    """Backward-looking features at every 15m bar close."""
    df, f = H.df, H.f
    c = df["close"]
    a15 = f["m15_atr"].replace(0, np.nan)
    X = pd.DataFrame(index=df.index)
    for k in (1, 4, 16, 96):
        X[f"ret{k}"] = (c - c.shift(k)) / a15
    X["d_ema20_15"] = (c - f["m15_ema20"]) / a15
    X["d_ema50_15"] = (c - f["m15_ema50"]) / a15
    X["d_ema20_h1"] = (c - f["h1_ema20"]) / f["h1_atr"]
    X["d_ema50_h1"] = (c - f["h1_ema50"]) / f["h1_atr"]
    X["d_ema20_d1"] = f["d1_ext"]
    X["d_ema50_d1"] = (f["d1_close"] - f["d1_ema50"]) / f["d1_atr"]
    for tf in ("m15", "h1", "h4", "d1"):
        X[f"rsi_{tf}"] = f[f"{tf}_rsi"]
    for tf in ("h1", "h4", "d1"):
        X[f"adx_{tf}"] = f[f"{tf}_adx"]
    X["bw_pct"] = f["m15_bw_pct"]
    X["bw_min16"] = f["m15_bw_pct"].shift(1).rolling(16, min_periods=1).min()
    X["atr_rel"] = f["m15_atr"] / f["m15_atr"].rolling(480, min_periods=50).mean()
    X["atr_pct_h1"] = f["h1_atr"] / c * 100
    X["trend_d1"], X["trend_h4"], X["trend_h1"] = f["trend_d1"], f["trend_h4"], f["trend_h1"]
    X["hour"], X["dow"] = f["ny_close_h"], f["ny_dow"]
    X["cpos_long"] = f["cpos_long"]
    X["d_chan_hi"] = (c - f["chan_hi"]) / a15
    X["d_chan_lo"] = (c - f["chan_lo"]) / a15
    X["vol_ratio"] = A.volume_ratio(H, 20)
    X["er48"] = A.efficiency_ratio(H, 48)
    # today's range so far (17:00 NY sessions)
    sess = pd.Series(H.clock.session, index=df.index)
    hi = df["high"].groupby(sess).cummax()
    lo = df["low"].groupby(sess).cummin()
    X["day_range"] = (hi - lo) / f["d1_atr"]
    X["day_pos"] = ((c - lo) / (hi - lo).replace(0, np.nan))
    X["hurst"] = A.hurst_series(H)
    return X.replace([np.inf, -np.inf], np.nan)


def year_of(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.year.to_numpy()


def gb(regress: bool, seed: int = 7):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    kw = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0,
              random_state=seed)
    return HistGradientBoostingRegressor(**kw) if regress else HistGradientBoostingClassifier(**kw)


# ----------------------------------------------------------------------------- ml: direction model

def part_ml(H: A.Harness) -> pd.DataFrame:
    X = feature_table(H)
    cols = list(X.columns)
    Xv = X.to_numpy(float)
    yrs = year_of(H.df.index)
    c = H.c
    a15 = H.f["m15_atr"].to_numpy(float)
    rows = [A.row("baseline efficient", H.baselines()["efficient"], {"kind": "baseline"})]
    for horizon in (4, 16, 96):
        y = (np.r_[c[horizon:], np.full(horizon, np.nan)] - c) / a15
        pred = np.full(H.n, np.nan)
        t0 = time.time()
        for Y in YEARS:
            tr_mask = (yrs < Y) & np.isfinite(y) & np.isfinite(Xv).all(axis=1)
            # purge: a training label must be known before the test year starts
            tr_mask &= np.r_[np.arange(H.n) + horizon < np.searchsorted(yrs, Y)]
            te_mask = yrs == Y
            idx = np.flatnonzero(tr_mask)[::2]                     # every 2nd bar is plenty
            m = gb(True).fit(Xv[idx], y[idx])
            tr_pred = m.predict(Xv[idx[::5]])
            lo, hi = np.percentile(tr_pred, 10), np.percentile(tr_pred, 90)
            te = np.flatnonzero(te_mask & np.isfinite(Xv).all(axis=1))
            p = m.predict(Xv[te])
            pred[te] = np.where(p >= hi, 1, np.where(p <= lo, -1, 0))
        sig = np.nan_to_num(pred, nan=0).astype(np.int64)
        sig = np.where(A.first_of_run(sig == 1), 1, np.where(A.first_of_run(sig == -1), -1, 0))
        raw = H.timed(sig)
        flt = H.filtered(raw)
        tag = f"梯度提升方向模型 (預測 {horizon * 15 // 60 if horizon >= 4 else 15}{'h' if horizon >= 4 else 'm'})"
        rows.append(A.row(f"{tag} | 原始", H.run1(raw), {"kind": "ml_raw", "signals": int((raw != 0).sum()), "secs": round(time.time() - t0)}))
        rows.append(A.row(f"{tag} | 加系統過濾", H.run1(flt), {"kind": "ml_filtered", "signals": int((flt != 0).sum())}))
        rows.append(A.row(f"{tag} | 當高效積極的第 2 張", H.run2(H.sys_sig(18, True), flt), {"kind": "ml_addon"}))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "XAUUSD_open_source_ml.csv", index=False)
    return res


# ----------------------------------------------------------------------------- knn: Lorentzian classification

def wavetrend(df: pd.DataFrame, n1=10, n2=11) -> pd.Series:
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3
    esa = ind.ema(hlc3, n1)
    d = ind.ema((hlc3 - esa).abs(), n1)
    ci = (hlc3 - esa) / (0.015 * d.replace(0, np.nan))
    return ind.ema(ci, n2)


def cci(df: pd.DataFrame, n=20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(n).mean()
    md = (tp - ma).abs().rolling(n).mean()
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def normalize_running(s: pd.Series) -> np.ndarray:
    """jdehorty's 'normalize': min-max over everything seen so far."""
    x = s.to_numpy(float)
    mn = pd.Series(x).cummin().to_numpy()
    mx = pd.Series(x).cummax().to_numpy()
    return (x - mn) / np.where(mx - mn > 0, mx - mn, 1.0)


def knn_features(df: pd.DataFrame) -> np.ndarray:
    f1 = ind.rsi(df["close"], 14) / 100.0
    f2 = normalize_running(wavetrend(df, 10, 11))
    f3 = normalize_running(cci(df, 20))
    f4 = ind.adx(df, 20)["adx"] / 100.0
    f5 = ind.rsi(df["close"], 9) / 100.0
    return np.column_stack([f1.to_numpy(), f2, f3, f4.to_numpy(), f5.to_numpy()])


def kernel_rq(c: np.ndarray, h=8.0, r=8.0, x=25) -> np.ndarray:
    """Nadaraya-Watson rational-quadratic kernel estimate using the last x bars (non-repainting)."""
    k = np.arange(x)
    w = (1 + k ** 2 / (2 * r * h ** 2)) ** (-r)
    out = np.full(len(c), np.nan)
    for i in range(x, len(c)):
        out[i] = (w * c[i - k]).sum() / w.sum()
    return out


try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*a, **k):
        return (lambda f: f) if not a or not callable(a[0]) else a[0]


@njit(cache=True)
def _lorentzian(F, y, max_back, k, horizon, faithful):
    n = F.shape[0]
    pred = np.zeros(n)
    for i in range(max_back + horizon, n):
        bad = False
        for q in range(F.shape[1]):
            if np.isnan(F[i, q]):
                bad = True
        if bad:
            continue
        if faithful:
            # jdehorty's selection: walk the window from oldest to newest in steps of 4, keep a
            # neighbour when its distance >= the running 'lastDistance', drop the oldest beyond k
            last = -1.0
            dist = np.empty(k + 1)
            lab = np.empty(k + 1)
            m = 0
            start = i - horizon - max_back
            for j in range(start, i - horizon + 1, 4):
                d = 0.0
                for q in range(F.shape[1]):
                    d += np.log(1.0 + abs(F[i, q] - F[j, q]))
                if d >= last:
                    dist[m] = d
                    lab[m] = y[j]
                    m += 1
                    if m > k:
                        last = dist[int(round(k * 3 / 4))]
                        for u in range(m - 1):
                            dist[u] = dist[u + 1]
                            lab[u] = lab[u + 1]
                        m -= 1
            s = 0.0
            for u in range(m):
                s += lab[u]
            pred[i] = s
        else:
            # true k nearest neighbours in the same window (every 4th bar)
            dist = np.full(k, 1e18)
            lab = np.zeros(k)
            start = i - horizon - max_back
            for j in range(start, i - horizon + 1, 4):
                d = 0.0
                for q in range(F.shape[1]):
                    d += np.log(1.0 + abs(F[i, q] - F[j, q]))
                worst = 0
                for u in range(1, k):
                    if dist[u] > dist[worst]:
                        worst = u
                if d < dist[worst]:
                    dist[worst] = d
                    lab[worst] = y[j]
            s = 0.0
            for u in range(k):
                s += lab[u]
            pred[i] = s
    return pred


def part_knn(H: A.Harness) -> pd.DataFrame:
    df = H.df
    F = knn_features(df)
    c = H.c
    horizon = 4
    y = np.sign(np.r_[c[horizon:], np.full(horizon, np.nan)] - c)
    y = np.nan_to_num(y, nan=0.0)
    # filters of the script (defaults): volatility (ATR1 > ATR10), regime (kernel slope > -0.1 proxy)
    atr1 = ind.atr(df, 1).to_numpy()
    atr10 = ind.atr(df, 10).to_numpy()
    vol_ok = atr1 > atr10
    yhat = kernel_rq(c, 8.0, 8.0, 25)
    bullish = np.r_[False, yhat[1:] > yhat[:-1]]
    bearish = np.r_[False, yhat[1:] < yhat[:-1]]
    o = H.o
    rows = [A.row("baseline efficient", H.baselines()["efficient"], {"kind": "baseline"})]
    for faithful in (True, False):
        t0 = time.time()
        pred = _lorentzian(F.astype(np.float64), y.astype(np.float64), 2000, 8, horizon, faithful)
        secs = round(time.time() - t0)
        signal = np.zeros(H.n, dtype=np.int64)
        for i in range(1, H.n):
            if pred[i] > 0 and vol_ok[i]:
                signal[i] = 1
            elif pred[i] < 0 and vol_ok[i]:
                signal[i] = -1
            else:
                signal[i] = signal[i - 1]
        flip = np.r_[0, np.diff(signal)]
        entry = np.where((flip > 0) & bullish, 1, np.where((flip < 0) & bearish, -1, 0)).astype(np.int64)
        name = "Lorentzian 分類 (原腳本的鄰居選法)" if faithful else "Lorentzian 分類 (真正的 kNN)"
        # the script's own exit: 4 bars later at the open (or earlier on a flip); costs as everywhere
        rs = []
        for i in np.flatnonzero(entry):
            if i + 1 + horizon >= H.n:
                continue
            d = entry[i]
            e_px = o[i + 1] + d * (H.sp[i + 1] / 2 + H.sl[i + 1])
            j = i + 1 + horizon
            for jj in range(i + 2, i + 1 + horizon):
                if entry[jj] == -d:
                    j = jj + 1
                    break
            x_px = o[j] - d * (H.sp[j] / 2 + H.sl[j])
            risk = 2.0 * H.atr1[i]
            rs.append({"entry_time": df.index[i + 1], "exit_time": df.index[j], "R": d * (x_px - e_px) / risk, "reason": "bars",
                       "sig_i": i, "exit_i": j, "dir": d})
        tr4 = pd.DataFrame(rs)
        tr4 = tr4[tr4["entry_time"] >= pd.Timestamp("2005-01-01", tz="UTC")]
        rows.append(A.row(f"{name} | 原腳本 4 根出場", tr4, {"kind": "knn_native", "signals": int((entry != 0).sum()), "secs": secs}))
        raw = H.timed(entry)
        flt = H.filtered(raw)
        rows.append(A.row(f"{name} | 系統出場", H.run1(raw), {"kind": "knn_raw"}))
        rows.append(A.row(f"{name} | 加系統過濾", H.run1(flt), {"kind": "knn_filtered", "signals": int((flt != 0).sum())}))
        rows.append(A.row(f"{name} | 當高效積極的第 2 張", H.run2(H.sys_sig(18, True), flt), {"kind": "knn_addon"}))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "XAUUSD_open_source_knn.csv", index=False)
    return res


# ----------------------------------------------------------------------------- meta-labeling

def candidate_pool(H: A.Harness) -> tuple[np.ndarray, pd.DataFrame]:
    """Direction per bar (0 = none) and one-hot generator flags for every candidate entry."""
    gens = {}
    for hrs, sq in ((8, False), (12, False), (16, False), (24, False), (12, True), (16, True), (18, True), (24, True)):
        gens[f"sys{hrs}{'s' if sq else ''}"] = H.sys_sig(hrs, sq)
    gens["supertrend15"] = H.filtered(H.timed(A.trig_supertrend(H, 10, 3.0, "15m")))
    gens["keltner15"] = H.filtered(H.timed(A.trig_keltner(H, tf="15m")))
    gens["lazybear15"] = H.filtered(H.timed(A.trig_lazybear(H, tf="15m")))
    gens["bos10"] = H.filtered(H.timed(A.trig_bos(H, 10)))
    gens["lw05"] = H.filtered(H.timed(A.trig_lw(H, 0.5)))
    gens["utbot1h"] = H.filtered(H.timed(A.trig_utbot(H, 2.0, 10, "1h")))
    G = pd.DataFrame(gens, index=H.df.index)
    pos = (G > 0).sum(axis=1)
    neg = (G < 0).sum(axis=1)
    d = np.where((pos > 0) & (neg == 0), 1, np.where((neg > 0) & (pos == 0), -1, 0)).astype(np.int64)
    return d, G


def part_meta(H: A.Harness) -> pd.DataFrame:
    d, G = candidate_pool(H)
    cand = np.flatnonzero(d)
    print(f"  candidate entries: {len(cand)} ({len(cand) / ((H.df.index[-1] - H.df.index[0]).days / 7):.2f} per week)")
    # 1) label every candidate by simulating it alone
    bk = E.Book(H.df, H.atr1, H.plan, H.sp, H.sl, H.clock)
    t0 = time.time()
    recs = []
    for i in cand:
        r = bk.open(int(i), int(d[i]))
        if r is not None:
            recs.append({"i": int(i), "dir": int(d[i]), "R": float(r["R"]), "exit_i": int(r["exit_i"]), "reason": r["reason"]})
    L = pd.DataFrame(recs)
    L = L[L["reason"] != "end"].reset_index(drop=True)
    print(f"  labelled {len(L)} candidates in {time.time() - t0:.0f}s; avg R alone {L['R'].mean():+.3f}, {100 * (L['R'] > 0).mean():.0f}% winners")
    # 2) features at the signal bar
    X = feature_table(H)
    Xc = X.iloc[L["i"].to_numpy()].reset_index(drop=True)
    Gc = G.iloc[L["i"].to_numpy()].reset_index(drop=True)
    for col in Gc.columns:
        Xc[f"g_{col}"] = (Gc[col].to_numpy() == L["dir"].to_numpy()).astype(float)
    Xc["n_gen"] = Xc[[f"g_{c}" for c in Gc.columns]].sum(axis=1)
    Xc["dir"] = L["dir"].to_numpy()
    # direction-symmetric features: express trend / momentum relative to the trade direction
    for col in ("ret1", "ret4", "ret16", "ret96", "d_ema20_15", "d_ema50_15", "d_ema20_h1", "d_ema50_h1", "d_ema20_d1",
                "d_ema50_d1", "trend_d1", "trend_h4", "trend_h1", "d_chan_hi", "d_chan_lo"):
        Xc[col] = Xc[col] * Xc["dir"]
    Xc["rsi_m15"] = np.where(Xc["dir"] > 0, Xc["rsi_m15"], 100 - Xc["rsi_m15"])
    Xc["rsi_h1"] = np.where(Xc["dir"] > 0, Xc["rsi_h1"], 100 - Xc["rsi_h1"])
    Xc["day_pos"] = np.where(Xc["dir"] > 0, Xc["day_pos"], 1 - Xc["day_pos"])
    cols = [c for c in Xc.columns if c != "dir"]
    Xv = Xc[cols].to_numpy(float)
    yrs = year_of(H.df.index[L["i"].to_numpy()])
    exit_yrs = year_of(H.df.index[L["exit_i"].to_numpy()])
    R = L["R"].to_numpy()
    ywin = (R > 0).astype(int)
    # 3) walk-forward probabilities
    prob = np.full(len(L), np.nan)
    rank = np.full(len(L), np.nan)
    t0 = time.time()
    for Y in YEARS:
        tr = (exit_yrs < Y) & np.isfinite(Xv).all(axis=1)       # purged: the trade must have closed before the test year
        te = yrs == Y
        if tr.sum() < 200 or te.sum() == 0:
            continue
        m = gb(False).fit(Xv[tr], ywin[tr])
        ptr = m.predict_proba(Xv[tr])[:, 1]
        p = m.predict_proba(np.nan_to_num(Xv[te], nan=0.0))[:, 1]
        prob[te] = p
        rank[te] = np.searchsorted(np.sort(ptr), p) / len(ptr)      # percentile within the training probabilities
    print(f"  walk-forward fits in {time.time() - t0:.0f}s")
    L["prob"], L["rank"] = prob, rank
    # importance (last model, for the write-up)
    imp = pd.Series(getattr(m, "feature_importances_", np.zeros(len(cols))), index=cols) if hasattr(m, "feature_importances_") else None
    # 4) trade the selected candidates like 高效積極模式 (max 2, same direction), compare on 2009+
    rows = []
    base = H.baselines()
    win = pd.Timestamp("2009-01-01", tz="UTC")
    def after(tr):
        return tr[tr["entry_time"] >= win].reset_index(drop=True)
    for k, tr in base.items():
        rows.append(A.row(f"baseline {k} (2009 起)", after(tr), {"kind": "baseline"}))
    rows.append(A.row("候選池全做 (2009 起), 最多 2 張", after(H.run2(d)), {"kind": "pool"}))
    for q in (0.5, 0.6, 0.7, 0.8):
        sel = L[(L["rank"] >= q)]
        sig = np.zeros(H.n, dtype=np.int64)
        sig[sel["i"].to_numpy()] = sel["dir"].to_numpy()
        rows.append(A.row(f"meta-labeling 取機率前 {int((1 - q) * 100)}% (2009 起), 最多 2 張", after(H.run2(sig)),
                          {"kind": "meta", "signals": int(len(sel))}))
        rows.append(A.row(f"meta-labeling 前 {int((1 - q) * 100)}% 當高效積極的第 2 張 (2009 起)", after(H.run2(H.sys_sig(18, True), sig)),
                          {"kind": "meta_addon"}))
    for p0 in (0.5, 0.55, 0.6):
        sel = L[(L["prob"] >= p0)]
        sig = np.zeros(H.n, dtype=np.int64)
        sig[sel["i"].to_numpy()] = sel["dir"].to_numpy()
        rows.append(A.row(f"meta-labeling 機率 ≥ {p0:.2f} (2009 起), 最多 2 張", after(H.run2(sig)), {"kind": "meta_p", "signals": int(len(sel))}))
    # how good is the ranking at all? R by probability quintile, out of sample
    ok = L["rank"].notna()
    L["q"] = pd.qcut(L.loc[ok, "rank"], 5, labels=False, duplicates="drop")
    byq = L[ok].groupby("q")["R"].agg(["count", "mean"]).round(3)
    print("  out-of-sample R by probability quintile (0 = lowest):\n", byq.to_string())
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "XAUUSD_open_source_meta.csv", index=False)
    L.to_csv(OUT / "XAUUSD_open_source_meta_candidates.csv", index=False)
    byq.to_csv(OUT / "XAUUSD_open_source_meta_quintiles.csv")
    if imp is not None:
        imp.sort_values(ascending=False).to_csv(OUT / "XAUUSD_open_source_meta_importance.csv")
    return res


def main(H: A.Harness, which: str):
    if which == "ml":
        part_ml(H)
    elif which == "knn":
        part_knn(H)
    elif which == "meta":
        part_meta(H)
    else:
        raise SystemExit(f"unknown part {which}")
