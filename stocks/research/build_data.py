"""Build normalized, total-return-adjusted daily bars for the research universe from the GitHub-hosted
datasets found on 2026-11-08 (see docs/RESEARCH.md section 1 and research/data_sources.json).

Method: daily RETURNS are chained from the best source available on each day (sources have
different adjustment bases, so levels cannot be concatenated), then the series is rebased so the
last close equals the last raw close. For unadjusted Taiwan sources (TWSE / FinMind) the
ex-dividend days are detected from the exchange's own "漲跌" field (reported as X / 0 on ex-days)
and the cash dividend is estimated from the TWSE dividend yield of that day; stock dividends and
splits in 2000-2021 are already adjusted in the Yahoo-based source.

Output: data/daily/<SYMBOL>.csv  (date, open, high, low, close, volume, close_raw, src)
        data/daily/_report.json  (coverage, overlap checks, detected ex-dates)

    python research/build_data.py --datasets /path/to/datasets
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stocksignal.universe import DEFAULT_UNIVERSE  # noqa: E402

OUT = ROOT / "data" / "daily"


# ----------------------------------------------------------------------------- readers
def _std(df: pd.DataFrame, date_col: str, o: str, h: str, l: str, c: str, v: str | None, adj: str | None = None,
         utc_strip: bool = False) -> pd.DataFrame:
    d = pd.DataFrame()
    idx = pd.to_datetime(df[date_col].astype(str).str.slice(0, 10), errors="coerce")
    d["open"], d["high"], d["low"], d["close"] = [pd.to_numeric(df[x], errors="coerce").to_numpy() for x in (o, h, l, c)]
    d["volume"] = pd.to_numeric(df[v], errors="coerce").to_numpy() if v else 0.0
    d["adj"] = pd.to_numeric(df[adj], errors="coerce").to_numpy() if adj else d["close"].to_numpy()
    d.index = idx
    d = d[~d.index.isna()].sort_index()
    d = d[~d.index.duplicated(keep="last")]
    d = d[(d["close"] > 0) & (d["adj"] > 0)]
    for c in ("open", "high", "low"):
        d.loc[d[c] <= 0, c] = d.loc[d[c] <= 0, "close"]
    return d.dropna(subset=["close"])


def _mark_last_raw(d: pd.DataFrame) -> pd.DataFrame:
    """For dividend-adjusted sources the LAST bar's adjusted close equals the actual close (as of download)."""
    if len(d):
        d["close_raw"] = np.nan
        d.iloc[-1, d.columns.get_loc("close_raw")] = d["close"].iloc[-1]
    return d


def read_plain(path: Path) -> pd.DataFrame:            # adjusted OHLCV (date column may be unnamed)
    df = pd.read_csv(path)
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    return _mark_last_raw(_std(df, "date", "open", "high", "low", "close", "volume"))


def read_yfin_lower(path: Path) -> pd.DataFrame:       # yennanliu: date,open,high,low,close,volume,div,split (adjusted)
    df = pd.read_csv(path)
    return _mark_last_raw(_std(df, "date", "open", "high", "low", "close", "volume"))


def read_yahoo_caps(path: Path) -> pd.DataFrame:       # Date,Open,High,Low,Close,(Adj Close),Volume
    df = pd.read_csv(path)
    adj = "Adj Close" if "Adj Close" in df.columns else None
    return _mark_last_raw(_std(df, "Date", "Open", "High", "Low", "Close", "Volume" if "Volume" in df.columns else None, adj))


def read_ndx(path: Path) -> pd.DataFrame:              # Date,CompanyName,Symbol,Open,High,Low,Close,Volume,Unadjusted Close,InIndex
    df = pd.read_csv(path, encoding="utf-8-sig")
    d = _std(df, "Date", "Open", "High", "Low", "Close", "Volume")
    d["close_raw"] = pd.to_numeric(df.set_index(pd.to_datetime(df["Date"]))["Unadjusted Close"], errors="coerce").reindex(d.index).to_numpy()
    return d


def read_kronos(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _mark_last_raw(_std(df, "timestamps", "open", "high", "low", "close", "volume"))


def read_close_only(path: Path) -> pd.DataFrame:       # culolom: Date,Close,Volume (adjusted)
    df = pd.read_csv(path)
    d = pd.DataFrame({"close": pd.to_numeric(df["Close"], errors="coerce").to_numpy(),
                      "volume": pd.to_numeric(df["Volume"], errors="coerce").fillna(0).to_numpy()},
                     index=pd.to_datetime(df["Date"], errors="coerce"))
    d = d[~d.index.isna()].sort_index()
    for c in ("open", "high", "low"):
        d[c] = d["close"]
    d["adj"] = d["close"]
    return _mark_last_raw(d.dropna(subset=["close"]))


def read_finmind(path: Path) -> pd.DataFrame:          # date,stock_id,Trading_Volume,Trading_money,open,max,min,close,spread,Trading_turnover
    df = pd.read_csv(path, encoding="utf-8-sig")
    d = _std(df, "date", "open", "max", "min", "close", "Trading_Volume")
    if "spread" in df.columns:
        d["spread"] = pd.to_numeric(df.set_index(pd.to_datetime(df["date"]))["spread"], errors="coerce").reindex(d.index).to_numpy()
    else:
        d["spread"] = np.nan
    d["unadjusted"] = True
    return d


def read_twii_bofrank(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = ["date", "open", "high", "low", "close"][: len(df.columns)]
    df["date"] = df["date"].astype(str)
    return _std(df, "date", "open", "high", "low", "close", None)


def read_timhun(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _std(df, "date", "open", "high", "low", "close", "volume")


# ------------------------------------------------------------------- TW dividends
def tw_ex_dates(d: pd.DataFrame, yield_df: pd.DataFrame | None, report: list) -> pd.Series:
    """Estimated cash dividend per share on detected ex-dates (FinMind 'spread' == 0 and close moved).

    Returns a Series indexed by ex-date with the dividend estimate; NaN where unknown (skipped)."""
    out = {}
    if "spread" not in d.columns:
        return pd.Series(dtype=float)
    prev = d["close"].shift(1)
    flag = (d["spread"] == 0) & (d["close"] != prev) & prev.notna()
    # ETF rows sometimes carry the reference-price difference instead of 0
    ref = d["close"] - d["spread"]
    flag2 = ((ref - prev).abs() / prev > 0.003) & ((ref - prev).abs() / prev < 0.15) & prev.notna() & (d["spread"] != 0)
    days = d.index[flag | flag2]
    for day in days:
        p = prev.loc[day]
        est = np.nan
        if flag2.loc[day]:
            est = float(p - ref.loc[day])
        elif yield_df is not None and len(yield_df):
            y = yield_df["yield"].reindex([day]).iloc[0]
            if pd.isna(y):
                y = yield_df["yield"][yield_df.index <= day].tail(1)
                y = float(y.iloc[0]) if len(y) else np.nan
            if not pd.isna(y) and y > 0:
                lo, hi = day - pd.Timedelta(days=370), day
                n = int(((d.index[flag | flag2] > lo) & (d.index[flag | flag2] <= hi)).sum())
                est = float(y / 100.0 * p / max(n, 1))
        if not pd.isna(est) and p > 0 and 0 < est / p < 0.15:
            out[day] = est
        report.append({"date": str(day.date()), "prev_close": float(p), "close": float(d["close"].loc[day]),
                       "dividend_est": None if pd.isna(est) else round(float(est), 3)})
    return pd.Series(out, dtype=float)


# -------------------------------------------------------------------------- chaining
def chain(segments: list[tuple[str, pd.DataFrame]], dividends: dict[str, pd.Series] | None = None) -> tuple[pd.DataFrame, dict]:
    """segments: [(name, frame)] in priority order (first = preferred when overlapping).

    Each frame has open/high/low/close/adj (adj = dividend-adjusted close when available; for
    unadjusted frames adj == close and dividends[name] supplies the ex-date adjustments).
    """
    dividends = dividends or {}
    all_days = sorted(set().union(*[set(f.index) for _, f in segments]))
    rows = []
    checks = {}
    # overlap consistency: correlation of daily returns between consecutive sources
    for (n1, f1), (n2, f2) in zip(segments, segments[1:]):
        common = f1.index.intersection(f2.index)
        if len(common) > 30:
            r1 = f1["adj"].reindex(common).pct_change()
            r2 = f2["adj"].reindex(common).pct_change()
            checks[f"{n1}~{n2}"] = {"overlap_days": int(len(common)), "ret_corr": round(float(r1.corr(r2)), 4),
                                    "mean_abs_diff_bp": round(float((r1 - r2).abs().mean() * 1e4), 2)}
    prev_close_by_src = {}
    for t in all_days:
        picked = None
        for name, f in segments:
            if t not in f.index:
                continue
            row = f.loc[t]
            # need a previous close within the same source (<= 10 calendar days gap) to form a return
            pos = f.index.get_loc(t)
            if pos == 0:
                if picked is None and not rows:          # very first day of the whole series
                    picked = (name, row, None)
                continue
            prev_t = f.index[pos - 1]
            if (t - prev_t).days > 12:
                continue
            prev_adj = f["adj"].iloc[pos - 1]
            r = row["adj"] / prev_adj
            if name in dividends and t in dividends[name].index:
                div = dividends[name].loc[t]
                r = row["close"] / (f["close"].iloc[pos - 1] - div)
            picked = (name, row, r)
            break
        if picked is None:
            continue
        name, row, r = picked
        shape = row
        if row["open"] == row["close"] == row["high"] == row["low"]:      # close-only source: borrow the bar shape
            for n2, f2 in segments:
                if n2 != name and t in f2.index and not (f2.loc[t, "open"] == f2.loc[t, "close"] == f2.loc[t, "high"]):
                    shape = f2.loc[t]
                    if "close_raw" not in row or pd.isna(row.get("close_raw", np.nan)):
                        row = row.copy(); row["close_raw"] = shape.get("close_raw", shape["close"])
                    break
        craw = row.get("close_raw", np.nan)
        if pd.isna(craw):
            craw = shape.get("close_raw", np.nan)
        rows.append({"date": t, "src": name, "r": r, "o": shape["open"] / shape["close"], "h": shape["high"] / shape["close"],
                     "l": shape["low"] / shape["close"], "volume": shape["volume"], "close_raw": craw})
    if not rows:
        return pd.DataFrame(), checks
    df = pd.DataFrame(rows).set_index("date")
    df["r"] = df["r"].fillna(1.0)
    # drop impossible returns (data glitches): > +/- 60% in a day
    bad = (df["r"] > 1.6) | (df["r"] < 0.4)
    if bad.any():
        checks["dropped_glitch_days"] = [str(x.date()) for x in df.index[bad]][:20]
        df.loc[bad, "r"] = 1.0
    level = df["r"].cumprod()
    raw_known = df["close_raw"].dropna()
    if len(raw_known):
        d_star = raw_known.index[-1]
        close = level / level.loc[d_star] * float(raw_known.iloc[-1])
    else:
        close = level
    out = pd.DataFrame({"open": close * df["o"], "high": close * df["h"], "low": close * df["l"], "close": close,
                        "volume": df["volume"], "close_raw": df["close_raw"], "src": df["src"]}, index=df.index)
    out.index.name = "date"
    # sanity: high >= max(open, close), low <= min(open, close)
    out["high"] = out[["high", "open", "close"]].max(axis=1)
    out["low"] = out[["low", "open", "close"]].min(axis=1)
    return out, checks


# ------------------------------------------------------------------------- builders
def build_us(ds: Path, sym: str) -> tuple[pd.DataFrame, dict] | None:
    segs = []
    pb = ds / "plain_backtest" / "repo" / "data" / "processed" / "daily"
    for p in (pb / "equities" / f"{sym}.csv", pb / f"{sym}.csv"):
        if p.exists():
            segs.append(("plain_backtest", read_plain(p)))
            break
    ndx = glob.glob(str(ds / "plain_backtest" / "ndx100_adj" / f"*{sym}*.csv"))
    ndx = [x for x in ndx if os.path.basename(x).split(".")[0].upper().split("-")[0] == sym or os.path.basename(x).upper().startswith(sym)]
    if not segs and ndx:
        segs.append(("ndx100", read_ndx(Path(ndx[0]))))
    misc = ds / "etf_misc"
    extra = {
        "SMH": [("azataiot", misc / "SMH_azataiot_2013-2023.csv", read_yahoo_caps), ("kronos", misc / "SMH_kronos_2015-2026.csv", read_kronos),
                ("etspotter", misc / "SMH_etspotter_2024-2026.csv", read_plain)],
        "SOXX": [("azataiot", misc / "SOXX_azataiot_2013-2023.csv", read_yahoo_caps), ("crowding", misc / "SOXX_crowding_2022-2026.csv", read_plain)],
        "XLK": [("ramki", misc / "XLK_ramki_2003-2022.csv", read_yahoo_caps), ("etspotter", misc / "XLK_etspotter_2024-2026.csv", read_plain)],
        "TSM": [("azataiot", misc / "TSM_azataiot_2013-2023.csv", read_yahoo_caps)],
    }
    for name, p, rd in extra.get(sym, []):
        if p.exists():
            f = rd(p)
            if sym == "XLK" and name == "etspotter":
                f.index = pd.to_datetime(f.index)
            segs.append((name, f))
    yl = ds / "yennanliu_finance_data" / "repo" / "data" / "prices" / f"{sym.lower()}.csv"
    if yl.exists():
        segs.append(("yennanliu", read_yfin_lower(yl)))
    if sym == "GOOGL" and not yl.exists():
        g = ds / "yennanliu_finance_data" / "repo" / "data" / "prices" / "goog.csv"
        if g.exists():
            segs.append(("yennanliu_goog", read_yfin_lower(g)))      # same company, class C; returns nearly identical
    cu = ds / "culolom" / "repo" / "data" / f"{sym}.csv"
    if cu.exists():
        segs.append(("culolom_close", read_close_only(cu)))
    if not segs:
        return None
    # chaining priority: long adjusted OHLC sources first, then recent extensions
    order = {"plain_backtest": 0, "ndx100": 1, "ramki": 2, "azataiot": 3, "kronos": 4, "yennanliu": 5, "yennanliu_goog": 5, "crowding": 6,
             "etspotter": 7, "culolom_close": 9}
    segs.sort(key=lambda x: order.get(x[0], 8))
    return chain(segs)


def read_tsec(path: Path) -> pd.DataFrame:
    """TWSE official daily file (no header): ROC date, shares, value, O, H, L, C, change ('X..' on ex-days), trades."""
    t = pd.read_csv(path, header=None, names=["d", "vol", "val", "o", "h", "l", "c", "chg", "n"], dtype=str)
    def roc(x):
        y, m, dd = str(x).split("/")
        return pd.Timestamp(int(y) + 1911, int(m), int(dd))
    idx = t["d"].map(roc)
    d = pd.DataFrame(index=idx)
    for src, dst in (("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"), ("vol", "volume")):
        d[dst] = pd.to_numeric(t[src].str.replace(",", ""), errors="coerce").to_numpy()
    d["ex"] = t["chg"].astype(str).str.contains("X").to_numpy()
    d = d[(d["close"] > 0)].sort_index()
    d = d[~d.index.duplicated(keep="last")]
    for c in ("open", "high", "low"):
        d.loc[d[c].isna() | (d[c] <= 0), c] = d["close"]
    return d


def build_tw(ds: Path, code: str, sym: str) -> tuple[pd.DataFrame, dict] | None:
    """Taiwan: raw TWSE/FinMind price path x dividend/split adjustment factor.

    Adjustment multipliers on ex-dates come from, in order: a yfinance-adjusted modern source
    (culolom / yennanliu) when it covers the date; the Yahoo Adj/raw factor step for stocks
    2000-2021 (robust medians around the ex-date, ignores Yahoo's isolated bad prints); the
    TWSE dividend yield x price for stocks after 2021; a market-adjusted price gap for ETFs.
    """
    rep: dict = {"ex_dates": []}
    is_etf = DEFAULT_UNIVERSE[sym].kind == "etf"
    # ---- raw path: TWSE (tsec) until 2014, FinMind from 2015 (both unadjusted, official closes)
    raws = []
    p = ds / "tsec" / "repo" / "data" / f"{code}.csv"
    if p.exists():
        t = read_tsec(p)
        raws.append(t[t.index < "2015-01-01"] if (ds / "fintech_ai").exists() else t)
    fm = ds / "fintech_ai" / "repo" / "data" / "external" / "raw" / "prices" / f"{code}_price.csv"
    ga = ds / "stock_GA" / "repo" / "data" / "stock_dayK" / f"{code}.csv"
    fin = None
    for pth in (fm, ga):
        if pth.exists():
            fin = read_finmind(pth)
            break
    if fin is not None:
        prev = fin["close"].shift(1)
        fin["ex"] = ((fin["spread"] == 0) & (fin["close"] != prev) & prev.notna()) | \
                    (((fin["close"] - fin["spread"] - prev).abs() / prev > 0.003) & (fin["spread"] != 0) & prev.notna())
        if raws and len(raws[0]):
            gap_start = raws[0].index[-1] + pd.Timedelta(days=1)
            # if FinMind starts after the TWSE file ends (ETFs from stock_GA start 2020), fill with Yahoo raw close
            if fin.index[0] > gap_start + pd.Timedelta(days=40):
                an = glob.glob(str(ds / "tw_analyzer" / "repo" / "output" / "歷史股價" / "**" / f"{code}.csv"), recursive=True)
                if an:
                    y = read_yahoo_caps(Path(an[0]))
                    y = y[(y["volume"] > 0) & (y.index >= gap_start) & (y.index < fin.index[0])].copy()
                    y["ex"] = False
                    y["from_yahoo"] = True
                    raws.append(y[["open", "high", "low", "close", "volume", "ex", "from_yahoo"]])
        raws.append(fin[["open", "high", "low", "close", "volume", "ex"]])
    if not raws:
        return None
    raw = pd.concat(raws).sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    raw = raw[raw.index >= "2000-01-01"]
    if "from_yahoo" not in raw.columns:
        raw["from_yahoo"] = False
    raw["from_yahoo"] = raw["from_yahoo"].fillna(False).astype(bool)
    if len(raw) < 300:
        return None
    prev_close = raw["close"].shift(1)
    # ---- market series for the ETF gap estimate
    twii = None
    tp = ds / "tw_index" / "TAIEX_bofrankbo_1999-2024_ohlc.csv"
    cp = ds / "culolom" / "repo" / "data" / "^TWII.csv"
    if cp.exists():
        twii = read_close_only(cp)["close"]
    elif tp.exists():
        twii = read_twii_bofrank(tp)["close"]
    # ---- modern yfinance-adjusted sources (factor truth where they exist)
    modern = None
    cu = ds / "culolom" / "repo" / "data" / f"{code}.TW.csv"
    yl = ds / "yennanliu_finance_data" / "repo" / "data" / "prices" / (f"{code}.tw.csv" if code != "0050" else "0050.csv")
    if cu.exists():
        modern = read_close_only(cu)["close"]
    if yl.exists():
        yy = read_yfin_lower(yl)["close"]
        modern = yy if modern is None else modern.combine_first(yy)
    ratio = None
    if modern is not None:
        ratio = (modern.reindex(raw.index) / raw["close"]).rolling(3, center=True).median()
    # ---- Yahoo (2021 download) adjustment steps for stocks
    yahoo_f = None
    an = glob.glob(str(ds / "tw_analyzer" / "repo" / "output" / "歷史股價" / "**" / f"{code}.csv"), recursive=True)
    if an and not is_etf:
        y = read_yahoo_caps(Path(an[0]))
        y = y[y["volume"] > 0]
        yahoo_f = (y["adj"] / raw["close"].reindex(y.index)).dropna()
    # ---- yield for stocks after 2021
    yield_df = None
    per = ds / "fintech_ai" / "repo" / "data" / "external" / "raw" / "per_pbr" / f"{code}_per_pbr.csv"
    if per.exists():
        yv = pd.read_csv(per, encoding="utf-8-sig")
        yield_df = pd.DataFrame({"yield": pd.to_numeric(yv["dividend_yield"], errors="coerce").to_numpy(),
                                 "pe": pd.to_numeric(yv["PER"], errors="coerce").to_numpy(),
                                 "pb": pd.to_numeric(yv["PBR"], errors="coerce").to_numpy()}, index=pd.to_datetime(yv["date"])).sort_index()
        yield_df.to_csv(OUT / f"{sym}.valuation.csv.gz", compression="gzip")
    # ---- ETF ex-dates inside the Yahoo gap window: market-adjusted drop > 1.5%
    if is_etf and twii is not None and raw["from_yahoo"].any():
        mk = twii.reindex(raw.index).ffill().pct_change()
        gap = prev_close * (1 + mk) - raw["close"]
        cand = raw["from_yahoo"] & (gap / prev_close > 0.015)
        raw.loc[cand, "ex"] = True
    # ---- daily multipliers
    m = pd.Series(1.0, index=raw.index)
    src = pd.Series("none", index=raw.index)
    ex_days = raw.index[raw["ex"].fillna(False).astype(bool) & prev_close.notna()]
    n_modern = n_yahoo = n_yield = n_gap = 0
    for d in ex_days:
        i = raw.index.get_loc(d)
        p = prev_close.iloc[i]
        mult = None
        if ratio is not None and i > 0 and not pd.isna(ratio.iloc[i]) and not pd.isna(ratio.iloc[i - 1]) and ratio.iloc[i - 1] > 0:
            mult = float(ratio.iloc[i] / ratio.iloc[i - 1]); n_modern += 1; how = "modern"
        elif yahoo_f is not None and d in yahoo_f.index:
            j = yahoo_f.index.get_loc(d)
            before = yahoo_f.iloc[max(0, j - 6): j].median()
            after = yahoo_f.iloc[j + 1: j + 7].median()
            if before > 0 and after > 0 and 0.3 < after / before < 1.8:          # reverse splits / capital reductions too
                mult = float(after / before); n_yahoo += 1; how = "yahoo_step"
        raw_move = abs(raw["close"].iloc[i] / p - 1.0)
        if mult is None and not is_etf and yield_df is not None and d >= pd.Timestamp("2015-01-01") and raw_move < 0.105:
            yv = yield_df["yield"][yield_df.index <= d].tail(1)
            if len(yv) and yv.iloc[0] > 0:
                n_ev = int(((ex_days > d - pd.Timedelta(days=370)) & (ex_days <= d)).sum())
                div = float(yv.iloc[0]) / 100.0 * p / max(n_ev, 1)
                mult = p / (p - div); n_yield += 1; how = "yield"
        if mult is None and twii is not None and (is_etf or raw_move >= 0.105):
            mk = twii.reindex([raw.index[i - 1], d]).ffill().pct_change().iloc[-1]
            if not pd.isna(mk):
                ref = raw["close"].iloc[i] / (1 + mk)                 # implied reference price
                cand = p / ref
                if is_etf:
                    prior = [e["mult"] for e in rep["ex_dates"] if e.get("mult") and e["how"] == "mkt_gap"]
                    cap = 1 + 2.0 * (np.median(prior) - 1) if len(prior) >= 3 else 1.12
                    if 1.002 < cand < cap:
                        mult = float(cand); n_gap += 1; how = "mkt_gap"
                elif 0.5 < cand < 2.0:
                    mult = float(cand); n_gap += 1; how = "mkt_gap_corporate_action"
        if mult is not None and 0.3 < mult < 1.8:
            m.iloc[i] = mult
            src.iloc[i] = how
            rep["ex_dates"].append({"date": str(d.date()), "prev_close": float(p), "close": float(raw["close"].iloc[i]),
                                    "mult": round(mult, 4), "how": how})
        else:
            rep["ex_dates"].append({"date": str(d.date()), "prev_close": float(p), "close": float(raw["close"].iloc[i]), "mult": None, "how": "skipped"})
    # non-ex days covered by a modern source: let its day-to-day ratio changes through too (splits, missed ex-days)
    if ratio is not None:
        rr = (ratio / ratio.shift(1))
        extra = rr.notna() & ((rr - 1).abs() > 0.15) & ~raw["ex"].fillna(False).astype(bool)      # splits only
        for d in raw.index[extra]:
            m.loc[d] = float(rr.loc[d]); src.loc[d] = "modern_extra"
    F = m.cumprod()
    seg = raw[["open", "high", "low", "close", "volume"]].copy()
    seg["adj"] = seg["close"] * F
    seg["close_raw"] = seg["close"]
    segs = [("twse_raw_adj", seg)]
    if yl.exists():
        segs.insert(0, ("yennanliu", read_yfin_lower(yl)))
    if cu.exists():
        segs.append(("culolom_close", read_close_only(cu)))
    out, checks = chain(segs)
    if out.empty:
        return None
    checks["tw_ex_dates_detected"] = int(len(ex_days))
    checks["tw_mult_sources"] = {"modern": n_modern, "yahoo_step": n_yahoo, "yield": n_yield, "mkt_gap": n_gap}
    checks["tw_ex_dates"] = rep["ex_dates"][-14:]
    return out, checks


def build_twii(ds: Path) -> tuple[pd.DataFrame, dict] | None:
    segs = []
    p = ds / "tw_index" / "TAIEX_bofrankbo_1999-2024_ohlc.csv"
    if p.exists():
        segs.append(("twse_official", read_twii_bofrank(p)))
    p = ds / "culolom" / "repo" / "data" / "^TWII.csv"
    if p.exists():
        segs.append(("culolom_close", read_close_only(p)))
    if not segs:
        return None
    return chain(segs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    a = ap.parse_args()
    ds = Path(a.datasets)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for sym, inst in DEFAULT_UNIVERSE.items():
        res = build_tw(ds, inst.code, sym) if inst.market == "TW" else build_us(ds, sym)
        if res is None or res[0].empty:
            report[sym] = {"status": "missing"}
            print(f"{sym:10s} MISSING")
            continue
        df, checks = res
        df.to_csv(OUT / f"{sym}.csv.gz", float_format="%.6f", compression="gzip")
        report[sym] = {"status": "ok", "start": str(df.index[0].date()), "end": str(df.index[-1].date()), "rows": int(len(df)),
                       "sources": df["src"].value_counts().to_dict(), **checks}
        print(f"{sym:10s} {df.index[0].date()} -> {df.index[-1].date()} rows={len(df)} src={dict(df['src'].value_counts())}")
    res = build_twii(ds)
    if res:
        df, checks = res
        df.to_csv(OUT / "^TWII.csv.gz", float_format="%.4f", compression="gzip")
        report["^TWII"] = {"status": "ok", "start": str(df.index[0].date()), "end": str(df.index[-1].date()), "rows": int(len(df)), **checks}
        print(f"^TWII      {df.index[0].date()} -> {df.index[-1].date()} rows={len(df)}")
    (OUT / "_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
