"""Per-instrument analysis: where is the price (zones), what is the trend, what happened historically
from here (probability cloud), and the one-line verdict. Pure functions over a daily frame so the
dashboard and the tests share them.

See docs/RESEARCH.md section 6 for why the verdict is worded the way it is.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import zones as Z
from .backtest import US_STOCK, Costs, metrics, simulate
from .indicators import cagr, max_drawdown, sma
from .regime import direction_label, forward_cloud, score_to_state, trend_score, trend_votes

ZONE_WORDS = {-2: "很便宜", -1: "便宜", 0: "合理", 1: "偏貴", 2: "很貴"}
HORIZONS = (63, 126, 252)


def _f(x, nd=2):
    if x is None:
        return None
    try:
        if np.isnan(x):
            return None
    except TypeError:
        pass
    return round(float(x), nd)


def _last(s: pd.Series):
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else None


def zone_block(name: str, zone: pd.Series, levels: dict, extra: dict | None = None) -> dict:
    z = _last(zone)
    zi = int(z) if z is not None else None
    return {"method": name, "zone": zi, "label": ZONE_WORDS.get(zi, "資料不足"), "levels": {k: _f(v) for k, v in levels.items()},
            **(extra or {})}


def analyze(df: pd.DataFrame, pe: pd.Series | None = None, bench: pd.Series | None = None, kind: str = "stock",
            market: str = "US", costs: Costs = US_STOCK, pooled: dict | None = None) -> dict:
    """df: adjusted daily bars with close_raw. pe: trailing P/E series (optional). bench: benchmark close."""
    c = df["close"]
    out: dict = {"n_bars": int(len(df)), "first": str(df.index[0].date()), "last": str(df.index[-1].date())}
    price = float(c.iloc[-1])
    raw = df["close_raw"].dropna()
    out["price"] = _f(raw.iloc[-1] if len(raw) else price)
    out["chg_1d"] = _f(c.iloc[-1] / c.iloc[-2] - 1, 4) if len(c) > 1 else None
    out["ret"] = {k: _f(c.iloc[-1] / c.iloc[-n - 1] - 1, 4) for k, n in (("1m", 21), ("3m", 63), ("6m", 126), ("1y", 252)) if len(c) > n}
    hi52 = c.tail(252).max()
    out["dd_from_high_52w"] = _f(price / hi52 - 1, 4)

    # ---------------------------------------------------------------- zones
    zones = []
    ch = Z.channel_bands(c)
    zc = Z.channel_zone(ch["zscore"])
    if ch["center"].notna().any():
        zones.append(zone_block("channel", zc, {"dn2": ch["dn2"].iloc[-1], "dn1": ch["dn1"].iloc[-1], "center": ch["center"].iloc[-1],
                                                  "up1": ch["up1"].iloc[-1], "up2": ch["up2"].iloc[-1]},
                                {"zscore": _f(ch["zscore"].iloc[-1]), "slope_annual": _f(ch["slope_annual"].iloc[-1], 3)}))
    dv = Z.deviation_bands(c)
    zd = Z.deviation_zone(dv["dev_rank"])
    if dv["dev_rank"].notna().any():
        zones.append(zone_block("deviation", zd, {"dn2": dv["dn2"].iloc[-1], "dn1": dv["dn1"].iloc[-1], "center": dv["ma"].iloc[-1],
                                                    "up1": dv["up1"].iloc[-1], "up2": dv["up2"].iloc[-1]},
                                {"dev": _f(dv["dev"].iloc[-1], 4), "dev_rank": _f(dv["dev_rank"].iloc[-1], 3)}))
    zp = None
    if pe is not None and len(pe.dropna()) > 300:
        close_for_pe = df["close_raw"].where(df["close_raw"].notna(), c)
        pb = Z.pe_bands(close_for_pe, pe)
        zp = Z.pe_zone(pb)
        if pb["pe_rank"].notna().any() and pd.notna(pb["pe"].iloc[-1]):
            # express the bands in today's (adjusted == raw at the last bar) price terms
            zones.append(zone_block("pe", zp, {"dn2": pb["dn2"].iloc[-1], "dn1": pb["dn1"].iloc[-1], "center": pb["mid"].iloc[-1],
                                                "up1": pb["up1"].iloc[-1], "up2": pb["up2"].iloc[-1]},
                                    {"pe": _f(pb["pe"].iloc[-1]), "pe_rank": _f(pb["pe_rank"].iloc[-1], 3), "eps_ttm": _f(pb["eps_ttm"].iloc[-1]),
                                     "pe_bands": {k: _f(pb[f"pe_{k}"].iloc[-1]) for k in ("dn2", "dn1", "mid", "up1", "up2")}}))
    out["zones"] = zones
    zvals = [z["zone"] for z in zones if z["zone"] is not None]
    out["zone_min"] = min(zvals) if zvals else None
    out["zone_max"] = max(zvals) if zvals else None

    # ---------------------------------------------------------------- trend
    votes = trend_votes(c)
    score = trend_score(c)
    st = score_to_state(score)
    cur_state = _last(st)
    vote_last = votes.iloc[-1]
    out["trend"] = {"score": _f(_last(score)), "state": None if cur_state is None else int(cur_state),
                    "votes": {k: (None if pd.isna(v) else int(v)) for k, v in vote_last.items()},
                    "sma200": _f(sma(c, 200).iloc[-1]), "above_sma200": bool(c.iloc[-1] > sma(c, 200).iloc[-1]) if pd.notna(sma(c, 200).iloc[-1]) else None}
    market_state = None
    if bench is not None and len(bench.dropna()) > 260:
        b = bench.dropna()
        me = b.index.to_period("M")
        is_me = np.r_[me[1:] != me[:-1], True]
        mclose = b[is_me]
        sma10 = mclose.rolling(10).mean()
        # use the last COMPLETED month for the official state, plus the live reading for information
        completed = mclose.iloc[-2] > sma10.iloc[-2] if len(mclose) > 10 and pd.notna(sma10.iloc[-2]) else None
        live = b.iloc[-1] > sma10.iloc[-1] if pd.notna(sma10.iloc[-1]) else None
        market_state = 1 if completed else (-1 if completed is not None else None)
        out["market_trend"] = {"faber_up": None if completed is None else bool(completed), "live_above_10m": None if live is None else bool(live), "sma10m": _f(sma10.iloc[-1]), "last": _f(b.iloc[-1]),
                         "dist_pct": _f(b.iloc[-1] / sma10.iloc[-1] - 1, 4) if pd.notna(sma10.iloc[-1]) else None}
    label, reason = direction_label(cur_state if cur_state is not None else np.nan, market_state)
    out["trend"]["label"], out["trend"]["reason"] = label, reason

    # ---------------------------------------------------------------- probability cloud (own history)
    clouds = {}
    if zones:
        clouds["channel"] = forward_cloud(c, zc, zones[0]["zone"]) if zones[0]["zone"] is not None else {}
    if len(zones) > 1 and zones[1]["method"] == "deviation" and zones[1]["zone"] is not None:
        clouds["deviation"] = forward_cloud(c, zd, zones[1]["zone"])
    if zp is not None:
        pz = next((z for z in zones if z["method"] == "pe"), None)
        if pz and pz["zone"] is not None:
            clouds["pe"] = forward_cloud(c, zp, pz["zone"])
    if cur_state is not None:
        clouds["trend"] = forward_cloud(c, st, cur_state)
    out["cloud"] = clouds
    out["pooled"] = pooled or {}

    # ---------------------------------------------------------------- verdict
    out["verdict"] = verdict(out, kind=kind, market=market)

    # ---------------------------------------------------------------- quick back-test: buy & hold vs 10-month SMA
    out["defense_backtest"] = defense_backtest(df, costs)
    return out


def verdict(a: dict, kind: str, market: str) -> dict:
    zmin, zmax = a.get("zone_min"), a.get("zone_max")
    pe_zone = next((z["zone"] for z in a["zones"] if z["method"] == "pe"), None)
    cloud = a.get("cloud", {})

    names = {"pe": "本益比河流圖", "channel": "五線譜", "deviation": "年線乖離"}
    zl = {z["method"]: z["zone"] for z in a["zones"] if z["zone"] is not None}
    # the method that drives the verdict: cheapest for buy-side wording, richest for the warning
    if zmin is not None and zmin <= -1:
        driver = min(zl, key=lambda k: zl[k]) if zl else None
    else:
        driver = max(zl, key=lambda k: zl[k]) if zl else None
    hist = ""
    if driver:
        cl = cloud.get(driver, {}).get("252")
        if cl and "hit" in cl and cl.get("n_eff", 0) >= 6:
            hist = (f"這檔自己的歷史（{names[driver]}在同一區）：之後 12 個月上漲 {cl['hit']*100:.0f}%（平時 {cl['hit_all']*100:.0f}%），"
                    f"中位數 {cl['median']*100:+.0f}%，10–90% 區間 {cl['p10']*100:+.0f}%～{cl['p90']*100:+.0f}%（約 {cl['n_eff']:.0f} 個獨立樣本）。")
        elif cl and "hit" in cl:
            hist = f"這檔自己在這個區間的歷史樣本太少（約 {cl['n_eff']:.0f} 個獨立樣本），參考同市場合併的統計。"
        else:
            hist = "這檔自己還沒有在這個區間的歷史紀錄，參考同市場合併的統計。"
        pooled = a.get("pooled", {}).get(driver, {}).get(str(zl[driver]))
        if pooled:
            hist += (f" 同市場合併（{pooled.get('n_sym', '?')} 檔，{names[driver]}同一區）：12 個月上漲 {pooled['hit']*100:.0f}%"
                     f"（基準 {pooled['base_hit']*100:.0f}%），平均比平時 {pooled['excess_mean']*100:+.0f} 個百分點。")
    if zmin is None:
        return {"stamp": "資料不足", "cls": "neutral", "title": "歷史不夠長，還算不出區間", "body": "需要至少 3.5 年日線才能畫五線譜。", "hist": hist}
    if zmin <= -2 or pe_zone is not None and pe_zone <= -1:
        return {"stamp": "便宜區", "cls": "buy",
                "title": "分批買進的好位置" if zmin <= -2 else "本益比在自己歷史的低檔，可以分批買",
                "body": ("價格落在自己長期趨勢帶的最下緣（或本益比在歷史最低四分之一）。回測裡這是唯一兩個市場、全市場都站得住的訊號："
                         "之後 12 個月平均比平時多賺（台股約 +20、美股約 +3～8 個百分點），上漲機率較高。分批、不要一次全進——便宜之後常常還會更便宜一陣子。"),
                "hist": hist}
    if zmin == -1:
        return {"stamp": "偏便宜", "cls": "buy-soft", "title": "可以開始分批買",
                "body": "價格在趨勢帶的下半部（−1σ～−2σ）。歷史上這裡之後 12 個月略好於平時，但沒有「很便宜」那麼明顯；照原本的計畫買即可。", "hist": hist}
    if zmax >= 2 or (pe_zone is not None and pe_zone >= 2):
        warn = ("台股本益比在自己歷史最高 10% 時買進，之後 12 個月平均少賺 15 個百分點、上漲機率掉 10 個百分點。" if (pe_zone is not None and pe_zone >= 2 and market == "TW")
                else "注意：這不代表會跌。對強勢股而言，美股「年線乖離很貴」之後 12 個月反而多賺 10 個百分點；歷史上在這裡賣出的人多半少賺。")
        return {"stamp": "偏貴區", "cls": "hold-warn", "title": "不追高；要用錢或部位太重時，優先從這裡減碼",
                "body": "價格在趨勢帶的最上緣。" + warn + " 已持有的照計畫持有，不要因為「很貴」就全部賣掉。", "hist": hist}
    if zmax == 1:
        return {"stamp": "略貴", "cls": "hold", "title": "照計畫持有，新資金不急著加碼",
                "body": "價格在趨勢帶上半部。歷史上這裡之後 12 個月略差於平時（台股五線譜約 −6 個百分點），但差距不大。", "hist": hist}
    return {"stamp": "合理區", "cls": "hold", "title": "照自己的計畫定期買、持有",
            "body": "價格在長期趨勢帶中間。區間本身沒有額外資訊；定期定額在這裡最省心。", "hist": hist}


def defense_backtest(df: pd.DataFrame, costs: Costs) -> dict:
    """Buy & hold vs month-end 10-month SMA filter on this instrument, full history and since 2019."""
    c = df["close"]
    if len(c) < 600:
        return {}
    me = df.index.to_period("M")
    is_me = np.r_[me[1:] != me[:-1], True]
    mclose = c[is_me]
    sma10 = mclose.rolling(10).mean()
    sig = (mclose > sma10).astype(float).where(sma10.notna(), 0.0)
    tgt = sig.reindex(df.index).ffill().fillna(0.0)
    out = {}
    for label, start in (("全期", None), ("2019 起", "2019-01-01")):
        sub = df if start is None else df[df.index >= start]
        if len(sub) < 400:
            continue
        bh = simulate(sub, pd.Series(1.0, index=sub.index), costs)
        fb = simulate(sub, tgt.reindex(sub.index), costs)
        out[label] = {"years": _f(len(sub) / 252, 1), "bh_cagr": _f(cagr(bh["equity"]), 4), "bh_dd": _f(max_drawdown(bh["equity"]), 4),
                      "faber_cagr": _f(cagr(fb["equity"]), 4), "faber_dd": _f(max_drawdown(fb["equity"]), 4),
                      "faber_time_in": _f(metrics(fb)["time_in_market"], 3)}
    return out


def chart_payload(df: pd.DataFrame, pe: pd.Series | None, tf: str = "1D", limit: int = 1500) -> dict:
    """Bars + bands for the chart (daily / weekly / monthly)."""
    c = df["close"]
    ch = Z.channel_bands(c)
    dv = Z.deviation_bands(c)
    bands = pd.DataFrame({"ch_dn2": ch["dn2"], "ch_dn1": ch["dn1"], "ch_center": ch["center"], "ch_up1": ch["up1"], "ch_up2": ch["up2"],
                          "dv_dn2": dv["dn2"], "dv_dn1": dv["dn1"], "dv_ma": dv["ma"], "dv_up1": dv["up1"], "dv_up2": dv["up2"],
                          "sma200": sma(c, 200)})
    if pe is not None and len(pe.dropna()) > 300:
        close_for_pe = df["close_raw"].where(df["close_raw"].notna(), c)
        pb = Z.pe_bands(close_for_pe, pe)
        for k in ("dn2", "dn1", "mid", "up1", "up2"):
            bands[f"pe_{k}"] = pb[k]
    frame = df[["open", "high", "low", "close", "volume"]].join(bands)
    if tf in ("1W", "1M"):
        rule = "W-FRI" if tf == "1W" else "ME"
        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        agg.update({k: "last" for k in bands.columns})
        frame = frame.resample(rule).agg(agg).dropna(subset=["close"])
    frame = frame.tail(limit)
    ts = ((frame.index - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).astype(int).tolist()
    def ser(col):
        v = frame[col]
        return [[t, (None if pd.isna(x) else round(float(x), 4))] for t, x in zip(ts, v)]
    bars = [{"time": t, "open": round(float(r.open), 4), "high": round(float(r.high), 4), "low": round(float(r.low), 4),
             "close": round(float(r.close), 4), "volume": float(r.volume) if pd.notna(r.volume) else 0.0}
            for t, r in zip(ts, frame.itertuples())]
    return {"tf": tf, "bars": bars, "lines": {k: ser(k) for k in bands.columns if k in frame.columns}}
