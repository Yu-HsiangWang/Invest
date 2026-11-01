"""Does Kronos (Shi et al. 2025, "a foundation model for the language of financial markets",
github.com/shiyu-coder/Kronos, AAAI 2026) help the gold system? (docs/RESEARCH.md section 12.3)

Kronos is a decoder-only transformer pre-trained on K-lines from 45 exchanges. Given the last
`lookback` OHLCV bars it samples the next `pred_len` bars (open/high/low/close/volume); the
forecast is the mean of `sample_count` sampled paths. Three questions, each with a fixed yardstick:

  A. forecast skill on gold 15m bars: at ~600 random bars, is the sign of the forecast
     close change over 1 h / 2 h / 4 h right more often than chance and than "the last
     4 hours' direction continues"? (hit rate with a 95 % interval, Spearman correlation,
     actual move in ATR units conditional on the forecast direction)
  B. as a filter on the system's signals (18h + squeeze = 高效積極, 12h = 積極): do signals
     that Kronos agrees with earn more R than those it disagrees with?
  C. as a trade source: 高效積極 with the signals Kronos disagrees with removed, and with
     Kronos-approved 12h signals as the second position - the usual dev / val / test
     efficiency table (research/efficient_active.py) against the 高效積極 baseline.

Kronos needs torch and ~5-10 s per forecast on a 2-core CPU (no KV cache: every sampled bar
re-runs the whole context), so forecasts are cached in data/cache/kronos_preds.parquet and the
script can be stopped and resumed. The pre-trained weights are not reachable from every network:
pass local copies of the Hugging Face folders (config.json + model.safetensors).

    python research/kronos_check.py --kronos-src /path/to/Kronos --model-dir /path/Kronos-small \\
        --tokenizer-dir /path/Kronos-Tokenizer-base [--lookback 256 --pred-len 16 --samples 5]
    python research/kronos_check.py --kronos-src /path/to/Kronos --random-weights --limit 5   # pipeline smoke test
    python research/kronos_check.py --analyze-only                                             # tables from the cache
"""
from __future__ import annotations

import argparse
import json
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

CACHE = ROOT / "data" / "cache" / "kronos_preds.parquet"
OUT = ROOT / "results"
HORIZONS = (4, 8, 16)          # bars: 1 h, 2 h, 4 h
N_RANDOM = 600
SEED = 11


# ----------------------------------------------------------------------------- which bars

def pick_bars(H: A.Harness, n_random: int = N_RANDOM, lookback: int = 256, pred_len: int = 16) -> pd.DataFrame:
    """Bars to forecast at: every signal bar of the 18h+squeeze and 12h channels (2005-) plus a
    random sample of ordinary bars whose next pred_len bars have no weekend gap."""
    idx = H.df.index
    e, a = H.sys_sig(18, True), H.sys_sig(12, False)
    ok = np.zeros(H.n, dtype=bool)
    ok[lookback:H.n - pred_len - 1] = True
    ok &= np.asarray(idx >= pd.Timestamp("2005-01-01", tz="UTC"))
    gap = np.r_[np.asarray((idx[pred_len:] - idx[:-pred_len]) <= pd.Timedelta(hours=6)), np.zeros(pred_len, dtype=bool)]
    sig_bars = np.flatnonzero(((e != 0) | (a != 0)) & ok)
    rng = np.random.default_rng(SEED)
    pool = np.flatnonzero(ok & gap & (e == 0) & (a == 0))
    rand_bars = np.sort(rng.choice(pool, size=min(n_random, len(pool)), replace=False))
    rows = [{"i": int(i), "kind": "signal", "eff": int(e[i]), "a12": int(a[i])} for i in sig_bars]
    rows += [{"i": int(i), "kind": "random", "eff": 0, "a12": 0} for i in rand_bars]
    return pd.DataFrame(rows).drop_duplicates("i").sort_values("i").reset_index(drop=True)


# ----------------------------------------------------------------------------- forecasts

def load_kronos(src: str, model_dir: str | None, tok_dir: str | None, random_weights: bool):
    sys.path.insert(0, src)
    import torch
    from model import Kronos, KronosPredictor, KronosTokenizer
    torch.manual_seed(SEED)
    if random_weights:   # pipeline test only: a Kronos-small-sized model with random weights
        tok = KronosTokenizer(d_in=6, d_model=256, n_heads=4, ff_dim=512, n_enc_layers=4, n_dec_layers=4, ffn_dropout_p=0,
                              attn_dropout_p=0, resid_dropout_p=0, s1_bits=10, s2_bits=10, beta=1.0, gamma0=1.0, gamma=1.0,
                              zeta=1.0, group_size=4)
        model = Kronos(s1_bits=10, s2_bits=10, n_layers=8, d_model=512, n_heads=8, ff_dim=2048, ffn_dropout_p=0,
                       attn_dropout_p=0, resid_dropout_p=0, token_dropout_p=0, learn_te=False)
    else:
        tok = KronosTokenizer.from_pretrained(tok_dir)
        model = Kronos.from_pretrained(model_dir)
    tok.eval(); model.eval()
    return KronosPredictor(model, tok, device="cpu", max_context=512)


def forecast(H: A.Harness, bars: pd.DataFrame, predictor, lookback: int, pred_len: int, samples: int,
             limit: int | None = None, batch: int = 4) -> pd.DataFrame:
    """Forecast at each bar (the lookback ends with that bar's close); cached and resumable."""
    done = pd.read_parquet(CACHE) if CACHE.exists() else pd.DataFrame({"i": pd.Series(dtype="int64")})
    todo = bars[~bars["i"].isin(done["i"])]["i"].to_numpy()
    if limit:
        todo = todo[:limit]
    df = H.df
    cols = ["open", "high", "low", "close", "volume"]
    rows, t0 = [], time.time()
    for k in range(0, len(todo), batch):
        ii = todo[k:k + batch]
        dfs = [df.iloc[i - lookback + 1:i + 1][cols].reset_index(drop=True) for i in ii]
        xs = [pd.Series(df.index[i - lookback + 1:i + 1].tz_localize(None)) for i in ii]
        ys = [pd.Series(df.index[i + 1:i + 1 + pred_len].tz_localize(None)) for i in ii]
        preds = predictor.predict_batch(dfs, xs, ys, pred_len=pred_len, T=1.0, top_p=0.9, sample_count=samples, verbose=False)
        for i, p in zip(ii, preds):
            c0 = float(df["close"].iloc[i])
            r = {"i": int(i), "c0": c0}
            for h in HORIZONS:
                if h <= pred_len:
                    r[f"p{h}"] = float(p["close"].iloc[h - 1]) / c0 - 1
            r["p_hi"] = float(p["high"].max()) / c0 - 1
            r["p_lo"] = float(p["low"].min()) / c0 - 1
            r["p_last"] = float(p["close"].iloc[-1]) / c0 - 1
            rows.append(r)
        if (k // batch) % 10 == 0:
            el = time.time() - t0
            print(f"  {k + len(ii)}/{len(todo)} forecasts, {el / (k + len(ii)):.1f} s each, ~{el / (k + len(ii)) * (len(todo) - k - len(ii)) / 60:.0f} min left", flush=True)
            if rows:
                pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_parquet(CACHE)
    if rows:
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        done.to_parquet(CACHE)
    done["i"] = done["i"].astype(int)
    return done


# ----------------------------------------------------------------------------- analysis

def period_of(ts: pd.Series) -> pd.Series:
    out = pd.Series("test", index=ts.index)
    out[ts < pd.Timestamp("2021-01-01", tz="UTC")] = "val"
    out[ts < pd.Timestamp("2017-01-01", tz="UTC")] = "dev"
    return out


def skill_table(H: A.Harness, P: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """A: forecast skill at the random bars (and at the signal bars, for reference)."""
    df, c = H.df, H.c
    atr = H.atr1
    P = P.copy(); P["i"] = P["i"].astype(int)
    P = P.merge(bars, on="i")
    P["t"] = df.index[P["i"].to_numpy()]
    P["period"] = period_of(P["t"])
    rows = []
    for kind in ("random", "signal"):
        X = P[P["kind"] == kind]
        for per in ("dev", "val", "test", "all"):
            Y = X if per == "all" else X[X["period"] == per]
            if len(Y) < 20:
                continue
            for h in HORIZONS:
                if f"p{h}" not in Y:
                    continue
                i = Y["i"].to_numpy()
                act = c[i + h] / c[i] - 1
                act_atr = (c[i + h] - c[i]) / atr[i]
                pred = Y[f"p{h}"].to_numpy()
                mom = c[i] - c[np.maximum(i - h, 0)]
                m = act != 0
                hit = float((np.sign(pred[m]) == np.sign(act[m])).mean())
                se = np.sqrt(hit * (1 - hit) / m.sum())
                mom_hit = float((np.sign(mom[m]) == np.sign(act[m])).mean())
                rho = float(pd.Series(pred).corr(pd.Series(act), method="spearman"))
                up, dn = pred > 0, pred < 0
                big = np.abs(pred) >= np.quantile(np.abs(pred), 0.67)
                rows.append({"bars": kind, "period": per, "horizon_h": h / 4, "n": int(m.sum()),
                             "hit%": round(100 * hit, 1), "hit_lo%": round(100 * (hit - 1.96 * se), 1), "hit_hi%": round(100 * (hit + 1.96 * se), 1),
                             "momentum_hit%": round(100 * mom_hit, 1), "spearman": round(rho, 3),
                             "act_ATR_if_up": round(float(act_atr[up].mean()), 3) if up.any() else None,
                             "act_ATR_if_down": round(float(act_atr[dn].mean()), 3) if dn.any() else None,
                             "signed_ATR_top3rd": round(float((np.sign(pred) * act_atr)[big].mean()), 3),
                             "signed_ATR_all": round(float((np.sign(pred) * act_atr).mean()), 3),
                             "mean_|act|%": round(100 * float(np.abs(act).mean()), 3)})
    return pd.DataFrame(rows)


def signal_tables(H: A.Harness, P: pd.DataFrame, bars: pd.DataFrame, pred_len: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """B: R of the system's trades split by Kronos agreement; C: back-tests with Kronos as filter / feed."""
    e, a = H.sys_sig(18, True), H.sys_sig(12, False)
    col = f"p{min(16, pred_len)}"
    P = P.copy(); P["i"] = P["i"].astype(int)
    P = P.merge(bars, on="i")
    kdir = pd.Series(np.sign(P[col].to_numpy()).astype(int), index=P["i"].to_numpy())
    kmag = pd.Series((P[col].to_numpy() * P["c0"].to_numpy()) / H.atr1[P["i"].to_numpy()], index=P["i"].to_numpy())
    known = np.zeros(H.n, dtype=bool); known[kdir.index.to_numpy()] = True
    K = np.zeros(H.n, dtype=int); K[kdir.index.to_numpy()] = kdir.to_numpy()
    M = np.zeros(H.n); M[kmag.index.to_numpy()] = kmag.to_numpy()

    trades = {"高效積極 (18h+收窄)": H.run2(e),
              "積極 (12h)": H._trim(E.run_variant(H.df, a, a, H.atr1, H.plan, H.sp, H.sl, H.clock, "any", 2, False))}
    rows = []
    for name, tr in trades.items():
        tr = tr[tr["sig_i"].isin(kdir.index)].copy()
        tr["agree"] = np.where(K[tr["sig_i"]] == tr["dir"], "同向", np.where(K[tr["sig_i"]] == 0, "無方向", "反向"))
        tr["mag"] = np.abs(M[tr["sig_i"]])
        tr["period"] = period_of(tr["entry_time"])
        for per in ("dev", "val", "test", "all"):
            X = tr if per == "all" else tr[tr["period"] == per]
            for g, Y in [("全部", X)] + [(k, v) for k, v in X.groupby("agree")]:
                if len(Y) >= 5:
                    r = Y["R"].to_numpy()
                    rows.append({"signals": name, "period": per, "group": g, "n": len(Y), "avgR": round(float(r.mean()), 3),
                                 "win%": round(100 * float((r > 0).mean()), 1), "totR": round(float(r.sum()), 1)})
            big = X[(X["agree"] == "同向") & (X["mag"] >= 0.5)]
            if len(big) >= 5:
                rows.append({"signals": name, "period": per, "group": "同向且預測 ≥ 0.5 ATR", "n": len(big), "avgR": round(float(big["R"].mean()), 3),
                             "win%": round(100 * float((big["R"] > 0).mean()), 1), "totR": round(float(big["R"].sum()), 1)})
    split = pd.DataFrame(rows)

    def keep(sig, cond):
        return np.where(cond, sig, 0).astype(np.int64)
    agree_e = keep(e, (K == e) & known)
    agree_a = keep(a, (K == a) & known)
    strong_a = keep(a, (K == a) & known & (np.abs(M) >= 0.5))
    notdis_e = keep(e, (K != -e) & known)
    bt = [A.row("baseline 高效積極", H.run2(e)),
          A.row("高效積極，只做 Kronos 同向的訊號", H.run2(agree_e)),
          A.row("高效積極，去掉 Kronos 反向的訊號", H.run2(notdis_e)),
          A.row("第 2 張 = 12h 訊號 (不看 Kronos)", H.run2(e, keep(a, known))),
          A.row("第 2 張 = Kronos 同向的 12h 訊號", H.run2(e, agree_a)),
          A.row("第 2 張 = Kronos 同向且 ≥ 0.5 ATR 的 12h 訊號", H.run2(e, strong_a)),
          A.row("第 1、2 張都可用 Kronos 同向的 12h 訊號", H.run2(agree_a, agree_a)),
          A.row("baseline 積極 (12h, 不冷卻)", H._trim(E.run_variant(H.df, a, a, H.atr1, H.plan, H.sp, H.sl, H.clock, "any", 2, False)))]
    return split, pd.DataFrame(bt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kronos-src", default=None, help="path of the cloned Kronos repo (its model/ package)")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--tokenizer-dir", default=None)
    ap.add_argument("--random-weights", action="store_true")
    ap.add_argument("--lookback", type=int, default=256)
    ap.add_argument("--pred-len", type=int, default=16)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)

    H = A.Harness(E.load())
    bars = pick_bars(H, lookback=args.lookback, pred_len=args.pred_len)
    print(f"{len(bars)} bars to forecast ({(bars['kind'] == 'signal').sum()} signal bars, {(bars['kind'] == 'random').sum()} random)")
    if not args.analyze_only:
        predictor = load_kronos(args.kronos_src, args.model_dir, args.tokenizer_dir, args.random_weights)
        P = forecast(H, bars, predictor, args.lookback, args.pred_len, args.samples, args.limit)
    else:
        P = pd.read_parquet(CACHE)
    print(f"{len(P)} forecasts in the cache")
    skill = skill_table(H, P, bars)
    print(skill.to_string(index=False))
    split, bt = signal_tables(H, P, bars, args.pred_len)
    print(split.to_string(index=False))
    OUT.mkdir(exist_ok=True)
    tag = "_random" if args.random_weights else ""
    skill.to_csv(OUT / f"XAUUSD_kronos_skill{tag}.csv", index=False)
    split.to_csv(OUT / f"XAUUSD_kronos_signals{tag}.csv", index=False)
    bt.to_csv(OUT / f"XAUUSD_kronos_backtest{tag}.csv", index=False)
    meta = {"lookback": args.lookback, "pred_len": args.pred_len, "samples": args.samples, "n_forecasts": int(len(P)),
            "model_dir": args.model_dir, "random_weights": args.random_weights}
    (OUT / f"XAUUSD_kronos_meta{tag}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
