"""Start the gold / silver signal dashboard.

    python run.py                 # live mode (free real-time data, no API key)
    python run.py --replay        # replay recent history (practice / weekends)
    python run.py --only XAUUSD   # only one instrument
    python run.py --port 8800     # use another port
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
import webbrowser
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def ensure_replay_data(sym: str, days: int = 260) -> Path:
    """Replay uses the 15m bars saved by scripts/validate_recent.py, or downloads them."""
    from goldsignal.instruments import INSTRUMENTS
    p = ROOT / "data" / f"dukascopy_{sym}_15m.csv.gz"
    if p.exists():
        return p
    from goldsignal.datafeed import Dukascopy, to_m15
    inst = INSTRUMENTS[sym]
    print(f"第一次使用回放：下載 {inst.name} 最近 {days} 天的歷史資料 ...")
    feed = Dukascopy(ROOT / "data" / "cache" / "dukascopy")
    today = datetime.now(timezone.utc).date()
    m1 = feed.minutes(inst.code, today - timedelta(days=days), today,
                      progress=lambda k, n: print(f"  {k}/{n}", end="\r", flush=True))
    m15 = to_m15(m1)
    p.parent.mkdir(parents=True, exist_ok=True)
    m15.to_csv(p)
    print()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--only", default=None, help="XAUUSD or XAGUSD")
    ap.add_argument("--replay", action="store_true", help="replay history instead of live data")
    ap.add_argument("--replay-csv", action="append", default=[], help="SYM=path (testing)")
    ap.add_argument("--replay-start", default=None, help="e.g. 2026-08-01")
    ap.add_argument("--speed", type=float, default=1.5, help="replay: seconds per 15m bar")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import uvicorn

    from goldsignal.engine import Hub
    from goldsignal.instruments import INSTRUMENTS
    from goldsignal.server import create_app

    symbols = [args.only] if args.only else list(INSTRUMENTS)
    replay_csvs = None
    if args.replay_csv:
        replay_csvs = dict(x.split("=", 1) for x in args.replay_csv)
        replay_csvs = {k: Path(v) for k, v in replay_csvs.items()}
        symbols = [s for s in symbols if s in replay_csvs]
    elif args.replay:
        replay_csvs = {s: ensure_replay_data(s) for s in symbols}
    hub = Hub(ROOT, symbols=symbols, replay_csvs=replay_csvs, replay_start=args.replay_start, replay_speed=args.speed)
    app = create_app(hub)
    url = f"http://{args.host}:{args.port}/"
    print(f"\n  金銀短線訊號儀表板 → {url}\n  (關閉這個視窗即可停止)\n")
    if not args.no_browser:
        threading.Thread(target=lambda: (time.sleep(1.5), webbrowser.open(url)), daemon=True).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
