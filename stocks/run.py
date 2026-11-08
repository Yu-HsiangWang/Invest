"""Start the Taiwan / US tech-stock zone dashboard.

    python run.py                  # live: Yahoo / TWSE / SEC / news, no API key needed
    python run.py --offline        # bundled history only (no network); good for a first look
    python run.py --market TW      # one market only
    python run.py --port 8801
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8801)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--market", default=None, help="TW or US (default: both)")
    ap.add_argument("--offline", action="store_true", help="use the bundled history only")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import uvicorn

    from stocksignal.engine import Hub
    from stocksignal.server import create_app

    hub = Hub(ROOT, offline=args.offline, markets=[args.market] if args.market else None)
    app = create_app(hub)
    url = f"http://{args.host}:{args.port}/"
    print(f"\n  科技股買賣區間儀表板 → {url}\n  (關閉這個視窗即可停止)\n")
    if not args.no_browser:
        threading.Thread(target=lambda: (time.sleep(1.5), webbrowser.open(url)), daemon=True).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
