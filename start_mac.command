#!/bin/bash
cd "$(dirname "$0")"
echo "金銀短線訊號儀表板 - 啟動中..."
if ! command -v python3 >/dev/null 2>&1; then
  echo "找不到 python3，請先到 https://www.python.org/downloads/ 安裝 Python 3.10 以上。"
  read -r -p "按 Enter 關閉"
  exit 1
fi
if [ ! -x .venv/bin/python ]; then
  echo "第一次啟動：建立 Python 環境並安裝套件（約 1-2 分鐘）..."
  python3 -m venv .venv || exit 1
fi
.venv/bin/python -m pip install -q --disable-pip-version-check -r requirements.txt
.venv/bin/python run.py "$@"
