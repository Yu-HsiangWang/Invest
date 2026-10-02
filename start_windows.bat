@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 金銀短線訊號儀表板 - 啟動中...
where py >nul 2>nul
if %errorlevel%==0 (set "PY=py -3") else (set "PY=python")
if not exist ".venv\Scripts\python.exe" (
  echo 第一次啟動：建立 Python 環境並安裝套件（約 1-2 分鐘）...
  %PY% -m venv .venv
  if errorlevel 1 (
    echo 找不到 Python。請先到 https://www.python.org/downloads/ 安裝 Python 3.10 以上，並勾選 Add Python to PATH。
    pause
    exit /b 1
  )
)
".venv\Scripts\python.exe" -m pip install -q --disable-pip-version-check -r requirements.txt
".venv\Scripts\python.exe" run.py %*
pause
