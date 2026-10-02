@echo off
rem Gold/silver signal dashboard - Windows launcher (ASCII only on purpose)
cd /d "%~dp0"
echo Starting the gold/silver signal dashboard...

set "PY="
py -3 --version >nul 2>nul && set "PY=py -3"
if not defined PY python --version >nul 2>nul && set "PY=python"
if not defined PY goto nopython

if exist ".venv\Scripts\python.exe" goto run
echo First run: creating the Python environment and installing packages (1-2 minutes)...
%PY% -m venv .venv
if errorlevel 1 goto nopython

:run
".venv\Scripts\python.exe" -m pip install -q --disable-pip-version-check -r requirements.txt
".venv\Scripts\python.exe" run.py %*
pause
exit /b 0

:nopython
echo.
echo Python 3.10 or newer was not found on this computer.
echo 1. Download it from https://www.python.org/downloads/
echo 2. In the installer, tick "Add python.exe to PATH", then click Install Now.
echo 3. Double-click start_windows.bat again.
echo.
pause
exit /b 1
