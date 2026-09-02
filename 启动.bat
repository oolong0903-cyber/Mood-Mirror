@echo off
rem ================================================================
rem  Chinese Emotion Text Analysis Tool - launcher
rem  (ASCII-only on purpose: cmd reads .bat in the system codepage,
rem   so any non-ASCII bytes here would corrupt the commands.)
rem ================================================================
cd /d "%~dp0"

rem Prefer the venv that already has streamlit+pandas; else fall back
rem to whatever "python" resolves to on PATH.
set "PY=D:\UFS\Study\02 Code\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo.
echo  Starting Streamlit... the browser will open at:
echo      http://localhost:8501
echo  To stop it later: press Ctrl+C here, or just close this window.
echo.
"%PY%" -m streamlit run 20260902_app.py

echo.
echo  App exited.
pause
