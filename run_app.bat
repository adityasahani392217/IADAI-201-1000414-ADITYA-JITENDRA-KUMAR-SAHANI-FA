@echo off
title SafeFall AI - Healthcare Monitoring System
echo ========================================================
echo Starting SafeFall AI Streamlit Web Application...
echo ========================================================
cd /d "%~dp0"

if exist ".venv\Scripts\streamlit.exe" (
    echo Using virtual environment: .venv\Scripts\streamlit.exe
    ".venv\Scripts\streamlit.exe" run app.py
) else if exist ".venv\Scripts\python.exe" (
    echo Using virtual environment: .venv\Scripts\python.exe
    ".venv\Scripts\python.exe" -m streamlit run app.py
) else (
    echo Using system Python Launcher py -3.13
    py -3.13 -m streamlit run app.py
)
pause
