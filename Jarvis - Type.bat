@echo off
rem Double-click to start Jarvis (type instead of talking).
title Jarvis
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Jarvis isn't installed in this folder yet. See "Setup on Windows" in README.md.
  pause
  exit /b 1
)
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set PYTHONWARNINGS=ignore
".venv\Scripts\python.exe" -m jarvis --mode text %*
if errorlevel 1 pause
