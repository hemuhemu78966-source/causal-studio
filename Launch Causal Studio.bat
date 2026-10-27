@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run the setup instructions in README.md first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" app.py
pause
