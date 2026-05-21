@echo off
setlocal

set "PYTHON=D:\ESPNet\.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo ESPNet venv python not found: %PYTHON%
  exit /b 1
)

set "PYTHONDONTWRITEBYTECODE=1"
"%PYTHON%" "%~dp0eval_val_hw_outputs.py" %*
exit /b %ERRORLEVEL%
