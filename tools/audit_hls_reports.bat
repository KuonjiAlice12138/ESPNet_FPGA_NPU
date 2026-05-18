@echo off
setlocal
python "%~dp0audit_hls_reports.py" --strict %*
exit /b %ERRORLEVEL%
