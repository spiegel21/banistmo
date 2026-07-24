@echo off
REM ---------------------------------------------------------------------------
REM  Double-click this file to show the USD/CRC signal RIGHT NOW — no scheduler,
REM  no terminal, no typing. It refreshes the data, computes the next-session
REM  signal, e-mails/pops it per config, prints the sheet in this window, and
REM  waits so you can read it.
REM
REM  Want a specific channel? Either edit config.py, or make a copy of this file
REM  and add an argument to the line below, e.g.:  ...run_notifier.py --channel popup
REM
REM  Tip: right-click this file -> "Send to" -> "Desktop (create shortcut)" to
REM  get a clickable icon on your desktop (rename it, give it an icon if you like).
REM ---------------------------------------------------------------------------
setlocal
set "HERE=%~dp0"
set "PROJ=%HERE%.."
set "PYEXE=%PROJ%\.venv\Scripts\python.exe"
if not exist "%PYEXE%" set "PYEXE=python"

title USD/CRC Daily Signal
"%PYEXE%" "%HERE%run_notifier.py" %*
set "RC=%ERRORLEVEL%"

echo.
if not "%RC%"=="0" echo [!] Something went wrong (exit %RC%) — see production\logs\notifier.log
echo Press any key to close this window . . .
pause >nul
exit /b %RC%
