@echo off
REM ---------------------------------------------------------------------------
REM  USD/CRC daily-signal launcher for Windows Task Scheduler.
REM
REM  Uses the project virtual environment if one exists at ..\.venv, otherwise
REM  falls back to whatever "python" is on PATH. Runs the notifier and lets it
REM  do its own logging (production\logs\notifier.log); this .bat only captures
REM  a crash-level launch log so a broken venv is still visible.
REM
REM  Register it with Register-DailySignalTask.ps1, or point a Task Scheduler
REM  "Start a program" action straight at this file.
REM ---------------------------------------------------------------------------
setlocal
set "HERE=%~dp0"
set "PROJ=%HERE%.."
set "PYEXE=%PROJ%\.venv\Scripts\python.exe"
if not exist "%PYEXE%" set "PYEXE=python"

REM Pass any arguments straight through (e.g. --channel popup, a date, --no-refresh).
"%PYEXE%" "%HERE%run_notifier.py" %* 1>>"%HERE%logs\launch.log" 2>&1
exit /b %ERRORLEVEL%
