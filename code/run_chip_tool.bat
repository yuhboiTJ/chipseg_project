@echo off
REM Double-click this to launch the live microchip area tool on Windows.
REM Needs python on PATH and a webcam or USB capture device plugged in.

cd /d "%~dp0"
python scripts\live_demo.py %*
if errorlevel 1 (
    echo.
    echo the app exited with an error. press any key to close this window.
    pause >nul
)
