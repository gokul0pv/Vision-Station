@echo off
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo  Python was not found. Run install.bat first.
    pause
    exit /b 1
)

python -m vision_station --classic
if errorlevel 1 (
    echo.
    echo  ========================================================
    echo  The application exited with an error. Checking the
    echo  installation to see whether something is missing...
    echo.
    python scripts\preflight.py
    echo.
    pause
)
