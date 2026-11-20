@echo off
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo  Python was not found. Run install.bat first.
    pause
    exit /b 1
)

rem GPU or CPU is chosen from this PC's graphics card when the application
rem starts (vision_station\core\system_profile.py). To choose by hand:
rem   set VS_DEVICE=GPU    or    set VS_DEVICE=CPU

python -m vision_station
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
