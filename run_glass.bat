@echo off
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo  Python was not found. Run install.bat first.
    pause
    exit /b 1
)

rem Run trained models on the GPU when one is usable (falls back to CPU).
rem Delete this line on a line PC that should stay on the CPU.
if not defined VS_DEVICE set VS_DEVICE=GPU

python -m vision_station.glass
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
