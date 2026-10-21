@echo off
cd /d "%~dp0"
where python >nul 2>&1
if errorlevel 1 (
    echo  Python was not found on the PATH.
    echo.
    echo  Install Python 3.10 or newer from python.org, and tick
    echo  "Add python.exe to PATH" on the first screen of the installer.
    echo.
    pause
    exit /b 1
)
python install.py %*
pause
