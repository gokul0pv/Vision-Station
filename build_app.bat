@echo off
cd /d "%~dp0"
rem Builds dist\VisionStation\VisionStation.exe and dist\VisionStation-<version>-win64.zip
python packaging\build.py %*
pause
