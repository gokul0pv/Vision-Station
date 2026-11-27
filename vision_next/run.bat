@echo off
rem Vision Next: build (first time, or after a change) and start.
cd /d "%~dp0"
set "PATH=%USERPROFILE%\.cargo\bin;%LOCALAPPDATA%\mingw-binutils\bin;%PATH%"
cargo build --release || (pause & exit /b 1)
start "" "target\release\vision-next.exe"
