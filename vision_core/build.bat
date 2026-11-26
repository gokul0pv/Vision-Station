@echo off
rem Vision Core build: portable MSVC + CMake + Ninja (see docs\ADR-001).
rem   build.bat            release build into build\
rem   build.bat debug      debug build into build-debug\
setlocal
cd /d "%~dp0"
set "MSVC_DIR=%LOCALAPPDATA%\msvc-portable"
set "TOOLS=%LOCALAPPDATA%\vc-tools"
call "%MSVC_DIR%\setup_x64.bat" || exit /b 1
set "PATH=%TOOLS%\cmake-4.5.1-windows-x86_64\bin;%TOOLS%\ninja;%PATH%"
set "CFG=Release" & set "OUT=build"
if /i "%1"=="debug" (set "CFG=Debug" & set "OUT=build-debug")
if not exist "%OUT%\build.ninja" (
  cmake -S . -B "%OUT%" -G Ninja -DCMAKE_BUILD_TYPE=%CFG% -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl || exit /b 1
)
cmake --build "%OUT%" || exit /b 1
echo built %OUT%
