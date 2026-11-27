@echo off
rem Builds vsx_cv.dll (the OpenCV bridge for Vision Next) with MSVC.
rem   MSVC_DIR   folder made by portable-msvc (holds setup_x64.bat)
rem   OPENCV_DIR the official OpenCV 4.10.0 Windows build's "opencv\build" folder
rem Output: ..\runtime\vsx_cv.dll, with opencv_world4100.dll copied beside it.
setlocal
cd /d "%~dp0"
if "%MSVC_DIR%"=="" (echo set MSVC_DIR & exit /b 1)
if "%OPENCV_DIR%"=="" (echo set OPENCV_DIR & exit /b 1)
call "%MSVC_DIR%\setup_x64.bat" || exit /b 1
set "OCV_LIB=%OPENCV_DIR%\x64\vc16\lib"
set "OCV_BIN=%OPENCV_DIR%\x64\vc16\bin"
cl /nologo /O2 /EHsc /MD /std:c++17 /LD /I "%OPENCV_DIR%\include" vsx_cv.cpp ^
   /Fe:vsx_cv.dll /link /LIBPATH:"%OCV_LIB%" opencv_world4100.lib || exit /b 1
if not exist ..\runtime mkdir ..\runtime
copy /y vsx_cv.dll ..\runtime\ >nul
copy /y "%OCV_BIN%\opencv_world4100.dll" ..\runtime\ >nul
del /q vsx_cv.obj vsx_cv.exp vsx_cv.lib 2>nul
echo built ..\runtime\vsx_cv.dll
