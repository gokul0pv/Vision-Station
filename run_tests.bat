@echo off
cd /d "%~dp0"
python -m pytest tests.py tests_vp.py tests_package.py -q
pause
