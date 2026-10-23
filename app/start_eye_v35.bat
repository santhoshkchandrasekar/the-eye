@echo off
REM The EYE v3.5 (COMPOUND split) on port 7867
cd /d "%~dp0"
set M2_PROJECT_DIR=%~dp0..\..\Milestone 3\M3 HOST
set PY=%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe
if not exist "%PY%" set PY=python
"%PY%" m2_eye_app_v35.py
