@echo off
setlocal EnableExtensions
chcp 65001 >nul
title AI Paper Tool

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo AI Paper Tool Windows Launcher
echo Script dir: %SCRIPT_DIR%
echo.

call :find_python
if not defined PYTHON_EXE (
  echo [ERROR] Cannot find virtual environment Python.
  echo.
  echo Checked:
  echo   %SCRIPT_DIR%..\.venv\Scripts\python.exe
  echo   %SCRIPT_DIR%.venv\Scripts\python.exe
  echo.
  echo Please run these commands from the project root:
  echo   python -m venv .venv
  echo   .venv\Scripts\activate
  echo   pip install -r paper_ai_tool\requirements.txt
  echo.
  pause
  exit /b 1
)

echo Python: %PYTHON_EXE%
echo.
echo Drag a PDF file into this window when prompted, then press Enter.
echo.

"%PYTHON_EXE%" cli.py
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo [ERROR] Program exited with code %EXIT_CODE%.
  echo The error output above should show the cause.
  echo.
  pause
  exit /b %EXIT_CODE%
)

echo Done.
echo.
pause
exit /b 0

:find_python
if defined PAPER_AI_TOOL_PYTHON (
  if exist "%PAPER_AI_TOOL_PYTHON%" (
    set "PYTHON_EXE=%PAPER_AI_TOOL_PYTHON%"
    exit /b 0
  )
)
if exist "%SCRIPT_DIR%..\.venv\Scripts\python.exe" (
  set "PYTHON_EXE=%SCRIPT_DIR%..\.venv\Scripts\python.exe"
  exit /b 0
)
if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
  set "PYTHON_EXE=%SCRIPT_DIR%.venv\Scripts\python.exe"
  exit /b 0
)
exit /b 0
