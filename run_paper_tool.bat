@echo off
setlocal

cd /d "%~dp0"

if exist "..\.venv\Scripts\python.exe" (
  set "PYTHON=..\.venv\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
) else (
  echo 未找到虚拟环境 Python。
  echo 请先在项目根目录执行：
  echo python -m venv .venv
  echo .venv\Scripts\activate
  echo pip install -r paper_ai_tool\requirements.txt
  echo.
  pause
  exit /b 1
)

echo AI 论文处理工具
echo 当前目录：%CD%
echo.
"%PYTHON%" cli.py

echo.
pause
