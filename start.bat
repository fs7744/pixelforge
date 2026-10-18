@echo off
rem PixelForge MVP · 一键启动（Windows）
setlocal
cd /d %~dp0
set PY=C:\Users\fs774\.workbuddy\binaries\python\envs\pixelforge\Scripts\python.exe
if not exist "%PY%" (
  echo [ERROR] 未找到虚拟环境，请先执行：python -m venv %USERPROFILE%\.workbuddy\binaries\python\envs\pixelforge
  pause
  exit /b 1
)
echo Starting PixelForge on http://127.0.0.1:8787 ...
start "" http://127.0.0.1:8787
"%PY%" -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8787
