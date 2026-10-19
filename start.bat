@echo off
rem ===========================================================================
rem  PixelForge launcher for Windows - just double-click this file.
rem
rem  First run creates .venv, installs dependencies, builds the frontend,
rem  then starts the server. Later runs reuse everything and start in seconds.
rem
rem  Options (run from cmd, e.g. "start.bat --port 9000"):
rem    --no-browser   start without opening the browser
rem    --port N       listen on a custom port (auto-shifts if busy)
rem    --reload       dev mode, auto-reload backend on code change
rem    --setup        only prepare the environment, do not start
rem    --check        only run a self-check and exit
rem    --gpu          also install AI inference deps (torch/diffusers, ~2.9GB)
rem
rem  Environment variables: PF_PYTHON, PF_VENV, PF_PIP_INDEX, TORCH_INDEX, HOST, PORT
rem ===========================================================================
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"
set "ROOT=%CD%"
if defined PF_VENV (set "VENV=%PF_VENV%") else (set "VENV=%ROOT%\.venv")
if not defined HOST set "HOST=127.0.0.1"
if not defined PORT set "PORT=8787"
set "OPEN_BROWSER=1"
set "RELOAD="
set "MODE=run"
set "INSTALL_GPU=0"
set "PY="
if not defined TORCH_INDEX set "TORCH_INDEX=https://download.pytorch.org/whl/cu124"
set "REMBG_STAMP=%VENV%\.pf-rembg-skip"

:parse
if "%~1"=="" goto parsed
if /i "%~1"=="--no-browser" ( set "OPEN_BROWSER=0" & shift & goto parse )
if /i "%~1"=="--reload"     ( set "RELOAD=--reload" & shift & goto parse )
if /i "%~1"=="--setup"      ( set "MODE=setup" & shift & goto parse )
if /i "%~1"=="--check"      ( set "MODE=check" & set "OPEN_BROWSER=0" & shift & goto parse )
if /i "%~1"=="--gpu"        ( set "INSTALL_GPU=1" & shift & goto parse )
if /i "%~1"=="--port" (
  if "%~2"=="" ( echo [ERROR] --port needs a value & goto failed )
  set "PORT=%~2" & shift & shift & goto parse
)
if /i "%~1"=="--host" (
  if "%~2"=="" ( echo [ERROR] --host needs a value & goto failed )
  set "HOST=%~2" & shift & shift & goto parse
)
if /i "%~1"=="-h"    goto usage
if /i "%~1"=="--help" goto usage
echo [ERROR] Unknown option: %~1
goto usage

:parsed
echo.
echo   PixelForge
echo   ----------
echo.

rem ---------- 1. Python ----------
rem Upper bound matters more than the lower one: on 3.14/3.15 some deps
rem (PyYAML, pulled in by uvicorn[standard]) have no prebuilt wheels yet,
rem so pip tries to compile from source and dies on "requires MSVC".
if defined PF_PYTHON (
  if exist "%PF_PYTHON%" set "PY=%PF_PYTHON%"
)
if not defined PY if exist "%VENV%\Scripts\python.exe" set "PY=%VENV%\Scripts\python.exe"
if not defined PY if exist "%VENV%\bin\python.exe" set "PY=%VENV%\bin\python.exe"

if defined PY (
  "%PY%" -c "import sys;sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,13) else 1)" >nul 2>nul
  if errorlevel 1 (
    rem Out-of-range environment left by an earlier run: move it aside and rebuild.
    echo        Existing .venv is outside the supported Python 3.10-3.13 range.
    echo        Rebuilding it now - the old folder is kept as .venv.bak.
    if exist "%VENV%.bak" rd /s /q "%VENV%.bak"
    move "%VENV%" "%VENV%.bak" >nul 2>nul
    set "PY="
  )
)

if defined PY (
  echo [1/4] Python ......... OK
  goto deps
)

rem Prefer a version with published wheels before falling back to whatever py defaults to.
set "BASE="
where py >nul 2>nul
if not errorlevel 1 (
  for %%v in (3.13 3.12 3.11 3.10) do (
    if not defined BASE (
      py -%%v -c "import sys;sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,13) else 1)" >nul 2>nul
      if not errorlevel 1 set "BASE=py -%%v"
    )
  )
  if not defined BASE (
    py -3 -c "import sys;sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,13) else 1)" >nul 2>nul
    if not errorlevel 1 set "BASE=py -3"
  )
)
if not defined BASE (
  where python >nul 2>nul
  if not errorlevel 1 (
    python -c "import sys;sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,13) else 1)" >nul 2>nul
    if not errorlevel 1 set "BASE=python"
  )
)
if not defined BASE (
  rem Nothing in range. Take any 3.10+ and warn, instead of hard-failing.
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3 -c "import sys;sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>nul
    if not errorlevel 1 set "BASE=py -3"
  )
)
if not defined BASE (
  where python >nul 2>nul
  if not errorlevel 1 (
    python -c "import sys;sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>nul
    if not errorlevel 1 set "BASE=python"
  )
)
if not defined BASE (
  echo [ERROR] Python 3.10+ not found.
  echo         Install Python 3.12 or 3.13 from https://www.python.org/downloads/
  echo         and tick "Add python.exe to PATH" during setup.
  goto failed
)

echo [1/4] Creating virtual environment .venv ...
%BASE% -m venv "%VENV%"
if errorlevel 1 (
  echo [ERROR] Failed to create the virtual environment.
  goto failed
)
if exist "%VENV%\Scripts\python.exe" set "PY=%VENV%\Scripts\python.exe"
if not defined PY (
  echo [ERROR] Virtual environment looks incomplete: %VENV%
  goto failed
)
echo        Python ......... OK

rem ---------- 2. Python dependencies ----------
:deps
echo [2/4] Checking Python dependencies ...
"%PY%" -c "import fastapi, uvicorn, PIL, numpy, cv2" >nul 2>nul
if errorlevel 1 (
  echo        Installing, this takes 1-3 minutes on first run ...
  "%PY%" -m pip install --disable-pip-version-check -U pip >nul 2>nul
  "%PY%" -m pip install --disable-pip-version-check -r "%ROOT%\backend\requirements.txt"
  if errorlevel 1 (
    echo        Retrying with the Aliyun mirror ...
    "%PY%" -m pip install --disable-pip-version-check -r "%ROOT%\backend\requirements.txt" -i https://mirrors.aliyun.com/pypi/simple
    if errorlevel 1 (
      rem Plain uvicorn instead of uvicorn[standard]: skips PyYAML/uvloop,
      rem which are the packages that need a C compiler on brand-new Python.
      echo        Retrying without optional extras ^(core packages only^) ...
      "%PY%" -m pip install --disable-pip-version-check fastapi uvicorn pillow numpy python-multipart aiofiles httpx opencv-python-headless -i https://mirrors.aliyun.com/pypi/simple
    )
  )
  "%PY%" -c "import fastapi, uvicorn, PIL, numpy, cv2" >nul 2>nul
  if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    echo         If your Python is newer than 3.13, install Python 3.12 or 3.13,
    echo         delete the .venv folder, and run start.bat again.
    goto failed
  )
)
echo        Dependencies ... OK

rem ---------- 2.5. AI inference deps (optional, ~2.9GB) ----------
rem torch is deliberately kept out of requirements.txt: it is a 2.9GB download
rem that only the AI tools need. Without it, hitting Generate fails with
rem "No module named 'torch'", so we surface that clearly here instead.
"%PY%" -c "import torch" >nul 2>nul
if not errorlevel 1 (
  echo        AI deps ....... OK
  goto rembg
)
if not "%INSTALL_GPU%"=="1" (
  echo        [warn] PyTorch not installed - AI generation is disabled.
  echo            Deterministic tools ^(matte / pixelize / seamless^) still work.
  echo            Run "start.bat --gpu" to install it ^(~2.9GB^).
  goto rembg
)
echo [2.5/4] Installing AI inference deps ^(torch ~2.9GB, 5-15 min^) ...
echo        torch source: %TORCH_INDEX%
"%PY%" -m pip install --disable-pip-version-check -U pip >nul 2>nul
"%PY%" -m pip install torch --index-url "%TORCH_INDEX%" --retries 5 --timeout 60
if errorlevel 1 (
  echo [ERROR] torch install failed.
  echo         Check network, free disk space, and an NVIDIA driver with CUDA 12.x.
  goto failed
)
rem torchvision 0.21.0 pairs with torch 2.6.0; diffusers needs it for the
rem Qwen2VL video processor config shipped with Qwen-Image-Edit.
"%PY%" -m pip install torchvision==0.21.0 --index-url "%TORCH_INDEX%" --retries 5 --timeout 60
"%PY%" -m pip install --disable-pip-version-check diffusers transformers accelerate safetensors huggingface_hub -i https://mirrors.aliyun.com/pypi/simple
if errorlevel 1 (
  echo [ERROR] diffusers install failed. Check your network and retry.
  goto failed
)
"%PY%" -m pip install --disable-pip-version-check peft -i https://mirrors.aliyun.com/pypi/simple
"%PY%" -m pip install --disable-pip-version-check torchao==0.16.0 -i https://mirrors.aliyun.com/pypi/simple
"%PY%" -c "import torch; print('        torch', torch.__version__, 'CUDA:', torch.cuda.is_available())"
if errorlevel 1 (
  echo [ERROR] torch installed but cannot be imported.
  goto failed
)

:rembg
rem Optional deep-matting stack. A failure here only downgrades matting quality.
if not exist "%REMBG_STAMP%" (
  "%PY%" -c "import rembg" >nul 2>nul
  if errorlevel 1 (
    echo        Optional: installing rembg for better matting ...
    "%PY%" -m pip install --disable-pip-version-check -q -r "%ROOT%\backend\requirements-optional.txt" >nul 2>nul
    if errorlevel 1 (
      echo        [warn] rembg unavailable - matting falls back to GrabCut.
      echo. > "%REMBG_STAMP%"
    )
  )
)

rem ---------- 3. Frontend ----------
echo [3/4] Checking frontend ...
if exist "%ROOT%\frontend\dist\index.html" (
  echo        Frontend ....... OK
  goto models
)
where npm >nul 2>nul
if errorlevel 1 (
  echo        [warn] Node.js/npm not found - the web UI will be unavailable.
  echo            API and /docs still work. Install Node.js 18+ to build the UI.
  goto models
)
pushd "%ROOT%\frontend"
if exist "node_modules" (
  echo        Reusing node_modules
) else (
  echo        Installing frontend dependencies ...
  if exist "package-lock.json" ( call npm ci --no-audit --no-fund ) else ( call npm install --no-audit --no-fund )
  if errorlevel 1 (
    popd
    echo [ERROR] npm install failed. Check your network and retry.
    goto failed
  )
)
echo        Building frontend ...
call npm run build
if errorlevel 1 (
  popd
  echo [ERROR] Frontend build failed. Try manually: cd frontend ^&^& npm install ^&^& npm run build
  goto failed
)
popd
echo        Frontend ....... built

rem ---------- 4. Models ----------
:models
echo [4/4] Checking local models ...
set "HAS_MODEL=0"
if not exist "%ROOT%\models" goto no_models
for /d %%D in ("%ROOT%\models\*") do set "HAS_MODEL=1"
if exist "%ROOT%\models\*.safetensors" set "HAS_MODEL=1"
if "%HAS_MODEL%"=="1" (
  echo        Models ......... found
) else (
  echo        [warn] No model yet - AI generation is disabled.
  echo            Deterministic tools ^(matte / pixelize / seamless^) still work.
  echo            Download Z-Image: bash backend/fetch_zimage.sh
)
goto ports

:no_models
echo        [warn] No model yet - AI generation is disabled.
echo            Deterministic tools ^(matte / pixelize / seamless^) still work.
echo            Download Z-Image: bash backend/fetch_zimage.sh

rem ---------- 5. Free port ----------
:ports
set /a TRIES=0
:findport
netstat -ano | findstr /c:":%PORT% " >nul 2>nul
if not errorlevel 1 (
  set /a PORT+=1
  set /a TRIES+=1
  if !TRIES! GEQ 30 (
    echo [ERROR] No free port found in range. Use --port to pick another one.
    goto failed
  )
  goto findport
)
if not "%TRIES%"=="0" echo        Port busy, shifted to %PORT%
echo.

if /i "%MODE%"=="check" (
  "%PY%" -c "import cv2; print('        OpenCV ......... OK')" 2>nul || echo        [warn] OpenCV missing
  "%PY%" -c "import rembg; print('        rembg .......... OK')" 2>nul || echo        [warn] rembg missing ^(optional^)
  "%PY%" -c "import torch; print('        torch .......... OK')" 2>nul || echo        [warn] torch missing - AI generation disabled ^(run start.bat --gpu^)
  "%PY%" -c "import diffusers; print('        diffusers ...... OK')" 2>nul || echo        [warn] diffusers missing - AI generation disabled ^(run start.bat --gpu^)
  echo.
  echo   Self-check done. Port %PORT% is free.
  goto done
)
if /i "%MODE%"=="setup" (
  echo   Environment is ready. Run start.bat to launch.
  goto done
)

echo   Serving at http://%HOST%:%PORT%
echo   Press Ctrl+C to stop.
echo.

if "%OPEN_BROWSER%"=="1" (
  start "" /b powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Milliseconds 2000; Start-Process 'http://%HOST%:%PORT%'" >nul 2>nul
)

"%PY%" -m uvicorn backend.app.main:app --host %HOST% --port %PORT% %RELOAD%
if errorlevel 1 (
  echo.
  echo [ERROR] Server exited unexpectedly. See the message above.
  goto failed
)
goto done

:usage
echo.
echo   Usage: start.bat [--no-browser] [--port N] [--reload] [--setup] [--check] [--gpu]
echo.
echo   Just double-click start.bat for the default behavior.
echo.
goto done

:failed
echo.
pause
exit /b 1

:done
endlocal
exit /b 0
