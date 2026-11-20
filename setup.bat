@echo off
rem One-time setup. Start ComfyUI first, then double-click this file.
setlocal
cd /d "%~dp0"

rem Use the Python that runs ComfyUI: it already has every package we need.
set "PY="
for /f "usebackq delims=" %%p in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\find_comfy_python.ps1"`) do set "PY=%%p"
if not defined PY (
  echo ComfyUI does not seem to be running on port 8188.
  echo Start ComfyUI, wait until it has finished loading, then run setup.bat again.
  echo.
  pause
  exit /b 1
)

"%PY%" "%~dp0setup.py" %*
echo.
pause
