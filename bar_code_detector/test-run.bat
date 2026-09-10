@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "VENV_DIR=.venv"
set "PYTHON_BIN=%VENV_DIR%\Scripts\python.exe"
set "REQ_FILE=requirements.txt"
set "MAIN_FILE=main.py"

where uv >nul 2>&1
if errorlevel 1 (
    echo [ERROR] uv is not installed or not on PATH.
    echo Install uv from https://docs.astral.sh/uv/ and re-run this script.
    exit /b 1
)

if not exist "%PYTHON_BIN%" (
    echo [INFO] Creating virtual environment with uv at %VENV_DIR% ...
    uv venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        exit /b 1
    )
) else (
    echo [INFO] Reusing existing virtual environment: %VENV_DIR%
)

if not exist "%REQ_FILE%" (
    echo [ERROR] Missing %REQ_FILE%
    exit /b 1
)

echo [INFO] Installing dependencies with uv (no YOLO / ultralytics) ...
uv pip install --python "%PYTHON_BIN%" -r "%REQ_FILE%"
if errorlevel 1 (
    echo [ERROR] Dependency install failed.
    exit /b 1
)

if not exist "%MAIN_FILE%" (
    echo [ERROR] Missing %MAIN_FILE%
    exit /b 1
)

echo [INFO] Running DataMatrix crop decode pipeline ...
"%PYTHON_BIN%" "%MAIN_FILE%" %*
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if "%EXIT_CODE%"=="0" (
    echo [INFO] Done. Check results\ for CSV report and logs.
) else (
    echo [ERROR] main.py exited with code %EXIT_CODE%.
)

exit /b %EXIT_CODE%
