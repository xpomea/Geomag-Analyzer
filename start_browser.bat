@echo off
setlocal
title GeoMag Analyzer
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [ERROR] Python virtual environment not found: %PY%
    echo Create .venv and install backend dependencies first.
    pause
    exit /b 1
)

if not exist "frontend\dist\index.html" (
    echo [ERROR] Built frontend not found: frontend\dist\index.html
    echo Run "npm run build" inside frontend\ first.
    pause
    exit /b 1
)

if not exist "data\processed\catalog.sqlite" (
    echo [ERROR] Processed dataset not found: data\processed\catalog.sqlite
    echo Run ETL first: %PY% -m geomag_etl.build
    pause
    exit /b 1
)

set "REQ_PORT=%~1"
if "%REQ_PORT%"=="" set "REQ_PORT=8000"

"%PY%" backend\launcher.py --browser --port "%REQ_PORT%"

if errorlevel 1 (
    echo.
    echo GeoMag Analyzer exited with an error.
    pause
    exit /b 1
)

endlocal
