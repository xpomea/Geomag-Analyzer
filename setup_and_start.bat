@echo off
setlocal
title GeoMag Analyzer - Setup
cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"

if exist "%VENV_PY%" goto :check_pkg

echo.
echo ============================================================
echo   GeoMag Analyzer - First-time Python environment setup
echo ============================================================
echo.

set "SYS_PY="

where py >nul 2>nul
if errorlevel 1 goto :try_python
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "SYS_PY=py -3"
    goto :have_python
)

:try_python
where python >nul 2>nul
if errorlevel 1 goto :no_python
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "SYS_PY=python"
    goto :have_python
)

:no_python
echo [ERROR] Python 3.11 or newer was not found on this computer.
echo.
echo Please install Python 3.11+ from https://www.python.org/downloads/
echo Check "Add python.exe to PATH" during installation
echo and run this script again.
echo.
pause
exit /b 1

:have_python
echo [1/2] Creating virtual environment in .venv using %SYS_PY% ...
%SYS_PY% -m venv .venv
if errorlevel 1 goto :venv_failed
goto :check_pkg

:venv_failed
echo [ERROR] Failed to create .venv.
pause
exit /b 1

:check_pkg
"%VENV_PY%" -c "import geomag_api, uvicorn" >nul 2>nul
if not errorlevel 1 goto :launch

echo [2/2] Installing backend dependencies...
"%VENV_PY%" -m pip install -e backend
if errorlevel 1 goto :pip_failed
goto :launch

:pip_failed
echo.
echo [ERROR] Failed to install backend dependencies.
echo Check your internet connection and try again.
pause
exit /b 1

:launch
echo.
echo Setup complete. Launching GeoMag Analyzer...
call "%~dp0start_browser.bat" %*
endlocal
