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

"%PY%" -c "exec('''import os, secrets, socket, subprocess, sys, threading, time, urllib.request, webbrowser\nreq = int(sys.argv[1])\ns = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\ntry:\n    s.bind(('127.0.0.1', req))\n    port = req\nexcept OSError:\n    s.bind(('127.0.0.1', 0))\n    port = s.getsockname()[1]\nfinally:\n    s.close()\ntok = secrets.token_urlsafe(32)\nenv = os.environ.copy()\nenv['GEOMAG_BOOTSTRAP_TOKEN'] = tok\nhealth_url = f'http://127.0.0.1:{port}/api/health'\napp_url = f'http://127.0.0.1:{port}/?token={tok}'\nprint(f'Starting GeoMag Analyzer at http://127.0.0.1:{port}', flush=True)\nprint('Press Ctrl+C in this window to stop the server.', flush=True)\nproc = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'geomag_api.main:app', '--host', '127.0.0.1', '--port', str(port), '--workers', '1'], env=env)\ndef open_when_ready():\n    for _ in range(120):\n        if proc.poll() is not None:\n            return\n        try:\n            with urllib.request.urlopen(health_url, timeout=1.0) as r:\n                if r.status == 200:\n                    webbrowser.open(app_url)\n                    return\n        except Exception:\n            pass\n        time.sleep(0.25)\nthreading.Thread(target=open_when_ready, daemon=True).start()\ntry:\n    rc = proc.wait()\nexcept KeyboardInterrupt:\n    proc.terminate()\n    rc = proc.wait()\nsys.exit(rc)\n''')" "%REQ_PORT%"

if errorlevel 1 (
    echo.
    echo GeoMag Analyzer exited with an error.
    pause
    exit /b 1
)

endlocal
