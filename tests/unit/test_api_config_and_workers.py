"""Unit tests for PyInstaller frozen path resolution and worker isolation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from geomag_api import config as api_config


def test_frozen_paths_and_portable_mode_marker(tmp_path: Path, monkeypatch):
    install_dir = tmp_path / "LocalAppPrograms"
    internal_dir = install_dir / "_internal"
    internal_dir.mkdir(parents=True)
    exe_path = install_dir / "GeoMagAnalyzer.exe"
    exe_path.write_text("", encoding="utf-8")
    local_app_data = tmp_path / "LocalAppData"

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(internal_dir), raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_path), raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    monkeypatch.delenv("GEOMAG_PROCESSED_DIR", raising=False)
    monkeypatch.delenv("GEOMAG_USER_DATA_DIR", raising=False)
    monkeypatch.delenv("GEOMAG_FRONTEND_DIST_DIR", raising=False)

    assert api_config.get_bundle_root() == internal_dir.resolve()
    assert api_config.get_app_dir() == install_dir.resolve()

    # Without portable.mode marker, writable install_dir resolves user_data to LOCALAPPDATA
    settings_default = api_config.ApiSettings.from_env()
    assert settings_default.processed_dir == internal_dir.resolve() / "data" / "processed"
    assert settings_default.frontend_dist_dir == internal_dir.resolve() / "frontend" / "dist"
    assert settings_default.user_data_dir == local_app_data / "GeoMagAnalyzer"

    # With portable.mode marker next to exe, writable install_dir uses ./user_data
    (install_dir / api_config.PORTABLE_MARKER_FILENAME).write_text("1", encoding="utf-8")
    settings_portable = api_config.ApiSettings.from_env()
    assert settings_portable.user_data_dir == install_dir.resolve() / "user_data"


def test_importing_workers_isolates_main_fastapi_and_has_no_numba():
    code = (
        "import importlib.util, os, sys, geomag_api.workers as w; "
        "assert 'geomag_api.main' not in sys.modules; "
        "assert 'fastapi' not in sys.modules; "
        "assert 'geomag_core' not in sys.modules; "
        "assert 'numba' not in sys.modules; "
        "assert 'llvmlite' not in sys.modules; "
        "assert importlib.util.find_spec('numba') is None; "
        "assert importlib.util.find_spec('llvmlite') is None; "
        "w.worker_init(); "
        "assert os.environ.get('OMP_NUM_THREADS') == '1'; "
        "assert os.environ.get('MKL_NUM_THREADS') == '1'; "
        "assert os.environ.get('OPENBLAS_NUM_THREADS') == '1'; "
        "assert os.environ.get('NUMEXPR_NUM_THREADS') == '1'; "
        "assert 'NUMBA_NUM_THREADS' not in os.environ; "
        "import geomag_core; "
        "assert not hasattr(geomag_core, 'HAS_NUMBA'); "
        "print('OK')"
    )
    env = dict(os.environ)
    env.pop("NUMBA_NUM_THREADS", None)
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert "OK" in res.stdout
