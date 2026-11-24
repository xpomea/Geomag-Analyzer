"""Unit tests for PyInstaller frozen path resolution, Numba cache guard, and worker isolation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from geomag_api import config as api_config
from geomag_api.data_store import canonicalize_station_pair
from geomag_core.cross_wavelet import _should_enable_numba_cache


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
    monkeypatch.delenv("GEOMAG_STATIC_CACHE_DIR", raising=False)
    monkeypatch.delenv("GEOMAG_USER_DATA_DIR", raising=False)
    monkeypatch.delenv("GEOMAG_FRONTEND_DIST_DIR", raising=False)

    assert api_config.get_bundle_root() == internal_dir.resolve()
    assert api_config.get_app_dir() == install_dir.resolve()

    # Without portable.mode marker, writable install_dir MUST resolve user_data to LOCALAPPDATA
    settings_inno = api_config.ApiSettings.from_env()
    assert settings_inno.processed_dir == internal_dir.resolve() / "data" / "processed"
    assert settings_inno.static_cache_dir == internal_dir.resolve() / "data" / "cache"
    assert settings_inno.frontend_dist_dir == internal_dir.resolve() / "frontend" / "dist"
    assert settings_inno.user_data_dir == local_app_data / "GeoMagAnalyzer"
    assert _should_enable_numba_cache() is False

    # With portable.mode marker next to exe, writable install_dir uses ./user_data
    (install_dir / api_config.PORTABLE_MARKER_FILENAME).write_text("1", encoding="utf-8")
    settings_portable = api_config.ApiSettings.from_env()
    assert settings_portable.user_data_dir == install_dir.resolve() / "user_data"


def test_numba_cache_enabled_when_not_frozen(monkeypatch):
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    assert _should_enable_numba_cache() is True


def test_canonicalize_station_pair_west_north_and_iaga_tiebreak():
    ott = {"iaga_code": "OTT", "lon_deg": -75.55, "lat_deg": 45.40}
    lvv = {"iaga_code": "LVV", "lon_deg": 23.75, "lat_deg": 49.90}
    a, b, swapped = canonicalize_station_pair(lvv, ott)
    assert (a["iaga_code"], b["iaga_code"], swapped) == ("OTT", "LVV", True)

    a2, b2, swapped2 = canonicalize_station_pair(ott, lvv)
    assert (a2["iaga_code"], b2["iaga_code"], swapped2) == ("OTT", "LVV", False)

    # Same longitude -> North-first tie-break
    south = {"iaga_code": "STH", "lon_deg": 20.0, "lat_deg": 45.0}
    north = {"iaga_code": "NTH", "lon_deg": 20.0, "lat_deg": 65.0}
    a3, b3, swapped3 = canonicalize_station_pair(south, north)
    assert (a3["iaga_code"], b3["iaga_code"], swapped3) == ("NTH", "STH", True)


def test_importing_workers_does_not_eagerly_import_main_fastapi_or_numba():
    code = (
        "import sys, geomag_api.workers as w; "
        "assert 'geomag_api.main' not in sys.modules; "
        "assert 'fastapi' not in sys.modules; "
        "assert 'numba' not in sys.modules; "
        "w.worker_init(); "
        "print('OK')"
    )
    env = dict(os.environ)
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert "OK" in res.stdout
