"""Runtime configuration and paths for geomag_api."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

PORTABLE_MARKER_FILENAME = "portable.mode"


def get_bundle_root() -> Path:
    """Return read-only static asset root (PyInstaller _internal/_MEIPASS or repo root)."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(str(sys._MEIPASS)).resolve()
    return Path(__file__).resolve().parents[2]


def get_app_dir() -> Path:
    """Return application executable directory when frozen, or repository root in dev mode."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


REPO_ROOT = get_app_dir()
DEFAULT_PROCESSED_DIR = get_bundle_root() / "data" / "processed"
DEFAULT_FRONTEND_DIST_DIR = get_bundle_root() / "frontend" / "dist"


def _is_writable_target(target: Path) -> bool:
    probe_dir = target if target.exists() else target.parent
    try:
        probe_dir.mkdir(parents=True, exist_ok=True)
        probe_file = probe_dir / f".write_probe_{os.getpid()}.tmp"
        probe_file.write_text("ok", encoding="utf-8")
        probe_file.unlink()
        return True
    except OSError:
        return False


def resolve_default_user_data_dir() -> Path:
    """Resolve writable user_data directory with explicit portable.mode guard in frozen mode."""
    app_dir = get_app_dir()
    local_user_data = app_dir / "user_data"
    if getattr(sys, "frozen", False):
        if (app_dir / PORTABLE_MARKER_FILENAME).is_file() and _is_writable_target(local_user_data):
            return local_user_data
    elif _is_writable_target(local_user_data):
        return local_user_data
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "GeoMagAnalyzer"
    return Path.home() / ".geomag_analyzer"


@dataclass
class ApiSettings:
    """Runtime configuration for FastAPI server, worker pool, cache, and local origin guard."""

    processed_dir: Path = DEFAULT_PROCESSED_DIR
    user_data_dir: Path | None = None
    frontend_dist_dir: Path = DEFAULT_FRONTEND_DIST_DIR
    max_workers: int = max(1, min(4, (os.cpu_count() or 2) - 1))
    max_mc_surrogates: int = 1000
    max_active_jobs: int = 8
    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost", "::1", "testserver")
    shutdown_callback: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        self.processed_dir = Path(self.processed_dir)
        self.frontend_dist_dir = Path(self.frontend_dist_dir)
        if self.user_data_dir is None:
            self.user_data_dir = resolve_default_user_data_dir()
        else:
            self.user_data_dir = Path(self.user_data_dir)

    @property
    def catalog_db_path(self) -> Path:
        return self.processed_dir / "catalog.sqlite"

    @property
    def runtime_db_path(self) -> Path:
        assert self.user_data_dir is not None
        return self.user_data_dir / "runtime.sqlite"

    @property
    def user_cache_dir(self) -> Path:
        assert self.user_data_dir is not None
        return self.user_data_dir / "user_cache"

    @classmethod
    def from_env(cls) -> ApiSettings:
        """Build ApiSettings from GEOMAG_* environment variables."""
        bundle_root = get_bundle_root()
        processed = Path(
            os.environ.get("GEOMAG_PROCESSED_DIR", str(bundle_root / "data" / "processed"))
        )
        user_data_env = os.environ.get("GEOMAG_USER_DATA_DIR")
        user_data = Path(user_data_env) if user_data_env else resolve_default_user_data_dir()
        frontend_dist = Path(
            os.environ.get("GEOMAG_FRONTEND_DIST_DIR", str(bundle_root / "frontend" / "dist"))
        )
        workers_env = os.environ.get("GEOMAG_MAX_WORKERS")
        max_workers = (
            int(workers_env)
            if workers_env
            else max(1, min(4, (os.cpu_count() or 2) - 1))
        )
        mc_cap = int(os.environ.get("GEOMAG_MAX_MC_SURROGATES", "1000"))
        active_cap = int(os.environ.get("GEOMAG_MAX_ACTIVE_JOBS", "8"))
        return cls(
            processed_dir=processed,
            user_data_dir=user_data,
            frontend_dist_dir=frontend_dist,
            max_workers=max(1, max_workers),
            max_mc_surrogates=max(1, mc_cap),
            max_active_jobs=max(1, active_cap),
        )
