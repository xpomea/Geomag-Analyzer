"""Runtime configuration and paths for geomag_api."""

from __future__ import annotations

import os
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_STATIC_CACHE_DIR = REPO_ROOT / "data" / "cache"
DEFAULT_FRONTEND_DIST_DIR = REPO_ROOT / "frontend" / "dist"


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
    """Resolve writable user_data directory with LOCALAPPDATA fallback."""
    local_user_data = REPO_ROOT / "user_data"
    if _is_writable_target(local_user_data):
        return local_user_data
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "GeoMagAnalyzer"
    return Path.home() / ".geomag_analyzer"


def _parse_csv_tuple(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass
class ApiSettings:
    """Runtime configuration for FastAPI server, worker pool, cache, and CSRF auth."""

    processed_dir: Path = DEFAULT_PROCESSED_DIR
    static_cache_dir: Path = DEFAULT_STATIC_CACHE_DIR
    user_data_dir: Path | None = None
    frontend_dist_dir: Path = DEFAULT_FRONTEND_DIST_DIR
    max_workers: int = max(1, min(4, (os.cpu_count() or 2) - 1))
    max_mc_surrogates: int = 1000
    max_active_jobs: int = 8
    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost", "::1", "testserver")
    allowed_origins: tuple[str, ...] = ()
    bootstrap_token: str = ""
    session_token: str = ""
    session_cookie_value: str = ""
    require_auth: bool = True
    shutdown_callback: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        self.processed_dir = Path(self.processed_dir)
        self.static_cache_dir = Path(self.static_cache_dir)
        self.frontend_dist_dir = Path(self.frontend_dist_dir)
        if self.user_data_dir is None:
            self.user_data_dir = resolve_default_user_data_dir()
        else:
            self.user_data_dir = Path(self.user_data_dir)
        if not self.bootstrap_token:
            self.bootstrap_token = secrets.token_urlsafe(32)
        if not self.session_token:
            self.session_token = secrets.token_urlsafe(32)
        if not self.session_cookie_value:
            self.session_cookie_value = secrets.token_urlsafe(32)

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
        processed = Path(os.environ.get("GEOMAG_PROCESSED_DIR", str(DEFAULT_PROCESSED_DIR)))
        static_cache = Path(
            os.environ.get("GEOMAG_STATIC_CACHE_DIR", str(DEFAULT_STATIC_CACHE_DIR))
        )
        user_data_env = os.environ.get("GEOMAG_USER_DATA_DIR")
        user_data = Path(user_data_env) if user_data_env else resolve_default_user_data_dir()
        frontend_dist = Path(
            os.environ.get("GEOMAG_FRONTEND_DIST_DIR", str(DEFAULT_FRONTEND_DIST_DIR))
        )
        workers_env = os.environ.get("GEOMAG_MAX_WORKERS")
        max_workers = (
            int(workers_env)
            if workers_env
            else max(1, min(4, (os.cpu_count() or 2) - 1))
        )
        mc_cap = int(os.environ.get("GEOMAG_MAX_MC_SURROGATES", "1000"))
        active_cap = int(os.environ.get("GEOMAG_MAX_ACTIVE_JOBS", "8"))
        hosts_env = _parse_csv_tuple(os.environ.get("GEOMAG_ALLOWED_HOSTS"))
        allowed_hosts = hosts_env or ("127.0.0.1", "localhost", "::1", "testserver")
        allowed_origins = _parse_csv_tuple(os.environ.get("GEOMAG_ALLOWED_ORIGINS"))
        require_auth = os.environ.get("GEOMAG_REQUIRE_AUTH", "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )
        return cls(
            processed_dir=processed,
            static_cache_dir=static_cache,
            user_data_dir=user_data,
            frontend_dist_dir=frontend_dist,
            max_workers=max(1, max_workers),
            max_mc_surrogates=max(1, mc_cap),
            max_active_jobs=max(1, active_cap),
            allowed_hosts=allowed_hosts,
            allowed_origins=allowed_origins,
            bootstrap_token=os.environ.get("GEOMAG_BOOTSTRAP_TOKEN", ""),
            session_token=os.environ.get("GEOMAG_SESSION_TOKEN", ""),
            session_cookie_value=os.environ.get("GEOMAG_SESSION_COOKIE", ""),
            require_auth=require_auth,
        )
