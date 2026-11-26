"""FastAPI application factory, lifespan wiring, and static bundle serving."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware

import geomag_core as gc

from . import (
    routes_catalog,
    routes_cross,
    routes_jobs,
    routes_polarization,
    routes_timeseries,
)
from .cache import CacheManager
from .config import ApiSettings
from .data_store import DataStore, NotFoundError
from .deps import SettingsDep, StoreDep
from .jobs import JobManager, TooManyJobsError
from .schemas import HealthResponse, ShutdownResponse
from .security import AuthState, HostOriginCSRFMiddleware


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    """Create configured FastAPI application instance."""
    cfg = settings if settings is not None else ApiSettings.from_env()
    auth_state = AuthState(allowed_hosts=cfg.allowed_hosts)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        store = DataStore(cfg.processed_dir)
        cache = CacheManager(
            user_cache_dir=cfg.user_cache_dir,
            runtime_db_path=cfg.runtime_db_path,
        )
        job_manager = JobManager(cfg, cache)
        application.state.settings = cfg
        application.state.auth = auth_state
        application.state.store = store
        application.state.cache = cache
        application.state.job_manager = job_manager
        try:
            yield
        finally:
            job_manager.shutdown()

    application = FastAPI(
        title="GeoMag Analyzer API",
        version=gc.__version__,
        description=(
            "Pc5/Pc6 geomagnetic polarization, two-phase Welch cross-spectral matrix, "
            "and XWT/WTC AR(1) Monte Carlo backend."
        ),
        lifespan=lifespan,
    )
    application.add_middleware(GZipMiddleware, minimum_size=1024)
    application.add_middleware(HostOriginCSRFMiddleware, auth=auth_state)

    @application.exception_handler(NotFoundError)
    async def _not_found_handler(_request: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @application.exception_handler(TooManyJobsError)
    async def _too_many_jobs_handler(_request: Request, exc: TooManyJobsError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=429)

    @application.exception_handler(ValueError)
    async def _value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @application.get("/api/health", response_model=HealthResponse, tags=["system"])
    def health(
        store: StoreDep,
        settings_dep: SettingsDep,
    ) -> HealthResponse:
        """Liveness and dataset summary endpoint."""
        return HealthResponse(
            status="ok",
            core_version=gc.__version__,
            catalog_stations=len(store.get_stations()),
            catalog_episodes=len(store.get_episodes()),
            can_shutdown=settings_dep.shutdown_callback is not None,
        )

    @application.post("/api/shutdown", response_model=ShutdownResponse, tags=["system"])
    def shutdown_server(
        settings_dep: SettingsDep,
    ) -> ShutdownResponse | JSONResponse:
        """Cleanly shut down worker pool and desktop/browser server instance."""
        if settings_dep.shutdown_callback is None:
            return JSONResponse(
                {"detail": "Shutdown endpoint is unavailable in headless/uvicorn mode."},
                status_code=501,
            )
        settings_dep.shutdown_callback()
        return ShutdownResponse(status="shutting_down", message="Server shutdown triggered.")

    application.include_router(routes_catalog.router)
    application.include_router(routes_timeseries.router)
    application.include_router(routes_polarization.router)
    application.include_router(routes_cross.router)
    application.include_router(routes_jobs.router)

    dist_index = cfg.frontend_dist_dir / "index.html"
    if dist_index.is_file():
        application.mount(
            "/",
            StaticFiles(directory=str(cfg.frontend_dist_dir), html=True),
            name="frontend",
        )

    return application


_default_app: FastAPI | None = None


def __getattr__(name: str) -> object:
    global _default_app
    if name == "app":
        if _default_app is None:
            _default_app = create_app()
        return _default_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
