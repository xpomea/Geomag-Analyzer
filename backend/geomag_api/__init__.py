"""FastAPI application and background job manager for GeoMag Analyzer."""

from __future__ import annotations

from typing import Any

from .config import ApiSettings

__all__ = ["ApiSettings", "app", "create_app"]


def __getattr__(name: str) -> Any:
    # Lazily resolve main/app so Windows spawn workers importing geomag_api.workers
    # never eagerly import FastAPI or geomag_core before worker_init runs.
    if name in ("app", "create_app"):
        from . import main as _main

        return getattr(_main, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
