"""Deterministic ETL pipeline for GeoMag_Storms_Dataset_Oct-2026."""

from __future__ import annotations

from typing import Any

ETL_VERSION = "1.0.0"

__all__ = ["ETL_VERSION", "build_all"]


def __getattr__(name: str) -> Any:
    if name == "build_all":
        from .build import build_all

        return build_all
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
