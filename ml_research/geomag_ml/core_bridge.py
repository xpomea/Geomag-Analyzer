"""Read-only import bridge to verified backend modules (geomag_core, geomag_etl, geomag_api)."""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, NamedTuple


class BackendUnavailableError(ImportError):
    """Raised when the read-only backend directory is missing or cannot be imported."""


class BallisticShiftAPI(NamedTuple):
    """Functions imported read-only from geomag_api.data_store for L1 -> BSN propagation."""

    compute_ballistic_shift: Callable[..., Any]
    resample_monotonic_series: Callable[..., Any]


def get_backend_dir() -> Path:
    """Return configured backend directory (GEOMAG_BACKEND_DIR or <repo_root>/backend)."""
    override = os.environ.get("GEOMAG_BACKEND_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "backend"


def is_backend_available() -> bool:
    """Return True if the read-only backend package directory exists on disk."""
    backend_dir = get_backend_dir()
    return backend_dir.is_dir() and (backend_dir / "geomag_core").is_dir()


def _ensure_backend_on_path() -> Path:
    backend_dir = get_backend_dir()
    if not backend_dir.is_dir():
        raise BackendUnavailableError(
            f"Read-only backend directory not found at {backend_dir}. "
            "In cloud environments (Colab/Kaggle), use prebuilt .npz bundles via "
            "geomag_ml.data.schema.load_ml_dataset rather than backend modules."
        )
    backend_str = str(backend_dir)
    if backend_str not in sys.path:
        sys.path.insert(0, backend_str)
    return backend_dir


def _import_backend_module(module_name: str) -> ModuleType:
    _ensure_backend_on_path()
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - defensive import guard
        raise BackendUnavailableError(
            f"Failed to import read-only backend module {module_name!r} from {get_backend_dir()}: {exc}"
        ) from exc


def import_mlt() -> ModuleType:
    """Import read-only `geomag_core.mlt`."""
    return _import_backend_module("geomag_core.mlt")


def import_conventions() -> ModuleType:
    """Import read-only `geomag_core.conventions`."""
    return _import_backend_module("geomag_core.conventions")


def import_spectral() -> ModuleType:
    """Import read-only `geomag_core.spectral`."""
    return _import_backend_module("geomag_core.spectral")


def import_polarization() -> ModuleType:
    """Import read-only `geomag_core.polarization`."""
    return _import_backend_module("geomag_core.polarization")


def import_analytic() -> ModuleType:
    """Import read-only `geomag_core.analytic`."""
    return _import_backend_module("geomag_core.analytic")


def import_cross_wavelet() -> ModuleType:
    """Import read-only `geomag_core.cross_wavelet`."""
    return _import_backend_module("geomag_core.cross_wavelet")


def import_etl_parsers() -> SimpleNamespace:
    """Import read-only ETL parsers and constants from `geomag_etl`."""
    intermagnet = _import_backend_module("geomag_etl.intermagnet")
    omni = _import_backend_module("geomag_etl.omni")
    config = _import_backend_module("geomag_etl.config")
    return SimpleNamespace(
        intermagnet=intermagnet,
        omni=omni,
        config=config,
        parse_iaga2002_file=intermagnet.parse_iaga2002_file,
        align_day_record_to_episode=intermagnet.align_day_record_to_episode,
        IagaDayRecord=intermagnet.IagaDayRecord,
        normalize_lon_180=intermagnet.normalize_lon_180,
        parse_omni_hro_file=omni.parse_omni_hro_file,
        EPISODES=config.EPISODES,
        OMNI_HRO_COLUMNS=config.OMNI_HRO_COLUMNS,
        OMNI_LINE_WIDTH=config.OMNI_LINE_WIDTH,
        BASELINE_NORM_THRESHOLD_NT=config.BASELINE_NORM_THRESHOLD_NT,
        INTERMAGNET_FILL_VALUES=config.INTERMAGNET_FILL_VALUES,
    )


def import_ballistic_shift() -> BallisticShiftAPI:
    """Import ballistic L1->BSN shift helpers from lazy `geomag_api.data_store`."""
    data_store = _import_backend_module("geomag_api.data_store")
    return BallisticShiftAPI(
        compute_ballistic_shift=data_store.compute_ballistic_shift,
        resample_monotonic_series=data_store._resample_monotonic_series,
    )


__all__ = [
    "BackendUnavailableError",
    "BallisticShiftAPI",
    "get_backend_dir",
    "import_analytic",
    "import_ballistic_shift",
    "import_conventions",
    "import_cross_wavelet",
    "import_etl_parsers",
    "import_mlt",
    "import_polarization",
    "import_spectral",
    "is_backend_available",
]
