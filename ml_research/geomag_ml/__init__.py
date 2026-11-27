"""Isolated ML/DL research subsystem for geomagnetic and ionospheric event analysis."""

from __future__ import annotations

from pathlib import Path

__version__ = "0.1.0"

ML_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_ROOT.parent

__all__ = ["ML_ROOT", "REPO_ROOT", "__version__"]
