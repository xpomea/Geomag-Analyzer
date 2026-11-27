"""Shared base interface and ONNX metadata contract for geomag_ml models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from geomag_ml.utils.io import load_npz, save_npz_atomic


class BaseGeomagModel(ABC):
    """Unified interface shared by Phase 1 linear baselines and Phase 2–4 neural models."""

    branch: str = "B"
    supports_zero_shot: bool = False
    model_name: str = "base"

    @abstractmethod
    def predict_masked(
        self,
        X_work: np.ndarray,
        obs_mask: np.ndarray,
        target_mask: np.ndarray,
        *,
        time_features: np.ndarray | None = None,
        mlt_hours: np.ndarray | None = None,
        station_coords: np.ndarray | None = None,
        station_dist_km: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict scaled/delta field `(T, N, 3)` using ONLY stations where `obs_mask` is True."""

    def encode_masked(
        self,
        X_work: np.ndarray,
        obs_mask: np.ndarray,
        *,
        time_features: np.ndarray | None = None,
        mlt_hours: np.ndarray | None = None,
        station_coords: np.ndarray | None = None,
    ) -> np.ndarray:
        """Encode observed network state into Branch-A latent coordinates `(T, k)`."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement encode_masked().")

    def count_parameters(self) -> int:
        """Return total scalar parameter count for reporting in `config.json`."""
        return 0

    def onnx_export_spec(self) -> dict[str, Any]:
        """Return ONNX export specification adhering to §5.4 (`supported=False` for linear baselines)."""
        return {
            "opset": 17,
            "supported": False,
            "input_names": [],
            "output_names": [],
            "dynamic_axes": {},
            "sample_shapes": {},
            "verified_max_abs_diff": None,
        }

    def to_state_arrays(self) -> dict[str, np.ndarray]:
        """Serialize fitted parameters to a dictionary of NumPy arrays."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement to_state_arrays().")

    def save_npz(self, path: Path) -> Path:
        """Atomically persist fitted baseline parameters into `model.npz`."""
        return save_npz_atomic(Path(path), self.to_state_arrays(), compressed=True)

    @classmethod
    def load_npz(cls, path: Path) -> BaseGeomagModel:
        """Restore a persisted baseline model from `model.npz`."""
        arrays = load_npz(Path(path))
        return cls.from_state_arrays(arrays)

    @classmethod
    def from_state_arrays(cls, arrays: dict[str, np.ndarray]) -> BaseGeomagModel:
        """Construct model instance from arrays saved by `to_state_arrays()`."""
        raise NotImplementedError(f"{cls.__name__} does not implement from_state_arrays().")
