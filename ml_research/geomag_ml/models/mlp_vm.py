"""Branch B Phase 2 model: Single unified Masked Wide Residual MLP."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from geomag_ml.models.base import BaseGeomagModel
from geomag_ml.models.mlp_ae import (
    PreNormResMLP,
    build_mlp_flat_input,
    sanitize_flat_mlp_input,
)


class MaskedWideMLP(nn.Module, BaseGeomagModel):
    """Branch B Level-1 DL: Single unified Masked Wide Residual MLP (`4N -> 512x3 -> 3N`)."""

    branch: str = "B"
    supports_zero_shot: bool = False
    model_name: str = "masked_wide_mlp"

    def __init__(
        self,
        n_stations: int,
        hidden_dim: int = 512,
        n_blocks: int = 3,
        dropout: float = 0.15,
    ) -> None:
        nn.Module.__init__(self)
        n_st = int(n_stations)
        h_dim = int(hidden_dim)
        blocks = int(n_blocks)
        drop = float(dropout)

        if n_st < 1:
            raise ValueError(f"n_stations must be >= 1, got {n_st}.")
        if h_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {h_dim}.")
        if blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {blocks}.")

        self.n_stations = n_st
        self.hidden_dim = h_dim
        self.n_blocks = blocks
        self.dropout = drop
        self.model_kwargs: dict[str, Any] = {
            "n_stations": n_st,
            "hidden_dim": h_dim,
            "n_blocks": blocks,
            "dropout": drop,
        }

        self.net = nn.Sequential(
            nn.Linear(4 * n_st, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            *[PreNormResMLP(h_dim, h_dim, dropout=drop) for _ in range(blocks)],
            nn.LayerNorm(h_dim),
            nn.Linear(h_dim, 3 * n_st),
        )

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Predict `(B, 3*N)` from sanitized `(B, 4*N)` flat masked input."""
        u = sanitize_flat_mlp_input(x_flat, self.n_stations)
        return self.net(u)

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
        eval_batch_size: int = 1024,
    ) -> np.ndarray:
        """Predict scaled `(T, N, 3)` field in eval mode using ONLY `obs_mask` entries."""
        _ = (target_mask, time_features, mlt_hours, station_coords, station_dist_km)
        x_np = np.asarray(X_work, dtype=np.float32)
        obs_np = np.asarray(obs_mask, dtype=bool)
        n_t, n_st, n_comp = x_np.shape
        if n_st != self.n_stations or n_comp != 3:
            raise ValueError(f"Expected X_work shape (T, {self.n_stations}, 3), got {x_np.shape}.")
        if obs_np.shape != (n_t, n_st):
            raise ValueError(f"Expected obs_mask shape {(n_t, n_st)}, got {obs_np.shape}.")
        if n_t == 0:
            return np.empty((0, n_st, 3), dtype=np.float32)

        device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        out = np.empty((n_t, n_st, 3), dtype=np.float32)
        bs = max(1, int(eval_batch_size))
        try:
            with torch.no_grad():
                for start in range(0, n_t, bs):
                    end = min(n_t, start + bs)
                    xb = torch.from_numpy(np.ascontiguousarray(x_np[start:end])).to(device)
                    mb = torch.from_numpy(np.ascontiguousarray(obs_np[start:end])).to(device)
                    u = build_mlp_flat_input(xb, mb)
                    out[start:end] = self.forward(u).reshape(end - start, n_st, 3).cpu().numpy()
        finally:
            self.train(was_training)
        return out

    def count_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def onnx_export_spec(self) -> dict[str, Any]:
        """Return opset-17 ONNX specification."""
        return {
            "opset": 17,
            "supported": True,
            "input_names": ["x_flat"],
            "output_names": ["y_hat"],
            "dynamic_axes": {"x_flat": {0: "B"}, "y_hat": {0: "B"}},
            "sample_shapes": {"x_flat": [2, 4 * self.n_stations]},
            "verified_max_abs_diff": None,
        }
