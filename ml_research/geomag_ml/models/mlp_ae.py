"""Branch A Phase 2 model: Masked Conditional MLP Autoencoder / beta-VAE."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from geomag_ml.models.base import BaseGeomagModel

_VALID_BOTTLENECK_MODES = ("ae", "ae_decorr", "beta_vae")
_VALID_TIME_COND_DIMS = (0, 2, 4)


def build_mlp_flat_input(x_work: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
    """Build NaN-safe `(B, 4*N)` input `[ΔB_obs_flat (3N), m_obs (N)]` via `torch.where`."""
    x_f = x_work.float()
    if x_f.ndim != 3 or x_f.shape[2] != 3:
        raise ValueError(f"Expected x_work shape (B, N, 3), got {tuple(x_f.shape)}.")
    b, n_st, _ = x_f.shape
    mask_b = obs_mask.bool()
    if mask_b.shape != (b, n_st):
        raise ValueError(f"Expected obs_mask shape {(b, n_st)}, got {tuple(mask_b.shape)}.")

    x_clean = torch.where(
        mask_b.unsqueeze(-1),
        torch.nan_to_num(x_f, nan=0.0, posinf=0.0, neginf=0.0),
        torch.zeros_like(x_f),
    )
    return torch.cat([x_clean.reshape(b, 3 * n_st), mask_b.to(x_f.dtype)], dim=-1)


def sanitize_flat_mlp_input(x_flat: torch.Tensor, n_stations: int) -> torch.Tensor:
    """Enforce zero leakage and NaN-safety directly on a `(B, 4*N)` flat vector."""
    if x_flat.ndim != 2 or x_flat.shape[1] != 4 * int(n_stations):
        raise ValueError(
            f"Expected x_flat shape (B, {4 * int(n_stations)}), got {tuple(x_flat.shape)}."
        )
    n_st = int(n_stations)
    x_vals = x_flat[:, : 3 * n_st].reshape(-1, n_st, 3)
    m_obs = x_flat[:, 3 * n_st :] > 0.5
    x_clean = torch.where(
        m_obs.unsqueeze(-1),
        torch.nan_to_num(x_vals, nan=0.0, posinf=0.0, neginf=0.0),
        torch.zeros_like(x_vals),
    )
    return torch.cat([x_clean.reshape(-1, 3 * n_st), m_obs.to(x_flat.dtype)], dim=-1)


class PreNormResMLP(nn.Module):
    """Pre-norm residual MLP block shared across Branch A and Branch B MLPs."""

    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        h_dim = int(dim if hidden_dim is None else hidden_dim)
        self.norm = nn.LayerNorm(int(dim))
        self.fc1 = nn.Linear(int(dim), h_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(h_dim, int(dim))
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(self.norm(x))
        h = self.act(h)
        h = self.drop(self.fc2(h))
        return x + h


class MaskedConditionalMLPAE(nn.Module, BaseGeomagModel):
    """Branch A Level-1 DL: Masked Conditional MLP Autoencoder / beta-VAE (`4N -> k -> 3N`)."""

    branch: str = "A"
    supports_zero_shot: bool = False
    model_name: str = "masked_conditional_mlp_ae"

    def __init__(
        self,
        n_stations: int,
        latent_dim: int = 5,
        time_cond_dim: int = 2,
        bottleneck_mode: str = "ae_decorr",
        hidden_dim: int = 256,
        mid_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        nn.Module.__init__(self)
        n_st = int(n_stations)
        k_dim = int(latent_dim)
        d_t = int(time_cond_dim)
        mode = str(bottleneck_mode).strip().lower()
        h_dim = int(hidden_dim)
        m_dim = int(mid_dim)
        drop = float(dropout)

        if n_st < 1:
            raise ValueError(f"n_stations must be >= 1, got {n_st}.")
        if k_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {k_dim}.")
        if d_t not in _VALID_TIME_COND_DIMS:
            raise ValueError(f"time_cond_dim must be one of {_VALID_TIME_COND_DIMS}, got {d_t}.")
        if mode not in _VALID_BOTTLENECK_MODES:
            raise ValueError(
                f"bottleneck_mode must be one of {_VALID_BOTTLENECK_MODES}, got {bottleneck_mode!r}."
            )

        self.n_stations = n_st
        self.latent_dim = k_dim
        self.time_cond_dim = d_t
        self.bottleneck_mode = mode
        self.hidden_dim = h_dim
        self.mid_dim = m_dim
        self.dropout = drop

        self.model_kwargs: dict[str, Any] = {
            "n_stations": n_st,
            "latent_dim": k_dim,
            "time_cond_dim": d_t,
            "bottleneck_mode": mode,
            "hidden_dim": h_dim,
            "mid_dim": m_dim,
            "dropout": drop,
        }

        self.enc_stem = nn.Sequential(
            nn.Linear(4 * n_st, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Dropout(drop),
            PreNormResMLP(h_dim, h_dim, dropout=drop),
            nn.Linear(h_dim, m_dim),
            nn.LayerNorm(m_dim),
            nn.GELU(),
        )
        self.fc_mu = nn.Linear(m_dim, k_dim)
        self.fc_logvar = nn.Linear(m_dim, k_dim) if mode == "beta_vae" else None

        self.decoder = nn.Sequential(
            nn.Linear(k_dim + d_t, m_dim),
            nn.LayerNorm(m_dim),
            nn.GELU(),
            nn.Linear(m_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            PreNormResMLP(h_dim, h_dim, dropout=drop),
            nn.Linear(h_dim, 3 * n_st),
        )

    def encode_stats(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Encode sanitized `(B, 4*N)` flat input into `(z, mu, logvar)`."""
        u = sanitize_flat_mlp_input(x_flat, self.n_stations)
        h = self.enc_stem(u)
        mu = self.fc_mu(h)
        if self.fc_logvar is not None:
            logvar = torch.clamp(self.fc_logvar(h), min=-6.0, max=4.0)
            if self.training:
                z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            else:
                z = mu
            return z, mu, logvar
        return mu, mu, None

    def encode_deterministic(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Encode sanitized `(B, 4*N)` flat input deterministically into `mu`."""
        u = sanitize_flat_mlp_input(x_flat, self.n_stations)
        return self.fc_mu(self.enc_stem(u))

    def decode(self, z: torch.Tensor, c_time: torch.Tensor | None = None) -> torch.Tensor:
        """Decode latent `z` (with optional `c_time` when `time_cond_dim > 0`) into `(B, 3*N)`."""
        if self.time_cond_dim > 0:
            if c_time is None:
                raise ValueError(
                    f"c_time of shape (B, {self.time_cond_dim}) is required when time_cond_dim > 0."
                )
            if c_time.ndim != 2 or c_time.shape[1] != self.time_cond_dim:
                raise ValueError(
                    f"Expected c_time shape (B, {self.time_cond_dim}), got {tuple(c_time.shape)}."
                )
            dec_in = torch.cat([z, c_time], dim=-1)
        else:
            if c_time is not None:
                raise ValueError("c_time must be None when time_cond_dim == 0 (ONNX contract).")
            dec_in = z
        return self.decoder(dec_in)

    def forward_with_stats(
        self,
        x_flat: torch.Tensor,
        c_time: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return `(y_hat_flat, z, mu, logvar)` for training losses."""
        z, mu, logvar = self.encode_stats(x_flat)
        y_hat = self.decode(z, c_time)
        return y_hat, z, mu, logvar

    def forward(
        self,
        x_flat: torch.Tensor,
        c_time: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ONNX-clean forward returning strictly `(y_hat_flat, z)`."""
        z, _, _ = self.encode_stats(x_flat)
        y_hat = self.decode(z, c_time)
        return y_hat, z

    def select_time_cond(
        self,
        time_features: np.ndarray | torch.Tensor | None,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor | None:
        """Slice `(B, 4)` time features down to `(B, time_cond_dim)` or `None` when `d_t == 0`."""
        if self.time_cond_dim == 0:
            return None
        if time_features is None:
            raise ValueError(
                f"time_features must be provided when time_cond_dim={self.time_cond_dim}."
            )
        if isinstance(time_features, torch.Tensor):
            tf = time_features.float()
        else:
            tf = torch.from_numpy(np.asarray(time_features, dtype=np.float32))
        if tf.ndim != 2 or tf.shape[1] < self.time_cond_dim:
            raise ValueError(
                f"Expected time_features with >= {self.time_cond_dim} columns, got {tuple(tf.shape)}."
            )
        cond = tf[:, : self.time_cond_dim]
        return cond.to(device) if device is not None else cond

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
        _ = (target_mask, mlt_hours, station_coords, station_dist_km)
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
                    c_b = (
                        self.select_time_cond(time_features[start:end], device=device)
                        if time_features is not None and self.time_cond_dim > 0
                        else self.select_time_cond(None, device=device)
                    )
                    y_hat_flat, _ = self.forward(u, c_b)
                    out[start:end] = y_hat_flat.reshape(end - start, n_st, 3).cpu().numpy()
        finally:
            self.train(was_training)
        return out

    def encode_masked(
        self,
        X_work: np.ndarray,
        obs_mask: np.ndarray,
        *,
        time_features: np.ndarray | None = None,
        mlt_hours: np.ndarray | None = None,
        station_coords: np.ndarray | None = None,
        eval_batch_size: int = 1024,
    ) -> np.ndarray:
        """Encode observed network states into deterministic latent trajectories `(T, k)`."""
        _ = (time_features, mlt_hours, station_coords)
        x_np = np.asarray(X_work, dtype=np.float32)
        obs_np = np.asarray(obs_mask, dtype=bool)
        n_t, n_st, n_comp = x_np.shape
        if n_st != self.n_stations or n_comp != 3:
            raise ValueError(f"Expected X_work shape (T, {self.n_stations}, 3), got {x_np.shape}.")
        if obs_np.shape != (n_t, n_st):
            raise ValueError(f"Expected obs_mask shape {(n_t, n_st)}, got {obs_np.shape}.")
        if n_t == 0:
            return np.empty((0, self.latent_dim), dtype=np.float32)

        device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        out = np.empty((n_t, self.latent_dim), dtype=np.float32)
        bs = max(1, int(eval_batch_size))
        try:
            with torch.no_grad():
                for start in range(0, n_t, bs):
                    end = min(n_t, start + bs)
                    xb = torch.from_numpy(np.ascontiguousarray(x_np[start:end])).to(device)
                    mb = torch.from_numpy(np.ascontiguousarray(obs_np[start:end])).to(device)
                    u = build_mlp_flat_input(xb, mb)
                    z = self.encode_deterministic(u)
                    out[start:end] = z.cpu().numpy()
        finally:
            self.train(was_training)
        return out

    def count_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def onnx_export_spec(self) -> dict[str, Any]:
        """Return opset-17 ONNX specification omitting `c_time` when `time_cond_dim == 0`."""
        if self.time_cond_dim == 0:
            return {
                "opset": 17,
                "supported": True,
                "input_names": ["x_flat"],
                "output_names": ["y_hat", "z"],
                "dynamic_axes": {"x_flat": {0: "B"}, "y_hat": {0: "B"}, "z": {0: "B"}},
                "sample_shapes": {"x_flat": [2, 4 * self.n_stations]},
                "verified_max_abs_diff": None,
            }
        return {
            "opset": 17,
            "supported": True,
            "input_names": ["x_flat", "c_time"],
            "output_names": ["y_hat", "z"],
            "dynamic_axes": {
                "x_flat": {0: "B"},
                "c_time": {0: "B"},
                "y_hat": {0: "B"},
                "z": {0: "B"},
            },
            "sample_shapes": {
                "x_flat": [2, 4 * self.n_stations],
                "c_time": [2, self.time_cond_dim],
            },
            "verified_max_abs_diff": None,
        }
