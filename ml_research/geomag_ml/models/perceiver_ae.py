"""Branch A Phase 3 model: Spatial Set Encoder + Perceiver-Bottleneck AE + FiLM Coordinate Decoder."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from geomag_ml.models.base import BaseGeomagModel
from geomag_ml.models.mlp_ae import PreNormResMLP
from geomag_ml.models.spatial_encoder import (
    PreNormCrossAttnBlock,
    PreNormSelfAttnBlock,
    SpatialSetEncoder,
    build_obs_tokens,
    build_station_coord_features,
    compute_pair_geometry,
)

_VALID_BOTTLENECK_MODES = ("ae", "ae_decorr", "beta_vae")
_VALID_TIME_COND_DIMS = (0, 2, 4)


class PerceiverBottleneckAE(nn.Module, BaseGeomagModel):
    """Branch A Level-2 DL: Spatial Set Transformer + Perceiver-Bottleneck AE (`k=3..5`)."""

    branch: str = "A"
    supports_zero_shot: bool = True
    model_name: str = "perceiver_bottleneck_ae"

    def __init__(
        self,
        d_in: int = 9,
        d_out: int = 3,
        latent_dim: int = 5,
        time_cond_dim: int = 2,
        bottleneck_mode: str = "ae_decorr",
        d_model: int = 128,
        n_enc_layers: int = 3,
        n_heads: int = 4,
        d_ff: int = 256,
        n_lat_queries: int = 4,
        dropout: float = 0.1,
        use_spatial_bias: bool = True,
        spatial_bias_hidden: int = 16,
        use_station_emb: bool = False,
        max_stations: int = 49,
        station_emb_dim: int = 8,
        station_emb_dropout: float = 0.3,
    ) -> None:
        nn.Module.__init__(self)
        d_i = int(d_in)
        d_o = int(d_out)
        k_dim = int(latent_dim)
        d_t = int(time_cond_dim)
        mode = str(bottleneck_mode).strip().lower()
        dm = int(d_model)
        n_enc = int(n_enc_layers)
        nh = int(n_heads)
        ff = int(d_ff)
        nq = int(n_lat_queries)
        drop = float(dropout)

        if d_i < 1 or d_o < 1 or k_dim < 1 or nq < 1:
            raise ValueError("d_in, d_out, latent_dim, and n_lat_queries must all be >= 1.")
        if d_t not in _VALID_TIME_COND_DIMS:
            raise ValueError(f"time_cond_dim must be one of {_VALID_TIME_COND_DIMS}, got {d_t}.")
        if mode not in _VALID_BOTTLENECK_MODES:
            raise ValueError(
                f"bottleneck_mode must be one of {_VALID_BOTTLENECK_MODES}, got {bottleneck_mode!r}."
            )

        self.d_in = d_i
        self.d_out = d_o
        self.latent_dim = k_dim
        self.time_cond_dim = d_t
        self.query_coord_dim = 6 + d_t
        self.bottleneck_mode = mode
        self.d_model = dm
        self.n_enc_layers = n_enc
        self.n_heads = nh
        self.d_ff = ff
        self.n_lat_queries = nq
        self.dropout = drop
        self.use_spatial_bias = bool(use_spatial_bias)
        self.spatial_bias_hidden = int(spatial_bias_hidden)
        self.use_station_emb = bool(use_station_emb)
        self.max_stations = int(max_stations)
        self.station_emb_dim = int(station_emb_dim)
        self.station_emb_dropout = float(station_emb_dropout)

        self.model_kwargs: dict[str, Any] = {
            "d_in": d_i,
            "d_out": d_o,
            "latent_dim": k_dim,
            "time_cond_dim": d_t,
            "bottleneck_mode": mode,
            "d_model": dm,
            "n_enc_layers": n_enc,
            "n_heads": nh,
            "d_ff": ff,
            "n_lat_queries": nq,
            "dropout": drop,
            "use_spatial_bias": self.use_spatial_bias,
            "spatial_bias_hidden": self.spatial_bias_hidden,
            "use_station_emb": self.use_station_emb,
            "max_stations": self.max_stations,
            "station_emb_dim": self.station_emb_dim,
            "station_emb_dropout": self.station_emb_dropout,
        }

        self.encoder = SpatialSetEncoder(
            d_in=d_i,
            d_model=dm,
            n_layers=n_enc,
            n_heads=nh,
            d_ff=ff,
            dropout=drop,
            use_spatial_bias=self.use_spatial_bias,
            spatial_bias_hidden=self.spatial_bias_hidden,
            use_station_emb=self.use_station_emb,
            max_stations=self.max_stations,
            station_emb_dim=self.station_emb_dim,
            station_emb_dropout=self.station_emb_dropout,
        )

        self.lat_queries = nn.Parameter(torch.randn(1, nq, dm) * 0.02)
        self.perceiver_cross = PreNormCrossAttnBlock(dm, nh, ff, dropout=drop)
        self.perceiver_self = PreNormSelfAttnBlock(dm, nh, ff, dropout=drop)
        self.bottleneck_proj = nn.Sequential(
            nn.Linear(nq * dm, dm),
            nn.GELU(),
        )
        self.fc_mu = nn.Linear(dm, k_dim)
        self.fc_logvar = nn.Linear(dm, k_dim) if mode == "beta_vae" else None

        # Coordinate Query Decoder with FiLM modulation from latent z
        self.q_proj = nn.Sequential(
            nn.Linear(self.query_coord_dim, dm),
            nn.LayerNorm(dm),
            nn.GELU(),
        )
        self.film = nn.Linear(k_dim, 2 * dm)
        self.dec_fuse = nn.Sequential(
            nn.Linear(dm + k_dim, dm),
            nn.LayerNorm(dm),
            nn.GELU(),
        )
        self.dec_res = PreNormResMLP(dm, dm, dropout=drop)
        self.dec_head = nn.Linear(dm, d_o)

    def set_in_train_stations(self, in_train_mask: torch.Tensor | list[bool]) -> None:
        """Forward `in_train_mask` to the encoder station-embedding guard."""
        self.encoder.set_in_train_stations(in_train_mask)

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

    def encode_from_hidden(
        self,
        h_obs: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Aggregate pre-encoded spatial hidden states `(B, N, d_model)` into `(z, mu, logvar)`."""
        b = h_obs.shape[0]
        q_lat = self.lat_queries.expand(b, -1, -1).to(dtype=h_obs.dtype)
        key_forbidden = ~obs_mask.bool()
        lat = self.perceiver_cross(q_lat, h_obs, key_forbidden=key_forbidden)
        lat = self.perceiver_self(lat)
        pooled = self.bottleneck_proj(lat.reshape(b, self.n_lat_queries * self.d_model))
        mu = self.fc_mu(pooled)
        if self.fc_logvar is not None:
            logvar = torch.clamp(self.fc_logvar(pooled), min=-6.0, max=4.0)
            if self.training:
                z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            else:
                z = mu
            return z, mu, logvar
        return mu, mu, None

    def encode_stats(
        self,
        obs_tokens: torch.Tensor,
        obs_mask: torch.Tensor,
        pair_geom: torch.Tensor | None = None,
        station_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Encode `(B, N, d_in)` observed station tokens into `(z, mu, logvar)`."""
        h_obs = self.encoder(obs_tokens, obs_mask, pair_geom=pair_geom, station_ids=station_ids)
        return self.encode_from_hidden(h_obs, obs_mask)

    def encode_deterministic(
        self,
        obs_tokens: torch.Tensor,
        obs_mask: torch.Tensor,
        pair_geom: torch.Tensor | None = None,
        station_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode observed tokens deterministically into `mu` (used for consistency loss and inference)."""
        _, mu, _ = self.encode_stats(obs_tokens, obs_mask, pair_geom=pair_geom, station_ids=station_ids)
        return mu

    def decode(self, z: torch.Tensor, q_coords: torch.Tensor) -> torch.Tensor:
        """Decode latent `z: (B, k)` at arbitrary query coordinates `q_coords: (B, N_q, d_q)`."""
        if q_coords.ndim != 3 or q_coords.shape[2] != self.query_coord_dim:
            raise ValueError(
                f"Expected q_coords shape (B, N_q, {self.query_coord_dim}), got {tuple(q_coords.shape)}."
            )
        b, n_q, _ = q_coords.shape
        if z.shape != (b, self.latent_dim):
            raise ValueError(f"Expected z shape {(b, self.latent_dim)}, got {tuple(z.shape)}.")

        q_feat = self.q_proj(q_coords)
        gamma, beta = self.film(z).chunk(2, dim=-1)
        h_q = (1.0 + gamma.unsqueeze(1)) * q_feat + beta.unsqueeze(1)
        z_exp = z.unsqueeze(1).expand(b, n_q, self.latent_dim)
        fused = self.dec_fuse(torch.cat([h_q, z_exp], dim=-1))
        fused = self.dec_res(fused)
        return self.dec_head(fused)

    def forward_with_stats(
        self,
        obs_tokens: torch.Tensor,
        obs_mask: torch.Tensor,
        pair_geom: torch.Tensor | None,
        q_coords: torch.Tensor,
        station_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return `(y_hat, z, mu, logvar)` for training losses."""
        z, mu, logvar = self.encode_stats(
            obs_tokens, obs_mask, pair_geom=pair_geom, station_ids=station_ids
        )
        y_hat = self.decode(z, q_coords)
        return y_hat, z, mu, logvar

    def forward(
        self,
        obs_tokens: torch.Tensor,
        obs_mask: torch.Tensor,
        pair_geom: torch.Tensor | None,
        q_coords: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ONNX opset-17 native forward returning strictly `(y_hat, z)`."""
        if q_coords is None:
            if self.use_spatial_bias:
                raise ValueError("q_coords must be provided.")
            q_coords_eff = pair_geom
            pair_geom_eff = None
        else:
            q_coords_eff = q_coords
            pair_geom_eff = pair_geom
        assert q_coords_eff is not None
        z, _, _ = self.encode_stats(obs_tokens, obs_mask, pair_geom=pair_geom_eff, station_ids=None)
        y_hat = self.decode(z, q_coords_eff)
        return y_hat, z

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
        eval_batch_size: int = 512,
    ) -> np.ndarray:
        """Predict `(T, N, d_out)` at `target_mask` positions in float32 eval mode."""
        _ = station_dist_km
        if self.d_in != 9:
            raise RuntimeError(
                f"{self.__class__.__name__} with d_in={self.d_in} expects specialized token input, not raw (T, N, 3)."
            )
        if station_coords is None or mlt_hours is None:
            raise ValueError("PerceiverBottleneckAE.predict_masked requires station_coords and mlt_hours.")

        x_np = np.asarray(X_work, dtype=np.float32)
        obs_np = np.asarray(obs_mask, dtype=bool)
        tgt_np = np.asarray(target_mask, dtype=bool)
        coords_np = np.asarray(station_coords, dtype=np.float32)
        mlt_np = np.asarray(mlt_hours, dtype=np.float32)

        n_t, n_st, n_comp = x_np.shape
        if n_comp != 3 or obs_np.shape != (n_t, n_st) or tgt_np.shape != (n_t, n_st):
            raise ValueError("Incompatible X_work / obs_mask / target_mask shapes.")
        if coords_np.shape != (n_st, 4) or mlt_np.shape != (n_t, n_st):
            raise ValueError("Incompatible station_coords / mlt_hours shapes.")

        out = np.zeros((n_t, n_st, self.d_out), dtype=np.float32)
        active_rows = np.flatnonzero(np.any(tgt_np, axis=1))
        if active_rows.size == 0:
            return out

        active_q = np.flatnonzero(np.any(tgt_np[active_rows], axis=0))
        device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        bs = max(1, int(eval_batch_size))

        try:
            with torch.no_grad():
                coords_t = torch.from_numpy(np.ascontiguousarray(coords_np)).to(device)
                pair_geom_t = (
                    compute_pair_geometry(coords_t, coords_t).unsqueeze(0)
                    if self.use_spatial_bias
                    else None
                )
                q_idx_t = torch.from_numpy(np.ascontiguousarray(active_q)).to(device=device, dtype=torch.int64)
                k_idx_t = torch.arange(n_st, dtype=torch.int64, device=device)
                q_coords_static = coords_t[q_idx_t]

                for start in range(0, active_rows.size, bs):
                    rows = active_rows[start : start + bs]
                    xb = torch.from_numpy(np.ascontiguousarray(x_np[rows])).to(device)
                    mb = torch.from_numpy(np.ascontiguousarray(obs_np[rows])).to(device)
                    mlt_b = torch.from_numpy(np.ascontiguousarray(mlt_np[rows])).to(device)
                    c_b = (
                        self.select_time_cond(time_features[rows], device=device)
                        if self.time_cond_dim > 0 and time_features is not None
                        else self.select_time_cond(None, device=device)
                    )

                    obs_tokens = build_obs_tokens(xb, coords_t, mlt_b, mb)
                    q_coords = build_station_coord_features(
                        q_coords_static, mlt_b[:, q_idx_t], c_time=c_b
                    )
                    _, mu, _ = self.encode_stats(
                        obs_tokens, mb, pair_geom=pair_geom_t, station_ids=k_idx_t
                    )
                    y_q = self.decode(mu, q_coords).cpu().numpy()
                    out[rows[:, None], active_q[None, :], :] = y_q
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
        eval_batch_size: int = 512,
    ) -> np.ndarray:
        """Encode observed network states into deterministic latent trajectories `(T, k)`."""
        _ = time_features
        if self.d_in != 9:
            raise RuntimeError(
                f"{self.__class__.__name__} with d_in={self.d_in} expects specialized token input."
            )
        if station_coords is None or mlt_hours is None:
            raise ValueError("PerceiverBottleneckAE.encode_masked requires station_coords and mlt_hours.")

        x_np = np.asarray(X_work, dtype=np.float32)
        obs_np = np.asarray(obs_mask, dtype=bool)
        coords_np = np.asarray(station_coords, dtype=np.float32)
        mlt_np = np.asarray(mlt_hours, dtype=np.float32)
        n_t, n_st, n_comp = x_np.shape
        if n_comp != 3 or obs_np.shape != (n_t, n_st):
            raise ValueError("Incompatible X_work / obs_mask shapes.")
        if coords_np.shape != (n_st, 4) or mlt_np.shape != (n_t, n_st):
            raise ValueError("Incompatible station_coords / mlt_hours shapes.")
        if n_t == 0:
            return np.empty((0, self.latent_dim), dtype=np.float32)

        device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        out = np.empty((n_t, self.latent_dim), dtype=np.float32)
        bs = max(1, int(eval_batch_size))

        try:
            with torch.no_grad():
                coords_t = torch.from_numpy(np.ascontiguousarray(coords_np)).to(device)
                pair_geom_t = (
                    compute_pair_geometry(coords_t, coords_t).unsqueeze(0)
                    if self.use_spatial_bias
                    else None
                )
                k_idx_t = torch.arange(n_st, dtype=torch.int64, device=device)

                for start in range(0, n_t, bs):
                    end = min(n_t, start + bs)
                    xb = torch.from_numpy(np.ascontiguousarray(x_np[start:end])).to(device)
                    mb = torch.from_numpy(np.ascontiguousarray(obs_np[start:end])).to(device)
                    mlt_b = torch.from_numpy(np.ascontiguousarray(mlt_np[start:end])).to(device)
                    obs_tokens = build_obs_tokens(xb, coords_t, mlt_b, mb)
                    mu = self.encode_deterministic(
                        obs_tokens, mb, pair_geom=pair_geom_t, station_ids=k_idx_t
                    )
                    out[start:end] = mu.cpu().numpy()
        finally:
            self.train(was_training)
        return out

    def predict_at_coordinates(
        self,
        X_obs: np.ndarray,
        obs_mask: np.ndarray,
        obs_coords: np.ndarray,
        obs_mlt: np.ndarray,
        query_coords: np.ndarray,
        query_mlt: np.ndarray,
        *,
        time_features: np.ndarray | None = None,
        eval_batch_size: int = 512,
    ) -> np.ndarray:
        """Predict `(T, N_q, d_out)` at arbitrary unseen query coordinates (`N_q` independent of `N_obs`)."""
        x_np = np.asarray(X_obs, dtype=np.float32)
        obs_np = np.asarray(obs_mask, dtype=bool)
        oc_np = np.asarray(obs_coords, dtype=np.float32)
        om_np = np.asarray(obs_mlt, dtype=np.float32)
        qc_np = np.asarray(query_coords, dtype=np.float32)
        qm_np = np.asarray(query_mlt, dtype=np.float32)

        n_t, n_obs, _ = x_np.shape
        n_q = qc_np.shape[0]
        if n_t == 0 or n_q == 0:
            return np.empty((n_t, n_q, self.d_out), dtype=np.float32)

        device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        out = np.empty((n_t, n_q, self.d_out), dtype=np.float32)
        bs = max(1, int(eval_batch_size))

        try:
            with torch.no_grad():
                oc_t = torch.from_numpy(np.ascontiguousarray(oc_np)).to(device)
                qc_t = torch.from_numpy(np.ascontiguousarray(qc_np)).to(device)
                pair_geom_t = (
                    compute_pair_geometry(oc_t, oc_t).unsqueeze(0)
                    if self.use_spatial_bias
                    else None
                )
                k_idx_t = torch.arange(n_obs, dtype=torch.int64, device=device)

                for start in range(0, n_t, bs):
                    end = min(n_t, start + bs)
                    xb = torch.from_numpy(np.ascontiguousarray(x_np[start:end])).to(device)
                    mb = torch.from_numpy(np.ascontiguousarray(obs_np[start:end])).to(device)
                    omb = torch.from_numpy(np.ascontiguousarray(om_np[start:end])).to(device)
                    qmb = torch.from_numpy(np.ascontiguousarray(qm_np[start:end])).to(device)
                    c_b = (
                        self.select_time_cond(time_features[start:end], device=device)
                        if self.time_cond_dim > 0 and time_features is not None
                        else self.select_time_cond(None, device=device)
                    )
                    obs_tokens = build_obs_tokens(xb, oc_t, omb, mb)
                    q_coords = build_station_coord_features(qc_t, qmb, c_time=c_b)
                    _, mu, _ = self.encode_stats(
                        obs_tokens, mb, pair_geom=pair_geom_t, station_ids=k_idx_t
                    )
                    out[start:end] = self.decode(mu, q_coords).cpu().numpy()
        finally:
            self.train(was_training)
        return out

    def count_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def onnx_export_spec(self) -> dict[str, Any]:
        """Return opset-17 ONNX specification with dynamic `B`, `N_obs`, and `N_q` axes."""
        if self.use_spatial_bias:
            return {
                "opset": 17,
                "supported": True,
                "input_names": ["obs_tokens", "obs_mask", "pair_geom", "q_coords"],
                "output_names": ["y_hat", "z"],
                "dynamic_axes": {
                    "obs_tokens": {0: "B", 1: "N_obs"},
                    "obs_mask": {0: "B", 1: "N_obs"},
                    "pair_geom": {0: "B_or_1", 1: "N_obs", 2: "N_obs"},
                    "q_coords": {0: "B", 1: "N_q"},
                    "y_hat": {0: "B", 1: "N_q"},
                    "z": {0: "B"},
                },
                "sample_shapes": {
                    "obs_tokens": [2, self.max_stations, self.d_in],
                    "obs_mask": [2, self.max_stations],
                    "pair_geom": [1, self.max_stations, self.max_stations, 5],
                    "q_coords": [2, self.max_stations, self.query_coord_dim],
                },
                "verified_max_abs_diff": None,
            }
        return {
            "opset": 17,
            "supported": True,
            "input_names": ["obs_tokens", "obs_mask", "q_coords"],
            "output_names": ["y_hat", "z"],
            "dynamic_axes": {
                "obs_tokens": {0: "B", 1: "N_obs"},
                "obs_mask": {0: "B", 1: "N_obs"},
                "q_coords": {0: "B", 1: "N_q"},
                "y_hat": {0: "B", 1: "N_q"},
                "z": {0: "B"},
            },
            "sample_shapes": {
                "obs_tokens": [2, self.max_stations, self.d_in],
                "obs_mask": [2, self.max_stations],
                "q_coords": [2, self.max_stations, self.query_coord_dim],
            },
            "verified_max_abs_diff": None,
        }


__all__ = ["PerceiverBottleneckAE"]
