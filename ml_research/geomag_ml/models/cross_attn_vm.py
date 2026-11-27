"""Branch B Phase 3 model: Spatial Set Encoder + Cross-Attention Virtual Magnetometer."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from geomag_ml.models.base import BaseGeomagModel
from geomag_ml.models.spatial_encoder import (
    PreNormCrossAttnBlock,
    SpatialBiasMLP,
    SpatialSetEncoder,
    build_attn_forbidden,
    build_obs_tokens,
    build_station_coord_features,
    compute_pair_geometry,
)

_VALID_TIME_COND_DIMS = (0, 2, 4)


class CrossAttentionVirtualMagnetometer(nn.Module, BaseGeomagModel):
    """Branch B Level-2 DL: Bottleneck-free Spatial Cross-Attention Virtual Magnetometer."""

    branch: str = "B"
    supports_zero_shot: bool = True
    model_name: str = "cross_attn_vm"

    def __init__(
        self,
        d_in: int = 9,
        d_out: int = 3,
        time_cond_dim: int = 2,
        d_model: int = 128,
        n_enc_layers: int = 3,
        n_cross_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 256,
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
        d_t = int(time_cond_dim)
        dm = int(d_model)
        n_enc = int(n_enc_layers)
        n_cross = int(n_cross_layers)
        nh = int(n_heads)
        ff = int(d_ff)
        drop = float(dropout)

        if d_i < 1 or d_o < 1 or n_cross < 1:
            raise ValueError("d_in, d_out, and n_cross_layers must all be >= 1.")
        if d_t not in _VALID_TIME_COND_DIMS:
            raise ValueError(f"time_cond_dim must be one of {_VALID_TIME_COND_DIMS}, got {d_t}.")

        self.d_in = d_i
        self.d_out = d_o
        self.time_cond_dim = d_t
        self.query_coord_dim = 6 + d_t
        self.d_model = dm
        self.n_enc_layers = n_enc
        self.n_cross_layers = n_cross
        self.n_heads = nh
        self.d_ff = ff
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
            "time_cond_dim": d_t,
            "d_model": dm,
            "n_enc_layers": n_enc,
            "n_cross_layers": n_cross,
            "n_heads": nh,
            "d_ff": ff,
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

        q_in_dim = self.query_coord_dim + (self.station_emb_dim if self.use_station_emb else 0)
        self.q_proj = nn.Sequential(
            nn.Linear(q_in_dim, dm),
            nn.LayerNorm(dm),
            nn.GELU(),
        )
        self.cross_spatial_bias_mlp = (
            SpatialBiasMLP(5, self.spatial_bias_hidden, nh) if self.use_spatial_bias else None
        )
        self.cross_layers = nn.ModuleList(
            [PreNormCrossAttnBlock(dm, nh, ff, dropout=drop) for _ in range(n_cross)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(dm),
            nn.Linear(dm, dm // 2),
            nn.GELU(),
            nn.Linear(dm // 2, d_o),
        )

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

    def decode_from_hidden(
        self,
        h_obs: torch.Tensor,
        q_coords: torch.Tensor,
        q_pair_geom: torch.Tensor | None,
        attn_forbidden: torch.Tensor,
        q_station_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cross-attend target queries `(B, N_q, d_q)` against encoded observed tokens `h_obs`."""
        if q_coords.ndim != 3 or q_coords.shape[2] != self.query_coord_dim:
            raise ValueError(
                f"Expected q_coords shape (B, N_q, {self.query_coord_dim}), got {tuple(q_coords.shape)}."
            )
        b, n_q, _ = q_coords.shape
        if self.use_station_emb:
            q_emb = self.encoder.embed_station_ids(
                q_station_ids, b, n_q, q_coords.device, q_coords.dtype
            )
            q_in = torch.cat([q_coords, q_emb], dim=-1)
        else:
            q_in = q_coords

        q = self.q_proj(q_in)

        cross_bias: torch.Tensor | None = None
        if self.cross_spatial_bias_mlp is not None:
            if q_pair_geom is None:
                raise ValueError("q_pair_geom must be provided when use_spatial_bias=True.")
            cross_bias = self.cross_spatial_bias_mlp(q_pair_geom)

        for layer in self.cross_layers:
            q = layer(q, h_obs, key_forbidden=attn_forbidden, attn_bias=cross_bias)
        return self.head(q)

    def forward(
        self,
        obs_tokens: torch.Tensor,
        obs_mask: torch.Tensor,
        pair_geom: torch.Tensor | None,
        q_coords: torch.Tensor,
        q_pair_geom: torch.Tensor | None = None,
        attn_forbidden: torch.Tensor | None = None,
        k_station_ids: torch.Tensor | None = None,
        q_station_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode observed stations and cross-attend to query stations with strict mask enforcement."""
        if q_pair_geom is None and attn_forbidden is None:
            if self.use_spatial_bias:
                raise ValueError("q_pair_geom and attn_forbidden must be provided when use_spatial_bias=True.")
            # Match 4-input ONNX signature (`obs_tokens, obs_mask, q_coords, attn_forbidden`)
            q_coords_eff = pair_geom
            attn_forbidden_eff = q_coords
            pair_geom_eff = None
            q_pair_geom_eff = None
        else:
            if attn_forbidden is None:
                raise ValueError("attn_forbidden must be provided.")
            q_coords_eff = q_coords
            attn_forbidden_eff = attn_forbidden
            pair_geom_eff = pair_geom
            q_pair_geom_eff = q_pair_geom

        assert q_coords_eff is not None
        h_obs = self.encoder(
            obs_tokens, obs_mask, pair_geom=pair_geom_eff, station_ids=k_station_ids
        )
        eff_forbidden = attn_forbidden_eff.bool() | (~obs_mask.bool()).unsqueeze(1)
        return self.decode_from_hidden(
            h_obs,
            q_coords_eff,
            q_pair_geom_eff,
            eff_forbidden,
            q_station_ids=q_station_ids,
        )

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
        """Predict `(T, N, d_out)` at `target_mask` positions strictly using ONLY `obs_mask`."""
        _ = station_dist_km
        if self.d_in != 9:
            raise RuntimeError(
                f"{self.__class__.__name__} with d_in={self.d_in} expects specialized token input, not raw (T, N, 3)."
            )
        if station_coords is None or mlt_hours is None:
            raise ValueError(
                "CrossAttentionVirtualMagnetometer.predict_masked requires station_coords and mlt_hours."
            )

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
        if int(np.sum(obs_np & tgt_np)) != 0:
            raise AssertionError("obs_mask and target_mask must be disjoint to prevent multi-hop leakage.")

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
                q_idx_t = torch.from_numpy(np.ascontiguousarray(active_q)).to(
                    device=device, dtype=torch.int64
                )
                k_idx_t = torch.arange(n_st, dtype=torch.int64, device=device)
                q_coords_static = coords_t[q_idx_t]

                pair_geom_t = (
                    compute_pair_geometry(coords_t, coords_t).unsqueeze(0)
                    if self.use_spatial_bias
                    else None
                )
                q_pair_geom_t = (
                    compute_pair_geometry(q_coords_static, coords_t).unsqueeze(0)
                    if self.use_spatial_bias
                    else None
                )

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
                    attn_forbidden = build_attn_forbidden(mb, q_idx_t, k_idx_t)
                    y_q = self.forward(
                        obs_tokens,
                        mb,
                        pair_geom_t,
                        q_coords,
                        q_pair_geom_t,
                        attn_forbidden,
                        k_station_ids=k_idx_t,
                        q_station_ids=q_idx_t,
                    ).cpu().numpy()
                    out[rows[:, None], active_q[None, :], :] = y_q
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
        """Zero-shot predict `(T, N_q, d_out)` at arbitrary query coordinates (`q_station_id = -1`)."""
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
                q_pair_geom_t = (
                    compute_pair_geometry(qc_t, oc_t).unsqueeze(0)
                    if self.use_spatial_bias
                    else None
                )
                k_idx_t = torch.arange(n_obs, dtype=torch.int64, device=device)
                q_idx_t = torch.full((n_q,), -1, dtype=torch.int64, device=device)

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
                    attn_forbidden = build_attn_forbidden(mb, q_idx_t, k_idx_t)
                    out[start:end] = self.forward(
                        obs_tokens,
                        mb,
                        pair_geom_t,
                        q_coords,
                        q_pair_geom_t,
                        attn_forbidden,
                        k_station_ids=k_idx_t,
                        q_station_ids=q_idx_t,
                    ).cpu().numpy()
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
                "input_names": [
                    "obs_tokens",
                    "obs_mask",
                    "pair_geom",
                    "q_coords",
                    "q_pair_geom",
                    "attn_forbidden",
                ],
                "output_names": ["y_hat"],
                "dynamic_axes": {
                    "obs_tokens": {0: "B", 1: "N_obs"},
                    "obs_mask": {0: "B", 1: "N_obs"},
                    "pair_geom": {0: "B_or_1", 1: "N_obs", 2: "N_obs"},
                    "q_coords": {0: "B", 1: "N_q"},
                    "q_pair_geom": {0: "B_or_1", 1: "N_q", 2: "N_obs"},
                    "attn_forbidden": {0: "B", 1: "N_q", 2: "N_obs"},
                    "y_hat": {0: "B", 1: "N_q"},
                },
                "sample_shapes": {
                    "obs_tokens": [2, self.max_stations, self.d_in],
                    "obs_mask": [2, self.max_stations],
                    "pair_geom": [1, self.max_stations, self.max_stations, 5],
                    "q_coords": [2, 3, self.query_coord_dim],
                    "q_pair_geom": [1, 3, self.max_stations, 5],
                    "attn_forbidden": [2, 3, self.max_stations],
                },
                "verified_max_abs_diff": None,
            }
        return {
            "opset": 17,
            "supported": True,
            "input_names": ["obs_tokens", "obs_mask", "q_coords", "attn_forbidden"],
            "output_names": ["y_hat"],
            "dynamic_axes": {
                "obs_tokens": {0: "B", 1: "N_obs"},
                "obs_mask": {0: "B", 1: "N_obs"},
                "q_coords": {0: "B", 1: "N_q"},
                "attn_forbidden": {0: "B", 1: "N_q", 2: "N_obs"},
                "y_hat": {0: "B", 1: "N_q"},
            },
            "sample_shapes": {
                "obs_tokens": [2, self.max_stations, self.d_in],
                "obs_mask": [2, self.max_stations],
                "q_coords": [2, 3, self.query_coord_dim],
                "attn_forbidden": [2, 3, self.max_stations],
            },
            "verified_max_abs_diff": None,
        }


__all__ = ["CrossAttentionVirtualMagnetometer"]
