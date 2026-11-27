"""Shared Phase 3–4 Spatial Set Encoder, pair-geometry bias, and FP16-safe masked attention."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

_EARTH_RADIUS_KM: float = 6371.0
_DEG_TO_RAD: float = math.pi / 180.0
_TWO_PI: float = 2.0 * math.pi


def compute_pair_geometry(
    station_coords_q: torch.Tensor,
    station_coords_k: torch.Tensor,
) -> torch.Tensor:
    """Compute 5D relative pair geometry `[d_km/2000, log1p(d_km/500), dMLat/30, sin_dMLon, cos_dMLon]`.

    Args:
        station_coords_q: `(N_q, 4)` or `(B, N_q, 4)` with `[lat_deg, lon_deg, mlat_deg, mlon_deg]`.
        station_coords_k: `(N_k, 4)` or `(B, N_k, 4)` with `[lat_deg, lon_deg, mlat_deg, mlon_deg]`.
    """
    if station_coords_q.ndim not in (2, 3) or station_coords_q.shape[-1] != 4:
        raise ValueError(
            f"Expected station_coords_q shape (N_q, 4) or (B, N_q, 4), got {tuple(station_coords_q.shape)}."
        )
    if station_coords_k.ndim not in (2, 3) or station_coords_k.shape[-1] != 4:
        raise ValueError(
            f"Expected station_coords_k shape (N_k, 4) or (B, N_k, 4), got {tuple(station_coords_k.shape)}."
        )
    if station_coords_q.ndim != station_coords_k.ndim:
        raise ValueError(
            "station_coords_q and station_coords_k must have matching rank (both 2D or both 3D)."
        )

    cq = station_coords_q.to(dtype=torch.float64)
    ck = station_coords_k.to(dtype=torch.float64)

    lat_q = cq[..., :, 0].unsqueeze(-1) * _DEG_TO_RAD
    lon_q = cq[..., :, 1].unsqueeze(-1) * _DEG_TO_RAD
    mlat_q = cq[..., :, 2].unsqueeze(-1)
    mlon_q = cq[..., :, 3].unsqueeze(-1)

    lat_k = ck[..., :, 0].unsqueeze(-2) * _DEG_TO_RAD
    lon_k = ck[..., :, 1].unsqueeze(-2) * _DEG_TO_RAD
    mlat_k = ck[..., :, 2].unsqueeze(-2)
    mlon_k = ck[..., :, 3].unsqueeze(-2)

    dphi = lat_q - lat_k
    dlam = lon_q - lon_k
    sin_dphi = torch.sin(0.5 * dphi)
    sin_dlam = torch.sin(0.5 * dlam)
    a = sin_dphi * sin_dphi + torch.cos(lat_q) * torch.cos(lat_k) * (sin_dlam * sin_dlam)
    a_clamped = torch.clamp(a, min=0.0, max=1.0)
    d_km = 2.0 * _EARTH_RADIUS_KM * torch.asin(torch.sqrt(a_clamped))

    d_mlat = (mlat_q - mlat_k) / 30.0
    d_mlon_rad = (mlon_q - mlon_k) * _DEG_TO_RAD
    geom = torch.stack(
        [
            d_km / 2000.0,
            torch.log1p(d_km / 500.0),
            d_mlat,
            torch.sin(d_mlon_rad),
            torch.cos(d_mlon_rad),
        ],
        dim=-1,
    )
    return geom.to(dtype=torch.float32)


def build_station_coord_features(
    station_coords: torch.Tensor,
    mlt_hours: torch.Tensor,
    c_time: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build 6D coordinate features `[MLat/90, sin_MLT, cos_MLT, lat/90, sin_lon, cos_lon]` (+ optional `c_time`)."""
    if mlt_hours.ndim != 2:
        raise ValueError(f"Expected mlt_hours shape (B, N), got {tuple(mlt_hours.shape)}.")
    b, n_st = mlt_hours.shape

    coords = station_coords.float()
    if coords.ndim == 2:
        if coords.shape != (n_st, 4):
            raise ValueError(f"Expected station_coords shape ({n_st}, 4), got {tuple(coords.shape)}.")
        coords = coords.unsqueeze(0).expand(b, n_st, 4)
    elif coords.ndim == 3:
        if coords.shape != (b, n_st, 4):
            raise ValueError(f"Expected station_coords shape ({b}, {n_st}, 4), got {tuple(coords.shape)}.")
    else:
        raise ValueError(f"Invalid station_coords rank {coords.ndim}.")

    mlt_rad = mlt_hours.float() * (_TWO_PI / 24.0)
    lon_rad = coords[..., 1] * _DEG_TO_RAD
    feat = torch.stack(
        [
            coords[..., 2] / 90.0,
            torch.sin(mlt_rad),
            torch.cos(mlt_rad),
            coords[..., 0] / 90.0,
            torch.sin(lon_rad),
            torch.cos(lon_rad),
        ],
        dim=-1,
    )
    if c_time is not None:
        if c_time.ndim != 2 or c_time.shape[0] != b:
            raise ValueError(f"Expected c_time shape ({b}, d_t), got {tuple(c_time.shape)}.")
        if c_time.shape[1] > 0:
            c_exp = c_time.float().unsqueeze(1).expand(b, n_st, c_time.shape[1])
            feat = torch.cat([feat, c_exp], dim=-1)
    return feat


def build_obs_tokens(
    x_work: torch.Tensor,
    station_coords: torch.Tensor,
    mlt_hours: torch.Tensor,
    obs_mask: torch.Tensor,
) -> torch.Tensor:
    """Build NaN-safe 9D observation tokens `[ΔX̃, ΔỸ, ΔZ̃, MLat/90, sin_MLT, cos_MLT, lat/90, sin_lon, cos_lon]`."""
    x_f = x_work.float()
    if x_f.ndim != 3 or x_f.shape[2] != 3:
        raise ValueError(f"Expected x_work shape (B, N, 3), got {tuple(x_f.shape)}.")
    mask_b = obs_mask.bool()
    if mask_b.shape != x_f.shape[:2]:
        raise ValueError(f"Expected obs_mask shape {x_f.shape[:2]}, got {tuple(mask_b.shape)}.")

    x_clean = torch.where(
        mask_b.unsqueeze(-1),
        torch.nan_to_num(x_f, nan=0.0, posinf=0.0, neginf=0.0),
        torch.zeros_like(x_f),
    )
    coord_feat = build_station_coord_features(station_coords, mlt_hours, c_time=None)
    tokens = torch.cat([x_clean, coord_feat], dim=-1)
    return torch.where(mask_b.unsqueeze(-1), tokens, torch.zeros_like(tokens))


def build_attn_forbidden(
    obs_mask: torch.Tensor,
    q_station_ids: torch.Tensor | None = None,
    k_station_ids: torch.Tensor | None = None,
    n_q: int | None = None,
) -> torch.Tensor:
    """Construct `attn_forbidden[b, q, k] = (~obs_mask[b, k]) | (q_id == k_id >= 0)`."""
    mask_b = obs_mask.bool()
    if mask_b.ndim != 2:
        raise ValueError(f"Expected obs_mask shape (B, N_k), got {tuple(mask_b.shape)}.")
    b, n_k = mask_b.shape

    if n_q is None:
        if q_station_ids is not None:
            n_q_eff = int(q_station_ids.shape[-1])
        else:
            n_q_eff = n_k
    else:
        n_q_eff = int(n_q)

    forbidden = (~mask_b).unsqueeze(1).expand(b, n_q_eff, n_k)
    if q_station_ids is not None and k_station_ids is not None:
        q_ids = q_station_ids.long()
        k_ids = k_station_ids.long()
        if q_ids.ndim == 1:
            q_ids = q_ids.unsqueeze(0).expand(b, n_q_eff)
        if k_ids.ndim == 1:
            k_ids = k_ids.unsqueeze(0).expand(b, n_k)
        same_station = (
            (q_ids.unsqueeze(-1) == k_ids.unsqueeze(-2))
            & (q_ids.unsqueeze(-1) >= 0)
            & (k_ids.unsqueeze(-2) >= 0)
        )
        forbidden = forbidden | same_station
    return forbidden


class SpatialBiasMLP(nn.Module):
    """Map 5D pairwise station geometry to per-head attention logit biases."""

    def __init__(self, in_dim: int = 5, hidden_dim: int = 16, n_heads: int = 4) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.n_heads),
        )

    def forward(self, pair_geom: torch.Tensor) -> torch.Tensor:
        """Transform `(N_q, N_k, 5)` or `(B|1, N_q, N_k, 5)` -> `(B|1, n_heads, N_q, N_k)`."""
        g = pair_geom
        if g.ndim == 3:
            g = g.unsqueeze(0)
        if g.ndim != 4 or g.shape[-1] != self.in_dim:
            raise ValueError(
                f"Expected pair_geom shape (B|1, N_q, N_k, {self.in_dim}), got {tuple(pair_geom.shape)}."
            )
        bias = self.net(g)
        return bias.permute(0, 3, 1, 2).contiguous()


class MaskedMultiHeadAttention(nn.Module):
    """FP16/AMP-safe multi-head attention computing masked softmax strictly in float32."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if int(d_model) % int(n_heads) != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}.")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.scale = 1.0 / math.sqrt(float(self.head_dim))

        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.attn_drop = nn.Dropout(float(dropout))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_forbidden: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend over `(B, N_k, d_model)` keys/values with float32 masking and all-masked protection."""
        b, n_q, _ = query.shape
        n_k = key.shape[1]
        in_dtype = query.dtype

        q = self.q_proj(query).view(b, n_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(b, n_k, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(b, n_k, self.n_heads, self.head_dim).transpose(1, 2)

        dev_type = "cuda" if query.is_cuda else "cpu"
        with torch.amp.autocast(device_type=dev_type, enabled=False):
            q_f = q.float()
            k_f = k.float()
            v_f = v.float()

            logits = torch.matmul(q_f, k_f.transpose(-2, -1)) * self.scale
            if attn_bias is not None:
                logits = logits + attn_bias.float()

            if key_forbidden is not None:
                kf = key_forbidden.bool()
                if kf.ndim == 2:
                    mask_f = kf.unsqueeze(1).unsqueeze(2)
                elif kf.ndim == 3:
                    mask_f = kf.unsqueeze(1)
                else:
                    raise ValueError(
                        f"Expected key_forbidden shape (B, N_k) or (B, N_q, N_k), got {tuple(kf.shape)}."
                    )
                all_forbidden = mask_f.all(dim=-1, keepdim=True)
                safe_mask = mask_f & (~all_forbidden)
                logits = logits.masked_fill(safe_mask, -1e9)
                attn = torch.softmax(logits, dim=-1)
                attn = torch.where(all_forbidden, torch.zeros_like(attn), attn)
            else:
                attn = torch.softmax(logits, dim=-1)

            attn = self.attn_drop(attn)
            out_f = torch.matmul(attn, v_f)

        out = out_f.to(dtype=in_dtype).transpose(1, 2).contiguous().view(b, n_q, self.d_model)
        return self.out_proj(out)


class PreNormSelfAttnBlock(nn.Module):
    """Pre-norm self-attention + FFN Transformer block."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(int(d_model))
        self.attn = MaskedMultiHeadAttention(int(d_model), int(n_heads), dropout=float(dropout))
        self.drop1 = nn.Dropout(float(dropout))
        self.norm2 = nn.LayerNorm(int(d_model))
        self.ffn = nn.Sequential(
            nn.Linear(int(d_model), int(d_ff)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(d_ff), int(d_model)),
            nn.Dropout(float(dropout)),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_forbidden: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.drop1(self.attn(h, h, h, key_forbidden=key_forbidden, attn_bias=attn_bias))
        x = x + self.ffn(self.norm2(x))
        return x


class PreNormCrossAttnBlock(nn.Module):
    """Pre-norm cross-attention + FFN block."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(int(d_model))
        self.norm_kv = nn.LayerNorm(int(d_model))
        self.cross_attn = MaskedMultiHeadAttention(int(d_model), int(n_heads), dropout=float(dropout))
        self.drop1 = nn.Dropout(float(dropout))
        self.norm_ffn = nn.LayerNorm(int(d_model))
        self.ffn = nn.Sequential(
            nn.Linear(int(d_model), int(d_ff)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(d_ff), int(d_model)),
            nn.Dropout(float(dropout)),
        )

    def forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        *,
        key_forbidden: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qn = self.norm_q(query)
        kvn = self.norm_kv(kv)
        x = query + self.drop1(
            self.cross_attn(qn, kvn, kvn, key_forbidden=key_forbidden, attn_bias=attn_bias)
        )
        x = x + self.ffn(self.norm_ffn(x))
        return x


class SpatialSetEncoder(nn.Module):
    """Permutation-equivariant Spatial Set Transformer shared across Phases 3–4 and T5.2."""

    def __init__(
        self,
        d_in: int = 9,
        d_model: int = 128,
        n_layers: int = 3,
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
        super().__init__()
        if int(d_in) < 1 or int(d_model) < 1 or int(n_layers) < 1:
            raise ValueError("d_in, d_model, and n_layers must all be >= 1.")

        self.d_in = int(d_in)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.d_ff = int(d_ff)
        self.dropout = float(dropout)
        self.use_spatial_bias = bool(use_spatial_bias)
        self.spatial_bias_hidden = int(spatial_bias_hidden)
        self.use_station_emb = bool(use_station_emb)
        self.max_stations = int(max_stations)
        self.station_emb_dim = int(station_emb_dim)
        self.station_emb_dropout = float(station_emb_dropout)
        self.unk_station_idx = self.max_stations

        in_proj_dim = self.d_in
        if self.use_station_emb:
            self.station_emb: nn.Embedding | None = nn.Embedding(
                self.max_stations + 1, self.station_emb_dim
            )
            self.register_buffer(
                "in_train_emb_mask",
                torch.ones(self.max_stations, dtype=torch.bool),
                persistent=True,
            )
            in_proj_dim += self.station_emb_dim
        else:
            self.station_emb = None
            self.register_buffer(
                "in_train_emb_mask",
                torch.ones(self.max_stations, dtype=torch.bool),
                persistent=False,
            )

        self.input_proj = nn.Linear(in_proj_dim, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.spatial_bias_mlp = (
            SpatialBiasMLP(5, self.spatial_bias_hidden, self.n_heads)
            if self.use_spatial_bias
            else None
        )
        self.layers = nn.ModuleList(
            [
                PreNormSelfAttnBlock(
                    self.d_model, self.n_heads, self.d_ff, dropout=self.dropout
                )
                for _ in range(self.n_layers)
            ]
        )

    def set_in_train_stations(self, in_train_mask: torch.Tensor | list[bool]) -> None:
        """Record which station indices are eligible for learned station embeddings (`<UNK>` otherwise)."""
        mask_t = torch.as_tensor(in_train_mask, dtype=torch.bool, device=self.in_train_emb_mask.device)
        if mask_t.ndim != 1:
            raise ValueError(f"Expected 1D in_train_mask, got {tuple(mask_t.shape)}.")
        new_mask = torch.zeros(
            self.max_stations, dtype=torch.bool, device=self.in_train_emb_mask.device
        )
        take = min(self.max_stations, int(mask_t.shape[0]))
        new_mask[:take] = mask_t[:take]
        self.in_train_emb_mask.copy_(new_mask)

    def embed_station_ids(
        self,
        station_ids: torch.Tensor | None,
        batch_size: int,
        n_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Look up station embeddings safely, mapping `-1`/untrained/dropped stations to `<UNK>=max_stations`."""
        if self.station_emb is None:
            raise RuntimeError("embed_station_ids called when use_station_emb=False.")
        unk = self.unk_station_idx
        if station_ids is None:
            idx = torch.full((batch_size, n_tokens), unk, dtype=torch.int64, device=device)
        else:
            raw = station_ids.to(device=device, dtype=torch.int64)
            if raw.ndim == 1:
                raw = raw.unsqueeze(0).expand(batch_size, n_tokens)
            elif raw.shape != (batch_size, n_tokens):
                raise ValueError(
                    f"Expected station_ids shape ({batch_size}, {n_tokens}), got {tuple(raw.shape)}."
                )
            valid_range = (raw >= 0) & (raw < self.max_stations)
            clamped = raw.clamp(0, self.max_stations - 1)
            in_tr = self.in_train_emb_mask[clamped]
            idx = torch.where(valid_range & in_tr, clamped, torch.full_like(clamped, unk))

        if self.training and self.station_emb_dropout > 0.0:
            drop_mask = torch.rand_like(idx.float()) < self.station_emb_dropout
            idx = torch.where(drop_mask, torch.full_like(idx, unk), idx)

        return self.station_emb(idx).to(dtype=dtype)

    def forward(
        self,
        obs_tokens: torch.Tensor,
        obs_mask: torch.Tensor,
        pair_geom: torch.Tensor | None = None,
        station_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode observed station tokens `(B, N, d_in)` -> `(B, N, d_model)` with zero masked leakage."""
        if obs_tokens.ndim != 3 or obs_tokens.shape[2] != self.d_in:
            raise ValueError(
                f"Expected obs_tokens shape (B, N, {self.d_in}), got {tuple(obs_tokens.shape)}."
            )
        b, n_st, _ = obs_tokens.shape
        mask_b = obs_mask.bool()
        if mask_b.shape != (b, n_st):
            raise ValueError(f"Expected obs_mask shape {(b, n_st)}, got {tuple(mask_b.shape)}.")

        # Strict NaN-safe zeroing before first linear projection (avoids IEEE-754 0 * NaN == NaN).
        x_clean = torch.where(
            mask_b.unsqueeze(-1),
            torch.nan_to_num(obs_tokens, nan=0.0, posinf=0.0, neginf=0.0),
            torch.zeros_like(obs_tokens),
        )
        if self.use_station_emb:
            emb = self.embed_station_ids(station_ids, b, n_st, x_clean.device, x_clean.dtype)
            emb = torch.where(mask_b.unsqueeze(-1), emb, torch.zeros_like(emb))
            x_in = torch.cat([x_clean, emb], dim=-1)
        else:
            x_in = x_clean

        h = self.input_norm(self.input_proj(x_in))
        h = torch.where(mask_b.unsqueeze(-1), h, torch.zeros_like(h))

        attn_bias: torch.Tensor | None = None
        if self.spatial_bias_mlp is not None:
            if pair_geom is None:
                raise ValueError("pair_geom must be provided when use_spatial_bias=True.")
            attn_bias = self.spatial_bias_mlp(pair_geom)

        key_forbidden = ~mask_b
        for layer in self.layers:
            h = layer(h, key_forbidden=key_forbidden, attn_bias=attn_bias)
            h = torch.where(mask_b.unsqueeze(-1), h, torch.zeros_like(h))

        return h


__all__ = [
    "MaskedMultiHeadAttention",
    "PreNormCrossAttnBlock",
    "PreNormSelfAttnBlock",
    "SpatialBiasMLP",
    "SpatialSetEncoder",
    "build_attn_forbidden",
    "build_obs_tokens",
    "build_station_coord_features",
    "compute_pair_geometry",
]
