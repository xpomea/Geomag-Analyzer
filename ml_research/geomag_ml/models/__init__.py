"""Geomag ML model registry, Phase 1 baselines, and Phase 2+ neural models."""

from __future__ import annotations

from .base import BaseGeomagModel
from .baselines import (
    EMPCABaseline,
    IDWBaseline,
    OrdinaryKrigingBaseline,
    RidgeSchurBaseline,
    compute_magnetic_pair_components,
    wrap_lon_diff_deg,
)
from .cross_attn_vm import CrossAttentionVirtualMagnetometer
from .mlp_ae import (
    MaskedConditionalMLPAE,
    PreNormResMLP,
    build_mlp_flat_input,
    sanitize_flat_mlp_input,
)
from .mlp_vm import MaskedWideMLP
from .perceiver_ae import PerceiverBottleneckAE
from .spatial_encoder import (
    MaskedMultiHeadAttention,
    PreNormCrossAttnBlock,
    PreNormSelfAttnBlock,
    SpatialBiasMLP,
    SpatialSetEncoder,
    build_attn_forbidden,
    build_obs_tokens,
    build_station_coord_features,
    compute_pair_geometry,
)

__all__ = [
    "BaseGeomagModel",
    "CrossAttentionVirtualMagnetometer",
    "EMPCABaseline",
    "IDWBaseline",
    "MaskedConditionalMLPAE",
    "MaskedMultiHeadAttention",
    "MaskedWideMLP",
    "OrdinaryKrigingBaseline",
    "PerceiverBottleneckAE",
    "PreNormCrossAttnBlock",
    "PreNormResMLP",
    "PreNormSelfAttnBlock",
    "RidgeSchurBaseline",
    "SpatialBiasMLP",
    "SpatialSetEncoder",
    "build_attn_forbidden",
    "build_mlp_flat_input",
    "build_obs_tokens",
    "build_station_coord_features",
    "compute_magnetic_pair_components",
    "compute_pair_geometry",
    "sanitize_flat_mlp_input",
    "wrap_lon_diff_deg",
]
