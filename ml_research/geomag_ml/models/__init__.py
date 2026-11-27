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
from .mlp_ae import (
    MaskedConditionalMLPAE,
    PreNormResMLP,
    build_mlp_flat_input,
    sanitize_flat_mlp_input,
)
from .mlp_vm import MaskedWideMLP

__all__ = [
    "BaseGeomagModel",
    "EMPCABaseline",
    "IDWBaseline",
    "MaskedConditionalMLPAE",
    "MaskedWideMLP",
    "OrdinaryKrigingBaseline",
    "PreNormResMLP",
    "RidgeSchurBaseline",
    "build_mlp_flat_input",
    "compute_magnetic_pair_components",
    "sanitize_flat_mlp_input",
    "wrap_lon_diff_deg",
]
