"""Evaluation metrics and leakage-safe fold evaluator for geomag_ml."""

from __future__ import annotations

from .evaluator import evaluate_model_on_fold, sanitize_masked_input
from .metrics import (
    REPORTING_BELTS,
    StreamingMetricsAccumulator,
    belt_station_masks,
    compute_latent_stability,
    compute_reconstruction_metrics,
    compute_space_weather_alignment,
)

__all__ = [
    "REPORTING_BELTS",
    "StreamingMetricsAccumulator",
    "belt_station_masks",
    "compute_latent_stability",
    "compute_reconstruction_metrics",
    "compute_space_weather_alignment",
    "evaluate_model_on_fold",
    "sanitize_masked_input",
]
