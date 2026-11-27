"""Training loop, losses, and artifact serialization for geomag_ml."""

from __future__ import annotations

from .artifacts import (
    BASELINE_MODEL_REGISTRY,
    MODEL_REGISTRY,
    build_model,
    build_stations_manifest,
    load_baseline_artifact,
    load_model_checkpoint,
    load_run_bundle,
    make_run_id,
    save_model_checkpoint,
    save_run_bundle,
    write_history_csv,
)
from .losses import (
    assert_finite_tensor,
    beta_schedule,
    beta_vae_kl_loss,
    covariance_decorr_loss,
    latent_consistency_loss,
    masked_weighted_huber_loss,
)
from .trainer import VALID_TRAIN_MASK_MODES, TrainConfig, Trainer

__all__ = [
    "BASELINE_MODEL_REGISTRY",
    "MODEL_REGISTRY",
    "TrainConfig",
    "Trainer",
    "VALID_TRAIN_MASK_MODES",
    "assert_finite_tensor",
    "beta_schedule",
    "beta_vae_kl_loss",
    "build_model",
    "build_stations_manifest",
    "covariance_decorr_loss",
    "latent_consistency_loss",
    "load_baseline_artifact",
    "load_model_checkpoint",
    "load_run_bundle",
    "make_run_id",
    "masked_weighted_huber_loss",
    "save_model_checkpoint",
    "save_run_bundle",
    "write_history_csv",
]
