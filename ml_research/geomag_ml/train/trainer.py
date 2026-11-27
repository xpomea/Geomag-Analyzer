"""Unified deterministic CPU/T4 training loop for Phase 2–4 neural models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import time
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from geomag_ml.data.datasets import SingleStepDataset
from geomag_ml.data.masks import MaskGenerator
from geomag_ml.data.preprocessing import FieldScaler
from geomag_ml.eval.metrics import StreamingMetricsAccumulator
from geomag_ml.models.mlp_ae import MaskedConditionalMLPAE, build_mlp_flat_input
from geomag_ml.models.mlp_vm import MaskedWideMLP
from geomag_ml.train.losses import (
    assert_finite_tensor,
    beta_schedule,
    beta_vae_kl_loss,
    covariance_decorr_loss,
    latent_consistency_loss,
    masked_weighted_huber_loss,
)


VALID_TRAIN_MASK_MODES: tuple[str, ...] = ("none", "m1_m2", "m1_m2_m3", "custom")
_ALLOWED_MASK_KEYS: frozenset[str] = frozenset({"M1", "M2", "M3", "M4", "none"})
_DEFAULT_MASK_PROBS_BY_MODE: dict[str, dict[str, float]] = {
    "none": {"none": 1.0},
    "m1_m2": {"M1": 0.30, "M2": 0.70},
    "m1_m2_m3": {"M1": 0.35, "M2": 0.35, "M3": 0.30},
}


@dataclass
class TrainConfig:
    """Hyperparameters and training-mask settings for `Trainer`."""

    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 100
    warmup_epochs: int = 5
    min_lr_ratio: float = 0.01
    early_stopping_patience: int = 15
    grad_clip: float = 1.0
    use_amp: bool = True
    huber_delta: float = 1.0
    w_tgt: float = 2.0
    w_obs: float = 1.0
    beta_max: float = 0.0
    beta_warmup_epochs: int = 15
    lambda_cov: float = 0.0
    lambda_cons: float = 0.0
    cov_min_batch_size: int = 8
    train_mask_mode: str = "m1_m2_m3"
    mask_probs: dict[str, float] | None = None
    m2_drop_range: tuple[float, float] = (0.20, 0.35)
    m3_radius_range_km: tuple[float, float] = (500.0, 1000.0)
    min_obs: int = 10
    val_m2_drop_frac: float = 0.25
    val_m2_repeats: int = 3
    eval_batch_size: int = 1024
    seed: int = 42

    def __post_init__(self) -> None:
        self.train_mask_mode = str(self.train_mask_mode).strip().lower()
        if self.train_mask_mode not in VALID_TRAIN_MASK_MODES:
            raise ValueError(
                f"Unsupported train_mask_mode {self.train_mask_mode!r}; "
                f"expected one of {VALID_TRAIN_MASK_MODES}."
            )
        if self.mask_probs is None:
            if self.train_mask_mode == "custom":
                raise ValueError(
                    "mask_probs must be explicitly provided when train_mask_mode='custom'."
                )
            self.mask_probs = dict(_DEFAULT_MASK_PROBS_BY_MODE[self.train_mask_mode])
            return

        cleaned = {str(k): float(v) for k, v in dict(self.mask_probs).items()}
        unknown_keys = set(cleaned) - _ALLOWED_MASK_KEYS
        if unknown_keys:
            raise ValueError(f"Unsupported mask_probs keys {sorted(unknown_keys)}.")
        if any(v < 0.0 for v in cleaned.values()):
            raise ValueError(f"mask_probs cannot contain negative weights: {cleaned}.")
        if self.train_mask_mode != "none":
            active_keys = {k for k, v in cleaned.items() if v > 0.0}
            if not active_keys:
                raise ValueError(f"mask_probs must have positive total weight, got {cleaned}.")
            if self.train_mask_mode == "m1_m2" and not (active_keys <= {"M1", "M2"}):
                raise ValueError(
                    f"train_mask_mode='m1_m2' only allows M1/M2 masks, got {sorted(active_keys)}."
                )
            if self.train_mask_mode == "m1_m2_m3" and not (active_keys <= {"M1", "M2", "M3"}):
                raise ValueError(
                    f"train_mask_mode='m1_m2_m3' only allows M1/M2/M3 masks, got {sorted(active_keys)}."
                )
        self.mask_probs = cleaned

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any], *, seed: int | None = None) -> TrainConfig:
        """Construct `TrainConfig` from a YAML dictionary with clean type normalization."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs: dict[str, Any] = {}
        for k, v in cfg.items():
            if k not in valid_fields:
                continue
            if k in ("m2_drop_range", "m3_radius_range_km"):
                kwargs[k] = (float(v[0]), float(v[1]))
            elif k == "mask_probs":
                kwargs[k] = None if v is None else {str(mk): float(mv) for mk, mv in dict(v).items()}
            else:
                kwargs[k] = v
        if seed is not None:
            kwargs["seed"] = int(seed)
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize `TrainConfig` to a plain dict."""
        data = asdict(self)
        data["m2_drop_range"] = list(self.m2_drop_range)
        data["m3_radius_range_km"] = list(self.m3_radius_range_km)
        return data


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    max_epochs: int,
    warmup_epochs: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_ep = max(1, int(max_epochs))
    warm = min(max(0, int(warmup_epochs)), total_ep)
    min_ratio = float(min_lr_ratio)

    def _lr_lambda(epoch_idx: int) -> float:
        if warm > 0 and epoch_idx < warm:
            return float(epoch_idx + 1) / float(warm)
        rem = max(1, total_ep - warm)
        progress = min(1.0, float(epoch_idx - warm) / float(rem))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


class Trainer:
    """Unified CPU / CUDA T4 trainer with strict float32 losses and val early stopping."""

    def __init__(
        self,
        model: nn.Module,
        train_cfg: TrainConfig,
        device: torch.device,
    ) -> None:
        if int(train_cfg.max_epochs) < 1:
            raise ValueError(f"max_epochs must be >= 1, got {train_cfg.max_epochs}.")
        self.model = model.to(device)
        self.cfg = train_cfg
        self.device = device
        self.amp_enabled = bool(train_cfg.use_amp and device.type == "cuda")

    def _compute_batch_loss(
        self,
        x_b: torch.Tensor,
        m_qc_b: torch.Tensor,
        tf_b: torch.Tensor,
        obs_t: torch.Tensor,
        tgt_t: torch.Tensor,
        in_train_t: torch.Tensor,
        *,
        epoch: int,
        mask_generator: MaskGenerator,
        m_qc_b_np: np.ndarray,
        in_train_np: np.ndarray,
        effective_mask_probs: dict[str, float],
        train_min_obs: int,
        cons_rng: np.random.Generator,
        effective_lambda_cons: float,
    ) -> torch.Tensor:
        cfg = self.cfg
        x_flat = build_mlp_flat_input(x_b, obs_t)

        if isinstance(self.model, MaskedConditionalMLPAE):
            c_time = self.model.select_time_cond(tf_b, device=self.device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=self.amp_enabled):
                y_hat_flat, _, mu, logvar = self.model.forward_with_stats(x_flat, c_time)

            # All losses and regularizers strictly in float32 outside AMP autocast.
            y_hat_3d = y_hat_flat.float().reshape(x_b.shape[0], x_b.shape[1], 3)
            loss_rec = masked_weighted_huber_loss(
                y_hat_3d,
                x_b,
                m_qc_b,
                obs_t,
                tgt_t,
                in_train_t,
                w_tgt=cfg.w_tgt,
                w_obs=cfg.w_obs,
                delta=cfg.huber_delta,
            )
            loss = loss_rec

            if logvar is not None and float(cfg.beta_max) > 0.0:
                beta_cur = beta_schedule(epoch, cfg.beta_max, cfg.beta_warmup_epochs)
                if beta_cur > 0.0:
                    loss = loss + float(beta_cur) * beta_vae_kl_loss(mu, logvar)

            if float(cfg.lambda_cov) > 0.0:
                loss = loss + float(cfg.lambda_cov) * covariance_decorr_loss(
                    mu, min_batch_size=cfg.cov_min_batch_size
                )

            if effective_lambda_cons > 0.0:
                obs2_np, _ = mask_generator.sample_train_batch_masks(
                    m_qc_b_np,
                    m_qc_b_np,
                    in_train_np,
                    probs=effective_mask_probs,
                    m2_drop_range=cfg.m2_drop_range,
                    m3_radius_range_km=cfg.m3_radius_range_km,
                    min_obs=train_min_obs,
                    rng=cons_rng,
                )
                obs2_t = torch.from_numpy(obs2_np).to(self.device)
                x_flat2 = build_mlp_flat_input(x_b, obs2_t)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=self.amp_enabled):
                    mu_2 = self.model.encode_deterministic(x_flat2)
                loss = loss + float(effective_lambda_cons) * latent_consistency_loss(mu, mu_2)

            return assert_finite_tensor(loss, "branch_a_train_loss")

        if isinstance(self.model, MaskedWideMLP):
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=self.amp_enabled):
                y_hat_flat = self.model(x_flat)
            y_hat_3d = y_hat_flat.float().reshape(x_b.shape[0], x_b.shape[1], 3)
            loss = masked_weighted_huber_loss(
                y_hat_3d,
                x_b,
                m_qc_b,
                obs_t,
                tgt_t,
                in_train_t,
                w_tgt=cfg.w_tgt,
                w_obs=cfg.w_obs,
                delta=cfg.huber_delta,
            )
            return assert_finite_tensor(loss, "branch_b_train_loss")

        raise TypeError(f"Unsupported model class for Phase 2 Trainer: {type(self.model).__name__}")

    def _evaluate_validation(
        self,
        val_dataset: SingleStepDataset,
        val_scenarios: list[Any],
        y_true_val_nT: np.ndarray,
        scaler: FieldScaler | None,
        station_belt_id: np.ndarray,
    ) -> tuple[float, float, float, float]:
        cfg = self.cfg
        self.model.eval()
        acc = StreamingMetricsAccumulator(station_belt_id)

        val_x_np = val_dataset.x_work.numpy()
        val_x_t = val_dataset.x_work
        val_qc_t = val_dataset.m_qc_clean
        val_tf_np = val_dataset.time_features.numpy()
        in_train_t = val_dataset.in_train_mask

        n_val = int(val_x_np.shape[0])
        n_st = int(val_x_np.shape[1])
        bs = max(1, int(cfg.eval_batch_size))

        loss_sum = 0.0
        n_tgt_total = 0

        with torch.no_grad():
            for sc in val_scenarios:
                if not np.any(sc.target_mask):
                    continue
                y_hat_work = self.model.predict_masked(
                    val_x_np,
                    sc.obs_mask,
                    sc.target_mask,
                    time_features=val_tf_np,
                    eval_batch_size=bs,
                )
                y_hat_nT = (
                    scaler.inverse_transform(y_hat_work)
                    if scaler is not None
                    else np.asarray(y_hat_work, dtype=np.float32)
                )
                eval_mask = sc.target_mask & val_qc_t.numpy() & in_train_t.numpy()[None, :]
                acc.update(y_true_val_nT, y_hat_nT, eval_mask)

                # Compute masked Huber validation loss in scaled space (strictly on target points).
                for start in range(0, n_val, bs):
                    end = min(n_val, start + bs)
                    yh_b = torch.from_numpy(np.ascontiguousarray(y_hat_work[start:end])).to(
                        self.device
                    )
                    yt_b = val_x_t[start:end].to(self.device)
                    qc_b = val_qc_t[start:end].to(self.device)
                    obs_b = torch.from_numpy(np.ascontiguousarray(sc.obs_mask[start:end])).to(
                        self.device
                    )
                    tgt_b = torch.from_numpy(np.ascontiguousarray(sc.target_mask[start:end])).to(
                        self.device
                    )
                    in_tr_b = in_train_t.to(self.device)
                    b_loss = masked_weighted_huber_loss(
                        yh_b,
                        yt_b,
                        qc_b,
                        obs_b,
                        tgt_b,
                        in_tr_b,
                        w_tgt=1.0,
                        w_obs=0.0,
                        delta=cfg.huber_delta,
                    )
                    cnt = int(
                        (qc_b & tgt_b & in_tr_b.unsqueeze(0)).sum().item()
                    ) * 3
                    if cnt > 0:
                        loss_sum += float(b_loss.item()) * float(cnt)
                        n_tgt_total += cnt

        metrics = acc.finalize()
        val_loss = loss_sum / float(max(1, n_tgt_total))
        g_rmse = metrics["global"]["vector_3d"]["rmse_nT"]
        eu_rmse = metrics["europe_mid"]["vector_3d"]["rmse_nT"]
        aur_rmse = metrics["auroral"]["vector_3d"]["rmse_nT"]

        if g_rmse is None:
            raise RuntimeError("Validation evaluation produced zero active target points.")
        return (
            float(val_loss),
            float(g_rmse),
            float(eu_rmse) if eu_rmse is not None else float(g_rmse),
            float(aur_rmse) if aur_rmse is not None else float(g_rmse),
        )

    def fit(
        self,
        train_dataset: SingleStepDataset,
        val_dataset: SingleStepDataset,
        scaler: FieldScaler | None,
        mask_generator: MaskGenerator,
        station_belt_id: np.ndarray,
        *,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Train model with early stopping on `val_masked_rmse_nT` and restore best weights."""
        cfg = self.cfg
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
        )
        scheduler = _build_lr_scheduler(
            optimizer,
            cfg.max_epochs,
            cfg.warmup_epochs,
            cfg.min_lr_ratio,
        )
        grad_scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

        dl_gen = torch.Generator()
        dl_gen.manual_seed(int(cfg.seed))
        # Always num_workers=0 on Windows / Ryzen CPU per §2.2
        train_loader = DataLoader(
            train_dataset,
            batch_size=max(1, int(cfg.batch_size)),
            shuffle=True,
            num_workers=0,
            generator=dl_gen,
            pin_memory=(self.device.type == "cuda"),
        )

        in_train_np = train_dataset.in_train_mask.numpy().astype(bool)
        n_in_train_active = max(1, int(np.sum(in_train_np)))
        train_min_obs = min(int(cfg.min_obs), max(1, n_in_train_active - 2))

        val_qc_np = val_dataset.m_qc_clean.numpy().astype(bool)
        val_in_tr_np = val_dataset.in_train_mask.numpy().astype(bool)
        val_min_obs = min(int(cfg.min_obs), max(1, int(np.sum(val_in_tr_np)) - 2))
        val_scenarios = mask_generator.generate_m2(
            val_qc_np,
            val_qc_np,
            val_in_tr_np,
            drop_frac=float(cfg.val_m2_drop_frac),
            n_repeats=int(cfg.val_m2_repeats),
            min_obs=val_min_obs,
            seed_offset=900,
        )
        if sum(int(sc.target_mask.sum()) for sc in val_scenarios) == 0:
            val_scenarios = mask_generator.generate_m1(val_qc_np, val_qc_np, val_in_tr_np)

        if val_dataset.x_delta_nT is not None:
            y_true_val_nT = val_dataset.x_delta_nT.numpy()
        elif scaler is not None:
            y_true_val_nT = scaler.inverse_transform(val_dataset.x_work.numpy())
        else:
            y_true_val_nT = val_dataset.x_work.numpy()

        mask_mode = str(cfg.train_mask_mode).strip().lower()
        if mask_mode == "none":
            effective_mask_probs = {"none": 1.0}
            effective_lambda_cons = 0.0
        else:
            effective_mask_probs = dict(cfg.mask_probs)
            effective_lambda_cons = float(cfg.lambda_cons)

        in_train_t = train_dataset.in_train_mask.to(self.device)
        best_val_metric_nT = float("inf")
        best_epoch = 0
        best_state_dict: dict[str, torch.Tensor] | None = None
        no_improve = 0
        history: list[dict[str, Any]] = []

        for ep in range(int(cfg.max_epochs)):
            t_ep = time.perf_counter()
            self.model.train()
            mask_rng = np.random.default_rng(int(cfg.seed) + 10000 + ep)
            cons_rng = np.random.default_rng(int(cfg.seed) + 20000 + ep)
            lr_cur = float(optimizer.param_groups[0]["lr"])

            tr_loss_sum = 0.0
            tr_samples = 0

            for batch in train_loader:
                x_b = batch["x_work"].to(self.device, non_blocking=(self.device.type == "cuda"))
                m_qc_b = batch["m_qc_clean"].to(
                    self.device, non_blocking=(self.device.type == "cuda")
                )
                tf_b = batch["time_features"].to(
                    self.device, non_blocking=(self.device.type == "cuda")
                )
                m_qc_b_np = batch["m_qc_clean"].numpy().astype(bool)

                obs_np, tgt_np = mask_generator.sample_train_batch_masks(
                    m_qc_b_np,
                    m_qc_b_np,
                    in_train_np,
                    probs=effective_mask_probs,
                    m2_drop_range=cfg.m2_drop_range,
                    m3_radius_range_km=cfg.m3_radius_range_km,
                    min_obs=train_min_obs,
                    rng=mask_rng,
                )
                obs_t = torch.from_numpy(obs_np).to(self.device)
                tgt_t = torch.from_numpy(tgt_np).to(self.device)

                optimizer.zero_grad(set_to_none=True)
                loss = self._compute_batch_loss(
                    x_b,
                    m_qc_b,
                    tf_b,
                    obs_t,
                    tgt_t,
                    in_train_t,
                    epoch=ep,
                    mask_generator=mask_generator,
                    m_qc_b_np=m_qc_b_np,
                    in_train_np=in_train_np,
                    effective_mask_probs=effective_mask_probs,
                    train_min_obs=train_min_obs,
                    cons_rng=cons_rng,
                    effective_lambda_cons=effective_lambda_cons,
                )

                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                if float(cfg.grad_clip) > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(cfg.grad_clip))
                grad_scaler.step(optimizer)
                grad_scaler.update()

                bs_cur = int(x_b.shape[0])
                tr_loss_sum += float(loss.item()) * float(bs_cur)
                tr_samples += bs_cur

            train_loss = tr_loss_sum / float(max(1, tr_samples))
            val_loss, val_rmse_g, val_rmse_eu, val_rmse_aur = self._evaluate_validation(
                val_dataset,
                val_scenarios,
                y_true_val_nT,
                scaler,
                station_belt_id,
            )
            ep_sec = time.perf_counter() - t_ep

            history.append(
                {
                    "epoch": ep + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_rmse_global_nT": val_rmse_g,
                    "val_rmse_eu_mid_nT": val_rmse_eu,
                    "val_rmse_auroral_nT": val_rmse_aur,
                    "lr": lr_cur,
                    "epoch_sec": ep_sec,
                }
            )

            if val_rmse_g < best_val_metric_nT - 1e-6:
                best_val_metric_nT = val_rmse_g
                best_epoch = ep + 1
                best_state_dict = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1

            scheduler.step()

            if verbose and ((ep + 1) == 1 or (ep + 1) % 10 == 0 or no_improve == 0):
                print(
                    f"  ep {ep + 1:03d}/{cfg.max_epochs:03d} | "
                    f"tr_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                    f"val_RMSE gl={val_rmse_g:.2f} eu={val_rmse_eu:.2f} aur={val_rmse_aur:.2f} nT | "
                    f"lr={lr_cur:.2e} ({ep_sec:.2f}s)"
                )

            if no_improve >= int(cfg.early_stopping_patience):
                if verbose:
                    print(
                        f"  Early stopping at epoch {ep + 1} (best epoch={best_epoch}, "
                        f"val_RMSE={best_val_metric_nT:.2f} nT)."
                    )
                break

        if best_state_dict is not None:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in best_state_dict.items()}
            )
        self.model.eval()

        return {
            "history": history,
            "best_epoch": int(best_epoch),
            "best_val_metric_nT": float(best_val_metric_nT),
            "epochs_ran": len(history),
        }
