"""Float32-safe masked reconstruction and latent regularization losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def assert_finite_tensor(tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    """Raise `FloatingPointError` if any element of `tensor` is NaN or Inf."""
    if not bool(torch.isfinite(tensor).all().item()):
        raise FloatingPointError(f"Non-finite values detected in {name}.")
    return tensor


def masked_weighted_huber_loss(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    qc_clean_mask: torch.Tensor,
    obs_mask: torch.Tensor,
    target_mask: torch.Tensor,
    in_train_mask: torch.Tensor,
    *,
    w_tgt: float = 2.0,
    w_obs: float = 1.0,
    delta: float = 1.0,
) -> torch.Tensor:
    """Compute weighted Huber loss in float32 strictly over QC-clean `in_train` stations."""
    y_hat_f = y_hat.float()
    y_true_f = y_true.float()
    if y_hat_f.shape != y_true_f.shape:
        raise ValueError(f"y_hat shape {tuple(y_hat_f.shape)} != y_true shape {tuple(y_true_f.shape)}.")

    qc = qc_clean_mask.bool()
    obs = obs_mask.bool()
    tgt = target_mask.bool()
    in_tr = in_train_mask.bool()
    if in_tr.ndim == 1:
        in_tr = in_tr.unsqueeze(0)

    active = qc & in_tr
    y_true_safe = torch.where(
        active.unsqueeze(-1),
        torch.nan_to_num(y_true_f, nan=0.0, posinf=0.0, neginf=0.0),
        torch.zeros_like(y_hat_f),
    )

    w_station = active.float() * (float(w_tgt) * tgt.float() + float(w_obs) * obs.float())
    w = w_station.unsqueeze(-1).expand_as(y_hat_f)

    elem_loss = F.huber_loss(y_hat_f, y_true_safe, reduction="none", delta=float(delta))
    denom = w.sum().clamp_min(1.0)
    loss = (w * elem_loss).sum() / denom
    return assert_finite_tensor(loss, "masked_weighted_huber_loss")


def beta_vae_kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute mean KL(N(mu, sigma^2) || N(0, I)) in float32 across batch and latent dims."""
    mu_f = mu.float()
    if mu_f.numel() == 0:
        return mu_f.sum() * 0.0
    lv = logvar.float().clamp(min=-6.0, max=4.0)
    kl = -0.5 * (1.0 + lv - mu_f.pow(2) - lv.exp())
    loss = kl.mean()
    return assert_finite_tensor(loss, "beta_vae_kl_loss")


def beta_schedule(epoch: int, beta_max: float, warmup_epochs: int = 15) -> float:
    """Linear KL warmup schedule from 0 to `beta_max` over `warmup_epochs`."""
    b_max = float(beta_max)
    if b_max <= 0.0:
        return 0.0
    warm = max(1, int(warmup_epochs))
    frac = min(1.0, float(int(epoch) + 1) / float(warm))
    return b_max * frac


def covariance_decorr_loss(mu: torch.Tensor, min_batch_size: int = 8) -> torch.Tensor:
    """Compute `(1 / k^2) * ||Cov(mu) - I_k||_F^2` in float32; strictly 0.0 when `B < min_batch_size`."""
    mu_f = mu.float()
    if mu_f.ndim != 2:
        raise ValueError(f"Expected 2D latent tensor (B, k), got shape {tuple(mu_f.shape)}.")
    b, k = mu_f.shape
    if b < int(min_batch_size) or k == 0:
        return mu_f.sum() * 0.0

    mu_c = mu_f - mu_f.mean(dim=0, keepdim=True)
    cov = (mu_c.T @ mu_c) / float(b - 1)
    eye = torch.eye(k, device=mu_f.device, dtype=torch.float32)
    diff = cov - eye
    loss = (diff * diff).mean()
    return assert_finite_tensor(loss, "covariance_decorr_loss")


def latent_consistency_loss(mu_1: torch.Tensor, mu_2: torch.Tensor) -> torch.Tensor:
    """Compute deterministic latent consistency `(1 / (B*k)) * sum ||mu_1 - mu_2||_2^2`."""
    m1 = mu_1.float()
    m2 = mu_2.float()
    if m1.shape != m2.shape:
        raise ValueError(f"mu_1 shape {tuple(m1.shape)} != mu_2 shape {tuple(m2.shape)}.")
    if m1.numel() == 0:
        return m1.sum() * 0.0
    diff = m1 - m2
    loss = (diff * diff).mean()
    return assert_finite_tensor(loss, "latent_consistency_loss")
