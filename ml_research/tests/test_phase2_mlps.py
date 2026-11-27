"""Tests for Phase 2 (`T2.1`, `T2.2`, `T2.3`): datasets, losses, Trainer, MLPs, and gate."""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from geomag_ml import ML_ROOT
from geomag_ml.data.datasets import SingleStepDataset
from geomag_ml.data.masks import MaskGenerator
from geomag_ml.data.preprocessing import BaselineEstimator, FieldScaler
from geomag_ml.data.splits import SplitFold
from geomag_ml.eval.evaluator import evaluate_model_on_fold
from geomag_ml.models.mlp_ae import MaskedConditionalMLPAE, build_mlp_flat_input
from geomag_ml.models.mlp_vm import MaskedWideMLP
from geomag_ml.train.artifacts import (
    build_stations_manifest,
    load_run_bundle,
    save_run_bundle,
)
from geomag_ml.train.losses import (
    beta_schedule,
    beta_vae_kl_loss,
    covariance_decorr_loss,
    latent_consistency_loss,
    masked_weighted_huber_loss,
)
from geomag_ml.train.trainer import TrainConfig, Trainer
from geomag_ml.utils.device import resolve_device
from geomag_ml.utils.io import write_json
from geomag_ml.utils.seed import set_seed


def _load_gate_module():
    gate_path = ML_ROOT / "scripts" / "check_phase2_gate.py"
    spec = importlib.util.spec_from_file_location("check_phase2_gate", gate_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_clean_ep_split(
    synthetic_dataset: tuple[dict[str, np.ndarray], dict[str, Any]],
    *,
    scaler_mode: str = "zscore_per_station",
):
    dataset, meta = synthetic_dataset
    n_st = dataset["X_raw"].shape[1]
    st_train_all = np.ones(n_st, dtype=bool)
    is_zs_none = np.zeros(n_st, dtype=bool)

    tr_idx = np.arange(0, 220, dtype=np.int64)
    va_idx = np.arange(220, 300, dtype=np.int64)
    te_idx = np.arange(300, 600, dtype=np.int64)

    est = BaselineEstimator(
        mode="temp_train_quiet_proxy",
        baseless_policy="train_quiet_proxy",
        min_train_points=20,
    ).fit(dataset, tr_idx, station_train_mask=st_train_all)
    x_delta = est.transform(
        dataset["X_raw"],
        dataset["M_valid"],
        dataset["episode_idx"],
        timestamps_unix_s=dataset["timestamps_unix_s"],
        station_has_baseline=dataset["station_has_baseline"],
    )
    scaler = FieldScaler(mode=scaler_mode, min_train_points=20).fit(
        x_delta,
        dataset["M_qc_clean"],
        tr_idx,
        dataset["station_belt_id"],
        station_train_mask=st_train_all,
    )
    x_work = scaler.transform(x_delta)
    in_train = est.in_train_stations_ & scaler.in_train_stations_
    fold = SplitFold(
        fold_id=0,
        protocol="looe_4fold",
        train_idx=tr_idx,
        val_idx=va_idx,
        test_idx=te_idx,
        train_episodes=(0,),
        val_episodes=(0,),
        test_episodes=(1,),
        station_train_mask=st_train_all,
        is_zero_shot_mask=is_zs_none,
        in_train_mask=in_train,
    )
    mask_gen = MaskGenerator(dataset["station_coords"], dataset["station_dist_km"], seed=42)
    return dataset, meta, fold, est, scaler, x_delta, x_work, mask_gen


# ---------------------------------------------------------------------------
# T2.1: Dataset, Losses, Trainer, Seed Parity, and Run Artifacts
# ---------------------------------------------------------------------------


def test_loss_masked_huber_nan_safe_and_empty_mask() -> None:
    y_hat = torch.randn(4, 6, 3, requires_grad=True)
    y_true = torch.randn(4, 6, 3)
    qc = torch.ones(4, 6, dtype=torch.bool)
    obs = torch.ones(4, 6, dtype=torch.bool)
    tgt = torch.zeros(4, 6, dtype=torch.bool)
    tgt[:, 1] = True
    obs[:, 1] = False
    in_tr = torch.tensor([True, True, True, True, True, False])

    # Inject NaN and +1e6 at ~qc and ~in_train positions: must not leak into loss or grad.
    qc[0, 2] = False
    y_true[0, 2, :] = float("nan")
    y_true[:, 5, :] = 1e6

    loss = masked_weighted_huber_loss(
        y_hat, y_true, qc, obs, tgt, in_tr, w_tgt=4.0, w_obs=0.25, delta=1.0
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert y_hat.grad is not None and torch.isfinite(y_hat.grad).all()
    assert float(y_hat.grad[0, 2].abs().max()) == 0.0
    assert float(y_hat.grad[:, 5].abs().max()) == 0.0

    # All-empty target/obs mask must safely return 0.0 with finite zero gradient.
    y_hat2 = torch.randn(2, 4, 3, requires_grad=True)
    empty = torch.zeros(2, 4, dtype=torch.bool)
    loss_empty = masked_weighted_huber_loss(
        y_hat2, torch.full((2, 4, 3), float("nan")), empty, empty, empty, torch.ones(4, dtype=torch.bool)
    )
    assert float(loss_empty.item()) == 0.0
    loss_empty.backward()
    assert y_hat2.grad is not None and float(y_hat2.grad.abs().max()) == 0.0


def test_loss_covariance_decorr_small_batch_and_identity() -> None:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for b_small in (1, 2, 7):
            mu_small = torch.randn(b_small, 5, requires_grad=True)
            loss_small = covariance_decorr_loss(mu_small, min_batch_size=8)
            assert float(loss_small.item()) == 0.0
            loss_small.backward()
            assert mu_small.grad is not None and float(mu_small.grad.abs().max()) == 0.0
    assert not any("degrees of freedom" in str(w.message).lower() for w in rec)

    # Construct exact zero-mean matrix whose sample covariance equals I_4
    q, _ = torch.linalg.qr(torch.randn(16, 4))
    q_c = q - q.mean(dim=0, keepdim=True)
    # Re-orthogonalize centered columns to have Cov = I_4
    u, _, _ = torch.linalg.svd(q_c, full_matrices=False)
    mu_orth = u * np.sqrt(16 - 1)
    assert float(covariance_decorr_loss(mu_orth, min_batch_size=8).item()) < 1e-6

    mu_corr = (torch.ones(12, 3) * torch.arange(12, dtype=torch.float32)[:, None]).requires_grad_(True)
    loss_corr = covariance_decorr_loss(mu_corr, min_batch_size=8)
    assert float(loss_corr.item()) > 0.1
    loss_corr.backward()
    assert mu_corr.grad is not None and torch.isfinite(mu_corr.grad).all()


def test_loss_beta_vae_kl_and_consistency() -> None:
    mu_zero = torch.zeros(8, 4)
    lv_zero = torch.zeros(8, 4)
    assert float(beta_vae_kl_loss(mu_zero, lv_zero).item()) == pytest.approx(0.0, abs=1e-7)

    mu = torch.randn(8, 4, requires_grad=True)
    lv = torch.full((8, 4), 10.0, requires_grad=True)  # clamped to [-6, 4]
    kl = beta_vae_kl_loss(mu, lv)
    assert torch.isfinite(kl) and float(kl.item()) > 0.0
    kl.backward()
    assert mu.grad is not None and torch.isfinite(mu.grad).all()

    assert beta_schedule(0, 0.01, warmup_epochs=15) == pytest.approx(0.01 / 15.0)
    assert beta_schedule(14, 0.01, warmup_epochs=15) == pytest.approx(0.01)
    assert beta_schedule(30, 0.01, warmup_epochs=15) == pytest.approx(0.01)
    assert beta_schedule(5, 0.0, warmup_epochs=15) == 0.0

    m1 = torch.randn(8, 5, requires_grad=True)
    assert float(latent_consistency_loss(m1, m1.detach()).item()) == pytest.approx(0.0, abs=1e-7)
    m2 = m1.detach() + 0.5
    c_loss = latent_consistency_loss(m1, m2)
    assert float(c_loss.item()) == pytest.approx(0.25, abs=1e-6)


def test_trainer_overfit_32_synthetic_points() -> None:
    set_seed(42)
    device = resolve_device("cpu")
    n_t, n_st = 32, 6
    rng = np.random.default_rng(7)
    coords = np.column_stack(
        [
            np.linspace(45.0, 68.0, n_st),
            np.linspace(-20.0, 30.0, n_st),
            np.linspace(48.0, 70.0, n_st),
            np.linspace(60.0, 110.0, n_st),
        ]
    ).astype(np.float32)
    dist = np.abs(coords[:, :1] - coords[:, :1].T) * 111.0
    mg = MaskGenerator(coords, dist, seed=42)
    belt_id = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)

    t_grid = np.linspace(0.0, 2.0 * np.pi, n_t, dtype=np.float32)
    latent = np.column_stack([np.sin(t_grid), np.cos(t_grid), np.sin(2 * t_grid)])
    w_proj = rng.normal(0.0, 0.6, size=(3, n_st * 3)).astype(np.float32)
    x_work = (latent @ w_proj).reshape(n_t, n_st, 3)
    m_valid = np.ones((n_t, n_st), dtype=bool)
    m_qc = np.ones((n_t, n_st), dtype=bool)
    tf = np.column_stack([np.sin(t_grid), np.cos(t_grid), np.zeros(n_t), np.ones(n_t)]).astype(
        np.float32
    )
    in_tr = np.ones(n_st, dtype=bool)

    idx = np.arange(n_t, dtype=np.int64)
    ds = SingleStepDataset(x_work, m_valid, m_qc, tf, idx, in_tr, x_delta_nT=x_work * 20.0)

    # Branch A overfit
    model_a = MaskedConditionalMLPAE(
        n_stations=n_st, latent_dim=4, time_cond_dim=2, bottleneck_mode="ae", hidden_dim=64, mid_dim=32, dropout=0.0
    )
    cfg_a = TrainConfig(
        lr=5e-3,
        batch_size=16,
        max_epochs=60,
        warmup_epochs=3,
        early_stopping_patience=60,
        w_tgt=2.0,
        w_obs=1.0,
        train_mask_mode="none",
        val_m2_repeats=2,
        min_obs=3,
        seed=42,
    )
    res_a = Trainer(model_a, cfg_a, device).fit(ds, ds, None, mg, belt_id, verbose=False)
    assert res_a["history"][-1]["train_loss"] < 0.05

    # Branch B overfit
    model_b = MaskedWideMLP(n_stations=n_st, hidden_dim=128, n_blocks=2, dropout=0.0)
    cfg_b = TrainConfig(
        lr=5e-3,
        batch_size=16,
        max_epochs=60,
        warmup_epochs=3,
        early_stopping_patience=60,
        w_tgt=4.0,
        w_obs=0.25,
        train_mask_mode="m1_m2",
        mask_probs={"M1": 1.0},
        val_m2_repeats=2,
        min_obs=3,
        seed=42,
    )
    res_b = Trainer(model_b, cfg_b, device).fit(ds, ds, None, mg, belt_id, verbose=False)
    assert res_b["history"][-1]["train_loss"] < 0.05


def test_trainer_seed_parity_and_artifacts(
    synthetic_dataset: tuple[dict[str, np.ndarray], dict[str, Any]],
    tmp_path: Path,
) -> None:
    dataset, meta, fold, est, scaler, x_delta, x_work, mask_gen = _make_clean_ep_split(
        synthetic_dataset
    )
    device = resolve_device("cpu")
    in_tr = fold.in_train_mask

    train_ds = SingleStepDataset(
        x_work, dataset["M_valid"], dataset["M_qc_clean"], dataset["time_features"],
        fold.train_idx, in_tr, mlt_hours=dataset["mlt_hours"], x_delta_nT=x_delta,
    )
    val_ds = SingleStepDataset(
        x_work, dataset["M_valid"], dataset["M_qc_clean"], dataset["time_features"],
        fold.val_idx, in_tr, mlt_hours=dataset["mlt_hours"], x_delta_nT=x_delta,
    )
    cfg = TrainConfig(
        lr=1e-3, batch_size=64, max_epochs=3, warmup_epochs=1,
        lambda_cov=0.05, lambda_cons=0.1, mask_probs={"M1": 0.3, "M2": 0.7},
        m2_drop_range=(0.15, 0.35), min_obs=4, val_m2_repeats=2, seed=42,
    )

    set_seed(42)
    m1 = MaskedConditionalMLPAE(n_stations=12, latent_dim=4, time_cond_dim=2, bottleneck_mode="ae_decorr")
    res1 = Trainer(m1, cfg, device).fit(train_ds, val_ds, scaler, mask_gen, dataset["station_belt_id"], verbose=False)

    set_seed(42)
    m2 = MaskedConditionalMLPAE(n_stations=12, latent_dim=4, time_cond_dim=2, bottleneck_mode="ae_decorr")
    Trainer(m2, cfg, device).fit(train_ds, val_ds, scaler, mask_gen, dataset["station_belt_id"], verbose=False)

    max_w_diff = max(
        float((p1.detach() - p2.detach()).abs().max().item())
        for p1, p2 in zip(m1.parameters(), m2.parameters())
    )
    assert max_w_diff < 1e-6

    metrics, outputs = evaluate_model_on_fold(
        m1, dataset, meta, fold, est, scaler, mask_gen,
        eval_split="test", eval_modes=("M1", "M2"), m2_repeats=2, min_obs=4,
        stability_drop_fracs=(0.2, 0.3), stability_repeats=2, return_outputs=True,
    )
    run_dir = tmp_path / "p2_A_test_run"
    save_run_bundle(
        run_dir,
        model=m1,
        best_epoch=res1["best_epoch"],
        best_val_metric_nT=res1["best_val_metric_nT"],
        resolved_config={"phase": "phase2", "branch": "A", "tag": "test", "fold_id": 0, "seed": 42},
        baseline=est,
        scaler=scaler,
        stations_manifest=build_stations_manifest(meta, dataset, in_tr, fold.is_zero_shot_mask),
        metrics=metrics,
        history=res1["history"],
        outputs=outputs,
    )

    for fname in (
        "model.pt", "config.json", "scaler.json", "stations.json",
        "metrics.json", "onnx_export_spec.json", "history.csv", "val_outputs.npz",
    ):
        assert (run_dir / fname).is_file()

    bundle = load_run_bundle(run_dir, device="cpu")
    re_metrics, _ = evaluate_model_on_fold(
        bundle["model"], dataset, meta, fold, bundle["baseline"], bundle["scaler"], mask_gen,
        eval_split="test", eval_modes=("M1", "M2"), m2_repeats=2, min_obs=4,
        stability_drop_fracs=(0.2, 0.3), stability_repeats=2, return_outputs=False,
    )
    r_orig = metrics["eval_masks"]["M2"]["global"]["vector_3d"]["rmse_nT"]
    r_re = re_metrics["eval_masks"]["M2"]["global"]["vector_3d"]["rmse_nT"]
    assert abs(float(r_orig) - float(r_re)) < 1e-4


# ---------------------------------------------------------------------------
# T2.2: Branch A — Masked Conditional MLP-AE / beta-VAE
# ---------------------------------------------------------------------------


def test_branch_a_shapes_and_param_count() -> None:
    model_49 = MaskedConditionalMLPAE(
        n_stations=49, latent_dim=5, time_cond_dim=2, bottleneck_mode="ae_decorr"
    )
    n_params = model_49.count_parameters()
    assert 410_000 <= n_params <= 435_000  # §6.2: ~422K at N=49, k=5, d_t=2

    for d_t in (0, 2, 4):
        m = MaskedConditionalMLPAE(n_stations=12, latent_dim=4, time_cond_dim=d_t)
        x_flat = torch.randn(5, 4 * 12)
        c_time = torch.randn(5, d_t) if d_t > 0 else None
        y_hat, z = m(x_flat, c_time)
        assert y_hat.shape == (5, 3 * 12)
        assert z.shape == (5, 4)
        spec = m.onnx_export_spec()
        assert spec["opset"] == 17 and spec["supported"] is True
        if d_t == 0:
            assert spec["input_names"] == ["x_flat"]
            assert "c_time" not in spec["dynamic_axes"]
        else:
            assert spec["input_names"] == ["x_flat", "c_time"]


def test_branch_a_zero_mask_leakage_and_nan_safety() -> None:
    model = MaskedConditionalMLPAE(n_stations=12, latent_dim=5, time_cond_dim=2, bottleneck_mode="ae_decorr")
    model.eval()
    x_raw = torch.randn(4, 12, 3)
    obs = torch.ones(4, 12, dtype=torch.bool)
    obs[:, 3] = False
    c_time = torch.randn(4, 2)

    u_clean = build_mlp_flat_input(x_raw, obs)
    y_clean, z_clean = model(u_clean, c_time)

    # Perturb hidden station 3 directly in x_raw with +1e6 and NaN while obs[:, 3] == False
    x_huge = x_raw.clone()
    x_huge[:, 3, :] = 1e6
    y_huge, z_huge = model(build_mlp_flat_input(x_huge, obs), c_time)
    assert float((z_clean - z_huge).abs().max().item()) == 0.0
    assert float((y_clean - y_huge).abs().max().item()) == 0.0

    # Also test direct flat tensor injection bypassing build_mlp_flat_input
    u_direct_nan = u_clean.clone()
    u_direct_nan[:, 3 * 3 : 3 * 3 + 3] = float("nan")
    u_direct_nan[:, 3 * 12 + 3] = 0.0
    y_nan, z_nan = model(u_direct_nan, c_time)
    assert torch.isfinite(y_nan).all() and torch.isfinite(z_nan).all()
    assert float((z_clean - z_nan).abs().max().item()) == 0.0
    assert float((y_clean - y_nan).abs().max().item()) == 0.0


def test_branch_a_bottleneck_modes_and_bvae_eval_determinism() -> None:
    x_flat = torch.randn(10, 4 * 12)
    c_time = torch.randn(10, 2)
    for mode in ("ae", "ae_decorr", "beta_vae"):
        m = MaskedConditionalMLPAE(n_stations=12, latent_dim=5, time_cond_dim=2, bottleneck_mode=mode)
        m.train()
        y_hat, z, mu, logvar = m.forward_with_stats(x_flat, c_time)
        assert torch.isfinite(y_hat).all() and torch.isfinite(z).all()
        loss = y_hat.pow(2).mean() + covariance_decorr_loss(mu)
        if logvar is not None:
            loss = loss + beta_vae_kl_loss(mu, logvar)
        loss.backward()
        for p in m.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()

    bvae = MaskedConditionalMLPAE(n_stations=12, latent_dim=5, time_cond_dim=2, bottleneck_mode="beta_vae")
    bvae.train()
    _, z_tr1 = bvae(x_flat, c_time)
    _, z_tr2 = bvae(x_flat, c_time)
    assert float((z_tr1 - z_tr2).abs().max().item()) > 1e-6

    bvae.eval()
    y_ev1, z_ev1 = bvae(x_flat, c_time)
    y_ev2, z_ev2 = bvae(x_flat, c_time)
    assert float((z_ev1 - z_ev2).abs().max().item()) == 0.0
    assert float((y_ev1 - y_ev2).abs().max().item()) == 0.0


def test_branch_a_smoke_ablation_modes(
    synthetic_dataset: tuple[dict[str, np.ndarray], dict[str, Any]],
) -> None:
    device = resolve_device("cpu")
    variants = [
        ("zscore_per_station", 0, "ae_decorr", 0.0, 0.05, 0.1, "m1_m2"),
        ("asinh_global", 2, "ae_decorr", 0.0, 0.05, 0.1, "m1_m2"),
        ("zscore_per_station", 2, "beta_vae", 0.01, 0.0, 0.1, "m1_m2"),
        ("zscore_per_station", 2, "ae_decorr", 0.0, 0.05, 0.0, "none"),
    ]
    for sc_mode, d_t, b_mode, b_max, l_cov, l_cons, tm_mode in variants:
        dataset, meta, fold, est, scaler, x_delta, x_work, mask_gen = _make_clean_ep_split(
            synthetic_dataset, scaler_mode=sc_mode
        )
        in_tr = fold.in_train_mask
        train_ds = SingleStepDataset(
            x_work, dataset["M_valid"], dataset["M_qc_clean"], dataset["time_features"],
            fold.train_idx, in_tr, x_delta_nT=x_delta,
        )
        val_ds = SingleStepDataset(
            x_work, dataset["M_valid"], dataset["M_qc_clean"], dataset["time_features"],
            fold.val_idx, in_tr, x_delta_nT=x_delta,
        )
        model = MaskedConditionalMLPAE(
            n_stations=12, latent_dim=3, time_cond_dim=d_t, bottleneck_mode=b_mode
        )
        cfg = TrainConfig(
            lr=1e-3, batch_size=64, max_epochs=2, warmup_epochs=1,
            beta_max=b_max, lambda_cov=l_cov, lambda_cons=l_cons,
            train_mask_mode=tm_mode, mask_probs={"M1": 0.3, "M2": 0.7},
            m2_drop_range=(0.15, 0.35), min_obs=4, val_m2_repeats=2, seed=42,
        )
        res = Trainer(model, cfg, device).fit(
            train_ds, val_ds, scaler, mask_gen, dataset["station_belt_id"], verbose=False
        )
        assert res["epochs_ran"] == 2
        metrics, _ = evaluate_model_on_fold(
            model, dataset, meta, fold, est, scaler, mask_gen,
            eval_split="test", eval_modes=("M1", "M2"), m2_repeats=2, min_obs=4,
            stability_drop_fracs=(0.2, 0.3), stability_repeats=2,
        )
        assert metrics["eval_masks"]["zero_shot"] == {"status": "not_supported_fixed_stations"}
        assert np.isfinite(metrics["eval_masks"]["M2"]["global"]["vector_3d"]["rmse_nT"])
        # SUA is absent in ep0 (train) -> excluded from M1/M2 evaluations
        assert in_tr[meta["station_codes"].index("SUA")] == False


# ---------------------------------------------------------------------------
# T2.3: Branch B — Masked Wide MLP & Phase 2 Gate Report
# ---------------------------------------------------------------------------


def test_branch_b_shapes_single_network_and_param_count() -> None:
    model_49 = MaskedWideMLP(n_stations=49, hidden_dim=512, n_blocks=3, dropout=0.15)
    assert isinstance(model_49, MaskedWideMLP)
    n_params = model_49.count_parameters()
    assert 1_700_000 <= n_params <= 1_820_000  # §6.2: ~1.75M at N=49

    x_flat = torch.randn(6, 4 * 49)
    y_hat = model_49(x_flat)
    assert y_hat.shape == (6, 3 * 49)
    spec = model_49.onnx_export_spec()
    assert spec["opset"] == 17 and spec["input_names"] == ["x_flat"]


def test_branch_b_zero_mask_leakage_nan_safety_and_finite_grad() -> None:
    model = MaskedWideMLP(n_stations=12, hidden_dim=64, n_blocks=2, dropout=0.15)
    model.eval()
    x_raw = torch.randn(4, 12, 3)
    obs = torch.ones(4, 12, dtype=torch.bool)
    obs[:, 4] = False

    y_clean = model(build_mlp_flat_input(x_raw, obs))
    x_huge = x_raw.clone()
    x_huge[:, 4, :] = 1e6
    y_huge = model(build_mlp_flat_input(x_huge, obs))
    assert float((y_clean - y_huge).abs().max().item()) == 0.0

    u_nan = build_mlp_flat_input(x_raw, obs)
    u_nan[:, 4 * 3 : 4 * 3 + 3] = float("nan")
    u_nan[:, 3 * 12 + 4] = 0.0
    y_nan = model(u_nan)
    assert torch.isfinite(y_nan).all()
    assert float((y_clean - y_nan).abs().max().item()) == 0.0

    model.train()
    y_tr = model(u_nan)
    y_tr.pow(2).mean().backward()
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_branch_b_seed_parity_and_in_train_barrier(
    synthetic_dataset: tuple[dict[str, np.ndarray], dict[str, Any]],
) -> None:
    dataset, meta, fold, est, scaler, x_delta, x_work, mask_gen = _make_clean_ep_split(
        synthetic_dataset
    )
    device = resolve_device("cpu")
    in_tr = fold.in_train_mask
    train_ds = SingleStepDataset(
        x_work, dataset["M_valid"], dataset["M_qc_clean"], dataset["time_features"],
        fold.train_idx, in_tr, x_delta_nT=x_delta,
    )
    val_ds = SingleStepDataset(
        x_work, dataset["M_valid"], dataset["M_qc_clean"], dataset["time_features"],
        fold.val_idx, in_tr, x_delta_nT=x_delta,
    )
    cfg = TrainConfig(
        lr=1e-3, batch_size=64, max_epochs=2, warmup_epochs=1,
        w_tgt=4.0, w_obs=0.25, train_mask_mode="m1_m2_m3",
        mask_probs={"M1": 0.35, "M2": 0.35, "M3": 0.30},
        m2_drop_range=(0.20, 0.35), m3_radius_range_km=(500.0, 1000.0),
        min_obs=4, val_m2_repeats=2, seed=42,
    )

    set_seed(42)
    m1 = MaskedWideMLP(n_stations=12, hidden_dim=64, n_blocks=2, dropout=0.15)
    Trainer(m1, cfg, device).fit(train_ds, val_ds, scaler, mask_gen, dataset["station_belt_id"], verbose=False)

    set_seed(42)
    m2 = MaskedWideMLP(n_stations=12, hidden_dim=64, n_blocks=2, dropout=0.15)
    Trainer(m2, cfg, device).fit(train_ds, val_ds, scaler, mask_gen, dataset["station_belt_id"], verbose=False)

    max_w_diff = max(
        float((p1.detach() - p2.detach()).abs().max().item())
        for p1, p2 in zip(m1.parameters(), m2.parameters())
    )
    assert max_w_diff < 1e-6

    metrics, _ = evaluate_model_on_fold(
        m1, dataset, meta, fold, est, scaler, mask_gen,
        eval_split="test", eval_modes=("M1", "M2", "M3_500km", "M3_1000km", "M4"),
        m2_repeats=2, min_obs=4,
    )
    assert metrics["eval_masks"]["zero_shot"] == {"status": "not_supported_fixed_stations"}
    assert metrics["n_in_train_stations"] == int(in_tr.sum())


def test_phase2_gate_report_generation(tmp_path: Path) -> None:
    def _dummy_metrics(rmse_eu: float, rmse_aur: float, branch: str) -> dict[str, Any]:
        rmse_gl = 0.5 * (rmse_eu + rmse_aur)
        eval_masks: dict[str, Any] = {}
        for mode in ("M1", "M2", "M3_500km", "M3_1000km", "M4"):
            eval_masks[mode] = {}
            for belt, val in (
                ("global", rmse_gl),
                ("europe_mid", rmse_eu),
                ("auroral", rmse_aur),
                ("canada", rmse_aur * 0.9),
            ):
                eval_masks[mode][belt] = {
                    c: {"rmse_nT": val, "mae_nT": val * 0.8, "r2": 0.85, "pearson_r": 0.92, "n_points": 100}
                    for c in ("vector_3d", "X", "Y", "Z")
                }
        out: dict[str, Any] = {"eval_masks": eval_masks}
        if branch == "A":
            out["latent_stability"] = {
                "M2_20": {"cos_sim_mean": 0.99, "rel_l2_drift": 0.05, "axis_decorr_frob": 0.1},
                "M2_30": {"cos_sim_mean": 0.98, "rel_l2_drift": 0.08, "axis_decorr_frob": 0.1},
                "M2_40": {"cos_sim_mean": 0.96, "rel_l2_drift": 0.12, "axis_decorr_frob": 0.1},
            }
            out["space_weather_alignment"] = {
                "omni": {"ae": {"status": "ok", "linear_r2": 0.45, "best_axis_abs_pearson_r": 0.65}},
                "solar1_bsn": {"ey_mVm": {"status": "ok", "linear_r2": 0.30, "best_axis_abs_pearson_r": 0.50}},
                "oulu": {"delta_n_pct": {"status": "ok", "linear_r2": 0.25, "best_axis_abs_pearson_r": 0.45}},
            }
        return out

    specs = [
        ("phase1", "A", "empca_k5_zscore", 15.0, 90.0, False, {"k": 5}, {}),
        ("phase1", "B", "ridge_zscore", 10.0, 75.0, False, {}, {}),
        ("phase2", "A", "mlp_ae_k5_ut_decorr_zscore", 14.0, 82.0, False, {}, {"latent_dim": 5}),
        ("phase2", "B", "wide_mlp_zscore", 8.0, 60.0, False, {}, {}),
        # Smoke run must be skipped by check_phase2_gate
        ("phase2", "B", "wide_mlp_zscore", 999.0, 999.0, True, {}, {}),
    ]
    for ph, br, tg, eu, aur, smoke, params, mkwargs in specs:
        for fid in (0, 1):
            suffix = "smoke" if smoke else "real"
            rdir = tmp_path / f"{ph}_{br}_{tg}_f{fid}_{suffix}"
            write_json(
                rdir / "config.json",
                {
                    "run_id": rdir.name,
                    "phase": ph,
                    "branch": br,
                    "tag": tg,
                    "fold_id": fid,
                    "dataset_sha256": "abc123",
                    "split_protocol": "looe_4fold",
                    "scaling": "zscore_per_station",
                    "baseless_policy": "train_quiet_proxy",
                    "zero_shot": False,
                    "smoke": smoke,
                    "params": params,
                    "model_kwargs": mkwargs,
                },
            )
            write_json(rdir / "metrics.json", _dummy_metrics(eu, aur, br))

    gate_mod = _load_gate_module()
    report = gate_mod.build_phase2_gate_report(tmp_path, expected_folds=2)
    assert report["gate_summary"]["branch_a_configs_complete"] is True
    assert report["gate_summary"]["branch_b_configs_complete"] is True
    assert report["gate_summary"]["wide_mlp_beats_ridge_m2_auroral"] is True
    # Compression price on M2 auroral: 82.0 - 60.0 = 22.0 nT
    cp_aur = report["compression_price"]["k5"]["delta_rmse_a_minus_b"]["M2"]["auroral"]["vector_3d"]
    assert cp_aur["delta_a_minus_b_nT"]["mean"] == pytest.approx(22.0)
