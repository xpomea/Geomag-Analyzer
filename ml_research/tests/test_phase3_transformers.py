"""Tests for Phase 3 (`T3.1`–`T3.4`): SpatialSetEncoder, Perceiver-AE, Cross-Attn VM, Zero-Shot, Cloud Packer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
import pytest
import torch

from geomag_ml import ML_ROOT
from geomag_ml.data.datasets import SingleStepDataset
from geomag_ml.data.masks import MaskGenerator
from geomag_ml.data.preprocessing import BaselineEstimator, FieldScaler
from geomag_ml.data.splits import SplitFold, build_station_train_mask
from geomag_ml.eval.evaluator import evaluate_model_on_fold
from geomag_ml.models.cross_attn_vm import CrossAttentionVirtualMagnetometer
from geomag_ml.models.perceiver_ae import PerceiverBottleneckAE
from geomag_ml.models.spatial_encoder import (
    MaskedMultiHeadAttention,
    SpatialSetEncoder,
    build_attn_forbidden,
    build_obs_tokens,
    build_station_coord_features,
    compute_pair_geometry,
)
from geomag_ml.train.artifacts import build_stations_manifest, load_run_bundle, save_run_bundle
from geomag_ml.train.trainer import TrainConfig, Trainer
from geomag_ml.utils.device import resolve_device
from geomag_ml.utils.seed import set_seed


def _make_phase3_split(
    synthetic_dataset: tuple[dict[str, np.ndarray], dict[str, Any]],
    *,
    scaler_mode: str = "asinh_global",
    zero_shot_stations: tuple[str, ...] = (),
):
    dataset, meta = synthetic_dataset
    codes = meta["station_codes"]
    n_st = len(codes)
    if zero_shot_stations:
        st_train_mask, is_zs_mask = build_station_train_mask(codes, zero_shot_stations)
    else:
        st_train_mask = np.ones(n_st, dtype=bool)
        is_zs_mask = np.zeros(n_st, dtype=bool)

    tr_idx = np.arange(0, 220, dtype=np.int64)
    va_idx = np.arange(220, 300, dtype=np.int64)
    te_idx = np.arange(300, 600, dtype=np.int64)

    est = BaselineEstimator(
        mode="temp_train_quiet_proxy",
        baseless_policy="train_quiet_proxy",
        min_train_points=20,
        causal_init_minutes=60,
    ).fit(dataset, tr_idx, station_train_mask=st_train_mask)
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
        station_train_mask=st_train_mask,
    )
    x_work = scaler.transform(x_delta)
    in_train = est.in_train_stations_ & scaler.in_train_stations_ & st_train_mask
    fold = SplitFold(
        fold_id=0,
        protocol="looe_4fold",
        train_idx=tr_idx,
        val_idx=va_idx,
        test_idx=te_idx,
        train_episodes=(0,),
        val_episodes=(0,),
        test_episodes=(1,),
        station_train_mask=st_train_mask,
        is_zero_shot_mask=is_zs_mask,
        in_train_mask=in_train,
    )
    mask_gen = MaskGenerator(dataset["station_coords"], dataset["station_dist_km"], seed=42)
    return dataset, meta, fold, est, scaler, x_delta, x_work, mask_gen


# ---------------------------------------------------------------------------
# T3.1: SpatialSetEncoder + SpatialBiasMLP + FP16-safe Attention
# ---------------------------------------------------------------------------


def test_encoder_pair_geometry_properties_and_antimeridian() -> None:
    coords = torch.tensor(
        [
            [60.0, 178.0, 55.0, 178.0],
            [60.0, -178.0, 55.0, -178.0],
            [45.0, 10.0, 42.0, 90.0],
        ],
        dtype=torch.float32,
    )
    geom = compute_pair_geometry(coords, coords)
    assert geom.shape == (3, 3, 5)
    assert torch.isfinite(geom).all()
    assert torch.allclose(torch.diagonal(geom[:, :, 0]), torch.zeros(3), atol=1e-7)
    assert torch.allclose(torch.diagonal(geom[:, :, 1]), torch.zeros(3), atol=1e-7)
    assert torch.allclose(torch.diagonal(geom[:, :, 2]), torch.zeros(3), atol=1e-7)
    assert torch.allclose(torch.diagonal(geom[:, :, 3]), torch.zeros(3), atol=1e-6)
    assert torch.allclose(torch.diagonal(geom[:, :, 4]), torch.ones(3), atol=1e-6)

    # 4 deg lon at lat 60 deg is ~222 km (< 250 km) even across +-180 antimeridian
    d_anti_km = float(geom[0, 1, 0].item()) * 2000.0
    assert 200.0 < d_anti_km < 250.0
    # sin^2 + cos^2 == 1 for dMLon
    assert torch.allclose(geom[..., 3] ** 2 + geom[..., 4] ** 2, torch.ones(3, 3), atol=1e-5)

    # Arbitrary N_q != N_k works seamlessly
    geom_rect = compute_pair_geometry(coords[:2], coords[1:])
    assert geom_rect.shape == (2, 2, 5)


def test_encoder_permutation_equivariance() -> None:
    set_seed(11)
    b, n_st = 3, 12
    enc = SpatialSetEncoder(
        d_in=9, d_model=128, n_layers=3, n_heads=4, d_ff=256, dropout=0.0, use_station_emb=False
    )
    enc.eval()

    x = torch.randn(b, n_st, 3)
    coords = torch.stack(
        [
            torch.linspace(45.0, 72.0, n_st),
            torch.linspace(-110.0, 30.0, n_st),
            torch.linspace(43.0, 70.0, n_st),
            torch.linspace(-20.0, 120.0, n_st),
        ],
        dim=-1,
    )
    mlt = torch.rand(b, n_st) * 24.0
    obs = torch.ones(b, n_st, dtype=torch.bool)
    obs[0, 3] = False
    obs[1, 7] = False

    tokens = build_obs_tokens(x, coords, mlt, obs)
    geom = compute_pair_geometry(coords, coords).unsqueeze(0)
    h_orig = enc(tokens, obs, pair_geom=geom)

    perm = torch.tensor([5, 0, 11, 2, 9, 1, 7, 3, 10, 4, 8, 6], dtype=torch.int64)
    tokens_p = tokens[:, perm, :]
    obs_p = obs[:, perm]
    coords_p = coords[perm]
    geom_p = compute_pair_geometry(coords_p, coords_p).unsqueeze(0)
    h_perm = enc(tokens_p, obs_p, pair_geom=geom_p)

    assert float((h_orig[:, perm, :] - h_perm).abs().max().item()) < 1e-5


def test_encoder_adversarial_mask_leakage_and_nan_safety() -> None:
    set_seed(12)
    enc = SpatialSetEncoder(d_in=9, d_model=64, n_layers=2, n_heads=4, d_ff=128, dropout=0.0)
    enc.eval()

    b, n_st = 3, 10
    coords = torch.rand(n_st, 4) * 30.0
    mlt = torch.rand(b, n_st) * 24.0
    x = torch.randn(b, n_st, 3)
    obs = torch.ones(b, n_st, dtype=torch.bool)
    obs[:, 2] = False
    obs[1, 6] = False
    geom = compute_pair_geometry(coords, coords).unsqueeze(0)

    t_clean = build_obs_tokens(x, coords, mlt, obs)
    h_clean = enc(t_clean, obs, pair_geom=geom)

    # Inject +1e6 and NaN directly at masked positions (bypassing build_obs_tokens too)
    t_adv = t_clean.clone()
    t_adv[:, 2, :] = 1e6
    t_adv[1, 6, :] = float("nan")
    t_adv.requires_grad_(True)
    h_adv = enc(t_adv, obs, pair_geom=geom)

    assert torch.isfinite(h_adv).all()
    assert float((h_clean - h_adv).abs().max().item()) < 1e-6
    assert float(h_adv[:, 2, :].abs().max().item()) == 0.0
    assert float(h_adv[1, 6, :].abs().max().item()) == 0.0

    loss = h_adv.pow(2).sum()
    loss.backward()
    assert t_adv.grad is not None and torch.isfinite(t_adv.grad).all()
    assert float(t_adv.grad[:, 2, :].abs().max().item()) == 0.0
    assert float(t_adv.grad[1, 6, :].abs().max().item()) == 0.0


def test_encoder_fp16_and_all_masked_safety() -> None:
    set_seed(13)
    # 1. Explicit float16 attention test (exercises the -1e9 overflow guard directly)
    attn = MaskedMultiHeadAttention(d_model=32, n_heads=4, dropout=0.0).half()
    q_h = torch.randn(2, 5, 32, dtype=torch.float16, requires_grad=True)
    k_h = torch.randn(2, 5, 32, dtype=torch.float16, requires_grad=True)
    v_h = torch.randn(2, 5, 32, dtype=torch.float16, requires_grad=True)
    # Row 0 has all keys forbidden, Row 1 has keys 0..2 forbidden
    kf = torch.tensor([[True, True, True, True, True], [True, True, True, False, False]])
    bias_h = torch.randn(1, 4, 5, 5, dtype=torch.float16)
    out_h = attn(q_h, k_h, v_h, key_forbidden=kf, attn_bias=bias_h)
    assert out_h.dtype == torch.float16
    assert torch.isfinite(out_h).all()
    out_h.float().sum().backward()
    assert q_h.grad is not None and torch.isfinite(q_h.grad).all()

    # 2. Full SpatialSetEncoder under autocast + all-masked row
    enc = SpatialSetEncoder(d_in=9, d_model=64, n_layers=2, n_heads=4, d_ff=128, dropout=0.0)
    coords = torch.rand(8, 4) * 30.0
    geom = compute_pair_geometry(coords, coords).unsqueeze(0)
    tokens = torch.randn(3, 8, 9)
    obs = torch.ones(3, 8, dtype=torch.bool)
    obs[0, :] = False  # completely empty observation row
    tokens[0, :, :] = float("nan")

    with torch.amp.autocast("cpu", dtype=torch.bfloat16, enabled=True):
        h = enc(tokens, obs, pair_geom=geom)
    assert torch.isfinite(h).all()
    assert float(h[0].abs().max().item()) == 0.0
    h.float().sum().backward()
    for p in enc.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_encoder_variable_station_count_and_configurable_d_in() -> None:
    enc9 = SpatialSetEncoder(d_in=9, d_model=128, n_layers=3, n_heads=4, d_ff=256)
    enc9.eval()
    for n_st in (49, 35, 60):
        coords = torch.rand(n_st, 4) * 20.0
        geom = compute_pair_geometry(coords, coords).unsqueeze(0)
        tokens = torch.randn(2, n_st, 9)
        obs = torch.ones(2, n_st, dtype=torch.bool)
        h = enc9(tokens, obs, pair_geom=geom)
        assert h.shape == (2, n_st, 128)

    # Phase 4 (d_in=105) and T5.2 (d_in=18) encoder configurations
    for d_in in (18, 105):
        enc_cfg = SpatialSetEncoder(d_in=d_in, d_model=128, n_layers=2, n_heads=4, d_ff=256)
        coords = torch.rand(12, 4) * 20.0
        geom = compute_pair_geometry(coords, coords).unsqueeze(0)
        h = enc_cfg(torch.randn(2, 12, d_in), torch.ones(2, 12, dtype=torch.bool), pair_geom=geom)
        assert h.shape == (2, 12, 128)


def test_encoder_station_emb_unk_and_zero_shot() -> None:
    enc = SpatialSetEncoder(
        d_in=9, d_model=64, n_layers=1, n_heads=4, d_ff=128, dropout=0.0,
        use_station_emb=True, max_stations=10, station_emb_dim=8, station_emb_dropout=0.0,
    )
    in_tr = [True] * 8 + [False, False]
    enc.set_in_train_stations(in_tr)
    enc.eval()

    # -1, >= max_stations, and ~in_train (idx 8, 9) must all map identically to <UNK>=max_stations
    ids_unk = torch.tensor([[-1, 10, 15, 8, 9]])
    emb_unk = enc.embed_station_ids(ids_unk, 1, 5, torch.device("cpu"), torch.float32)
    ref = enc.station_emb.weight[10]
    for col in range(5):
        assert torch.allclose(emb_unk[0, col], ref, atol=1e-7)

    # None station_ids also maps cleanly to <UNK>
    emb_none = enc.embed_station_ids(None, 2, 4, torch.device("cpu"), torch.float32)
    assert emb_none.shape == (2, 4, 8)
    assert torch.allclose(emb_none[0, 0], ref, atol=1e-7)


# ---------------------------------------------------------------------------
# T3.2: Branch A — PerceiverBottleneckAE
# ---------------------------------------------------------------------------


def test_perceiver_shapes_param_count_and_configurable_io() -> None:
    model_std = PerceiverBottleneckAE(
        d_in=9, d_out=3, latent_dim=5, time_cond_dim=2, bottleneck_mode="ae_decorr"
    )
    n_params = model_std.count_parameters()
    assert 770_000 <= n_params <= 800_000  # §6.3: ~785K params

    coords = torch.rand(12, 4) * 20.0
    geom = compute_pair_geometry(coords, coords).unsqueeze(0)
    obs = torch.ones(4, 12, dtype=torch.bool)
    tokens = torch.randn(4, 12, 9)

    for d_t in (0, 2, 4):
        m = PerceiverBottleneckAE(d_in=9, d_out=3, latent_dim=4, time_cond_dim=d_t)
        q_c = torch.randn(4, 7, 6 + d_t)  # variable N_q = 7 != N_obs = 12
        y_hat, z = m(tokens, obs, geom, q_c)
        assert y_hat.shape == (4, 7, 3)
        assert z.shape == (4, 4)
        spec = m.onnx_export_spec()
        assert spec["opset"] == 17 and spec["supported"] is True
        assert spec["dynamic_axes"]["q_coords"][1] == "N_q"

    # T5.2 Neuro-ULF configuration: d_in=18, d_out=8
    m_ulf = PerceiverBottleneckAE(d_in=18, d_out=8, latent_dim=5, time_cond_dim=0, use_spatial_bias=False)
    y_ulf, z_ulf = m_ulf(torch.randn(3, 12, 18), obs[:3], torch.randn(3, 5, 6))
    assert y_ulf.shape == (3, 5, 8)
    assert z_ulf.shape == (3, 5)
    assert m_ulf.onnx_export_spec()["input_names"] == ["obs_tokens", "obs_mask", "q_coords"]

    # beta-VAE eval determinism
    bvae = PerceiverBottleneckAE(d_in=9, d_out=3, latent_dim=5, time_cond_dim=2, bottleneck_mode="beta_vae")
    q_c2 = torch.randn(4, 5, 8)
    bvae.train()
    _, z_tr1 = bvae(tokens, obs, geom, q_c2)
    _, z_tr2 = bvae(tokens, obs, geom, q_c2)
    assert float((z_tr1 - z_tr2).abs().max().item()) > 1e-6
    bvae.eval()
    y_ev1, z_ev1 = bvae(tokens, obs, geom, q_c2)
    y_ev2, z_ev2 = bvae(tokens, obs, geom, q_c2)
    assert float((z_ev1 - z_ev2).abs().max().item()) == 0.0
    assert float((y_ev1 - y_ev2).abs().max().item()) == 0.0


def test_perceiver_adversarial_mask_leakage_and_nan_safety() -> None:
    set_seed(21)
    m = PerceiverBottleneckAE(d_in=9, d_out=3, latent_dim=5, time_cond_dim=2, dropout=0.0)
    m.eval()

    b, n_st = 4, 12
    coords = torch.rand(n_st, 4) * 30.0
    mlt = torch.rand(b, n_st) * 24.0
    x = torch.randn(b, n_st, 3)
    obs = torch.ones(b, n_st, dtype=torch.bool)
    obs[:, 4] = False
    geom = compute_pair_geometry(coords, coords).unsqueeze(0)
    q_coords = build_station_coord_features(coords, mlt, c_time=torch.randn(b, 2))

    t_clean = build_obs_tokens(x, coords, mlt, obs)
    y_clean, z_clean = m(t_clean, obs, geom, q_coords)

    x_huge = x.clone()
    x_huge[:, 4, :] = 1e6
    y_huge, z_huge = m(build_obs_tokens(x_huge, coords, mlt, obs), obs, geom, q_coords)
    assert float((z_clean - z_huge).abs().max().item()) < 1e-6
    assert float((y_clean - y_huge).abs().max().item()) < 1e-6

    t_nan = t_clean.clone()
    t_nan[:, 4, :] = float("nan")
    y_nan, z_nan = m(t_nan, obs, geom, q_coords)
    assert torch.isfinite(y_nan).all() and torch.isfinite(z_nan).all()
    assert float((z_clean - z_nan).abs().max().item()) < 1e-6


def test_perceiver_trainer_smoke_and_bundle_roundtrip(
    synthetic_dataset: tuple[dict[str, np.ndarray], dict[str, Any]],
    tmp_path: Path,
) -> None:
    set_seed(42)
    device = resolve_device("cpu")
    dataset, meta, fold, est, scaler, x_delta, x_work, mask_gen = _make_phase3_split(
        synthetic_dataset, scaler_mode="asinh_global"
    )
    in_tr = fold.in_train_mask

    train_ds = SingleStepDataset(
        x_work, dataset["M_valid"], dataset["M_qc_clean"], dataset["time_features"],
        fold.train_idx, in_tr, mlt_hours=dataset["mlt_hours"], x_delta_nT=x_delta,
    )
    val_ds = SingleStepDataset(
        x_work, dataset["M_valid"], dataset["M_qc_clean"], dataset["time_features"],
        fold.val_idx, in_tr, mlt_hours=dataset["mlt_hours"], x_delta_nT=x_delta,
    )

    model = PerceiverBottleneckAE(
        d_in=9, d_out=3, latent_dim=4, time_cond_dim=2, bottleneck_mode="ae_decorr",
        d_model=64, n_enc_layers=2, n_heads=4, d_ff=128, n_lat_queries=4, dropout=0.1,
    )
    cfg = TrainConfig(
        lr=5e-4, batch_size=64, max_epochs=2, warmup_epochs=1,
        w_tgt=2.0, w_obs=1.0, lambda_cov=0.05, lambda_cons=0.1,
        train_mask_mode="m1_m2", mask_probs={"M1": 0.3, "M2": 0.7},
        m2_drop_range=(0.15, 0.35), min_obs=4, val_m2_repeats=2, eval_batch_size=128, seed=42,
    )
    res = Trainer(model, cfg, device).fit(
        train_ds, val_ds, scaler, mask_gen, dataset["station_belt_id"], verbose=False
    )
    assert res["epochs_ran"] == 2

    metrics, outputs = evaluate_model_on_fold(
        model, dataset, meta, fold, est, scaler, mask_gen,
        eval_split="test", eval_modes=("M1", "M2"), m2_repeats=2, min_obs=4,
        stability_drop_fracs=(0.2, 0.3, 0.4), stability_repeats=2, return_outputs=True,
    )
    assert set(metrics["latent_stability"].keys()) == {"M2_20", "M2_30", "M2_40"}
    assert np.isfinite(metrics["eval_masks"]["M2"]["global"]["vector_3d"]["rmse_nT"])
    assert "z_test" in outputs

    run_dir = tmp_path / "p3_A_perceiver_smoke"
    save_run_bundle(
        run_dir, model=model, best_epoch=res["best_epoch"], best_val_metric_nT=res["best_val_metric_nT"],
        resolved_config={"phase": "phase3", "branch": "A", "tag": "smoke", "fold_id": 0, "seed": 42},
        baseline=est, scaler=scaler,
        stations_manifest=build_stations_manifest(meta, dataset, in_tr, fold.is_zero_shot_mask),
        metrics=metrics, history=res["history"], outputs=outputs,
    )
    bundle = load_run_bundle(run_dir, device="cpu")
    re_metrics, _ = evaluate_model_on_fold(
        bundle["model"], dataset, meta, fold, bundle["baseline"], bundle["scaler"], mask_gen,
        eval_split="test", eval_modes=("M1", "M2"), m2_repeats=2, min_obs=4,
        stability_drop_fracs=(0.2, 0.3, 0.4), stability_repeats=2, return_outputs=False,
    )
    r1 = float(metrics["eval_masks"]["M2"]["global"]["vector_3d"]["rmse_nT"])
    r2 = float(re_metrics["eval_masks"]["M2"]["global"]["vector_3d"]["rmse_nT"])
    assert abs(r1 - r2) < 1e-4


# ---------------------------------------------------------------------------
# T3.3: Branch B — CrossAttentionVirtualMagnetometer (+ Zero-Shot)
# ---------------------------------------------------------------------------


def test_cross_attn_shapes_param_count_and_onnx_spec() -> None:
    model_std = CrossAttentionVirtualMagnetometer(
        d_in=9, d_out=3, time_cond_dim=2, use_spatial_bias=True, use_station_emb=False
    )
    n_params = model_std.count_parameters()
    assert 660_000 <= n_params <= 690_000  # §6.3: ~675K params

    spec = model_std.onnx_export_spec()
    assert spec["opset"] == 17 and spec["supported"] is True
    assert spec["dynamic_axes"]["obs_tokens"][1] == "N_obs"
    assert spec["dynamic_axes"]["q_coords"][1] == "N_q"

    b, n_obs, n_q = 3, 12, 4
    obs = torch.ones(b, n_obs, dtype=torch.bool)
    obs[:, :4] = False
    tokens = torch.randn(b, n_obs, 9)
    q_coords = torch.randn(b, n_q, 8)
    forbidden = build_attn_forbidden(obs, n_q=n_q)

    # Spatial bias enabled
    geom = torch.randn(1, n_obs, n_obs, 5)
    q_geom = torch.randn(1, n_q, n_obs, 5)
    y = model_std(tokens, obs, geom, q_coords, q_geom, forbidden)
    assert y.shape == (b, n_q, 3)

    # Spatial bias disabled (4-arg ONNX signature)
    m_nobias = CrossAttentionVirtualMagnetometer(
        d_in=9, d_out=3, time_cond_dim=2, use_spatial_bias=False, use_station_emb=False
    )
    assert m_nobias.onnx_export_spec()["input_names"] == [
        "obs_tokens", "obs_mask", "q_coords", "attn_forbidden"
    ]
    y_nb = m_nobias(tokens, obs, q_coords, forbidden)
    assert y_nb.shape == (b, n_q, 3)


def test_cross_attn_adversarial_leakage_multihop_and_attn_forbidden() -> None:
    set_seed(31)
    model = CrossAttentionVirtualMagnetometer(
        d_in=9, d_out=3, time_cond_dim=2, d_model=64, n_enc_layers=2, n_cross_layers=2,
        n_heads=4, d_ff=128, dropout=0.0, use_spatial_bias=True, use_station_emb=False,
    )
    model.eval()

    b, n_st = 4, 12
    coords = torch.rand(n_st, 4) * 30.0
    mlt = torch.rand(b, n_st) * 24.0
    x = torch.randn(b, n_st, 3)
    c_time = torch.randn(b, 2)
    target_j = 5

    obs = torch.ones(b, n_st, dtype=torch.bool)
    obs[:, target_j] = False  # Must be masked BEFORE SpatialSetEncoder (multi-hop guard)
    tgt = torch.zeros(b, n_st, dtype=torch.bool)
    tgt[:, target_j] = True

    geom = compute_pair_geometry(coords, coords).unsqueeze(0)
    k_ids = torch.arange(n_st, dtype=torch.int64)
    q_ids = torch.arange(n_st, dtype=torch.int64)
    q_coords = build_station_coord_features(coords, mlt, c_time=c_time)
    forbidden = build_attn_forbidden(obs, q_ids, k_ids)

    t_clean = build_obs_tokens(x, coords, mlt, obs)
    h_clean = model.encoder(t_clean, obs, pair_geom=geom)
    y_clean = model(t_clean, obs, geom, q_coords, geom, forbidden, k_station_ids=k_ids, q_station_ids=q_ids)

    # Adversarial perturbation of hidden station j with +1e6 and NaN
    x_huge = x.clone()
    x_huge[:, target_j, :] = 1e6
    t_huge = build_obs_tokens(x_huge, coords, mlt, obs)
    h_huge = model.encoder(t_huge, obs, pair_geom=geom)
    y_huge = model(t_huge, obs, geom, q_coords, geom, forbidden, k_station_ids=k_ids, q_station_ids=q_ids)

    assert float((h_clean - h_huge).abs().max().item()) == 0.0
    assert float((y_clean[:, target_j] - y_huge[:, target_j]).abs().max().item()) < 1e-6

    t_nan = t_clean.clone()
    t_nan[:, target_j, :] = float("nan")
    y_nan = model(t_nan, obs, geom, q_coords, geom, forbidden, k_station_ids=k_ids, q_station_ids=q_ids)
    assert torch.isfinite(y_nan).all()
    assert float((y_clean[:, target_j] - y_nan[:, target_j]).abs().max().item()) < 1e-6

    # Direct attn_forbidden test: even if obs_mask[j] == True, q_id == k_id blocks direct self-key attention
    obs_all = torch.ones(b, n_st, dtype=torch.bool)
    forb_self = build_attn_forbidden(obs_all, q_ids, k_ids)
    assert bool(forb_self[:, target_j, target_j].all().item())
    # Perturbing encoder output directly at token `target_j` cannot leak directly to query `target_j`
    h_base = torch.randn(b, n_st, 64)
    h_pert = h_base.clone()
    h_pert[:, target_j, :] += 1e5
    y_d1 = model.decode_from_hidden(h_base, q_coords, geom, forb_self, q_station_ids=q_ids)
    y_d2 = model.decode_from_hidden(h_pert, q_coords, geom, forb_self, q_station_ids=q_ids)
    assert float((y_d1[:, target_j] - y_d2[:, target_j]).abs().max().item()) < 1e-5

    # Trainer enforces R8 guard (w_obs == 0.0 and active masks required)
    with pytest.raises(ValueError, match="w_obs == 0.0"):
        Trainer(model, TrainConfig(w_obs=0.25), resolve_device("cpu"))
    with pytest.raises(ValueError, match="train_mask_mode != 'none'"):
        Trainer(model, TrainConfig(w_obs=0.0, train_mask_mode="none"), resolve_device("cpu"))


def test_cross_attn_zero_shot_invariance_and_arbitrary_coords() -> None:
    set_seed(32)
    # Train/init on 10 stations, predict 3 unseen stations with arbitrary coordinates (q_station_id = -1)
    for use_emb in (False, True):
        model = CrossAttentionVirtualMagnetometer(
            d_in=9, d_out=3, time_cond_dim=2, d_model=64, n_enc_layers=2, n_cross_layers=2,
            n_heads=4, d_ff=128, dropout=0.0, use_spatial_bias=True,
            use_station_emb=use_emb, max_stations=10, station_emb_dim=8, station_emb_dropout=0.0,
        )
        model.set_in_train_stations([True] * 10)
        model.eval()

        obs_coords = torch.tensor(
            [[45.0 + i * 2.0, 5.0 + i * 3.0, 42.0 + i * 2.0, 85.0 + i * 3.0] for i in range(10)],
            dtype=torch.float32,
        )
        query_coords = torch.tensor(
            [
                [50.0, 12.0, 48.0, 92.0],
                [61.0, 18.0, 59.0, 99.0],
                [53.5, -113.0, 61.0, -55.0],
            ],
            dtype=torch.float32,
        )
        t_len = 5
        x_obs = np.random.default_rng(1).normal(0.0, 1.0, size=(t_len, 10, 3)).astype(np.float32)
        obs_mask = np.ones((t_len, 10), dtype=bool)
        obs_mlt = np.full((t_len, 10), 12.0, dtype=np.float32)
        query_mlt = np.full((t_len, 3), 12.0, dtype=np.float32)
        tf = np.tile(np.array([[0.0, 1.0, 0.0, 1.0]], dtype=np.float32), (t_len, 1))

        pred = model.predict_at_coordinates(
            x_obs, obs_mask, obs_coords.numpy(), obs_mlt, query_coords.numpy(), query_mlt,
            time_features=tf,
        )
        assert pred.shape == (t_len, 3, 3)
        assert np.all(np.isfinite(pred))

        # Reordering observed stations must leave Zero-Shot query prediction invariant when use_station_emb=False
        if not use_emb:
            perm = np.array([9, 3, 1, 7, 0, 5, 8, 2, 6, 4])
            pred_perm = model.predict_at_coordinates(
                x_obs[:, perm, :], obs_mask[:, perm], obs_coords.numpy()[perm], obs_mlt[:, perm],
                query_coords.numpy(), query_mlt, time_features=tf,
            )
            assert np.max(np.abs(pred - pred_perm)) < 1e-5


def test_cross_attn_trainer_smoke_zero_shot_and_bundle_roundtrip(
    synthetic_dataset: tuple[dict[str, np.ndarray], dict[str, Any]],
    tmp_path: Path,
) -> None:
    set_seed(42)
    device = resolve_device("cpu")
    dataset, meta, fold, est, scaler, x_delta, x_work, mask_gen = _make_phase3_split(
        synthetic_dataset, scaler_mode="asinh_global", zero_shot_stations=("BEL", "LVV", "MEA")
    )
    in_tr = fold.in_train_mask
    assert not in_tr[meta["station_codes"].index("BEL")]
    assert not in_tr[meta["station_codes"].index("LVV")]
    assert not in_tr[meta["station_codes"].index("MEA")]

    train_ds = SingleStepDataset(
        x_work, dataset["M_valid"], dataset["M_qc_clean"], dataset["time_features"],
        fold.train_idx, in_tr, mlt_hours=dataset["mlt_hours"], x_delta_nT=x_delta,
    )
    val_ds = SingleStepDataset(
        x_work, dataset["M_valid"], dataset["M_qc_clean"], dataset["time_features"],
        fold.val_idx, in_tr, mlt_hours=dataset["mlt_hours"], x_delta_nT=x_delta,
    )

    model = CrossAttentionVirtualMagnetometer(
        d_in=9, d_out=3, time_cond_dim=2, d_model=64, n_enc_layers=2, n_cross_layers=2,
        n_heads=4, d_ff=128, dropout=0.1, use_spatial_bias=True,
        use_station_emb=False, max_stations=12,
    )
    cfg = TrainConfig(
        lr=5e-4, batch_size=64, max_epochs=2, warmup_epochs=1,
        w_tgt=1.0, w_obs=0.0, train_mask_mode="m1_m2_m3",
        mask_probs={"M1": 0.30, "M2": 0.35, "M3": 0.35},
        m2_drop_range=(0.20, 0.35), m3_radius_range_km=(500.0, 1000.0),
        min_obs=4, station_dropout_count=(1, 2), val_m2_repeats=2, eval_batch_size=128, seed=42,
    )
    res = Trainer(model, cfg, device).fit(
        train_ds, val_ds, scaler, mask_gen, dataset["station_belt_id"], verbose=False
    )
    assert res["epochs_ran"] == 2

    metrics, outputs = evaluate_model_on_fold(
        model, dataset, meta, fold, est, scaler, mask_gen,
        eval_split="test", eval_modes=("M1", "M2", "M3_500km", "M4"),
        m2_repeats=2, min_obs=4, return_outputs=True,
    )
    zs = metrics["eval_masks"]["zero_shot"]
    assert "heldout_zero_shot" in zs and "single_episode_unseen" in zs
    assert np.isfinite(zs["heldout_zero_shot"]["global"]["vector_3d"]["rmse_nT"])
    assert set(zs["per_station"].keys()) == {"BEL", "LVV", "MEA", "SUA"}
    assert "y_pred_m1" in outputs and "y_pred_zs" in outputs

    run_dir = tmp_path / "p3_B_cross_attn_zs_smoke"
    save_run_bundle(
        run_dir, model=model, best_epoch=res["best_epoch"], best_val_metric_nT=res["best_val_metric_nT"],
        resolved_config={"phase": "phase3", "branch": "B", "tag": "zs_smoke", "fold_id": 0, "seed": 42},
        baseline=est, scaler=scaler,
        stations_manifest=build_stations_manifest(meta, dataset, in_tr, fold.is_zero_shot_mask),
        metrics=metrics, history=res["history"], outputs=outputs,
    )
    bundle = load_run_bundle(run_dir, device="cpu")
    re_metrics, _ = evaluate_model_on_fold(
        bundle["model"], dataset, meta, fold, bundle["baseline"], bundle["scaler"], mask_gen,
        eval_split="test", eval_modes=("M1", "M2"), m2_repeats=2, min_obs=4, return_outputs=False,
    )
    zs_orig = float(metrics["eval_masks"]["zero_shot"]["heldout_zero_shot"]["global"]["vector_3d"]["rmse_nT"])
    zs_re = float(re_metrics["eval_masks"]["zero_shot"]["heldout_zero_shot"]["global"]["vector_3d"]["rmse_nT"])
    assert abs(zs_orig - zs_re) < 1e-4


# ---------------------------------------------------------------------------
# T3.4: Cloud Transfer Bundle Packer & Thin Notebooks
# ---------------------------------------------------------------------------


def test_cloud_bundle_packer_and_notebooks_valid(
    synthetic_dataset: tuple[dict[str, np.ndarray], dict[str, Any]],
    tmp_path: Path,
) -> None:
    pack_path = ML_ROOT / "scripts" / "pack_for_cloud.py"
    spec = importlib.util.spec_from_file_location("pack_for_cloud", pack_path)
    assert spec is not None and spec.loader is not None
    pack_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pack_mod)

    arrays, meta = synthetic_dataset
    ds_file = tmp_path / "syn_clean.npz"
    np.savez_compressed(ds_file, **arrays)
    (tmp_path / "syn_clean.meta.json").write_text(json.dumps(meta), encoding="utf-8")

    out_zip = tmp_path / "cloud_bundle.zip"
    summary = pack_mod.build_cloud_bundle(
        ds_file, out_zip, ml_root=ML_ROOT, include_tests=True, max_size_mb=25.0
    )
    assert out_zip.is_file()
    assert summary["size_mb"] < 25.0

    with zipfile.ZipFile(out_zip, "r") as zf:
        names = set(zf.namelist())
        assert "pyproject.toml" in names
        assert "pytest.ini" in names
        assert "requirements_colab_gpu.txt" in names
        assert "bundle_manifest.json" in names
        assert "data_cache/syn_clean.npz" in names
        assert "data_cache/syn_clean.meta.json" in names
        assert "geomag_ml/models/spatial_encoder.py" in names
        assert "configs/phase3_branch_a_perceiver.yaml" in names
        assert "configs/phase3_branch_b_cross_attn.yaml" in names
        assert "scripts/train.py" in names
        assert not any(n.startswith(".venv") or "/.venv" in n for n in names)
        assert not any(n.startswith("artifacts/") for n in names)
        assert not any("__pycache__" in n for n in names)

    nb_dir = ML_ROOT / "notebooks"
    if nb_dir.is_dir():
        for nb_name in ("colab_runner.ipynb", "kaggle_runner.ipynb"):
            nb_path = nb_dir / nb_name
            assert nb_path.is_file()
            nb_json = json.loads(nb_path.read_text(encoding="utf-8"))
            assert nb_json["nbformat"] == 4
            all_src = "\n".join("".join(c.get("source", [])) for c in nb_json["cells"])
            assert "scripts/train.py" in all_src
            assert "class SpatialSetEncoder" not in all_src
            assert "class PerceiverBottleneckAE" not in all_src
