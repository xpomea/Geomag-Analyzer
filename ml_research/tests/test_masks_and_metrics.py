"""Phase 1 (T1.2 & T1.3) tests: MaskGenerator invariants, metrics, and evaluator barriers."""

from __future__ import annotations

import numpy as np
import pytest

from geomag_ml.data.masks import MaskGenerator, assert_mask_invariants
from geomag_ml.data.schema import AURORAL_MLAT_THRESHOLD_DEG


def test_mask_generator_m1_m2_m3_m4_invariants(
    synthetic_dataset: tuple[dict[str, np.ndarray], dict],
) -> None:
    """Verify M1–M4 masks satisfy disjointness, count/cluster constraints, determinism, and in_train isolation."""
    arrays, meta = synthetic_dataset
    codes = meta["station_codes"]
    sua_idx = codes.index("SUA")

    val = arrays["M_valid"]
    qc = arrays["M_qc_clean"]
    coords = arrays["station_coords"]
    dist = arrays["station_dist_km"]

    # Exclude SUA as ~in_train (mimics Fold 3 where SUA has 0 training samples)
    in_train = np.ones(len(codes), dtype=bool)
    in_train[sua_idx] = False

    gen_a = MaskGenerator(coords, dist, seed=42)
    gen_b = MaskGenerator(coords, dist, seed=42)

    # M1: deterministic leave-one-station-out
    m1_a = gen_a.generate_m1(val, qc, in_train)
    m1_b = gen_b.generate_m1(val, qc, in_train)
    assert len(m1_a) == int(np.sum(in_train))
    for sa, sb in zip(m1_a, m1_b):
        assert np.array_equal(sa.obs_mask, sb.obs_mask)
        assert np.array_equal(sa.target_mask, sb.target_mask)
        assert np.all(np.sum(sa.target_mask, axis=1) <= 1)
        assert not np.any(sa.target_mask[:, sua_idx])
        assert not np.any(sa.obs_mask[:, sua_idx])

    # M2: random 25% drop, deterministic across repeats
    m2_a = gen_a.generate_m2(val, qc, in_train, drop_frac=0.25, n_repeats=3, min_obs=5)
    m2_b = gen_b.generate_m2(val, qc, in_train, drop_frac=0.25, n_repeats=3, min_obs=5)
    assert len(m2_a) == 3
    for sa, sb in zip(m2_a, m2_b):
        assert np.array_equal(sa.obs_mask, sb.obs_mask)
        assert np.array_equal(sa.target_mask, sb.target_mask)
        assert np.all(np.sum(sa.obs_mask, axis=1) >= 5)
        n_dropped = np.sum(sa.target_mask, axis=1)
        assert np.mean(n_dropped) > 1.5
        assert not np.any(sa.target_mask[:, sua_idx])

    # M3 (500km & 1000km): clusters >= 2 stations
    m3_500 = gen_a.generate_m3(val, qc, in_train, radius_km=500.0)
    assert len(m3_500) >= 1
    for sc in m3_500:
        active = np.any(sc.target_mask, axis=1)
        assert np.all(np.sum(sc.target_mask[active], axis=1) >= 2)
        assert not np.any(sc.target_mask[:, sua_idx])

    m3_1000 = gen_a.generate_m3(val, qc, in_train, radius_km=1000.0)
    assert len(m3_1000) >= len(m3_500)

    # M4: all auroral in_train stations in target, only non-auroral in obs
    m4 = gen_a.generate_m4(val, qc, in_train)
    assert len(m4) == 1
    aur_mask = np.abs(coords[:, 2]) >= AURORAL_MLAT_THRESHOLD_DEG
    assert not np.any(m4[0].obs_mask[:, aur_mask])
    assert not np.any(m4[0].target_mask[:, ~aur_mask])
    assert not np.any(m4[0].target_mask[:, sua_idx])


def test_mask_generator_zero_shot_and_train_mixture(
    synthetic_dataset: tuple[dict[str, np.ndarray], dict],
) -> None:
    """Verify Zero-Shot submask split (heldout vs single-episode unseen) and train batch mixture."""
    arrays, meta = synthetic_dataset
    codes = meta["station_codes"]
    val = arrays["M_valid"]
    qc = arrays["M_qc_clean"]

    is_zs = np.array([c in ("BEL", "LVV", "MEA") for c in codes], dtype=bool)
    in_train = ~is_zs
    in_train[codes.index("SUA")] = False  # SUA is unseen but NOT in heldout_zero_shot

    gen = MaskGenerator(arrays["station_coords"], arrays["station_dist_km"], seed=7)
    zs = gen.generate_zero_shot(val, qc, in_train, is_zs)

    assert int(np.sum(zs.obs_mask[:, ~in_train])) == 0
    assert int(np.sum(zs.target_mask[:, in_train])) == 0

    heldout = zs.meta["heldout_zero_shot_mask"]
    single_ep = zs.meta["single_episode_unseen_mask"]
    assert np.array_equal(heldout | single_ep, zs.target_mask)
    assert int(np.sum(heldout & single_ep)) == 0
    assert np.any(heldout[:, codes.index("BEL")])
    assert np.any(single_ep[:, codes.index("SUA")])
    assert not np.any(heldout[:, codes.index("SUA")])

    # Train batch mixture with station-dropout augmentation
    rng = np.random.default_rng(123)
    obs_tr, tgt_tr = gen.sample_train_batch_masks(
        val[:64],
        qc[:64],
        in_train,
        probs={"M1": 0.30, "M2": 0.35, "M3": 0.35},
        min_obs=4,
        station_dropout_count=(1, 2),
        rng=rng,
    )
    assert obs_tr.shape == (64, len(codes))
    assert np.any(tgt_tr)
    assert int(np.sum(obs_tr[:, ~in_train])) == 0
    assert int(np.sum(tgt_tr[:, ~in_train])) == 0

    # Explicit negative test for assert_mask_invariants
    bad_obs = zs.obs_mask.copy()
    bad_obs[0, codes.index("BEL")] = True
    with pytest.raises(AssertionError):
        assert_mask_invariants(bad_obs, zs.target_mask, val, qc, in_train, mode_kind="zero_shot")


# ---------------------------------------------------------------------------
# Phase 1 (T1.3): Metrics & Evaluator tests
# ---------------------------------------------------------------------------


def test_compute_reconstruction_metrics_analytic_and_belts() -> None:
    """Verify exact analytical RMSE/MAE/R2/Pearson and belt aggregation."""
    from geomag_ml.eval.metrics import compute_reconstruction_metrics

    # 4 stations covering all 4 belt IDs (0=EU_mid, 1=EU_aur, 2=CA_mid, 3=CA_aur)
    belts = np.array([0, 1, 2, 3], dtype=np.int8)
    t_len = 20
    t_grid = np.linspace(0.0, 2.0 * np.pi, t_len)[:, None]
    y_true = np.zeros((t_len, 4, 3), dtype=np.float32)
    y_true[:, :, 0] = 10.0 * np.sin(t_grid)
    y_true[:, :, 1] = 20.0 * np.cos(t_grid)
    y_true[:, :, 2] = 5.0 * np.sin(2.0 * t_grid)

    eval_mask = np.ones((t_len, 4), dtype=bool)
    perfect = compute_reconstruction_metrics(y_true, y_true.copy(), eval_mask, belts)
    for belt_name, comps in perfect.items():
        assert comps["X"]["n_points"] == (t_len * 2 if belt_name in ("auroral", "canada") else (t_len * 4 if belt_name == "global" else t_len))
        for key in ("X", "Y", "Z", "vector_3d"):
            assert comps[key]["rmse_nT"] == pytest.approx(0.0, abs=1e-6)
            assert comps[key]["mae_nT"] == pytest.approx(0.0, abs=1e-6)
            assert comps[key]["r2"] == pytest.approx(1.0, abs=1e-6)
            assert comps[key]["pearson_r"] == pytest.approx(1.0, abs=1e-6)

    # Known vector offsets: (+3, -4, 0) on EU_mid (idx 0), (+10, 0, 0) on CA_aur (idx 3)
    y_pred = y_true.copy()
    y_pred[:, 0, 0] += 3.0
    y_pred[:, 0, 1] -= 4.0
    y_pred[:, 3, 0] += 10.0
    res = compute_reconstruction_metrics(y_true, y_pred, eval_mask, belts)
    assert res["europe_mid"]["X"]["rmse_nT"] == pytest.approx(3.0, abs=1e-5)
    assert res["europe_mid"]["Y"]["rmse_nT"] == pytest.approx(4.0, abs=1e-5)
    assert res["europe_mid"]["Z"]["rmse_nT"] == pytest.approx(0.0, abs=1e-6)
    assert res["europe_mid"]["vector_3d"]["rmse_nT"] == pytest.approx(5.0, abs=1e-5)
    assert res["europe_mid"]["vector_3d"]["mae_nT"] == pytest.approx(5.0, abs=1e-5)
    # Auroral combines idx 1 (err=0) and idx 3 (err=10 in X): sqrt(10^2 / 2) = sqrt(50)
    assert res["auroral"]["X"]["rmse_nT"] == pytest.approx(np.sqrt(50.0), abs=1e-5)

    y_bad = y_pred.copy()
    y_bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        compute_reconstruction_metrics(y_true, y_bad, eval_mask, belts)


def test_evaluator_in_train_barrier_and_zero_shot_separation(
    synthetic_dataset: tuple[dict[str, np.ndarray], dict],
) -> None:
    """Verify ~in_train stations never affect M1-M4 and heldout_zero_shot stays separate from SUA."""
    from geomag_ml.data.preprocessing import BaselineEstimator
    from geomag_ml.data.splits import SplitFold, build_station_train_mask
    from geomag_ml.eval.evaluator import evaluate_model_on_fold

    arrays, meta = synthetic_dataset
    codes = meta["station_codes"]
    sua_idx = codes.index("SUA")
    bel_idx = codes.index("BEL")
    lvv_idx = codes.index("LVV")
    mea_idx = codes.index("MEA")

    train_idx = np.arange(0, 300, dtype=np.int64)
    val_idx = np.arange(0, 300, dtype=np.int64)  # dummy for SplitFold construction
    test_idx = np.arange(300, 600, dtype=np.int64)

    st_train_mask, is_zs_mask = build_station_train_mask(codes, ("BEL", "LVV", "MEA"))
    est = BaselineEstimator(mode="temp_train_quiet_proxy", causal_init_minutes=120).fit(
        arrays, train_idx, station_train_mask=st_train_mask
    )
    in_train = est.in_train_stations_.copy()
    assert not in_train[sua_idx]
    assert not in_train[bel_idx]

    fold = SplitFold(
        fold_id=3,
        protocol="synthetic_test",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_episodes=(0,),
        val_episodes=(0,),
        test_episodes=(1,),
        station_train_mask=st_train_mask,
        is_zero_shot_mask=is_zs_mask,
        in_train_mask=in_train,
    )

    class _MockZSModel:
        branch = "B"
        supports_zero_shot = True

        def __init__(self, poison_sua_pred: bool) -> None:
            self.poison_sua_pred = poison_sua_pred
            self.seen_obs_non_train = 0

        def predict_masked(
            self,
            X_work: np.ndarray,
            obs_mask: np.ndarray,
            target_mask: np.ndarray,
            **_: object,
        ) -> np.ndarray:
            self.seen_obs_non_train += int(np.sum(obs_mask & (~in_train[None, :])))
            # Simple spatial average of observed stations
            cnt = np.maximum(np.sum(obs_mask, axis=1, keepdims=True), 1)
            avg = np.sum(np.where(obs_mask[..., None], X_work, 0.0), axis=1, keepdims=True) / cnt[..., None]
            out = np.broadcast_to(avg, X_work.shape).copy().astype(np.float32)
            if self.poison_sua_pred and not np.any(target_mask[:, sua_idx]):
                # In standard M1-M4, SUA is not in target_mask; NaN here must not leak into metrics!
                out[:, sua_idx, :] = np.nan
            return out

    class _MockFixedModel:
        branch = "B"
        supports_zero_shot = False

        def predict_masked(self, X_work: np.ndarray, obs_mask: np.ndarray, target_mask: np.ndarray, **_: object) -> np.ndarray:
            _ = obs_mask, target_mask
            return np.zeros_like(X_work, dtype=np.float32)

    gen = MaskGenerator(arrays["station_coords"], arrays["station_dist_km"], seed=42)

    # Perturbing SUA and Zero-Shot station inputs in test_idx must not change any M1-M4 metric
    perturbed = {k: v.copy() for k, v in arrays.items()}
    perturbed["X_raw"][test_idx, sua_idx, :] += 1e6
    perturbed["X_raw"][test_idx, bel_idx, :] += 1e6

    m_clean, _ = evaluate_model_on_fold(
        _MockZSModel(poison_sua_pred=False),
        arrays,
        meta,
        fold,
        est,
        None,
        gen,
        eval_modes=("M1", "M2", "M4"),
        m2_repeats=2,
        min_obs=4,
    )
    mock_poison = _MockZSModel(poison_sua_pred=True)
    m_pert, _ = evaluate_model_on_fold(
        mock_poison,
        perturbed,
        meta,
        fold,
        est,
        None,
        gen,
        eval_modes=("M1", "M2", "M4"),
        m2_repeats=2,
        min_obs=4,
    )
    assert mock_poison.seen_obs_non_train == 0
    for mode in ("M1", "M2", "M4"):
        assert m_clean["eval_masks"][mode]["global"]["vector_3d"]["rmse_nT"] == pytest.approx(
            m_pert["eval_masks"][mode]["global"]["vector_3d"]["rmse_nT"], abs=1e-6
        )

    # Zero-shot bucket separates BEL/LVV/MEA from single-episode SUA
    zs_bucket = m_clean["eval_masks"]["zero_shot"]
    n_held_expected = int(np.sum(arrays["M_qc_clean"][test_idx][:, [bel_idx, lvv_idx, mea_idx]]))
    n_sua_expected = int(np.sum(arrays["M_qc_clean"][test_idx][:, sua_idx]))
    assert zs_bucket["heldout_zero_shot"]["global"]["X"]["n_points"] == n_held_expected
    assert zs_bucket["single_episode_unseen"]["global"]["X"]["n_points"] == n_sua_expected
    assert zs_bucket["single_episode_unseen"]["baseline_note"] == "causal_initial_6h"
    assert set(zs_bucket["per_station"].keys()) == {"BEL", "LVV", "MEA", "SUA"}
    assert zs_bucket["per_station"]["SUA"]["is_zero_shot"] is False
    assert zs_bucket["per_station"]["BEL"]["is_zero_shot"] is True

    m_fixed, _ = evaluate_model_on_fold(
        _MockFixedModel(),
        arrays,
        meta,
        fold,
        est,
        None,
        gen,
        eval_modes=("M1",),
        min_obs=4,
    )
    assert m_fixed["eval_masks"]["zero_shot"] == {"status": "not_supported_fixed_stations"}


def test_latent_stability_and_space_weather_alignment_nan_safe(
    synthetic_dataset: tuple[dict[str, np.ndarray], dict],
) -> None:
    """Verify cos_sim=1 invariant and NaN-safe OMNI status codes."""
    from geomag_ml.eval.metrics import (
        compute_latent_stability,
        compute_space_weather_alignment,
    )

    arrays, _ = synthetic_dataset
    t_len = 600
    phase = np.linspace(0.0, 4.0 * np.pi, t_len)
    z = np.column_stack([np.sin(phase), np.cos(phase), np.sin(2.0 * phase)])

    stab = compute_latent_stability(z, [z.copy(), z.copy()])
    assert stab["cos_sim_mean"] == pytest.approx(1.0, abs=1e-7)
    assert stab["rel_l2_drift"] == pytest.approx(0.0, abs=1e-7)
    assert stab["axis_corr_mean"] == pytest.approx(1.0, abs=1e-7)

    non_test_idx = np.arange(0, 300, dtype=np.int64)
    test_idx = np.arange(300, 600, dtype=np.int64)
    sw_res = compute_space_weather_alignment(
        z[non_test_idx],
        z[test_idx],
        arrays["sw_omni"],
        arrays["sw_solar1_bsn"],
        arrays["oulu_delta_n_pct"],
        non_test_idx,
        test_idx,
    )
    assert sw_res["omni"]["sym_h"]["status"] == "unavailable_all_nan"
    assert sw_res["omni"]["ae"]["status"] == "unavailable_in_test_fold"
    assert sw_res["solar1_bsn"]["bz_gsm"]["status"] == "ok"
    assert sw_res["oulu"]["delta_n_pct"]["status"] == "ok"
    assert sw_res["oulu"]["delta_n_pct"]["linear_r2"] > 0.90


def test_space_weather_alignment_rejects_overlapping_indices(
    synthetic_dataset: tuple[dict[str, np.ndarray], dict],
) -> None:
    """Regression test for ML-02: compute_space_weather_alignment must reject overlapping splits."""
    from geomag_ml.eval.metrics import compute_space_weather_alignment

    arrays, _ = synthetic_dataset
    z_all = np.column_stack(
        [np.sin(np.linspace(0, 4 * np.pi, 600)), np.cos(np.linspace(0, 4 * np.pi, 600))]
    )
    nt_idx = np.arange(0, 350, dtype=np.int64)
    te_idx = np.arange(300, 600, dtype=np.int64)

    with pytest.raises(ValueError, match="strictly disjoint"):
        compute_space_weather_alignment(
            z_all[nt_idx],
            z_all[te_idx],
            arrays["sw_omni"],
            arrays["sw_solar1_bsn"],
            arrays["oulu_delta_n_pct"],
            nt_idx,
            te_idx,
        )


def test_evaluator_branch_a_eval_split_val_no_leakage(
    synthetic_dataset: tuple[dict[str, np.ndarray], dict],
) -> None:
    """Regression test for ML-02: eval_split='val' must train space-weather alignment on train_idx only."""
    from geomag_ml.data.preprocessing import BaselineEstimator, FieldScaler
    from geomag_ml.data.splits import SplitFold
    from geomag_ml.eval.evaluator import evaluate_model_on_fold, sanitize_masked_input
    from geomag_ml.eval.metrics import compute_space_weather_alignment
    from geomag_ml.models.baselines import EMPCABaseline

    arrays, meta = synthetic_dataset
    n_st = len(meta["station_codes"])
    sua_idx = meta["station_codes"].index("SUA")

    # 3-way disjoint split: train=[0..200), val=[200..400), test=[400..600)
    train_idx = np.arange(0, 200, dtype=np.int64)
    val_idx = np.arange(200, 400, dtype=np.int64)
    test_idx = np.arange(400, 600, dtype=np.int64)

    st_train_mask = np.ones(n_st, dtype=bool)
    est = BaselineEstimator(mode="temp_train_quiet_proxy", causal_init_minutes=60).fit(
        arrays, train_idx, station_train_mask=st_train_mask
    )
    x_delta = est.transform(
        arrays["X_raw"],
        arrays["M_valid"],
        arrays["episode_idx"],
        timestamps_unix_s=arrays["timestamps_unix_s"],
        station_has_baseline=arrays["station_has_baseline"],
    )
    sc = FieldScaler(mode="zscore_per_station").fit(
        x_delta, arrays["M_qc_clean"], train_idx, arrays["station_belt_id"]
    )
    x_work = sc.transform(x_delta)
    in_train = est.in_train_stations_ & sc.in_train_stations_
    assert not in_train[sua_idx]

    fold = SplitFold(
        fold_id=0,
        protocol="synthetic_3way",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_episodes=(0,),
        val_episodes=(0, 1),
        test_episodes=(1,),
        station_train_mask=st_train_mask,
        is_zero_shot_mask=np.zeros(n_st, dtype=bool),
        in_train_mask=in_train,
    )
    model = EMPCABaseline(k=3, use_sq_detrend=False).fit(
        x_work, arrays["M_qc_clean"], train_idx, in_train
    )
    gen = MaskGenerator(arrays["station_coords"], arrays["station_dist_km"], seed=42)

    metrics, outputs = evaluate_model_on_fold(
        model,
        arrays,
        meta,
        fold,
        est,
        sc,
        gen,
        eval_split="val",
        eval_modes=("M1",),
        stability_drop_fracs=(0.20,),
        stability_repeats=1,
        min_obs=4,
        return_outputs=True,
    )
    assert "z_val" in outputs
    assert "z_test" not in outputs

    oulu_res = metrics["space_weather_alignment"]["oulu"]["delta_n_pct"]
    assert oulu_res["status"] == "ok"
    assert oulu_res["n_train"] == len(train_idx)
    assert oulu_res["n_test"] == len(val_idx)

    # Confirm exact parity with manual train_idx -> val_idx alignment (zero val contamination)
    obs_tr = arrays["M_qc_clean"][train_idx] & in_train[None, :]
    obs_va = arrays["M_qc_clean"][val_idx] & in_train[None, :]
    z_tr = model.encode_masked(sanitize_masked_input(x_work[train_idx], obs_tr), obs_tr)
    z_va = model.encode_masked(sanitize_masked_input(x_work[val_idx], obs_va), obs_va)
    manual = compute_space_weather_alignment(
        z_tr,
        z_va,
        arrays["sw_omni"],
        arrays["sw_solar1_bsn"],
        arrays["oulu_delta_n_pct"],
        train_idx,
        val_idx,
    )
    assert oulu_res["linear_r2"] == pytest.approx(
        manual["oulu"]["delta_n_pct"]["linear_r2"], abs=1e-7
    )


