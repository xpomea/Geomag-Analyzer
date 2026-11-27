"""Phase 0 (T0.4) tests: leakage-free BaselineEstimator and dual FieldScaler."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geomag_ml.data.preprocessing import (
    BaselineEstimator,
    FieldScaler,
    load_scaler_json,
    save_scaler_json,
)


def _deepcopy_dataset(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: v.copy() for k, v in arrays.items()}


@pytest.mark.parametrize("baseline_mode", ["temp_train_quiet_proxy", "target_kp_quiet_night"])
@pytest.mark.parametrize("scaler_mode", ["asinh_global", "zscore_per_station"])
def test_baseline_and_scaler_zero_test_leakage(
    synthetic_dataset: tuple[dict[str, np.ndarray], dict],
    baseline_mode: str,
    scaler_mode: str,
) -> None:
    arrays, _ = synthetic_dataset
    # Episode 0 (0..300) is test, Episode 1 (300..600) is train
    test_idx = np.arange(0, 300, dtype=np.int64)
    train_idx = np.arange(300, 600, dtype=np.int64)

    perturbed = _deepcopy_dataset(arrays)
    m_test = perturbed["M_valid"][test_idx]
    perturbed["X_raw"][test_idx] = np.where(
        m_test[..., None],
        perturbed["X_raw"][test_idx] + np.float32(5000.0),
        np.nan,
    )

    est_orig = BaselineEstimator(mode=baseline_mode).fit(arrays, train_idx)
    est_pert = BaselineEstimator(mode=baseline_mode).fit(perturbed, train_idx)
    assert np.allclose(est_orig.B0_nT_, est_pert.B0_nT_, atol=1e-7, equal_nan=True)

    delta_orig = est_orig.transform(
        arrays["X_raw"],
        arrays["M_valid"],
        arrays["episode_idx"],
        timestamps_unix_s=arrays["timestamps_unix_s"],
        station_has_baseline=arrays["station_has_baseline"],
    )
    delta_pert = est_pert.transform(
        perturbed["X_raw"],
        perturbed["M_valid"],
        perturbed["episode_idx"],
        timestamps_unix_s=perturbed["timestamps_unix_s"],
        station_has_baseline=perturbed["station_has_baseline"],
    )

    sc_orig = FieldScaler(mode=scaler_mode).fit(
        delta_orig, arrays["M_qc_clean"], train_idx, arrays["station_belt_id"]
    )
    sc_pert = FieldScaler(mode=scaler_mode).fit(
        delta_pert, perturbed["M_qc_clean"], train_idx, perturbed["station_belt_id"]
    )
    if scaler_mode == "asinh_global":
        assert np.allclose(sc_orig.scale_s_, sc_pert.scale_s_, atol=1e-7)
    else:
        assert np.allclose(sc_orig.mean_, sc_pert.mean_, atol=1e-7)
        assert np.allclose(sc_orig.std_, sc_pert.std_, atol=1e-7)


def test_baseline_zero_shot_station_excluded_from_vnet_and_episode_boundary(
    synthetic_dataset: tuple[dict[str, np.ndarray], dict],
) -> None:
    arrays, meta = synthetic_dataset
    codes = meta["station_codes"]
    zs_mask = np.array([c not in ("BEL", "LVV", "MEA") for c in codes], dtype=bool)
    train_idx = np.arange(0, 600, dtype=np.int64)

    est1 = BaselineEstimator(mode="temp_train_quiet_proxy").fit(
        arrays, train_idx, station_train_mask=zs_mask
    )

    # Perturbing a zero-shot station inside train_idx must not change v_net or B0 of in_train stations
    mod = _deepcopy_dataset(arrays)
    bel_i = codes.index("BEL")
    mod["X_raw"][:, bel_i, :] += np.linspace(-1000.0, 1000.0, 600, dtype=np.float32)[:, None]
    est2 = BaselineEstimator(mode="temp_train_quiet_proxy").fit(
        mod, train_idx, station_train_mask=zs_mask
    )
    assert np.allclose(est1.B0_nT_[zs_mask], est2.B0_nT_[zs_mask], atol=1e-7, equal_nan=True)
    assert not est1.in_train_stations_[bel_i]
    # Static observatory constant B0 for zero-shot BEL is still fitted from train_idx (no test touch)
    assert np.all(np.isfinite(est1.B0_nT_[bel_i]))


@pytest.mark.parametrize("scaler_mode", ["asinh_global", "zscore_per_station"])
def test_scaler_invertibility_and_json_roundtrip(
    synthetic_dataset: tuple[dict[str, np.ndarray], dict],
    tmp_path: Path,
    scaler_mode: str,
) -> None:
    arrays, _ = synthetic_dataset
    train_idx = np.arange(300, 600, dtype=np.int64)
    est = BaselineEstimator(mode="temp_train_quiet_proxy").fit(arrays, train_idx)
    delta = est.transform(
        arrays["X_raw"],
        arrays["M_valid"],
        arrays["episode_idx"],
        timestamps_unix_s=arrays["timestamps_unix_s"],
        station_has_baseline=arrays["station_has_baseline"],
    )

    sc = FieldScaler(mode=scaler_mode).fit(
        delta, arrays["M_qc_clean"], train_idx, arrays["station_belt_id"]
    )
    scaled = sc.transform(delta)
    recovered = sc.inverse_transform(scaled)

    m_valid = arrays["M_valid"]
    max_err = float(np.max(np.abs(recovered[m_valid] - delta[m_valid])))
    assert max_err < 1e-3  # float32 roundtrip on ~50-300 nT
    # Tight check on well-conditioned bounded field
    small = m_valid & (np.max(np.abs(delta), axis=-1) < 200.0)
    assert float(np.max(np.abs(recovered[small] - delta[small]))) < 1e-4
    assert np.all(np.isnan(scaled[~m_valid]))

    # Exact bit-for-bit persistence roundtrip via scaler.json
    scaler_json = tmp_path / "scaler.json"
    save_scaler_json(scaler_json, est, sc)
    est_loaded, sc_loaded = load_scaler_json(scaler_json)
    scaled_loaded = sc_loaded.transform(delta)
    assert np.array_equal(scaled, scaled_loaded, equal_nan=True)
    assert est_loaded.fit_split_hash_ == est.fit_split_hash_


def test_baseline_and_scaler_zero_train_station_sua_case(
    synthetic_dataset: tuple[dict[str, np.ndarray], dict],
) -> None:
    arrays, meta = synthetic_dataset
    codes = meta["station_codes"]
    sua_idx = codes.index("SUA")

    # Train strictly on episode 0 (where SUA has 0 valid samples), test on episode 1 (SUA present)
    train_idx = np.arange(0, 300, dtype=np.int64)
    est = BaselineEstimator(mode="temp_train_quiet_proxy", causal_init_minutes=120).fit(
        arrays, train_idx
    )
    assert not est.in_train_stations_[sua_idx]
    assert not est.has_train_baseline_[sua_idx]
    assert np.all(np.isnan(est.B0_nT_[sua_idx]))

    delta = est.transform(
        arrays["X_raw"],
        arrays["M_valid"],
        arrays["episode_idx"],
        timestamps_unix_s=arrays["timestamps_unix_s"],
        station_has_baseline=arrays["station_has_baseline"],
    )
    # Episode 1 SUA receives causal initial baseline and stays completely finite
    assert np.all(np.isfinite(delta[300:, sua_idx, :]))

    sc = FieldScaler(mode="zscore_per_station").fit(
        delta, arrays["M_qc_clean"], train_idx, arrays["station_belt_id"]
    )
    assert not sc.in_train_stations_[sua_idx]
    assert np.all(np.isfinite(sc.mean_[sua_idx]))
    assert np.all(np.isfinite(sc.std_[sua_idx]))
    assert np.all(sc.std_[sua_idx] >= sc.min_std_nT)

    scaled = sc.transform(delta)
    assert np.all(np.isfinite(scaled[300:, sua_idx, :]))


def test_baseline_policies_baseless_and_variation_step(
    synthetic_dataset: tuple[dict[str, np.ndarray], dict],
) -> None:
    arrays, meta = synthetic_dataset
    codes = meta["station_codes"]
    gck_idx = codes.index("GCK")
    clf_idx = codes.index("CLF")
    train_idx = np.arange(0, 300, dtype=np.int64)

    # 1. baseless_policy == "exclude" masks out GCK
    est_excl = BaselineEstimator(baseless_policy="exclude").fit(arrays, train_idx)
    assert not est_excl.in_train_stations_[gck_idx]
    delta_excl = est_excl.transform(
        arrays["X_raw"],
        arrays["M_valid"],
        arrays["episode_idx"],
        station_has_baseline=arrays["station_has_baseline"],
    )
    assert np.all(np.isnan(delta_excl[:, gck_idx, :]))

    # 2. variation_step_policy == "causal_initial_6h_reanchor" re-anchors a >30 nT step on CLF in ep1
    mod = _deepcopy_dataset(arrays)
    mod["X_raw"][300:, clf_idx, 1] += 80.0
    est_re = BaselineEstimator(variation_step_policy="causal_initial_6h_reanchor").fit(
        mod, train_idx
    )
    delta_re = est_re.transform(
        mod["X_raw"],
        mod["M_valid"],
        mod["episode_idx"],
        timestamps_unix_s=mod["timestamps_unix_s"],
        station_has_baseline=mod["station_has_baseline"],
    )
    # Re-anchoring cancels the +80 nT DC jump on ep1
    assert abs(float(np.median(delta_re[300:, clf_idx, 1]))) < 30.0

    # 3. per_episode_median_ablation removes storm shift per episode
    est_ep = BaselineEstimator(mode="per_episode_median_ablation").fit(arrays, train_idx)
    delta_ep = est_ep.transform(
        arrays["X_raw"],
        arrays["M_valid"],
        arrays["episode_idx"],
        station_has_baseline=arrays["station_has_baseline"],
    )
    assert abs(float(np.median(delta_ep[300:, clf_idx, 0]))) < 1e-3

    # 4. target_kp_quiet_night secular trend fit when quiet span >= 60 days
    long_ds = _deepcopy_dataset(arrays)
    long_ds["timestamps_unix_s"][300:] += 70 * 86400
    long_ds["event_flags"][:] = 1  # EVENT_FLAG_QUIET
    est_sec = BaselineEstimator(mode="target_kp_quiet_night", use_secular_trend=True).fit(
        long_ds, np.arange(0, 600, dtype=np.int64)
    )
    assert np.any(est_sec.secular_slope_nT_per_day_[clf_idx] != 0.0)
    delta_sec = est_sec.transform(
        long_ds["X_raw"],
        long_ds["M_valid"],
        long_ds["episode_idx"],
        timestamps_unix_s=long_ds["timestamps_unix_s"],
        station_has_baseline=long_ds["station_has_baseline"],
    )
    assert np.all(np.isfinite(delta_sec[:, clf_idx, :]))


def test_baseline_secular_trend_does_not_trigger_false_reanchor(
    synthetic_dataset: tuple[dict[str, np.ndarray], dict],
) -> None:
    """Regression test for ML-05(1): secular trend > 30 nT must not trigger false 6h re-anchor."""
    arrays, meta = synthetic_dataset
    clf_idx = meta["station_codes"].index("CLF")

    # Build 3 quiet/storm episodes spanning 400 days with +0.2 nT/day secular drift on CLF Y
    def _clone_ep(start_slice: slice, day_offset: int, ep_id: int) -> dict[str, np.ndarray]:
        out = {}
        for k in ("X_raw", "M_valid", "M_qc_clean", "mlt_hours", "sw_omni", "event_flags"):
            out[k] = arrays[k][start_slice].copy()
        out["timestamps_unix_s"] = arrays["timestamps_unix_s"][start_slice].copy() + day_offset * 86400
        out["episode_idx"] = np.full(300, ep_id, dtype=np.int16)
        return out

    ep0 = _clone_ep(slice(0, 300), 0, 0)
    ep1 = _clone_ep(slice(0, 300), 100, 1)
    ep2 = _clone_ep(slice(0, 300), 400, 2)

    ep0["event_flags"][:] = 1
    ep1["event_flags"][:] = 1
    ep2["event_flags"][:] = 2

    # Zero out random storm variations on CLF to isolate secular drift + Dst cleanly
    base_val = np.float32(1500.0)
    ep0["X_raw"][:, clf_idx, :] = base_val
    ep1["X_raw"][:, clf_idx, :] = base_val
    ep2["X_raw"][:, clf_idx, :] = base_val

    # Apply +0.2 nT/day secular drift across all episodes (+20 nT at ep1, +80 nT at ep2)
    ep1["X_raw"][:, clf_idx, 1] += np.float32(20.0)
    ep2["X_raw"][:, clf_idx, 1] += np.float32(80.0)
    # Apply physical -25 nT storm Dst shift in test episode ep2 (no instrumental jump)
    ep2["X_raw"][:, clf_idx, 1] -= np.float32(25.0)

    long_ds = {
        k: np.concatenate([ep0[k], ep1[k], ep2[k]], axis=0)
        for k in ep0
    }
    long_ds["station_has_baseline"] = arrays["station_has_baseline"].copy()

    train_idx = np.arange(0, 600, dtype=np.int64)
    est = BaselineEstimator(
        mode="target_kp_quiet_night",
        variation_step_policy="causal_initial_6h_reanchor",
        use_secular_trend=True,
    ).fit(long_ds, train_idx)

    assert np.isclose(est.secular_slope_nT_per_day_[clf_idx, 1], 0.2, atol=1e-3)
    delta = est.transform(
        long_ds["X_raw"],
        long_ds["M_valid"],
        long_ds["episode_idx"],
        timestamps_unix_s=long_ds["timestamps_unix_s"],
        station_has_baseline=long_ds["station_has_baseline"],
    )
    # Must preserve the -25 nT storm shift instead of falsely re-anchoring to 0 nT
    assert np.isclose(float(np.median(delta[600:900, clf_idx, 1])), -25.0, atol=0.5)

    # Now inject an actual +80 nT instrumental step on ep2: re-anchoring must cancel it
    long_ds_step = _deepcopy_dataset(long_ds)
    long_ds_step["X_raw"][600:900, clf_idx, 1] += np.float32(80.0)
    delta_step = est.transform(
        long_ds_step["X_raw"],
        long_ds_step["M_valid"],
        long_ds_step["episode_idx"],
        timestamps_unix_s=long_ds_step["timestamps_unix_s"],
        station_has_baseline=long_ds_step["station_has_baseline"],
    )
    assert abs(float(np.median(delta_step[600:900, clf_idx, 1]))) < 1.0


def test_scaler_min_train_points_alignment_and_save_scaler_json(
    synthetic_dataset: tuple[dict[str, np.ndarray], dict],
    tmp_path: Path,
) -> None:
    """Regression test for ML-05(2): FieldScaler matches min_train_points=60 and save_scaler_json agrees."""
    arrays, meta = synthetic_dataset
    sua_idx = meta["station_codes"].index("SUA")

    # Give SUA only 10 clean points in ep1 (< 60 threshold)
    mod = _deepcopy_dataset(arrays)
    mod["M_qc_clean"][310:600, sua_idx] = False
    train_idx = np.arange(300, 600, dtype=np.int64)

    est = BaselineEstimator(mode="temp_train_quiet_proxy").fit(mod, train_idx)
    delta = est.transform(
        mod["X_raw"],
        mod["M_valid"],
        mod["episode_idx"],
        timestamps_unix_s=mod["timestamps_unix_s"],
        station_has_baseline=mod["station_has_baseline"],
    )
    sc = FieldScaler(mode="zscore_per_station").fit(
        delta, mod["M_qc_clean"], train_idx, mod["station_belt_id"]
    )

    assert not est.in_train_stations_[sua_idx]
    assert not sc.in_train_stations_[sua_idx]

    # Even if a scaler was created with a smaller min_train_points, save_scaler_json intersects flags
    sc_loose = FieldScaler(mode="zscore_per_station", min_train_points=2).fit(
        delta, mod["M_qc_clean"], train_idx, mod["station_belt_id"]
    )
    out_json = tmp_path / "scaler.json"
    save_scaler_json(out_json, est, sc_loose)
    from geomag_ml.utils.io import read_json

    saved = read_json(out_json)
    assert saved["B0_nT"][sua_idx] == [None, None, None]
    assert saved["in_train_stations"][sua_idx] is False
