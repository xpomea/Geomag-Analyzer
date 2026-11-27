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


# ---------------------------------------------------------------------------
# Phase 1 (T1.1): Splits & Window Boundary Guard tests
# ---------------------------------------------------------------------------

def test_looe_4fold_oct2026_real_qc_and_omni_invariants(ml_root: Path) -> None:
    """Verify all T1.1 QC, OMNI, gap, and night-MLT invariants on the real Oct-2026 dataset."""
    npz_path = ml_root / "data_cache" / "geomag_ml_oct2026_clean_v1.npz"
    if not npz_path.is_file():
        pytest.skip("Oct-2026 clean dataset not present in this environment")

    from geomag_ml.data.schema import OMNI_COLS, load_ml_dataset
    from geomag_ml.data.splits import (
        build_looe_4fold_splits,
        compute_min_temporal_gap_sec,
    )

    arrays, meta = load_ml_dataset(npz_path, validate=True)
    codes = meta["station_codes"]
    sua_idx = codes.index("SUA")
    ae_col = OMNI_COLS.index("ae")

    folds = build_looe_4fold_splits(arrays, meta)
    assert len(folds) == 4

    ts = arrays["timestamps_unix_s"]
    m_valid = arrays["M_valid"]
    m_qc = arrays["M_qc_clean"]
    mlt = arrays["mlt_hours"]
    is_night = (mlt >= 22.0) | (mlt <= 4.0)

    for fold in folds:
        assert compute_min_temporal_gap_sec(ts, fold.train_idx, fold.test_idx) >= 86400
        assert compute_min_temporal_gap_sec(ts, fold.train_idx, fold.val_idx) >= 86400
        assert compute_min_temporal_gap_sec(ts, fold.val_idx, fold.test_idx) >= 86400

        omni_train_valid = int(np.sum(np.isfinite(arrays["sw_omni"][fold.train_idx, ae_col])))
        assert omni_train_valid > 0

        tr_valid_cnt = np.sum(m_valid[fold.train_idx], axis=0)
        tr_qc_cnt = np.sum(m_qc[fold.train_idx], axis=0)
        te_valid_cnt = np.sum(m_valid[fold.test_idx], axis=0)
        te_qc_cnt = np.sum(m_qc[fold.test_idx], axis=0)

        zero_tr_valid = [i for i in range(len(codes)) if te_valid_cnt[i] > 0 and tr_valid_cnt[i] == 0]
        zero_tr_qc = [i for i in range(len(codes)) if te_qc_cnt[i] > 0 and tr_qc_cnt[i] == 0]

        if fold.fold_id == 3:
            assert zero_tr_valid == [sua_idx]
            assert zero_tr_qc == [sua_idx]
        else:
            assert zero_tr_valid == []
            assert zero_tr_qc == []

        for i in range(len(codes)):
            if te_qc_cnt[i] > 0 and i not in zero_tr_qc:
                assert tr_qc_cnt[i] >= 60
                night_qc = int(np.sum(m_qc[fold.train_idx, i] & is_night[fold.train_idx, i]))
                assert night_qc >= 60


def test_looe_4fold_zero_shot_station_mask_and_random_forbidden(
    synthetic_dataset: tuple[dict[str, np.ndarray], dict],
) -> None:
    """Verify Zero-Shot station masking and rejection of random minute-level splitting."""
    from geomag_ml.data.splits import DEFAULT_ZERO_SHOT_STATIONS, create_splits

    arrays, meta = synthetic_dataset
    with pytest.raises(ValueError, match="forbidden"):
        create_splits("random", arrays, meta)

    # Expand synthetic 2-episode dataset to 4 episodes separated by 3 days each.
    ep0 = slice(0, 300)
    ep_list = []
    for k in range(4):
        shift = k * 3 * 86400
        d = {
            "X_raw": arrays["X_raw"][ep0].copy(),
            "M_valid": arrays["M_valid"][ep0].copy(),
            "M_qc_clean": arrays["M_qc_clean"][ep0].copy(),
            "timestamps_unix_s": arrays["timestamps_unix_s"][ep0].copy() + shift,
            "episode_idx": np.full(300, k, dtype=np.int16),
            "event_flags": arrays["event_flags"][ep0].copy(),
        }
        ep_list.append(d)
    ds4 = {
        k: np.concatenate([ep[k] for ep in ep_list], axis=0)
        for k in ep_list[0]
    }
    ds4["episode_bounds"] = np.array([[0, 300], [300, 600], [600, 900], [900, 1200]], dtype=np.int64)
    ds4["station_has_baseline"] = arrays["station_has_baseline"].copy()

    folds = create_splits("looe_4fold", ds4, meta, zero_shot_stations=DEFAULT_ZERO_SHOT_STATIONS)
    codes = meta["station_codes"]
    zs_expected = [codes.index(c) for c in DEFAULT_ZERO_SHOT_STATIONS]
    for f in folds:
        for idx in zs_expected:
            assert f.is_zero_shot_mask[idx]
            assert not f.station_train_mask[idx]
            assert not f.in_train_mask[idx]


def test_block_temporal_purge_gap_and_eclipse_spe_enforcement(
    synthetic_dataset: tuple[dict[str, np.ndarray], dict],
) -> None:
    """Verify >= 1 day purge gap, forced eclipse/SPE placement in test, and violation AssertionError."""
    from geomag_ml.data.schema import EVENT_FLAG_ECLIPSE, EVENT_FLAG_QUIET, EVENT_FLAG_SPE
    from geomag_ml.data.splits import (
        assert_no_eclipse_spe_in_train_or_val,
        compute_min_temporal_gap_sec,
        create_splits,
    )

    arrays, meta = synthetic_dataset
    # Construct continuous 30-day dataset (43,200 min) so 1-day purge buffers around relocated events leave all splits non-empty
    n_days = 30
    n_t = n_days * 1440
    t0 = int(arrays["timestamps_unix_s"][0])
    ts = t0 + np.arange(n_t, dtype=np.int64) * 60
    n_st = len(meta["station_codes"])

    ev = np.full(n_t, EVENT_FLAG_QUIET, dtype=np.uint8)
    # Inject eclipse during day 2 (inside initial train range) and SPE during day 20 (inside initial val range)
    ev[2 * 1440 : 2 * 1440 + 180] = EVENT_FLAG_ECLIPSE
    ev[20 * 1440 : 20 * 1440 + 120] = EVENT_FLAG_SPE

    ds = {
        "X_raw": np.ones((n_t, n_st, 3), dtype=np.float32),
        "M_valid": np.ones((n_t, n_st), dtype=bool),
        "M_qc_clean": np.ones((n_t, n_st), dtype=bool),
        "timestamps_unix_s": ts,
        "episode_idx": np.zeros(n_t, dtype=np.int16),
        "episode_bounds": np.array([[0, n_t]], dtype=np.int64),
        "station_has_baseline": arrays["station_has_baseline"].copy(),
        "event_flags": ev,
    }

    with pytest.raises(ValueError, match="purge_gap_min"):
        create_splits("block_temporal", ds, meta, purge_gap_min=720)

    folds = create_splits("block_temporal", ds, meta, purge_gap_min=1440)
    fold = folds[0]

    assert compute_min_temporal_gap_sec(ts, fold.train_idx, fold.test_idx) >= 86400
    assert compute_min_temporal_gap_sec(ts, fold.train_idx, fold.val_idx) >= 86400
    assert compute_min_temporal_gap_sec(ts, fold.val_idx, fold.test_idx) >= 86400

    special_idx = np.flatnonzero((ev & (EVENT_FLAG_ECLIPSE | EVENT_FLAG_SPE)) != 0)
    assert np.all(np.isin(special_idx, fold.test_idx))
    assert not np.any(np.isin(special_idx, fold.train_idx))
    assert not np.any(np.isin(special_idx, fold.val_idx))

    with pytest.raises(AssertionError, match="Eclipse/SPE"):
        assert_no_eclipse_spe_in_train_or_val(
            ev, np.concatenate([fold.train_idx, special_idx[:1]]), fold.val_idx
        )


def test_build_valid_window_anchors_causal_and_centered() -> None:
    """Verify exact window containment with zero episode, split, or timestamp-step crossings."""
    from geomag_ml.data.splits import build_valid_window_anchors

    # Episode 0: [0..100), Episode 1: [100..200) with a 120s jump at index 150
    t_ep0 = 1_700_000_000 + np.arange(100, dtype=np.int64) * 60
    t_ep1 = t_ep0[-1] + 86400 + np.arange(100, dtype=np.int64) * 60
    t_ep1[50:] += 60  # introduce 120s gap between index 149 and 150
    ts = np.concatenate([t_ep0, t_ep1])
    ep = np.concatenate([np.zeros(100, dtype=np.int16), np.ones(100, dtype=np.int16)])

    # Split leaves out [80..100) to also test split boundary inside ep0
    split_idx = np.concatenate([np.arange(0, 80, dtype=np.int64), np.arange(100, 200, dtype=np.int64)])

    w = 16
    anchors_causal = build_valid_window_anchors(ts, ep, split_idx, w, mode="causal")
    expected_causal = np.concatenate(
        [
            np.arange(w - 1, 80, dtype=np.int64),
            np.arange(100 + w - 1, 150, dtype=np.int64),
            np.arange(150 + w - 1, 200, dtype=np.int64),
        ]
    )
    assert np.array_equal(anchors_causal, expected_causal)
    for t in anchors_causal:
        win = np.arange(t - w + 1, t + 1)
        assert np.all(np.isin(win, split_idx))
        assert np.all(ep[win] == ep[t])
        assert np.all(np.diff(ts[win]) == 60)

    anchors_centered = build_valid_window_anchors(ts, ep, split_idx, w, mode="centered")
    left, right = w // 2, w // 2 - 1
    expected_centered = np.concatenate(
        [
            np.arange(left, 80 - right, dtype=np.int64),
            np.arange(100 + left, 150 - right, dtype=np.int64),
            np.arange(150 + left, 200 - right, dtype=np.int64),
        ]
    )
    assert np.array_equal(anchors_centered, expected_centered)
    for t in anchors_centered:
        win = np.arange(t - left, t + right + 1)
        assert np.all(np.isin(win, split_idx))
        assert np.all(ep[win] == ep[t])
        assert np.all(np.diff(ts[win]) == 60)


def test_geomag_ml_data_package_not_gitignored(repo_root: Path) -> None:
    """Regression test for ML-01: geomag_ml/data/*.py must not be shadowed by root `data/` ignore."""
    import shutil
    import subprocess

    if not (repo_root / ".git").exists() or shutil.which("git") is None:
        pytest.skip("Git repository metadata or git CLI unavailable")

    tracked_candidates = [
        "ml_research/geomag_ml/data/__init__.py",
        "ml_research/geomag_ml/data/schema.py",
        "ml_research/geomag_ml/data/build_dataset.py",
        "ml_research/geomag_ml/data/ingest_extended.py",
        "ml_research/geomag_ml/data/preprocessing.py",
        "ml_research/geomag_ml/data/splits.py",
        "ml_research/geomag_ml/data/masks.py",
    ]
    res = subprocess.run(
        ["git", "check-ignore", "-v", *tracked_candidates],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert res.returncode == 1, f"geomag_ml/data files are gitignored:\n{res.stdout}"
    assert res.stdout.strip() == ""

    cache_res = subprocess.run(
        ["git", "check-ignore", "-v", "ml_research/geomag_ml/data/__pycache__/dummy.pyc"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert cache_res.returncode == 0


