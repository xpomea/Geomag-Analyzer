"""Phase 1 (T1.4) tests: EM-PCA (+/- Sq), Ridge-Schur, IDW, Ordinary Kriging, Zero-Shot."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geomag_ml.data.preprocessing import FieldScaler
from geomag_ml.data.schema import compute_time_features
from geomag_ml.models.baselines import (
    EMPCABaseline,
    IDWBaseline,
    OrdinaryKrigingBaseline,
    RidgeSchurBaseline,
)


def test_empca_monotonic_rmse_centering_and_zero_leakage() -> None:
    """Verify EM-PCA rank-k error decrease, non-zero centering, and zero leakage on masked inputs."""
    rng = np.random.default_rng(101)
    t_len, n_st = 300, 10
    z_true = rng.normal(size=(t_len, 5))
    loadings = rng.normal(size=(5, n_st * 3))
    mu_true = rng.uniform(-80.0, 80.0, size=(n_st, 3))
    x_clean = (z_true @ loadings).reshape(t_len, n_st, 3) + mu_true[None, :, :]
    x_clean += rng.normal(0.0, 0.05, size=x_clean.shape)

    m_qc = np.ones((t_len, n_st), dtype=bool)
    m_qc[10:25, 0] = False
    m_qc[50:70, 4] = False
    x_obs = np.where(m_qc[..., None], x_clean, np.nan)

    train_idx = np.arange(0, 200, dtype=np.int64)
    test_idx = np.arange(200, 300, dtype=np.int64)
    in_train = np.ones(n_st, dtype=bool)

    rmses = []
    for k in (3, 4, 5):
        pca = EMPCABaseline(k=k, use_sq_detrend=False, n_em_iters=25).fit(
            x_obs, m_qc, train_idx, in_train
        )
        obs_test = m_qc[test_idx].copy()
        obs_test[:, 2] = False
        x_in = np.where(obs_test[..., None], np.nan_to_num(x_obs[test_idx], nan=0.0), 0.0)
        pred = pca.predict_masked(x_in, obs_test, ~obs_test)
        rmse = float(np.sqrt(np.mean((pred[:, 2, :] - x_clean[test_idx, 2, :]) ** 2)))
        rmses.append(rmse)

        # Masked station perturbation by +1e6 must not change z_hat by anything
        x_pert = x_in.copy()
        x_pert[:, 2, :] += 1e6
        z_a = pca.encode_masked(x_in, obs_test)
        z_b = pca.encode_masked(x_pert, obs_test)
        assert np.max(np.abs(z_a - z_b)) == 0.0

    assert rmses[0] > rmses[1] > rmses[2]


def test_empca_sq_detrend_incomplete_ut_no_explosion() -> None:
    """Adaptive M_h=1 + scaled L2 prevents +261 nT extrapolation when only 15/24 UT hours are observed."""
    rng = np.random.default_rng(202)
    # 2 days = 2880 min; SFS-like station observed ONLY during UT hours 0..14 (15/24 bins)
    n_t, n_st = 2880, 6
    ts = 1_791_072_000 + np.arange(n_t, dtype=np.int64) * 60
    tf = compute_time_features(ts)
    ut_h = (ts % 86400) / 3600.0

    x = rng.normal(0.0, 2.0, size=(n_t, n_st, 3))
    # Smooth diurnal variation ~ 20 nT across all stations
    x += (20.0 * np.sin(2.0 * np.pi * ut_h / 24.0))[:, None, None]

    m_qc = np.ones((n_t, n_st), dtype=bool)
    sfs_idx = 1
    m_qc[:, sfs_idx] = ut_h < 15.0  # 15/24 UT hour bins (< 20/24 threshold!)

    tr_idx = np.arange(0, n_t, dtype=np.int64)
    in_train = np.ones(n_st, dtype=bool)

    pca_sq = EMPCABaseline(k=3, use_sq_detrend=True).fit(
        x, m_qc, tr_idx, in_train, time_features=tf, timestamps_unix_s=ts
    )
    assert pca_sq.sq_orders_[sfs_idx] == 1
    assert pca_sq.sq_orders_[0] == 4

    sq_full = pca_sq._compute_sq_offset(n_t, tf, ts)
    assert sq_full is not None
    max_train = float(np.max(np.abs(x[m_qc[:, sfs_idx], sfs_idx, :])))
    max_sq_24h = float(np.max(np.abs(sq_full[:, sfs_idx, :])))
    assert max_sq_24h < 2.0 * max_train


def test_ridge_schur_zero_overlap_pairs_and_accuracy() -> None:
    """EM-Gaussian covariance succeeds with 0 joint-overlap station pairs and beats zero baseline."""
    rng = np.random.default_rng(303)
    n_t, n_st = 400, 6
    driver = rng.normal(0.0, 20.0, size=(n_t, 3))
    x = driver[:, None, :] * np.linspace(0.8, 1.3, n_st)[None, :, None] + rng.normal(
        0.0, 2.0, size=(n_t, n_st, 3)
    )

    m_qc = np.ones((n_t, n_st), dtype=bool)
    # Station 0 only present in [0..150), Station 1 only present in [150..300) -> 0 joint train minutes!
    m_qc[150:300, 0] = False
    m_qc[0:150, 1] = False

    train_idx = np.arange(0, 300, dtype=np.int64)
    test_idx = np.arange(300, 400, dtype=np.int64)
    in_train = np.ones(n_st, dtype=bool)

    ridge = RidgeSchurBaseline(alpha_grid=(0.1, 1.0), n_em_iters=15).fit(
        x, m_qc, train_idx, in_train
    )
    assert np.all(np.isfinite(ridge.mu_))
    assert np.all(np.isfinite(ridge.Sigma_))

    # Test M1 on station 0 and random M2
    obs = np.ones((100, n_st), dtype=bool)
    obs[:, 0] = False
    pred = ridge.predict_masked(np.where(obs[..., None], x[test_idx], 0.0), obs, ~obs)
    rmse_model = float(np.sqrt(np.mean((pred[:, 0, :] - x[test_idx, 0, :]) ** 2)))
    rmse_zero = float(np.sqrt(np.mean(x[test_idx, 0, :] ** 2)))
    assert rmse_model < 0.25 * rmse_zero

    # Verify precision-matrix M2 path matches explicit direct Schur complement on random row masks
    rng_m2 = np.random.default_rng(777)
    obs_m2 = rng_m2.random((40, n_st)) > 0.35
    obs_m2[:, 0] = True  # ensure >= 1 observed station and mix of unique patterns (> 40//4 = 10)
    obs_m2[np.arange(40), np.arange(40) % (n_st - 1) + 1] = False
    x_m2_in = np.where(obs_m2[..., None], x[test_idx[:40]], 0.0)
    pred_m2 = ridge.predict_masked(x_m2_in, obs_m2, ~obs_m2)

    for row in range(40):
        pat_d = np.repeat(obs_m2[row], 3)
        o_idx = np.flatnonzero(pat_d)
        u_idx = np.flatnonzero(~pat_d)
        if u_idx.size == 0:
            continue
        x_flat_r = x_m2_in[row].reshape(-1).astype(np.float64)
        s_oo = ridge.Sigma_[np.ix_(o_idx, o_idx)].copy()
        s_oo.flat[:: o_idx.size + 1] += ridge.best_alpha_
        s_uo = ridge.Sigma_[np.ix_(u_idx, o_idx)]
        direct_u = ridge.mu_[u_idx] + s_uo @ np.linalg.solve(s_oo, x_flat_r[o_idx] - ridge.mu_[o_idx])
        assert np.max(np.abs(pred_m2[row].reshape(-1)[u_idx] - direct_u)) < 1e-4


def test_idw_and_kriging_zero_shot_antimeridian_and_constant_field() -> None:
    """Verify IDW & Ordinary Kriging Zero-Shot prediction, +-180 wrap, constant field, and nugget floor."""
    # 5 stations straddling the +-180 deg magnetic longitude boundary; station 4 is Zero-Shot (~in_train)
    coords = np.array(
        [
            [60.0, 178.0, 55.0, 178.0],
            [60.0, -178.0, 55.0, -178.0],
            [55.0, 179.0, 50.0, 179.0],
            [65.0, -179.0, 60.0, -179.0],
            [60.0, 179.5, 55.0, 179.8],  # Zero-Shot station right between 178 and -178!
        ],
        dtype=np.float32,
    )
    n_t, n_st = 100, 5
    rng = np.random.default_rng(404)
    base_sig = np.linspace(-30.0, 30.0, n_t)[:, None, None]
    x_field = (base_sig + rng.normal(0.0, 1.0, size=(n_t, n_st, 3))).astype(np.float32)
    m_qc = np.ones((n_t, n_st), dtype=bool)
    train_idx = np.arange(0, 80, dtype=np.int64)
    in_train = np.array([True, True, True, True, False], dtype=bool)

    # Reject zscore_per_station scaler
    zsc = FieldScaler(mode="zscore_per_station", min_train_points=10).fit(
        x_field, m_qc, train_idx, np.zeros(n_st, dtype=np.int8), station_train_mask=in_train
    )
    with pytest.raises(ValueError, match="zscore_per_station"):
        IDWBaseline().fit(x_field, m_qc, train_idx, in_train, station_coords=coords, scaler=zsc)
    with pytest.raises(ValueError, match="zscore_per_station"):
        OrdinaryKrigingBaseline().fit(x_field, m_qc, train_idx, in_train, station_coords=coords, scaler=zsc)

    idw = IDWBaseline(coord_metric="magnetic").fit(
        x_field, m_qc, train_idx, in_train, station_coords=coords
    )
    krig = OrdinaryKrigingBaseline(min_nugget_ratio=1e-2).fit(
        x_field, m_qc, train_idx, in_train, station_coords=coords
    )
    # Check nugget ratio floor
    for c in range(3):
        assert krig.variogram_params_[c, 3] / krig.variogram_params_[c, 2] >= 1e-2 - 1e-9

    obs_zs = np.broadcast_to(in_train[None, :], (n_t, n_st)).copy()
    tgt_zs = ~obs_zs
    x_in = np.where(obs_zs[..., None], x_field, 0.0)

    pred_idw = idw.predict_masked(x_in, obs_zs, tgt_zs)
    pred_krig = krig.predict_masked(x_in, obs_zs, tgt_zs)
    assert np.all(np.isfinite(pred_idw[:, 4, :]))
    assert np.all(np.isfinite(pred_krig[:, 4, :]))
    assert float(np.sqrt(np.mean((pred_idw[:, 4, :] - x_field[:, 4, :]) ** 2))) < 3.0
    assert float(np.sqrt(np.mean((pred_krig[:, 4, :] - x_field[:, 4, :]) ** 2))) < 3.0

    # Constant field test -> exact 0 error
    x_const = np.full((20, n_st, 3), 42.5, dtype=np.float32)
    x_const_in = np.where(obs_zs[:20, ..., None], x_const, 0.0)
    assert np.max(np.abs(idw.predict_masked(x_const_in, obs_zs[:20], tgt_zs[:20])[:, 4, :] - 42.5)) < 1e-4
    assert np.max(np.abs(krig.predict_masked(x_const_in, obs_zs[:20], tgt_zs[:20])[:, 4, :] - 42.5)) < 1e-3


def test_baseline_state_npz_roundtrip(tmp_path: Path) -> None:
    """Verify `.npz` serialization roundtrip preserves predictions exactly for all 4 baselines."""
    rng = np.random.default_rng(505)
    n_t, n_st = 120, 6
    ts = 1_791_072_000 + np.arange(n_t, dtype=np.int64) * 60
    tf = compute_time_features(ts)
    coords = np.column_stack(
        [
            np.linspace(45.0, 65.0, n_st),
            np.linspace(5.0, 30.0, n_st),
            np.linspace(42.0, 63.0, n_st),
            np.linspace(85.0, 115.0, n_st),
        ]
    ).astype(np.float32)
    x = rng.normal(0.0, 15.0, size=(n_t, n_st, 3)).astype(np.float32)
    m_qc = np.ones((n_t, n_st), dtype=bool)
    tr_idx = np.arange(0, 90, dtype=np.int64)
    in_tr = np.ones(n_st, dtype=bool)

    obs = np.ones((30, n_st), dtype=bool)
    obs[:, 0] = False
    x_in = np.where(obs[..., None], x[90:], 0.0)

    models = [
        EMPCABaseline(k=3, use_sq_detrend=True).fit(x, m_qc, tr_idx, in_tr, time_features=tf, timestamps_unix_s=ts),
        RidgeSchurBaseline(alpha_grid=(1.0,)).fit(x, m_qc, tr_idx, in_tr),
        IDWBaseline().fit(x, m_qc, tr_idx, in_tr, station_coords=coords),
        OrdinaryKrigingBaseline().fit(x, m_qc, tr_idx, in_tr, station_coords=coords),
    ]
    for idx, mdl in enumerate(models):
        p = tmp_path / f"m_{idx}.npz"
        mdl.save_npz(p)
        loaded = mdl.__class__.load_npz(p)
        y1 = mdl.predict_masked(x_in, obs, ~obs, time_features=tf[90:])
        y2 = loaded.predict_masked(x_in, obs, ~obs, time_features=tf[90:])
        assert np.allclose(y1, y2, atol=1e-6, equal_nan=True)


def test_portable_dataset_path_in_config_serialization(
    ml_root: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    """Regression test for ML-03: portable_path_str omits machine-local absolute prefixes."""
    from geomag_ml.utils.io import portable_path_str

    ds_abs = (ml_root / "data_cache" / "geomag_ml_oct2026_clean_v1.npz").resolve()
    rel = portable_path_str(ds_abs, repo_root=repo_root)
    expected_rel = (
        "ml_research/data_cache/geomag_ml_oct2026_clean_v1.npz"
        if (repo_root / "ml_research").is_dir()
        else "data_cache/geomag_ml_oct2026_clean_v1.npz"
    )
    assert rel == expected_rel
    assert ":" not in rel

    ext_file = (tmp_path / "external_bundle.npz").resolve()
    ext_str = portable_path_str(ext_file, repo_root=repo_root)
    assert ext_str == ext_file.as_posix()

