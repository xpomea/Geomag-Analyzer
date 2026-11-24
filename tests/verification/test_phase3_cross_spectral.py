"""Phase 3 verification tests: two-phase Welch, m-number, FLR, and XWT/WTC + AR(1) MC."""

from __future__ import annotations

import math
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import geomag_core as gc

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def test_synthetic_45deg_phase_shift_welch_and_wtc_agreement():
    t = np.arange(1440, dtype=np.float64) * gc.DT_1MIN_SEC
    omega = 2.0 * math.pi / 300.0
    # Localized Pc5 wave packet centered at t = 720 min where signal 1 leads signal 2 by +45 deg
    env = np.exp(-0.5 * ((t - 720.0 * 60.0) / (180.0 * 60.0)) ** 2)
    x1 = 20.0 * env * np.cos(omega * t + math.radians(45.0))
    x2 = 20.0 * env * np.cos(omega * t)

    # 1) Two-phase Welch on central window around packet
    sp1 = gc.compute_station_phase1_spectra(x1[360:1080], station_iaga="S1", L=128, M=64)
    sp2 = gc.compute_station_phase1_spectra(x2[360:1080], station_iaga="S2", L=128, M=64)
    pair_pos = gc.compute_welch_cross_pair(sp1, sp2, band="Pc5")
    pair_neg = gc.compute_welch_cross_pair(sp2, sp1, band="Pc5")

    assert bool(pair_pos.is_valid[0]) is True
    assert bool(pair_pos.is_significant[0]) is True
    assert float(pair_pos.gamma2[0]) > 0.99
    assert abs(float(pair_pos.phase_deg[0]) - 45.0) <= 0.5
    assert abs(float(pair_neg.phase_deg[0]) + 45.0) <= 0.5

    # 2) Morlet XWT and WTC at T = 300 s, t = 720 min
    xw = gc.compute_xwt_wtc(x1, x2, n_surrogates=0)
    xw_rev = gc.compute_xwt_wtc(x2, x1, n_surrogates=0)
    s_idx = int(np.argmin(np.abs(xw.periods_sec - 300.0)))
    c_idx = 720

    assert bool(xw.inside_coi[s_idx, c_idx]) is True
    assert float(xw.wtc_r2[s_idx, c_idx]) > 0.98
    assert abs(float(xw.xwt_phase_deg[s_idx, c_idx]) - 45.0) <= 0.5
    assert abs(float(xw.wtc_phase_deg[s_idx, c_idx]) - 45.0) <= 0.5
    assert abs(float(xw_rev.wtc_phase_deg[s_idx, c_idx]) + 45.0) <= 0.5


def test_welch_joint_mask_recalculation_and_cauchy_schwarz():
    # Synthetic asymmetric gaps across two stations
    rng = np.random.default_rng(11)
    n_pts = 1440
    x1 = rng.standard_normal(n_pts)
    x2 = 0.7 * x1 + 0.3 * rng.standard_normal(n_pts)
    v1 = np.ones(n_pts, dtype=bool)
    v2 = np.ones(n_pts, dtype=bool)
    # Station 1 missing [0, 260), Station 2 missing [1000, 1440)
    x1[:260] = np.nan
    v1[:260] = False
    x2[1000:] = np.nan
    v2[1000:] = False

    sp1 = gc.compute_station_phase1_spectra(x1, station_iaga="A", is_valid=v1, L=128, M=64)
    sp2 = gc.compute_station_phase1_spectra(x2, station_iaga="B", is_valid=v2, L=128, M=64)
    pair = gc.compute_welch_cross_pair(sp1, sp2, band="Pc5")

    common_mask = v1 & v2
    expected_k = int(np.sum(sp1.valid_segments[0] & sp2.valid_segments[0]))
    assert int(pair.K_ij[0]) == expected_k
    assert expected_k < min(int(sp1.valid_segments.sum()), int(sp2.valid_segments.sum()))
    assert 0.0 <= float(pair.gamma2[0]) <= 1.0

    # Verify S_ii^{(ij)} and S_jj^{(ij)} strictly match Welch computed on the common mask alone
    w1_common = gc.compute_welch_spectral_matrix(
        np.column_stack([np.where(common_mask, x1, np.nan), np.where(common_mask, x2, np.nan)]),
        band="Pc5",
        is_valid_xyz=common_mask,
    )
    assert math.isclose(float(pair.S_ii[0]), float(np.real(w1_common.S[0, 0])), rel_tol=1e-12)
    assert math.isclose(float(pair.S_jj[0]), float(np.real(w1_common.S[1, 1])), rel_tol=1e-12)
    assert math.isclose(abs(complex(pair.S_ij[0])), abs(complex(w1_common.S[0, 1])), rel_tol=1e-12)

    # Real INTERMAGNET asymmetric pair: NUR (972-min gap) vs LVV in storm_20261009_11
    ep2 = pd.read_parquet(PROCESSED_DIR / "geomag" / "storm_20261009_11.parquet")
    nur = ep2[ep2["station_iaga"] == "NUR"].sort_values("timestamp_utc")
    lvv = ep2[ep2["station_iaga"] == "LVV"].sort_values("timestamp_utc")
    sp_nur = gc.compute_station_phase1_spectra(
        nur["x_nT"].to_numpy(),
        station_iaga="NUR",
        is_valid=nur["is_valid_xyz"].to_numpy(),
        window_samples=360,
        step_samples=120,
    )
    sp_lvv = gc.compute_station_phase1_spectra(
        lvv["x_nT"].to_numpy(),
        station_iaga="LVV",
        is_valid=lvv["is_valid_xyz"].to_numpy(),
        window_samples=360,
        step_samples=120,
    )
    pair_real = gc.compute_welch_cross_pair(sp_nur, sp_lvv, band="Pc5")
    valid_g2 = pair_real.gamma2[pair_real.is_valid]
    assert valid_g2.size > 0
    assert np.all((valid_g2 >= 0.0) & (valid_g2 <= 1.0))

    # Matrix endpoint must exactly match pairwise endpoint
    mat = gc.compute_welch_cross_matrix([sp_nur, sp_lvv], band="Pc5")
    assert np.allclose(mat.gamma2[:, 0, 1], pair_real.gamma2, equal_nan=True, atol=1e-12)
    assert np.array_equal(mat.K_ij[:, 0, 1], pair_real.K_ij)
    assert np.allclose(mat.gamma2_crit[:, 0, 1], pair_real.gamma2_crit, equal_nan=True, atol=1e-12)


def test_welch_49x49_1176_pairs_performance_under_2s():
    # 49 stations, 3-day grid (4320 samples), 288 dynamic macro-windows (window=360, step=14 -> 283)
    rng = np.random.default_rng(7)
    n_stations = 49
    n_samples = 4320
    data = rng.standard_normal((n_stations, n_samples))
    spectra = [
        gc.compute_station_phase1_spectra(
            data[i],
            station_iaga=f"ST{i:02d}",
            window_samples=360,
            step_samples=13,  # (4320 - 360) // 13 + 1 = 305 >= 288 steps
            L=128,
            M=64,
        )
        for i in range(n_stations)
    ]

    t0 = time.perf_counter()
    mat = gc.compute_welch_cross_matrix(spectra, band="Pc5", time_chunk_size=32)
    elapsed = time.perf_counter() - t0

    assert mat.gamma2.shape == (305, 49, 49)
    assert elapsed < 2.0, f"49x49 Phase 2 took {elapsed:.3f}s (limit < 2.0s)"
    assert np.allclose(mat.gamma2, np.swapaxes(mat.gamma2, 1, 2), atol=1e-12)
    assert np.allclose(mat.phase_deg, -np.swapaxes(mat.phase_deg, 1, 2), atol=1e-10)


def test_azimuthal_wavenumber_m_guard_and_aliasing():
    # Meridional pair (|d_lon| = 1.2 deg < 3.0 deg) must be guarded
    res_merid = gc.compute_azimuthal_wavenumber_m(
        np.array([0.5, 0.6]),
        lon1_deg=20.0,
        lon2_deg=21.2,
        gamma2=np.array([0.9, 0.9]),
        gamma2_crit=np.array([0.2, 0.2]),
        min_delta_lon_deg=3.0,
    )
    assert res_merid.is_meridional_guarded is True
    assert np.all(np.isnan(res_merid.m))
    assert math.isnan(res_merid.m_alias)

    # Zonal pair (10 deg separation): +45 deg = +pi/4 rad -> m = (pi/4) / (10*pi/180) = 4.5
    d_phi = np.array([math.radians(45.0), math.radians(45.0)])
    g2 = np.array([0.95, 0.10])
    gcrit = np.array([0.30, 0.30])
    res_zonal = gc.compute_azimuthal_wavenumber_m(
        d_phi, lon1_deg=25.0, lon2_deg=15.0, gamma2=g2, gamma2_crit=gcrit
    )
    assert res_zonal.is_meridional_guarded is False
    assert math.isclose(res_zonal.m_alias, 18.0, abs_tol=1e-9)
    assert math.isclose(float(res_zonal.m[0]), 4.5, abs_tol=1e-9)
    assert math.isnan(float(res_zonal.m[1]))

    # Antimeridian wrap check (-175 vs +175 -> d_lon = +10 deg)
    res_wrap = gc.compute_azimuthal_wavenumber_m(
        math.radians(45.0), lon1_deg=-175.0, lon2_deg=175.0, gamma2=0.9, gamma2_crit=0.2
    )
    assert math.isclose(res_wrap.delta_lon_deg, 10.0, abs_tol=1e-9)
    assert math.isclose(float(res_wrap.m), 4.5, abs_tol=1e-9)


def test_meridional_flr_profile_synthetic_and_real_chain():
    # Synthetic FLR profile across 5 latitudes peaking at 60 deg with phase progression
    t = np.arange(1440, dtype=np.float64) * gc.DT_1MIN_SEC
    omega = 2.0 * math.pi / 300.0
    lats = [50.0, 55.0, 60.0, 65.0, 70.0]
    amps = [5.0, 12.0, 25.0, 11.0, 4.0]
    phases_deg = [0.0, 30.0, 90.0, 150.0, 180.0]
    chain_sp = []
    for i, (amp, ph) in enumerate(zip(amps, phases_deg, strict=True)):
        sig = amp * np.cos(omega * t + math.radians(ph))
        chain_sp.append(gc.compute_station_phase1_spectra(sig, station_iaga=f"FLR{i}"))

    prof = gc.compute_meridional_flr_profile(chain_sp, lats, band="Pc5")
    assert prof.peak_station == "FLR2"
    assert math.isclose(prof.peak_latitude_deg, 60.0)
    assert np.all(prof.pair_is_significant)
    assert np.all(prof.pair_coherence > 0.99)
    assert math.isclose(float(prof.cumulative_phase_deg[-1]), 180.0, abs_tol=1.0)
    assert np.all(prof.phase_gradient_deg_per_lat > 0.0)

    # Real Scandinavian-Eastern European chain in storm_20261009_11
    ep2 = pd.read_parquet(PROCESSED_DIR / "geomag" / "storm_20261009_11.parquet")
    chain_codes = ["HRN", "ABK", "KIR", "LYC", "UPS", "BEL", "LVV", "PAG", "PEG"]
    lats_real = []
    specs_real = []
    for code in chain_codes:
        sub = ep2[ep2["station_iaga"] == code].sort_values("timestamp_utc")
        # Subset to first 24h (1440 samples)
        sub24 = sub.iloc[:1440]
        specs_real.append(
            gc.compute_station_phase1_spectra(
                sub24["x_nT"].to_numpy(),
                station_iaga=code,
                is_valid=sub24["is_valid_xyz"].to_numpy(),
            )
        )
    # Use representative geodetic latitudes from North to South
    lats_real = [77.0, 68.36, 67.84, 64.61, 59.90, 51.84, 49.90, 42.52, 38.08]
    prof_real = gc.compute_meridional_flr_profile(specs_real, lats_real, band="Pc5")
    assert np.all(np.isfinite(prof_real.band_power))
    assert prof_real.peak_station in chain_codes


def test_xwt_wtc_nan_subsegments_and_gap_coi():
    t = np.arange(1440, dtype=np.float64) * gc.DT_1MIN_SEC
    omega = 2.0 * math.pi / 300.0
    x1 = 15.0 * np.cos(omega * t + math.radians(45.0))
    x2 = 15.0 * np.cos(omega * t)
    # Introduce a 100-min gap in the middle [650, 750)
    x1[650:750] = np.nan
    x2[650:750] = np.nan

    res_sub = gc.compute_xwt_wtc(x1, x2, gap_policy="subsegments", n_surrogates=0)
    assert res_sub.subsegments == ((0, 650), (750, 1440))
    assert np.all(np.isnan(res_sub.wtc_r2[:, 650:750]))
    assert not np.any(res_sub.inside_coi[:, 650:750])
    s_idx = int(np.argmin(np.abs(res_sub.periods_sec - 300.0)))
    assert abs(float(res_sub.wtc_phase_deg[s_idx, 325]) - 45.0) <= 0.5
    assert abs(float(res_sub.wtc_phase_deg[s_idx, 1095]) - 45.0) <= 0.5

    res_gap = gc.compute_xwt_wtc(x1, x2, gap_policy="gap_coi", n_surrogates=0)
    assert not np.any(res_gap.inside_coi[:, 700])  # inside gap -> coi = 0 -> masked False
    assert np.all(np.isnan(res_gap.wtc_r2[:, 650:750]))
    assert np.all(np.isnan(res_gap.xwt_power[:, 650:750]))
    assert np.all(np.isnan(res_gap.wtc_phase_deg[:, 650:750]))
    assert np.all(np.isnan(res_gap.xwt_phase_deg[:, 650:750]))
    assert bool(res_gap.inside_coi[s_idx, 325]) is True
    assert bool(res_gap.inside_coi[s_idx, 1095]) is True

    # Real INTERMAGNET pair with NaN gap: NUR vs LVV on storm_20261009_11
    ep2 = pd.read_parquet(PROCESSED_DIR / "geomag" / "storm_20261009_11.parquet")
    nur = ep2[ep2["station_iaga"] == "NUR"].sort_values("timestamp_utc")["x_nT"].to_numpy()
    lvv = ep2[ep2["station_iaga"] == "LVV"].sort_values("timestamp_utc")["x_nT"].to_numpy()
    res_real = gc.compute_xwt_wtc(nur, lvv, gap_policy="subsegments", n_surrogates=0)
    assert len(res_real.subsegments) >= 1
    assert np.all(np.isfinite(res_real.wtc_r2[res_real.inside_coi]))


def test_wtc_monte_carlo_significance_progress_cancel_and_numba_parity():
    rng = np.random.default_rng(99)
    n_pts = 720
    t = np.arange(n_pts, dtype=np.float64) * gc.DT_1MIN_SEC
    omega = 2.0 * math.pi / 300.0
    env = np.exp(-0.5 * ((t - 360.0 * 60.0) / (90.0 * 60.0)) ** 2)
    # Coherent +45 deg Pc5 wave packet embedded in AR(1) red noise
    noise1 = np.zeros(n_pts)
    noise2 = np.zeros(n_pts)
    for k in range(1, n_pts):
        noise1[k] = 0.7 * noise1[k - 1] + 0.7 * rng.standard_normal()
        noise2[k] = 0.7 * noise2[k - 1] + 0.7 * rng.standard_normal()
    x1 = noise1 + 8.0 * env * np.cos(omega * t + math.radians(45.0))
    x2 = noise2 + 8.0 * env * np.cos(omega * t)

    progress_log: list[tuple[int, int, str]] = []

    def _on_prog(done: int, total: int, stage: str) -> None:
        progress_log.append((done, total, stage))

    res_nb = gc.compute_xwt_wtc(
        x1,
        x2,
        n_surrogates=100,
        seed=42,
        batch_size=25,
        use_numba=True,
        progress_callback=_on_prog,
    )
    res_np = gc.compute_xwt_wtc(
        x1,
        x2,
        n_surrogates=100,
        seed=42,
        batch_size=25,
        use_numba=False,
    )

    assert progress_log == [
        (25, 100, "monte_carlo"),
        (50, 100, "monte_carlo"),
        (75, 100, "monte_carlo"),
        (100, 100, "monte_carlo"),
    ]
    assert res_nb.wtc_sig95 is not None
    assert np.all((res_nb.wtc_sig95 > 0.0) & (res_nb.wtc_sig95 <= 1.0))

    s_idx = int(np.argmin(np.abs(res_nb.periods_sec - 300.0)))
    assert bool(res_nb.wtc_is_significant[s_idx, 360]) is True

    # Parity between Numba JIT and pure-NumPy fallback
    assert np.allclose(res_nb.wtc_r2, res_np.wtc_r2, atol=1e-12)
    assert np.allclose(res_nb.wtc_sig95, res_np.wtc_sig95, atol=1e-12)

    # Cancellation token test: cancel after batch 2
    batch_counter = {"n": 0}

    def _cancel_after_two() -> bool:
        batch_counter["n"] += 1
        return batch_counter["n"] > 2

    with pytest.raises(gc.CalculationCancelledError):
        gc.compute_xwt_wtc(
            x1,
            x2,
            n_surrogates=200,
            seed=42,
            batch_size=25,
            cancel_check=_cancel_after_two,
        )


def test_phase1_and_wtc_cache_keys_and_npz_roundtrip(tmp_path: Path):
    k1 = gc.build_phase1_cache_key(
        gc.__version__,
        episode_id="storm_20261009_11",
        station_iaga="LVV",
        component="X",
        coord_frame="XYZ",
        d0_deg=0.0,
        L=128,
        M=64,
        window_samples=360,
        step_samples=15,
    )
    k2 = gc.build_phase1_cache_key(
        gc.__version__,
        episode_id="storm_20261009_11",
        station_iaga="LVV",
        component="Y",
        coord_frame="XYZ",
        d0_deg=0.0,
        L=128,
        M=64,
        window_samples=360,
        step_samples=15,
    )
    assert len(k1) == 64 and k1 != k2

    t = np.arange(512, dtype=np.float64) * gc.DT_1MIN_SEC
    x1 = np.cos(2.0 * math.pi * t / 300.0)
    x2 = np.sin(2.0 * math.pi * t / 300.0)

    sp = gc.compute_station_phase1_spectra(
        x1, station_iaga="LVV", window_samples=256, step_samples=64
    )
    p1_path = tmp_path / "phase1_lvv.npz"
    gc.save_phase1_cache(p1_path, sp)
    sp_loaded = gc.load_phase1_cache(p1_path)
    assert sp_loaded.cache_key == sp.cache_key
    assert np.allclose(sp_loaded.F, sp.F)
    assert np.array_equal(sp_loaded.valid_segments, sp.valid_segments)

    xw = gc.compute_xwt_wtc(x1, x2, n_surrogates=50, seed=10)
    wtc_path = tmp_path / "wtc_test.npz"
    gc.save_wtc_cache(wtc_path, xw)
    xw_loaded = gc.load_wtc_cache(wtc_path)
    assert xw_loaded.cache_key == xw.cache_key
    assert xw_loaded.subsegments == xw.subsegments
    assert np.allclose(xw_loaded.wtc_r2, xw.wtc_r2, equal_nan=True)
    assert np.allclose(xw_loaded.wtc_sig95, xw.wtc_sig95)


def test_estimate_ar1_unbiased_on_gapped_series():
    rng = np.random.default_rng(123)
    n = 4320
    alpha_true = 0.85
    x = np.zeros(n, dtype=np.float64)
    scale = math.sqrt(1.0 - alpha_true ** 2)
    for i in range(1, n):
        x[i] = alpha_true * x[i - 1] + scale * float(rng.standard_normal())

    for gap_frac in (0.0, 0.20, 0.50):
        xg = x.copy()
        if gap_frac > 0.0:
            xg[rng.random(n) < gap_frac] = np.nan
        est = gc.estimate_ar1(xg)
        assert abs(est - alpha_true) < 0.03, f"gap_frac={gap_frac}: got {est:.4f}"


def test_azimuthal_wavenumber_unwrap_time_1d_and_2d():
    raw_1d = np.array([2.5, 2.9, -3.0, -2.6, -2.2], dtype=np.float64)
    g2_1d = np.full(5, 0.9)
    gc_1d = np.full(5, 0.2)
    res_1d = gc.compute_azimuthal_wavenumber_m(
        raw_1d, lon1_deg=20.0, lon2_deg=10.0, gamma2=g2_1d, gamma2_crit=gc_1d, unwrap_time=True
    )
    assert np.all(np.diff(res_1d.delta_phi_rad) > 0.0)
    assert float(res_1d.delta_phi_rad[-1]) > math.pi

    raw_2d = np.column_stack([raw_1d, raw_1d + 0.1])
    g2_2d = np.full((5, 2), 0.9)
    gc_2d = np.full((5, 2), 0.2)
    res_2d = gc.compute_azimuthal_wavenumber_m(
        raw_2d, lon1_deg=20.0, lon2_deg=10.0, gamma2=g2_2d, gamma2_crit=gc_2d, unwrap_time=True
    )
    assert res_2d.m.shape == (5, 2)
    assert np.all(np.diff(res_2d.delta_phi_rad, axis=0) > 0.0)
    assert np.all(res_2d.delta_phi_rad[-1, :] > math.pi)


def test_meridional_flr_profile_stops_cumulative_phase_on_insignificant_link():
    rng = np.random.default_rng(31)
    t = np.arange(1440, dtype=np.float64) * gc.DT_1MIN_SEC
    omega = 2.0 * math.pi / 300.0
    s0 = 15.0 * np.cos(omega * t)
    s1 = rng.standard_normal(1440) * 5.0  # completely incoherent middle station
    s2 = 15.0 * np.cos(omega * t + math.radians(40.0))
    chain = [
        gc.compute_station_phase1_spectra(sig, station_iaga=f"M{i}")
        for i, sig in enumerate((s0, s1, s2))
    ]
    prof = gc.compute_meridional_flr_profile(chain, [50.0, 55.0, 60.0], band="Pc5")
    assert bool(prof.pair_is_significant[0]) is False
    assert math.isnan(float(prof.phase_gradient_deg_per_lat[0]))
    assert math.isnan(float(prof.cumulative_phase_deg[1]))
    assert math.isnan(float(prof.cumulative_phase_deg[2]))


def test_wtc_monte_carlo_peak_memory_3day():
    rng = np.random.default_rng(7)
    n_pts = 4320
    x1 = rng.standard_normal(n_pts)
    x2 = rng.standard_normal(n_pts)
    tracemalloc.start()
    try:
        res = gc.compute_xwt_wtc(x1, x2, n_surrogates=20, seed=7)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert res.wtc_sig95 is not None
    assert peak_bytes < 200 * 1024 * 1024, (
        f"3-day WTC MC peak memory {peak_bytes / 1e6:.1f} MB exceeded 200 MB limit"
    )


def test_morlet_dj0_scales_with_omega0():
    from geomag_core.cross_wavelet import _morlet_dj0

    assert math.isclose(_morlet_dj0(6.0), 0.60, abs_tol=1e-12)
    assert math.isclose(_morlet_dj0(8.0), 0.45, abs_tol=1e-12)

    k6 = gc.build_wtc_cache_key(
        gc.__version__,
        episode_id="ep",
        pair=("A", "B"),
        component="X",
        coord_frame="XYZ",
        d0_deg=0.0,
        period_min_sec=120.0,
        period_max_sec=1800.0,
        dj=1.0 / 12.0,
        omega0=6.0,
        gap_policy="subsegments",
        n_surrogates=0,
        alpha=0.05,
        seed=42,
    )
    k8 = gc.build_wtc_cache_key(
        gc.__version__,
        episode_id="ep",
        pair=("A", "B"),
        component="X",
        coord_frame="XYZ",
        d0_deg=0.0,
        period_min_sec=120.0,
        period_max_sec=1800.0,
        dj=1.0 / 12.0,
        omega0=8.0,
        gap_policy="subsegments",
        n_surrogates=0,
        alpha=0.05,
        seed=42,
    )
    assert k6 != k8

    t = np.arange(1440, dtype=np.float64) * gc.DT_1MIN_SEC
    omega = 2.0 * math.pi / 300.0
    x1 = 10.0 * np.cos(omega * t + math.radians(45.0))
    x2 = 10.0 * np.cos(omega * t)
    xw = gc.compute_xwt_wtc(x1, x2, omega0=8.0, n_surrogates=0)
    s_idx = int(np.argmin(np.abs(xw.periods_sec - 300.0)))
    assert abs(float(xw.wtc_phase_deg[s_idx, 720]) - 45.0) <= 0.5
