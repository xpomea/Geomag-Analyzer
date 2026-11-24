"""Phase 2 unit tests: D0 guard, despiking, multitaper, aggregation, and NaN subsegments."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import geomag_core as gc

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RAW_INTERMAGNET_DIR = (
    REPO_ROOT / "GeoMag_Storms_Dataset_Oct-2026" / "GeoMag Storms" / "INTERMAGNET_1min"
)


def test_geomag_core_has_no_web_or_etl_dependencies():
    core_dir = REPO_ROOT / "backend" / "geomag_core"
    forbidden = ("fastapi", "starlette", "uvicorn", "pandas", "geomag_etl", "geomag_api")
    for py_file in core_dir.glob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        for mod in forbidden:
            assert f"import {mod}" not in text and f"from {mod}" not in text, (
                f"Forbidden dependency {mod!r} found in {py_file.name}"
            )


def test_d0_rotation_invariance_and_spectral_equivalence():
    t = np.arange(1440, dtype=np.float64) * gc.DT_1MIN_SEC
    xy = gc.synthetic_ellipse_2d(t, T_sec=300.0, a=20.0, b=10.0, psi0_deg=30.0, h=1)
    # Add quiet full-field background along D0 = 18.5 deg
    d0_deg = 18.5
    d0_rad = math.radians(d0_deg)
    bg_x = 20000.0 * math.cos(d0_rad)
    bg_y = 20000.0 * math.sin(d0_rad)
    bg_z = 45000.0
    xyz = np.column_stack([bg_x + xy[:, 0], bg_y + xy[:, 1], bg_z + np.zeros_like(t)])

    d0_est = gc.compute_quiet_declination_rad(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    assert abs(math.degrees(d0_est) - d0_deg) < 1e-3

    # Rotate time series first vs rotate spectral matrix S directly
    xyz_rot = gc.rotate_xyz_to_magnetic(xyz, d0_est)
    s_geo = gc.compute_welch_spectral_matrix(xyz, band="Pc5").S
    s_from_series = gc.compute_welch_spectral_matrix(xyz_rot, band="Pc5").S
    s_from_matrix = gc.rotate_spectral_matrix(s_geo, d0_est)
    assert np.allclose(s_from_series, s_from_matrix, rtol=1e-11, atol=1e-9)

    st_geo = gc.stokes_from_spectral_matrix_2d(s_geo)
    st_mag = gc.stokes_from_spectral_matrix_2d(s_from_matrix)
    assert math.isclose(st_mag.I, st_geo.I, rel_tol=1e-9)
    assert math.isclose(st_mag.V, st_geo.V, rel_tol=1e-9)
    assert math.isclose(st_mag.p, st_geo.p, abs_tol=1e-9)
    diff_psi = abs((st_mag.psi_deg - (st_geo.psi_deg - math.degrees(d0_est)) + 90.0) % 180.0 - 90.0)
    assert diff_psi < 1e-9


def test_d0_baseline_guard_blocks_gck_nck_and_allows_res_and_external_d0():
    ep1 = pd.read_parquet(PROCESSED_DIR / "geomag" / "storm_20261004_05.parquet")

    for baseless_code in ("GCK",):
        sub = ep1[ep1["station_iaga"] == baseless_code].sort_values("timestamp_utc")
        xyz = sub[["x_nT", "y_nT", "z_nT"]].to_numpy(dtype=np.float64)
        assert not gc.has_valid_baseline(xyz)
        with pytest.raises(ValueError):
            gc.compute_quiet_declination_rad(xyz[:, 0], xyz[:, 1], xyz[:, 2])
        with pytest.raises(ValueError):
            gc.compute_quiet_declination_rad(xyz[:, 0], xyz[:, 1])
        with pytest.raises(ValueError):
            gc.compute_quiet_declination_rad(xyz[:, 0], xyz[:, 1], has_baseline=False)
        # Explicit external D0 must succeed even when has_baseline=False
        ext_d0 = gc.compute_quiet_declination_rad(
            xyz[:, 0], xyz[:, 1], has_baseline=False, external_d0_deg=5.0
        )
        assert math.isclose(ext_d0, math.radians(5.0), abs_tol=1e-12)

    ep3 = pd.read_parquet(PROCESSED_DIR / "geomag" / "storm_20261021_23.parquet")
    nck = ep3[ep3["station_iaga"] == "NCK"].sort_values("timestamp_utc")
    nck_xyz = nck[["x_nT", "y_nT", "z_nT"]].to_numpy(dtype=np.float64)
    assert not gc.has_valid_baseline(nck_xyz)
    with pytest.raises(ValueError):
        gc.compute_quiet_declination_rad(nck_xyz[:, 0], nck_xyz[:, 1], nck_xyz[:, 2])

    # Arctic full-field station RES (low H_mean < 5000 nT, high |meanXYZ| > 57000 nT) must pass
    res = ep1[ep1["station_iaga"] == "RES"].sort_values("timestamp_utc")
    res_xyz = res[["x_nT", "y_nT", "z_nT"]].to_numpy(dtype=np.float64)
    assert gc.has_valid_baseline(res_xyz)
    d0_res = gc.compute_quiet_declination_rad(res_xyz[:, 0], res_xyz[:, 1], res_xyz[:, 2])
    assert math.isfinite(d0_res)
    d0_res_xy = gc.compute_quiet_declination_rad(
        res_xyz[:, 0], res_xyz[:, 1], has_baseline=True
    )
    assert math.isclose(d0_res_xy, d0_res, abs_tol=1e-12)


def test_core_despiking_val_ott_and_sc_protection():
    # Read raw VAL (2026-10-09) and HAD (neighbor) directly from IAGA-2002
    def _load_raw_day_xyz(region: str, iaga: str, date_tag: str) -> np.ndarray:
        pattern = f"{iaga.lower()}{date_tag}*min.min"
        path = sorted((RAW_INTERMAGNET_DIR / region / iaga).glob(pattern))[0]
        lines = path.read_text(encoding="utf-8").splitlines()
        hdr = next(i for i, line in enumerate(lines) if line.startswith("DATE"))
        arr = np.fromiter(
            (float(tok) for line in lines[hdr + 1 :] for tok in line.split()[3:6]),
            dtype=np.float64,
            count=1440 * 3,
        ).reshape(1440, 3)
        arr[arr >= 88888.0] = np.nan
        if "HDZ" in lines[hdr - 10 : hdr] or lines[hdr].split()[3].endswith("H"):
            x, y, z = gc.hdz_to_xyz(arr[:, 0], arr[:, 1], arr[:, 2])
            return np.column_stack([x, y, z])
        return arr

    val_xyz = _load_raw_day_xyz("Europe", "VAL", "20261009")
    had_xyz = _load_raw_day_xyz("Europe", "HAD", "20261009")
    cleaned, flags = gc.despike_station_network(
        {"VAL": val_xyz, "HAD": had_xyz},
        {"VAL": (51.933, -10.25), "HAD": (51.0, -4.48)},
    )
    idx_1039 = 10 * 60 + 39
    assert bool(flags["VAL"][idx_1039]) is True
    assert 19500.0 < cleaned["VAL"][idx_1039, 0] < 19680.0
    assert -1000.0 < cleaned["VAL"][idx_1039, 1] < -900.0
    assert 44100.0 < cleaned["VAL"][idx_1039, 2] < 44200.0

    # Synthetic 3-component spike 2 minutes before a +800 nT baseline step (step-immune test)
    t_step = np.arange(120, dtype=np.float64)
    st_base = np.column_stack(
        [20000.0 + 0.2 * np.sin(t_step), -800.0 + 0.2 * np.cos(t_step), 44000.0 + 0 * t_step]
    )
    sp1 = st_base.copy()
    sp2 = st_base.copy()
    sp1[62:, 0] += 60.0
    sp1[62:, 1] += 60.0
    sp1[62:, 2] += 800.0
    true_60 = sp1[60].copy()
    sp1[60] += np.array([-320.0, -340.0, -1200.0])
    cl_step, fl_step = gc.despike_station_network(
        {"A": sp1, "B": sp2}, {"A": (50.0, 20.0), "B": (50.5, 20.5)}
    )
    assert bool(fl_step["A"][60]) is True
    assert np.allclose(cl_step["A"][60], true_60, atol=0.5)

    # Synthetic multi-station SC protection
    t = np.arange(180, dtype=np.float64)
    base = np.column_stack([20000.0 + np.sin(t / 8.0), 500.0 + np.cos(t / 8.0), 42000.0 + 0 * t])
    s1 = base.copy()
    s2 = base.copy()
    s1[40, 2] -= 600.0
    s1[100, 0] += 250.0
    s2[100, 0] += 230.0
    cl2, fl2 = gc.despike_station_network(
        {"S1": s1, "S2": s2}, {"S1": (49.0, 24.0), "S2": (50.0, 25.0)}
    )
    assert bool(fl2["S1"][40]) is True and abs(cl2["S1"][40, 2] - 42000.0) < 1e-6
    assert bool(fl2["S1"][100]) is False and abs(cl2["S1"][100, 0] - (base[100, 0] + 250.0)) < 1e-6


def test_multitaper_recovers_exact_case_a_ellipse():
    t = np.arange(512, dtype=np.float64) * gc.DT_1MIN_SEC
    xy = gc.synthetic_ellipse_2d(t, T_sec=300.0, a=20.0, b=10.0, psi0_deg=30.0, h=1)
    mt = gc.compute_multitaper_spectral_matrix(xy, band="Pc5", NW=2.5, K_tapers=4)
    assert mt.K_tapers == 4
    st = gc.stokes_from_spectral_matrix_2d(mt.S)
    assert math.isclose(st.p, 1.0, abs_tol=1e-4)
    assert abs(st.psi_deg - 30.0) < 0.05
    assert abs(st.chi_deg - 26.565) < 0.05
    assert math.isclose(st.epsilon, 0.5, abs_tol=1e-3)


def test_stokes_and_matrix_aggregation_invariance():
    rng = np.random.default_rng(9)
    t = np.arange(1440, dtype=np.float64) * gc.DT_1MIN_SEC
    w1 = gc.synthetic_ellipse_2d(t, T_sec=300.0, a=18.0, b=9.0, psi0_deg=25.0, h=1)
    w2 = gc.synthetic_ellipse_2d(t, T_sec=400.0, a=12.0, b=6.0, psi0_deg=-15.0, h=-1) + rng.normal(
        0.0, 1.5, size=(1440, 2)
    )
    s1 = gc.compute_welch_spectral_matrix(w1, band="Pc5").S
    s2 = gc.compute_welch_spectral_matrix(w2, band="Pc5").S

    s_agg = gc.aggregate_spectral_matrices([s1, s2], weights=[2.0, 1.0])
    st_from_mat = gc.stokes_from_spectral_matrix_2d(s_agg)
    st_from_stokes = gc.aggregate_stokes(
        [gc.stokes_from_spectral_matrix_2d(s1), gc.stokes_from_spectral_matrix_2d(s2)],
        weights=[2.0, 1.0],
    )
    assert math.isclose(st_from_mat.p, st_from_stokes.p, abs_tol=1e-12)
    assert math.isclose(st_from_mat.psi_deg, st_from_stokes.psi_deg, abs_tol=1e-12)
    assert math.isclose(st_from_mat.chi_deg, st_from_stokes.chi_deg, abs_tol=1e-12)


def test_variant_b_nan_subsegment_splitting_synthetic_and_real():
    # 1) Synthetic 48h wave split by a 10-min NaN gap and a short 4h isolated fragment
    n_total = 2880
    t = np.arange(n_total, dtype=np.float64) * gc.DT_1MIN_SEC
    xy = gc.synthetic_ellipse_2d(t, T_sec=300.0, a=20.0, b=10.0, psi0_deg=30.0, h=1)
    # Segment 1: [0, 1200) (20h -> valid >= 7h); Gap1: [1200, 1210);
    # Segment 2: [1210, 1450) (4h -> too short for 3h+3h+1h = 7h -> must be discarded);
    # Gap2: [1450, 1465); Segment 3: [1465, 2880) (23.6h -> valid)
    xy[1200:1210] = np.nan
    xy[1450:1465] = np.nan

    res = gc.run_variant_b_pipeline(
        xy, band="Pc5", filter_method="sos", order_n=3, oversample_m=4, trim_hours=3.0
    )
    assert res.is_valid
    assert res.subsegment_ranges_orig == ((0, 1200), (1465, 2880))
    # Zero NaNs inside usable mask, NaN everywhere outside usable mask
    assert np.all(np.isfinite(res.filtered[res.usable_mask]))
    assert np.all(np.isfinite(res.analytic[res.usable_mask]))
    assert np.all(np.isnan(res.filtered[~res.usable_mask]))

    st = gc.stokes_from_spectral_matrix_2d(res.S, n_eff=res.n_eff)
    assert math.isclose(st.p, 1.0, abs_tol=1e-3)
    assert abs(st.psi_deg - 30.0) < 0.05
    assert abs(st.chi_deg - 26.565) < 0.05
    assert st.is_significant_p is True

    # 2) Real INTERMAGNET stations with gaps > 3 min: NUR (972-min block) & BLC (6-15 min gaps)
    ep2 = pd.read_parquet(PROCESSED_DIR / "geomag" / "storm_20261009_11.parquet")
    for station_code in ("NUR", "BLC"):
        sub = ep2[ep2["station_iaga"] == station_code].sort_values("timestamp_utc")
        xyz = sub[["x_nT", "y_nT", "z_nT"]].to_numpy(dtype=np.float64)
        assert not np.all(np.isfinite(xyz)), f"Expected NaN gaps in {station_code}"
        res_real = gc.run_variant_b_pipeline(
            xyz, band="Pc5", filter_method="sos", order_n=3, oversample_m=4
        )
        assert res_real.is_valid
        assert len(res_real.subsegment_ranges_orig) >= 1
        assert np.all(np.isfinite(res_real.filtered[res_real.usable_mask]))
        assert np.all(np.isfinite(res_real.analytic[res_real.usable_mask]))
        st_real = gc.stokes_from_spectral_matrix_2d(res_real.S, n_eff=res_real.n_eff)
        assert math.isfinite(st_real.p) and 0.0 <= st_real.p <= 1.0
        assert math.isfinite(st_real.p_crit)


def test_variant_b_respects_is_valid_xyz_qc_thresholds():
    n_total = 1440
    t = np.arange(n_total, dtype=np.float64) * gc.DT_1MIN_SEC
    xy = gc.synthetic_ellipse_2d(t, T_sec=300.0, a=20.0, b=10.0, psi0_deg=30.0, h=1)

    # Interior [180:1260) has 1080 points; mark 8% (~86 pts) as interpolated False
    valid_bad = np.ones(n_total, dtype=bool)
    valid_bad[200 : 200 + 86 * 4 : 4] = False  # single-min gaps, >5% missing in usable window
    res_bad = gc.run_variant_b_pipeline(
        xy, band="Pc5", filter_method="sos", is_valid_xyz=valid_bad
    )
    assert res_bad.is_valid is False

    # 10 isolated 1-min gaps (~0.9% < 5%) must remain valid
    valid_ok = np.ones(n_total, dtype=bool)
    valid_ok[200:250:5] = False
    res_ok = gc.run_variant_b_pipeline(
        xy, band="Pc5", filter_method="sos", is_valid_xyz=valid_ok
    )
    assert res_ok.is_valid is True


def test_default_trim_hours_and_variant_b_custom_bands():
    # Standard Table 14 / §9.3 bands preserved
    assert gc.default_trim_hours("Pc5", "fft_mask") == 3.0
    assert gc.default_trim_hours("Pc6", "fft_mask", t_max_pc6_sec=1800.0) == 5.0
    assert gc.default_trim_hours("Pc6", "sos", order_n=3, t_max_pc6_sec=1800.0) == 3.0
    assert gc.default_trim_hours("Pc6", "sos", order_n=4, t_max_pc6_sec=1800.0) == 3.5

    # Long/custom passbands scale cleanly with T_max instead of falling back to 3.0 h
    assert gc.default_trim_hours("Pc6", "fft_mask", t_max_pc6_sec=3600.0) == 10.0
    assert gc.default_trim_hours("Pc6", "sos", order_n=3, t_max_pc6_sec=3600.0) == 6.0
    custom_fft_trim = gc.default_trim_hours(
        "custom", "fft_mask", f1_hz=1.0 / 3600.0, f2_hz=1.0 / 600.0
    )
    assert custom_fft_trim == 10.0

    # End-to-end Variant B recovery on T=3000s wave inside T_max=3600s Pc6 band
    n_total = 36 * 60
    t = np.arange(n_total, dtype=np.float64) * gc.DT_1MIN_SEC
    xy = gc.synthetic_ellipse_2d(t, T_sec=3000.0, a=15.0, b=5.0, psi0_deg=-20.0, h=-1)
    res = gc.run_variant_b_pipeline(
        xy,
        band="Pc6",
        filter_method="fft_mask",
        t_max_pc6_sec=3600.0,
        oversample_m=4,
    )
    assert res.is_valid
    assert res.trim_hours == 10.0
    st = gc.stokes_from_spectral_matrix_2d(res.S)
    assert math.isclose(st.p, 1.0, abs_tol=0.01)
    assert abs(st.psi_deg - (-20.0)) < 0.5
