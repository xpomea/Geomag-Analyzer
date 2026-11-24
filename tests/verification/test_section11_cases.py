"""Mandatory §11 deterministic verification gate against Pc5_Pc6_Polarization_Formulas (.docx)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import signal

import geomag_core as gc

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "docx_section11.json"


def _load_fixtures() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _time_grid_1day() -> np.ndarray:
    return np.arange(1440, dtype=np.float64) * gc.DT_1MIN_SEC


def _build_case_signal(case_key: str, t_sec: np.ndarray, fixtures: dict) -> np.ndarray:
    spec = fixtures["table_18_synthetic_signals"][case_key]
    if spec["kind"] in ("2d_monochromatic", "2d_superposition", "2d_dual_band"):
        out = np.zeros((t_sec.size, 2), dtype=np.float64)
        for comp in spec["components"]:
            out += gc.synthetic_ellipse_2d(
                t_sec,
                T_sec=comp["T_sec"],
                a=comp["a"],
                b=comp["b"],
                psi0_deg=comp["psi0_deg"],
                h=comp["h"],
            )
        return out
    if spec["kind"] == "3d_monochromatic":
        w = 2.0 * math.pi / float(spec["T_sec"])
        x = spec["X_amp"] * np.cos(w * t_sec + math.radians(spec["X_phase_deg"]))
        y = spec["Y_amp"] * np.cos(w * t_sec + math.radians(spec["Y_phase_deg"]))
        z = spec["Z_amp"] * np.cos(w * t_sec + math.radians(spec["Z_phase_deg"]))
        return np.column_stack([x, y, z])
    if spec["kind"] == "2d_superposition_filtered":
        return _build_case_signal(spec["base_case"], t_sec, fixtures)
    raise ValueError(f"Unknown case kind {spec['kind']}")


def _assert_stokes_matches(
    stokes: gc.Stokes2D, expected: dict, tol_angle: float, tol_ratio: float
):
    if "Q_over_I" in expected:
        exp_pl = math.hypot(expected.get("Q_over_I", 0.0), expected.get("U_over_I", 0.0))
        assert math.isclose(stokes.p_L, exp_pl, abs_tol=tol_ratio)
    if "Q_over_I" in expected:
        assert math.isclose(stokes.Q / stokes.I, expected["Q_over_I"], abs_tol=tol_ratio)
    if "U_over_I" in expected:
        assert math.isclose(stokes.U / stokes.I, expected["U_over_I"], abs_tol=tol_ratio)
    if "V_over_I" in expected:
        assert math.isclose(stokes.p_C, expected["V_over_I"], abs_tol=tol_ratio)
    if "p" in expected:
        assert math.isclose(stokes.p, expected["p"], abs_tol=tol_ratio)
    if "psi_deg" in expected:
        if expected["psi_deg"] is None:
            assert math.isnan(stokes.psi_deg)
        else:
            # Handle axial 180-deg wrap (e.g. -90 vs +90)
            diff = abs((stokes.psi_deg - expected["psi_deg"] + 90.0) % 180.0 - 90.0)
            assert diff <= tol_angle
    if "chi_deg" in expected:
        assert abs(stokes.chi_deg - expected["chi_deg"]) <= tol_angle
    if "epsilon" in expected:
        assert math.isclose(stokes.epsilon, expected["epsilon"], abs_tol=tol_ratio)
    if "gamma2" in expected:
        assert math.isclose(stokes.gamma2, expected["gamma2"], abs_tol=tol_ratio)


def test_case_0_sign_convention():
    fixtures = _load_fixtures()
    t = _time_grid_1day()
    proc = fixtures["table_19_expected_2d"]["processing"]
    tol = fixtures["table_19_expected_2d"]["tolerances"]

    sig = _build_case_signal("0", t, fixtures)
    welch = gc.compute_welch_spectral_matrix(
        sig, dt=gc.DT_1MIN_SEC, band=proc["band"], L=proc["L"], M=proc["M"]
    )
    assert welch.is_valid
    assert welch.K_valid == proc["K"]
    assert welch.N_B == proc["N_B"]

    stokes = gc.stokes_from_spectral_matrix_2d(welch.S)
    _assert_stokes_matches(
        stokes,
        fixtures["table_19_expected_2d"]["cases"]["0"],
        tol["angle_deg"],
        tol["ratio"],
    )


def test_cases_a_b_c_2d():
    fixtures = _load_fixtures()
    t = _time_grid_1day()
    proc = fixtures["table_19_expected_2d"]["processing"]
    tol = fixtures["table_19_expected_2d"]["tolerances"]

    for case_id, exp_key in [("A", "A"), ("B", "B"), ("C", "C_xy")]:
        sig = _build_case_signal(case_id, t, fixtures)
        welch = gc.compute_welch_spectral_matrix(
            sig, dt=gc.DT_1MIN_SEC, band=proc["band"], L=proc["L"], M=proc["M"]
        )
        assert welch.K_valid == proc["K"]
        assert welch.N_B == proc["N_B"]

        expected = fixtures["table_19_expected_2d"]["cases"][exp_key]
        stokes = gc.stokes_from_spectral_matrix_2d(welch.S)
        _assert_stokes_matches(stokes, expected, tol["angle_deg"], tol["ratio"])

        # Normalize S to match Formula (21)-(23) trace convention (S_xx + S_yy)
        exp_trace = expected["S_xx"] + expected["S_yy"]
        s2 = welch.S[:2, :2]
        scale = exp_trace / float(np.real(np.trace(s2)))
        s_norm = s2 * scale
        assert abs(float(np.real(s_norm[0, 0])) - expected["S_xx"]) < 0.25
        assert abs(float(np.real(s_norm[1, 1])) - expected["S_yy"]) < 0.25
        assert abs(float(np.real(s_norm[0, 1])) - expected["S_xy_re"]) < 0.25
        assert abs(float(np.imag(s_norm[0, 1])) - expected["S_xy_im"]) < 0.25


def test_case_c_3d():
    fixtures = _load_fixtures()
    t = _time_grid_1day()
    proc = fixtures["table_19_expected_2d"]["processing"]
    exp3 = fixtures["case_c_expected_3d"]
    tol = exp3["tolerances"]

    sig3 = _build_case_signal("C", t, fixtures)
    welch = gc.compute_welch_spectral_matrix(
        sig3, dt=gc.DT_1MIN_SEC, band=proc["band"], L=proc["L"], M=proc["M"]
    )
    pol3 = gc.polarization_3d(welch.S)

    assert math.isclose(pol3.P, exp3["P"], abs_tol=tol["ratio"])
    assert abs(pol3.theta_deg - exp3["theta_deg"]) <= tol["angle_deg"]
    assert abs(pol3.phi_n_deg - exp3["phi_n_deg"]) <= tol["angle_deg"]
    assert math.isclose(pol3.b_over_a, exp3["b_over_a"], abs_tol=tol["ratio"])
    assert math.isclose(pol3.s, exp3["s"], abs_tol=tol["ratio"])

    nv_norm_ratio = pol3.n_v / pol3.tr_S
    assert np.allclose(nv_norm_ratio, exp3["n_v_over_tr_S"], atol=tol["ratio"])

    scale = exp3["tr_S"] / pol3.tr_S
    assert np.allclose(pol3.n_v * scale, exp3["n_v"], atol=tol["power_nT2"])
    assert np.allclose(
        pol3.eigenvalues_re_S * scale, exp3["eigenvalues_re_S"], atol=tol["power_nT2"]
    )


def test_table_11_sos_coefficients_and_scipy_agreement():
    fixtures = _load_fixtures()
    t11 = fixtures["table_11_sos_coefficients"]
    tol = t11["tolerance"]
    fs_m4 = float(t11["oversample_m"]) / gc.DT_1MIN_SEC

    for band in ("Pc5", "Pc6"):
        spec = gc.get_band_spec(band)
        sos = gc.design_butterworth_sos(
            spec.f_min_hz,
            spec.f_max_hz,
            fs_m4,
            order_n=t11["order_n"],
            zero_phase_corrected=True,
        )
        assert sos.shape == (3, 6)
        for row, ref in zip(sos, t11[band], strict=True):
            assert abs(row[0] - ref["b0"]) < tol
            assert abs(row[1] - ref["b1"]) < tol
            assert abs(row[2] - ref["b2"]) < tol
            assert abs(row[3] - ref["a0"]) < tol
            assert abs(row[4] - ref["a1"]) < tol
            assert abs(row[5] - ref["a2"]) < tol

    # Cross-check against Formula (39) and SciPy butter(..., output="sos") across n=1..4, m=1,4,8
    for band in ("Pc5", "Pc6"):
        spec = gc.get_band_spec(band)
        for order_n in (1, 2, 3, 4):
            for m in (1, 4, 8):
                fs = float(m) / gc.DT_1MIN_SEC
                f_grid = np.linspace(
                    spec.f_min_hz / 5.0, min(spec.f_max_hz * 3.0, 0.45 * fs), 500
                )

                sos_uncorr = gc.design_butterworth_sos(
                    spec.f_min_hz, spec.f_max_hz, fs, order_n=order_n, zero_phase_corrected=False
                )
                sos_scipy = signal.butter(
                    order_n, [spec.f_min_hz, spec.f_max_hz], btype="bandpass", fs=fs, output="sos"
                )
                _, h_doc = signal.sosfreqz(sos_uncorr, worN=f_grid, fs=fs)
                _, h_sp = signal.sosfreqz(sos_scipy, worN=f_grid, fs=fs)
                assert float(np.max(np.abs(np.abs(h_doc) - np.abs(h_sp)))) < 5e-13

                h_f39 = gc.butterworth_single_pass_response(
                    f_grid,
                    spec.f_min_hz,
                    spec.f_max_hz,
                    fs,
                    order_n=order_n,
                    zero_phase_corrected=False,
                )
                assert float(np.max(np.abs(np.abs(h_doc) - h_f39))) < 1e-12

                # Verify -3 dB boundaries after zero-phase correction (40): A(f1)=A(f2)=1/sqrt(2)
                sos_corr = gc.design_butterworth_sos(
                    spec.f_min_hz, spec.f_max_hz, fs, order_n=order_n, zero_phase_corrected=True
                )
                _, h_corr = signal.sosfreqz(
                    sos_corr, worN=np.array([spec.f_min_hz, spec.f_max_hz]), fs=fs
                )
                assert np.allclose(np.abs(h_corr) ** 2, 1.0 / math.sqrt(2.0), atol=1e-13)


def test_case_d_all_table_20_modes():
    fixtures = _load_fixtures()
    t = _time_grid_1day()
    sig_d = _build_case_signal("D", t, fixtures)
    t20 = fixtures["table_20_case_d"]
    trim_h = t20["processing"]["trim_hours"]
    tol = t20["tolerances"]
    spec = gc.get_band_spec("Pc5")

    for mode_row in t20["modes"]:
        method = mode_row["method"]
        if method == "analog":
            # Verify analog prototype analytical response + weighted line matrix formula (36)
            a_vals = gc.analog_butterworth_zero_phase_response(
                np.array([1.0 / 300.0, 1.0 / 450.0]),
                spec.f_min_hz,
                spec.f_max_hz,
                order_n=mode_row["order"],
            )
            assert math.isclose(float(a_vals[0]), mode_row["A_300"], abs_tol=1e-5)
            assert math.isclose(float(a_vals[1]), mode_row["A_450"], abs_tol=1e-5)
            # Theoretical weighted S from Case A and Case B second line (Formulas 22-23, 36)
            s_300 = np.array(
                [
                    [325.0, 129.9038105676658 + 200.0j],
                    [129.9038105676658 - 200.0j, 175.0],
                ]
            )
            s_450 = np.array([[100.0, -100.0j], [100.0j, 100.0]])
            s_ana = (float(a_vals[0]) ** 2) * s_300 + (float(a_vals[1]) ** 2) * s_450
            st = gc.stokes_from_spectral_matrix_2d(s_ana)
            assert math.isclose(st.p, mode_row["p"], abs_tol=tol["ratio"])
            assert math.isclose(st.p_C, mode_row["V_over_I"], abs_tol=tol["ratio"])
            assert abs(st.chi_deg - mode_row["chi_deg"]) <= tol["angle_deg"]
            assert abs(st.psi_deg - mode_row["psi_deg"]) <= tol["psi_deg"]
            continue

        order_n = 3 if mode_row["order"] is None else int(mode_row["order"])
        for m in mode_row["m_values"]:
            if method == "fft_mask":
                a_vals = gc.fft_mask_response(
                    np.array([1.0 / 300.0, 1.0 / 450.0]), spec.f_min_hz, spec.f_max_hz
                )
            else:
                a_vals = gc.butterworth_zero_phase_response(
                    np.array([1.0 / 300.0, 1.0 / 450.0]),
                    spec.f_min_hz,
                    spec.f_max_hz,
                    fs_hz=float(m) / gc.DT_1MIN_SEC,
                    order_n=order_n,
                )
            assert math.isclose(float(a_vals[0]), mode_row["A_300"], abs_tol=1e-5)
            assert math.isclose(float(a_vals[1]), mode_row["A_450"], abs_tol=1e-5)

            res = gc.run_variant_b_pipeline(
                sig_d,
                band="Pc5",
                filter_method=method,
                order_n=order_n,
                oversample_m=m,
                trim_hours=trim_h,
            )
            assert res.is_valid
            st = gc.stokes_from_spectral_matrix_2d(res.S)
            assert math.isclose(st.p, mode_row["p"], abs_tol=tol["ratio"])
            assert math.isclose(st.p_C, mode_row["V_over_I"], abs_tol=tol["ratio"])
            assert abs(st.chi_deg - mode_row["chi_deg"]) <= tol["angle_deg"]
            assert abs(st.psi_deg - mode_row["psi_deg"]) <= tol["psi_deg"]


def test_case_e_all_table_21_modes():
    fixtures = _load_fixtures()
    t = _time_grid_1day()
    sig_e = _build_case_signal("E", t, fixtures)
    t21 = fixtures["table_21_case_e"]
    trim_h = t21["processing"]["trim_hours"]
    m = t21["processing"]["oversample_m"]
    tol = t21["tolerances"]
    trim_os = int(round(trim_h * 3600.0 / (gc.DT_1MIN_SEC / m)))

    for mode_row in t21["modes"]:
        method = mode_row["method"]
        if method == "none":
            # Unfiltered periodic mixture: direct Formula (44) without odd-reflection wings
            sig_os = gc.oversample_fourier(sig_e, m=m)
            z = gc.analytic_signal_marple(sig_os, reflect_samples=0)
            s_mat = gc.spectral_matrix_from_analytic(z[trim_os:-trim_os])
        else:
            res = gc.run_variant_b_pipeline(
                sig_e,
                band=mode_row["band"],
                filter_method=method,
                order_n=3 if mode_row["order"] is None else int(mode_row["order"]),
                oversample_m=m,
                trim_hours=trim_h,
            )
            assert res.is_valid
            s_mat = res.S

        st = gc.stokes_from_spectral_matrix_2d(s_mat)
        assert math.isclose(st.p, mode_row["p"], abs_tol=tol["ratio"])
        assert math.isclose(st.p_C, mode_row["V_over_I"], abs_tol=tol["ratio"])
        assert abs(st.chi_deg - mode_row["chi_deg"]) <= tol["angle_deg"]
        diff_psi = abs((st.psi_deg - mode_row["psi_deg"] + 90.0) % 180.0 - 90.0)
        assert diff_psi <= tol["angle_deg"]


def test_table_5_critical_thresholds():
    fixtures = _load_fixtures()
    for row in fixtures["table_5_critical_thresholds"]:
        n_eff = float(row["n"])
        p95, g95 = gc.critical_thresholds(n_eff, alpha=0.05)
        p99, g99 = gc.critical_thresholds(n_eff, alpha=0.01)
        assert abs(p95 - row["p_crit_95"]) <= 0.0006
        assert abs(p99 - row["p_crit_99"]) <= 0.0006
        assert abs(g95 - row["gamma2_crit_95"]) <= 0.0006
        assert abs(g99 - row["gamma2_crit_99"]) <= 0.0006


def test_formula_45_vs_46_real_form_equivalence():
    rng = np.random.default_rng(123)
    x = rng.standard_normal((1080, 3))
    z = gc.analytic_signal_marple(x, reflect_samples=360)
    assert float(np.max(np.abs(np.real(z) - x))) < 1e-13

    w = np.hanning(1080)
    for weights in (None, w):
        s_complex = gc.spectral_matrix_from_analytic(z, weights=weights)
        s_real = gc.spectral_matrix_real_form(np.real(z), np.imag(z), weights=weights)
        assert float(np.max(np.abs(s_complex - s_real))) < 1e-14


def test_fft_mask_power_complementarity():
    freqs = np.linspace(1e-6, 1.0 / 120.0, 4000)
    f_pc5 = gc.get_band_spec("Pc5")
    f_pc6 = gc.get_band_spec("Pc6")
    w5 = gc.fft_mask_response(freqs, f_pc5.f_min_hz, f_pc5.f_max_hz)
    w6 = gc.fft_mask_response(freqs, f_pc6.f_min_hz, f_pc6.f_max_hz)

    # Inside the shared transition around 600 s (480..800 s), W5^2 + W6^2 == 1 to machine precision
    shared = (freqs >= 1.0 / 1400.0) & (freqs <= 1.0 / 220.0)
    assert float(np.max(np.abs(w5[shared] ** 2 + w6[shared] ** 2 - 1.0))) < 1e-15


def test_oversampling_exactness_and_kaiser_rmse():
    # Node identity y[m*n] == x[n] for both even and odd N
    for n_len in (120, 121):
        t = np.arange(n_len, dtype=np.float64) * 60.0
        sig = np.sin(2.0 * np.pi * t / 300.0) + 0.5 * np.cos(2.0 * np.pi * t / 450.0)
        y_f = gc.oversample_fourier(sig, m=4, pad_samples=60)
        assert y_f.size == 4 * n_len
        assert float(np.max(np.abs(y_f[::4] - sig))) < 1e-14

        y_k = gc.oversample_kaiser(sig, m=4, Q=16)
        assert y_k.size == 4 * (n_len - 1) + 1
        assert float(np.max(np.abs(y_k[::4] - sig))) < 1e-14

    # Interior accuracy of Kaiser Q=16 vs exact analytical Pc5 wave
    t_orig = np.arange(720, dtype=np.float64) * 60.0
    t_fine = np.arange(4 * 719 + 1, dtype=np.float64) * 15.0
    x_orig = np.sin(2.0 * np.pi * t_orig / 240.0) + 0.4 * np.cos(2.0 * np.pi * t_orig / 180.0)
    x_true = np.sin(2.0 * np.pi * t_fine / 240.0) + 0.4 * np.cos(2.0 * np.pi * t_fine / 180.0)
    y_kaiser = gc.oversample_kaiser(x_orig, m=4, Q=16)
    rel_rmse = float(
        np.linalg.norm(y_kaiser[240:-240] - x_true[240:-240]) / np.linalg.norm(x_true[240:-240])
    )
    assert rel_rmse < 2e-4


def test_table_17_formula48_and_m_invariance():
    fixtures = _load_fixtures()
    t17 = fixtures["table_17_filtered_effective_n"]

    for key, entries in t17.items():
        band, hrs_str = key.split("_")
        window_sec = float(hrs_str.rstrip("h")) * 3600.0

        for filt_key, (mc_ref, f48_ref) in entries.items():
            method = "fft_mask" if filt_key.startswith("fft") else "sos"
            spec_type = "flat" if filt_key.endswith("flat") else "red"
            n_eff, _ = gc.effective_n_formula48(
                window_sec,
                band=band,
                filter_method=method,
                dt_orig=60.0,
                oversample_m=1,
                order_n=3,
                spectrum_type=spec_type,
            )
            assert abs(n_eff - float(f48_ref)) <= 1.0
            assert abs(n_eff - float(mc_ref)) / float(mc_ref) <= 0.045

    # Strict invariance of FFT-mask n_eff across m=1, 4, 8
    n_m1, _ = gc.effective_n_formula48(
        18 * 3600.0, band="Pc5", filter_method="fft_mask", oversample_m=1
    )
    n_m4, _ = gc.effective_n_formula48(
        18 * 3600.0, band="Pc5", filter_method="fft_mask", oversample_m=4
    )
    n_m8, _ = gc.effective_n_formula48(
        18 * 3600.0, band="Pc5", filter_method="fft_mask", oversample_m=8
    )
    assert abs(n_m1 - n_m4) < 1e-12 and abs(n_m1 - n_m8) < 1e-12


def test_mc_calibration_effective_n_table3_and_table17():
    fixtures = _load_fixtures()
    t3_windows = {w["n_samples"]: w for w in fixtures["table_3_welch_effective_n"]["windows"]}

    # Verify Table 3 Welch effective n within +-5% for flat and red (random-walk) spectra
    for n_samples in (720, 1440):
        ref_flat = t3_windows[n_samples]["pc5_flat"]
        n_mc_flat = gc.calibrate_welch_effective_n_mc(
            n_samples, band="Pc5", spectrum_type="flat", n_trials=2000, seed=42
        )
        assert abs(n_mc_flat - ref_flat) / ref_flat <= 0.05

    for n_samples in (256, 1440):
        ref_red = t3_windows[n_samples]["pc5_red_f2"]
        n_mc_red = gc.calibrate_welch_effective_n_mc(
            n_samples, band="Pc5", spectrum_type="red", n_trials=2500, seed=43
        )
        assert abs(n_mc_red - ref_red) / ref_red <= 0.05

    n_pc6_flat = gc.calibrate_welch_effective_n_mc(
        1440, band="Pc6", spectrum_type="flat", n_trials=2000, seed=44
    )
    n_pc6_red = gc.calibrate_welch_effective_n_mc(
        1440, band="Pc6", spectrum_type="red", n_trials=2000, seed=45
    )
    assert abs(n_pc6_flat - t3_windows[1440]["pc6_flat"]) / t3_windows[1440]["pc6_flat"] <= 0.05
    assert abs(n_pc6_red - t3_windows[1440]["pc6_red_f2"]) / t3_windows[1440]["pc6_red_f2"] <= 0.05

    # Verify Variant B Formula (48) matches independent MC on 18h window within +-5%
    rng = np.random.default_rng(77)
    n_trials = 1500
    noise = rng.standard_normal((n_trials, 1440, 2))
    x2d = np.transpose(noise, (1, 0, 2)).reshape(1440, n_trials * 2)
    filt = gc.apply_fft_mask(x2d, dt_prime=60.0, band="Pc5")
    z = gc.analytic_signal_marple(filt, reflect_samples=360)[180:-180].reshape(1080, n_trials, 2)
    s_xx = np.mean(np.abs(z[..., 0]) ** 2, axis=0)
    s_yy = np.mean(np.abs(z[..., 1]) ** 2, axis=0)
    s_xy = np.mean(z[..., 0] * np.conj(z[..., 1]), axis=0)
    i_v = s_xx + s_yy
    p2 = ((s_xx - s_yy) ** 2 + 4.0 * np.abs(s_xy) ** 2) / (i_v ** 2)
    n_mc_var_b = 0.5 * (3.0 / float(np.mean(p2)) - 1.0)
    n_f48, _ = gc.effective_n_formula48(18 * 3600.0, band="Pc5", filter_method="fft_mask")
    assert abs(n_mc_var_b - n_f48) / n_f48 <= 0.05


def test_case_d_kaiser_oversampling_mode():
    fixtures = _load_fixtures()
    t = _time_grid_1day()
    sig_d = _build_case_signal("D", t, fixtures)
    t20 = fixtures["table_20_case_d"]
    mode_sos3_m4 = next(m for m in t20["modes"] if m["mode"] == "sos_n3_m4")

    res = gc.run_variant_b_pipeline(
        sig_d,
        band="Pc5",
        filter_method="sos",
        order_n=3,
        oversample_m=4,
        oversample_method="kaiser",
        trim_hours=t20["processing"]["trim_hours"],
    )
    assert res.is_valid
    assert res.filtered.shape[0] == 4 * (1440 - 1) + 1
    st = gc.stokes_from_spectral_matrix_2d(res.S)
    assert math.isclose(st.p, mode_sos3_m4["p"], abs_tol=0.002)
    assert math.isclose(st.p_C, mode_sos3_m4["V_over_I"], abs_tol=0.002)
    assert abs(st.chi_deg - mode_sos3_m4["chi_deg"]) <= 0.05
    assert abs(st.psi_deg - mode_sos3_m4["psi_deg"]) <= 0.05


def test_welch_effective_n_overlap_consistency():
    for m_step in (128, 64, 32, 16):
        k_seg = (1440 - 128) // m_step + 1
        n_lookup = gc.lookup_welch_effective_n(1440, band="Pc5", L=128, M=m_step)
        n_band = float(
            gc.welch_band_dof(k_seg, N_B=39, band="Pc5", spectrum_type="flat", L=128, M=m_step)
        )
        n_mc = gc.calibrate_welch_effective_n_mc(
            1440, band="Pc5", L=128, M=m_step, n_trials=800, seed=50 + m_step
        )
        assert abs(n_lookup - n_mc) / n_mc <= 0.15
        assert abs(n_band - n_mc) / n_mc <= 0.15
        assert n_lookup < 500.0 and n_band < 500.0
