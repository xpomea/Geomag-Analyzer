"""Polarization analysis REST router (/api/polarization) for Tier 2 calculations."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter

from geomag_core import (
    DT_1MIN_SEC,
    Polarization3D,
    Stokes2D,
    aggregate_spectral_matrices,
    compute_multitaper_spectral_matrix,
    compute_station_phase1_spectra,
    compute_welch_spectral_matrix,
    effective_n_formula48,
    find_valid_subsegments,
    is_window_valid,
    periodic_hann_window,
    polarization_3d,
    run_variant_b_pipeline,
    spectral_matrix_from_analytic,
    stokes_from_spectral_matrix_2d,
    welch_effective_dof,
)

from .deps import StoreDep
from .schemas import (
    DynamicPolarizationOut,
    FilteredSeriesOut,
    NetworkSnapshotRequest,
    NetworkSnapshotResponse,
    NetworkStationSnapshot,
    Polarization3DOut,
    PolarizationRequest,
    PolarizationResponse,
    SpectrogramOut,
    Stokes2DOut,
)
from .serializers import timestamps_to_iso, to_json_float, to_json_list

router = APIRouter(prefix="/api/polarization", tags=["polarization"])


def _stokes_to_schema(st: Stokes2D) -> Stokes2DOut:
    return Stokes2DOut(
        I=to_json_float(st.I),
        Q=to_json_float(st.Q),
        U=to_json_float(st.U),
        V=to_json_float(st.V),
        A=to_json_float(st.A),
        R=to_json_float(st.R),
        p=to_json_float(st.p),
        p_L=to_json_float(st.p_L),
        p_C=to_json_float(st.p_C),
        psi_deg=to_json_float(st.psi_deg),
        chi_deg=to_json_float(st.chi_deg),
        epsilon=to_json_float(st.epsilon),
        gamma2=to_json_float(st.gamma2),
        arg_sxy_deg=to_json_float(st.arg_Sxy_deg),
        n_eff=to_json_float(st.n_eff),
        p_crit=to_json_float(st.p_crit),
        gamma2_crit=to_json_float(st.gamma2_crit),
        is_significant_p=st.is_significant_p,
        is_significant_gamma2=st.is_significant_gamma2,
    )


def _pol3_to_schema(p3: Polarization3D) -> Polarization3DOut:
    return Polarization3DOut(
        P=to_json_float(p3.P),
        theta_deg=to_json_float(p3.theta_deg),
        phi_n_deg=to_json_float(p3.phi_n_deg),
        b_over_a=to_json_float(p3.b_over_a),
        s=to_json_float(p3.s),
        tr_S=to_json_float(p3.tr_S),
        n_v=to_json_list(p3.n_v),
        eigenvalues_re_S=to_json_list(p3.eigenvalues_re_S),
    )


def _empty_dynamic_out() -> DynamicPolarizationOut:
    return DynamicPolarizationOut(
        timestamps_utc=[],
        I=[],
        p=[],
        p_crit=[],
        psi_deg=[],
        chi_deg=[],
        epsilon=[],
        gamma2=[],
        gamma2_crit=[],
        n_eff=[],
        is_significant_p=[],
        is_valid=[],
    )


def _compute_station_spectrogram(
    xyz: np.ndarray,
    timestamps_utc,
    is_valid_xyz: np.ndarray,
    *,
    station_iaga: str,
    L: int,
    M: int,
    window_samples: int,
    step_samples: int,
) -> SpectrogramOut:
    """Compute calibrated one-sided dynamic Welch PSD spectrogram Sxx(t,f), Syy(t,f)."""
    n_samples = xyz.shape[0]
    if n_samples < window_samples or window_samples < L:
        return SpectrogramOut(
            timestamps_utc=[],
            freqs_hz=[],
            power_total=[],
            power_x=[],
            power_y=[],
        )

    sp_x = compute_station_phase1_spectra(
        xyz[:, 0],
        station_iaga=station_iaga,
        component="X",
        L=L,
        M=M,
        window_samples=window_samples,
        step_samples=step_samples,
        is_valid=is_valid_xyz,
    )
    sp_y = compute_station_phase1_spectra(
        xyz[:, 1],
        station_iaga=station_iaga,
        component="Y",
        L=L,
        M=M,
        window_samples=window_samples,
        step_samples=step_samples,
        is_valid=is_valid_xyz,
    )
    freqs = sp_x.freqs_hz
    win = periodic_hann_window(L)
    u_norm = float(np.sum(win ** 2))
    scale = np.full(freqs.shape, 2.0 * DT_1MIN_SEC / u_norm, dtype=np.float64)
    scale[0] = DT_1MIN_SEC / u_norm
    if L % 2 == 0:
        scale[-1] = DT_1MIN_SEC / u_norm

    f_mask = freqs > 0.0
    pos_freqs = freqs[f_mask]
    pos_scale = scale[f_mask]

    valid = sp_x.valid_segments & sp_y.valid_segments
    k_valid = np.sum(valid, axis=1, dtype=np.int64)
    fx = np.where(valid[:, :, None], np.abs(sp_x.F[:, :, f_mask]) ** 2, 0.0)
    fy = np.where(valid[:, :, None], np.abs(sp_y.F[:, :, f_mask]) ** 2, 0.0)
    sum_x = np.sum(fx, axis=1)
    sum_y = np.sum(fy, axis=1)

    px = np.full(sum_x.shape, np.nan, dtype=np.float64)
    py = np.full(sum_y.shape, np.nan, dtype=np.float64)
    ok = k_valid > 0
    px[ok] = (sum_x[ok] / k_valid[ok, None]) * pos_scale[None, :]
    py[ok] = (sum_y[ok] / k_valid[ok, None]) * pos_scale[None, :]
    ptot = px + py

    return SpectrogramOut(
        timestamps_utc=timestamps_to_iso(timestamps_utc[sp_x.window_center_indices]),
        freqs_hz=to_json_list(pos_freqs),
        power_total=to_json_list(ptot),
        power_x=to_json_list(px),
        power_y=to_json_list(py),
    )


@router.post("", response_model=PolarizationResponse)
def compute_polarization(
    req: PolarizationRequest,
    store: StoreDep,
) -> PolarizationResponse:
    """Execute Tier 2 Variant A (Welch/Multitaper) or Variant B polarization analysis."""
    data = store.load_station_episode(
        req.episode_id,
        req.station_iaga,
        start_utc=req.start_utc,
        end_utc=req.end_utc,
    )
    xyz, d0_deg, canon_frame = store.apply_coord_frame(
        data,
        coord_frame=req.coord_frame,
        external_d0_deg=req.external_d0_deg,
    )
    warnings = list(data.warnings)

    if not data.available:
        return PolarizationResponse(
            episode_id=req.episode_id,
            station_iaga=data.station_iaga,
            method=req.method,
            band=req.band,
            coord_frame=canon_frame,
            d0_deg=float(d0_deg),
            is_valid=False,
            warnings=warnings,
        )

    n_samples = xyz.shape[0]
    ts = data.timestamps_utc

    if req.method == "welch":
        if req.include_dynamic and req.window_samples < req.L:
            raise ValueError(
                f"window_samples ({req.window_samples}) must be >= L ({req.L}) for Welch."
            )
        overall_res = compute_welch_spectral_matrix(
            xyz,
            band=req.band,
            L=req.L,
            M=req.M,
            t_max_pc6_sec=req.t_max_pc6_sec,
            f_min_hz=req.f_min_hz,
            f_max_hz=req.f_max_hz,
            is_valid_xyz=data.is_valid_xyz,
        )
        n_eff_overall: float | None = None
        stokes_out: Stokes2DOut | None = None
        pol3_out: Polarization3DOut | None = None
        is_valid = bool(overall_res.is_valid and np.all(np.isfinite(overall_res.S)))
        if is_valid:
            n_eff_raw = float(
                welch_effective_dof(
                    overall_res.K_valid,
                    L=req.L,
                    M=req.M,
                    band=req.band,
                    spectrum_type=req.spectrum_type,
                    t_max_pc6_sec=req.t_max_pc6_sec,
                    f_min_hz=req.f_min_hz,
                    f_max_hz=req.f_max_hz,
                ).item()
            )
            n_eff_arg = n_eff_raw if n_eff_raw > 1.0 else None
            n_eff_overall = n_eff_raw
            stokes_out = _stokes_to_schema(
                stokes_from_spectral_matrix_2d(
                    overall_res.S, n_eff=n_eff_arg, alpha=req.alpha
                )
            )
            if req.include_3d:
                pol3_out = _pol3_to_schema(polarization_3d(overall_res.S))
        else:
            warnings.append("Insufficient valid Welch segments for overall station estimate.")

        dyn_out: DynamicPolarizationOut | None = None
        if req.include_dynamic:
            if n_samples < req.window_samples:
                dyn_out = _empty_dynamic_out()
            else:
                starts = list(range(0, n_samples - req.window_samples + 1, req.step_samples))
                win_ts = timestamps_to_iso(ts[[s + req.window_samples // 2 for s in starts]])
                i_vals, p_vals, pcrit_vals, psi_vals, chi_vals, eps_vals, g2_vals = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                g2crit_vals, neff_vals = [], []
                sig_flags, valid_flags = [], []
                for s in starts:
                    e = s + req.window_samples
                    w_sub = compute_welch_spectral_matrix(
                        xyz[s:e],
                        band=req.band,
                        L=req.L,
                        M=req.M,
                        t_max_pc6_sec=req.t_max_pc6_sec,
                        f_min_hz=req.f_min_hz,
                        f_max_hz=req.f_max_hz,
                        is_valid_xyz=data.is_valid_xyz[s:e],
                    )
                    if w_sub.is_valid and np.all(np.isfinite(w_sub.S[:2, :2])):
                        nw_raw = float(
                            welch_effective_dof(
                                w_sub.K_valid,
                                L=req.L,
                                M=req.M,
                                band=req.band,
                                spectrum_type=req.spectrum_type,
                                t_max_pc6_sec=req.t_max_pc6_sec,
                                f_min_hz=req.f_min_hz,
                                f_max_hz=req.f_max_hz,
                            ).item()
                        )
                        st_w = stokes_from_spectral_matrix_2d(
                            w_sub.S,
                            n_eff=nw_raw if nw_raw > 1.0 else None,
                            alpha=req.alpha,
                        )
                        i_vals.append(to_json_float(st_w.I))
                        p_vals.append(to_json_float(st_w.p))
                        pcrit_vals.append(to_json_float(st_w.p_crit))
                        psi_vals.append(to_json_float(st_w.psi_deg))
                        chi_vals.append(to_json_float(st_w.chi_deg))
                        eps_vals.append(to_json_float(st_w.epsilon))
                        g2_vals.append(to_json_float(st_w.gamma2))
                        g2crit_vals.append(to_json_float(st_w.gamma2_crit))
                        neff_vals.append(to_json_float(st_w.n_eff))
                        sig_flags.append(bool(st_w.is_significant_p))
                        valid_flags.append(True)
                    else:
                        i_vals.append(None)
                        p_vals.append(None)
                        pcrit_vals.append(None)
                        psi_vals.append(None)
                        chi_vals.append(None)
                        eps_vals.append(None)
                        g2_vals.append(None)
                        g2crit_vals.append(None)
                        neff_vals.append(None)
                        sig_flags.append(False)
                        valid_flags.append(False)
                dyn_out = DynamicPolarizationOut(
                    timestamps_utc=win_ts,
                    I=i_vals,
                    p=p_vals,
                    p_crit=pcrit_vals,
                    psi_deg=psi_vals,
                    chi_deg=chi_vals,
                    epsilon=eps_vals,
                    gamma2=g2_vals,
                    gamma2_crit=g2crit_vals,
                    n_eff=neff_vals,
                    is_significant_p=sig_flags,
                    is_valid=valid_flags,
                )

        spec_out = (
            _compute_station_spectrogram(
                xyz,
                ts,
                data.is_valid_xyz,
                station_iaga=data.station_iaga,
                L=req.L,
                M=req.M,
                window_samples=req.window_samples,
                step_samples=req.step_samples,
            )
            if req.include_spectrogram
            else None
        )

        return PolarizationResponse(
            episode_id=req.episode_id,
            station_iaga=data.station_iaga,
            method=req.method,
            band=req.band,
            coord_frame=canon_frame,
            d0_deg=float(d0_deg),
            is_valid=is_valid,
            n_eff=to_json_float(n_eff_overall),
            stokes_2d=stokes_out,
            pol_3d=pol3_out,
            dynamic=dyn_out,
            spectrogram=spec_out,
            warnings=warnings,
        )

    if req.method == "multitaper":
        finite_rows = np.all(np.isfinite(xyz), axis=1)
        min_seg = max(64, req.window_samples if req.include_dynamic else 64)
        subsegs = [
            (s, e)
            for s, e in find_valid_subsegments(finite_rows, min_length=min_seg)
            if is_window_valid(data.is_valid_xyz[s:e], finite_rows[s:e])
        ]
        if not subsegs:
            warnings.append("No contiguous QC-valid subsegments available for Multitaper.")
            return PolarizationResponse(
                episode_id=req.episode_id,
                station_iaga=data.station_iaga,
                method=req.method,
                band=req.band,
                coord_frame=canon_frame,
                d0_deg=float(d0_deg),
                is_valid=False,
                dynamic=_empty_dynamic_out() if req.include_dynamic else None,
                warnings=warnings,
            )

        s_mats, w_lens, dof_sum = [], [], 0.0
        for s_idx, e_idx in subsegs:
            mt = compute_multitaper_spectral_matrix(
                xyz[s_idx:e_idx],
                band=req.band,
                NW=req.nw,
                K_tapers=req.k_tapers,
                t_max_pc6_sec=req.t_max_pc6_sec,
                f_min_hz=req.f_min_hz,
                f_max_hz=req.f_max_hz,
            )
            s_mats.append(mt.S)
            w_lens.append(float(e_idx - s_idx))
            dof_sum += float(mt.K_tapers) * max(1.0, float(mt.N_B) / max(1.0, 2.0 * req.nw))

        s_overall = aggregate_spectral_matrices(s_mats, weights=w_lens)
        n_eff_mt = max(2.0, dof_sum)
        stokes_out = _stokes_to_schema(
            stokes_from_spectral_matrix_2d(s_overall, n_eff=n_eff_mt, alpha=req.alpha)
        )
        pol3_out = _pol3_to_schema(polarization_3d(s_overall)) if req.include_3d else None

        dyn_out = None
        if req.include_dynamic:
            if n_samples < req.window_samples:
                dyn_out = _empty_dynamic_out()
            else:
                starts = list(range(0, n_samples - req.window_samples + 1, req.step_samples))
                win_ts = timestamps_to_iso(ts[[s + req.window_samples // 2 for s in starts]])
                i_vals, p_vals, pcrit_vals, psi_vals, chi_vals, eps_vals, g2_vals = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                g2crit_vals, neff_vals = [], []
                sig_flags, valid_flags = [], []
                for s in starts:
                    e = s + req.window_samples
                    seg_ok = bool(
                        np.all(finite_rows[s:e])
                        and is_window_valid(data.is_valid_xyz[s:e], finite_rows[s:e])
                    )
                    if seg_ok:
                        mt_w = compute_multitaper_spectral_matrix(
                            xyz[s:e],
                            band=req.band,
                            NW=req.nw,
                            K_tapers=req.k_tapers,
                            t_max_pc6_sec=req.t_max_pc6_sec,
                            f_min_hz=req.f_min_hz,
                            f_max_hz=req.f_max_hz,
                        )
                        nw_mt = max(
                            2.0,
                            float(mt_w.K_tapers)
                            * max(1.0, float(mt_w.N_B) / max(1.0, 2.0 * req.nw)),
                        )
                        st_w = stokes_from_spectral_matrix_2d(
                            mt_w.S, n_eff=nw_mt, alpha=req.alpha
                        )
                        i_vals.append(to_json_float(st_w.I))
                        p_vals.append(to_json_float(st_w.p))
                        pcrit_vals.append(to_json_float(st_w.p_crit))
                        psi_vals.append(to_json_float(st_w.psi_deg))
                        chi_vals.append(to_json_float(st_w.chi_deg))
                        eps_vals.append(to_json_float(st_w.epsilon))
                        g2_vals.append(to_json_float(st_w.gamma2))
                        g2crit_vals.append(to_json_float(st_w.gamma2_crit))
                        neff_vals.append(to_json_float(st_w.n_eff))
                        sig_flags.append(bool(st_w.is_significant_p))
                        valid_flags.append(True)
                    else:
                        i_vals.append(None)
                        p_vals.append(None)
                        pcrit_vals.append(None)
                        psi_vals.append(None)
                        chi_vals.append(None)
                        eps_vals.append(None)
                        g2_vals.append(None)
                        g2crit_vals.append(None)
                        neff_vals.append(None)
                        sig_flags.append(False)
                        valid_flags.append(False)
                dyn_out = DynamicPolarizationOut(
                    timestamps_utc=win_ts,
                    I=i_vals,
                    p=p_vals,
                    p_crit=pcrit_vals,
                    psi_deg=psi_vals,
                    chi_deg=chi_vals,
                    epsilon=eps_vals,
                    gamma2=g2_vals,
                    gamma2_crit=g2crit_vals,
                    n_eff=neff_vals,
                    is_significant_p=sig_flags,
                    is_valid=valid_flags,
                )

        spec_out = (
            _compute_station_spectrogram(
                xyz,
                ts,
                data.is_valid_xyz,
                station_iaga=data.station_iaga,
                L=req.L,
                M=req.M,
                window_samples=req.window_samples,
                step_samples=req.step_samples,
            )
            if req.include_spectrogram
            else None
        )

        return PolarizationResponse(
            episode_id=req.episode_id,
            station_iaga=data.station_iaga,
            method=req.method,
            band=req.band,
            coord_frame=canon_frame,
            d0_deg=float(d0_deg),
            is_valid=True,
            n_eff=to_json_float(n_eff_mt),
            stokes_2d=stokes_out,
            pol_3d=pol3_out,
            dynamic=dyn_out,
            spectrogram=spec_out,
            warnings=warnings,
        )

    # Variant B (FFT-mask or Butterworth SOS + Marple analytic signal)
    filter_method = "fft_mask" if req.method == "variant_b_fft" else "sos"
    vb = run_variant_b_pipeline(
        xyz,
        band=req.band,
        filter_method=filter_method,
        order_n=req.order_n,
        oversample_m=req.oversample_m,
        oversample_method=req.oversample_method,
        trim_hours=req.trim_hours,
        spectrum_type=req.spectrum_type,
        t_max_pc6_sec=req.t_max_pc6_sec,
        f1_hz=req.f_min_hz,
        f2_hz=req.f_max_hz,
        is_valid_xyz=data.is_valid_xyz,
    )
    if not vb.is_valid or not np.all(np.isfinite(vb.S[:2, :2])):
        warnings.append(
            f"No valid continuous subsegment survived 2*T_cut ({2 * vb.trim_hours:.1f} h) "
            "+ min_window edge trim."
        )
        return PolarizationResponse(
            episode_id=req.episode_id,
            station_iaga=data.station_iaga,
            method=req.method,
            band=req.band,
            coord_frame=canon_frame,
            d0_deg=float(d0_deg),
            is_valid=False,
            trim_hours=float(vb.trim_hours),
            dynamic=_empty_dynamic_out() if req.include_dynamic else None,
            warnings=warnings,
        )

    n_eff_b = float(vb.n_eff) if vb.n_eff > 1.0 else None
    stokes_out = _stokes_to_schema(
        stokes_from_spectral_matrix_2d(vb.S, n_eff=n_eff_b, alpha=req.alpha)
    )
    pol3_out = _pol3_to_schema(polarization_3d(vb.S)) if req.include_3d else None

    dyn_out = None
    if req.include_dynamic:
        if n_samples < req.window_samples:
            dyn_out = _empty_dynamic_out()
        else:
            starts = list(range(0, n_samples - req.window_samples + 1, req.step_samples))
            win_ts = timestamps_to_iso(ts[[s + req.window_samples // 2 for s in starts]])
            w_len_os = int(round(req.window_samples * 60.0 / vb.dt_prime))
            n_w_vb, _ = effective_n_formula48(
                float(req.window_samples) * 60.0,
                band=req.band,
                filter_method=filter_method,
                dt_orig=60.0,
                oversample_m=req.oversample_m,
                order_n=req.order_n,
                spectrum_type=req.spectrum_type,
                t_max_pc6_sec=req.t_max_pc6_sec,
                f1_hz=req.f_min_hz,
                f2_hz=req.f_max_hz,
            )
            n_w_arg = n_w_vb if n_w_vb > 1.0 else None
            i_vals, p_vals, pcrit_vals, psi_vals, chi_vals, eps_vals, g2_vals = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
            g2crit_vals, neff_vals = [], []
            sig_flags, valid_flags = [], []
            for s in starts:
                os_s = s * vb.oversample_m
                os_e = os_s + w_len_os
                ok = bool(
                    os_e <= vb.usable_mask.shape[0]
                    and np.all(vb.usable_mask[os_s:os_e])
                    and is_window_valid(data.is_valid_xyz[s : s + req.window_samples])
                )
                if ok:
                    s_w = spectral_matrix_from_analytic(vb.analytic[os_s:os_e])
                    st_w = stokes_from_spectral_matrix_2d(
                        s_w, n_eff=n_w_arg, alpha=req.alpha
                    )
                    i_vals.append(to_json_float(st_w.I))
                    p_vals.append(to_json_float(st_w.p))
                    pcrit_vals.append(to_json_float(st_w.p_crit))
                    psi_vals.append(to_json_float(st_w.psi_deg))
                    chi_vals.append(to_json_float(st_w.chi_deg))
                    eps_vals.append(to_json_float(st_w.epsilon))
                    g2_vals.append(to_json_float(st_w.gamma2))
                    g2crit_vals.append(to_json_float(st_w.gamma2_crit))
                    neff_vals.append(to_json_float(st_w.n_eff))
                    sig_flags.append(bool(st_w.is_significant_p))
                    valid_flags.append(True)
                else:
                    i_vals.append(None)
                    p_vals.append(None)
                    pcrit_vals.append(None)
                    psi_vals.append(None)
                    chi_vals.append(None)
                    eps_vals.append(None)
                    g2_vals.append(None)
                    g2crit_vals.append(None)
                    neff_vals.append(None)
                    sig_flags.append(False)
                    valid_flags.append(False)
            dyn_out = DynamicPolarizationOut(
                timestamps_utc=win_ts,
                I=i_vals,
                p=p_vals,
                p_crit=pcrit_vals,
                psi_deg=psi_vals,
                chi_deg=chi_vals,
                epsilon=eps_vals,
                gamma2=g2_vals,
                gamma2_crit=g2crit_vals,
                n_eff=neff_vals,
                is_significant_p=sig_flags,
                is_valid=valid_flags,
            )

    filtered_out = None
    if req.include_filtered:
        stride = max(1, vb.oversample_m)
        f_1m = vb.filtered[::stride][:n_samples]
        a_1m = vb.analytic[::stride][:n_samples]
        m_1m = vb.usable_mask[::stride][:n_samples]
        filtered_out = FilteredSeriesOut(
            timestamps_utc=timestamps_to_iso(ts),
            x_filt=to_json_list(f_1m[:, 0]),
            y_filt=to_json_list(f_1m[:, 1]),
            z_filt=to_json_list(f_1m[:, 2]),
            x_amp=to_json_list(np.abs(a_1m[:, 0])),
            y_amp=to_json_list(np.abs(a_1m[:, 1])),
            usable_mask=to_json_list(m_1m),
        )

    spec_out = (
        _compute_station_spectrogram(
            xyz,
            ts,
            data.is_valid_xyz,
            station_iaga=data.station_iaga,
            L=req.L,
            M=req.M,
            window_samples=req.window_samples,
            step_samples=req.step_samples,
        )
        if req.include_spectrogram
        else None
    )

    return PolarizationResponse(
        episode_id=req.episode_id,
        station_iaga=data.station_iaga,
        method=req.method,
        band=req.band,
        coord_frame=canon_frame,
        d0_deg=float(d0_deg),
        is_valid=True,
        n_eff=to_json_float(vb.n_eff),
        trim_hours=float(vb.trim_hours),
        stokes_2d=stokes_out,
        pol_3d=pol3_out,
        dynamic=dyn_out,
        filtered=filtered_out,
        spectrogram=spec_out,
        warnings=warnings,
    )


@router.post("/network-snapshot", response_model=NetworkSnapshotResponse)
def compute_network_snapshot(
    req: NetworkSnapshotRequest,
    store: StoreDep,
) -> NetworkSnapshotResponse:
    """Compute Welch 2D Stokes polarization across all magnetometers for map ellipses."""
    store.require_episode(req.episode_id)
    if req.station_iagas is None:
        mags = store.get_stations(station_type="magnetometer")
        codes = [st["iaga_code"] for st in mags]
    else:
        codes = [store.require_magnetometer(c)["iaga_code"] for c in req.station_iagas]

    items: list[NetworkStationSnapshot] = []
    for code in codes:
        data = store.load_station_episode(
            req.episode_id,
            code,
            start_utc=req.start_utc,
            end_utc=req.end_utc,
        )
        meta = data.station_meta
        if not data.available:
            items.append(
                NetworkStationSnapshot(
                    station_iaga=code,
                    lat_deg=float(meta["lat_deg"]),
                    lon_deg=float(meta["lon_deg"]),
                    has_baseline=bool(meta["has_baseline"]),
                    available=False,
                    is_valid=False,
                    warning=data.warnings[0] if data.warnings else "No data in episode.",
                )
            )
            continue

        try:
            xyz, _, canon_frame = store.apply_coord_frame(
                data, coord_frame=req.coord_frame
            )
        except ValueError as exc:
            items.append(
                NetworkStationSnapshot(
                    station_iaga=code,
                    lat_deg=float(meta["lat_deg"]),
                    lon_deg=float(meta["lon_deg"]),
                    has_baseline=bool(meta["has_baseline"]),
                    available=True,
                    is_valid=False,
                    warning=str(exc),
                )
            )
            continue

        w_res = compute_welch_spectral_matrix(
            xyz[:, :2],
            band=req.band,
            L=req.L,
            M=req.M,
            t_max_pc6_sec=req.t_max_pc6_sec,
            is_valid_xyz=data.is_valid_xyz,
        )
        if not w_res.is_valid or not np.all(np.isfinite(w_res.S)):
            items.append(
                NetworkStationSnapshot(
                    station_iaga=code,
                    lat_deg=float(meta["lat_deg"]),
                    lon_deg=float(meta["lon_deg"]),
                    has_baseline=bool(meta["has_baseline"]),
                    available=True,
                    is_valid=False,
                    warning="Insufficient valid Welch segments in selected window.",
                )
            )
            continue

        n_eff_raw = float(
            welch_effective_dof(
                w_res.K_valid,
                L=req.L,
                M=req.M,
                band=req.band,
                spectrum_type=req.spectrum_type,
                t_max_pc6_sec=req.t_max_pc6_sec,
            ).item()
        )
        st = stokes_from_spectral_matrix_2d(
            w_res.S,
            n_eff=n_eff_raw if n_eff_raw > 1.0 else None,
            alpha=req.alpha,
        )
        items.append(
            NetworkStationSnapshot(
                station_iaga=code,
                lat_deg=float(meta["lat_deg"]),
                lon_deg=float(meta["lon_deg"]),
                has_baseline=bool(meta["has_baseline"]),
                available=True,
                is_valid=True,
                stokes_2d=_stokes_to_schema(st),
            )
        )

    return NetworkSnapshotResponse(
        episode_id=req.episode_id,
        band=req.band,
        coord_frame="XmYm" if req.coord_frame.upper() == "XMYM" else "XYZ",
        stations=items,
    )
