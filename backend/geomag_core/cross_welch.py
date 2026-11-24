"""Two-phase Welch cross-spectral engine, azimuthal wave number m, and FLR profiles (§6.1)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .conventions import DT_1MIN_SEC, canonical_cache_key, is_window_valid
from .spectral import (
    band_frequency_mask,
    detrend_linear_ols,
    periodic_hann_window,
    welch_effective_dof,
    welch_segment_dof,
)


@dataclass(frozen=True)
class Phase1StationSpectra:
    """Cached Phase 1 per-station windowed segment FFTs and §3.2 segment validity masks."""

    station_iaga: str
    component: str
    dt: float
    L: int
    M: int
    window_samples: int
    step_samples: int
    window_center_indices: np.ndarray  # (T_w,) int64
    freqs_hz: np.ndarray               # (N_f,) float64
    F: np.ndarray                      # (T_w, K, N_f) complex
    valid_segments: np.ndarray         # (T_w, K) bool
    cache_key: str


@dataclass(frozen=True)
class WelchPairResult:
    """Phase 2 pairwise Welch cross-spectrum computed strictly over joint valid segments."""

    station_i: str
    station_j: str
    band: str
    window_center_indices: np.ndarray  # (T_w,)
    freqs_hz: np.ndarray               # (N_f_band,)
    S_ij: np.ndarray                   # (T_w,) complex128
    S_ii: np.ndarray                   # (T_w,) float64
    S_jj: np.ndarray                   # (T_w,) float64
    K_ij: np.ndarray                   # (T_w,) int64
    n_eff: np.ndarray                  # (T_w,) float64
    gamma2: np.ndarray                 # (T_w,) float64
    gamma2_crit: np.ndarray            # (T_w,) float64
    phase_rad: np.ndarray              # (T_w,) float64
    phase_deg: np.ndarray              # (T_w,) float64
    phase_deg_masked: np.ndarray       # (T_w,) float64
    is_significant: np.ndarray         # (T_w,) bool
    is_valid: np.ndarray               # (T_w,) bool
    gamma2_tf: np.ndarray              # (T_w, N_f_band) float64
    gamma2_crit_tf: np.ndarray         # (T_w,) float64
    phase_deg_tf: np.ndarray           # (T_w, N_f_band) float64
    phase_deg_tf_masked: np.ndarray    # (T_w, N_f_band) float64
    S_ij_tf: np.ndarray                # (T_w, N_f_band) complex128


@dataclass(frozen=True)
class WelchMatrixResult:
    """Dynamic S x S all-pairs Welch coherence and phase matrices over joint masks."""

    station_codes: tuple[str, ...]
    band: str
    window_center_indices: np.ndarray  # (T_w,)
    gamma2: np.ndarray                 # (T_w, S, S) float64
    gamma2_crit: np.ndarray            # (T_w, S, S) float64
    phase_deg: np.ndarray              # (T_w, S, S) float64
    phase_deg_masked: np.ndarray       # (T_w, S, S) float64
    K_ij: np.ndarray                   # (T_w, S, S) int64
    n_eff: np.ndarray                  # (T_w, S, S) float64
    is_significant: np.ndarray         # (T_w, S, S) bool


@dataclass(frozen=True)
class AzimuthalWavenumberResult:
    """Azimuthal wave number m = delta_phi / delta_lon with meridional guard and aliasing info."""

    m: np.ndarray | float
    m_alias: float
    delta_lon_deg: float
    is_meridional_guarded: bool
    delta_phi_rad: np.ndarray | float


@dataclass(frozen=True)
class MeridionalFLRProfile:
    """Meridional amplitude and phase profile across a latitudinal station chain."""

    station_codes: tuple[str, ...]
    latitudes_deg: np.ndarray
    band_power: np.ndarray
    amplitude: np.ndarray
    pair_coherence: np.ndarray
    pair_gamma2_crit: np.ndarray
    pair_phase_deg: np.ndarray
    pair_is_significant: np.ndarray
    cumulative_phase_deg: np.ndarray
    phase_gradient_deg_per_lat: np.ndarray
    peak_station: str
    peak_latitude_deg: float


def build_phase1_cache_key(
    core_version: str,
    *,
    episode_id: str,
    station_iaga: str,
    component: str,
    coord_frame: str,
    d0_deg: float,
    L: int,
    M: int,
    window_samples: int,
    step_samples: int,
    detrend: str = "linear_ols",
    window: str = "hann_periodic",
    qc_params: dict[str, Any] | None = None,
) -> str:
    """Build unified Phase 1 cache key per ARCHITECTURE.md §4.2."""
    params = {
        "stage": "welch_phase1",
        "episode_id": episode_id,
        "station_iaga": station_iaga,
        "component": component,
        "coord_frame": coord_frame,
        "d0_deg": round(float(d0_deg), 6),
        "L": int(L),
        "M": int(M),
        "window_samples": int(window_samples),
        "step_samples": int(step_samples),
        "detrend": detrend,
        "window": window,
        "qc_params": qc_params or {},
    }
    return canonical_cache_key(core_version, params)


def compute_station_phase1_spectra(
    series_1d: np.ndarray,
    *,
    station_iaga: str,
    component: str = "X",
    dt: float = DT_1MIN_SEC,
    L: int = 128,
    M: int = 64,
    window_samples: int | None = None,
    step_samples: int | None = None,
    is_valid: np.ndarray | None = None,
    max_missing_frac: float = 0.05,
    max_gap: int = 3,
    episode_id: str = "",
    coord_frame: str = "XYZ",
    d0_deg: float = 0.0,
    core_version: str = "0.1.0",
    dtype: np.dtype[Any] | type[np.complex128] | type[np.complex64] = np.complex128,
) -> Phase1StationSpectra:
    """Compute Phase 1 windowed complex segment FFTs and validity masks for one station."""
    arr = np.asarray(series_1d, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D time series, got shape {arr.shape}")
    n_total = arr.shape[0]
    if L <= 0 or M <= 0:
        raise ValueError(f"L and M must be positive, got L={L}, M={M}")

    w_len = n_total if window_samples is None else int(window_samples)
    w_step = w_len if step_samples is None else int(step_samples)
    if w_len < L:
        raise ValueError(f"window_samples ({w_len}) must be >= L ({L})")
    if w_step <= 0:
        raise ValueError(f"step_samples must be positive, got {w_step}")
    if n_total < w_len:
        raise ValueError(f"Series length ({n_total}) must be >= window_samples ({w_len})")

    k_seg = (w_len - L) // M + 1
    win_starts = np.arange(0, n_total - w_len + 1, w_step, dtype=np.int64)
    win_centers = win_starts + w_len // 2

    seg_offsets = np.arange(k_seg, dtype=np.int64) * M
    sample_offsets = np.arange(L, dtype=np.int64)
    # Shape: (T_w, K, L)
    idx = win_starts[:, None, None] + seg_offsets[None, :, None] + sample_offsets[None, None, :]

    segs = arr[idx]
    seg_finite = np.all(np.isfinite(segs), axis=-1)

    if is_valid is None:
        valid_segments = seg_finite
    else:
        raw_valid = np.asarray(is_valid, dtype=bool)
        if raw_valid.shape != (n_total,):
            raise ValueError(f"is_valid shape {raw_valid.shape} must match ({n_total},).")
        raw_segs = raw_valid[idx]
        all_raw_ok = np.all(raw_segs, axis=-1)
        miss_frac_ok = np.mean(~raw_segs, axis=-1) <= max_missing_frac + 1e-12
        valid_segments = seg_finite & all_raw_ok
        need_gap_check = seg_finite & (~all_raw_ok) & miss_frac_ok
        if np.any(need_gap_check):
            rows, cols = np.where(need_gap_check)
            for r_i, c_i in zip(rows, cols, strict=True):
                valid_segments[r_i, c_i] = is_window_valid(
                    raw_segs[r_i, c_i],
                    np.isfinite(segs[r_i, c_i]),
                    max_missing_frac=max_missing_frac,
                    max_gap=max_gap,
                )

    clean_segs = np.where(valid_segments[:, :, None], segs, 0.0)
    detrended = detrend_linear_ols(clean_segs, axis=-1)
    win = periodic_hann_window(L)
    windowed = detrended * win[None, None, :]
    fft_segs = np.fft.rfft(windowed, axis=-1)
    fft_segs[~valid_segments] = 0.0
    if np.dtype(dtype) == np.complex64:
        fft_segs = fft_segs.astype(np.complex64, copy=False)

    freqs = np.fft.rfftfreq(L, d=dt)
    cache_key = build_phase1_cache_key(
        core_version,
        episode_id=episode_id,
        station_iaga=station_iaga,
        component=component,
        coord_frame=coord_frame,
        d0_deg=d0_deg,
        L=L,
        M=M,
        window_samples=w_len,
        step_samples=w_step,
        qc_params={"max_missing_frac": max_missing_frac, "max_gap": max_gap},
    )

    return Phase1StationSpectra(
        station_iaga=station_iaga,
        component=component,
        dt=float(dt),
        L=int(L),
        M=int(M),
        window_samples=w_len,
        step_samples=w_step,
        window_center_indices=win_centers,
        freqs_hz=freqs,
        F=fft_segs,
        valid_segments=valid_segments,
        cache_key=cache_key,
    )


def save_phase1_cache(path: Path, spectra: Phase1StationSpectra) -> None:
    """Persist Phase1StationSpectra to compressed NPZ."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        station_iaga=np.array(spectra.station_iaga),
        component=np.array(spectra.component),
        dt=np.array(spectra.dt, dtype=np.float64),
        L=np.array(spectra.L, dtype=np.int64),
        M=np.array(spectra.M, dtype=np.int64),
        window_samples=np.array(spectra.window_samples, dtype=np.int64),
        step_samples=np.array(spectra.step_samples, dtype=np.int64),
        window_center_indices=spectra.window_center_indices,
        freqs_hz=spectra.freqs_hz,
        F=spectra.F,
        valid_segments=spectra.valid_segments,
        cache_key=np.array(spectra.cache_key),
    )


def load_phase1_cache(path: Path) -> Phase1StationSpectra:
    """Load Phase1StationSpectra from compressed NPZ."""
    with np.load(path, allow_pickle=False) as data:
        return Phase1StationSpectra(
            station_iaga=str(data["station_iaga"].item()),
            component=str(data["component"].item()),
            dt=float(data["dt"].item()),
            L=int(data["L"].item()),
            M=int(data["M"].item()),
            window_samples=int(data["window_samples"].item()),
            step_samples=int(data["step_samples"].item()),
            window_center_indices=data["window_center_indices"],
            freqs_hz=data["freqs_hz"],
            F=data["F"],
            valid_segments=data["valid_segments"],
            cache_key=str(data["cache_key"].item()),
        )


def welch_band_dof(
    K_ij: np.ndarray | int,
    *,
    N_B: int | None = None,
    band: str = "Pc5",
    spectrum_type: str = "flat",
    L: int = 128,
    M: int = 64,
    dt: float = DT_1MIN_SEC,
    t_max_pc6_sec: float = 1800.0,
    f_min_hz: float | None = None,
    f_max_hz: float | None = None,
) -> np.ndarray:
    """Band-averaged Welch effective DOF via exact Block-Toeplitz Gram formulation."""
    _ = N_B  # Retained for API backward compatibility; bin structure follows L, dt, band.
    return welch_effective_dof(
        K_ij,
        L=L,
        M=M,
        dt=dt,
        band=band,
        spectrum_type=spectrum_type,
        t_max_pc6_sec=t_max_pc6_sec,
        f_min_hz=f_min_hz,
        f_max_hz=f_max_hz,
    )


def _gamma2_crit_from_n(n_eff: np.ndarray, alpha: float) -> np.ndarray:
    out = np.full_like(n_eff, np.nan, dtype=np.float64)
    valid = np.isfinite(n_eff) & (n_eff > 1.0)
    out[valid] = 1.0 - (float(alpha) ** (1.0 / (n_eff[valid] - 1.0)))
    return out


def _validate_phase1_pair(spec_i: Phase1StationSpectra, spec_j: Phase1StationSpectra) -> None:
    if spec_i.L != spec_j.L or spec_i.M != spec_j.M or spec_i.dt != spec_j.dt:
        raise ValueError("Incompatible Phase 1 Welch parameters (L, M, dt) between stations.")
    if spec_i.F.shape != spec_j.F.shape:
        raise ValueError(
            f"Incompatible Phase 1 array shapes: {spec_i.F.shape} vs {spec_j.F.shape}"
        )


def compute_welch_cross_pair(
    spec_i: Phase1StationSpectra,
    spec_j: Phase1StationSpectra,
    *,
    band: str = "Pc5",
    t_max_pc6_sec: float = 1800.0,
    f_min_hz: float | None = None,
    f_max_hz: float | None = None,
    alpha: float = 0.05,
    min_common_segments: int = 2,
    spectrum_type: str = "flat",
) -> WelchPairResult:
    """Phase 2 pairwise Welch cross-spectrum strictly over joint mask valid_i & valid_j."""
    _validate_phase1_pair(spec_i, spec_j)

    f_mask = band_frequency_mask(
        spec_i.freqs_hz,
        band=band,
        t_max_pc6_sec=t_max_pc6_sec,
        f_min_hz=f_min_hz,
        f_max_hz=f_max_hz,
    )
    n_b = int(np.sum(f_mask))
    if n_b == 0:
        raise ValueError(f"No FFT bins in band {band!r} for L={spec_i.L}, dt={spec_i.dt}")

    mask_ij = spec_i.valid_segments & spec_j.valid_segments  # (T_w, K)
    k_ij = np.sum(mask_ij, axis=1, dtype=np.int64)          # (T_w,)
    is_valid = k_ij >= int(min_common_segments)

    fi = spec_i.F[:, :, f_mask].astype(np.complex128, copy=False)  # (T_w, K, N_B)
    fj = spec_j.F[:, :, f_mask].astype(np.complex128, copy=False)
    m_exp = mask_ij[:, :, None]

    # Joint-mask frequency-resolved sums across common segments
    s_ij_sum_tf = np.sum(np.where(m_exp, fi * np.conj(fj), 0.0), axis=1)
    s_ii_sum_tf = np.sum(np.where(m_exp, (fi.real ** 2 + fi.imag ** 2), 0.0), axis=1)
    s_jj_sum_tf = np.sum(np.where(m_exp, (fj.real ** 2 + fj.imag ** 2), 0.0), axis=1)

    win = periodic_hann_window(spec_i.L)
    psd_scale = 2.0 * spec_i.dt / float(np.sum(win ** 2))
    denom_k = np.where(is_valid, np.maximum(k_ij, 1), 1).astype(np.float64)[:, None]

    s_ij_tf = (s_ij_sum_tf / denom_k) * psd_scale
    s_ii_tf = (s_ii_sum_tf / denom_k) * psd_scale
    s_jj_tf = (s_jj_sum_tf / denom_k) * psd_scale
    s_ij_tf[~is_valid] = np.nan + 0j

    denom_tf = s_ii_tf * s_jj_tf
    gamma2_tf = np.full_like(denom_tf, np.nan, dtype=np.float64)
    good_tf = is_valid[:, None] & (denom_tf > 0.0)
    gamma2_tf[good_tf] = np.clip(
        (np.abs(s_ij_tf[good_tf]) ** 2) / denom_tf[good_tf], 0.0, 1.0
    )

    n_seg = welch_segment_dof(k_ij, L=spec_i.L, M=spec_i.M)
    gamma2_crit_tf = _gamma2_crit_from_n(n_seg, alpha=alpha)
    phase_deg_tf = np.where(is_valid[:, None], np.rad2deg(np.angle(s_ij_tf)), np.nan)
    sig_tf = good_tf & (gamma2_tf > gamma2_crit_tf[:, None])
    phase_deg_tf_masked = np.where(sig_tf, phase_deg_tf, np.nan)

    # Band-averaged quantities strictly over mask_ij
    s_ij = np.mean(s_ij_tf, axis=1)
    s_ii = np.mean(s_ii_tf, axis=1)
    s_jj = np.mean(s_jj_tf, axis=1)
    s_ij[~is_valid] = np.nan + 0j
    s_ii[~is_valid] = np.nan
    s_jj[~is_valid] = np.nan

    denom_band = s_ii * s_jj
    gamma2 = np.full(k_ij.shape, np.nan, dtype=np.float64)
    good_band = is_valid & (denom_band > 0.0)
    gamma2[good_band] = np.clip(
        (np.abs(s_ij[good_band]) ** 2) / denom_band[good_band], 0.0, 1.0
    )

    n_eff = np.where(
        is_valid,
        welch_band_dof(
            k_ij,
            N_B=n_b,
            band=band,
            spectrum_type=spectrum_type,
            L=spec_i.L,
            M=spec_i.M,
            dt=spec_i.dt,
            t_max_pc6_sec=t_max_pc6_sec,
            f_min_hz=f_min_hz,
            f_max_hz=f_max_hz,
        ),
        np.nan,
    )
    gamma2_crit = _gamma2_crit_from_n(n_eff, alpha=alpha)
    is_significant = good_band & np.isfinite(gamma2_crit) & (gamma2 > gamma2_crit)

    phase_rad = np.where(is_valid, np.angle(s_ij), np.nan)
    phase_deg = np.rad2deg(phase_rad)
    phase_deg_masked = np.where(is_significant, phase_deg, np.nan)

    return WelchPairResult(
        station_i=spec_i.station_iaga,
        station_j=spec_j.station_iaga,
        band=band,
        window_center_indices=spec_i.window_center_indices,
        freqs_hz=spec_i.freqs_hz[f_mask],
        S_ij=s_ij,
        S_ii=s_ii,
        S_jj=s_jj,
        K_ij=k_ij,
        n_eff=n_eff,
        gamma2=gamma2,
        gamma2_crit=gamma2_crit,
        phase_rad=phase_rad,
        phase_deg=phase_deg,
        phase_deg_masked=phase_deg_masked,
        is_significant=is_significant,
        is_valid=is_valid,
        gamma2_tf=gamma2_tf,
        gamma2_crit_tf=gamma2_crit_tf,
        phase_deg_tf=phase_deg_tf,
        phase_deg_tf_masked=phase_deg_tf_masked,
        S_ij_tf=s_ij_tf,
    )


def compute_welch_cross_matrix(
    spectra_list: Sequence[Phase1StationSpectra],
    *,
    band: str = "Pc5",
    t_max_pc6_sec: float = 1800.0,
    f_min_hz: float | None = None,
    f_max_hz: float | None = None,
    alpha: float = 0.05,
    min_common_segments: int = 2,
    spectrum_type: str = "flat",
    time_chunk_size: int = 32,
) -> WelchMatrixResult:
    """Compute S x S all-pairs dynamic Welch matrix via chunked BLAS GEMM with joint masks."""
    if not spectra_list:
        raise ValueError("spectra_list must be non-empty.")
    ref = spectra_list[0]
    for sp in spectra_list[1:]:
        _validate_phase1_pair(ref, sp)

    f_mask = band_frequency_mask(
        ref.freqs_hz,
        band=band,
        t_max_pc6_sec=t_max_pc6_sec,
        f_min_hz=f_min_hz,
        f_max_hz=f_max_hz,
    )
    n_b = int(np.sum(f_mask))
    if n_b == 0:
        raise ValueError(f"No FFT bins in band {band!r} for L={ref.L}, dt={ref.dt}")

    n_st = len(spectra_list)
    t_w, k_seg, _ = ref.F.shape
    station_codes = tuple(sp.station_iaga for sp in spectra_list)

    gamma2_out = np.full((t_w, n_st, n_st), np.nan, dtype=np.float64)
    gamma2_crit_out = np.full((t_w, n_st, n_st), np.nan, dtype=np.float64)
    phase_deg_out = np.full((t_w, n_st, n_st), np.nan, dtype=np.float64)
    phase_masked_out = np.full((t_w, n_st, n_st), np.nan, dtype=np.float64)
    k_ij_out = np.zeros((t_w, n_st, n_st), dtype=np.int64)
    n_eff_out = np.full((t_w, n_st, n_st), np.nan, dtype=np.float64)
    sig_out = np.zeros((t_w, n_st, n_st), dtype=bool)

    chunk = max(1, int(time_chunk_size))
    for t0 in range(0, t_w, chunk):
        t1 = min(t_w, t0 + chunk)
        c_len = t1 - t0

        # Stack chunk across stations: (c_len, S, K, N_B)
        f_chunk = np.stack(
            [sp.F[t0:t1, :, f_mask].astype(np.complex128, copy=False) for sp in spectra_list],
            axis=1,
        )
        v_chunk = np.stack([sp.valid_segments[t0:t1, :] for sp in spectra_list], axis=1)
        # Ensure masked entries are strictly zero so joint sums need no explicit 3D mask
        f_chunk = np.where(v_chunk[:, :, :, None], f_chunk, 0.0)

        f_flat = f_chunk.reshape(c_len, n_st, k_seg * n_b)
        s_ij_num = f_flat @ np.conj(f_flat).transpose(0, 2, 1)  # (c_len, S, S)

        p_seg = np.sum(f_chunk.real ** 2 + f_chunk.imag ** 2, axis=-1)  # (c_len, S, K)
        v_f = v_chunk.astype(np.float64)
        s_ii_num = p_seg @ v_f.transpose(0, 2, 1)  # entry (i, j) = sum_{k in Vi&Vj} sum_b |F_i|^2
        s_jj_num = s_ii_num.transpose(0, 2, 1)

        k_mat = np.rint(v_f @ v_f.transpose(0, 2, 1)).astype(np.int64)
        valid_mat = k_mat >= int(min_common_segments)
        denom = s_ii_num * s_jj_num
        good = valid_mat & (denom > 0.0)

        g2 = np.full((c_len, n_st, n_st), np.nan, dtype=np.float64)
        g2[good] = np.clip((np.abs(s_ij_num[good]) ** 2) / denom[good], 0.0, 1.0)

        n_eff = np.where(
            valid_mat,
            welch_band_dof(
                k_mat,
                N_B=n_b,
                band=band,
                spectrum_type=spectrum_type,
                L=ref.L,
                M=ref.M,
                dt=ref.dt,
                t_max_pc6_sec=t_max_pc6_sec,
                f_min_hz=f_min_hz,
                f_max_hz=f_max_hz,
            ),
            np.nan,
        )
        g2_crit = _gamma2_crit_from_n(n_eff, alpha=alpha)
        ph_deg = np.where(valid_mat, np.rad2deg(np.angle(s_ij_num)), np.nan)
        sig = good & np.isfinite(g2_crit) & (g2 > g2_crit)

        gamma2_out[t0:t1] = g2
        gamma2_crit_out[t0:t1] = g2_crit
        phase_deg_out[t0:t1] = ph_deg
        phase_masked_out[t0:t1] = np.where(sig, ph_deg, np.nan)
        k_ij_out[t0:t1] = k_mat
        n_eff_out[t0:t1] = n_eff
        sig_out[t0:t1] = sig

    return WelchMatrixResult(
        station_codes=station_codes,
        band=band,
        window_center_indices=ref.window_center_indices,
        gamma2=gamma2_out,
        gamma2_crit=gamma2_crit_out,
        phase_deg=phase_deg_out,
        phase_deg_masked=phase_masked_out,
        K_ij=k_ij_out,
        n_eff=n_eff_out,
        is_significant=sig_out,
    )


def compute_azimuthal_wavenumber_m(
    delta_phi_rad: np.ndarray | float,
    lon1_deg: float,
    lon2_deg: float,
    gamma2: np.ndarray | float,
    gamma2_crit: np.ndarray | float,
    *,
    min_delta_lon_deg: float = 3.0,
    unwrap_time: bool = False,
) -> AzimuthalWavenumberResult:
    """Compute azimuthal wave number m = delta_phi / delta_lon with |d_lon| guard (§6.1)."""
    delta_lon_deg = (float(lon1_deg) - float(lon2_deg) + 180.0) % 360.0 - 180.0
    is_scalar = np.ndim(delta_phi_rad) == 0

    if abs(delta_lon_deg) < float(min_delta_lon_deg):
        nan_out: np.ndarray | float = (
            float("nan")
            if is_scalar
            else np.full_like(np.asarray(delta_phi_rad, dtype=np.float64), np.nan)
        )
        return AzimuthalWavenumberResult(
            m=nan_out,
            m_alias=float("nan"),
            delta_lon_deg=delta_lon_deg,
            is_meridional_guarded=True,
            delta_phi_rad=delta_phi_rad,
        )

    delta_lon_rad = math.radians(delta_lon_deg)
    m_alias = math.pi / abs(delta_lon_rad)

    phi_b, g2_b, gcrit_b = np.broadcast_arrays(
        np.asarray(delta_phi_rad, dtype=np.float64),
        np.asarray(gamma2, dtype=np.float64),
        np.asarray(gamma2_crit, dtype=np.float64),
    )
    phi_arr = np.array(phi_b, dtype=np.float64, copy=True)
    sig = np.isfinite(phi_arr) & np.isfinite(g2_b) & np.isfinite(gcrit_b) & (g2_b > gcrit_b)

    if unwrap_time and phi_arr.ndim >= 1 and np.any(sig):
        for tail_idx in np.ndindex(phi_arr.shape[1:]):
            sig_1d = sig[(slice(None), *tail_idx)]
            if not np.any(sig_1d):
                continue
            padded = np.concatenate(([False], sig_1d, [False]))
            diffs = np.diff(padded.astype(np.int8))
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]
            for s_i, e_i in zip(starts, ends, strict=True):
                if e_i - s_i > 1:
                    seg_sl = (slice(s_i, e_i), *tail_idx)
                    phi_arr[seg_sl] = np.unwrap(phi_arr[seg_sl])

    m_arr = np.where(sig, phi_arr / delta_lon_rad, np.nan)
    is_out_scalar = is_scalar and m_arr.ndim == 0
    return AzimuthalWavenumberResult(
        m=float(m_arr.item()) if is_out_scalar else m_arr,
        m_alias=m_alias,
        delta_lon_deg=delta_lon_deg,
        is_meridional_guarded=False,
        delta_phi_rad=float(phi_arr.item()) if is_out_scalar else phi_arr,
    )


def compute_meridional_flr_profile(
    spectra_chain: Sequence[Phase1StationSpectra],
    latitudes_deg: Sequence[float],
    *,
    band: str = "Pc5",
    window_index: int = 0,
    alpha: float = 0.05,
    min_common_segments: int = 2,
    t_max_pc6_sec: float = 1800.0,
) -> MeridionalFLRProfile:
    """Compute latitudinal FLR amplitude and phase gradient profile along a station chain."""
    if len(spectra_chain) < 2:
        raise ValueError("Meridional FLR profile requires at least 2 stations.")
    if len(spectra_chain) != len(latitudes_deg):
        raise ValueError("spectra_chain and latitudes_deg must have equal length.")

    lats = np.asarray(latitudes_deg, dtype=np.float64)
    codes = tuple(sp.station_iaga for sp in spectra_chain)
    ref = spectra_chain[0]
    f_mask = band_frequency_mask(ref.freqs_hz, band=band, t_max_pc6_sec=t_max_pc6_sec)
    win = periodic_hann_window(ref.L)
    psd_scale = 2.0 * ref.dt / float(np.sum(win ** 2))

    powers = np.full(len(spectra_chain), np.nan, dtype=np.float64)
    for idx, sp in enumerate(spectra_chain):
        v_mask = sp.valid_segments[window_index]
        if np.any(v_mask):
            f_band = sp.F[window_index, v_mask][:, f_mask]
            powers[idx] = float(np.mean(f_band.real ** 2 + f_band.imag ** 2)) * psd_scale
    amps = np.sqrt(np.maximum(0.0, powers))

    n_pairs = len(spectra_chain) - 1
    pair_coh = np.full(n_pairs, np.nan, dtype=np.float64)
    pair_crit = np.full(n_pairs, np.nan, dtype=np.float64)
    pair_phase = np.full(n_pairs, np.nan, dtype=np.float64)
    pair_sig = np.zeros(n_pairs, dtype=bool)
    grad = np.full(n_pairs, np.nan, dtype=np.float64)

    cum_phase = np.zeros(len(spectra_chain), dtype=np.float64)
    for i in range(n_pairs):
        pair_res = compute_welch_cross_pair(
            spectra_chain[i],
            spectra_chain[i + 1],
            band=band,
            t_max_pc6_sec=t_max_pc6_sec,
            alpha=alpha,
            min_common_segments=min_common_segments,
        )
        pair_coh[i] = float(pair_res.gamma2[window_index])
        pair_crit[i] = float(pair_res.gamma2_crit[window_index])
        pair_phase[i] = float(pair_res.phase_deg[window_index])
        pair_sig[i] = bool(pair_res.is_significant[window_index])

        d_lat = float(lats[i + 1] - lats[i])
        if pair_sig[i] and np.isfinite(pair_phase[i]):
            cum_phase[i + 1] = cum_phase[i] - pair_phase[i]
            if abs(d_lat) > 1e-6:
                grad[i] = -pair_phase[i] / d_lat
        else:
            cum_phase[i + 1] = np.nan

    if np.any(np.isfinite(amps)):
        peak_idx = int(np.nanargmax(amps))
        peak_station = codes[peak_idx]
        peak_lat = float(lats[peak_idx])
    else:
        peak_station = ""
        peak_lat = float("nan")

    return MeridionalFLRProfile(
        station_codes=codes,
        latitudes_deg=lats,
        band_power=powers,
        amplitude=amps,
        pair_coherence=pair_coh,
        pair_gamma2_crit=pair_crit,
        pair_phase_deg=pair_phase,
        pair_is_significant=pair_sig,
        cumulative_phase_deg=cum_phase,
        phase_gradient_deg_per_lat=grad,
        peak_station=peak_station,
        peak_latitude_deg=peak_lat,
    )
