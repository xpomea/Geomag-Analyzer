"""Variant B time-domain polarization: NaN subsegments, Marple Hilbert, S^(b), Formula 48 (§9)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .conventions import DT_1MIN_SEC, get_band_spec, is_window_valid
from .filters import (
    apply_butterworth_sos_zero_phase,
    apply_fft_mask,
    butterworth_zero_phase_response,
    fft_mask_response,
)
from .oversampling import odd_reflect_pad, oversample_fourier, oversample_kaiser


@dataclass(frozen=True)
class VariantBResult:
    """Result of Variant B bandpass + Marple analytic signal polarization pipeline."""

    S: np.ndarray
    n_eff: float
    B_eff_hz: float
    filtered: np.ndarray
    analytic: np.ndarray
    usable_mask: np.ndarray
    subsegment_ranges_orig: tuple[tuple[int, int], ...]
    subsegment_ranges_oversampled: tuple[tuple[int, int], ...]
    dt_orig: float
    dt_prime: float
    oversample_m: int
    band: str
    filter_method: str
    trim_hours: float
    is_valid: bool


def default_trim_hours(
    band: str = "Pc5",
    filter_method: str = "sos",
    order_n: int = 3,
    *,
    t_max_pc6_sec: float = 1800.0,
    f1_hz: float | None = None,
    f2_hz: float | None = None,
    rho: float = 0.25,
) -> float:
    """Compute band-aware practical edge trim T_cut [hours] from filter decay scales (§9.3)."""
    spec = get_band_spec(
        band=band,
        t_max_pc6_sec=t_max_pc6_sec,
        f_min_hz=f1_hz,
        f_max_hz=f2_hz,
    )
    t_max_h = spec.t_max_sec / 3600.0
    bw_hz = spec.f_max_hz - spec.f_min_hz
    t_bw_h = 1.0 / (3600.0 * bw_hz)
    tau_h = max(t_max_h, t_bw_h)
    f_key = filter_method.strip().lower()
    if f_key in ("fft", "fft_mask"):
        return round(max(3.0, 2.5 * tau_h / max(float(rho), 0.05)), 4)
    return round(max(3.0, (float(order_n) + 3.0) * tau_h), 4)


def analytic_signal_marple(x: np.ndarray, *, reflect_samples: int = 0) -> np.ndarray:
    """Compute discrete-time analytic signal z = x + i*H{x} via Marple (1999) (Formula 44)."""
    arr = np.asarray(x, dtype=np.float64)
    squeeze_1d = arr.ndim == 1
    work_in = arr[:, None] if squeeze_1d else arr
    n_orig = work_in.shape[0]
    if n_orig == 0:
        raise ValueError("Cannot compute analytic signal on empty array.")

    work = odd_reflect_pad(work_in, reflect_samples) if reflect_samples > 0 else work_in
    n_p = work.shape[0]

    spec = np.fft.fft(work, axis=0)
    h = np.zeros(n_p, dtype=np.float64)
    h[0] = 1.0
    if n_p % 2 == 0:
        h[1 : n_p // 2] = 2.0
        h[n_p // 2] = 1.0
    else:
        h[1 : (n_p + 1) // 2] = 2.0

    z_full = np.fft.ifft(spec * h[:, None], axis=0)
    out = z_full[reflect_samples : reflect_samples + n_orig]
    return out[:, 0] if squeeze_1d else out


def spectral_matrix_from_analytic(
    z: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute spectral matrix S^(b) from analytic signals via Formula (45)."""
    arr = np.asarray(z, dtype=np.complex128)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError(f"Expected (N, 2) or (N, 3) analytic array, got shape {arr.shape}")
    n = arr.shape[0]
    if n == 0:
        raise ValueError("Empty analytic signal window.")

    if weights is None:
        return (arr.T @ np.conj(arr)) / float(n)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n,):
        raise ValueError(f"weights shape {w.shape} must match ({n},).")
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        raise ValueError("Sum of weights must be positive.")
    zw = arr * w[:, None]
    return (zw.T @ np.conj(arr)) / w_sum


def spectral_matrix_real_form(
    x: np.ndarray,
    hx: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute S^(b) explicitly from real x and Hilbert transform H{x} via Formula (46)."""
    x_arr = np.asarray(x, dtype=np.float64)
    hx_arr = np.asarray(hx, dtype=np.float64)
    if x_arr.shape != hx_arr.shape or x_arr.ndim != 2 or x_arr.shape[1] not in (2, 3):
        raise ValueError(f"Invalid shapes x={x_arr.shape}, hx={hx_arr.shape}")
    n = x_arr.shape[0]
    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (n,):
            raise ValueError(f"weights shape {w.shape} must match ({n},).")
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        raise ValueError("Sum of weights must be positive.")

    xw = x_arr * w[:, None]
    hxw = hx_arr * w[:, None]
    re_s = (xw.T @ x_arr + hxw.T @ hx_arr) / w_sum
    im_s = (hxw.T @ x_arr - xw.T @ hx_arr) / w_sum
    return re_s + 1j * im_s


def effective_n_formula48(
    window_sec: float,
    *,
    band: str = "Pc5",
    filter_method: str = "fft_mask",
    dt_orig: float = DT_1MIN_SEC,
    oversample_m: int = 1,
    order_n: int = 3,
    spectrum_type: str = "flat",
    t_max_pc6_sec: float = 1800.0,
    f1_hz: float | None = None,
    f2_hz: float | None = None,
    rho: float = 0.25,
    weights: np.ndarray | None = None,
    dt_prime: float | None = None,
    n_grid: int = 50000,
) -> tuple[float, float]:
    """Compute effective DOF n and effective bandwidth B_eff via Formula (48).

    Strictly integrates over the original signal Nyquist interval (0, 1/(2*dt_orig)]
    so n is invariant to oversampling factor m for FFT-mask and never scales by m*N.
    """
    spec = get_band_spec(
        band=band,
        t_max_pc6_sec=t_max_pc6_sec,
        f_min_hz=f1_hz,
        f_max_hz=f2_hz,
    )
    f_nyq_orig = 1.0 / (2.0 * float(dt_orig))
    f_grid = np.linspace(1e-7, f_nyq_orig * (1.0 - 1e-8), n_grid, dtype=np.float64)

    f_key = filter_method.strip().lower()
    if f_key in ("fft", "fft_mask"):
        a_resp = fft_mask_response(f_grid, spec.f_min_hz, spec.f_max_hz, rho=rho)
    elif f_key in ("sos", "butterworth", "butterworth_sos"):
        fs_hz = float(oversample_m) / float(dt_orig)
        a_resp = butterworth_zero_phase_response(
            f_grid, spec.f_min_hz, spec.f_max_hz, fs_hz, order_n=order_n
        )
    else:
        raise ValueError(f"Unsupported filter_method {filter_method!r}")

    s_key = spectrum_type.strip().lower()
    if s_key in ("flat", "white"):
        s0 = np.ones_like(f_grid)
    elif s_key in ("red", "f-2", "f^-2"):
        s0 = 1.0 / (f_grid ** 2)
    else:
        raise ValueError(f"Unsupported spectrum_type {spectrum_type!r}")

    p_f = (a_resp ** 2) * s0
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    num = float(trapz_fn(p_f, f_grid))
    den = float(trapz_fn(p_f ** 2, f_grid))
    b_eff = (num * num) / den

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        step = (dt_orig / float(oversample_m)) if dt_prime is None else float(dt_prime)
        t_eff = step * (float(np.sum(w)) ** 2) / float(np.sum(w ** 2))
    else:
        t_eff = float(window_sec)

    return t_eff * b_eff, b_eff


def find_valid_subsegments(
    is_finite_mask: np.ndarray,
    *,
    min_length: int = 1,
) -> list[tuple[int, int]]:
    """Find contiguous [start, end) runs where is_finite_mask is True and len >= min_length."""
    mask = np.asarray(is_finite_mask, dtype=bool)
    if mask.size == 0 or not bool(np.any(mask)):
        return []
    padded = np.concatenate(([False], mask, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return [
        (int(s), int(e))
        for s, e in zip(starts, ends, strict=True)
        if int(e - s) >= min_length
    ]


def run_variant_b_pipeline(
    xyz: np.ndarray,
    *,
    dt: float = DT_1MIN_SEC,
    band: str = "Pc5",
    filter_method: str = "sos",
    order_n: int = 3,
    oversample_m: int = 4,
    oversample_method: str = "fourier",
    oversample_pad_samples: int = 240,
    trim_hours: float | None = None,
    reflect_hours: float | None = None,
    analytic_reflect_hours: float = 6.0,
    min_window_hours: float = 1.0,
    detrend_pc6: bool | None = None,
    spectrum_type: str = "flat",
    t_max_pc6_sec: float = 1800.0,
    f1_hz: float | None = None,
    f2_hz: float | None = None,
    is_valid_xyz: np.ndarray | None = None,
    max_missing_frac: float = 0.05,
    max_gap: int = 3,
) -> VariantBResult:
    """Execute full Variant B pipeline with NaN-gap subsegment splitting (§§7-9).

    Any residual NaN gaps (>3 min after ETL) split the series into contiguous finite
    subsegments of length >= 2*T_cut + min_window. Each valid subsegment is oversampled,
    zero-phase filtered, transformed via Marple Hilbert, and trimmed by T_cut on BOTH ends
    so NaNs never enter FFT/sosfilt/Hilbert.
    """
    arr = np.asarray(xyz, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError(f"Expected (N, 2) or (N, 3) array, got shape {arr.shape}")
    n_orig, n_comp = arr.shape

    eff_trim_hours = (
        default_trim_hours(
            band=band,
            filter_method=filter_method,
            order_n=order_n,
            t_max_pc6_sec=t_max_pc6_sec,
            f1_hz=f1_hz,
            f2_hz=f2_hz,
        )
        if trim_hours is None
        else float(trim_hours)
    )
    eff_reflect_hours = (
        max(6.0, eff_trim_hours) if reflect_hours is None else float(reflect_hours)
    )
    trim_orig = int(round(eff_trim_hours * 3600.0 / dt))
    min_win_orig = max(1, int(round(min_window_hours * 3600.0 / dt)))
    min_subseg_orig = 2 * trim_orig + min_win_orig

    os_key = oversample_method.strip().lower()
    if os_key == "kaiser" and oversample_m > 1:
        n_total_os = oversample_m * max(0, n_orig - 1) + 1
    else:
        n_total_os = oversample_m * n_orig
    dt_prime = dt / float(oversample_m)

    filtered_full = np.full((n_total_os, n_comp), np.nan, dtype=np.float64)
    analytic_full = np.full((n_total_os, n_comp), np.nan + 0j, dtype=np.complex128)
    usable_mask = np.zeros(n_total_os, dtype=bool)

    finite_rows = np.all(np.isfinite(arr), axis=1)
    subsegments = find_valid_subsegments(finite_rows, min_length=min_subseg_orig)
    if is_valid_xyz is not None:
        raw_valid = np.asarray(is_valid_xyz, dtype=bool)
        if raw_valid.shape != (n_orig,):
            raise ValueError(f"is_valid_xyz shape {raw_valid.shape} must match ({n_orig},).")
        subsegments = [
            (s_idx, e_idx)
            for s_idx, e_idx in subsegments
            if is_window_valid(
                raw_valid[s_idx + trim_orig : e_idx - trim_orig],
                finite_rows[s_idx + trim_orig : e_idx - trim_orig],
                max_missing_frac=max_missing_frac,
                max_gap=max_gap,
            )
        ]
    if not subsegments:
        return VariantBResult(
            S=np.full((n_comp, n_comp), np.nan + 0j, dtype=np.complex128),
            n_eff=float("nan"),
            B_eff_hz=float("nan"),
            filtered=filtered_full,
            analytic=analytic_full,
            usable_mask=usable_mask,
            subsegment_ranges_orig=(),
            subsegment_ranges_oversampled=(),
            dt_orig=dt,
            dt_prime=dt_prime,
            oversample_m=oversample_m,
            band=band,
            filter_method=filter_method,
            trim_hours=eff_trim_hours,
            is_valid=False,
        )

    f_key = filter_method.strip().lower()
    analytic_pad_os = (
        int(round(analytic_reflect_hours * 3600.0 / dt_prime))
        if analytic_reflect_hours > 0.0
        else 0
    )
    os_ranges: list[tuple[int, int]] = []
    usable_z_chunks: list[np.ndarray] = []

    for start_idx, end_idx in subsegments:
        seg = arr[start_idx:end_idx]
        if oversample_m == 1:
            seg_os = seg.copy()
        elif os_key == "fourier":
            seg_os = oversample_fourier(seg, m=oversample_m, pad_samples=oversample_pad_samples)
        elif os_key == "kaiser":
            seg_os = oversample_kaiser(seg, m=oversample_m)
        else:
            raise ValueError(f"Unsupported oversample_method {oversample_method!r}")

        if f_key in ("fft", "fft_mask"):
            seg_filt = apply_fft_mask(
                seg_os,
                dt_prime=dt_prime,
                band=band,
                f1_hz=f1_hz,
                f2_hz=f2_hz,
                t_max_pc6_sec=t_max_pc6_sec,
                reflect_hours=eff_reflect_hours,
                detrend=detrend_pc6,
            )
        elif f_key in ("sos", "butterworth", "butterworth_sos"):
            seg_filt = apply_butterworth_sos_zero_phase(
                seg_os,
                dt_prime=dt_prime,
                band=band,
                order_n=order_n,
                f1_hz=f1_hz,
                f2_hz=f2_hz,
                t_max_pc6_sec=t_max_pc6_sec,
                reflect_hours=eff_reflect_hours,
                detrend=detrend_pc6,
            )
        else:
            raise ValueError(f"Unsupported filter_method {filter_method!r}")

        seg_z = analytic_signal_marple(seg_filt, reflect_samples=analytic_pad_os)

        trim_os = int(round(eff_trim_hours * 3600.0 / dt_prime))
        usable_len_os = seg_os.shape[0] - 2 * trim_os
        if usable_len_os <= 0:
            continue

        os_base = start_idx * oversample_m
        u_start = os_base + trim_os
        u_end = u_start + usable_len_os
        filtered_full[u_start:u_end] = seg_filt[trim_os : trim_os + usable_len_os]
        analytic_full[u_start:u_end] = seg_z[trim_os : trim_os + usable_len_os]
        usable_mask[u_start:u_end] = True
        os_ranges.append((u_start, u_end))
        usable_z_chunks.append(seg_z[trim_os : trim_os + usable_len_os])

    if not usable_z_chunks:
        return VariantBResult(
            S=np.full((n_comp, n_comp), np.nan + 0j, dtype=np.complex128),
            n_eff=float("nan"),
            B_eff_hz=float("nan"),
            filtered=filtered_full,
            analytic=analytic_full,
            usable_mask=usable_mask,
            subsegment_ranges_orig=tuple(subsegments),
            subsegment_ranges_oversampled=(),
            dt_orig=dt,
            dt_prime=dt_prime,
            oversample_m=oversample_m,
            band=band,
            filter_method=filter_method,
            trim_hours=eff_trim_hours,
            is_valid=False,
        )

    z_all = np.vstack(usable_z_chunks)
    s_mat = spectral_matrix_from_analytic(z_all)
    usable_window_sec = float(z_all.shape[0]) * dt_prime
    n_eff, b_eff = effective_n_formula48(
        usable_window_sec,
        band=band,
        filter_method=filter_method,
        dt_orig=dt,
        oversample_m=oversample_m,
        order_n=order_n,
        spectrum_type=spectrum_type,
        t_max_pc6_sec=t_max_pc6_sec,
        f1_hz=f1_hz,
        f2_hz=f2_hz,
    )

    return VariantBResult(
        S=s_mat,
        n_eff=n_eff,
        B_eff_hz=b_eff,
        filtered=filtered_full,
        analytic=analytic_full,
        usable_mask=usable_mask,
        subsegment_ranges_orig=tuple(subsegments),
        subsegment_ranges_oversampled=tuple(os_ranges),
        dt_orig=dt,
        dt_prime=dt_prime,
        oversample_m=oversample_m,
        band=band,
        filter_method=filter_method,
        trim_hours=eff_trim_hours,
        is_valid=True,
    )
