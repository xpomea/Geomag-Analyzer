"""Morlet CWT, XWT, WTC (NaN subsegments & gap-COI), and batched AR(1) Monte Carlo (§6.2)."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import lfilter

from .analytic import find_valid_subsegments
from .conventions import DT_1MIN_SEC, CalculationCancelledError, canonical_cache_key

MORLET_OMEGA0 = 6.0
MORLET_FOURIER_FACTOR = (4.0 * math.pi) / (MORLET_OMEGA0 + math.sqrt(2.0 + MORLET_OMEGA0 ** 2))
MORLET_COI_FACTOR = 1.0 / math.sqrt(2.0)
MORLET_DJ0 = 0.60
WTC_CACHE_FORMAT = "compact_f32_v2"

ProgressCallback = Callable[[int, int, str], None]
CancelCheck = Callable[[], bool]


@dataclass(frozen=True)
class XWTWTCResult:
    """Cross-wavelet transform (XWT), wavelet coherence (WTC), COI, and MC significance."""

    scales: np.ndarray                 # (n_scales,) seconds
    periods_sec: np.ndarray            # (n_scales,) seconds
    coi_periods_sec: np.ndarray        # (N,) seconds
    inside_coi: np.ndarray             # (n_scales, N) bool
    W_i: np.ndarray | None             # (n_scales, N) complex128 (None when loaded from disk cache)
    W_j: np.ndarray | None             # (n_scales, N) complex128 (None when loaded from disk cache)
    W_ij: np.ndarray | None            # (n_scales, N) complex128 (None when loaded from disk cache)
    xwt_power: np.ndarray              # (n_scales, N) float64
    xwt_phase_rad: np.ndarray          # (n_scales, N) float64
    xwt_phase_deg: np.ndarray          # (n_scales, N) float64
    wtc_r2: np.ndarray                 # (n_scales, N) float64
    wtc_phase_rad: np.ndarray          # (n_scales, N) float64
    wtc_phase_deg: np.ndarray          # (n_scales, N) float64
    alpha1_i: float
    alpha1_j: float
    wtc_sig95: np.ndarray | None       # (n_scales,) float64 or None
    wtc_is_significant: np.ndarray     # (n_scales, N) bool
    gap_policy: str
    subsegments: tuple[tuple[int, int], ...]
    n_surrogates: int
    alpha: float
    cache_key: str


def _morlet_fourier_factor(omega0: float = MORLET_OMEGA0) -> float:
    return (4.0 * math.pi) / (omega0 + math.sqrt(2.0 + omega0 * omega0))


def _morlet_dj0(omega0: float = MORLET_OMEGA0) -> float:
    """Scale decorrelation factor dj0 scaled cleanly with Morlet center frequency omega0."""
    if omega0 <= 0.0:
        raise ValueError(f"omega0 must be positive, got {omega0}")
    return MORLET_DJ0 * (MORLET_OMEGA0 / float(omega0))


def build_morlet_scales(
    n_samples: int,
    *,
    dt: float = DT_1MIN_SEC,
    period_min_sec: float = 120.0,
    period_max_sec: float = 3600.0,
    dj: float = 1.0 / 12.0,
    omega0: float = MORLET_OMEGA0,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct dyadic Morlet wavelet scales and equivalent Fourier periods [s]."""
    if n_samples <= 0 or dt <= 0.0 or dj <= 0.0:
        raise ValueError("n_samples, dt, and dj must be positive.")
    ff = _morlet_fourier_factor(omega0)
    s0 = max(2.0 * dt, float(period_min_sec)) / ff
    max_p = min(float(period_max_sec), float(n_samples) * float(dt))
    s_max = max(s0, max_p / ff)
    n_j = max(0, int(math.floor(math.log2(s_max / s0) / float(dj))))
    scales = s0 * (2.0 ** (np.arange(n_j + 1, dtype=np.float64) * float(dj)))
    periods_sec = scales * ff
    return scales, periods_sec


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _build_morlet_daughters(
    scales: np.ndarray,
    n_pad: int,
    dt: float,
    omega0: float = MORLET_OMEGA0,
) -> np.ndarray:
    """Build Fourier-space Morlet daughter matrix (n_scales, n_pad) for positive omega."""
    freqs = np.fft.fftfreq(n_pad, d=dt)
    omega = 2.0 * math.pi * freqs
    pos = omega > 0.0
    s_omega = scales[:, None] * omega[None, pos]
    norm = (math.pi ** -0.25) * np.sqrt((2.0 * math.pi * scales) / dt)[:, None]
    daughters = np.zeros((scales.shape[0], n_pad), dtype=np.float64)
    daughters[:, pos] = norm * np.exp(-0.5 * (s_omega - omega0) ** 2)
    return daughters


def morlet_cwt_single(
    x_1d: np.ndarray,
    scales: np.ndarray,
    *,
    dt: float = DT_1MIN_SEC,
    omega0: float = MORLET_OMEGA0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute complex Morlet CWT (n_scales, N) and boundary COI periods (N,)."""
    arr = np.asarray(x_1d, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("morlet_cwt_single requires finite input without NaNs.")
    n = arr.shape[0]
    n_pad = _next_pow2(n)
    x_zero = arr - float(np.mean(arr))
    x_hat = np.fft.fft(x_zero, n=n_pad)
    daughters = _build_morlet_daughters(scales, n_pad, dt, omega0=omega0)
    w_mat = np.fft.ifft(x_hat[None, :] * daughters, axis=1)[:, :n]

    idx = np.arange(n, dtype=np.float64)
    dist_edge = np.minimum(idx, float(n - 1) - idx)
    ff = _morlet_fourier_factor(omega0)
    coi_periods = dist_edge * dt * MORLET_COI_FACTOR * ff
    return w_mat, coi_periods


def _smooth_scale_boxcar_numpy(arr_2d: np.ndarray, width: float) -> np.ndarray:
    """Pure-NumPy symmetric fractional boxcar smoothing along scale axis 0."""
    n_scales, n_cols = arr_2d.shape
    if width <= 1.0 or n_scales <= 1:
        return arr_2d.copy()
    half_w = 0.5 * (width - 1.0)
    k_full = int(math.floor(half_w))
    frac = half_w - float(k_full)

    out = np.empty_like(arr_2d, dtype=np.float64)
    for i in range(n_scales):
        lo_full = max(0, i - k_full)
        hi_full = min(n_scales, i + k_full + 1)
        acc = np.sum(arr_2d[lo_full:hi_full], axis=0)
        w_sum = float(hi_full - lo_full)
        if frac > 0.0:
            left_idx = i - k_full - 1
            right_idx = i + k_full + 1
            if left_idx >= 0:
                acc = acc + frac * arr_2d[left_idx]
                w_sum += frac
            if right_idx < n_scales:
                acc = acc + frac * arr_2d[right_idx]
                w_sum += frac
        out[i] = acc / w_sum
    return out


def _generate_ar1_numpy(noise_2d: np.ndarray, alpha: float) -> np.ndarray:
    n_rows, n_cols = noise_2d.shape
    if n_cols <= 1:
        return noise_2d.copy()
    scale = math.sqrt(max(0.0, 1.0 - alpha * alpha))
    out = np.empty_like(noise_2d, dtype=np.float64)
    out[:, 0] = noise_2d[:, 0]
    zi = alpha * noise_2d[:, :1]
    out[:, 1:], _ = lfilter([scale], [1.0, -alpha], noise_2d[:, 1:], axis=1, zi=zi)
    return out


def _build_time_gauss_kernels(
    scales: np.ndarray,
    n_samples: int,
    dt: float,
) -> tuple[np.ndarray, int]:
    """Precompute FFT Gaussian time-smoothing kernels (n_scales, n_pad_t)."""
    max_half = int(math.ceil(4.0 * float(np.max(scales)) / dt))
    n_pad_t = _next_pow2(n_samples + max_half)
    omega = 2.0 * math.pi * np.fft.fftfreq(n_pad_t, d=dt)
    gauss_hat = np.exp(-0.5 * (scales[:, None] * omega[None, :]) ** 2)
    return gauss_hat, n_pad_t


def smooth_wavelet_matrix(
    arr: np.ndarray,
    scales: np.ndarray,
    *,
    dt: float = DT_1MIN_SEC,
    dj: float = 1.0 / 12.0,
    omega0: float = MORLET_OMEGA0,
) -> np.ndarray:
    """Apply Grinsted et al. (2004) Gaussian time smoothing + fractional boxcar scale smoothing."""
    data = np.asarray(arr)
    if data.ndim != 2 or data.shape[0] != scales.shape[0]:
        raise ValueError(f"Shape {data.shape} incompatible with scales {scales.shape}")
    n_samples = data.shape[1]
    gauss_hat, n_pad_t = _build_time_gauss_kernels(scales, n_samples, dt)
    width = _morlet_dj0(omega0) / float(dj)

    if np.iscomplexobj(data):
        spec = np.fft.fft(data, n=n_pad_t, axis=1)
        t_smooth = np.fft.ifft(spec * gauss_hat, axis=1)[:, :n_samples]
        re_s = _smooth_scale_boxcar_numpy(t_smooth.real, width)
        im_s = _smooth_scale_boxcar_numpy(t_smooth.imag, width)
        return re_s + 1j * im_s

    spec = np.fft.rfft(data.astype(np.float64, copy=False), n=n_pad_t, axis=1)
    gauss_r = gauss_hat[:, : n_pad_t // 2 + 1]
    t_smooth = np.fft.irfft(spec * gauss_r, n=n_pad_t, axis=1)[:, :n_samples]
    return _smooth_scale_boxcar_numpy(t_smooth, width)


def estimate_ar1(x_1d: np.ndarray) -> float:
    """Estimate lag-1 autocorrelation alpha1 over contiguous finite pairs, clamped to [0, 0.999]."""
    arr = np.asarray(x_1d, dtype=np.float64)
    finite = np.isfinite(arr)
    if int(np.sum(finite)) < 4:
        return 0.0
    pair_ok = finite[:-1] & finite[1:]
    if int(np.sum(pair_ok)) < 3:
        return 0.0
    mean_v = float(np.mean(arr[finite]))
    xc0 = arr[:-1][pair_ok] - mean_v
    xc1 = arr[1:][pair_ok] - mean_v
    num = float(np.sum(xc0 * xc1))
    den = math.sqrt(float(np.sum(xc0 ** 2)) * float(np.sum(xc1 ** 2)))
    if den <= 0.0:
        return 0.0
    alpha = num / den
    return min(0.999, max(0.0, alpha))


def compute_gap_coi_scales(valid_mask: np.ndarray, *, dt: float = DT_1MIN_SEC) -> np.ndarray:
    """Compute distance-to-nearest-gap/edge COI scale [s] via O(N) forward/backward pass."""
    mask = np.asarray(valid_mask, dtype=bool)
    n = mask.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    dist_left = np.empty(n, dtype=np.float64)
    dist_right = np.empty(n, dtype=np.float64)

    cur = -1.0
    for i in range(n):
        if not mask[i]:
            cur = 0.0
        else:
            cur = 0.0 if i == 0 else cur + 1.0
        dist_left[i] = cur

    cur = -1.0
    for i in range(n - 1, -1, -1):
        if not mask[i]:
            cur = 0.0
        else:
            cur = 0.0 if i == n - 1 else cur + 1.0
        dist_right[i] = cur

    d_min = np.minimum(dist_left, dist_right)
    return d_min * float(dt) * MORLET_COI_FACTOR


def _bridge_gaps_linear(x_1d: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Linearly interpolate gaps (constant-pad ends) for gap-COI mode; inside_coi masks edges."""
    out = np.asarray(x_1d, dtype=np.float64).copy()
    if bool(np.all(valid_mask)):
        return out
    idx_all = np.arange(out.shape[0], dtype=np.float64)
    idx_good = idx_all[valid_mask]
    if idx_good.size == 0:
        raise ValueError("No valid samples in series.")
    out[~valid_mask] = np.interp(idx_all[~valid_mask], idx_good, out[valid_mask])
    return out


def _compute_xwt_wtc_segment(
    seg_i: np.ndarray,
    seg_j: np.ndarray,
    scales: np.ndarray,
    *,
    dt: float,
    dj: float,
    omega0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w_i, coi_p = morlet_cwt_single(seg_i, scales, dt=dt, omega0=omega0)
    w_j, _ = morlet_cwt_single(seg_j, scales, dt=dt, omega0=omega0)
    w_ij = w_i * np.conj(w_j)

    inv_s = (1.0 / scales)[:, None]
    s_wij = smooth_wavelet_matrix(
        inv_s * w_ij, scales, dt=dt, dj=dj, omega0=omega0
    )
    s_wi2 = smooth_wavelet_matrix(
        inv_s * (w_i.real ** 2 + w_i.imag ** 2),
        scales,
        dt=dt,
        dj=dj,
        omega0=omega0,
    )
    s_wj2 = smooth_wavelet_matrix(
        inv_s * (w_j.real ** 2 + w_j.imag ** 2),
        scales,
        dt=dt,
        dj=dj,
        omega0=omega0,
    )

    denom = np.maximum(s_wi2 * s_wj2, 1e-30)
    wtc_r2 = np.clip((s_wij.real ** 2 + s_wij.imag ** 2) / denom, 0.0, 1.0)
    wtc_phase = np.angle(s_wij)
    return w_i, w_j, w_ij, wtc_r2, wtc_phase, coi_p


def _run_wtc_monte_carlo(
    n_samples: int,
    scales: np.ndarray,
    alpha1_i: float,
    alpha1_j: float,
    *,
    dt: float,
    dj: float,
    omega0: float,
    n_surrogates: int,
    alpha: float,
    seed: int,
    batch_size: int | None,
    progress_callback: ProgressCallback | None,
    cancel_check: CancelCheck | None,
) -> np.ndarray:
    """Batched AR(1) Monte Carlo significance level for WTC inside COI."""
    n_scales = scales.shape[0]
    n_bins = 1000
    hist = np.zeros((n_scales, n_bins), dtype=np.int64)

    n_pad_cwt = _next_pow2(n_samples)
    daughters = _build_morlet_daughters(scales, n_pad_cwt, dt, omega0=omega0)
    gauss_hat, n_pad_t = _build_time_gauss_kernels(scales, n_samples, dt)
    gauss_r = gauss_hat[:, : n_pad_t // 2 + 1]
    inv_s = (1.0 / scales)[None, :, None]
    width = _morlet_dj0(omega0) / float(dj)

    if batch_size is None:
        # Cap batch working-set buffers around ~80 MB so parallel workers avoid OOM
        target_bytes = 80 * 1024 * 1024
        bytes_per_surr = max(1, n_scales * max(n_pad_cwt, n_pad_t) * 80)
        eff_batch = max(1, min(25, target_bytes // bytes_per_surr))
    else:
        if int(batch_size) < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        eff_batch = int(batch_size)

    idx = np.arange(n_samples, dtype=np.float64)
    ff = _morlet_fourier_factor(omega0)
    coi_p = np.minimum(idx, float(n_samples - 1) - idx) * dt * MORLET_COI_FACTOR * ff
    periods = scales * ff
    inside_coi = periods[:, None] <= coi_p[None, :]
    # Fallback to all indices if signal is shorter than smallest scale COI
    valid_t_masks = [
        inside_coi[s_idx] if np.any(inside_coi[s_idx]) else np.ones(n_samples, dtype=bool)
        for s_idx in range(n_scales)
    ]

    rng = np.random.default_rng(seed)
    remaining = int(n_surrogates)
    completed = 0

    while remaining > 0:
        if cancel_check is not None and cancel_check():
            raise CalculationCancelledError("XWT/WTC Monte Carlo cancelled before batch.")

        cur = min(eff_batch, remaining)
        sur_i = _generate_ar1_numpy(rng.standard_normal((cur, n_samples)), float(alpha1_i))
        sur_j = _generate_ar1_numpy(rng.standard_normal((cur, n_samples)), float(alpha1_j))
        sur_i -= np.mean(sur_i, axis=1, keepdims=True)
        sur_j -= np.mean(sur_j, axis=1, keepdims=True)

        hat_i = np.fft.fft(sur_i, n=n_pad_cwt, axis=1)
        hat_j = np.fft.fft(sur_j, n=n_pad_cwt, axis=1)
        del sur_i, sur_j

        # Compute w_i -> p_i -> t_pi and drop hat_i / p_i early to minimize peak RAM
        w_i = np.fft.ifft(hat_i[:, None, :] * daughters[None, :, :], axis=2)[:, :, :n_samples]
        del hat_i
        p_i = (w_i.real ** 2 + w_i.imag ** 2) * inv_s
        t_pi = np.fft.irfft(
            np.fft.rfft(p_i, n=n_pad_t, axis=2) * gauss_r[None, :, :], n=n_pad_t, axis=2
        )[:, :, :n_samples]
        del p_i

        w_j = np.fft.ifft(hat_j[:, None, :] * daughters[None, :, :], axis=2)[:, :, :n_samples]
        del hat_j
        w_ij = w_i * np.conj(w_j)
        del w_i
        w_ij *= inv_s
        p_j = (w_j.real ** 2 + w_j.imag ** 2) * inv_s
        del w_j

        t_pj = np.fft.irfft(
            np.fft.rfft(p_j, n=n_pad_t, axis=2) * gauss_r[None, :, :], n=n_pad_t, axis=2
        )[:, :, :n_samples]
        del p_j
        t_wij = np.fft.ifft(np.fft.fft(w_ij, n=n_pad_t, axis=2) * gauss_hat[None, :, :], axis=2)[
            :, :, :n_samples
        ]
        del w_ij

        for b_idx in range(cur):
            s_re = _smooth_scale_boxcar_numpy(t_wij[b_idx].real, width)
            s_im = _smooth_scale_boxcar_numpy(t_wij[b_idx].imag, width)
            s_ii = _smooth_scale_boxcar_numpy(t_pi[b_idx], width)
            s_jj = _smooth_scale_boxcar_numpy(t_pj[b_idx], width)
            r2 = np.clip((s_re ** 2 + s_im ** 2) / np.maximum(s_ii * s_jj, 1e-30), 0.0, 1.0)
            bin_idx = np.minimum(n_bins - 1, (r2 * float(n_bins)).astype(np.int64))
            for s_idx in range(n_scales):
                cols = bin_idx[s_idx][valid_t_masks[s_idx]]
                hist[s_idx] += np.bincount(cols, minlength=n_bins)

        del t_pi, t_pj, t_wij
        remaining -= cur
        completed += cur
        if progress_callback is not None:
            progress_callback(completed, int(n_surrogates), "monte_carlo")
        if cancel_check is not None and cancel_check():
            raise CalculationCancelledError("XWT/WTC Monte Carlo cancelled after batch.")

    sig_thresh = np.empty(n_scales, dtype=np.float64)
    target_q = 1.0 - float(alpha)
    for s_idx in range(n_scales):
        cdf = np.cumsum(hist[s_idx]).astype(np.float64)
        total = float(cdf[-1])
        if total <= 0.0:
            sig_thresh[s_idx] = 1.0
        else:
            cdf /= total
            pos = int(np.searchsorted(cdf, target_q, side="left"))
            sig_thresh[s_idx] = (float(pos) + 0.5) / float(n_bins)
    return sig_thresh


def build_wtc_cache_key(
    core_version: str,
    *,
    episode_id: str,
    pair: tuple[str, str],
    component: str,
    coord_frame: str,
    d0_deg: float,
    period_min_sec: float,
    period_max_sec: float,
    dj: float,
    omega0: float,
    gap_policy: str,
    n_surrogates: int,
    alpha: float,
    seed: int,
    qc_params: dict[str, Any] | None = None,
) -> str:
    """Build unified XWT/WTC cache key per ARCHITECTURE.md §4.2."""
    params = {
        "stage": "cross_wtc",
        "cache_format": WTC_CACHE_FORMAT,
        "episode_id": episode_id,
        "pair": [pair[0], pair[1]],
        "component": component,
        "coord_frame": coord_frame,
        "d0_deg": round(float(d0_deg), 6),
        "band_range": [float(period_min_sec), float(period_max_sec)],
        "scales": {"dj": float(dj)},
        "wavelet": {"name": "morlet", "omega0": float(omega0)},
        "smoothing_params": {"dj0": _morlet_dj0(omega0)},
        "gap_policy": gap_policy,
        "n_surrogates": int(n_surrogates),
        "alpha": float(alpha),
        "seed": int(seed),
        "qc_params": qc_params or {},
    }
    return canonical_cache_key(core_version, params)


def compute_xwt_wtc(
    x_i: np.ndarray,
    x_j: np.ndarray,
    *,
    dt: float = DT_1MIN_SEC,
    period_min_sec: float = 120.0,
    period_max_sec: float = 3600.0,
    dj: float = 1.0 / 12.0,
    omega0: float = MORLET_OMEGA0,
    gap_policy: str = "subsegments",
    min_subsegment_samples: int = 120,
    n_surrogates: int = 300,
    alpha: float = 0.05,
    seed: int = 42,
    batch_size: int | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
    core_version: str = "0.1.0",
    cache_params: dict[str, Any] | None = None,
) -> XWTWTCResult:
    """Compute Morlet XWT and WTC with NaN subsegment/gap-COI policy and batched AR(1) MC."""
    arr_i = np.asarray(x_i, dtype=np.float64)
    arr_j = np.asarray(x_j, dtype=np.float64)
    if arr_i.ndim != 1 or arr_i.shape != arr_j.shape:
        raise ValueError(f"Expected equal-length 1D arrays, got {arr_i.shape} and {arr_j.shape}")
    n_total = arr_i.shape[0]
    valid = np.isfinite(arr_i) & np.isfinite(arr_j)
    if not np.any(valid):
        raise ValueError("No jointly finite samples in station pair.")

    scales, periods_sec = build_morlet_scales(
        n_total,
        dt=dt,
        period_min_sec=period_min_sec,
        period_max_sec=period_max_sec,
        dj=dj,
        omega0=omega0,
    )
    n_scales = scales.shape[0]
    ff = _morlet_fourier_factor(omega0)

    w_i_full = np.full((n_scales, n_total), np.nan + 0j, dtype=np.complex128)
    w_j_full = np.full((n_scales, n_total), np.nan + 0j, dtype=np.complex128)
    w_ij_full = np.full((n_scales, n_total), np.nan + 0j, dtype=np.complex128)
    wtc_r2_full = np.full((n_scales, n_total), np.nan, dtype=np.float64)
    wtc_ph_full = np.full((n_scales, n_total), np.nan, dtype=np.float64)
    coi_full = np.zeros(n_total, dtype=np.float64)
    inside_coi_full = np.zeros((n_scales, n_total), dtype=bool)

    policy = gap_policy.strip().lower()
    if policy == "subsegments":
        subsegments = tuple(find_valid_subsegments(valid, min_length=int(min_subsegment_samples)))
        if not subsegments:
            raise ValueError(
                f"No contiguous finite subsegments of length >= {min_subsegment_samples}."
            )
        for s_idx, e_idx in subsegments:
            wi, wj, wij, r2, ph, coi_p = _compute_xwt_wtc_segment(
                arr_i[s_idx:e_idx],
                arr_j[s_idx:e_idx],
                scales,
                dt=dt,
                dj=dj,
                omega0=omega0,
            )
            w_i_full[:, s_idx:e_idx] = wi
            w_j_full[:, s_idx:e_idx] = wj
            w_ij_full[:, s_idx:e_idx] = wij
            wtc_r2_full[:, s_idx:e_idx] = r2
            wtc_ph_full[:, s_idx:e_idx] = ph
            coi_full[s_idx:e_idx] = coi_p
            inside_coi_full[:, s_idx:e_idx] = periods_sec[:, None] <= coi_p[None, :]
        mc_len = max(e_idx - s_idx for s_idx, e_idx in subsegments)
    elif policy == "gap_coi":
        subsegments = ((0, n_total),)
        bridge_i = _bridge_gaps_linear(arr_i, valid)
        bridge_j = _bridge_gaps_linear(arr_j, valid)
        wi, wj, wij, r2, ph, _ = _compute_xwt_wtc_segment(
            bridge_i,
            bridge_j,
            scales,
            dt=dt,
            dj=dj,
            omega0=omega0,
        )
        w_i_full[:] = wi
        w_j_full[:] = wj
        w_ij_full[:] = wij
        wtc_r2_full[:] = r2
        wtc_ph_full[:] = ph
        coi_full[:] = compute_gap_coi_scales(valid, dt=dt) * ff
        inside_coi_full[:] = periods_sec[:, None] <= coi_full[None, :]
        gap_cols = ~valid
        if np.any(gap_cols):
            w_i_full[:, gap_cols] = np.nan + 0j
            w_j_full[:, gap_cols] = np.nan + 0j
            w_ij_full[:, gap_cols] = np.nan + 0j
            wtc_r2_full[:, gap_cols] = np.nan
            wtc_ph_full[:, gap_cols] = np.nan
            inside_coi_full[:, gap_cols] = False
        mc_len = n_total
    else:
        raise ValueError(
            f"Unsupported gap_policy {gap_policy!r}; expected 'subsegments' or 'gap_coi'."
        )

    masked_i = np.where(valid, arr_i, np.nan)
    masked_j = np.where(valid, arr_j, np.nan)
    alpha1_i = estimate_ar1(masked_i)
    alpha1_j = estimate_ar1(masked_j)

    if n_surrogates > 0:
        wtc_sig95 = _run_wtc_monte_carlo(
            mc_len,
            scales,
            alpha1_i,
            alpha1_j,
            dt=dt,
            dj=dj,
            omega0=omega0,
            n_surrogates=n_surrogates,
            alpha=alpha,
            seed=seed,
            batch_size=batch_size,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )
        wtc_sig = inside_coi_full & np.isfinite(wtc_r2_full) & (wtc_r2_full >= wtc_sig95[:, None])
    else:
        wtc_sig95 = None
        wtc_sig = np.zeros((n_scales, n_total), dtype=bool)

    xwt_power = np.abs(w_ij_full)
    xwt_phase_rad = np.where(np.isfinite(xwt_power), np.angle(w_ij_full), np.nan)
    xwt_phase_deg = np.rad2deg(xwt_phase_rad)
    wtc_phase_deg = np.rad2deg(wtc_ph_full)

    key_params = cache_params or {
        "stage": "cross_wtc",
        "cache_format": WTC_CACHE_FORMAT,
        "n_total": n_total,
        "dt": dt,
        "period_min_sec": period_min_sec,
        "period_max_sec": period_max_sec,
        "dj": dj,
        "omega0": omega0,
        "gap_policy": policy,
        "n_surrogates": n_surrogates,
        "alpha": alpha,
        "seed": seed,
    }
    cache_key = canonical_cache_key(core_version, key_params)

    return XWTWTCResult(
        scales=scales,
        periods_sec=periods_sec,
        coi_periods_sec=coi_full,
        inside_coi=inside_coi_full,
        W_i=w_i_full,
        W_j=w_j_full,
        W_ij=w_ij_full,
        xwt_power=xwt_power,
        xwt_phase_rad=xwt_phase_rad,
        xwt_phase_deg=xwt_phase_deg,
        wtc_r2=wtc_r2_full,
        wtc_phase_rad=wtc_ph_full,
        wtc_phase_deg=wtc_phase_deg,
        alpha1_i=alpha1_i,
        alpha1_j=alpha1_j,
        wtc_sig95=wtc_sig95,
        wtc_is_significant=wtc_sig,
        gap_policy=policy,
        subsegments=subsegments,
        n_surrogates=int(n_surrogates),
        alpha=float(alpha),
        cache_key=cache_key,
    )


def save_wtc_cache(path: Path, result: XWTWTCResult) -> None:
    """Persist compact display/export XWT/WTC arrays and metadata to compressed NPZ."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sub_arr = np.asarray(result.subsegments, dtype=np.int64)
    sig95 = np.array([]) if result.wtc_sig95 is None else result.wtc_sig95
    np.savez_compressed(
        path,
        scales=result.scales,
        periods_sec=result.periods_sec,
        coi_periods_sec=result.coi_periods_sec,
        inside_coi=result.inside_coi,
        xwt_power=np.asarray(result.xwt_power, dtype=np.float32),
        xwt_phase_deg=np.asarray(result.xwt_phase_deg, dtype=np.float32),
        wtc_r2=np.asarray(result.wtc_r2, dtype=np.float32),
        wtc_phase_deg=np.asarray(result.wtc_phase_deg, dtype=np.float32),
        alpha1_i=np.array(result.alpha1_i, dtype=np.float64),
        alpha1_j=np.array(result.alpha1_j, dtype=np.float64),
        wtc_sig95=sig95,
        wtc_is_significant=result.wtc_is_significant,
        gap_policy=np.array(result.gap_policy),
        subsegments=sub_arr,
        n_surrogates=np.array(result.n_surrogates, dtype=np.int64),
        alpha=np.array(result.alpha, dtype=np.float64),
        cache_key=np.array(result.cache_key),
    )


def load_wtc_cache(path: Path) -> XWTWTCResult:
    """Load compact XWTWTCResult from compressed NPZ (W_i/W_j/W_ij omitted; phases recomputed)."""
    with np.load(path, allow_pickle=False) as data:
        sig95_raw = data["wtc_sig95"]
        wtc_sig95 = None if sig95_raw.size == 0 else sig95_raw
        sub_tuples = tuple((int(r[0]), int(r[1])) for r in data["subsegments"])
        xwt_phase_deg = np.asarray(data["xwt_phase_deg"], dtype=np.float64)
        wtc_phase_deg = np.asarray(data["wtc_phase_deg"], dtype=np.float64)
        return XWTWTCResult(
            scales=data["scales"],
            periods_sec=data["periods_sec"],
            coi_periods_sec=data["coi_periods_sec"],
            inside_coi=data["inside_coi"],
            W_i=None,
            W_j=None,
            W_ij=None,
            xwt_power=np.asarray(data["xwt_power"], dtype=np.float64),
            xwt_phase_rad=np.deg2rad(xwt_phase_deg),
            xwt_phase_deg=xwt_phase_deg,
            wtc_r2=np.asarray(data["wtc_r2"], dtype=np.float64),
            wtc_phase_rad=np.deg2rad(wtc_phase_deg),
            wtc_phase_deg=wtc_phase_deg,
            alpha1_i=float(data["alpha1_i"].item()),
            alpha1_j=float(data["alpha1_j"].item()),
            wtc_sig95=wtc_sig95,
            wtc_is_significant=data["wtc_is_significant"],
            gap_policy=str(data["gap_policy"].item()),
            subsegments=sub_tuples,
            n_surrogates=int(data["n_surrogates"].item()),
            alpha=float(data["alpha"].item()),
            cache_key=str(data["cache_key"].item()),
        )
