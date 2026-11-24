"""Variant A spectral matrix estimation: Welch (Hann + OLS detrend) & Thomson Multitaper (§4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal.windows import dpss

from .conventions import DT_1MIN_SEC, get_band_spec, is_window_valid

# Reference Table 3 effective degrees of freedom (L=128, M=64, Hann, OLS linear detrend)
_TABLE_3_N_EFF: dict[tuple[int, str, str], int] = {
    (256, "Pc5", "flat"): 56,
    (256, "Pc5", "red"): 38,
    (256, "Pc6", "flat"): 13,
    (256, "Pc6", "red"): 11,
    (360, "Pc5", "flat"): 75,
    (360, "Pc5", "red"): 53,
    (360, "Pc6", "flat"): 17,
    (360, "Pc6", "red"): 14,
    (720, "Pc5", "flat"): 190,
    (720, "Pc5", "red"): 135,
    (720, "Pc6", "flat"): 42,
    (720, "Pc6", "red"): 34,
    (1440, "Pc5", "flat"): 395,
    (1440, "Pc5", "red"): 263,
    (1440, "Pc6", "flat"): 87,
    (1440, "Pc6", "red"): 68,
}


@dataclass(frozen=True)
class WelchResult:
    """Result of band-averaged Welch spectral matrix estimation."""

    S: np.ndarray
    K_total: int
    K_valid: int
    N_B: int
    L: int
    M: int
    band: str
    freqs_hz: np.ndarray
    valid_segment_mask: np.ndarray
    is_valid: bool


@dataclass(frozen=True)
class MultitaperResult:
    """Result of Thomson DPSS multitaper spectral matrix estimation."""

    S: np.ndarray
    K_tapers: int
    N_B: int
    N: int
    NW: float
    band: str
    freqs_hz: np.ndarray


def detrend_linear_ols(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Remove an OLS straight line along `axis` without matrix inversions (Formula 7)."""
    arr = np.asarray(x, dtype=np.float64)
    n = arr.shape[axis]
    if n <= 1:
        return np.zeros_like(arr, dtype=np.float64)
    t_centered = np.arange(n, dtype=np.float64) - 0.5 * (n - 1.0)
    denom = float(np.dot(t_centered, t_centered))
    shape = [1] * arr.ndim
    shape[axis] = n
    t_b = t_centered.reshape(shape)
    mean_x = np.mean(arr, axis=axis, keepdims=True)
    x_c = arr - mean_x
    slope = np.sum(x_c * t_b, axis=axis, keepdims=True) / denom
    return x_c - slope * t_b


def periodic_hann_window(L: int) -> np.ndarray:
    """Return periodic Hann window w_n = 0.5 * (1 - cos(2*pi*n/L)) (Formula 7)."""
    if L <= 0:
        raise ValueError(f"Segment length L must be positive, got {L}")
    n = np.arange(L, dtype=np.float64)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * n / float(L)))


def band_frequency_mask(
    freqs_hz: np.ndarray,
    *,
    band: str = "Pc5",
    t_max_pc6_sec: float = 1800.0,
    f_min_hz: float | None = None,
    f_max_hz: float | None = None,
) -> np.ndarray:
    """Boolean mask selecting frequency bins inside [f1, f2] (Formula 1 & §4.2)."""
    spec = get_band_spec(
        band=band,
        t_max_pc6_sec=t_max_pc6_sec,
        f_min_hz=f_min_hz,
        f_max_hz=f_max_hz,
    )
    f = np.asarray(freqs_hz, dtype=np.float64)
    return (f >= spec.f_min_hz * (1.0 - 1e-12)) & (f <= spec.f_max_hz * (1.0 + 1e-12))


def compute_welch_spectral_matrix(
    xyz: np.ndarray,
    *,
    dt: float = DT_1MIN_SEC,
    band: str = "Pc5",
    L: int = 128,
    M: int = 64,
    t_max_pc6_sec: float = 1800.0,
    f_min_hz: float | None = None,
    f_max_hz: float | None = None,
    is_valid_xyz: np.ndarray | None = None,
    max_missing_frac: float = 0.05,
    max_gap: int = 3,
    min_valid_segments: int = 1,
    scale_psd: bool = True,
) -> WelchResult:
    """Compute band-averaged Welch spectral matrix S_ij (Formulas 6-9)."""
    arr = np.asarray(xyz, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError(f"Expected (N, 2) or (N, 3) array, got shape {arr.shape}")
    n_samples, n_comp = arr.shape
    if L <= 0 or M <= 0:
        raise ValueError(f"L and M must be positive, got L={L}, M={M}")

    freqs = np.fft.rfftfreq(L, d=dt)
    f_mask = band_frequency_mask(
        freqs,
        band=band,
        t_max_pc6_sec=t_max_pc6_sec,
        f_min_hz=f_min_hz,
        f_max_hz=f_max_hz,
    )
    band_freqs = freqs[f_mask]
    n_b = int(np.sum(f_mask))
    if n_b == 0:
        raise ValueError(f"No FFT bins in band {band!r} for L={L}, dt={dt}")

    if n_samples < L:
        return WelchResult(
            S=np.full((n_comp, n_comp), np.nan + 0j, dtype=np.complex128),
            K_total=0,
            K_valid=0,
            N_B=n_b,
            L=L,
            M=M,
            band=band,
            freqs_hz=band_freqs,
            valid_segment_mask=np.zeros(0, dtype=bool),
            is_valid=False,
        )

    k_total = (n_samples - L) // M + 1
    starts = np.arange(k_total, dtype=np.int64) * M
    idx = starts[:, None] + np.arange(L, dtype=np.int64)[None, :]

    segs = arr[idx]  # (K_total, L, C)
    seg_finite = np.all(np.isfinite(segs), axis=2)  # (K_total, L)
    if is_valid_xyz is None:
        valid_mask = np.all(seg_finite, axis=1)
    else:
        raw_valid = np.asarray(is_valid_xyz, dtype=bool)
        if raw_valid.shape[0] != n_samples:
            raise ValueError("is_valid_xyz length must match xyz length.")
        raw_segs = raw_valid[idx]
        valid_mask = np.array(
            [
                is_window_valid(
                    raw_segs[m],
                    seg_finite[m],
                    max_missing_frac=max_missing_frac,
                    max_gap=max_gap,
                )
                for m in range(k_total)
            ],
            dtype=bool,
        )

    k_valid = int(np.sum(valid_mask))
    if k_valid < min_valid_segments:
        return WelchResult(
            S=np.full((n_comp, n_comp), np.nan + 0j, dtype=np.complex128),
            K_total=k_total,
            K_valid=k_valid,
            N_B=n_b,
            L=L,
            M=M,
            band=band,
            freqs_hz=band_freqs,
            valid_segment_mask=valid_mask,
            is_valid=False,
        )

    valid_segs = detrend_linear_ols(segs[valid_mask], axis=1)
    win = periodic_hann_window(L)
    windowed = valid_segs * win[None, :, None]
    fft_full = np.fft.rfft(windowed, axis=1)
    fft_band = fft_full[:, f_mask, :]  # (K_valid, N_B, C)

    # Formula (9): S_ij = (1 / (K * N_B)) * sum_{m, k in B} F_{i,m}(k) F_{j,m}^*(k)
    s_mat = np.einsum("kmc,kmd->cd", fft_band, np.conj(fft_band)) / float(k_valid * n_b)
    if scale_psd:
        s_mat = s_mat * (2.0 * dt / float(np.sum(win ** 2)))

    return WelchResult(
        S=s_mat,
        K_total=k_total,
        K_valid=k_valid,
        N_B=n_b,
        L=L,
        M=M,
        band=band,
        freqs_hz=band_freqs,
        valid_segment_mask=valid_mask,
        is_valid=True,
    )


def compute_multitaper_spectral_matrix(
    xyz: np.ndarray,
    *,
    dt: float = DT_1MIN_SEC,
    band: str = "Pc5",
    NW: float = 2.5,
    K_tapers: int | None = None,
    t_max_pc6_sec: float = 1800.0,
    f_min_hz: float | None = None,
    f_max_hz: float | None = None,
    detrend: bool = True,
    scale_psd: bool = True,
) -> MultitaperResult:
    """Compute Thomson DPSS multitaper spectral matrix (Formula 10)."""
    arr = np.asarray(xyz, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError(f"Expected (N, 2) or (N, 3) array, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Multitaper input contains NaN or Inf values.")

    n_samples = arr.shape[0]
    k_t = max(1, int(round(2.0 * NW)) - 1) if K_tapers is None else int(K_tapers)
    tapers = dpss(n_samples, NW=NW, Kmax=k_t)  # (K_t, N)

    work = detrend_linear_ols(arr, axis=0) if detrend else arr
    tapered = tapers[:, :, None] * work[None, :, :]  # (K_t, N, C)

    freqs = np.fft.rfftfreq(n_samples, d=dt)
    f_mask = band_frequency_mask(
        freqs,
        band=band,
        t_max_pc6_sec=t_max_pc6_sec,
        f_min_hz=f_min_hz,
        f_max_hz=f_max_hz,
    )
    n_b = int(np.sum(f_mask))
    if n_b == 0:
        raise ValueError(f"No FFT bins in band {band!r} for N={n_samples}, dt={dt}")

    fft_band = np.fft.rfft(tapered, axis=1)[:, f_mask, :]
    s_mat = np.einsum("qkc,qkd->cd", fft_band, np.conj(fft_band)) / float(k_t * n_b)
    if scale_psd:
        s_mat = s_mat * (2.0 * dt)

    return MultitaperResult(
        S=s_mat,
        K_tapers=k_t,
        N_B=n_b,
        N=n_samples,
        NW=NW,
        band=band,
        freqs_hz=freqs[f_mask],
    )


def welch_segment_dof(K_ij: np.ndarray | int, *, L: int = 128, M: int = 64) -> np.ndarray:
    """Overlap-corrected effective DOF n_seg for single-frequency Welch coherence."""
    k_arr = np.asarray(K_ij, dtype=np.float64)
    out = np.zeros_like(k_arr, dtype=np.float64)
    win = periodic_hann_window(L)
    norm_w = float(np.sum(win ** 2))

    # Precompute lag correlations rho(r * M) for lags where r*M < L
    max_r = max(0, (L - 1) // M)
    rho_sq = []
    for r in range(1, max_r + 1):
        shift = r * M
        overlap = float(np.sum(win[: L - shift] * win[shift:])) / norm_w
        rho_sq.append((r, overlap * overlap))

    it = np.nditer(k_arr, flags=["multi_index"])
    for val in it:
        k_val = int(round(float(val)))
        if k_val <= 1:
            out[it.multi_index] = float(k_val)
            continue
        denom = 1.0
        for r, r2 in rho_sq:
            if r < k_val:
                denom += 2.0 * (1.0 - float(r) / float(k_val)) * r2
        out[it.multi_index] = float(k_val) / denom
    return out


def lookup_welch_effective_n(
    n_samples: int,
    *,
    band: str = "Pc5",
    spectrum_type: str = "flat",
    L: int = 128,
    M: int = 64,
    dt: float = DT_1MIN_SEC,
    t_max_pc6_sec: float = 1800.0,
) -> float:
    """Return Table 3 effective DOF n (or overlap-corrected §4.4 estimate for custom K/M)."""
    band_key = "Pc5" if band.strip().upper() == "PC5" else "Pc6"
    spec_key = "flat" if spectrum_type.strip().lower() in ("flat", "white") else "red"
    if L == 128 and M == 64 and dt == 60.0 and t_max_pc6_sec == 1800.0:
        exact = _TABLE_3_N_EFF.get((n_samples, band_key, spec_key))
        if exact is not None:
            return float(exact)

    k_seg = max(1, (n_samples - L) // M + 1)
    k_eff = float(welch_segment_dof(k_seg, L=L, M=M).item())
    freqs = np.fft.rfftfreq(L, d=dt)
    n_b = int(np.sum(band_frequency_mask(freqs, band=band, t_max_pc6_sec=t_max_pc6_sec)))
    if spec_key == "flat":
        factor = 0.5
    else:
        factor = 0.34 if band_key == "Pc5" else 0.42
    return float(max(2.0, factor * k_eff * n_b))


def calibrate_welch_effective_n_mc(
    n_samples: int,
    *,
    band: str = "Pc5",
    L: int = 128,
    M: int = 64,
    dt: float = DT_1MIN_SEC,
    spectrum_type: str = "flat",
    n_trials: int = 2000,
    seed: int = 42,
    t_max_pc6_sec: float = 1800.0,
    batch_size: int = 500,
) -> float:
    """Calibrate Welch effective DOF n via Monte Carlo: n = 0.5 * (3 / E[p^2] - 1) (§4.4)."""
    if n_samples < L:
        raise ValueError(f"n_samples ({n_samples}) must be >= L ({L})")

    rng = np.random.default_rng(seed)
    k_total = (n_samples - L) // M + 1
    starts = np.arange(k_total, dtype=np.int64) * M
    seg_idx = starts[:, None] + np.arange(L, dtype=np.int64)[None, :]

    win = periodic_hann_window(L)
    freqs_seg = np.fft.rfftfreq(L, d=dt)
    f_mask = band_frequency_mask(freqs_seg, band=band, t_max_pc6_sec=t_max_pc6_sec)

    spec_key = spectrum_type.strip().lower()
    use_cumsum = spec_key in ("red", "brownian", "random_walk")
    if spec_key in ("f-2", "f^-2", "spectral_red"):
        freqs_full = np.fft.rfftfreq(n_samples, d=dt)
        shape_amp = np.zeros_like(freqs_full)
        shape_amp[1:] = 1.0 / freqs_full[1:]
    else:
        shape_amp = None

    p2_list: list[np.ndarray] = []
    remaining = int(n_trials)
    while remaining > 0:
        cur = min(batch_size, remaining)
        remaining -= cur

        noise = rng.standard_normal((cur, n_samples, 2))
        if use_cumsum:
            noise = np.cumsum(noise, axis=1)
        elif shape_amp is not None:
            spec = np.fft.rfft(noise, axis=1) * shape_amp[None, :, None]
            noise = np.fft.irfft(spec, n=n_samples, axis=1)

        segs = noise[:, seg_idx, :]  # (cur, K, L, 2)
        segs = detrend_linear_ols(segs, axis=2) * win[None, None, :, None]
        fft_band = np.fft.rfft(segs, axis=2)[:, :, f_mask, :]  # (cur, K, N_B, 2)

        s_xx = np.mean(np.abs(fft_band[..., 0]) ** 2, axis=(1, 2))
        s_yy = np.mean(np.abs(fft_band[..., 1]) ** 2, axis=(1, 2))
        s_xy = np.mean(fft_band[..., 0] * np.conj(fft_band[..., 1]), axis=(1, 2))

        i_val = s_xx + s_yy
        q_val = s_xx - s_yy
        u_val = 2.0 * np.real(s_xy)
        v_val = 2.0 * np.imag(s_xy)
        r2 = q_val ** 2 + u_val ** 2 + v_val ** 2
        p2_list.append(r2 / (i_val ** 2))

    mean_p2 = float(np.mean(np.concatenate(p2_list)))
    return 0.5 * (3.0 / mean_p2 - 1.0)
