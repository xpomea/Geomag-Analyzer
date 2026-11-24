"""Variant A spectral matrix estimation: Welch (Hann + OLS detrend) & Thomson Multitaper (§4)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.signal.windows import dpss

from .conventions import DT_1MIN_SEC, get_band_spec, is_window_valid


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


@lru_cache(maxsize=64)
def _welch_gram_invariants(
    L: int,
    M: int,
    dt: float,
    f_min_hz: float,
    f_max_hz: float,
    is_red: bool,
) -> tuple[float, float, tuple[tuple[int, float], ...]]:
    """Precompute exact Block-Toeplitz Gram invariants for Welch effective DOF.

    Builds A = (E_B * w) @ P_det for OLS linear detrending, periodic Hann window w,
    and DFT bins in [f_min_hz, f_max_hz], then evaluates the trace and Frobenius
    norms of G_r = A C_lag(r) A^* across all overlapping segment lags r * M < L.
    """
    freqs = np.fft.rfftfreq(L, d=dt)
    f_mask = (freqs >= f_min_hz * (1.0 - 1e-12)) & (freqs <= f_max_hz * (1.0 + 1e-12))
    k_bins = np.where(f_mask)[0]
    if k_bins.size == 0:
        raise ValueError(
            f"No FFT bins in [{f_min_hz}, {f_max_hz}] Hz for L={L}, dt={dt}"
        )

    t_c = np.arange(L, dtype=np.float64) - 0.5 * (L - 1.0)
    slope_norm = float(np.dot(t_c, t_c)) if L > 1 else 1.0
    p_det = np.eye(L, dtype=np.float64) - 1.0 / float(L)
    if L > 1:
        p_det = p_det - np.outer(t_c, t_c) / slope_norm

    win = periodic_hann_window(L)
    n_idx = np.arange(L, dtype=np.float64)
    e_b = np.exp(-2j * np.pi * np.outer(k_bins, n_idx) / float(L))
    op_a = (e_b * win[None, :]) @ p_det

    max_r = max(0, (L - 1) // M)
    if is_red:
        u = n_idx[:, None]
        v = n_idx[None, :]

    def _lag_cov(r: int) -> np.ndarray:
        shift = r * M
        if not is_red:
            c_mat = np.zeros((L, L), dtype=np.float64)
            if shift < L:
                idx = np.arange(L - shift)
                c_mat[idx + shift, idx] = 1.0
            return c_mat
        # For Brownian motion, the random-walk constant shift m*M is annihilated by P_det.
        return np.minimum(u, float(shift) + v)

    g0 = op_a @ _lag_cov(0) @ op_a.conj().T
    tr0 = float(np.real(np.trace(g0)))
    frob0_sq = float(np.sum(np.abs(g0) ** 2))
    lag_frob_sq: list[tuple[int, float]] = []
    for r in range(1, max_r + 1):
        gr = op_a @ _lag_cov(r) @ op_a.conj().T
        lag_frob_sq.append((r, float(np.sum(np.abs(gr) ** 2))))
    return tr0, frob0_sq, tuple(lag_frob_sq)


def welch_effective_dof(
    K: np.ndarray | int,
    *,
    L: int = 128,
    M: int = 64,
    dt: float = DT_1MIN_SEC,
    band: str = "Pc5",
    spectrum_type: str = "flat",
    t_max_pc6_sec: float = 1800.0,
    f_min_hz: float | None = None,
    f_max_hz: float | None = None,
) -> np.ndarray:
    """Compute exact Block-Toeplitz Gram-matrix Welch effective DOF n_eff (§4.4)."""
    if L <= 0 or M <= 0 or dt <= 0.0:
        raise ValueError(f"L, M, and dt must be positive, got L={L}, M={M}, dt={dt}")

    s_key = spectrum_type.strip().lower()
    if s_key in ("flat", "white"):
        is_red = False
    elif s_key in ("red", "f-2", "f^-2", "brownian", "random_walk"):
        is_red = True
    else:
        raise ValueError(f"Unsupported spectrum_type {spectrum_type!r}")

    spec = get_band_spec(
        band=band,
        t_max_pc6_sec=t_max_pc6_sec,
        f_min_hz=f_min_hz,
        f_max_hz=f_max_hz,
    )
    tr0, frob0_sq, lag_frob_sq = _welch_gram_invariants(
        int(L),
        int(M),
        float(dt),
        spec.f_min_hz,
        spec.f_max_hz,
        is_red,
    )

    k_arr = np.asarray(K, dtype=np.float64)
    denom = k_arr * frob0_sq
    for r, fr2 in lag_frob_sq:
        denom = denom + 2.0 * np.maximum(0.0, k_arr - float(r)) * fr2
    num = (k_arr * tr0) ** 2
    return np.where(k_arr > 0.0, num / np.maximum(denom, 1e-30), 0.0)


@lru_cache(maxsize=64)
def _multitaper_effective_dof_cached(
    n_samples: int,
    nw: float,
    k_tapers: int,
    dt: float,
    f_min_hz: float,
    f_max_hz: float,
    is_red: bool,
) -> float:
    """Compute exact Gram-matrix effective DOF n_eff for Thomson DPSS multitaper."""
    tapers = dpss(n_samples, NW=nw, Kmax=k_tapers)  # (K_t, N)
    freqs = np.fft.rfftfreq(n_samples, d=dt)
    f_mask = (freqs >= f_min_hz * (1.0 - 1e-12)) & (freqs <= f_max_hz * (1.0 + 1e-12))
    k_bins = np.where(f_mask)[0]
    n_b = int(k_bins.size)
    if n_b == 0:
        raise ValueError(
            f"No FFT bins in [{f_min_hz}, {f_max_hz}] Hz for N={n_samples}, dt={dt}"
        )

    if not is_red:
        mask_full = np.zeros(n_samples, dtype=np.float64)
        mask_full[k_bins] = 1.0
        h_b = np.fft.ifft(mask_full) * float(n_samples)
        h_b_abs2 = h_b.real**2 + h_b.imag**2
        tr2 = 0.0
        for q in range(k_tapers):
            for p in range(q, k_tapers):
                w_qp = tapers[q] * tapers[p]
                fw = np.fft.rfft(w_qp)
                c_qp = np.fft.irfft(fw.real**2 + fw.imag**2, n=n_samples)
                val = float(np.dot(h_b_abs2, c_qp))
                tr2 += val if q == p else 2.0 * val
        tr = float(n_b * k_tapers)
        return (tr * tr) / max(tr2, 1e-30)

    # Red noise (Brownian motion under OLS linear detrending):
    # P_det C_BM P_det == P_det C_circ P_det with S_BM[m] = 1 / (4 sin^2(pi f_m dt)).
    t_c = np.arange(n_samples, dtype=np.float64) - 0.5 * (n_samples - 1.0)
    u1 = t_c / max(float(np.linalg.norm(t_c)), 1e-30)
    w_fft = np.fft.fft(tapers, axis=1)
    u1_fft = np.fft.fft(u1)
    c_qk = np.fft.fft(tapers * u1[None, :], axis=1)[:, k_bins]

    f_all = np.fft.fftfreq(n_samples, d=dt)
    s_bm = np.zeros(n_samples, dtype=np.float64)
    nz = f_all != 0.0
    s_bm[nz] = 1.0 / (4.0 * np.sin(np.pi * f_all[nz] * dt) ** 2)
    sqrt_s = np.sqrt(s_bm) / math.sqrt(float(n_samples))

    if n_samples <= 256:
        m_all = np.arange(1, n_samples, dtype=np.int64)
        diff_all = (k_bins[:, None] - m_all[None, :]) % n_samples
        b_full = (
            w_fft[:, diff_all] - c_qk[:, :, None] * u1_fft[m_all].conj()[None, None, :]
        ) * sqrt_s[m_all][None, None, :]
        b_2d = b_full.reshape(k_tapers * n_b, m_all.size)
        tr_tot = float(np.sum(b_2d.real**2 + b_2d.imag**2))
        gram = b_2d.conj().T @ b_2d
        tr2_tot = float(np.sum(gram.real**2 + gram.imag**2))
        return (tr_tot * tr_tot) / max(tr2_tot, 1e-30)

    m_low_lim = min(n_samples // 4, 64)
    low_mask = np.zeros(n_samples, dtype=bool)
    low_mask[1 : m_low_lim + 1] = True
    low_mask[n_samples - m_low_lim :] = True
    m_low = np.where(low_mask)[0]

    diff_low = (k_bins[:, None] - m_low[None, :]) % n_samples
    b_low = (
        w_fft[:, diff_low] - c_qk[:, :, None] * u1_fft[m_low].conj()[None, None, :]
    ) * sqrt_s[m_low][None, None, :]
    b_low_2d = b_low.reshape(k_tapers * n_b, m_low.size)
    tr_low = float(np.sum(b_low_2d.real**2 + b_low_2d.imag**2))
    gram_low = b_low_2d.conj().T @ b_low_2d
    tr2_low = float(np.sum(gram_low.real**2 + gram_low.imag**2))

    half_sup = int(math.ceil(nw)) + 8
    m_offs = np.arange(-half_sup, half_sup + 1, dtype=np.int64)
    idx_loc = (k_bins[:, None] + m_offs[None, :]) % n_samples
    in_band_only = (~low_mask[idx_loc]) & (idx_loc != 0)
    s_loc = np.where(in_band_only, s_bm[idx_loc] / float(n_samples), 0.0)

    w_loc = w_fft[:, (-m_offs) % n_samples]
    w_abs2_sum = np.sum(w_loc.real**2 + w_loc.imag**2, axis=0)
    tr_band = float(np.sum(s_loc @ w_abs2_sum))

    d_max = min(n_b - 1, 2 * half_sup)
    tr2_band = 0.0
    for d in range(0, d_max + 1):
        if d >= m_offs.size:
            break
        wq = w_loc[:, d:]
        wp = w_loc[:, : m_offs.size - d].conj()
        w_qp = (wq[:, None, :] * wp[None, :, :]).reshape(k_tapers * k_tapers, -1)
        s_ov = s_loc[: n_b - d, d:]
        g_d = s_ov @ w_qp.T
        frob2 = float(np.sum(g_d.real**2 + g_d.imag**2))
        tr2_band += frob2 if d == 0 else 2.0 * frob2

    m_min = max(1, int(k_bins[0]) - half_sup)
    m_max = min(n_samples - 1, int(k_bins[-1]) + half_sup)
    m_band_len = m_max - m_min + 1
    acc = np.zeros((m_band_len, m_low.size), dtype=np.complex128)
    k0_shift = int(k_bins[0]) - m_min
    for u, off in enumerate(m_offs):
        pos_start = k0_shift + int(off)
        i_start = max(0, -pos_start)
        i_end = min(n_b, m_band_len - pos_start)
        if i_end <= i_start:
            continue
        contrib = np.tensordot(w_loc[:, u].conj(), b_low[:, i_start:i_end, :], axes=([0], [0]))
        acc[pos_start + i_start : pos_start + i_end] += contrib
    m_band_vals = np.arange(m_min, m_max + 1, dtype=np.int64)
    band_valid = (~low_mask[m_band_vals]) & (m_band_vals != 0)
    acc *= np.where(band_valid, sqrt_s[m_band_vals], 0.0)[:, None]
    tr2_cross = float(np.sum(acc.real**2 + acc.imag**2))

    tr_tot = tr_low + tr_band
    tr2_tot = tr2_low + 2.0 * tr2_cross + tr2_band
    return (tr_tot * tr_tot) / max(tr2_tot, 1e-30)


def multitaper_effective_dof(
    n_samples: int,
    *,
    NW: float = 2.5,
    K_tapers: int | None = None,
    dt: float = DT_1MIN_SEC,
    band: str = "Pc5",
    spectrum_type: str = "flat",
    t_max_pc6_sec: float = 1800.0,
    f_min_hz: float | None = None,
    f_max_hz: float | None = None,
) -> float:
    """Compute exact Gram-matrix effective DOF n_eff for Thomson DPSS multitaper (§4.4)."""
    if n_samples <= 0 or NW <= 0.0 or dt <= 0.0:
        raise ValueError(
            f"n_samples, NW, and dt must be positive, got n_samples={n_samples}, NW={NW}, dt={dt}"
        )
    k_t = max(1, int(round(2.0 * NW)) - 1) if K_tapers is None else int(K_tapers)
    if k_t < 1:
        raise ValueError(f"K_tapers must be >= 1, got {k_t}")

    s_key = spectrum_type.strip().lower()
    if s_key in ("flat", "white"):
        is_red = False
    elif s_key in ("red", "f-2", "f^-2", "brownian", "random_walk"):
        is_red = True
    else:
        raise ValueError(f"Unsupported spectrum_type {spectrum_type!r}")

    spec = get_band_spec(
        band=band,
        t_max_pc6_sec=t_max_pc6_sec,
        f_min_hz=f_min_hz,
        f_max_hz=f_max_hz,
    )
    return _multitaper_effective_dof_cached(
        int(n_samples),
        float(NW),
        int(k_t),
        float(dt),
        float(spec.f_min_hz),
        float(spec.f_max_hz),
        is_red,
    )


def lookup_welch_effective_n(
    n_samples: int,
    *,
    band: str = "Pc5",
    spectrum_type: str = "flat",
    L: int = 128,
    M: int = 64,
    dt: float = DT_1MIN_SEC,
    t_max_pc6_sec: float = 1800.0,
    f_min_hz: float | None = None,
    f_max_hz: float | None = None,
) -> float:
    """Compute Welch effective DOF n for a window of n_samples from first principles (§4.4)."""
    if L <= 0 or M <= 0:
        raise ValueError(f"L and M must be positive, got L={L}, M={M}")
    k_seg = max(1, (int(n_samples) - int(L)) // int(M) + 1)
    n_eff = welch_effective_dof(
        k_seg,
        L=L,
        M=M,
        dt=dt,
        band=band,
        spectrum_type=spectrum_type,
        t_max_pc6_sec=t_max_pc6_sec,
        f_min_hz=f_min_hz,
        f_max_hz=f_max_hz,
    )
    return float(n_eff.item())


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
