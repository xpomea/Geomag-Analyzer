"""Exact oversampling and odd-reflection boundary padding (§7.4)."""

from __future__ import annotations

import math

import numpy as np
from scipy.special import i0


def odd_reflect_pad(x: np.ndarray, pad: int) -> np.ndarray:
    """Apply odd (point-symmetric) boundary extension by `pad` samples on each end (Formula 33).

    Supports 1D (N,) and 2D (N, C) arrays along axis 0. Preserves C1 continuity
    and linear trends. If `pad >= N`, applies iterative reflections cleanly.
    """
    if pad < 0:
        raise ValueError(f"pad must be non-negative, got {pad}")
    arr = np.asarray(x, dtype=np.float64)
    if pad == 0:
        return arr.copy()

    squeeze_1d = arr.ndim == 1
    work = arr[:, None].copy() if squeeze_1d else arr.copy()
    if work.shape[0] == 0:
        raise ValueError("Cannot reflect-pad an empty array.")
    if work.shape[0] == 1:
        pad_block = np.repeat(work, pad, axis=0)
        out = np.concatenate([pad_block, work, pad_block], axis=0)
        return out[:, 0] if squeeze_1d else out

    remaining = int(pad)
    while remaining > 0:
        n_cur = work.shape[0]
        step = min(remaining, n_cur - 1)
        left = 2.0 * work[0:1] - work[1 : 1 + step][::-1]
        right = 2.0 * work[-1:] - work[n_cur - 1 - step : n_cur - 1][::-1]
        work = np.concatenate([left, work, right], axis=0)
        remaining -= step

    return work[:, 0] if squeeze_1d else work


def oversample_fourier(
    x: np.ndarray,
    *,
    m: int = 4,
    pad_samples: int = 240,
) -> np.ndarray:
    """Zero-padded FFT interpolation with Nyquist bin splitting and reflection (Formulas 32-33)."""
    if m < 1:
        raise ValueError(f"Oversampling factor m must be >= 1, got {m}")
    arr = np.asarray(x, dtype=np.float64)
    if m == 1:
        return arr.copy()

    squeeze_1d = arr.ndim == 1
    work_in = arr[:, None] if squeeze_1d else arr
    n_orig, n_comp = work_in.shape

    work = odd_reflect_pad(work_in, pad_samples) if pad_samples > 0 else work_in
    n_p = work.shape[0]
    n_out = m * n_p

    spec = np.fft.fft(work, axis=0)
    padded_spec = np.zeros((n_out, n_comp), dtype=np.complex128)
    if n_p % 2 == 0:
        half = n_p // 2
        padded_spec[:half] = spec[:half]
        padded_spec[half] = 0.5 * spec[half]
        padded_spec[n_out - half] = 0.5 * spec[half]
        padded_spec[n_out - half + 1 :] = spec[half + 1 :]
    else:
        half = (n_p - 1) // 2
        padded_spec[: half + 1] = spec[: half + 1]
        padded_spec[n_out - half :] = spec[half + 1 :]

    y_padded = np.real(np.fft.ifft(padded_spec, axis=0)) * float(m)
    start = m * pad_samples
    out = y_padded[start : start + m * n_orig]
    return out[:, 0] if squeeze_1d else out


def kaiser_beta(attenuation_db: float = 80.0) -> float:
    """Compute Kaiser window shape parameter beta for attenuation A > 50 dB (Formula 35)."""
    if attenuation_db > 50.0:
        return 0.1102 * (attenuation_db - 8.7)
    if attenuation_db >= 21.0:
        diff = attenuation_db - 21.0
        return 0.5842 * (diff ** 0.4) + 0.07886 * diff
    return 0.0


def kaiser_min_half_width(
    f2_hz: float,
    *,
    dt: float = 60.0,
    attenuation_db: float = 80.0,
) -> int:
    """Compute minimum Kaiser FIR half-width Q from Formula (35)."""
    delta_omega = 2.0 * math.pi * (1.0 - 2.0 * f2_hz * dt)
    if delta_omega <= 0.0:
        raise ValueError(f"Transition width non-positive for f2_hz={f2_hz}, dt={dt}")
    return int(math.ceil((attenuation_db - 8.0) / (4.57 * delta_omega)))


def oversample_kaiser(
    x: np.ndarray,
    *,
    m: int = 4,
    Q: int = 16,
    attenuation_db: float = 80.0,
    beta: float | None = None,
) -> np.ndarray:
    """Polyphase Kaiser-windowed sinc FIR interpolator giving m*(N-1)+1 samples (Formulas 34-35)."""
    if m < 1:
        raise ValueError(f"Oversampling factor m must be >= 1, got {m}")
    if Q < 1:
        raise ValueError(f"Half-width Q must be >= 1, got {Q}")
    arr = np.asarray(x, dtype=np.float64)
    if m == 1:
        return arr.copy()

    squeeze_1d = arr.ndim == 1
    work_in = arr[:, None] if squeeze_1d else arr
    n_orig, n_comp = work_in.shape
    if n_orig < 2:
        raise ValueError("Kaiser oversampling requires at least 2 samples.")

    beta_val = kaiser_beta(attenuation_db) if beta is None else float(beta)
    i0_beta = float(i0(beta_val))

    pad_len = Q + 1
    work_pad = odd_reflect_pad(work_in, pad_len)
    # x[n] sits at index n + pad_len; for n in 0..N-2 and j in -Q+1..Q,
    # index n + pad_len + j spans [2, N + 2Q - 1] inside work_pad (length N + 2Q + 2).
    n_intervals = n_orig - 1
    j_offsets = np.arange(-Q + 1, Q + 1, dtype=np.float64)  # length 2Q
    out = np.empty((m * n_intervals + 1, n_comp), dtype=np.float64)
    out[0::m] = work_in

    # Build (n_intervals, 2Q, n_comp) view via sliding_window_view
    windows = np.lib.stride_tricks.sliding_window_view(
        work_pad[2 : 2 + n_intervals + 2 * Q - 1],
        window_shape=2 * Q,
        axis=0,
    )
    # sliding_window_view appends window axis at the end: (n_intervals, n_comp, 2Q)
    for k in range(1, m):
        tau = float(k) / float(m)
        u = tau - j_offsets
        radicand = np.maximum(0.0, 1.0 - (u / float(Q)) ** 2)
        h = np.sinc(u) * i0(beta_val * np.sqrt(radicand)) / i0_beta
        out[k : m * n_intervals : m] = np.einsum("ncq,q->nc", windows, h)

    return out[:, 0] if squeeze_1d else out
