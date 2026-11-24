"""Zero-phase bandpass filtering for Pc5/Pc6: FFT-mask and Butterworth SOS (§8)."""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import sosfilt

from .conventions import get_band_spec
from .oversampling import odd_reflect_pad
from .spectral import detrend_linear_ols


def _resolve_band_bounds(
    band: str,
    *,
    f1_hz: float | None = None,
    f2_hz: float | None = None,
    t_max_pc6_sec: float = 1800.0,
) -> tuple[float, float, str]:
    spec = get_band_spec(
        band=band,
        t_max_pc6_sec=t_max_pc6_sec,
        f_min_hz=f1_hz,
        f_max_hz=f2_hz,
    )
    return spec.f_min_hz, spec.f_max_hz, spec.name


def _cos_transition_lambda(freq: np.ndarray, fc: float, rho: float) -> np.ndarray:
    dc = rho * fc
    out = np.zeros_like(freq, dtype=np.float64)
    out[freq >= fc + dc] = 1.0
    mid = np.abs(freq - fc) < dc
    out[mid] = 0.5 * (1.0 + np.sin(math.pi * (freq[mid] - fc) / (2.0 * dc)))
    return out


def fft_mask_response(
    freqs_hz: np.ndarray,
    f1_hz: float,
    f2_hz: float,
    *,
    rho: float = 0.25,
) -> np.ndarray:
    """Power-complementary cosine-tapered FFT mask W(f) from Formula (37)."""
    a = np.abs(np.asarray(freqs_hz, dtype=np.float64))
    lam_low = _cos_transition_lambda(a, f1_hz, rho)
    lam_high = _cos_transition_lambda(a, f2_hz, rho)
    w = np.sqrt(np.maximum(0.0, lam_low * (1.0 - lam_high)))
    w[a == 0.0] = 0.0
    return w


def apply_fft_mask(
    x: np.ndarray,
    *,
    dt_prime: float,
    band: str = "Pc5",
    f1_hz: float | None = None,
    f2_hz: float | None = None,
    t_max_pc6_sec: float = 1800.0,
    rho: float = 0.25,
    reflect_hours: float = 6.0,
    detrend: bool | None = None,
) -> np.ndarray:
    """Zero-phase FFT-mask bandpass filter with odd boundary reflection (§8.2)."""
    f1, f2, band_name = _resolve_band_bounds(
        band, f1_hz=f1_hz, f2_hz=f2_hz, t_max_pc6_sec=t_max_pc6_sec
    )
    arr = np.asarray(x, dtype=np.float64)
    squeeze_1d = arr.ndim == 1
    work = arr[:, None].copy() if squeeze_1d else arr.copy()

    do_detrend = (band_name.upper() == "PC6") if detrend is None else bool(detrend)
    if do_detrend:
        work = detrend_linear_ols(work, axis=0)
    else:
        work = work - np.mean(work, axis=0, keepdims=True)

    n_orig = work.shape[0]
    pad = int(round(reflect_hours * 3600.0 / dt_prime)) if reflect_hours > 0.0 else 0
    padded = odd_reflect_pad(work, pad) if pad > 0 else work
    n_p = padded.shape[0]

    freqs = np.fft.rfftfreq(n_p, d=dt_prime)
    mask = fft_mask_response(freqs, f1, f2, rho=rho)
    spec = np.fft.rfft(padded, axis=0)
    filtered_pad = np.fft.irfft(spec * mask[:, None], n=n_p, axis=0)
    out = filtered_pad[pad : pad + n_orig]
    return out[:, 0] if squeeze_1d else out


def design_butterworth_sos(
    f1_hz: float,
    f2_hz: float,
    fs_hz: float,
    *,
    order_n: int = 3,
    zero_phase_corrected: bool = True,
) -> np.ndarray:
    """Design bandpass Butterworth SOS cascade via Steps 1-7 of §8.3 (Formulas 38-43).

    Returns shape (order_n, 6) with rows [b0, b1, b2, a0, a1, a2] and gain G
    absorbed into b0, b2 of section 0, matching Table 11 ordering.
    """
    if order_n < 1:
        raise ValueError(f"Filter order n must be >= 1, got {order_n}")
    if not (0.0 < f1_hz < f2_hz < 0.5 * fs_hz):
        raise ValueError(
            f"Require 0 < f1 ({f1_hz}) < f2 ({f2_hz}) < fs/2 ({0.5 * fs_hz})."
        )

    omega1 = math.tan(math.pi * f1_hz / fs_hz)
    omega2 = math.tan(math.pi * f2_hz / fs_hz)
    omega0_sq = omega1 * omega2
    omega0 = math.sqrt(omega0_sq)
    bw = omega2 - omega1
    if zero_phase_corrected:
        bw = bw / ((math.sqrt(2.0) - 1.0) ** (1.0 / (2.0 * order_n)))

    sections: list[list[float]] = []
    for k in range(1, order_n + 1):
        pk = np.exp(1j * math.pi * (2 * k + order_n - 1) / (2.0 * order_n))
        if np.imag(pk) > 1e-12:
            disc = np.sqrt(bw * bw * pk * pk - 4.0 * omega0_sq)
            for sign in (1.0, -1.0):
                s = 0.5 * (bw * pk + sign * disc)
                z = (1.0 + s) / (1.0 - s)
                a1 = -2.0 * float(np.real(z))
                a2 = float(np.abs(z) ** 2)
                sections.append([1.0, 0.0, -1.0, 1.0, a1, a2])
        elif abs(float(np.imag(pk))) <= 1e-12:
            disc = np.sqrt(complex(bw * bw * pk * pk - 4.0 * omega0_sq))
            s1 = 0.5 * (bw * pk + disc)
            s2 = 0.5 * (bw * pk - disc)
            z1 = (1.0 + s1) / (1.0 - s1)
            z2 = (1.0 + s2) / (1.0 - s2)
            a1 = float(np.real(-(z1 + z2)))
            a2 = float(np.real(z1 * z2))
            sections.append([1.0, 0.0, -1.0, 1.0, a1, a2])

    sos = np.asarray(sections, dtype=np.float64)
    w0 = 2.0 * math.atan(omega0)
    ejw = np.exp(-1j * w0)
    ej2w = ejw * ejw
    h_prod = 1.0
    for row in sos:
        num = row[0] + row[1] * ejw + row[2] * ej2w
        den = row[3] + row[4] * ejw + row[5] * ej2w
        h_prod *= abs(num / den)
    gain = 1.0 / h_prod
    sos[0, :3] *= gain
    return sos


def butterworth_single_pass_response(
    freqs_hz: np.ndarray,
    f1_hz: float,
    f2_hz: float,
    fs_hz: float,
    *,
    order_n: int = 3,
    zero_phase_corrected: bool = True,
) -> np.ndarray:
    """Exact theoretical |H(e^{i*omega})| of single-pass Butterworth filter (Formula 39)."""
    f = np.asarray(freqs_hz, dtype=np.float64)
    omega1 = math.tan(math.pi * f1_hz / fs_hz)
    omega2 = math.tan(math.pi * f2_hz / fs_hz)
    omega0_sq = omega1 * omega2
    bw = omega2 - omega1
    if zero_phase_corrected:
        bw = bw / ((math.sqrt(2.0) - 1.0) ** (1.0 / (2.0 * order_n)))

    omega = np.tan(math.pi * f / fs_hz)
    out = np.zeros_like(f, dtype=np.float64)
    nz = (f > 0.0) & (f < 0.5 * fs_hz)
    xi = (omega[nz] ** 2 - omega0_sq) / (bw * omega[nz])
    out[nz] = 1.0 / np.sqrt(1.0 + xi ** (2 * order_n))
    return out


def butterworth_zero_phase_response(
    freqs_hz: np.ndarray,
    f1_hz: float,
    f2_hz: float,
    fs_hz: float,
    *,
    order_n: int = 3,
) -> np.ndarray:
    """Zero-phase forward-backward amplitude response A(f) = |H(f)|^2 (§8.1, §8.3)."""
    h_single = butterworth_single_pass_response(
        freqs_hz,
        f1_hz,
        f2_hz,
        fs_hz,
        order_n=order_n,
        zero_phase_corrected=True,
    )
    return h_single ** 2


def analog_butterworth_zero_phase_response(
    freqs_hz: np.ndarray,
    f1_hz: float,
    f2_hz: float,
    *,
    order_n: int = 3,
) -> np.ndarray:
    """Analog prototype forward-backward response A(f) = 1 / (1 + xi_a^(2n))."""
    f = np.asarray(freqs_hz, dtype=np.float64)
    f0_sq = f1_hz * f2_hz
    bw = (f2_hz - f1_hz) / ((math.sqrt(2.0) - 1.0) ** (1.0 / (2.0 * order_n)))
    out = np.zeros_like(f, dtype=np.float64)
    nz = f > 0.0
    xi = (f[nz] ** 2 - f0_sq) / (bw * f[nz])
    out[nz] = 1.0 / (1.0 + xi ** (2 * order_n))
    return out


def apply_butterworth_sos_zero_phase(
    x: np.ndarray,
    *,
    dt_prime: float,
    band: str = "Pc5",
    order_n: int = 3,
    f1_hz: float | None = None,
    f2_hz: float | None = None,
    t_max_pc6_sec: float = 1800.0,
    reflect_hours: float = 6.0,
    detrend: bool | None = None,
) -> np.ndarray:
    """Apply zero-phase forward-backward Butterworth SOS filter with odd reflection (Step 8)."""
    f1, f2, band_name = _resolve_band_bounds(
        band, f1_hz=f1_hz, f2_hz=f2_hz, t_max_pc6_sec=t_max_pc6_sec
    )
    fs_hz = 1.0 / float(dt_prime)
    sos = design_butterworth_sos(f1, f2, fs_hz, order_n=order_n, zero_phase_corrected=True)

    arr = np.asarray(x, dtype=np.float64)
    squeeze_1d = arr.ndim == 1
    work = arr[:, None].copy() if squeeze_1d else arr.copy()

    do_detrend = (band_name.upper() == "PC6") if detrend is None else bool(detrend)
    if do_detrend:
        work = detrend_linear_ols(work, axis=0)
    else:
        # Remove constant DC offset so sosfilt(zi=0) avoids a 45,000 nT step transient
        work = work - np.mean(work, axis=0, keepdims=True)

    n_orig = work.shape[0]
    pad = int(round(reflect_hours * 3600.0 / dt_prime)) if reflect_hours > 0.0 else 0
    padded = odd_reflect_pad(work, pad) if pad > 0 else work

    fwd = sosfilt(sos, padded, axis=0)
    rev = sosfilt(sos, fwd[::-1], axis=0)[::-1]
    out = rev[pad : pad + n_orig]
    return out[:, 0] if squeeze_1d else out
