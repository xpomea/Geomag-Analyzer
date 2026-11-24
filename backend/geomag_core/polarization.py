"""2D Stokes polarization, 3D Means/Samson parameters, thresholds, and aggregation (§§5-6)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta as beta_dist


@dataclass(frozen=True)
class Stokes2D:
    """2D horizontal polarization parameters from Formulas (12)-(19)."""

    I: float  # noqa: E741
    Q: float
    U: float
    V: float
    A: float
    R: float
    p: float
    p_L: float
    p_C: float
    psi_deg: float
    chi_deg: float
    epsilon: float
    gamma2: float
    arg_Sxy_deg: float
    n_eff: float | None = None
    p_crit: float | None = None
    gamma2_crit: float | None = None
    is_significant_p: bool | None = None
    is_significant_gamma2: bool | None = None


@dataclass(frozen=True)
class Polarization3D:
    """3D polarization parameters from Samson (1973) & Means (1972) (Formulas 26-29)."""

    P: float
    n_v: np.ndarray
    n_v_unit: np.ndarray
    theta_deg: float
    phi_n_deg: float
    eigenvalues_re_S: np.ndarray
    b_over_a: float
    s: float
    tr_S: float


def critical_thresholds(n_eff: float, alpha: float = 0.05) -> tuple[float, float]:
    """Compute (p_crit, gamma2_crit) for null hypothesis of unpolarized noise (Formulas 24-25)."""
    if n_eff <= 1.0:
        raise ValueError(f"Effective degrees of freedom n_eff must be > 1, got {n_eff}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"Significance level alpha must be in (0, 1), got {alpha}")
    u_alpha = float(beta_dist.ppf(alpha, n_eff - 1.0, 1.5))
    p_crit = math.sqrt(max(0.0, 1.0 - u_alpha))
    gamma2_crit = 1.0 - (alpha ** (1.0 / (n_eff - 1.0)))
    return p_crit, gamma2_crit


def stokes_from_spectral_matrix_2d(
    S: np.ndarray,
    *,
    n_eff: float | None = None,
    alpha: float = 0.05,
) -> Stokes2D:
    """Compute 2D horizontal Stokes parameters from 2x2 or upper-left 2x2 of 3x3 S."""
    mat = np.asarray(S, dtype=np.complex128)
    if mat.shape not in ((2, 2), (3, 3)):
        raise ValueError(f"Expected 2x2 or 3x3 matrix, got shape {mat.shape}")
    s_xx = float(np.real(mat[0, 0]))
    s_yy = float(np.real(mat[1, 1]))
    s_xy = complex(mat[0, 1])

    all_finite = (
        math.isfinite(s_xx)
        and math.isfinite(s_yy)
        and math.isfinite(s_xy.real)
        and math.isfinite(s_xy.imag)
    )
    if not all_finite:
        return Stokes2D(
            I=float("nan"), Q=float("nan"), U=float("nan"), V=float("nan"),
            A=float("nan"), R=float("nan"), p=float("nan"), p_L=float("nan"),
            p_C=float("nan"), psi_deg=float("nan"), chi_deg=float("nan"),
            epsilon=float("nan"), gamma2=float("nan"), arg_Sxy_deg=float("nan"),
            n_eff=n_eff,
        )

    i_val = s_xx + s_yy
    q_val = s_xx - s_yy
    u_val = 2.0 * s_xy.real
    v_val = 2.0 * s_xy.imag
    a_val = math.hypot(q_val, u_val)
    r_val = math.sqrt(q_val * q_val + u_val * u_val + v_val * v_val)

    if i_val <= 0.0:
        return Stokes2D(
            I=i_val, Q=q_val, U=u_val, V=v_val,
            A=a_val, R=r_val, p=float("nan"), p_L=float("nan"),
            p_C=float("nan"), psi_deg=float("nan"), chi_deg=float("nan"),
            epsilon=float("nan"), gamma2=float("nan"), arg_Sxy_deg=float("nan"),
            n_eff=n_eff,
        )

    p_val = min(1.0, max(0.0, r_val / i_val))
    p_l = a_val / i_val
    p_c = v_val / i_val

    if a_val <= 1e-8 * i_val:
        psi_deg = float("nan")
    else:
        psi_deg = 0.5 * math.degrees(math.atan2(u_val, q_val))
        if psi_deg <= -90.0 + 1e-12:
            psi_deg += 180.0

    if r_val <= 0.0:
        chi_deg = 0.0
        epsilon = 0.0
    else:
        ratio_vr = max(-1.0, min(1.0, v_val / r_val))
        chi_deg = 0.5 * math.degrees(math.asin(ratio_vr))
        epsilon = v_val / (r_val + a_val)

    denom_g = s_xx * s_yy
    if denom_g > 0.0:
        gamma2 = min(1.0, max(0.0, (abs(s_xy) ** 2) / denom_g))
    else:
        gamma2 = float("nan")
    arg_sxy_deg = math.degrees(math.atan2(s_xy.imag, s_xy.real))

    p_crit: float | None = None
    g_crit: float | None = None
    sig_p: bool | None = None
    sig_g: bool | None = None
    if n_eff is not None and math.isfinite(n_eff) and n_eff > 1.0:
        p_crit, g_crit = critical_thresholds(n_eff, alpha=alpha)
        sig_p = p_val > p_crit
        sig_g = (gamma2 > g_crit) if math.isfinite(gamma2) else False

    return Stokes2D(
        I=i_val,
        Q=q_val,
        U=u_val,
        V=v_val,
        A=a_val,
        R=r_val,
        p=p_val,
        p_L=p_l,
        p_C=p_c,
        psi_deg=psi_deg,
        chi_deg=chi_deg,
        epsilon=epsilon,
        gamma2=gamma2,
        arg_Sxy_deg=arg_sxy_deg,
        n_eff=n_eff,
        p_crit=p_crit,
        gamma2_crit=g_crit,
        is_significant_p=sig_p,
        is_significant_gamma2=sig_g,
    )


def polarization_3d(S: np.ndarray) -> Polarization3D:
    """Compute 3D Samson/Means polarization parameters from 3x3 S (Formulas 26-29)."""
    mat = np.asarray(S, dtype=np.complex128)
    if mat.shape != (3, 3):
        raise ValueError(f"Expected 3x3 spectral matrix, got shape {mat.shape}")
    if not np.all(np.isfinite(mat)):
        return Polarization3D(
            P=float("nan"),
            n_v=np.full(3, np.nan),
            n_v_unit=np.full(3, np.nan),
            theta_deg=float("nan"),
            phi_n_deg=float("nan"),
            eigenvalues_re_S=np.full(3, np.nan),
            b_over_a=float("nan"),
            s=float("nan"),
            tr_S=float("nan"),
        )

    tr_s = float(np.real(np.trace(mat)))
    tr_s2 = float(np.real(np.trace(mat @ mat)))
    if tr_s <= 0.0:
        raise ValueError(f"Trace of spectral matrix must be positive, got {tr_s}")

    p3_sq = max(0.0, (3.0 * tr_s2 - tr_s * tr_s) / (2.0 * tr_s * tr_s))
    p3 = min(1.0, math.sqrt(p3_sq))

    # Formula (27): n_v = [Im(S_yz), -Im(S_xz), Im(S_xy)]^T
    n_v = np.array(
        [
            float(np.imag(mat[1, 2])),
            -float(np.imag(mat[0, 2])),
            float(np.imag(mat[0, 1])),
        ],
        dtype=np.float64,
    )
    norm_nv = float(np.linalg.norm(n_v))
    if norm_nv > 0.0:
        unit = n_v / norm_nv
        theta_deg = math.degrees(math.acos(min(1.0, abs(float(unit[2])))))
        phi_n_deg = math.degrees(math.atan2(float(unit[1]), float(unit[0]))) % 180.0
    else:
        unit = np.full(3, np.nan, dtype=np.float64)
        theta_deg = float("nan")
        phi_n_deg = float("nan")

    re_sym = 0.5 * (np.real(mat) + np.real(mat).T)
    eigvals = np.linalg.eigvalsh(re_sym)[::-1].copy()  # descending: lambda1 >= lambda2 >= lambda3
    lam1, lam2, lam3 = float(eigvals[0]), float(eigvals[1]), float(eigvals[2])
    if lam1 > lam3 + 1e-15:
        b_over_a = math.sqrt(max(0.0, lam2 - lam3) / (lam1 - lam3))
    else:
        b_over_a = float("nan")
    s_val = (2.0 * norm_nv) / tr_s

    return Polarization3D(
        P=p3,
        n_v=n_v,
        n_v_unit=unit,
        theta_deg=theta_deg,
        phi_n_deg=phi_n_deg,
        eigenvalues_re_S=eigvals,
        b_over_a=b_over_a,
        s=s_val,
        tr_S=tr_s,
    )


def aggregate_spectral_matrices(
    matrices: Sequence[np.ndarray] | np.ndarray,
    weights: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Aggregate spectral matrices S across windows, days, or stations (§5.7)."""
    arr = np.asarray(matrices, dtype=np.complex128)
    if arr.ndim != 3 or arr.shape[0] == 0:
        raise ValueError(f"Expected (M, C, C) array of matrices, got shape {arr.shape}")
    if weights is None:
        return np.mean(arr, axis=0)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (arr.shape[0],):
        raise ValueError(f"weights shape {w.shape} must match ({arr.shape[0]},).")
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        raise ValueError("Sum of weights must be positive.")
    return np.sum(arr * w[:, None, None], axis=0) / w_sum


def aggregate_stokes(
    stokes_list: Sequence[Stokes2D],
    weights: Sequence[float] | np.ndarray | None = None,
    *,
    n_eff: float | None = None,
    alpha: float = 0.05,
) -> Stokes2D:
    """Aggregate 2D Stokes parameters (I, Q, U, V) without averaging angles directly (§5.7)."""
    if not stokes_list:
        raise ValueError("stokes_list must be non-empty.")
    m = len(stokes_list)
    if weights is None:
        w = np.ones(m, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (m,):
            raise ValueError(f"weights shape {w.shape} must match ({m},).")
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        raise ValueError("Sum of weights must be positive.")

    i_avg = sum(wk * s.I for wk, s in zip(w, stokes_list, strict=True)) / w_sum
    q_avg = sum(wk * s.Q for wk, s in zip(w, stokes_list, strict=True)) / w_sum
    u_avg = sum(wk * s.U for wk, s in zip(w, stokes_list, strict=True)) / w_sum
    v_avg = sum(wk * s.V for wk, s in zip(w, stokes_list, strict=True)) / w_sum

    s_xx = 0.5 * (i_avg + q_avg)
    s_yy = 0.5 * (i_avg - q_avg)
    s_xy = 0.5 * (u_avg + 1j * v_avg)
    s_mat = np.array([[s_xx, s_xy], [np.conj(s_xy), s_yy]], dtype=np.complex128)
    return stokes_from_spectral_matrix_2d(s_mat, n_eff=n_eff, alpha=alpha)


def synthetic_ellipse_2d(
    t_sec: np.ndarray,
    *,
    T_sec: float,
    a: float,
    b: float,
    psi0_deg: float,
    h: int,
) -> np.ndarray:
    """Generate monochromatic 2D elliptical wave [X, Y] via Formula (20)."""
    if h not in (-1, 1):
        raise ValueError(f"Handedness h must be +1 or -1, got {h}")
    t = np.asarray(t_sec, dtype=np.float64)
    omega = 2.0 * math.pi / float(T_sec)
    psi0 = math.radians(float(psi0_deg))
    c_p, s_p = math.cos(psi0), math.sin(psi0)
    c_wt = np.cos(omega * t)
    s_wt = np.sin(omega * t)
    x = a * c_p * c_wt - float(h) * b * s_p * s_wt
    y = a * s_p * c_wt + float(h) * b * c_p * s_wt
    return np.column_stack([x, y])
