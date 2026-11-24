"""Physical conventions, band definitions, coordinate rotations, and QC rules (§§1-3)."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

DT_1MIN_SEC = 60.0
BASELINE_NORM_THRESHOLD_NT = 10000.0
INTERMAGNET_FILL_VALUES = (99999.0, 88888.0)


@dataclass(frozen=True)
class BandSpec:
    """Frequency and period bounds for a ULF pulsation band."""

    name: str
    f_min_hz: float
    f_max_hz: float
    t_min_sec: float
    t_max_sec: float


def get_band_spec(
    band: str = "Pc5",
    *,
    t_max_pc6_sec: float = 1800.0,
    f_min_hz: float | None = None,
    f_max_hz: float | None = None,
) -> BandSpec:
    """Return BandSpec for Pc5, Pc6, or an explicit custom frequency range (Formula 1)."""
    if f_min_hz is not None or f_max_hz is not None:
        if f_min_hz is None or f_max_hz is None:
            raise ValueError("Both f_min_hz and f_max_hz must be provided for a custom band.")
        if not (0.0 < f_min_hz < f_max_hz):
            raise ValueError(f"Invalid frequency bounds: f_min_hz={f_min_hz}, f_max_hz={f_max_hz}")
        return BandSpec(
            name=band,
            f_min_hz=float(f_min_hz),
            f_max_hz=float(f_max_hz),
            t_min_sec=1.0 / float(f_max_hz),
            t_max_sec=1.0 / float(f_min_hz),
        )

    key = band.strip().upper()
    if key == "PC5":
        return BandSpec(
            name="Pc5",
            f_min_hz=1.0 / 600.0,
            f_max_hz=1.0 / 150.0,
            t_min_sec=150.0,
            t_max_sec=600.0,
        )
    if key == "PC6":
        if t_max_pc6_sec <= 600.0:
            raise ValueError(f"t_max_pc6_sec must exceed 600 s, got {t_max_pc6_sec}")
        return BandSpec(
            name="Pc6",
            f_min_hz=1.0 / float(t_max_pc6_sec),
            f_max_hz=1.0 / 600.0,
            t_min_sec=600.0,
            t_max_sec=float(t_max_pc6_sec),
        )
    raise ValueError(f"Unsupported band {band!r}; expected 'Pc5', 'Pc6', or explicit f_min/f_max.")


def hdz_to_xyz(
    h_nT: np.ndarray | float,
    d_arcmin: np.ndarray | float,
    z_nT: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert (H [nT], D [arcmin], Z [nT]) to geographic (X, Y, Z) in nT (Formula 2)."""
    h_arr = np.asarray(h_nT, dtype=np.float64)
    d_rad = np.deg2rad(np.asarray(d_arcmin, dtype=np.float64) / 60.0)
    z_arr = np.asarray(z_nT, dtype=np.float64).copy()
    x_arr = h_arr * np.cos(d_rad)
    y_arr = h_arr * np.sin(d_rad)
    return x_arr, y_arr, z_arr


def xyz_to_hdz(
    x_nT: np.ndarray | float,
    y_nT: np.ndarray | float,
    z_nT: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert geographic (X, Y, Z) [nT] to (H [nT], D [arcmin], Z [nT]) (Formula 3)."""
    x_arr = np.asarray(x_nT, dtype=np.float64)
    y_arr = np.asarray(y_nT, dtype=np.float64)
    z_arr = np.asarray(z_nT, dtype=np.float64).copy()
    h_arr = np.hypot(x_arr, y_arr)
    d_arcmin = np.rad2deg(np.arctan2(y_arr, x_arr)) * 60.0
    return h_arr, d_arcmin, z_arr


def has_valid_baseline(
    xyz: np.ndarray,
    *,
    threshold_nT: float = BASELINE_NORM_THRESHOLD_NT,
) -> bool:
    """Return True if the 3D mean vector norm ||mean(X, Y, Z)|| >= threshold_nT."""
    arr = np.asarray(xyz, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected (N, 2+) array, got shape {arr.shape}")
    valid = np.all(np.isfinite(arr), axis=1)
    if not np.any(valid):
        return False
    mean_vec = np.mean(arr[valid], axis=0)
    return float(np.linalg.norm(mean_vec)) >= float(threshold_nT)


def compute_quiet_declination_rad(
    x_nT: np.ndarray,
    y_nT: np.ndarray,
    z_nT: np.ndarray | None = None,
    *,
    has_baseline: bool | None = None,
    external_d0_deg: float | None = None,
    threshold_nT: float = BASELINE_NORM_THRESHOLD_NT,
) -> float:
    """Compute quiet magnetic declination D0 = atan2(<Y>, <X>) in radians (Formula 4).

    Blocks automatic D0 computation for baseless variation stations (has_baseline=False
    or ||mean(X,Y,Z)|| < threshold_nT) unless an explicit external_d0_deg is supplied.
    """
    if external_d0_deg is not None:
        return math.radians(float(external_d0_deg))

    if has_baseline is False:
        raise ValueError(
            "Automatic D0 rotation is disabled for variation stations without an absolute "
            "baseline (has_baseline=False); pass external_d0_deg explicitly."
        )
    if has_baseline is None and z_nT is None:
        raise ValueError(
            "Cannot verify absolute baseline without z_nT; pass z_nT, explicit "
            "has_baseline=True, or external_d0_deg."
        )

    x_arr = np.asarray(x_nT, dtype=np.float64)
    y_arr = np.asarray(y_nT, dtype=np.float64)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if z_nT is not None:
        z_arr = np.asarray(z_nT, dtype=np.float64)
        valid &= np.isfinite(z_arr)
        if has_baseline is None:
            xyz = np.column_stack([x_arr, y_arr, z_arr])
            if not has_valid_baseline(xyz, threshold_nT=threshold_nT):
                raise ValueError(
                    "Mean 3D field vector norm is below baseline threshold; refusing automatic D0."
                )

    if not np.any(valid):
        raise ValueError("No finite vector samples available to compute quiet declination D0.")

    return math.atan2(float(np.mean(y_arr[valid])), float(np.mean(x_arr[valid])))


def rotation_matrix_2d(d0_rad: float) -> np.ndarray:
    """Return 2x2 horizontal rotation matrix R(D0) from Formula (5)."""
    c = math.cos(d0_rad)
    s = math.sin(d0_rad)
    return np.array([[c, s], [-s, c]], dtype=np.float64)


def rotate_xy_to_magnetic(
    x_nT: np.ndarray,
    y_nT: np.ndarray,
    d0_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate horizontal (X, Y) to local magnetic meridian (Xm, Ym) (Formula 5)."""
    x_arr = np.asarray(x_nT, dtype=np.float64)
    y_arr = np.asarray(y_nT, dtype=np.float64)
    c = math.cos(d0_rad)
    s = math.sin(d0_rad)
    xm = c * x_arr + s * y_arr
    ym = -s * x_arr + c * y_arr
    return xm, ym


def rotate_xyz_to_magnetic(xyz: np.ndarray, d0_rad: float) -> np.ndarray:
    """Rotate (N, 3) XYZ array around vertical Z axis by D0 (Formula 5)."""
    arr = np.asarray(xyz, dtype=np.float64).copy()
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected (N, 2+) array, got shape {arr.shape}")
    xm, ym = rotate_xy_to_magnetic(arr[:, 0], arr[:, 1], d0_rad)
    arr[:, 0] = xm
    arr[:, 1] = ym
    return arr


def rotate_spectral_matrix(S: np.ndarray, d0_rad: float) -> np.ndarray:
    """Rotate 2x2 or 3x3 spectral matrix S_m = R @ S @ R.T (Section 3.3)."""
    mat = np.asarray(S)
    if mat.shape == (2, 2):
        rot = rotation_matrix_2d(d0_rad)
    elif mat.shape == (3, 3):
        r2 = rotation_matrix_2d(d0_rad)
        rot = np.eye(3, dtype=np.float64)
        rot[:2, :2] = r2
    else:
        raise ValueError(f"Expected 2x2 or 3x3 spectral matrix, got shape {mat.shape}")
    return rot @ mat @ rot.T


def max_invalid_run(is_valid: np.ndarray) -> int:
    """Return length of the longest contiguous False run in a 1D boolean mask."""
    mask = np.asarray(is_valid, dtype=bool)
    if mask.size == 0 or bool(np.all(mask)):
        return 0
    invalid = ~mask
    padded = np.concatenate(([False], invalid, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return int(np.max(ends - starts)) if starts.size else 0


def interpolate_short_gaps(
    xyz: np.ndarray,
    is_valid_xyz: np.ndarray | None = None,
    *,
    max_gap: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate interior gaps of length <= max_gap across all components (§3.2)."""
    arr = np.asarray(xyz, dtype=np.float64)
    squeeze_1d = arr.ndim == 1
    if squeeze_1d:
        arr = arr[:, None]
    out = arr.copy()
    if is_valid_xyz is None:
        valid = np.all(np.isfinite(out), axis=1)
    else:
        valid = np.asarray(is_valid_xyz, dtype=bool)
        if valid.shape[0] != out.shape[0]:
            raise ValueError("is_valid_xyz length must match xyz.")
    # Enforce common validity mask across all components
    out[~valid] = np.nan

    n = out.shape[0]
    is_interp = np.zeros(n, dtype=bool)
    if n == 0 or bool(np.all(valid)):
        return (out[:, 0] if squeeze_1d else out), is_interp

    invalid = ~valid
    padded = np.concatenate(([False], invalid, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    for start, end in zip(starts, ends, strict=True):
        run_len = int(end - start)
        if 1 <= run_len <= max_gap and start > 0 and end < n:
            left = out[start - 1]
            right = out[end]
            if np.all(np.isfinite(left)) and np.all(np.isfinite(right)):
                w = (np.arange(1, run_len + 1, dtype=np.float64) / (run_len + 1))[:, None]
                out[start:end] = left[None, :] * (1.0 - w) + right[None, :] * w
                is_interp[start:end] = True

    return (out[:, 0] if squeeze_1d else out), is_interp


def is_window_valid(
    is_valid_xyz: np.ndarray,
    is_finite_after_interp: np.ndarray | None = None,
    *,
    max_missing_frac: float = 0.05,
    max_gap: int = 3,
) -> bool:
    """Check §3.2 window acceptance: <= max_missing_frac invalid and no gap > max_gap."""
    valid = np.asarray(is_valid_xyz, dtype=bool)
    if valid.size == 0:
        return False
    missing_frac = float(np.mean(~valid))
    if missing_frac > max_missing_frac + 1e-12:
        return False
    if max_invalid_run(valid) > max_gap:
        return False
    if is_finite_after_interp is not None and not bool(np.all(is_finite_after_interp)):
        return False
    return True


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers between two points on Earth."""
    r_earth = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    return 2.0 * r_earth * math.asin(min(1.0, math.sqrt(a)))


def despike_station_network(
    station_xyz: dict[str, np.ndarray],
    station_coords: dict[str, tuple[float, float]],
    *,
    min_jump_nT: float = 30.0,
    min_resid_nT: float = 50.0,
    mad_jump_mult: float = 12.0,
    mad_resid_mult: float = 20.0,
    hampel_z_thresh: float = 10.0,
    neighbor_confirm_nT: float = 15.0,
    neighbor_confirm_z: float = 8.0,
    n_neighbors: int = 3,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """MAD/Hampel 1st-difference despiker with cross-station SC/substorm protection."""
    cleaned = {k: np.asarray(v, dtype=np.float64).copy() for k, v in station_xyz.items()}
    flags = {k: np.zeros(v.shape[0], dtype=bool) for k, v in cleaned.items()}
    if not cleaned:
        return cleaned, flags

    d1_map: dict[str, np.ndarray] = {}
    mad_map: dict[str, np.ndarray] = {}
    for code, xyz in cleaned.items():
        n_comp = xyz.shape[1]
        d1 = np.diff(xyz, axis=0, prepend=np.nan)
        mads = np.empty(n_comp, dtype=np.float64)
        for c in range(n_comp):
            col = d1[:, c]
            fin = col[np.isfinite(col)]
            if fin.size >= 10:
                med = float(np.median(fin))
                mads[c] = max(float(np.median(np.abs(fin - med))) * 1.4826, 0.5)
            else:
                mads[c] = 0.5
        d1_map[code] = d1
        mad_map[code] = mads

    station_list = sorted(cleaned.keys())
    for code in station_list:
        xyz = cleaned[code]
        n, n_comp = xyz.shape
        if n < 3:
            continue
        mad_vec = mad_map[code]
        lat1, lon1 = station_coords[code]
        dists = sorted(
            (haversine_km(lat1, lon1, station_coords[other][0], station_coords[other][1]), other)
            for other in station_list
            if other != code
        )
        neighbors = [other for _, other in dists[:n_neighbors]]

        for t in range(1, n - 1):
            spiked_comps: list[int] = []
            for c in range(n_comp):
                prev_v, cur_v, next_v = xyz[t - 1, c], xyz[t, c], xyz[t + 1, c]
                if not (math.isfinite(prev_v) and math.isfinite(cur_v) and math.isfinite(next_v)):
                    continue
                d_in = cur_v - prev_v
                d_out = next_v - cur_v
                if d_in * d_out >= 0.0:
                    continue
                mad = float(mad_vec[c])
                if min(abs(d_in), abs(d_out)) <= max(min_jump_nT, mad_jump_mult * mad):
                    continue
                resid = abs(cur_v - 0.5 * (prev_v + next_v))
                if resid <= max(min_resid_nT, mad_resid_mult * mad):
                    continue
                win = xyz[max(0, t - 5) : min(n, t + 6), c]
                fin_w = win[np.isfinite(win)]
                if fin_w.size < 5:
                    continue
                w_med = float(np.median(fin_w))
                w_mad_raw = max(float(np.median(np.abs(fin_w - w_med))) * 1.4826, 0.5)
                z_raw = abs(cur_v - w_med) / w_mad_raw

                # Step-immune 1st-difference local scale protects detection near baseline steps
                win_d1 = d1_map[code][max(1, t - 5) : min(n, t + 6), c]
                fin_d1 = win_d1[np.isfinite(win_d1)]
                if fin_d1.size >= 5:
                    d1_med = float(np.median(fin_d1))
                    w_mad_d1 = max(float(np.median(np.abs(fin_d1 - d1_med))) * 1.4826, 0.5)
                    z_d1 = resid / w_mad_d1
                else:
                    z_d1 = 0.0

                if z_raw > hampel_z_thresh or z_d1 > hampel_z_thresh:
                    spiked_comps.append(c)

            if not spiked_comps:
                continue

            confirmed_geophysical = False
            lo_t = max(0, t - 2)
            hi_t = min(n, t + 3)
            for nb in neighbors:
                nb_slice = np.abs(d1_map[nb][lo_t:hi_t, :])
                if not np.any(np.isfinite(nb_slice)):
                    continue
                nb_max_jump = float(np.nanmax(nb_slice))
                nb_max_z = float(np.nanmax(nb_slice / mad_map[nb][None, :]))
                if nb_max_jump > neighbor_confirm_nT and nb_max_z > neighbor_confirm_z:
                    confirmed_geophysical = True
                    break

            if not confirmed_geophysical:
                # Joint 3-component vector cleaning at confirmed instrumental spike timestamp
                for c in range(n_comp):
                    prev_v, next_v = xyz[t - 1, c], xyz[t + 1, c]
                    if math.isfinite(prev_v) and math.isfinite(next_v):
                        cleaned[code][t, c] = 0.5 * (prev_v + next_v)
                flags[code][t] = True

    return cleaned, flags


class CalculationCancelledError(RuntimeError):
    """Raised when a long-running calculation observes a cancellation token."""


def _normalize_json_value(val: Any) -> Any:
    if isinstance(val, dict):
        return {str(k): _normalize_json_value(v) for k, v in sorted(val.items())}
    if isinstance(val, (list, tuple)):
        return [_normalize_json_value(v) for v in val]
    if isinstance(val, np.ndarray):
        return _normalize_json_value(val.tolist())
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        f_val = float(val)
        return None if not math.isfinite(f_val) else f_val
    if isinstance(val, float):
        return None if not math.isfinite(val) else val
    return val


def canonical_cache_key(core_version: str, params: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 cache key from core_version and canonical JSON params."""
    payload = _normalize_json_value({"core_version": core_version, **params})
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
