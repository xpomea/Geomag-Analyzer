"""Dataset contract constants, geometry/time feature helpers, and `.npz` validators."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geomag_ml.utils.io import load_npz, read_json

BELT_ID_TO_NAME: dict[int, str] = {
    0: "EU_mid",
    1: "EU_aur",
    2: "CA_mid",
    3: "CA_aur",
}
BELT_NAME_TO_ID: dict[str, int] = {v: k for k, v in BELT_ID_TO_NAME.items()}
AURORAL_MLAT_THRESHOLD_DEG: float = 60.0

OMNI_COLS: tuple[str, ...] = (
    "ae",
    "al",
    "au",
    "sym_h",
    "pc_n",
    "kp",
    "bz_gsm",
    "flow_speed",
    "pdyn",
    "electric_field",
)
OMNI_PARQUET_COL_MAP: dict[str, str] = {
    "ae": "ae_nT",
    "al": "al_nT",
    "au": "au_nT",
    "sym_h": "sym_h_nT",
    "pc_n": "pc_n_index",
    "kp": "kp",
    "bz_gsm": "bz_gsm_nT",
    "flow_speed": "flow_speed_kms",
    "pdyn": "pdyn_nPa",
    "electric_field": "electric_field_mVm",
}

SOLAR1_BSN_COLS: tuple[str, ...] = (
    "bz_gsm",
    "b_mag_gsm",
    "vx_gse",
    "proton_n",
    "pdyn_nPa",
    "ey_mVm",
)
SOLAR1_L1_COLS: tuple[str, ...] = (
    "bz_gsm",
    "b_mag_gsm",
    "vx_gse",
    "proton_n",
    "pdyn_nPa",
)
SOLAR1_PARQUET_COL_MAP: dict[str, str] = {
    "bz_gsm": "bz_gsm_nT",
    "b_mag_gsm": "b_mag_gsm_nT",
    "vx_gse": "vx_gse_kms",
    "proton_n": "proton_n_cm3",
    "pdyn_nPa": "pdyn_nPa",
}

STATION_COORD_COLS: tuple[str, ...] = ("lat_deg", "lon_deg", "mlat_deg", "mlon_deg")
TIME_FEATURE_COLS: tuple[str, ...] = ("sin_ut", "cos_ut", "sin_doy", "cos_doy")

EVENT_FLAG_QUIET: int = 1
EVENT_FLAG_STORM: int = 2
EVENT_FLAG_ECLIPSE: int = 4
EVENT_FLAG_SPE: int = 8

EVENT_NAME_TO_BIT: dict[str, int] = {
    "quiet": EVENT_FLAG_QUIET,
    "quiet_kp_le2": EVENT_FLAG_QUIET,
    "storm": EVENT_FLAG_STORM,
    "eclipse": EVENT_FLAG_ECLIPSE,
    "spe": EVENT_FLAG_SPE,
}

NPZ_DTYPE_SPEC: dict[str, np.dtype] = {
    "X_raw": np.dtype(np.float32),
    "M_valid": np.dtype(np.bool_),
    "M_raw_valid": np.dtype(np.bool_),
    "M_interpolated": np.dtype(np.bool_),
    "M_despiked": np.dtype(np.bool_),
    "M_qc_clean": np.dtype(np.bool_),
    "M_known_spikes": np.dtype(np.bool_),
    "timestamps_unix_s": np.dtype(np.int64),
    "episode_idx": np.dtype(np.int16),
    "episode_bounds": np.dtype(np.int64),
    "station_coords": np.dtype(np.float32),
    "station_dist_km": np.dtype(np.float32),
    "station_has_baseline": np.dtype(np.bool_),
    "station_belt_id": np.dtype(np.int8),
    "mlt_hours": np.dtype(np.float32),
    "time_features": np.dtype(np.float32),
    "sw_omni": np.dtype(np.float32),
    "sw_solar1_bsn": np.dtype(np.float32),
    "sw_solar1_l1": np.dtype(np.float32),
    "oulu_delta_n_pct": np.dtype(np.float32),
    "event_flags": np.dtype(np.uint8),
}


def infer_station_region(lon_deg: float, explicit_region: str | None = None) -> str:
    """Normalize station region to `'Europe'` or `'Canada'`."""
    if explicit_region is not None:
        cleaned = str(explicit_region).strip()
        low = cleaned.lower()
        if low in ("europe", "eu"):
            return "Europe"
        if low in ("canada", "ca", "north_america", "america"):
            return "Canada"
        if cleaned:
            return cleaned
    return "Canada" if float(lon_deg) < -30.0 else "Europe"


def classify_station_belt(region: str, mlat_deg: float) -> int:
    """Map station `(region, mlat_deg)` to canonical belt ID `{0, 1, 2, 3}`."""
    is_aur = abs(float(mlat_deg)) >= AURORAL_MLAT_THRESHOLD_DEG
    reg = infer_station_region(0.0, region).lower()
    if reg == "canada":
        return 3 if is_aur else 2
    return 1 if is_aur else 0


def timestamps_to_unix_s(ts_values: pd.DatetimeIndex | pd.Series | np.ndarray) -> np.ndarray:
    """Convert UTC timestamps cleanly into `int64` Unix seconds regardless of pandas unit."""
    idx = pd.DatetimeIndex(ts_values)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return idx.as_unit("s").astype("int64").to_numpy(dtype=np.int64)


def compute_time_features(timestamps_unix_s: np.ndarray) -> np.ndarray:
    """Compute `[sin_ut, cos_ut, sin_doy, cos_doy]` as `float32[T, 4]`."""
    ts = np.asarray(timestamps_unix_s, dtype=np.int64)
    dt_idx = pd.to_datetime(ts, unit="s", utc=True)
    ut_hours = (
        dt_idx.hour.to_numpy(dtype=np.float64)
        + dt_idx.minute.to_numpy(dtype=np.float64) / 60.0
        + dt_idx.second.to_numpy(dtype=np.float64) / 3600.0
    )
    doy = (dt_idx.day_of_year.to_numpy(dtype=np.float64) - 1.0) + ut_hours / 24.0
    ang_ut = 2.0 * math.pi * (ut_hours / 24.0)
    ang_doy = 2.0 * math.pi * (doy / 365.25)
    feats = np.column_stack(
        [
            np.sin(ang_ut),
            np.cos(ang_ut),
            np.sin(ang_doy),
            np.cos(ang_doy),
        ]
    )
    return feats.astype(np.float32)


def compute_station_geometry(
    lats_deg: np.ndarray | list[float],
    lons_deg: np.ndarray | list[float],
    mlt_mod: Any,
    conv_mod: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute `station_coords: float32[N, 4]` and symmetric `station_dist_km: float32[N, N]`."""
    n_st = len(lats_deg)
    coords = np.empty((n_st, 4), dtype=np.float32)
    for i in range(n_st):
        lat = float(lats_deg[i])
        lon = float(lons_deg[i])
        mlat, mlon = mlt_mod.geo_to_magnetic_dipole(lat, lon)
        coords[i] = (lat, lon, float(mlat), float(mlon))

    dist = np.zeros((n_st, n_st), dtype=np.float32)
    for i in range(n_st):
        lat_i, lon_i = float(lats_deg[i]), float(lons_deg[i])
        for j in range(i + 1, n_st):
            d_ij = float(conv_mod.haversine_km(lat_i, lon_i, float(lats_deg[j]), float(lons_deg[j])))
            dist[i, j] = d_ij
            dist[j, i] = d_ij
    np.fill_diagonal(dist, 0.0)
    return coords, dist


def compute_mlt_matrix(
    lats_deg: np.ndarray | list[float],
    lons_deg: np.ndarray | list[float],
    timestamps_unix_s: np.ndarray,
    mlt_mod: Any,
) -> np.ndarray:
    """Compute `mlt_hours: float32[T, N]` strictly within `[0, 24)`."""
    ts = np.asarray(timestamps_unix_s, dtype=np.int64)
    n_t = ts.shape[0]
    n_st = len(lats_deg)
    out = np.empty((n_t, n_st), dtype=np.float32)
    for i in range(n_st):
        mlt_64 = mlt_mod.compute_mlt_array(float(lats_deg[i]), float(lons_deg[i]), ts)
        mlt_32 = np.mod(mlt_64, 24.0).astype(np.float32)
        # Float32 rounding at 23.9999999 can produce 24.0; wrap cleanly to 0.0.
        mlt_32 = np.where((mlt_32 >= 24.0) | (mlt_32 < 0.0), np.float32(0.0), mlt_32)
        out[:, i] = mlt_32
    return out


def validate_dataset_arrays(
    arrays: dict[str, np.ndarray],
    meta: dict[str, Any] | None = None,
) -> None:
    """Validate shapes, dtypes, masks, timeline continuity, and geometry against §5.1."""
    missing = [k for k in NPZ_DTYPE_SPEC if k not in arrays]
    if missing:
        raise ValueError(f"Missing required dataset keys: {missing}")

    for key, expected_dtype in NPZ_DTYPE_SPEC.items():
        actual = arrays[key]
        if actual.dtype != expected_dtype:
            raise ValueError(
                f"Key {key!r} has dtype {actual.dtype}, expected {expected_dtype}"
            )

    x_raw = arrays["X_raw"]
    if x_raw.ndim != 3 or x_raw.shape[2] != 3 or x_raw.shape[0] == 0 or x_raw.shape[1] == 0:
        raise ValueError(f"Invalid X_raw shape {x_raw.shape}, expected (T>0, N>0, 3)")
    n_t, n_st, _ = x_raw.shape

    eb = arrays["episode_bounds"]
    if eb.ndim != 2 or eb.shape[1] != 2 or eb.shape[0] == 0:
        raise ValueError(f"Invalid episode_bounds shape {eb.shape}, expected (E>0, 2)")
    n_ep = eb.shape[0]

    expected_shapes: dict[str, tuple[int, ...]] = {
        "X_raw": (n_t, n_st, 3),
        "M_valid": (n_t, n_st),
        "M_raw_valid": (n_t, n_st),
        "M_interpolated": (n_t, n_st),
        "M_despiked": (n_t, n_st),
        "M_qc_clean": (n_t, n_st),
        "M_known_spikes": (n_t, n_st),
        "timestamps_unix_s": (n_t,),
        "episode_idx": (n_t,),
        "episode_bounds": (n_ep, 2),
        "station_coords": (n_st, 4),
        "station_dist_km": (n_st, n_st),
        "station_has_baseline": (n_st,),
        "station_belt_id": (n_st,),
        "mlt_hours": (n_t, n_st),
        "time_features": (n_t, 4),
        "sw_omni": (n_t, len(OMNI_COLS)),
        "sw_solar1_bsn": (n_t, len(SOLAR1_BSN_COLS)),
        "sw_solar1_l1": (n_t, len(SOLAR1_L1_COLS)),
        "oulu_delta_n_pct": (n_t,),
        "event_flags": (n_t,),
    }
    for key, exp_shape in expected_shapes.items():
        if arrays[key].shape != exp_shape:
            raise ValueError(f"Key {key!r} has shape {arrays[key].shape}, expected {exp_shape}")

    # Finiteness and mask hierarchy
    x_fin = np.all(np.isfinite(x_raw), axis=-1)
    m_valid = arrays["M_valid"]
    if not np.array_equal(x_fin, m_valid):
        n_mismatch = int(np.sum(x_fin ^ m_valid))
        raise ValueError(f"X_raw finiteness mismatches M_valid at {n_mismatch} positions")
    # Ensure partial NaN cannot hide where ~M_valid
    if np.any(np.isfinite(x_raw[~m_valid])):
        raise ValueError("X_raw contains finite components where M_valid is False")

    m_qc = arrays["M_qc_clean"]
    if np.any(m_qc & (~m_valid)):
        raise ValueError("M_qc_clean must be a subset of M_valid")
    m_spikes = arrays["M_known_spikes"]
    if np.any(m_spikes & (~m_valid)):
        raise ValueError("M_known_spikes must be a subset of M_valid")

    # Episode timeline continuity
    ts = arrays["timestamps_unix_s"]
    ep_idx = arrays["episode_idx"]
    if int(eb[0, 0]) != 0 or int(eb[-1, 1]) != n_t:
        raise ValueError(f"episode_bounds {eb.tolist()} does not span [0, {n_t})")
    for k in range(n_ep):
        s, e = int(eb[k, 0]), int(eb[k, 1])
        if s >= e:
            raise ValueError(f"Empty/invalid episode_bounds[{k}] = [{s}, {e})")
        if k > 0 and s != int(eb[k - 1, 1]):
            raise ValueError(f"Non-contiguous episode_bounds between {k-1} and {k}")
        if not np.all(ep_idx[s:e] == k):
            raise ValueError(f"episode_idx mismatch inside episode {k}")
        if e - s > 1 and not np.all(np.diff(ts[s:e]) == 60):
            raise ValueError(f"Non-60s timestamp step inside episode {k}")
        if k > 0 and int(ts[s]) <= int(ts[s - 1]):
            raise ValueError(f"Non-increasing timestamp across episode boundary {k-1}->{k}")

    # Station geometry
    coords = arrays["station_coords"]
    if not np.all(np.isfinite(coords)):
        raise ValueError("station_coords contains non-finite values")
    if np.any(np.abs(coords[:, 0]) > 90.0) or np.any(np.abs(coords[:, 2]) > 90.0):
        raise ValueError("station latitudes/MLat out of [-90, 90]")
    if np.any(np.abs(coords[:, 1]) > 180.001) or np.any(np.abs(coords[:, 3]) > 180.001):
        raise ValueError("station longitudes/MLon out of [-180, 180]")

    dist = arrays["station_dist_km"]
    if not np.all(np.isfinite(dist)) or np.any(dist < 0.0):
        raise ValueError("station_dist_km must be finite and non-negative")
    if not np.allclose(dist, dist.T, atol=1e-4):
        raise ValueError("station_dist_km is not symmetric")
    if not np.all(np.diag(dist) == 0.0):
        raise ValueError("station_dist_km diagonal must be exactly 0.0")

    belts = arrays["station_belt_id"]
    if np.any((belts < 0) | (belts > 3)):
        raise ValueError("station_belt_id contains values outside {0, 1, 2, 3}")

    # MLT and cyclic time features
    mlt = arrays["mlt_hours"]
    if not np.all(np.isfinite(mlt)) or np.any(mlt < 0.0) or np.any(mlt >= 24.0):
        raise ValueError("mlt_hours must be finite and strictly within [0, 24)")

    tf = arrays["time_features"]
    if not np.all(np.isfinite(tf)):
        raise ValueError("time_features contains non-finite values")
    ut_norm = tf[:, 0] ** 2 + tf[:, 1] ** 2
    doy_norm = tf[:, 2] ** 2 + tf[:, 3] ** 2
    if not np.allclose(ut_norm, 1.0, atol=1e-5) or not np.allclose(doy_norm, 1.0, atol=1e-5):
        raise ValueError("time_features sin^2+cos^2 deviates from 1.0 by > 1e-5")

    if meta is not None:
        codes = meta.get("station_codes")
        if codes is not None:
            if len(codes) != n_st:
                raise ValueError(f"meta.station_codes length {len(codes)} != N={n_st}")
            if list(codes) != sorted(codes):
                raise ValueError("meta.station_codes must be sorted lexicographically")
        ep_ids = meta.get("episode_ids")
        if ep_ids is not None and len(ep_ids) != n_ep:
            raise ValueError(f"meta.episode_ids length {len(ep_ids)} != E={n_ep}")
        if "omni_cols" in meta and list(meta["omni_cols"]) != list(OMNI_COLS):
            raise ValueError("meta.omni_cols mismatch")
        if "solar1_cols" in meta and list(meta["solar1_cols"]) != list(SOLAR1_BSN_COLS):
            raise ValueError("meta.solar1_cols mismatch")


def load_ml_dataset(
    npz_path: Path,
    *,
    validate: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load `.npz` dataset bundle and companion `.meta.json` with optional validation."""
    target = Path(npz_path)
    arrays = load_npz(target)
    meta_path = target.with_suffix(".meta.json")
    meta = read_json(meta_path) if meta_path.is_file() else {}
    if validate:
        validate_dataset_arrays(arrays, meta if meta else None)
    return arrays, meta
