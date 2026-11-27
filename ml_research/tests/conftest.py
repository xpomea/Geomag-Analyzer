"""Shared pytest fixtures for isolated `ml_research` tests."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from geomag_ml.core_bridge import import_conventions, import_mlt, is_backend_available
from geomag_ml.data.schema import (
    EVENT_FLAG_STORM,
    OMNI_COLS,
    SOLAR1_BSN_COLS,
    SOLAR1_L1_COLS,
    classify_station_belt,
    compute_mlt_matrix,
    compute_station_geometry,
    compute_time_features,
    validate_dataset_arrays,
)


class _FallbackMLT:
    """Standalone IGRF-14 centered-dipole and MLT fallback when backend/ is absent (cloud pytest)."""

    _POLE_LAT = 80.86 * math.pi / 180.0
    _POLE_LON = -72.83 * math.pi / 180.0

    def __init__(self) -> None:
        zm = (
            math.cos(self._POLE_LAT) * math.cos(self._POLE_LON),
            math.cos(self._POLE_LAT) * math.sin(self._POLE_LON),
            math.sin(self._POLE_LAT),
        )
        y_raw = (-zm[1], zm[0], 0.0)
        y_norm = math.hypot(*y_raw)
        ym = (y_raw[0] / y_norm, y_raw[1] / y_norm, 0.0)
        xm = (
            ym[1] * zm[2] - ym[2] * zm[1],
            ym[2] * zm[0] - ym[0] * zm[2],
            ym[0] * zm[1] - ym[1] * zm[0],
        )
        self._xm, self._ym, self._zm = xm, ym, zm

    def geo_to_magnetic_dipole(self, lat_deg: float, lon_deg: float) -> tuple[float, float]:
        phi = math.radians(float(lat_deg))
        lam = math.radians(float(lon_deg))
        vx, vy, vz = math.cos(phi) * math.cos(lam), math.cos(phi) * math.sin(lam), math.sin(phi)
        mx = self._xm[0] * vx + self._xm[1] * vy + self._xm[2] * vz
        my = self._ym[0] * vx + self._ym[1] * vy + self._ym[2] * vz
        mz = self._zm[0] * vx + self._zm[1] * vy + self._zm[2] * vz
        mlat = math.degrees(math.atan2(mz, math.hypot(mx, my)))
        mlon = math.degrees(math.atan2(my, mx))
        return mlat, mlon

    def compute_mlt_array(self, lat_deg: float, lon_deg: float, utc_unix_s: np.ndarray) -> np.ndarray:
        _, st_mlon = self.geo_to_magnetic_dipole(lat_deg, lon_deg)
        utc_h = (np.asarray(utc_unix_s, dtype=np.float64) % 86400.0) / 3600.0
        sun_lon = 180.0 - utc_h * 15.0
        sun_rad = np.deg2rad(sun_lon)
        vx = np.cos(sun_rad)
        vy = np.sin(sun_rad)
        mx = self._xm[0] * vx + self._xm[1] * vy
        my = self._ym[0] * vx + self._ym[1] * vy
        sun_mlon = np.rad2deg(np.arctan2(my, mx))
        return np.mod(12.0 + (st_mlon - sun_mlon) / 15.0, 24.0)


class _FallbackConventions:
    """Standalone haversine fallback when backend/ is absent."""

    @staticmethod
    def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r_earth = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
        return 2.0 * r_earth * math.asin(min(1.0, math.sqrt(a)))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def ml_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_synthetic_dataset() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build synthetic mini-dataset (T=600, N=12) strictly adhering to §5.1 without requiring backend/."""
    rng = np.random.default_rng(12345)

    # 12 stations sorted lexicographically across all 4 belts, 1 baseless, 1 ep1-only ("SUA")
    stations_spec = [
        ("ABK", "Europe", 68.36, 18.82, True),   # EU_aur
        ("BEL", "Europe", 51.84, 20.79, True),   # EU_mid (ZS candidate)
        ("BRD", "Canada", 49.87, -99.97, True),  # CA_mid
        ("CLF", "Europe", 48.02, 2.26, True),    # EU_mid
        ("FCC", "Canada", 58.76, -94.09, True),  # CA_aur
        ("GCK", "Europe", 44.63, 20.77, False),  # EU_mid, baseless
        ("HRN", "Europe", 77.00, 15.55, True),   # EU_aur
        ("LVV", "Europe", 49.90, 23.75, True),   # EU_mid (ZS candidate)
        ("MEA", "Canada", 54.62, -113.35, True), # CA_aur (ZS candidate)
        ("OTT", "Canada", 45.40, -75.55, True),  # CA_mid
        ("SUA", "Europe", 44.68, 26.25, True),   # EU_mid, ep1 only
        ("YKC", "Canada", 62.48, -114.48, True), # CA_aur
    ]
    station_codes = [s[0] for s in stations_spec]
    station_regions = [s[1] for s in stations_spec]
    lats = [s[2] for s in stations_spec]
    lons = [s[3] for s in stations_spec]
    has_baseline = np.array([s[4] for s in stations_spec], dtype=np.bool_)

    if is_backend_available():
        mlt_mod: Any = import_mlt()
        conv_mod: Any = import_conventions()
    else:
        mlt_mod = _FallbackMLT()
        conv_mod = _FallbackConventions()
    station_coords, station_dist_km = compute_station_geometry(lats, lons, mlt_mod, conv_mod)
    station_belt_id = np.array(
        [classify_station_belt(station_regions[i], float(station_coords[i, 2])) for i in range(12)],
        dtype=np.int8,
    )

    # 2 episodes of 300 min each separated by 3 days (exceeds 1-day purge gap)
    t_ep0 = 1_791_072_000 + np.arange(300, dtype=np.int64) * 60  # 2026-10-04 00:00Z
    t_ep1 = 1_791_331_200 + np.arange(300, dtype=np.int64) * 60  # 2026-10-07 00:00Z
    timestamps_unix_s = np.concatenate([t_ep0, t_ep1])
    episode_idx = np.concatenate(
        [np.zeros(300, dtype=np.int16), np.ones(300, dtype=np.int16)]
    )
    episode_bounds = np.array([[0, 300], [300, 600]], dtype=np.int64)

    n_t, n_st = 600, 12
    mlt_hours = compute_mlt_matrix(lats, lons, timestamps_unix_s, mlt_mod)
    time_features = compute_time_features(timestamps_unix_s)

    # Construct smooth realistic global + local magnetic variations around static B0
    phase = np.linspace(0.0, 4.0 * math.pi, n_t, dtype=np.float64)[:, None]
    b0_base = np.where(has_baseline[:, None], np.array([[20000.0, 1500.0, 43000.0]]), np.array([[300.0, -50.0, 120.0]]))
    b0_base = b0_base + np.arange(n_st, dtype=np.float64)[:, None] * 100.0

    # Storm depression in episode 1
    storm_shift = np.where(episode_idx[:, None, None] == 1, -40.0, 0.0)
    var_field = (
        25.0 * np.sin(phase)[:, :, None]
        + rng.normal(0.0, 5.0, size=(n_t, n_st, 3))
        + storm_shift
    )
    # Amplify auroral stations
    aur_mask = (station_belt_id == 1) | (station_belt_id == 3)
    var_field[:, aur_mask, :] *= 3.0

    x_raw = (b0_base[None, :, :] + var_field).astype(np.float32)
    m_raw_valid = np.ones((n_t, n_st), dtype=np.bool_)
    # SUA (index 10) absent in episode 0
    sua_idx = station_codes.index("SUA")
    m_raw_valid[:300, sua_idx] = False
    # Add short gap & QC exclusion window on BEL
    m_interpolated = np.zeros((n_t, n_st), dtype=np.bool_)
    m_interpolated[50, 1] = True
    m_raw_valid[50, 1] = False
    x_raw[50, 1, :] = (
        0.5 * (x_raw[49, 1, :].astype(np.float64) + x_raw[51, 1, :].astype(np.float64))
    ).astype(np.float32)
    m_despiked = np.zeros((n_t, n_st), dtype=np.bool_)

    m_valid = m_raw_valid | m_interpolated
    x_raw[~m_valid] = np.nan

    m_qc_clean = m_valid.copy()
    m_qc_clean[400:415, 1] = False  # synthetic QC window
    m_known_spikes = np.zeros((n_t, n_st), dtype=np.bool_)

    sw_omni = np.full((n_t, len(OMNI_COLS)), np.nan, dtype=np.float32)
    sw_omni[:300, OMNI_COLS.index("ae")] = 150.0 + 50.0 * np.sin(phase[:300, 0])
    sw_omni[:300, OMNI_COLS.index("bz_gsm")] = -4.0 * np.cos(phase[:300, 0])
    sw_omni[:300, OMNI_COLS.index("electric_field")] = 2.0 * np.cos(phase[:300, 0])

    sw_solar1_bsn = np.zeros((n_t, len(SOLAR1_BSN_COLS)), dtype=np.float32)
    sw_solar1_bsn[:, 0] = -4.0 * np.cos(phase[:, 0])
    sw_solar1_bsn[:, 1] = 8.0
    sw_solar1_bsn[:, 2] = -500.0
    sw_solar1_bsn[:, 3] = 6.0
    sw_solar1_bsn[:, 4] = 2.5
    sw_solar1_bsn[:, 5] = sw_solar1_bsn[:, 2] * sw_solar1_bsn[:, 0] * 1e-3

    sw_solar1_l1 = sw_solar1_bsn[:, : len(SOLAR1_L1_COLS)].copy()
    oulu_delta_n_pct = (-1.5 * np.sin(phase[:, 0])).astype(np.float32)
    event_flags = np.full(n_t, EVENT_FLAG_STORM, dtype=np.uint8)

    arrays = {
        "X_raw": x_raw,
        "M_valid": m_valid,
        "M_raw_valid": m_raw_valid,
        "M_interpolated": m_interpolated,
        "M_despiked": m_despiked,
        "M_qc_clean": m_qc_clean,
        "M_known_spikes": m_known_spikes,
        "timestamps_unix_s": timestamps_unix_s,
        "episode_idx": episode_idx,
        "episode_bounds": episode_bounds,
        "station_coords": station_coords,
        "station_dist_km": station_dist_km,
        "station_has_baseline": has_baseline,
        "station_belt_id": station_belt_id,
        "mlt_hours": mlt_hours,
        "time_features": time_features,
        "sw_omni": sw_omni,
        "sw_solar1_bsn": sw_solar1_bsn,
        "sw_solar1_l1": sw_solar1_l1,
        "oulu_delta_n_pct": oulu_delta_n_pct,
        "event_flags": event_flags,
    }
    meta = {
        "tag": "synthetic_test",
        "station_codes": station_codes,
        "station_regions": station_regions,
        "episode_ids": ["syn_ep0", "syn_ep1"],
        "omni_cols": list(OMNI_COLS),
        "solar1_cols": list(SOLAR1_BSN_COLS),
        "solar1_l1_cols": list(SOLAR1_L1_COLS),
        "calendar_span_days": 3.21,
        "recommended_use_doy": False,
    }
    validate_dataset_arrays(arrays, meta)
    return arrays, meta


@pytest.fixture()
def synthetic_dataset() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Synthetic mini-dataset (T=600, N=12) adhering strictly to the §5.1 `.npz` contract."""
    return build_synthetic_dataset()
