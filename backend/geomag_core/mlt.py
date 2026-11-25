"""Centered-dipole IGRF-14 (epoch 2026.78) geomagnetic coordinates and MLT utilities.

Matches ``frontend/src/utils/astronomy.ts`` identically using NOAA/Meeus solar
equations and the Laundal & Richmond (2017) centered-dipole orthonormal basis.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

IGRF14_EPOCH = 2026.78
IGRF14_NORTH_POLE_LAT_DEG = 80.86
IGRF14_NORTH_POLE_LON_DEG = -72.83

_DEG2RAD = math.pi / 180.0
_RAD2DEG = 180.0 / math.pi


@dataclass(frozen=True)
class SubsolarPoint:
    """Subsolar coordinates and NOAA/Meeus astronomical quantities at a UTC instant."""

    lat_deg: float
    lon_deg: float
    declination_deg: float
    eot_min: float
    julian_date: float


def wrap_lon_180(lon_deg: float) -> float:
    """Wrap longitude in degrees into (-180.0, 180.0]."""
    wrapped = ((float(lon_deg) + 180.0) % 360.0) - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def _clamp_unit(val: float) -> float:
    v = float(val)
    if math.isnan(v):
        return float("nan")
    return max(-1.0, min(1.0, v))


def utc_to_unix_ms(utc_timestamp: Any) -> float:
    """Convert ISO str, datetime, pandas.Timestamp, np.datetime64, or Unix s to Unix ms."""
    if type(utc_timestamp).__name__ == "NaTType":
        return float("nan")

    if isinstance(utc_timestamp, str):
        s = utc_timestamp.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        else:
            dt = dt.astimezone(UTC)
        return dt.timestamp() * 1000.0

    if isinstance(utc_timestamp, datetime):
        dt = (
            utc_timestamp.replace(tzinfo=UTC)
            if utc_timestamp.tzinfo is None
            else utc_timestamp.astimezone(UTC)
        )
        return dt.timestamp() * 1000.0

    if isinstance(utc_timestamp, np.datetime64):
        if np.isnat(utc_timestamp):
            return float("nan")
        return float(utc_timestamp.astype("datetime64[ms]").astype(np.int64))

    if hasattr(utc_timestamp, "timestamp") and callable(utc_timestamp.timestamp):
        ts_obj = utc_timestamp
        if getattr(ts_obj, "tzinfo", None) is None and hasattr(ts_obj, "tz_localize"):
            ts_obj = ts_obj.tz_localize("UTC")
        elif getattr(ts_obj, "tzinfo", None) is not None and hasattr(ts_obj, "tz_convert"):
            ts_obj = ts_obj.tz_convert("UTC")
        return float(ts_obj.timestamp()) * 1000.0

    if isinstance(utc_timestamp, (int, float, np.integer, np.floating)):
        return float(utc_timestamp) * 1000.0

    raise TypeError(f"Unsupported UTC timestamp type: {type(utc_timestamp).__name__}")


def compute_subsolar_point(utc_timestamp: Any) -> SubsolarPoint:
    """Compute solar declination, Equation of Time, and subsolar [lon, lat] via NOAA/Meeus."""
    utc_ms = utc_to_unix_ms(utc_timestamp)
    julian_date = utc_ms / 86_400_000.0 + 2_440_587.5
    t_cent = (julian_date - 2_451_545.0) / 36_525.0

    l0_deg = (280.46646 + t_cent * (36000.76983 + 0.0003032 * t_cent)) % 360.0
    m_deg = 357.52911 + t_cent * (35999.05029 - 0.0001537 * t_cent)
    m_rad = m_deg * _DEG2RAD
    ecc = 0.016708634 - t_cent * (0.000042037 + 0.0000001267 * t_cent)

    c_deg = (
        math.sin(m_rad) * (1.914602 - t_cent * (0.004817 + 0.000014 * t_cent))
        + math.sin(2.0 * m_rad) * (0.019993 - 0.000101 * t_cent)
        + math.sin(3.0 * m_rad) * 0.000289
    )

    sun_true_lon_deg = l0_deg + c_deg
    omega_deg = 125.04 - 1934.136 * t_cent
    omega_rad = omega_deg * _DEG2RAD
    lambda_app_deg = sun_true_lon_deg - 0.00569 - 0.00478 * math.sin(omega_rad)

    eps0_deg = (
        23.0
        + (26.0 + (21.448 - t_cent * (46.815 + t_cent * (0.00059 - t_cent * 0.001813))) / 60.0)
        / 60.0
    )
    eps_deg = eps0_deg + 0.00256 * math.cos(omega_rad)
    eps_rad = eps_deg * _DEG2RAD

    sin_decl = math.sin(eps_rad) * math.sin(lambda_app_deg * _DEG2RAD)
    declination_deg = math.asin(_clamp_unit(sin_decl)) * _RAD2DEG

    y = math.tan(eps_rad * 0.5) ** 2
    l0_rad = l0_deg * _DEG2RAD
    eot_min = (
        4.0
        * _RAD2DEG
        * (
            y * math.sin(2.0 * l0_rad)
            - 2.0 * ecc * math.sin(m_rad)
            + 4.0 * ecc * y * math.sin(m_rad) * math.cos(2.0 * l0_rad)
            - 0.5 * y * y * math.sin(4.0 * l0_rad)
            - 1.25 * ecc * ecc * math.sin(2.0 * m_rad)
        )
    )

    utc_min = (utc_ms % 86_400_000.0) / 60_000.0
    lon_deg = wrap_lon_180(180.0 - (utc_min + eot_min) / 4.0)

    return SubsolarPoint(
        lat_deg=declination_deg,
        lon_deg=lon_deg,
        declination_deg=declination_deg,
        eot_min=eot_min,
        julian_date=julian_date,
    )


def solar_elevation_deg(lat_deg: float, lon_deg: float, utc_timestamp: Any) -> float:
    """Compute solar elevation angle [deg] at geographic (lat_deg, lon_deg) and UTC instant."""
    sub = compute_subsolar_point(utc_timestamp)
    phi = float(lat_deg) * _DEG2RAD
    decl = sub.declination_deg * _DEG2RAD
    d_lon = (float(lon_deg) - sub.lon_deg) * _DEG2RAD
    sin_elev = (
        math.sin(phi) * math.sin(decl)
        + math.cos(phi) * math.cos(decl) * math.cos(d_lon)
    )
    return math.asin(_clamp_unit(sin_elev)) * _RAD2DEG


def _build_dipole_basis(
    pole_lat_deg: float,
    pole_lon_deg: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    lat_p = float(pole_lat_deg) * _DEG2RAD
    lon_p = float(pole_lon_deg) * _DEG2RAD
    zm = (
        math.cos(lat_p) * math.cos(lon_p),
        math.cos(lat_p) * math.sin(lon_p),
        math.sin(lat_p),
    )
    y_raw = (-zm[1], zm[0], 0.0)
    y_norm = math.hypot(y_raw[0], y_raw[1], y_raw[2]) or 1.0
    ym = (y_raw[0] / y_norm, y_raw[1] / y_norm, y_raw[2] / y_norm)
    xm = (
        ym[1] * zm[2] - ym[2] * zm[1],
        ym[2] * zm[0] - ym[0] * zm[2],
        ym[0] * zm[1] - ym[1] * zm[0],
    )
    return xm, ym, zm


_DEFAULT_BASIS = _build_dipole_basis(IGRF14_NORTH_POLE_LAT_DEG, IGRF14_NORTH_POLE_LON_DEG)


def _get_basis(
    pole_lat_deg: float,
    pole_lon_deg: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    if (
        pole_lat_deg == IGRF14_NORTH_POLE_LAT_DEG
        and pole_lon_deg == IGRF14_NORTH_POLE_LON_DEG
    ):
        return _DEFAULT_BASIS
    return _build_dipole_basis(pole_lat_deg, pole_lon_deg)


def geo_to_magnetic_dipole(
    lat_deg: float,
    lon_deg: float,
    *,
    pole_lat_deg: float = IGRF14_NORTH_POLE_LAT_DEG,
    pole_lon_deg: float = IGRF14_NORTH_POLE_LON_DEG,
) -> tuple[float, float]:
    """Convert geographic (lat_deg, lon_deg) to centered-dipole (mlat_deg, mlon_deg)."""
    xm, ym, zm = _get_basis(pole_lat_deg, pole_lon_deg)
    phi = float(lat_deg) * _DEG2RAD
    lam = float(lon_deg) * _DEG2RAD
    vx = math.cos(phi) * math.cos(lam)
    vy = math.cos(phi) * math.sin(lam)
    vz = math.sin(phi)

    mx = xm[0] * vx + xm[1] * vy + xm[2] * vz
    my = ym[0] * vx + ym[1] * vy + ym[2] * vz
    mz = zm[0] * vx + zm[1] * vy + zm[2] * vz

    horiz = math.hypot(mx, my)
    mlat_deg = math.atan2(mz, horiz) * _RAD2DEG
    mlon_deg = 0.0 if horiz < 1e-12 else wrap_lon_180(math.atan2(my, mx) * _RAD2DEG)
    return mlat_deg, mlon_deg


def magnetic_dipole_to_geo(
    mlat_deg: float,
    mlon_deg: float,
    *,
    pole_lat_deg: float = IGRF14_NORTH_POLE_LAT_DEG,
    pole_lon_deg: float = IGRF14_NORTH_POLE_LON_DEG,
) -> tuple[float, float]:
    """Convert centered-dipole (mlat_deg, mlon_deg) back to geographic (lat_deg, lon_deg)."""
    xm, ym, zm = _get_basis(pole_lat_deg, pole_lon_deg)
    mlat = float(mlat_deg) * _DEG2RAD
    mlon = float(mlon_deg) * _DEG2RAD
    mx = math.cos(mlat) * math.cos(mlon)
    my = math.cos(mlat) * math.sin(mlon)
    mz = math.sin(mlat)

    vx = xm[0] * mx + ym[0] * my + zm[0] * mz
    vy = xm[1] * mx + ym[1] * my + zm[1] * mz
    vz = xm[2] * mx + ym[2] * my + zm[2] * mz

    horiz = math.hypot(vx, vy)
    lat_deg = math.atan2(vz, horiz) * _RAD2DEG
    lon_deg = 0.0 if horiz < 1e-12 else wrap_lon_180(math.atan2(vy, vx) * _RAD2DEG)
    return lat_deg, lon_deg


def compute_mlt(
    lat_deg: float,
    lon_deg: float,
    utc_timestamp: Any,
    *,
    pole_lat_deg: float = IGRF14_NORTH_POLE_LAT_DEG,
    pole_lon_deg: float = IGRF14_NORTH_POLE_LON_DEG,
) -> float:
    """Compute centered-dipole Magnetic Local Time (MLT in [0, 24) hours)."""
    sub = compute_subsolar_point(utc_timestamp)
    _, st_mlon = geo_to_magnetic_dipole(
        lat_deg,
        lon_deg,
        pole_lat_deg=pole_lat_deg,
        pole_lon_deg=pole_lon_deg,
    )
    _, sun_mlon = geo_to_magnetic_dipole(
        sub.lat_deg,
        sub.lon_deg,
        pole_lat_deg=pole_lat_deg,
        pole_lon_deg=pole_lon_deg,
    )
    return (12.0 + (st_mlon - sun_mlon) / 15.0) % 24.0


def compute_mlt_array(
    lat_deg: float,
    lon_deg: float,
    utc_timestamps: Iterable[Any] | np.ndarray,
    *,
    pole_lat_deg: float = IGRF14_NORTH_POLE_LAT_DEG,
    pole_lon_deg: float = IGRF14_NORTH_POLE_LON_DEG,
) -> np.ndarray:
    """Vectorized Magnetic Local Time [0, 24) h for one station across UTC timestamps."""
    if isinstance(utc_timestamps, np.ndarray) and np.issubdtype(
        utc_timestamps.dtype, np.datetime64
    ):
        nat_mask = np.isnat(utc_timestamps)
        utc_ms = np.where(
            nat_mask,
            np.nan,
            utc_timestamps.astype("datetime64[ms]").astype(np.float64),
        )
    elif isinstance(utc_timestamps, np.ndarray) and np.issubdtype(
        utc_timestamps.dtype, np.number
    ):
        utc_ms = utc_timestamps.astype(np.float64) * 1000.0
    else:
        utc_ms = np.array([utc_to_unix_ms(t) for t in utc_timestamps], dtype=np.float64)

    if utc_ms.size == 0:
        return np.empty(0, dtype=np.float64)

    with np.errstate(invalid="ignore"):
        julian_date = utc_ms / 86_400_000.0 + 2_440_587.5
        t_cent = (julian_date - 2_451_545.0) / 36_525.0

        l0_deg = (280.46646 + t_cent * (36000.76983 + 0.0003032 * t_cent)) % 360.0
        m_rad = (357.52911 + t_cent * (35999.05029 - 0.0001537 * t_cent)) * _DEG2RAD
        ecc = 0.016708634 - t_cent * (0.000042037 + 0.0000001267 * t_cent)

        c_deg = (
            np.sin(m_rad) * (1.914602 - t_cent * (0.004817 + 0.000014 * t_cent))
            + np.sin(2.0 * m_rad) * (0.019993 - 0.000101 * t_cent)
            + np.sin(3.0 * m_rad) * 0.000289
        )

        omega_rad = (125.04 - 1934.136 * t_cent) * _DEG2RAD
        lambda_app_deg = (l0_deg + c_deg) - 0.00569 - 0.00478 * np.sin(omega_rad)

        eps0_deg = (
            23.0
            + (
                26.0
                + (21.448 - t_cent * (46.815 + t_cent * (0.00059 - t_cent * 0.001813))) / 60.0
            )
            / 60.0
        )
        eps_rad = (eps0_deg + 0.00256 * np.cos(omega_rad)) * _DEG2RAD

        decl_rad = np.arcsin(
            np.clip(np.sin(eps_rad) * np.sin(lambda_app_deg * _DEG2RAD), -1.0, 1.0),
        )

        y = np.tan(eps_rad * 0.5) ** 2
        l0_rad = l0_deg * _DEG2RAD
        eot_min = (
            4.0
            * _RAD2DEG
            * (
                y * np.sin(2.0 * l0_rad)
                - 2.0 * ecc * np.sin(m_rad)
                + 4.0 * ecc * y * np.sin(m_rad) * np.cos(2.0 * l0_rad)
                - 0.5 * y * y * np.sin(4.0 * l0_rad)
                - 1.25 * ecc * ecc * np.sin(2.0 * m_rad)
            )
        )

        utc_min = (utc_ms % 86_400_000.0) / 60_000.0
        sun_lon_rad = (180.0 - (utc_min + eot_min) / 4.0) * _DEG2RAD

        xm, ym, _ = _get_basis(pole_lat_deg, pole_lon_deg)
        cos_decl = np.cos(decl_rad)
        vx = cos_decl * np.cos(sun_lon_rad)
        vy = cos_decl * np.sin(sun_lon_rad)
        vz = np.sin(decl_rad)

        mx = xm[0] * vx + xm[1] * vy + xm[2] * vz
        my = ym[0] * vx + ym[1] * vy + ym[2] * vz
        sun_mlon = np.arctan2(my, mx) * _RAD2DEG

        _, st_mlon = geo_to_magnetic_dipole(
            lat_deg,
            lon_deg,
            pole_lat_deg=pole_lat_deg,
            pole_lon_deg=pole_lon_deg,
        )
        return (12.0 + (st_mlon - sun_mlon) / 15.0) % 24.0


def mlt_sector(mlt_hours: float) -> str:
    """Return 'morning' for [0, 12) MLT or 'afternoon' for [12, 24) MLT."""
    h = float(mlt_hours) % 24.0
    return "morning" if h < 12.0 else "afternoon"
