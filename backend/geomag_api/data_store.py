"""Read-only catalog, magnetometer Parquet, SOLAR-1 ballistic shift, OMNI, and OULU loaders."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geomag_core import compute_quiet_declination_rad, rotate_xyz_to_magnetic

_SWIPS_FLAGS_WARNING = (
    "SOLAR-1 SWiPS flag bits {0,1,2,3,8,10,11,16} are undocumented in NOAA README "
    "and NetCDF metadata; default ETL filters flags == 0 plus physical bounds."
)

_SOLAR1_NUMERIC_COLS = (
    "bx_gse_nT",
    "by_gse_nT",
    "bz_gse_nT",
    "bx_gsm_nT",
    "by_gsm_nT",
    "bz_gsm_nT",
    "b_mag_gsm_nT",
    "proton_n_cm3",
    "proton_T_K",
    "vx_gse_kms",
    "vy_gse_kms",
    "vz_gse_kms",
    "vx_gsm_kms",
    "vy_gsm_kms",
    "vz_gsm_kms",
    "pdyn_nPa",
)

_OMNI_SERIES_COLS = (
    "b_mag_nT",
    "bx_gse_nT",
    "by_gse_nT",
    "bz_gse_nT",
    "by_gsm_nT",
    "bz_gsm_nT",
    "flow_speed_kms",
    "vx_gse_kms",
    "vy_gse_kms",
    "vz_gse_kms",
    "proton_n_cm3",
    "proton_T_K",
    "pdyn_nPa",
    "electric_field_mVm",
    "plasma_beta",
    "alfven_mach",
    "magnetosonic_mach",
    "ae_nT",
    "al_nT",
    "au_nT",
    "sym_d_nT",
    "sym_h_nT",
    "asy_d_nT",
    "asy_h_nT",
    "pc_n_index",
)


class NotFoundError(LookupError):
    """Raised when an episode, station, job, or resource ID does not exist."""


@dataclass(frozen=True)
class StationEpisodeData:
    """Magnetometer time series slice loaded strictly read-only for one station/episode."""

    episode_id: str
    station_iaga: str
    station_meta: dict[str, Any]
    available: bool
    timestamps_utc: pd.DatetimeIndex
    x_nT: np.ndarray
    y_nT: np.ndarray
    z_nT: np.ndarray
    f_scalar_nT: np.ndarray
    is_valid_xyz: np.ndarray
    is_interpolated: np.ndarray
    was_despiked: np.ndarray
    valid_pct: float
    warnings: tuple[str, ...]

    def to_xyz_matrix(self) -> np.ndarray:
        """Return contiguous (N, 3) float64 array [X, Y, Z]."""
        return np.column_stack([self.x_nT, self.y_nT, self.z_nT])


def _parse_utc_bound(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def slice_utc_mask(
    timestamps: pd.DatetimeIndex,
    start_utc: str | None = None,
    end_utc: str | None = None,
) -> np.ndarray:
    """Build boolean mask for [start_utc, end_utc] UTC bounds."""
    mask = np.ones(len(timestamps), dtype=bool)
    start_ts = _parse_utc_bound(start_utc) if start_utc is not None else None
    end_ts = _parse_utc_bound(end_utc) if end_utc is not None else None
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError(f"start_utc ({start_utc}) must be <= end_utc ({end_utc}).")
    if start_ts is not None:
        mask &= np.asarray(timestamps >= start_ts, dtype=bool)
    if end_ts is not None:
        mask &= np.asarray(timestamps <= end_ts, dtype=bool)
    if not np.any(mask):
        raise ValueError("Requested UTC window contains 0 samples inside the episode.")
    return mask


def compute_ballistic_shift(
    timestamps_utc: pd.DatetimeIndex,
    vx_kms: np.ndarray,
    *,
    ref_distance_km: float = 1_400_000.0,
    smooth_window_min: int = 31,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute smoothed, strictly monotonic L1 -> BSN propagation arrival timestamps [s]."""
    if ref_distance_km <= 0.0:
        raise ValueError(f"ref_distance_km must be positive, got {ref_distance_km}")
    if smooth_window_min < 1:
        raise ValueError(f"smooth_window_min must be >= 1, got {smooth_window_min}")

    idx = pd.DatetimeIndex(timestamps_utc).tz_convert("UTC")
    t_sec = idx.as_unit("ns").astype("int64").to_numpy(dtype=np.float64) / 1e9
    n = t_sec.shape[0]
    if n == 0:
        return t_sec, np.zeros(0, dtype=np.float64)

    vx = np.asarray(vx_kms, dtype=np.float64)
    speed = np.where(np.isfinite(vx) & (np.abs(vx) > 10.0), np.abs(vx), np.nan)
    fin = np.isfinite(speed)
    if not np.any(fin):
        speed_filled = np.full(n, 450.0, dtype=np.float64)
    else:
        grid = np.arange(n, dtype=np.float64)
        speed_filled = np.interp(grid, grid[fin], speed[fin])

    speed_smooth = (
        pd.Series(speed_filled)
        .rolling(window=int(smooth_window_min), center=True, min_periods=1)
        .median()
        .to_numpy(dtype=np.float64)
    )
    speed_clamped = np.clip(speed_smooth, 200.0, 2000.0)
    delay_sec = float(ref_distance_km) / speed_clamped
    t_arr_sec = t_sec + delay_sec

    # Enforce strict forward monotonicity so sharp V_x speed jumps never invert sample order
    for i in range(1, n):
        if t_arr_sec[i] <= t_arr_sec[i - 1]:
            t_arr_sec[i] = t_arr_sec[i - 1] + 1.0

    effective_delay_min = (t_arr_sec - t_sec) / 60.0
    return t_arr_sec, effective_delay_min


def _resample_monotonic_series(
    t_arr_sec: np.ndarray,
    values: np.ndarray,
    t_grid_sec: np.ndarray,
) -> np.ndarray:
    """Interpolate strictly increasing shifted series onto target grid without bridging gaps."""
    n_src = t_arr_sec.shape[0]
    out = np.full(t_grid_sec.shape[0], np.nan, dtype=np.float64)
    if n_src == 0 or t_grid_sec.size == 0:
        return out

    in_bounds = (t_grid_sec >= t_arr_sec[0]) & (t_grid_sec <= t_arr_sec[-1])
    if not np.any(in_bounds):
        return out

    src_pos = np.interp(
        t_grid_sec[in_bounds],
        t_arr_sec,
        np.arange(n_src, dtype=np.float64),
    )
    i0 = np.clip(np.floor(src_pos).astype(np.int64), 0, n_src - 1)
    i1 = np.clip(i0 + 1, 0, n_src - 1)
    frac = src_pos - i0.astype(np.float64)

    arr = np.asarray(values, dtype=np.float64)
    v0 = arr[i0]
    v1 = arr[i1]
    both_ok = np.isfinite(v0) & np.isfinite(v1)
    exact_0 = (frac <= 1e-9) & np.isfinite(v0)

    interp_vals = np.full(src_pos.shape[0], np.nan, dtype=np.float64)
    interp_vals[both_ok] = v0[both_ok] * (1.0 - frac[both_ok]) + v1[both_ok] * frac[both_ok]
    interp_vals[exact_0] = v0[exact_0]
    out[in_bounds] = interp_vals
    return out


def _clean_record(row: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for k, v in row.items():
        if v is None:
            cleaned[k] = None
        elif isinstance(v, float) and not math.isfinite(v):
            cleaned[k] = None
        elif isinstance(v, (np.floating, float)):
            f_v = float(v)
            cleaned[k] = f_v if math.isfinite(f_v) else None
        elif isinstance(v, (np.integer,)):
            cleaned[k] = int(v)
        elif isinstance(v, (np.bool_,)):
            cleaned[k] = bool(v)
        else:
            cleaned[k] = v
    return cleaned


class DataStore:
    """Thread-safe, strictly read-only repository of normalized storm datasets."""

    def __init__(self, processed_dir: Path) -> None:
        self.processed_dir = Path(processed_dir)
        self.catalog_db_path = self.processed_dir / "catalog.sqlite"
        if not self.catalog_db_path.exists():
            raise FileNotFoundError(
                f"Normalized catalog.sqlite not found at {self.catalog_db_path}"
            )
        self._lock = threading.RLock()
        self._geomag_cache: dict[str, pd.DataFrame] = {}
        self._solar1_df: pd.DataFrame | None = None
        self._omni_df: pd.DataFrame | None = None
        self._oulu_df: pd.DataFrame | None = None

        self._episodes: list[dict[str, Any]] = []
        self._stations: list[dict[str, Any]] = []
        self._availability: list[dict[str, Any]] = []
        self._metadata: dict[str, Any] = {}
        self._load_catalog_readonly()

        self._episodes_by_id = {ep["episode_id"]: ep for ep in self._episodes}
        self._stations_by_code = {st["iaga_code"]: st for st in self._stations}

    def _load_catalog_readonly(self) -> None:
        uri = self.catalog_db_path.resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            ep_df = pd.read_sql_query("SELECT * FROM episodes ORDER BY start_utc", conn)
            ep_df["omni_available"] = ep_df["omni_available"].astype(bool)
            self._episodes = [_clean_record(r) for r in ep_df.to_dict(orient="records")]

            st_df = pd.read_sql_query("SELECT * FROM stations ORDER BY iaga_code", conn)
            st_df["has_baseline"] = st_df["has_baseline"].astype(bool)
            self._stations = [_clean_record(r) for r in st_df.to_dict(orient="records")]

            av_df = pd.read_sql_query(
                "SELECT * FROM availability ORDER BY episode_id, station_iaga, file_date",
                conn,
            )
            self._availability = [_clean_record(r) for r in av_df.to_dict(orient="records")]

            meta_rows = conn.execute("SELECT key, value FROM dataset_metadata").fetchall()
            parsed_meta: dict[str, Any] = {}
            for k, v in meta_rows:
                try:
                    parsed_meta[str(k)] = json.loads(v)
                except (TypeError, json.JSONDecodeError):
                    parsed_meta[str(k)] = v
            self._metadata = parsed_meta
        finally:
            conn.close()

    def get_episodes(self) -> list[dict[str, Any]]:
        return [dict(ep) for ep in self._episodes]

    def get_stations(self, station_type: str | None = None) -> list[dict[str, Any]]:
        if station_type is None:
            return [dict(st) for st in self._stations]
        t_norm = station_type.strip().lower()
        return [dict(st) for st in self._stations if st["station_type"].lower() == t_norm]

    def get_availability(
        self,
        *,
        episode_id: str | None = None,
        station_iaga: str | None = None,
    ) -> list[dict[str, Any]]:
        rows = self._availability
        if episode_id is not None:
            self.require_episode(episode_id)
            rows = [r for r in rows if r["episode_id"] == episode_id]
        if station_iaga is not None:
            code = self.require_station(station_iaga)["iaga_code"]
            rows = [r for r in rows if r["station_iaga"] == code]
        return [dict(r) for r in rows]

    def get_metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def require_episode(self, episode_id: str) -> dict[str, Any]:
        ep = self._episodes_by_id.get(episode_id)
        if ep is None:
            raise NotFoundError(f"Unknown episode_id {episode_id!r}")
        return dict(ep)

    def require_station(self, station_iaga: str) -> dict[str, Any]:
        code = station_iaga.strip().upper()
        st = self._stations_by_code.get(code)
        if st is None:
            raise NotFoundError(f"Unknown station {station_iaga!r}")
        return dict(st)

    def require_magnetometer(self, station_iaga: str) -> dict[str, Any]:
        st = self.require_station(station_iaga)
        if st["station_type"] != "magnetometer":
            raise ValueError(
                f"Station {st['iaga_code']} has station_type={st['station_type']!r}, "
                "expected 'magnetometer'."
            )
        return st

    def _get_geomag_df(self, episode_id: str) -> pd.DataFrame:
        self.require_episode(episode_id)
        with self._lock:
            df = self._geomag_cache.get(episode_id)
            if df is None:
                path = self.processed_dir / "geomag" / f"{episode_id}.parquet"
                df = pd.read_parquet(path)
                df["timestamp_utc"] = pd.DatetimeIndex(df["timestamp_utc"]).tz_convert("UTC")
                self._geomag_cache[episode_id] = df
            return df

    def _get_solar1_df(self) -> pd.DataFrame:
        with self._lock:
            if self._solar1_df is None:
                df = pd.read_parquet(
                    self.processed_dir / "solar_wind" / "solar1_l1_1min.parquet"
                )
                df["timestamp_utc"] = pd.DatetimeIndex(df["timestamp_utc"]).tz_convert("UTC")
                self._solar1_df = df
            return self._solar1_df

    def _get_omni_df(self) -> pd.DataFrame:
        with self._lock:
            if self._omni_df is None:
                df = pd.read_parquet(
                    self.processed_dir / "solar_wind" / "omni_bsn_1min.parquet"
                )
                df["timestamp_utc"] = pd.DatetimeIndex(df["timestamp_utc"]).tz_convert("UTC")
                self._omni_df = df
            return self._omni_df

    def _get_oulu_df(self) -> pd.DataFrame:
        with self._lock:
            if self._oulu_df is None:
                df = pd.read_parquet(
                    self.processed_dir / "cosmic_rays" / "oulu_nm_1min.parquet"
                )
                df["timestamp_utc"] = pd.DatetimeIndex(df["timestamp_utc"]).tz_convert("UTC")
                self._oulu_df = df
            return self._oulu_df

    def episode_magnetometer_codes(self, episode_id: str) -> list[str]:
        """Return sorted magnetometer codes present in `{episode_id}.parquet`."""
        df = self._get_geomag_df(episode_id)
        return sorted(str(c) for c in df["station_iaga"].unique().tolist())

    def load_station_episode(
        self,
        episode_id: str,
        station_iaga: str,
        *,
        start_utc: str | None = None,
        end_utc: str | None = None,
    ) -> StationEpisodeData:
        """Load magnetometer vector series for one station/episode, handling absent stations."""
        ep = self.require_episode(episode_id)
        st = self.require_magnetometer(station_iaga)
        code = st["iaga_code"]
        df = self._get_geomag_df(episode_id)
        sub = df[df["station_iaga"] == code]

        warnings: list[str] = []
        if sub.empty:
            ep_ts = pd.date_range(ep["start_utc"], ep["end_utc"], freq="1min", tz="UTC")
            sl_mask = slice_utc_mask(ep_ts, start_utc, end_utc)
            ts_out = ep_ts[sl_mask]
            n_out = len(ts_out)
            present_eps = sorted(
                {r["episode_id"] for r in self._availability if r["station_iaga"] == code}
            )
            warnings.append(
                f"Station {code} has no data in {episode_id}; "
                f"available episodes: {', '.join(present_eps) if present_eps else 'none'}."
            )
            return StationEpisodeData(
                episode_id=episode_id,
                station_iaga=code,
                station_meta=st,
                available=False,
                timestamps_utc=ts_out,
                x_nT=np.full(n_out, np.nan, dtype=np.float64),
                y_nT=np.full(n_out, np.nan, dtype=np.float64),
                z_nT=np.full(n_out, np.nan, dtype=np.float64),
                f_scalar_nT=np.full(n_out, np.nan, dtype=np.float64),
                is_valid_xyz=np.zeros(n_out, dtype=bool),
                is_interpolated=np.zeros(n_out, dtype=bool),
                was_despiked=np.zeros(n_out, dtype=bool),
                valid_pct=0.0,
                warnings=tuple(warnings),
            )

        sub = sub.sort_values("timestamp_utc")
        full_ts = pd.DatetimeIndex(sub["timestamp_utc"]).tz_convert("UTC")
        sl_mask = slice_utc_mask(full_ts, start_utc, end_utc)
        ts_out = full_ts[sl_mask]
        x_arr = sub["x_nT"].to_numpy(dtype=np.float64)[sl_mask]
        y_arr = sub["y_nT"].to_numpy(dtype=np.float64)[sl_mask]
        z_arr = sub["z_nT"].to_numpy(dtype=np.float64)[sl_mask]
        f_arr = sub["f_scalar_nT"].to_numpy(dtype=np.float64)[sl_mask]
        valid = sub["is_valid_xyz"].to_numpy(dtype=bool)[sl_mask]
        interp = sub["is_interpolated"].to_numpy(dtype=bool)[sl_mask]
        despiked = sub["was_despiked"].to_numpy(dtype=bool)[sl_mask]
        valid_pct = round(100.0 * float(np.mean(valid)), 4)

        if valid_pct < 99.0:
            warnings.append(
                f"Station {code} has partial vector coverage in selected window ({valid_pct:.2f}%)."
            )
        if not bool(st["has_baseline"]):
            warnings.append(
                f"Station {code} is a variation-only observatory without absolute baseline; "
                "automatic D0 rotation is disabled unless external_d0_deg is supplied."
            )

        return StationEpisodeData(
            episode_id=episode_id,
            station_iaga=code,
            station_meta=st,
            available=True,
            timestamps_utc=ts_out,
            x_nT=x_arr,
            y_nT=y_arr,
            z_nT=z_arr,
            f_scalar_nT=f_arr,
            is_valid_xyz=valid,
            is_interpolated=interp,
            was_despiked=despiked,
            valid_pct=valid_pct,
            warnings=tuple(warnings),
        )

    def apply_coord_frame(
        self,
        data: StationEpisodeData,
        *,
        coord_frame: str = "XYZ",
        external_d0_deg: float | None = None,
    ) -> tuple[np.ndarray, float, str]:
        """Return (N, 3) vector array in requested frame, effective d0_deg, canonical frame name."""
        frame_key = coord_frame.strip().upper()
        if frame_key == "XYZ":
            return data.to_xyz_matrix(), 0.0, "XYZ"
        if frame_key != "XMYM":
            raise ValueError(
                f"Unsupported coord_frame {coord_frame!r}; expected 'XYZ' or 'XmYm'."
            )

        has_base = bool(data.station_meta["has_baseline"])
        if not data.available:
            if not has_base and external_d0_deg is None:
                raise ValueError(
                    f"Automatic D0 rotation is disabled for baseless station {data.station_iaga}."
                )
            d0_deg = float(external_d0_deg) if external_d0_deg is not None else 0.0
            return data.to_xyz_matrix(), d0_deg, "XmYm"

        d0_rad = compute_quiet_declination_rad(
            data.x_nT,
            data.y_nT,
            data.z_nT,
            has_baseline=has_base,
            external_d0_deg=external_d0_deg,
        )
        xyz_rot = rotate_xyz_to_magnetic(data.to_xyz_matrix(), d0_rad)
        return xyz_rot, math.degrees(d0_rad), "XmYm"

    def compute_station_d0_deg(
        self,
        episode_id: str,
        station_iaga: str,
        *,
        coord_frame: str = "XYZ",
        external_d0_deg: float | None = None,
    ) -> float:
        """Compute canonical episode-level D0 [deg] for Phase 1 / WTC cache keys."""
        frame_key = coord_frame.strip().upper()
        if frame_key == "XYZ":
            return 0.0
        full_data = self.load_station_episode(episode_id, station_iaga)
        _, d0_deg, _ = self.apply_coord_frame(
            full_data, coord_frame="XmYm", external_d0_deg=external_d0_deg
        )
        return d0_deg

    def load_solar1(
        self,
        episode_id: str,
        *,
        shift_mode: str = "l1_raw",
        manual_lag_min: float | None = None,
        ref_distance_km: float = 1_400_000.0,
        smooth_window_min: int = 31,
        start_utc: str | None = None,
        end_utc: str | None = None,
    ) -> dict[str, Any]:
        """Load SOLAR-1 L1 data with optional ballistic or manual propagation shift to BSN."""
        self.require_episode(episode_id)
        mode = shift_mode.strip().lower()
        if mode not in ("l1_raw", "ballistic_auto", "manual"):
            raise ValueError(
                f"Unsupported solar1_shift {shift_mode!r}; "
                "expected 'l1_raw', 'ballistic_auto', or 'manual'."
            )

        df_all = self._get_solar1_df()
        ep_df = df_all[df_all["episode_id"] == episode_id].sort_values("timestamp_utc")
        ep_ts = pd.DatetimeIndex(ep_df["timestamp_utc"]).tz_convert("UTC")
        sl_mask = slice_utc_mask(ep_ts, start_utc, end_utc)
        target_ts = ep_ts[sl_mask]

        if mode == "l1_raw":
            series_map = {
                col: ep_df[col].to_numpy(dtype=np.float64)[sl_mask]
                for col in _SOLAR1_NUMERIC_COLS
            }
            flags_out = ep_df["swips_flags"].to_numpy()[sl_mask]
            valid_out = ep_df["is_swips_valid"].to_numpy(dtype=bool)[sl_mask]
            delay_out = np.zeros(len(target_ts), dtype=np.float64)
            time_frame = "L1"
        else:
            t_src_sec = ep_ts.as_unit("ns").astype("int64").to_numpy(dtype=np.float64) / 1e9
            if mode == "ballistic_auto":
                t_arr_sec, eff_delay = compute_ballistic_shift(
                    ep_ts,
                    ep_df["vx_gse_kms"].to_numpy(dtype=np.float64),
                    ref_distance_km=ref_distance_km,
                    smooth_window_min=smooth_window_min,
                )
            else:
                if manual_lag_min is None:
                    raise ValueError("manual_lag_min must be provided when shift_mode='manual'.")
                lag_sec = float(manual_lag_min) * 60.0
                t_arr_sec = t_src_sec + lag_sec
                eff_delay = np.full(len(ep_ts), float(manual_lag_min), dtype=np.float64)

            t_target_sec = (
                target_ts.as_unit("ns").astype("int64").to_numpy(dtype=np.float64) / 1e9
            )
            series_map = {
                col: _resample_monotonic_series(
                    t_arr_sec,
                    ep_df[col].to_numpy(dtype=np.float64),
                    t_target_sec,
                )
                for col in _SOLAR1_NUMERIC_COLS
            }
            delay_out = _resample_monotonic_series(t_arr_sec, eff_delay, t_target_sec)
            valid_out = np.isfinite(series_map["proton_n_cm3"]) & np.isfinite(
                series_map["vx_gse_kms"]
            )
            flags_out = np.where(valid_out, 0, None)
            time_frame = f"L1_shifted_{mode}"

        return {
            "timestamps_utc": target_ts,
            "time_frame": time_frame,
            "shift_mode": mode,
            "ref_distance_km": float(ref_distance_km),
            "propagation_delay_min": delay_out,
            "series": series_map,
            "swips_flags": flags_out,
            "is_swips_valid": valid_out,
            "warning": _SWIPS_FLAGS_WARNING,
        }

    def load_omni(
        self,
        episode_id: str,
        *,
        start_utc: str | None = None,
        end_utc: str | None = None,
    ) -> dict[str, Any]:
        """Load OMNI_1min (BSN) data for episode if covered (Oct 4-5, Oct 9-11)."""
        ep = self.require_episode(episode_id)
        if not bool(ep["omni_available"]):
            ep_ts = pd.date_range(ep["start_utc"], ep["end_utc"], freq="1min", tz="UTC")
            sl_mask = slice_utc_mask(ep_ts, start_utc, end_utc)
            return {
                "available": False,
                "time_frame": "BSN",
                "timestamps_utc": ep_ts[sl_mask],
                "series": {},
                "warning": (
                    f"OMNI_1min (BSN) data are not available for {episode_id} "
                    "(coverage: Oct 4-5 and Oct 9-11 only)."
                ),
            }

        df_all = self._get_omni_df()
        sub = df_all[df_all["episode_id"] == episode_id].sort_values("timestamp_utc")
        ep_ts = pd.DatetimeIndex(sub["timestamp_utc"]).tz_convert("UTC")
        sl_mask = slice_utc_mask(ep_ts, start_utc, end_utc)
        series_map = {
            col: sub[col].to_numpy(dtype=np.float64)[sl_mask]
            for col in _OMNI_SERIES_COLS
            if col in sub.columns
        }
        return {
            "available": True,
            "time_frame": "BSN",
            "timestamps_utc": ep_ts[sl_mask],
            "series": series_map,
            "warning": None,
        }

    def load_oulu(
        self,
        episode_id: str,
        *,
        start_utc: str | None = None,
        end_utc: str | None = None,
    ) -> dict[str, Any]:
        """Load OULU Neutron Monitor 1-min count rate and delta_n_pct variations."""
        ep = self.require_episode(episode_id)
        df_all = self._get_oulu_df()
        sub = df_all[df_all["episode_id"] == episode_id].sort_values("timestamp_utc")
        ep_ts = pd.DatetimeIndex(sub["timestamp_utc"]).tz_convert("UTC")
        sl_mask = slice_utc_mask(ep_ts, start_utc, end_utc)
        return {
            "available": True,
            "timestamps_utc": ep_ts[sl_mask],
            "baseline_cpm": ep.get("oulu_baseline_cpm"),
            "corrected_cpm": sub["corrected_cpm"].to_numpy(dtype=np.float64)[sl_mask],
            "uncorrected_cpm": sub["uncorrected_cpm"].to_numpy(dtype=np.float64)[sl_mask],
            "pressure_mbar": sub["pressure_mbar"].to_numpy(dtype=np.float64)[sl_mask],
            "delta_n_pct": sub["delta_n_pct"].to_numpy(dtype=np.float64)[sl_mask],
        }
