"""IAGA-2002 INTERMAGNET parser, HDZ->XYZ converter, QC gap handler, and despiker."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from geomag_core.conventions import (
    despike_station_network as despike_episode_stations,
)
from geomag_core.conventions import (
    haversine_km,
    hdz_to_xyz,
    interpolate_short_gaps,
    max_invalid_run,
)

from .config import BASELINE_NORM_THRESHOLD_NT, EPISODES, INTERMAGNET_FILL_VALUES

__all__ = [
    "IagaDayRecord",
    "align_day_record_to_episode",
    "despike_episode_stations",
    "haversine_km",
    "interpolate_short_gaps",
    "max_invalid_run",
    "normalize_lon_180",
    "parse_iaga2002_file",
    "process_intermagnet",
]

_DATA_TYPE_TO_QUALITY = {
    "variation": "v",
    "quasi-definitive": "q",
    "provisional": "p",
    "definitive": "d",
}


@dataclass
class IagaDayRecord:
    """Parsed daily IAGA-2002 magnetometer record."""

    iaga_code: str
    station_name: str
    source_of_data: str
    lat_deg: float
    lon_deg: float
    elev_m: float
    orig_format: str
    quality_code: str
    timestamps_utc: pd.DatetimeIndex
    x_nT: np.ndarray
    y_nT: np.ndarray
    z_nT: np.ndarray
    f_scalar_nT: np.ndarray
    is_valid_xyz: np.ndarray


def normalize_lon_180(lon_deg: float) -> float:
    """Normalize longitude from [0, 360) to [-180, 180]."""
    lon = float(lon_deg)
    while lon > 180.0:
        lon -= 360.0
    while lon < -180.0:
        lon += 360.0
    return lon


def align_day_record_to_episode(
    ep_timestamps: pd.DatetimeIndex,
    rec: IagaDayRecord,
    *,
    episode_id: str = "",
) -> np.ndarray:
    """Return integer index positions aligning rec.timestamps_utc onto ep_timestamps."""
    pos = ep_timestamps.get_indexer(rec.timestamps_utc)
    if np.any(pos < 0):
        raise ValueError(
            f"Timestamps for {rec.iaga_code} fall outside episode {episode_id!r}"
        )
    if np.unique(pos).size != pos.size:
        raise ValueError(
            f"Duplicate aligned episode timestamps for {rec.iaga_code} in {episode_id!r}"
        )
    return pos


def _detect_format(reported: str, value_cols: list[str], path: Path) -> str:
    rep_upper = reported.strip().upper()
    suffixes = [c[-1].upper() for c in value_cols[:3]]
    if rep_upper.startswith("HDZ") and suffixes == ["H", "D", "Z"]:
        return "HDZF"
    if rep_upper.startswith("XYZ") and suffixes == ["X", "Y", "Z"]:
        return "XYZF"
    raise ValueError(
        f"Format mismatch in {path.name}: Reported={reported!r}, columns={value_cols}"
    )


def parse_iaga2002_file(path: Path) -> IagaDayRecord:
    """Parse one IAGA-2002 1-minute .min file strictly by Reported and column headers."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_end = next(
        (idx for idx, line in enumerate(lines) if line.startswith("DATE")),
        None,
    )
    if header_end is None:
        raise ValueError(f"Missing IAGA-2002 DATE column header in {path}")

    meta: dict[str, str] = {}
    for line in lines[:header_end]:
        if line.lstrip().startswith("#"):
            continue
        if len(line) >= 24:
            key = line[:24].strip()
            val = line[24:69].strip()
            if key:
                meta[key] = val

    col_tokens = lines[header_end].split()
    if len(col_tokens) < 7 or col_tokens[:3] != ["DATE", "TIME", "DOY"]:
        raise ValueError(f"Unexpected IAGA-2002 header columns in {path}: {col_tokens}")
    value_cols = col_tokens[3:7]

    reported = meta.get("Reported", "")
    orig_format = _detect_format(reported, value_cols, path)

    data_lines = [line for line in lines[header_end + 1 :] if line.strip()]
    if not data_lines:
        raise ValueError(f"Empty data section in {path}")

    n_rows = len(data_lines)
    ts_strings = [f"{line[:10]}T{line[11:23]}Z" for line in data_lines]
    timestamps = pd.DatetimeIndex(pd.to_datetime(ts_strings, utc=True, format="ISO8601"))
    if not timestamps.is_monotonic_increasing or timestamps.has_duplicates:
        raise ValueError(f"Non-monotonic or duplicate timestamps in {path}")

    flat_vals = np.fromiter(
        (float(tok) for line in data_lines for tok in line.split()[3:7]),
        dtype=np.float64,
        count=n_rows * 4,
    )
    arr = flat_vals.reshape(n_rows, 4)
    for fill_val in INTERMAGNET_FILL_VALUES:
        arr[np.isclose(arr, fill_val, atol=1e-3)] = np.nan

    if orig_format == "HDZF":
        x_nT, y_nT, z_nT = hdz_to_xyz(arr[:, 0], arr[:, 1], arr[:, 2])
    else:
        x_nT = arr[:, 0]
        y_nT = arr[:, 1]
        z_nT = arr[:, 2]

    f_scalar_nT = arr[:, 3].copy()
    is_valid_xyz = np.isfinite(x_nT) & np.isfinite(y_nT) & np.isfinite(z_nT)
    # Enforce joint vector mask: if one vector component is NaN, mask all 3 vector components
    x_nT = np.where(is_valid_xyz, x_nT, np.nan)
    y_nT = np.where(is_valid_xyz, y_nT, np.nan)
    z_nT = np.where(is_valid_xyz, z_nT, np.nan)

    iaga_code = meta["IAGA Code"].strip().upper()
    lat_deg = float(meta["Geodetic Latitude"])
    lon_deg = normalize_lon_180(float(meta["Geodetic Longitude"]))
    elev_str = meta.get("Elevation", "").strip()
    elev_m = float(elev_str) if elev_str else 0.0

    dtype_raw = meta.get("Data Type", "").strip().lower()
    quality_code = _DATA_TYPE_TO_QUALITY.get(dtype_raw)
    if quality_code is None:
        quality_code = path.name[-8].lower()

    return IagaDayRecord(
        iaga_code=iaga_code,
        station_name=meta.get("Station Name", iaga_code),
        source_of_data=meta.get("Source of Data", ""),
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        elev_m=elev_m,
        orig_format=orig_format,
        quality_code=quality_code,
        timestamps_utc=timestamps,
        x_nT=x_nT,
        y_nT=y_nT,
        z_nT=z_nT,
        f_scalar_nT=f_scalar_nT,
        is_valid_xyz=is_valid_xyz,
    )


def _canonical_coords(records: list[tuple[str, float, float]]) -> tuple[float, float]:
    """Select mode (lat, lon) across files; break ties by latest date."""
    counts = Counter((lat, lon) for _, lat, lon in records)
    max_cnt = max(counts.values())
    top_coords = {coord for coord, cnt in counts.items() if cnt == max_cnt}
    for _, lat, lon in sorted(records, key=lambda r: r[0], reverse=True):
        if (lat, lon) in top_coords:
            return lat, lon
    return records[-1][1], records[-1][2]


def process_intermagnet(
    raw_root: Path,
    out_dir: Path,
    *,
    enable_despike: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, object]]]:
    """Normalize all INTERMAGNET files into episode Parquet files and station metadata."""
    intermagnet_root = raw_root / "GeoMag Storms" / "INTERMAGNET_1min"
    min_files = sorted(intermagnet_root.rglob("*.min"))
    if not min_files:
        raise FileNotFoundError(f"No .min files found under {intermagnet_root}")

    parsed_by_station_date: dict[tuple[str, str], IagaDayRecord] = {}
    station_coord_history: dict[str, list[tuple[str, float, float]]] = {}
    station_meta_raw: dict[str, dict[str, object]] = {}
    station_valid_chunks: dict[str, list[np.ndarray]] = {}
    file_stats: dict[str, dict[str, object]] = {}

    for path in min_files:
        rec = parse_iaga2002_file(path)
        file_date = rec.timestamps_utc[0].strftime("%Y-%m-%d")
        region = path.parent.parent.name
        parsed_by_station_date[(rec.iaga_code, file_date)] = rec
        station_coord_history.setdefault(rec.iaga_code, []).append(
            (file_date, rec.lat_deg, rec.lon_deg)
        )
        station_meta_raw.setdefault(
            rec.iaga_code,
            {
                "iaga_code": rec.iaga_code,
                "station_name": rec.station_name,
                "station_type": "magnetometer",
                "region": region,
                "elev_m": rec.elev_m,
                "orig_format": rec.orig_format,
                "source_of_data": rec.source_of_data,
            },
        )
        if np.any(rec.is_valid_xyz):
            station_valid_chunks.setdefault(rec.iaga_code, []).append(
                np.column_stack(
                    [
                        rec.x_nT[rec.is_valid_xyz],
                        rec.y_nT[rec.is_valid_xyz],
                        rec.z_nT[rec.is_valid_xyz],
                    ]
                )
            )
        rel_posix = path.relative_to(raw_root).as_posix()
        valid_pct = 100.0 * float(np.mean(rec.is_valid_xyz))
        file_stats[rel_posix] = {
            "row_count": int(rec.x_nT.size),
            "valid_pct": round(valid_pct, 4),
            "file_lat_deg": rec.lat_deg,
            "file_lon_deg": rec.lon_deg,
        }

    # Build station summary with canonical coordinates and baseline check
    station_coords_canon: dict[str, tuple[float, float]] = {}
    station_rows: list[dict[str, object]] = []
    for iaga in sorted(station_meta_raw.keys()):
        canon_lat, canon_lon = _canonical_coords(station_coord_history[iaga])
        station_coords_canon[iaga] = (canon_lat, canon_lon)
        chunks = station_valid_chunks.get(iaga, [])
        if chunks:
            all_xyz = np.vstack(chunks)
            mean_norm = float(np.linalg.norm(all_xyz.mean(axis=0)))
        else:
            mean_norm = 0.0
        has_baseline = mean_norm >= BASELINE_NORM_THRESHOLD_NT
        meta = station_meta_raw[iaga]
        station_rows.append(
            {
                **meta,
                "lat_deg": canon_lat,
                "lon_deg": canon_lon,
                "has_baseline": has_baseline,
                "mean_vec_norm_nT": round(mean_norm, 3),
                "days_present": len(station_coord_history[iaga]),
                "cutoff_rigidity_gv": np.nan,
                "barometric_coeff": np.nan,
            }
        )
    stations_df = pd.DataFrame.from_records(station_rows)
    for col in ("mean_vec_norm_nT", "cutoff_rigidity_gv", "barometric_coeff"):
        stations_df[col] = stations_df[col].astype("float64")

    geomag_dir = out_dir / "geomag"
    geomag_dir.mkdir(parents=True, exist_ok=True)
    availability_rows: list[dict[str, object]] = []

    for ep in EPISODES:
        ep_timestamps = pd.date_range(
            ep.start_utc, ep.end_utc, freq="1min", tz="UTC"
        )
        n_ep = len(ep_timestamps)
        raw_ep_xyz: dict[str, np.ndarray] = {}
        raw_ep_f: dict[str, np.ndarray] = {}
        raw_ep_valid: dict[str, np.ndarray] = {}
        day_records_ep: dict[str, list[tuple[int, str, IagaDayRecord]]] = {}

        for iaga in sorted(station_meta_raw.keys()):
            if not any((iaga, d) in parsed_by_station_date for d in ep.dates):
                continue
            xyz = np.full((n_ep, 3), np.nan, dtype=np.float64)
            f_arr = np.full(n_ep, np.nan, dtype=np.float64)
            valid = np.zeros(n_ep, dtype=bool)
            day_recs: list[tuple[int, str, IagaDayRecord]] = []

            for day_idx, date_str in enumerate(ep.dates):
                rec = parsed_by_station_date.get((iaga, date_str))
                if rec is None:
                    continue
                pos = align_day_record_to_episode(
                    ep_timestamps, rec, episode_id=ep.episode_id
                )
                xyz[pos, 0] = rec.x_nT
                xyz[pos, 1] = rec.y_nT
                xyz[pos, 2] = rec.z_nT
                f_arr[pos] = rec.f_scalar_nT
                valid[pos] = rec.is_valid_xyz
                day_recs.append((day_idx, date_str, rec))

            raw_ep_xyz[iaga] = xyz
            raw_ep_f[iaga] = f_arr
            raw_ep_valid[iaga] = valid
            day_records_ep[iaga] = day_recs

        # Despike across stitched episode before interpolating short <=3 min gaps
        if enable_despike:
            despiked_xyz, despiked_flags = despike_episode_stations(
                raw_ep_xyz, station_coords_canon
            )
        else:
            despiked_xyz = raw_ep_xyz
            despiked_flags = {k: np.zeros(n_ep, dtype=bool) for k in raw_ep_xyz}

        ep_frames: list[pd.DataFrame] = []
        for iaga in sorted(raw_ep_xyz.keys()):
            valid = raw_ep_valid[iaga]
            interp_xyz, is_interp = interpolate_short_gaps(despiked_xyz[iaga], valid, max_gap=3)
            was_despiked = despiked_flags[iaga]
            f_arr = raw_ep_f[iaga]

            ep_frames.append(
                pd.DataFrame(
                    {
                        "timestamp_utc": ep_timestamps,
                        "station_iaga": iaga,
                        "x_nT": interp_xyz[:, 0],
                        "y_nT": interp_xyz[:, 1],
                        "z_nT": interp_xyz[:, 2],
                        "f_scalar_nT": f_arr,
                        "is_valid_xyz": valid,
                        "is_interpolated": is_interp,
                        "was_despiked": was_despiked,
                    }
                )
            )

            for day_idx, date_str, rec in day_records_ep[iaga]:
                sl = slice(day_idx * 1440, (day_idx + 1) * 1440)
                day_valid = valid[sl]
                day_f = f_arr[sl]
                day_interp = is_interp[sl]
                day_despike = was_despiked[sl]
                valid_cnt = int(np.sum(day_valid))
                f_missing_cnt = int(np.sum(~np.isfinite(day_f)))
                f_only_missing_cnt = int(np.sum((~np.isfinite(day_f)) & day_valid))
                availability_rows.append(
                    {
                        "station_iaga": iaga,
                        "file_date": date_str,
                        "episode_id": ep.episode_id,
                        "quality_code": rec.quality_code,
                        "orig_format": rec.orig_format,
                        "valid_vector_count": valid_cnt,
                        "valid_vector_pct": round(100.0 * valid_cnt / 1440.0, 4),
                        "f_missing_count": f_missing_cnt,
                        "f_only_missing_count": f_only_missing_cnt,
                        "interpolated_count": int(np.sum(day_interp)),
                        "max_gap_min": max_invalid_run(day_valid),
                        "despiked_count": int(np.sum(day_despike)),
                        "file_lat_deg": rec.lat_deg,
                        "file_lon_deg": rec.lon_deg,
                    }
                )

        ep_df = pd.concat(ep_frames, ignore_index=True)
        ep_df["station_iaga"] = ep_df["station_iaga"].astype("string")
        ep_df.to_parquet(
            geomag_dir / f"{ep.episode_id}.parquet",
            index=False,
            compression="zstd",
        )

    availability_df = pd.DataFrame.from_records(availability_rows)
    return stations_df, availability_df, file_stats
