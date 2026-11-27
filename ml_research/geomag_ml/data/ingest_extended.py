"""Adapter for ingesting extended external data (quiet days, storms, new stations, OMNI, Kp, events)."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geomag_ml import REPO_ROOT
from geomag_ml.core_bridge import (
    import_ballistic_shift,
    import_conventions,
    import_etl_parsers,
    import_mlt,
)
from geomag_ml.data.build_dataset import build_qc_exclusion_masks, detect_baseline_step_warnings
from geomag_ml.data.schema import (
    EVENT_FLAG_ECLIPSE,
    EVENT_FLAG_QUIET,
    EVENT_FLAG_SPE,
    EVENT_FLAG_STORM,
    EVENT_NAME_TO_BIT,
    OMNI_COLS,
    OMNI_PARQUET_COL_MAP,
    SOLAR1_BSN_COLS,
    SOLAR1_L1_COLS,
    SOLAR1_PARQUET_COL_MAP,
    classify_station_belt,
    compute_mlt_matrix,
    compute_station_geometry,
    compute_time_features,
    infer_station_region,
    load_ml_dataset,
    timestamps_to_unix_s,
    validate_dataset_arrays,
)
from geomag_ml.utils.io import compute_sha256, read_yaml, save_npz_atomic, write_json


def _resolve_path(path_val: str | Path, base_dir: Path) -> Path:
    p = Path(path_val)
    return p if p.is_absolute() else (base_dir / p).resolve()


def _parse_utc_unix_s(val: str | pd.Timestamp | datetime) -> int:
    ts = pd.Timestamp(val)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp())


def parse_events_yaml(path: Path) -> dict[str, Any]:
    """Parse and validate `events.yaml` specification."""
    raw = read_yaml(Path(path))
    parsed: dict[str, Any] = {
        "quiet_kp_le2": list(raw.get("quiet_kp_le2", []) or []),
        "storm": list(raw.get("storm", []) or []),
        "eclipse": list(raw.get("eclipse", []) or []),
        "spe": list(raw.get("spe", []) or []),
    }
    return parsed


def interpolate_eclipse_track(
    eclipse_event: dict[str, Any],
    timestamps_unix_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve static center or moving eclipse track onto 1-min `timestamps_unix_s`."""
    ts = np.asarray(timestamps_unix_s, dtype=np.int64)
    n_t = ts.shape[0]
    active = np.zeros(n_t, dtype=bool)
    lat = np.full(n_t, np.nan, dtype=np.float64)
    lon = np.full(n_t, np.nan, dtype=np.float64)
    rad = np.full(n_t, np.nan, dtype=np.float64)

    track = eclipse_event.get("track")
    if track:
        pts = sorted(
            (
                _parse_utc_unix_s(pt["timestamp_utc"]),
                float(pt["lat_deg"]),
                float(pt["lon_deg"]),
                float(pt.get("radius_km", eclipse_event.get("radius_km", 750.0))),
            )
            for pt in track
        )
        t_wp = np.array([p[0] for p in pts], dtype=np.float64)
        lat_wp = np.array([p[1] for p in pts], dtype=np.float64)
        lon_wp = np.unwrap(np.deg2rad([p[2] for p in pts])) * (180.0 / math.pi)
        rad_wp = np.array([p[3] for p in pts], dtype=np.float64)

        t_start = _parse_utc_unix_s(eclipse_event["start_utc"]) if "start_utc" in eclipse_event else int(t_wp[0])
        t_end = _parse_utc_unix_s(eclipse_event["end_utc"]) if "end_utc" in eclipse_event else int(t_wp[-1])
        active = (ts >= t_start) & (ts <= t_end) & (ts >= int(t_wp[0])) & (ts <= int(t_wp[-1]))
        if np.any(active):
            t_q = ts[active].astype(np.float64)
            lat[active] = np.interp(t_q, t_wp, lat_wp)
            lon_interp = np.interp(t_q, t_wp, lon_wp)
            lon[active] = ((lon_interp + 180.0) % 360.0) - 180.0
            rad[active] = np.interp(t_q, t_wp, rad_wp)
        return active, lat, lon, rad

    t_start = _parse_utc_unix_s(eclipse_event["start_utc"])
    t_end = _parse_utc_unix_s(eclipse_event["end_utc"])
    active = (ts >= t_start) & (ts <= t_end)
    if np.any(active):
        lat[active] = float(eclipse_event["lat_deg"])
        lon[active] = float(eclipse_event["lon_deg"])
        rad[active] = float(eclipse_event["radius_km"])
    return active, lat, lon, rad


def compute_event_flags(
    timestamps_unix_s: np.ndarray,
    events: dict[str, Any] | None,
    *,
    base_flags: np.ndarray | None = None,
) -> np.ndarray:
    """Compute `uint8[T]` event bitmask from `events` intervals."""
    ts = np.asarray(timestamps_unix_s, dtype=np.int64)
    flags = (
        np.asarray(base_flags, dtype=np.uint8).copy()
        if base_flags is not None
        else np.zeros(ts.shape[0], dtype=np.uint8)
    )
    if not events:
        return flags

    mapping = (
        ("quiet_kp_le2", EVENT_FLAG_QUIET),
        ("storm", EVENT_FLAG_STORM),
        ("eclipse", EVENT_FLAG_ECLIPSE),
        ("spe", EVENT_FLAG_SPE),
    )
    for key, bit in mapping:
        for item in events.get(key, []) or []:
            if key == "eclipse" and "track" in item and ("start_utc" not in item or "end_utc" not in item):
                t_pts = [_parse_utc_unix_s(pt["timestamp_utc"]) for pt in item["track"]]
                t0, t1 = min(t_pts), max(t_pts)
            else:
                t0 = _parse_utc_unix_s(item["start_utc"])
                t1 = _parse_utc_unix_s(item["end_utc"])
            mask = (ts >= t0) & (ts <= t1)
            flags[mask] |= np.uint8(bit)
    return flags


def parse_kp_csv(path: Path) -> pd.DataFrame:
    """Parse 3-hour Kp CSV and expand stepwise onto 1-minute UTC timestamps."""
    df = pd.read_csv(Path(path), encoding="utf-8")
    col_lower = {c.lower().strip(): c for c in df.columns}
    if "kp" not in col_lower:
        raise ValueError(f"Kp CSV {path} must contain a 'kp' column")
    kp_vals = pd.to_numeric(df[col_lower["kp"]], errors="coerce").to_numpy(dtype=np.float64)

    if "timestamp_utc" in col_lower:
        ts_idx = pd.to_datetime(df[col_lower["timestamp_utc"]], utc=True)
    elif "datetime" in col_lower:
        ts_idx = pd.to_datetime(df[col_lower["datetime"]], utc=True)
    elif "date" in col_lower and "hour" in col_lower:
        base = pd.to_datetime(df[col_lower["date"]], utc=True)
        ts_idx = base + pd.to_timedelta(pd.to_numeric(df[col_lower["hour"]]), unit="h")
    else:
        raise ValueError(
            f"Kp CSV {path} must contain 'timestamp_utc', 'datetime', or ('date', 'hour') columns"
        )

    start_s = timestamps_to_unix_s(ts_idx)
    offsets_s = np.arange(0, 180 * 60, 60, dtype=np.int64)
    expanded_s = (start_s[:, None] + offsets_s[None, :]).reshape(-1)
    expanded_kp = np.repeat(kp_vals, offsets_s.shape[0])
    out = pd.DataFrame(
        {
            "timestamp_unix_s": expanded_s,
            "kp": expanded_kp,
        }
    )
    out = out.dropna(subset=["kp"]).drop_duplicates(subset=["timestamp_unix_s"], keep="last")
    return out.sort_values("timestamp_unix_s").reset_index(drop=True)


def load_external_omni(paths: list[Path]) -> pd.DataFrame:
    """Load external OMNI (.asc / .parquet / .csv) strictly keyed by `timestamp_utc`, ignoring hardcoded `episode_id`."""
    if not paths:
        return pd.DataFrame(columns=["timestamp_unix_s", *OMNI_COLS])

    etl = import_etl_parsers()
    frames: list[pd.DataFrame] = []
    for raw_path in paths:
        p = Path(raw_path)
        suf = p.suffix.lower()
        if suf in (".asc", ".txt", ".dat"):
            df = etl.parse_omni_hro_file(p)
        elif suf == ".parquet":
            df = pd.read_parquet(p)
        elif suf == ".csv":
            df = pd.read_csv(p, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported OMNI file extension: {p}")

        if "timestamp_utc" not in df.columns:
            raise ValueError(f"OMNI file {p} lacks 'timestamp_utc' column")
        # Explicitly ignore hardcoded `episode_id` from parse_omni_hro_file!
        ts_unix_s = timestamps_to_unix_s(df["timestamp_utc"])
        norm = pd.DataFrame({"timestamp_unix_s": ts_unix_s})
        for col in OMNI_COLS:
            pq_col = OMNI_PARQUET_COL_MAP[col]
            if col in df.columns:
                norm[col] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
            elif pq_col in df.columns:
                norm[col] = pd.to_numeric(df[pq_col], errors="coerce").to_numpy(dtype=np.float64)
            else:
                norm[col] = np.nan
        frames.append(norm)

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values("timestamp_unix_s")
    # Merge duplicates by taking last non-null per timestamp_unix_s
    merged = merged.groupby("timestamp_unix_s", as_index=False).last()
    return merged.sort_values("timestamp_unix_s").reset_index(drop=True)


def group_timestamps_into_episodes(
    sorted_unix_s: np.ndarray,
    *,
    gap_threshold_min: int = 60,
    known_episodes: list[tuple[int, int, str]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Group sorted Unix second timestamps into contiguous 60s episodes (split if gap > threshold)."""
    ts = np.unique(np.asarray(sorted_unix_s, dtype=np.int64))
    if ts.size == 0:
        raise ValueError("Cannot group empty timestamp array into episodes")

    gap_sec = int(gap_threshold_min) * 60
    splits = np.flatnonzero(np.diff(ts) > gap_sec) + 1
    groups = np.split(ts, splits)

    full_ts_list: list[np.ndarray] = []
    ep_idx_list: list[np.ndarray] = []
    bounds: list[list[int]] = []
    episode_ids: list[str] = []
    used_names: set[str] = set()
    offset = 0

    for ep_i, grp in enumerate(groups):
        t0 = int(grp[0])
        t1 = int(grp[-1])
        # Snap to 60s grid from t0 to t1 inclusive
        n_ep = ((t1 - t0) // 60) + 1
        grid = t0 + np.arange(n_ep, dtype=np.int64) * 60

        ep_name: str | None = None
        if known_episodes:
            for k_start, k_end, k_name in known_episodes:
                if t0 == k_start and t1 == k_end:
                    ep_name = k_name
                    break
        if ep_name is None:
            d0 = datetime.fromtimestamp(t0, tz=UTC).strftime("%Y%m%d")
            d1 = datetime.fromtimestamp(t1, tz=UTC).strftime("%Y%m%d")
            base_name = f"ext_{d0}" if d0 == d1 else f"ext_{d0}_{d1}"
            ep_name = base_name
            suffix = 2
            while ep_name in used_names:
                ep_name = f"{base_name}_{suffix}"
                suffix += 1
        used_names.add(ep_name)
        episode_ids.append(ep_name)

        full_ts_list.append(grid)
        ep_idx_list.append(np.full(n_ep, ep_i, dtype=np.int16))
        bounds.append([offset, offset + n_ep])
        offset += n_ep

    return (
        np.concatenate(full_ts_list),
        np.concatenate(ep_idx_list),
        np.array(bounds, dtype=np.int64),
        episode_ids,
    )


def _expand_file_patterns(
    entries: list[str | Path],
    base_dir: Path,
    *,
    dir_patterns: tuple[str, ...] = ("*.min",),
) -> list[Path]:
    paths: list[Path] = []
    for entry in entries:
        s = str(entry)
        p = Path(s)
        if not p.is_absolute():
            p = base_dir / p
        if any(ch in s for ch in "*?["):
            parts = p.parts
            wild_idx = next(
                (idx for idx, part in enumerate(parts) if any(ch in part for ch in "*?[")),
                len(parts),
            )
            if wild_idx == 0:
                anchor = base_dir
                rel_pat = s
            else:
                anchor = Path(*parts[:wild_idx])
                rel_pat = str(Path(*parts[wild_idx:]))
            if anchor.exists():
                paths.extend(sorted(fp for fp in anchor.glob(rel_pat) if fp.is_file()))
        elif p.is_dir():
            for pat in dir_patterns:
                paths.extend(sorted(fp for fp in p.rglob(pat) if fp.is_file()))
        elif p.is_file():
            paths.append(p)
    return sorted(set(paths))


def _extract_solar1_col(df: pd.DataFrame, canonical_col: str) -> np.ndarray:
    pq_col = SOLAR1_PARQUET_COL_MAP[canonical_col]
    if canonical_col in df.columns:
        return pd.to_numeric(df[canonical_col], errors="coerce").to_numpy(dtype=np.float64)
    if pq_col in df.columns:
        return pd.to_numeric(df[pq_col], errors="coerce").to_numpy(dtype=np.float64)
    return np.full(len(df), np.nan, dtype=np.float64)


def build_extended_dataset(
    config: dict[str, Any],
    *,
    repo_root: Path | None = None,
) -> tuple[Path, Path]:
    """Merge base dataset with external `.min`, OMNI, Kp CSV, SOLAR-1/OULU, and `events.yaml`."""
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    ext_dir = _resolve_path(config.get("external_dir", "ml_research/data_external"), root)
    out_dir = _resolve_path(config.get("out_dir", "ml_research/data_cache"), root)
    tag = str(config["tag"])
    include_short_interp = bool(config.get("include_short_interp", True))
    gap_threshold_min = int(config.get("episode_gap_minutes", 60))
    variation_step_policy = str(config.get("variation_step_policy", "flag_qc_exclude"))
    step_warn_nT = float(config.get("variation_step_warn_nT", 30.0))
    ref_distance_km = float(config.get("ballistic_ref_distance_km", 1_400_000.0))
    smooth_window_min = int(config.get("ballistic_smooth_window_min", 31))
    region_overrides: dict[str, str] = {
        str(k).strip().upper(): str(v)
        for k, v in (config.get("station_region_overrides") or {}).items()
    }

    etl = import_etl_parsers()
    conv = import_conventions()
    mlt_mod = import_mlt()

    base_arrays: dict[str, np.ndarray] | None = None
    base_meta: dict[str, Any] = {}
    base_npz_cfg = config.get("base_npz")
    if base_npz_cfg is not None and str(base_npz_cfg).strip() != "":
        base_path = _resolve_path(base_npz_cfg, root)
        if not base_path.is_file():
            raise FileNotFoundError(f"Configured base_npz not found: {base_path}")
        base_arrays, base_meta = load_ml_dataset(base_path, validate=True)

    # 1. Discover and parse external .min files
    min_entries = config.get("intermagnet_dirs") or []
    min_files = _expand_file_patterns(min_entries, root, dir_patterns=("*.min",))
    if not min_entries and not min_files and ext_dir.is_dir():
        min_files = sorted(ext_dir.rglob("*.min"))

    parsed_recs: list[Any] = [etl.parse_iaga2002_file(p) for p in min_files]

    # 2. Build unified sorted station registry
    station_info: dict[str, dict[str, Any]] = {}
    if base_arrays is not None:
        base_codes = list(base_meta["station_codes"])
        base_regions = list(base_meta["station_regions"])
        base_coords = base_arrays["station_coords"]
        base_has_bl = base_arrays["station_has_baseline"]
        for i, code in enumerate(base_codes):
            station_info[code] = {
                "lat_deg": float(base_coords[i, 0]),
                "lon_deg": float(base_coords[i, 1]),
                "region": base_regions[i],
                "has_baseline": bool(base_has_bl[i]),
            }

    ext_station_chunks: dict[str, list[np.ndarray]] = {}
    for rec in parsed_recs:
        code = rec.iaga_code
        if np.any(rec.is_valid_xyz):
            ext_station_chunks.setdefault(code, []).append(
                np.column_stack(
                    [
                        rec.x_nT[rec.is_valid_xyz],
                        rec.y_nT[rec.is_valid_xyz],
                        rec.z_nT[rec.is_valid_xyz],
                    ]
                )
            )
        if code not in station_info:
            reg = infer_station_region(rec.lon_deg, region_overrides.get(code))
            station_info[code] = {
                "lat_deg": float(rec.lat_deg),
                "lon_deg": float(rec.lon_deg),
                "region": reg,
                "has_baseline": None,
            }

    for code, info in station_info.items():
        if info["has_baseline"] is None:
            chunks = ext_station_chunks.get(code, [])
            if chunks:
                all_xyz = np.vstack(chunks)
                mean_norm = float(np.linalg.norm(all_xyz.mean(axis=0)))
                info["has_baseline"] = mean_norm >= float(etl.BASELINE_NORM_THRESHOLD_NT)
            else:
                info["has_baseline"] = False

    station_codes = sorted(station_info.keys())
    if not station_codes:
        raise ValueError("No stations found in either base_npz or external .min files")
    n_st = len(station_codes)
    st_to_idx = {c: i for i, c in enumerate(station_codes)}

    station_regions = [
        infer_station_region(station_info[c]["lon_deg"], region_overrides.get(c) or station_info[c]["region"])
        for c in station_codes
    ]
    lats = np.array([station_info[c]["lat_deg"] for c in station_codes], dtype=np.float64)
    lons = np.array([station_info[c]["lon_deg"] for c in station_codes], dtype=np.float64)
    station_has_baseline = np.array(
        [bool(station_info[c]["has_baseline"]) for c in station_codes],
        dtype=np.bool_,
    )
    station_coords, station_dist_km = compute_station_geometry(lats, lons, mlt_mod, conv)
    station_belt_id = np.array(
        [
            classify_station_belt(reg, float(station_coords[i, 2]))
            for i, reg in enumerate(station_regions)
        ],
        dtype=np.int8,
    )

    # 3. Build unified timeline and episodes
    all_ts_parts: list[np.ndarray] = []
    known_eps: list[tuple[int, int, str]] = []
    if base_arrays is not None:
        base_ts = base_arrays["timestamps_unix_s"]
        all_ts_parts.append(base_ts)
        base_ep_ids = list(base_meta.get("episode_ids", []))
        base_eb = base_arrays["episode_bounds"]
        for k, ep_id in enumerate(base_ep_ids):
            s_b, e_b = int(base_eb[k, 0]), int(base_eb[k, 1])
            known_eps.append((int(base_ts[s_b]), int(base_ts[e_b - 1]), str(ep_id)))

    for rec in parsed_recs:
        all_ts_parts.append(timestamps_to_unix_s(rec.timestamps_utc))

    merged_unix = np.unique(np.concatenate(all_ts_parts))
    timestamps_unix_s, episode_idx, episode_bounds, episode_ids = group_timestamps_into_episodes(
        merged_unix,
        gap_threshold_min=gap_threshold_min,
        known_episodes=known_eps,
    )
    n_t = timestamps_unix_s.shape[0]

    # 4. Populate magnetometer tensor X_raw and masks
    x_raw = np.full((n_t, n_st, 3), np.nan, dtype=np.float32)
    m_raw_valid = np.zeros((n_t, n_st), dtype=np.bool_)
    m_interpolated = np.zeros((n_t, n_st), dtype=np.bool_)
    m_despiked = np.zeros((n_t, n_st), dtype=np.bool_)
    m_known_spikes = np.zeros((n_t, n_st), dtype=np.bool_)
    base_qc_disabled = np.zeros((n_t, n_st), dtype=bool)

    if base_arrays is not None:
        base_ts = base_arrays["timestamps_unix_s"]
        pos_t = np.searchsorted(timestamps_unix_s, base_ts)
        base_codes = list(base_meta["station_codes"])
        for j, code in enumerate(base_codes):
            i = st_to_idx[code]
            x_raw[pos_t, i, :] = base_arrays["X_raw"][:, j, :]
            m_raw_valid[pos_t, i] = base_arrays["M_raw_valid"][:, j]
            m_interpolated[pos_t, i] = base_arrays["M_interpolated"][:, j]
            m_despiked[pos_t, i] = base_arrays["M_despiked"][:, j]
            m_known_spikes[pos_t, i] = base_arrays["M_known_spikes"][:, j]
            base_qc_disabled[pos_t, i] = base_arrays["M_valid"][:, j] & (~base_arrays["M_qc_clean"][:, j])

    # Group external records per timestamp across episodes (handles mid-day >60 min gaps)
    chunks_by_ep_st: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for rec in parsed_recs:
        rec_s = timestamps_to_unix_s(rec.timestamps_utc)
        if rec_s.size == 0:
            continue
        pos_t = np.searchsorted(timestamps_unix_s, rec_s)
        if np.any(pos_t >= n_t) or not np.array_equal(timestamps_unix_s[pos_t], rec_s):
            raise ValueError(f"Timestamps for {rec.iaga_code} failed to align on merged timeline")
        ep_of_pts = episode_idx[pos_t]
        st_i = st_to_idx[rec.iaga_code]
        rec_xyz = np.column_stack([rec.x_nT, rec.y_nT, rec.z_nT])
        for e_val in np.unique(ep_of_pts):
            sel = ep_of_pts == e_val
            chunks_by_ep_st.setdefault((int(e_val), st_i), []).append(
                (rec_s[sel], rec_xyz[sel], rec.is_valid_xyz[sel])
            )

    for (e_i, st_i), chunks in chunks_by_ep_st.items():
        s_ep, e_ep = int(episode_bounds[e_i, 0]), int(episode_bounds[e_i, 1])
        ep_grid_s = timestamps_unix_s[s_ep:e_ep]
        # Preserve any existing base_npz data inside this episode before overlaying external chunks
        xyz_ep = x_raw[s_ep:e_ep, st_i, :].astype(np.float64)
        valid_ep = m_raw_valid[s_ep:e_ep, st_i].copy()
        xyz_ep[~valid_ep] = np.nan
        for chunk_s, chunk_xyz, chunk_valid in chunks:
            pos = np.searchsorted(ep_grid_s, chunk_s)
            if np.any(pos >= ep_grid_s.shape[0]) or not np.array_equal(ep_grid_s[pos], chunk_s):
                raise ValueError(
                    f"Aligned external chunk timestamps outside episode {episode_ids[e_i]!r}"
                )
            take = chunk_valid | (~valid_ep[pos])
            pos_take = pos[take]
            xyz_ep[pos_take] = chunk_xyz[take]
            valid_ep[pos_take] = chunk_valid[take]
        if include_short_interp:
            xyz_filled, interp_ep = conv.interpolate_short_gaps(xyz_ep, valid_ep, max_gap=3)
        else:
            xyz_filled = xyz_ep
            xyz_filled[~valid_ep] = np.nan
            interp_ep = np.zeros(e_ep - s_ep, dtype=bool)

        m_raw_valid[s_ep:e_ep, st_i] = valid_ep
        m_interpolated[s_ep:e_ep, st_i] = interp_ep
        x_raw[s_ep:e_ep, st_i, :] = xyz_filled.astype(np.float32)

    m_valid = (m_raw_valid | m_interpolated) if include_short_interp else m_raw_valid.copy()
    x_raw[~m_valid] = np.nan

    multi_mask, var_step_mask = build_qc_exclusion_masks(
        timestamps_unix_s,
        episode_idx,
        station_codes,
        episode_ids,
        config.get("qc_exclusions"),
    )
    m_qc_clean = m_valid & (~base_qc_disabled) & (~multi_mask)
    if variation_step_policy == "flag_qc_exclude":
        m_qc_clean &= ~var_step_mask

    # 5. Geometry, MLT, and time features
    mlt_hours = compute_mlt_matrix(lats, lons, timestamps_unix_s, mlt_mod)
    time_features = compute_time_features(timestamps_unix_s)

    # 6. Merge OMNI, Kp, SOLAR-1, OULU strictly by timestamp_unix_s
    sw_omni = np.full((n_t, len(OMNI_COLS)), np.nan, dtype=np.float32)
    sw_solar1_bsn = np.full((n_t, len(SOLAR1_BSN_COLS)), np.nan, dtype=np.float32)
    sw_solar1_l1 = np.full((n_t, len(SOLAR1_L1_COLS)), np.nan, dtype=np.float32)
    oulu_delta_n_pct = np.full(n_t, np.nan, dtype=np.float32)
    event_flags = np.zeros(n_t, dtype=np.uint8)

    if base_arrays is not None:
        pos_b = np.searchsorted(timestamps_unix_s, base_arrays["timestamps_unix_s"])
        sw_omni[pos_b] = base_arrays["sw_omni"]
        sw_solar1_bsn[pos_b] = base_arrays["sw_solar1_bsn"]
        sw_solar1_l1[pos_b] = base_arrays["sw_solar1_l1"]
        oulu_delta_n_pct[pos_b] = base_arrays["oulu_delta_n_pct"]
        event_flags[pos_b] = base_arrays["event_flags"]

    omni_paths = _expand_file_patterns(
        config.get("omni_files") or [],
        root,
        dir_patterns=("*.asc", "*.txt", "*.dat", "*.parquet", "*.csv"),
    )
    if omni_paths:
        ext_omni = load_external_omni(omni_paths)
        o_ts = ext_omni["timestamp_unix_s"].to_numpy(dtype=np.int64)
        pos_o = np.searchsorted(timestamps_unix_s, o_ts)
        in_rng = pos_o < n_t
        pos_ok = pos_o[in_rng]
        exact = timestamps_unix_s[pos_ok] == o_ts[in_rng]
        if np.any(exact):
            target_idx = pos_ok[exact]
            ext_mat = ext_omni[list(OMNI_COLS)].to_numpy(dtype=np.float32)[in_rng][exact]
            valid_ext = np.isfinite(ext_mat)
            cur = sw_omni[target_idx]
            sw_omni[target_idx] = np.where(valid_ext, ext_mat, cur)

    kp_paths = _expand_file_patterns(
        config.get("kp_files") or [],
        root,
        dir_patterns=("*.csv",),
    )
    if kp_paths:
        kp_idx = OMNI_COLS.index("kp")
        for kp_path in kp_paths:
            kp_df = parse_kp_csv(kp_path)
            k_ts = kp_df["timestamp_unix_s"].to_numpy(dtype=np.int64)
            pos_k = np.searchsorted(timestamps_unix_s, k_ts)
            in_rng = pos_k < n_t
            pos_ok = pos_k[in_rng]
            exact = timestamps_unix_s[pos_ok] == k_ts[in_rng]
            if np.any(exact):
                sw_omni[pos_ok[exact], kp_idx] = kp_df["kp"].to_numpy(dtype=np.float32)[in_rng][exact]

    s1_paths = _expand_file_patterns(
        config.get("solar1_files") or [],
        root,
        dir_patterns=("*.parquet", "*.csv"),
    )
    if s1_paths:
        ballistic_api = import_ballistic_shift()
        s1_frames = []
        for s1_p in s1_paths:
            df_s1 = (
                pd.read_parquet(s1_p)
                if s1_p.suffix.lower() == ".parquet"
                else pd.read_csv(s1_p, encoding="utf-8")
            )
            df_s1["timestamp_utc"] = pd.to_datetime(df_s1["timestamp_utc"], utc=True)
            norm_s1 = pd.DataFrame({"timestamp_utc": df_s1["timestamp_utc"]})
            for col in SOLAR1_L1_COLS:
                norm_s1[col] = _extract_solar1_col(df_s1, col)
            s1_frames.append(norm_s1)
        s1_all = (
            pd.concat(s1_frames, ignore_index=True)
            .sort_values("timestamp_utc")
            .groupby("timestamp_utc", as_index=False)
            .last()
            .sort_values("timestamp_utc")
            .reset_index(drop=True)
        )
        s1_unix = timestamps_to_unix_s(s1_all["timestamp_utc"])
        pos_s1 = np.searchsorted(timestamps_unix_s, s1_unix)
        in_rng = pos_s1 < n_t
        pos_ok = pos_s1[in_rng]
        exact = timestamps_unix_s[pos_ok] == s1_unix[in_rng]
        if np.any(exact):
            target_idx = pos_ok[exact]
            l1_vals = s1_all[list(SOLAR1_L1_COLS)].to_numpy(dtype=np.float32)[in_rng][exact]
            cur_l1 = sw_solar1_l1[target_idx]
            sw_solar1_l1[target_idx] = np.where(np.isfinite(l1_vals), l1_vals, cur_l1)
        for e_i in range(episode_bounds.shape[0]):
            s_ep, e_ep = int(episode_bounds[e_i, 0]), int(episode_bounds[e_i, 1])
            ep_grid_s = timestamps_unix_s[s_ep:e_ep]
            in_ep = (s1_unix >= int(ep_grid_s[0])) & (s1_unix <= int(ep_grid_s[-1]))
            if not np.any(in_ep):
                continue
            sub = s1_all.loc[in_ep].sort_values("timestamp_utc")
            ep_ts = pd.DatetimeIndex(sub["timestamp_utc"]).tz_convert("UTC")
            t_arr_sec, _ = ballistic_api.compute_ballistic_shift(
                ep_ts,
                sub["vx_gse"].to_numpy(dtype=np.float64),
                ref_distance_km=ref_distance_km,
                smooth_window_min=smooth_window_min,
            )
            t_target_sec = ep_grid_s.astype(np.float64)
            res = {
                c: ballistic_api.resample_monotonic_series(
                    t_arr_sec,
                    sub[c].to_numpy(dtype=np.float64),
                    t_target_sec,
                )
                for c in SOLAR1_L1_COLS
            }
            ey_mvm = res["vx_gse"] * res["bz_gsm"] * 1e-3
            new_bsn = np.column_stack(
                [res["bz_gsm"], res["b_mag_gsm"], res["vx_gse"], res["proton_n"], res["pdyn_nPa"], ey_mvm]
            ).astype(np.float32)
            cur_bsn = sw_solar1_bsn[s_ep:e_ep]
            sw_solar1_bsn[s_ep:e_ep] = np.where(np.isfinite(new_bsn), new_bsn, cur_bsn)

    oulu_paths = _expand_file_patterns(
        config.get("oulu_files") or [],
        root,
        dir_patterns=("*.parquet", "*.csv"),
    )
    if oulu_paths:
        for o_p in oulu_paths:
            o_df = pd.read_parquet(o_p) if o_p.suffix.lower() == ".parquet" else pd.read_csv(o_p, encoding="utf-8")
            o_ts = timestamps_to_unix_s(pd.to_datetime(o_df["timestamp_utc"], utc=True))
            pos_ou = np.searchsorted(timestamps_unix_s, o_ts)
            in_rng = pos_ou < n_t
            pos_ok = pos_ou[in_rng]
            exact = timestamps_unix_s[pos_ok] == o_ts[in_rng]
            if np.any(exact):
                target_idx = pos_ok[exact]
                ext_oulu = pd.to_numeric(
                    o_df["delta_n_pct"], errors="coerce"
                ).to_numpy(dtype=np.float32)[in_rng][exact]
                cur_oulu = oulu_delta_n_pct[target_idx]
                oulu_delta_n_pct[target_idx] = np.where(np.isfinite(ext_oulu), ext_oulu, cur_oulu)

    # 7. Event flags
    events_file_cfg = config.get("events_file")
    events_dict: dict[str, Any] | None = None
    if events_file_cfg is not None and str(events_file_cfg).strip() != "":
        ev_path = _resolve_path(events_file_cfg, root)
        if not ev_path.is_file():
            raise FileNotFoundError(f"Configured events_file not found: {ev_path}")
        events_dict = parse_events_yaml(ev_path)
        event_flags = compute_event_flags(timestamps_unix_s, events_dict, base_flags=event_flags)

    if config.get("auto_quiet_from_kp", True):
        kp_vals = sw_omni[:, OMNI_COLS.index("kp")]
        kp_ok = np.isfinite(kp_vals) & (kp_vals <= 2.0) & ((event_flags & EVENT_FLAG_STORM) == 0)
        event_flags[kp_ok] |= np.uint8(EVENT_FLAG_QUIET)

    if "default_event_flag" in config and config["default_event_flag"] is not None:
        bit = int(EVENT_NAME_TO_BIT[str(config["default_event_flag"])])
        unset = event_flags == 0
        event_flags[unset] = np.uint8(bit)

    arrays: dict[str, np.ndarray] = {
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
        "station_has_baseline": station_has_baseline,
        "station_belt_id": station_belt_id,
        "mlt_hours": mlt_hours,
        "time_features": time_features,
        "sw_omni": sw_omni,
        "sw_solar1_bsn": sw_solar1_bsn,
        "sw_solar1_l1": sw_solar1_l1,
        "oulu_delta_n_pct": oulu_delta_n_pct,
        "event_flags": event_flags,
    }

    step_warnings = detect_baseline_step_warnings(
        x_raw,
        m_valid,
        episode_idx,
        episode_bounds,
        station_codes,
        episode_ids,
        threshold_nT=step_warn_nT,
    )
    calendar_span_days = float((int(timestamps_unix_s[-1]) - int(timestamps_unix_s[0])) / 86400.0)
    meta: dict[str, Any] = {
        "tag": tag,
        "dataset_kind": "extended",
        "created_utc": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "station_codes": station_codes,
        "station_regions": station_regions,
        "episode_ids": episode_ids,
        "omni_cols": list(OMNI_COLS),
        "solar1_cols": list(SOLAR1_BSN_COLS),
        "solar1_l1_cols": list(SOLAR1_L1_COLS),
        "include_short_interp": include_short_interp,
        "variation_step_policy": variation_step_policy,
        "calendar_span_days": round(calendar_span_days, 4),
        "recommended_use_doy": bool(calendar_span_days >= 60.0),
        "qc_exclusions": config.get("qc_exclusions", {}),
        "events": events_dict or {},
        "baseline_step_warnings": step_warnings,
    }

    validate_dataset_arrays(arrays, meta)
    npz_path = out_dir / f"geomag_ml_{tag}.npz"
    save_npz_atomic(npz_path, arrays, compressed=True)
    meta["dataset_sha256"] = compute_sha256(npz_path)
    meta_path = npz_path.with_suffix(".meta.json")
    write_json(meta_path, meta, nan_to_null=True)
    return npz_path, meta_path
