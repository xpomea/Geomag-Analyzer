"""Dataset builders for clean and undespiked October 2026 `.npz` bundles (T0.2 & T0.3)."""

from __future__ import annotations

import sqlite3
import warnings
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
from geomag_ml.data.schema import (
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


def _parse_utc_unix_s(value: str | pd.Timestamp | datetime) -> int:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp())


def detect_baseline_step_warnings(
    X_raw: np.ndarray,
    M_valid: np.ndarray,
    episode_idx: np.ndarray,
    episode_bounds: np.ndarray,
    station_codes: list[str],
    episode_ids: list[str],
    *,
    threshold_nT: float = 30.0,
    min_ep_points: int = 60,
) -> list[dict[str, Any]]:
    """Detect inter-episode baseline jumps > threshold_nT per station and component."""
    _ = episode_idx
    n_ep = episode_bounds.shape[0]
    n_st = len(station_codes)
    warnings_list: list[dict[str, Any]] = []
    comp_names = ("X", "Y", "Z")

    for i in range(n_st):
        ep_medians: list[tuple[int, np.ndarray]] = []
        for e in range(n_ep):
            s, end = int(episode_bounds[e, 0]), int(episode_bounds[e, 1])
            m_e = M_valid[s:end, i]
            if int(np.sum(m_e)) >= min_ep_points:
                med = np.median(X_raw[s:end, i, :][m_e], axis=0)
                ep_medians.append((e, med.astype(np.float64)))
        if len(ep_medians) < 2:
            continue

        best_jump = 0.0
        best_info: dict[str, Any] | None = None
        for idx_a in range(len(ep_medians)):
            e_a, med_a = ep_medians[idx_a]
            for idx_b in range(idx_a + 1, len(ep_medians)):
                e_b, med_b = ep_medians[idx_b]
                diffs = np.abs(med_b - med_a)
                comp_idx = int(np.argmax(diffs))
                jump = float(diffs[comp_idx])
                if jump > best_jump:
                    best_jump = jump
                    best_info = {
                        "station": station_codes[i],
                        "component": comp_names[comp_idx],
                        "max_step_nT": round(jump, 3),
                        "episode_a": episode_ids[e_a],
                        "episode_b": episode_ids[e_b],
                        "median_a_nT": round(float(med_a[comp_idx]), 3),
                        "median_b_nT": round(float(med_b[comp_idx]), 3),
                    }
        if best_info is not None and best_jump > float(threshold_nT):
            warnings_list.append(best_info)

    return warnings_list


def build_qc_exclusion_masks(
    timestamps_unix_s: np.ndarray,
    episode_idx: np.ndarray,
    station_codes: list[str],
    episode_ids: list[str],
    qc_exclusions: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build `(multi_minute_mask, variation_step_mask)` boolean arrays of shape `(T, N)`."""
    n_t = timestamps_unix_s.shape[0]
    n_st = len(station_codes)
    multi_mask = np.zeros((n_t, n_st), dtype=bool)
    var_step_mask = np.zeros((n_t, n_st), dtype=bool)

    if not qc_exclusions:
        return multi_mask, var_step_mask

    st_to_idx = {code: idx for idx, code in enumerate(station_codes)}
    ep_to_idx = {ep_id: idx for idx, ep_id in enumerate(episode_ids)}

    for entry in qc_exclusions.get("multi_minute_anomalies", []) or []:
        st_code = str(entry["station"]).strip().upper()
        if st_code not in st_to_idx:
            continue
        st_idx = st_to_idx[st_code]
        start_s = _parse_utc_unix_s(entry["start_utc"])
        end_s = _parse_utc_unix_s(entry["end_utc"])
        mask_t = (timestamps_unix_s >= start_s) & (timestamps_unix_s <= end_s)
        ep_id = entry.get("episode_id")
        if ep_id is not None:
            if ep_id not in ep_to_idx:
                continue
            mask_t &= episode_idx == ep_to_idx[ep_id]
        multi_mask[:, st_idx] |= mask_t

    for entry in qc_exclusions.get("variation_baseline_steps", []) or []:
        st_code = str(entry["station"]).strip().upper()
        if st_code not in st_to_idx:
            continue
        st_idx = st_to_idx[st_code]
        ep_id = entry.get("episode_id")
        if ep_id is not None:
            if ep_id not in ep_to_idx:
                continue
            var_step_mask[:, st_idx] |= episode_idx == ep_to_idx[ep_id]
        elif "start_utc" in entry and "end_utc" in entry:
            start_s = _parse_utc_unix_s(entry["start_utc"])
            end_s = _parse_utc_unix_s(entry["end_utc"])
            var_step_mask[:, st_idx] |= (timestamps_unix_s >= start_s) & (
                timestamps_unix_s <= end_s
            )

    return multi_mask, var_step_mask


def _align_series_by_unix_s(
    target_unix_s: np.ndarray,
    src_unix_s: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Align 1D or 2D array by exact matching Unix second timestamps."""
    out_shape = (target_unix_s.shape[0],) + values.shape[1:]
    out = np.full(out_shape, np.nan, dtype=np.float32)
    if src_unix_s.size == 0 or target_unix_s.size == 0:
        return out

    order = np.argsort(src_unix_s)
    src_sorted = src_unix_s[order]
    val_sorted = values[order]

    pos = np.searchsorted(target_unix_s, src_sorted)
    in_range = pos < target_unix_s.shape[0]
    pos_ok = pos[in_range]
    exact = target_unix_s[pos_ok] == src_sorted[in_range]
    if np.any(exact):
        out[pos_ok[exact]] = val_sorted[in_range][exact].astype(np.float32)
    return out


def _compute_solar1_arrays(
    solar1_df: pd.DataFrame,
    timestamps_unix_s: np.ndarray,
    episode_bounds: np.ndarray,
    *,
    ref_distance_km: float,
    smooth_window_min: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build `sw_solar1_l1: float32[T, 5]` and ballistic-shifted `sw_solar1_bsn: float32[T, 6]`."""
    n_t = timestamps_unix_s.shape[0]
    ballistic_api = import_ballistic_shift()

    s1_df = solar1_df.copy()
    s1_df["timestamp_utc"] = pd.DatetimeIndex(s1_df["timestamp_utc"]).tz_convert("UTC")
    s1_unix_s = timestamps_to_unix_s(s1_df["timestamp_utc"])

    l1_mat = np.column_stack(
        [s1_df[SOLAR1_PARQUET_COL_MAP[col]].to_numpy(dtype=np.float64) for col in SOLAR1_L1_COLS]
    )
    sw_solar1_l1 = _align_series_by_unix_s(timestamps_unix_s, s1_unix_s, l1_mat)

    sw_solar1_bsn = np.full((n_t, len(SOLAR1_BSN_COLS)), np.nan, dtype=np.float32)
    for ep_idx in range(episode_bounds.shape[0]):
        s, e = int(episode_bounds[ep_idx, 0]), int(episode_bounds[ep_idx, 1])
        ep_grid_s = timestamps_unix_s[s:e]
        in_ep = (s1_unix_s >= int(ep_grid_s[0])) & (s1_unix_s <= int(ep_grid_s[-1]))
        if not np.any(in_ep):
            continue
        sub = s1_df.loc[in_ep].sort_values("timestamp_utc")
        ep_ts = pd.DatetimeIndex(sub["timestamp_utc"]).tz_convert("UTC")
        vx_arr = sub["vx_gse_kms"].to_numpy(dtype=np.float64)
        t_arr_sec, _ = ballistic_api.compute_ballistic_shift(
            ep_ts,
            vx_arr,
            ref_distance_km=float(ref_distance_km),
            smooth_window_min=int(smooth_window_min),
        )
        t_target_sec = ep_grid_s.astype(np.float64)

        resampled: dict[str, np.ndarray] = {}
        for col in SOLAR1_L1_COLS:
            pq_col = SOLAR1_PARQUET_COL_MAP[col]
            resampled[col] = ballistic_api.resample_monotonic_series(
                t_arr_sec,
                sub[pq_col].to_numpy(dtype=np.float64),
                t_target_sec,
            )

        # Exact sign convention: vx_gse < 0 and bz_gsm < 0 => positive Ey [mV/m]
        ey_mvm = resampled["vx_gse"] * resampled["bz_gsm"] * 1e-3
        bsn_block = np.column_stack(
            [
                resampled["bz_gsm"],
                resampled["b_mag_gsm"],
                resampled["vx_gse"],
                resampled["proton_n"],
                resampled["pdyn_nPa"],
                ey_mvm,
            ]
        )
        sw_solar1_bsn[s:e] = bsn_block.astype(np.float32)

    return sw_solar1_l1, sw_solar1_bsn


def build_clean_dataset(
    config: dict[str, Any],
    *,
    repo_root: Path | None = None,
) -> tuple[Path, Path]:
    """Build clean `.npz` dataset bundle and `.meta.json` strictly read-only from `data/processed/`."""
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    processed_dir = _resolve_path(config.get("processed_dir", "data/processed"), root)
    out_dir = _resolve_path(config.get("out_dir", "ml_research/data_cache"), root)
    tag = str(config["tag"])
    include_short_interp = bool(config.get("include_short_interp", True))
    variation_step_policy = str(config.get("variation_step_policy", "flag_qc_exclude"))
    step_warn_nT = float(config.get("variation_step_warn_nT", 30.0))
    ref_distance_km = float(config.get("ballistic_ref_distance_km", 1_400_000.0))
    smooth_window_min = int(config.get("ballistic_smooth_window_min", 31))
    default_event_name = str(config.get("default_event_flag", "storm"))
    default_event_bit = int(EVENT_NAME_TO_BIT[default_event_name])

    catalog_path = processed_dir / "catalog.sqlite"
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Read-only catalog not found at {catalog_path}")

    uri = catalog_path.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        ep_df = pd.read_sql_query("SELECT * FROM episodes ORDER BY start_utc", conn)
        st_df = pd.read_sql_query(
            "SELECT * FROM stations WHERE station_type = 'magnetometer' ORDER BY iaga_code",
            conn,
        )
    finally:
        conn.close()

    station_codes = [str(c).strip().upper() for c in st_df["iaga_code"].tolist()]
    if station_codes != sorted(station_codes):
        raise ValueError("Station codes from catalog must be sorted lexicographically")
    station_regions = [
        infer_station_region(float(r["lon_deg"]), r["region"])
        for _, r in st_df.iterrows()
    ]
    lats = st_df["lat_deg"].to_numpy(dtype=np.float64)
    lons = st_df["lon_deg"].to_numpy(dtype=np.float64)
    station_has_baseline = st_df["has_baseline"].astype(bool).to_numpy(dtype=np.bool_)

    mlt_mod = import_mlt()
    conv_mod = import_conventions()
    station_coords, station_dist_km = compute_station_geometry(lats, lons, mlt_mod, conv_mod)
    station_belt_id = np.array(
        [
            classify_station_belt(reg, float(station_coords[i, 2]))
            for i, reg in enumerate(station_regions)
        ],
        dtype=np.int8,
    )

    episode_ids = [str(e) for e in ep_df["episode_id"].tolist()]
    ep_unix_chunks: list[np.ndarray] = []
    ep_idx_chunks: list[np.ndarray] = []
    bounds_list: list[list[int]] = []
    offset = 0
    for e_i, row in ep_df.iterrows():
        ep_ts = pd.date_range(row["start_utc"], row["end_utc"], freq="1min", tz="UTC")
        ep_s = timestamps_to_unix_s(ep_ts)
        n_ep = ep_s.shape[0]
        ep_unix_chunks.append(ep_s)
        ep_idx_chunks.append(np.full(n_ep, e_i, dtype=np.int16))
        bounds_list.append([offset, offset + n_ep])
        offset += n_ep

    timestamps_unix_s = np.concatenate(ep_unix_chunks)
    episode_idx = np.concatenate(ep_idx_chunks)
    episode_bounds = np.array(bounds_list, dtype=np.int64)

    n_t = timestamps_unix_s.shape[0]
    n_st = len(station_codes)
    st_to_idx = {code: i for i, code in enumerate(station_codes)}

    x_raw = np.full((n_t, n_st, 3), np.nan, dtype=np.float32)
    m_raw_valid = np.zeros((n_t, n_st), dtype=np.bool_)
    m_interpolated = np.zeros((n_t, n_st), dtype=np.bool_)
    m_despiked = np.zeros((n_t, n_st), dtype=np.bool_)

    for e_i, ep_id in enumerate(episode_ids):
        s, e = int(episode_bounds[e_i, 0]), int(episode_bounds[e_i, 1])
        ep_grid_s = timestamps_unix_s[s:e]
        pq_path = processed_dir / "geomag" / f"{ep_id}.parquet"
        gdf = pd.read_parquet(pq_path)
        gdf_s = timestamps_to_unix_s(gdf["timestamp_utc"])
        pos_t = np.searchsorted(ep_grid_s, gdf_s)
        if np.any(pos_t >= ep_grid_s.shape[0]) or not np.array_equal(ep_grid_s[pos_t], gdf_s):
            raise ValueError(f"Timestamp mismatch when aligning {pq_path}")

        codes_col = gdf["station_iaga"].astype(str).str.strip().str.upper().tolist()
        pos_s = np.array([st_to_idx[c] for c in codes_col], dtype=np.int64)
        global_t = s + pos_t

        xyz = np.column_stack(
            [
                gdf["x_nT"].to_numpy(dtype=np.float32),
                gdf["y_nT"].to_numpy(dtype=np.float32),
                gdf["z_nT"].to_numpy(dtype=np.float32),
            ]
        )
        raw_v = gdf["is_valid_xyz"].to_numpy(dtype=np.bool_)
        interp_v = gdf["is_interpolated"].to_numpy(dtype=np.bool_)
        despike_v = gdf["was_despiked"].to_numpy(dtype=np.bool_)

        x_raw[global_t, pos_s, :] = xyz
        m_raw_valid[global_t, pos_s] = raw_v
        m_interpolated[global_t, pos_s] = interp_v
        m_despiked[global_t, pos_s] = despike_v

    m_valid = (m_raw_valid | m_interpolated) if include_short_interp else m_raw_valid.copy()
    x_raw[~m_valid] = np.nan

    multi_mask, var_step_mask = build_qc_exclusion_masks(
        timestamps_unix_s,
        episode_idx,
        station_codes,
        episode_ids,
        config.get("qc_exclusions"),
    )
    m_qc_clean = m_valid & (~multi_mask)
    if variation_step_policy == "flag_qc_exclude":
        m_qc_clean &= ~var_step_mask
    elif variation_step_policy not in ("causal_initial_6h_reanchor", "keep_raw"):
        raise ValueError(f"Unsupported variation_step_policy={variation_step_policy!r}")

    m_known_spikes = np.zeros((n_t, n_st), dtype=np.bool_)

    mlt_hours = compute_mlt_matrix(lats, lons, timestamps_unix_s, mlt_mod)
    time_features = compute_time_features(timestamps_unix_s)

    solar1_df = pd.read_parquet(processed_dir / "solar_wind" / "solar1_l1_1min.parquet")
    sw_solar1_l1, sw_solar1_bsn = _compute_solar1_arrays(
        solar1_df,
        timestamps_unix_s,
        episode_bounds,
        ref_distance_km=ref_distance_km,
        smooth_window_min=smooth_window_min,
    )

    omni_df = pd.read_parquet(processed_dir / "solar_wind" / "omni_bsn_1min.parquet")
    omni_unix_s = timestamps_to_unix_s(omni_df["timestamp_utc"])
    omni_cols_data: list[np.ndarray] = []
    for col in OMNI_COLS:
        pq_col = OMNI_PARQUET_COL_MAP[col]
        if pq_col in omni_df.columns:
            omni_cols_data.append(omni_df[pq_col].to_numpy(dtype=np.float64))
        else:
            omni_cols_data.append(np.full(len(omni_df), np.nan, dtype=np.float64))
    omni_mat = np.column_stack(omni_cols_data)
    sw_omni = _align_series_by_unix_s(timestamps_unix_s, omni_unix_s, omni_mat)

    oulu_df = pd.read_parquet(processed_dir / "cosmic_rays" / "oulu_nm_1min.parquet")
    oulu_unix_s = timestamps_to_unix_s(oulu_df["timestamp_utc"])
    oulu_delta_n_pct = _align_series_by_unix_s(
        timestamps_unix_s,
        oulu_unix_s,
        oulu_df["delta_n_pct"].to_numpy(dtype=np.float64),
    )

    event_flags = np.full(n_t, default_event_bit, dtype=np.uint8)

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
    if step_warnings:
        warnings.warn(
            f"Detected {len(step_warnings)} station baseline step(s) > {step_warn_nT} nT: "
            + ", ".join(f"{w['station']}({w['max_step_nT']} nT)" for w in step_warnings),
            RuntimeWarning,
            stacklevel=2,
        )

    calendar_span_days = float((int(timestamps_unix_s[-1]) - int(timestamps_unix_s[0])) / 86400.0)
    meta: dict[str, Any] = {
        "tag": tag,
        "dataset_kind": "clean",
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
        "baseline_step_warnings": step_warnings,
    }

    validate_dataset_arrays(arrays, meta)
    npz_path = out_dir / f"geomag_ml_{tag}.npz"
    save_npz_atomic(npz_path, arrays, compressed=True)
    meta["dataset_sha256"] = compute_sha256(npz_path)
    meta_path = npz_path.with_suffix(".meta.json")
    write_json(meta_path, meta, nan_to_null=True)
    return npz_path, meta_path


def build_undespiked_dataset(
    config: dict[str, Any],
    *,
    clean_npz_path: Path | None = None,
    repo_root: Path | None = None,
) -> tuple[Path, Path]:
    """Build undespiked `.npz` directly from raw `.min` files (strictly read-only) on the clean grid."""
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    raw_dir = _resolve_path(config.get("raw_dir", "GeoMag_Storms_Dataset_Oct-2026"), root)
    out_dir = _resolve_path(config.get("out_dir", "ml_research/data_cache"), root)
    clean_tag = str(config["tag"])
    undespiked_tag = str(config.get("undespiked_tag", f"{clean_tag}_undespiked"))
    include_short_interp = bool(config.get("include_short_interp", True))

    target_clean = (
        Path(clean_npz_path)
        if clean_npz_path is not None
        else (out_dir / f"geomag_ml_{clean_tag}.npz")
    )
    if not target_clean.is_file():
        target_clean, _ = build_clean_dataset(config, repo_root=root)

    clean_arrays, clean_meta = load_ml_dataset(target_clean, validate=True)
    station_codes: list[str] = list(clean_meta["station_codes"])
    episode_ids: list[str] = list(clean_meta["episode_ids"])
    st_to_idx = {c: i for i, c in enumerate(station_codes)}

    etl = import_etl_parsers()
    conv = import_conventions()

    intermagnet_root = raw_dir / "GeoMag Storms" / "INTERMAGNET_1min"
    min_files = sorted(intermagnet_root.rglob("*.min"))
    if not min_files:
        raise FileNotFoundError(f"No raw .min files found under {intermagnet_root}")

    parsed_by_station_date: dict[tuple[str, str], Any] = {}
    for min_path in min_files:
        rec = etl.parse_iaga2002_file(min_path)
        if rec.iaga_code not in st_to_idx:
            continue
        date_str = rec.timestamps_utc[0].strftime("%Y-%m-%d")
        parsed_by_station_date[(rec.iaga_code, date_str)] = rec

    timestamps_unix_s = clean_arrays["timestamps_unix_s"]
    episode_idx = clean_arrays["episode_idx"]
    episode_bounds = clean_arrays["episode_bounds"]
    x_clean = clean_arrays["X_raw"]
    m_valid = clean_arrays["M_valid"]
    m_despiked = clean_arrays["M_despiked"]

    x_undespiked = np.full_like(x_clean, np.nan)
    for e_i in range(episode_bounds.shape[0]):
        s, end = int(episode_bounds[e_i, 0]), int(episode_bounds[e_i, 1])
        ep_grid_s = timestamps_unix_s[s:end]
        ep_ts = pd.to_datetime(ep_grid_s, unit="s", utc=True)
        ep_dates = sorted({ts.strftime("%Y-%m-%d") for ts in ep_ts})

        for st_i, code in enumerate(station_codes):
            recs = [
                parsed_by_station_date[(code, d)]
                for d in ep_dates
                if (code, d) in parsed_by_station_date
            ]
            if not recs:
                continue
            xyz_ep = np.full((end - s, 3), np.nan, dtype=np.float64)
            valid_ep = np.zeros(end - s, dtype=bool)
            for rec in recs:
                pos = etl.align_day_record_to_episode(ep_ts, rec, episode_id=episode_ids[e_i])
                xyz_ep[pos, 0] = rec.x_nT
                xyz_ep[pos, 1] = rec.y_nT
                xyz_ep[pos, 2] = rec.z_nT
                valid_ep[pos] = rec.is_valid_xyz

            if include_short_interp:
                xyz_filled, _ = conv.interpolate_short_gaps(xyz_ep, valid_ep, max_gap=3)
            else:
                xyz_filled = xyz_ep

            x_undespiked[s:end, st_i, :] = xyz_filled.astype(np.float32)

    x_undespiked[~m_valid] = np.nan

    # Verify exact diff against clean version
    diff_mag = np.zeros(m_valid.shape, dtype=np.float32)
    diff_mag[m_valid] = np.max(np.abs(x_undespiked[m_valid] - x_clean[m_valid]), axis=-1)
    diff_mask = m_valid & (diff_mag > 1e-3)
    if not np.array_equal(diff_mask, m_despiked):
        n_extra = int(np.sum(diff_mask & (~m_despiked)))
        n_missing = int(np.sum((~diff_mask) & m_despiked))
        raise RuntimeError(
            f"Undespiked diff mask mismatch vs M_despiked: extra={n_extra}, missing={n_missing}"
        )

    multi_mask, _ = build_qc_exclusion_masks(
        timestamps_unix_s,
        episode_idx,
        station_codes,
        episode_ids,
        config.get("qc_exclusions"),
    )
    m_known_spikes = (m_despiked | multi_mask) & m_valid
    m_qc_clean = clean_arrays["M_qc_clean"] & (~m_despiked)

    out_arrays = dict(clean_arrays)
    out_arrays["X_raw"] = x_undespiked
    out_arrays["M_known_spikes"] = m_known_spikes
    out_arrays["M_qc_clean"] = m_qc_clean

    out_meta = dict(clean_meta)
    out_meta["tag"] = undespiked_tag
    out_meta["dataset_kind"] = "undespiked"
    out_meta["created_utc"] = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    out_meta["clean_parent_sha256"] = clean_meta.get("dataset_sha256")
    out_meta["n_despiked_points"] = int(np.sum(m_despiked))
    out_meta["n_known_spikes"] = int(np.sum(m_known_spikes))

    validate_dataset_arrays(out_arrays, out_meta)
    npz_path = out_dir / f"geomag_ml_{undespiked_tag}.npz"
    save_npz_atomic(npz_path, out_arrays, compressed=True)
    out_meta["dataset_sha256"] = compute_sha256(npz_path)
    meta_path = npz_path.with_suffix(".meta.json")
    write_json(meta_path, out_meta, nan_to_null=True)
    return npz_path, meta_path


def load_config_and_build(
    config_path: Path,
    *,
    also_undespiked: bool = False,
    repo_root: Path | None = None,
) -> dict[str, Path]:
    """Load YAML dataset config and execute clean (+ optional undespiked) or extended build."""
    cfg = read_yaml(Path(config_path))
    mode = str(cfg.get("mode", "clean")).strip().lower()
    if mode == "extended":
        from geomag_ml.data.ingest_extended import build_extended_dataset

        ext_npz, ext_meta = build_extended_dataset(cfg, repo_root=repo_root)
        return {"npz": ext_npz, "meta": ext_meta}

    clean_npz, clean_meta = build_clean_dataset(cfg, repo_root=repo_root)
    result = {"clean_npz": clean_npz, "clean_meta": clean_meta}
    if also_undespiked:
        un_npz, un_meta = build_undespiked_dataset(
            cfg, clean_npz_path=clean_npz, repo_root=repo_root
        )
        result["undespiked_npz"] = un_npz
        result["undespiked_meta"] = un_meta
    return result
