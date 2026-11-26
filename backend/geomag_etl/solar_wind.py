"""SOLAR-1 L1 IMF (MAG) and plasma (SWiPS) ETL."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    EPISODES,
    SWIPS_DEFAULT_ALLOWED_FLAGS,
    SWIPS_FLAGS_WARNING,
    SWIPS_PHYSICAL_LIMITS,
)

_SWIPS_VALUE_COLS = (
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


def build_episode_minute_grid() -> pd.DataFrame:
    """Return the canonical 14,400-minute UTC grid across all 4 storm episodes."""
    frames = []
    for ep in EPISODES:
        ts = pd.date_range(ep.start_utc, ep.end_utc, freq="1min", tz="UTC")
        frames.append(pd.DataFrame({"timestamp_utc": ts, "episode_id": ep.episode_id}))
    return pd.concat(frames, ignore_index=True)


def parse_solar1_mag_csvs(
    raw_root: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Read all SOLAR-1 MAG 1-min CSVs and reindex onto the full UTC episode grid."""
    mag_dir = raw_root / "Solar Wind Data" / "SolarWind_Oct2026" / "SOLAR-1_IMF_MAG_1min_CSV"
    csv_files = sorted(mag_dir.glob("solar1_imf_1min_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No SOLAR-1 MAG CSV files found in {mag_dir}")

    file_stats: dict[str, dict[str, object]] = {}
    parts: list[pd.DataFrame] = []
    for path in csv_files:
        df = pd.read_csv(path)
        df["timestamp_utc"] = pd.to_datetime(df["time_utc"], utc=True)
        rel = path.relative_to(raw_root).as_posix()
        valid_rows = df[["bx_gsm_nT", "by_gsm_nT", "bz_gsm_nT"]].notna().all(axis=1).sum()
        file_stats[rel] = {
            "row_count": int(len(df)),
            "valid_pct": round(100.0 * float(valid_rows) / 1440.0, 4),
        }
        parts.append(df)

    raw_mag = pd.concat(parts, ignore_index=True).sort_values("timestamp_utc")
    raw_mag = raw_mag.drop_duplicates(subset=["timestamp_utc"], keep="first")

    grid = build_episode_minute_grid()
    merged = grid.merge(
        raw_mag.drop(columns=["time_utc"]),
        on="timestamp_utc",
        how="left",
    )
    merged["b_mag_gsm_nT"] = np.sqrt(
        merged["bx_gsm_nT"] ** 2 + merged["by_gsm_nT"] ** 2 + merged["bz_gsm_nT"] ** 2
    )
    return merged, file_stats


def parse_solar1_swips_csvs(
    raw_root: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Read all SOLAR-1 SWiPS plasma CSVs, sort strictly by UTC, and convert Pa -> nPa."""
    swips_dir = raw_root / "Solar Wind Data" / "SolarWind_Oct2026" / "SOLAR-1_plasma_SWiPS_1min_CSV"
    csv_files = sorted(swips_dir.glob("solar1_swips_proton_1min_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No SOLAR-1 SWiPS CSV files found in {swips_dir}")

    file_stats: dict[str, dict[str, object]] = {}
    parts: list[pd.DataFrame] = []
    for path in csv_files:
        df = pd.read_csv(path)
        df["timestamp_utc"] = pd.to_datetime(df["time_utc"], utc=True)
        rel = path.relative_to(raw_root).as_posix()
        valid_cnt = int((df["flags"] == 0).sum())
        file_stats[rel] = {
            "row_count": int(len(df)),
            "valid_pct": round(100.0 * valid_cnt / max(len(df), 1), 4),
        }
        parts.append(df)

    combined = pd.concat(parts, ignore_index=True)
    # Sort strictly ascending to fix backward timestamp jumps (e.g. -27.962 s on flags=11)
    combined = combined.sort_values("timestamp_utc", kind="mergesort").reset_index(drop=True)
    combined = combined.rename(columns={"proton_n_cm-3": "proton_n_cm3"})
    combined["pdyn_nPa"] = combined["proton_dyn_pressure_Pa"].astype(np.float64) * 1e9
    return combined, file_stats


def bin_swips_to_minute_grid(
    swips_df: pd.DataFrame,
    target_timestamps: pd.DatetimeIndex | pd.Series,
    *,
    allowed_flags: Sequence[int] = SWIPS_DEFAULT_ALLOWED_FLAGS,
    apply_physical_limits: bool = True,
    half_window_sec: float = 30.0,
) -> pd.DataFrame:
    """Deterministically snap irregular ~60.055s SWiPS scans onto UTC minute bins.

    For each target minute t_k, chooses the nearest valid scan within +-half_window_sec,
    breaking exact distance ties in favor of the earlier scan timestamp.
    """
    df = swips_df.sort_values("timestamp_utc", kind="mergesort").reset_index(drop=True)
    scan_ns = (
        df["timestamp_utc"].dt.tz_convert("UTC").dt.as_unit("ns").astype("int64").to_numpy()
    )
    target_idx = pd.DatetimeIndex(target_timestamps).tz_convert("UTC")
    target_ns = target_idx.as_unit("ns").astype("int64").to_numpy()
    half_ns = int(round(half_window_sec * 1_000_000_000))

    flags_arr = df["flags"].to_numpy(dtype=np.int64)
    allowed_set = set(int(f) for f in allowed_flags)
    is_valid_scan = np.isin(flags_arr, list(allowed_set))
    for col in _SWIPS_VALUE_COLS:
        is_valid_scan &= np.isfinite(df[col].to_numpy(dtype=np.float64))

    if apply_physical_limits:
        n_val = df["proton_n_cm3"].to_numpy(dtype=np.float64)
        t_val = df["proton_T_K"].to_numpy(dtype=np.float64)
        vx_val = df["vx_gse_kms"].to_numpy(dtype=np.float64)
        p_val = df["pdyn_nPa"].to_numpy(dtype=np.float64)
        n_lo, n_hi = SWIPS_PHYSICAL_LIMITS["proton_n_cm3"]
        t_lo, t_hi = SWIPS_PHYSICAL_LIMITS["proton_T_K"]
        vx_lo, vx_hi = SWIPS_PHYSICAL_LIMITS["vx_gse_kms"]
        p_lo, p_hi = SWIPS_PHYSICAL_LIMITS["pdyn_nPa"]
        is_valid_scan &= (
            (n_val > n_lo) & (n_val <= n_hi)
            & (t_val >= t_lo) & (t_val <= t_hi)
            & (vx_val >= vx_lo) & (vx_val <= vx_hi)
            & (p_val > p_lo) & (p_val <= p_hi)
        )

    n_target = len(target_ns)
    out_arrays = {col: np.full(n_target, np.nan, dtype=np.float64) for col in _SWIPS_VALUE_COLS}
    out_offset = np.full(n_target, np.nan, dtype=np.float64)
    out_flags = np.full(n_target, -1, dtype=np.int64)
    out_valid = np.zeros(n_target, dtype=bool)

    col_arrays = {col: df[col].to_numpy(dtype=np.float64) for col in _SWIPS_VALUE_COLS}
    lo_indices = np.searchsorted(scan_ns, target_ns - half_ns, side="left")
    hi_indices = np.searchsorted(scan_ns, target_ns + half_ns, side="right")

    for k in range(n_target):
        lo = int(lo_indices[k])
        hi = int(hi_indices[k])
        if lo >= hi:
            continue
        t_k = target_ns[k]
        cand_slice = slice(lo, hi)
        cand_valid = np.flatnonzero(is_valid_scan[cand_slice]) + lo
        if cand_valid.size > 0:
            diffs = np.abs(scan_ns[cand_valid] - t_k)
            # np.argmin picks the earliest index on exact tie because scan_ns is sorted
            best_idx = int(cand_valid[int(np.argmin(diffs))])
            for col in _SWIPS_VALUE_COLS:
                out_arrays[col][k] = col_arrays[col][best_idx]
            out_offset[k] = float(scan_ns[best_idx] - t_k) / 1e9
            out_flags[k] = int(flags_arr[best_idx])
            out_valid[k] = True
        else:
            # Record nearest raw scan flag in window while leaving plasma values NaN
            diffs_all = np.abs(scan_ns[cand_slice] - t_k)
            nearest_idx = lo + int(np.argmin(diffs_all))
            out_flags[k] = int(flags_arr[nearest_idx])

    result = pd.DataFrame({"timestamp_utc": target_idx})
    for col in _SWIPS_VALUE_COLS:
        result[col] = out_arrays[col]
    result["swips_flags"] = pd.Series(
        np.where(out_flags >= 0, out_flags, pd.NA), dtype="Int16"
    )
    result["sweep_offset_sec"] = out_offset
    result["is_swips_valid"] = out_valid
    return result


def process_solar1(
    raw_root: Path,
    out_dir: Path,
    *,
    allowed_flags: Sequence[int] = SWIPS_DEFAULT_ALLOWED_FLAGS,
    apply_physical_limits: bool = True,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]], dict[str, object]]:
    """Build normalized L1 solar1_l1_1min.parquet combining MAG and SWiPS."""
    mag_df, mag_stats = parse_solar1_mag_csvs(raw_root)
    swips_raw, swips_stats = parse_solar1_swips_csvs(raw_root)
    swips_binned = bin_swips_to_minute_grid(
        swips_raw,
        mag_df["timestamp_utc"],
        allowed_flags=allowed_flags,
        apply_physical_limits=apply_physical_limits,
    )

    out_df = pd.concat(
        [
            mag_df[
                [
                    "timestamp_utc",
                    "episode_id",
                    "bx_gse_nT",
                    "by_gse_nT",
                    "bz_gse_nT",
                    "bx_gsm_nT",
                    "by_gsm_nT",
                    "bz_gsm_nT",
                    "b_mag_gsm_nT",
                ]
            ].reset_index(drop=True),
            swips_binned.drop(columns=["timestamp_utc"]).reset_index(drop=True),
        ],
        axis=1,
    )
    out_df.insert(2, "time_frame", "L1")
    out_df["episode_id"] = out_df["episode_id"].astype("string")
    out_df["time_frame"] = out_df["time_frame"].astype("string")

    sw_dir = out_dir / "solar_wind"
    sw_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(sw_dir / "solar1_l1_1min.parquet", index=False, compression="zstd")

    all_stats = {**mag_stats, **swips_stats}
    summary = {
        "rows": int(len(out_df)),
        "time_frame": "L1",
        "mag_valid_rows": int(out_df["bz_gsm_nT"].notna().sum()),
        "swips_valid_rows": int(out_df["is_swips_valid"].sum()),
        "swips_flags_warning": SWIPS_FLAGS_WARNING,
    }
    return out_df, all_stats, summary
