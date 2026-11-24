"""OULU Neutron Monitor (1-min cosmic rays) ETL parser and baseline normalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import EPISODES, OULU_STATION_META, date_to_episode_id


def process_oulu(
    raw_root: Path,
    out_dir: Path,
    *,
    quiet_window_minutes: int = 360,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, float], dict[str, dict[str, object]]]:
    """Merge OULU NM 1-min episode CSVs and compute per-episode quiet baseline variations."""
    oulu_dir = raw_root / "Cosmic Rays Activity" / "OULU_cosmic_rays_1min"
    csv_files = sorted(oulu_dir.glob("OULU_NM_1min_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No OULU NM CSV files found in {oulu_dir}")

    parts: list[pd.DataFrame] = []
    file_stats: dict[str, dict[str, object]] = {}

    for path in csv_files:
        df = pd.read_csv(path)
        df = df.rename(
            columns={
                "Timestamp": "timestamp_utc",
                "FractionalDate": "fractional_date",
                "UncorrectedCountRate[cts/min]": "uncorrected_cpm",
                "CorrectedCountRate[cts/min]": "corrected_cpm",
                "Pressure[mbar]": "pressure_mbar",
            }
        )
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df["episode_id"] = df["timestamp_utc"].dt.strftime("%Y-%m-%d").map(date_to_episode_id)
        if df["episode_id"].isna().any():
            raise ValueError(f"Unmapped timestamp dates in OULU file {path.name}")

        rel = path.relative_to(raw_root).as_posix()
        valid_pct = 100.0 * float(df["corrected_cpm"].notna().mean())
        file_stats[rel] = {
            "row_count": int(len(df)),
            "valid_pct": round(valid_pct, 4),
            "file_lat_deg": float(OULU_STATION_META["lat_deg"]),
            "file_lon_deg": float(OULU_STATION_META["lon_deg"]),
        }
        parts.append(df)

    merged = (
        pd.concat(parts, ignore_index=True)
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    # Verify strict 60s cadence inside every episode and compute quiet baseline
    episode_baselines: dict[str, float] = {}
    delta_pct = np.empty(len(merged), dtype=np.float64)

    for ep in EPISODES:
        mask = merged["episode_id"] == ep.episode_id
        sub = merged.loc[mask]
        if len(sub) != ep.n_minutes:
            raise ValueError(
                f"OULU episode {ep.episode_id} has {len(sub)} rows, expected {ep.n_minutes}"
            )
        diffs = sub["timestamp_utc"].diff().dropna().dt.total_seconds().to_numpy()
        if not np.allclose(diffs, 60.0):
            raise ValueError(f"Non-60s cadence in OULU episode {ep.episode_id}")
        n0 = float(np.median(sub["corrected_cpm"].iloc[:quiet_window_minutes].to_numpy()))
        episode_baselines[ep.episode_id] = round(n0, 4)
        delta_pct[mask.to_numpy()] = (
            100.0 * (sub["corrected_cpm"].to_numpy(dtype=np.float64) - n0) / n0
        )

    merged["delta_n_pct"] = delta_pct
    merged["episode_id"] = merged["episode_id"].astype("string")
    merged = merged[
        [
            "timestamp_utc",
            "episode_id",
            "fractional_date",
            "uncorrected_cpm",
            "corrected_cpm",
            "pressure_mbar",
            "delta_n_pct",
        ]
    ]

    cr_dir = out_dir / "cosmic_rays"
    cr_dir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(cr_dir / "oulu_nm_1min.parquet", index=False, compression="zstd")

    station_row = {**OULU_STATION_META, "days_present": sum(ep.n_days for ep in EPISODES)}
    return merged, station_row, episode_baselines, file_stats
