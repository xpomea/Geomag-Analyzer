"""Main deterministic ETL orchestrator: `python -m geomag_etl.build`."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from .catalog import write_catalog_sqlite
from .config import (
    BASELINE_NORM_THRESHOLD_NT,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RAW_DIR,
    EPISODES,
)
from .intermagnet import process_intermagnet
from .manifest import enrich_manifest, scan_raw_manifest
from .omni import process_omni
from .oulu import process_oulu
from .solar_wind import process_solar1


def build_all(
    raw_root: Path = DEFAULT_RAW_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    *,
    enable_despike: bool = True,
) -> dict[str, object]:
    """Run full deterministic ETL from raw_root into processed_dir."""
    t0 = time.perf_counter()
    processed_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = scan_raw_manifest(raw_root)
    stations_geomag_df, availability_df, intermagnet_stats = process_intermagnet(
        raw_root, processed_dir, enable_despike=enable_despike
    )
    _, solar1_stats, solar1_summary = process_solar1(raw_root, processed_dir)
    _, omni_stats, omni_summary = process_omni(raw_root, processed_dir)
    _, oulu_station, oulu_baselines, oulu_stats = process_oulu(raw_root, processed_dir)

    all_file_stats = {
        **intermagnet_stats,
        **solar1_stats,
        **omni_stats,
        **oulu_stats,
    }
    enriched_manifest = enrich_manifest(manifest_df, all_file_stats)

    oulu_df = pd.DataFrame([oulu_station])
    for col in ("mean_vec_norm_nT", "cutoff_rigidity_gv", "barometric_coeff"):
        oulu_df[col] = oulu_df[col].astype("float64")
    stations_df = pd.DataFrame.from_records(
        [*stations_geomag_df.to_dict("records"), *oulu_df.to_dict("records")]
    )
    episodes_df = pd.DataFrame(
        [
            {
                "episode_id": ep.episode_id,
                "label": ep.label,
                "start_utc": ep.start_utc,
                "end_utc": ep.end_utc,
                "n_minutes": ep.n_minutes,
                "n_days": ep.n_days,
                "omni_available": ep.omni_available,
                "oulu_baseline_cpm": oulu_baselines[ep.episode_id],
            }
            for ep in EPISODES
        ]
    )

    metadata = {
        "etl_version": "1.0.0",
        "raw_files_count": int(len(enriched_manifest)),
        "raw_total_bytes": int(enriched_manifest["size_bytes"].sum()),
        "baseline_norm_threshold_nT": BASELINE_NORM_THRESHOLD_NT,
        "despike_enabled": bool(enable_despike),
        "solar1_summary": solar1_summary,
        "omni_summary": omni_summary,
        "oulu_baselines_cpm": oulu_baselines,
    }
    write_catalog_sqlite(
        processed_dir / "catalog.sqlite",
        stations_df=stations_df,
        episodes_df=episodes_df,
        manifest_df=enriched_manifest,
        availability_df=availability_df,
        metadata=metadata,
    )

    elapsed_sec = round(time.perf_counter() - t0, 3)
    return {
        "elapsed_sec": elapsed_sec,
        "processed_dir": str(processed_dir),
        "raw_files": int(len(enriched_manifest)),
        "raw_bytes": int(enriched_manifest["size_bytes"].sum()),
        "stations_total": int(len(stations_df)),
        "availability_rows": int(len(availability_df)),
        "f_only_missing_rows": int(availability_df["f_only_missing_count"].sum()),
        "despiked_points": int(availability_df["despiked_count"].sum()),
        "interpolated_points": int(availability_df["interpolated_count"].sum()),
        "solar1_summary": solar1_summary,
        "omni_summary": omni_summary,
        "oulu_baselines_cpm": oulu_baselines,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build normalized Parquet/SQLite dataset from GeoMag_Storms_Dataset_Oct-2026."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Path to read-only GeoMag_Storms_Dataset_Oct-2026 directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Output directory for data/processed/.",
    )
    parser.add_argument(
        "--no-despike",
        action="store_true",
        help="Disable diagnostic MAD/Hampel cross-station despiking.",
    )
    args = parser.parse_args(argv)
    summary = build_all(
        raw_root=args.raw_dir,
        processed_dir=args.out_dir,
        enable_despike=not args.no_despike,
    )
    print(
        f"ETL complete in {summary['elapsed_sec']}s -> {summary['processed_dir']} "
        f"({summary['raw_files']} raw files, {summary['stations_total']} stations, "
        f"despiked={summary['despiked_points']}, interpolated={summary['interpolated_points']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
