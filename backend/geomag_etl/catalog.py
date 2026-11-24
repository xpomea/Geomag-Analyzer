"""SQLite static catalog builder (read-only at runtime)."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pandas as pd

_SCHEMA_SQL = """
CREATE TABLE stations (
    iaga_code TEXT PRIMARY KEY,
    station_name TEXT NOT NULL,
    station_type TEXT NOT NULL,
    region TEXT NOT NULL,
    lat_deg REAL NOT NULL,
    lon_deg REAL NOT NULL,
    elev_m REAL NOT NULL,
    orig_format TEXT NOT NULL,
    has_baseline INTEGER NOT NULL,
    mean_vec_norm_nT REAL,
    days_present INTEGER NOT NULL,
    source_of_data TEXT NOT NULL,
    cutoff_rigidity_gv REAL,
    barometric_coeff REAL
);

CREATE TABLE episodes (
    episode_id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    start_utc TEXT NOT NULL,
    end_utc TEXT NOT NULL,
    n_minutes INTEGER NOT NULL,
    n_days INTEGER NOT NULL,
    omni_available INTEGER NOT NULL,
    oulu_baseline_cpm REAL
);

CREATE TABLE raw_manifest (
    relative_path TEXT PRIMARY KEY,
    source_group TEXT NOT NULL,
    extension TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    station_iaga TEXT,
    region TEXT,
    file_date TEXT,
    episode_id TEXT,
    quality_code TEXT,
    row_count INTEGER,
    valid_pct REAL,
    file_lat_deg REAL,
    file_lon_deg REAL
);

CREATE TABLE availability (
    station_iaga TEXT NOT NULL,
    file_date TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    quality_code TEXT NOT NULL,
    orig_format TEXT NOT NULL,
    valid_vector_count INTEGER NOT NULL,
    valid_vector_pct REAL NOT NULL,
    f_missing_count INTEGER NOT NULL,
    f_only_missing_count INTEGER NOT NULL,
    interpolated_count INTEGER NOT NULL,
    max_gap_min INTEGER NOT NULL,
    despiked_count INTEGER NOT NULL,
    file_lat_deg REAL NOT NULL,
    file_lon_deg REAL NOT NULL,
    PRIMARY KEY (station_iaga, file_date)
);

CREATE TABLE dataset_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX idx_raw_manifest_source ON raw_manifest (source_group);
CREATE INDEX idx_raw_manifest_station ON raw_manifest (station_iaga);
CREATE INDEX idx_availability_episode ON availability (episode_id);
"""


def write_catalog_sqlite(
    db_path: Path,
    stations_df: pd.DataFrame,
    episodes_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    availability_df: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    """Atomically write normalized catalog tables into catalog.sqlite."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_suffix(".sqlite.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    conn = sqlite3.connect(str(tmp_path))
    try:
        conn.executescript(_SCHEMA_SQL)

        stations_out = stations_df.copy()
        stations_out["has_baseline"] = stations_out["has_baseline"].astype(int)
        stations_out[
            [
                "iaga_code",
                "station_name",
                "station_type",
                "region",
                "lat_deg",
                "lon_deg",
                "elev_m",
                "orig_format",
                "has_baseline",
                "mean_vec_norm_nT",
                "days_present",
                "source_of_data",
                "cutoff_rigidity_gv",
                "barometric_coeff",
            ]
        ].to_sql("stations", conn, if_exists="append", index=False)

        episodes_out = episodes_df.copy()
        episodes_out["omni_available"] = episodes_out["omni_available"].astype(int)
        episodes_out[
            [
                "episode_id",
                "label",
                "start_utc",
                "end_utc",
                "n_minutes",
                "n_days",
                "omni_available",
                "oulu_baseline_cpm",
            ]
        ].to_sql("episodes", conn, if_exists="append", index=False)

        manifest_df[
            [
                "relative_path",
                "source_group",
                "extension",
                "size_bytes",
                "sha256",
                "station_iaga",
                "region",
                "file_date",
                "episode_id",
                "quality_code",
                "row_count",
                "valid_pct",
                "file_lat_deg",
                "file_lon_deg",
            ]
        ].to_sql("raw_manifest", conn, if_exists="append", index=False)

        availability_df[
            [
                "station_iaga",
                "file_date",
                "episode_id",
                "quality_code",
                "orig_format",
                "valid_vector_count",
                "valid_vector_pct",
                "f_missing_count",
                "f_only_missing_count",
                "interpolated_count",
                "max_gap_min",
                "despiked_count",
                "file_lat_deg",
                "file_lon_deg",
            ]
        ].to_sql("availability", conn, if_exists="append", index=False)

        meta_rows = [
            (
                str(k),
                v if isinstance(v, str) else json.dumps(v, ensure_ascii=False),
            )
            for k, v in sorted(metadata.items())
        ]
        conn.executemany(
            "INSERT INTO dataset_metadata (key, value) VALUES (?, ?)",
            meta_rows,
        )
        conn.commit()
    finally:
        conn.close()

    os.replace(tmp_path, db_path)
