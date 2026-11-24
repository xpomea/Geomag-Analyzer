"""Unit tests for raw manifest inventory and SHA-256 verification."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

from geomag_etl.config import DEFAULT_PROCESSED_DIR, DEFAULT_RAW_DIR
from geomag_etl.manifest import scan_raw_manifest

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def test_manifest_counts_and_hashes():
    manifest_pq = pd.read_parquet(DEFAULT_PROCESSED_DIR / "manifest.parquet")
    assert len(manifest_pq) == 483
    assert int(manifest_pq["size_bytes"].sum()) == 108_484_030
    assert manifest_pq["sha256"].str.match(_SHA256_RE).all()
    assert manifest_pq["relative_path"].is_unique

    counts = manifest_pq["source_group"].value_counts().to_dict()
    assert counts == {
        "intermagnet": 435,
        "solar1_mag_csv": 10,
        "solar1_swips_csv": 10,
        "solar1_netcdf_mag": 10,
        "solar1_netcdf_swips": 10,
        "oulu_nm": 4,
        "omni_1min": 1,
        "documentation": 3,
    }


def test_manifest_matches_catalog_sqlite_and_is_deterministic():
    conn = sqlite3.connect(str(DEFAULT_PROCESSED_DIR / "catalog.sqlite"))
    try:
        db_df = pd.read_sql_query("SELECT * FROM raw_manifest ORDER BY relative_path", conn)
    finally:
        conn.close()

    pq_df = pd.read_parquet(DEFAULT_PROCESSED_DIR / "manifest.parquet").sort_values(
        "relative_path"
    ).reset_index(drop=True)
    assert len(db_df) == 483
    assert (db_df["relative_path"] == pq_df["relative_path"]).all()
    assert (db_df["sha256"] == pq_df["sha256"]).all()

    rescan = scan_raw_manifest(Path(DEFAULT_RAW_DIR)).sort_values("relative_path").reset_index(
        drop=True
    )
    assert (rescan["sha256"] == pq_df["sha256"]).all()
