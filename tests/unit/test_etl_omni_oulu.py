"""Unit tests for OMNI_1min positional parser and OULU NM ETL."""

from __future__ import annotations

import math
import sqlite3

import pandas as pd

from geomag_etl.config import DEFAULT_PROCESSED_DIR


def test_omni_per_column_fill_masking_and_coverage():
    omni = pd.read_parquet(DEFAULT_PROCESSED_DIR / "solar_wind" / "omni_bsn_1min.parquet")
    assert len(omni) == 7_200
    assert (omni["time_frame"] == "BSN").all()

    dates = sorted(omni["timestamp_utc"].dt.strftime("%Y-%m-%d").unique().tolist())
    assert dates == [
        "2026-10-04",
        "2026-10-05",
        "2026-10-09",
        "2026-10-10",
        "2026-10-11",
    ]

    # Exact per-column fill marker checks (99.9, 999.9, 99999)
    assert int(omni["magnetosonic_mach"].isna().sum()) == 2_540
    assert int(omni["alfven_mach"].isna().sum()) == 2_540
    assert int(omni["sym_h_nT"].isna().sum()) == 7_200
    assert int(omni["sym_d_nT"].isna().sum()) == 7_200
    assert int(omni["asy_h_nT"].isna().sum()) == 7_200
    assert int(omni["pc_n_index"].isna().sum()) == 7_200

    # Valid AE=99 / AE=999 must NOT be destroyed by naive string replacement
    assert int((omni["ae_nT"] == 99).sum()) == 25
    assert int((omni["ae_nT"] == 999).sum()) == 1
    assert int((omni["au_nT"] == 99).sum()) == 26
    assert int(omni["ae_nT"].notna().sum()) == 7_200

    # Oct 11 partial Bz_GSM coverage
    oct11 = omni[omni["timestamp_utc"].dt.strftime("%Y-%m-%d") == "2026-10-11"]
    assert int(oct11["bz_gsm_nT"].notna().sum()) == 596
    assert int(oct11["proton_n_cm3"].notna().sum()) == 490


def test_oulu_nm_merge_cadence_and_catalog_metadata():
    oulu = pd.read_parquet(DEFAULT_PROCESSED_DIR / "cosmic_rays" / "oulu_nm_1min.parquet")
    assert len(oulu) == 14_400
    assert oulu.isna().sum().sum() == 0
    assert oulu["episode_id"].value_counts().to_dict() == {
        "storm_20261004_05": 2880,
        "storm_20261009_11": 4320,
        "storm_20261021_23": 4320,
        "storm_20261030_31": 2880,
    }
    assert oulu["delta_n_pct"].abs().max() < 20.0

    conn = sqlite3.connect(str(DEFAULT_PROCESSED_DIR / "catalog.sqlite"))
    try:
        oulu_meta = pd.read_sql_query(
            "SELECT * FROM stations WHERE iaga_code = 'OULU'", conn
        ).iloc[0]
        episodes = pd.read_sql_query("SELECT * FROM episodes ORDER BY episode_id", conn)
    finally:
        conn.close()

    assert oulu_meta["station_type"] == "neutron_monitor"
    assert math.isclose(float(oulu_meta["cutoff_rigidity_gv"]), 0.81, abs_tol=1e-6)
    assert math.isclose(float(oulu_meta["lat_deg"]), 65.0544, abs_tol=1e-4)
    assert episodes["omni_available"].tolist() == [1, 1, 0, 0]
