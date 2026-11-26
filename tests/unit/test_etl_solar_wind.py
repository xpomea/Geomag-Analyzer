"""Unit tests for SOLAR-1 MAG, SWiPS, and NetCDF-3 ETL."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from geomag_etl.config import DEFAULT_PROCESSED_DIR, DEFAULT_RAW_DIR
from geomag_etl.solar_wind import (
    bin_swips_to_minute_grid,
    parse_solar1_swips_csvs,
)


def test_solar1_parquet_schema_units_and_l1_frame():
    df = pd.read_parquet(DEFAULT_PROCESSED_DIR / "solar_wind" / "solar1_l1_1min.parquet")
    assert len(df) == 14_400
    assert (df["time_frame"] == "L1").all()

    valid = df[df["is_swips_valid"]]
    assert len(valid) > 12_000
    # Default flags == 0 + physical limits exclude flag=8/1 spikes (n=119.1, Pdyn=56.9, T=4.63e6)
    assert (valid["swips_flags"] == 0).all()
    assert valid["proton_n_cm3"].max() <= 100.0
    assert valid["pdyn_nPa"].max() <= 50.0
    assert valid["proton_T_K"].max() <= 3_000_000.0
    # Units must be nPa (median ~1.5 nPa, NOT ~1e-9 Pa)
    assert 0.5 < float(valid["pdyn_nPa"].median()) < 5.0
    # Sweep offset within +-30 s
    assert (valid["sweep_offset_sec"].abs() <= 30.0).all()


def test_swips_non_monotonic_rows_275_276_and_midnight_spillover():
    raw_sorted, _ = parse_solar1_swips_csvs(DEFAULT_RAW_DIR)
    assert raw_sorted["timestamp_utc"].is_monotonic_increasing

    # Check Oct 4 around 04:29-04:31 UTC (rows 275-276 had -27.962 s jump in raw CSV)
    oct4_win = raw_sorted[
        (raw_sorted["timestamp_utc"] >= pd.Timestamp("2026-10-04T04:28:00Z"))
        & (raw_sorted["timestamp_utc"] <= pd.Timestamp("2026-10-04T04:32:00Z"))
    ]
    assert len(oct4_win) >= 4
    assert oct4_win["timestamp_utc"].is_monotonic_increasing

    # Midnight spillovers Oct 21 -> Oct 22 and Oct 30 -> Oct 31 appear cleanly on minute grid
    df = pd.read_parquet(DEFAULT_PROCESSED_DIR / "solar_wind" / "solar1_l1_1min.parquet")
    for ts in ["2026-10-22T00:00:00Z", "2026-10-31T00:00:00Z"]:
        row = df[df["timestamp_utc"] == pd.Timestamp(ts)].iloc[0]
        assert bool(row["is_swips_valid"]) is True


def test_mag_missing_minutes_oct31_are_nan():
    df = pd.read_parquet(DEFAULT_PROCESSED_DIR / "solar_wind" / "solar1_l1_1min.parquet")
    after_end = df[df["timestamp_utc"] >= pd.Timestamp("2026-10-31T23:16:00Z")]
    assert len(after_end) == 44
    assert after_end["bz_gsm_nT"].isna().all()


def test_swips_deterministic_30s_binning_and_tiebreak():
    synth = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                [
                    "2026-10-04T00:00:50Z",  # -10s from 00:01:00 (tie-break winner)
                    "2026-10-04T00:01:10Z",  # +10s from 00:01:00
                    "2026-10-04T00:02:05Z",  # +5s from 00:02:00, flags=8 (rejected)
                    "2026-10-04T00:03:31Z",  # +31s from 00:03:00 (out of window)
                ],
                utc=True,
            ),
            "proton_n_cm3": [10.0, 20.0, 99.0, 15.0],
            "proton_T_K": [1e5, 1e5, 1e5, 1e5],
            "vx_gse_kms": [-400.0, -400.0, -400.0, -400.0],
            "vy_gse_kms": [0.0, 0.0, 0.0, 0.0],
            "vz_gse_kms": [0.0, 0.0, 0.0, 0.0],
            "vx_gsm_kms": [-400.0, -400.0, -400.0, -400.0],
            "vy_gsm_kms": [0.0, 0.0, 0.0, 0.0],
            "vz_gsm_kms": [0.0, 0.0, 0.0, 0.0],
            "pdyn_nPa": [2.0, 3.0, 4.0, 2.5],
            "flags": [0, 0, 8, 0],
        }
    )
    target = pd.date_range("2026-10-04T00:01:00Z", "2026-10-04T00:03:00Z", freq="1min", tz="UTC")
    binned = bin_swips_to_minute_grid(synth, target)
    assert bool(binned.loc[0, "is_swips_valid"]) is True
    assert math.isclose(float(binned.loc[0, "proton_n_cm3"]), 10.0)
    assert math.isclose(float(binned.loc[0, "sweep_offset_sec"]), -10.0)
    assert bool(binned.loc[1, "is_swips_valid"]) is False
    assert np.isnan(binned.loc[1, "proton_n_cm3"])
    assert bool(binned.loc[2, "is_swips_valid"]) is False
    assert pd.isna(binned.loc[2, "swips_flags"])
