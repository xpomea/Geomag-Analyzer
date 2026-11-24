"""Unit tests for INTERMAGNET IAGA-2002 ETL edge cases."""

from __future__ import annotations

import math
import sqlite3

import numpy as np
import pandas as pd
import pytest

import geomag_core as gc
from geomag_etl.config import DEFAULT_PROCESSED_DIR, DEFAULT_RAW_DIR
from geomag_etl.intermagnet import (
    IagaDayRecord,
    align_day_record_to_episode,
    despike_episode_stations,
    interpolate_short_gaps,
    normalize_lon_180,
    parse_iaga2002_file,
)


def test_hdzf_stations_converted_by_formula_2():
    hdzf_expected = {"DUR", "ESK", "HAD", "IZN", "LER", "SBL", "SFS", "VAL"}
    conn = sqlite3.connect(str(DEFAULT_PROCESSED_DIR / "catalog.sqlite"))
    try:
        stations = pd.read_sql_query(
            "SELECT iaga_code, orig_format FROM stations WHERE station_type = 'magnetometer'",
            conn,
        )
    finally:
        conn.close()

    actual_hdzf = set(stations.loc[stations["orig_format"] == "HDZF", "iaga_code"])
    assert actual_hdzf == hdzf_expected

    # Spot-check every HDZF station directly against its first raw .min file at 00:00:00Z
    ep1 = pd.read_parquet(DEFAULT_PROCESSED_DIR / "geomag" / "storm_20261004_05.parquet")
    intermagnet_root = DEFAULT_RAW_DIR / "GeoMag Storms" / "INTERMAGNET_1min"
    for iaga in sorted(hdzf_expected):
        raw_path = sorted(intermagnet_root.rglob(f"{iaga.lower()}20261004*min.min"))[0]
        lines = raw_path.read_text(encoding="utf-8").splitlines()
        hdr = next(i for i, line in enumerate(lines) if line.startswith("DATE"))
        toks = lines[hdr + 1].split()
        h_val, d_arcmin, z_val = float(toks[3]), float(toks[4]), float(toks[5])
        d_rad = math.radians(d_arcmin / 60.0)
        exp_x = h_val * math.cos(d_rad)
        exp_y = h_val * math.sin(d_rad)

        row = ep1[
            (ep1["station_iaga"] == iaga)
            & (ep1["timestamp_utc"] == pd.Timestamp("2026-10-04T00:00:00Z"))
        ].iloc[0]
        assert math.isclose(row["x_nT"], exp_x, abs_tol=1e-9)
        assert math.isclose(row["y_nT"], exp_y, abs_tol=1e-9)
        assert math.isclose(row["z_nT"], z_val, abs_tol=1e-9)


def test_xyzf_misleading_sensor_orientation_ignored():
    # 10 stations with Sensor Orientation HDZ/HDZF + LON with DIF must remain XYZF
    misleading_xyzf = {"CLF", "EBR", "FUR", "GUI", "LON", "NGK", "PAG", "SPT", "SUA", "WIC", "WNG"}
    conn = sqlite3.connect(str(DEFAULT_PROCESSED_DIR / "catalog.sqlite"))
    try:
        stations = pd.read_sql_query("SELECT iaga_code, orig_format FROM stations", conn)
    finally:
        conn.close()

    fmt_map = dict(zip(stations["iaga_code"], stations["orig_format"], strict=True))
    for code in misleading_xyzf:
        assert fmt_map[code] == "XYZF"

    # Verify GUI (Oct 4) and SUA (Oct 30) raw X,Y,Z were not rotated
    ep1 = pd.read_parquet(DEFAULT_PROCESSED_DIR / "geomag" / "storm_20261004_05.parquet")
    ep4 = pd.read_parquet(DEFAULT_PROCESSED_DIR / "geomag" / "storm_20261030_31.parquet")
    for code, ep_df, date_tag in [
        ("GUI", ep1, "20261004"),
        ("LON", ep1, "20261004"),
        ("SUA", ep4, "20261030"),
    ]:
        raw_path = sorted(
            (DEFAULT_RAW_DIR / "GeoMag Storms" / "INTERMAGNET_1min").rglob(
                f"{code.lower()}{date_tag}*min.min"
            )
        )[0]
        rec = parse_iaga2002_file(raw_path)
        assert rec.orig_format == "XYZF"
        sub = ep_df[ep_df["station_iaga"] == code].sort_values("timestamp_utc")
        assert math.isclose(float(sub.iloc[0]["x_nT"]), float(rec.x_nT[0]), abs_tol=1e-9)
        assert math.isclose(float(sub.iloc[0]["y_nT"]), float(rec.y_nT[0]), abs_tol=1e-9)


def test_f_scalar_missing_does_not_invalidate_vector():
    conn = sqlite3.connect(str(DEFAULT_PROCESSED_DIR / "catalog.sqlite"))
    try:
        avail = pd.read_sql_query("SELECT * FROM availability", conn)
    finally:
        conn.close()

    assert int(avail["f_only_missing_count"].sum()) == 87_592
    assert int(avail["f_missing_count"].sum()) == 94_183

    lon_rows = avail[avail["station_iaga"] == "LON"]
    assert int(lon_rows["valid_vector_count"].sum()) == 14_400
    assert int(lon_rows["f_only_missing_count"].sum()) == 14_400

    ep1 = pd.read_parquet(DEFAULT_PROCESSED_DIR / "geomag" / "storm_20261004_05.parquet")
    lon_ep1 = ep1[ep1["station_iaga"] == "LON"]
    assert lon_ep1["is_valid_xyz"].all()
    assert lon_ep1["f_scalar_nT"].isna().all()
    assert lon_ep1["x_nT"].notna().all()


def test_has_baseline_flag_separates_gck_nck_from_res():
    conn = sqlite3.connect(str(DEFAULT_PROCESSED_DIR / "catalog.sqlite"))
    try:
        stations = pd.read_sql_query(
            "SELECT iaga_code, has_baseline, mean_vec_norm_nT FROM stations "
            "WHERE station_type = 'magnetometer'",
            conn,
        )
    finally:
        conn.close()

    assert len(stations) == 49
    no_base = set(stations.loc[stations["has_baseline"] == 0, "iaga_code"])
    assert no_base == {"GCK", "NCK"}
    res_row = stations.loc[stations["iaga_code"] == "RES"].iloc[0]
    assert res_row["has_baseline"] == 1
    assert res_row["mean_vec_norm_nT"] > 50_000.0


def test_despiking_cleans_val_ott_and_preserves_sc():
    ep2 = pd.read_parquet(DEFAULT_PROCESSED_DIR / "geomag" / "storm_20261009_11.parquet")
    val_spike = ep2[
        (ep2["station_iaga"] == "VAL")
        & (ep2["timestamp_utc"] == pd.Timestamp("2026-10-09T10:39:00Z"))
    ].iloc[0]
    assert bool(val_spike["was_despiked"]) is True
    # All 3 components (X, Y, Z) at VAL 10:39Z must be cleaned, not just Z
    assert 19_500.0 < float(val_spike["x_nT"]) < 19_680.0
    assert -1_000.0 < float(val_spike["y_nT"]) < -900.0
    assert 44_100.0 < float(val_spike["z_nT"]) < 44_200.0

    # Variant B polarization on VAL 2026-10-09 must reflect cleaned X, Y (no +65 deg spike bias)
    val_oct9 = ep2[
        (ep2["station_iaga"] == "VAL")
        & (ep2["timestamp_utc"] < pd.Timestamp("2026-10-10T00:00:00Z"))
    ].sort_values("timestamp_utc")
    xy_val = val_oct9[["x_nT", "y_nT"]].to_numpy(dtype=np.float64)
    st_val = gc.stokes_from_spectral_matrix_2d(
        gc.run_variant_b_pipeline(xy_val, band="Pc5", filter_method="sos", order_n=3).S
    )
    assert st_val.I < 450.0
    assert st_val.psi_deg < 0.0

    ep3 = pd.read_parquet(DEFAULT_PROCESSED_DIR / "geomag" / "storm_20261021_23.parquet")
    ott_spike = ep3[
        (ep3["station_iaga"] == "OTT")
        & (ep3["timestamp_utc"] == pd.Timestamp("2026-10-22T13:48:00Z"))
    ].iloc[0]
    assert bool(ott_spike["was_despiked"]) is True
    assert 49_950.0 < float(ott_spike["z_nT"]) < 49_965.0

    # OTT 14:00Z and MEA 17:33Z must clean all 3 components simultaneously
    ott_1400 = ep3[
        (ep3["station_iaga"] == "OTT")
        & (ep3["timestamp_utc"] == pd.Timestamp("2026-10-22T14:00:00Z"))
    ].iloc[0]
    assert bool(ott_1400["was_despiked"]) is True
    assert 18_325.0 < float(ott_1400["x_nT"]) < 18_350.0
    assert -4_205.0 < float(ott_1400["y_nT"]) < -4_188.0
    assert 49_940.0 < float(ott_1400["z_nT"]) < 49_960.0

    mea_1733 = ep3[
        (ep3["station_iaga"] == "MEA")
        & (ep3["timestamp_utc"] == pd.Timestamp("2026-10-23T17:33:00Z"))
    ].iloc[0]
    assert bool(mea_1733["was_despiked"]) is True
    assert 13_860.0 < float(mea_1733["x_nT"]) < 13_880.0
    assert 3_345.0 < float(mea_1733["y_nT"]) < 3_360.0
    assert 54_370.0 < float(mea_1733["z_nT"]) < 54_385.0

    # Real auroral jumps at ABK must NOT be marked as despiked
    ep1 = pd.read_parquet(DEFAULT_PROCESSED_DIR / "geomag" / "storm_20261004_05.parquet")
    abk_sc = ep1[
        (ep1["station_iaga"] == "ABK")
        & (ep1["timestamp_utc"] == pd.Timestamp("2026-10-04T18:32:00Z"))
    ].iloc[0]
    assert bool(abk_sc["was_despiked"]) is False

    abk_ep3 = ep3[
        (ep3["station_iaga"] == "ABK")
        & (ep3["timestamp_utc"] == pd.Timestamp("2026-10-21T18:03:00Z"))
    ].iloc[0]
    assert bool(abk_ep3["was_despiked"]) is False

    # Synthetic multi-station SC protection test
    t = np.arange(120, dtype=np.float64)
    base = np.column_stack([20000.0 + np.sin(t / 10.0), 1000.0 + np.cos(t / 10.0), 40000.0 + 0 * t])
    s1 = base.copy()
    s2 = base.copy()
    s1[50, 2] -= 500.0  # isolated single-station spike
    s1[90, 0] += 300.0  # simultaneous SC event on both stations
    s2[90, 0] += 280.0
    coords = {"AAA": (50.0, 20.0), "BBB": (50.5, 20.5)}
    cleaned, flags = despike_episode_stations({"AAA": s1, "BBB": s2}, coords)
    assert bool(flags["AAA"][50]) is True
    assert abs(cleaned["AAA"][50, 2] - 40000.0) < 1e-6
    assert bool(flags["AAA"][90]) is False
    assert abs(cleaned["AAA"][90, 0] - (base[90, 0] + 300.0)) < 1e-6


def test_sod_coordinate_shift_longitudes_and_partial_stations():
    conn = sqlite3.connect(str(DEFAULT_PROCESSED_DIR / "catalog.sqlite"))
    try:
        stations = pd.read_sql_query(
            "SELECT * FROM stations WHERE station_type = 'magnetometer'", conn
        )
        avail = pd.read_sql_query("SELECT * FROM availability", conn)
    finally:
        conn.close()

    assert (stations["lon_deg"] >= -180.0).all() and (stations["lon_deg"] <= 180.0).all()
    assert math.isclose(normalize_lon_180(349.75), -10.25)

    sod = stations.loc[stations["iaga_code"] == "SOD"].iloc[0]
    assert math.isclose(float(sod["lat_deg"]), 67.400, abs_tol=1e-4)
    assert math.isclose(float(sod["lon_deg"]), 26.600, abs_tol=1e-4)

    sod_avail = avail[avail["station_iaga"] == "SOD"].set_index("file_date")
    assert sod_avail.loc["2026-10-04", "quality_code"] == "v"
    assert math.isclose(float(sod_avail.loc["2026-10-04", "file_lat_deg"]), 67.400, abs_tol=1e-4)
    assert sod_avail.loc["2026-10-23", "quality_code"] == "p"
    assert math.isclose(float(sod_avail.loc["2026-10-23", "file_lat_deg"]), 67.370, abs_tol=1e-4)
    assert math.isclose(float(sod_avail.loc["2026-10-23", "file_lon_deg"]), 26.630, abs_tol=1e-4)
    assert sod_avail.loc["2026-10-30", "quality_code"] == "p"
    assert math.isclose(float(sod_avail.loc["2026-10-30", "file_lat_deg"]), 67.400, abs_tol=1e-4)

    expected_partial = {
        "LVV": 9, "RES": 8, "SOD": 8, "DOU": 7, "MAB": 7,
        "CLF": 6, "GUI": 6, "SFS": 6, "SPT": 6,
        "DUR": 5, "EBR": 5, "HRB": 5, "NCK": 5, "SUA": 2,
    }
    days_map = dict(zip(stations["iaga_code"], stations["days_present"], strict=True))
    assert len([k for k, v in days_map.items() if v < 10]) == 14
    for code, expected_days in expected_partial.items():
        assert days_map[code] == expected_days


def test_short_gap_interpolation_vs_large_gaps():
    # VAL 2026-10-04 00:02Z is a 1-min missing sample -> interpolated
    ep1 = pd.read_parquet(DEFAULT_PROCESSED_DIR / "geomag" / "storm_20261004_05.parquet")
    val_gap = ep1[
        (ep1["station_iaga"] == "VAL")
        & (ep1["timestamp_utc"] == pd.Timestamp("2026-10-04T00:02:00Z"))
    ].iloc[0]
    assert bool(val_gap["is_valid_xyz"]) is False
    assert bool(val_gap["is_interpolated"]) is True
    assert np.isfinite(val_gap["x_nT"])

    # NUR 2026-10-11 has a 972-min block gap -> remains NaN and not interpolated
    ep2 = pd.read_parquet(DEFAULT_PROCESSED_DIR / "geomag" / "storm_20261009_11.parquet")
    nur_nan = ep2[(ep2["station_iaga"] == "NUR") & (~ep2["is_valid_xyz"])]
    assert len(nur_nan) >= 972
    assert (~nur_nan["is_interpolated"]).all()
    assert nur_nan["x_nT"].isna().all()

    # Synthetic §3.2 check: <=3 min bounded interpolated, >3 min or boundary stay NaN
    xyz = np.array(
        [
            [np.nan, np.nan, np.nan],   # boundary gap -> stays NaN
            [10.0, 20.0, 30.0],
            [np.nan, np.nan, np.nan],   # len 2 -> interpolated
            [np.nan, np.nan, np.nan],
            [40.0, 50.0, 60.0],
            [np.nan, np.nan, np.nan],   # len 4 -> stays NaN
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [100.0, 110.0, 120.0],
        ],
        dtype=np.float64,
    )
    valid = np.isfinite(xyz[:, 0])
    out, is_interp = interpolate_short_gaps(xyz, valid, max_gap=3)
    assert np.isnan(out[0, 0]) and not is_interp[0]
    assert is_interp[2] and is_interp[3]
    assert math.isclose(out[2, 0], 20.0) and math.isclose(out[3, 0], 30.0)
    assert np.all(np.isnan(out[5:9, 0])) and not np.any(is_interp[5:9])


def test_align_day_record_to_episode_handles_partial_and_out_of_range():
    ep_ts = pd.date_range("2026-10-04T00:00:00Z", "2026-10-05T23:59:00Z", freq="1min", tz="UTC")
    partial_ts = pd.date_range("2026-10-04T06:00:00Z", periods=120, freq="1min", tz="UTC")
    n = len(partial_ts)
    rec = IagaDayRecord(
        iaga_code="TST",
        station_name="Test",
        source_of_data="",
        lat_deg=50.0,
        lon_deg=20.0,
        elev_m=100.0,
        orig_format="XYZF",
        quality_code="v",
        timestamps_utc=partial_ts,
        x_nT=np.zeros(n),
        y_nT=np.zeros(n),
        z_nT=np.zeros(n),
        f_scalar_nT=np.zeros(n),
        is_valid_xyz=np.ones(n, dtype=bool),
    )
    pos = align_day_record_to_episode(ep_ts, rec, episode_id="storm_20261004_05")
    assert pos.shape == (120,)
    assert int(pos[0]) == 360 and int(pos[-1]) == 479

    bad_rec = IagaDayRecord(
        iaga_code="TST",
        station_name="Test",
        source_of_data="",
        lat_deg=50.0,
        lon_deg=20.0,
        elev_m=100.0,
        orig_format="XYZF",
        quality_code="v",
        timestamps_utc=pd.date_range("2026-10-06T00:00:00Z", periods=10, freq="1min", tz="UTC"),
        x_nT=np.zeros(10),
        y_nT=np.zeros(10),
        z_nT=np.zeros(10),
        f_scalar_nT=np.zeros(10),
        is_valid_xyz=np.ones(10, dtype=bool),
    )
    with pytest.raises(ValueError, match="outside episode"):
        align_day_record_to_episode(ep_ts, bad_rec, episode_id="storm_20261004_05")
