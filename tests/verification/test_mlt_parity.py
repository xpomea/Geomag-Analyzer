"""Cross-language verification of centered-dipole IGRF-14 MLT against shared JSON fixture."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geomag_core import (
    IGRF14_EPOCH,
    IGRF14_NORTH_POLE_LAT_DEG,
    IGRF14_NORTH_POLE_LON_DEG,
    compute_mlt,
    compute_mlt_array,
    compute_subsolar_point,
    geo_to_magnetic_dipole,
    magnetic_dipole_to_geo,
    mlt_sector,
    solar_elevation_deg,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "mlt_dipole_reference.json"


@pytest.fixture(scope="module")
def mlt_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_mlt_model_metadata_matches_igrf14_centered_dipole(mlt_fixture: dict) -> None:
    model = mlt_fixture["model"]
    assert model["name"] == "centered_dipole_igrf14"
    assert model["epoch"] == pytest.approx(IGRF14_EPOCH, abs=1e-9)
    assert model["pole_lat_deg"] == pytest.approx(IGRF14_NORTH_POLE_LAT_DEG, abs=1e-9)
    assert model["pole_lon_deg"] == pytest.approx(IGRF14_NORTH_POLE_LON_DEG, abs=1e-9)


def test_subsolar_point_matches_shared_fixture(mlt_fixture: dict) -> None:
    for case in mlt_fixture["subsolar_cases"]:
        sub = compute_subsolar_point(case["utc_iso"])
        assert sub.declination_deg == pytest.approx(case["declination_deg"], abs=1e-6)
        assert sub.lon_deg == pytest.approx(case["lon_deg"], abs=1e-6)
        assert sub.eot_min == pytest.approx(case["eot_min"], abs=1e-6)
        assert sub.julian_date == pytest.approx(case["julian_date"], abs=1e-6)


def test_dipole_coordinates_and_round_trip_match_shared_fixture(mlt_fixture: dict) -> None:
    for case in mlt_fixture["dipole_cases"]:
        mlat, mlon = geo_to_magnetic_dipole(case["lat_deg"], case["lon_deg"])
        assert mlat == pytest.approx(case["mlat_deg"], abs=1e-6)
        assert mlon == pytest.approx(case["mlon_deg"], abs=1e-6)

        lat_rt, lon_rt = magnetic_dipole_to_geo(mlat, mlon)
        assert lat_rt == pytest.approx(case["lat_deg"], abs=1e-6)
        if abs(case["lat_deg"]) < 89.99:
            assert lon_rt == pytest.approx(case["lon_deg"], abs=1e-6)


def test_mlt_and_solar_elevation_match_shared_fixture(mlt_fixture: dict) -> None:
    for case in mlt_fixture["mlt_cases"]:
        mlt_h = compute_mlt(case["lat_deg"], case["lon_deg"], case["utc_iso"])
        elev = solar_elevation_deg(case["lat_deg"], case["lon_deg"], case["utc_iso"])
        assert mlt_h == pytest.approx(case["mlt_hours"], abs=1e-6)
        assert mlt_sector(mlt_h) == case["sector"]
        assert elev == pytest.approx(case["solar_elevation_deg"], abs=1e-6)


def test_compute_mlt_input_types_and_vectorized_array(mlt_fixture: dict) -> None:
    lvv_cases = [c for c in mlt_fixture["mlt_cases"] if c["station"] == "LVV"]
    iso_list = [c["utc_iso"] for c in lvv_cases]
    expected = np.array([c["mlt_hours"] for c in lvv_cases], dtype=np.float64)

    arr_iso = compute_mlt_array(49.9, 23.8, iso_list)
    np.testing.assert_allclose(arr_iso, expected, atol=1e-6)

    dti = pd.DatetimeIndex(iso_list)
    arr_dti = compute_mlt_array(49.9, 23.8, dti)
    np.testing.assert_allclose(arr_dti, expected, atol=1e-6)

    np_dt = np.array([np.datetime64(s.rstrip("Z")) for s in iso_list])
    arr_np = compute_mlt_array(49.9, 23.8, np_dt)
    np.testing.assert_allclose(arr_np, expected, atol=1e-6)

    # Scalar accepts datetime, pd.Timestamp, and float Unix seconds identically
    sample_iso = iso_list[0]
    dt_obj = datetime.fromisoformat(sample_iso.replace("Z", "+00:00")).astimezone(UTC)
    ts_pd = pd.Timestamp(sample_iso)
    unix_sec = dt_obj.timestamp()
    for val in (dt_obj, ts_pd, unix_sec):
        assert compute_mlt(49.9, 23.8, val) == pytest.approx(expected[0], abs=1e-9)


def test_mlt_handles_nat_and_nan_timestamps(mlt_fixture: dict) -> None:
    """Missing/NaT timestamps must propagate as NaN rather than INT64_MIN dummy dates."""
    from geomag_core.mlt import utc_to_unix_ms

    valid_iso = mlt_fixture["mlt_cases"][0]["utc_iso"]
    expected_mlt = compute_mlt(49.9, 23.8, valid_iso)

    for missing in (np.datetime64("NaT"), np.datetime64("NaT", "ns"), pd.NaT, float("nan")):
        assert math.isnan(utc_to_unix_ms(missing))
        sub = compute_subsolar_point(missing)
        assert math.isnan(sub.declination_deg)
        assert math.isnan(sub.lon_deg)
        assert math.isnan(solar_elevation_deg(49.9, 23.8, missing))
        assert math.isnan(compute_mlt(49.9, 23.8, missing))

    arr_np = np.array([valid_iso.rstrip("Z"), "NaT"], dtype="datetime64[ns]")
    out_np = compute_mlt_array(49.9, 23.8, arr_np)
    assert out_np[0] == pytest.approx(expected_mlt, abs=1e-9)
    assert math.isnan(out_np[1])

    idx_pd = pd.DatetimeIndex([valid_iso, pd.NaT])
    out_pd = compute_mlt_array(49.9, 23.8, idx_pd)
    assert out_pd[0] == pytest.approx(expected_mlt, abs=1e-9)
    assert math.isnan(out_pd[1])
