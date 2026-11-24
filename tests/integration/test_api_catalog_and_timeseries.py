"""Integration tests for /api/catalog and /api/timeseries (incl. ballistic shift & SUA)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from geomag_api.data_store import compute_ballistic_shift


def test_catalog_endpoints_full_metadata(api_client):
    client, _, _ = api_client
    resp = client.get("/api/catalog")
    assert resp.status_code == 200
    payload = resp.json()

    assert len(payload["episodes"]) == 4
    assert len(payload["stations"]) == 50
    assert len(payload["availability"]) == 435

    by_code = {s["iaga_code"]: s for s in payload["stations"]}
    assert by_code["GCK"]["has_baseline"] is False
    assert by_code["NCK"]["has_baseline"] is False
    assert by_code["RES"]["has_baseline"] is True
    assert by_code["OULU"]["station_type"] == "neutron_monitor"

    mags = client.get("/api/catalog/stations?station_type=magnetometer")
    assert mags.status_code == 200 and len(mags.json()) == 49

    eps = client.get("/api/catalog/episodes")
    assert eps.status_code == 200 and len(eps.json()) == 4

    avail_lvv = client.get("/api/catalog/availability?station_iaga=LVV")
    assert avail_lvv.status_code == 200 and len(avail_lvv.json()) == 9


def test_geomag_timeseries_despiked_baseless_guard_and_absent_station(api_client):
    client, _, _ = api_client

    # VAL on storm_20261009_11 has cleaned spike at 10:39Z across X, Y, Z
    r_val = client.get(
        "/api/timeseries/geomag",
        params={"episode_id": "storm_20261009_11", "station_iaga": "VAL"},
    )
    assert r_val.status_code == 200
    body = r_val.json()
    assert body["available"] is True
    assert len(body["timestamps_utc"]) == 4320
    idx_1039 = 10 * 60 + 39
    assert body["was_despiked"][idx_1039] is True
    assert 19_500.0 < body["x_nT"][idx_1039] < 19_680.0
    assert -1_000.0 < body["y_nT"][idx_1039] < -900.0
    assert 44_100.0 < body["z_nT"][idx_1039] < 44_200.0

    # GCK baseless guard: XYZ works, XmYm without external D0 -> 422, XmYm with D0 -> 200
    r_gck_xyz = client.get(
        "/api/timeseries/geomag",
        params={"episode_id": "storm_20261004_05", "station_iaga": "GCK", "coord_frame": "XYZ"},
    )
    assert r_gck_xyz.status_code == 200

    r_gck_xmy_fail = client.get(
        "/api/timeseries/geomag",
        params={
            "episode_id": "storm_20261004_05",
            "station_iaga": "GCK",
            "coord_frame": "XmYm",
        },
    )
    assert r_gck_xmy_fail.status_code == 422

    r_gck_xmy_ok = client.get(
        "/api/timeseries/geomag",
        params={
            "episode_id": "storm_20261004_05",
            "station_iaga": "GCK",
            "coord_frame": "XmYm",
            "external_d0_deg": 5.0,
        },
    )
    assert r_gck_xmy_ok.status_code == 200
    assert abs(r_gck_xmy_ok.json()["d0_deg"] - 5.0) < 1e-9

    # Incomplete station SUA on storm_20261004_05 (present only Oct 30-31)
    r_sua = client.get(
        "/api/timeseries/geomag",
        params={"episode_id": "storm_20261004_05", "station_iaga": "SUA"},
    )
    assert r_sua.status_code == 200
    sua = r_sua.json()
    assert sua["available"] is False
    assert sua["valid_pct"] == 0.0
    assert len(sua["timestamps_utc"]) == 2880
    assert any("storm_20261030_31" in w for w in sua["warnings"])

    # Unknown station -> 404; non-magnetometer OULU -> 422
    assert (
        client.get(
            "/api/timeseries/geomag",
            params={"episode_id": "storm_20261004_05", "station_iaga": "ZZZ"},
        ).status_code
        == 404
    )
    assert (
        client.get(
            "/api/timeseries/geomag",
            params={"episode_id": "storm_20261004_05", "station_iaga": "OULU"},
        ).status_code
        == 422
    )


def test_space_weather_ballistic_shift_monotonic_and_omni_coverage(api_client):
    client, _, _ = api_client

    # L1 raw vs ballistic_auto vs manual on storm_20261009_11
    r_raw = client.get(
        "/api/timeseries/space-weather",
        params={"episode_id": "storm_20261009_11", "solar1_shift": "l1_raw"},
    )
    assert r_raw.status_code == 200
    sw_raw = r_raw.json()
    assert sw_raw["solar1"]["time_frame"] == "L1"
    assert sw_raw["omni"]["available"] is True
    assert sw_raw["oulu"]["available"] is True
    assert all(v is not None for v in sw_raw["oulu"]["delta_n_pct"])

    r_auto = client.get(
        "/api/timeseries/space-weather",
        params={"episode_id": "storm_20261009_11", "solar1_shift": "ballistic_auto"},
    )
    assert r_auto.status_code == 200
    sw_auto = r_auto.json()["solar1"]
    assert sw_auto["time_frame"] == "L1_shifted_ballistic_auto"
    assert sw_auto["ref_distance_km"] == 1_400_000.0
    finite_delays = [d for d in sw_auto["propagation_delay_min"] if d is not None]
    assert len(finite_delays) > 4000
    assert all(20.0 <= d <= 90.0 for d in finite_delays)

    r_man = client.get(
        "/api/timeseries/space-weather",
        params={
            "episode_id": "storm_20261009_11",
            "solar1_shift": "manual",
            "manual_lag_min": 45.0,
        },
    )
    assert r_man.status_code == 200
    man_delays = [d for d in r_man.json()["solar1"]["propagation_delay_min"] if d is not None]
    assert all(abs(d - 45.0) < 1e-6 for d in man_delays)

    # Combined endpoint + missing OMNI warning on Oct 21-23
    r_comb = client.get(
        "/api/timeseries",
        params={
            "episode_id": "storm_20261021_23",
            "station_iaga": "LVV",
            "solar1_shift": "ballistic_auto",
        },
    )
    assert r_comb.status_code == 200
    comb = r_comb.json()
    assert comb["geomag"]["available"] is True
    assert comb["space_weather"]["omni"]["available"] is False
    assert comb["space_weather"]["omni"]["warning"] is not None

    # Direct synthetic monotonicity check on sharp speed step (-350 -> -900 km/s)
    ts = pd.date_range("2026-10-09T00:00:00Z", periods=100, freq="1min", tz="UTC")
    vx = np.concatenate([np.full(50, -350.0), np.full(50, -900.0)])
    t_arr, _ = compute_ballistic_shift(ts, vx, smooth_window_min=5)
    assert np.all(np.diff(t_arr) > 0.0)
