"""Integration tests for Tier 2 polarization and two-phase Welch cross-spectral endpoints."""

from __future__ import annotations

import math


def test_polarization_all_methods_real_stations_and_incomplete_handling(api_client):
    client, headers, _ = api_client

    for method in ("welch", "multitaper", "variant_b_fft", "variant_b_sos"):
        resp = client.post(
            "/api/polarization",
            headers=headers,
            json={
                "episode_id": "storm_20261009_11",
                "station_iaga": "LVV",
                "method": method,
                "band": "Pc5",
                "coord_frame": "XmYm",
                "include_3d": True,
                "include_dynamic": True,
                "include_filtered": method.startswith("variant_b"),
                "include_spectrogram": method == "welch",
                "step_samples": 120,
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["is_valid"] is True
        assert data["coord_frame"] == "XmYm"
        assert data["stokes_2d"] is not None
        assert 0.0 <= data["stokes_2d"]["p"] <= 1.0
        assert data["pol_3d"] is not None
        assert 0.0 <= data["pol_3d"]["P"] <= 1.0
        assert len(data["dynamic"]["timestamps_utc"]) > 20
        assert len(data["dynamic"]["n_eff"]) == len(data["dynamic"]["timestamps_utc"])
        assert len(data["dynamic"]["gamma2_crit"]) == len(data["dynamic"]["timestamps_utc"])
        assert any(v is not None and v > 1.0 for v in data["dynamic"]["n_eff"])
        if method == "welch":
            spec = data["spectrogram"]
            assert spec is not None
            assert len(spec["timestamps_utc"]) == len(data["dynamic"]["timestamps_utc"])
            assert len(spec["freqs_hz"]) > 10
            assert all(f > 0.0 for f in spec["freqs_hz"])
            assert len(spec["power_total"]) == len(spec["timestamps_utc"])
        if method.startswith("variant_b"):
            assert len(data["filtered"]["x_filt"]) == 4320
            assert data["trim_hours"] == 3.0
        else:
            assert data["filtered"] is None

    # NUR has a 972-min NaN block in storm_20261009_11 -> must serialize NaNs as JSON null
    r_nur = client.post(
        "/api/polarization",
        headers=headers,
        json={
            "episode_id": "storm_20261009_11",
            "station_iaga": "NUR",
            "method": "variant_b_sos",
            "band": "Pc5",
            "step_samples": 120,
        },
    )
    assert r_nur.status_code == 200
    assert r_nur.json()["is_valid"] is True

    # SUA on storm_20261004_05 -> valid HTTP 200 with is_valid=False and warning
    r_sua = client.post(
        "/api/polarization",
        headers=headers,
        json={
            "episode_id": "storm_20261004_05",
            "station_iaga": "SUA",
            "method": "welch",
        },
    )
    assert r_sua.status_code == 200
    assert r_sua.json()["is_valid"] is False
    assert r_sua.json()["warnings"]

    # GCK without external D0 -> 422
    r_gck = client.post(
        "/api/polarization",
        headers=headers,
        json={
            "episode_id": "storm_20261004_05",
            "station_iaga": "GCK",
            "method": "welch",
            "coord_frame": "XmYm",
        },
    )
    assert r_gck.status_code == 422


def test_network_snapshot_handles_baseless_and_absent_stations(api_client):
    client, headers, _ = api_client
    r = client.post(
        "/api/polarization/network-snapshot",
        headers=headers,
        json={
            "episode_id": "storm_20261004_05",
            "station_iagas": ["LVV", "BEL", "GCK", "SUA"],
            "band": "Pc5",
            "coord_frame": "XmYm",
        },
    )
    assert r.status_code == 200
    by_code = {s["station_iaga"]: s for s in r.json()["stations"]}
    assert by_code["LVV"]["is_valid"] is True
    assert by_code["BEL"]["is_valid"] is True
    assert by_code["GCK"]["available"] is True and by_code["GCK"]["is_valid"] is False
    assert by_code["SUA"]["available"] is False and by_code["SUA"]["is_valid"] is False


def test_cross_welch_pair_matrix_and_flr_with_phase1_cache(api_client):
    client, headers, settings = api_client

    pair_req = {
        "episode_id": "storm_20261009_11",
        "station_i": "LVV",
        "station_j": "BEL",
        "component": "X",
        "band": "Pc5",
        "window_samples": 360,
        "step_samples": 60,
    }
    r1 = client.post("/api/cross/welch/pair", headers=headers, json=pair_req)
    assert r1.status_code == 200, r1.text
    p1 = r1.json()
    assert p1["available"] is True
    assert p1["cache_hits"] == {"LVV": False, "BEL": False}
    assert any(p1["is_significant"])
    assert len(p1["phase_deg_tf"]) == len(p1["window_timestamps_utc"])
    assert len(p1["phase_deg_tf"][0]) == len(p1["freqs_hz"])
    assert abs(p1["azimuthal"]["delta_lon_deg"]) > 3.0

    # Second call must reuse Phase 1 NPZ cache from user_cache
    r2 = client.post("/api/cross/welch/pair", headers=headers, json=pair_req)
    assert r2.status_code == 200
    p2 = r2.json()
    assert p2["cache_hits"] == {"LVV": True, "BEL": True}
    assert list((settings.user_cache_dir / "spectra_phase1").glob("*.npz"))

    # Warm all-45-station matrix (1st pass caches stations, 2nd runs Phase 2 in < 2s)
    mat_req = {
        "episode_id": "storm_20261009_11",
        "component": "Xm",
        "coord_frame": "XmYm",
        "band": "Pc5",
        "window_samples": 360,
        "step_samples": 60,
    }
    rm1 = client.post("/api/cross/welch/matrix", headers=headers, json=mat_req)
    assert rm1.status_code == 200, rm1.text
    m1 = rm1.json()
    assert "GCK" in m1["skipped_stations"]  # baseless station safely excluded in XmYm
    assert len(m1["station_codes"]) >= 43

    rm2 = client.post("/api/cross/welch/matrix", headers=headers, json=mat_req)
    assert rm2.status_code == 200
    m2 = rm2.json()
    assert all(m2["cache_hits"].values())
    assert m2["elapsed_ms"] < 2000.0

    # Meridional FLR profile along European chain
    r_flr = client.post(
        "/api/cross/welch/flr",
        headers=headers,
        json={
            "episode_id": "storm_20261009_11",
            "station_chain": ["HRN", "ABK", "KIR", "LYC", "BEL", "LVV", "PAG"],
            "component": "X",
            "band": "Pc5",
            "window_samples": 360,
            "step_samples": 120,
            "window_index": 5,
        },
    )
    assert r_flr.status_code == 200, r_flr.text
    flr = r_flr.json()
    assert flr["peak_station"] in {"HRN", "ABK", "KIR", "LYC", "BEL", "LVV", "PAG"}
    assert math.isfinite(flr["peak_latitude_deg"])
