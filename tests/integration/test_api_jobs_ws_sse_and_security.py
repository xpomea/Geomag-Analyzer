"""Integration tests for Tier 3 WTC jobs, WebSocket/SSE progress, cancel, security, and uvicorn."""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import httpx
import pytest
from starlette.websockets import WebSocketDisconnect

from geomag_etl.config import DEFAULT_PROCESSED_DIR


def test_security_host_origin_csrf_bootstrap_and_shutdown(make_test_client):
    shutdown_called = {"flag": False}

    def _on_shutdown() -> None:
        shutdown_called["flag"] = True

    client, settings = make_test_client(shutdown_callback=_on_shutdown)

    # Health reports can_shutdown
    h0 = client.get("/api/health")
    assert h0.status_code == 200 and h0.json()["can_shutdown"] is True

    # OpenAPI / Swagger UI must be documented and reachable
    assert client.get("/docs").status_code == 200
    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    assert "/api/jobs/wtc" in openapi.json()["paths"]

    # Bad Host or Origin (incl. cross-port localhost attack) must return 403
    assert client.get("/api/health", headers={"Host": "evil.example"}).status_code == 403
    assert (
        client.get("/api/health", headers={"Origin": "https://evil.example"}).status_code == 403
    )
    assert (
        client.get(
            "/api/health",
            headers={"Host": "127.0.0.1:8000", "Origin": "http://localhost:9999"},
        ).status_code
        == 403
    )
    assert (
        client.get(
            "/api/health",
            headers={"Host": "127.0.0.1:8000", "Origin": "http://127.0.0.1:9999"},
        ).status_code
        == 403
    )
    assert (
        client.get(
            "/api/health",
            headers={"Host": "127.0.0.1:8000", "Origin": "http://127.0.0.1:8000"},
        ).status_code
        == 200
    )

    # Unauthenticated GET on data/jobs endpoints and mutations must return 401
    assert client.get("/api/catalog").status_code == 401
    assert client.get("/api/jobs").status_code == 401
    assert (
        client.post(
            "/api/polarization",
            json={"episode_id": "storm_20261004_05", "station_iaga": "BEL"},
        ).status_code
        == 401
    )
    assert client.post("/api/shutdown").status_code == 401

    # Cookie before bootstrap is unauthorized
    assert client.get("/api/auth/session").status_code == 401

    # One-time bootstrap succeeds once and fails on reuse
    boot1 = client.get(f"/api/auth/bootstrap?token={settings.bootstrap_token}")
    assert boot1.status_code == 200
    sess_tok = boot1.json()["session_token"]
    boot2 = client.get(f"/api/auth/bootstrap?token={settings.bootstrap_token}")
    assert boot2.status_code == 401

    # Session cookie now unlocks GET /api/auth/session and GET /api/catalog
    sess_res = client.get("/api/auth/session")
    assert sess_res.status_code == 200
    assert sess_res.json()["session_token"] == sess_tok
    assert client.get("/api/catalog/episodes").status_code == 200

    # Evil Origin / cross-port Origin on WebSocket must be rejected before accept
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            "/ws/jobs/nonexistent", headers={"Origin": "https://evil.example"}
        ):
            pass

    # Valid X-Session-Token triggers /api/shutdown
    sd = client.post("/api/shutdown", headers={"X-Session-Token": sess_tok})
    assert sd.status_code == 200
    assert shutdown_called["flag"] is True

    # Headless server without shutdown_callback reports can_shutdown=False and 501
    c_no_sd, s_no_sd = make_test_client(
        user_data_dir=settings.user_data_dir / "no_sd",
    )
    assert c_no_sd.get("/api/health").json()["can_shutdown"] is False
    tok2 = c_no_sd.get(f"/api/auth/bootstrap?token={s_no_sd.bootstrap_token}").json()[
        "session_token"
    ]
    assert c_no_sd.post("/api/shutdown", headers={"X-Session-Token": tok2}).status_code == 501


def test_wtc_lifecycle_ws_progress_cache_hit_and_sse(api_client):
    client, headers, settings = api_client
    catalog_before = (DEFAULT_PROCESSED_DIR / "catalog.sqlite").read_bytes()

    wtc_req = {
        "episode_id": "storm_20261004_05",
        "station_i": "BEL",
        "station_j": "WNG",
        "component": "X",
        "coord_frame": "XYZ",
        "period_min_sec": 120.0,
        "period_max_sec": 1800.0,
        "dj": 1.0 / 8.0,
        "gap_policy": "subsegments",
        "n_surrogates": 30,
        "alpha": 0.05,
        "seed": 7,
        "start_utc": "2026-10-04T00:00:00Z",
        "end_utc": "2026-10-04T11:59:00Z",
    }

    # 1) Cache miss -> queued job
    r_sub = client.post("/api/jobs/wtc", headers=headers, json=wtc_req)
    assert r_sub.status_code == 200, r_sub.text
    job1 = r_sub.json()
    assert job1["cache_hit"] is False
    assert job1["status"] in ("queued", "running")

    # 2) WebSocket streams progress until completed
    ws_frames = []
    with client.websocket_connect(f"/ws/jobs/{job1['job_id']}") as ws:
        while True:
            msg = ws.receive_json()
            ws_frames.append(msg)
            if msg["status"] in ("completed", "cached", "failed", "cancelled"):
                break
    assert ws_frames[-1]["status"] == "completed"
    assert ws_frames[-1]["progress_pct"] == 100.0

    # 3) Fetch full WTC + 95% MC contour result
    res1 = client.get(f"/api/jobs/{job1['job_id']}/result", params={"stride": 4})
    assert res1.status_code == 200
    payload1 = res1.json()
    assert len(payload1["timestamps_utc"]) == 720 // 4
    assert len(payload1["wtc_sig95"]) == len(payload1["periods_sec"])
    assert payload1["subsegments"] == [[0, 720 // 4]]
    assert (settings.user_data_dir / "runtime.sqlite").is_file()

    # Read-only catalog.sqlite must remain byte-for-byte untouched
    assert (DEFAULT_PROCESSED_DIR / "catalog.sqlite").read_bytes() == catalog_before

    # 4) Identical second request -> instant cache hit
    r_cached = client.post("/api/jobs/wtc", headers=headers, json=wtc_req)
    assert r_cached.status_code == 200
    job2 = r_cached.json()
    assert job2["cache_hit"] is True
    assert job2["status"] == "cached"
    assert job2["progress_pct"] == 100.0

    # 4b) Reversed station pair (WNG, BEL) -> instant cache hit with negated phases
    r_rev = client.post(
        "/api/jobs/wtc",
        headers=headers,
        json={**wtc_req, "station_i": "WNG", "station_j": "BEL"},
    )
    assert r_rev.status_code == 200
    job_rev = r_rev.json()
    assert job_rev["cache_hit"] is True
    assert job_rev["status"] == "cached"
    res_rev = client.get(f"/api/jobs/{job_rev['job_id']}/result", params={"stride": 4}).json()
    assert res_rev["station_i"] == "WNG"
    assert res_rev["station_j"] == "BEL"
    assert res_rev["alpha1_i"] == pytest.approx(payload1["alpha1_j"], abs=1e-9)
    assert res_rev["alpha1_j"] == pytest.approx(payload1["alpha1_i"], abs=1e-9)
    assert res_rev["wtc_phase_deg"][0][10] == pytest.approx(
        -payload1["wtc_phase_deg"][0][10], abs=1e-9
    )

    # 5) Verify SSE fallback on cached/completed job (even with Accept-Encoding: gzip)
    with client.stream(
        "GET",
        f"/api/jobs/{job2['job_id']}/events",
        headers={"Accept-Encoding": "gzip"},
    ) as sse:
        assert sse.status_code == 200
        data_lines = [ln for ln in sse.iter_lines() if ln.startswith("data: ")]
        assert data_lines
        evt = json.loads(data_lines[-1][6:])
        assert evt["status"] == "cached"

    # 6) Cache index lists cross_wtc artifact
    c_list = client.get("/api/cache?category=cross_wtc")
    assert c_list.status_code == 200
    assert any(entry["cache_key"] == job1["cache_key"] for entry in c_list.json())


def test_wtc_mid_monte_carlo_cancellation_cleans_up_temp_files(api_client):
    client, headers, settings = api_client

    r_sub = client.post(
        "/api/jobs/wtc",
        headers=headers,
        json={
            "episode_id": "storm_20261009_11",
            "station_i": "LVV",
            "station_j": "BEL",
            "component": "X",
            "coord_frame": "XYZ",
            "n_surrogates": 300,
            "seed": 19,
        },
    )
    assert r_sub.status_code == 200
    job_id = r_sub.json()["job_id"]
    cache_key = r_sub.json()["cache_key"]

    # Wait briefly until worker enters running Monte Carlo loop
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        cur = client.get(f"/api/jobs/{job_id}").json()
        if cur["status"] == "running" and cur["completed_steps"] > 0:
            break
        if cur["status"] == "completed":
            break
        time.sleep(0.1)

    r_del = client.delete(f"/api/jobs/{job_id}", headers=headers)
    assert r_del.status_code == 200

    # Poll until terminal cancelled state within 10s
    final_status = None
    cancel_deadline = time.monotonic() + 10.0
    while time.monotonic() < cancel_deadline:
        st = client.get(f"/api/jobs/{job_id}").json()["status"]
        if st in ("cancelled", "completed", "failed"):
            final_status = st
            break
        time.sleep(0.1)

    assert final_status == "cancelled"
    assert not list((settings.user_cache_dir / "cross_wtc").glob("*.tmp*"))
    assert not (settings.user_cache_dir / "cross_wtc" / f"{cache_key}.npz").exists()
    assert client.get(f"/api/jobs/{job_id}/result").status_code == 409


def test_wtc_rejects_absent_station_and_baseless_xmy_and_quota(api_client):
    client, headers, _ = api_client

    # SUA absent in storm_20261004_05
    r_sua = client.post(
        "/api/jobs/wtc",
        headers=headers,
        json={
            "episode_id": "storm_20261004_05",
            "station_i": "SUA",
            "station_j": "BEL",
            "n_surrogates": 10,
        },
    )
    assert r_sua.status_code == 422

    # GCK without external D0 in XmYm
    r_gck = client.post(
        "/api/jobs/wtc",
        headers=headers,
        json={
            "episode_id": "storm_20261004_05",
            "station_i": "GCK",
            "station_j": "BEL",
            "coord_frame": "XmYm",
            "component": "Xm",
            "n_surrogates": 10,
        },
    )
    assert r_gck.status_code == 422

    # Exceeding max_mc_surrogates quota
    r_quota = client.post(
        "/api/jobs/wtc",
        headers=headers,
        json={
            "episode_id": "storm_20261004_05",
            "station_i": "BEL",
            "station_j": "WNG",
            "n_surrogates": 2000,
        },
    )
    assert r_quota.status_code == 422

    # Component / coord_frame mismatch -> 422
    assert (
        client.post(
            "/api/jobs/wtc",
            headers=headers,
            json={
                "episode_id": "storm_20261004_05",
                "station_i": "BEL",
                "station_j": "WNG",
                "coord_frame": "XYZ",
                "component": "Xm",
                "n_surrogates": 10,
            },
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/jobs/wtc",
            headers=headers,
            json={
                "episode_id": "storm_20261004_05",
                "station_i": "BEL",
                "station_j": "WNG",
                "coord_frame": "XmYm",
                "component": "X",
                "n_surrogates": 10,
            },
        ).status_code
        == 422
    )


def test_orphaned_jobs_reconciled_on_startup_and_cancel(tmp_path: Path, make_test_client):
    from geomag_api.cache import CacheManager, utc_now_iso

    ud = tmp_path / "orphan_ud"
    sc = tmp_path / "orphan_sc"
    sc.mkdir(parents=True, exist_ok=True)
    cm = CacheManager(
        static_cache_dir=sc,
        user_cache_dir=ud / "user_cache",
        runtime_db_path=ud / "runtime.sqlite",
    )
    now = utc_now_iso()
    conn = cm.connect_runtime()
    try:
        for jid, st, ck in [
            ("stale-run", "running", "0" * 64),
            ("stale-canc", "cancelling", "1" * 64),
        ]:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, job_type, cache_key, status, stage, progress_pct,
                    completed_steps, total_steps, eta_sec, cache_hit, result_path,
                    error_message, params_json, created_at, updated_at
                ) VALUES (?, 'cross_wtc', ?, ?, ?, 50.0, 5, 10, 5.0, 0, NULL, NULL, ?, ?, ?)
                """,
                (jid, ck, st, st, '{"n_surrogates": 10}', now, now),
            )
        conn.commit()
    finally:
        conn.close()

    client, settings = make_test_client(user_data_dir=ud, static_cache_dir=sc)
    tok = client.get(
        f"/api/auth/bootstrap?token={settings.bootstrap_token}"
    ).json()["session_token"]
    hdrs = {"X-Session-Token": tok}

    # On startup JobManager reconciled prior queued/running/cancelling jobs to failed
    for jid in ("stale-run", "stale-canc"):
        j = client.get(f"/api/jobs/{jid}").json()
        assert j["status"] == "failed"
        assert "restart" in j["error_message"].lower()

    # Stale row without in-memory Future -> DELETE cancels immediately
    conn2 = cm.connect_runtime()
    try:
        conn2.execute(
            """
            INSERT INTO jobs (
                job_id, job_type, cache_key, status, stage, progress_pct,
                completed_steps, total_steps, eta_sec, cache_hit, result_path,
                error_message, params_json, created_at, updated_at
            ) VALUES (
                'orphan-live', 'cross_wtc', ?, 'running', 'mc', 20.0, 2, 10, 4.0,
                0, NULL, NULL, ?, ?, ?
            )
            """,
            ("2" * 64, '{"n_surrogates": 10}', now, now),
        )
        conn2.commit()
    finally:
        conn2.close()

    r_del = client.delete("/api/jobs/orphan-live", headers=hdrs)
    assert r_del.status_code == 200
    assert r_del.json()["status"] == "cancelled"

    # Orphan with matching cache_key is marked failed and fresh worker starts
    import geomag_core as gc
    from geomag_core import build_wtc_cache_key

    ck_77 = build_wtc_cache_key(
        gc.__version__,
        episode_id="storm_20261004_05",
        pair=("BEL", "WNG"),
        component="X",
        coord_frame="XYZ",
        d0_deg=0.0,
        period_min_sec=120.0,
        period_max_sec=3600.0,
        dj=1.0 / 12.0,
        omega0=6.0,
        gap_policy="subsegments",
        n_surrogates=0,
        alpha=0.05,
        seed=77,
        qc_params={
            "d0_deg_i": 0.0,
            "d0_deg_j": 0.0,
            "start_utc": "2026-10-04T00:00:00Z",
            "end_utc": "2026-10-04T05:59:00Z",
            "min_subsegment_samples": 120,
        },
    )
    conn3 = cm.connect_runtime()
    try:
        conn3.execute(
            """
            INSERT INTO jobs (
                job_id, job_type, cache_key, status, stage, progress_pct,
                completed_steps, total_steps, eta_sec, cache_hit, result_path,
                error_message, params_json, created_at, updated_at
            ) VALUES (
                'fake-stale-ck', 'cross_wtc', ?, 'queued', 'queued', 0.0, 0, 1,
                NULL, 0, NULL, NULL, ?, ?, ?
            )
            """,
            (ck_77, '{"n_surrogates": 0}', now, now),
        )
        conn3.commit()
    finally:
        conn3.close()

    wtc_req = {
        "episode_id": "storm_20261004_05",
        "station_i": "BEL",
        "station_j": "WNG",
        "component": "X",
        "coord_frame": "XYZ",
        "n_surrogates": 0,
        "seed": 77,
        "start_utc": "2026-10-04T00:00:00Z",
        "end_utc": "2026-10-04T05:59:00Z",
    }
    sub1 = client.post("/api/jobs/wtc", headers=hdrs, json=wtc_req).json()
    assert sub1["job_id"] != "fake-stale-ck"
    stale_after = client.get("/api/jobs/fake-stale-ck").json()
    assert stale_after["status"] == "failed"

    # Wait briefly for fresh worker to finish cleanly before fixture exit
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        st_sub = client.get(f"/api/jobs/{sub1['job_id']}").json()
        if st_sub["status"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(0.05)
    assert client.get(f"/api/jobs/{sub1['job_id']}").json()["status"] == "completed"


@pytest.mark.asyncio
async def test_async_httpx_transport_and_endpoints(tmp_path: Path):
    from geomag_api import ApiSettings, create_app

    settings = ApiSettings(
        processed_dir=DEFAULT_PROCESSED_DIR,
        static_cache_dir=tmp_path / "static_cache",
        user_data_dir=tmp_path / "user_data",
        max_workers=1,
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://127.0.0.1:8000"
        ) as ac:
            h = await ac.get("/api/health")
            assert h.status_code == 200
            assert h.json()["catalog_stations"] == 50

            assert (await ac.get("/api/catalog/episodes")).status_code == 401
            boot = await ac.get(f"/api/auth/bootstrap?token={settings.bootstrap_token}")
            assert boot.status_code == 200

            cat = await ac.get("/api/catalog/episodes")
            assert cat.status_code == 200 and len(cat.json()) == 4


def test_uvicorn_live_subprocess_smoke(tmp_path: Path):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = int(s.getsockname()[1])

    env = dict(os.environ)
    env["GEOMAG_USER_DATA_DIR"] = str(tmp_path / "uvicorn_user_data")
    env["GEOMAG_STATIC_CACHE_DIR"] = str(tmp_path / "uvicorn_static_cache")

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "geomag_api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--workers",
            "1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        base_url = f"http://127.0.0.1:{port}"
        ready = False
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            try:
                r = httpx.get(f"{base_url}/api/health", timeout=1.0)
                if r.status_code == 200:
                    assert r.json()["catalog_stations"] == 50
                    ready = True
                    break
            except httpx.HTTPError:
                time.sleep(0.2)
        assert ready, "uvicorn subprocess failed to become healthy within 15s"
        oa = httpx.get(f"{base_url}/openapi.json", timeout=2.0)
        assert oa.status_code == 200
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)


def test_spawn_worker_enforces_single_numba_thread_without_loading_main():
    from geomag_api.workers import inspect_worker_runtime, worker_init

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx, initializer=worker_init) as pool:
        info = pool.submit(inspect_worker_runtime).result(timeout=30.0)

    assert info["env_numba_threads"] == "1"
    assert info["main_loaded"] is False
    assert info["fastapi_loaded"] is False
    if info["numba_get_num_threads"] is not None:
        assert info["numba_get_num_threads"] == 1
        assert info["numba_config_threads"] == 1


def test_bootstrap_token_logged_when_autogenerated_and_cookie_secure_flag(
    make_test_client, caplog
):
    with caplog.at_level(logging.INFO, logger="geomag_api"):
        c_auto, s_auto = make_test_client()
        assert any(s_auto.bootstrap_token in rec.getMessage() for rec in caplog.records)

        # Plain HTTP omits Secure; X-Forwarded-Proto: https sets Secure
        r_https = c_auto.get(
            f"/api/auth/bootstrap?token={s_auto.bootstrap_token}",
            headers={"X-Forwarded-Proto": "https"},
        )
        assert r_https.status_code == 200
        set_cookie_https = r_https.headers.get("set-cookie", "").lower()
        assert "httponly" in set_cookie_https and "secure" in set_cookie_https

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="geomag_api"):
        c_explicit, s_exp = make_test_client(
            bootstrap_token="explicit-fixed-token",
            user_data_dir=s_auto.user_data_dir / "explicit",
        )
        assert not any("explicit-fixed-token" in rec.getMessage() for rec in caplog.records)
        r_http = c_explicit.get("/api/auth/bootstrap?token=explicit-fixed-token")
        assert r_http.status_code == 200
        set_cookie_http = r_http.headers.get("set-cookie", "").lower()
        assert "httponly" in set_cookie_http and "; secure" not in set_cookie_http
