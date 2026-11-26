"""Integration tests for Phase 7 launcher (--browser) and headless uvicorn web-service."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import httpx
import websockets.sync.client as ws_sync

from geomag_api.main import create_app
from geomag_etl.config import DEFAULT_PROCESSED_DIR
from launcher import find_free_port

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_PY = REPO_ROOT / "backend" / "launcher.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_launcher_browser_subprocess_lifecycle_wtc_cache_and_shutdown(tmp_path: Path):
    user_data = tmp_path / "launcher_user_data"
    ready_file = tmp_path / "ready.json"
    catalog_before = _sha256(DEFAULT_PROCESSED_DIR / "catalog.sqlite")

    env = dict(os.environ)
    env.pop("GEOMAG_REQUIRE_AUTH", None)
    env.pop("GEOMAG_BOOTSTRAP_TOKEN", None)
    env.pop("GEOMAG_ALLOWED_HOSTS", None)
    env.pop("GEOMAG_ALLOWED_ORIGINS", None)
    env["GEOMAG_STATIC_CACHE_DIR"] = str(tmp_path / "empty_static_cache")

    proc = subprocess.Popen(
        [
            sys.executable,
            str(LAUNCHER_PY),
            "--browser",
            "--no-browser-open",
            "--no-tray",
            "--user-data-dir",
            str(user_data),
            "--ready-file",
            str(ready_file),
        ],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    try:
        deadline = time.monotonic() + 25.0
        ready: dict | None = None
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                out, err = proc.communicate(timeout=5.0)
                err_txt = err.decode("utf-8", errors="replace")
                out_txt = out.decode("utf-8", errors="replace")
                raise AssertionError(
                    f"Launcher exited prematurely ({proc.returncode}): {err_txt}\n{out_txt}"
                )
            if ready_file.is_file():
                try:
                    ready = json.loads(ready_file.read_text(encoding="utf-8"))
                    break
                except json.JSONDecodeError:
                    pass
            time.sleep(0.15)
        assert ready is not None, "Launcher failed to write ready-file within 25s"
        assert ready["mode"] == "browser"
        assert Path(ready["user_data_dir"]) == user_data

        base_url = str(ready["base_url"])
        port = int(ready["port"])

        with httpx.Client(base_url=base_url, trust_env=False, timeout=15.0) as client:
            # Static SPA served at /
            idx = client.get("/")
            assert idx.status_code == 200
            assert "GeoMag Analyzer" in idx.text

            # Health reports can_shutdown=True in launcher mode
            health = client.get("/api/health")
            assert health.status_code == 200
            assert health.json()["catalog_stations"] == 50
            assert health.json()["can_shutdown"] is True

            # One-time bootstrap authentication
            boot = client.get(f"/api/auth/bootstrap?token={ready['bootstrap_token']}")
            assert boot.status_code == 200
            sess_tok = boot.json()["session_token"]
            hdrs = {"X-Session-Token": sess_tok}

            assert (
                client.get(f"/api/auth/bootstrap?token={ready['bootstrap_token']}").status_code
                == 401
            )

            stations = client.get("/api/catalog/stations")
            assert stations.status_code == 200 and len(stations.json()) == 50

            # Cancel mid-MC job
            sub_cancel = client.post(
                "/api/jobs/wtc",
                headers=hdrs,
                json={
                    "episode_id": "storm_20261009_11",
                    "station_i": "LVV",
                    "station_j": "BEL",
                    "component": "X",
                    "coord_frame": "XYZ",
                    "n_surrogates": 250,
                    "seed": 31,
                },
            )
            assert sub_cancel.status_code == 200
            cancel_jid = sub_cancel.json()["job_id"]

            wait_run_dl = time.monotonic() + 15.0
            while time.monotonic() < wait_run_dl:
                st = client.get(f"/api/jobs/{cancel_jid}").json()
                if st["status"] == "running" and st["completed_steps"] > 0:
                    break
                if st["status"] == "completed":
                    break
                time.sleep(0.1)

            assert client.delete(f"/api/jobs/{cancel_jid}", headers=hdrs).status_code == 200
            wait_canc_dl = time.monotonic() + 10.0
            final_canc = None
            while time.monotonic() < wait_canc_dl:
                final_canc = client.get(f"/api/jobs/{cancel_jid}").json()["status"]
                if final_canc in ("cancelled", "completed", "failed"):
                    break
                time.sleep(0.1)
            assert final_canc == "cancelled"
            assert not list((user_data / "user_cache" / "cross_wtc").glob("*.tmp*"))

            # Complete WTC job via WebSocket + verify cache hit on repeat
            wtc_req = {
                "episode_id": "storm_20261004_05",
                "station_i": "BEL",
                "station_j": "WNG",
                "component": "X",
                "coord_frame": "XYZ",
                "n_surrogates": 20,
                "seed": 42,
                "start_utc": "2026-10-04T00:00:00Z",
                "end_utc": "2026-10-04T07:59:00Z",
            }
            sub_ok = client.post("/api/jobs/wtc", headers=hdrs, json=wtc_req)
            assert sub_ok.status_code == 200
            ok_jid = sub_ok.json()["job_id"]
            assert sub_ok.json()["cache_hit"] is False

            cookie_val = client.cookies.get("geomag_session")
            assert cookie_val
            ws_url = f"ws://127.0.0.1:{port}/ws/jobs/{ok_jid}"
            ws_frames = []
            with ws_sync.connect(
                ws_url,
                additional_headers={
                    "Origin": base_url,
                    "Cookie": f"geomag_session={cookie_val}",
                },
            ) as ws:
                while True:
                    msg = json.loads(ws.recv(timeout=20.0))
                    ws_frames.append(msg)
                    if msg["status"] in ("completed", "cached", "failed", "cancelled"):
                        break
            assert ws_frames[-1]["status"] == "completed"

            sub_hit = client.post("/api/jobs/wtc", headers=hdrs, json=wtc_req)
            assert sub_hit.status_code == 200
            assert sub_hit.json()["cache_hit"] is True
            assert sub_hit.json()["status"] == "cached"

            # Verify runtime.sqlite + user_cache/ populated and catalog.sqlite untouched
            rt_db = user_data / "runtime.sqlite"
            assert rt_db.is_file()
            conn = sqlite3.connect(str(rt_db))
            try:
                cnt = conn.execute("SELECT COUNT(*) FROM cache_index").fetchone()[0]
                assert cnt >= 1
            finally:
                conn.close()
            assert list((user_data / "user_cache" / "cross_wtc").glob("*.npz"))
            assert _sha256(DEFAULT_PROCESSED_DIR / "catalog.sqlite") == catalog_before

            # Shutdown via UI endpoint terminates server cleanly
            sd = client.post("/api/shutdown", headers=hdrs)
            assert sd.status_code == 200

        rc = proc.wait(timeout=12.0)
        assert rc == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)


def test_web_service_mode_uvicorn_reverse_proxy_quotas_and_ws(tmp_path: Path):
    port = find_free_port("127.0.0.1")
    user_data = tmp_path / "web_user_data"
    static_cache = tmp_path / "web_static_cache"
    static_cache.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["GEOMAG_USER_DATA_DIR"] = str(user_data)
    env["GEOMAG_STATIC_CACHE_DIR"] = str(static_cache)
    env["GEOMAG_ALLOWED_HOSTS"] = "geomag.example.edu,127.0.0.1"
    env["GEOMAG_ALLOWED_ORIGINS"] = "https://geomag.example.edu"
    env["GEOMAG_MAX_WORKERS"] = "1"
    env["GEOMAG_MAX_MC_SURROGATES"] = "40"
    env["GEOMAG_MAX_ACTIVE_JOBS"] = "2"
    env["GEOMAG_REQUIRE_AUTH"] = "1"
    env["GEOMAG_COOKIE_SECURE"] = "1"
    env["GEOMAG_BOOTSTRAP_TOKEN"] = "web-proxy-bootstrap-token"

    popen_kwargs: dict[str, object] = {
        "cwd": str(REPO_ROOT),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "env": env,
    }
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

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
        **popen_kwargs,  # type: ignore[arg-type]
    )

    try:
        base_url = f"http://127.0.0.1:{port}"
        with httpx.Client(base_url=base_url, trust_env=False, timeout=15.0) as client:
            ready = False
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                try:
                    r = client.get("/api/health", headers={"Host": "geomag.example.edu"})
                    if r.status_code == 200:
                        ready = True
                        assert r.json()["can_shutdown"] is False
                        break
                except httpx.HTTPError:
                    time.sleep(0.2)
            assert ready, "Web-service uvicorn failed to become ready within 20s"

            # Serves built SPA at /
            spa = client.get("/", headers={"Host": "geomag.example.edu"})
            assert spa.status_code == 200 and "GeoMag Analyzer" in spa.text

            # Rejects forged origin/host
            assert (
                client.get(
                    "/api/health",
                    headers={"Host": "geomag.example.edu", "Origin": "https://evil.example"},
                ).status_code
                == 403
            )
            assert (
                client.get("/api/health", headers={"Host": "attacker.example"}).status_code == 403
            )

            # Reverse-proxied HTTPS bootstrap sets Secure + HttpOnly + SameSite=strict
            proxy_hdrs = {
                "Host": "geomag.example.edu",
                "Origin": "https://geomag.example.edu",
                "X-Forwarded-Proto": "https",
            }
            boot = client.get(
                "/api/auth/bootstrap?token=web-proxy-bootstrap-token",
                headers=proxy_hdrs,
            )
            assert boot.status_code == 200
            set_cookie = boot.headers.get("set-cookie", "").lower()
            assert "secure" in set_cookie
            assert "httponly" in set_cookie
            assert "samesite=strict" in set_cookie
            sess_tok = boot.json()["session_token"]

            # Shutdown endpoint MUST be disabled (501) in headless web-service mode
            sd = client.post(
                "/api/shutdown",
                headers={**proxy_hdrs, "X-Session-Token": sess_tok},
            )
            assert sd.status_code == 501

            # Enforces Tier 3 surrogate quota
            over_quota = client.post(
                "/api/jobs/wtc",
                headers={**proxy_hdrs, "X-Session-Token": sess_tok},
                json={
                    "episode_id": "storm_20261004_05",
                    "station_i": "BEL",
                    "station_j": "WNG",
                    "n_surrogates": 41,
                },
            )
            assert over_quota.status_code == 422

            # Executes valid WTC job over proxied WS & hits cache on repeat
            wtc_req = {
                "episode_id": "storm_20261004_05",
                "station_i": "BEL",
                "station_j": "WNG",
                "component": "X",
                "coord_frame": "XYZ",
                "n_surrogates": 10,
                "seed": 9,
                "start_utc": "2026-10-04T00:00:00Z",
                "end_utc": "2026-10-04T05:59:00Z",
            }
            sub = client.post(
                "/api/jobs/wtc",
                headers={**proxy_hdrs, "X-Session-Token": sess_tok},
                json=wtc_req,
            )
            assert sub.status_code == 200
            jid = sub.json()["job_id"]

            cookie_val = client.cookies.get("geomag_session")
            assert cookie_val
            frames = []
            with ws_sync.connect(
                f"ws://127.0.0.1:{port}/ws/jobs/{jid}",
                additional_headers={
                    "Host": "geomag.example.edu",
                    "Origin": "https://geomag.example.edu",
                    "Cookie": f"geomag_session={cookie_val}",
                },
            ) as ws:
                while True:
                    frame = json.loads(ws.recv(timeout=20.0))
                    frames.append(frame)
                    if frame["status"] in ("completed", "cached", "failed", "cancelled"):
                        break
            assert frames[-1]["status"] == "completed"

            hit = client.post(
                "/api/jobs/wtc",
                headers={**proxy_hdrs, "X-Session-Token": sess_tok},
                json=wtc_req,
            )
            assert hit.status_code == 200 and hit.json()["cache_hit"] is True
    finally:
        if proc.poll() is None:
            if sys.platform == "win32" and hasattr(signal, "CTRL_BREAK_EVENT"):
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.terminate()
        try:
            proc.communicate(timeout=10.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate(timeout=5.0)


def test_web_service_mode_require_auth_disabled_still_enforces_origin(tmp_path: Path):
    from fastapi.testclient import TestClient

    from geomag_api import ApiSettings

    settings = ApiSettings(
        processed_dir=DEFAULT_PROCESSED_DIR,
        static_cache_dir=tmp_path / "sc",
        user_data_dir=tmp_path / "ud",
        allowed_hosts=("geomag.example.edu", "127.0.0.1", "testserver"),
        allowed_origins=("https://geomag.example.edu",),
        require_auth=False,
        max_workers=1,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        sess = client.get("/api/auth/session", headers={"Host": "geomag.example.edu"})
        assert sess.status_code == 200
        assert sess.json()["authenticated"] is True
        assert sess.json()["require_auth"] is False

        eps = client.get("/api/catalog/episodes", headers={"Host": "geomag.example.edu"})
        assert eps.status_code == 200 and len(eps.json()) == 4

        evil = client.get(
            "/api/catalog/episodes",
            headers={"Host": "geomag.example.edu", "Origin": "https://evil.example"},
        )
        assert evil.status_code == 403
