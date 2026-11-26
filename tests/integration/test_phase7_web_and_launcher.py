"""Integration tests for Phase 7 launcher (--browser) lifecycle and caching."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import httpx
import websockets.sync.client as ws_sync

from geomag_etl.config import DEFAULT_PROCESSED_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_PY = REPO_ROOT / "backend" / "launcher.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_launcher_browser_subprocess_lifecycle_wtc_cache_and_shutdown(tmp_path: Path):
    user_data = tmp_path / "launcher_user_data"
    ready_file = tmp_path / "ready.json"
    catalog_before = _sha256(DEFAULT_PROCESSED_DIR / "catalog.sqlite")

    proc = subprocess.Popen(
        [
            sys.executable,
            str(LAUNCHER_PY),
            "--browser",
            "--no-browser-open",
            "--user-data-dir",
            str(user_data),
            "--ready-file",
            str(ready_file),
        ],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
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

            cat = client.get("/api/catalog")
            assert cat.status_code == 200 and len(cat.json()["stations"]) == 50

            # Cancel mid-MC job
            sub_cancel = client.post(
                "/api/jobs/wtc",
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

            assert client.delete(f"/api/jobs/{cancel_jid}").status_code == 200
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
            sub_ok = client.post("/api/jobs/wtc", json=wtc_req)
            assert sub_ok.status_code == 200
            ok_jid = sub_ok.json()["job_id"]
            assert sub_ok.json()["cache_hit"] is False

            ws_url = f"ws://127.0.0.1:{port}/ws/jobs/{ok_jid}"
            ws_frames = []
            with ws_sync.connect(
                ws_url,
                additional_headers={"Origin": base_url},
            ) as ws:
                while True:
                    msg = json.loads(ws.recv(timeout=20.0))
                    ws_frames.append(msg)
                    if msg["status"] in ("completed", "cached", "failed", "cancelled"):
                        break
            assert ws_frames[-1]["status"] == "completed"

            sub_hit = client.post("/api/jobs/wtc", json=wtc_req)
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
            sd = client.post("/api/shutdown")
            assert sd.status_code == 200

        rc = proc.wait(timeout=12.0)
        assert rc == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)
