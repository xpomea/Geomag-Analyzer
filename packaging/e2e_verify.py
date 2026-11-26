"""Frozen Windows E2E acceptance suite for GeoMag Analyzer (Phase 7).

Tests the built distributable (`GeoMagAnalyzer_Portable_Win64.zip`) in a stripped
environment WITHOUT installed Python, PYTHONPATH, or network access:
  1. Portable ZIP extraction into a path with spaces, `portable.mode` -> `./user_data/`,
     WTC run/cancel/cache hit, `runtime.sqlite`, read-only `catalog.sqlite`, clean shutdown.
  2. Real headless Microsoft Edge (CDP) offline UI verification (all non-127.0.0.1 requests
     actively aborted), D3-geo 49-station SVG map + terminator + MLT=12, SVG/CSV export,
     and UI Exit button server shutdown.
  3. Frozen desktop `pywebview` (Edge WebView2) launch, `webview2_data` confinement inside
     `user_data_dir`, and clean `/api/shutdown`.
  4. Optional Windows Defender scan (`MpCmdRun.exe`) and raw dataset integrity check.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import websockets.sync.client as ws_sync

REPO_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = REPO_ROOT / "dist"
PORTABLE_ZIP = DIST_DIR / "GeoMagAnalyzer_Portable_Win64.zip"
RAW_DATASET_DIR = REPO_ROOT / "GeoMag_Storms_Dataset_Oct-2026"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _find_msedge() -> Path | None:
    candidates = [
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    ]
    which_edge = shutil.which("msedge")
    if which_edge:
        candidates.insert(0, Path(which_edge))
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def _stripped_env(local_app_data: Path, temp_root: Path) -> dict[str, str]:
    local_app_data.mkdir(parents=True, exist_ok=True)
    app_data = temp_root / "AppDataRoaming"
    user_prof = temp_root / "UserProfile"
    tmp_dir = temp_root / "Temp"
    for d in (app_data, user_prof, tmp_dir):
        d.mkdir(parents=True, exist_ok=True)

    return {
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", r"C:\Windows"),
        "WINDIR": os.environ.get("WINDIR", r"C:\Windows"),
        "COMSPEC": os.environ.get("COMSPEC", r"C:\Windows\System32\cmd.exe"),
        "PATHEXT": os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD"),
        "PATH": r"C:\Windows\System32;C:\Windows;C:\Windows\System32\Wbem",
        "TEMP": str(tmp_dir),
        "TMP": str(tmp_dir),
        "LOCALAPPDATA": str(local_app_data),
        "APPDATA": str(app_data),
        "USERPROFILE": str(user_prof),
        "HOME": str(user_prof),
    }


def _wait_ready_file(
    proc: subprocess.Popen[bytes], ready_file: Path, timeout: float = 40.0
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise AssertionError(f"Frozen process exited early with code {proc.returncode}")
        if ready_file.is_file():
            try:
                return json.loads(ready_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        time.sleep(0.15)
    raise AssertionError(f"Timed out waiting for ready-file {ready_file}")


def _list_processes_under(root_dir: Path) -> list[str]:
    norm_prefix = str(root_dir.resolve()).lower()
    ps_cmd = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.ExecutablePath } | "
        "ForEach-Object { \"$($_.ProcessId)|$($_.ExecutablePath)\" }"
    )
    res = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_cmd],
        capture_output=True,
        text=True,
        check=False,
    )
    matches: list[str] = []
    for line in res.stdout.splitlines():
        if "|" not in line:
            continue
        pid_str, exe_str = line.split("|", 1)
        if exe_str.strip().lower().startswith(norm_prefix):
            matches.append(f"{pid_str.strip()}:{exe_str.strip()}")
    return matches


def _assert_no_orphan_processes(root_dir: Path, timeout: float = 6.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        leftover = _list_processes_under(root_dir)
        if not leftover:
            return
        time.sleep(0.2)
    raise AssertionError(f"Orphan processes still running under {root_dir}: {leftover}")


class _CdpSession:
    """Minimal synchronous Chrome DevTools Protocol client with Fetch offline enforcement."""

    def __init__(self, ws_url: str) -> None:
        self._cm = ws_sync.connect(ws_url, max_size=32 * 1024 * 1024)
        self.ws = self._cm.__enter__()
        self._next_id = 1
        self.blocked_external_requests: list[str] = []
        self.js_exceptions: list[str] = []
        self.console_errors: list[str] = []

    def close(self) -> None:
        try:
            self._cm.__exit__(None, None, None)
        except Exception:
            pass

    def _handle_event(self, msg: dict[str, Any]) -> None:
        method = msg.get("method")
        params = msg.get("params", {})
        if method == "Fetch.requestPaused":
            req_id = params["requestId"]
            url = str(params.get("request", {}).get("url", ""))
            parsed = urlparse(url)
            is_local = (
                parsed.scheme in ("data", "blob", "about", "chrome", "edge")
                or (
                    parsed.scheme in ("http", "https")
                    and (parsed.hostname or "").lower() in ("127.0.0.1", "localhost", "::1")
                )
            )
            if is_local:
                self._send_no_wait("Fetch.continueRequest", {"requestId": req_id})
            else:
                self.blocked_external_requests.append(url)
                self._send_no_wait(
                    "Fetch.failRequest",
                    {"requestId": req_id, "errorReason": "InternetDisconnected"},
                )
        elif method == "Runtime.exceptionThrown":
            details = params.get("exceptionDetails", {})
            desc = details.get("exception", {}).get("description") or details.get("text")
            self.js_exceptions.append(str(desc))
        elif method == "Runtime.consoleAPICalled" and params.get("type") == "error":
            args = [str(a.get("value", a.get("description", ""))) for a in params.get("args", [])]
            self.console_errors.append(" ".join(args))

    def _send_no_wait(self, method: str, params: dict[str, Any] | None = None) -> int:
        msg_id = self._next_id
        self._next_id += 1
        payload: dict[str, Any] = {"id": msg_id, "method": method}
        if params is not None:
            payload["params"] = params
        self.ws.send(json.dumps(payload))
        return msg_id

    def call(self, method: str, params: dict[str, Any] | None = None, timeout: float = 15.0) -> Any:
        target_id = self._send_no_wait(method, params)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            raw = self.ws.recv(timeout=max(0.1, deadline - time.monotonic()))
            msg = json.loads(raw)
            if "id" in msg:
                if msg["id"] == target_id:
                    if "error" in msg:
                        raise RuntimeError(f"CDP {method} failed: {msg['error']}")
                    return msg.get("result", {})
                continue
            self._handle_event(msg)
        raise TimeoutError(f"CDP call {method} timed out")

    def pump(self, duration: float = 0.1) -> None:
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            rem = deadline - time.monotonic()
            try:
                raw = self.ws.recv(timeout=min(0.05, max(0.005, rem)))
            except TimeoutError:
                continue
            msg = json.loads(raw)
            if "id" not in msg:
                self._handle_event(msg)

    def eval_js(self, expr: str) -> Any:
        res = self.call(
            "Runtime.evaluate",
            {"expression": expr, "returnByValue": True, "awaitPromise": True},
        )
        if "exceptionDetails" in res:
            raise RuntimeError(f"JS eval exception: {res['exceptionDetails']}")
        return res.get("result", {}).get("value")


def _verify_api_wtc_and_cache(
    base_url: str,
    ready: dict[str, Any],
    expected_user_data: Path,
    catalog_path: Path,
) -> None:
    catalog_before = _sha256(catalog_path)
    port = int(ready["port"])
    assert Path(ready["user_data_dir"]).resolve() == expected_user_data.resolve()

    with httpx.Client(base_url=base_url, trust_env=False, timeout=20.0) as client:
        h = client.get("/api/health")
        assert h.status_code == 200
        assert h.json()["catalog_stations"] == 50
        assert h.json()["can_shutdown"] is True

        spa = client.get("/")
        assert spa.status_code == 200 and "GeoMag Analyzer" in spa.text

        cat = client.get("/api/catalog")
        assert cat.status_code == 200 and len(cat.json()["stations"]) == 50

        # Cancel mid-MC job
        sub_c = client.post(
            "/api/jobs/wtc",
            json={
                "episode_id": "storm_20261009_11",
                "station_i": "LVV",
                "station_j": "BEL",
                "component": "X",
                "coord_frame": "XYZ",
                "n_surrogates": 250,
                "seed": 77,
            },
        )
        assert sub_c.status_code == 200
        canc_jid = sub_c.json()["job_id"]
        dl = time.monotonic() + 15.0
        while time.monotonic() < dl:
            st = client.get(f"/api/jobs/{canc_jid}").json()
            if (st["status"] == "running" and st["completed_steps"] > 0) or (
                st["status"] == "completed"
            ):
                break
            time.sleep(0.1)
        assert client.delete(f"/api/jobs/{canc_jid}").status_code == 200
        dl2 = time.monotonic() + 10.0
        final_st = None
        while time.monotonic() < dl2:
            final_st = client.get(f"/api/jobs/{canc_jid}").json()["status"]
            if final_st in ("cancelled", "completed", "failed"):
                break
            time.sleep(0.1)
        assert final_st == "cancelled"
        assert not list((expected_user_data / "user_cache" / "cross_wtc").glob("*.tmp*"))

        # Complete WTC job via WS & verify instant cache hit
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

        frames = []
        with ws_sync.connect(
            f"ws://127.0.0.1:{port}/ws/jobs/{ok_jid}",
            additional_headers={"Origin": base_url},
        ) as ws:
            while True:
                msg = json.loads(ws.recv(timeout=25.0))
                frames.append(msg)
                if msg["status"] in ("completed", "cached", "failed", "cancelled"):
                    break
        assert frames[-1]["status"] == "completed"

        hit = client.post("/api/jobs/wtc", json=wtc_req)
        assert hit.status_code == 200 and hit.json()["cache_hit"] is True
        assert hit.json()["status"] == "cached"

        assert (expected_user_data / "runtime.sqlite").is_file()
        assert list((expected_user_data / "user_cache" / "cross_wtc").glob("*.npz"))
        assert _sha256(catalog_path) == catalog_before

        sd = client.post("/api/shutdown")
        assert sd.status_code == 200


def _verify_offline_edge_ui(extract_dir: Path, env: dict[str, str], edge_root: Path) -> None:
    edge_exe = _find_msedge()
    if edge_exe is None:
        raise AssertionError("msedge.exe not found; cannot run headless browser UI offline test.")

    ready_file = edge_root / "ready_ui.json"
    edge_ud = edge_root / "edge_profile"
    edge_downloads = edge_root / "downloads"
    edge_ud.mkdir(parents=True, exist_ok=True)
    edge_downloads.mkdir(parents=True, exist_ok=True)

    app_proc = subprocess.Popen(
        [
            str(extract_dir / "GeoMagAnalyzer.exe"),
            "--browser",
            "--no-browser-open",
            "--ready-file",
            str(ready_file),
        ],
        cwd=str(extract_dir),
        env=env,
    )

    cdp_port = _find_free_port()
    edge_proc = subprocess.Popen(
        [
            str(edge_exe),
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            f"--remote-debugging-port={cdp_port}",
            f"--user-data-dir={edge_ud}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    cdp: _CdpSession | None = None
    try:
        ready = _wait_ready_file(app_proc, ready_file, timeout=35.0)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        ws_url: str | None = None
        dl = time.monotonic() + 20.0
        while time.monotonic() < dl:
            try:
                with opener.open(f"http://127.0.0.1:{cdp_port}/json/list", timeout=1.5) as resp:
                    targets = json.loads(resp.read().decode("utf-8"))
                pages = [
                    t
                    for t in targets
                    if t.get("type") == "page" and "webSocketDebuggerUrl" in t
                ]
                if pages:
                    ws_url = str(pages[0]["webSocketDebuggerUrl"])
                    break
            except Exception:
                time.sleep(0.15)
        assert ws_url, "Edge CDP page target did not start"

        cdp = _CdpSession(ws_url)
        cdp.call("Page.enable")
        cdp.call("Runtime.enable")
        cdp.call("Fetch.enable", {"patterns": [{"urlPattern": "*", "requestStage": "Request"}]})
        try:
            cdp.call(
                "Browser.setDownloadBehavior",
                {"behavior": "allow", "downloadPath": str(edge_downloads)},
            )
        except Exception:
            cdp.call(
                "Page.setDownloadBehavior",
                {"behavior": "allow", "downloadPath": str(edge_downloads)},
            )

        cdp.call("Page.navigate", {"url": str(ready["url"])})

        # Wait until 49-station D3-geo map, terminator, MLT=12, and export toolbar render
        dom_ok = False
        dl_ui = time.monotonic() + 30.0
        while time.monotonic() < dl_ui:
            cdp.pump(0.15)
            state = cdp.eval_js(
                """(() => {
                    const sel = '[data-testid^="station-node-"]';
                    const stations = document.querySelectorAll(sel).length;
                    const term = !!document.querySelector('[data-testid="map-terminator"]');
                    const mlt12 = !!document.querySelector('[data-testid="map-mlt12"]');
                    const oulu = !!document.querySelector('[data-testid="station-marker-OULU"]');
                    const svgBtn = !!document.querySelector('button[title="Export vector SVG"]');
                    const csvBtn = !!document.querySelector('button[title="Export CSV"]');
                    return { stations, term, mlt12, oulu, svgBtn, csvBtn };
                })()"""
            )
            if (
                state
                and state["stations"] == 49
                and state["term"]
                and state["mlt12"]
                and state["oulu"]
                and state["svgBtn"]
                and state["csvBtn"]
            ):
                dom_ok = True
                break
        assert dom_ok, f"Offline D3-geo map did not render completely: {state}"

        # Trigger SVG and CSV export buttons from the map toolbar
        cdp.eval_js("document.querySelector('button[title=\"Export vector SVG\"]').click()")
        cdp.eval_js("document.querySelector('button[title=\"Export CSV\"]').click()")

        dl_exp = time.monotonic() + 10.0
        svg_files: list[Path] = []
        csv_files: list[Path] = []
        while time.monotonic() < dl_exp:
            cdp.pump(0.15)
            svg_files = list(edge_downloads.glob("*.svg"))
            csv_files = list(edge_downloads.glob("*.csv"))
            if svg_files and csv_files:
                break
        assert svg_files and csv_files, f"Expected .svg and .csv downloads in {edge_downloads}"
        svg_txt = svg_files[0].read_text(encoding="utf-8")
        csv_txt = csv_files[0].read_text(encoding="utf-8")
        assert "<svg" in svg_txt and "LVV" in svg_txt
        assert csv_txt.startswith("# ") and "iaga_code" in csv_txt and "LVV" in csv_txt

        # Trigger clean server shutdown via TopBar Exit button -> confirmation modal Exit
        cdp.eval_js("document.querySelector('button[title=\"Close GeoMag Analyzer\"]').click()")
        cdp.pump(0.2)
        cdp.eval_js(
            """(() => {
                const btn = document.querySelector('.modal-card button.btn-danger');
                if (btn) btn.click();
                return !!btn;
            })()"""
        )

        closed_ui = False
        dl_sd = time.monotonic() + 10.0
        while time.monotonic() < dl_sd:
            cdp.pump(0.1)
            if "GeoMag Analyzer Closed" in str(cdp.eval_js("document.body.innerText")):
                closed_ui = True
                break
        assert closed_ui, "UI did not transition to 'GeoMag Analyzer Closed' state"

        rc = app_proc.wait(timeout=12.0)
        assert rc == 0, f"Expected exit code 0 on UI Exit shutdown, got {rc}"
        _assert_no_orphan_processes(extract_dir)
        assert cdp.blocked_external_requests == [], (
            f"External network requests attempted: {cdp.blocked_external_requests}"
        )
        assert cdp.js_exceptions == [], f"Uncaught JS exceptions: {cdp.js_exceptions}"
    finally:
        if cdp is not None:
            cdp.close()
        if edge_proc.poll() is None:
            edge_proc.terminate()
            try:
                edge_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                edge_proc.kill()
        if app_proc.poll() is None:
            app_proc.kill()
            app_proc.wait(timeout=5.0)


def _verify_frozen_desktop_pywebview(
    extract_dir: Path, env: dict[str, str], run_root: Path
) -> None:
    ready_file = run_root / "ready_desktop.json"
    proc = subprocess.Popen(
        [str(extract_dir / "GeoMagAnalyzer.exe"), "--ready-file", str(ready_file)],
        cwd=str(extract_dir),
        env=env,
    )
    try:
        ready = _wait_ready_file(proc, ready_file, timeout=35.0)
        assert ready["mode"] == "desktop", f"Expected desktop mode, got {ready['mode']}"
        base_url = str(ready["base_url"])

        with httpx.Client(base_url=base_url, trust_env=False, timeout=10.0) as client:
            ud = Path(ready["user_data_dir"])
            wv_initialized = False
            dl_wv = time.monotonic() + 20.0
            while time.monotonic() < dl_wv:
                wv_dir = ud / "webview2_data"
                if (wv_dir / "EBWebView" / "Default").is_dir() or (
                    wv_dir.is_dir() and any(wv_dir.iterdir())
                ):
                    wv_initialized = True
                    break
                time.sleep(0.2)
            assert wv_initialized, (
                f"WebView2 did not initialize storage inside {ud / 'webview2_data'}"
            )

            log_path = ud / "logs" / "launcher.log"
            assert log_path.is_file()
            log_text = log_path.read_text(encoding="utf-8")
            assert "ERROR" not in log_text and "Traceback" not in log_text, log_text

            sd = client.post("/api/shutdown")
            assert sd.status_code == 200

        rc = proc.wait(timeout=15.0)
        assert rc == 0, f"Frozen desktop mode returned {rc}"
        _assert_no_orphan_processes(extract_dir)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)


def _scan_with_defender(targets: list[Path]) -> str:
    mp_candidates = [
        Path(r"C:\Program Files\Windows Defender\MpCmdRun.exe"),
        Path(r"C:\ProgramData\Microsoft\Windows Defender\Platform"),
    ]
    mp_exe: Path | None = None
    if mp_candidates[0].is_file():
        mp_exe = mp_candidates[0]
    elif mp_candidates[1].is_dir():
        versions = sorted(mp_candidates[1].glob("*/MpCmdRun.exe"))
        if versions:
            mp_exe = versions[-1]

    if mp_exe is None:
        return "MpCmdRun.exe not found (skipped)"

    for tgt in targets:
        res = subprocess.run(
            [str(mp_exe), "-Scan", "-ScanType", "3", "-DisableRemediation", "-File", str(tgt)],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            raise AssertionError(
                f"Windows Defender scan failed on {tgt} (rc={res.returncode}): "
                f"{res.stdout}\n{res.stderr}"
            )
    return f"Scanned {len(targets)} artifacts cleanly via {mp_exe}"


def _verify_raw_dataset_readonly() -> tuple[int, int]:
    files = [p for p in RAW_DATASET_DIR.rglob("*") if p.is_file()]
    total_bytes = sum(p.stat().st_size for p in files)
    assert len(files) == 483, f"Expected 483 raw files, got {len(files)}"
    assert total_bytes == 108_484_030, f"Expected 108484030 raw bytes, got {total_bytes}"
    return len(files), total_bytes


def main() -> int:
    if not PORTABLE_ZIP.is_file():
        raise FileNotFoundError(f"Missing {PORTABLE_ZIP}")

    with tempfile.TemporaryDirectory(prefix="geomag_p7_e2e_") as tmp_str:
        tmp = Path(tmp_str)

        # 1. Portable ZIP extraction into path with spaces + stripped env
        extract_root = tmp / "Portable Test Dir"
        extract_root.mkdir(parents=True)
        with zipfile.ZipFile(PORTABLE_ZIP, "r") as zf:
            zf.extractall(extract_root)

        portable_dir = extract_root / "GeoMagAnalyzer"
        assert (portable_dir / "GeoMagAnalyzer.exe").is_file()
        assert (portable_dir / "portable.mode").is_file()
        assert (portable_dir / "GeoMagAnalyzer_Browser.bat").is_file()
        assert (portable_dir / "README_PORTABLE.txt").is_file()
        port_catalog = portable_dir / "_internal" / "data" / "processed" / "catalog.sqlite"
        assert port_catalog.is_file()
        assert (portable_dir / "_internal" / "frontend" / "dist" / "index.html").is_file()

        unused_local_app_data = tmp / "PortableUnusedLocalAppData"
        env_port = _stripped_env(unused_local_app_data, tmp / "PortEnv")

        t0 = time.monotonic()
        ready_file_1 = tmp / "ready_portable.json"
        proc_port = subprocess.Popen(
            [
                str(portable_dir / "GeoMagAnalyzer.exe"),
                "--browser",
                "--no-browser-open",
                "--ready-file",
                str(ready_file_1),
            ],
            cwd=str(portable_dir),
            env=env_port,
        )
        try:
            ready_port = _wait_ready_file(proc_port, ready_file_1, timeout=35.0)
            cold_start_sec = time.monotonic() - t0
            _verify_api_wtc_and_cache(
                str(ready_port["base_url"]),
                ready_port,
                portable_dir / "user_data",
                port_catalog,
            )
            assert proc_port.wait(timeout=12.0) == 0
            assert not (unused_local_app_data / "GeoMagAnalyzer").exists()
            _assert_no_orphan_processes(portable_dir)
            print(f"[PASS] Portable ZIP API + WTC + cache + shutdown ({cold_start_sec:.2f}s start)")
        finally:
            if proc_port.poll() is None:
                proc_port.kill()
                proc_port.wait(timeout=5.0)

        # 2. Real offline headless Microsoft Edge UI test against Portable EXE
        _verify_offline_edge_ui(portable_dir, env_port, tmp / "EdgeOffline")
        print("[PASS] Headless Edge offline UI + 49-station D3-geo map + SVG/CSV export + UI Exit")

        # 3. Frozen Desktop pywebview (Edge WebView2) smoke test
        _verify_frozen_desktop_pywebview(portable_dir, env_port, tmp / "DesktopTest")
        print("[PASS] Frozen pywebview Edge WebView2 launch + storage isolation + /api/shutdown")

    defender_msg = _scan_with_defender([DIST_DIR / "GeoMagAnalyzer", PORTABLE_ZIP])
    print(f"[PASS] Windows Defender: {defender_msg}")

    n_files, n_bytes = _verify_raw_dataset_readonly()
    print(f"[PASS] Read-only raw dataset verified: {n_files} files, {n_bytes:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
