"""Unit tests for Phase 7 launcher, icon generator, offline bundle, and packaging specs."""

from __future__ import annotations

import importlib.util
import json
import logging
import re
import socket
import struct
import subprocess
import sys
import threading
import types
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import launcher

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGING_DIR = REPO_ROOT / "packaging"
FRONTEND_DIST_DIR = REPO_ROOT / "frontend" / "dist"


def test_launcher_top_level_imports_are_stdlib_only():
    code = "\n".join(
        [
            "import sys",
            "sys.path.insert(0, 'backend')",
            "import launcher",
            (
                "for mod in ('fastapi', 'uvicorn', 'webview', 'numpy', "
                "'scipy', 'geomag_api', 'geomag_core'):"
            ),
            "    assert mod not in sys.modules, f'Unexpected top-level import: {mod}'",
            "print('OK')",
        ]
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "OK" in res.stdout


def test_find_free_port_ephemeral_and_preferred_fallback():
    ephemeral = launcher.find_free_port("127.0.0.1")
    assert 1 <= ephemeral <= 65535

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        busy_port = int(occupied.getsockname()[1])
        fallback = launcher.find_free_port("127.0.0.1", preferred=busy_port)
        assert fallback != busy_port
        assert 1 <= fallback <= 65535


def test_select_launch_mode_decisions():
    assert launcher.select_launch_mode(force_browser=True, webview2_ok=True) == "browser"
    assert launcher.select_launch_mode(force_browser=False, webview2_ok=False) == "browser"
    assert launcher.select_launch_mode(force_browser=False, webview2_ok=True) == "desktop"


def test_is_webview2_available_mocked_registry(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(launcher.sys, "platform", "linux")
    assert launcher.is_webview2_available() is False

    monkeypatch.setattr(launcher.sys, "platform", "win32")
    monkeypatch.setattr(launcher.importlib.util, "find_spec", lambda name: None)
    assert launcher.is_webview2_available() is False

    monkeypatch.setattr(
        launcher.importlib.util,
        "find_spec",
        lambda name: object() if name == "webview" else None,
    )

    # Explicit WEBVIEW2_RUNTIME_PATH override
    rt_dir = tmp_path / "webview2_rt"
    rt_dir.mkdir()
    monkeypatch.setenv("WEBVIEW2_RUNTIME_PATH", str(rt_dir))
    assert launcher.is_webview2_available() is True
    monkeypatch.delenv("WEBVIEW2_RUNTIME_PATH")

    class _FakeKey:
        def __init__(self, values: dict[str, object]) -> None:
            self.values = values

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def _make_winreg(net_release: int, client_pv: str | None):
        client_sub = r"SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
        net_sub = r"SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full"

        def open_key(root, subkey):
            if subkey == net_sub:
                return _FakeKey({"Release": net_release})
            if root == 1 and subkey == client_sub and client_pv is not None:
                return _FakeKey({"pv": client_pv})
            raise OSError("not found")

        def query_value_ex(key_obj, name):
            if name not in key_obj.values:
                raise OSError("missing value")
            return key_obj.values[name], 1

        return SimpleNamespace(
            HKEY_CURRENT_USER=1,
            HKEY_LOCAL_MACHINE=2,
            OpenKey=open_key,
            QueryValueEx=query_value_ex,
        )

    monkeypatch.setattr(launcher, "winreg", _make_winreg(394802, "158.0.4370.33"))
    assert launcher.is_webview2_available() is True

    monkeypatch.setattr(launcher, "winreg", _make_winreg(378389, "158.0.4370.33"))
    assert launcher.is_webview2_available() is False

    monkeypatch.setattr(launcher, "winreg", _make_winreg(394802, "0.0.0.0"))
    assert launcher.is_webview2_available() is False

    monkeypatch.setattr(launcher, "winreg", _make_winreg(394802, None))
    assert launcher.is_webview2_available() is False


def test_configure_logging_handles_none_streams_console_false(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)

    log_path = launcher.configure_logging(tmp_path, debug=True)
    try:
        logging.getLogger("geomag_launcher.test").info("console-false-log-check")
        for h in logging.getLogger().handlers:
            h.flush()
        content = log_path.read_text(encoding="utf-8")
        assert "console-false-log-check" in content
    finally:
        root = logging.getLogger()
        for h in list(root.handlers):
            if getattr(h, "_geomag_launcher_handler", False):
                root.removeHandler(h)
                h.close()


def test_resolve_runtime_settings_portable_and_explicit_dir(monkeypatch, tmp_path: Path):
    import geomag_api.config as api_cfg

    fake_app_dir = tmp_path / "portable_app"
    fake_app_dir.mkdir()
    monkeypatch.setattr(api_cfg, "get_app_dir", lambda: fake_app_dir)

    args_portable = launcher.parse_args(["--portable", "--host", "127.0.0.1"])
    s_port = launcher.resolve_runtime_settings(
        args_portable,
        host="127.0.0.1",
        shutdown_callback=lambda: None,
    )
    assert s_port.user_data_dir == fake_app_dir / "user_data"
    assert s_port.require_auth is True
    assert s_port.shutdown_callback is not None
    assert "127.0.0.1" in s_port.allowed_hosts
    assert "localhost" in s_port.allowed_hosts

    explicit_ud = tmp_path / "custom_ud"
    args_custom = launcher.parse_args(["--user-data-dir", str(explicit_ud), "--portable"])
    s_custom = launcher.resolve_runtime_settings(
        args_custom,
        host="127.0.0.1",
        shutdown_callback=lambda: None,
    )
    assert s_custom.user_data_dir == explicit_ud


def test_launcher_runtime_fallback_when_pywebview_fails(monkeypatch, tmp_path: Path):
    ready_file = tmp_path / "ready.json"
    user_data = tmp_path / "ud"

    fake_webview = types.ModuleType("webview")
    fake_webview.settings = {"ALLOW_DOWNLOADS": False, "OPEN_EXTERNAL_LINKS_IN_BROWSER": True}

    def _boom_create_window(*_args, **_kwargs):
        raise RuntimeError("Simulated missing WebView2 loader")

    fake_webview.create_window = _boom_create_window  # type: ignore[attr-defined]
    fake_webview.start = lambda **_kwargs: None  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "webview", fake_webview)
    monkeypatch.setattr(launcher, "select_launch_mode", lambda force_browser=False: "desktop")

    stop_result: dict[str, object] = {}

    def _shutdown_driver() -> None:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = 15.0
        elapsed = 0.0
        import time

        while elapsed < deadline:
            if ready_file.is_file():
                try:
                    data = json.loads(ready_file.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    data = {}
                if data.get("mode") == "browser":
                    stop_result.update(data)
                    req = urllib.request.Request(
                        f"{data['base_url']}/api/shutdown",
                        method="POST",
                        headers={"X-Session-Token": str(data["session_token"])},
                    )
                    with opener.open(req, timeout=5.0) as resp:
                        stop_result["shutdown_status"] = int(resp.status)
                    return
            time.sleep(0.1)
            elapsed += 0.1

    driver = threading.Thread(target=_shutdown_driver, daemon=True)
    driver.start()
    try:
        rc = launcher.main(
            [
                "--user-data-dir",
                str(user_data),
                "--no-browser-open",
                "--no-tray",
                "--ready-file",
                str(ready_file),
            ]
        )
        assert rc == 0
        driver.join(timeout=5.0)
        assert stop_result.get("mode") == "browser"
        assert stop_result.get("shutdown_status") == 200
        assert fake_webview.settings["ALLOW_DOWNLOADS"] is True
    finally:
        root = logging.getLogger()
        for h in list(root.handlers):
            if getattr(h, "_geomag_launcher_handler", False):
                root.removeHandler(h)
                h.close()


def test_generate_icon_produces_valid_multi_res_ico(tmp_path: Path):
    spec = importlib.util.spec_from_file_location(
        "generate_icon",
        PACKAGING_DIR / "generate_icon.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    out = mod.generate_icon_file(tmp_path / "test.ico")
    raw = out.read_bytes()
    committed = (PACKAGING_DIR / "geomag_analyzer.ico").read_bytes()
    assert raw == committed

    reserved, img_type, count = struct.unpack("<HHH", raw[:6])
    assert (reserved, img_type, count) == (0, 1, 4)

    sizes_seen: list[int] = []
    for idx in range(count):
        off = 6 + idx * 16
        w_b, h_b, _cc, _res, planes, bpp, byte_len, data_off = struct.unpack(
            "<BBBBHHII", raw[off : off + 16]
        )
        sz = 256 if w_b == 0 else w_b
        assert h_b == w_b
        assert (planes, bpp) == (1, 32)
        payload = raw[data_off : data_off + byte_len]
        if sz == 256:
            assert payload.startswith(b"\x89PNG\r\n\x1a\n")
        else:
            bi_size, bi_w, bi_h = struct.unpack("<III", payload[:12])
            assert (bi_size, bi_w, bi_h) == (40, sz, sz * 2)
        sizes_seen.append(sz)
    assert sizes_seen == [16, 32, 48, 256]


def test_packaging_specs_and_web_configs_meet_phase7_contracts():
    spec_text = (PACKAGING_DIR / "geomag_analyzer.spec").read_text(encoding="utf-8")
    assert "exclude_binaries=True" in spec_text
    assert "COLLECT(" in spec_text
    assert "console=False" in spec_text
    assert "upx=False" in spec_text
    assert "PROCESSED_DIR" in spec_text
    assert "FRONTEND_DIST_DIR" in spec_text
    assert "geomag_analyzer.ico" in spec_text

    iss_text = (PACKAGING_DIR / "installer.iss").read_text(encoding="utf-8")
    assert "PrivilegesRequired=lowest" in iss_text
    assert "ArchitecturesAllowed=x64compatible" in iss_text
    assert r"DefaultDirName={localappdata}\Programs\GeoMagAnalyzer" in iss_text
    assert "desktopicon" in iss_text
    assert "OutputBaseFilename=GeoMagAnalyzer_Setup" in iss_text
    assert "portable.mode" not in iss_text

    nginx_text = (PACKAGING_DIR / "web" / "nginx.conf").read_text(encoding="utf-8")
    assert "--workers 1" in nginx_text
    assert "proxy_http_version 1.1" in nginx_text
    assert "Upgrade $http_upgrade" in nginx_text
    assert "proxy_buffering off" in nginx_text

    caddy_text = (PACKAGING_DIR / "web" / "Caddyfile").read_text(encoding="utf-8")
    assert "--workers 1" in caddy_text
    assert "flush_interval -1" in caddy_text
    assert "/api/jobs/*/events" in caddy_text

    svc_text = (PACKAGING_DIR / "web" / "geomag-analyzer.service").read_text(encoding="utf-8")
    assert "--workers 1" in svc_text
    assert "uvicorn geomag_api.main:app" in svc_text


def test_frontend_dist_is_100_percent_offline_with_world_topojson():
    index_html = FRONTEND_DIST_DIR / "index.html"
    assert index_html.is_file()

    forbidden_hosts = (
        "fonts.googleapis.com",
        "fonts.gstatic.com",
        "unpkg.com",
        "jsdelivr.net",
        "cdnjs.cloudflare.com",
        "openstreetmap.org",
        "cartocdn.com",
        "mapbox.com",
    )
    allowed_url_prefixes = (
        "http://www.w3.org/",
        "https://github.com/ecomfe/zrender/",
        "https://reactjs.org/docs/error-decoder.html",
    )
    url_re = re.compile(r'https?://[^\s"\'<>)]+')
    has_topo = False

    for path in FRONTEND_DIST_DIR.rglob("*"):
        if path.suffix not in (".html", ".js", ".css"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for host in forbidden_hosts:
            assert host not in text, f"Unexpected external CDN/tile host {host} in {path}"
        for url in url_re.findall(text):
            assert url.startswith(allowed_url_prefixes), f"Unexpected URL {url} in {path}"
        if path.suffix == ".js" and "Topology" in text and "countries" in text:
            has_topo = True

    assert has_topo, "Embedded world-110m.topojson geometry missing from frontend/dist/"
    assert list((FRONTEND_DIST_DIR / "assets").glob("*.woff2")), "Local woff2 fonts missing"
