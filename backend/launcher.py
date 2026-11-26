"""Desktop (pywebview Edge WebView2) and browser-fallback launcher for GeoMag Analyzer.

IMPORTANT:
- Top-level imports MUST stay stdlib-only so PyInstaller Windows `spawn` workers
  exit immediately inside `multiprocessing.freeze_support()` without importing
  FastAPI, NumPy/SciPy, or pywebview.
- Under PyInstaller `console=False`, `sys.stdout` and `sys.stderr` may be `None`.
  Never pass default uvicorn logging config or attach unguarded StreamHandlers.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import multiprocessing
import os
import signal
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    import winreg
else:
    winreg = None  # type: ignore[assignment]

_BACKEND_DIR = Path(__file__).resolve().parent
_WEBVIEW2_MIN_VERSION = (86, 0, 622, 0)
_DOTNET_462_MIN_RELEASE = 394802
_WEBVIEW2_CLIENT_GUIDS = (
    "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}",  # WebView2 Runtime
    "{2CD8A007-E189-409D-A2C8-9AF4EF3C72AA}",  # Beta
    "{0D50BFEC-CD6A-4F9A-964C-C7416E3ACB10}",  # Dev
    "{65C35B14-6C1D-4122-AC46-7148CC9D6497}",  # Canary
)
_BLAS_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _ensure_backend_on_path() -> None:
    if not getattr(sys, "frozen", False):
        backend_str = str(_BACKEND_DIR)
        if backend_str not in sys.path:
            sys.path.insert(0, backend_str)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse desktop/browser launcher CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="GeoMagAnalyzer",
        description="Launch GeoMag Analyzer (Edge WebView2 desktop window or system browser).",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Force launch in the default system web browser instead of pywebview.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host interface (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Preferred TCP port (0 picks a free ephemeral port).",
    )
    parser.add_argument(
        "--user-data-dir",
        type=Path,
        default=None,
        help="Override writable directory for runtime.sqlite, user_cache/, and logs.",
    )
    parser.add_argument(
        "--portable",
        action="store_true",
        help="Prefer ./user_data beside the application executable when writable.",
    )
    parser.add_argument(
        "--no-browser-open",
        action="store_true",
        help="Do not invoke webbrowser.open() in browser mode (useful for automated tests).",
    )
    parser.add_argument(
        "--ready-file",
        type=Path,
        default=None,
        help="Atomically write server readiness JSON metadata to this path.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging and pywebview developer tools.",
    )
    return parser.parse_args(argv)


def find_free_port(host: str = "127.0.0.1", preferred: int = 0) -> int:
    """Return `preferred` if bindable on `host`, otherwise an OS-assigned free TCP port."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    if preferred > 0:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, int(preferred)))
                return int(sock.getsockname()[1])
            except OSError:
                pass
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _parse_version_tuple(raw: str | None) -> tuple[int, ...] | None:
    if not raw:
        return None
    parts: list[int] = []
    for piece in str(raw).strip().split("."):
        if not piece.isdigit():
            return None
        parts.append(int(piece))
    if not parts:
        return None
    while len(parts) < 4:
        parts.append(0)
    return tuple(parts[:4])


def is_webview2_available() -> bool:
    """Return True when pywebview, .NET >= 4.6.2, and Edge WebView2 Runtime are present."""
    if sys.platform != "win32" or importlib.util.find_spec("webview") is None:
        return False

    runtime_override = os.environ.get("WEBVIEW2_RUNTIME_PATH", "").strip()
    if runtime_override and Path(runtime_override).exists():
        return True

    if winreg is None:
        return False

    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full",
        ) as net_key:
            release, _ = winreg.QueryValueEx(net_key, "Release")
        if int(release) < _DOTNET_462_MIN_RELEASE:
            return False
    except (OSError, ValueError, TypeError):
        return False

    reg_roots = (
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\EdgeUpdate\Clients"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\EdgeUpdate\Clients"),
    )
    for guid in _WEBVIEW2_CLIENT_GUIDS:
        for root, prefix in reg_roots:
            try:
                with winreg.OpenKey(root, f"{prefix}\\{guid}") as key:
                    pv, _ = winreg.QueryValueEx(key, "pv")
                ver = _parse_version_tuple(str(pv))
                if ver is not None and ver >= _WEBVIEW2_MIN_VERSION:
                    return True
            except (OSError, ValueError, TypeError):
                continue
    return False


def select_launch_mode(force_browser: bool, webview2_ok: bool | None = None) -> str:
    """Choose 'desktop' (pywebview) or 'browser' fallback mode."""
    if force_browser:
        return "browser"
    ok = is_webview2_available() if webview2_ok is None else bool(webview2_ok)
    return "desktop" if ok else "browser"


def configure_logging(user_data_dir: Path, debug: bool = False) -> Path:
    """Configure UTF-8 file logging (and stderr only when console stream exists)."""
    log_dir = Path(user_data_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "launcher.log"

    level = logging.DEBUG if debug else logging.INFO
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_geomag_launcher_handler", False):
            root.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    file_handler._geomag_launcher_handler = True  # type: ignore[attr-defined]
    root.addHandler(file_handler)

    if sys.stderr is not None:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        stream_handler._geomag_launcher_handler = True  # type: ignore[attr-defined]
        root.addHandler(stream_handler)

    if root.level == logging.NOTSET or root.level > level:
        root.setLevel(level)

    return log_path


def resolve_runtime_settings(
    args: argparse.Namespace,
    *,
    host: str,
    shutdown_callback: Callable[[], None],
) -> Any:
    """Build ApiSettings honoring --user-data-dir, --portable, and desktop shutdown."""
    from geomag_api.config import ApiSettings, _is_writable_target, get_app_dir

    settings = ApiSettings.from_env()
    if args.user_data_dir is not None:
        settings.user_data_dir = Path(args.user_data_dir)
    elif getattr(args, "portable", False):
        portable_dir = get_app_dir() / "user_data"
        if _is_writable_target(portable_dir):
            settings.user_data_dir = portable_dir

    assert settings.user_data_dir is not None
    settings.user_data_dir.mkdir(parents=True, exist_ok=True)

    merged_hosts: list[str] = []
    seen: set[str] = set()
    for candidate in (host, "127.0.0.1", "localhost", "::1", *settings.allowed_hosts):
        cleaned = str(candidate).strip()
        lowered = cleaned.lower()
        if cleaned and lowered not in seen:
            seen.add(lowered)
            merged_hosts.append(cleaned)

    settings.allowed_hosts = tuple(merged_hosts)
    settings.shutdown_callback = shutdown_callback
    return settings


def resolve_icon_path() -> Path | None:
    """Locate packaged or repository .ico asset if present."""
    from geomag_api.config import get_app_dir, get_bundle_root

    candidates = (
        get_bundle_root() / "assets" / "geomag_analyzer.ico",
        get_app_dir() / "packaging" / "geomag_analyzer.ico",
        _BACKEND_DIR.parent / "packaging" / "geomag_analyzer.ico",
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def wait_for_server_ready(
    health_url: str,
    server_thread: threading.Thread,
    timeout_sec: float = 30.0,
) -> bool:
    """Poll local health URL (bypassing any system HTTP proxy) until HTTP 200 or failure."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if not server_thread.is_alive():
            return False
        try:
            with opener.open(health_url, timeout=1.0) as resp:
                if int(resp.status) == 200:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(0.15)
    return False


def write_ready_file(ready_file: Path | None, payload: dict[str, Any]) -> None:
    """Atomically persist launch metadata JSON for test harnesses or parent scripts."""
    if ready_file is None:
        return
    target = Path(ready_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp_path, target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def run_browser_fallback(
    *,
    app_url: str,
    ready_payload: dict[str, Any],
    ready_file: Path | None,
    open_browser: bool,
    shutdown_event: threading.Event,
    server_thread: threading.Thread,
    trigger_shutdown: Callable[[], None],
) -> int:
    """Open the UI in the default system browser and block until UI Exit or signal shutdown."""
    write_ready_file(ready_file, {**ready_payload, "mode": "browser"})
    if open_browser:
        webbrowser.open(app_url)

    try:
        while not shutdown_event.wait(0.25):
            if not server_thread.is_alive():
                break
    except KeyboardInterrupt:
        trigger_shutdown()
    return 0


def _install_signal_handlers(trigger_shutdown: Callable[[], None]) -> None:
    if threading.current_thread() is not threading.main_thread():
        return

    def _handle(_signum: int, _frame: Any) -> None:
        trigger_shutdown()

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            try:
                signal.signal(sig, _handle)
            except (ValueError, OSError):
                pass


def main(argv: list[str] | None = None) -> int:
    """Launch embedded FastAPI server and open desktop pywebview or browser fallback."""
    _ensure_backend_on_path()
    for env_var in _BLAS_THREAD_VARS:
        os.environ.setdefault(env_var, "1")

    args = parse_args(argv)
    import uvicorn

    from geomag_api.main import create_app

    shutdown_event = threading.Event()
    shutdown_requested = {"flag": False}
    window_ref: dict[str, Any] = {"window": None}
    server_ref: dict[str, Any] = {"server": None}

    def trigger_shutdown() -> None:
        shutdown_requested["flag"] = True
        shutdown_event.set()
        srv = server_ref.get("server")
        if srv is not None:
            srv.should_exit = True
        win = window_ref.get("window")
        if win is not None:

            def _close_win() -> None:
                try:
                    win.destroy()
                except Exception:
                    pass

            timer = threading.Timer(0.25, _close_win)
            timer.daemon = True
            timer.start()

    settings = resolve_runtime_settings(
        args,
        host=args.host,
        shutdown_callback=trigger_shutdown,
    )
    assert settings.user_data_dir is not None
    configure_logging(settings.user_data_dir, debug=args.debug)
    logger = logging.getLogger("geomag_launcher")

    port = find_free_port(args.host, preferred=int(args.port))
    display_host = (
        f"[{args.host}]" if ":" in args.host and not args.host.startswith("[") else args.host
    )
    base_url = f"http://{display_host}:{port}"
    app_url = f"{base_url}/"
    health_url = f"{base_url}/api/health"

    app = create_app(settings)

    class _ThreadedUvicornServer(uvicorn.Server):
        def install_signal_handlers(self) -> None:
            return

    uv_config = uvicorn.Config(
        app=app,
        host=args.host,
        port=port,
        log_config=None,
        access_log=bool(args.debug),
        timeout_graceful_shutdown=5,
        ws="websockets",
        lifespan="on",
    )
    server = _ThreadedUvicornServer(uv_config)
    server_ref["server"] = server

    server_thread = threading.Thread(
        target=server.run,
        name="geomag-uvicorn",
        daemon=True,
    )
    _install_signal_handlers(trigger_shutdown)
    server_thread.start()

    mode = select_launch_mode(force_browser=bool(args.browser))
    icon_path = resolve_icon_path()
    ready_payload: dict[str, Any] = {
        "url": app_url,
        "base_url": base_url,
        "host": args.host,
        "port": port,
        "mode": mode,
        "pid": os.getpid(),
        "user_data_dir": str(settings.user_data_dir),
    }

    try:
        if not wait_for_server_ready(health_url, server_thread, timeout_sec=30.0):
            logger.error("Embedded FastAPI server failed to become ready at %s", health_url)
            return 1

        logger.info("GeoMag Analyzer ready at %s (mode=%s)", base_url, mode)

        if mode == "desktop":
            try:
                import webview

                webview.settings["ALLOW_DOWNLOADS"] = True
                webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True

                storage_dir = settings.user_data_dir / "webview2_data"
                storage_dir.mkdir(parents=True, exist_ok=True)

                window = webview.create_window(
                    "GeoMag Analyzer — Pc5/Pc6 Geomagnetic Pulsation Analysis",
                    url=app_url,
                    width=1440,
                    height=900,
                    min_size=(1024, 680),
                    background_color="#0b1120",
                    text_select=True,
                    zoomable=True,
                )
                window_ref["window"] = window
                write_ready_file(args.ready_file, ready_payload)

                start_kwargs: dict[str, Any] = {
                    "gui": "edgechromium",
                    "debug": bool(args.debug),
                    "private_mode": False,
                    "storage_path": str(storage_dir),
                }
                if icon_path is not None:
                    start_kwargs["icon"] = str(icon_path)
                webview.start(**start_kwargs)
                return 0
            except Exception as exc:
                if shutdown_requested["flag"]:
                    return 0
                logger.exception(
                    "pywebview failed at runtime (%s); falling back to system browser", exc
                )
                window_ref["window"] = None
                return run_browser_fallback(
                    app_url=app_url,
                    ready_payload=ready_payload,
                    ready_file=args.ready_file,
                    open_browser=not bool(args.no_browser_open),
                    shutdown_event=shutdown_event,
                    server_thread=server_thread,
                    trigger_shutdown=trigger_shutdown,
                )

        return run_browser_fallback(
            app_url=app_url,
            ready_payload=ready_payload,
            ready_file=args.ready_file,
            open_browser=not bool(args.no_browser_open),
            shutdown_event=shutdown_event,
            server_thread=server_thread,
            trigger_shutdown=trigger_shutdown,
        )
    finally:
        server.should_exit = True
        if server_thread.is_alive():
            server_thread.join(timeout=15.0)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
