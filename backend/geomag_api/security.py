"""Local Host/Origin and Sec-Fetch-Site guard for GeoMag Analyzer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlparse

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


@dataclass
class AuthState:
    """Host allowlist state for local desktop/browser origin validation."""

    allowed_hosts: tuple[str, ...]


def _extract_hostname(host_header: str | None) -> str:
    if not host_header:
        return ""
    raw = host_header.strip().lower()
    if raw.startswith("["):
        end_br = raw.find("]")
        if end_br > 0:
            return raw[1:end_br]
        return ""
    return raw.split(":", 1)[0]


def _parse_host_port(host_header: str | None, default_port: int = 80) -> tuple[str, int]:
    if not host_header:
        return "", -1
    raw = host_header.strip().lower()
    if raw.startswith("["):
        end_br = raw.find("]")
        if end_br <= 0:
            return "", -1
        host = raw[1:end_br]
        rest = raw[end_br + 1 :]
        if rest.startswith(":"):
            try:
                return host, int(rest[1:])
            except ValueError:
                return "", -1
        return (host, default_port) if rest == "" else ("", -1)
    if ":" in raw:
        host, port_str = raw.rsplit(":", 1)
        try:
            return host, int(port_str)
        except ValueError:
            return "", -1
    return raw, default_port


def is_host_allowed(host_header: str | None, auth: AuthState) -> bool:
    """Validate HTTP/WS Host header against localhost allowlist."""
    if not host_header:
        return False
    hostname = _extract_hostname(host_header)
    return hostname in {h.lower() for h in auth.allowed_hosts}


def is_origin_allowed(
    origin_header: str | None,
    auth: AuthState,
    host_header: str | None = None,
) -> bool:
    """Validate Origin header including host:port match against Host when provided."""
    if origin_header is None or origin_header == "":
        return True
    parsed = urlparse(origin_header.strip())
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        return False
    orig_host = (parsed.hostname or "").lower()
    if not orig_host or orig_host not in {h.lower() for h in auth.allowed_hosts}:
        return False
    default_port = 443 if scheme == "https" else 80
    try:
        orig_port = parsed.port if parsed.port is not None else default_port
    except ValueError:
        return False
    if host_header:
        req_host, req_port = _parse_host_port(host_header, default_port=default_port)
        if not req_host or orig_host != req_host or orig_port != req_port:
            return False
    return True


def is_sec_fetch_site_allowed(headers: Mapping[str, str]) -> bool:
    """Block browser cross-site requests indicated by Sec-Fetch-Site."""
    lowered = {k.lower(): v for k, v in headers.items()}
    return lowered.get("sec-fetch-site", "").strip().lower() != "cross-site"


def verify_ws_origin(
    headers: Mapping[str, str],
    auth: AuthState,
) -> tuple[bool, int, str]:
    """Validate WebSocket Host, Origin, and Sec-Fetch-Site headers."""
    lowered = {k.lower(): v for k, v in headers.items()}
    host_hdr = lowered.get("host")
    if (
        not is_host_allowed(host_hdr, auth)
        or not is_origin_allowed(lowered.get("origin"), auth, host_header=host_hdr)
        or not is_sec_fetch_site_allowed(lowered)
    ):
        return False, 403, "Forbidden origin or host"
    return True, 200, "ok"


class HostOriginCSRFMiddleware:
    """Pure ASGI middleware enforcing localhost Host, Origin, and Sec-Fetch-Site checks."""

    def __init__(self, app: ASGIApp, auth: AuthState) -> None:
        self.app = app
        self.auth = auth

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        scope_type = scope["type"]
        if scope_type not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1")
            for k, v in scope.get("headers", [])
        }
        host_hdr = headers.get("host")
        if (
            not is_host_allowed(host_hdr, self.auth)
            or not is_origin_allowed(headers.get("origin"), self.auth, host_header=host_hdr)
            or not is_sec_fetch_site_allowed(headers)
        ):
            if scope_type == "http":
                resp = JSONResponse({"detail": "Forbidden origin or host"}, status_code=403)
                await resp(scope, receive, send)
            else:
                await send({"type": "websocket.close", "code": 1008})
            return

        await self.app(scope, receive, send)
