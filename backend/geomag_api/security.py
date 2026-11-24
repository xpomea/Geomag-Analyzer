"""Host/Origin validation, one-time bootstrap cookies, and X-Session-Token CSRF guard."""

from __future__ import annotations

import hmac
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlparse

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

SESSION_COOKIE_NAME = "geomag_session"


@dataclass
class AuthState:
    """In-memory session credentials and host/origin allowlist state."""

    bootstrap_token: str
    session_token: str
    session_cookie_value: str
    allowed_hosts: tuple[str, ...]
    allowed_origins: tuple[str, ...] = ()
    require_auth: bool = True
    bootstrap_consumed: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


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


def _parse_cookie_header(cookie_header: str | None) -> dict[str, str]:
    cookies: dict[str, str] = {}
    if not cookie_header:
        return cookies
    for part in cookie_header.split(";"):
        if "=" not in part:
            continue
        name, value = part.split("=", 1)
        cleaned_name = name.strip()
        if cleaned_name:
            cookies[cleaned_name] = value.strip()
    return cookies


def is_host_allowed(host_header: str | None, auth: AuthState) -> bool:
    """Validate HTTP/WS Host header against localhost/configured allowlist."""
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
    cleaned = origin_header.strip()
    if cleaned in auth.allowed_origins:
        return True
    parsed = urlparse(cleaned)
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


def consume_bootstrap_token(token: str | None, auth: AuthState) -> bool:
    """Atomically verify and consume the one-time bootstrap token."""
    if not token:
        return False
    with auth._lock:
        if auth.bootstrap_consumed:
            return False
        if not hmac.compare_digest(token, auth.bootstrap_token):
            return False
        auth.bootstrap_consumed = True
        return True


def verify_session_cookie(cookies: Mapping[str, str], auth: AuthState) -> bool:
    """Check whether geomag_session HttpOnly cookie matches active session secret."""
    cookie_val = cookies.get(SESSION_COOKIE_NAME)
    if not cookie_val:
        return False
    return hmac.compare_digest(cookie_val, auth.session_cookie_value)


def verify_session_header(headers: Mapping[str, str], auth: AuthState) -> bool:
    """Check whether X-Session-Token header matches active session token."""
    lowered = {k.lower(): v for k, v in headers.items()}
    tok = lowered.get("x-session-token")
    if not tok:
        return False
    return hmac.compare_digest(tok, auth.session_token)


def verify_ws_or_sse_auth(
    headers: Mapping[str, str],
    cookies: Mapping[str, str],
    auth: AuthState,
) -> tuple[bool, int, str]:
    """Authenticate WS or SSE connections compatible with browser EventSource/WebSocket APIs."""
    lowered = {k.lower(): v for k, v in headers.items()}
    host_hdr = lowered.get("host")
    if not is_host_allowed(host_hdr, auth) or not is_origin_allowed(
        lowered.get("origin"), auth, host_header=host_hdr
    ):
        return False, 403, "Forbidden origin or host"

    if not auth.require_auth:
        return True, 200, "ok"

    if verify_session_cookie(cookies, auth) or verify_session_header(lowered, auth):
        return True, 200, "ok"

    return False, 401, "Missing or invalid session credentials"


class HostOriginCSRFMiddleware:
    """Pure ASGI middleware enforcing Host/Origin checks and session/CSRF auth on /api/*."""

    _EXEMPT_GET_PATHS = frozenset({"/api/health", "/api/auth/bootstrap", "/api/auth/session"})

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
        host_ok = is_host_allowed(host_hdr, self.auth)
        origin_ok = is_origin_allowed(headers.get("origin"), self.auth, host_header=host_hdr)
        if not host_ok or not origin_ok:
            if scope_type == "http":
                resp = JSONResponse({"detail": "Forbidden origin or host"}, status_code=403)
                await resp(scope, receive, send)
            else:
                await send({"type": "websocket.close", "code": 1008})
            return

        if scope_type == "http":
            method = scope.get("method", "GET").upper()
            path = scope.get("path", "")
            if self.auth.require_auth and path.startswith("/api/"):
                if method in ("POST", "DELETE", "PUT", "PATCH"):
                    if not verify_session_header(headers, self.auth):
                        resp = JSONResponse(
                            {"detail": "Missing or invalid X-Session-Token"},
                            status_code=401,
                        )
                        await resp(scope, receive, send)
                        return
                elif method in ("GET", "HEAD") and path not in self._EXEMPT_GET_PATHS:
                    cookies = _parse_cookie_header(headers.get("cookie"))
                    if not (
                        verify_session_cookie(cookies, self.auth)
                        or verify_session_header(headers, self.auth)
                    ):
                        resp = JSONResponse(
                            {"detail": "Missing or invalid session credentials"},
                            status_code=401,
                        )
                        await resp(scope, receive, send)
                        return

        await self.app(scope, receive, send)
