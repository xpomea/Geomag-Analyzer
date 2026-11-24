"""Shared fixtures for Phase 4 FastAPI integration tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from geomag_api import ApiSettings, create_app
from geomag_etl.config import DEFAULT_PROCESSED_DIR


@pytest.fixture
def make_test_client(tmp_path: Path) -> Callable[..., tuple[TestClient, ApiSettings]]:
    """Create a TestClient backed by isolated user_data and static_cache temp dirs."""
    created_clients: list[TestClient] = []

    def _factory(**overrides) -> tuple[TestClient, ApiSettings]:
        static_cache = Path(overrides.pop("static_cache_dir", tmp_path / "static_cache"))
        user_data = Path(overrides.pop("user_data_dir", tmp_path / "user_data"))
        static_cache.mkdir(parents=True, exist_ok=True)
        user_data.mkdir(parents=True, exist_ok=True)

        settings = ApiSettings(
            processed_dir=overrides.pop("processed_dir", DEFAULT_PROCESSED_DIR),
            static_cache_dir=static_cache,
            user_data_dir=user_data,
            max_workers=overrides.pop("max_workers", 2),
            max_mc_surrogates=overrides.pop("max_mc_surrogates", 1000),
            **overrides,
        )
        app = create_app(settings)
        client = TestClient(app)
        client.__enter__()
        created_clients.append(client)
        return client, settings

    yield _factory  # type: ignore[misc]

    for c in reversed(created_clients):
        c.__exit__(None, None, None)


@pytest.fixture
def api_client(make_test_client) -> tuple[TestClient, dict[str, str], ApiSettings]:
    """Authenticated TestClient with X-Session-Token headers and one-time bootstrap cookie."""
    client, settings = make_test_client()
    boot = client.get(f"/api/auth/bootstrap?token={settings.bootstrap_token}")
    assert boot.status_code == 200
    headers = {"X-Session-Token": boot.json()["session_token"]}
    return client, headers, settings
