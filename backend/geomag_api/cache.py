"""Unified SHA-256 NPZ disk cache and runtime.sqlite index management."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from geomag_core import Phase1StationSpectra, load_phase1_cache, save_phase1_cache

_VALID_CATEGORIES = ("spectra_phase1", "cross_wtc")

_RUNTIME_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    job_type TEXT NOT NULL,
    cache_key TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL,
    progress_pct REAL NOT NULL,
    completed_steps INTEGER NOT NULL,
    total_steps INTEGER NOT NULL,
    eta_sec REAL,
    cache_hit INTEGER NOT NULL DEFAULT 0,
    result_path TEXT,
    error_message TEXT,
    params_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_cache_key ON jobs (cache_key);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (status);

CREATE TABLE IF NOT EXISTS cache_index (
    cache_key TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    file_path TEXT NOT NULL,
    source TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    params_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cache_category ON cache_index (category);
"""


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_cache_args(category: str, cache_key: str) -> None:
    if category not in _VALID_CATEGORIES:
        raise ValueError(f"Unsupported cache category {category!r}")
    if len(cache_key) != 64 or not all(c in "0123456789abcdef" for c in cache_key):
        raise ValueError(f"Invalid SHA-256 cache_key {cache_key!r}")


class CacheManager:
    """Manages user (read-write) NPZ caches plus runtime.sqlite index."""

    def __init__(
        self,
        *,
        user_cache_dir: Path,
        runtime_db_path: Path,
    ) -> None:
        self.user_cache_dir = Path(user_cache_dir)
        self.runtime_db_path = Path(runtime_db_path)
        self._lock = threading.RLock()
        self._key_locks: dict[str, threading.RLock] = {}

        self.runtime_db_path.parent.mkdir(parents=True, exist_ok=True)
        for cat in _VALID_CATEGORIES:
            (self.user_cache_dir / cat).mkdir(parents=True, exist_ok=True)
        self._init_runtime_db()

    def _lock_for_key(self, cache_key: str) -> threading.RLock:
        with self._lock:
            lock = self._key_locks.get(cache_key)
            if lock is None:
                lock = threading.RLock()
                self._key_locks[cache_key] = lock
            return lock

    def connect_runtime(self) -> sqlite3.Connection:
        """Open a short-lived WAL-mode SQLite connection to runtime.sqlite."""
        conn = sqlite3.connect(str(self.runtime_db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_runtime_db(self) -> None:
        with self._lock:
            conn = self.connect_runtime()
            try:
                conn.executescript(_RUNTIME_SCHEMA_SQL)
                conn.commit()
            finally:
                conn.close()

    def user_path_for(self, category: str, cache_key: str) -> Path:
        _validate_cache_args(category, cache_key)
        return self.user_cache_dir / category / f"{cache_key}.npz"

    def find(self, category: str, cache_key: str) -> Path | None:
        """Look up cached NPZ file in user cache."""
        _validate_cache_args(category, cache_key)
        user_path = self.user_cache_dir / category / f"{cache_key}.npz"
        if user_path.is_file():
            return user_path
        return None

    def register_entry(
        self,
        *,
        cache_key: str,
        category: str,
        file_path: Path,
        source: str,
        params: dict[str, Any],
    ) -> None:
        """Record or update an entry in runtime.sqlite:cache_index."""
        _validate_cache_args(category, cache_key)
        size_bytes = int(file_path.stat().st_size) if file_path.exists() else 0
        now = utc_now_iso()
        params_json = json.dumps(params, sort_keys=True, ensure_ascii=True)
        with self._lock:
            conn = self.connect_runtime()
            try:
                conn.execute(
                    """
                    INSERT INTO cache_index (
                        cache_key, category, file_path, source, size_bytes, params_json, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        file_path = excluded.file_path,
                        source = excluded.source,
                        size_bytes = excluded.size_bytes,
                        params_json = excluded.params_json
                    """,
                    (
                        cache_key,
                        category,
                        str(file_path),
                        source,
                        size_bytes,
                        params_json,
                        now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def get_or_compute_phase1(
        self,
        cache_key: str,
        params: dict[str, Any],
        compute_fn: Any,
    ) -> tuple[Phase1StationSpectra, bool]:
        """Load Phase 1 station spectra from cache or compute and store atomically."""
        with self._lock_for_key(cache_key):
            path = self.find("spectra_phase1", cache_key)
            if path is not None:
                spectra = load_phase1_cache(path)
                if spectra.cache_key == cache_key:
                    self.register_entry(
                        cache_key=cache_key,
                        category="spectra_phase1",
                        file_path=path,
                        source="user",
                        params=params,
                    )
                    return spectra, True

            spectra = compute_fn()
            target = self.user_path_for("spectra_phase1", cache_key)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = target.with_name(
                f".{cache_key}.{os.getpid()}.{threading.get_ident()}.tmp.npz"
            )
            try:
                save_phase1_cache(tmp_path, spectra)
                os.replace(tmp_path, target)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)

            self.register_entry(
                cache_key=cache_key,
                category="spectra_phase1",
                file_path=target,
                source="user",
                params=params,
            )
            return spectra, False

    def list_entries(self, category: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """List indexed cache entries from runtime.sqlite."""
        with self._lock:
            conn = self.connect_runtime()
            try:
                if category is not None:
                    rows = conn.execute(
                        "SELECT * FROM cache_index WHERE category = ? "
                        "ORDER BY created_at DESC LIMIT ?",
                        (category, int(limit)),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM cache_index ORDER BY created_at DESC LIMIT ?",
                        (int(limit),),
                    ).fetchall()
                return [
                    {
                        "cache_key": r["cache_key"],
                        "category": r["category"],
                        "file_path": r["file_path"],
                        "source": r["source"],
                        "size_bytes": int(r["size_bytes"]),
                        "params": json.loads(r["params_json"]),
                        "created_at": r["created_at"],
                    }
                    for r in rows
                ]
            finally:
                conn.close()
