"""Async ProcessPoolExecutor JobManager with runtime.sqlite persistence and WS/SSE push."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import queue as py_queue
import threading
import uuid
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any

from .cache import CacheManager, utc_now_iso
from .config import ApiSettings
from .workers import execute_wtc_job, worker_init

TERMINAL_JOB_STATUSES = frozenset({"cached", "completed", "cancelled", "failed"})
Subscriber = tuple[asyncio.AbstractEventLoop, asyncio.Queue[dict[str, Any]]]


class TooManyJobsError(RuntimeError):
    """Raised when max_active_jobs capacity is exceeded."""


class JobManager:
    """Coordinates Tier 3 background jobs across ProcessPoolExecutor, runtime.sqlite, and WS/SSE."""

    def __init__(self, settings: ApiSettings, cache: CacheManager) -> None:
        self.settings = settings
        self.cache = cache
        self._lock = threading.RLock()
        self._manager: Any = None
        self._executor: ProcessPoolExecutor | None = None
        self._progress_queue: Any = None
        self._pump_stop = threading.Event()
        self._pump_thread: threading.Thread | None = None

        self._futures: dict[str, Future[dict[str, Any]]] = {}
        self._cancel_events: dict[str, Any] = {}
        self._subscribers: dict[str, set[Subscriber]] = {}
        self._reconcile_orphaned_jobs()

    def _reconcile_orphaned_jobs(self) -> None:
        """Mark stale non-terminal jobs left from a prior crashed/restarted server as failed."""
        now = utc_now_iso()
        with self._lock:
            conn = self.cache.connect_runtime()
            try:
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'failed',
                        stage = 'failed',
                        eta_sec = NULL,
                        error_message = 'Interrupted by server restart',
                        updated_at = ?
                    WHERE status IN ('queued', 'running', 'cancelling')
                    """,
                    (now,),
                )
                conn.commit()
            finally:
                conn.close()

    def _ensure_pool(self) -> None:
        with self._lock:
            if self._executor is not None:
                return
            ctx = multiprocessing.get_context("spawn")
            self._manager = ctx.Manager()
            self._progress_queue = self._manager.Queue()
            self._executor = ProcessPoolExecutor(
                max_workers=self.settings.max_workers,
                mp_context=ctx,
                initializer=worker_init,
            )
            self._pump_stop.clear()
            self._pump_thread = threading.Thread(
                target=self._pump_progress_loop,
                name="geomag-job-progress-pump",
                daemon=True,
            )
            self._pump_thread.start()

    def _pump_progress_loop(self) -> None:
        q = self._progress_queue
        while not self._pump_stop.is_set():
            if q is None:
                break
            try:
                msg = q.get(timeout=0.1)
            except (py_queue.Empty, EOFError, BrokenPipeError, OSError):
                continue
            if not isinstance(msg, dict) or "job_id" not in msg:
                continue
            job_id = str(msg["job_id"])
            updated = self._apply_progress_update(job_id, msg)
            if updated is not None:
                self._notify_subscribers(job_id, updated)

    def _apply_progress_update(
        self,
        job_id: str,
        msg: dict[str, Any],
    ) -> dict[str, Any] | None:
        now = utc_now_iso()
        with self._lock:
            conn = self.cache.connect_runtime()
            try:
                row = conn.execute(
                    "SELECT status FROM jobs WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
                if row is None or row["status"] in TERMINAL_JOB_STATUSES:
                    return None
                next_status = "cancelling" if row["status"] == "cancelling" else "running"
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = ?, stage = ?, progress_pct = ?,
                        completed_steps = ?, total_steps = ?, eta_sec = ?, updated_at = ?
                    WHERE job_id = ?
                    """,
                    (
                        next_status,
                        str(msg.get("stage", "running")),
                        float(msg.get("progress_pct", 0.0)),
                        int(msg.get("completed_steps", 0)),
                        int(msg.get("total_steps", 1)),
                        msg.get("eta_sec"),
                        now,
                        job_id,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
            return self.get_job(job_id)

    def _notify_subscribers(self, job_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            subs = list(self._subscribers.get(job_id, ()))
        for loop, q in subs:
            try:
                loop.call_soon_threadsafe(q.put_nowait, dict(payload))
            except RuntimeError:
                pass

    def _insert_job_row(
        self,
        *,
        job_id: str,
        job_type: str,
        cache_key: str,
        status: str,
        stage: str,
        progress_pct: float,
        completed_steps: int,
        total_steps: int,
        eta_sec: float | None,
        cache_hit: bool,
        result_path: str | None,
        error_message: str | None,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        now = utc_now_iso()
        params_json = json.dumps(params, sort_keys=True, ensure_ascii=True)
        with self._lock:
            conn = self.cache.connect_runtime()
            try:
                conn.execute(
                    """
                    INSERT INTO jobs (
                        job_id, job_type, cache_key, status, stage, progress_pct,
                        completed_steps, total_steps, eta_sec, cache_hit, result_path,
                        error_message, params_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        job_type,
                        cache_key,
                        status,
                        stage,
                        progress_pct,
                        completed_steps,
                        total_steps,
                        eta_sec,
                        1 if cache_hit else 0,
                        result_path,
                        error_message,
                        params_json,
                        now,
                        now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        job = self.get_job(job_id)
        assert job is not None
        return job

    def _row_to_dict(self, row: Any) -> dict[str, Any]:
        return {
            "job_id": row["job_id"],
            "job_type": row["job_type"],
            "cache_key": row["cache_key"],
            "status": row["status"],
            "stage": row["stage"],
            "progress_pct": float(row["progress_pct"]),
            "completed_steps": int(row["completed_steps"]),
            "total_steps": int(row["total_steps"]),
            "eta_sec": float(row["eta_sec"]) if row["eta_sec"] is not None else None,
            "cache_hit": bool(row["cache_hit"]),
            "result_path": row["result_path"],
            "error_message": row["error_message"],
            "params": json.loads(row["params_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self.cache.connect_runtime()
            try:
                row = conn.execute(
                    "SELECT * FROM jobs WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
                return self._row_to_dict(row) if row is not None else None
            finally:
                conn.close()

    def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            conn = self.cache.connect_runtime()
            try:
                rows = conn.execute(
                    "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                    (int(limit),),
                ).fetchall()
                return [self._row_to_dict(r) for r in rows]
            finally:
                conn.close()

    def submit_wtc_job(
        self,
        *,
        params: dict[str, Any],
        cache_key: str,
        reverse_cache_key: str | None = None,
    ) -> dict[str, Any]:
        """Submit a Tier 3 XWT/WTC job or immediately return cached/in-flight handle."""
        total_steps = max(1, int(params["n_surrogates"]))
        with self._lock:
            hit = self.cache.find("cross_wtc", cache_key)
            if hit is not None:
                path, source = hit
                self.cache.register_entry(
                    cache_key=cache_key,
                    category="cross_wtc",
                    file_path=path,
                    source=source,
                    params=params["canonical_cache_params"],
                )
                return self._insert_job_row(
                    job_id=uuid.uuid4().hex,
                    job_type="cross_wtc",
                    cache_key=cache_key,
                    status="cached",
                    stage="cached",
                    progress_pct=100.0,
                    completed_steps=total_steps,
                    total_steps=total_steps,
                    eta_sec=0.0,
                    cache_hit=True,
                    result_path=str(path),
                    error_message=None,
                    params=params,
                )

            if reverse_cache_key is not None and reverse_cache_key != cache_key:
                rev_hit = self.cache.find("cross_wtc", reverse_cache_key)
                if rev_hit is not None:
                    rev_path, _ = rev_hit
                    rev_params = {**params, "phase_negated": True}
                    return self._insert_job_row(
                        job_id=uuid.uuid4().hex,
                        job_type="cross_wtc",
                        cache_key=reverse_cache_key,
                        status="cached",
                        stage="cached",
                        progress_pct=100.0,
                        completed_steps=total_steps,
                        total_steps=total_steps,
                        eta_sec=0.0,
                        cache_hit=True,
                        result_path=str(rev_path),
                        error_message=None,
                        params=rev_params,
                    )

            now = utc_now_iso()
            conn = self.cache.connect_runtime()
            try:
                existing = conn.execute(
                    "SELECT * FROM jobs WHERE cache_key = ? "
                    "AND status IN ('queued', 'running') ORDER BY created_at DESC LIMIT 1",
                    (cache_key,),
                ).fetchone()
                if existing is not None:
                    existing_id = str(existing["job_id"])
                    if existing_id in self._futures:
                        return self._row_to_dict(existing)
                    conn.execute(
                        """
                        UPDATE jobs
                        SET status = 'failed',
                            stage = 'failed',
                            eta_sec = NULL,
                            error_message = 'Orphaned job reconciled',
                            updated_at = ?
                        WHERE job_id = ?
                        """,
                        (now, existing_id),
                    )
                    conn.commit()

                active_count = len(self._futures)
                if active_count >= self.settings.max_active_jobs:
                    raise TooManyJobsError(
                        f"Maximum active jobs limit ({self.settings.max_active_jobs}) reached."
                    )
            finally:
                conn.close()

            self._ensure_pool()
            assert self._executor is not None
            assert self._manager is not None
            assert self._progress_queue is not None

            job_id = uuid.uuid4().hex
            out_path = self.cache.user_path_for("cross_wtc", cache_key)
            cancel_event = self._manager.Event()
            job_record = self._insert_job_row(
                job_id=job_id,
                job_type="cross_wtc",
                cache_key=cache_key,
                status="queued",
                stage="queued",
                progress_pct=0.0,
                completed_steps=0,
                total_steps=total_steps,
                eta_sec=None,
                cache_hit=False,
                result_path=None,
                error_message=None,
                params=params,
            )

            self._cancel_events[job_id] = cancel_event
            future = self._executor.submit(
                execute_wtc_job,
                job_id,
                params,
                str(self.settings.processed_dir),
                str(out_path),
                self._progress_queue,
                cancel_event,
            )
            self._futures[job_id] = future
            future.add_done_callback(
                lambda fut, jid=job_id, ck=cache_key, prm=params: self._on_job_done(
                    jid, ck, prm, fut
                )
            )
            return job_record

    def _on_job_done(
        self,
        job_id: str,
        cache_key: str,
        params: dict[str, Any],
        future: Future[dict[str, Any]],
    ) -> None:
        now = utc_now_iso()
        total_steps = max(1, int(params["n_surrogates"]))
        if future.cancelled():
            outcome: dict[str, Any] = {"status": "cancelled"}
        else:
            try:
                outcome = future.result()
            except Exception as exc:
                outcome = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

        status = str(outcome.get("status", "failed"))
        if status == "completed":
            result_path = Path(str(outcome["result_path"]))
            self.cache.register_entry(
                cache_key=cache_key,
                category="cross_wtc",
                file_path=result_path,
                source="user",
                params=params["canonical_cache_params"],
            )
            with self._lock:
                conn = self.cache.connect_runtime()
                try:
                    conn.execute(
                        """
                        UPDATE jobs
                        SET status = 'completed', stage = 'completed', progress_pct = 100.0,
                            completed_steps = ?, total_steps = ?, eta_sec = 0.0,
                            result_path = ?, error_message = NULL, updated_at = ?
                        WHERE job_id = ?
                        """,
                        (total_steps, total_steps, str(result_path), now, job_id),
                    )
                    conn.commit()
                finally:
                    conn.close()
        elif status == "cancelled":
            with self._lock:
                conn = self.cache.connect_runtime()
                try:
                    conn.execute(
                        """
                        UPDATE jobs
                        SET status = 'cancelled', stage = 'cancelled', eta_sec = NULL,
                            error_message = NULL, updated_at = ?
                        WHERE job_id = ?
                        """,
                        (now, job_id),
                    )
                    conn.commit()
                finally:
                    conn.close()
        else:
            err = str(outcome.get("error", "Unknown worker error"))
            with self._lock:
                conn = self.cache.connect_runtime()
                try:
                    conn.execute(
                        """
                        UPDATE jobs
                        SET status = 'failed', stage = 'failed', eta_sec = NULL,
                            error_message = ?, updated_at = ?
                        WHERE job_id = ?
                        """,
                        (err, now, job_id),
                    )
                    conn.commit()
                finally:
                    conn.close()

        with self._lock:
            self._futures.pop(job_id, None)
            self._cancel_events.pop(job_id, None)

        final_job = self.get_job(job_id)
        if final_job is not None:
            self._notify_subscribers(job_id, final_job)

    def cancel_job(self, job_id: str) -> dict[str, Any] | None:
        """Signal cooperative cancellation for queued or running job."""
        now = utc_now_iso()
        with self._lock:
            current = self.get_job(job_id)
            if current is None:
                return None
            if current["status"] in TERMINAL_JOB_STATUSES:
                return current

            ev = self._cancel_events.get(job_id)
            if ev is not None:
                ev.set()
            fut = self._futures.get(job_id)
            cancelled_immediately = fut is None or bool(fut.cancel())

            conn = self.cache.connect_runtime()
            try:
                if cancelled_immediately:
                    conn.execute(
                        "UPDATE jobs SET status = 'cancelled', stage = 'cancelled', "
                        "eta_sec = NULL, updated_at = ? WHERE job_id = ?",
                        (now, job_id),
                    )
                else:
                    conn.execute(
                        "UPDATE jobs SET status = 'cancelling', stage = 'cancelling', "
                        "updated_at = ? WHERE job_id = ?",
                        (now, job_id),
                    )
                conn.commit()
            finally:
                conn.close()

        updated = self.get_job(job_id)
        if updated is not None:
            self._notify_subscribers(job_id, updated)
        return updated

    def subscribe_with_snapshot(
        self,
        job_id: str,
        loop: asyncio.AbstractEventLoop,
    ) -> tuple[dict[str, Any] | None, Subscriber | None]:
        """Atomically fetch job snapshot and subscribe to live updates if non-terminal."""
        with self._lock:
            snap = self.get_job(job_id)
            if snap is None:
                return None, None
            if snap["status"] in TERMINAL_JOB_STATUSES:
                return snap, None
            q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
            sub: Subscriber = (loop, q)
            self._subscribers.setdefault(job_id, set()).add(sub)
            return snap, sub

    def unsubscribe(self, job_id: str, sub: Subscriber | None) -> None:
        if sub is None:
            return
        with self._lock:
            subs = self._subscribers.get(job_id)
            if subs is not None:
                subs.discard(sub)
                if not subs:
                    self._subscribers.pop(job_id, None)

    def shutdown(self) -> None:
        """Gracefully cancel in-flight jobs and terminate worker pool and sync manager."""
        with self._lock:
            for ev in self._cancel_events.values():
                try:
                    ev.set()
                except Exception:
                    pass
            executor = self._executor
            manager = self._manager
            pump_thread = self._pump_thread
            self._pump_stop.set()

        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if pump_thread is not None and pump_thread.is_alive():
            pump_thread.join(timeout=3.0)
        if manager is not None:
            try:
                manager.shutdown()
            except Exception:
                pass

        with self._lock:
            self._executor = None
            self._manager = None
            self._progress_queue = None
            self._pump_thread = None
            self._futures.clear()
            self._cancel_events.clear()
            self._subscribers.clear()
