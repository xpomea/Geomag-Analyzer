"""Async Tier 3 jobs router: /api/jobs, WS /ws/jobs/{job_id}, and /api/cache."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import (
    APIRouter,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)

import geomag_core as gc
from geomag_core import (
    MORLET_DJ0,
    WTC_CACHE_FORMAT,
    build_wtc_cache_key,
    find_valid_subsegments,
    load_wtc_cache,
)

from .data_store import NotFoundError, resolve_component_index
from .deps import CacheDep, JobManagerDep, SettingsDep, StoreDep
from .jobs import TERMINAL_JOB_STATUSES, JobManager
from .schemas import CacheEntryOut, JobStatusResponse, WTCJobRequest, WTCResultResponse
from .security import AuthState, verify_ws_origin
from .serializers import timestamps_to_iso, to_json_list

router = APIRouter(tags=["jobs"])


@router.post("/api/jobs/wtc", response_model=JobStatusResponse)
def submit_wtc_job(
    req: WTCJobRequest,
    settings: SettingsDep,
    store: StoreDep,
    job_manager: JobManagerDep,
) -> JobStatusResponse:
    """Submit a Tier 3 XWT/WTC + AR(1) Monte Carlo job or return instant cache hit."""
    canon_frame, c_idx = resolve_component_index(req.component, req.coord_frame)
    if req.n_surrogates > settings.max_mc_surrogates:
        raise ValueError(
            f"n_surrogates ({req.n_surrogates}) exceeds configured quota "
            f"({settings.max_mc_surrogates})."
        )
    if req.period_min_sec >= req.period_max_sec:
        raise ValueError("period_min_sec must be strictly less than period_max_sec.")

    data_i = store.load_station_episode(
        req.episode_id,
        req.station_i,
        start_utc=req.start_utc,
        end_utc=req.end_utc,
    )
    data_j = store.load_station_episode(
        req.episode_id,
        req.station_j,
        start_utc=req.start_utc,
        end_utc=req.end_utc,
    )

    # Validate coordinate frame & baseline D0 guards upfront
    xyz_i, d0_deg_i, canon_frame = store.apply_coord_frame(
        data_i,
        coord_frame=canon_frame,
        external_d0_deg=req.external_d0_deg_i,
    )
    xyz_j, d0_deg_j, _ = store.apply_coord_frame(
        data_j,
        coord_frame=canon_frame,
        external_d0_deg=req.external_d0_deg_j,
    )

    if not data_i.available or not data_j.available:
        missing = [d.station_iaga for d in (data_i, data_j) if not d.available]
        raise ValueError(
            f"Station(s) {', '.join(missing)} have no magnetometer data in {req.episode_id}."
        )

    joint_valid = np.isfinite(xyz_i[:, c_idx]) & np.isfinite(xyz_j[:, c_idx])
    if not np.any(joint_valid):
        raise ValueError("Station pair has 0 jointly finite samples in selected window.")
    if req.gap_policy == "subsegments":
        segs = find_valid_subsegments(joint_valid, min_length=int(req.min_subsegment_samples))
        if not segs:
            raise ValueError(
                f"No contiguous finite subsegment of length >= {req.min_subsegment_samples}."
            )

    qc_params: dict[str, Any] = {
        "d0_deg_i": round(float(d0_deg_i), 6),
        "d0_deg_j": round(float(d0_deg_j), 6),
        "start_utc": req.start_utc,
        "end_utc": req.end_utc,
        "min_subsegment_samples": int(req.min_subsegment_samples),
    }
    cache_key = build_wtc_cache_key(
        gc.__version__,
        episode_id=req.episode_id,
        pair=(data_i.station_iaga, data_j.station_iaga),
        component=req.component,
        coord_frame=canon_frame,
        d0_deg=d0_deg_i,
        period_min_sec=req.period_min_sec,
        period_max_sec=req.period_max_sec,
        dj=req.dj,
        omega0=req.omega0,
        gap_policy=req.gap_policy,
        n_surrogates=req.n_surrogates,
        alpha=req.alpha,
        seed=req.seed,
        qc_params=qc_params,
    )
    canonical_cache_params = {
        "stage": "cross_wtc",
        "cache_format": WTC_CACHE_FORMAT,
        "episode_id": req.episode_id,
        "pair": [data_i.station_iaga, data_j.station_iaga],
        "component": req.component,
        "coord_frame": canon_frame,
        "d0_deg": round(float(d0_deg_i), 6),
        "band_range": [float(req.period_min_sec), float(req.period_max_sec)],
        "scales": {"dj": float(req.dj)},
        "wavelet": {"name": "morlet", "omega0": float(req.omega0)},
        "smoothing_params": {"dj0": MORLET_DJ0 * (6.0 / float(req.omega0))},
        "gap_policy": req.gap_policy,
        "n_surrogates": int(req.n_surrogates),
        "alpha": float(req.alpha),
        "seed": int(req.seed),
        "qc_params": qc_params,
    }
    worker_params: dict[str, Any] = {
        "episode_id": req.episode_id,
        "station_i": data_i.station_iaga,
        "station_j": data_j.station_iaga,
        "component": req.component,
        "coord_frame": canon_frame,
        "d0_deg_i": float(d0_deg_i),
        "d0_deg_j": float(d0_deg_j),
        "external_d0_deg_i": req.external_d0_deg_i,
        "external_d0_deg_j": req.external_d0_deg_j,
        "period_min_sec": float(req.period_min_sec),
        "period_max_sec": float(req.period_max_sec),
        "dj": float(req.dj),
        "omega0": float(req.omega0),
        "gap_policy": req.gap_policy,
        "min_subsegment_samples": int(req.min_subsegment_samples),
        "n_surrogates": int(req.n_surrogates),
        "alpha": float(req.alpha),
        "seed": int(req.seed),
        "start_utc": req.start_utc,
        "end_utc": req.end_utc,
        "core_version": gc.__version__,
        "canonical_cache_params": canonical_cache_params,
    }
    reverse_cache_key: str | None = None
    if data_i.station_iaga != data_j.station_iaga:
        rev_qc_params = {
            "d0_deg_i": round(float(d0_deg_j), 6),
            "d0_deg_j": round(float(d0_deg_i), 6),
            "start_utc": req.start_utc,
            "end_utc": req.end_utc,
            "min_subsegment_samples": int(req.min_subsegment_samples),
        }
        reverse_cache_key = build_wtc_cache_key(
            gc.__version__,
            episode_id=req.episode_id,
            pair=(data_j.station_iaga, data_i.station_iaga),
            component=req.component,
            coord_frame=canon_frame,
            d0_deg=d0_deg_j,
            period_min_sec=req.period_min_sec,
            period_max_sec=req.period_max_sec,
            dj=req.dj,
            omega0=req.omega0,
            gap_policy=req.gap_policy,
            n_surrogates=req.n_surrogates,
            alpha=req.alpha,
            seed=req.seed,
            qc_params=rev_qc_params,
        )

    job = job_manager.submit_wtc_job(
        params=worker_params,
        cache_key=cache_key,
        reverse_cache_key=reverse_cache_key,
    )
    return JobStatusResponse(**job)


@router.get("/api/jobs", response_model=list[JobStatusResponse])
def list_jobs(
    job_manager: JobManagerDep,
    limit: int = Query(default=50, ge=1, le=500),
) -> list[JobStatusResponse]:
    """List recent background jobs from runtime.sqlite."""
    return [JobStatusResponse(**row) for row in job_manager.list_jobs(limit=limit)]


@router.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(
    job_id: str,
    job_manager: JobManagerDep,
) -> JobStatusResponse:
    """Get current status and progress for one job."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise NotFoundError(f"Unknown job_id {job_id!r}")
    return JobStatusResponse(**job)


@router.delete("/api/jobs/{job_id}", response_model=JobStatusResponse)
def cancel_job(
    job_id: str,
    job_manager: JobManagerDep,
) -> JobStatusResponse:
    """Request cooperative cancellation of a queued or running background job."""
    job = job_manager.cancel_job(job_id)
    if job is None:
        raise NotFoundError(f"Unknown job_id {job_id!r}")
    return JobStatusResponse(**job)


@router.get("/api/jobs/{job_id}/result", response_model=WTCResultResponse)
def get_wtc_job_result(
    job_id: str,
    store: StoreDep,
    job_manager: JobManagerDep,
    stride: int = Query(default=1, ge=1, le=30),
) -> WTCResultResponse:
    """Load completed/cached XWT/WTC matrices and 95% Monte Carlo significance contour."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise NotFoundError(f"Unknown job_id {job_id!r}")
    if job["status"] not in ("completed", "cached") or not job["result_path"]:
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is in status {job['status']!r}; result not ready.",
        )

    res_path = Path(job["result_path"])
    if not res_path.is_file():
        raise NotFoundError(f"Cached result file missing for job {job_id!r}")

    wtc = load_wtc_cache(res_path)
    params = job["params"]
    data_i = store.load_station_episode(
        params["episode_id"],
        params["station_i"],
        start_utc=params.get("start_utc"),
        end_utc=params.get("end_utc"),
    )
    ts_iso = timestamps_to_iso(data_i.timestamps_utc[::stride])
    sl = slice(None, None, stride)
    n_out = len(ts_iso)
    scaled_subsegments: list[list[int]] = []
    for s_orig, e_orig in wtc.subsegments:
        s_st = (int(s_orig) + stride - 1) // stride
        e_st = min(n_out, (int(e_orig) + stride - 1) // stride)
        if e_st > s_st:
            scaled_subsegments.append([s_st, e_st])

    phase_negated = bool(params.get("phase_negated", False))
    xwt_ph_arr = wtc.xwt_phase_deg[:, sl]
    wtc_ph_arr = wtc.wtc_phase_deg[:, sl]
    if phase_negated:
        xwt_ph_arr = np.where(np.isfinite(xwt_ph_arr), -xwt_ph_arr, np.nan)
        wtc_ph_arr = np.where(np.isfinite(wtc_ph_arr), -wtc_ph_arr, np.nan)

    return WTCResultResponse(
        job_id=job["job_id"],
        cache_key=wtc.cache_key,
        episode_id=params["episode_id"],
        station_i=params["station_i"],
        station_j=params["station_j"],
        component=params["component"],
        coord_frame=params["coord_frame"],
        timestamps_utc=ts_iso,
        scales=to_json_list(wtc.scales),
        periods_sec=to_json_list(wtc.periods_sec),
        coi_periods_sec=to_json_list(wtc.coi_periods_sec[sl]),
        inside_coi=to_json_list(wtc.inside_coi[:, sl]),
        xwt_power=to_json_list(wtc.xwt_power[:, sl]),
        xwt_phase_deg=to_json_list(xwt_ph_arr),
        wtc_r2=to_json_list(wtc.wtc_r2[:, sl]),
        wtc_phase_deg=to_json_list(wtc_ph_arr),
        wtc_sig95=to_json_list(wtc.wtc_sig95) if wtc.wtc_sig95 is not None else None,
        wtc_is_significant=to_json_list(wtc.wtc_is_significant[:, sl]),
        alpha1_i=float(wtc.alpha1_j) if phase_negated else float(wtc.alpha1_i),
        alpha1_j=float(wtc.alpha1_i) if phase_negated else float(wtc.alpha1_j),
        gap_policy=wtc.gap_policy,
        subsegments=scaled_subsegments,
        n_surrogates=int(wtc.n_surrogates),
        alpha=float(wtc.alpha),
    )


@router.websocket("/ws/jobs/{job_id}")
async def ws_job_progress(websocket: WebSocket, job_id: str) -> None:
    """WebSocket progress stream (/ws/jobs/{job_id}) guarded by Host/Origin validation."""
    auth: AuthState = websocket.app.state.auth
    job_manager: JobManager = websocket.app.state.job_manager

    ok, _, _ = verify_ws_origin(websocket.headers, auth)
    if not ok:
        await websocket.close(code=1008)
        return

    loop = asyncio.get_running_loop()
    snap, sub = job_manager.subscribe_with_snapshot(job_id, loop)
    if snap is None:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    try:
        await websocket.send_json(snap)
        if snap["status"] in TERMINAL_JOB_STATUSES or sub is None:
            return

        _, queue = sub
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=1.0)
            except TimeoutError:
                cur = job_manager.get_job(job_id)
                if cur is None:
                    break
                if cur["status"] in TERMINAL_JOB_STATUSES:
                    await websocket.send_json(cur)
                    break
                continue
            await websocket.send_json(event)
            if event["status"] in TERMINAL_JOB_STATUSES:
                break
    except WebSocketDisconnect:
        pass
    finally:
        job_manager.unsubscribe(job_id, sub)


@router.get("/api/cache", response_model=list[CacheEntryOut])
def list_cache_entries(
    cache: CacheDep,
    category: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[CacheEntryOut]:
    """List indexed SHA-256 NPZ artifacts in runtime.sqlite."""
    return [
        CacheEntryOut(**row)
        for row in cache.list_entries(category=category, limit=limit)
    ]
