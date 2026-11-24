"""Top-level Windows spawn-safe worker functions for ProcessPoolExecutor Tier 3 jobs."""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Any


def worker_init() -> None:
    """Enforce single-threaded BLAS/Numba execution inside each ProcessPool worker."""
    for env_var in (
        "NUMBA_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[env_var] = "1"
    numba_mod = sys.modules.get("numba")
    if numba_mod is not None:
        try:
            numba_mod.set_num_threads(1)
        except Exception:
            pass


def inspect_worker_runtime() -> dict[str, Any]:
    """Return worker thread configuration and loaded module state (for spawn diagnostics)."""
    main_loaded = "geomag_api.main" in sys.modules
    fastapi_loaded = "fastapi" in sys.modules
    numba_threads: int | None = None
    numba_config_threads: int | None = None
    try:
        import numba

        numba_config_threads = int(numba.config.NUMBA_NUM_THREADS)
        numba_threads = int(numba.get_num_threads())
    except Exception:
        pass
    return {
        "env_numba_threads": os.environ.get("NUMBA_NUM_THREADS"),
        "numba_config_threads": numba_config_threads,
        "numba_get_num_threads": numba_threads,
        "main_loaded": main_loaded,
        "fastapi_loaded": fastapi_loaded,
    }


def _load_station_series_for_worker(
    processed_dir: Path,
    *,
    episode_id: str,
    station_iaga: str,
    component: str,
    coord_frame: str,
    d0_deg: float,
    start_utc: str | None,
    end_utc: str | None,
) -> tuple[Any, Any]:
    import numpy as np
    import pandas as pd

    from geomag_core import rotate_xyz_to_magnetic

    parquet_path = processed_dir / "geomag" / f"{episode_id}.parquet"
    df = pd.read_parquet(parquet_path, filters=[("station_iaga", "==", station_iaga)])
    if df.empty:
        raise ValueError(f"Station {station_iaga} has no data in episode {episode_id}.")
    df = df.sort_values("timestamp_utc")
    ts = pd.DatetimeIndex(df["timestamp_utc"]).tz_convert("UTC")

    mask = np.ones(len(ts), dtype=bool)
    if start_utc is not None:
        s_ts = pd.Timestamp(start_utc)
        s_ts = s_ts.tz_localize("UTC") if s_ts.tzinfo is None else s_ts.tz_convert("UTC")
        mask &= np.asarray(ts >= s_ts, dtype=bool)
    if end_utc is not None:
        e_ts = pd.Timestamp(end_utc)
        e_ts = e_ts.tz_localize("UTC") if e_ts.tzinfo is None else e_ts.tz_convert("UTC")
        mask &= np.asarray(ts <= e_ts, dtype=bool)
    if not np.any(mask):
        raise ValueError("Requested UTC window has 0 samples.")

    xyz = np.column_stack(
        [
            df["x_nT"].to_numpy(dtype=np.float64)[mask],
            df["y_nT"].to_numpy(dtype=np.float64)[mask],
            df["z_nT"].to_numpy(dtype=np.float64)[mask],
        ]
    )
    if coord_frame.upper() == "XMYM":
        xyz = rotate_xyz_to_magnetic(xyz, math.radians(float(d0_deg)))

    comp_upper = component.upper()
    if comp_upper in ("X", "XM"):
        col_idx = 0
    elif comp_upper in ("Y", "YM"):
        col_idx = 1
    elif comp_upper == "Z":
        col_idx = 2
    else:
        raise ValueError(f"Unsupported component {component!r}")

    return ts[mask], xyz[:, col_idx]


def execute_wtc_job(
    job_id: str,
    params: dict[str, Any],
    processed_dir_str: str,
    output_path_str: str,
    progress_queue: Any,
    cancel_event: Any,
) -> dict[str, Any]:
    """Run XWT/WTC + AR(1) Monte Carlo inside an isolated worker and write NPZ atomically."""
    worker_init()

    from geomag_core import CalculationCancelledError, compute_xwt_wtc, save_wtc_cache

    processed_dir = Path(processed_dir_str)
    out_path = Path(output_path_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.stem}.{os.getpid()}.tmp.npz")

    t_start = time.monotonic()

    def _emit_progress(done: int, total: int, stage: str) -> None:
        elapsed = time.monotonic() - t_start
        pct = round(100.0 * float(done) / max(float(total), 1.0), 2)
        eta = round(elapsed * float(total - done) / float(done), 2) if done > 0 else None
        try:
            progress_queue.put_nowait(
                {
                    "job_id": job_id,
                    "status": "running",
                    "stage": stage,
                    "progress_pct": pct,
                    "completed_steps": int(done),
                    "total_steps": int(total),
                    "eta_sec": eta,
                }
            )
        except Exception:
            pass

    try:
        if cancel_event.is_set():
            return {"status": "cancelled"}

        n_surrogates = int(params["n_surrogates"])
        _emit_progress(0, max(1, n_surrogates), "cwt_wtc")

        _, arr_i = _load_station_series_for_worker(
            processed_dir,
            episode_id=params["episode_id"],
            station_iaga=params["station_i"],
            component=params["component"],
            coord_frame=params["coord_frame"],
            d0_deg=float(params["d0_deg_i"]),
            start_utc=params.get("start_utc"),
            end_utc=params.get("end_utc"),
        )
        _, arr_j = _load_station_series_for_worker(
            processed_dir,
            episode_id=params["episode_id"],
            station_iaga=params["station_j"],
            component=params["component"],
            coord_frame=params["coord_frame"],
            d0_deg=float(params["d0_deg_j"]),
            start_utc=params.get("start_utc"),
            end_utc=params.get("end_utc"),
        )

        res = compute_xwt_wtc(
            arr_i,
            arr_j,
            period_min_sec=float(params["period_min_sec"]),
            period_max_sec=float(params["period_max_sec"]),
            dj=float(params["dj"]),
            omega0=float(params["omega0"]),
            gap_policy=str(params["gap_policy"]),
            min_subsegment_samples=int(params.get("min_subsegment_samples", 120)),
            n_surrogates=n_surrogates,
            alpha=float(params["alpha"]),
            seed=int(params["seed"]),
            use_numba=True,
            progress_callback=_emit_progress,
            cancel_check=cancel_event.is_set,
            core_version=str(params["core_version"]),
            cache_params=params["canonical_cache_params"],
        )

        if cancel_event.is_set():
            return {"status": "cancelled"}

        save_wtc_cache(tmp_path, res)
        os.replace(tmp_path, out_path)
        size_bytes = int(out_path.stat().st_size)
        return {
            "status": "completed",
            "result_path": str(out_path),
            "size_bytes": size_bytes,
        }
    except CalculationCancelledError:
        return {"status": "cancelled"}
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
