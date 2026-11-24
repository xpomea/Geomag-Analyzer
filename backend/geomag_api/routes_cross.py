"""Two-phase Welch cross-spectral router (/api/cross/welch)."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter

import geomag_core as gc
from geomag_core import (
    Phase1StationSpectra,
    build_phase1_cache_key,
    compute_azimuthal_wavenumber_m,
    compute_meridional_flr_profile,
    compute_station_phase1_spectra,
    compute_welch_cross_matrix,
    compute_welch_cross_pair,
)

from .cache import CacheManager
from .data_store import DataStore, StationEpisodeData, resolve_component_index
from .deps import CacheDep, StoreDep
from .schemas import (
    AzimuthalOut,
    MeridionalFLRRequest,
    MeridionalFLRResponse,
    WelchMatrixRequest,
    WelchMatrixResponse,
    WelchPairRequest,
    WelchPairResponse,
)
from .serializers import timestamps_to_iso, to_json_float, to_json_list

router = APIRouter(prefix="/api/cross/welch", tags=["cross-welch"])

_DEFAULT_QC_PARAMS = {"max_missing_frac": 0.05, "max_gap": 3}


def _get_or_compute_phase1(
    store: DataStore,
    cache: CacheManager,
    data: StationEpisodeData,
    *,
    component: str,
    coord_frame: str,
    external_d0_deg: float | None,
    L: int,
    M: int,
    window_samples: int,
    step_samples: int,
) -> tuple[Phase1StationSpectra, bool]:
    canon_frame, comp_idx = resolve_component_index(component, coord_frame)
    xyz, d0_deg, canon_frame = store.apply_coord_frame(
        data,
        coord_frame=canon_frame,
        external_d0_deg=external_d0_deg,
    )
    cache_key = build_phase1_cache_key(
        gc.__version__,
        episode_id=data.episode_id,
        station_iaga=data.station_iaga,
        component=component,
        coord_frame=canon_frame,
        d0_deg=d0_deg,
        L=L,
        M=M,
        window_samples=window_samples,
        step_samples=step_samples,
        qc_params=_DEFAULT_QC_PARAMS,
    )
    params: dict[str, Any] = {
        "episode_id": data.episode_id,
        "station_iaga": data.station_iaga,
        "component": component,
        "coord_frame": canon_frame,
        "d0_deg": round(float(d0_deg), 6),
        "L": int(L),
        "M": int(M),
        "window_samples": int(window_samples),
        "step_samples": int(step_samples),
        "qc_params": _DEFAULT_QC_PARAMS,
    }
    return cache.get_or_compute_phase1(
        cache_key,
        params,
        lambda: compute_station_phase1_spectra(
            xyz[:, comp_idx],
            station_iaga=data.station_iaga,
            component=component,
            L=L,
            M=M,
            window_samples=window_samples,
            step_samples=step_samples,
            is_valid=data.is_valid_xyz,
            episode_id=data.episode_id,
            coord_frame=canon_frame,
            d0_deg=d0_deg,
            core_version=gc.__version__,
        ),
    )


@router.post("/pair", response_model=WelchPairResponse)
def cross_welch_pair(
    req: WelchPairRequest,
    store: StoreDep,
    cache: CacheDep,
) -> WelchPairResponse:
    """Compute Phase 2 pairwise Welch cross-spectrum strictly over joint valid segments."""
    canon_frame, _ = resolve_component_index(req.component, req.coord_frame)
    if req.window_samples < req.L:
        raise ValueError(f"window_samples ({req.window_samples}) must be >= L ({req.L}).")

    data_i = store.load_station_episode(req.episode_id, req.station_i)
    data_j = store.load_station_episode(req.episode_id, req.station_j)
    warnings = list(data_i.warnings) + list(data_j.warnings)

    if not data_i.available or not data_j.available:
        store.apply_coord_frame(
            data_i, coord_frame=req.coord_frame, external_d0_deg=req.external_d0_deg_i
        )
        store.apply_coord_frame(
            data_j, coord_frame=req.coord_frame, external_d0_deg=req.external_d0_deg_j
        )
        return WelchPairResponse(
            episode_id=req.episode_id,
            station_i=data_i.station_iaga,
            station_j=data_j.station_iaga,
            component=req.component,
            coord_frame="XmYm" if req.coord_frame.upper() == "XMYM" else "XYZ",
            band=req.band,
            available=False,
            cache_hits={data_i.station_iaga: False, data_j.station_iaga: False},
            window_timestamps_utc=[],
            freqs_hz=[],
            K_ij=[],
            n_eff=[],
            gamma2=[],
            gamma2_crit=[],
            phase_deg=[],
            phase_deg_masked=[],
            is_significant=[],
            is_valid=[],
            gamma2_tf=[],
            gamma2_crit_tf=[],
            phase_deg_tf=[],
            phase_deg_tf_masked=[],
            azimuthal=None,
            warnings=warnings,
        )

    sp_i, hit_i = _get_or_compute_phase1(
        store,
        cache,
        data_i,
        component=req.component,
        coord_frame=req.coord_frame,
        external_d0_deg=req.external_d0_deg_i,
        L=req.L,
        M=req.M,
        window_samples=req.window_samples,
        step_samples=req.step_samples,
    )
    sp_j, hit_j = _get_or_compute_phase1(
        store,
        cache,
        data_j,
        component=req.component,
        coord_frame=req.coord_frame,
        external_d0_deg=req.external_d0_deg_j,
        L=req.L,
        M=req.M,
        window_samples=req.window_samples,
        step_samples=req.step_samples,
    )

    pair = compute_welch_cross_pair(
        sp_i,
        sp_j,
        band=req.band,
        t_max_pc6_sec=req.t_max_pc6_sec,
        f_min_hz=req.f_min_hz,
        f_max_hz=req.f_max_hz,
        alpha=req.alpha,
        min_common_segments=req.min_common_segments,
        spectrum_type=req.spectrum_type,
    )
    az = compute_azimuthal_wavenumber_m(
        pair.phase_rad,
        float(data_i.station_meta["lon_deg"]),
        float(data_j.station_meta["lon_deg"]),
        pair.gamma2,
        pair.gamma2_crit,
        min_delta_lon_deg=req.min_delta_lon_deg,
        unwrap_time=req.unwrap_time,
    )
    win_ts = timestamps_to_iso(data_i.timestamps_utc[pair.window_center_indices])
    az_out = AzimuthalOut(
        m=to_json_list(az.m),
        m_alias=to_json_float(az.m_alias),
        delta_lon_deg=float(az.delta_lon_deg),
        is_meridional_guarded=bool(az.is_meridional_guarded),
    )

    return WelchPairResponse(
        episode_id=req.episode_id,
        station_i=pair.station_i,
        station_j=pair.station_j,
        component=req.component,
        coord_frame="XmYm" if req.coord_frame.upper() == "XMYM" else "XYZ",
        band=req.band,
        available=True,
        cache_hits={pair.station_i: hit_i, pair.station_j: hit_j},
        window_timestamps_utc=win_ts,
        freqs_hz=to_json_list(pair.freqs_hz),
        K_ij=to_json_list(pair.K_ij),
        n_eff=to_json_list(pair.n_eff),
        gamma2=to_json_list(pair.gamma2),
        gamma2_crit=to_json_list(pair.gamma2_crit),
        phase_deg=to_json_list(pair.phase_deg),
        phase_deg_masked=to_json_list(pair.phase_deg_masked),
        is_significant=to_json_list(pair.is_significant),
        is_valid=to_json_list(pair.is_valid),
        gamma2_tf=to_json_list(pair.gamma2_tf),
        gamma2_crit_tf=to_json_list(pair.gamma2_crit_tf),
        phase_deg_tf=to_json_list(pair.phase_deg_tf),
        phase_deg_tf_masked=to_json_list(pair.phase_deg_tf_masked),
        azimuthal=az_out,
        warnings=warnings,
    )


@router.post("/matrix", response_model=WelchMatrixResponse)
def cross_welch_matrix(
    req: WelchMatrixRequest,
    store: StoreDep,
    cache: CacheDep,
) -> WelchMatrixResponse:
    """Compute S x S dynamic all-pairs Welch cross-spectral matrix via BLAS GEMM."""
    resolve_component_index(req.component, req.coord_frame)
    if req.window_samples < req.L:
        raise ValueError(f"window_samples ({req.window_samples}) must be >= L ({req.L}).")
    store.require_episode(req.episode_id)

    if req.station_iagas is None:
        codes = store.episode_magnetometer_codes(req.episode_id)
    else:
        codes = [store.require_magnetometer(c)["iaga_code"] for c in req.station_iagas]

    spectra_list: list[Phase1StationSpectra] = []
    cache_hits: dict[str, bool] = {}
    skipped: dict[str, str] = {}
    ref_ts = None

    for code in codes:
        data = store.load_station_episode(req.episode_id, code)
        if not data.available:
            skipped[code] = data.warnings[0] if data.warnings else "No data in episode."
            continue
        try:
            sp, hit = _get_or_compute_phase1(
                store,
                cache,
                data,
                component=req.component,
                coord_frame=req.coord_frame,
                external_d0_deg=None,
                L=req.L,
                M=req.M,
                window_samples=req.window_samples,
                step_samples=req.step_samples,
            )
        except ValueError as exc:
            skipped[code] = str(exc)
            continue
        if ref_ts is None:
            ref_ts = data.timestamps_utc
        spectra_list.append(sp)
        cache_hits[code] = hit

    if len(spectra_list) < 2 or ref_ts is None:
        raise ValueError("Need at least 2 valid magnetometer stations for Welch matrix.")

    t0 = time.perf_counter()
    mat = compute_welch_cross_matrix(
        spectra_list,
        band=req.band,
        t_max_pc6_sec=req.t_max_pc6_sec,
        alpha=req.alpha,
        min_common_segments=req.min_common_segments,
        spectrum_type=req.spectrum_type,
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    win_ts = timestamps_to_iso(ref_ts[mat.window_center_indices])

    return WelchMatrixResponse(
        episode_id=req.episode_id,
        component=req.component,
        coord_frame="XmYm" if req.coord_frame.upper() == "XMYM" else "XYZ",
        band=req.band,
        station_codes=list(mat.station_codes),
        window_timestamps_utc=win_ts,
        gamma2=to_json_list(mat.gamma2),
        gamma2_crit=to_json_list(mat.gamma2_crit),
        phase_deg_masked=to_json_list(mat.phase_deg_masked),
        K_ij=to_json_list(mat.K_ij),
        is_significant=to_json_list(mat.is_significant),
        cache_hits=cache_hits,
        skipped_stations=skipped,
        elapsed_ms=elapsed_ms,
    )


@router.post("/flr", response_model=MeridionalFLRResponse)
def cross_welch_flr(
    req: MeridionalFLRRequest,
    store: StoreDep,
    cache: CacheDep,
) -> MeridionalFLRResponse:
    """Compute meridional amplitude and phase gradient profile across a latitudinal chain."""
    resolve_component_index(req.component, req.coord_frame)
    if req.window_samples < req.L:
        raise ValueError(f"window_samples ({req.window_samples}) must be >= L ({req.L}).")
    store.require_episode(req.episode_id)

    spectra_chain: list[Phase1StationSpectra] = []
    lats: list[float] = []
    ref_ts = None

    for code in req.station_chain:
        data = store.load_station_episode(req.episode_id, code)
        if not data.available:
            raise ValueError(
                data.warnings[0]
                if data.warnings
                else f"Station {code} unavailable in {req.episode_id}."
            )
        sp, _ = _get_or_compute_phase1(
            store,
            cache,
            data,
            component=req.component,
            coord_frame=req.coord_frame,
            external_d0_deg=None,
            L=req.L,
            M=req.M,
            window_samples=req.window_samples,
            step_samples=req.step_samples,
        )
        if ref_ts is None:
            ref_ts = data.timestamps_utc
        spectra_chain.append(sp)
        lats.append(float(data.station_meta["lat_deg"]))

    assert ref_ts is not None
    n_windows = len(spectra_chain[0].window_center_indices)
    if req.window_index >= n_windows:
        raise ValueError(
            f"window_index ({req.window_index}) out of bounds [0, {n_windows - 1}]."
        )

    prof = compute_meridional_flr_profile(
        spectra_chain,
        lats,
        band=req.band,
        window_index=req.window_index,
        alpha=req.alpha,
        min_common_segments=req.min_common_segments,
        t_max_pc6_sec=req.t_max_pc6_sec,
    )
    center_idx = int(spectra_chain[0].window_center_indices[req.window_index])
    win_iso = timestamps_to_iso(ref_ts[[center_idx]])[0]

    return MeridionalFLRResponse(
        episode_id=req.episode_id,
        band=req.band,
        window_timestamp_utc=win_iso,
        station_codes=list(prof.station_codes),
        latitudes_deg=[float(v) for v in prof.latitudes_deg],
        band_power=to_json_list(prof.band_power),
        amplitude=to_json_list(prof.amplitude),
        pair_coherence=to_json_list(prof.pair_coherence),
        pair_gamma2_crit=to_json_list(prof.pair_gamma2_crit),
        pair_phase_deg=to_json_list(prof.pair_phase_deg),
        pair_is_significant=to_json_list(prof.pair_is_significant),
        cumulative_phase_deg=to_json_list(prof.cumulative_phase_deg),
        phase_gradient_deg_per_lat=to_json_list(prof.phase_gradient_deg_per_lat),
        peak_station=prof.peak_station,
        peak_latitude_deg=to_json_float(prof.peak_latitude_deg),
    )
