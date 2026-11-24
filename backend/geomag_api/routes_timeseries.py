"""Time-series REST router (/api/timeseries): INTERMAGNET, SOLAR-1, OMNI, OULU."""

from __future__ import annotations

from fastapi import APIRouter, Query

from .data_store import DataStore
from .deps import StoreDep
from .schemas import (
    CombinedTimeseriesResponse,
    GeomagTimeseriesResponse,
    OmniSeries,
    OuluSeries,
    Solar1Series,
    SpaceWeatherResponse,
    StationInfo,
)
from .serializers import timestamps_to_iso, to_json_float, to_json_list

router = APIRouter(prefix="/api/timeseries", tags=["timeseries"])


def _build_geomag_response(
    store: DataStore,
    *,
    episode_id: str,
    station_iaga: str,
    coord_frame: str,
    external_d0_deg: float | None,
    start_utc: str | None,
    end_utc: str | None,
) -> GeomagTimeseriesResponse:
    data = store.load_station_episode(
        episode_id,
        station_iaga,
        start_utc=start_utc,
        end_utc=end_utc,
    )
    xyz, d0_deg, canon_frame = store.apply_coord_frame(
        data,
        coord_frame=coord_frame,
        external_d0_deg=external_d0_deg,
    )
    return GeomagTimeseriesResponse(
        episode_id=episode_id,
        station_iaga=data.station_iaga,
        station_meta=StationInfo(**data.station_meta),
        available=data.available,
        coord_frame=canon_frame,
        d0_deg=float(d0_deg),
        valid_pct=float(data.valid_pct),
        timestamps_utc=timestamps_to_iso(data.timestamps_utc),
        x_nT=to_json_list(xyz[:, 0]),
        y_nT=to_json_list(xyz[:, 1]),
        z_nT=to_json_list(xyz[:, 2]),
        f_scalar_nT=to_json_list(data.f_scalar_nT),
        is_valid_xyz=to_json_list(data.is_valid_xyz),
        is_interpolated=to_json_list(data.is_interpolated),
        was_despiked=to_json_list(data.was_despiked),
        warnings=list(data.warnings),
    )


def _build_space_weather_response(
    store: DataStore,
    *,
    episode_id: str,
    solar1_shift: str,
    manual_lag_min: float | None,
    ref_distance_km: float,
    smooth_window_min: int,
    include_omni: bool,
    include_oulu: bool,
    start_utc: str | None,
    end_utc: str | None,
) -> SpaceWeatherResponse:
    sw = store.load_solar1(
        episode_id,
        shift_mode=solar1_shift,
        manual_lag_min=manual_lag_min,
        ref_distance_km=ref_distance_km,
        smooth_window_min=smooth_window_min,
        start_utc=start_utc,
        end_utc=end_utc,
    )
    solar1_out = Solar1Series(
        timestamps_utc=timestamps_to_iso(sw["timestamps_utc"]),
        time_frame=sw["time_frame"],
        shift_mode=sw["shift_mode"],
        ref_distance_km=sw["ref_distance_km"],
        propagation_delay_min=to_json_list(sw["propagation_delay_min"]),
        series={k: to_json_list(v) for k, v in sw["series"].items()},
        swips_flags=to_json_list(sw["swips_flags"]),
        is_swips_valid=to_json_list(sw["is_swips_valid"]),
        warning=sw["warning"],
    )

    omni_out = None
    if include_omni:
        om = store.load_omni(episode_id, start_utc=start_utc, end_utc=end_utc)
        omni_out = OmniSeries(
            available=om["available"],
            time_frame=om["time_frame"],
            timestamps_utc=timestamps_to_iso(om["timestamps_utc"]),
            series={k: to_json_list(v) for k, v in om["series"].items()},
            warning=om["warning"],
        )

    oulu_out = None
    if include_oulu:
        ou = store.load_oulu(episode_id, start_utc=start_utc, end_utc=end_utc)
        oulu_out = OuluSeries(
            available=ou["available"],
            timestamps_utc=timestamps_to_iso(ou["timestamps_utc"]),
            baseline_cpm=to_json_float(ou["baseline_cpm"]),
            corrected_cpm=to_json_list(ou["corrected_cpm"]),
            uncorrected_cpm=to_json_list(ou["uncorrected_cpm"]),
            pressure_mbar=to_json_list(ou["pressure_mbar"]),
            delta_n_pct=to_json_list(ou["delta_n_pct"]),
        )

    return SpaceWeatherResponse(
        episode_id=episode_id,
        solar1=solar1_out,
        omni=omni_out,
        oulu=oulu_out,
    )


@router.get("/geomag", response_model=GeomagTimeseriesResponse)
def get_geomag_timeseries(
    store: StoreDep,
    episode_id: str = Query(...),
    station_iaga: str = Query(...),
    coord_frame: str = Query(default="XYZ"),
    external_d0_deg: float | None = Query(default=None),
    start_utc: str | None = Query(default=None),
    end_utc: str | None = Query(default=None),
) -> GeomagTimeseriesResponse:
    """Fetch INTERMAGNET magnetometer series for one station and episode."""
    return _build_geomag_response(
        store,
        episode_id=episode_id,
        station_iaga=station_iaga,
        coord_frame=coord_frame,
        external_d0_deg=external_d0_deg,
        start_utc=start_utc,
        end_utc=end_utc,
    )


@router.get("/space-weather", response_model=SpaceWeatherResponse)
def get_space_weather_timeseries(
    store: StoreDep,
    episode_id: str = Query(...),
    solar1_shift: str = Query(default="l1_raw"),
    manual_lag_min: float | None = Query(default=None),
    ref_distance_km: float = Query(default=1_400_000.0, gt=0.0),
    smooth_window_min: int = Query(default=31, ge=1),
    include_omni: bool = Query(default=True),
    include_oulu: bool = Query(default=True),
    start_utc: str | None = Query(default=None),
    end_utc: str | None = Query(default=None),
) -> SpaceWeatherResponse:
    """Fetch SOLAR-1 (L1 or ballistic/manual shifted), OMNI (BSN), and OULU NM."""
    return _build_space_weather_response(
        store,
        episode_id=episode_id,
        solar1_shift=solar1_shift,
        manual_lag_min=manual_lag_min,
        ref_distance_km=ref_distance_km,
        smooth_window_min=smooth_window_min,
        include_omni=include_omni,
        include_oulu=include_oulu,
        start_utc=start_utc,
        end_utc=end_utc,
    )


@router.get("", response_model=CombinedTimeseriesResponse)
def get_combined_timeseries(
    store: StoreDep,
    episode_id: str = Query(...),
    station_iaga: str = Query(...),
    coord_frame: str = Query(default="XYZ"),
    external_d0_deg: float | None = Query(default=None),
    solar1_shift: str = Query(default="l1_raw"),
    manual_lag_min: float | None = Query(default=None),
    ref_distance_km: float = Query(default=1_400_000.0, gt=0.0),
    smooth_window_min: int = Query(default=31, ge=1),
    include_omni: bool = Query(default=True),
    include_oulu: bool = Query(default=True),
    start_utc: str | None = Query(default=None),
    end_utc: str | None = Query(default=None),
) -> CombinedTimeseriesResponse:
    """Fetch aligned magnetometer + space-weather panels in one synchronous request."""
    geomag = _build_geomag_response(
        store,
        episode_id=episode_id,
        station_iaga=station_iaga,
        coord_frame=coord_frame,
        external_d0_deg=external_d0_deg,
        start_utc=start_utc,
        end_utc=end_utc,
    )
    space_weather = _build_space_weather_response(
        store,
        episode_id=episode_id,
        solar1_shift=solar1_shift,
        manual_lag_min=manual_lag_min,
        ref_distance_km=ref_distance_km,
        smooth_window_min=smooth_window_min,
        include_omni=include_omni,
        include_oulu=include_oulu,
        start_utc=start_utc,
        end_utc=end_utc,
    )
    return CombinedTimeseriesResponse(geomag=geomag, space_weather=space_weather)
