"""Catalog REST router (/api/catalog)."""

from __future__ import annotations

from fastapi import APIRouter, Query

from .deps import StoreDep
from .schemas import AvailabilityRow, CatalogResponse, EpisodeInfo, StationInfo

router = APIRouter(prefix="/api/catalog", tags=["catalog"])


@router.get("", response_model=CatalogResponse)
def get_full_catalog(store: StoreDep) -> CatalogResponse:
    """Return complete static metadata: episodes, 50 stations, availability, and dataset info."""
    return CatalogResponse(
        episodes=[EpisodeInfo(**row) for row in store.get_episodes()],
        stations=[StationInfo(**row) for row in store.get_stations()],
        availability=[AvailabilityRow(**row) for row in store.get_availability()],
        metadata=store.get_metadata(),
    )


@router.get("/episodes", response_model=list[EpisodeInfo])
def list_episodes(store: StoreDep) -> list[EpisodeInfo]:
    """List storm episodes."""
    return [EpisodeInfo(**row) for row in store.get_episodes()]


@router.get("/stations", response_model=list[StationInfo])
def list_stations(
    store: StoreDep,
    station_type: str | None = Query(default=None),
) -> list[StationInfo]:
    """List stations optionally filtered by station_type ('magnetometer' or 'neutron_monitor')."""
    return [StationInfo(**row) for row in store.get_stations(station_type=station_type)]


@router.get("/availability", response_model=list[AvailabilityRow])
def list_availability(
    store: StoreDep,
    episode_id: str | None = Query(default=None),
    station_iaga: str | None = Query(default=None),
) -> list[AvailabilityRow]:
    """Return station x file_date data availability and QC matrix."""
    return [
        AvailabilityRow(**row)
        for row in store.get_availability(episode_id=episode_id, station_iaga=station_iaga)
    ]
