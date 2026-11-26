"""Catalog REST router (/api/catalog)."""

from __future__ import annotations

from fastapi import APIRouter

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
