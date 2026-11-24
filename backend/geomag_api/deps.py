"""FastAPI request dependency helpers and Annotated dependency aliases."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from .cache import CacheManager
from .config import ApiSettings
from .data_store import DataStore
from .jobs import JobManager
from .security import AuthState


def get_settings(request: Request) -> ApiSettings:
    return request.app.state.settings


def get_auth(request: Request) -> AuthState:
    return request.app.state.auth


def get_store(request: Request) -> DataStore:
    return request.app.state.store


def get_cache(request: Request) -> CacheManager:
    return request.app.state.cache


def get_job_manager(request: Request) -> JobManager:
    return request.app.state.job_manager


SettingsDep = Annotated[ApiSettings, Depends(get_settings)]
AuthDep = Annotated[AuthState, Depends(get_auth)]
StoreDep = Annotated[DataStore, Depends(get_store)]
CacheDep = Annotated[CacheManager, Depends(get_cache)]
JobManagerDep = Annotated[JobManager, Depends(get_job_manager)]
