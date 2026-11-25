"""Pydantic v2 OpenAPI schemas for all geomag_api endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    core_version: str
    catalog_stations: int
    catalog_episodes: int
    require_auth: bool
    can_shutdown: bool = False


class BootstrapResponse(BaseModel):
    authenticated: bool
    session_token: str


class SessionResponse(BaseModel):
    authenticated: bool
    session_token: str | None = None
    require_auth: bool


class ShutdownResponse(BaseModel):
    status: str
    message: str


class EpisodeInfo(BaseModel):
    episode_id: str
    label: str
    start_utc: str
    end_utc: str
    n_minutes: int
    n_days: int
    omni_available: bool
    oulu_baseline_cpm: float | None = None


class StationInfo(BaseModel):
    iaga_code: str
    station_name: str
    station_type: str
    region: str
    lat_deg: float
    lon_deg: float
    elev_m: float
    orig_format: str
    has_baseline: bool
    mean_vec_norm_nT: float | None = None
    days_present: int
    source_of_data: str
    cutoff_rigidity_gv: float | None = None
    barometric_coeff: float | None = None


class AvailabilityRow(BaseModel):
    station_iaga: str
    file_date: str
    episode_id: str
    quality_code: str
    orig_format: str
    valid_vector_count: int
    valid_vector_pct: float
    f_missing_count: int
    f_only_missing_count: int
    interpolated_count: int
    max_gap_min: int
    despiked_count: int
    file_lat_deg: float
    file_lon_deg: float


class CatalogResponse(BaseModel):
    episodes: list[EpisodeInfo]
    stations: list[StationInfo]
    availability: list[AvailabilityRow]
    metadata: dict[str, Any]


class GeomagTimeseriesResponse(BaseModel):
    episode_id: str
    station_iaga: str
    station_meta: StationInfo
    available: bool
    coord_frame: str
    d0_deg: float
    valid_pct: float
    timestamps_utc: list[str]
    x_nT: list[float | None]
    y_nT: list[float | None]
    z_nT: list[float | None]
    f_scalar_nT: list[float | None]
    is_valid_xyz: list[bool]
    is_interpolated: list[bool]
    was_despiked: list[bool]
    warnings: list[str] = Field(default_factory=list)


class Solar1Series(BaseModel):
    timestamps_utc: list[str]
    time_frame: str
    shift_mode: str
    ref_distance_km: float
    propagation_delay_min: list[float | None]
    series: dict[str, list[float | None]]
    swips_flags: list[int | None]
    is_swips_valid: list[bool]
    warning: str


class OmniSeries(BaseModel):
    available: bool
    time_frame: str
    timestamps_utc: list[str]
    series: dict[str, list[float | None]]
    warning: str | None = None


class OuluSeries(BaseModel):
    available: bool
    timestamps_utc: list[str]
    baseline_cpm: float | None = None
    corrected_cpm: list[float | None]
    uncorrected_cpm: list[float | None]
    pressure_mbar: list[float | None]
    delta_n_pct: list[float | None]


class SpaceWeatherResponse(BaseModel):
    episode_id: str
    solar1: Solar1Series
    omni: OmniSeries | None = None
    oulu: OuluSeries | None = None


class CombinedTimeseriesResponse(BaseModel):
    geomag: GeomagTimeseriesResponse
    space_weather: SpaceWeatherResponse


class PolarizationRequest(BaseModel):
    episode_id: str
    station_iaga: str
    method: Literal["welch", "multitaper", "variant_b_fft", "variant_b_sos"] = "welch"
    band: str = "Pc5"
    t_max_pc6_sec: float = Field(default=1800.0, gt=600.0)
    f_min_hz: float | None = Field(default=None, gt=0.0)
    f_max_hz: float | None = Field(default=None, gt=0.0)
    coord_frame: Literal["XYZ", "XmYm"] = "XYZ"
    external_d0_deg: float | None = None
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    spectrum_type: Literal["flat", "red"] = "flat"
    include_3d: bool = True
    include_dynamic: bool = True
    include_filtered: bool = False
    include_spectrogram: bool = False
    L: int = Field(default=128, ge=16)
    M: int = Field(default=64, ge=1)
    window_samples: int = Field(default=360, ge=32)
    step_samples: int = Field(default=30, ge=1)
    nw: float = Field(default=2.5, gt=0.5)
    k_tapers: int | None = Field(default=None, ge=1)
    order_n: int = Field(default=3, ge=1, le=6)
    oversample_m: int = Field(default=4, ge=1, le=16)
    oversample_method: Literal["fourier", "kaiser"] = "fourier"
    trim_hours: float | None = Field(default=None, ge=0.0)
    start_utc: str | None = None
    end_utc: str | None = None


class Stokes2DOut(BaseModel):
    I: float | None = None  # noqa: E741
    Q: float | None = None
    U: float | None = None
    V: float | None = None
    A: float | None = None
    R: float | None = None
    p: float | None = None
    p_L: float | None = None
    p_C: float | None = None
    psi_deg: float | None = None
    chi_deg: float | None = None
    epsilon: float | None = None
    gamma2: float | None = None
    arg_sxy_deg: float | None = None
    n_eff: float | None = None
    p_crit: float | None = None
    gamma2_crit: float | None = None
    is_significant_p: bool | None = None
    is_significant_gamma2: bool | None = None


class Polarization3DOut(BaseModel):
    P: float | None = None
    theta_deg: float | None = None
    phi_n_deg: float | None = None
    b_over_a: float | None = None
    s: float | None = None
    tr_S: float | None = None
    n_v: list[float | None]
    eigenvalues_re_S: list[float | None]


class DynamicPolarizationOut(BaseModel):
    timestamps_utc: list[str]
    I: list[float | None]  # noqa: E741
    p: list[float | None]
    p_crit: list[float | None]
    psi_deg: list[float | None]
    chi_deg: list[float | None]
    epsilon: list[float | None]
    gamma2: list[float | None]
    gamma2_crit: list[float | None] = Field(default_factory=list)
    n_eff: list[float | None] = Field(default_factory=list)
    is_significant_p: list[bool]
    is_valid: list[bool]


class FilteredSeriesOut(BaseModel):
    timestamps_utc: list[str]
    x_filt: list[float | None]
    y_filt: list[float | None]
    z_filt: list[float | None]
    x_amp: list[float | None]
    y_amp: list[float | None]
    usable_mask: list[bool]


class SpectrogramOut(BaseModel):
    timestamps_utc: list[str]
    freqs_hz: list[float]
    power_total: list[list[float | None]]
    power_x: list[list[float | None]]
    power_y: list[list[float | None]]


class PolarizationResponse(BaseModel):
    episode_id: str
    station_iaga: str
    method: str
    band: str
    coord_frame: str
    d0_deg: float
    is_valid: bool
    n_eff: float | None = None
    trim_hours: float | None = None
    stokes_2d: Stokes2DOut | None = None
    pol_3d: Polarization3DOut | None = None
    dynamic: DynamicPolarizationOut | None = None
    filtered: FilteredSeriesOut | None = None
    spectrogram: SpectrogramOut | None = None
    warnings: list[str] = Field(default_factory=list)


class NetworkSnapshotRequest(BaseModel):
    episode_id: str
    station_iagas: list[str] | None = None
    band: str = "Pc5"
    t_max_pc6_sec: float = Field(default=1800.0, gt=600.0)
    coord_frame: Literal["XYZ", "XmYm"] = "XYZ"
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    spectrum_type: Literal["flat", "red"] = "flat"
    L: int = Field(default=128, ge=16)
    M: int = Field(default=64, ge=1)
    start_utc: str | None = None
    end_utc: str | None = None


class NetworkStationSnapshot(BaseModel):
    station_iaga: str
    lat_deg: float
    lon_deg: float
    has_baseline: bool
    available: bool
    is_valid: bool
    d0_deg: float = 0.0
    stokes_2d: Stokes2DOut | None = None
    warning: str | None = None


class NetworkSnapshotResponse(BaseModel):
    episode_id: str
    band: str
    coord_frame: str
    stations: list[NetworkStationSnapshot]


class WelchPairRequest(BaseModel):
    episode_id: str
    station_i: str
    station_j: str
    component: Literal["X", "Y", "Z", "Xm", "Ym"] = "X"
    coord_frame: Literal["XYZ", "XmYm"] = "XYZ"
    external_d0_deg_i: float | None = None
    external_d0_deg_j: float | None = None
    band: str = "Pc5"
    t_max_pc6_sec: float = Field(default=1800.0, gt=600.0)
    f_min_hz: float | None = Field(default=None, gt=0.0)
    f_max_hz: float | None = Field(default=None, gt=0.0)
    L: int = Field(default=128, ge=16)
    M: int = Field(default=64, ge=1)
    window_samples: int = Field(default=360, ge=32)
    step_samples: int = Field(default=30, ge=1)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    min_common_segments: int = Field(default=2, ge=1)
    spectrum_type: Literal["flat", "red"] = "flat"
    window_index: int | None = Field(default=None, ge=0)
    min_delta_lon_deg: float = Field(default=3.0, ge=0.0)
    unwrap_time: bool = True


class AzimuthalOut(BaseModel):
    m: list[float | None]
    m_alias: float | None = None
    delta_lon_deg: float
    is_meridional_guarded: bool


class WelchPairResponse(BaseModel):
    episode_id: str
    station_i: str
    station_j: str
    component: str
    coord_frame: str
    band: str
    available: bool
    cache_hits: dict[str, bool]
    window_timestamps_utc: list[str]
    freqs_hz: list[float]
    K_ij: list[int]
    n_eff: list[float | None]
    gamma2: list[float | None]
    gamma2_crit: list[float | None]
    phase_deg: list[float | None]
    phase_deg_masked: list[float | None]
    is_significant: list[bool]
    is_valid: list[bool]
    gamma2_tf: list[list[float | None]]
    gamma2_crit_tf: list[float | None]
    phase_deg_tf: list[list[float | None]] = Field(default_factory=list)
    phase_deg_tf_masked: list[list[float | None]]
    azimuthal: AzimuthalOut | None = None
    warnings: list[str] = Field(default_factory=list)


class WelchMatrixRequest(BaseModel):
    episode_id: str
    station_iagas: list[str] | None = None
    component: Literal["X", "Y", "Z", "Xm", "Ym"] = "X"
    coord_frame: Literal["XYZ", "XmYm"] = "XYZ"
    band: str = "Pc5"
    t_max_pc6_sec: float = Field(default=1800.0, gt=600.0)
    L: int = Field(default=128, ge=16)
    M: int = Field(default=64, ge=1)
    window_samples: int = Field(default=360, ge=32)
    step_samples: int = Field(default=30, ge=1)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    min_common_segments: int = Field(default=2, ge=1)
    spectrum_type: Literal["flat", "red"] = "flat"
    window_index: int | None = Field(default=None, ge=0)


class WelchMatrixResponse(BaseModel):
    episode_id: str
    component: str
    coord_frame: str
    band: str
    station_codes: list[str]
    window_timestamps_utc: list[str]
    gamma2: list[list[list[float | None]]]
    gamma2_crit: list[list[list[float | None]]]
    phase_deg: list[list[list[float | None]]] = Field(default_factory=list)
    phase_deg_masked: list[list[list[float | None]]]
    K_ij: list[list[list[int]]]
    is_significant: list[list[list[bool]]]
    cache_hits: dict[str, bool]
    skipped_stations: dict[str, str] = Field(default_factory=dict)
    elapsed_ms: float


class MeridionalFLRRequest(BaseModel):
    episode_id: str
    station_chain: list[str] = Field(min_length=2)
    component: Literal["X", "Y", "Z", "Xm", "Ym"] = "X"
    coord_frame: Literal["XYZ", "XmYm"] = "XYZ"
    band: str = "Pc5"
    t_max_pc6_sec: float = Field(default=1800.0, gt=600.0)
    L: int = Field(default=128, ge=16)
    M: int = Field(default=64, ge=1)
    window_samples: int = Field(default=360, ge=32)
    step_samples: int = Field(default=30, ge=1)
    window_index: int = Field(default=0, ge=0)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    min_common_segments: int = Field(default=2, ge=1)
    spectrum_type: Literal["flat", "red"] = "flat"


class MeridionalFLRResponse(BaseModel):
    episode_id: str
    band: str
    window_timestamp_utc: str
    station_codes: list[str]
    latitudes_deg: list[float]
    band_power: list[float | None]
    amplitude: list[float | None]
    pair_coherence: list[float | None]
    pair_gamma2_crit: list[float | None]
    pair_phase_deg: list[float | None]
    pair_is_significant: list[bool]
    cumulative_phase_deg: list[float | None]
    phase_gradient_deg_per_lat: list[float | None]
    peak_station: str
    peak_latitude_deg: float | None = None


class WTCJobRequest(BaseModel):
    episode_id: str
    station_i: str
    station_j: str
    component: Literal["X", "Y", "Z", "Xm", "Ym"] = "X"
    coord_frame: Literal["XYZ", "XmYm"] = "XYZ"
    external_d0_deg_i: float | None = None
    external_d0_deg_j: float | None = None
    period_min_sec: float = Field(default=120.0, ge=60.0)
    period_max_sec: float = Field(default=3600.0, gt=120.0)
    dj: float = Field(default=1.0 / 12.0, gt=0.0)
    omega0: float = Field(default=6.0, gt=0.0)
    gap_policy: Literal["subsegments", "gap_coi"] = "subsegments"
    min_subsegment_samples: int = Field(default=120, ge=32)
    n_surrogates: int = Field(default=300, ge=0)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    seed: int = 42
    start_utc: str | None = None
    end_utc: str | None = None


class JobStatusResponse(BaseModel):
    job_id: str
    job_type: str
    cache_key: str
    status: str
    stage: str
    progress_pct: float
    completed_steps: int
    total_steps: int
    eta_sec: float | None = None
    cache_hit: bool
    result_path: str | None = None
    error_message: str | None = None
    params: dict[str, Any]
    created_at: str
    updated_at: str


class WTCResultResponse(BaseModel):
    job_id: str
    cache_key: str
    episode_id: str
    station_i: str
    station_j: str
    component: str
    coord_frame: str
    timestamps_utc: list[str]
    scales: list[float]
    periods_sec: list[float]
    coi_periods_sec: list[float | None]
    inside_coi: list[list[bool]]
    xwt_power: list[list[float | None]]
    xwt_phase_deg: list[list[float | None]]
    wtc_r2: list[list[float | None]]
    wtc_phase_deg: list[list[float | None]]
    wtc_sig95: list[float | None] | None = None
    wtc_is_significant: list[list[bool]]
    alpha1_i: float
    alpha1_j: float
    gap_policy: str
    subsegments: list[list[int]]
    n_surrogates: int
    alpha: float


class CacheEntryOut(BaseModel):
    cache_key: str
    category: str
    file_path: str
    source: str
    size_bytes: int
    params: dict[str, Any]
    created_at: str
