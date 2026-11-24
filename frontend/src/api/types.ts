/** Exact TypeScript equivalents of backend/geomag_api/schemas.py */

export type PolarizationMethod =
  | "welch"
  | "multitaper"
  | "variant_b_fft"
  | "variant_b_sos";

export type BandType = "Pc5" | "Pc6";

export type CoordFrame = "XYZ" | "XmYm";

export type CrossComponent = "X" | "Y" | "Z" | "Xm" | "Ym";

export type Solar1ShiftMode = "l1_raw" | "ballistic_auto" | "manual";

export type SpectrumType = "flat" | "red";

export type OversampleMethod = "fourier" | "kaiser";

export type GapPolicy = "subsegments" | "gap_coi";

export type JobStatus =
  | "queued"
  | "running"
  | "completed"
  | "cached"
  | "cancelled"
  | "failed";

export interface HealthResponse {
  status: string;
  core_version: string;
  catalog_stations: number;
  catalog_episodes: number;
  require_auth: boolean;
  can_shutdown?: boolean;
}

export interface BootstrapResponse {
  authenticated: boolean;
  session_token: string;
}

export interface SessionResponse {
  authenticated: boolean;
  session_token: string | null;
  require_auth: boolean;
}

export interface ShutdownResponse {
  status: string;
  message: string;
}

export interface EpisodeInfo {
  episode_id: string;
  label: string;
  start_utc: string;
  end_utc: string;
  n_minutes: number;
  n_days: number;
  omni_available: boolean;
  oulu_baseline_cpm: number | null;
}

export interface StationInfo {
  iaga_code: string;
  station_name: string;
  station_type: string;
  region: string;
  lat_deg: number;
  lon_deg: number;
  elev_m: number;
  orig_format: string;
  has_baseline: boolean;
  mean_vec_norm_nT: number | null;
  days_present: number;
  source_of_data: string;
  cutoff_rigidity_gv: number | null;
  barometric_coeff: number | null;
}

export interface AvailabilityRow {
  station_iaga: string;
  file_date: string;
  episode_id: string;
  quality_code: string;
  orig_format: string;
  valid_vector_count: number;
  valid_vector_pct: number;
  f_missing_count: number;
  f_only_missing_count: number;
  interpolated_count: number;
  max_gap_min: number;
  despiked_count: number;
  file_lat_deg: number;
  file_lon_deg: number;
}

export interface CatalogResponse {
  episodes: EpisodeInfo[];
  stations: StationInfo[];
  availability: AvailabilityRow[];
  metadata: Record<string, unknown>;
}

export interface GeomagTimeseriesResponse {
  episode_id: string;
  station_iaga: string;
  station_meta: StationInfo;
  available: boolean;
  coord_frame: string;
  d0_deg: number;
  valid_pct: number;
  timestamps_utc: string[];
  x_nT: (number | null)[];
  y_nT: (number | null)[];
  z_nT: (number | null)[];
  f_scalar_nT: (number | null)[];
  is_valid_xyz: boolean[];
  is_interpolated: boolean[];
  was_despiked: boolean[];
  warnings: string[];
}

export interface Solar1Series {
  timestamps_utc: string[];
  time_frame: string;
  shift_mode: string;
  ref_distance_km: number;
  propagation_delay_min: (number | null)[];
  series: Record<string, (number | null)[]>;
  swips_flags: (number | null)[];
  is_swips_valid: boolean[];
  warning: string;
}

export interface OmniSeries {
  available: boolean;
  time_frame: string;
  timestamps_utc: string[];
  series: Record<string, (number | null)[]>;
  warning: string | null;
}

export interface OuluSeries {
  available: boolean;
  timestamps_utc: string[];
  baseline_cpm: number | null;
  corrected_cpm: (number | null)[];
  uncorrected_cpm: (number | null)[];
  pressure_mbar: (number | null)[];
  delta_n_pct: (number | null)[];
}

export interface SpaceWeatherResponse {
  episode_id: string;
  solar1: Solar1Series;
  omni: OmniSeries | null;
  oulu: OuluSeries | null;
}

export interface CombinedTimeseriesResponse {
  geomag: GeomagTimeseriesResponse;
  space_weather: SpaceWeatherResponse;
}

export interface PolarizationRequest {
  episode_id: string;
  station_iaga: string;
  method?: PolarizationMethod;
  band?: string;
  t_max_pc6_sec?: number;
  f_min_hz?: number | null;
  f_max_hz?: number | null;
  coord_frame?: CoordFrame;
  external_d0_deg?: number | null;
  alpha?: number;
  spectrum_type?: SpectrumType;
  include_3d?: boolean;
  include_dynamic?: boolean;
  include_filtered?: boolean;
  include_spectrogram?: boolean;
  L?: number;
  M?: number;
  window_samples?: number;
  step_samples?: number;
  nw?: number;
  k_tapers?: number | null;
  order_n?: number;
  oversample_m?: number;
  oversample_method?: OversampleMethod;
  trim_hours?: number | null;
  start_utc?: string | null;
  end_utc?: string | null;
}

export interface Stokes2DOut {
  I: number | null;
  Q: number | null;
  U: number | null;
  V: number | null;
  A: number | null;
  R: number | null;
  p: number | null;
  p_L: number | null;
  p_C: number | null;
  psi_deg: number | null;
  chi_deg: number | null;
  epsilon: number | null;
  gamma2: number | null;
  arg_sxy_deg: number | null;
  n_eff: number | null;
  p_crit: number | null;
  gamma2_crit: number | null;
  is_significant_p: boolean | null;
  is_significant_gamma2: boolean | null;
}

export interface Polarization3DOut {
  P: number | null;
  theta_deg: number | null;
  phi_n_deg: number | null;
  b_over_a: number | null;
  s: number | null;
  tr_S: number | null;
  n_v: (number | null)[];
  eigenvalues_re_S: (number | null)[];
}

export interface DynamicPolarizationOut {
  timestamps_utc: string[];
  I: (number | null)[];
  p: (number | null)[];
  p_crit: (number | null)[];
  psi_deg: (number | null)[];
  chi_deg: (number | null)[];
  epsilon: (number | null)[];
  gamma2: (number | null)[];
  gamma2_crit: (number | null)[];
  n_eff: (number | null)[];
  is_significant_p: boolean[];
  is_valid: boolean[];
}

export interface FilteredSeriesOut {
  timestamps_utc: string[];
  x_filt: (number | null)[];
  y_filt: (number | null)[];
  z_filt: (number | null)[];
  x_amp: (number | null)[];
  y_amp: (number | null)[];
  usable_mask: boolean[];
}

export interface SpectrogramOut {
  timestamps_utc: string[];
  freqs_hz: number[];
  power_total: (number | null)[][];
  power_x: (number | null)[][];
  power_y: (number | null)[][];
}

export interface PolarizationResponse {
  episode_id: string;
  station_iaga: string;
  method: string;
  band: string;
  coord_frame: string;
  d0_deg: number;
  is_valid: boolean;
  n_eff: number | null;
  trim_hours: number | null;
  stokes_2d: Stokes2DOut | null;
  pol_3d: Polarization3DOut | null;
  dynamic: DynamicPolarizationOut | null;
  filtered: FilteredSeriesOut | null;
  spectrogram: SpectrogramOut | null;
  warnings: string[];
}

export interface NetworkSnapshotRequest {
  episode_id: string;
  station_iagas?: string[] | null;
  band?: string;
  t_max_pc6_sec?: number;
  coord_frame?: CoordFrame;
  alpha?: number;
  spectrum_type?: SpectrumType;
  L?: number;
  M?: number;
  start_utc?: string | null;
  end_utc?: string | null;
}

export interface NetworkStationSnapshot {
  station_iaga: string;
  lat_deg: number;
  lon_deg: number;
  has_baseline: boolean;
  available: boolean;
  is_valid: boolean;
  stokes_2d: Stokes2DOut | null;
  warning: string | null;
}

export interface NetworkSnapshotResponse {
  episode_id: string;
  band: string;
  coord_frame: string;
  stations: NetworkStationSnapshot[];
}

export interface WelchPairRequest {
  episode_id: string;
  station_i: string;
  station_j: string;
  component?: CrossComponent;
  coord_frame?: CoordFrame;
  external_d0_deg_i?: number | null;
  external_d0_deg_j?: number | null;
  band?: string;
  t_max_pc6_sec?: number;
  f_min_hz?: number | null;
  f_max_hz?: number | null;
  L?: number;
  M?: number;
  window_samples?: number;
  step_samples?: number;
  alpha?: number;
  min_common_segments?: number;
  spectrum_type?: SpectrumType;
  min_delta_lon_deg?: number;
  unwrap_time?: boolean;
}

export interface AzimuthalOut {
  m: (number | null)[];
  m_alias: number | null;
  delta_lon_deg: number;
  is_meridional_guarded: boolean;
}

export interface WelchPairResponse {
  episode_id: string;
  station_i: string;
  station_j: string;
  component: string;
  coord_frame: string;
  band: string;
  available: boolean;
  cache_hits: Record<string, boolean>;
  window_timestamps_utc: string[];
  freqs_hz: number[];
  K_ij: number[];
  n_eff: (number | null)[];
  gamma2: (number | null)[];
  gamma2_crit: (number | null)[];
  phase_deg: (number | null)[];
  phase_deg_masked: (number | null)[];
  is_significant: boolean[];
  is_valid: boolean[];
  gamma2_tf: (number | null)[][];
  gamma2_crit_tf: (number | null)[];
  phase_deg_tf: (number | null)[][];
  phase_deg_tf_masked: (number | null)[][];
  azimuthal: AzimuthalOut | null;
  warnings: string[];
}

export interface MeridionalFLRRequest {
  episode_id: string;
  station_chain: string[];
  component?: CrossComponent;
  coord_frame?: CoordFrame;
  band?: string;
  t_max_pc6_sec?: number;
  L?: number;
  M?: number;
  window_samples?: number;
  step_samples?: number;
  window_index?: number;
  alpha?: number;
  min_common_segments?: number;
}

export interface MeridionalFLRResponse {
  episode_id: string;
  band: string;
  window_timestamp_utc: string;
  station_codes: string[];
  latitudes_deg: number[];
  band_power: (number | null)[];
  amplitude: (number | null)[];
  pair_coherence: (number | null)[];
  pair_gamma2_crit: (number | null)[];
  pair_phase_deg: (number | null)[];
  pair_is_significant: boolean[];
  cumulative_phase_deg: (number | null)[];
  phase_gradient_deg_per_lat: (number | null)[];
  peak_station: string;
  peak_latitude_deg: number | null;
}

export interface WTCJobRequest {
  episode_id: string;
  station_i: string;
  station_j: string;
  component?: CrossComponent;
  coord_frame?: CoordFrame;
  external_d0_deg_i?: number | null;
  external_d0_deg_j?: number | null;
  period_min_sec?: number;
  period_max_sec?: number;
  dj?: number;
  omega0?: number;
  gap_policy?: GapPolicy;
  min_subsegment_samples?: number;
  n_surrogates?: number;
  alpha?: number;
  seed?: number;
  start_utc?: string | null;
  end_utc?: string | null;
}

export interface JobStatusResponse {
  job_id: string;
  job_type: string;
  cache_key: string;
  status: JobStatus | string;
  stage: string;
  progress_pct: number;
  completed_steps: number;
  total_steps: number;
  eta_sec: number | null;
  cache_hit: boolean;
  result_path: string | null;
  error_message: string | null;
  params: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface WTCResultResponse {
  job_id: string;
  cache_key: string;
  episode_id: string;
  station_i: string;
  station_j: string;
  component: string;
  coord_frame: string;
  timestamps_utc: string[];
  scales: number[];
  periods_sec: number[];
  coi_periods_sec: (number | null)[];
  inside_coi: boolean[][];
  xwt_power: (number | null)[][];
  xwt_phase_deg: (number | null)[][];
  wtc_r2: (number | null)[][];
  wtc_phase_deg: (number | null)[][];
  wtc_sig95: (number | null)[] | null;
  wtc_is_significant: boolean[][];
  alpha1_i: number;
  alpha1_j: number;
  gap_policy: string;
  subsegments: [number, number][];
  n_surrogates: number;
  alpha: number;
}

export interface CacheEntryOut {
  cache_key: string;
  category: string;
  file_path: string;
  source: string;
  size_bytes: number;
  params: Record<string, unknown>;
  created_at: string;
}
