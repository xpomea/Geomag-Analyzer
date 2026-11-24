"""Static configuration, schemas, and reference metadata for ETL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO_ROOT / "GeoMag_Storms_Dataset_Oct-2026"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"

BASELINE_NORM_THRESHOLD_NT = 10000.0
INTERMAGNET_FILL_VALUES = (99999.0, 88888.0)

SWIPS_DEFAULT_ALLOWED_FLAGS = (0,)
SWIPS_PHYSICAL_LIMITS = {
    "proton_n_cm3": (0.0, 100.0),      # exclusive lower (> 0), inclusive upper
    "proton_T_K": (1000.0, 3_000_000.0),
    "vx_gse_kms": (-2500.0, -150.0),
    "pdyn_nPa": (0.0, 50.0),           # exclusive lower (> 0), inclusive upper
}
SWIPS_FLAGS_WARNING = (
    "SOLAR-1 SWiPS flag bits {0,1,2,3,8,10,11,16} are undocumented in NOAA README "
    "and NetCDF metadata; default ETL filters flags == 0 plus physical bounds."
)

OULU_STATION_META = {
    "iaga_code": "OULU",
    "station_name": "Oulu Neutron Monitor (9-NM64)",
    "station_type": "neutron_monitor",
    "region": "Europe",
    "lat_deg": 65.0544,
    "lon_deg": 25.4681,
    "elev_m": 15.0,
    "orig_format": "CSV",
    "has_baseline": True,
    "mean_vec_norm_nT": None,
    "source_of_data": "Sodankyla Geophysical Observatory / University of Oulu",
    "cutoff_rigidity_gv": 0.81,
    "barometric_coeff": -0.74,
}


@dataclass(frozen=True)
class EpisodeSpec:
    """Definition of one contiguous geomagnetic storm episode."""

    episode_id: str
    label: str
    dates: tuple[str, ...]
    start_utc: str
    end_utc: str
    n_minutes: int
    n_days: int
    omni_available: bool


EPISODES: tuple[EpisodeSpec, ...] = (
    EpisodeSpec(
        episode_id="storm_20261004_05",
        label="2026-10-04 .. 2026-10-05",
        dates=("2026-10-04", "2026-10-05"),
        start_utc="2026-10-04T00:00:00Z",
        end_utc="2026-10-05T23:59:00Z",
        n_minutes=2880,
        n_days=2,
        omni_available=True,
    ),
    EpisodeSpec(
        episode_id="storm_20261009_11",
        label="2026-10-09 .. 2026-10-11",
        dates=("2026-10-09", "2026-10-10", "2026-10-11"),
        start_utc="2026-10-09T00:00:00Z",
        end_utc="2026-10-11T23:59:00Z",
        n_minutes=4320,
        n_days=3,
        omni_available=True,
    ),
    EpisodeSpec(
        episode_id="storm_20261021_23",
        label="2026-10-21 .. 2026-10-23",
        dates=("2026-10-21", "2026-10-22", "2026-10-23"),
        start_utc="2026-10-21T00:00:00Z",
        end_utc="2026-10-23T23:59:00Z",
        n_minutes=4320,
        n_days=3,
        omni_available=False,
    ),
    EpisodeSpec(
        episode_id="storm_20261030_31",
        label="2026-10-30 .. 2026-10-31",
        dates=("2026-10-30", "2026-10-31"),
        start_utc="2026-10-30T00:00:00Z",
        end_utc="2026-10-31T23:59:00Z",
        n_minutes=2880,
        n_days=2,
        omni_available=False,
    ),
)

_DATE_TO_EPISODE: dict[str, str] = {
    d: ep.episode_id for ep in EPISODES for d in ep.dates
}
_EPISODE_BY_ID: dict[str, EpisodeSpec] = {ep.episode_id: ep for ep in EPISODES}


def date_to_episode_id(date_str: str) -> str | None:
    """Map YYYY-MM-DD or YYYYMMDD date string to an episode_id."""
    normalized = date_str.strip()
    if len(normalized) == 8 and normalized.isdigit():
        normalized = f"{normalized[:4]}-{normalized[4:6]}-{normalized[6:8]}"
    return _DATE_TO_EPISODE.get(normalized)


def get_episode_spec(episode_id: str) -> EpisodeSpec:
    """Return EpisodeSpec by ID or raise KeyError."""
    return _EPISODE_BY_ID[episode_id]


# 46 fixed-width columns per OMNI HRO_format.txt:
# (2I4,4I3,3I4,2I7,F6.2,I7, 8F8.2,4F8.1,F7.2,F9.0,F6.2,2F7.2,F6.1,6F8.2,7I6,F7.2, F5.1)
# Tuple format: (column_name, char_width, dtype_kind ['int'|'float'], per_column_fill_value)
OMNI_HRO_COLUMNS: tuple[tuple[str, int, str, int | float | None], ...] = (
    ("year", 4, "int", None),
    ("doy", 4, "int", None),
    ("hour", 3, "int", None),
    ("minute", 3, "int", None),
    ("imf_spacecraft_id", 3, "int", 99),
    ("sw_spacecraft_id", 3, "int", 99),
    ("n_imf_points", 4, "int", 999),
    ("n_plasma_points", 4, "int", 999),
    ("percent_interp", 4, "int", 999),
    ("timeshift_sec", 7, "int", 999999),
    ("rms_timeshift_sec", 7, "int", 999999),
    ("rms_pfn", 6, "float", 99.99),
    ("dbot1_sec", 7, "int", 999999),
    ("b_mag_nT", 8, "float", 9999.99),
    ("bx_gse_nT", 8, "float", 9999.99),
    ("by_gse_nT", 8, "float", 9999.99),
    ("bz_gse_nT", 8, "float", 9999.99),
    ("by_gsm_nT", 8, "float", 9999.99),
    ("bz_gsm_nT", 8, "float", 9999.99),
    ("rms_b_scalar_nT", 8, "float", 9999.99),
    ("rms_b_vector_nT", 8, "float", 9999.99),
    ("flow_speed_kms", 8, "float", 99999.9),
    ("vx_gse_kms", 8, "float", 99999.9),
    ("vy_gse_kms", 8, "float", 99999.9),
    ("vz_gse_kms", 8, "float", 99999.9),
    ("proton_n_cm3", 7, "float", 999.99),
    ("proton_T_K", 9, "float", 9999999.0),
    ("pdyn_nPa", 6, "float", 99.99),
    ("electric_field_mVm", 7, "float", 999.99),
    ("plasma_beta", 7, "float", 999.99),
    ("alfven_mach", 6, "float", 999.9),
    ("sc_x_gse_re", 8, "float", 9999.99),
    ("sc_y_gse_re", 8, "float", 9999.99),
    ("sc_z_gse_re", 8, "float", 9999.99),
    ("bsn_x_gse_re", 8, "float", 9999.99),
    ("bsn_y_gse_re", 8, "float", 9999.99),
    ("bsn_z_gse_re", 8, "float", 9999.99),
    ("ae_nT", 6, "int", 99999),
    ("al_nT", 6, "int", 99999),
    ("au_nT", 6, "int", 99999),
    ("sym_d_nT", 6, "int", 99999),
    ("sym_h_nT", 6, "int", 99999),
    ("asy_d_nT", 6, "int", 99999),
    ("asy_h_nT", 6, "int", 99999),
    ("pc_n_index", 7, "float", 999.99),
    ("magnetosonic_mach", 5, "float", 99.9),
)
OMNI_LINE_WIDTH = sum(col[1] for col in OMNI_HRO_COLUMNS)
