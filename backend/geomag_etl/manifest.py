"""SHA-256 raw inventory and manifest generation for GeoMag_Storms_Dataset_Oct-2026."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

from .config import date_to_episode_id

_IAGA_RE = re.compile(r"^([a-z0-9]{3})(\d{8})([a-z])min\.min$", re.IGNORECASE)
_SOLAR1_MAG_RE = re.compile(r"^solar1_imf_1min_(\d{8})\.csv$", re.IGNORECASE)
_SOLAR1_SWIPS_RE = re.compile(r"^solar1_swips_proton_1min_(\d{8})\.csv$", re.IGNORECASE)
_NC_MAG_RE = re.compile(r"^oe_mag-l3_solar1_s?(\d{8}).*\.nc$", re.IGNORECASE)
_NC_SWIPS_RE = re.compile(r"^oe_swips-l2_solar1_s?(\d{8}).*\.nc$", re.IGNORECASE)
_OULU_RE = re.compile(r"^OULU_NM_1min_(\d{8})-(\d{8})\.csv$", re.IGNORECASE)


def sha256_file(path: Path, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hex digest of a file via streaming reads."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _format_iso_date(yyyymmdd: str) -> str:
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


def _classify_raw_file(rel_posix: str, path: Path) -> dict[str, object]:
    name = path.name
    ext = path.suffix.lower()

    if m := _IAGA_RE.match(name):
        iaga = m.group(1).upper()
        fdate = _format_iso_date(m.group(2))
        region = path.parent.parent.name
        return {
            "source_group": "intermagnet",
            "station_iaga": iaga,
            "region": region,
            "file_date": fdate,
            "episode_id": date_to_episode_id(fdate),
            "quality_code": m.group(3).lower(),
        }
    if m := _SOLAR1_MAG_RE.match(name):
        fdate = _format_iso_date(m.group(1))
        return {
            "source_group": "solar1_mag_csv",
            "station_iaga": "SOLAR1",
            "region": "L1",
            "file_date": fdate,
            "episode_id": date_to_episode_id(fdate),
            "quality_code": None,
        }
    if m := _SOLAR1_SWIPS_RE.match(name):
        fdate = _format_iso_date(m.group(1))
        return {
            "source_group": "solar1_swips_csv",
            "station_iaga": "SOLAR1",
            "region": "L1",
            "file_date": fdate,
            "episode_id": date_to_episode_id(fdate),
            "quality_code": None,
        }
    if m := _NC_MAG_RE.match(name):
        fdate = _format_iso_date(m.group(1))
        return {
            "source_group": "solar1_netcdf_mag",
            "station_iaga": "SOLAR1",
            "region": "L1",
            "file_date": fdate,
            "episode_id": date_to_episode_id(fdate),
            "quality_code": None,
        }
    if m := _NC_SWIPS_RE.match(name):
        fdate = _format_iso_date(m.group(1))
        return {
            "source_group": "solar1_netcdf_swips",
            "station_iaga": "SOLAR1",
            "region": "L1",
            "file_date": fdate,
            "episode_id": date_to_episode_id(fdate),
            "quality_code": None,
        }
    if m := _OULU_RE.match(name):
        fdate = _format_iso_date(m.group(1))
        return {
            "source_group": "oulu_nm",
            "station_iaga": "OULU",
            "region": "Europe",
            "file_date": fdate,
            "episode_id": date_to_episode_id(fdate),
            "quality_code": None,
        }
    if "OMNI_1min" in rel_posix:
        return {
            "source_group": "omni_1min" if ext == ".asc" else "documentation",
            "station_iaga": "OMNI" if ext == ".asc" else None,
            "region": "BSN" if ext == ".asc" else None,
            "file_date": None,
            "episode_id": None,
            "quality_code": None,
        }
    return {
        "source_group": "documentation",
        "station_iaga": None,
        "region": None,
        "file_date": None,
        "episode_id": None,
        "quality_code": None,
    }


def scan_raw_manifest(raw_root: Path) -> pd.DataFrame:
    """Recursively index all files in raw_root with deterministic ordering and SHA-256."""
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_root}")

    files = sorted(
        (p for p in raw_root.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(raw_root).as_posix(),
    )
    records: list[dict[str, object]] = []
    for path in files:
        rel_posix = path.relative_to(raw_root).as_posix()
        meta = _classify_raw_file(rel_posix, path)
        records.append(
            {
                "relative_path": rel_posix,
                "source_group": meta["source_group"],
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "station_iaga": meta["station_iaga"],
                "region": meta["region"],
                "file_date": meta["file_date"],
                "episode_id": meta["episode_id"],
                "quality_code": meta["quality_code"],
                "row_count": None,
                "valid_pct": None,
                "file_lat_deg": None,
                "file_lon_deg": None,
            }
        )

    df = pd.DataFrame.from_records(records)
    df["row_count"] = df["row_count"].astype("Int64")
    df["valid_pct"] = df["valid_pct"].astype("float64")
    df["file_lat_deg"] = df["file_lat_deg"].astype("float64")
    df["file_lon_deg"] = df["file_lon_deg"].astype("float64")
    return df


def enrich_manifest(
    manifest_df: pd.DataFrame,
    file_stats: dict[str, dict[str, object]],
) -> pd.DataFrame:
    """Merge per-file parsed statistics (row_count, valid_pct, coords) into manifest."""
    enriched = manifest_df.copy()
    for rel_path, stats in file_stats.items():
        mask = enriched["relative_path"] == rel_path
        if not mask.any():
            continue
        for key in ("row_count", "valid_pct", "file_lat_deg", "file_lon_deg"):
            if key in stats and stats[key] is not None:
                enriched.loc[mask, key] = stats[key]
    return enriched
