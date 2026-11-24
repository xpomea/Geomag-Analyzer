"""Positional Fortran HRO parser for OMNI_1min data (BSN timeframe)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import OMNI_HRO_COLUMNS, OMNI_LINE_WIDTH, date_to_episode_id


def parse_omni_hro_file(path: Path) -> pd.DataFrame:
    """Parse fixed-width 299-char OMNI HRO ASCII lines with strictly per-column fill masking."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        raise ValueError(f"Empty OMNI file: {path}")

    col_names = [col[0] for col in OMNI_HRO_COLUMNS]
    slices: list[tuple[int, int]] = []
    pos = 0
    for _, width, _, _ in OMNI_HRO_COLUMNS:
        slices.append((pos, pos + width))
        pos += width

    raw_columns: dict[str, list[str]] = {name: [] for name in col_names}
    for line_no, line in enumerate(lines, start=1):
        if len(line) != OMNI_LINE_WIDTH:
            raise ValueError(
                f"OMNI line {line_no} in {path.name} has length {len(line)}, "
                f"expected {OMNI_LINE_WIDTH}"
            )
        for name, (start, end) in zip(col_names, slices, strict=True):
            raw_columns[name].append(line[start:end])

    data: dict[str, np.ndarray] = {}
    for name, _, kind, fill_val in OMNI_HRO_COLUMNS:
        if kind == "int" and fill_val is None:
            arr_i = np.array([int(tok) for tok in raw_columns[name]], dtype=np.int64)
            data[name] = arr_i
        elif kind == "int":
            vals = np.array([int(tok) for tok in raw_columns[name]], dtype=np.float64)
            vals[vals == float(fill_val)] = np.nan
            data[name] = vals
        else:
            vals = np.array([float(tok) for tok in raw_columns[name]], dtype=np.float64)
            if fill_val is not None:
                vals[np.isclose(vals, float(fill_val), rtol=0.0, atol=1e-6)] = np.nan
            data[name] = vals

    df = pd.DataFrame(data)
    base_days = pd.to_datetime(
        df["year"] * 1000 + df["doy"], format="%Y%j", utc=True
    )
    timestamps = (
        base_days
        + pd.to_timedelta(df["hour"], unit="h")
        + pd.to_timedelta(df["minute"], unit="m")
    )
    df.insert(0, "timestamp_utc", timestamps)
    episode_ids = timestamps.dt.strftime("%Y-%m-%d").map(date_to_episode_id)
    df.insert(1, "episode_id", episode_ids.astype("string"))
    df.insert(2, "time_frame", pd.Series("BSN", index=df.index, dtype="string"))
    return df


def process_omni(
    raw_root: Path,
    out_dir: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]], dict[str, object]]:
    """Parse OMNI_1min data and write omni_bsn_1min.parquet."""
    omni_dir = raw_root / "Solar Wind Data" / "SolarWind_Oct2026" / "OMNI_1min"
    asc_path = omni_dir / "omni_min_selected_days_Oct2026.asc"
    if not asc_path.exists():
        raise FileNotFoundError(f"Missing OMNI ASCII file: {asc_path}")

    df = parse_omni_hro_file(asc_path)
    sw_dir = out_dir / "solar_wind"
    sw_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(sw_dir / "omni_bsn_1min.parquet", index=False, compression="zstd")

    available_dates = sorted(df["timestamp_utc"].dt.strftime("%Y-%m-%d").unique().tolist())
    rel = asc_path.relative_to(raw_root).as_posix()
    file_stats = {
        rel: {
            "row_count": int(len(df)),
            "valid_pct": round(100.0 * float(df["bz_gsm_nT"].notna().mean()), 4),
        }
    }
    summary = {
        "rows": int(len(df)),
        "time_frame": "BSN",
        "available_dates": available_dates,
        "sym_h_valid_count": int(df["sym_h_nT"].notna().sum()),
        "ae_valid_count": int(df["ae_nT"].notna().sum()),
        "bz_gsm_valid_count": int(df["bz_gsm_nT"].notna().sum()),
    }
    return df, file_stats, summary
