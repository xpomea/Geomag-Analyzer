"""NaN-safe JSON serialization helpers for scientific NumPy and pandas structures."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def to_json_float(val: Any) -> float | None:
    """Convert numeric scalar to finite float or None (JSON null) if NaN/Inf/None."""
    if val is None or pd.isna(val):
        return None
    f_val = float(val)
    return f_val if math.isfinite(f_val) else None


def _sanitize_nested(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, list):
        return [_sanitize_nested(x) for x in obj]
    if isinstance(obj, np.floating):
        f_val = float(obj)
        return f_val if math.isfinite(f_val) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def to_json_list(arr: np.ndarray | pd.Series | list[Any]) -> list[Any]:
    """Convert a 1D/2D/3D array to pure JSON-safe nested lists with NaN/Inf -> None."""
    np_arr = np.asarray(arr)
    if np_arr.dtype == bool:
        return np_arr.tolist()
    if np.issubdtype(np_arr.dtype, np.integer):
        return np_arr.tolist()
    if np.issubdtype(np_arr.dtype, np.floating):
        if np_arr.ndim == 0:
            v = to_json_float(np_arr.item())
            return [v]
        if np_arr.ndim == 1:
            return [x if math.isfinite(x) else None for x in np_arr.astype(np.float64).tolist()]
        if np_arr.ndim == 2:
            raw = np_arr.astype(np.float64).tolist()
            return [[x if math.isfinite(x) else None for x in row] for row in raw]
        return _sanitize_nested(np_arr.astype(np.float64).tolist())
    return _sanitize_nested(np_arr.tolist())


def timestamps_to_iso(ts: pd.DatetimeIndex | pd.Series) -> list[str]:
    """Convert UTC timestamps to canonical YYYY-MM-DDTHH:MM:SSZ strings."""
    idx = pd.DatetimeIndex(ts).tz_convert("UTC")
    return idx.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
