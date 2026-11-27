"""Atomic JSON, YAML, NPZ, and SHA-256 helpers with explicit UTF-8 encoding."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def compute_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hex digest of a file using chunked streaming reads."""
    target = Path(path)
    hasher = hashlib.sha256()
    with target.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def to_jsonable(obj: Any, *, nan_to_null: bool = False) -> Any:
    """Recursively convert numpy/pathlib objects into JSON-serializable Python values."""
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        if not math.isfinite(val):
            if nan_to_null:
                return None
            raise ValueError(f"Non-finite float {val!r} is not allowed in strict JSON.")
        return val
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v, nan_to_null=nan_to_null) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, nan_to_null=nan_to_null) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v, nan_to_null=nan_to_null) for v in obj]
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable.")


def read_json(path: Path) -> Any:
    """Read UTF-8 JSON file."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(
    path: Path,
    payload: Any,
    *,
    indent: int = 2,
    nan_to_null: bool = False,
) -> Path:
    """Atomically write UTF-8 JSON file with strict NaN/Inf handling."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    clean_payload = to_jsonable(payload, nan_to_null=nan_to_null)
    tmp_path = target.parent / f".{target.name}.{os.getpid()}.tmp"
    try:
        text = json.dumps(clean_payload, indent=indent, ensure_ascii=False, allow_nan=False)
        tmp_path.write_text(text + "\n", encoding="utf-8")
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return target


def read_yaml(path: Path) -> dict[str, Any]:
    """Read UTF-8 YAML mapping file."""
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at {path}, got {type(data).__name__}.")
    return data


def write_yaml(path: Path, payload: Any) -> Path:
    """Atomically write UTF-8 YAML file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    clean_payload = to_jsonable(payload, nan_to_null=True)
    tmp_path = target.parent / f".{target.name}.{os.getpid()}.tmp"
    try:
        text = yaml.safe_dump(clean_payload, sort_keys=False, allow_unicode=True)
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return target


def save_npz_atomic(
    path: Path,
    arrays: dict[str, np.ndarray],
    *,
    compressed: bool = True,
) -> Path:
    """Atomically save dictionary of NumPy arrays into a `.npz` archive."""
    target = Path(path)
    if target.suffix.lower() != ".npz":
        raise ValueError(f"Target path must end with .npz, got {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    # Temp name MUST end with `.npz` so NumPy does not silently append another `.npz`.
    tmp_path = target.parent / f".{target.stem}.{os.getpid()}.tmp.npz"
    saver = np.savez_compressed if compressed else np.savez
    try:
        saver(tmp_path, **arrays)
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return target


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """Load `.npz` archive and immediately release the underlying file handle."""
    target = Path(path)
    with np.load(target, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def portable_path_str(path: Path | str, *, repo_root: Path | None = None) -> str:
    """Serialize a filesystem path portably relative to `repo_root` (or `ML_ROOT` in cloud bundles)."""
    target = Path(path).resolve()
    ml_root = Path(__file__).resolve().parents[2]
    root = Path(repo_root).resolve() if repo_root is not None else ml_root.parent
    if (root / "ml_research").is_dir():
        try:
            return target.relative_to(root).as_posix()
        except ValueError:
            pass
    try:
        return target.relative_to(ml_root).as_posix()
    except ValueError:
        return target.as_posix()


def get_git_commit(repo_root: Path | None = None) -> str | None:
    """Return short git HEAD commit hash if available, otherwise None."""
    cwd = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=5,
            check=False,
        )
        if res.returncode == 0:
            val = res.stdout.strip()
            return val or None
    except Exception:
        return None
    return None
