"""Utility helpers for determinism, atomic I/O, and CPU/CUDA device setup."""

from __future__ import annotations

from .device import recommended_cpu_threads, recommended_num_workers, resolve_device
from .io import (
    compute_sha256,
    get_git_commit,
    load_npz,
    portable_path_str,
    read_json,
    read_yaml,
    resolve_project_path,
    save_npz_atomic,
    to_jsonable,
    write_json,
    write_yaml,
)
from .seed import make_numpy_rng, set_seed

__all__ = [
    "compute_sha256",
    "get_git_commit",
    "load_npz",
    "make_numpy_rng",
    "portable_path_str",
    "read_json",
    "read_yaml",
    "recommended_cpu_threads",
    "recommended_num_workers",
    "resolve_device",
    "resolve_project_path",
    "save_npz_atomic",
    "set_seed",
    "to_jsonable",
    "write_json",
    "write_yaml",
]
