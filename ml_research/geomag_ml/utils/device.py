"""Device selection and Ryzen CPU thread tuning utilities."""

from __future__ import annotations

import os
import warnings

import torch


def recommended_cpu_threads(max_threads: int = 6) -> int:
    """Return physical-core thread count cap (6 on 12-logical-core Ryzen)."""
    logical = os.cpu_count() or 2
    return max(1, min(int(max_threads), max(1, logical // 2)))


def resolve_device(
    requested: str = "auto",
    *,
    cpu_threads: int | None = None,
) -> torch.device:
    """Resolve `auto|cpu|cuda` device safely with CPU fallback and thread capping."""
    req = str(requested).strip().lower()
    if req == "auto":
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
    elif req == "cpu":
        device = torch.device("cpu")
    elif req.startswith("cuda"):
        if torch.cuda.is_available():
            device = torch.device(req)
        else:
            warnings.warn(
                f"Requested device {requested!r} but CUDA is unavailable; falling back to 'cpu'.",
                RuntimeWarning,
                stacklevel=2,
            )
            device = torch.device("cpu")
    else:
        raise ValueError(f"Unsupported device specifier {requested!r}; expected 'auto', 'cpu', or 'cuda'.")

    if device.type == "cpu":
        n_threads = int(cpu_threads) if cpu_threads is not None else recommended_cpu_threads()
        torch.set_num_threads(max(1, n_threads))
    return device


def recommended_num_workers(device: torch.device | str = "cpu") -> int:
    """Return DataLoader num_workers (0 on Windows and CPU to avoid spawn overhead)."""
    _ = device
    if os.name == "nt":
        return 0
    return 0
