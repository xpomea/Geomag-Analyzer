"""Deterministic RNG seeding utilities."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, *, deterministic_cuda: bool = True) -> int:
    """Seed Python, NumPy, and PyTorch RNGs deterministically."""
    seed_int = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed_int)
    random.seed(seed_int)
    np.random.seed(seed_int)
    torch.manual_seed(seed_int)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_int)
        torch.cuda.manual_seed_all(seed_int)
        if deterministic_cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return seed_int


def make_numpy_rng(seed: int = 42) -> np.random.Generator:
    """Create a fresh isolated NumPy Generator with the given integer seed."""
    return np.random.default_rng(int(seed))
