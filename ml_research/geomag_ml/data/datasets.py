"""PyTorch Dataset classes for single-timestep geomagnetic network reconstruction."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class SingleStepDataset(Dataset[dict[str, Any]]):
    """Single-minute network snapshots sliced strictly to a split's `indices`."""

    def __init__(
        self,
        x_work: np.ndarray,
        m_valid: np.ndarray,
        m_qc_clean: np.ndarray,
        time_features: np.ndarray,
        indices: np.ndarray,
        in_train_mask: np.ndarray,
        *,
        mlt_hours: np.ndarray | None = None,
        x_delta_nT: np.ndarray | None = None,
    ) -> None:
        idx_arr = np.asarray(indices)
        if idx_arr.dtype == np.bool_:
            if idx_arr.ndim != 1 or idx_arr.shape[0] != x_work.shape[0]:
                raise ValueError(f"Boolean indices shape {idx_arr.shape} != ({x_work.shape[0]},).")
            sel = np.flatnonzero(idx_arr).astype(np.int64)
        else:
            sel = np.asarray(idx_arr, dtype=np.int64).reshape(-1)
            if sel.size > 0 and (int(sel.min()) < 0 or int(sel.max()) >= x_work.shape[0]):
                raise ValueError(f"indices out of bounds [0, {x_work.shape[0]}).")

        x_sub = np.asarray(x_work[sel], dtype=np.float32)
        if x_sub.ndim != 3 or x_sub.shape[2] != 3:
            raise ValueError(f"Expected x_work shape (T, N, 3), got {x_sub.shape}.")
        n_t, n_st, _ = x_sub.shape

        m_val_sub = np.asarray(m_valid[sel], dtype=bool)
        m_qc_sub = np.asarray(m_qc_clean[sel], dtype=bool)
        tf_sub = np.asarray(time_features[sel], dtype=np.float32)
        in_tr = np.asarray(in_train_mask, dtype=bool)

        if m_val_sub.shape != (n_t, n_st) or m_qc_sub.shape != (n_t, n_st):
            raise ValueError("m_valid and m_qc_clean shapes must match x_work[indices, :, :].")
        if tf_sub.shape != (n_t, 4):
            raise ValueError(f"Expected time_features shape ({n_t}, 4), got {tf_sub.shape}.")
        if in_tr.shape != (n_st,):
            raise ValueError(f"Expected in_train_mask shape ({n_st},), got {in_tr.shape}.")

        # Enforce finiteness inside valid masks and zero out invalid entries cleanly.
        m_val_sub = m_val_sub & np.all(np.isfinite(x_sub), axis=-1)
        xd_sub: np.ndarray | None = None
        if x_delta_nT is not None:
            xd_sub = np.asarray(x_delta_nT[sel], dtype=np.float32)
            if xd_sub.shape != (n_t, n_st, 3):
                raise ValueError(f"Expected x_delta_nT shape {(n_t, n_st, 3)}, got {xd_sub.shape}.")
            m_val_sub = m_val_sub & np.all(np.isfinite(xd_sub), axis=-1)

        m_qc_sub = m_qc_sub & m_val_sub
        x_clean = np.where(
            m_val_sub[..., None],
            np.nan_to_num(x_sub, nan=0.0, posinf=0.0, neginf=0.0),
            np.float32(0.0),
        ).astype(np.float32)

        self.indices: np.ndarray = sel
        self.n_stations: int = int(n_st)
        self.x_work: torch.Tensor = torch.from_numpy(np.ascontiguousarray(x_clean))
        self.m_valid: torch.Tensor = torch.from_numpy(np.ascontiguousarray(m_val_sub))
        self.m_qc_clean: torch.Tensor = torch.from_numpy(np.ascontiguousarray(m_qc_sub))
        self.time_features: torch.Tensor = torch.from_numpy(
            np.ascontiguousarray(np.nan_to_num(tf_sub, nan=0.0, posinf=0.0, neginf=0.0))
        )
        self.in_train_mask: torch.Tensor = torch.from_numpy(np.ascontiguousarray(in_tr))

        if mlt_hours is not None:
            mlt_sub = np.asarray(mlt_hours[sel], dtype=np.float32)
            if mlt_sub.shape != (n_t, n_st):
                raise ValueError(f"Expected mlt_hours shape {(n_t, n_st)}, got {mlt_sub.shape}.")
            self.mlt_hours: torch.Tensor | None = torch.from_numpy(np.ascontiguousarray(mlt_sub))
        else:
            self.mlt_hours = None

        if xd_sub is not None:
            xd_clean = np.where(
                m_val_sub[..., None],
                np.nan_to_num(xd_sub, nan=0.0, posinf=0.0, neginf=0.0),
                np.float32(0.0),
            ).astype(np.float32)
            self.x_delta_nT: torch.Tensor | None = torch.from_numpy(np.ascontiguousarray(xd_clean))
        else:
            self.x_delta_nT = None

    def __len__(self) -> int:
        return int(self.x_work.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item: dict[str, torch.Tensor] = {
            "x_work": self.x_work[idx],
            "m_valid": self.m_valid[idx],
            "m_qc_clean": self.m_qc_clean[idx],
            "time_features": self.time_features[idx],
            "sample_idx": torch.tensor(int(self.indices[idx]), dtype=torch.int64),
        }
        if self.mlt_hours is not None:
            item["mlt_hours"] = self.mlt_hours[idx]
        if self.x_delta_nT is not None:
            item["x_delta_nT"] = self.x_delta_nT[idx]
        return item
