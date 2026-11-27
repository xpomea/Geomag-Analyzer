"""Leakage-free baseline estimation (`B0`) and field scaling (`asinh_global` / `zscore_per_station`)."""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from geomag_ml.data.schema import EVENT_FLAG_QUIET, OMNI_COLS
from geomag_ml.utils.io import read_json, write_json

_VALID_BASELINE_MODES = (
    "temp_train_quiet_proxy",
    "target_kp_quiet_night",
    "per_episode_median_ablation",
)
_VALID_BASELESS_POLICIES = ("train_quiet_proxy", "episode_initial_6h", "exclude")
_VALID_VAR_STEP_POLICIES = ("flag_qc_exclude", "causal_initial_6h_reanchor", "keep_raw")
_VALID_SCALER_MODES = ("asinh_global", "zscore_per_station")


def _normalize_train_indices(train_idx: np.ndarray, total_len: int) -> np.ndarray:
    idx_arr = np.asarray(train_idx)
    if idx_arr.dtype == np.bool_:
        if idx_arr.shape != (total_len,):
            raise ValueError(f"Boolean train_idx shape {idx_arr.shape} != ({total_len},)")
        idx = np.flatnonzero(idx_arr).astype(np.int64)
    else:
        idx = np.unique(idx_arr.astype(np.int64))
        if idx.size > 0 and (int(idx.min()) < 0 or int(idx.max()) >= total_len):
            raise ValueError(f"train_idx out of bounds [0, {total_len})")
    if idx.size == 0:
        raise ValueError("train_idx must contain at least one index")
    return idx


def _compute_split_hash(
    tr_idx: np.ndarray,
    station_mask: np.ndarray,
    extra: str = "",
) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(tr_idx, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(station_mask, dtype=np.bool_).tobytes())
    if extra:
        h.update(extra.encode("utf-8"))
    return h.hexdigest()[:16]


class BaselineEstimator:
    """Estimate and subtract static/quiet baseline `B0` strictly from `train_idx`."""

    def __init__(
        self,
        mode: str = "temp_train_quiet_proxy",
        baseless_policy: str = "train_quiet_proxy",
        variation_step_policy: str = "flag_qc_exclude",
        *,
        step_warn_nT: float = 30.0,
        min_train_points: int = 60,
        quiet_quantile: float = 0.30,
        causal_init_minutes: int = 360,
        kp_max: float = 2.0,
        sym_h_abs_max: float = 15.0,
        use_secular_trend: bool = True,
    ) -> None:
        if mode not in _VALID_BASELINE_MODES:
            raise ValueError(f"Unsupported baseline mode {mode!r}")
        if baseless_policy not in _VALID_BASELESS_POLICIES:
            raise ValueError(f"Unsupported baseless_policy {baseless_policy!r}")
        if variation_step_policy not in _VALID_VAR_STEP_POLICIES:
            raise ValueError(f"Unsupported variation_step_policy {variation_step_policy!r}")

        self.mode = mode
        self.baseless_policy = baseless_policy
        self.variation_step_policy = variation_step_policy
        self.step_warn_nT = float(step_warn_nT)
        self.min_train_points = int(min_train_points)
        self.quiet_quantile = float(quiet_quantile)
        self.causal_init_minutes = int(causal_init_minutes)
        self.kp_max = float(kp_max)
        self.sym_h_abs_max = float(sym_h_abs_max)
        self.use_secular_trend = bool(use_secular_trend)

        self.B0_nT_: np.ndarray | None = None
        self.in_train_stations_: np.ndarray | None = None
        self.has_train_baseline_: np.ndarray | None = None
        self.station_has_baseline_: np.ndarray | None = None
        self.secular_slope_nT_per_day_: np.ndarray | None = None
        self.secular_ref_unix_s_: np.ndarray | None = None
        self.fit_split_hash_: str | None = None

    def fit(
        self,
        dataset: dict[str, np.ndarray],
        train_idx: np.ndarray,
        *,
        station_train_mask: np.ndarray | None = None,
    ) -> BaselineEstimator:
        """Fit baseline constants `B0_nT_` exclusively on `train_idx` (0% test access)."""
        x_all = dataset["X_raw"]
        n_t, n_st, _ = x_all.shape
        tr_idx = _normalize_train_indices(train_idx, n_t)

        # Slice ONLY train indices; test indices are never read.
        x_tr = x_all[tr_idx].astype(np.float64)
        m_qc_tr = dataset["M_qc_clean"][tr_idx].astype(bool)
        ep_tr = dataset["episode_idx"][tr_idx].astype(np.int64)
        ts_tr = dataset["timestamps_unix_s"][tr_idx].astype(np.int64)
        mlt_tr = dataset["mlt_hours"][tr_idx].astype(np.float64)
        has_base = dataset["station_has_baseline"].astype(bool).copy()

        if station_train_mask is None:
            st_req = np.ones(n_st, dtype=bool)
        else:
            st_req = np.asarray(station_train_mask, dtype=bool).copy()
            if st_req.shape != (n_st,):
                raise ValueError(f"station_train_mask shape {st_req.shape} != ({n_st},)")

        if self.baseless_policy == "exclude":
            st_req &= has_base

        valid_cnt_tr = np.sum(m_qc_tr, axis=0)
        in_train = st_req & (valid_cnt_tr >= self.min_train_points)

        b0 = np.full((n_st, 3), np.nan, dtype=np.float64)
        has_train_b0 = np.zeros(n_st, dtype=bool)
        slope = np.zeros((n_st, 3), dtype=np.float64)
        ref_s = np.zeros(n_st, dtype=np.float64)

        if self.mode == "per_episode_median_ablation":
            has_train_b0 = in_train.copy()
        elif self.mode == "temp_train_quiet_proxy":
            # Intra-episode 1st differences within train_idx only
            consec = np.zeros(tr_idx.shape[0], dtype=bool)
            if tr_idx.shape[0] > 1:
                consec[1:] = (ep_tr[1:] == ep_tr[:-1]) & ((ts_tr[1:] - ts_tr[:-1]) == 60)

            d1_x = np.full((tr_idx.shape[0], n_st), np.nan, dtype=np.float64)
            d1_y = np.full((tr_idx.shape[0], n_st), np.nan, dtype=np.float64)
            if tr_idx.shape[0] > 1:
                pair_ok = consec[1:, None] & m_qc_tr[1:] & m_qc_tr[:-1]
                dx_raw = x_tr[1:, :, 0] - x_tr[:-1, :, 0]
                dy_raw = x_tr[1:, :, 1] - x_tr[:-1, :, 1]
                d1_x[1:] = np.where(pair_ok, dx_raw, np.nan)
                d1_y[1:] = np.where(pair_ok, dy_raw, np.nan)

            if np.any(in_train):
                speed = np.hypot(d1_x[:, in_train], d1_y[:, in_train])
            else:
                speed = np.hypot(d1_x, d1_y)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                v_net = np.nanmedian(speed, axis=1)

            fin_v = np.isfinite(v_net)
            if np.any(fin_v):
                q_thresh = float(np.nanpercentile(v_net[fin_v], self.quiet_quantile * 100.0))
                is_quiet = fin_v & (v_net <= q_thresh)
            else:
                is_quiet = np.ones(tr_idx.shape[0], dtype=bool)

            is_night = (mlt_tr >= 22.0) | (mlt_tr <= 4.0)

            for i in range(n_st):
                if self.baseless_policy == "exclude" and not has_base[i]:
                    continue
                m_i = m_qc_tr[:, i]
                if int(np.sum(m_i)) < self.min_train_points:
                    in_train[i] = False
                    continue

                sel = m_i & is_night[:, i] & is_quiet
                if int(np.sum(sel)) < self.min_train_points:
                    sel = m_i & is_night[:, i]
                if int(np.sum(sel)) < self.min_train_points:
                    sel = m_i

                b0[i] = np.median(x_tr[sel, i, :], axis=0)
                has_train_b0[i] = True
        else:  # target_kp_quiet_night
            sw_omni_tr = dataset["sw_omni"][tr_idx]
            ev_tr = dataset["event_flags"][tr_idx]
            kp_col = sw_omni_tr[:, OMNI_COLS.index("kp")].astype(np.float64)
            sym_col = sw_omni_tr[:, OMNI_COLS.index("sym_h")].astype(np.float64)

            kp_quiet = np.where(
                np.isfinite(kp_col),
                kp_col <= self.kp_max,
                (ev_tr & EVENT_FLAG_QUIET) != 0,
            )
            sym_quiet = np.where(
                np.isfinite(sym_col),
                np.abs(sym_col) <= self.sym_h_abs_max,
                True,
            )
            is_quiet = kp_quiet & sym_quiet
            is_night = (mlt_tr >= 22.0) | (mlt_tr <= 4.0)

            for i in range(n_st):
                if self.baseless_policy == "exclude" and not has_base[i]:
                    continue
                m_i = m_qc_tr[:, i]
                if int(np.sum(m_i)) < self.min_train_points:
                    in_train[i] = False
                    continue

                sel = m_i & is_quiet & is_night[:, i]
                if int(np.sum(sel)) < self.min_train_points:
                    sel = m_i & is_quiet
                if int(np.sum(sel)) < self.min_train_points:
                    sel = m_i & is_night[:, i]
                if int(np.sum(sel)) < self.min_train_points:
                    sel = m_i

                pts = x_tr[sel, i, :]
                t_sel = ts_tr[sel].astype(np.float64)
                span_days = float((t_sel.max() - t_sel.min()) / 86400.0) if t_sel.size >= 2 else 0.0
                if self.use_secular_trend and span_days >= 60.0:
                    t_ref = float(np.median(t_sel))
                    dt_days = (t_sel - t_ref) / 86400.0
                    dt_mean = float(np.mean(dt_days))
                    dt_centered = dt_days - dt_mean
                    denom = float(np.dot(dt_centered, dt_centered))
                    if denom > 1e-9:
                        y_mean = np.mean(pts, axis=0)
                        sl = np.sum(dt_centered[:, None] * (pts - y_mean[None, :]), axis=0) / denom
                        slope[i] = sl
                        ref_s[i] = t_ref
                        b0[i] = y_mean - sl * dt_mean
                    else:
                        b0[i] = np.median(pts, axis=0)
                else:
                    b0[i] = np.median(pts, axis=0)
                has_train_b0[i] = True

        self.B0_nT_ = b0
        self.in_train_stations_ = in_train
        self.has_train_baseline_ = has_train_b0
        self.station_has_baseline_ = has_base
        self.secular_slope_nT_per_day_ = slope
        self.secular_ref_unix_s_ = ref_s
        self.fit_split_hash_ = _compute_split_hash(
            tr_idx, in_train, extra=f"{self.mode}:{self.baseless_policy}:{self.variation_step_policy}"
        )
        return self

    def _causal_initial_mask(
        self,
        m_ep: np.ndarray,
        ts_ep: np.ndarray | None,
    ) -> np.ndarray:
        """Select valid indices in the causal initial window (or earliest valid samples)."""
        n_ep = m_ep.shape[0]
        if ts_ep is not None and n_ep > 0:
            in_init = (ts_ep - ts_ep[0]) < (self.causal_init_minutes * 60)
        else:
            in_init = np.zeros(n_ep, dtype=bool)
            in_init[: min(self.causal_init_minutes, n_ep)] = True
        init_valid = m_ep & in_init
        if np.any(init_valid):
            return init_valid
        mask = np.zeros(n_ep, dtype=bool)
        valid_idx = np.flatnonzero(m_ep)
        take = valid_idx[: min(self.causal_init_minutes, valid_idx.size)]
        mask[take] = True
        return mask

    def _causal_initial_baseline(
        self,
        x_ep: np.ndarray,
        m_ep: np.ndarray,
        ts_ep: np.ndarray | None,
    ) -> np.ndarray:
        """Compute causal baseline from the first `causal_init_minutes` of an episode."""
        init_mask = self._causal_initial_mask(m_ep, ts_ep)
        return np.median(x_ep[init_mask], axis=0)

    def transform(
        self,
        X_raw: np.ndarray,
        M_valid: np.ndarray,
        episode_idx: np.ndarray,
        *,
        timestamps_unix_s: np.ndarray | None = None,
        station_has_baseline: np.ndarray | None = None,
    ) -> np.ndarray:
        """Subtract baseline `B0` from `X_raw` while preserving `NaN` at invalid positions."""
        if self.B0_nT_ is None or self.has_train_baseline_ is None:
            raise RuntimeError("BaselineEstimator must be fitted before transform().")

        x = np.asarray(X_raw, dtype=np.float64)
        m = np.asarray(M_valid, dtype=bool)
        ep = np.asarray(episode_idx, dtype=np.int64)
        n_t, n_st, n_comp = x.shape
        if n_comp != 3 or m.shape != (n_t, n_st) or ep.shape != (n_t,):
            raise ValueError("Incompatible shapes passed to BaselineEstimator.transform()")

        has_base = (
            np.asarray(station_has_baseline, dtype=bool)
            if station_has_baseline is not None
            else self.station_has_baseline_
        )
        if has_base is None:
            has_base = np.ones(n_st, dtype=bool)

        ts = (
            np.asarray(timestamps_unix_s, dtype=np.int64)
            if timestamps_unix_s is not None
            else None
        )
        out = np.full((n_t, n_st, 3), np.nan, dtype=np.float64)
        unique_eps = np.unique(ep)

        if self.mode == "per_episode_median_ablation":
            for e_id in unique_eps:
                ep_mask = ep == e_id
                for i in range(n_st):
                    if self.baseless_policy == "exclude" and not has_base[i]:
                        continue
                    valid_e = ep_mask & m[:, i]
                    if not np.any(valid_e):
                        continue
                    med = np.median(x[valid_e, i, :], axis=0)
                    out[valid_e, i, :] = x[valid_e, i, :] - med
            return out.astype(np.float32)

        assert self.secular_slope_nT_per_day_ is not None
        assert self.secular_ref_unix_s_ is not None

        for i in range(n_st):
            if self.baseless_policy == "exclude" and not has_base[i]:
                continue
            use_causal_station = (not bool(self.has_train_baseline_[i])) or (
                (not bool(has_base[i])) and self.baseless_policy == "episode_initial_6h"
            )

            for e_id in unique_eps:
                ep_mask = ep == e_id
                valid_e = ep_mask & m[:, i]
                if not np.any(valid_e):
                    continue

                if use_causal_station:
                    ep_idx_pos = np.flatnonzero(ep_mask)
                    ts_ep = ts[ep_idx_pos] if ts is not None else None
                    b0_eff = self._causal_initial_baseline(
                        x[ep_idx_pos, i, :],
                        m[ep_idx_pos, i],
                        ts_ep,
                    )
                    out[valid_e, i, :] = x[valid_e, i, :] - b0_eff[None, :]
                    continue

                b0_static = self.B0_nT_[i]
                slope_i = self.secular_slope_nT_per_day_[i]
                if ts is not None and np.any(slope_i != 0.0):
                    dt_days = (ts[valid_e].astype(np.float64) - self.secular_ref_unix_s_[i]) / 86400.0
                    b0_series = b0_static[None, :] + dt_days[:, None] * slope_i[None, :]
                else:
                    b0_series = b0_static[None, :]

                if self.variation_step_policy == "causal_initial_6h_reanchor":
                    ep_idx_pos = np.flatnonzero(ep_mask)
                    ts_ep = ts[ep_idx_pos] if ts is not None else None
                    init_mask = self._causal_initial_mask(m[ep_idx_pos, i], ts_ep)
                    init_med = np.median(x[ep_idx_pos[init_mask], i, :], axis=0)
                    if ts_ep is not None and np.any(slope_i != 0.0):
                        dt_init = (
                            ts_ep[init_mask].astype(np.float64) - self.secular_ref_unix_s_[i]
                        ) / 86400.0
                        b0_init_ref = np.median(
                            b0_static[None, :] + dt_init[:, None] * slope_i[None, :], axis=0
                        )
                    else:
                        b0_init_ref = b0_static
                    if float(np.max(np.abs(init_med - b0_init_ref))) > self.step_warn_nT:
                        b0_series = b0_series + (init_med - b0_init_ref)[None, :]

                out[valid_e, i, :] = x[valid_e, i, :] - b0_series

        out[~m] = np.nan
        return out.astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        """Serialize fitted parameters into a JSON-compatible dictionary."""
        if self.B0_nT_ is None or self.in_train_stations_ is None:
            raise RuntimeError("BaselineEstimator must be fitted before to_dict().")
        return {
            "baseline_mode": self.mode,
            "baseless_policy": self.baseless_policy,
            "variation_step_policy": self.variation_step_policy,
            "step_warn_nT": self.step_warn_nT,
            "min_train_points": self.min_train_points,
            "quiet_quantile": self.quiet_quantile,
            "causal_init_minutes": self.causal_init_minutes,
            "kp_max": self.kp_max,
            "sym_h_abs_max": self.sym_h_abs_max,
            "use_secular_trend": self.use_secular_trend,
            "B0_nT": self.B0_nT_.tolist(),
            "in_train_stations": self.in_train_stations_.tolist(),
            "has_train_baseline": self.has_train_baseline_.tolist()
            if self.has_train_baseline_ is not None
            else None,
            "station_has_baseline": self.station_has_baseline_.tolist()
            if self.station_has_baseline_ is not None
            else None,
            "secular_slope_nT_per_day": self.secular_slope_nT_per_day_.tolist()
            if self.secular_slope_nT_per_day_ is not None
            else None,
            "secular_ref_unix_s": self.secular_ref_unix_s_.tolist()
            if self.secular_ref_unix_s_ is not None
            else None,
            "fit_split_hash": self.fit_split_hash_,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BaselineEstimator:
        """Restore fitted `BaselineEstimator` from serialized dictionary."""
        est = cls(
            mode=str(payload["baseline_mode"]),
            baseless_policy=str(payload["baseless_policy"]),
            variation_step_policy=str(payload.get("variation_step_policy", "flag_qc_exclude")),
            step_warn_nT=float(payload.get("step_warn_nT", 30.0)),
            min_train_points=int(payload.get("min_train_points", 60)),
            quiet_quantile=float(payload.get("quiet_quantile", 0.30)),
            causal_init_minutes=int(payload.get("causal_init_minutes", 360)),
            kp_max=float(payload.get("kp_max", 2.0)),
            sym_h_abs_max=float(payload.get("sym_h_abs_max", 15.0)),
            use_secular_trend=bool(payload.get("use_secular_trend", True)),
        )
        est.B0_nT_ = np.array(
            [[np.nan if v is None else float(v) for v in row] for row in payload["B0_nT"]],
            dtype=np.float64,
        )
        est.in_train_stations_ = np.asarray(payload["in_train_stations"], dtype=bool)
        if payload.get("has_train_baseline") is not None:
            est.has_train_baseline_ = np.asarray(payload["has_train_baseline"], dtype=bool)
        if payload.get("station_has_baseline") is not None:
            est.station_has_baseline_ = np.asarray(payload["station_has_baseline"], dtype=bool)
        if payload.get("secular_slope_nT_per_day") is not None:
            est.secular_slope_nT_per_day_ = np.asarray(
                payload["secular_slope_nT_per_day"], dtype=np.float64
            )
        if payload.get("secular_ref_unix_s") is not None:
            est.secular_ref_unix_s_ = np.asarray(payload["secular_ref_unix_s"], dtype=np.float64)
        est.fit_split_hash_ = payload.get("fit_split_hash")
        return est


class FieldScaler:
    """Dual invertible field scaler (`asinh_global` or `zscore_per_station`) fit on `train_idx`."""

    def __init__(
        self,
        mode: str = "asinh_global",
        *,
        min_std_nT: float = 5.0,
        min_asinh_s_nT: float = 10.0,
        fixed_asinh_scale: tuple[float, float, float] | list[float] | None = None,
        robust_clip_mad: float = 10.0,
        min_train_points: int = 60,
    ) -> None:
        if mode not in _VALID_SCALER_MODES:
            raise ValueError(f"Unsupported scaler mode {mode!r}")
        self.mode = mode
        self.min_std_nT = float(min_std_nT)
        self.min_asinh_s_nT = float(min_asinh_s_nT)
        self.fixed_asinh_scale = (
            tuple(float(v) for v in fixed_asinh_scale)
            if fixed_asinh_scale is not None
            else None
        )
        self.robust_clip_mad = float(robust_clip_mad)
        self.min_train_points = int(min_train_points)

        self.in_train_stations_: np.ndarray | None = None
        self.scale_s_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.fit_split_hash_: str | None = None

    def fit(
        self,
        X_delta: np.ndarray,
        M_qc_clean: np.ndarray,
        train_idx: np.ndarray,
        station_belt_id: np.ndarray,
        station_train_mask: np.ndarray | None = None,
    ) -> FieldScaler:
        """Fit scaling parameters strictly on `train_idx` & `M_qc_clean` & `in_train`."""
        x_all = np.asarray(X_delta)
        n_t, n_st, n_comp = x_all.shape
        if n_comp != 3:
            raise ValueError(f"Expected (T, N, 3) X_delta, got {x_all.shape}")

        tr_idx = _normalize_train_indices(train_idx, n_t)
        x_tr = x_all[tr_idx].astype(np.float64)
        m_tr = np.asarray(M_qc_clean[tr_idx], dtype=bool)
        belts = np.asarray(station_belt_id, dtype=np.int64)
        if belts.shape != (n_st,):
            raise ValueError(f"station_belt_id shape {belts.shape} != ({n_st},)")

        # Require actual finite values in X_delta too (guards against excluded baseless stations)
        m_tr = m_tr & np.all(np.isfinite(x_tr), axis=-1)
        valid_cnt = np.sum(m_tr, axis=0)
        in_train = valid_cnt >= max(2, self.min_train_points)
        if station_train_mask is not None:
            st_mask = np.asarray(station_train_mask, dtype=bool)
            if st_mask.shape != (n_st,):
                raise ValueError(f"station_train_mask shape {st_mask.shape} != ({n_st},)")
            in_train &= st_mask

        self.in_train_stations_ = in_train

        if self.mode == "asinh_global":
            if self.fixed_asinh_scale is not None:
                if len(self.fixed_asinh_scale) != 3:
                    raise ValueError("fixed_asinh_scale must have length 3")
                s = np.maximum(np.asarray(self.fixed_asinh_scale, dtype=np.float64), self.min_asinh_s_nT)
            else:
                cand = np.flatnonzero((belts == 0) & in_train)
                if cand.size == 0:
                    cand = np.flatnonzero(in_train)
                if cand.size == 0:
                    s = np.full(3, self.min_asinh_s_nT, dtype=np.float64)
                else:
                    iqr_scales = []
                    for i in cand:
                        pts = x_tr[m_tr[:, i], i, :]
                        q75, q25 = np.percentile(pts, [75.0, 25.0], axis=0)
                        iqr_scales.append((q75 - q25) / 1.3489795)
                    s = np.median(np.vstack(iqr_scales), axis=0)
                    s = np.maximum(s, self.min_asinh_s_nT)
            self.scale_s_ = s.astype(np.float64)
            self.mean_ = None
            self.std_ = None
        else:
            mean = np.full((n_st, 3), np.nan, dtype=np.float64)
            std = np.full((n_st, 3), np.nan, dtype=np.float64)
            for i in np.flatnonzero(in_train):
                pts = x_tr[m_tr[:, i], i, :]
                for c in range(3):
                    col = pts[:, c]
                    med = float(np.median(col))
                    mad = float(np.median(np.abs(col - med))) * 1.4826
                    if mad > 1e-6 and self.robust_clip_mad > 0.0:
                        lo = med - self.robust_clip_mad * mad
                        hi = med + self.robust_clip_mad * mad
                        col_c = np.clip(col, lo, hi)
                    else:
                        col_c = col
                    mean[i, c] = float(np.mean(col_c))
                    std[i, c] = max(float(np.std(col_c)), self.min_std_nT)

            # Belt-median fallback for unseen stations (e.g. SUA in Fold 3, Zero-Shot LVV/BEL/MEA)
            trained_idx = np.flatnonzero(in_train)
            global_mean = (
                np.median(mean[trained_idx], axis=0)
                if trained_idx.size > 0
                else np.zeros(3, dtype=np.float64)
            )
            global_std = (
                np.median(std[trained_idx], axis=0)
                if trained_idx.size > 0
                else np.full(3, self.min_std_nT, dtype=np.float64)
            )
            for i in np.flatnonzero(~in_train):
                same_belt = np.flatnonzero((belts == belts[i]) & in_train)
                if same_belt.size > 0:
                    mean[i] = np.median(mean[same_belt], axis=0)
                    std[i] = np.median(std[same_belt], axis=0)
                else:
                    mean[i] = global_mean
                    std[i] = global_std

            std = np.maximum(std, self.min_std_nT)
            if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(std))):
                raise RuntimeError("FieldScaler produced non-finite mean/std parameters")
            self.mean_ = mean
            self.std_ = std
            self.scale_s_ = None

        self.fit_split_hash_ = _compute_split_hash(tr_idx, in_train, extra=self.mode)
        return self

    def transform(self, X_delta: np.ndarray) -> np.ndarray:
        """Transform `ΔB [nT]` into scaled space while preserving `NaN`."""
        x = np.asarray(X_delta, dtype=np.float64)
        if self.mode == "asinh_global":
            if self.scale_s_ is None:
                raise RuntimeError("FieldScaler(asinh_global) is not fitted.")
            shape_s = (1,) * (x.ndim - 1) + (3,)
            s = self.scale_s_.reshape(shape_s)
            with np.errstate(invalid="ignore"):
                out = np.arcsinh(x / s)
        else:
            if self.mean_ is None or self.std_ is None:
                raise RuntimeError("FieldScaler(zscore_per_station) is not fitted.")
            if x.ndim == 2:
                mean = self.mean_
                std = self.std_
            else:
                shape_b = (1,) * (x.ndim - 2) + self.mean_.shape
                mean = self.mean_.reshape(shape_b)
                std = self.std_.reshape(shape_b)
            with np.errstate(invalid="ignore"):
                out = (x - mean) / std
        return out.astype(np.float32)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Invert scaled values back to physical `ΔB [nT]`."""
        z = np.asarray(X_scaled, dtype=np.float64)
        if self.mode == "asinh_global":
            if self.scale_s_ is None:
                raise RuntimeError("FieldScaler(asinh_global) is not fitted.")
            shape_s = (1,) * (z.ndim - 1) + (3,)
            s = self.scale_s_.reshape(shape_s)
            with np.errstate(invalid="ignore", over="ignore"):
                out = s * np.sinh(z)
        else:
            if self.mean_ is None or self.std_ is None:
                raise RuntimeError("FieldScaler(zscore_per_station) is not fitted.")
            if z.ndim == 2:
                mean = self.mean_
                std = self.std_
            else:
                shape_b = (1,) * (z.ndim - 2) + self.mean_.shape
                mean = self.mean_.reshape(shape_b)
                std = self.std_.reshape(shape_b)
            with np.errstate(invalid="ignore"):
                out = z * std + mean
        return out.astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        """Serialize scaler parameters to JSON-safe dictionary."""
        if self.in_train_stations_ is None:
            raise RuntimeError("FieldScaler must be fitted before to_dict().")
        return {
            "scaler_mode": self.mode,
            "min_std_nT": self.min_std_nT,
            "min_asinh_s_nT": self.min_asinh_s_nT,
            "fixed_asinh_scale": list(self.fixed_asinh_scale)
            if self.fixed_asinh_scale is not None
            else None,
            "robust_clip_mad": self.robust_clip_mad,
            "min_train_points": self.min_train_points,
            "in_train_stations": self.in_train_stations_.tolist(),
            "scale_s": self.scale_s_.tolist() if self.scale_s_ is not None else None,
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "std": self.std_.tolist() if self.std_ is not None else None,
            "fit_split_hash": self.fit_split_hash_,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FieldScaler:
        """Restore fitted `FieldScaler` from dictionary."""
        scaler = cls(
            mode=str(payload["scaler_mode"]),
            min_std_nT=float(payload.get("min_std_nT", 5.0)),
            min_asinh_s_nT=float(payload.get("min_asinh_s_nT", 10.0)),
            fixed_asinh_scale=payload.get("fixed_asinh_scale"),
            robust_clip_mad=float(payload.get("robust_clip_mad", 10.0)),
            min_train_points=int(payload.get("min_train_points", 60)),
        )
        scaler.in_train_stations_ = np.asarray(payload["in_train_stations"], dtype=bool)
        if payload.get("scale_s") is not None:
            scaler.scale_s_ = np.asarray(payload["scale_s"], dtype=np.float64)
        if payload.get("mean") is not None:
            scaler.mean_ = np.asarray(payload["mean"], dtype=np.float64)
        if payload.get("std") is not None:
            scaler.std_ = np.asarray(payload["std"], dtype=np.float64)
        scaler.fit_split_hash_ = payload.get("fit_split_hash")
        return scaler


def save_scaler_json(
    path: Path,
    baseline: BaselineEstimator,
    scaler: FieldScaler | None,
) -> Path:
    """Save combined `scaler.json` artifact adhering to §5.4 contract."""
    b_dict = baseline.to_dict()
    if scaler is None:
        effective_in_train = np.asarray(b_dict["in_train_stations"], dtype=bool).tolist()
        combined: dict[str, Any] = {
            "baseline_mode": b_dict["baseline_mode"],
            "baseless_policy": b_dict["baseless_policy"],
            "variation_step_policy": b_dict["variation_step_policy"],
            "B0_nT": b_dict["B0_nT"],
            "in_train_stations": effective_in_train,
            "scaler_mode": "none",
            "scale_s": None,
            "mean": None,
            "std": None,
            "fit_split_hash": b_dict["fit_split_hash"],
            "baseline_state": b_dict,
            "scaler_state": None,
        }
        return write_json(path, combined, nan_to_null=True)

    s_dict = scaler.to_dict()
    effective_in_train = (
        np.asarray(b_dict["in_train_stations"], dtype=bool)
        & np.asarray(s_dict["in_train_stations"], dtype=bool)
    ).tolist()
    combined = {
        "baseline_mode": b_dict["baseline_mode"],
        "baseless_policy": b_dict["baseless_policy"],
        "variation_step_policy": b_dict["variation_step_policy"],
        "B0_nT": b_dict["B0_nT"],
        "in_train_stations": effective_in_train,
        "scaler_mode": s_dict["scaler_mode"],
        "scale_s": s_dict["scale_s"],
        "mean": s_dict["mean"],
        "std": s_dict["std"],
        "fit_split_hash": s_dict["fit_split_hash"],
        "baseline_state": b_dict,
        "scaler_state": s_dict,
    }
    return write_json(path, combined, nan_to_null=True)


def load_scaler_json(path: Path) -> tuple[BaselineEstimator, FieldScaler | None]:
    """Restore `(BaselineEstimator, FieldScaler | None)` from `scaler.json`."""
    payload = read_json(path)
    b_payload = payload.get("baseline_state", payload)
    baseline = BaselineEstimator.from_dict(b_payload)
    if payload.get("scaler_mode") == "none":
        return baseline, None
    s_payload = payload.get("scaler_state", payload)
    if s_payload is None:
        return baseline, None
    return baseline, FieldScaler.from_dict(s_payload)
