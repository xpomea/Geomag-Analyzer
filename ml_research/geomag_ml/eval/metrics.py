"""Standardized geomagnetic reconstruction, latent stability, and space-weather alignment metrics."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from geomag_ml.data.schema import OMNI_COLS, SOLAR1_BSN_COLS

REPORTING_BELTS: tuple[str, ...] = (
    "global",
    "europe_mid",
    "auroral",
    "europe_auroral",
    "canada_auroral",
    "canada",
    "canada_mid",
)
_COMPONENT_NAMES: tuple[str, ...] = ("X", "Y", "Z")


def belt_station_masks(station_belt_id: np.ndarray) -> dict[str, np.ndarray]:
    """Return boolean `(N,)` station selectors for each canonical reporting belt."""
    belts = np.asarray(station_belt_id, dtype=np.int64)
    if belts.ndim != 1:
        raise ValueError(f"Expected 1D station_belt_id, got shape {belts.shape}.")
    return {
        "global": np.ones(belts.shape[0], dtype=bool),
        "europe_mid": belts == 0,
        "auroral": (belts == 1) | (belts == 3),
        "europe_auroral": belts == 1,
        "canada_auroral": belts == 3,
        "canada": (belts == 2) | (belts == 3),
        "canada_mid": belts == 2,
    }


def _empty_component_metric() -> dict[str, float | int | None]:
    return {
        "rmse_nT": None,
        "mae_nT": None,
        "r2": None,
        "pearson_r": None,
        "n_points": 0,
    }


class StreamingMetricsAccumulator:
    """Numerically stable Chan-parallel accumulator across scenarios/batches and reporting belts."""

    def __init__(self, station_belt_id: np.ndarray) -> None:
        self._belt_masks = belt_station_masks(station_belt_id)
        self._n_stations = int(np.asarray(station_belt_id).shape[0])
        self._state: dict[str, dict[str, Any]] = {
            belt: {
                "n": 0,
                "mean_t": np.zeros(3, dtype=np.float64),
                "mean_p": np.zeros(3, dtype=np.float64),
                "m2_t": np.zeros(3, dtype=np.float64),
                "m2_p": np.zeros(3, dtype=np.float64),
                "c_tp": np.zeros(3, dtype=np.float64),
                "sse": np.zeros(3, dtype=np.float64),
                "sae": np.zeros(3, dtype=np.float64),
                "vec_sae": 0.0,
            }
            for belt in REPORTING_BELTS
        }

    def update(
        self,
        y_true_nT: np.ndarray,
        y_pred_nT: np.ndarray,
        eval_mask: np.ndarray,
    ) -> None:
        """Accumulate errors strictly at positions where `eval_mask` is True."""
        yt = np.asarray(y_true_nT, dtype=np.float64)
        yp = np.asarray(y_pred_nT, dtype=np.float64)
        mask = np.asarray(eval_mask, dtype=bool)

        if yt.ndim == 4:
            yt = yt.reshape(-1, yt.shape[2], yt.shape[3])
            yp = yp.reshape(-1, yp.shape[2], yp.shape[3])
            mask = mask.reshape(-1, mask.shape[2])

        if yt.ndim != 3 or yt.shape[2] != 3 or yp.shape != yt.shape:
            raise ValueError(f"Expected matching (T, N, 3) arrays, got {yt.shape} and {yp.shape}.")
        if mask.shape != yt.shape[:2] or yt.shape[1] != self._n_stations:
            raise ValueError(
                f"Incompatible eval_mask shape {mask.shape} for N={self._n_stations}, T={yt.shape[0]}."
            )

        if not np.any(mask):
            return

        if not (np.all(np.isfinite(yt[mask])) and np.all(np.isfinite(yp[mask]))):
            raise ValueError("Non-finite values found inside eval_mask during metric computation.")

        for belt_name, st_mask in self._belt_masks.items():
            if not np.any(st_mask):
                continue
            bm = mask[:, st_mask]
            if not np.any(bm):
                continue

            yt_b = yt[:, st_mask, :][bm]
            yp_b = yp[:, st_mask, :][bm]
            n_b = int(yt_b.shape[0])

            mt_b = np.mean(yt_b, axis=0)
            mp_b = np.mean(yp_b, axis=0)
            dt_b = yt_b - mt_b[None, :]
            dp_b = yp_b - mp_b[None, :]
            m2t_b = np.sum(dt_b * dt_b, axis=0)
            m2p_b = np.sum(dp_b * dp_b, axis=0)
            ctp_b = np.sum(dt_b * dp_b, axis=0)

            err = yp_b - yt_b
            sse_b = np.sum(err * err, axis=0)
            sae_b = np.sum(np.abs(err), axis=0)
            vec_sae_b = float(np.sum(np.linalg.norm(err, axis=1)))

            st = self._state[belt_name]
            n_a = int(st["n"])
            if n_a == 0:
                st["n"] = n_b
                st["mean_t"] = mt_b
                st["mean_p"] = mp_b
                st["m2_t"] = m2t_b
                st["m2_p"] = m2p_b
                st["c_tp"] = ctp_b
                st["sse"] = sse_b
                st["sae"] = sae_b
                st["vec_sae"] = vec_sae_b
            else:
                n_tot = n_a + n_b
                delta_t = mt_b - st["mean_t"]
                delta_p = mp_b - st["mean_p"]
                scale = (float(n_a) * float(n_b)) / float(n_tot)

                st["mean_t"] = st["mean_t"] + delta_t * (float(n_b) / float(n_tot))
                st["mean_p"] = st["mean_p"] + delta_p * (float(n_b) / float(n_tot))
                st["m2_t"] = st["m2_t"] + m2t_b + (delta_t * delta_t) * scale
                st["m2_p"] = st["m2_p"] + m2p_b + (delta_p * delta_p) * scale
                st["c_tp"] = st["c_tp"] + ctp_b + (delta_t * delta_p) * scale
                st["sse"] = st["sse"] + sse_b
                st["sae"] = st["sae"] + sae_b
                st["vec_sae"] = float(st["vec_sae"]) + vec_sae_b
                st["n"] = n_tot

    def finalize(self) -> dict[str, dict[str, dict[str, float | int | None]]]:
        """Compute final `{belt: {X|Y|Z|vector_3d: {rmse_nT, mae_nT, r2, pearson_r, n_points}}}`."""
        out: dict[str, dict[str, dict[str, float | int | None]]] = {}
        for belt_name in REPORTING_BELTS:
            st = self._state[belt_name]
            n = int(st["n"])
            if n == 0:
                out[belt_name] = {
                    "X": _empty_component_metric(),
                    "Y": _empty_component_metric(),
                    "Z": _empty_component_metric(),
                    "vector_3d": _empty_component_metric(),
                }
                continue

            belt_dict: dict[str, dict[str, float | int | None]] = {}
            for c_idx, c_name in enumerate(_COMPONENT_NAMES):
                sse_c = float(st["sse"][c_idx])
                sae_c = float(st["sae"][c_idx])
                m2t_c = float(st["m2_t"][c_idx])
                m2p_c = float(st["m2_p"][c_idx])
                ctp_c = float(st["c_tp"][c_idx])

                rmse = math.sqrt(sse_c / n)
                mae = sae_c / n
                r2 = (1.0 - (sse_c / m2t_c)) if m2t_c > 1e-12 else None
                denom = math.sqrt(m2t_c * m2p_c)
                pearson = max(-1.0, min(1.0, ctp_c / denom)) if denom > 1e-12 else None
                belt_dict[c_name] = {
                    "rmse_nT": float(rmse),
                    "mae_nT": float(mae),
                    "r2": float(r2) if r2 is not None else None,
                    "pearson_r": float(pearson) if pearson is not None else None,
                    "n_points": n,
                }

            tot_sse = float(np.sum(st["sse"]))
            tot_m2t = float(np.sum(st["m2_t"]))
            tot_m2p = float(np.sum(st["m2_p"]))
            tot_ctp = float(np.sum(st["c_tp"]))
            vec_rmse = math.sqrt(tot_sse / n)
            vec_mae = float(st["vec_sae"]) / n
            vec_r2 = (1.0 - (tot_sse / tot_m2t)) if tot_m2t > 1e-12 else None
            vec_denom = math.sqrt(tot_m2t * tot_m2p)
            vec_r = max(-1.0, min(1.0, tot_ctp / vec_denom)) if vec_denom > 1e-12 else None
            belt_dict["vector_3d"] = {
                "rmse_nT": float(vec_rmse),
                "mae_nT": float(vec_mae),
                "r2": float(vec_r2) if vec_r2 is not None else None,
                "pearson_r": float(vec_r) if vec_r is not None else None,
                "n_points": n,
            }
            out[belt_name] = belt_dict
        return out


def compute_reconstruction_metrics(
    y_true_nT: np.ndarray,
    y_pred_nT: np.ndarray,
    eval_mask: np.ndarray,
    station_belt_id: np.ndarray,
    station_regions: Sequence[str] | None = None,
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    """Compute standardized belt & component reconstruction metrics strictly on `eval_mask`."""
    _ = station_regions
    acc = StreamingMetricsAccumulator(station_belt_id)
    acc.update(y_true_nT, y_pred_nT, eval_mask)
    return acc.finalize()


def _safe_pearson_1d(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2:
        return None
    da = a - np.mean(a)
    db = b - np.mean(b)
    va = float(np.dot(da, da))
    vb = float(np.dot(db, db))
    denom = math.sqrt(va * vb)
    if denom <= 1e-12:
        return None
    return float(max(-1.0, min(1.0, float(np.dot(da, db)) / denom)))


def _safe_spearman_1d(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2:
        return None
    if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return None
    res = spearmanr(a, b)
    val = float(res.statistic)
    if not math.isfinite(val):
        return None
    return float(max(-1.0, min(1.0, val)))


def compute_latent_stability(
    z_full: np.ndarray,
    z_masked_list: Sequence[np.ndarray],
) -> dict[str, float]:
    """Evaluate invariance of Branch-A latent vector `z(t)` under station masking."""
    zf = np.asarray(z_full, dtype=np.float64)
    if zf.ndim != 2 or zf.shape[0] == 0 or zf.shape[1] == 0:
        raise ValueError(f"Expected non-empty 2D z_full, got shape {zf.shape}.")
    if not z_masked_list:
        raise ValueError("z_masked_list must contain at least one masked encoding.")

    n_t, k = zf.shape
    fin_f = np.all(np.isfinite(zf), axis=1)
    cos_vals: list[float] = []
    drift_vals: list[float] = []
    axis_corrs: list[float] = []

    for zm_raw in z_masked_list:
        zm = np.asarray(zm_raw, dtype=np.float64)
        if zm.shape != zf.shape:
            raise ValueError(f"z_masked shape {zm.shape} != z_full shape {zf.shape}.")
        valid = fin_f & np.all(np.isfinite(zm), axis=1)
        if not np.any(valid):
            continue
        zf_v = zf[valid]
        zm_v = zm[valid]

        nf = np.linalg.norm(zf_v, axis=1)
        nm = np.linalg.norm(zm_v, axis=1)
        dots = np.sum(zf_v * zm_v, axis=1)
        both_pos = (nf > 1e-12) & (nm > 1e-12)
        both_zero = (nf <= 1e-12) & (nm <= 1e-12)
        cos_row = np.where(both_pos, dots / np.maximum(nf * nm, 1e-24), np.where(both_zero, 1.0, 0.0))
        cos_vals.append(float(np.mean(np.clip(cos_row, -1.0, 1.0))))

        diff_norm = np.linalg.norm(zm_v - zf_v, axis=1)
        rel_row = np.where(
            nf > 1e-12,
            diff_norm / np.maximum(nf, 1e-12),
            np.where(diff_norm <= 1e-12, 0.0, 1.0),
        )
        drift_vals.append(float(np.mean(rel_row)))

        for a in range(k):
            r = _safe_pearson_1d(zf_v[:, a], zm_v[:, a])
            if r is None:
                # Constant equal arrays have perfect agreement; one-sided constant has 0.
                r = 1.0 if np.allclose(zf_v[:, a], zm_v[:, a], atol=1e-9) else 0.0
            axis_corrs.append(float(r))

    if np.any(fin_f) and int(np.sum(fin_f)) >= 2:
        zf_c = zf[fin_f] - np.mean(zf[fin_f], axis=0, keepdims=True)
        cov = (zf_c.T @ zf_c) / max(1, int(np.sum(fin_f)) - 1)
        decorr_frob = float(np.linalg.norm(cov - np.eye(k), ord="fro"))
        std = np.sqrt(np.maximum(np.diag(cov), 0.0))
        denom = np.outer(std, std)
        corr = np.where(denom > 1e-12, cov / np.maximum(denom, 1e-12), np.eye(k))
        np.fill_diagonal(corr, 1.0)
        offdiag_corr_frob = float(np.linalg.norm(corr - np.eye(k), ord="fro"))
    else:
        decorr_frob = 0.0
        offdiag_corr_frob = 0.0

    return {
        "cos_sim_mean": float(np.mean(cos_vals)) if cos_vals else 0.0,
        "rel_l2_drift": float(np.mean(drift_vals)) if drift_vals else 0.0,
        "axis_corr_mean": float(np.mean(axis_corrs)) if axis_corrs else 0.0,
        "axis_decorr_frob": decorr_frob,
        "axis_offdiag_corr_frob": offdiag_corr_frob,
    }


def _evaluate_single_sw_target(
    z_non_test: np.ndarray,
    z_test: np.ndarray,
    y_full: np.ndarray,
    non_test_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    ridge_alpha: float,
    min_samples: int,
) -> dict[str, Any]:
    y_arr = np.asarray(y_full, dtype=np.float64)
    if int(np.sum(np.isfinite(y_arr))) == 0:
        return {"status": "unavailable_all_nan", "n_train": 0, "n_test": 0}

    y_tr_raw = y_arr[non_test_idx]
    y_te_raw = y_arr[test_idx]

    m_tr = np.all(np.isfinite(z_non_test), axis=1) & np.isfinite(y_tr_raw)
    m_te = np.all(np.isfinite(z_test), axis=1) & np.isfinite(y_te_raw)
    n_tr = int(np.sum(m_tr))
    n_te = int(np.sum(m_te))

    if n_te < int(min_samples):
        return {"status": "unavailable_in_test_fold", "n_train": n_tr, "n_test": n_te}
    if n_tr < int(min_samples):
        return {"status": "unavailable_in_train_fold", "n_train": n_tr, "n_test": n_te}

    z_tr = z_non_test[m_tr]
    y_tr = y_tr_raw[m_tr]
    z_te = z_test[m_te]
    y_te = y_te_raw[m_te]

    z_mean = np.mean(z_tr, axis=0)
    z_std = np.maximum(np.std(z_tr, axis=0), 1e-6)
    z_tr_s = (z_tr - z_mean) / z_std
    z_te_s = (z_te - z_mean) / z_std

    reg = Ridge(alpha=float(ridge_alpha))
    reg.fit(z_tr_s, y_tr)
    y_hat = reg.predict(z_te_s)

    err = y_hat - y_te
    sse = float(np.dot(err, err))
    dy = y_te - np.mean(y_te)
    sst = float(np.dot(dy, dy))
    linear_r2 = (1.0 - sse / sst) if sst > 1e-12 else None
    linear_pearson = _safe_pearson_1d(y_te, y_hat)
    linear_spearman = _safe_spearman_1d(y_te, y_hat)

    k = z_te.shape[1]
    per_axis_p = [_safe_pearson_1d(z_te[:, a], y_te) for a in range(k)]
    per_axis_s = [_safe_spearman_1d(z_te[:, a], y_te) for a in range(k)]
    abs_p = [abs(r) for r in per_axis_p if r is not None]
    best_abs_p = float(max(abs_p)) if abs_p else None

    return {
        "status": "ok",
        "n_train": n_tr,
        "n_test": n_te,
        "linear_r2": float(linear_r2) if linear_r2 is not None else None,
        "linear_pearson_r": linear_pearson,
        "linear_spearman_r": linear_spearman,
        "per_axis_pearson_r": per_axis_p,
        "per_axis_spearman_r": per_axis_s,
        "best_axis_abs_pearson_r": best_abs_p,
    }


def compute_space_weather_alignment(
    z_non_test: np.ndarray,
    z_test: np.ndarray,
    sw_omni: np.ndarray,
    sw_solar1_bsn: np.ndarray,
    oulu: np.ndarray,
    non_test_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    ridge_alpha: float = 1.0,
    min_samples: int = 10,
) -> dict[str, Any]:
    """Compute leakage-free non-test->test Ridge R2 and NaN-safe correlations with OMNI, SOLAR-1, and OULU."""
    z_nt = np.asarray(z_non_test, dtype=np.float64)
    z_te = np.asarray(z_test, dtype=np.float64)
    nt_idx = np.asarray(non_test_idx, dtype=np.int64)
    te_idx = np.asarray(test_idx, dtype=np.int64)

    if z_nt.shape[0] != nt_idx.shape[0]:
        raise ValueError(f"z_non_test length {z_nt.shape[0]} != non_test_idx length {nt_idx.shape[0]}.")
    if z_te.shape[0] != te_idx.shape[0]:
        raise ValueError(f"z_test length {z_te.shape[0]} != test_idx length {te_idx.shape[0]}.")
    if z_nt.shape[1] != z_te.shape[1]:
        raise ValueError("z_non_test and z_test latent dimensions differ.")
    if np.intersect1d(nt_idx, te_idx).size > 0:
        raise ValueError(
            "non_test_idx and test_idx must be strictly disjoint (space-weather alignment leakage)."
        )

    omni_res: dict[str, Any] = {}
    for c_idx, col in enumerate(OMNI_COLS):
        omni_res[col] = _evaluate_single_sw_target(
            z_nt,
            z_te,
            sw_omni[:, c_idx],
            nt_idx,
            te_idx,
            ridge_alpha=ridge_alpha,
            min_samples=min_samples,
        )

    solar1_res: dict[str, Any] = {}
    for c_idx, col in enumerate(SOLAR1_BSN_COLS):
        solar1_res[col] = _evaluate_single_sw_target(
            z_nt,
            z_te,
            sw_solar1_bsn[:, c_idx],
            nt_idx,
            te_idx,
            ridge_alpha=ridge_alpha,
            min_samples=min_samples,
        )

    oulu_res = {
        "delta_n_pct": _evaluate_single_sw_target(
            z_nt,
            z_te,
            oulu,
            nt_idx,
            te_idx,
            ridge_alpha=ridge_alpha,
            min_samples=min_samples,
        )
    }

    return {
        "omni": omni_res,
        "solar1_bsn": solar1_res,
        "oulu": oulu_res,
    }
