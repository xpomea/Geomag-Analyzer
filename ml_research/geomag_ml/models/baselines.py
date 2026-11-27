"""Linear and geostatistical Phase 1 baselines (`EMPCA`, `RidgeSchur`, `IDW`, `OrdinaryKriging`)."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import least_squares

from geomag_ml.data.masks import MaskGenerator
from geomag_ml.data.preprocessing import FieldScaler
from geomag_ml.eval.metrics import StreamingMetricsAccumulator
from geomag_ml.models.base import BaseGeomagModel


def wrap_lon_diff_deg(delta_lon_deg: np.ndarray) -> np.ndarray:
    """Wrap longitude differences strictly into `[-180, 180)` degrees."""
    return ((np.asarray(delta_lon_deg, dtype=np.float64) + 180.0) % 360.0) - 180.0


def compute_magnetic_pair_components(
    coords_q: np.ndarray,
    coords_k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute `(d_MLat, wrap(d_MLon) * cos(mean_MLat))` arrays of shape `(N_q, N_k)`."""
    cq = np.asarray(coords_q, dtype=np.float64)
    ck = np.asarray(coords_k, dtype=np.float64)
    mlat_q = cq[:, 2][:, None]
    mlat_k = ck[:, 2][None, :]
    d_mlat = mlat_q - mlat_k
    mean_lat_rad = np.deg2rad(0.5 * (mlat_q + mlat_k))
    d_mlon_cos = wrap_lon_diff_deg(cq[:, 3][:, None] - ck[:, 3][None, :]) * np.cos(mean_lat_rad)
    return d_mlat, d_mlon_cos


def haversine_matrix_km(coords_q: np.ndarray, coords_k: np.ndarray) -> np.ndarray:
    """Vectorized orthodromic distance matrix `(N_q, N_k)` in km from `(lat, lon)`."""
    cq = np.asarray(coords_q, dtype=np.float64)
    ck = np.asarray(coords_k, dtype=np.float64)
    lat1 = np.deg2rad(cq[:, 0])[:, None]
    lon1 = np.deg2rad(cq[:, 1])[:, None]
    lat2 = np.deg2rad(ck[:, 0])[None, :]
    lon2 = np.deg2rad(ck[:, 1])[None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon / 2.0) ** 2)
    return 2.0 * 6371.0 * np.arcsin(np.minimum(1.0, np.sqrt(np.maximum(0.0, a))))


def ut_hours_from_time_features(time_features: np.ndarray) -> np.ndarray:
    """Recover UTC hours in `[0, 24)` from `[sin_ut, cos_ut, ...]`."""
    tf = np.asarray(time_features, dtype=np.float64)
    ang = np.mod(np.arctan2(tf[:, 0], tf[:, 1]), 2.0 * math.pi)
    return (ang / (2.0 * math.pi)) * 24.0


def _sq_design_matrix(ut_hours: np.ndarray, n_harmonics: int) -> np.ndarray:
    ut = np.asarray(ut_hours, dtype=np.float64)
    cols = [np.ones(ut.shape[0], dtype=np.float64)]
    base_ang = (2.0 * math.pi / 24.0) * ut
    for m in range(1, int(n_harmonics) + 1):
        ang = float(m) * base_ang
        cols.append(np.cos(ang))
        cols.append(np.sin(ang))
    return np.column_stack(cols)


def _pack_mask_rows(mask: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.packbits(np.asarray(mask, dtype=bool), axis=1))


def _group_unique_patterns(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group rows by exact boolean observation pattern; returns list of `(pattern_bool, row_indices)`."""
    m = np.asarray(mask, dtype=bool)
    if m.shape[0] == 0:
        return []
    packed = _pack_mask_rows(m)
    row_view = packed.view(np.dtype((np.void, packed.shape[1]))).ravel()
    _, inv = np.unique(row_view, return_inverse=True)
    order = np.argsort(inv, kind="mergesort")
    inv_sorted = inv[order]
    splits = np.flatnonzero(np.diff(inv_sorted)) + 1
    groups = np.split(order, splits)
    return [(m[int(grp[0])], grp.astype(np.int64)) for grp in groups]


def _count_unique_patterns(mask: np.ndarray) -> int:
    packed = _pack_mask_rows(mask)
    row_view = packed.view(np.dtype((np.void, packed.shape[1]))).ravel()
    return int(np.unique(row_view).shape[0])


def _resolve_ut_hours(
    n_t: int,
    time_features: np.ndarray | None,
    timestamps_unix_s: np.ndarray | None,
) -> np.ndarray:
    if time_features is not None:
        return ut_hours_from_time_features(time_features)
    if timestamps_unix_s is not None:
        ts = np.asarray(timestamps_unix_s, dtype=np.int64)
        return (ts % 86400).astype(np.float64) / 3600.0
    raise ValueError("EMPCABaseline with use_sq_detrend=True requires time_features or timestamps_unix_s.")


class EMPCABaseline(BaseGeomagModel):
    """Branch-A centered EM-PCA (`k=3..5`) with optional adaptive UT `Sq` harmonic detrending."""

    branch = "A"
    supports_zero_shot = False

    def __init__(
        self,
        k: int = 4,
        use_sq_detrend: bool = False,
        *,
        n_em_iters: int = 20,
        ridge_eps: float = 1e-4,
        sq_l2_scale: float = 0.01,
        sq_max_harmonics: int = 4,
        sq_min_ut_hour_bins: int = 20,
        sq_fallback_harmonics: int = 1,
    ) -> None:
        self.k = int(k)
        self.use_sq_detrend = bool(use_sq_detrend)
        self.n_em_iters = int(n_em_iters)
        self.ridge_eps = float(ridge_eps)
        self.sq_l2_scale = float(sq_l2_scale)
        self.sq_max_harmonics = int(sq_max_harmonics)
        self.sq_min_ut_hour_bins = int(sq_min_ut_hour_bins)
        self.sq_fallback_harmonics = int(sq_fallback_harmonics)
        self.model_name = "em_pca_sq" if self.use_sq_detrend else "em_pca"

        self.n_total_stations_: int = 0
        self.active_idx_: np.ndarray | None = None
        self.mu_st_: np.ndarray | None = None
        self.W_st_: np.ndarray | None = None
        self.G_st_: np.ndarray | None = None
        self.sq_coefs_: np.ndarray | None = None
        self.sq_orders_: np.ndarray | None = None
        self.k_eff_: int = 0

    def fit(
        self,
        X_work: np.ndarray,
        M_qc_clean: np.ndarray,
        train_idx: np.ndarray,
        in_train_mask: np.ndarray,
        *,
        time_features: np.ndarray | None = None,
        timestamps_unix_s: np.ndarray | None = None,
    ) -> EMPCABaseline:
        """Fit adaptive Sq harmonics (if enabled) and centered EM-PCA strictly on `train_idx`."""
        x_all = np.asarray(X_work)
        m_all = np.asarray(M_qc_clean, dtype=bool)
        tr_idx = np.asarray(train_idx, dtype=np.int64)
        in_tr = np.asarray(in_train_mask, dtype=bool)

        n_st_total = int(x_all.shape[1])
        active_idx = np.flatnonzero(in_tr).astype(np.int64)
        if active_idx.size == 0:
            raise ValueError("EMPCABaseline requires at least one in_train station.")

        x_tr = np.asarray(x_all[tr_idx][:, active_idx, :], dtype=np.float64)
        m_tr = m_all[tr_idx][:, active_idx] & np.all(np.isfinite(x_tr), axis=-1)
        x_tr = np.where(m_tr[..., None], x_tr, 0.0)

        n_t_tr, n_tr, _ = x_tr.shape
        sq_coefs = np.zeros((n_tr, 1 + 2 * self.sq_max_harmonics, 3), dtype=np.float64)
        sq_orders = np.zeros(n_tr, dtype=np.int64)

        if self.use_sq_detrend:
            tf_tr = time_features[tr_idx] if time_features is not None else None
            ts_tr = timestamps_unix_s[tr_idx] if timestamps_unix_s is not None else None
            ut_tr = _resolve_ut_hours(n_t_tr, tf_tr, ts_tr)
            for s in range(n_tr):
                m_s = m_tr[:, s]
                n_obs = int(np.sum(m_s))
                if n_obs == 0:
                    continue
                ut_s = ut_tr[m_s]
                ut_bins = np.unique(np.floor(ut_s).astype(np.int64) % 24)
                m_h = (
                    self.sq_max_harmonics
                    if ut_bins.size >= self.sq_min_ut_hour_bins
                    else self.sq_fallback_harmonics
                )
                phi = _sq_design_matrix(ut_s, m_h)
                reg = np.zeros(1 + 2 * m_h, dtype=np.float64)
                reg[1:] = self.sq_l2_scale * float(n_obs)
                gram = phi.T @ phi + np.diag(reg)
                rhs = phi.T @ x_tr[m_s, s, :]
                coef = np.linalg.solve(gram, rhs)
                sq_coefs[s, : 1 + 2 * m_h, :] = coef
                sq_orders[s] = m_h
                x_tr[m_s, s, :] = x_tr[m_s, s, :] - (phi @ coef)

        d_dim = 3 * n_tr
        r_obs = x_tr.reshape(n_t_tr, d_dim)
        m_mat = np.repeat(m_tr, 3, axis=1)

        counts = np.sum(m_mat, axis=0)
        sums = np.sum(np.where(m_mat, r_obs, 0.0), axis=0)
        mu = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
        r_comp = np.where(m_mat, r_obs, mu[None, :])

        k_eff = min(self.k, n_t_tr, d_dim)
        if k_eff < 1:
            raise ValueError("Insufficient rank to fit EMPCABaseline.")

        w_mat = np.zeros((d_dim, k_eff), dtype=np.float64)
        use_gram = n_t_tr > 2 * d_dim
        for _ in range(max(1, self.n_em_iters)):
            mu = np.mean(r_comp, axis=0)
            r_c = r_comp - mu[None, :]
            if use_gram:
                _, evecs = np.linalg.eigh(r_c.T @ r_c)
                w_mat = evecs[:, -k_eff:][:, ::-1]
            else:
                _, _, vt = np.linalg.svd(r_c, full_matrices=False)
                w_mat = vt[:k_eff].T
            z = r_c @ w_mat
            r_recon = mu[None, :] + (z @ w_mat.T)
            r_comp = np.where(m_mat, r_obs, r_recon)

        mu = np.mean(r_comp, axis=0)
        r_c = r_comp - mu[None, :]
        if use_gram:
            _, evecs = np.linalg.eigh(r_c.T @ r_c)
            w_mat = evecs[:, -k_eff:][:, ::-1].copy()
        else:
            _, _, vt = np.linalg.svd(r_c, full_matrices=False)
            w_mat = vt[:k_eff].T.copy()

        # Canonicalize column signs for reproducibility
        for col in range(k_eff):
            max_idx = int(np.argmax(np.abs(w_mat[:, col])))
            if w_mat[max_idx, col] < 0.0:
                w_mat[:, col] *= -1.0

        w_st = w_mat.reshape(n_tr, 3, k_eff)
        g_st = np.einsum("sck,scl->skl", w_st, w_st)

        self.n_total_stations_ = n_st_total
        self.active_idx_ = active_idx
        self.mu_st_ = mu.reshape(n_tr, 3)
        self.W_st_ = w_st
        self.G_st_ = g_st
        self.sq_coefs_ = sq_coefs
        self.sq_orders_ = sq_orders
        self.k_eff_ = k_eff
        return self

    def _compute_sq_offset(
        self,
        n_t: int,
        time_features: np.ndarray | None,
        timestamps_unix_s: np.ndarray | None,
    ) -> np.ndarray | None:
        if not self.use_sq_detrend or self.sq_coefs_ is None:
            return None
        ut = _resolve_ut_hours(n_t, time_features, timestamps_unix_s)
        phi = _sq_design_matrix(ut, self.sq_max_harmonics)
        return np.einsum("tp,spc->tsc", phi, self.sq_coefs_)

    def encode_masked(
        self,
        X_work: np.ndarray,
        obs_mask: np.ndarray,
        *,
        time_features: np.ndarray | None = None,
        mlt_hours: np.ndarray | None = None,
        station_coords: np.ndarray | None = None,
        timestamps_unix_s: np.ndarray | None = None,
    ) -> np.ndarray:
        """Project observed coordinates onto learned PCA basis via masked ridge OLS."""
        _ = mlt_hours, station_coords
        if self.active_idx_ is None or self.mu_st_ is None or self.W_st_ is None or self.G_st_ is None:
            raise RuntimeError("EMPCABaseline must be fitted before encode_masked().")

        x_a = np.asarray(X_work[:, self.active_idx_, :], dtype=np.float64)
        m_a = np.asarray(obs_mask[:, self.active_idx_], dtype=bool)
        x_clean = np.where(m_a[..., None], np.nan_to_num(x_a, nan=0.0), 0.0)

        sq = self._compute_sq_offset(x_clean.shape[0], time_features, timestamps_unix_s)
        if sq is not None:
            x_clean = np.where(m_a[..., None], x_clean - sq, 0.0)

        diff = np.where(m_a[..., None], x_clean - self.mu_st_[None, :, :], 0.0)
        rhs = np.einsum("tsc,sck->tk", diff, self.W_st_)

        gram = np.tensordot(m_a.astype(np.float64), self.G_st_, axes=([1], [0]))
        eye_k = self.ridge_eps * np.eye(self.k_eff_, dtype=np.float64)
        gram = gram + eye_k[None, :, :]
        z = np.linalg.solve(gram, rhs[..., None])[..., 0]
        return z.astype(np.float32)

    def predict_masked(
        self,
        X_work: np.ndarray,
        obs_mask: np.ndarray,
        target_mask: np.ndarray,
        *,
        time_features: np.ndarray | None = None,
        mlt_hours: np.ndarray | None = None,
        station_coords: np.ndarray | None = None,
        station_dist_km: np.ndarray | None = None,
        timestamps_unix_s: np.ndarray | None = None,
    ) -> np.ndarray:
        """Reconstruct `(T, N, 3)` field across `in_train` stations from latent `z(t)`."""
        _ = target_mask, mlt_hours, station_coords, station_dist_km
        if self.active_idx_ is None or self.mu_st_ is None or self.W_st_ is None:
            raise RuntimeError("EMPCABaseline must be fitted before predict_masked().")

        z = self.encode_masked(
            X_work,
            obs_mask,
            time_features=time_features,
            timestamps_unix_s=timestamps_unix_s,
        ).astype(np.float64)
        recon_a = self.mu_st_[None, :, :] + np.einsum("tk,sck->tsc", z, self.W_st_)
        sq = self._compute_sq_offset(recon_a.shape[0], time_features, timestamps_unix_s)
        if sq is not None:
            recon_a = recon_a + sq

        out = np.full(
            (X_work.shape[0], self.n_total_stations_, 3),
            np.nan,
            dtype=np.float32,
        )
        out[:, self.active_idx_, :] = recon_a.astype(np.float32)
        return out

    def count_parameters(self) -> int:
        if self.W_st_ is None or self.mu_st_ is None:
            return 0
        cnt = int(self.W_st_.size + self.mu_st_.size)
        if self.use_sq_detrend and self.sq_coefs_ is not None:
            cnt += int(self.sq_coefs_.size)
        return cnt

    def to_state_arrays(self) -> dict[str, np.ndarray]:
        if self.active_idx_ is None or self.mu_st_ is None or self.W_st_ is None:
            raise RuntimeError("EMPCABaseline is not fitted.")
        assert self.sq_coefs_ is not None and self.sq_orders_ is not None
        return {
            "meta_scalars": np.array(
                [
                    self.k,
                    int(self.use_sq_detrend),
                    self.n_em_iters,
                    self.ridge_eps,
                    self.sq_l2_scale,
                    self.sq_max_harmonics,
                    self.sq_min_ut_hour_bins,
                    self.sq_fallback_harmonics,
                    self.n_total_stations_,
                    self.k_eff_,
                ],
                dtype=np.float64,
            ),
            "active_idx": self.active_idx_.astype(np.int64),
            "mu_st": self.mu_st_.astype(np.float64),
            "W_st": self.W_st_.astype(np.float64),
            "sq_coefs": self.sq_coefs_.astype(np.float64),
            "sq_orders": self.sq_orders_.astype(np.int64),
        }

    @classmethod
    def from_state_arrays(cls, arrays: dict[str, np.ndarray]) -> EMPCABaseline:
        ms = arrays["meta_scalars"]
        obj = cls(
            k=int(ms[0]),
            use_sq_detrend=bool(int(ms[1])),
            n_em_iters=int(ms[2]),
            ridge_eps=float(ms[3]),
            sq_l2_scale=float(ms[4]),
            sq_max_harmonics=int(ms[5]),
            sq_min_ut_hour_bins=int(ms[6]),
            sq_fallback_harmonics=int(ms[7]),
        )
        obj.n_total_stations_ = int(ms[8])
        obj.k_eff_ = int(ms[9])
        obj.active_idx_ = arrays["active_idx"].astype(np.int64)
        obj.mu_st_ = arrays["mu_st"].astype(np.float64)
        obj.W_st_ = arrays["W_st"].astype(np.float64)
        obj.G_st_ = np.einsum("sck,scl->skl", obj.W_st_, obj.W_st_)
        obj.sq_coefs_ = arrays["sq_coefs"].astype(np.float64)
        obj.sq_orders_ = arrays["sq_orders"].astype(np.int64)
        return obj


class RidgeSchurBaseline(BaseGeomagModel):
    """Branch-B linear Gaussian Schur-complement predictor with EM covariance estimation."""

    branch = "B"
    supports_zero_shot = False
    model_name = "ridge_schur"

    def __init__(
        self,
        alpha_grid: Sequence[float] = (1e-2, 1e-1, 1.0, 10.0, 100.0),
        *,
        n_em_iters: int = 20,
        shrinkage_gamma: float = 0.05,
        min_eig: float = 1e-4,
    ) -> None:
        self.alpha_grid = tuple(float(a) for a in alpha_grid)
        self.n_em_iters = int(n_em_iters)
        self.shrinkage_gamma = float(shrinkage_gamma)
        self.min_eig = float(min_eig)

        self.n_total_stations_: int = 0
        self.active_idx_: np.ndarray | None = None
        self.mu_: np.ndarray | None = None
        self.Sigma_: np.ndarray | None = None
        self.best_alpha_: float = float(self.alpha_grid[0])
        self._cached_alpha_: float | None = None
        self._cached_P_: np.ndarray | None = None

    def _invalidate_precision_cache(self) -> None:
        self._cached_alpha_ = None
        self._cached_P_ = None

    def _get_precision_matrix(self) -> np.ndarray:
        if self.Sigma_ is None:
            raise RuntimeError("RidgeSchurBaseline must be fitted before computing precision matrix.")
        alpha = float(self.best_alpha_)
        if self._cached_P_ is not None and self._cached_alpha_ == alpha:
            return self._cached_P_
        d_dim = self.Sigma_.shape[0]
        a_mat = self.Sigma_.copy()
        a_mat.flat[:: d_dim + 1] += alpha
        l_fac = cho_factor(a_mat, check_finite=False)
        p_mat = cho_solve(l_fac, np.eye(d_dim, dtype=np.float64), check_finite=False)
        p_mat = 0.5 * (p_mat + p_mat.T)
        self._cached_alpha_ = alpha
        self._cached_P_ = p_mat
        return p_mat

    def fit(
        self,
        X_work: np.ndarray,
        M_qc_clean: np.ndarray,
        train_idx: np.ndarray,
        in_train_mask: np.ndarray,
        *,
        val_idx: np.ndarray | None = None,
        mask_generator: MaskGenerator | None = None,
        scaler: FieldScaler | None = None,
    ) -> RidgeSchurBaseline:
        """Fit EM-Gaussian `(mu, Sigma)` strictly on `train_idx` and tune `alpha` on `val_idx`."""
        x_all = np.asarray(X_work)
        m_all = np.asarray(M_qc_clean, dtype=bool)
        tr_idx = np.asarray(train_idx, dtype=np.int64)
        in_tr = np.asarray(in_train_mask, dtype=bool)

        n_st_total = int(x_all.shape[1])
        active_idx = np.flatnonzero(in_tr).astype(np.int64)
        n_tr = active_idx.size
        if n_tr < 2:
            raise ValueError("RidgeSchurBaseline requires at least 2 in_train stations.")

        x_tr = np.asarray(x_all[tr_idx][:, active_idx, :], dtype=np.float64)
        m_tr = m_all[tr_idx][:, active_idx] & np.all(np.isfinite(x_tr), axis=-1)
        x_tr = np.where(m_tr[..., None], x_tr, 0.0)

        n_t_tr = x_tr.shape[0]
        d_dim = 3 * n_tr
        r_obs = x_tr.reshape(n_t_tr, d_dim)
        m_mat = np.repeat(m_tr, 3, axis=1)

        counts = np.sum(m_mat, axis=0)
        sums = np.sum(np.where(m_mat, r_obs, 0.0), axis=0)
        mu = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
        sq_diff = np.where(m_mat, (r_obs - mu[None, :]) ** 2, 0.0)
        var = np.where(counts > 1, np.sum(sq_diff, axis=0) / np.maximum(counts - 1, 1), 1.0)
        var = np.maximum(var, self.min_eig)
        sigma = np.diag(var)

        patterns = _group_unique_patterns(m_tr)
        for _ in range(max(1, self.n_em_iters)):
            s1 = np.zeros(d_dim, dtype=np.float64)
            s2 = np.zeros((d_dim, d_dim), dtype=np.float64)
            for pat_st, rows_p in patterns:
                n_p = rows_p.shape[0]
                pat_d = np.repeat(pat_st, 3)
                o_idx = np.flatnonzero(pat_d)
                u_idx = np.flatnonzero(~pat_d)
                x_p = r_obs[rows_p]

                if u_idx.size == 0:
                    s1 += np.sum(x_p, axis=0)
                    s2 += x_p.T @ x_p
                elif o_idx.size == 0:
                    s1 += float(n_p) * mu
                    s2 += float(n_p) * (sigma + np.outer(mu, mu))
                else:
                    s_oo = sigma[np.ix_(o_idx, o_idx)].copy()
                    s_oo.flat[:: o_idx.size + 1] += 1e-6
                    s_ou = sigma[np.ix_(o_idx, u_idx)]
                    s_uu = sigma[np.ix_(u_idx, u_idx)]

                    l_fac = cho_factor(s_oo, check_finite=False)
                    k_gain = cho_solve(l_fac, s_ou, check_finite=False)
                    diff_o = x_p[:, o_idx] - mu[None, o_idx]
                    e_x = np.empty((n_p, d_dim), dtype=np.float64)
                    e_x[:, o_idx] = x_p[:, o_idx]
                    e_x[:, u_idx] = mu[None, u_idx] + diff_o @ k_gain
                    c_uu = s_uu - s_ou.T @ k_gain

                    s1 += np.sum(e_x, axis=0)
                    s2 += e_x.T @ e_x
                    s2[np.ix_(u_idx, u_idx)] += float(n_p) * c_uu

            mu = s1 / float(n_t_tr)
            sigma = s2 / float(n_t_tr) - np.outer(mu, mu)
            sigma = 0.5 * (sigma + sigma.T)
            evals, evecs = np.linalg.eigh(sigma)
            evals = np.maximum(evals, self.min_eig)
            sigma = (evecs * evals) @ evecs.T
            sigma = 0.5 * (sigma + sigma.T)

        sigma = (1.0 - self.shrinkage_gamma) * sigma + self.shrinkage_gamma * np.diag(np.diag(sigma))
        sigma = 0.5 * (sigma + sigma.T)
        if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))):
            raise RuntimeError("EM-Gaussian covariance estimation produced non-finite values.")

        self.n_total_stations_ = n_st_total
        self.active_idx_ = active_idx
        self.mu_ = mu
        self.Sigma_ = sigma
        self._invalidate_precision_cache()

        if val_idx is not None and mask_generator is not None and len(self.alpha_grid) > 1:
            self.best_alpha_ = self._tune_alpha(
                x_all, m_all, np.asarray(val_idx, dtype=np.int64), in_tr, mask_generator, scaler
            )
        else:
            self.best_alpha_ = float(self.alpha_grid[0])
        self._invalidate_precision_cache()
        return self

    def _tune_alpha(
        self,
        x_all: np.ndarray,
        m_all: np.ndarray,
        val_idx: np.ndarray,
        in_tr: np.ndarray,
        mask_generator: MaskGenerator,
        scaler: FieldScaler | None,
    ) -> float:
        stride = max(1, val_idx.shape[0] // 400)
        sub_idx = val_idx[::stride]
        x_val = x_all[sub_idx]
        m_val = m_all[sub_idx] & np.all(np.isfinite(x_val), axis=-1)
        y_true_nT = scaler.inverse_transform(x_val) if scaler is not None else np.asarray(x_val, dtype=np.float32)

        m1_list = mask_generator.generate_m1(m_val, m_val, in_tr)
        m2_list = mask_generator.generate_m2(m_val, m_val, in_tr, drop_frac=0.25, n_repeats=2, min_obs=4)
        scenarios = m1_list + m2_list
        if not scenarios:
            return float(self.alpha_grid[0])

        best_alpha = float(self.alpha_grid[0])
        best_rmse = math.inf
        dummy_belt = np.zeros(x_all.shape[1], dtype=np.int8)

        for alpha in self.alpha_grid:
            self.best_alpha_ = float(alpha)
            self._invalidate_precision_cache()
            acc = StreamingMetricsAccumulator(dummy_belt)
            for sc in scenarios:
                x_in = np.where(sc.obs_mask[..., None], np.nan_to_num(x_val, nan=0.0), np.float32(0.0))
                pred_work = self.predict_masked(x_in, sc.obs_mask, sc.target_mask)
                pred_nT = scaler.inverse_transform(pred_work) if scaler is not None else pred_work
                eval_m = sc.target_mask & m_val & in_tr[None, :]
                acc.update(y_true_nT, pred_nT, eval_m)
            res = acc.finalize()["global"]["vector_3d"]["rmse_nT"]
            if res is not None and float(res) < best_rmse:
                best_rmse = float(res)
                best_alpha = float(alpha)
        return best_alpha

    def predict_masked(
        self,
        X_work: np.ndarray,
        obs_mask: np.ndarray,
        target_mask: np.ndarray,
        *,
        time_features: np.ndarray | None = None,
        mlt_hours: np.ndarray | None = None,
        station_coords: np.ndarray | None = None,
        station_dist_km: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict missing stations via exact precision/Schur complement using only `obs_mask`."""
        _ = target_mask, time_features, mlt_hours, station_coords, station_dist_km
        if self.active_idx_ is None or self.mu_ is None or self.Sigma_ is None:
            raise RuntimeError("RidgeSchurBaseline must be fitted before predict_masked().")

        x_a = np.asarray(X_work[:, self.active_idx_, :], dtype=np.float64)
        m_a = np.asarray(obs_mask[:, self.active_idx_], dtype=bool)
        n_t, n_tr, _ = x_a.shape
        d_dim = 3 * n_tr
        x_flat = np.where(m_a[..., None], np.nan_to_num(x_a, nan=0.0), 0.0).reshape(n_t, d_dim)
        m_flat = np.repeat(m_a, 3, axis=1)

        pred_flat = np.broadcast_to(self.mu_[None, :], (n_t, d_dim)).copy()
        p_mat = self._get_precision_matrix()
        n_unique = _count_unique_patterns(m_a)

        if n_unique <= max(256, n_t // 4):
            patterns = _group_unique_patterns(m_a)
            for pat_st, rows_p in patterns:
                pat_d = np.repeat(pat_st, 3)
                o_idx = np.flatnonzero(pat_d)
                u_idx = np.flatnonzero(~pat_d)
                if o_idx.size == 0 or u_idx.size == 0:
                    continue
                diff_o = x_flat[rows_p][:, o_idx] - self.mu_[None, o_idx]
                if u_idx.size <= o_idx.size:
                    # Exact identity: A_{UO} A_{OO}^{-1} = -P_{UU}^{-1} P_{UO}
                    p_uu = p_mat[np.ix_(u_idx, u_idx)]
                    p_uo = p_mat[np.ix_(u_idx, o_idx)]
                    l_uu = cho_factor(p_uu, check_finite=False)
                    k_gain_t = -cho_solve(l_uu, p_uo, check_finite=False)
                    pred_flat[np.ix_(rows_p, u_idx)] = self.mu_[None, u_idx] + diff_o @ k_gain_t.T
                else:
                    s_oo = self.Sigma_[np.ix_(o_idx, o_idx)].copy()
                    s_oo.flat[:: o_idx.size + 1] += self.best_alpha_
                    s_ou = self.Sigma_[np.ix_(o_idx, u_idx)]
                    l_fac = cho_factor(s_oo, check_finite=False)
                    k_gain = cho_solve(l_fac, s_ou, check_finite=False)
                    pred_flat[np.ix_(rows_p, u_idx)] = self.mu_[None, u_idx] + diff_o @ k_gain
        else:
            diff_obs = np.where(m_flat, x_flat - self.mu_[None, :], 0.0)
            neg_p_diff = -(diff_obs @ p_mat)
            for t in range(n_t):
                u_idx = np.flatnonzero(~m_flat[t])
                if u_idx.size == 0 or u_idx.size == d_dim:
                    continue
                dx_u = np.linalg.solve(p_mat[np.ix_(u_idx, u_idx)], neg_p_diff[t, u_idx])
                pred_flat[t, u_idx] = self.mu_[u_idx] + dx_u

        out = np.full(
            (n_t, self.n_total_stations_, 3),
            np.nan,
            dtype=np.float32,
        )
        out[:, self.active_idx_, :] = pred_flat.reshape(n_t, n_tr, 3).astype(np.float32)
        return out

    def count_parameters(self) -> int:
        if self.mu_ is None or self.Sigma_ is None:
            return 0
        return int(self.mu_.size + self.Sigma_.size)

    def to_state_arrays(self) -> dict[str, np.ndarray]:
        if self.active_idx_ is None or self.mu_ is None or self.Sigma_ is None:
            raise RuntimeError("RidgeSchurBaseline is not fitted.")
        return {
            "meta_scalars": np.array(
                [
                    self.n_em_iters,
                    self.shrinkage_gamma,
                    self.min_eig,
                    self.n_total_stations_,
                    self.best_alpha_,
                ],
                dtype=np.float64,
            ),
            "alpha_grid": np.asarray(self.alpha_grid, dtype=np.float64),
            "active_idx": self.active_idx_.astype(np.int64),
            "mu": self.mu_.astype(np.float64),
            "Sigma": self.Sigma_.astype(np.float64),
        }

    @classmethod
    def from_state_arrays(cls, arrays: dict[str, np.ndarray]) -> RidgeSchurBaseline:
        ms = arrays["meta_scalars"]
        obj = cls(
            alpha_grid=arrays["alpha_grid"].tolist(),
            n_em_iters=int(ms[0]),
            shrinkage_gamma=float(ms[1]),
            min_eig=float(ms[2]),
        )
        obj.n_total_stations_ = int(ms[3])
        obj.best_alpha_ = float(ms[4])
        obj.active_idx_ = arrays["active_idx"].astype(np.int64)
        obj.mu_ = arrays["mu"].astype(np.float64)
        obj.Sigma_ = arrays["Sigma"].astype(np.float64)
        return obj


class IDWBaseline(BaseGeomagModel):
    """Branch-B Zero-Shot Inverse Distance Weighting in physical `nT` or `asinh_global` space."""

    branch = "B"
    supports_zero_shot = True

    def __init__(
        self,
        coord_metric: str = "magnetic",
        *,
        power_grid: Sequence[float] = (1.5, 2.0, 3.0),
        k_grid: Sequence[int | None] = (5, 10, None),
        anisotropy_grid: Sequence[float] = (0.5, 1.0),
        d0: float | None = None,
    ) -> None:
        metric = str(coord_metric).strip().lower()
        if metric not in ("magnetic", "haversine"):
            raise ValueError(f"Unsupported coord_metric {coord_metric!r}; expected 'magnetic' or 'haversine'.")
        self.coord_metric = metric
        self.power_grid = tuple(float(p) for p in power_grid)
        self.k_grid = tuple(None if k is None else int(k) for k in k_grid)
        self.anisotropy_grid = (
            tuple(float(a) for a in anisotropy_grid) if metric == "magnetic" else (1.0,)
        )
        self.d0 = float(d0) if d0 is not None else (0.25 if metric == "magnetic" else 25.0)
        self.model_name = f"idw_{metric}"

        self.station_coords_: np.ndarray | None = None
        self.station_dist_km_: np.ndarray | None = None
        self.best_power_: float = float(self.power_grid[0])
        self.best_k_: int | None = self.k_grid[0]
        self.best_anisotropy_: float = float(self.anisotropy_grid[0])

    @staticmethod
    def _reject_zscore(scaler: FieldScaler | None) -> None:
        if scaler is not None and scaler.mode == "zscore_per_station":
            raise ValueError(
                "IDWBaseline must operate in physical nT or asinh_global space, never zscore_per_station."
            )

    def _pairwise_distance(
        self,
        coords_q: np.ndarray,
        coords_k: np.ndarray,
        anisotropy: float,
    ) -> np.ndarray:
        if self.coord_metric == "haversine":
            if (
                coords_q is self.station_coords_
                and coords_k is self.station_coords_
                and self.station_dist_km_ is not None
            ):
                return self.station_dist_km_.astype(np.float64)
            return haversine_matrix_km(coords_q, coords_k)
        d_mlat, d_mlon_cos = compute_magnetic_pair_components(coords_q, coords_k)
        return np.hypot(d_mlat, float(anisotropy) * d_mlon_cos)

    def fit(
        self,
        X_work: np.ndarray,
        M_qc_clean: np.ndarray,
        train_idx: np.ndarray,
        in_train_mask: np.ndarray,
        *,
        station_coords: np.ndarray,
        station_dist_km: np.ndarray | None = None,
        val_idx: np.ndarray | None = None,
        mask_generator: MaskGenerator | None = None,
        scaler: FieldScaler | None = None,
    ) -> IDWBaseline:
        """Store station geometry and tune `(power, k, anisotropy)` strictly on `val_idx`."""
        _ = train_idx
        self._reject_zscore(scaler)
        self.station_coords_ = np.asarray(station_coords, dtype=np.float64).copy()
        self.station_dist_km_ = (
            np.asarray(station_dist_km, dtype=np.float64).copy()
            if station_dist_km is not None
            else None
        )

        if val_idx is not None and mask_generator is not None:
            self._tune_hyperparams(
                np.asarray(X_work),
                np.asarray(M_qc_clean, dtype=bool),
                np.asarray(val_idx, dtype=np.int64),
                np.asarray(in_train_mask, dtype=bool),
                mask_generator,
                scaler,
            )
        return self

    def _tune_hyperparams(
        self,
        x_all: np.ndarray,
        m_all: np.ndarray,
        val_idx: np.ndarray,
        in_tr: np.ndarray,
        mask_generator: MaskGenerator,
        scaler: FieldScaler | None,
    ) -> None:
        stride = max(1, val_idx.shape[0] // 300)
        sub_idx = val_idx[::stride]
        x_val = x_all[sub_idx]
        m_val = m_all[sub_idx] & np.all(np.isfinite(x_val), axis=-1)
        y_true_nT = scaler.inverse_transform(x_val) if scaler is not None else np.asarray(x_val, dtype=np.float32)

        m1_list = mask_generator.generate_m1(m_val, m_val, in_tr)
        m2_list = mask_generator.generate_m2(m_val, m_val, in_tr, drop_frac=0.25, n_repeats=2, min_obs=4)
        scenarios = m1_list + m2_list
        if not scenarios:
            return

        best_rmse = math.inf
        best_cfg = (self.best_power_, self.best_k_, self.best_anisotropy_)
        dummy_belt = np.zeros(x_all.shape[1], dtype=np.int8)

        for aniso in self.anisotropy_grid:
            for pwr in self.power_grid:
                for k_val in self.k_grid:
                    self.best_anisotropy_ = float(aniso)
                    self.best_power_ = float(pwr)
                    self.best_k_ = k_val
                    acc = StreamingMetricsAccumulator(dummy_belt)
                    for sc in scenarios:
                        x_in = np.where(sc.obs_mask[..., None], np.nan_to_num(x_val, nan=0.0), np.float32(0.0))
                        pred_work = self.predict_masked(x_in, sc.obs_mask, sc.target_mask)
                        pred_nT = scaler.inverse_transform(pred_work) if scaler is not None else pred_work
                        eval_m = sc.target_mask & m_val & in_tr[None, :]
                        acc.update(y_true_nT, pred_nT, eval_m)
                    res = acc.finalize()["global"]["vector_3d"]["rmse_nT"]
                    if res is not None and float(res) < best_rmse:
                        best_rmse = float(res)
                        best_cfg = (float(pwr), k_val, float(aniso))

        self.best_power_, self.best_k_, self.best_anisotropy_ = best_cfg

    def predict_masked(
        self,
        X_work: np.ndarray,
        obs_mask: np.ndarray,
        target_mask: np.ndarray,
        *,
        time_features: np.ndarray | None = None,
        mlt_hours: np.ndarray | None = None,
        station_coords: np.ndarray | None = None,
        station_dist_km: np.ndarray | None = None,
    ) -> np.ndarray:
        """Interpolate field for any station (including Zero-Shot) strictly from `obs_mask`."""
        _ = target_mask, time_features, mlt_hours, station_dist_km
        coords = (
            np.asarray(station_coords, dtype=np.float64)
            if station_coords is not None
            else self.station_coords_
        )
        if coords is None:
            raise RuntimeError("IDWBaseline requires station_coords via fit() or predict_masked().")

        x_clean = np.where(
            np.asarray(obs_mask, dtype=bool)[..., None],
            np.nan_to_num(np.asarray(X_work, dtype=np.float64), nan=0.0),
            0.0,
        )
        m_obs = np.asarray(obs_mask, dtype=bool)
        n_t, n_st, _ = x_clean.shape

        dist = self._pairwise_distance(coords, coords, self.best_anisotropy_)
        w_base = (dist + self.d0) ** (-self.best_power_)
        np.fill_diagonal(w_base, 0.0)

        m_tgt = np.asarray(target_mask, dtype=bool)
        active_targets = np.flatnonzero(np.any(m_tgt, axis=0))
        if active_targets.size == 0:
            active_targets = np.arange(n_st, dtype=np.int64)

        ord_by_j = {
            int(j): np.array(
                [idx for idx in np.argsort(dist[int(j)], kind="mergesort") if idx != int(j)],
                dtype=np.int64,
            )
            for j in active_targets
        }

        out = np.zeros((n_t, n_st, 3), dtype=np.float32)
        chunk = 1024
        for s in range(0, n_t, chunk):
            e = min(n_t, s + chunk)
            m_c = m_obs[s:e]
            x_c = x_clean[s:e]
            for j_idx in active_targets:
                j = int(j_idx)
                ord_j = ord_by_j[j]
                if ord_j.size == 0:
                    continue
                obs_j = m_c[:, ord_j]
                if self.best_k_ is not None:
                    keep_j = obs_j & (np.cumsum(obs_j, axis=1) <= self.best_k_)
                else:
                    keep_j = obs_j
                w_j = np.where(keep_j, w_base[j, ord_j][None, :], 0.0)
                w_sum = np.sum(w_j, axis=1, keepdims=True)
                w_norm = np.where(w_sum > 0.0, w_j / np.maximum(w_sum, 1e-12), 0.0)
                out[s:e, j, :] = np.einsum("ti,tic->tc", w_norm, x_c[:, ord_j, :]).astype(np.float32)
        return out

    def count_parameters(self) -> int:
        return 3

    def to_state_arrays(self) -> dict[str, np.ndarray]:
        if self.station_coords_ is None:
            raise RuntimeError("IDWBaseline is not fitted.")
        out = {
            "coord_metric": np.array([0 if self.coord_metric == "magnetic" else 1], dtype=np.int64),
            "meta_scalars": np.array(
                [
                    self.d0,
                    self.best_power_,
                    -1 if self.best_k_ is None else self.best_k_,
                    self.best_anisotropy_,
                ],
                dtype=np.float64,
            ),
            "power_grid": np.asarray(self.power_grid, dtype=np.float64),
            "k_grid": np.asarray([-1 if k is None else k for k in self.k_grid], dtype=np.int64),
            "anisotropy_grid": np.asarray(self.anisotropy_grid, dtype=np.float64),
            "station_coords": self.station_coords_.astype(np.float64),
        }
        if self.station_dist_km_ is not None:
            out["station_dist_km"] = self.station_dist_km_.astype(np.float64)
        return out

    @classmethod
    def from_state_arrays(cls, arrays: dict[str, np.ndarray]) -> IDWBaseline:
        metric = "magnetic" if int(arrays["coord_metric"][0]) == 0 else "haversine"
        ms = arrays["meta_scalars"]
        k_grid = [None if int(v) < 0 else int(v) for v in arrays["k_grid"].tolist()]
        obj = cls(
            coord_metric=metric,
            power_grid=arrays["power_grid"].tolist(),
            k_grid=k_grid,
            anisotropy_grid=arrays["anisotropy_grid"].tolist(),
            d0=float(ms[0]),
        )
        obj.best_power_ = float(ms[1])
        obj.best_k_ = None if int(ms[2]) < 0 else int(ms[2])
        obj.best_anisotropy_ = float(ms[3])
        obj.station_coords_ = arrays["station_coords"].astype(np.float64)
        obj.station_dist_km_ = (
            arrays["station_dist_km"].astype(np.float64) if "station_dist_km" in arrays else None
        )
        return obj


class OrdinaryKrigingBaseline(BaseGeomagModel):
    """Branch-B Zero-Shot Ordinary Kriging with anisotropic Matérn-3/2 variograms per `(X, Y, Z)`."""

    branch = "B"
    supports_zero_shot = True
    model_name = "ordinary_kriging"

    def __init__(
        self,
        *,
        min_nugget_ratio: float = 1e-2,
        max_nugget_ratio: float = 1.5,
        l_phi_bounds: tuple[float, float] = (1.5, 45.0),
        l_lon_bounds: tuple[float, float] = (2.0, 90.0),
    ) -> None:
        self.min_nugget_ratio = float(min_nugget_ratio)
        self.max_nugget_ratio = float(max_nugget_ratio)
        self.l_phi_bounds = (float(l_phi_bounds[0]), float(l_phi_bounds[1]))
        self.l_lon_bounds = (float(l_lon_bounds[0]), float(l_lon_bounds[1]))

        self.station_coords_: np.ndarray | None = None
        # Shape (3, 4): [L_phi, L_lon, sigma_s, sigma_n] per component X, Y, Z
        self.variogram_params_: np.ndarray | None = None

    @staticmethod
    def _matern32_kernel(
        d_mlat: np.ndarray,
        d_mlon_cos: np.ndarray,
        l_phi: float,
        l_lon: float,
        sigma_s: float,
    ) -> np.ndarray:
        r = np.hypot(d_mlat / float(l_phi), d_mlon_cos / float(l_lon))
        sr = math.sqrt(3.0) * r
        return (float(sigma_s) ** 2) * (1.0 + sr) * np.exp(-sr)

    def fit(
        self,
        X_work: np.ndarray,
        M_qc_clean: np.ndarray,
        train_idx: np.ndarray,
        in_train_mask: np.ndarray,
        *,
        station_coords: np.ndarray,
        scaler: FieldScaler | None = None,
    ) -> OrdinaryKrigingBaseline:
        """Estimate anisotropic Matérn-3/2 variogram parameters per component strictly on `train_idx`."""
        if scaler is not None and scaler.mode == "zscore_per_station":
            raise ValueError(
                "OrdinaryKrigingBaseline must operate in physical nT or asinh_global space, never zscore_per_station."
            )

        coords = np.asarray(station_coords, dtype=np.float64).copy()
        x_all = np.asarray(X_work)
        m_all = np.asarray(M_qc_clean, dtype=bool)
        tr_idx = np.asarray(train_idx, dtype=np.int64)
        in_tr = np.asarray(in_train_mask, dtype=bool)

        active_idx = np.flatnonzero(in_tr)
        if active_idx.size < 2:
            raise ValueError("OrdinaryKrigingBaseline requires at least 2 in_train stations.")

        x_tr = np.asarray(x_all[tr_idx][:, active_idx, :], dtype=np.float64)
        m_tr = m_all[tr_idx][:, active_idx] & np.all(np.isfinite(x_tr), axis=-1)

        c_act = coords[active_idx]
        d_mlat_all, d_mlon_all = compute_magnetic_pair_components(c_act, c_act)

        d_lat_pairs: list[float] = []
        d_lon_pairs: list[float] = []
        weights_list: list[float] = []
        gamma_rows: list[np.ndarray] = []

        n_act = active_idx.size
        for i in range(n_act):
            for j in range(i + 1, n_act):
                joint = m_tr[:, i] & m_tr[:, j]
                if int(np.sum(joint)) < 30:
                    continue
                diff = x_tr[joint, i, :] - x_tr[joint, j, :]
                gamma_ij = 0.5 * np.mean(diff * diff, axis=0)
                d_lat_p = float(d_mlat_all[i, j])
                d_lon_p = float(d_mlon_all[i, j])
                d_deg = math.hypot(d_lat_p, d_lon_p)
                d_lat_pairs.append(d_lat_p)
                d_lon_pairs.append(d_lon_p)
                weights_list.append(1.0 / (1.0 + d_deg / 15.0))
                gamma_rows.append(gamma_ij)

        params = np.zeros((3, 4), dtype=np.float64)
        if not gamma_rows:
            for c in range(3):
                params[c] = (10.0, 15.0, 1.0, max(self.min_nugget_ratio, 1e-2))
        else:
            d_lat_arr = np.asarray(d_lat_pairs, dtype=np.float64)
            d_lon_arr = np.asarray(d_lon_pairs, dtype=np.float64)
            w_sqrt = np.sqrt(np.asarray(weights_list, dtype=np.float64))
            gamma_mat = np.vstack(gamma_rows)

            lo = np.array(
                [
                    math.log(self.l_phi_bounds[0]),
                    math.log(self.l_lon_bounds[0]),
                    math.log(1e-3),
                    math.log(self.min_nugget_ratio),
                ],
                dtype=np.float64,
            )
            hi = np.array(
                [
                    math.log(self.l_phi_bounds[1]),
                    math.log(self.l_lon_bounds[1]),
                    math.log(1e4),
                    math.log(self.max_nugget_ratio),
                ],
                dtype=np.float64,
            )

            for c in range(3):
                g_c = gamma_mat[:, c]
                if float(np.max(g_c)) <= 1e-10:
                    params[c] = (10.0, 15.0, 1.0, max(self.min_nugget_ratio, 1e-2))
                    continue

                s_init = max(1e-2, min(5e3, math.sqrt(float(np.percentile(g_c, 85)))))
                ratio_init = max(self.min_nugget_ratio * 1.5, min(0.5, 0.2))
                x0 = np.array(
                    [math.log(12.0), math.log(20.0), math.log(s_init), math.log(ratio_init)],
                    dtype=np.float64,
                )
                x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)

                def _residuals(theta: np.ndarray, target_g: np.ndarray = g_c) -> np.ndarray:
                    lp = math.exp(float(theta[0]))
                    ll = math.exp(float(theta[1]))
                    ss = math.exp(float(theta[2]))
                    rat = math.exp(float(theta[3]))
                    sn = rat * ss
                    r = np.hypot(d_lat_arr / lp, d_lon_arr / ll)
                    sr = math.sqrt(3.0) * r
                    pred_g = (sn * sn) + (ss * ss) * (1.0 - (1.0 + sr) * np.exp(-sr))
                    return w_sqrt * (pred_g - target_g)

                res = least_squares(_residuals, x0, bounds=(lo, hi), max_nfev=150)
                lp = math.exp(float(res.x[0]))
                ll = math.exp(float(res.x[1]))
                ss = math.exp(float(res.x[2]))
                rat = max(self.min_nugget_ratio, math.exp(float(res.x[3])))
                params[c] = (lp, ll, ss, rat * ss)

        self.station_coords_ = coords
        self.variogram_params_ = params
        return self

    def predict_masked(
        self,
        X_work: np.ndarray,
        obs_mask: np.ndarray,
        target_mask: np.ndarray,
        *,
        time_features: np.ndarray | None = None,
        mlt_hours: np.ndarray | None = None,
        station_coords: np.ndarray | None = None,
        station_dist_km: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict any stations (including Zero-Shot) via Ordinary Kriging from `obs_mask`."""
        _ = target_mask, time_features, mlt_hours, station_dist_km
        if self.variogram_params_ is None:
            raise RuntimeError("OrdinaryKrigingBaseline must be fitted before predict_masked().")
        coords = (
            np.asarray(station_coords, dtype=np.float64)
            if station_coords is not None
            else self.station_coords_
        )
        if coords is None:
            raise RuntimeError("OrdinaryKrigingBaseline requires station_coords.")

        m_obs = np.asarray(obs_mask, dtype=bool)
        x_clean = np.where(
            m_obs[..., None],
            np.nan_to_num(np.asarray(X_work, dtype=np.float64), nan=0.0),
            0.0,
        )
        n_t, n_st, _ = x_clean.shape
        d_mlat, d_mlon_cos = compute_magnetic_pair_components(coords, coords)
        out = np.zeros((n_t, n_st, 3), dtype=np.float32)

        n_unique = _count_unique_patterns(m_obs)
        use_patterns = n_unique <= max(256, n_t // 4)
        patterns = _group_unique_patterns(m_obs) if use_patterns else []

        for c in range(3):
            lp, ll, ss, sn = (float(v) for v in self.variogram_params_[c])
            sn_eff = max(sn, self.min_nugget_ratio * max(ss, 1e-6))
            k_sig = self._matern32_kernel(d_mlat, d_mlon_cos, lp, ll, ss)
            k_obs = k_sig.copy()
            k_obs.flat[:: n_st + 1] += sn_eff * sn_eff

            if use_patterns:
                for pat_obs, rows_p in patterns:
                    o_idx = np.flatnonzero(pat_obs)
                    m_len = o_idx.size
                    if m_len == 0:
                        continue
                    a_mat = np.zeros((m_len + 1, m_len + 1), dtype=np.float64)
                    a_mat[:m_len, :m_len] = k_obs[np.ix_(o_idx, o_idx)]
                    a_mat[:m_len, m_len] = 1.0
                    a_mat[m_len, :m_len] = 1.0

                    rhs = np.empty((m_len + 1, n_st), dtype=np.float64)
                    rhs[:m_len, :] = k_sig[o_idx, :]
                    rhs[m_len, :] = 1.0

                    sol = np.linalg.solve(a_mat, rhs)
                    weights = sol[:m_len, :].T  # (N_st, |O|)
                    out[rows_p, :, c] = (x_clean[rows_p][:, o_idx, c] @ weights.T).astype(np.float32)
            else:
                k_aug = np.zeros((n_st + 1, n_st + 1), dtype=np.float64)
                k_aug[:n_st, :n_st] = k_obs
                k_aug[:n_st, n_st] = 1.0
                k_aug[n_st, :n_st] = 1.0
                eye_aug = np.eye(n_st + 1, dtype=np.float64)

                chunk = 256
                for s in range(0, n_t, chunk):
                    e = min(n_t, s + chunk)
                    m_c = m_obs[s:e]
                    has_obs = np.any(m_c, axis=1)
                    if not np.any(has_obs):
                        continue
                    m_active = m_c[has_obs]
                    b_act = m_active.shape[0]
                    m1 = np.concatenate([m_active, np.ones((b_act, 1), dtype=bool)], axis=1)
                    a_mat = np.where(m1[:, :, None] & m1[:, None, :], k_aug[None, :, :], eye_aug[None, :, :])

                    rhs = np.empty((b_act, n_st + 1, n_st), dtype=np.float64)
                    rhs[:, :n_st, :] = np.where(m_active[:, :, None], k_sig[None, :, :], 0.0)
                    rhs[:, n_st, :] = 1.0

                    sol = np.linalg.solve(a_mat, rhs)
                    weights = sol[:, :n_st, :]  # (B_act, N_obs_pad, N_q)
                    x_act = x_clean[s:e][has_obs][:, :, c]  # (B_act, N_obs_pad)
                    pred_act = np.einsum("bi,biq->bq", x_act, weights).astype(np.float32)

                    out_chunk = np.zeros((e - s, n_st), dtype=np.float32)
                    out_chunk[has_obs] = pred_act
                    out[s:e, :, c] = out_chunk

        return out

    def count_parameters(self) -> int:
        return 12 if self.variogram_params_ is not None else 0

    def to_state_arrays(self) -> dict[str, np.ndarray]:
        if self.variogram_params_ is None or self.station_coords_ is None:
            raise RuntimeError("OrdinaryKrigingBaseline is not fitted.")
        return {
            "meta_scalars": np.array(
                [
                    self.min_nugget_ratio,
                    self.max_nugget_ratio,
                    self.l_phi_bounds[0],
                    self.l_phi_bounds[1],
                    self.l_lon_bounds[0],
                    self.l_lon_bounds[1],
                ],
                dtype=np.float64,
            ),
            "variogram_params": self.variogram_params_.astype(np.float64),
            "station_coords": self.station_coords_.astype(np.float64),
        }

    @classmethod
    def from_state_arrays(cls, arrays: dict[str, np.ndarray]) -> OrdinaryKrigingBaseline:
        ms = arrays["meta_scalars"]
        obj = cls(
            min_nugget_ratio=float(ms[0]),
            max_nugget_ratio=float(ms[1]),
            l_phi_bounds=(float(ms[2]), float(ms[3])),
            l_lon_bounds=(float(ms[4]), float(ms[5])),
        )
        obj.variogram_params_ = arrays["variogram_params"].astype(np.float64)
        obj.station_coords_ = arrays["station_coords"].astype(np.float64)
        return obj
