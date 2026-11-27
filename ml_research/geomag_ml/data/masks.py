"""Unified observation/target mask generator (`M1`, `M2`, `M3`, `M4`, `zero_shot`, train mixtures)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from geomag_ml.data.schema import AURORAL_MLAT_THRESHOLD_DEG


@dataclass(frozen=True)
class MaskScenario:
    """Single deterministic or stochastic masking scenario with paired `obs_mask` and `target_mask`."""

    name: str
    obs_mask: np.ndarray
    target_mask: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


def assert_mask_invariants(
    obs_mask: np.ndarray,
    target_mask: np.ndarray,
    valid_mask: np.ndarray,
    qc_clean_mask: np.ndarray,
    in_train_mask: np.ndarray,
    *,
    mode_kind: str = "standard",
) -> None:
    """Verify strict disjointness, QC containment, and `in_train` isolation invariants."""
    obs = np.asarray(obs_mask, dtype=bool)
    tgt = np.asarray(target_mask, dtype=bool)
    val = np.asarray(valid_mask, dtype=bool)
    qc = np.asarray(qc_clean_mask, dtype=bool)
    in_tr = np.asarray(in_train_mask, dtype=bool)

    if obs.shape != val.shape or tgt.shape != val.shape or qc.shape != val.shape:
        raise ValueError("Mask shapes do not match valid_mask shape.")
    if in_tr.shape != (val.shape[1],):
        raise ValueError(f"in_train_mask shape {in_tr.shape} != ({val.shape[1]},).")

    allowed_obs = val & in_tr[None, :]
    if np.any(obs & (~allowed_obs)):
        raise AssertionError("obs_mask contains invalid or non-in_train stations.")
    if np.any(tgt & (~qc)):
        raise AssertionError("target_mask contains non-QC-clean entries.")
    if int(np.sum(obs & tgt)) != 0:
        raise AssertionError("obs_mask and target_mask are not disjoint (M_obs & M_target != empty).")

    active_rows = np.any(tgt, axis=-1)
    if np.any(active_rows):
        if not np.all(np.sum(obs[active_rows], axis=-1) >= 1):
            raise AssertionError("Active target row has zero observed stations in obs_mask.")

    if mode_kind in ("standard", "train"):
        if int(np.sum(tgt & (~in_tr[None, :]))) != 0:
            raise AssertionError(
                f"Standard/train target_mask leaked non-in_train stations under mode_kind={mode_kind!r}."
            )
    elif mode_kind == "zero_shot":
        if int(np.sum(tgt & in_tr[None, :])) != 0:
            raise AssertionError("zero_shot target_mask must strictly exclude in_train stations.")
        if int(np.sum(obs & (~in_tr[None, :]))) != 0:
            raise AssertionError("zero_shot obs_mask must strictly exclude ~in_train stations.")
    else:
        raise ValueError(f"Unsupported mode_kind {mode_kind!r}.")


class MaskGenerator:
    """Deterministic evaluation and stochastic training mask generator across M1–M4 and Zero-Shot."""

    def __init__(
        self,
        station_coords: np.ndarray,
        station_dist_km: np.ndarray,
        seed: int = 42,
    ) -> None:
        coords = np.asarray(station_coords, dtype=np.float64)
        dist = np.asarray(station_dist_km, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 4:
            raise ValueError(f"Expected station_coords shape (N, 4), got {coords.shape}.")
        n_st = coords.shape[0]
        if dist.shape != (n_st, n_st):
            raise ValueError(f"Expected station_dist_km shape ({n_st}, {n_st}), got {dist.shape}.")

        self.station_coords = coords
        self.station_dist_km = dist
        self.seed = int(seed)
        self.n_stations = n_st
        self.auroral_mask = np.abs(coords[:, 2]) >= AURORAL_MLAT_THRESHOLD_DEG
        self.neighbor_order = np.argsort(dist, axis=1, kind="mergesort")

    def _validate_eval_inputs(
        self,
        valid_mask: np.ndarray,
        qc_clean_mask: np.ndarray,
        in_train_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        val = np.asarray(valid_mask, dtype=bool)
        qc = np.asarray(qc_clean_mask, dtype=bool)
        in_tr = np.asarray(in_train_mask, dtype=bool)
        if val.ndim != 2 or val.shape[1] != self.n_stations:
            raise ValueError(f"Expected mask shape (T, {self.n_stations}), got {val.shape}.")
        if qc.shape != val.shape:
            raise ValueError(f"qc_clean_mask shape {qc.shape} != {val.shape}.")
        if in_tr.shape != (self.n_stations,):
            raise ValueError(f"in_train_mask shape {in_tr.shape} != ({self.n_stations},).")
        if np.any(qc & (~val)):
            raise ValueError("qc_clean_mask must be a subset of valid_mask.")
        return val, qc, in_tr

    def generate_m1(
        self,
        valid_mask: np.ndarray,
        qc_clean_mask: np.ndarray,
        in_train_mask: np.ndarray,
    ) -> list[MaskScenario]:
        """Leave-one-station-out masks across all active `in_train` stations."""
        val, qc, in_tr = self._validate_eval_inputs(valid_mask, qc_clean_mask, in_train_mask)
        obs_base = val & in_tr[None, :]
        scenarios: list[MaskScenario] = []

        for j in np.flatnonzero(in_tr):
            if not np.any(qc[:, j]):
                continue
            obs = obs_base.copy()
            obs[:, j] = False
            has_obs = np.sum(obs, axis=-1) >= 1
            tgt = np.zeros_like(obs_base)
            tgt[:, j] = qc[:, j] & has_obs
            if not np.any(tgt):
                continue
            assert_mask_invariants(obs, tgt, val, qc, in_tr, mode_kind="standard")
            scenarios.append(
                MaskScenario(
                    name=f"M1_st{int(j):02d}",
                    obs_mask=obs,
                    target_mask=tgt,
                    meta={"mode": "M1", "station_idx": int(j)},
                )
            )
        return scenarios

    def generate_m2(
        self,
        valid_mask: np.ndarray,
        qc_clean_mask: np.ndarray,
        in_train_mask: np.ndarray,
        *,
        drop_frac: float = 0.25,
        n_repeats: int = 5,
        min_obs: int = 10,
        seed_offset: int = 0,
    ) -> list[MaskScenario]:
        """Randomly drop `drop_frac` of observed `in_train` stations while keeping >= `min_obs`."""
        val, qc, in_tr = self._validate_eval_inputs(valid_mask, qc_clean_mask, in_train_mask)
        if not (0.0 < float(drop_frac) < 1.0):
            raise ValueError(f"drop_frac must be in (0, 1), got {drop_frac}.")
        if int(n_repeats) < 1:
            raise ValueError(f"n_repeats must be >= 1, got {n_repeats}.")

        obs_base = val & in_tr[None, :]
        n_t, n_st = obs_base.shape
        n_avail = np.sum(obs_base, axis=1)
        min_keep = max(1, int(min_obs))
        n_drop = np.where(
            n_avail > min_keep,
            np.minimum(np.rint(float(drop_frac) * n_avail).astype(np.int64), n_avail - min_keep),
            0,
        )
        n_drop = np.where(n_avail > min_keep, np.maximum(n_drop, 1), 0)

        frac_key = int(round(float(drop_frac) * 1000))
        rng = np.random.default_rng(self.seed + 2000 + int(seed_offset) + frac_key)

        scenarios: list[MaskScenario] = []
        for rep in range(int(n_repeats)):
            u = rng.random((n_t, n_st))
            u[~obs_base] = np.inf
            ranks = np.argsort(np.argsort(u, axis=1, kind="mergesort"), axis=1, kind="mergesort")
            dropped = obs_base & (ranks < n_drop[:, None])
            obs = obs_base & (~dropped)
            tgt = dropped & qc & in_tr[None, :]
            assert_mask_invariants(obs, tgt, val, qc, in_tr, mode_kind="standard")
            scenarios.append(
                MaskScenario(
                    name=f"M2_p{frac_key}_r{rep}",
                    obs_mask=obs,
                    target_mask=tgt,
                    meta={
                        "mode": "M2",
                        "drop_frac": float(drop_frac),
                        "repeat": int(rep),
                        "min_obs": min_keep,
                    },
                )
            )
        return scenarios

    def generate_m3(
        self,
        valid_mask: np.ndarray,
        qc_clean_mask: np.ndarray,
        in_train_mask: np.ndarray,
        *,
        radius_km: float = 500.0,
    ) -> list[MaskScenario]:
        """Hide spatial clusters within `radius_km` (>= 2 stations in cluster, >= 1 outside)."""
        val, qc, in_tr = self._validate_eval_inputs(valid_mask, qc_clean_mask, in_train_mask)
        if float(radius_km) <= 0.0:
            raise ValueError(f"radius_km must be positive, got {radius_km}.")

        obs_base = val & in_tr[None, :]
        seen_clusters: dict[bytes, tuple[np.ndarray, list[int]]] = {}

        for c in np.flatnonzero(in_tr):
            cluster = (self.station_dist_km[c] <= float(radius_km)) & in_tr
            if int(np.sum(cluster)) < 2:
                continue
            key = cluster.tobytes()
            if key in seen_clusters:
                seen_clusters[key][1].append(int(c))
            else:
                seen_clusters[key] = (cluster, [int(c)])

        scenarios: list[MaskScenario] = []
        rad_tag = int(round(float(radius_km)))
        for idx, (cluster, anchors) in enumerate(seen_clusters.values()):
            active = (np.sum(qc & cluster[None, :], axis=1) >= 2) & (
                np.sum(obs_base & (~cluster[None, :]), axis=1) >= 1
            )
            if not np.any(active):
                continue

            obs = obs_base.copy()
            obs[active] &= ~cluster[None, :]
            tgt = np.zeros_like(obs_base)
            tgt[active] = qc[active] & cluster[None, :]
            assert_mask_invariants(obs, tgt, val, qc, in_tr, mode_kind="standard")
            scenarios.append(
                MaskScenario(
                    name=f"M3_{rad_tag}km_c{idx:02d}",
                    obs_mask=obs,
                    target_mask=tgt,
                    meta={
                        "mode": f"M3_{rad_tag}km",
                        "radius_km": float(radius_km),
                        "anchor_indices": anchors,
                        "cluster_size": int(np.sum(cluster)),
                    },
                )
            )
        return scenarios

    def generate_m4(
        self,
        valid_mask: np.ndarray,
        qc_clean_mask: np.ndarray,
        in_train_mask: np.ndarray,
    ) -> list[MaskScenario]:
        """Hide all auroral stations (`|MLat| >= 60 deg`), observe only mid-latitude (`|MLat| < 60 deg`)."""
        val, qc, in_tr = self._validate_eval_inputs(valid_mask, qc_clean_mask, in_train_mask)
        obs = val & in_tr[None, :] & (~self.auroral_mask[None, :])
        has_obs = np.sum(obs, axis=-1, keepdims=True) >= 1
        tgt = qc & in_tr[None, :] & self.auroral_mask[None, :] & has_obs
        if not np.any(tgt):
            return []
        assert_mask_invariants(obs, tgt, val, qc, in_tr, mode_kind="standard")
        return [
            MaskScenario(
                name="M4_auroral",
                obs_mask=obs,
                target_mask=tgt,
                meta={"mode": "M4", "auroral_stations": np.flatnonzero(self.auroral_mask & in_tr).tolist()},
            )
        ]

    def generate_zero_shot(
        self,
        valid_mask: np.ndarray,
        qc_clean_mask: np.ndarray,
        in_train_mask: np.ndarray,
        is_zero_shot_mask: np.ndarray,
    ) -> MaskScenario:
        """Mask all `~in_train` stations out of `obs_mask` and separate `heldout_zero_shot` vs `single_episode_unseen`."""
        val, qc, in_tr = self._validate_eval_inputs(valid_mask, qc_clean_mask, in_train_mask)
        is_zs = np.asarray(is_zero_shot_mask, dtype=bool)
        if is_zs.shape != (self.n_stations,):
            raise ValueError(f"is_zero_shot_mask shape {is_zs.shape} != ({self.n_stations},).")

        obs = val & in_tr[None, :]
        has_obs = np.sum(obs, axis=-1, keepdims=True) >= 1

        heldout_zs = qc & is_zs[None, :] & (~in_tr[None, :]) & has_obs
        single_ep_unseen = qc & (~is_zs[None, :]) & (~in_tr[None, :]) & has_obs
        tgt = heldout_zs | single_ep_unseen
        assert_mask_invariants(obs, tgt, val, qc, in_tr, mode_kind="zero_shot")

        return MaskScenario(
            name="zero_shot",
            obs_mask=obs,
            target_mask=tgt,
            meta={
                "mode": "zero_shot",
                "heldout_zero_shot_mask": heldout_zs,
                "single_episode_unseen_mask": single_ep_unseen,
            },
        )

    def generate_eval_suite(
        self,
        valid_mask: np.ndarray,
        qc_clean_mask: np.ndarray,
        in_train_mask: np.ndarray,
        *,
        modes: Sequence[str] = ("M1", "M2", "M3_500km", "M3_1000km", "M4"),
        m2_drop_frac: float = 0.25,
        m2_repeats: int = 5,
        min_obs: int = 10,
    ) -> dict[str, list[MaskScenario]]:
        """Generate all requested evaluation mask suites cleanly separated under `in_train`."""
        suite: dict[str, list[MaskScenario]] = {}
        for mode in modes:
            key = str(mode)
            if key == "M1":
                suite[key] = self.generate_m1(valid_mask, qc_clean_mask, in_train_mask)
            elif key == "M2":
                suite[key] = self.generate_m2(
                    valid_mask,
                    qc_clean_mask,
                    in_train_mask,
                    drop_frac=m2_drop_frac,
                    n_repeats=m2_repeats,
                    min_obs=min_obs,
                )
            elif key == "M3_500km":
                suite[key] = self.generate_m3(valid_mask, qc_clean_mask, in_train_mask, radius_km=500.0)
            elif key == "M3_1000km":
                suite[key] = self.generate_m3(valid_mask, qc_clean_mask, in_train_mask, radius_km=1000.0)
            elif key == "M4":
                suite[key] = self.generate_m4(valid_mask, qc_clean_mask, in_train_mask)
            else:
                raise ValueError(f"Unsupported evaluation mask mode {key!r}.")
        return suite

    def sample_train_batch_masks(
        self,
        valid_mask: np.ndarray,
        qc_clean_mask: np.ndarray,
        in_train_mask: np.ndarray,
        *,
        probs: Mapping[str, float] | None = None,
        m2_drop_range: tuple[float, float] = (0.20, 0.35),
        m3_radius_range_km: tuple[float, float] = (500.0, 1000.0),
        m3_fallback_k: int = 3,
        m3_max_drop_frac: float = 0.35,
        min_obs: int = 10,
        station_dropout_count: tuple[int, int] = (0, 0),
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample stochastic training `(obs_mask, target_mask)` batch respecting `in_train` and `min_obs`."""
        val, qc, in_tr = self._validate_eval_inputs(valid_mask, qc_clean_mask, in_train_mask)
        n_b, n_st = val.shape
        gen = rng if rng is not None else np.random.default_rng(self.seed)

        prob_map = dict(probs) if probs is not None else {"M1": 0.35, "M2": 0.35, "M3": 0.30}
        mode_names = list(prob_map.keys())
        weights = np.array([max(0.0, float(prob_map[k])) for k in mode_names], dtype=np.float64)
        total_w = float(np.sum(weights))
        if total_w <= 0.0:
            raise ValueError(f"Mask mixture probabilities must sum > 0, got {prob_map}")
        weights /= total_w

        obs_base = val & in_tr[None, :]
        eff_in_tr = in_tr.copy()
        batch_dropped_stations = np.zeros(n_st, dtype=bool)

        drop_lo, drop_hi = int(station_dropout_count[0]), int(station_dropout_count[1])
        if drop_hi > 0:
            n_drop_st = int(gen.integers(max(0, drop_lo), drop_hi + 1))
            cand_st = np.flatnonzero(in_tr & np.any(qc, axis=0))
            min_row_avail = int(np.min(np.sum(obs_base, axis=1))) if n_b > 0 else 0
            max_safe_drop = max(0, min_row_avail - max(1, int(min_obs)))
            take_drop = min(n_drop_st, cand_st.size, max_safe_drop)
            if take_drop > 0:
                chosen = gen.choice(cand_st, size=take_drop, replace=False)
                batch_dropped_stations[chosen] = True
                eff_in_tr[chosen] = False

        obs_avail = val & eff_in_tr[None, :]
        obs_out = obs_avail.copy()
        tgt_out = np.zeros_like(obs_base)
        if np.any(batch_dropped_stations):
            has_obs = np.sum(obs_avail, axis=1, keepdims=True) >= 1
            tgt_out |= qc & batch_dropped_stations[None, :] & has_obs

        sampled_modes = gen.choice(mode_names, size=n_b, p=weights)
        min_keep = max(1, int(min_obs))

        for b in range(n_b):
            mode = str(sampled_modes[b])
            if mode == "none":
                continue

            avail_idx = np.flatnonzero(obs_avail[b])
            n_av = avail_idx.size
            if n_av <= 1:
                continue

            qc_avail_idx = np.flatnonzero(obs_avail[b] & qc[b])
            if qc_avail_idx.size == 0:
                continue

            dropped_b = np.zeros(n_st, dtype=bool)
            if mode == "M1":
                j = int(gen.choice(qc_avail_idx))
                dropped_b[j] = True
            elif mode == "M2":
                if n_av > min_keep:
                    frac = float(gen.uniform(m2_drop_range[0], m2_drop_range[1]))
                    cnt = min(max(1, int(round(frac * n_av))), n_av - min_keep)
                    chosen = gen.choice(avail_idx, size=cnt, replace=False)
                    dropped_b[chosen] = True
            elif mode == "M3":
                if n_av > min_keep:
                    anchor = int(gen.choice(qc_avail_idx))
                    radius = float(gen.uniform(m3_radius_range_km[0], m3_radius_range_km[1]))
                    in_circle = (self.station_dist_km[anchor] <= radius) & obs_avail[b]
                    max_drop = min(
                        max(1, int(float(m3_max_drop_frac) * n_av)),
                        max(1, n_av - min_keep),
                    )
                    if int(np.sum(in_circle)) >= 2:
                        ordered = [int(idx) for idx in self.neighbor_order[anchor] if in_circle[idx]]
                    else:
                        ordered = [int(idx) for idx in self.neighbor_order[anchor] if obs_avail[b, idx]][
                            : int(m3_fallback_k)
                        ]
                    chosen = ordered[:max_drop]
                    if chosen:
                        dropped_b[chosen] = True
            elif mode == "M4":
                mid_cnt = int(np.sum(obs_avail[b] & (~self.auroral_mask)))
                if mid_cnt >= 1:
                    dropped_b = obs_avail[b] & self.auroral_mask
            else:
                raise ValueError(f"Unsupported train mask mode {mode!r}.")

            if np.any(dropped_b) and int(np.sum(obs_avail[b] & (~dropped_b))) >= 1:
                obs_out[b] = obs_avail[b] & (~dropped_b)
                tgt_out[b] |= dropped_b & qc[b] & in_tr

        assert_mask_invariants(obs_out, tgt_out, val, qc, in_tr, mode_kind="train")
        return obs_out, tgt_out
