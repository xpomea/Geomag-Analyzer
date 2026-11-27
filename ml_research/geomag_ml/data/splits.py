"""Leakage-free split protocols (`looe_4fold`, `block_temporal`) and window boundary guard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from geomag_ml.data.schema import EVENT_FLAG_ECLIPSE, EVENT_FLAG_SPE

DEFAULT_ZERO_SHOT_STATIONS: tuple[str, ...] = ("LVV", "BEL", "MEA")
MIN_PURGE_GAP_MIN: int = 1440  # 24 hours in minutes

# QC-verified 4-fold LOOE assignment for October 2026 (§7 T1.1):
# Guarantees OMNI > 0 in train of every fold and 0 untrained test stations under M_valid & M_qc_clean
# (except structurally single-episode SUA in Fold 3).
LOOE_4FOLD_SPEC: tuple[tuple[int, tuple[int, ...], int, int], ...] = (
    (0, (1, 2), 3, 0),
    (1, (0, 2), 3, 1),
    (2, (0, 3), 1, 2),
    (3, (1, 2), 0, 3),
)


@dataclass(frozen=True)
class SplitFold:
    """Single train/val/test split fold with explicit station training masks."""

    fold_id: int
    protocol: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_episodes: tuple[int, ...]
    val_episodes: tuple[int, ...]
    test_episodes: tuple[int, ...]
    station_train_mask: np.ndarray
    is_zero_shot_mask: np.ndarray
    in_train_mask: np.ndarray

    @property
    def non_test_idx(self) -> np.ndarray:
        """Sorted union of `train_idx` and `val_idx` (strictly disjoint from `test_idx`)."""
        return np.sort(np.concatenate([self.train_idx, self.val_idx])).astype(np.int64)


def build_station_train_mask(
    station_codes: Sequence[str],
    zero_shot_stations: Sequence[str] = DEFAULT_ZERO_SHOT_STATIONS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(station_train_mask, is_zero_shot_mask)` boolean arrays of shape `(N,)`."""
    codes = [str(c) for c in station_codes]
    code_set = set(codes)
    zs_list = [str(c) for c in zero_shot_stations]
    unknown = [c for c in zs_list if c not in code_set]
    if unknown:
        raise ValueError(f"Unknown zero_shot_stations not present in dataset: {unknown}")

    zs_set = set(zs_list)
    is_zs = np.array([c in zs_set for c in codes], dtype=bool)
    st_train = ~is_zs
    return st_train, is_zs


def assert_no_eclipse_spe_in_train_or_val(
    event_flags: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> None:
    """Enforce that `eclipse` and `spe` minutes never appear in `train` or `val`."""
    flags = np.asarray(event_flags, dtype=np.uint8)
    special_mask = (flags & (EVENT_FLAG_ECLIPSE | EVENT_FLAG_SPE)) != 0
    tr_hits = int(np.sum(special_mask[train_idx]))
    va_hits = int(np.sum(special_mask[val_idx]))
    if tr_hits > 0 or va_hits > 0:
        raise AssertionError(
            f"Eclipse/SPE samples leaked outside test split: train_hits={tr_hits}, val_hits={va_hits}"
        )


def compute_min_temporal_gap_sec(
    timestamps_unix_s: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
) -> int:
    """Compute exact minimum `|t_a - t_b|` in seconds between two index sets in O((|A|+|B|) log |B|)."""
    a = np.asarray(idx_a, dtype=np.int64)
    b = np.asarray(idx_b, dtype=np.int64)
    if a.size == 0 or b.size == 0:
        raise ValueError("Cannot compute temporal gap for an empty index set.")

    ts = np.asarray(timestamps_unix_s, dtype=np.int64)
    ta = np.sort(ts[a])
    tb = np.sort(ts[b])
    pos = np.searchsorted(tb, ta)

    min_gap = np.iinfo(np.int64).max
    valid_right = pos < tb.size
    if np.any(valid_right):
        diff_right = np.min(tb[pos[valid_right]] - ta[valid_right])
        min_gap = min(min_gap, int(diff_right))
    valid_left = pos > 0
    if np.any(valid_left):
        diff_left = np.min(ta[valid_left] - tb[pos[valid_left] - 1])
        min_gap = min(min_gap, int(diff_left))
    return int(min_gap)


def _normalize_unique_indices(indices: np.ndarray, total_len: int, name: str) -> np.ndarray:
    arr = np.unique(np.asarray(indices, dtype=np.int64))
    if arr.size == 0:
        raise ValueError(f"Split {name!r} is empty.")
    if int(arr[0]) < 0 or int(arr[-1]) >= total_len:
        raise ValueError(f"Split {name!r} indices out of bounds [0, {total_len}).")
    return arr


def validate_split_temporal_gap(
    timestamps_unix_s: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    min_gap_sec: int = MIN_PURGE_GAP_MIN * 60,
    event_flags: np.ndarray | None = None,
) -> dict[str, int]:
    """Validate split disjointness, minimum purge gap (>= 86400s), and eclipse/SPE isolation."""
    ts = np.asarray(timestamps_unix_s, dtype=np.int64)
    tr = _normalize_unique_indices(train_idx, ts.shape[0], "train_idx")
    va = _normalize_unique_indices(val_idx, ts.shape[0], "val_idx")
    te = _normalize_unique_indices(test_idx, ts.shape[0], "test_idx")

    if np.intersect1d(tr, va).size > 0:
        raise ValueError("train_idx and val_idx overlap.")
    if np.intersect1d(tr, te).size > 0:
        raise ValueError("train_idx and test_idx overlap.")
    if np.intersect1d(va, te).size > 0:
        raise ValueError("val_idx and test_idx overlap.")

    gap_tr_te = compute_min_temporal_gap_sec(ts, tr, te)
    gap_tr_va = compute_min_temporal_gap_sec(ts, tr, va)
    gap_va_te = compute_min_temporal_gap_sec(ts, va, te)

    for label, g in (
        ("train-test", gap_tr_te),
        ("train-val", gap_tr_va),
        ("val-test", gap_va_te),
    ):
        if g < int(min_gap_sec):
            raise ValueError(
                f"Temporal gap {label}={g}s violates required purge gap {int(min_gap_sec)}s."
            )

    if event_flags is not None:
        assert_no_eclipse_spe_in_train_or_val(event_flags, tr, va)

    return {
        "train_test_gap_sec": gap_tr_te,
        "train_val_gap_sec": gap_tr_va,
        "val_test_gap_sec": gap_va_te,
    }


def _compute_effective_in_train(
    dataset: dict[str, np.ndarray],
    train_idx: np.ndarray,
    station_train_mask: np.ndarray,
    *,
    min_train_points: int,
    baseless_policy: str,
) -> np.ndarray:
    m_qc_tr = dataset["M_qc_clean"][train_idx].astype(bool)
    valid_cnt = np.sum(m_qc_tr, axis=0)
    in_train = station_train_mask.astype(bool) & (valid_cnt >= int(min_train_points))
    if baseless_policy == "exclude":
        in_train &= dataset["station_has_baseline"].astype(bool)
    return in_train


def _resolve_station_masks(
    n_st: int,
    meta: dict[str, Any] | None,
    zero_shot_stations: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if not zero_shot_stations:
        return np.ones(n_st, dtype=bool), np.zeros(n_st, dtype=bool)
    codes = (
        list(meta["station_codes"])
        if meta is not None and "station_codes" in meta
        else [f"ST{i}" for i in range(n_st)]
    )
    return build_station_train_mask(codes, zero_shot_stations)


def build_looe_4fold_splits(
    dataset: dict[str, np.ndarray],
    meta: dict[str, Any] | None = None,
    *,
    zero_shot_stations: Sequence[str] | None = None,
    min_train_points: int = 60,
    baseless_policy: str = "train_quiet_proxy",
    min_purge_gap_min: int = MIN_PURGE_GAP_MIN,
) -> list[SplitFold]:
    """Construct the QC-verified 4-fold Leave-One-Episode-Out CV splits."""
    if int(min_purge_gap_min) < MIN_PURGE_GAP_MIN:
        raise ValueError(
            f"min_purge_gap_min={min_purge_gap_min} must be >= {MIN_PURGE_GAP_MIN} minutes."
        )

    eb = dataset["episode_bounds"]
    if eb.shape[0] != 4:
        raise ValueError(
            f"looe_4fold requires exactly 4 episodes, found {eb.shape[0]}."
        )

    ep_idx = dataset["episode_idx"].astype(np.int64)
    ts = dataset["timestamps_unix_s"].astype(np.int64)
    ev = dataset["event_flags"].astype(np.uint8)
    n_st = int(dataset["X_raw"].shape[1])

    st_train_mask, is_zs_mask = _resolve_station_masks(n_st, meta, zero_shot_stations)
    min_gap_sec = int(min_purge_gap_min) * 60

    folds: list[SplitFold] = []
    for fold_id, tr_eps, va_ep, te_ep in LOOE_4FOLD_SPEC:
        tr_idx = np.flatnonzero(np.isin(ep_idx, tr_eps)).astype(np.int64)
        va_idx = np.flatnonzero(ep_idx == va_ep).astype(np.int64)
        te_idx = np.flatnonzero(ep_idx == te_ep).astype(np.int64)

        validate_split_temporal_gap(
            ts,
            tr_idx,
            va_idx,
            te_idx,
            min_gap_sec=min_gap_sec,
            event_flags=ev,
        )
        in_train = _compute_effective_in_train(
            dataset,
            tr_idx,
            st_train_mask,
            min_train_points=min_train_points,
            baseless_policy=baseless_policy,
        )
        folds.append(
            SplitFold(
                fold_id=int(fold_id),
                protocol="looe_4fold",
                train_idx=tr_idx,
                val_idx=va_idx,
                test_idx=te_idx,
                train_episodes=tuple(int(e) for e in tr_eps),
                val_episodes=(int(va_ep),),
                test_episodes=(int(te_ep),),
                station_train_mask=st_train_mask.copy(),
                is_zero_shot_mask=is_zs_mask.copy(),
                in_train_mask=in_train,
            )
        )
    return folds


def _filter_far_from_reference(
    ts_all: np.ndarray,
    candidate_idx: np.ndarray,
    ref_idx: np.ndarray,
    min_gap_sec: int,
) -> np.ndarray:
    if candidate_idx.size == 0 or ref_idx.size == 0:
        return candidate_idx
    ts_cand = ts_all[candidate_idx]
    ts_ref = np.sort(ts_all[ref_idx])
    pos = np.searchsorted(ts_ref, ts_cand)

    min_dist = np.full(candidate_idx.shape[0], np.iinfo(np.int64).max, dtype=np.int64)
    has_right = pos < ts_ref.size
    if np.any(has_right):
        min_dist[has_right] = np.minimum(
            min_dist[has_right], ts_ref[pos[has_right]] - ts_cand[has_right]
        )
    has_left = pos > 0
    if np.any(has_left):
        min_dist[has_left] = np.minimum(
            min_dist[has_left], ts_cand[has_left] - ts_ref[pos[has_left] - 1]
        )
    return candidate_idx[min_dist >= int(min_gap_sec)].astype(np.int64)


def build_block_temporal_split(
    dataset: dict[str, np.ndarray],
    meta: dict[str, Any] | None = None,
    *,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    test_frac: float = 0.20,
    purge_gap_min: int = MIN_PURGE_GAP_MIN,
    zero_shot_stations: Sequence[str] | None = None,
    min_train_points: int = 60,
    baseless_policy: str = "train_quiet_proxy",
) -> SplitFold:
    """Construct chronological `train/val/test` block split with >= 1d purge gaps and forced test for eclipse/SPE."""
    if int(purge_gap_min) < MIN_PURGE_GAP_MIN:
        raise ValueError(
            f"purge_gap_min={purge_gap_min} violates minimum {MIN_PURGE_GAP_MIN} minutes (R3)."
        )
    if train_frac <= 0.0 or val_frac <= 0.0 or test_frac <= 0.0:
        raise ValueError("train_frac, val_frac, and test_frac must all be positive.")
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0.")

    ts = dataset["timestamps_unix_s"].astype(np.int64)
    ep_idx = dataset["episode_idx"].astype(np.int64)
    ev = dataset["event_flags"].astype(np.uint8)
    n_t, n_st, _ = dataset["X_raw"].shape
    if n_t > 1 and np.any(np.diff(ts) <= 0):
        raise ValueError("timestamps_unix_s must be strictly increasing for block_temporal split.")

    purge_sec = int(purge_gap_min) * 60
    half_purge_sec = purge_sec // 2

    n_tr_end = int(round(n_t * float(train_frac)))
    n_va_end = int(round(n_t * float(train_frac + val_frac)))
    n_tr_end = max(1, min(n_t - 2, n_tr_end))
    n_va_end = max(n_tr_end + 1, min(n_t - 1, n_va_end))

    raw_train = np.arange(0, n_tr_end, dtype=np.int64)
    raw_val = np.arange(n_tr_end, n_va_end, dtype=np.int64)
    raw_test = np.arange(n_va_end, n_t, dtype=np.int64)

    # All eclipse & SPE timestamps are strictly pinned to test.
    special_idx = np.flatnonzero((ev & (EVENT_FLAG_ECLIPSE | EVENT_FLAG_SPE)) != 0).astype(np.int64)
    test_idx = np.unique(np.concatenate([raw_test, special_idx])).astype(np.int64)

    # Remove test indices and their purge neighborhood from train & val candidates.
    train_cand = np.setdiff1d(raw_train, test_idx, assume_unique=True)
    val_cand = np.setdiff1d(raw_val, test_idx, assume_unique=True)

    train_cand = _filter_far_from_reference(ts, train_cand, test_idx, purge_sec)
    val_cand = _filter_far_from_reference(ts, val_cand, test_idx, purge_sec)

    # Symmetrically trim the train/val boundary before exact distance filtering.
    t_cut_tr_va = int(ts[n_tr_end])
    train_sym = train_cand[ts[train_cand] <= (t_cut_tr_va - half_purge_sec)]
    if train_sym.size > 0:
        train_idx = train_sym
    else:
        train_idx = train_cand

    val_idx = _filter_far_from_reference(ts, val_cand, train_idx, purge_sec)

    validate_split_temporal_gap(
        ts,
        train_idx,
        val_idx,
        test_idx,
        min_gap_sec=purge_sec,
        event_flags=ev,
    )

    st_train_mask, is_zs_mask = _resolve_station_masks(n_st, meta, zero_shot_stations)
    in_train = _compute_effective_in_train(
        dataset,
        train_idx,
        st_train_mask,
        min_train_points=min_train_points,
        baseless_policy=baseless_policy,
    )

    return SplitFold(
        fold_id=0,
        protocol="block_temporal",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_episodes=tuple(int(e) for e in np.unique(ep_idx[train_idx])),
        val_episodes=tuple(int(e) for e in np.unique(ep_idx[val_idx])),
        test_episodes=tuple(int(e) for e in np.unique(ep_idx[test_idx])),
        station_train_mask=st_train_mask,
        is_zero_shot_mask=is_zs_mask,
        in_train_mask=in_train,
    )


def create_splits(
    protocol: str,
    dataset: dict[str, np.ndarray],
    meta: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[SplitFold]:
    """Dispatch split construction; explicitly forbids random minute-level splitting."""
    proto = str(protocol).strip().lower()
    if proto in ("random", "random_minute", "shuffle", "iid"):
        raise ValueError(
            "Random minute-level splitting is strictly forbidden due to 1-min autocorrelation (R3)."
        )
    if proto == "looe_4fold":
        return build_looe_4fold_splits(dataset, meta, **kwargs)
    if proto == "block_temporal":
        return [build_block_temporal_split(dataset, meta, **kwargs)]
    raise ValueError(f"Unsupported split protocol {protocol!r}; expected 'looe_4fold' or 'block_temporal'.")


def build_valid_window_anchors(
    timestamps_unix_s: np.ndarray,
    episode_idx: np.ndarray,
    split_indices: np.ndarray,
    W: int,
    mode: str = "causal",
) -> np.ndarray:
    """Return sorted anchor indices `t` whose full window lies inside one episode, one split, and continuous 60s steps."""
    ts = np.asarray(timestamps_unix_s, dtype=np.int64)
    ep = np.asarray(episode_idx, dtype=np.int64)
    if ts.ndim != 1 or ep.shape != ts.shape:
        raise ValueError("timestamps_unix_s and episode_idx must be 1D arrays of equal length.")
    n_t = ts.shape[0]

    win_len = int(W)
    mode_norm = str(mode).strip().lower()
    if mode_norm not in ("causal", "centered"):
        raise ValueError(f"Unsupported window mode {mode!r}; expected 'causal' or 'centered'.")
    if mode_norm == "causal" and win_len < 1:
        raise ValueError(f"Causal window length W must be >= 1, got {win_len}.")
    if mode_norm == "centered" and win_len < 2:
        raise ValueError(f"Centered window length W must be >= 2, got {win_len}.")

    split_arr = np.asarray(split_indices)
    if split_arr.dtype == np.bool_:
        if split_arr.shape != (n_t,):
            raise ValueError(f"Boolean split_indices shape {split_arr.shape} != ({n_t},).")
        split_idx = np.flatnonzero(split_arr).astype(np.int64)
    else:
        split_idx = np.unique(split_arr.astype(np.int64))
        if split_idx.size > 0 and (int(split_idx[0]) < 0 or int(split_idx[-1]) >= n_t):
            raise ValueError(f"split_indices out of bounds [0, {n_t}).")

    if split_idx.size < win_len:
        return np.empty(0, dtype=np.int64)

    if mode_norm == "causal":
        left = win_len - 1
        right = 0
    else:
        left = win_len // 2
        right = win_len - 1 - left

    in_split = np.zeros(n_t, dtype=np.int64)
    in_split[split_idx] = 1
    pref_split = np.zeros(n_t + 1, dtype=np.int64)
    np.cumsum(in_split, out=pref_split[1:])

    step_ok = np.zeros(n_t, dtype=np.int64)
    if n_t > 1:
        step_ok[1:] = ((np.diff(ts) == 60) & (np.diff(ep) == 0)).astype(np.int64)
    pref_step = np.zeros(n_t + 1, dtype=np.int64)
    np.cumsum(step_ok, out=pref_step[1:])

    cand = split_idx[(split_idx >= left) & (split_idx <= (n_t - 1 - right))]
    if cand.size == 0:
        return np.empty(0, dtype=np.int64)

    starts = cand - left
    ends_excl = cand + right + 1
    full_in_split = (pref_split[ends_excl] - pref_split[starts]) == win_len
    full_cont_steps = (pref_step[ends_excl] - pref_step[starts + 1]) == (win_len - 1)
    return cand[full_in_split & full_cont_steps].astype(np.int64)
