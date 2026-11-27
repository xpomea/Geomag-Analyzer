"""Unified leakage-safe evaluator enforcing the `in_train` barrier across M1–M4 and Zero-Shot."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from geomag_ml.data.masks import MaskGenerator
from geomag_ml.data.preprocessing import BaselineEstimator, FieldScaler
from geomag_ml.data.splits import SplitFold
from geomag_ml.eval.metrics import (
    StreamingMetricsAccumulator,
    compute_latent_stability,
    compute_reconstruction_metrics,
    compute_space_weather_alignment,
)


def sanitize_masked_input(x: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
    """Safely zero masked/NaN inputs via `np.where` + `np.nan_to_num` (never `0 * NaN`)."""
    x_arr = np.asarray(x, dtype=np.float32)
    mask = np.asarray(obs_mask, dtype=bool)
    return np.where(mask[..., None], np.nan_to_num(x_arr, nan=0.0), np.float32(0.0)).astype(
        np.float32
    )


def _select_eval_indices(fold: SplitFold, eval_split: str) -> np.ndarray:
    split = str(eval_split).strip().lower()
    if split == "test":
        return fold.test_idx
    if split == "val":
        return fold.val_idx
    raise ValueError(f"Unsupported eval_split {eval_split!r}; expected 'test' or 'val'.")


def evaluate_model_on_fold(
    model: Any,
    dataset: dict[str, np.ndarray],
    meta: dict[str, Any],
    fold: SplitFold,
    baseline: BaselineEstimator,
    scaler: FieldScaler | None,
    mask_generator: MaskGenerator,
    *,
    eval_split: str = "test",
    eval_modes: Sequence[str] = ("M1", "M2", "M3_500km", "M3_1000km", "M4"),
    m2_drop_frac: float = 0.25,
    m2_repeats: int = 5,
    min_obs: int = 10,
    stability_drop_fracs: Sequence[float] = (0.20, 0.30, 0.40),
    stability_repeats: int = 5,
    use_qc_clean_obs: bool = True,
    return_outputs: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run unified evaluation on a single fold strictly respecting the `in_train` barrier."""
    eval_idx = _select_eval_indices(fold, eval_split)
    x_delta_all = baseline.transform(
        dataset["X_raw"],
        dataset["M_valid"],
        dataset["episode_idx"],
        timestamps_unix_s=dataset["timestamps_unix_s"],
        station_has_baseline=dataset["station_has_baseline"],
    )
    x_work_all = scaler.transform(x_delta_all) if scaler is not None else x_delta_all

    if baseline.in_train_stations_ is None:
        raise RuntimeError("BaselineEstimator must be fitted before evaluation.")
    in_train = fold.in_train_mask.astype(bool) & baseline.in_train_stations_.astype(bool)
    if scaler is not None:
        if scaler.in_train_stations_ is None:
            raise RuntimeError("FieldScaler must be fitted before evaluation.")
        in_train &= scaler.in_train_stations_.astype(bool)
    is_zero_shot = fold.is_zero_shot_mask.astype(bool)

    belt_id = dataset["station_belt_id"]
    coords = dataset["station_coords"]
    dist_km = dataset["station_dist_km"]
    codes = list(meta.get("station_codes", [f"ST{i}" for i in range(belt_id.shape[0])]))

    fin_eval = np.all(np.isfinite(x_delta_all[eval_idx]), axis=-1)
    valid_eval = dataset["M_valid"][eval_idx].astype(bool) & fin_eval
    qc_eval = dataset["M_qc_clean"][eval_idx].astype(bool) & valid_eval
    obs_valid_eval = qc_eval if use_qc_clean_obs else valid_eval

    y_true_nT = x_delta_all[eval_idx]
    x_work_eval = x_work_all[eval_idx]
    tf_eval = dataset["time_features"][eval_idx]
    mlt_eval = dataset["mlt_hours"][eval_idx]

    def _predict_nT(obs_m: np.ndarray, tgt_m: np.ndarray) -> np.ndarray:
        x_in = sanitize_masked_input(x_work_eval, obs_m)
        y_hat_work = model.predict_masked(
            x_in,
            obs_m,
            tgt_m,
            time_features=tf_eval,
            mlt_hours=mlt_eval,
            station_coords=coords,
            station_dist_km=dist_km,
        )
        return scaler.inverse_transform(y_hat_work) if scaler is not None else np.asarray(
            y_hat_work, dtype=np.float32
        )

    eval_masks_res: dict[str, Any] = {}
    outputs: dict[str, np.ndarray] = {}

    suite = mask_generator.generate_eval_suite(
        obs_valid_eval,
        qc_eval,
        in_train,
        modes=eval_modes,
        m2_drop_frac=m2_drop_frac,
        m2_repeats=m2_repeats,
        min_obs=min_obs,
    )

    for mode_name, scenarios in suite.items():
        acc = StreamingMetricsAccumulator(belt_id)
        m1_pred_accum = (
            np.full(y_true_nT.shape, np.nan, dtype=np.float32)
            if (return_outputs and mode_name == "M1" and getattr(model, "branch", "") == "B")
            else None
        )
        for sc in scenarios:
            if int(np.sum(sc.obs_mask & (~in_train[None, :]))) != 0:
                raise AssertionError(f"obs_mask leaked ~in_train station in {mode_name}/{sc.name}")
            if int(np.sum(sc.target_mask & (~in_train[None, :]))) != 0:
                raise AssertionError(f"target_mask leaked ~in_train station in {mode_name}/{sc.name}")

            y_hat_nT = _predict_nT(sc.obs_mask, sc.target_mask)
            eval_mask = sc.target_mask & qc_eval & in_train[None, :]
            acc.update(y_true_nT, y_hat_nT, eval_mask)
            if m1_pred_accum is not None and np.any(eval_mask):
                m1_pred_accum[eval_mask] = y_hat_nT[eval_mask]
        eval_masks_res[mode_name] = acc.finalize()
        if m1_pred_accum is not None:
            outputs["y_pred_m1"] = m1_pred_accum

    # Separate Zero-Shot bucket: strictly ~in_train targets, strictly in_train observations.
    if not bool(getattr(model, "supports_zero_shot", False)):
        eval_masks_res["zero_shot"] = {"status": "not_supported_fixed_stations"}
    else:
        zs_sc = mask_generator.generate_zero_shot(
            obs_valid_eval, qc_eval, in_train, is_zero_shot
        )
        if int(np.sum(zs_sc.obs_mask & (~in_train[None, :]))) != 0:
            raise AssertionError("zero_shot obs_mask leaked ~in_train station.")

        held_mask = zs_sc.meta["heldout_zero_shot_mask"] & qc_eval & (~in_train[None, :])
        single_mask = zs_sc.meta["single_episode_unseen_mask"] & qc_eval & (~in_train[None, :])

        if np.any(zs_sc.target_mask):
            y_hat_zs = _predict_nT(zs_sc.obs_mask, zs_sc.target_mask)
        else:
            y_hat_zs = np.zeros_like(y_true_nT, dtype=np.float32)

        held_res = compute_reconstruction_metrics(y_true_nT, y_hat_zs, held_mask, belt_id)
        single_res = compute_reconstruction_metrics(y_true_nT, y_hat_zs, single_mask, belt_id)
        single_res["baseline_note"] = "causal_initial_6h"

        per_station: dict[str, Any] = {}
        for st_idx in np.flatnonzero(~in_train):
            st_code = codes[int(st_idx)]
            st_mask = np.zeros_like(zs_sc.target_mask)
            st_mask[:, int(st_idx)] = zs_sc.target_mask[:, int(st_idx)] & qc_eval[:, int(st_idx)]
            st_metrics = compute_reconstruction_metrics(y_true_nT, y_hat_zs, st_mask, belt_id)[
                "global"
            ]
            per_station[st_code] = {
                "iaga_code": st_code,
                "station_idx": int(st_idx),
                "is_zero_shot": bool(is_zero_shot[int(st_idx)]),
                "baseline_note": "train_episodes"
                if bool(is_zero_shot[int(st_idx)])
                else "causal_initial_6h",
                **st_metrics,
            }

        eval_masks_res["zero_shot"] = {
            "heldout_zero_shot": held_res,
            "single_episode_unseen": single_res,
            "per_station": per_station,
        }
        if return_outputs and np.any(zs_sc.target_mask):
            zs_out = np.full(y_true_nT.shape, np.nan, dtype=np.float32)
            zs_out[zs_sc.target_mask] = y_hat_zs[zs_sc.target_mask]
            outputs["y_pred_zs"] = zs_out

    metrics: dict[str, Any] = {
        "eval_split": str(eval_split),
        "n_eval_rows": int(eval_idx.shape[0]),
        "n_in_train_stations": int(np.sum(in_train)),
        "eval_masks": eval_masks_res,
    }

    if getattr(model, "branch", "") == "A":
        obs_full = obs_valid_eval & in_train[None, :]
        x_full_in = sanitize_masked_input(x_work_eval, obs_full)
        z_full = np.asarray(
            model.encode_masked(
                x_full_in,
                obs_full,
                time_features=tf_eval,
                mlt_hours=mlt_eval,
                station_coords=coords,
            ),
            dtype=np.float32,
        )

        latent_stab: dict[str, Any] = {}
        last_decorr = 0.0
        for frac in stability_drop_fracs:
            frac_key = f"M2_{int(round(float(frac) * 100))}"
            m2_scens = mask_generator.generate_m2(
                obs_valid_eval,
                qc_eval,
                in_train,
                drop_frac=float(frac),
                n_repeats=int(stability_repeats),
                min_obs=min_obs,
                seed_offset=500,
            )
            z_masked_list: list[np.ndarray] = []
            for sc in m2_scens:
                x_m_in = sanitize_masked_input(x_work_eval, sc.obs_mask)
                z_m = np.asarray(
                    model.encode_masked(
                        x_m_in,
                        sc.obs_mask,
                        time_features=tf_eval,
                        mlt_hours=mlt_eval,
                        station_coords=coords,
                    ),
                    dtype=np.float32,
                )
                z_masked_list.append(z_m)
            stab_dict = compute_latent_stability(z_full, z_masked_list)
            last_decorr = float(stab_dict["axis_decorr_frob"])
            latent_stab[frac_key] = stab_dict

        metrics["latent_stability"] = latent_stab
        metrics["axis_decorr_frob"] = last_decorr

        # Encode strictly disjoint training rows for space-weather alignment regression:
        # on eval_split="test" use train+val (non_test_idx); on eval_split="val" use train_idx only.
        split_norm = str(eval_split).strip().lower()
        nt_idx = fold.non_test_idx if split_norm == "test" else fold.train_idx
        fin_nt = np.all(np.isfinite(x_delta_all[nt_idx]), axis=-1)
        m_nt_raw = (
            dataset["M_qc_clean"][nt_idx].astype(bool)
            if use_qc_clean_obs
            else dataset["M_valid"][nt_idx].astype(bool)
        )
        obs_nt = m_nt_raw & fin_nt & in_train[None, :]
        x_nt_in = sanitize_masked_input(x_work_all[nt_idx], obs_nt)
        z_non_test = np.asarray(
            model.encode_masked(
                x_nt_in,
                obs_nt,
                time_features=dataset["time_features"][nt_idx],
                mlt_hours=dataset["mlt_hours"][nt_idx],
                station_coords=coords,
            ),
            dtype=np.float32,
        )

        metrics["space_weather_alignment"] = compute_space_weather_alignment(
            z_non_test,
            z_full,
            dataset["sw_omni"],
            dataset["sw_solar1_bsn"],
            dataset["oulu_delta_n_pct"],
            nt_idx,
            eval_idx,
        )
        if return_outputs:
            outputs["z_test" if split_norm == "test" else "z_val"] = z_full

    return metrics, outputs
