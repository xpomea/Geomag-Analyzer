"""Unified CLI training entrypoint for Phase 2–4 neural models (local CPU and Colab/Kaggle T4)."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import time
from typing import Any

import numpy as np

from geomag_ml import ML_ROOT, REPO_ROOT
from geomag_ml.data.datasets import SingleStepDataset
from geomag_ml.data.masks import MaskGenerator
from geomag_ml.data.preprocessing import BaselineEstimator, FieldScaler
from geomag_ml.data.schema import load_ml_dataset
from geomag_ml.data.splits import DEFAULT_ZERO_SHOT_STATIONS, create_splits
from geomag_ml.eval.evaluator import evaluate_model_on_fold
from geomag_ml.train.artifacts import (
    build_model,
    build_stations_manifest,
    make_run_id,
    save_run_bundle,
)
from geomag_ml.train.trainer import TrainConfig, Trainer
from geomag_ml.utils.device import resolve_device
from geomag_ml.utils.io import (
    compute_sha256,
    get_git_commit,
    portable_path_str,
    read_yaml,
    resolve_project_path,
)
from geomag_ml.utils.seed import set_seed


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _select_ablation_variants(
    cfg: dict[str, Any],
    ablation_arg: str,
) -> list[tuple[str, dict[str, Any]]]:
    base_cfg = deepcopy(dict(cfg.get("base", {})))
    ablations_map = dict(cfg.get("ablations", {}))
    req = str(ablation_arg).strip()
    if req.lower() == "base":
        return [("base", base_cfg)]
    if req.lower() == "all":
        variants = [("base", base_cfg)]
        for name, ov in ablations_map.items():
            variants.append((str(name), _deep_merge(base_cfg, dict(ov))))
        return variants

    variants = []
    for raw_name in [s.strip() for s in req.split(",") if s.strip()]:
        if raw_name.lower() == "base":
            variants.append(("base", base_cfg))
        elif raw_name in ablations_map:
            variants.append((raw_name, _deep_merge(base_cfg, dict(ablations_map[raw_name]))))
        else:
            raise ValueError(
                f"Unknown ablation {raw_name!r}; available={sorted(['base', *ablations_map.keys()])}."
            )
    return variants


def _resolve_time_cond_dim(model_cfg: dict[str, Any], meta: dict[str, Any]) -> int:
    use_ut = bool(model_cfg.get("use_ut_cond", True))
    if not use_ut:
        return 0
    use_doy_raw = model_cfg.get("use_doy", "auto")
    if isinstance(use_doy_raw, bool):
        use_doy = use_doy_raw
    else:
        use_doy = bool(meta.get("recommended_use_doy", False))
    return 4 if use_doy else 2


def _build_model_kwargs(
    model_type: str,
    model_cfg: dict[str, Any],
    n_stations: int,
    meta: dict[str, Any],
) -> dict[str, Any]:
    mtype = str(model_type).strip().lower()
    if mtype in ("mlp_ae", "maskedconditionalmlpae"):
        return {
            "n_stations": int(n_stations),
            "latent_dim": int(model_cfg.get("latent_dim", 5)),
            "time_cond_dim": _resolve_time_cond_dim(model_cfg, meta),
            "bottleneck_mode": str(model_cfg.get("bottleneck_mode", "ae_decorr")),
            "hidden_dim": int(model_cfg.get("hidden_dim", 256)),
            "mid_dim": int(model_cfg.get("mid_dim", 128)),
            "dropout": float(model_cfg.get("dropout", 0.1)),
        }
    if mtype in ("mlp_vm", "maskedwidemlp"):
        return {
            "n_stations": int(n_stations),
            "hidden_dim": int(model_cfg.get("hidden_dim", 512)),
            "n_blocks": int(model_cfg.get("n_blocks", 3)),
            "dropout": float(model_cfg.get("dropout", 0.15)),
        }
    raise ValueError(f"Unsupported model_type {model_type!r} in train.py.")


def _extract_metric(
    metrics: dict[str, Any],
    mode: str,
    belt: str,
    comp: str,
    field: str,
) -> float | None:
    bucket = metrics.get("eval_masks", {}).get(mode)
    if not isinstance(bucket, dict) or "status" in bucket:
        return None
    val = bucket.get(belt, {}).get(comp, {}).get(field)
    return float(val) if val is not None else None


def _fmt(val: float | None, fmt: str = ".2f") -> str:
    return "None" if val is None else format(val, fmt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Phase 2–4 neural geomagnetic model.")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML config.")
    parser.add_argument("--dataset", type=Path, default=None, help="Optional .npz dataset override.")
    parser.add_argument(
        "--fold",
        type=str,
        default=None,
        help="Fold index (0..3) or 'all' (defaults to '0' under --smoke, else 'all').",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="base",
        help="Variant to run: 'base', 'all', or comma-separated ablation keys.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: 'auto', 'cpu', or 'cuda'.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Artifacts output root directory (default: ml_research/artifacts).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick smoke run (3 epochs, patience=3, defaults to fold 0 if --fold omitted).",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-epoch logs.")
    args = parser.parse_args()

    cfg = read_yaml(resolve_project_path(args.config))
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    device = resolve_device(args.device)

    dataset_path = resolve_project_path(args.dataset or cfg["dataset_path"])
    out_dir = resolve_project_path(args.out_dir or cfg.get("out_dir", ML_ROOT / "artifacts"))
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset, meta = load_ml_dataset(dataset_path, validate=True)
    dataset_sha256 = meta.get("dataset_sha256") or compute_sha256(dataset_path)
    raw_tag = str(meta.get("tag", dataset_path.stem))
    dataset_tag = "oct2026" if "oct2026" in raw_tag else raw_tag.split("_")[0]
    git_commit = get_git_commit(REPO_ROOT)

    phase = str(cfg.get("phase", "phase2"))
    phase_prefix = str(cfg.get("phase_prefix", "p2"))
    branch = str(cfg.get("branch", "B"))
    model_type = str(cfg["model_type"])
    split_protocol = str(cfg.get("split_protocol", "looe_4fold"))
    zs_stations = tuple(cfg.get("default_zero_shot_stations", list(DEFAULT_ZERO_SHOT_STATIONS)))
    eval_cfg = dict(cfg.get("eval", {}))

    fold_arg = args.fold if args.fold is not None else ("0" if args.smoke else "all")
    variants = _select_ablation_variants(cfg, args.ablation)

    splits_cache: dict[tuple[str, bool], list[Any]] = {}
    prep_cache: dict[
        tuple[str, str, str, str, bool, int],
        tuple[BaselineEstimator, FieldScaler | None, np.ndarray, np.ndarray],
    ] = {}

    n_st = int(dataset["X_raw"].shape[1])
    t_all_start = time.perf_counter()

    for var_name, var_cfg in variants:
        tag = str(var_cfg.get("tag", var_name))
        scaler_mode = str(var_cfg.get("scaler_mode", "zscore_per_station"))
        baseline_mode = str(
            var_cfg.get("baseline_mode", cfg.get("baseline_mode", "temp_train_quiet_proxy"))
        )
        baseless_policy = str(
            var_cfg.get("baseless_policy", cfg.get("baseless_policy", "train_quiet_proxy"))
        )
        variation_step_policy = str(
            var_cfg.get(
                "variation_step_policy", cfg.get("variation_step_policy", "flag_qc_exclude")
            )
        )
        is_zs = bool(var_cfg.get("zero_shot", cfg.get("zero_shot", False)))

        split_key = (baseless_policy, is_zs)
        if split_key not in splits_cache:
            splits_cache[split_key] = create_splits(
                split_protocol,
                dataset,
                meta,
                zero_shot_stations=zs_stations if is_zs else None,
                baseless_policy=baseless_policy,
            )
        folds = splits_cache[split_key]

        if str(fold_arg).strip().lower() == "all":
            selected_folds = list(range(len(folds)))
        else:
            selected_folds = [int(s.strip()) for s in str(fold_arg).split(",") if s.strip()]

        model_cfg = dict(var_cfg.get("model", {}))
        model_kwargs = _build_model_kwargs(model_type, model_cfg, n_st, meta)

        train_dict = dict(var_cfg.get("train", {}))
        if args.smoke:
            train_dict["max_epochs"] = min(3, int(train_dict.get("max_epochs", 3)))
            train_dict["early_stopping_patience"] = min(
                3, int(train_dict.get("early_stopping_patience", 3))
            )
        train_cfg = TrainConfig.from_dict(train_dict, seed=seed)

        for f_id in selected_folds:
            fold = folds[f_id]
            t0 = time.perf_counter()
            set_seed(seed)

            prep_key = (
                baseline_mode,
                baseless_policy,
                variation_step_policy,
                scaler_mode,
                is_zs,
                f_id,
            )
            if prep_key not in prep_cache:
                est = BaselineEstimator(
                    mode=baseline_mode,
                    baseless_policy=baseless_policy,
                    variation_step_policy=variation_step_policy,
                ).fit(dataset, fold.train_idx, station_train_mask=fold.station_train_mask)
                x_delta = est.transform(
                    dataset["X_raw"],
                    dataset["M_valid"],
                    dataset["episode_idx"],
                    timestamps_unix_s=dataset["timestamps_unix_s"],
                    station_has_baseline=dataset["station_has_baseline"],
                )
                if scaler_mode == "none":
                    scaler_obj = None
                    x_work = x_delta
                else:
                    scaler_obj = FieldScaler(mode=scaler_mode).fit(
                        x_delta,
                        dataset["M_qc_clean"],
                        fold.train_idx,
                        dataset["station_belt_id"],
                        station_train_mask=fold.station_train_mask,
                    )
                    x_work = scaler_obj.transform(x_delta)
                prep_cache[prep_key] = (est, scaler_obj, x_delta, x_work)
            else:
                est, scaler_obj, x_delta, x_work = prep_cache[prep_key]

            in_train = fold.in_train_mask & est.in_train_stations_
            if scaler_obj is not None and scaler_obj.in_train_stations_ is not None:
                in_train = in_train & scaler_obj.in_train_stations_

            train_ds = SingleStepDataset(
                x_work,
                dataset["M_valid"],
                dataset["M_qc_clean"],
                dataset["time_features"],
                fold.train_idx,
                in_train,
                mlt_hours=dataset["mlt_hours"],
                x_delta_nT=x_delta,
            )
            val_ds = SingleStepDataset(
                x_work,
                dataset["M_valid"],
                dataset["M_qc_clean"],
                dataset["time_features"],
                fold.val_idx,
                in_train,
                mlt_hours=dataset["mlt_hours"],
                x_delta_nT=x_delta,
            )

            mask_gen = MaskGenerator(
                dataset["station_coords"],
                dataset["station_dist_km"],
                seed=seed + f_id * 100,
            )

            model = build_model(model_type, model_kwargs)
            if not args.quiet:
                print(
                    f"\n=== [{branch}] {tag} (ablation={var_name}) | fold={f_id} | "
                    f"device={device} | params={model.count_parameters():,} ==="
                )

            trainer = Trainer(model, train_cfg, device)
            fit_res = trainer.fit(
                train_ds,
                val_ds,
                scaler_obj,
                mask_gen,
                dataset["station_belt_id"],
                verbose=not args.quiet,
            )

            metrics, outputs = evaluate_model_on_fold(
                model,
                dataset,
                meta,
                fold,
                est,
                scaler_obj,
                mask_gen,
                eval_split="test",
                eval_modes=eval_cfg.get("modes", ("M1", "M2", "M3_500km", "M3_1000km", "M4")),
                m2_drop_frac=float(eval_cfg.get("m2_drop_frac", 0.25)),
                m2_repeats=int(eval_cfg.get("m2_repeats", 5)),
                min_obs=int(eval_cfg.get("min_obs", 10)),
                stability_drop_fracs=eval_cfg.get("stability_drop_fracs", (0.20, 0.30, 0.40)),
                stability_repeats=int(eval_cfg.get("stability_repeats", 5)),
                use_qc_clean_obs=True,
                return_outputs=True,
            )

            elapsed = time.perf_counter() - t0
            run_id = make_run_id(phase_prefix, branch, tag, dataset_tag, f_id, seed, out_dir)
            run_dir = out_dir / run_id

            resolved_cfg: dict[str, Any] = {
                "run_id": run_id,
                "phase": phase,
                "branch": branch,
                "tag": tag,
                "ablation": var_name,
                "model_type": model_type,
                "model_class": model.__class__.__name__,
                "model_kwargs": model_kwargs,
                "scaling": scaler_mode,
                "zero_shot": is_zs,
                "zero_shot_stations": list(zs_stations) if is_zs else [],
                "split_protocol": split_protocol,
                "baseline_mode": baseline_mode,
                "baseless_policy": baseless_policy,
                "variation_step_policy": variation_step_policy,
                "train_config": train_cfg.to_dict(),
                "eval_config": eval_cfg,
                "smoke": bool(args.smoke),
                "git_commit": git_commit,
                "dataset_path": portable_path_str(dataset_path, repo_root=REPO_ROOT),
                "dataset_sha256": dataset_sha256,
                "fold_id": int(f_id),
                "seed": seed,
                "device": str(device),
                "elapsed_sec": round(elapsed, 4),
                "n_params": int(model.count_parameters()),
                "best_epoch": int(fit_res["best_epoch"]),
                "best_val_metric_nT": round(float(fit_res["best_val_metric_nT"]), 6),
                "epochs_ran": int(fit_res["epochs_ran"]),
            }

            save_run_bundle(
                run_dir,
                model=model,
                best_epoch=int(fit_res["best_epoch"]),
                best_val_metric_nT=float(fit_res["best_val_metric_nT"]),
                resolved_config=resolved_cfg,
                baseline=est,
                scaler=scaler_obj,
                stations_manifest=build_stations_manifest(
                    meta, dataset, in_train, fold.is_zero_shot_mask
                ),
                metrics=metrics,
                history=fit_res["history"],
                outputs=outputs,
            )

            m1_eu = _extract_metric(metrics, "M1", "europe_mid", "vector_3d", "rmse_nT")
            m1_aur = _extract_metric(metrics, "M1", "auroral", "vector_3d", "rmse_nT")
            m2_gl = _extract_metric(metrics, "M2", "global", "vector_3d", "rmse_nT")
            m2_r2 = _extract_metric(metrics, "M2", "global", "vector_3d", "r2")
            print(
                f"[{branch}] {tag:32s} f{f_id} | best_ep={fit_res['best_epoch']:03d} | "
                f"M1 EU={_fmt(m1_eu)}nT AUR={_fmt(m1_aur)}nT | "
                f"M2 GL={_fmt(m2_gl)}nT (R2={_fmt(m2_r2, '.3f')}) | {elapsed:.1f}s -> {run_id}"
            )

    total_sec = time.perf_counter() - t_all_start
    print(f"\nTraining run completed in {total_sec:.1f}s.")


if __name__ == "__main__":
    main()
