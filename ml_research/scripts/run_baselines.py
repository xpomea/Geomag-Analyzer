"""CLI runner for Phase 1 linear and geostatistical baselines across 4-fold LOOE-CV."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

import numpy as np

from geomag_ml import ML_ROOT, REPO_ROOT
from geomag_ml.data.masks import MaskGenerator
from geomag_ml.data.preprocessing import BaselineEstimator, FieldScaler, save_scaler_json
from geomag_ml.data.schema import BELT_ID_TO_NAME, load_ml_dataset
from geomag_ml.data.splits import DEFAULT_ZERO_SHOT_STATIONS, create_splits
from geomag_ml.eval.evaluator import evaluate_model_on_fold
from geomag_ml.models.baselines import (
    EMPCABaseline,
    IDWBaseline,
    OrdinaryKrigingBaseline,
    RidgeSchurBaseline,
)
from geomag_ml.utils.device import resolve_device
from geomag_ml.utils.io import (
    compute_sha256,
    get_git_commit,
    portable_path_str,
    read_yaml,
    resolve_project_path,
    save_npz_atomic,
    write_json,
)
from geomag_ml.utils.seed import set_seed


def _build_stations_manifest(
    meta: dict[str, Any],
    dataset: dict[str, np.ndarray],
    in_train: np.ndarray,
    is_zero_shot: np.ndarray,
) -> list[dict[str, Any]]:
    codes = list(meta["station_codes"])
    regions = list(meta["station_regions"])
    coords = dataset["station_coords"]
    has_base = dataset["station_has_baseline"]
    belts = dataset["station_belt_id"]

    rows: list[dict[str, Any]] = []
    for i, code in enumerate(codes):
        rows.append(
            {
                "idx": int(i),
                "iaga_code": str(code),
                "region": str(regions[i]),
                "belt": BELT_ID_TO_NAME[int(belts[i])],
                "lat_deg": float(coords[i, 0]),
                "lon_deg": float(coords[i, 1]),
                "mlat_deg": float(coords[i, 2]),
                "mlon_deg": float(coords[i, 3]),
                "has_baseline": bool(has_base[i]),
                "in_train": bool(in_train[i]),
                "is_zero_shot": bool(is_zero_shot[i]),
            }
        )
    return rows


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


def _extract_zs_metric(
    metrics: dict[str, Any],
    sub_key: str,
    field: str,
) -> float | None:
    zs = metrics.get("eval_masks", {}).get("zero_shot")
    if not isinstance(zs, dict) or sub_key not in zs:
        return None
    val = zs[sub_key].get("global", {}).get("vector_3d", {}).get(field)
    return float(val) if val is not None else None


def _instantiate_and_fit(
    exp_cfg: dict[str, Any],
    x_work: np.ndarray,
    m_qc: np.ndarray,
    fold: Any,
    in_train: np.ndarray,
    dataset: dict[str, np.ndarray],
    mask_gen: MaskGenerator,
    scaler: FieldScaler | None,
) -> Any:
    mtype = str(exp_cfg["model_type"]).strip().lower()
    params = dict(exp_cfg.get("params", {}))

    if mtype == "empca":
        model = EMPCABaseline(**params)
        model.fit(
            x_work,
            m_qc,
            fold.train_idx,
            in_train,
            time_features=dataset["time_features"],
            timestamps_unix_s=dataset["timestamps_unix_s"],
        )
        return model
    if mtype == "ridge_schur":
        model = RidgeSchurBaseline(**params)
        model.fit(
            x_work,
            m_qc,
            fold.train_idx,
            in_train,
            val_idx=fold.val_idx,
            mask_generator=mask_gen,
            scaler=scaler,
        )
        return model
    if mtype == "idw":
        model = IDWBaseline(**params)
        model.fit(
            x_work,
            m_qc,
            fold.train_idx,
            in_train,
            station_coords=dataset["station_coords"],
            station_dist_km=dataset["station_dist_km"],
            val_idx=fold.val_idx,
            mask_generator=mask_gen,
            scaler=scaler,
        )
        return model
    if mtype == "ordinary_kriging":
        model = OrdinaryKrigingBaseline(**params)
        model.fit(
            x_work,
            m_qc,
            fold.train_idx,
            in_train,
            station_coords=dataset["station_coords"],
            scaler=scaler,
        )
        return model
    raise ValueError(f"Unsupported Phase 1 model_type {mtype!r}.")


def _write_phase1_history_csv(
    path: Path,
    metrics: dict[str, Any],
    elapsed_sec: float,
) -> None:
    m1_global = _extract_metric(metrics, "M1", "global", "vector_3d", "rmse_nT")
    m1_eu = _extract_metric(metrics, "M1", "europe_mid", "vector_3d", "rmse_nT")
    m1_aur = _extract_metric(metrics, "M1", "auroral", "vector_3d", "rmse_nT")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "val_rmse_global_nT",
                "val_rmse_eu_mid_nT",
                "val_rmse_auroral_nT",
                "lr",
                "epoch_sec",
            ]
        )
        writer.writerow(
            [
                0,
                "",
                "",
                f"{m1_global:.6f}" if m1_global is not None else "",
                f"{m1_eu:.6f}" if m1_eu is not None else "",
                f"{m1_aur:.6f}" if m1_aur is not None else "",
                "0.0",
                f"{elapsed_sec:.4f}",
            ]
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 linear & geostatistical baselines.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ML_ROOT / "configs" / "phase1_baselines.yaml",
        help="Path to phase1_baselines.yaml",
    )
    parser.add_argument("--dataset", type=Path, default=None, help="Optional .npz dataset override.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Optional artifacts output directory.")
    parser.add_argument("--fold", type=str, default="all", help="Fold index 0..3 or 'all'.")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Optional comma-separated tag substring filter.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    args = parser.parse_args()

    cfg = read_yaml(resolve_project_path(args.config))
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    set_seed(seed)
    resolve_device("cpu")

    dataset_path = resolve_project_path(args.dataset or cfg["dataset_path"])
    out_dir = resolve_project_path(args.out_dir or cfg.get("out_dir", ML_ROOT / "artifacts"))
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset, meta = load_ml_dataset(dataset_path, validate=True)
    dataset_sha256 = meta.get("dataset_sha256") or compute_sha256(dataset_path)
    dataset_tag = str(meta.get("tag", dataset_path.stem)).split("_")[0]
    if "oct2026" in str(meta.get("tag", "")):
        dataset_tag = "oct2026"
    git_commit = get_git_commit(REPO_ROOT)

    split_protocol = str(cfg.get("split_protocol", "looe_4fold"))
    baseline_mode = str(cfg.get("baseline_mode", "temp_train_quiet_proxy"))
    baseless_policy = str(cfg.get("baseless_policy", "train_quiet_proxy"))
    variation_step_policy = str(cfg.get("variation_step_policy", "flag_qc_exclude"))
    zs_stations = tuple(cfg.get("default_zero_shot_stations", list(DEFAULT_ZERO_SHOT_STATIONS)))
    eval_cfg = dict(cfg.get("eval", {}))

    folds_std = create_splits(
        split_protocol,
        dataset,
        meta,
        zero_shot_stations=None,
        baseless_policy=baseless_policy,
    )
    folds_zs = create_splits(
        split_protocol,
        dataset,
        meta,
        zero_shot_stations=zs_stations,
        baseless_policy=baseless_policy,
    )

    if str(args.fold).strip().lower() == "all":
        selected_folds = list(range(len(folds_std)))
    else:
        selected_folds = [int(args.fold)]

    experiments = list(cfg.get("experiments", []))
    if args.models:
        needles = [s.strip().lower() for s in str(args.models).split(",") if s.strip()]
        experiments = [e for e in experiments if any(n in str(e["tag"]).lower() for n in needles)]

    summary_rows: list[dict[str, Any]] = []
    t_total_start = time.perf_counter()

    # Cache fitted baseline & scalers per (is_zero_shot, fold_id, scaling) to avoid redundant work
    prep_cache: dict[tuple[bool, int, str], tuple[BaselineEstimator, FieldScaler | None, np.ndarray]] = {}

    for exp in experiments:
        tag = str(exp["tag"])
        branch = str(exp["branch"])
        scaling = str(exp["scaling"])
        is_zs = bool(exp.get("zero_shot", False))

        for f_id in selected_folds:
            fold = folds_zs[f_id] if is_zs else folds_std[f_id]
            t0 = time.perf_counter()

            cache_key = (is_zs, f_id, scaling)
            if cache_key not in prep_cache:
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
                if scaling == "none":
                    scaler_obj = None
                    x_work = x_delta
                else:
                    scaler_obj = FieldScaler(mode=scaling).fit(
                        x_delta,
                        dataset["M_qc_clean"],
                        fold.train_idx,
                        dataset["station_belt_id"],
                        station_train_mask=fold.station_train_mask,
                    )
                    x_work = scaler_obj.transform(x_delta)
                prep_cache[cache_key] = (est, scaler_obj, x_work)
            else:
                est, scaler_obj, x_work = prep_cache[cache_key]

            in_train = fold.in_train_mask & est.in_train_stations_
            if scaler_obj is not None and scaler_obj.in_train_stations_ is not None:
                in_train = in_train & scaler_obj.in_train_stations_

            mask_gen = MaskGenerator(
                dataset["station_coords"],
                dataset["station_dist_km"],
                seed=seed + f_id * 100,
            )

            model = _instantiate_and_fit(
                exp, x_work, dataset["M_qc_clean"], fold, in_train, dataset, mask_gen, scaler_obj
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
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            run_id = f"p1_{branch}_{tag}_{dataset_tag}_f{f_id}_s{seed}_{stamp}"
            run_dir = out_dir / run_id
            suffix = 1
            while run_dir.exists():
                run_id = f"p1_{branch}_{tag}_{dataset_tag}_f{f_id}_s{seed}_{stamp}_{suffix}"
                run_dir = out_dir / run_id
                suffix += 1
            run_dir.mkdir(parents=True, exist_ok=True)

            model.save_npz(run_dir / "model.npz")
            save_scaler_json(run_dir / "scaler.json", est, scaler_obj)
            write_json(
                run_dir / "stations.json",
                _build_stations_manifest(meta, dataset, in_train, fold.is_zero_shot_mask),
            )
            write_json(run_dir / "metrics.json", metrics, nan_to_null=False)
            write_json(run_dir / "onnx_export_spec.json", model.onnx_export_spec())
            _write_phase1_history_csv(run_dir / "history.csv", metrics, elapsed)
            if outputs:
                save_npz_atomic(run_dir / "val_outputs.npz", outputs, compressed=True)

            resolved_cfg = {
                "run_id": run_id,
                "phase": "phase1",
                "branch": branch,
                "tag": tag,
                "model_type": exp["model_type"],
                "model_class": model.__class__.__name__,
                "scaling": scaling,
                "zero_shot": is_zs,
                "zero_shot_stations": list(zs_stations) if is_zs else [],
                "params": exp.get("params", {}),
                "split_protocol": split_protocol,
                "baseline_mode": baseline_mode,
                "baseless_policy": baseless_policy,
                "variation_step_policy": variation_step_policy,
                "git_commit": git_commit,
                "dataset_path": portable_path_str(dataset_path, repo_root=REPO_ROOT),
                "dataset_sha256": dataset_sha256,
                "fold_id": int(f_id),
                "seed": seed,
                "device": "cpu",
                "elapsed_sec": round(elapsed, 4),
                "n_params": int(model.count_parameters()),
            }
            if isinstance(model, RidgeSchurBaseline):
                resolved_cfg["selected_alpha"] = float(model.best_alpha_)
            elif isinstance(model, IDWBaseline):
                resolved_cfg["selected_idw"] = {
                    "power": float(model.best_power_),
                    "k": model.best_k_,
                    "anisotropy": float(model.best_anisotropy_),
                }
            write_json(run_dir / "config.json", resolved_cfg)

            sw = metrics.get("space_weather_alignment", {})
            omni_ae = sw.get("omni", {}).get("ae", {})
            sol_ey = sw.get("solar1_bsn", {}).get("ey_mVm", {})
            stab30 = metrics.get("latent_stability", {}).get("M2_30", {})

            row = {
                "run_id": run_id,
                "branch": branch,
                "tag": tag,
                "model_type": exp["model_type"],
                "scaling": scaling,
                "is_zero_shot_run": is_zs,
                "fold_id": f_id,
                "m1_global_rmse_nT": _extract_metric(metrics, "M1", "global", "vector_3d", "rmse_nT"),
                "m1_global_r2": _extract_metric(metrics, "M1", "global", "vector_3d", "r2"),
                "m1_eu_mid_rmse_nT": _extract_metric(metrics, "M1", "europe_mid", "vector_3d", "rmse_nT"),
                "m1_eu_mid_r2": _extract_metric(metrics, "M1", "europe_mid", "vector_3d", "r2"),
                "m1_auroral_rmse_nT": _extract_metric(metrics, "M1", "auroral", "vector_3d", "rmse_nT"),
                "m1_auroral_r2": _extract_metric(metrics, "M1", "auroral", "vector_3d", "r2"),
                "m2_global_rmse_nT": _extract_metric(metrics, "M2", "global", "vector_3d", "rmse_nT"),
                "m2_global_r2": _extract_metric(metrics, "M2", "global", "vector_3d", "r2"),
                "m2_eu_mid_rmse_nT": _extract_metric(metrics, "M2", "europe_mid", "vector_3d", "rmse_nT"),
                "m2_eu_mid_r2": _extract_metric(metrics, "M2", "europe_mid", "vector_3d", "r2"),
                "m2_auroral_rmse_nT": _extract_metric(metrics, "M2", "auroral", "vector_3d", "rmse_nT"),
                "m2_auroral_r2": _extract_metric(metrics, "M2", "auroral", "vector_3d", "r2"),
                "m3_500_global_rmse_nT": _extract_metric(metrics, "M3_500km", "global", "vector_3d", "rmse_nT"),
                "m3_1000_global_rmse_nT": _extract_metric(metrics, "M3_1000km", "global", "vector_3d", "rmse_nT"),
                "m4_auroral_rmse_nT": _extract_metric(metrics, "M4", "auroral", "vector_3d", "rmse_nT"),
                "zs_heldout_rmse_nT": _extract_zs_metric(metrics, "heldout_zero_shot", "rmse_nT"),
                "zs_heldout_r2": _extract_zs_metric(metrics, "heldout_zero_shot", "r2"),
                "sua_unseen_rmse_nT": _extract_zs_metric(metrics, "single_episode_unseen", "rmse_nT"),
                "m2_30_cos_sim": stab30.get("cos_sim_mean"),
                "omni_ae_r2": omni_ae.get("linear_r2") if omni_ae.get("status") == "ok" else None,
                "solar1_ey_r2": sol_ey.get("linear_r2") if sol_ey.get("status") == "ok" else None,
                "elapsed_sec": round(elapsed, 3),
            }
            summary_rows.append(row)
            def _fmt(v: float | None, fmt: str = ".2f") -> str:
                return "None" if v is None else format(v, fmt)

            print(
                f"[{branch}] {tag:22s} f{f_id} | "
                f"M1 EU={_fmt(row['m1_eu_mid_rmse_nT'])}nT (R2={_fmt(row['m1_eu_mid_r2'], '.3f')}) "
                f"AUR={_fmt(row['m1_auroral_rmse_nT'])}nT (R2={_fmt(row['m1_auroral_r2'], '.3f')}) | "
                f"M2 GL={_fmt(row['m2_global_rmse_nT'])}nT | {elapsed:.2f}s"
            )

    summary_path = out_dir / "phase1_summary.csv"
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with summary_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    total_sec = time.perf_counter() - t_total_start
    print(f"\nPhase 1 complete: {len(summary_rows)} runs written to {summary_path} in {total_sec:.2f}s")


if __name__ == "__main__":
    main()
