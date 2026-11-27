"""Re-evaluate any saved artifact bundle `artifacts/<run_id>/` and verify metric parity."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from geomag_ml.data.masks import MaskGenerator
from geomag_ml.data.schema import load_ml_dataset
from geomag_ml.data.splits import create_splits
from geomag_ml.eval.evaluator import evaluate_model_on_fold
from geomag_ml.train.artifacts import load_run_bundle
from geomag_ml.utils.device import resolve_device
from geomag_ml.utils.io import compute_sha256, resolve_project_path, write_json


def _safe_vec_rmse(metrics_obj: dict[str, Any], mode: str, belt: str) -> float | None:
    bucket = metrics_obj.get("eval_masks", {}).get(mode)
    if not isinstance(bucket, dict) or "status" in bucket:
        return None
    val = bucket.get(belt, {}).get("vector_3d", {}).get("rmse_nT")
    return float(val) if val is not None else None


def _fmt_rmse(val: float | None) -> str:
    return "N/A" if val is None else f"{val:.3f} nT"


def _collect_metric_diffs(
    ref_obj: Any,
    cur_obj: Any,
    prefix: str = "",
) -> list[tuple[str, float, float, float]]:
    diffs: list[tuple[str, float, float, float]] = []
    if isinstance(ref_obj, dict) and isinstance(cur_obj, dict):
        for key in ref_obj:
            if key not in cur_obj:
                continue
            sub_prefix = f"{prefix}.{key}" if prefix else str(key)
            if key in ("rmse_nT", "mae_nT"):
                r_val = ref_obj[key]
                c_val = cur_obj[key]
                if r_val is None and c_val is None:
                    continue
                if r_val is None or c_val is None:
                    raise AssertionError(f"Metric null mismatch at {sub_prefix}: saved={r_val}, recomputed={c_val}")
                diff = abs(float(r_val) - float(c_val))
                diffs.append((sub_prefix, float(r_val), float(c_val), diff))
            else:
                diffs.extend(_collect_metric_diffs(ref_obj[key], cur_obj[key], sub_prefix))
    return diffs


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate saved run artifact and check parity.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to artifacts/<run_id>.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional .npz dataset override (defaults to config.dataset_path).",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Compute device: 'cpu', 'cuda', or 'auto'.")
    parser.add_argument("--eval-split", type=str, default="test", help="Split to evaluate: 'test' or 'val'.")
    parser.add_argument(
        "--tol-nT",
        type=float,
        default=1e-3,
        help="Maximum allowed |saved - recomputed| difference in nT when comparing test metrics.",
    )
    parser.add_argument(
        "--allow-sha-mismatch",
        action="store_true",
        help="Allow evaluating against a dataset whose SHA-256 differs from config.json.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional output path to save recomputed metrics JSON.",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    run_dir = resolve_project_path(args.run_dir)
    bundle = load_run_bundle(run_dir, device=device)

    cfg = bundle["config"]
    dataset_path = resolve_project_path(args.dataset or cfg["dataset_path"])
    dataset, meta = load_ml_dataset(dataset_path, validate=True)
    actual_sha = meta.get("dataset_sha256") or compute_sha256(dataset_path)
    expected_sha = cfg.get("dataset_sha256")
    if expected_sha and actual_sha != expected_sha and not args.allow_sha_mismatch:
        raise ValueError(
            f"dataset_sha256 mismatch: artifact recorded {expected_sha}, got {actual_sha}."
        )

    split_protocol = str(cfg.get("split_protocol", "looe_4fold"))
    baseless_policy = str(cfg.get("baseless_policy", "train_quiet_proxy"))
    zs_stations = cfg.get("zero_shot_stations") or None
    fold_id = int(cfg["fold_id"])
    seed = int(cfg.get("seed", 42))
    eval_cfg = dict(cfg.get("eval_config", {}))

    folds = create_splits(
        split_protocol,
        dataset,
        meta,
        zero_shot_stations=zs_stations,
        baseless_policy=baseless_policy,
    )
    fold = folds[fold_id]
    mask_gen = MaskGenerator(
        dataset["station_coords"],
        dataset["station_dist_km"],
        seed=seed + fold_id * 100,
    )

    recomputed, _ = evaluate_model_on_fold(
        bundle["model"],
        dataset,
        meta,
        fold,
        bundle["baseline"],
        bundle["scaler"],
        mask_gen,
        eval_split=args.eval_split,
        eval_modes=eval_cfg.get("modes", ("M1", "M2", "M3_500km", "M3_1000km", "M4")),
        m2_drop_frac=float(eval_cfg.get("m2_drop_frac", 0.25)),
        m2_repeats=int(eval_cfg.get("m2_repeats", 5)),
        min_obs=int(eval_cfg.get("min_obs", 10)),
        stability_drop_fracs=eval_cfg.get("stability_drop_fracs", (0.20, 0.30, 0.40)),
        stability_repeats=int(eval_cfg.get("stability_repeats", 5)),
        use_qc_clean_obs=True,
        return_outputs=False,
    )

    if args.out_json is not None:
        write_json(resolve_project_path(args.out_json), recomputed, nan_to_null=False)

    m1_eu = _safe_vec_rmse(recomputed, "M1", "europe_mid")
    m1_aur = _safe_vec_rmse(recomputed, "M1", "auroral")
    m2_gl = _safe_vec_rmse(recomputed, "M2", "global")
    print(
        f"Evaluated {run_dir.name} ({args.eval_split}) on {device}: "
        f"M1 EU={_fmt_rmse(m1_eu)}, M1 AUR={_fmt_rmse(m1_aur)}, M2 GL={_fmt_rmse(m2_gl)}"
    )

    saved_metrics = bundle.get("metrics")
    if args.eval_split.strip().lower() == "test" and isinstance(saved_metrics, dict):
        diffs = _collect_metric_diffs(
            saved_metrics.get("eval_masks", {}),
            recomputed.get("eval_masks", {}),
        )
        max_diff = max((d[3] for d in diffs), default=0.0)
        print(f"Parity check against saved metrics.json: {len(diffs)} entries, max_abs_diff_nT={max_diff:.6e}")
        if max_diff >= float(args.tol_nT):
            worst = max(diffs, key=lambda d: d[3])
            raise AssertionError(
                f"Recomputed metrics deviated by {max_diff:.6e} nT (>= tol {args.tol_nT}) at {worst[0]}."
            )


if __name__ == "__main__":
    main()
