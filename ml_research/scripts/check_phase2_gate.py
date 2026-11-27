"""Phase 2 Gate verification script: compare Phase 1 vs Phase 2 and measure compression price (A vs B)."""

from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path
from typing import Any

import numpy as np

from geomag_ml import ML_ROOT, REPO_ROOT
from geomag_ml.utils.io import read_json, write_json

DEFAULT_MODES: tuple[str, ...] = ("M1", "M2", "M3_500km", "M3_1000km", "M4")
DEFAULT_BELTS: tuple[str, ...] = ("global", "europe_mid", "auroral", "canada")
DEFAULT_COMPS: tuple[str, ...] = ("vector_3d", "X", "Y", "Z")


def _resolve_repo_path(path_str: str | Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    cand_repo = REPO_ROOT / p
    if cand_repo.exists() or str(p).startswith("ml_research"):
        return cand_repo
    return ML_ROOT / p


def _agg_values(vals: list[float]) -> dict[str, Any] | None:
    if not vals:
        return None
    arr = np.asarray(vals, dtype=np.float64)
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {
        "mean": float(np.mean(arr)),
        "std": std,
        "n_folds": int(arr.size),
    }


def _get_recon_val(
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
    if val is None or not math.isfinite(float(val)):
        return None
    return float(val)


def _discover_runs(
    artifacts_dir: Path,
    *,
    phases: tuple[str, ...] = ("phase1", "phase2"),
) -> tuple[dict[tuple[str, str, str], dict[int, dict[str, Any]]], str | None, str | None]:
    """Scan `artifacts_dir` for complete non-smoke Phase 1/2 runs, validating shared SHA and protocol."""
    candidates: dict[tuple[str, str, str, int], tuple[str, dict[str, Any]]] = {}
    ref_sha: str | None = None
    ref_proto: str | None = None

    for run_dir in sorted(artifacts_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        cfg_path = run_dir / "config.json"
        met_path = run_dir / "metrics.json"
        if not (cfg_path.is_file() and met_path.is_file()):
            continue

        cfg = read_json(cfg_path)
        phase = str(cfg.get("phase", ""))
        if phase not in phases:
            continue
        if bool(cfg.get("smoke", False)):
            continue

        sha = cfg.get("dataset_sha256")
        proto = cfg.get("split_protocol")
        if ref_sha is None:
            ref_sha = sha
            ref_proto = proto
        else:
            if sha != ref_sha:
                raise ValueError(
                    f"Inconsistent dataset_sha256 in {run_dir.name}: {sha} != {ref_sha}"
                )
            if proto != ref_proto:
                raise ValueError(
                    f"Inconsistent split_protocol in {run_dir.name}: {proto} != {ref_proto}"
                )

        metrics = read_json(met_path)
        branch = str(cfg["branch"])
        tag = str(cfg["tag"])
        fold_id = int(cfg["fold_id"])
        key = (phase, branch, tag, fold_id)
        run_id = str(cfg.get("run_id", run_dir.name))
        entry = {"run_id": run_id, "run_dir": run_dir, "config": cfg, "metrics": metrics}
        if key not in candidates or run_id > candidates[key][0]:
            candidates[key] = (run_id, entry)

    grouped: dict[tuple[str, str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for (phase, branch, tag, fold_id), (_, entry) in candidates.items():
        grouped[(phase, branch, tag)][fold_id] = entry
    return dict(grouped), ref_sha, ref_proto


def _summarize_model_group(
    fold_map: dict[int, dict[str, Any]],
    *,
    expected_folds: int,
) -> dict[str, Any]:
    fold_ids = sorted(fold_map.keys())
    sample_cfg = fold_map[fold_ids[0]]["config"]

    eval_summary: dict[str, Any] = {}
    for mode in DEFAULT_MODES:
        mode_dict: dict[str, Any] = {}
        for belt in DEFAULT_BELTS:
            belt_dict: dict[str, Any] = {}
            for comp in DEFAULT_COMPS:
                rmses = [
                    v
                    for fid in fold_ids
                    if (v := _get_recon_val(fold_map[fid]["metrics"], mode, belt, comp, "rmse_nT"))
                    is not None
                ]
                maes = [
                    v
                    for fid in fold_ids
                    if (v := _get_recon_val(fold_map[fid]["metrics"], mode, belt, comp, "mae_nT"))
                    is not None
                ]
                r2s = [
                    v
                    for fid in fold_ids
                    if (v := _get_recon_val(fold_map[fid]["metrics"], mode, belt, comp, "r2"))
                    is not None
                ]
                belt_dict[comp] = {
                    "rmse_nT": _agg_values(rmses),
                    "mae_nT": _agg_values(maes),
                    "r2": _agg_values(r2s),
                }
            mode_dict[belt] = belt_dict
        eval_summary[mode] = mode_dict

    summary: dict[str, Any] = {
        "phase": sample_cfg["phase"],
        "branch": sample_cfg["branch"],
        "tag": sample_cfg["tag"],
        "model_class": sample_cfg.get("model_class"),
        "scaling": sample_cfg.get("scaling"),
        "baseless_policy": sample_cfg.get("baseless_policy"),
        "n_params": sample_cfg.get("n_params"),
        "folds_present": fold_ids,
        "missing_folds": [f for f in range(expected_folds) if f not in fold_map],
        "run_ids": {str(fid): fold_map[fid]["run_id"] for fid in fold_ids},
        "eval_masks": eval_summary,
    }

    if sample_cfg["branch"] == "A":
        stab_summary: dict[str, Any] = {}
        for frac_key in ("M2_20", "M2_30", "M2_40"):
            cos_vals = []
            drift_vals = []
            decorr_vals = []
            for fid in fold_ids:
                stab = fold_map[fid]["metrics"].get("latent_stability", {}).get(frac_key)
                if isinstance(stab, dict):
                    if stab.get("cos_sim_mean") is not None:
                        cos_vals.append(float(stab["cos_sim_mean"]))
                    if stab.get("rel_l2_drift") is not None:
                        drift_vals.append(float(stab["rel_l2_drift"]))
                    if stab.get("axis_decorr_frob") is not None:
                        decorr_vals.append(float(stab["axis_decorr_frob"]))
            stab_summary[frac_key] = {
                "cos_sim_mean": _agg_values(cos_vals),
                "rel_l2_drift": _agg_values(drift_vals),
                "axis_decorr_frob": _agg_values(decorr_vals),
            }
        summary["latent_stability"] = stab_summary

        sw_targets = (
            ("omni", "ae"),
            ("omni", "al"),
            ("solar1_bsn", "bz_gsm"),
            ("solar1_bsn", "ey_mVm"),
            ("solar1_bsn", "pdyn_nPa"),
            ("oulu", "delta_n_pct"),
        )
        sw_summary: dict[str, Any] = {}
        for grp, col in sw_targets:
            r2_vals = []
            best_abs_r_vals = []
            for fid in fold_ids:
                entry = (
                    fold_map[fid]["metrics"]
                    .get("space_weather_alignment", {})
                    .get(grp, {})
                    .get(col, {})
                )
                if entry.get("status") == "ok":
                    if entry.get("linear_r2") is not None:
                        r2_vals.append(float(entry["linear_r2"]))
                    if entry.get("best_axis_abs_pearson_r") is not None:
                        best_abs_r_vals.append(float(entry["best_axis_abs_pearson_r"]))
            sw_summary[f"{grp}.{col}"] = {
                "n_folds_covered": len(r2_vals),
                "linear_r2": _agg_values(r2_vals),
                "best_axis_abs_pearson_r": _agg_values(best_abs_r_vals),
            }
        summary["space_weather_alignment"] = sw_summary

    return summary


def _compare_pair_rmse(
    model_a_folds: dict[int, dict[str, Any]],
    model_b_folds: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Compute paired-fold `RMSE(model_a) - RMSE(model_b)` across modes, belts, and components."""
    common_folds = sorted(set(model_a_folds.keys()) & set(model_b_folds.keys()))
    out: dict[str, Any] = {}
    for mode in DEFAULT_MODES:
        out[mode] = {}
        for belt in DEFAULT_BELTS:
            out[mode][belt] = {}
            for comp in DEFAULT_COMPS:
                a_vals: list[float] = []
                b_vals: list[float] = []
                diffs: list[float] = []
                for fid in common_folds:
                    va = _get_recon_val(model_a_folds[fid]["metrics"], mode, belt, comp, "rmse_nT")
                    vb = _get_recon_val(model_b_folds[fid]["metrics"], mode, belt, comp, "rmse_nT")
                    if va is not None and vb is not None:
                        a_vals.append(va)
                        b_vals.append(vb)
                        diffs.append(va - vb)
                a_agg = _agg_values(a_vals)
                b_agg = _agg_values(b_vals)
                d_agg = _agg_values(diffs)
                rel_pct = None
                if a_agg is not None and b_agg is not None and b_agg["mean"] > 1e-12:
                    rel_pct = 100.0 * (b_agg["mean"] - a_agg["mean"]) / b_agg["mean"]
                out[mode][belt][comp] = {
                    "model_a_rmse_nT": a_agg,
                    "model_b_rmse_nT": b_agg,
                    "delta_a_minus_b_nT": d_agg,
                    "a_rel_improvement_over_b_pct": rel_pct,
                }
    return out


def build_phase2_gate_report(
    artifacts_dir: Path,
    *,
    expected_folds: int = 4,
) -> dict[str, Any]:
    """Aggregate Phase 1 & Phase 2 runs and compute Gate comparisons + compression price."""
    grouped, ref_sha, ref_proto = _discover_runs(artifacts_dir)
    if not grouped:
        raise FileNotFoundError(f"No Phase 1 or Phase 2 artifacts found in {artifacts_dir}.")

    summaries: dict[str, dict[str, Any]] = {}
    for (phase, branch, tag), f_map in sorted(grouped.items()):
        summaries[f"{phase}:{branch}:{tag}"] = _summarize_model_group(
            f_map, expected_folds=expected_folds
        )

    # Identify key reference models (standard baseless_policy=="train_quiet_proxy", zero_shot==False)
    def _find_group(
        phase: str,
        branch: str,
        preferred_tag: str,
        *,
        latent_dim: int | None = None,
    ) -> tuple[str, dict[int, dict[str, Any]]] | None:
        key = (phase, branch, preferred_tag)
        if key in grouped:
            return preferred_tag, grouped[key]
        for (ph, br, tg), f_map in sorted(grouped.items()):
            if ph == phase and br == branch:
                cfg0 = next(iter(f_map.values()))["config"]
                if bool(cfg0.get("zero_shot", False)) or cfg0.get("baseless_policy") == "exclude":
                    continue
                if latent_dim is not None:
                    k_val = cfg0.get("model_kwargs", {}).get(
                        "latent_dim", cfg0.get("params", {}).get("k")
                    )
                    if k_val != latent_dim:
                        continue
                return tg, f_map
        return None

    ridge_ref = _find_group("phase1", "B", "ridge_zscore")
    wide_ref = _find_group("phase2", "B", "wide_mlp_zscore")

    branch_b_vs_ridge: dict[str, Any] = {}
    if ridge_ref is not None and wide_ref is not None:
        ridge_tag, ridge_folds = ridge_ref
        wide_tag, wide_folds = wide_ref
        branch_b_vs_ridge = {
            "wide_mlp_tag": wide_tag,
            "ridge_tag": ridge_tag,
            "comparison": _compare_pair_rmse(wide_folds, ridge_folds),
        }

    branch_a_vs_pca: dict[str, Any] = {}
    compression_price: dict[str, Any] = {}
    for k in (3, 4, 5):
        mlp_k = _find_group("phase2", "A", f"mlp_ae_k{k}_ut_decorr_zscore", latent_dim=k)
        pca_k = _find_group("phase1", "A", f"empca_k{k}_zscore", latent_dim=k)
        pcasq_k = _find_group("phase1", "A", f"empca_k{k}_sq_zscore", latent_dim=k)

        if mlp_k is not None:
            k_tag, k_folds = mlp_k
            entry_k: dict[str, Any] = {"mlp_ae_tag": k_tag}
            if pca_k is not None:
                entry_k["vs_empca"] = {
                    "pca_tag": pca_k[0],
                    "comparison": _compare_pair_rmse(k_folds, pca_k[1]),
                }
            if pcasq_k is not None:
                entry_k["vs_empca_sq"] = {
                    "pca_sq_tag": pcasq_k[0],
                    "comparison": _compare_pair_rmse(k_folds, pcasq_k[1]),
                }
            branch_a_vs_pca[f"k{k}"] = entry_k

            if wide_ref is not None:
                compression_price[f"k{k}"] = {
                    "mlp_ae_tag": k_tag,
                    "wide_mlp_tag": wide_ref[0],
                    "delta_rmse_a_minus_b": _compare_pair_rmse(k_folds, wide_ref[1]),
                }

    branch_a_ablations = {
        tag: summaries[key]
        for (ph, br, tag), key in [
            ((s["phase"], s["branch"], s["tag"]), k) for k, s in summaries.items()
        ]
        if ph == "phase2" and br == "A"
    }
    branch_b_ablations = {
        tag: summaries[key]
        for (ph, br, tag), key in [
            ((s["phase"], s["branch"], s["tag"]), k) for k, s in summaries.items()
        ]
        if ph == "phase2" and br == "B"
    }

    # Gate checks
    gate_summary: dict[str, Any] = {
        "expected_folds": int(expected_folds),
        "branch_a_configs_complete": all(
            len(v["missing_folds"]) == 0 for v in branch_a_ablations.values()
        )
        and len(branch_a_ablations) > 0,
        "branch_b_configs_complete": all(
            len(v["missing_folds"]) == 0 for v in branch_b_ablations.values()
        )
        and len(branch_b_ablations) > 0,
    }
    if branch_b_vs_ridge:
        comp = branch_b_vs_ridge["comparison"]
        gate_summary["wide_mlp_beats_ridge_m2_auroral"] = (
            comp["M2"]["auroral"]["vector_3d"]["delta_a_minus_b_nT"]["mean"] < 0.0
        )
        gate_summary["wide_mlp_beats_ridge_m3_500km_auroral"] = (
            comp["M3_500km"]["auroral"]["vector_3d"]["delta_a_minus_b_nT"]["mean"] < 0.0
        )
        gate_summary["wide_mlp_beats_ridge_m3_1000km_auroral"] = (
            comp["M3_1000km"]["auroral"]["vector_3d"]["delta_a_minus_b_nT"]["mean"] < 0.0
        )
        gate_summary["wide_mlp_beats_ridge_m3_1000km_global"] = (
            comp["M3_1000km"]["global"]["vector_3d"]["delta_a_minus_b_nT"]["mean"] < 0.0
        )
        gate_summary["wide_mlp_beats_ridge_m4_auroral"] = (
            comp["M4"]["auroral"]["vector_3d"]["delta_a_minus_b_nT"]["mean"] < 0.0
        )
    if "k5" in branch_a_vs_pca and "vs_empca" in branch_a_vs_pca["k5"]:
        m2_cmp = branch_a_vs_pca["k5"]["vs_empca"]["comparison"]["M2"]["global"]["vector_3d"]
        gate_summary["mlp_ae_k5_vs_pca_k5_m2_global_delta_nT"] = m2_cmp["delta_a_minus_b_nT"]["mean"]

    return {
        "dataset_sha256": ref_sha,
        "split_protocol": ref_proto,
        "branch_b_vs_ridge": branch_b_vs_ridge,
        "branch_a_vs_pca": branch_a_vs_pca,
        "compression_price": compression_price,
        "branch_a_ablations": branch_a_ablations,
        "branch_b_ablations": branch_b_ablations,
        "gate_summary": gate_summary,
    }


def _fmt_mean_std(agg: dict[str, Any] | None) -> str:
    if agg is None:
        return "N/A"
    return f"{agg['mean']:.2f} +/- {agg['std']:.2f}"


def print_gate_summary(report: dict[str, Any]) -> None:
    """Print concise Markdown comparison tables to stdout."""
    print("\n=== Phase 2 Gate Summary ===")
    print(f"dataset_sha256: {report.get('dataset_sha256')}")
    print(f"split_protocol: {report.get('split_protocol')}")
    print(f"gate_summary:   {report.get('gate_summary')}")

    b_cmp = report.get("branch_b_vs_ridge", {})
    if b_cmp:
        print(
            f"\n[Branch B] {b_cmp['wide_mlp_tag']} vs {b_cmp['ridge_tag']} (vector_3d RMSE [nT]):"
        )
        for mode in DEFAULT_MODES:
            row_parts = []
            for belt in ("global", "europe_mid", "auroral", "canada"):
                c = b_cmp["comparison"][mode][belt]["vector_3d"]
                w_str = _fmt_mean_std(c["model_a_rmse_nT"])
                r_str = _fmt_mean_std(c["model_b_rmse_nT"])
                imp = c["a_rel_improvement_over_b_pct"]
                imp_str = f"{imp:+.1f}%" if imp is not None else "N/A"
                row_parts.append(f"{belt}={w_str} vs {r_str} ({imp_str})")
            print(f"  {mode:9s} | " + " | ".join(row_parts))

    a_cmp = report.get("branch_a_vs_pca", {})
    if a_cmp:
        print("\n[Branch A] MLP-AE vs EM-PCA (M2 RMSE [nT] & M2_30 cos_sim):")
        for k_key, info in sorted(a_cmp.items()):
            tag = info["mlp_ae_tag"]
            stab = (
                report["branch_a_ablations"][tag]
                .get("latent_stability", {})
                .get("M2_30", {})
                .get("cos_sim_mean")
            )
            stab_str = f"{stab['mean']:.4f}" if stab else "N/A"
            if "vs_empca" in info:
                c_gl = info["vs_empca"]["comparison"]["M2"]["global"]["vector_3d"]
                c_eu = info["vs_empca"]["comparison"]["M2"]["europe_mid"]["vector_3d"]
                c_au = info["vs_empca"]["comparison"]["M2"]["auroral"]["vector_3d"]
                print(
                    f"  {k_key} ({tag}) | "
                    f"GL={_fmt_mean_std(c_gl['model_a_rmse_nT'])} vs PCA={_fmt_mean_std(c_gl['model_b_rmse_nT'])} | "
                    f"EU={_fmt_mean_std(c_eu['model_a_rmse_nT'])} | AUR={_fmt_mean_std(c_au['model_a_rmse_nT'])} | "
                    f"cos_sim(M2_30)={stab_str}"
                )

    cp = report.get("compression_price", {})
    if cp:
        print("\n[Compression Price] Delta RMSE = RMSE(MLP-AE_k) - RMSE(Wide MLP) [nT] on M2:")
        for k_key, info in sorted(cp.items()):
            m2 = info["delta_rmse_a_minus_b"]["M2"]
            parts = [
                (
                    f"{belt}: {diff['mean']:+.2f} +/- {diff['std']:.2f}"
                    if (diff := m2[belt]["vector_3d"]["delta_a_minus_b_nT"]) is not None
                    else f"{belt}: N/A"
                )
                for belt in ("global", "europe_mid", "auroral", "canada")
            ]
            print(f"  {k_key} ({info['mlp_ae_tag']} - {info['wide_mlp_tag']}) | " + " | ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Phase 1 & Phase 2 runs and write phase2_gate_report.json."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ML_ROOT / "artifacts",
        help="Path to artifacts directory.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ML_ROOT / "artifacts" / "phase2_gate_report.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--expected-folds",
        type=int,
        default=4,
        help="Expected number of CV folds per model tag.",
    )
    args = parser.parse_args()

    artifacts_dir = _resolve_repo_path(args.artifacts_dir)
    out_path = _resolve_repo_path(args.out)

    report = build_phase2_gate_report(
        artifacts_dir,
        expected_folds=int(args.expected_folds),
    )
    write_json(out_path, report, nan_to_null=False)
    print_gate_summary(report)
    print(f"\nPhase 2 gate report written to {out_path}")


if __name__ == "__main__":
    main()
