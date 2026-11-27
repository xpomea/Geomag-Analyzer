"""Self-contained §5.4 run artifact bundle serialization, checkpointing, and loading."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from geomag_ml.data.preprocessing import (
    BaselineEstimator,
    FieldScaler,
    load_scaler_json,
    save_scaler_json,
)
from geomag_ml.data.schema import BELT_ID_TO_NAME
from geomag_ml.models.base import BaseGeomagModel
from geomag_ml.models.baselines import (
    EMPCABaseline,
    IDWBaseline,
    OrdinaryKrigingBaseline,
    RidgeSchurBaseline,
)
from geomag_ml.models.mlp_ae import MaskedConditionalMLPAE
from geomag_ml.models.mlp_vm import MaskedWideMLP
from geomag_ml.utils.io import read_json, save_npz_atomic, write_json

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "MaskedConditionalMLPAE": MaskedConditionalMLPAE,
    "mlp_ae": MaskedConditionalMLPAE,
    "MaskedWideMLP": MaskedWideMLP,
    "mlp_vm": MaskedWideMLP,
}

BASELINE_MODEL_REGISTRY: dict[str, type[BaseGeomagModel]] = {
    "EMPCABaseline": EMPCABaseline,
    "empca": EMPCABaseline,
    "RidgeSchurBaseline": RidgeSchurBaseline,
    "ridge_schur": RidgeSchurBaseline,
    "IDWBaseline": IDWBaseline,
    "idw": IDWBaseline,
    "OrdinaryKrigingBaseline": OrdinaryKrigingBaseline,
    "ordinary_kriging": OrdinaryKrigingBaseline,
}

HISTORY_COLUMNS: tuple[str, ...] = (
    "epoch",
    "train_loss",
    "val_loss",
    "val_rmse_global_nT",
    "val_rmse_eu_mid_nT",
    "val_rmse_auroral_nT",
    "lr",
    "epoch_sec",
)


def build_model(model_type_or_class: str, model_kwargs: dict[str, Any]) -> nn.Module:
    """Instantiate a registered neural model from its class name or config type."""
    key = str(model_type_or_class)
    if key not in MODEL_REGISTRY:
        low = key.strip().lower()
        if low not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model {model_type_or_class!r}; available={sorted(MODEL_REGISTRY)}."
            )
        cls = MODEL_REGISTRY[low]
    else:
        cls = MODEL_REGISTRY[key]
    return cls(**dict(model_kwargs))


def make_run_id(
    phase_prefix: str,
    branch: str,
    tag: str,
    dataset_tag: str,
    fold_id: int,
    seed: int,
    out_dir: Path,
) -> str:
    """Build collision-safe `<phase>_<branch>_<tag>_<dataset_tag>_f<fold>_s<seed>_<YYYYMMDD_HHMMSS>` ID."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_id = f"{phase_prefix}_{branch}_{tag}_{dataset_tag}_f{int(fold_id)}_s{int(seed)}_{stamp}"
    run_id = base_id
    suffix = 1
    while (Path(out_dir) / run_id).exists():
        run_id = f"{base_id}_{suffix}"
        suffix += 1
    return run_id


def build_stations_manifest(
    meta: dict[str, Any],
    dataset: dict[str, np.ndarray],
    in_train: np.ndarray,
    is_zero_shot: np.ndarray,
) -> list[dict[str, Any]]:
    """Construct the ordered N-station manifest adhering to §5.4."""
    n_st = int(dataset["station_coords"].shape[0])
    codes = list(meta.get("station_codes", [f"ST{i}" for i in range(n_st)]))
    regions = list(meta.get("station_regions", ["Europe" for _ in range(n_st)]))
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


def save_model_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    epoch: int,
    best_val_metric_nT: float,
) -> Path:
    """Atomically save `model.pt` with CPU state_dict and JSON-primitive `model_kwargs`."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    model_kwargs = getattr(model, "model_kwargs", None)
    if model_kwargs is None:
        raise ValueError(f"{model.__class__.__name__} must define `.model_kwargs` for checkpointing.")

    payload = {
        "model_class": model.__class__.__name__,
        "model_kwargs": dict(model_kwargs),
        "state_dict": cpu_state,
        "epoch": int(epoch),
        "best_val_metric_nT": float(best_val_metric_nT),
    }
    tmp_path = target.parent / f".{target.name}.{os.getpid()}.tmp"
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return target


def load_model_checkpoint(
    path: Path,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Restore `(model, checkpoint_meta)` cleanly using `weights_only=True`."""
    target = Path(path)
    dev = torch.device(device)
    ckpt = torch.load(target, map_location=dev, weights_only=True)
    model_class = str(ckpt["model_class"])
    model_kwargs = dict(ckpt["model_kwargs"])
    model = build_model(model_class, model_kwargs)
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev)
    model.eval()
    meta = {
        "model_class": model_class,
        "model_kwargs": model_kwargs,
        "epoch": int(ckpt["epoch"]),
        "best_val_metric_nT": float(ckpt["best_val_metric_nT"]),
    }
    return model, meta


def write_history_csv(path: Path, history: list[dict[str, Any]]) -> Path:
    """Atomically write UTF-8 `history.csv` following §5.4 column schema."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.parent / f".{target.name}.{os.getpid()}.tmp"
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(HISTORY_COLUMNS))
            writer.writeheader()
            for row in history:
                formatted: dict[str, Any] = {}
                for col in HISTORY_COLUMNS:
                    val = row.get(col)
                    if val is None:
                        formatted[col] = ""
                    elif col == "epoch":
                        formatted[col] = int(val)
                    elif col == "lr":
                        formatted[col] = f"{float(val):.8e}"
                    else:
                        formatted[col] = f"{float(val):.6f}"
                writer.writerow(formatted)
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return target


def save_run_bundle(
    run_dir: Path,
    *,
    model: nn.Module,
    best_epoch: int,
    best_val_metric_nT: float,
    resolved_config: dict[str, Any],
    baseline: BaselineEstimator,
    scaler: FieldScaler | None,
    stations_manifest: list[dict[str, Any]],
    metrics: dict[str, Any],
    history: list[dict[str, Any]],
    outputs: dict[str, np.ndarray] | None = None,
) -> Path:
    """Persist complete §5.4 artifact bundle into `run_dir`."""
    target = Path(run_dir)
    target.mkdir(parents=True, exist_ok=True)

    save_model_checkpoint(
        target / "model.pt",
        model,
        epoch=best_epoch,
        best_val_metric_nT=best_val_metric_nT,
    )
    write_json(target / "config.json", resolved_config)
    save_scaler_json(target / "scaler.json", baseline, scaler)
    write_json(target / "stations.json", stations_manifest)
    write_json(target / "metrics.json", metrics, nan_to_null=False)

    onnx_spec = (
        model.onnx_export_spec()
        if hasattr(model, "onnx_export_spec")
        else {
            "opset": 17,
            "supported": False,
            "input_names": [],
            "output_names": [],
            "dynamic_axes": {},
            "sample_shapes": {},
            "verified_max_abs_diff": None,
        }
    )
    write_json(target / "onnx_export_spec.json", onnx_spec)
    write_history_csv(target / "history.csv", history)
    if outputs:
        save_npz_atomic(target / "val_outputs.npz", outputs, compressed=True)
    return target


def load_baseline_artifact(
    path: Path,
    model_type_or_class: str,
) -> BaseGeomagModel:
    """Restore a Phase 1 baseline model from `model.npz`."""
    key = str(model_type_or_class).strip()
    cls = BASELINE_MODEL_REGISTRY.get(key) or BASELINE_MODEL_REGISTRY.get(key.lower())
    if cls is None:
        raise ValueError(
            f"Unknown baseline model {model_type_or_class!r}; available={sorted(BASELINE_MODEL_REGISTRY)}."
        )
    return cls.load_npz(Path(path))


def load_run_bundle(
    run_dir: Path,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load Phase 1 (`model.npz`) or Phase 2–4 (`model.pt`) run bundle."""
    target = Path(run_dir)
    if not target.is_dir():
        raise FileNotFoundError(f"Artifact run directory does not exist: {target}")

    config = read_json(target / "config.json")
    model_pt = target / "model.pt"
    model_npz = target / "model.npz"
    if model_pt.is_file():
        model: Any
        model, ckpt_meta = load_model_checkpoint(model_pt, device=device)
    elif model_npz.is_file():
        model_id = str(config.get("model_class") or config.get("model_type") or "")
        model = load_baseline_artifact(model_npz, model_id)
        ckpt_meta = {
            "model_class": model.__class__.__name__,
            "model_kwargs": dict(config.get("params", {})),
            "epoch": 0,
            "best_val_metric_nT": None,
        }
    else:
        raise FileNotFoundError(f"Neither model.pt nor model.npz found in {target}")

    baseline, scaler = load_scaler_json(target / "scaler.json")
    stations = read_json(target / "stations.json")
    metrics_path = target / "metrics.json"
    metrics = read_json(metrics_path) if metrics_path.is_file() else None
    onnx_path = target / "onnx_export_spec.json"
    onnx_spec = read_json(onnx_path) if onnx_path.is_file() else None

    return {
        "model": model,
        "checkpoint_meta": ckpt_meta,
        "config": config,
        "baseline": baseline,
        "scaler": scaler,
        "stations": stations,
        "metrics": metrics,
        "onnx_export_spec": onnx_spec,
    }
