"""CLI entry point to build clean, undespiked, or extended ML `.npz` datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from geomag_ml import REPO_ROOT
from geomag_ml.data.build_dataset import load_config_and_build
from geomag_ml.data.schema import BELT_ID_TO_NAME, OMNI_COLS, SOLAR1_BSN_COLS, load_ml_dataset


def _print_summary(label: str, npz_path: Path, meta_path: Path) -> None:
    arrays, meta = load_ml_dataset(npz_path, validate=True)
    size_mb = npz_path.stat().st_size / (1024.0 * 1024.0)
    n_t, n_st, _ = arrays["X_raw"].shape
    belts = arrays["station_belt_id"]
    belt_counts = {name: int(np.sum(belts == bid)) for bid, name in BELT_ID_TO_NAME.items()}

    ey_idx = SOLAR1_BSN_COLS.index("ey_mVm")
    ef_idx = OMNI_COLS.index("electric_field")
    ey_arr = arrays["sw_solar1_bsn"][:, ey_idx]
    ef_arr = arrays["sw_omni"][:, ef_idx]
    both = np.isfinite(ey_arr) & np.isfinite(ef_arr)
    corr_ey = float(np.corrcoef(ey_arr[both], ef_arr[both])[0, 1]) if int(np.sum(both)) >= 2 else float("nan")

    print(f"[{label}] {npz_path} ({size_mb:.2f} MB) | meta={meta_path.name}")
    print(
        f"  sha256={meta.get('dataset_sha256', 'n/a')} | shape=({n_t}, {n_st}, 3) | "
        f"episodes={len(meta.get('episode_ids', []))} | belts={belt_counts}"
    )
    print(
        f"  M_valid={int(arrays['M_valid'].sum())} | M_qc_clean={int(arrays['M_qc_clean'].sum())} | "
        f"M_despiked={int(arrays['M_despiked'].sum())} | M_known_spikes={int(arrays['M_known_spikes'].sum())} | "
        f"Ey_OMNI_corr={corr_ey:.4f} (n={int(both.sum())})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build self-contained GeoMag ML `.npz` dataset.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to dataset YAML config (e.g. ml_research/configs/dataset_oct2026.yaml).",
    )
    parser.add_argument(
        "--also-undespiked",
        action="store_true",
        help="Also build undespiked dataset from raw .min files (Phase 0 T0.3).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root directory (defaults to project root).",
    )
    args = parser.parse_args(argv)

    outputs = load_config_and_build(
        args.config,
        also_undespiked=args.also_undespiked,
        repo_root=args.repo_root,
    )
    if "clean_npz" in outputs:
        _print_summary("clean", outputs["clean_npz"], outputs["clean_meta"])
    if "undespiked_npz" in outputs:
        _print_summary("undespiked", outputs["undespiked_npz"], outputs["undespiked_meta"])
    if "npz" in outputs:
        _print_summary("extended", outputs["npz"], outputs["meta"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
