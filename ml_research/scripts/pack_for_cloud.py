"""Build a compact self-contained cloud transfer bundle (.zip) for Colab / Kaggle T4."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Sequence
import zipfile

from geomag_ml import ML_ROOT, REPO_ROOT
from geomag_ml.utils.io import compute_sha256, get_git_commit, resolve_project_path

_EXCLUDE_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".venv",
        ".venv_ml",
        "artifacts",
        "data_external",
        "data_cache",
        "__pycache__",
        ".pytest_cache",
        ".git",
        ".mypy_cache",
        ".ruff_cache",
    }
)
_EXCLUDE_SUFFIXES: frozenset[str] = frozenset({".pt", ".onnx", ".zip", ".pyc", ".pyo", ".tmp"})


def _is_excluded_rel(rel_parts: tuple[str, ...], file_name: str) -> bool:
    for part in rel_parts:
        if part in _EXCLUDE_DIR_NAMES or part.endswith(".egg-info"):
            return True
    low_name = file_name.lower()
    return any(low_name.endswith(sfx) for sfx in _EXCLUDE_SUFFIXES)


def _iter_tree_files(root_dir: Path, base_dir: Path) -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    if not root_dir.is_dir():
        return items
    for p in sorted(root_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(base_dir)
        if _is_excluded_rel(rel.parts[:-1], p.name):
            continue
        items.append((p, rel.as_posix()))
    return items


def build_cloud_bundle(
    dataset_path: Path,
    out_zip: Path,
    *,
    ml_root: Path = ML_ROOT,
    include_tests: bool = True,
    extra_datasets: Sequence[Path] = (),
    max_size_mb: float = 25.0,
) -> dict[str, Any]:
    """Package isolated `ml_research` code, configs, scripts, and `.npz` dataset into one `.zip`."""
    ml_r = Path(ml_root).resolve()
    ds_path = Path(dataset_path).resolve()
    if not ds_path.is_file():
        raise FileNotFoundError(f"Dataset .npz file does not exist: {ds_path}")

    out_path = Path(out_zip).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files_to_pack: list[tuple[Path, str]] = []
    for subdir in ("geomag_ml", "configs", "scripts"):
        files_to_pack.extend(_iter_tree_files(ml_r / subdir, ml_r))
    if include_tests:
        files_to_pack.extend(_iter_tree_files(ml_r / "tests", ml_r))

    for root_file in ("pyproject.toml", "requirements_colab_gpu.txt", "pytest.ini", "README.md"):
        cand = ml_r / root_file
        if cand.is_file():
            files_to_pack.append((cand, root_file))

    all_datasets = [ds_path, *[Path(p).resolve() for p in extra_datasets]]
    seen_arc_names: set[str] = {arc for _, arc in files_to_pack}
    for d_path in all_datasets:
        if not d_path.is_file():
            raise FileNotFoundError(f"Extra dataset file not found: {d_path}")
        arc_npz = f"data_cache/{d_path.name}"
        if arc_npz not in seen_arc_names:
            files_to_pack.append((d_path, arc_npz))
            seen_arc_names.add(arc_npz)
        meta_cand = d_path.with_suffix(".meta.json")
        if meta_cand.is_file():
            arc_meta = f"data_cache/{meta_cand.name}"
            if arc_meta not in seen_arc_names:
                files_to_pack.append((meta_cand, arc_meta))
                seen_arc_names.add(arc_meta)

    dataset_sha256 = compute_sha256(ds_path)
    manifest: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_filename": ds_path.name,
        "dataset_relpath": f"data_cache/{ds_path.name}",
        "dataset_sha256": dataset_sha256,
        "include_tests": bool(include_tests),
        "git_commit": get_git_commit(REPO_ROOT),
        "file_count": len(files_to_pack) + 1,
    }

    tmp_zip = out_path.parent / f".{out_path.stem}.{os.getpid()}.tmp.zip"
    try:
        with zipfile.ZipFile(tmp_zip, mode="w", allowZip64=True) as zf:
            for src_path, arc_name in files_to_pack:
                compress = (
                    zipfile.ZIP_STORED
                    if src_path.suffix.lower() == ".npz"
                    else zipfile.ZIP_DEFLATED
                )
                zf.write(src_path, arcname=arc_name, compress_type=compress)
            manifest_bytes = (
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
            ).encode("utf-8")
            zf.writestr("bundle_manifest.json", manifest_bytes, compress_type=zipfile.ZIP_DEFLATED)
        os.replace(tmp_zip, out_path)
    except Exception:
        if tmp_zip.exists():
            tmp_zip.unlink(missing_ok=True)
        raise

    size_bytes = int(out_path.stat().st_size)
    size_mb = size_bytes / (1024.0 * 1024.0)
    if float(max_size_mb) > 0.0 and size_mb >= float(max_size_mb):
        raise RuntimeError(
            f"Cloud bundle size {size_mb:.2f} MB exceeds configured limit {max_size_mb:.2f} MB."
        )

    manifest["out_zip"] = out_path.as_posix()
    manifest["size_bytes"] = size_bytes
    manifest["size_mb"] = round(size_mb, 3)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack compact ml_research code + .npz dataset bundle for Colab / Kaggle T4."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ML_ROOT / "data_cache" / "geomag_ml_oct2026_clean_v1.npz",
        help="Primary .npz dataset to include in the cloud bundle.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ML_ROOT / "data_cache" / "ml_research_cloud_bundle.zip",
        help="Output .zip path.",
    )
    parser.add_argument(
        "--extra-dataset",
        type=Path,
        action="append",
        default=[],
        help="Optional extra .npz dataset path to include.",
    )
    parser.add_argument(
        "--max-size-mb",
        type=float,
        default=25.0,
        help="Maximum allowed bundle size in MiB (<= 0 disables limit).",
    )
    parser.add_argument(
        "--no-tests",
        action="store_true",
        help="Exclude ml_research/tests from the cloud bundle.",
    )
    args = parser.parse_args()

    ds_resolved = resolve_project_path(args.dataset)
    out_resolved = resolve_project_path(args.out)
    extra_resolved = [resolve_project_path(p) for p in args.extra_dataset]

    summary = build_cloud_bundle(
        ds_resolved,
        out_resolved,
        include_tests=not args.no_tests,
        extra_datasets=extra_resolved,
        max_size_mb=float(args.max_size_mb),
    )
    print(
        f"Packed {summary['file_count']} files -> {out_resolved} "
        f"({summary['size_mb']:.2f} MB, dataset_sha256={summary['dataset_sha256'][:12]}...)"
    )


if __name__ == "__main__":
    main()
