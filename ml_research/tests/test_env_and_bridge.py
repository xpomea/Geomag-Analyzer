"""Phase 0 (T0.1) tests: package import, RNG seed parity, core_bridge, device, and atomic I/O."""

from __future__ import annotations

import random
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

import geomag_ml
from geomag_ml.core_bridge import (
    BackendUnavailableError,
    import_analytic,
    import_ballistic_shift,
    import_conventions,
    import_cross_wavelet,
    import_etl_parsers,
    import_mlt,
    import_polarization,
    import_spectral,
    is_backend_available,
)
from geomag_ml.utils import (
    compute_sha256,
    load_npz,
    read_json,
    read_yaml,
    recommended_cpu_threads,
    recommended_num_workers,
    resolve_device,
    resolve_project_path,
    save_npz_atomic,
    set_seed,
    write_json,
    write_yaml,
)


def test_package_import_and_seed_reproducibility() -> None:
    assert geomag_ml.__version__ == "0.1.0"

    set_seed(42)
    r1 = [random.random() for _ in range(5)]
    n1 = np.random.randn(6)
    t1 = torch.randn(6)

    set_seed(42)
    r2 = [random.random() for _ in range(5)]
    n2 = np.random.randn(6)
    t2 = torch.randn(6)

    assert r1 == r2
    assert np.allclose(n1, n2, atol=0.0, rtol=0.0)
    assert torch.equal(t1, t2)


def test_core_bridge_imports_and_math() -> None:
    assert is_backend_available()
    mlt = import_mlt()
    conv = import_conventions()
    spectral = import_spectral()
    pol = import_polarization()
    analytic = import_analytic()
    xwt = import_cross_wavelet()
    etl = import_etl_parsers()
    ballistic = import_ballistic_shift()

    # Verify IGRF-14 dipole coordinates and haversine distance
    mlat_abk, mlon_abk = mlt.geo_to_magnetic_dipole(68.36, 18.82)
    assert 65.0 < mlat_abk < 68.0
    assert -180.0 <= mlon_abk <= 180.0

    dist_abk_kir = conv.haversine_km(68.36, 18.82, 67.84, 20.42)
    assert 85.0 < dist_abk_kir < 95.0

    # Verify imported callables exist without triggering FastAPI imports
    assert callable(spectral.compute_welch_spectral_matrix)
    assert callable(pol.stokes_from_spectral_matrix_2d)
    assert callable(analytic.find_valid_subsegments)
    assert callable(xwt.morlet_cwt_single)
    assert callable(etl.parse_iaga2002_file)
    assert callable(etl.parse_omni_hro_file)
    assert callable(ballistic.compute_ballistic_shift)
    assert callable(ballistic.resample_monotonic_series)


def test_core_bridge_missing_backend_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GEOMAG_BACKEND_DIR", str(tmp_path / "missing_backend"))
    assert not is_backend_available()
    with pytest.raises(BackendUnavailableError):
        import_mlt()


def test_device_resolution_and_cpu_threads() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dev = resolve_device("cuda")
    if not torch.cuda.is_available():
        assert dev.type == "cpu"
        assert len(caught) >= 1
        assert torch.get_num_threads() == recommended_cpu_threads()
        assert torch.get_num_threads() <= 6
    assert recommended_num_workers(dev) == 0


def test_atomic_io_roundtrips(tmp_path: Path) -> None:
    j_path = tmp_path / "sub" / "meta.json"
    write_json(j_path, {"a": np.int64(7), "b": float("nan")}, nan_to_null=True)
    assert read_json(j_path) == {"a": 7, "b": None}

    with pytest.raises(ValueError):
        write_json(j_path, {"b": float("nan")}, nan_to_null=False)

    y_path = tmp_path / "cfg.yaml"
    write_yaml(y_path, {"tag": "test", "val": np.float32(1.5)})
    assert read_yaml(y_path)["tag"] == "test"

    npz_path = tmp_path / "arr.npz"
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    save_npz_atomic(npz_path, {"x": arr})
    loaded = load_npz(npz_path)
    assert np.array_equal(loaded["x"], arr)
    assert len(compute_sha256(npz_path)) == 64


def test_synthetic_dataset_works_without_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Regression test for ML-02: cloud pytest without backend/ must build synthetic_dataset cleanly."""
    try:
        from tests.conftest import build_synthetic_dataset
    except ImportError:
        from conftest import build_synthetic_dataset

    monkeypatch.setenv("GEOMAG_BACKEND_DIR", str(tmp_path / "missing_backend"))
    assert not is_backend_available()
    arrays, meta = build_synthetic_dataset()
    assert arrays["X_raw"].shape == (600, 12, 3)
    assert len(meta["station_codes"]) == 12


def test_resolve_project_path_monorepo_and_cloud_bundle(tmp_path: Path) -> None:
    """Regression test for ML-01: resolve `ml_research/` paths across monorepo and cloud bundles."""
    # Monorepo layout: repo_root/ml_research exists
    mono_repo = tmp_path / "repo"
    mono_ml = mono_repo / "ml_research"
    (mono_ml / "data_cache").mkdir(parents=True)
    (mono_ml / "data_cache" / "ds.npz").write_bytes(b"npz")

    assert (
        resolve_project_path("ml_research/data_cache/ds.npz", ml_root=mono_ml, repo_root=mono_repo)
        == (mono_ml / "data_cache" / "ds.npz").resolve()
    )
    assert (
        resolve_project_path("data_cache/ds.npz", ml_root=mono_ml, repo_root=mono_repo)
        == (mono_ml / "data_cache" / "ds.npz").resolve()
    )

    # Cloud bundle layout: unpacked directly into ml_work/ without repo/ml_research/
    cloud_root = tmp_path / "content"
    cloud_ml = cloud_root / "ml_work"
    (cloud_ml / "data_cache").mkdir(parents=True)
    (cloud_ml / "data_cache" / "ds.npz").write_bytes(b"npz")

    assert (
        resolve_project_path("ml_research/data_cache/ds.npz", ml_root=cloud_ml, repo_root=cloud_root)
        == (cloud_ml / "data_cache" / "ds.npz").resolve()
    )
    assert (
        resolve_project_path("ml_research/artifacts", ml_root=cloud_ml, repo_root=cloud_root)
        == (cloud_ml / "artifacts").resolve()
    )
    assert (
        resolve_project_path("data_cache/ds.npz", ml_root=cloud_ml, repo_root=cloud_root)
        == (cloud_ml / "data_cache" / "ds.npz").resolve()
    )
    # Must NOT strip non-directory prefix like "ml_research_extra"
    assert (
        resolve_project_path("ml_research_extra/out", ml_root=cloud_ml, repo_root=cloud_root)
        == (cloud_ml / "ml_research_extra" / "out").resolve()
    )
