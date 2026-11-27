# `ml_research` — Isolated ML/DL Research Subsystem

Isolated machine-learning and deep-learning subsystem for the thesis:
*"Recognition of ionospheric and geomagnetic events using wavelet analysis and deep learning methods"*.

## Isolation Rules

- All ML/DL code, configs, tests, cached `.npz` datasets, and run artifacts live strictly inside `ml_research/`.
- Uses dedicated virtual environment `.venv_ml` (Python 3.12 + CPU PyTorch locally).
- Reads `data/processed/`, `GeoMag_Storms_Dataset_Oct-2026/`, and `backend/` strictly **read-only** (never modifies files outside `ml_research/`).

## Local CPU Setup (Windows)

```powershell
py -3.12 -m venv .venv_ml
.venv_ml\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.venv_ml\Scripts\pip.exe install -r ml_research/requirements_local_cpu.txt
.venv_ml\Scripts\pip.exe install -e ml_research --no-deps
```

## Phase 0 Commands

Build clean and undespiked datasets:

```powershell
.venv_ml\Scripts\python.exe ml_research/scripts/build_dataset.py --config ml_research/configs/dataset_oct2026.yaml --also-undespiked
```

Run tests:

```powershell
.venv_ml\Scripts\pytest.exe -c ml_research\pytest.ini ml_research\tests
```

## Cloud Portability (Colab / Kaggle GPU T4)

Training scripts accept `--device cuda` and consume the self-contained `.npz` dataset bundle (`data_cache/geomag_ml_*.npz`) without requiring `backend/` at training time.
