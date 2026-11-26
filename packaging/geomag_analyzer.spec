# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller --onedir build specification for GeoMag Analyzer (Windows x64)."""

from __future__ import annotations

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

REPO_ROOT = Path(SPECPATH).resolve().parent  # type: ignore[name-defined]
BACKEND_DIR = REPO_ROOT / "backend"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
FRONTEND_DIST_DIR = REPO_ROOT / "frontend" / "dist"
ICON_PATH = REPO_ROOT / "packaging" / "geomag_analyzer.ico"

if not (PROCESSED_DIR / "catalog.sqlite").is_file():
    raise RuntimeError(
        f"Missing normalized dataset at {PROCESSED_DIR / 'catalog.sqlite'}. "
        "Run `python -m geomag_etl.build` first."
    )
if not (FRONTEND_DIST_DIR / "index.html").is_file():
    raise RuntimeError(
        f"Missing built frontend bundle at {FRONTEND_DIST_DIR / 'index.html'}. "
        "Run `npm run build` inside frontend/ first."
    )

datas: list[tuple[str, str]] = []
for src_dir in (PROCESSED_DIR, FRONTEND_DIST_DIR):
    for item in sorted(src_dir.rglob("*")):
        if item.is_file():
            rel_parent = item.parent.relative_to(REPO_ROOT).as_posix()
            datas.append((str(item), rel_parent))

if ICON_PATH.is_file():
    datas.append((str(ICON_PATH), "assets"))

datas += collect_data_files("webview", subdir="lib")

binaries: list[tuple[str, str]] = []
binaries += collect_dynamic_libs("webview")
binaries += collect_dynamic_libs("clr_loader")

hiddenimports = (
    collect_submodules("geomag_core")
    + collect_submodules("geomag_api")
    + collect_submodules("uvicorn")
    + collect_submodules("websockets")
    + [
        "anyio._backends._asyncio",
        "httptools",
        "webview.platforms.winforms",
        "webview.platforms.edgechromium",
        "clr",
        "clr_loader",
        "pythonnet",
        "pyarrow.vendored.version",
        "scipy.special._cdflib",
        "scipy.special._ufuncs",
        "scipy.signal",
        "scipy.stats",
    ]
)

excludes = [
    "pytest",
    "ruff",
    "docx",
    "numba",
    "llvmlite",
    "tkinter",
    "matplotlib",
    "IPython",
    "jupyter",
    "notebook",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "wx",
    "gtk",
    "gi",
]

a = Analysis(
    [str(BACKEND_DIR / "launcher.py")],
    pathex=[str(BACKEND_DIR)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="GeoMagAnalyzer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ICON_PATH) if ICON_PATH.is_file() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="GeoMagAnalyzer",
)
