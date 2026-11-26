"""Build Windows --onedir, Portable ZIP, and Inno Setup installer for GeoMag Analyzer."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PACKAGING_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGING_DIR.parent
if str(PACKAGING_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGING_DIR))

from generate_icon import generate_icon_file  # noqa: E402

REQUIRED_PROCESSED_FILES = (
    "catalog.sqlite",
    "manifest.parquet",
    "geomag/storm_20261004_05.parquet",
    "geomag/storm_20261009_11.parquet",
    "geomag/storm_20261021_23.parquet",
    "geomag/storm_20261030_31.parquet",
    "solar_wind/solar1_l1_1min.parquet",
    "solar_wind/omni_bsn_1min.parquet",
    "cosmic_rays/oulu_nm_1min.parquet",
)

PORTABLE_README_TEXT = """GeoMag Analyzer - Portable Edition (Windows x64)
=================================================

1. Double-click GeoMagAnalyzer.exe to launch the desktop application
   (uses Microsoft Edge WebView2 when available; automatically falls back
   to your default system web browser if WebView2 is absent).

2. To force launching in your system web browser (Chrome / Edge / Firefox),
   double-click GeoMagAnalyzer_Browser.bat or run:
       GeoMagAnalyzer.exe --browser

   In browser mode, stop the server cleanly via the "Exit" button in the UI
   or from the Windows notification tray icon ("Stop Server / Exit").

3. 100% Offline Operation:
   - Normalized storm data and offline D3-geo world map (world-110m.topojson)
     are bundled inside _internal/ as strictly read-only assets.
   - Because `portable.mode` is present beside GeoMagAnalyzer.exe, runtime state
     (`runtime.sqlite`, `user_cache/`, logs) is written to `./user_data/` when
     writable (automatically falling back to `%LOCALAPPDATA%\\GeoMagAnalyzer`).
"""


def find_iscc_executable() -> Path | None:
    """Locate Inno Setup compiler (ISCC.exe) on PATH or standard install directories."""
    for cmd in ("ISCC.exe", "iscc"):
        found = shutil.which(cmd)
        if found:
            return Path(found)

    candidates = [
        Path(r"C:\Program Files\Inno Setup 7\ISCC.exe"),
        Path(r"C:\Program Files (x86)\Inno Setup 7\ISCC.exe"),
        Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"),
        Path(r"C:\Program Files\Inno Setup 6\ISCC.exe"),
    ]
    local_app = os.environ.get("LOCALAPPDATA")
    if local_app:
        for base in Path(local_app, "Programs").glob("Inno Setup*"):
            candidates.append(base / "ISCC.exe")

    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def verify_processed_dataset(processed_dir: Path | None = None) -> None:
    """Verify all required normalized ETL artifacts are present before bundling."""
    root = processed_dir or (REPO_ROOT / "data" / "processed")
    missing = [rel for rel in REQUIRED_PROCESSED_FILES if not (root / rel).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing normalized dataset files in data/processed/: "
            + ", ".join(missing)
            + ". Run `python -m geomag_etl.build` first."
        )


def build_frontend() -> None:
    """Build production React/TypeScript SPA into frontend/dist/."""
    npm_exe = shutil.which("npm.cmd") or shutil.which("npm")
    if not npm_exe:
        raise FileNotFoundError("npm executable not found on PATH.")
    subprocess.run([npm_exe, "run", "build"], cwd=str(REPO_ROOT / "frontend"), check=True)
    dist_index = REPO_ROOT / "frontend" / "dist" / "index.html"
    if not dist_index.is_file():
        raise FileNotFoundError(f"Expected built frontend at {dist_index}")


def build_pyinstaller_onedir() -> Path:
    """Run PyInstaller --onedir spec into dist/GeoMagAnalyzer/."""
    spec_path = PACKAGING_DIR / "geomag_analyzer.spec"
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", str(spec_path)],
        cwd=str(REPO_ROOT),
        check=True,
    )
    onedir = REPO_ROOT / "dist" / "GeoMagAnalyzer"
    exe_path = onedir / "GeoMagAnalyzer.exe"
    if not exe_path.is_file():
        raise FileNotFoundError(f"Expected PyInstaller executable at {exe_path}")

    internal_catalog = onedir / "_internal" / "data" / "processed" / "catalog.sqlite"
    internal_index = onedir / "_internal" / "frontend" / "dist" / "index.html"
    if not internal_catalog.is_file():
        raise FileNotFoundError(f"Missing bundled catalog.sqlite at {internal_catalog}")
    if not internal_index.is_file():
        raise FileNotFoundError(f"Missing bundled frontend/dist/index.html at {internal_index}")

    # portable.mode MUST NOT exist in dist/GeoMagAnalyzer/ (reserved strictly for the ZIP)
    stray_marker = onedir / "portable.mode"
    if stray_marker.exists():
        stray_marker.unlink()

    return onedir


def build_portable_zip(onedir: Path, zip_path: Path) -> Path:
    """Package dist/GeoMagAnalyzer/ plus portable.mode marker into a Portable ZIP."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_zip = zip_path.with_name(f".{zip_path.name}.{os.getpid()}.tmp")
    if tmp_zip.exists():
        tmp_zip.unlink()

    try:
        with zipfile.ZipFile(tmp_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for item in sorted(onedir.rglob("*")):
                if not item.is_file():
                    continue
                rel = item.relative_to(onedir).as_posix()
                if rel in ("portable.mode", "GeoMagAnalyzer_Browser.bat", "README_PORTABLE.txt"):
                    continue
                zf.write(item, arcname=f"GeoMagAnalyzer/{rel}")

            zf.writestr("GeoMagAnalyzer/portable.mode", "portable\n")
            zf.writestr(
                "GeoMagAnalyzer/GeoMagAnalyzer_Browser.bat",
                "@echo off\r\nstart \"\" \"%~dp0GeoMagAnalyzer.exe\" --browser %*\r\n",
            )
            zf.writestr(
                "GeoMagAnalyzer/README_PORTABLE.txt",
                PORTABLE_README_TEXT.replace("\n", "\r\n"),
            )
        os.replace(tmp_zip, zip_path)
    finally:
        if tmp_zip.exists():
            tmp_zip.unlink(missing_ok=True)
    return zip_path


def build_inno_installer() -> Path:
    """Compile packaging/installer.iss with Inno Setup ISCC.exe."""
    iscc = find_iscc_executable()
    if iscc is None:
        raise FileNotFoundError("Inno Setup compiler (ISCC.exe) was not found.")
    iss_path = PACKAGING_DIR / "installer.iss"
    subprocess.run([str(iscc), str(iss_path)], cwd=str(REPO_ROOT), check=True)
    setup_exe = REPO_ROOT / "dist" / "GeoMagAnalyzer_Setup.exe"
    if not setup_exe.is_file():
        raise FileNotFoundError(f"Expected Inno Setup installer at {setup_exe}")
    return setup_exe


def sha256_file(path: Path) -> str:
    """Compute lowercase SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def write_checksums(dist_dir: Path, artifacts: list[Path]) -> Path:
    """Write dist/SHA256SUMS.txt for generated distributables."""
    lines = [f"{sha256_file(p)} *{p.name}" for p in artifacts if p.is_file()]
    out = dist_dir / "SHA256SUMS.txt"
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Windows distributables for GeoMag Analyzer."
    )
    parser.add_argument(
        "--skip-frontend",
        action="store_true",
        help="Reuse existing frontend/dist/",
    )
    parser.add_argument(
        "--skip-pyinstaller",
        action="store_true",
        help="Reuse existing dist/GeoMagAnalyzer/",
    )
    parser.add_argument("--skip-zip", action="store_true", help="Skip Portable ZIP creation")
    parser.add_argument("--skip-inno", action="store_true", help="Skip Inno Setup compilation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    icon_path = PACKAGING_DIR / "geomag_analyzer.ico"
    if not icon_path.is_file():
        generate_icon_file(icon_path)

    verify_processed_dataset()

    if not args.skip_frontend:
        print("[1/4] Building frontend/dist/ ...", flush=True)
        build_frontend()

    onedir = REPO_ROOT / "dist" / "GeoMagAnalyzer"
    if not args.skip_pyinstaller:
        print("[2/4] Building PyInstaller --onedir bundle ...", flush=True)
        onedir = build_pyinstaller_onedir()
    elif not (onedir / "GeoMagAnalyzer.exe").is_file():
        raise FileNotFoundError(
            f"Cannot --skip-pyinstaller; missing {onedir / 'GeoMagAnalyzer.exe'}"
        )

    artifacts: list[Path] = []
    zip_path = REPO_ROOT / "dist" / "GeoMagAnalyzer_Portable_Win64.zip"
    if not args.skip_zip:
        print("[3/4] Creating Portable ZIP ...", flush=True)
        artifacts.append(build_portable_zip(onedir, zip_path))
    elif zip_path.is_file():
        artifacts.append(zip_path)

    setup_path = REPO_ROOT / "dist" / "GeoMagAnalyzer_Setup.exe"
    if not args.skip_inno:
        print("[4/4] Compiling Inno Setup installer ...", flush=True)
        artifacts.append(build_inno_installer())
    elif setup_path.is_file():
        artifacts.append(setup_path)

    sums_path = write_checksums(REPO_ROOT / "dist", artifacts)
    print(f"Build complete. Checksums written to {sums_path}", flush=True)
    for art in artifacts:
        print(f"  - {art} ({art.stat().st_size:,} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
