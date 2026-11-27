"""Phase 0 (T0.2, T0.3, T0.5) tests: dataset `.npz` contract, undespiked spikes, and extended ingestion."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geomag_ml.core_bridge import import_etl_parsers
from geomag_ml.data.ingest_extended import (
    build_extended_dataset,
    compute_event_flags,
    interpolate_eclipse_track,
    load_external_omni,
    parse_kp_csv,
)
from geomag_ml.data.preprocessing import BaselineEstimator
from geomag_ml.data.schema import (
    BELT_ID_TO_NAME,
    EVENT_FLAG_ECLIPSE,
    EVENT_FLAG_QUIET,
    NPZ_DTYPE_SPEC,
    OMNI_COLS,
    SOLAR1_BSN_COLS,
    SOLAR1_L1_COLS,
    load_ml_dataset,
    timestamps_to_unix_s,
    validate_dataset_arrays,
)
from geomag_ml.utils.io import compute_sha256, save_npz_atomic, write_json, write_yaml


def test_oct2026_clean_dataset_contract(ml_root: Path) -> None:
    npz_path = ml_root / "data_cache" / "geomag_ml_oct2026_clean_v1.npz"
    meta_path = ml_root / "data_cache" / "geomag_ml_oct2026_clean_v1.meta.json"
    assert npz_path.is_file(), f"Missing clean dataset at {npz_path}"
    assert meta_path.is_file(), f"Missing clean metadata at {meta_path}"
    assert npz_path.stat().st_size < 25 * 1024 * 1024

    arrays, meta = load_ml_dataset(npz_path, validate=True)
    assert meta["dataset_sha256"] == compute_sha256(npz_path)
    assert meta["recommended_use_doy"] is False
    assert meta["station_codes"] == sorted(meta["station_codes"])
    assert len(meta["episode_ids"]) == 4

    # Exact shapes and dtypes
    assert arrays["X_raw"].shape == (14400, 49, 3)
    assert arrays["M_valid"].shape == (14400, 49)
    assert arrays["station_dist_km"].shape == (49, 49)
    for key, dt in NPZ_DTYPE_SPEC.items():
        assert arrays[key].dtype == dt

    # Finiteness invariants
    x_raw = arrays["X_raw"]
    m_valid = arrays["M_valid"]
    assert np.all(np.isfinite(x_raw[m_valid]))
    assert np.all(np.isnan(x_raw[~m_valid]))
    assert int(arrays["M_known_spikes"].sum()) == 0
    assert int(arrays["M_interpolated"].sum()) == 100
    assert int(arrays["M_despiked"].sum()) == 10

    # Baseline and belt counts
    codes = meta["station_codes"]
    has_bl = arrays["station_has_baseline"]
    assert int(has_bl.sum()) == 47
    baseless = [codes[i] for i in np.flatnonzero(~has_bl)]
    assert baseless == ["GCK", "NCK"]

    belts = arrays["station_belt_id"]
    counts = {BELT_ID_TO_NAME[b]: int((belts == b).sum()) for b in range(4)}
    assert counts == {"EU_mid": 31, "EU_aur": 6, "CA_mid": 5, "CA_aur": 7}

    # MLT and cyclic features
    mlt = arrays["mlt_hours"]
    assert np.all((mlt >= 0.0) & (mlt < 24.0))
    mlt_rad = 2.0 * np.pi * (mlt / 24.0)
    assert np.allclose(np.sin(mlt_rad) ** 2 + np.cos(mlt_rad) ** 2, 1.0, atol=1e-5)

    tf = arrays["time_features"]
    assert np.allclose(tf[:, 0] ** 2 + tf[:, 1] ** 2, 1.0, atol=1e-5)
    assert np.allclose(tf[:, 2] ** 2 + tf[:, 3] ** 2, 1.0, atol=1e-5)

    # Distance matrix
    dist = arrays["station_dist_km"]
    assert np.allclose(dist, dist.T, atol=1e-4)
    assert np.all(np.diag(dist) == 0.0)
    abk_idx, kir_idx = codes.index("ABK"), codes.index("KIR")
    assert 90.0 <= float(dist[abk_idx, kir_idx]) <= 93.0

    # OMNI coverage and positive Ey sign agreement with SOLAR-1 BSN
    sw_omni = arrays["sw_omni"]
    ae_col = sw_omni[:, OMNI_COLS.index("ae")]
    sym_col = sw_omni[:, OMNI_COLS.index("sym_h")]
    assert int(np.isfinite(ae_col).sum()) == 7200
    assert int(np.isfinite(sym_col).sum()) == 0

    ey_bsn = arrays["sw_solar1_bsn"][:, SOLAR1_BSN_COLS.index("ey_mVm")]
    ef_omni = sw_omni[:, OMNI_COLS.index("electric_field")]
    common = np.isfinite(ey_bsn) & np.isfinite(ef_omni)
    assert int(common.sum()) >= 3500
    corr_ey = float(np.corrcoef(ey_bsn[common], ef_omni[common])[0, 1])
    assert corr_ey > 0.75

    # QC exclusions and baseline step warnings
    m_qc = arrays["M_qc_clean"]
    ts = arrays["timestamps_unix_s"]
    ep_idx = arrays["episode_idx"]
    # Verify MEA 50-min window and LON@ep3 / SFS@ep1 are excluded from M_qc_clean
    mea_idx = codes.index("MEA")
    t_mea_0 = int(pd.Timestamp("2026-10-23T17:54:00Z").timestamp())
    t_mea_1 = int(pd.Timestamp("2026-10-23T18:43:00Z").timestamp())
    assert not np.any(m_qc[(ts >= t_mea_0) & (ts <= t_mea_1), mea_idx])
    assert not np.any(m_qc[ep_idx == 3, codes.index("LON")])
    assert not np.any(m_qc[ep_idx == 1, codes.index("SFS")])

    warn_stations = {w["station"] for w in meta.get("baseline_step_warnings", [])}
    assert {"LON", "SFS"}.issubset(warn_stations)


def test_oct2026_undespiked_dataset_contract(ml_root: Path) -> None:
    clean_path = ml_root / "data_cache" / "geomag_ml_oct2026_clean_v1.npz"
    un_path = ml_root / "data_cache" / "geomag_ml_oct2026_undespiked_v1.npz"
    assert clean_path.is_file(), f"Missing clean dataset at {clean_path}"
    assert un_path.is_file(), f"Missing undespiked dataset at {un_path}"
    assert un_path.stat().st_size < 25 * 1024 * 1024

    clean_arr, clean_meta = load_ml_dataset(clean_path, validate=True)
    un_arr, un_meta = load_ml_dataset(un_path, validate=True)
    assert un_meta["dataset_sha256"] == compute_sha256(un_path)

    m_valid = clean_arr["M_valid"]
    m_despiked = clean_arr["M_despiked"]
    outside = m_valid & (~m_despiked)
    assert np.array_equal(un_arr["X_raw"][outside], clean_arr["X_raw"][outside])

    diff = np.max(np.abs(un_arr["X_raw"][m_valid] - clean_arr["X_raw"][m_valid]), axis=-1)
    assert int(np.sum(diff > 1e-3)) == 10
    assert int(un_arr["M_known_spikes"].sum()) == 174

    codes = un_meta["station_codes"]
    ts = un_arr["timestamps_unix_s"]
    t_val = int(np.flatnonzero(ts == int(pd.Timestamp("2026-10-09T10:39:00Z").timestamp()))[0])
    t_ott = int(np.flatnonzero(ts == int(pd.Timestamp("2026-10-22T13:48:00Z").timestamp()))[0])
    dz_val = abs(
        float(un_arr["X_raw"][t_val, codes.index("VAL"), 2])
        - float(clean_arr["X_raw"][t_val, codes.index("VAL"), 2])
    )
    dz_ott = abs(
        float(un_arr["X_raw"][t_ott, codes.index("OTT"), 2])
        - float(clean_arr["X_raw"][t_ott, codes.index("OTT"), 2])
    )
    assert dz_val > 1200.0
    assert dz_ott > 350.0


def _write_synthetic_iaga2002(
    path: Path,
    *,
    iaga_code: str,
    lat_deg: float,
    lon_deg: float,
    date_str: str,
    base_xyz: tuple[float, float, float] = (18000.0, 500.0, 46000.0),
) -> None:
    start = datetime.fromisoformat(f"{date_str}T00:00:00+00:00")
    doy = start.timetuple().tm_yday
    header = [
        f" Format                 IAGA-2002                                    |",
        f" Source of Data         Synthetic Test                               |",
        f" Station Name           Synthetic {iaga_code:<34}|",
        f" IAGA Code              {iaga_code:<45}|",
        f" Geodetic Latitude      {lat_deg:<45.3f}|",
        f" Geodetic Longitude     {lon_deg:<45.3f}|",
        f" Elevation              100                                          |",
        f" Reported               XYZF                                         |",
        f" Data Type              definitive                                   |",
        "DATE       TIME         DOY     XYZ-X     XYZ-Y     XYZ-Z     XYZ-F  |",
    ]
    rows = []
    for m in range(1440):
        dt = start + timedelta(minutes=m)
        x = base_xyz[0] + 5.0 * np.sin(2.0 * np.pi * m / 1440.0)
        y = base_xyz[1] + 3.0 * np.cos(2.0 * np.pi * m / 1440.0)
        z = base_xyz[2] + 2.0 * np.sin(4.0 * np.pi * m / 1440.0)
        f = float(np.sqrt(x * x + y * y + z * z))
        rows.append(
            f"{dt:%Y-%m-%d} {dt:%H:%M:%S}.000 {doy:03d}  "
            f"{x:8.2f}  {y:8.2f}  {z:8.2f}  {f:8.2f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(header + rows) + "\n", encoding="utf-8")


def _write_synthetic_omni_hro(
    path: Path,
    timestamps: pd.DatetimeIndex,
    *,
    sym_h_val: float = -6.0,
    ae_val: float = 42.0,
    bz_val: float = 1.5,
) -> None:
    etl = import_etl_parsers()
    cols = etl.OMNI_HRO_COLUMNS
    lines: list[str] = []
    for ts in timestamps:
        parts: list[str] = []
        values = {
            "year": ts.year,
            "doy": ts.timetuple().tm_yday,
            "hour": ts.hour,
            "minute": ts.minute,
            "bz_gsm_nT": bz_val,
            "flow_speed_kms": 380.0,
            "vx_gse_kms": -380.0,
            "pdyn_nPa": 1.8,
            "electric_field_mVm": -0.57,
            "ae_nT": int(ae_val),
            "al_nT": -20,
            "au_nT": 22,
            "sym_h_nT": int(sym_h_val),
            "pc_n_index": 0.4,
        }
        for name, width, kind, fill in cols:
            val = values.get(name, fill)
            if kind == "int":
                parts.append(f"{int(val):{width}d}")
            else:
                dec = 2
                if name in ("flow_speed_kms", "vx_gse_kms", "vy_gse_kms", "vz_gse_kms", "alfven_mach", "magnetosonic_mach"):
                    dec = 1
                elif name == "proton_T_K":
                    dec = 0
                parts.append(f"{float(val):{width}.{dec}f}")
        line = "".join(parts)
        assert len(line) == etl.OMNI_LINE_WIDTH
        lines.append(line)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_extended_dataset_ingestion_and_quiet_baseline(
    synthetic_dataset: tuple[dict[str, np.ndarray], dict],
    tmp_path: Path,
) -> None:
    base_arrays, base_meta = synthetic_dataset
    base_npz = tmp_path / "base.npz"
    save_npz_atomic(base_npz, base_arrays)
    write_json(base_npz.with_suffix(".meta.json"), base_meta, nan_to_null=True)

    ext_root = tmp_path / "external"
    # Two new quiet days (2026-12-10..11 => >60 days from 2026-10-04 => recommended_use_doy=True)
    # Include existing station BEL and brand-new station NEW (Canada mid-lat)
    for d_str in ("2026-12-10", "2026-12-11"):
        compact = d_str.replace("-", "")
        _write_synthetic_iaga2002(
            ext_root / "intermagnet" / f"bel{compact}dmin.min",
            iaga_code="BEL",
            lat_deg=51.84,
            lon_deg=20.79,
            date_str=d_str,
            base_xyz=(19500.0, 2100.0, 44500.0),
        )
        _write_synthetic_iaga2002(
            ext_root / "intermagnet" / f"new{compact}dmin.min",
            iaga_code="NEW",
            lat_deg=47.0,
            lon_deg=-95.0,
            date_str=d_str,
            base_xyz=(17000.0, 1200.0, 52000.0),
        )

    quiet_ts = pd.date_range("2026-12-10T00:00:00Z", "2026-12-11T23:59:00Z", freq="1min", tz="UTC")
    omni_path = ext_root / "omni" / "omni_quiet.asc"
    _write_synthetic_omni_hro(omni_path, quiet_ts, sym_h_val=-7.0, ae_val=35.0, bz_val=2.0)

    # Verify raw ETL parser gives <NA> episode_id on Dec 2026, while load_external_omni ignores it
    omni_loaded = load_external_omni([omni_path])
    assert len(omni_loaded) == 2880
    assert np.allclose(omni_loaded["sym_h"].to_numpy(), -7.0)

    # 3-hour Kp=1.0 CSV across both quiet days
    kp_rows = []
    for day_offset in range(2):
        for hr in range(0, 24, 3):
            ts_k = datetime(2026, 12, 10, tzinfo=UTC) + timedelta(days=day_offset, hours=hr)
            kp_rows.append({"timestamp_utc": ts_k.strftime("%Y-%m-%dT%H:%M:%SZ"), "kp": 1.0})
    kp_path = ext_root / "kp" / "kp_dec2026.csv"
    kp_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(kp_rows).to_csv(kp_path, index=False, encoding="utf-8")
    assert len(parse_kp_csv(kp_path)) == 2880

    # Events YAML with quiet interval + static eclipse + moving eclipse track
    events_path = ext_root / "events.yaml"
    write_yaml(
        events_path,
        {
            "quiet_kp_le2": [
                {
                    "label": "Quiet Dec 2026",
                    "start_utc": "2026-12-10T00:00:00Z",
                    "end_utc": "2026-12-11T23:59:00Z",
                }
            ],
            "eclipse": [
                {
                    "label": "Static eclipse",
                    "start_utc": "2026-12-11T10:00:00Z",
                    "end_utc": "2026-12-11T12:00:00Z",
                    "lat_deg": 51.84,
                    "lon_deg": 20.79,
                    "radius_km": 600.0,
                }
            ],
            "storm": [],
            "spe": [],
        },
    )

    cfg = {
        "mode": "extended",
        "tag": "ext_test_v1",
        "base_npz": str(base_npz),
        "external_dir": str(ext_root),
        "out_dir": str(tmp_path / "out"),
        "intermagnet_dirs": [str(ext_root / "intermagnet" / "*.min")],
        "omni_files": [str(omni_path)],
        "kp_files": [str(kp_path)],
        "events_file": str(events_path),
        "episode_gap_minutes": 60,
    }
    out_npz, out_meta_path = build_extended_dataset(cfg, repo_root=tmp_path)
    arrays, meta = load_ml_dataset(out_npz, validate=True)
    assert out_meta_path.is_file()

    # Check new station NEW and merged episode count (2 base + 1 new quiet = 3)
    assert "NEW" in meta["station_codes"]
    assert len(meta["station_codes"]) == 13
    assert len(meta["episode_ids"]) == 3
    assert arrays["X_raw"].shape == (3480, 13, 3)  # 600 + 2880
    assert meta["recommended_use_doy"] is True

    # Verify SYM-H and Kp alignment on the newly appended episode (indices 600..3480)
    sym_col = arrays["sw_omni"][600:, OMNI_COLS.index("sym_h")]
    kp_col = arrays["sw_omni"][600:, OMNI_COLS.index("kp")]
    assert np.allclose(sym_col, -7.0)
    assert np.allclose(kp_col, 1.0)

    # Check event flags (quiet bit in Dec 10-11, eclipse bit during 10:00..12:00 on Dec 11)
    ts = arrays["timestamps_unix_s"]
    t_ecl0 = int(pd.Timestamp("2026-12-11T10:00:00Z").timestamp())
    t_ecl1 = int(pd.Timestamp("2026-12-11T12:00:00Z").timestamp())
    in_ecl = (ts >= t_ecl0) & (ts <= t_ecl1)
    assert np.all((arrays["event_flags"][in_ecl] & EVENT_FLAG_ECLIPSE) != 0)
    assert np.all((arrays["event_flags"][600:] & EVENT_FLAG_QUIET) != 0)

    # Verify moving eclipse track interpolation across the +/-180 dateline
    moving_ev = {
        "track": [
            {"timestamp_utc": "2026-12-11T10:00:00Z", "lat_deg": 60.0, "lon_deg": 179.0, "radius_km": 700.0},
            {"timestamp_utc": "2026-12-11T11:00:00Z", "lat_deg": 62.0, "lon_deg": -179.0, "radius_km": 900.0},
        ]
    }
    act, lat_t, lon_t, rad_t = interpolate_eclipse_track(moving_ev, ts)
    t_mid = int(np.flatnonzero(ts == int(pd.Timestamp("2026-12-11T10:30:00Z").timestamp()))[0])
    assert act[t_mid]
    assert np.isclose(lat_t[t_mid], 61.0, atol=1e-4)
    assert abs(abs(float(lon_t[t_mid])) - 180.0) < 1e-4
    assert np.isclose(rad_t[t_mid], 800.0, atol=1e-4)

    # Verify BaselineEstimator(mode="target_kp_quiet_night") works on extended dataset without leakage
    train_idx = np.arange(300, 3480, dtype=np.int64)
    est = BaselineEstimator(mode="target_kp_quiet_night").fit(arrays, train_idx)
    delta = est.transform(
        arrays["X_raw"],
        arrays["M_valid"],
        arrays["episode_idx"],
        timestamps_unix_s=arrays["timestamps_unix_s"],
        station_has_baseline=arrays["station_has_baseline"],
    )
    bel_idx = meta["station_codes"].index("BEL")
    new_idx = meta["station_codes"].index("NEW")
    assert np.isfinite(est.B0_nT_[bel_idx]).all()
    assert np.isfinite(est.B0_nT_[new_idx]).all()
    assert np.all(np.isfinite(delta[600:, new_idx, :]))
    assert compute_event_flags(ts[:10], None).sum() == 0


def test_extended_merge_adjacent_dates_preserves_base_data(
    synthetic_dataset: tuple[dict[str, np.ndarray], dict],
    tmp_path: Path,
) -> None:
    """Regression test for ML-01: merging adjacent dates must preserve base episode observations."""
    base_arrays, base_meta = synthetic_dataset
    base_npz = tmp_path / "base.npz"
    save_npz_atomic(base_npz, base_arrays)
    write_json(base_npz.with_suffix(".meta.json"), base_meta, nan_to_null=True)

    ext_root = tmp_path / "ext_adj"
    # Base syn_ep0 starts 2026-10-04T00:00Z; add contiguous preceding day 2026-10-03 for BEL
    _write_synthetic_iaga2002(
        ext_root / "intermagnet" / "bel20261003dmin.min",
        iaga_code="BEL",
        lat_deg=51.84,
        lon_deg=20.79,
        date_str="2026-10-03",
        base_xyz=(20100.0, 1600.0, 43100.0),
    )

    cfg = {
        "mode": "extended",
        "tag": "ext_adj_v1",
        "base_npz": str(base_npz),
        "external_dir": str(ext_root),
        "out_dir": str(tmp_path / "out"),
        "intermagnet_dirs": [str(ext_root / "intermagnet")],
        "episode_gap_minutes": 60,
    }
    out_npz, _ = build_extended_dataset(cfg, repo_root=tmp_path)
    arrays, meta = load_ml_dataset(out_npz, validate=True)

    # 2026-10-03 (1440m) + syn_ep0 2026-10-04 00:00..04:59Z (300m) merge into 1740-min episode
    assert len(meta["episode_ids"]) == 2
    assert int(arrays["episode_bounds"][0, 1]) == 1740
    bel_idx = meta["station_codes"].index("BEL")

    # External 2026-10-03 samples present (1440/1440) AND base syn_ep0 samples preserved (300/300)
    assert int(arrays["M_valid"][:1440, bel_idx].sum()) == 1440
    assert int(arrays["M_valid"][1440:1740, bel_idx].sum()) == 300
    assert np.allclose(
        arrays["X_raw"][1440:1740, bel_idx, :],
        base_arrays["X_raw"][:300, base_meta["station_codes"].index("BEL"), :],
        atol=1e-4,
    )


def test_extended_min_file_with_internal_gap_gt_60min(tmp_path: Path) -> None:
    """Regression test for ML-03: `.min` with >60 min missing block splits cleanly without IndexError."""
    ext_root = tmp_path / "ext_gap"
    min_path = ext_root / "intermagnet" / "new20261210dmin.min"
    _write_synthetic_iaga2002(
        min_path,
        iaga_code="NEW",
        lat_deg=50.0,
        lon_deg=15.0,
        date_str="2026-12-10",
    )
    # Remove rows for 05:00..06:59 UTC (120-min gap > 60-min threshold)
    lines = min_path.read_text(encoding="utf-8").splitlines()
    header, body = lines[:10], lines[10:]
    filtered_body = body[:300] + body[420:]
    min_path.write_text("\n".join(header + filtered_body) + "\n", encoding="utf-8")

    cfg = {
        "mode": "extended",
        "tag": "ext_gap_v1",
        "base_npz": None,
        "external_dir": str(ext_root),
        "out_dir": str(tmp_path / "out"),
        "intermagnet_dirs": [str(ext_root / "intermagnet" / "*.min")],
        "episode_gap_minutes": 60,
        "default_event_flag": "quiet_kp_le2",
    }
    out_npz, _ = build_extended_dataset(cfg, repo_root=tmp_path)
    arrays, meta = load_ml_dataset(out_npz, validate=True)

    assert len(meta["episode_ids"]) == 2
    assert arrays["episode_bounds"].tolist() == [[0, 300], [300, 1320]]
    assert int(arrays["M_valid"].sum()) == 1320


def test_extended_loaders_dir_expansion_solar1_csv_and_missing_base(
    synthetic_dataset: tuple[dict[str, np.ndarray], dict],
    tmp_path: Path,
) -> None:
    """Regression test for ML-04: directory globs, tz-naive SOLAR-1 CSV, and missing base_npz error."""
    base_arrays, base_meta = synthetic_dataset
    base_npz = tmp_path / "base.npz"
    save_npz_atomic(base_npz, base_arrays)
    write_json(base_npz.with_suffix(".meta.json"), base_meta, nan_to_null=True)

    ext_root = tmp_path / "ext_dirs"
    _write_synthetic_iaga2002(
        ext_root / "min_dir" / "new20261210dmin.min",
        iaga_code="NEW",
        lat_deg=47.0,
        lon_deg=-95.0,
        date_str="2026-12-10",
    )
    dec_ts = pd.date_range("2026-12-10T00:00:00Z", "2026-12-10T23:59:00Z", freq="1min", tz="UTC")
    _write_synthetic_omni_hro(ext_root / "omni_dir" / "omni_dec10.asc", dec_ts, sym_h_val=-4.0)
    (ext_root / "kp_dir").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"timestamp_utc": "2026-12-10T00:00:00Z", "kp": 1.3}]
    ).to_csv(ext_root / "kp_dir" / "kp.csv", index=False, encoding="utf-8")

    # tz-naive SOLAR-1 CSV with short canonical column names
    (ext_root / "s1_dir").mkdir(parents=True, exist_ok=True)
    s1_df = pd.DataFrame(
        {
            "timestamp_utc": dec_ts.tz_localize(None).strftime("%Y-%m-%d %H:%M:%S"),
            "bz_gsm": np.full(1440, -5.0),
            "b_mag_gsm": np.full(1440, 7.0),
            "vx_gse": np.full(1440, -400.0),
            "proton_n": np.full(1440, 5.0),
            "pdyn_nPa": np.full(1440, 2.0),
        }
    )
    s1_df.to_csv(ext_root / "s1_dir" / "solar1_dec10.csv", index=False, encoding="utf-8")

    (ext_root / "oulu_dir").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp_utc": dec_ts.tz_localize(None).strftime("%Y-%m-%d %H:%M:%S"),
            "delta_n_pct": np.full(1440, 0.5),
        }
    ).to_csv(ext_root / "oulu_dir" / "oulu_dec10.csv", index=False, encoding="utf-8")

    cfg = {
        "mode": "extended",
        "tag": "ext_dirs_v1",
        "base_npz": str(base_npz),
        "external_dir": str(ext_root),
        "out_dir": str(tmp_path / "out"),
        "intermagnet_dirs": [str(ext_root / "min_dir")],
        "omni_files": [str(ext_root / "omni_dir")],
        "kp_files": [str(ext_root / "kp_dir")],
        "solar1_files": [str(ext_root / "s1_dir")],
        "oulu_files": [str(ext_root / "oulu_dir")],
        "episode_gap_minutes": 60,
    }
    out_npz, _ = build_extended_dataset(cfg, repo_root=tmp_path)
    arrays, _ = load_ml_dataset(out_npz, validate=True)

    # Check directory-discovered files loaded properly while base SOLAR-1 stayed intact
    assert np.allclose(arrays["sw_omni"][600:, OMNI_COLS.index("sym_h")], -4.0)
    assert np.allclose(arrays["sw_omni"][600:780, OMNI_COLS.index("kp")], 1.3)
    assert np.allclose(arrays["sw_solar1_l1"][600:, SOLAR1_L1_COLS.index("vx_gse")], -400.0)
    assert np.isfinite(arrays["sw_solar1_bsn"][700:, SOLAR1_BSN_COLS.index("ey_mVm")]).all()
    assert np.allclose(arrays["sw_solar1_bsn"][:600], base_arrays["sw_solar1_bsn"], equal_nan=True)
    assert np.allclose(arrays["oulu_delta_n_pct"][600:], 0.5)

    # Missing explicit base_npz / events_file must raise FileNotFoundError
    bad_cfg = dict(cfg, base_npz=str(tmp_path / "missing_base.npz"))
    with pytest.raises(FileNotFoundError):
        build_extended_dataset(bad_cfg, repo_root=tmp_path)

    bad_ev_cfg = dict(cfg, events_file=str(tmp_path / "missing_events.yaml"))
    with pytest.raises(FileNotFoundError):
        build_extended_dataset(bad_ev_cfg, repo_root=tmp_path)
