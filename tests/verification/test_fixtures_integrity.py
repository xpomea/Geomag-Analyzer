"""Integrity tests ensuring docx_section11.json matches the .docx specification and raw data."""

from __future__ import annotations

import json
import math
import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURE_PATH = pathlib.Path(__file__).resolve().parent / "fixtures" / "docx_section11.json"
RAW_ROOT = REPO_ROOT / "GeoMag_Storms_Dataset_Oct-2026"
DOCX_PATH = RAW_ROOT / "GeoMag Storms" / "Pc5_Pc6_Polarization_Formulas_INTERMAGNET_XYZ_RU.docx"


def _clean_num(text: str) -> float:
    cleaned = (
        text.strip()
        .replace("\u2212", "-")
        .replace("−", "-")
        .replace("\u2005", "")
        .replace("\u2009", "")
        .replace("\xa0", "")
        .replace("×10", "e")
        .replace("+", "")
        .replace("°", "")
        .replace("\u2218", "")
        .replace(" ", "")
    )
    return float(cleaned)


def _split_slash_pair(text: str) -> tuple[float, float]:
    parts = text.split("/")
    assert len(parts) == 2, f"Expected slash-separated pair in {text!r}"
    return _clean_num(parts[0]), _clean_num(parts[1])


def _cell_text(cell_elem) -> str:
    return "".join(
        node.text
        for node in cell_elem.iter()
        if node.tag.split("}")[-1] == "t" and node.text
    ).strip()


def _find_table_with_header(doc, prefix: str):
    for table in doc.tables:
        rows = [
            [_cell_text(cell._element) for cell in row.cells]
            for row in table.rows
        ]
        if rows and rows[0] and rows[0][0].startswith(prefix):
            return rows
    raise AssertionError(f"Table starting with {prefix!r} not found in docx")


def test_fixture_schema_and_completeness():
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    required_keys = {
        "metadata",
        "table_1_bands",
        "table_2_intermagnet_gaussian_filter",
        "table_3_welch_effective_n",
        "table_5_critical_thresholds",
        "table_8_butterworth_max_diff_vs_analog",
        "table_10_interpolation_rmse_pct",
        "table_11_sos_coefficients",
        "table_14_edge_zones_hours",
        "table_16_analytic_edge_error",
        "table_17_filtered_effective_n",
        "table_18_synthetic_signals",
        "table_19_expected_2d",
        "case_c_expected_3d",
        "table_20_case_d",
        "table_21_case_e",
    }
    assert required_keys.issubset(data.keys())
    assert len(data["table_5_critical_thresholds"]) == 7
    assert len(data["table_11_sos_coefficients"]["Pc5"]) == 3
    assert len(data["table_11_sos_coefficients"]["Pc6"]) == 3
    assert len(data["table_20_case_d"]["modes"]) == 8
    assert len(data["table_21_case_e"]["modes"]) == 7


def test_raw_dataset_untouched_baseline():
    assert RAW_ROOT.exists(), f"Missing raw dataset at {RAW_ROOT}"
    files = [p for p in RAW_ROOT.rglob("*") if p.is_file()]
    assert len(files) == 483
    assert sum(p.stat().st_size for p in files) == 108_484_030


def test_fixture_matches_docx_tables():
    try:
        import docx
    except ImportError:
        pytest.skip("python-docx is not installed")

    assert DOCX_PATH.exists()
    doc = docx.Document(str(DOCX_PATH))
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    all_tables = [
        [
            [_cell_text(cell._element) for cell in row.cells]
            for row in tbl.rows
        ]
        for tbl in doc.tables
    ]

    # Table 1: Frequency bands
    t1_rows = next(rows for rows in all_tables if len(rows) == 4 and rows[0][0] == "Полоса")
    assert "150" in t1_rows[1][1] and "600" in t1_rows[1][1]
    assert "600" in t1_rows[2][1] and "1800" in t1_rows[2][3] and "900" in t1_rows[2][3]
    assert "120" in t1_rows[3][1]
    assert math.isclose(data["table_1_bands"]["pc5"]["period_min_sec"], 150.0)
    assert math.isclose(data["table_1_bands"]["pc5"]["period_max_sec"], 600.0)
    assert math.isclose(data["table_1_bands"]["pc6"]["default_period_max_sec"], 1800.0)
    assert math.isclose(data["table_1_bands"]["pc6"]["alt_period_max_sec"], 900.0)
    assert math.isclose(data["table_1_bands"]["nyquist_1min"]["period_sec"], 120.0)

    # Table 2: INTERMAGNET 1-minute Gaussian filter response
    t2_rows = next(rows for rows in all_tables if len(rows) == 2 and rows[0][0] == "Период T (с)")
    for p_str, val_str in zip(t2_rows[0][1:], t2_rows[1][1:], strict=True):
        assert math.isclose(
            data["table_2_intermagnet_gaussian_filter"][p_str.strip()],
            _clean_num(val_str),
            abs_tol=1e-9,
        )

    # Table 3: Welch effective n
    t3_rows = next(rows for rows in all_tables if len(rows) == 5 and rows[0][0] == "Окно")
    assert "(NB=39)" in t3_rows[0][3] and "(NB=8)" in t3_rows[0][4]
    for r_doc, w_fix in zip(t3_rows[1:], data["table_3_welch_effective_n"]["windows"], strict=True):
        assert int(r_doc[1]) == w_fix["n_samples"]
        assert int(r_doc[2]) == w_fix["K"]
        pc5_f, pc5_r = _split_slash_pair(r_doc[3])
        pc6_f, pc6_r = _split_slash_pair(r_doc[4])
        assert int(pc5_f) == w_fix["pc5_flat"] and int(pc5_r) == w_fix["pc5_red_f2"]
        assert int(pc6_f) == w_fix["pc6_flat"] and int(pc6_r) == w_fix["pc6_red_f2"]

    # Table 5: Critical thresholds
    t5 = _find_table_with_header(doc, "n")
    assert len(t5) == 8
    for row_doc, row_fix in zip(t5[1:], data["table_5_critical_thresholds"], strict=True):
        assert int(row_doc[0]) == row_fix["n"]
        assert abs(_clean_num(row_doc[1]) - row_fix["p_crit_95"]) < 1e-9
        assert abs(_clean_num(row_doc[2]) - row_fix["p_crit_99"]) < 1e-9
        assert abs(_clean_num(row_doc[3]) - row_fix["gamma2_crit_95"]) < 1e-9
        assert abs(_clean_num(row_doc[4]) - row_fix["gamma2_crit_99"]) < 1e-9

    # Table 8: Butterworth max diff vs analog
    t8_rows = next(
        rows
        for rows in all_tables
        if len(rows) == 7 and rows[0] == ["Полоса, порядок", "m=1", "m=2", "m=4", "m=8"]
    )
    t8_keys = ["Pc5_n2", "Pc5_n3", "Pc5_n4", "Pc6_n2", "Pc6_n3", "Pc6_n4"]
    for row_doc, k8 in zip(t8_rows[1:], t8_keys, strict=True):
        for col_idx, m_key in enumerate(["1", "2", "4", "8"], start=1):
            exp8 = data["table_8_butterworth_max_diff_vs_analog"][k8][m_key]
            assert abs(_clean_num(row_doc[col_idx]) - exp8) < 1e-9

    # Table 10: Oversampling interpolation RMSE %
    t10_rows = next(rows for rows in all_tables if len(rows) == 6 and rows[0][0] == "Метод")
    t10_keys = [
        "linear",
        "cubic_spline",
        "fourier_no_pad",
        "fourier_odd_reflect_P240",
        "kaiser_fir_Q16",
    ]
    t10_cols = ["Pc5_m4", "Pc5_m8", "Pc6_m4", "Pc6_m8"]
    for row_doc, m_key in zip(t10_rows[1:], t10_keys, strict=True):
        for col_idx, b_key in enumerate(t10_cols, start=1):
            v0, v1 = _split_slash_pair(row_doc[col_idx])
            exp = data["table_10_interpolation_rmse_pct"][m_key][b_key]
            assert math.isclose(v0, exp[0], abs_tol=1e-9)
            assert math.isclose(v1, exp[1], abs_tol=1e-9)

    # Table 11: SOS coefficients
    t11_rows = next(
        rows
        for rows in all_tables
        if len(rows) == 7 and rows[0] == ["Фильтр", "b0", "b2", "a1", "a2"]
    )
    all_sos = data["table_11_sos_coefficients"]["Pc5"] + data["table_11_sos_coefficients"]["Pc6"]
    for row_doc, sec_fix in zip(t11_rows[1:], all_sos, strict=True):
        assert abs(_clean_num(row_doc[1]) - sec_fix["b0"]) < 1e-12
        assert abs(_clean_num(row_doc[2]) - sec_fix["b2"]) < 1e-12
        assert abs(_clean_num(row_doc[3]) - sec_fix["a1"]) < 1e-12
        assert abs(_clean_num(row_doc[4]) - sec_fix["a2"]) < 1e-12

    # Table 14: Edge zones [hours]
    t14_rows = next(
        rows
        for rows in all_tables
        if len(rows) == 5 and rows[0][0] == "Фильтр" and "Pc5" in rows[0][1]
    )
    t14_keys = ["fft_mask", "sos_n2", "sos_n3", "sos_n4"]
    t14_cols = ["pc5_1e-2", "pc5_1e-3", "pc6_1e-2", "pc6_1e-3"]
    for row_doc, f_key in zip(t14_rows[1:], t14_keys, strict=True):
        for col_idx, c_key in enumerate(t14_cols, start=1):
            exp14 = data["table_14_edge_zones_hours"][f_key][c_key]
            assert abs(_clean_num(row_doc[col_idx]) - exp14) < 1e-9

    # Table 16: Analytic edge error
    t16_rows = next(
        rows for rows in all_tables if len(rows) == 3 and rows[0][0].startswith("Расстояние")
    )
    t16_dists = data["table_16_analytic_edge_error"]["distances_hours"]
    for col_idx, dist_h in enumerate(t16_dists, start=1):
        assert math.isclose(_clean_num(t16_rows[0][col_idx]), dist_h, abs_tol=1e-9)
        assert math.isclose(
            _clean_num(t16_rows[1][col_idx]),
            data["table_16_analytic_edge_error"]["without_reflection"][col_idx - 1],
            rel_tol=1e-9,
        )
        assert math.isclose(
            _clean_num(t16_rows[2][col_idx]),
            data["table_16_analytic_edge_error"]["with_reflection"][col_idx - 1],
            rel_tol=1e-9,
        )

    # Table 17: Filtered effective n (MC / Formula 48)
    t17_rows = next(
        rows for rows in all_tables if len(rows) == 6 and rows[0][0] == "Полоса, окно"
    )
    t17_keys = ["Pc5_18h", "Pc5_8h", "Pc5_4h", "Pc6_18h", "Pc6_8h"]
    t17_cols = ["fft_flat", "fft_red", "sos3_flat", "sos3_red"]
    for row_doc, k17 in zip(t17_rows[1:], t17_keys, strict=True):
        for col_idx, c17 in enumerate(t17_cols, start=1):
            mc_v, f48_v = _split_slash_pair(row_doc[col_idx])
            assert [int(mc_v), int(f48_v)] == data["table_17_filtered_effective_n"][k17][c17]

    # Table 18: Synthetic verification signals summary
    t18_rows = next(rows for rows in all_tables if len(rows) == 7 and rows[0][0] == "Случай")
    assert [r[0] for r in t18_rows[1:]] == ["0", "A", "B", "C", "D", "E"]
    assert len(data["table_18_synthetic_signals"]) == 6

    # Table 19: Expected 2D results
    t19_rows = next(
        rows for rows in all_tables if len(rows) == 11 and rows[0][0] == "Величина"
    )
    case_keys = ["0", "A", "B", "C_xy"]
    metric_map = {
        0: ("Q/I", "Q_over_I"),
        1: ("U/I", "U_over_I"),
        2: ("V/I", "V_over_I"),
        3: ("p", "p"),
        4: ("ψ", "psi_deg"),
        5: ("χ", "chi_deg"),
        6: ("ε", "epsilon"),
        7: ("γ", "gamma2"),
    }
    for row_idx, (prefix, field) in metric_map.items():
        row_doc = t19_rows[row_idx + 1]
        assert row_doc[0].startswith(prefix)
        for col_idx, ckey in enumerate(case_keys, start=1):
            expected_val = data["table_19_expected_2d"]["cases"][ckey][field]
            if expected_val is None:
                assert "не определён" in row_doc[col_idx]
            else:
                assert abs(_clean_num(row_doc[col_idx]) - expected_val) < 1e-9

    # Case C 3D paragraph text verification
    c3d = data["case_c_expected_3d"]
    para_texts = [
        "".join(
            node.text
            for node in p._element.iter()
            if node.tag.split("}")[-1] == "t" and node.text
        )
        for p in doc.paragraphs
    ]
    c3d_para = next(txt for txt in para_texts if "418.84" in txt)
    nums_c3d = [
        float(tok.replace("\u2212", "-").replace("−", "-"))
        for tok in re.findall(r"[-−\u2212+]?\d+(?:\.\d+)?", c3d_para)
    ]
    for expected_num in [
        c3d["tr_S"],
        *c3d["n_v"],
        *c3d["n_v_over_tr_S"],
        c3d["theta_deg"],
        c3d["phi_n_deg"],
        c3d["P"],
        *c3d["eigenvalues_re_S"],
        c3d["b_over_a"],
        c3d["s"],
    ]:
        assert any(math.isclose(n, expected_num, abs_tol=1e-9) for n in nums_c3d), (
            f"Missing Case C 3D value {expected_num} in paragraph: {c3d_para}"
        )

    # Table 20: Case D
    t20_rows = next(
        rows
        for rows in all_tables
        if len(rows) == 9 and rows[0][0] == "Режим" and rows[0][1] == "A300"
    )
    for row_doc, mode_fix in zip(t20_rows[1:], data["table_20_case_d"]["modes"], strict=True):
        assert abs(_clean_num(row_doc[1]) - mode_fix["A_300"]) < 1e-9
        assert abs(_clean_num(row_doc[2]) - mode_fix["A_450"]) < 1e-9
        assert abs(_clean_num(row_doc[3]) - mode_fix["p"]) < 1e-9
        assert abs(_clean_num(row_doc[4]) - mode_fix["V_over_I"]) < 1e-9
        assert abs(_clean_num(row_doc[5]) - mode_fix["chi_deg"]) < 1e-9

    # Table 21: Case E
    t21_rows = next(
        rows
        for rows in all_tables
        if len(rows) == 10 and rows[0][0] == "Режим" and rows[0][1] == "p"
    )
    expected_rows = [
        data["table_21_case_e"]["expected"]["Pc5"],
        data["table_21_case_e"]["expected"]["Pc6"],
        *data["table_21_case_e"]["modes"],
    ]
    for row_doc, item_fix in zip(t21_rows[1:], expected_rows, strict=True):
        assert abs(_clean_num(row_doc[1]) - item_fix["p"]) < 1e-9
        assert abs(_clean_num(row_doc[2]) - item_fix["V_over_I"]) < 1e-9
        assert abs(_clean_num(row_doc[3]) - item_fix["chi_deg"]) < 1e-9
        assert abs(_clean_num(row_doc[4]) - item_fix["psi_deg"]) < 1e-9
