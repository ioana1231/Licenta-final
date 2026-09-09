"""Teste unitare minimale pentru functiile de preprocesare.

Important:
- build_features_v4_final_audit.py ruleaza pipeline-ul la nivel de modul, deci nu este importat direct.
- Pentru a testa functiile reale, acest fisier extrage din AST doar definitiile functiilor tinta si le executa intr-un namespace controlat.

Rulare:
    python -m pytest tests/test_pipeline_unit.py
"""

from __future__ import annotations

import ast
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILE = ROOT / "build_features_v4_final_audit.py"
TARGET_FUNCTIONS = [
    "normalize_text",
    "slugify_col",
    "make_unique",
    "clean_id",
    "to_num",
    "flag",
    "calculate_bmi_series",
    "is_binary_like",
    "build_cluster_table",
]


def load_functions_from_source() -> dict[str, object]:
    source = SOURCE_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in TARGET_FUNCTIONS]
    future_import = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.Module(body=[future_import, *selected], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "pd": pd,
        "np": np,
        "re": re,
        "unicodedata": unicodedata,
        "MAX_MISSING_CLUSTERING": 0.50,
    }
    exec(compile(module, filename=str(SOURCE_FILE), mode="exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def funcs() -> dict[str, object]:
    return load_functions_from_source()


def test_normalize_text_removes_diacritics_and_spaces(funcs):
    normalize_text = funcs["normalize_text"]
    assert normalize_text("  Țară   clinică  ") == "TARA CLINICA"
    assert normalize_text("Bronho-alveolară") == "BRONHO-ALVEOLARA"
    assert normalize_text(np.nan) == ""


def test_slugify_and_unique_columns(funcs):
    slugify_col = funcs["slugify_col"]
    make_unique = funcs["make_unique"]
    assert slugify_col("FEV1 PRE % / final") == "fev1_pre_final"
    assert slugify_col("  Boli cardiovasculare / HTA  ") == "boli_cardiovasculare_hta"
    assert make_unique(["a", "a", "b", "a"]) == ["a", "a_2", "b", "a_3"]


def test_clean_id(funcs):
    clean_id = funcs["clean_id"]
    assert clean_id("123.0") == "123"
    assert clean_id("AB-12 34") == "1234"
    assert clean_id("") is None
    assert clean_id(np.nan) is None


def test_to_num_and_flag(funcs):
    to_num = funcs["to_num"]
    flag = funcs["flag"]
    s = pd.Series(["1,5", " 2 ", "abc", None, ""])
    out = to_num(s)
    assert out.iloc[0] == 1.5
    assert out.iloc[1] == 2
    assert pd.isna(out.iloc[2])
    assert pd.isna(out.iloc[3])
    assert pd.isna(out.iloc[4])

    flags = flag(pd.Series(["1", "2", "0", "-1", None]))
    assert flags.tolist() == [1, 1, 0, 0, 0]



def test_calculate_bmi_series_from_weight_and_height(funcs):
    calculate_bmi_series = funcs["calculate_bmi_series"]
    weight = pd.Series([60, 70, 80, 55, 10, 60])
    height = pd.Series([165, 1.75, 0, 170, 160, 900])
    bmi = calculate_bmi_series(weight, height)

    assert pytest.approx(bmi.iloc[0], rel=1e-3) == 22.04
    assert pytest.approx(bmi.iloc[1], rel=1e-3) == 22.86
    assert pd.isna(bmi.iloc[2])
    assert pytest.approx(bmi.iloc[3], rel=1e-3) == 19.03
    assert pd.isna(bmi.iloc[4])
    assert pd.isna(bmi.iloc[5])

def test_is_binary_like(funcs):
    is_binary_like = funcs["is_binary_like"]
    assert is_binary_like(pd.Series([0, 1, 1, np.nan])) is True
    assert is_binary_like(pd.Series([0, 2, 1])) is False
    assert is_binary_like(pd.Series([np.nan, np.nan])) is False


def test_build_cluster_table_imputation_missing_indicators_and_scaling(funcs):
    build_cluster_table = funcs["build_cluster_table"]

    source = pd.DataFrame({
        "id_pacient": [1, 2, 3, 4, 5],
        "sex": ["F", "M", "F", "M", np.nan],
        "statut_fumator": ["NEFUMATOR", "FUMATOR", "EX_FUMATOR", "NECUNOSCUT", "NEFUMATOR"],
        "varsta": [50, 60, np.nan, 80, np.nan],
        "imc": [20.0, np.nan, 25.0, 30.0, np.nan],
        "como_cardio": [1, 0, np.nan, 1, 0],
    })
    table, raw_table, report = build_cluster_table(
        source,
        cols=["sex", "statut_fumator", "varsta", "imc", "como_cardio"],
        dataset_name="unit_test",
    )

    assert "id_pacient" in table.columns
    assert not table.drop(columns=["id_pacient"]).isna().any().any()
    assert "varsta_missing" in table.columns
    assert "imc_missing" in table.columns
    assert set(table["como_cardio"].unique()).issubset({0, 1})
    assert abs(table["varsta"].mean()) < 1e-9
    assert pytest.approx(table["varsta"].std(ddof=0), rel=1e-6) == 1.0
    assert "sex_F" in table.columns
    assert "statut_fumator_NEFUMATOR" in table.columns
    assert len(raw_table) == len(source)
    assert {"feature", "missing_rate", "status", "imputation", "scaled"}.issubset(report.columns)


def test_default_db_path_prefers_project_directory():
    source = SOURCE_FILE.read_text(encoding="utf-8")
    assert 'CALE_DB = SCRIPT_DIR / "baza_date_licenta.db"' in source
    assert '/home/ioana/licenta/baza_date_licenta.db' not in source
    assert 'directorul proiectului curent' in source.lower() or 'projectului curent' in source.lower()
