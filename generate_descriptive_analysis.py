#!/usr/bin/env python3
"""
generate_descriptive_analysis.py - modul suplimentar pentru statistica descriptiva,
validarea exploratorie a clusterelor si raportarea structurii SQLite.

Rol:
- citeste tabela features_ai din baza SQLite generata de pipeline;
- foloseste, daca exista, tabela ai_cluster_labels generata de train_ai_final.py;
- genereaza tabele CSV si grafice pentru:
  * distributia varstei, total si pe sexe;
  * distributia IMC pe categorii, total si pe sexe;
  * distributia comorbiditatilor, inclusiv pe grupe de varsta;
  * comparatia comorbiditatilor cu scorurile FACED/BSI;
  * frecventa simptomelor;
  * comparatia FEV1/FVC dupa sex, varsta, colonizare Pseudomonas si exacerbari;
  * matricea de covarianta pentru variabilele numerice cheie;
  * distributia scorurilor de severitate FACED/BSI;
  * distributia etiologiilor;
  * teste Kruskal-Wallis / chi-patrat intre clustere;
  * Tabel 6 extins cu medie ± SD, mediana [IQR] si p-value;
  * boxplot-uri si heatmap al profilului clusterelor;
  * schema fizica a tabelelor SQLite si o diagrama logica simplificata.

Rulare uzuala:
    python3 generate_descriptive_analysis.py --db baza_date_licenta.db

Dupa ce rulezi clusteringul:
    python3 train_ai_final.py --db baza_date_licenta.db --force-k 3
    python3 generate_descriptive_analysis.py --db baza_date_licenta.db

Nota metodologica:
- Rezultatele sunt descriptive/exploratorii, nu diagnostice medicale.
- Pentru comparatii cu 2 grupuri se foloseste Mann-Whitney U.
- Pentru comparatii cu peste 2 grupuri se foloseste Kruskal-Wallis.
- Pentru variabile categorice pe clustere se foloseste chi-patrat.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    from scipy.stats import chi2_contingency, kruskal, mannwhitneyu
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# =========================
# CONFIGURARE
# =========================

CONTINUOUS_DESCRIPTIVE = [
    ("varsta", "Vârstă"),
    ("imc", "IMC"),
    ("mmrc_max_final", "mMRC maxim final"),
    ("exacerbari_an_final", "Exacerbări/an"),
    ("fev1_min_final", "FEV1 minim final"),
    ("fvc_min_final", "FVC minim final"),
    ("ipb_min_final", "IPB minim final"),
    ("faced_score_partial", "FACED parțial"),
    ("bsi_score_partial", "BSI parțial"),
]

TABLE6_VARIABLES = [
    ("varsta", "Vârstă medie"),
    ("imc", "IMC mediu"),
    ("mmrc_max_final", "mMRC maxim final"),
    ("exacerbari_an_final", "Exacerbări/an"),
    ("fev1_min_final", "FEV1 minim final"),
    ("fvc_min_final", "FVC minim final"),
    ("pseudomonas_final", "Pseudomonas final"),
    ("hrct_lobi_sever", "HRCT lobi sever"),
    ("hrct_bilateral", "HRCT bilateral"),
    ("hrct_tip_chistic", "HRCT tip chistic"),
    ("trat_old", "Tratament OLD"),
    ("trat_terapie_tripla", "Terapie triplă"),
    ("faced_score_partial", "FACED parțial"),
    ("bsi_score_partial", "BSI parțial"),
]

SYMPTOM_COLUMNS = [
    ("tuse_cronica", "Tuse cronică"),
    ("expectoratie_cronica", "Expectorație cronică"),
    ("hemoptizie_vreodata", "Hemoptizie vreodată"),
    ("raluri_bronsice", "Raluri bronșice"),
    ("raluri_bronhoalveolare", "Raluri bronhoalveolare"),
]

COMORBIDITY_COLUMNS = [
    ("como_pulmonare", "Pulmonare"),
    ("como_cardio", "Cardiovasculare"),
    ("como_orl", "ORL"),
    ("como_neurologice", "Neurologice"),
    ("como_psihiatrice", "Psihiatrice"),
    ("como_diabet", "Diabet"),
    ("como_gastro", "Gastroenterologice"),
    ("como_renal", "Renale"),
    ("como_neoplazii", "Neoplazii"),
    ("como_autoimuna", "Autoimune"),
]

ETIOLOGY_COLUMNS = [
    ("etio_idiopatica", "Idiopatică"),
    ("etio_postinfectioasa", "Postinfecțioasă"),
    ("etio_imunodeficienta", "Imunodeficiență"),
    ("etio_autoimuna", "Autoimună"),
    ("etio_congenitala_non_fc", "Congenitală non-FC"),
    ("etio_obstructiva", "Obstructivă"),
]

CATEGORICAL_CLUSTER_TESTS = [
    ("sex", "Sex"),
    ("pseudomonas_final", "Pseudomonas final"),
    ("hrct_lobi_sever", "HRCT lobi sever"),
    ("hrct_bilateral", "HRCT bilateral"),
    ("hrct_tip_chistic", "HRCT tip chistic"),
    ("trat_old", "Tratament OLD"),
    ("trat_terapie_tripla", "Terapie triplă"),
    ("como_cardio", "Comorbidități cardiovasculare"),
    ("como_diabet", "Diabet"),
    ("como_pulmonare", "Comorbidități pulmonare"),
]


# =========================
# UTILITARE
# =========================

def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Genereaza statistici descriptive si validari exploratorii pentru licenta.")
    parser.add_argument("--db", type=Path, default=base_dir / "baza_date_licenta.db", help="Calea catre baza SQLite")
    parser.add_argument("--features-table", default="features_ai", help="Tabela master descriptiva")
    parser.add_argument("--labels-table", default="ai_cluster_labels", help="Tabela cu etichete de cluster")
    parser.add_argument("--output-dir", type=Path, default=base_dir / "outputs_descriptive_analysis", help="Folder output")
    parser.add_argument("--write-sqlite", action="store_true", help="Salveaza tabelele generate si in SQLite")
    return parser.parse_args()


def ensure_dirs(output_dir: Path) -> tuple[Path, Path]:
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, figures_dir


def read_sql_table(conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    try:
        return pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
    except Exception as exc:
        available = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn)["name"].tolist()
        raise RuntimeError(f"Nu pot citi tabela {table_name}. Tabele disponibile: {available}") from exc


def existing_table(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)).fetchone()
    return row is not None


def save_table(df: pd.DataFrame, path: Path, conn: sqlite3.Connection | None = None, sqlite_name: str | None = None) -> None:
    df.to_csv(path, index=False)
    if conn is not None and sqlite_name is not None:
        df.to_sql(sqlite_name, conn, if_exists="replace", index=False)


def present_columns(df: pd.DataFrame, candidates: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(c, label) for c, label in candidates if c in df.columns]


def format_p(p: float | None) -> str:
    if p is None or pd.isna(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"


def round_or_blank(value: float | int | None, digits: int = 3) -> float | str:
    if value is None or pd.isna(value):
        return ""
    return round(float(value), digits)


def value_counts_table(series: pd.Series, label_col: str, total_n: int | None = None) -> pd.DataFrame:
    total_n = total_n if total_n is not None else len(series)
    counts = series.value_counts(dropna=False).reset_index()
    counts.columns = [label_col, "n"]
    counts["%"] = (counts["n"] / total_n * 100).round(2)
    return counts


def frequency_for_flags(df: pd.DataFrame, cols: list[tuple[str, str]], label_name: str) -> pd.DataFrame:
    rows = []
    total = len(df)
    for col, label in cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").fillna(0)
        n = int((s == 1).sum())
        rows.append({label_name: label, "variabila": col, "n": n, "%": round(n / total * 100, 2)})
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def continuous_summary(df: pd.DataFrame, cols: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    total = len(df)
    for col, label in cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        valid = s.dropna()
        rows.append({
            "variabila": col,
            "denumire": label,
            "n_valid": int(valid.shape[0]),
            "missing_n": int(total - valid.shape[0]),
            "missing_%": round((total - valid.shape[0]) / total * 100, 2),
            "mean": round_or_blank(valid.mean()),
            "sd": round_or_blank(valid.std(ddof=1)),
            "median": round_or_blank(valid.median()),
            "q1": round_or_blank(valid.quantile(0.25)),
            "q3": round_or_blank(valid.quantile(0.75)),
            "min": round_or_blank(valid.min()),
            "max": round_or_blank(valid.max()),
        })
    return pd.DataFrame(rows)


def classify_bmi(value: float) -> str:
    if pd.isna(value):
        return "IMC lipsă"
    if value < 18.5:
        return "Subponderal"
    if value < 25:
        return "Normoponderal"
    if value < 30:
        return "Supraponderal"
    if value < 35:
        return "Obezitate grad I"
    if value < 40:
        return "Obezitate grad II"
    return "Obezitate grad III"


def classify_age(value: float) -> str:
    if pd.isna(value):
        return "Vârstă lipsă"
    if value < 50:
        return "<50"
    if value < 60:
        return "50-59"
    if value < 70:
        return "60-69"
    if value < 80:
        return "70-79"
    return ">=80"


def classify_exacerbari(value: float) -> str:
    if pd.isna(value):
        return "Exacerbări lipsă"
    if value <= 0:
        return "0"
    if value < 3:
        return "1-2"
    return ">=3"


def classify_faced(value: float) -> str:
    if pd.isna(value):
        return "Necalculabil complet"
    if value <= 2:
        return "Ușoară"
    if value <= 4:
        return "Moderată"
    return "Severă"


def classify_bsi(value: float) -> str:
    if pd.isna(value):
        return "Necalculabil complet"
    if value <= 4:
        return "Ușoară"
    if value <= 8:
        return "Moderată"
    return "Severă"


def mann_whitney_or_blank(a: pd.Series, b: pd.Series) -> tuple[str, float | None]:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if not SCIPY_AVAILABLE or len(a) == 0 or len(b) == 0:
        return "Mann-Whitney U", None
    try:
        return "Mann-Whitney U", float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:
        return "Mann-Whitney U", None


def kruskal_or_blank(groups: list[pd.Series]) -> tuple[str, float | None]:
    clean = [pd.to_numeric(g, errors="coerce").dropna() for g in groups]
    clean = [g for g in clean if len(g) > 0]
    if not SCIPY_AVAILABLE or len(clean) < 2:
        return "Kruskal-Wallis", None
    try:
        return "Kruskal-Wallis", float(kruskal(*clean).pvalue)
    except Exception:
        return "Kruskal-Wallis", None


def chi_square_or_blank(table: pd.DataFrame) -> tuple[str, float | None]:
    if not SCIPY_AVAILABLE or table.shape[0] < 2 or table.shape[1] < 2:
        return "Chi-pătrat", None
    try:
        return "Chi-pătrat", float(chi2_contingency(table)[1])
    except Exception:
        return "Chi-pătrat", None


def grouped_continuous_comparison(df: pd.DataFrame, group_col: str, group_label: str, outcomes: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    if group_col not in df.columns:
        return pd.DataFrame()
    for outcome, outcome_label in outcomes:
        if outcome not in df.columns:
            continue
        temp = df[[group_col, outcome]].copy()
        temp[outcome] = pd.to_numeric(temp[outcome], errors="coerce")
        temp = temp.dropna(subset=[group_col, outcome])
        group_names = sorted(temp[group_col].dropna().unique().tolist(), key=lambda x: str(x))
        groups = [temp.loc[temp[group_col].eq(g), outcome] for g in group_names]
        if len(groups) == 2:
            test, p = mann_whitney_or_blank(groups[0], groups[1])
        else:
            test, p = kruskal_or_blank(groups)
        row = {"grupare": group_label, "variabila_grupare": group_col, "parametru": outcome_label, "variabila": outcome, "test": test, "p_value": p, "p_formatat": format_p(p)}
        for g, vals in zip(group_names, groups):
            vals = vals.dropna()
            row[f"{g}_n"] = int(len(vals))
            row[f"{g}_mean_sd"] = f"{vals.mean():.3f} ± {vals.std(ddof=1):.3f}" if len(vals) > 1 else ""
            row[f"{g}_median_iqr"] = f"{vals.median():.3f} [{vals.quantile(0.25):.3f}-{vals.quantile(0.75):.3f}]" if len(vals) > 0 else ""
        rows.append(row)
    return pd.DataFrame(rows)


def plot_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    if df.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.bar(df[x_col].astype(str), df[y_col])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_boxplot(df: pd.DataFrame, value_col: str, group_col: str, title: str, ylabel: str, path: Path) -> None:
    if value_col not in df.columns or group_col not in df.columns:
        return
    temp = df[[value_col, group_col]].copy()
    temp[value_col] = pd.to_numeric(temp[value_col], errors="coerce")
    temp = temp.dropna(subset=[value_col, group_col])
    if temp.empty:
        return
    group_names = sorted(temp[group_col].unique().tolist(), key=lambda x: str(x))
    data = [temp.loc[temp[group_col].eq(g), value_col].dropna().values for g in group_names]
    if not any(len(x) > 0 for x in data):
        return
    plt.figure(figsize=(7, 5))
    group_labels = [str(g) for g in group_names]
    try:
        plt.boxplot(data, tick_labels=group_labels)
    except TypeError:
        # Compatibilitate cu versiuni mai vechi de Matplotlib.
        plt.boxplot(data, labels=group_labels)
    plt.xlabel(group_col)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# =========================
# GENERARE RAPOARTE
# =========================

def plot_grouped_bar(df: pd.DataFrame, x_col: str, group_col: str, y_col: str, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    if df.empty or x_col not in df.columns or group_col not in df.columns or y_col not in df.columns:
        return
    pivot = df.pivot_table(index=x_col, columns=group_col, values=y_col, aggfunc="sum", fill_value=0)
    pivot.plot(kind="bar", figsize=(9, 5))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_matrix_heatmap(matrix: pd.DataFrame, title: str, path: Path) -> None:
    if matrix.empty:
        return
    fig_width = max(7, min(14, 0.75 * len(matrix.columns) + 3))
    fig_height = max(5, min(12, 0.65 * len(matrix.index) + 2))
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(matrix.to_numpy(dtype=float), aspect="auto")
    plt.colorbar(label="Valoare")
    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    plt.yticks(range(len(matrix.index)), matrix.index)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_surface_3d(matrix: pd.DataFrame, title: str, path: Path, zlabel: str) -> None:
    if matrix.empty:
        return
    x_values = np.arange(len(matrix.columns))
    y_values = np.arange(len(matrix.index))
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    z_values = matrix.to_numpy(dtype=float)

    figure = go.Figure(
        data=[
            go.Surface(
                x=list(matrix.columns),
                y=list(matrix.index),
                z=z_values,
                colorscale="Viridis",
                colorbar={"title": zlabel},
            )
        ]
    )
    figure.update_layout(
        title=title,
        scene={
            "xaxis_title": "Grupă de vârstă",
            "yaxis_title": "Comorbiditate",
            "zaxis_title": zlabel,
        },
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )
    figure.write_html(path, include_plotlyjs="cdn", full_html=True)


def covariance_matrix(df: pd.DataFrame, columns: list[tuple[str, str]]) -> pd.DataFrame:
    available = present_columns(df, columns)
    if len(available) < 2:
        return pd.DataFrame()
    source_cols = [col for col, _ in available]
    labels = {col: label for col, label in available}
    numeric = df[source_cols].apply(pd.to_numeric, errors="coerce").rename(columns=labels)
    return numeric.cov(min_periods=5).round(3)


def save_covariance_report(
    df: pd.DataFrame,
    columns: list[tuple[str, str]],
    tables_dir: Path,
    figures_dir: Path,
    file_stem: str,
    title: str,
    conn_out: sqlite3.Connection | None,
    sqlite_name: str,
) -> pd.DataFrame:
    matrix = covariance_matrix(df, columns)
    if matrix.empty:
        return matrix
    matrix_out = matrix.reset_index().rename(columns={"index": "variabila"})
    save_table(matrix_out, tables_dir / f"{file_stem}.csv", conn_out, sqlite_name)
    plot_matrix_heatmap(matrix, title, figures_dir / f"{file_stem}_heatmap.png")
    return matrix_out


def write_recommended_outputs(output_dir: Path) -> None:
    """Copiaza automat output-urile finale intr-un folder stabil folosit de dashboard."""
    import shutil

    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_rec = output_dir / "tables_recommended"
    figures_rec = output_dir / "figures_recommended"
    tables_rec.mkdir(parents=True, exist_ok=True)
    figures_rec.mkdir(parents=True, exist_ok=True)

    for src in tables_dir.glob("*.csv"):
        shutil.copy2(src, tables_rec / src.name)
    for pattern in ("*.png", "*.html"):
        for src in figures_dir.glob(pattern):
            shutil.copy2(src, figures_rec / src.name)


def generate_cohort_reports(features: pd.DataFrame, tables_dir: Path, figures_dir: Path, conn_out: sqlite3.Connection | None) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    total = len(features)

    continuous = continuous_summary(features, CONTINUOUS_DESCRIPTIVE)
    save_table(continuous, tables_dir / "cohort_descriptive_continuous.csv", conn_out, "desc_cohort_descriptive_continuous")
    outputs["cohort_descriptive_continuous"] = continuous

    selected_missing = continuous[["variabila", "denumire", "n_valid", "missing_n", "missing_%"]].copy()
    save_table(selected_missing, tables_dir / "missing_rates_selected_variables.csv", conn_out, "desc_missing_rates_selected_variables")
    outputs["missing_rates_selected_variables"] = selected_missing

    if "sex" in features.columns:
        sex_dist = value_counts_table(features["sex"].fillna("Necunoscut"), "sex", total)
        save_table(sex_dist, tables_dir / "sex_distribution.csv", conn_out, "desc_sex_distribution")
        outputs["sex_distribution"] = sex_dist

    if "varsta" in features.columns:
        temp_age = features.copy()
        temp_age["grupa_varsta"] = pd.to_numeric(temp_age["varsta"], errors="coerce").apply(classify_age)
        age_groups = value_counts_table(temp_age["grupa_varsta"], "grupa_varsta", total)
        age_order = ["<50", "50-59", "60-69", "70-79", ">=80", "Vârstă lipsă"]
        age_groups["ordine"] = age_groups["grupa_varsta"].map({v: i for i, v in enumerate(age_order)})
        age_groups = age_groups.sort_values("ordine").drop(columns="ordine")
        save_table(age_groups, tables_dir / "age_group_distribution_overall.csv", conn_out, "desc_age_group_distribution_overall")
        outputs["age_group_distribution_overall"] = age_groups
        plot_bar(age_groups, "grupa_varsta", "n", "Distribuția vârstei pe grupe", "Grupă de vârstă", "Număr pacienți", figures_dir / "age_group_distribution_overall.png")

        age_summary = grouped_continuous_comparison(features, "sex", "Sex", [("varsta", "Vârstă")]) if "sex" in features.columns else pd.DataFrame()
        if not age_summary.empty:
            save_table(age_summary, tables_dir / "age_descriptive_by_sex.csv", conn_out, "desc_age_descriptive_by_sex")
            outputs["age_descriptive_by_sex"] = age_summary

        if "sex" in temp_age.columns:
            age_by_sex = temp_age.groupby(["sex", "grupa_varsta"], dropna=False).size().reset_index(name="n")
            sex_totals = temp_age.groupby("sex", dropna=False).size().reset_index(name="total_sex")
            age_by_sex = age_by_sex.merge(sex_totals, on="sex", how="left")
            age_by_sex["%_din_sex"] = (age_by_sex["n"] / age_by_sex["total_sex"] * 100).round(2)
            age_by_sex["ordine"] = age_by_sex["grupa_varsta"].map({v: i for i, v in enumerate(age_order)})
            age_by_sex = age_by_sex.sort_values(["sex", "ordine"]).drop(columns="ordine")
            save_table(age_by_sex, tables_dir / "age_group_distribution_by_sex.csv", conn_out, "desc_age_group_distribution_by_sex")
            outputs["age_group_distribution_by_sex"] = age_by_sex
            plot_grouped_bar(age_by_sex, "grupa_varsta", "sex", "n", "Distribuția vârstei pe sexe", "Grupă de vârstă", "Număr pacienți", figures_dir / "age_group_distribution_by_sex.png")

    if "imc" in features.columns:
        temp = features.copy()

        def classify_imc_with_source(row: pd.Series) -> str:
            value = pd.to_numeric(row.get("imc"), errors="coerce")
            if pd.isna(value):
                return "IMC lipsă"
            return classify_bmi(value)

        temp["categorie_imc"] = temp.apply(classify_imc_with_source, axis=1)
        order = ["Subponderal", "Normoponderal", "Supraponderal", "Obezitate grad I", "Obezitate grad II", "Obezitate grad III", "IMC lipsă"]
        imc_overall = value_counts_table(temp["categorie_imc"], "categorie_imc", total)
        imc_overall["ordine"] = imc_overall["categorie_imc"].map({v: i for i, v in enumerate(order)})
        imc_overall = imc_overall.sort_values("ordine").drop(columns="ordine")
        save_table(imc_overall, tables_dir / "imc_distribution_overall.csv", conn_out, "desc_imc_distribution_overall")
        outputs["imc_distribution_overall"] = imc_overall
        plot_bar(imc_overall, "categorie_imc", "n", "Distribuția IMC pe categorii", "Categorie IMC", "Număr pacienți", figures_dir / "imc_distribution_overall.png")

        if "sex" in temp.columns:
            by_sex = temp.groupby(["sex", "categorie_imc"], dropna=False).size().reset_index(name="n")
            sex_totals = temp.groupby("sex", dropna=False).size().reset_index(name="total_sex")
            by_sex = by_sex.merge(sex_totals, on="sex", how="left")
            by_sex["%_din_sex"] = (by_sex["n"] / by_sex["total_sex"] * 100).round(2)
            by_sex["ordine"] = by_sex["categorie_imc"].map({v: i for i, v in enumerate(order)})
            by_sex = by_sex.sort_values(["sex", "ordine"]).drop(columns="ordine")
            save_table(by_sex, tables_dir / "imc_distribution_by_sex.csv", conn_out, "desc_imc_distribution_by_sex")
            outputs["imc_distribution_by_sex"] = by_sex

    symptoms = frequency_for_flags(features, SYMPTOM_COLUMNS, "simptom")
    if "mmrc_max_final" in features.columns:
        mmrc = pd.to_numeric(features["mmrc_max_final"], errors="coerce")
        n = int((mmrc >= 3).sum())
        symptoms = pd.concat([
            pd.DataFrame([{"simptom": "Dispnee importantă (mMRC ≥ 3)", "variabila": "mmrc_max_final", "n": n, "%": round(n / total * 100, 2)}]),
            symptoms,
        ], ignore_index=True)
    save_table(symptoms, tables_dir / "symptoms_frequency.csv", conn_out, "desc_symptoms_frequency")
    outputs["symptoms_frequency"] = symptoms
    plot_bar(symptoms, "simptom", "n", "Frecvența principalelor simptome", "Simptom", "Număr pacienți", figures_dir / "symptoms_frequency.png")

    comorb = frequency_for_flags(features, COMORBIDITY_COLUMNS, "comorbiditate")
    save_table(comorb, tables_dir / "comorbidity_distribution.csv", conn_out, "desc_comorbidity_distribution")
    outputs["comorbidity_distribution"] = comorb
    plot_bar(comorb, "comorbiditate", "n", "Distribuția comorbidităților", "Comorbiditate", "Număr pacienți", figures_dir / "comorbidity_distribution.png")

    if "varsta" in features.columns:
        temp_com_age = features.copy()
        temp_com_age["grupa_varsta"] = pd.to_numeric(temp_com_age["varsta"], errors="coerce").apply(classify_age)
        rows = []
        age_totals = temp_com_age.groupby("grupa_varsta", dropna=False).size().to_dict()
        for com_col, com_label in present_columns(temp_com_age, COMORBIDITY_COLUMNS):
            flag_col = pd.to_numeric(temp_com_age[com_col], errors="coerce").fillna(0)
            for age_group, group in temp_com_age.assign(_flag=flag_col).groupby("grupa_varsta", dropna=False):
                total_age = int(age_totals.get(age_group, len(group)))
                n = int((group["_flag"] == 1).sum())
                rows.append({
                    "grupa_varsta": age_group,
                    "comorbiditate": com_label,
                    "variabila": com_col,
                    "n": n,
                    "total_grupa_varsta": total_age,
                    "%_din_grupa_varsta": round(n / total_age * 100, 2) if total_age else 0.0,
                })
        comorb_age = pd.DataFrame(rows)
        if not comorb_age.empty:
            save_table(comorb_age, tables_dir / "comorbidity_distribution_by_age_group.csv", conn_out, "desc_comorbidity_distribution_by_age_group")
            outputs["comorbidity_distribution_by_age_group"] = comorb_age
            heat = comorb_age.pivot_table(index="comorbiditate", columns="grupa_varsta", values="%_din_grupa_varsta", aggfunc="mean", fill_value=0)
            age_order = ["<50", "50-59", "60-69", "70-79", ">=80", "Vârstă lipsă"]
            heat = heat[[c for c in age_order if c in heat.columns]]
            plot_matrix_heatmap(heat, "Comorbidități pe grupe de vârstă (%)", figures_dir / "comorbidity_by_age_group_heatmap.png")
            plot_surface_3d(
                heat,
                "Suprafață 3D: comorbidități pe grupe de vârstă",
                figures_dir / "comorbidity_by_age_group_surface_3d.html",
                "% în grupa de vârstă",
            )

    covariance_out = save_covariance_report(
        features,
        CONTINUOUS_DESCRIPTIVE,
        tables_dir,
        figures_dir,
        "covariance_key_variables",
        "Matrice de covarianță - variabile numerice cheie",
        conn_out,
        "desc_covariance_key_variables",
    )
    if not covariance_out.empty:
        available_cov = present_columns(features, CONTINUOUS_DESCRIPTIVE)
        cov_cols = [c for c, _ in available_cov]
        cov_labels = {c: label for c, label in available_cov}
        numeric_cov = features[cov_cols].apply(pd.to_numeric, errors="coerce").rename(columns=cov_labels)
        correlation_out = numeric_cov.corr(min_periods=5).round(3).reset_index().rename(columns={"index": "variabila"})
        save_table(correlation_out, tables_dir / "correlation_key_variables.csv", conn_out, "desc_correlation_key_variables")
        outputs["covariance_key_variables"] = covariance_out

    etiology = frequency_for_flags(features, ETIOLOGY_COLUMNS, "etiologie")
    save_table(etiology, tables_dir / "etiology_distribution.csv", conn_out, "desc_etiology_distribution")
    outputs["etiology_distribution"] = etiology
    plot_bar(etiology, "etiologie", "n", "Distribuția după etiologie", "Etiologie", "Număr pacienți", figures_dir / "etiology_distribution.png")

    if "nr_grupe_etiologie" in features.columns:
        etio_groups = value_counts_table(pd.to_numeric(features["nr_grupe_etiologie"], errors="coerce").fillna(-1).astype(int), "nr_grupe_etiologie", total)
        save_table(etio_groups, tables_dir / "etiology_number_of_groups_distribution.csv", conn_out, "desc_etiology_number_of_groups_distribution")
        outputs["etiology_number_of_groups_distribution"] = etio_groups

    # Severitate FACED / BSI - varianta stricta daca exista, altfel partiala cu mentiune in raport.
    severity_rows = []
    if "faced_score_calculat" in features.columns:
        faced_cat = pd.to_numeric(features["faced_score_calculat"], errors="coerce").apply(classify_faced)
        faced_dist = value_counts_table(faced_cat, "severitate", total)
        faced_dist.insert(0, "scor", "FACED")
        severity_rows.append(faced_dist)
        plot_bar(faced_dist, "severitate", "n", "Distribuția severității FACED", "Severitate", "Număr pacienți", figures_dir / "faced_severity_distribution.png")
        faced_calc = faced_dist[faced_dist["severitate"].ne("Necalculabil complet")].copy()
        if not faced_calc.empty:
            total_calc = int(faced_calc["n"].sum())
            faced_calc["%"] = (faced_calc["n"] / total_calc * 100).round(2)
            save_table(faced_calc, tables_dir / "faced_severity_distribution_calculable_only.csv", conn_out, "desc_faced_severity_distribution_calculable_only")
            outputs["faced_severity_distribution_calculable_only"] = faced_calc
            plot_bar(faced_calc, "severitate", "n", "Distribuția FACED pentru pacienții calculabili", "Severitate", "Număr pacienți", figures_dir / "faced_severity_distribution_calculable_only.png")
    if "bsi_score_calculat" in features.columns:
        bsi_cat = pd.to_numeric(features["bsi_score_calculat"], errors="coerce").apply(classify_bsi)
        bsi_dist = value_counts_table(bsi_cat, "severitate", total)
        bsi_dist.insert(0, "scor", "BSI")
        severity_rows.append(bsi_dist)
        plot_bar(bsi_dist, "severitate", "n", "Distribuția severității BSI", "Severitate", "Număr pacienți", figures_dir / "bsi_severity_distribution.png")
        bsi_calc = bsi_dist[bsi_dist["severitate"].ne("Necalculabil complet")].copy()
        if not bsi_calc.empty:
            total_calc = int(bsi_calc["n"].sum())
            bsi_calc["%"] = (bsi_calc["n"] / total_calc * 100).round(2)
            save_table(bsi_calc, tables_dir / "bsi_severity_distribution_calculable_only.csv", conn_out, "desc_bsi_severity_distribution_calculable_only")
            outputs["bsi_severity_distribution_calculable_only"] = bsi_calc
            plot_bar(bsi_calc, "severitate", "n", "Distribuția BSI pentru pacienții calculabili", "Severitate", "Număr pacienți", figures_dir / "bsi_severity_distribution_calculable_only.png")
    if severity_rows:
        severity = pd.concat(severity_rows, ignore_index=True)
        save_table(severity, tables_dir / "severity_scores_distribution_complete_scores.csv", conn_out, "desc_severity_scores_distribution")
        outputs["severity_scores_distribution_complete_scores"] = severity

    # Comorbiditati vs scoruri FACED/BSI - exploratoriu, nu cauzal.
    score_cols = [("faced_score_partial", "FACED parțial"), ("bsi_score_partial", "BSI parțial")]
    rows = []
    for com_col, com_label in present_columns(features, COMORBIDITY_COLUMNS):
        flag = pd.to_numeric(features[com_col], errors="coerce").fillna(0)
        for score_col, score_label in present_columns(features, score_cols):
            vals = pd.to_numeric(features[score_col], errors="coerce")
            yes = vals[flag == 1].dropna()
            no = vals[flag == 0].dropna()
            test, p = mann_whitney_or_blank(yes, no)
            rows.append({
                "comorbiditate": com_label,
                "variabila_comorbiditate": com_col,
                "scor": score_label,
                "n_cu_comorbiditate": int(len(yes)),
                "medie_cu_comorbiditate": round_or_blank(yes.mean()),
                "mediana_cu_comorbiditate": round_or_blank(yes.median()),
                "n_fara_comorbiditate": int(len(no)),
                "medie_fara_comorbiditate": round_or_blank(no.mean()),
                "mediana_fara_comorbiditate": round_or_blank(no.median()),
                "test": test,
                "p_value": p,
                "p_formatat": format_p(p),
            })
    com_score = pd.DataFrame(rows)
    save_table(com_score, tables_dir / "comorbidity_score_comparison_partial_scores.csv", conn_out, "desc_comorbidity_score_comparison")
    outputs["comorbidity_score_comparison_partial_scores"] = com_score

    # Comparatii FEV1/FVC dupa sex, varsta, colonizare Pseudomonas, exacerbari.
    outcomes = [("fev1_min_final", "FEV1 minim final"), ("fvc_min_final", "FVC minim final")]
    respiratory_outputs = []
    if "sex" in features.columns:
        res = grouped_continuous_comparison(features, "sex", "Sex", outcomes)
        if not res.empty:
            save_table(res, tables_dir / "respiratory_comparison_by_sex.csv", conn_out, "desc_respiratory_comparison_by_sex")
            respiratory_outputs.append(res)
            for out, label in outcomes:
                plot_boxplot(features, out, "sex", f"{label} după sex", label, figures_dir / f"{out}_by_sex_boxplot.png")
    if "varsta" in features.columns:
        temp = features.copy()
        temp["grupa_varsta"] = pd.to_numeric(temp["varsta"], errors="coerce").apply(classify_age)
        res = grouped_continuous_comparison(temp, "grupa_varsta", "Grupă de vârstă", outcomes)
        if not res.empty:
            save_table(res, tables_dir / "respiratory_comparison_by_age_group.csv", conn_out, "desc_respiratory_comparison_by_age_group")
            respiratory_outputs.append(res)
            for out, label in outcomes:
                plot_boxplot(temp, out, "grupa_varsta", f"{label} după grupa de vârstă", label, figures_dir / f"{out}_by_age_group_boxplot.png")
    colonization_col = "bact_colonizare_pseudomonas" if "bact_colonizare_pseudomonas" in features.columns else "pseudomonas_final"
    if colonization_col in features.columns:
        temp = features.copy()
        temp["colonizare_pseudomonas"] = pd.to_numeric(temp[colonization_col], errors="coerce").fillna(0).map({0: "Nu", 1: "Da"})
        res = grouped_continuous_comparison(temp, "colonizare_pseudomonas", "Colonizare Pseudomonas", outcomes)
        if not res.empty:
            save_table(res, tables_dir / "respiratory_comparison_by_colonizare_pseudomonas.csv", conn_out, "desc_respiratory_comparison_by_colonizare_pseudomonas")
            respiratory_outputs.append(res)
            for out, label in outcomes:
                plot_boxplot(temp, out, "colonizare_pseudomonas", f"{label} după colonizarea Pseudomonas", label, figures_dir / f"{out}_by_colonization_boxplot.png")
    if "exacerbari_an_final" in features.columns:
        temp = features.copy()
        temp["grupa_exacerbari"] = pd.to_numeric(temp["exacerbari_an_final"], errors="coerce").apply(classify_exacerbari)
        res = grouped_continuous_comparison(temp, "grupa_exacerbari", "Exacerbări/an", outcomes)
        if not res.empty:
            save_table(res, tables_dir / "respiratory_comparison_by_exacerbari_group.csv", conn_out, "desc_respiratory_comparison_by_exacerbari_group")
            respiratory_outputs.append(res)
            for out, label in outcomes:
                plot_boxplot(temp, out, "grupa_exacerbari", f"{label} după grupa de exacerbări", label, figures_dir / f"{out}_by_exacerbari_boxplot.png")
    if respiratory_outputs:
        outputs["respiratory_comparisons"] = pd.concat(respiratory_outputs, ignore_index=True)

    return outputs


def generate_cluster_reports(data: pd.DataFrame, tables_dir: Path, figures_dir: Path, conn_out: sqlite3.Connection | None) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    if "cluster" not in data.columns:
        return outputs

    cluster_counts = data["cluster"].value_counts().sort_index().reset_index()
    cluster_counts.columns = ["cluster", "n"]
    cluster_counts["%"] = (cluster_counts["n"] / len(data) * 100).round(2)
    save_table(cluster_counts, tables_dir / "cluster_distribution.csv", conn_out, "desc_cluster_distribution")
    outputs["cluster_distribution"] = cluster_counts

    # Covarianța este raportată atât pe cohorta completă, cât și separat în fiecare cluster.
    covariance_overall = save_covariance_report(
        data,
        CONTINUOUS_DESCRIPTIVE,
        tables_dir,
        figures_dir,
        "covariance_all_variables_overall",
        "Matrice de covarianță - toate variabilele numerice, populația totală",
        conn_out,
        "desc_covariance_all_variables_overall",
    )
    if not covariance_overall.empty:
        outputs["covariance_all_variables_overall"] = covariance_overall
    for cluster_id, cluster_data in data.groupby("cluster", sort=True):
        covariance_cluster = save_covariance_report(
            cluster_data,
            CONTINUOUS_DESCRIPTIVE,
            tables_dir,
            figures_dir,
            f"covariance_all_variables_cluster_{cluster_id}",
            f"Matrice de covarianță - toate variabilele numerice, cluster {cluster_id}",
            conn_out,
            f"desc_covariance_all_variables_cluster_{cluster_id}",
        )
        if not covariance_cluster.empty:
            outputs[f"covariance_all_variables_cluster_{cluster_id}"] = covariance_cluster

    # Tabel 6 extins.
    table6_rows = []
    for col, label in present_columns(data, TABLE6_VARIABLES):
        row = {"variabila": label, "coloana": col}
        groups = []
        for cluster_id in sorted(data["cluster"].dropna().unique()):
            vals = pd.to_numeric(data.loc[data["cluster"].eq(cluster_id), col], errors="coerce").dropna()
            groups.append(vals)
            row[f"cluster_{cluster_id}_n"] = int(len(vals))
            row[f"cluster_{cluster_id}_medie_sd"] = f"{vals.mean():.3f} ± {vals.std(ddof=1):.3f}" if len(vals) > 1 else ""
            row[f"cluster_{cluster_id}_mediana_iqr"] = f"{vals.median():.3f} [{vals.quantile(0.25):.3f}-{vals.quantile(0.75):.3f}]" if len(vals) > 0 else ""
        test, p = kruskal_or_blank(groups)
        row["test"] = test
        row["p_value"] = p
        row["p_formatat"] = format_p(p)
        table6_rows.append(row)
    table6 = pd.DataFrame(table6_rows)
    save_table(table6, tables_dir / "table6_cluster_profile_with_variation_and_pvalues.csv", conn_out, "desc_table6_cluster_profile")
    outputs["table6_cluster_profile_with_variation_and_pvalues"] = table6

    # Teste continue pe clustere.
    cont_rows = []
    for col, label in present_columns(data, CONTINUOUS_DESCRIPTIVE):
        groups = [pd.to_numeric(group[col], errors="coerce").dropna() for _, group in data.groupby("cluster")]
        test, p = kruskal_or_blank(groups)
        cont_rows.append({"variabila": col, "denumire": label, "test": test, "p_value": p, "p_formatat": format_p(p)})
    cont_tests = pd.DataFrame(cont_rows)
    save_table(cont_tests, tables_dir / "stat_tests_continuous_by_cluster.csv", conn_out, "desc_stat_tests_continuous_by_cluster")
    outputs["stat_tests_continuous_by_cluster"] = cont_tests

    # Teste categorice pe clustere.
    cat_rows = []
    for col, label in present_columns(data, CATEGORICAL_CLUSTER_TESTS):
        temp = data[["cluster", col]].dropna().copy()
        if temp.empty:
            continue
        table = pd.crosstab(temp["cluster"], temp[col])
        test, p = chi_square_or_blank(table)
        cat_rows.append({"variabila": col, "denumire": label, "test": test, "p_value": p, "p_formatat": format_p(p), "dimensiune_tabel": f"{table.shape[0]}x{table.shape[1]}"})
    cat_tests = pd.DataFrame(cat_rows)
    save_table(cat_tests, tables_dir / "stat_tests_categorical_chi_square_by_cluster.csv", conn_out, "desc_stat_tests_categorical_by_cluster")
    outputs["stat_tests_categorical_chi_square_by_cluster"] = cat_tests

    # Boxplot-uri pe cluster pentru toate variabilele numerice clinice disponibile.
    for col, label in present_columns(data, CONTINUOUS_DESCRIPTIVE):
        plot_boxplot(data, col, "cluster", f"{label} pe clustere", label, figures_dir / f"boxplot_{col}_by_cluster.png")

    # Heatmap profil standardizat pe clustere.
    heat_cols = [col for col, _ in present_columns(data, TABLE6_VARIABLES)]
    numeric = data[["cluster"] + heat_cols].copy()
    for c in heat_cols:
        numeric[c] = pd.to_numeric(numeric[c], errors="coerce")
    cluster_means = numeric.groupby("cluster")[heat_cols].mean()
    std = numeric[heat_cols].std(ddof=0).replace(0, np.nan)
    mean = numeric[heat_cols].mean()
    standardized = ((cluster_means - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0)
    standardized_out = standardized.reset_index()
    save_table(standardized_out, tables_dir / "heatmap_standardized_cluster_profile_values.csv", conn_out, "desc_heatmap_cluster_profile_values")
    outputs["heatmap_standardized_cluster_profile_values"] = standardized_out

    if not standardized.empty:
        plt.figure(figsize=(11, 4.8))
        plt.imshow(standardized.values, aspect="auto")
        plt.colorbar(label="Diferență standardizată față de media globală")
        plt.yticks(range(len(standardized.index)), [f"Cluster {c}" for c in standardized.index])
        plt.xticks(range(len(standardized.columns)), standardized.columns, rotation=45, ha="right")
        plt.title("Heatmap al profilului standardizat al clusterelor")
        plt.tight_layout()
        plt.savefig(figures_dir / "heatmap_cluster_profile.png", dpi=200)
        plt.close()

    return outputs


def generate_sqlite_schema_reports(conn: sqlite3.Connection, tables_dir: Path, figures_dir: Path, conn_out: sqlite3.Connection | None) -> None:
    table_names = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn)["name"].tolist()
    detail_rows = []
    summary_rows = []
    for table in table_names:
        info = pd.read_sql_query(f'PRAGMA table_info("{table}")', conn)
        summary_rows.append({
            "tabel": table,
            "nr_coloane": int(len(info)),
            "contine_id_pacient": int("id_pacient" in info["name"].tolist()),
            "cheie_logica": "id_pacient" if "id_pacient" in info["name"].tolist() else "",
            "observatie": "Cheile primare/străine nu sunt declarate formal; id_pacient este folosit ca legătură logică." if "id_pacient" in info["name"].tolist() else "",
        })
        for _, row in info.iterrows():
            detail_rows.append({
                "tabel": table,
                "coloana": row["name"],
                "tip_sqlite": row["type"],
                "not_null": int(row["notnull"]),
                "pk_declarat": int(row["pk"]),
                "rol_logic": "cheie logica de legatura" if row["name"] == "id_pacient" else "atribut",
            })
    summary = pd.DataFrame(summary_rows)
    detail = pd.DataFrame(detail_rows)
    save_table(summary, tables_dir / "sqlite_schema_summary.csv", conn_out, "desc_sqlite_schema_summary")
    save_table(detail, tables_dir / "sqlite_schema_detail.csv", conn_out, "desc_sqlite_schema_detail")

    # Diagrama logica simplificata.
    logical_tables = [
        ("*_clean", "Tabele anonimizate\nprovenite din foile Excel"),
        ("*_features", "Caracteristici\npe domenii clinice"),
        ("features_ai", "Tabel master\n1 rând / pacient"),
        ("features_clustering", "Input numeric\npentru KMeans"),
        ("ai_cluster_labels", "Etichete\nde cluster"),
        ("rapoarte/grafice", "Statistici descriptive\nși vizualizări"),
    ]
    positions = [(0.08, 0.65), (0.32, 0.65), (0.56, 0.65), (0.80, 0.65), (0.56, 0.25), (0.80, 0.25)]
    plt.figure(figsize=(11, 5))
    ax = plt.gca()
    ax.axis("off")
    for (name, desc), (x, y) in zip(logical_tables, positions):
        ax.text(x, y, f"{name}\n{desc}", ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round", alpha=0.15))
    arrows = [(positions[0], positions[1]), (positions[1], positions[2]), (positions[2], positions[3]), (positions[3], positions[4]), (positions[2], positions[5]), (positions[4], positions[5])]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2 - 0.08 if x2 > x1 else x2, y2), xytext=(x1 + 0.08 if x2 > x1 else x1, y1), arrowprops=dict(arrowstyle="->"))
    ax.set_title("Diagramă logică simplificată a tabelelor și fluxului de date")
    plt.tight_layout()
    plt.savefig(figures_dir / "sqlite_logical_er_diagram.png", dpi=200)
    plt.close()


def write_markdown_report(output_dir: Path, generated: dict[str, pd.DataFrame], has_clusters: bool) -> None:
    lines = [
        "# Raport statistici descriptive și validare exploratorie",
        "",
        "Acest raport este generat automat pe baza tabelei `features_ai` și, dacă există, a tabelei `ai_cluster_labels`.",
        "Rezultatele au rol descriptiv și exploratoriu. Ele nu reprezintă diagnostic medical și nu stabilesc relații cauzale.",
        "",
        "## Fișiere generate",
        "",
        "Tabelele CSV sunt salvate în folderul `tables/`, iar figurile în folderul `figures/`.",
        "",
    ]
    for name, df in generated.items():
        lines.append(f"- `{name}`: {df.shape[0]} rânduri × {df.shape[1]} coloane")
    if not has_clusters:
        lines.extend([
            "",
            "## Observație",
            "",
            "Nu a fost găsită tabela `ai_cluster_labels`. Au fost generate doar statisticile descriptive ale cohortei și schema bazei de date.",
            "Pentru tabelele și graficele pe clustere, rulează mai întâi `train_ai_final.py`.",
        ])
    if not SCIPY_AVAILABLE:
        lines.extend([
            "",
            "## Observație despre teste statistice",
            "",
            "Biblioteca `scipy` nu a fost disponibilă, astfel că p-value-urile nu au fost calculate.",
            "Instalare: `pip install scipy`.",
        ])
    (output_dir / "descriptive_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    tables_dir, figures_dir = ensure_dirs(args.output_dir)

    if not args.db.exists():
        raise FileNotFoundError(f"Nu gasesc baza de date: {args.db}")

    generated: dict[str, pd.DataFrame] = {}
    with sqlite3.connect(args.db) as conn:
        features = read_sql_table(conn, args.features_table)
        conn_out = conn if args.write_sqlite else None

        generated.update(generate_cohort_reports(features, tables_dir, figures_dir, conn_out))
        generate_sqlite_schema_reports(conn, tables_dir, figures_dir, conn_out)

        has_clusters = existing_table(conn, args.labels_table)
        if has_clusters:
            labels = read_sql_table(conn, args.labels_table)
            if "id_pacient" not in labels.columns or "cluster" not in labels.columns:
                raise ValueError(f"Tabela {args.labels_table} trebuie sa contina id_pacient si cluster.")
            data = features.merge(labels[["id_pacient", "cluster"]], on="id_pacient", how="inner")
            generated.update(generate_cluster_reports(data, tables_dir, figures_dir, conn_out))
        else:
            print(f"⚠ Tabela {args.labels_table} nu exista. Sar peste rapoartele pe clustere.")

    write_markdown_report(args.output_dir, generated, has_clusters)
    write_recommended_outputs(args.output_dir)

    print("\n✅ Analiza descriptiva a fost generata.")
    print(f"Folder output: {args.output_dir}")
    print(f"Tabele: {tables_dir}")
    print(f"Figuri: {figures_dir}")
    print("Raport: descriptive_analysis_report.md")


if __name__ == "__main__":
    main()
