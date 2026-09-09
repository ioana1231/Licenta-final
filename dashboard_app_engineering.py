#!/usr/bin/env python3
"""
Dashboard local pentru lucrarea de licenta - versiune pentru filmare/prezentare.

Rulare recomandata:
    streamlit run dashboard_app_recording.py --server.address 127.0.0.1 --server.port 8501

Ruleaza-l din acelasi folder cu:
    baza_date_licenta.db
    outputs_ai/
    outputs_descriptive_analysis/

Rol:
- vizualizare locala pentru tabele, metrici, grafice si structura bazei SQLite;
- fara rol diagnostic;
- nu modifica datele si nu retreneaza modelul.
"""
from __future__ import annotations

import sqlite3
import warnings
from pathlib import Path
from typing import Iterable

warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D.*",
    category=UserWarning,
    module="matplotlib\\.projections.*",
)

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

SCRIPT_DIR = Path(__file__).resolve().parent
CWD = Path.cwd()


# =========================
# Config helpers
# =========================

def find_first_existing(candidates: Iterable[Path]) -> Path:
    candidates = list(candidates)
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


DEFAULT_DB = find_first_existing([
    CWD / "baza_date_licenta.db",
    SCRIPT_DIR / "baza_date_licenta.db",
])
DEFAULT_OUTPUTS_AI = find_first_existing([
    CWD / "outputs_ai",
    SCRIPT_DIR / "outputs_ai",
])
DEFAULT_OUTPUTS_DESC = find_first_existing([
    CWD / "outputs_descriptive_analysis",
    SCRIPT_DIR / "outputs_descriptive_analysis",
    CWD / "outputs_descriptive_analysis(1)",
    SCRIPT_DIR / "outputs_descriptive_analysis(1)",
])
DEFAULT_OUTPUTS_ENGINEERING = find_first_existing([
    CWD / "outputs_engineering",
    SCRIPT_DIR / "outputs_engineering",
])

st.set_page_config(
    page_title="Dashboard licenta",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 2.5rem;
    }
    h1 {
        letter-spacing: -0.03em;
    }
    h2, h3 {
        letter-spacing: -0.015em;
    }
    .section-card {
        border: 1px solid rgba(49, 51, 63, 0.15);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0 1.4rem 0;
        background: rgba(250, 250, 250, 0.45);
    }
    .small-note {
        color: rgba(49, 51, 63, 0.68);
        font-size: 0.95rem;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
    [data-testid="stDecoration"] {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# Read helpers
# =========================

@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def list_sqlite_tables(db_path: str) -> list[str]:
    path = Path(db_path)
    if not path.exists():
        return []
    with sqlite3.connect(path) as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    return [row[0] for row in rows]


@st.cache_data(show_spinner=False)
def read_sqlite_table(db_path: str, table_name: str, limit: int | None = None) -> pd.DataFrame:
    query = f'SELECT * FROM "{table_name}"'
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)


def first_existing(base: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        candidate = base / name
        if candidate.exists():
            return candidate
    return None


def st_dataframe(df: pd.DataFrame, **kwargs) -> None:
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        try:
            st.dataframe(df, use_container_width=True, **kwargs)
        except TypeError:
            st.dataframe(df, **kwargs)


def st_image(path: Path, caption: str | None = None, width: int | None = None) -> None:
    if not path.exists():
        st.info(f"Nu am gasit figura: {path}")
        return

    def render() -> None:
        try:
            st.image(str(path), caption=caption, width=width or "stretch")
        except TypeError:
            try:
                if width is not None:
                    st.image(str(path), caption=caption, width=width)
                else:
                    st.image(str(path), caption=caption, use_container_width=True)
            except TypeError:
                st.image(str(path), caption=caption, use_column_width=True)

    if width is None:
        render()
    else:
        _, image_column, _ = st.columns([1, 2, 1])
        with image_column:
            render()


def st_image_any(base: Path, names: Iterable[str], caption: str | None = None) -> None:
    path = first_existing(base, names)
    if path is None:
        st.info("Nu am gasit figura: " + " / ".join(str(base / name) for name in names))
        return
    st_image(path, caption=caption)


def show_csv(path: Path, title: str, max_rows: int | None = None, expanded: bool = False) -> pd.DataFrame | None:
    df = read_csv(str(path))
    if df is None:
        st.info(f"Nu am gasit fisierul: {path}")
        return None
    container = st.expander(title, expanded=expanded)
    with container:
        st_dataframe(df.head(max_rows) if max_rows else df)
    return df


def metric_card(label: str, value: object) -> None:
    st.metric(label, "-" if value is None or pd.isna(value) else value)


# =========================
# Plot helpers
# =========================

def _display_fig(fig) -> None:
    try:
        st.pyplot(fig, width="stretch")
    except TypeError:
        try:
            st.pyplot(fig, use_container_width=True)
        except TypeError:
            st.pyplot(fig)
    plt.close(fig)


def barh_chart(
    df: pd.DataFrame | None,
    label_col: str,
    value_col: str,
    title: str,
    xlabel: str = "Numar pacienti",
    sort_desc: bool = True,
    note: str | None = None,
) -> None:
    if df is None or df.empty:
        st.info(f"Nu exista date pentru: {title}")
        return
    if label_col not in df.columns or value_col not in df.columns:
        st.info(f"Fisierul pentru {title} nu are coloanele asteptate: {label_col}, {value_col}")
        with st.expander("Vezi tabelul brut"):
            st_dataframe(df)
        return

    plot_df = df[[label_col, value_col]].dropna().copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[value_col])
    if sort_desc:
        plot_df = plot_df.sort_values(value_col, ascending=False)

    height = max(3.0, min(7.5, 1.4 + 0.48 * len(plot_df)))
    fig, ax = plt.subplots(figsize=(10, height))
    ax.barh(plot_df[label_col].astype(str), plot_df[value_col])
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    _display_fig(fig)
    if note:
        st.caption(note)


def grouped_bmi_by_sex(df: pd.DataFrame | None) -> None:
    if df is None or df.empty:
        st.info("Nu exista date pentru IMC pe sexe.")
        return
    required = {"sex", "categorie_imc", "n", "%_din_sex"}
    if not required.issubset(df.columns):
        st.info("Fisierul imc_distribution_by_sex.csv nu are coloanele asteptate.")
        with st.expander("Vezi tabelul brut"):
            st_dataframe(df)
        return

    order = [
        "Subponderal",
        "Normoponderal",
        "Supraponderal",
        "Obezitate grad I",
        "Obezitate grad II",
        "Obezitate grad III",
        "IMC lipsă",
    ]
    prepared = df.copy()
    prepared["categorie_imc"] = pd.Categorical(prepared["categorie_imc"], categories=order, ordered=True)
    pivot = (
        prepared.pivot_table(index="categorie_imc", columns="sex", values="%_din_sex", aggfunc="sum", observed=False)
        .reindex(order)
        .dropna(how="all")
        .fillna(0)
    )
    height = max(3.2, 1.5 + 0.5 * len(pivot))
    fig, ax = plt.subplots(figsize=(10, height))
    pivot.plot(kind="barh", ax=ax)
    ax.invert_yaxis()
    ax.set_title("Distribuția IMC pe sexe")
    ax.set_xlabel("Procent în cadrul fiecărui sex (%)")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(title="Sex")
    fig.tight_layout()
    _display_fig(fig)
    st.caption("Procente calculate separat în cadrul fiecărui sex. Categoria «IMC lipsă» indică lipsa valorii IMC, nu o categorie ponderală.")


def severity_chart(df: pd.DataFrame | None, title: str) -> None:
    if df is None or df.empty:
        st.info(f"Nu exista date pentru: {title}")
        return
    value_col = "%_dintre_calculabili" if "%_dintre_calculabili" in df.columns else "n"
    xlabel = "Procent dintre pacienții calculabili (%)" if value_col.startswith("%") else "Număr pacienți"
    order = ["Ușoară", "Moderată", "Severă"]
    plot_df = df.copy()
    if "severitate" in plot_df.columns:
        plot_df["severitate"] = pd.Categorical(plot_df["severitate"], categories=order, ordered=True)
        plot_df = plot_df.sort_values("severitate")
    barh_chart(plot_df, "severitate", value_col, title, xlabel=xlabel, sort_desc=False)
    if "n_necalculabil_complet" in df.columns and not df.empty:
        try:
            missing = int(df["n_necalculabil_complet"].dropna().iloc[0])
            st.caption(f"Pacienți necalculabili complet: {missing}. Graficul folosește doar pacienții pentru care scorul complet a putut fi calculat.")
        except Exception:
            pass


def stat_badge(text: str) -> None:
    st.markdown(f"<span class='small-note'>{text}</span>", unsafe_allow_html=True)


def comorbidity_surface_3d(df: pd.DataFrame | None) -> None:
    if df is None or df.empty:
        st.info("Nu există date pentru suprafața 3D a comorbidităților.")
        return
    required = {"grupa_varsta", "comorbiditate", "%_din_grupa_varsta"}
    if not required.issubset(df.columns):
        st.info("Tabelul comorbidităților pe grupe de vârstă nu are coloanele necesare pentru suprafața 3D.")
        return

    age_order = ["<50", "50-59", "60-69", "70-79", ">=80", "Vârstă lipsă"]
    ages = [age for age in age_order if age in df["grupa_varsta"].astype(str).unique()]
    comorbidities = df["comorbiditate"].dropna().astype(str).drop_duplicates().tolist()
    matrix = (
        df.assign(grupa_varsta=df["grupa_varsta"].astype(str))
        .pivot_table(index="comorbiditate", columns="grupa_varsta", values="%_din_grupa_varsta", aggfunc="mean", fill_value=0)
        .reindex(index=comorbidities, columns=ages, fill_value=0)
    )
    figure = go.Figure(data=[go.Surface(x=ages, y=matrix.index.tolist(), z=matrix.to_numpy(), colorscale="Viridis", colorbar={"title": "%"})])
    figure.update_layout(
        title="Suprafață 3D: comorbidități pe grupe de vârstă",
        scene={"xaxis_title": "Grupă de vârstă", "yaxis_title": "Comorbiditate", "zaxis_title": "% în grupa de vârstă"},
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
        height=650,
    )
    st.plotly_chart(figure, use_container_width=True)


def show_covariance_outputs(tables_dir: Path, figures_dir: Path, include_clusters: bool = True) -> None:
    overall_figure = figures_dir / "covariance_all_variables_overall_heatmap.png"
    if overall_figure.exists():
        st.subheader("Populația totală")
        st_image(overall_figure, "Covarianța tuturor variabilelor numerice în populația totală", width=760)
        show_csv(tables_dir / "covariance_all_variables_overall.csv", "Tabel covarianță - populația totală")

    if not include_clusters:
        return

    cluster_figures = sorted(figures_dir.glob("covariance_all_variables_cluster_*_heatmap.png"))
    if cluster_figures:
        st.subheader("Covarianța în fiecare cluster")
        st.caption("Valorile sunt calculate separat în interiorul fiecărui cluster; comparația trebuie făcută ținând cont de dimensiunea clusterului.")
        for figure in cluster_figures:
            cluster_name = figure.stem.replace("covariance_all_variables_", "").replace("_heatmap", "").replace("_", " ")
            cluster_id = cluster_name.replace("cluster ", "Cluster ")
            st_image(figure, f"Covarianța tuturor variabilelor numerice - {cluster_id}", width=760)
            table_name = figure.name.replace("_heatmap.png", ".csv")
            show_csv(tables_dir / table_name, f"Tabel covarianță - {cluster_id}")


# =========================
# Presentation mode paths
# =========================

st.title("Dashboard pentru analiza exploratorie")
st.caption("Vizualizare locală pentru tabele, metrici, grafice și structura bazei SQLite. Fără rol diagnostic.")

# Pentru filmare/prezentare folosim automat fișierele din folderul curent.
# Dashboard-ul trebuie rulat din același folder cu baza SQLite și folderele de output.
db_path = DEFAULT_DB
outputs_ai = DEFAULT_OUTPUTS_AI
outputs_desc = DEFAULT_OUTPUTS_DESC
outputs_eng = DEFAULT_OUTPUTS_ENGINEERING
tables_desc = outputs_desc / "tables_recommended"
figures_desc = outputs_desc / "figures_recommended"

if not db_path.exists():
    st.warning("Nu am găsit baza SQLite. Rulează dashboard-ul din folderul licenței, lângă baza_date_licenta.db și folderele outputs_ai / outputs_descriptive_analysis.")
    st.code("streamlit run dashboard_app_recording.py --server.address 127.0.0.1 --server.port 8501", language="bash")
    st.stop()

try:
    available_tables = list_sqlite_tables(str(db_path))
except Exception as exc:
    st.error(f"Nu pot citi baza SQLite: {exc}")
    st.stop()

if not available_tables:
    st.error("Baza SQLite există, dar nu conține tabele sau nu poate fi citită.")
    st.stop()

features_ai = read_sqlite_table(str(db_path), "features_ai") if "features_ai" in available_tables else pd.DataFrame()
cluster_labels = read_sqlite_table(str(db_path), "ai_cluster_labels") if "ai_cluster_labels" in available_tables else pd.DataFrame()
cluster_profile = read_sqlite_table(str(db_path), "ai_cluster_profile") if "ai_cluster_profile" in available_tables else pd.DataFrame()
cluster_metrics = read_sqlite_table(str(db_path), "ai_cluster_metrics") if "ai_cluster_metrics" in available_tables else pd.DataFrame()

if cluster_metrics.empty:
    metrics_csv = read_csv(str(outputs_ai / "ai_cluster_metrics.csv"))
    if metrics_csv is not None:
        cluster_metrics = metrics_csv
if cluster_profile.empty:
    profile_csv = read_csv(str(outputs_ai / "ai_cluster_profile.csv"))
    if profile_csv is not None:
        cluster_profile = profile_csv

# Tables loaded once for cleaner charts
sex_dist = read_csv(str(tables_desc / "sex_distribution.csv"))
cohort_cont = read_csv(str(tables_desc / "cohort_descriptive_continuous.csv"))
missing_rates = read_csv(str(tables_desc / "missing_rates_selected_variables.csv"))
age_overall = read_csv(str(tables_desc / "age_group_distribution_overall.csv"))
age_by_sex = read_csv(str(tables_desc / "age_group_distribution_by_sex.csv"))
age_summary = read_csv(str(tables_desc / "age_descriptive_by_sex.csv"))
imc_overall = read_csv(str(tables_desc / "imc_distribution_overall.csv"))
imc_by_sex = read_csv(str(tables_desc / "imc_distribution_by_sex.csv"))
symptoms = read_csv(str(tables_desc / "symptoms_frequency.csv"))
comorbidity = read_csv(str(tables_desc / "comorbidity_distribution.csv"))
comorbidity_by_age = read_csv(str(tables_desc / "comorbidity_distribution_by_age_group.csv"))
etiology = read_csv(str(tables_desc / "etiology_distribution.csv"))
covariance = read_csv(str(tables_desc / "covariance_key_variables.csv"))
faced_calc = read_csv(str(tables_desc / "faced_severity_distribution_calculable_only.csv"))
bsi_calc = read_csv(str(tables_desc / "bsi_severity_distribution_calculable_only.csv"))
cluster_dist = read_csv(str(tables_desc / "cluster_distribution.csv"))

# Fișiere inginerești suplimentare: comparare algoritmi + performanță
alg_comparison = read_csv(str(outputs_eng / "clustering_algorithm_comparison.csv"))
alg_best = read_csv(str(outputs_eng / "clustering_algorithm_best_by_method.csv"))
perf_stages = read_csv(str(outputs_eng / "performance_stage_summary.csv"))
scalability = read_csv(str(outputs_eng / "scalability_kmeans_runtime.csv"))
dashboard_perf = read_csv(str(outputs_eng / "dashboard_responsiveness.csv"))
refresh_rate = read_csv(str(outputs_eng / "refresh_rate_summary.csv"))

(tab_overview, tab_desc, tab_resp, tab_cluster, tab_algorithms, tab_perf, tab_db) = st.tabs([
    "Cohortă",
    "Statistici descriptive",
    "Parametri respiratori",
    "Clustering",
    "Comparare algoritmi",
    "Performanță",
    "Arhitectură / SQLite",
])

# =========================
# Cohort tab
# =========================
with tab_overview:
    st.header("Cohortă")
    if features_ai.empty:
        st.info("Tabela features_ai nu există în bază.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Pacienți", int(features_ai["id_pacient"].nunique()) if "id_pacient" in features_ai else len(features_ai))
        with c2:
            metric_card("Coloane features_ai", features_ai.shape[1])
        with c3:
            metric_card("Sex M", int((features_ai["sex"].astype(str).str.upper() == "M").sum()) if "sex" in features_ai else None)
        with c4:
            metric_card("Sex F", int((features_ai["sex"].astype(str).str.upper() == "F").sum()) if "sex" in features_ai else None)

        st.subheader("Distribuția pe sexe")
        barh_chart(sex_dist, "sex", "n", "Distribuția cohortei pe sexe", xlabel="Număr pacienți", sort_desc=True)

        if "imc_sursa" in features_ai.columns:
            st.subheader("Sursa IMC")
            source_counts = features_ai["imc_sursa"].fillna("lipsa").value_counts().rename_axis("sursa_imc").reset_index(name="n")
            source_counts["%_din_total"] = (source_counts["n"] / source_counts["n"].sum() * 100).round(2)
            st.dataframe(source_counts, use_container_width=True, hide_index=True)
            st.caption("IMC calculat din greutate și înălțime este folosit doar atunci când nu există valoare validă în Excel.")

        st.subheader("Distribuția vârstei")
        col_age1, col_age2 = st.columns(2)
        with col_age1:
            barh_chart(age_overall, "grupa_varsta", "n", "Distribuția vârstei pe grupe", xlabel="Număr pacienți", sort_desc=False)
        with col_age2:
            st_image(figures_desc / "age_group_distribution_by_sex.png", "Distribuția vârstei pe sexe")
        show_csv(tables_desc / "age_descriptive_by_sex.csv", "Vârstă - sinteză pe sexe")
        show_csv(tables_desc / "age_group_distribution_by_sex.csv", "Vârstă - grupe pe sexe")

        col_a, col_b = st.columns(2)
        with col_a:
            show_csv(tables_desc / "cohort_descriptive_continuous.csv", "Statistici descriptive continue", expanded=True)
        with col_b:
            show_csv(tables_desc / "missing_rates_selected_variables.csv", "Rate valori lipsă", expanded=True)

        with st.expander("Primele rânduri din features_ai"):
            st_dataframe(features_ai.head(30))

# =========================
# Descriptive tab
# =========================
with tab_desc:
    st.header("Statistici descriptive")
    stat_badge("Graficele de mai jos caracterizează lotul analizat înainte de interpretarea clusterelor.")

    st.subheader("1. Distribuția IMC")
    barh_chart(
        imc_overall,
        "categorie_imc",
        "n",
        "Distribuția IMC pe categorii",
        xlabel="Număr pacienți",
        sort_desc=False,
        note="Categoria «IMC lipsă» indică pacienții pentru care IMC nu a fost disponibil după integrarea datelor.",
    )
    grouped_bmi_by_sex(imc_by_sex)
    show_csv(tables_desc / "imc_distribution_overall.csv", "Tabel IMC - total")
    show_csv(tables_desc / "imc_distribution_by_sex.csv", "Tabel IMC - pe sexe")

    st.divider()
    st.subheader("2. Frecvența simptomelor")
    barh_chart(symptoms, "simptom", "n", "Frecvența principalelor simptome", xlabel="Număr pacienți")
    show_csv(tables_desc / "symptoms_frequency.csv", "Tabel simptome")

    st.divider()
    st.subheader("3. Comorbidități")
    barh_chart(comorbidity, "comorbiditate", "n", "Distribuția comorbidităților", xlabel="Număr pacienți")
    show_csv(tables_desc / "comorbidity_distribution.csv", "Tabel comorbidități")
    show_csv(tables_desc / "comorbidity_score_comparison_partial_scores.csv", "Comorbidități vs scoruri FACED/BSI parțiale")
    comorb_age_col1, comorb_age_col2 = st.columns(2)
    with comorb_age_col1:
        st_image(figures_desc / "comorbidity_by_age_group_heatmap.png", "Comorbidități pe grupe de vârstă", width=560)
    with comorb_age_col2:
        comorbidity_surface_3d(comorbidity_by_age)
    show_csv(tables_desc / "comorbidity_distribution_by_age_group.csv", "Comorbidități pe grupe de vârstă")

    st.divider()
    st.subheader("4. Etiologie")
    barh_chart(etiology, "etiologie", "n", "Distribuția după etiologie", xlabel="Număr pacienți")
    st.caption("Procentele nu trebuie însumate ca distribuție exclusivă dacă un pacient poate avea mai multe grupe etiologice marcate.")
    show_csv(tables_desc / "etiology_distribution.csv", "Tabel etiologie")

    st.divider()
    st.subheader("5. Scoruri de severitate")
    sev1, sev2 = st.columns(2)
    with sev1:
        severity_chart(faced_calc, "Distribuția severității FACED - pacienți calculabili")
        show_csv(tables_desc / "faced_severity_distribution_calculable_only.csv", "Tabel FACED")
    with sev2:
        severity_chart(bsi_calc, "Distribuția severității BSI - pacienți calculabili")
        show_csv(tables_desc / "bsi_severity_distribution_calculable_only.csv", "Tabel BSI")

    st.divider()
    st.subheader("6. Matrice de covarianță")
    st.caption("Matricea este calculată pentru variabile numerice cheie și are rol exploratoriu; variabilele sunt pe scări diferite.")
    st_image(figures_desc / "covariance_key_variables_heatmap.png", "Matrice de covarianță - variabile numerice cheie", width=720)
    show_csv(tables_desc / "covariance_key_variables.csv", "Tabel covarianță")

# =========================
# Respiratory tab
# =========================
with tab_resp:
    st.header("Parametri respiratori")
    st.caption("Compararea FEV1/FVC este descriptivă. Rezultatele nu stabilesc relații cauzale.")

    st.subheader("Comparații statistice")
    col1, col2, col3 = st.columns(3)
    with col1:
        show_csv(tables_desc / "respiratory_comparison_by_sex.csv", "FEV1/FVC după sex", expanded=True)
    with col2:
        show_csv(tables_desc / "respiratory_comparison_by_age_group.csv", "FEV1/FVC după grupe de vârstă", expanded=True)
    with col3:
        show_csv(tables_desc / "respiratory_comparison_by_colonizare_pseudomonas.csv", "FEV1/FVC după Pseudomonas", expanded=True)

    st.divider()
    st.subheader("Boxploturi")
    st_image(figures_desc / "fev1_min_final_by_age_group_boxplot.png", "FEV1 după grupe de vârstă", width=620)
    st_image(figures_desc / "fvc_min_final_by_age_group_boxplot.png", "FVC după grupe de vârstă", width=620)
    st_image(figures_desc / "fev1_min_final_by_colonization_boxplot.png", "FEV1 după colonizare Pseudomonas", width=620)
    st_image(figures_desc / "fvc_min_final_by_colonization_boxplot.png", "FVC după colonizare Pseudomonas", width=620)

# =========================
# Clustering tab
# =========================
with tab_cluster:
    st.header("Clustering")
    if not cluster_metrics.empty:
        row_k2 = cluster_metrics[cluster_metrics["k"].eq(2)] if "k" in cluster_metrics else pd.DataFrame()
        row_k3 = cluster_metrics[cluster_metrics["k"].eq(3)] if "k" in cluster_metrics else pd.DataFrame()
        best = cluster_metrics.sort_values("silhouette", ascending=False).iloc[0] if "silhouette" in cluster_metrics else None
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("k automat", int(best["k"]) if best is not None and "k" in best else None)
        with c2:
            metric_card("silhouette maxim", round(float(best["silhouette"]), 4) if best is not None and "silhouette" in best else None)
        with c3:
            metric_card("silhouette k=3", round(float(row_k3.iloc[0]["silhouette"]), 4) if not row_k3.empty else None)
        with c4:
            metric_card("k evaluate", len(cluster_metrics))
        st.caption("Pentru analiza finală a fost utilizat k=3. Valoarea k=2 a obținut un scor silhouette marginal mai mare, însă k=3 oferă o segmentare mai nuanțată pentru interpretarea exploratorie a profilurilor.")
        st_dataframe(cluster_metrics)
    else:
        st.info("Nu am găsit metricile de clustering.")

    st.divider()
    st.subheader("Distribuția clusterelor")
    barh_chart(cluster_dist, "cluster", "n", "Numărul pacienților pe clustere", xlabel="Număr pacienți", sort_desc=False)

    st.subheader("Metrici și vizualizări KMeans")
    km1, km2 = st.columns(2)
    with km1:
        st_image(outputs_ai / "silhouette_plot.png", "Silhouette score")
        st_image(outputs_ai / "cluster_sizes.png", "Dimensiunea clusterelor")
    with km2:
        st_image(outputs_ai / "elbow_plot.png", "Metoda elbow")
        st_image(outputs_ai / "pca_clusters.png", "Vizualizare PCA")

    st.divider()
    st.subheader("Profilul clusterelor")
    st_image(figures_desc / "heatmap_cluster_profile.png", "Heatmap profil standardizat al clusterelor")
    if not cluster_profile.empty:
        with st.expander("Profil clustere - tabel"):
            st_dataframe(cluster_profile)
    show_csv(tables_desc / "table6_cluster_profile_with_variation_and_pvalues.csv", "Profil descriptiv al clusterelor - tabel extins")
    show_csv(outputs_ai / "ai_cluster_top_features.csv", "Top features distinctive", max_rows=80)

    st.divider()
    st.subheader("Covarianța variabilelor pe clustere")
    show_covariance_outputs(tables_desc, figures_desc)

    st.divider()
    st.subheader("Boxploturi pe clustere")
    boxplot_files = sorted(figures_desc.glob("boxplot_*_by_cluster.png"))
    if not boxplot_files:
        st.info("Nu există boxploturi pe clustere. Rulează pipeline-ul pentru a genera graficele.")
    else:
        boxplot_columns = st.columns(2)
        for index, figure in enumerate(boxplot_files):
            with boxplot_columns[index % 2]:
                variable_name = figure.name.removeprefix("boxplot_").removesuffix("_by_cluster.png")
                st_image(figure, f"{variable_name} pe clustere")


# =========================
# Algorithm comparison tab
# =========================
with tab_algorithms:
    st.header("Comparare metode de clustering")
    st.caption("Comparația susține alegerea metodei finale din perspectivă computațională. Interpretarea rămâne exploratorie.")

    if alg_best is None or alg_best.empty:
        st.info("Nu am găsit output-urile de comparare algoritmi. Rulează: python compare_clustering_methods.py --db baza_date_licenta.db --output-dir outputs_engineering")
    else:
        display_cols = [c for c in ["method", "configuration", "n_clusters", "noise_points", "silhouette", "calinski_harabasz", "davies_bouldin", "runtime_seconds", "selection_reason"] if c in alg_best.columns]
        st.subheader("Cele mai bune configurații pe metodă")
        st_dataframe(alg_best[display_cols])
        st.caption("Pentru DBSCAN, punctele marcate ca zgomot nu sunt incluse în scorurile interne. Un scor bun pe un subset mic nu înseamnă automat metodă mai potrivită pentru analiza finală.")

        col_a, col_b = st.columns(2)
        with col_a:
            st_image(outputs_eng / "clustering_methods_silhouette_comparison.png", "Silhouette pe metoda selectată")
        with col_b:
            st_image(outputs_eng / "clustering_methods_runtime_comparison.png", "Runtime pe metoda selectată")

    show_csv(outputs_eng / "clustering_algorithm_comparison.csv", "Toate configurațiile evaluate")
    show_csv(outputs_eng / "dbscan_parameter_grid.csv", "Grid DBSCAN")

# =========================
# Performance tab
# =========================
with tab_perf:
    st.header("Performanță și cost computațional")
    st.caption("Măsurătorile sunt locale și pot varia în funcție de laptop. Testul de scalabilitate folosește replicarea matricei pentru evaluare computațională, nu pacienți reali suplimentari.")

    if perf_stages is None or perf_stages.empty:
        st.info("Nu am găsit output-urile de performanță. Rulează: python measure_engineering_performance.py --db baza_date_licenta.db --outputs-ai outputs_ai --outputs-desc outputs_descriptive_analysis --output-dir outputs_engineering")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            current = perf_stages[perf_stages["stage"].eq("kmeans_final_k3")] if "stage" in perf_stages else pd.DataFrame()
            metric_card("Runtime KMeans k=3", round(float(current.iloc[0]["duration_seconds"]), 4) if not current.empty else None)
        with c2:
            read_row = perf_stages[perf_stages["stage"].eq("read_features_clustering")] if "stage" in perf_stages else pd.DataFrame()
            metric_card("Citire matrice", round(float(read_row.iloc[0]["duration_seconds"]), 4) if not read_row.empty else None)
        with c3:
            if refresh_rate is not None and not refresh_rate.empty and "indicator" in refresh_rate.columns:
                ds = refresh_rate[refresh_rate["indicator"].eq("downstream_refresh_seconds")]
                metric_card("Refresh downstream", round(float(ds.iloc[0]["value_seconds"]), 4) if not ds.empty else None)

        st_image(outputs_eng / "stage_runtime_summary.png", "Cost computațional pe etape")
        show_csv(outputs_eng / "performance_stage_summary.csv", "Tabel runtime / throughput", expanded=True)
        show_csv(outputs_eng / "refresh_rate_summary.csv", "Refresh rate / regenerare locală", expanded=True)

    st.subheader("Scalabilitate")
    st_image(outputs_eng / "kmeans_scalability_runtime.png", "Scalabilitate KMeans prin replicare computațională")
    show_csv(outputs_eng / "scalability_kmeans_runtime.csv", "Tabel scalabilitate KMeans", expanded=True)

    st.subheader("Responsivitate dashboard")
    show_csv(outputs_eng / "dashboard_responsiveness.csv", "Timp încărcare resurse dashboard", expanded=True)

# =========================
# DB tab
# =========================
with tab_db:
    st.header("Arhitectură și baza SQLite")
    st.write("Contribuția principală este pipeline-ul de date: import, curățare, anonimizare, structurare în SQLite, feature engineering, clustering, export și vizualizare locală.")
    st_image(outputs_eng / "data_pipeline_diagram.png", "Diagramă pipeline de date")
    st_image(outputs_eng / "use_case_diagram.png", "Diagramă de cazuri de utilizare")
    st.write("Relațiile sunt păstrate logic prin `id_pacient`. Cheile SQL formale pot să nu fie declarate, deoarece tabelele sunt scrise cu pandas.to_sql.")
    st_image(figures_desc / "sqlite_logical_er_diagram.png", "Diagramă logică SQLite")
    st.subheader("Tabele disponibile")
    st_dataframe(pd.DataFrame({"tabel": available_tables}))
    show_csv(tables_desc / "sqlite_schema_summary.csv", "Schema fizică - sinteză", expanded=True)
    show_csv(tables_desc / "sqlite_schema_detail.csv", "Schema fizică detaliată")
    table_choice = st.selectbox("Inspectare tabelă", available_tables)
    preview = read_sqlite_table(str(db_path), table_choice, limit=100)
    st.caption(f"Se afișează primele {len(preview)} rânduri.")
    st_dataframe(preview)
