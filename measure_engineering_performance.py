#!/usr/bin/env python3
"""
Masurare performanta si cost computational pentru pipeline/dashboard.

Scop:
- masoara costul de citire a tabelei features_clustering;
- masoara runtime pentru KMeans final si metode alternative;
- simuleaza scalabilitatea prin replicarea matricei numerice;
- masoara timpul de incarcare al resurselor folosite de dashboard;
- genereaza o diagrama a pipeline-ului de date.

Rulare:
    python measure_engineering_performance.py --db baza_date_licenta.db --outputs-ai outputs_ai --outputs-desc outputs_descriptive_analysis --output-dir outputs_engineering
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Converteste un DataFrame in tabel Markdown fara dependinta tabulate."""
    if df.empty:
        return "_Nu exista date._"

    formatted = df.copy()
    formatted.columns = [str(column).replace("|", r"\|") for column in formatted.columns]
    formatted = formatted.fillna("")

    def escape(value: object) -> str:
        return str(value).replace("|", r"\|").replace("\n", "<br>")

    rows = [[escape(value) for value in row] for row in formatted.itertuples(index=False, name=None)]
    header = "| " + " | ".join(formatted.columns) + " |"
    separator = "| " + " | ".join("---" for _ in formatted.columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Masurare performanta pentru proiectul de licenta")
    parser.add_argument("--db", type=Path, default=base_dir / "baza_date_licenta.db")
    parser.add_argument("--table", default="features_clustering")
    parser.add_argument("--outputs-ai", type=Path, default=base_dir / "outputs_ai")
    parser.add_argument("--outputs-desc", type=Path, default=base_dir / "outputs_descriptive_analysis")
    parser.add_argument("--output-dir", type=Path, default=base_dir / "outputs_engineering")
    parser.add_argument("--scalability-factors", default="1,2,5,10,20", help="Factori de multiplicare pentru testul de scalabilitate")
    return parser.parse_args()


def timed(label: str, fn):
    start = time.perf_counter()
    result = fn()
    duration = time.perf_counter() - start
    return label, duration, result


def read_table(db_path: Path, table_name: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)


def read_feature_matrix(db_path: Path, table_name: str) -> tuple[pd.DataFrame, float]:
    label, duration, df = timed("read_features_clustering", lambda: read_table(db_path, table_name))
    if "id_pacient" not in df.columns:
        raise ValueError(f"Tabela {table_name} nu contine id_pacient")
    X = df.drop(columns=["id_pacient"])
    return X, duration


def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return float("nan")
    return float(silhouette_score(X, labels))


def benchmark_current_algorithms(X: pd.DataFrame) -> pd.DataFrame:
    rows = []
    arr = X.to_numpy()
    n_patients = len(X)

    start = time.perf_counter()
    km = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=50, algorithm="lloyd")
    km_labels = km.fit_predict(arr)
    duration = time.perf_counter() - start
    rows.append({
        "stage": "kmeans_final_k3",
        "description": "Antrenare KMeans final cu k=3",
        "n_patients": n_patients,
        "n_features": X.shape[1],
        "duration_seconds": duration,
        "throughput_patients_per_second": n_patients / duration if duration > 0 else np.nan,
        "silhouette": safe_silhouette(arr, km_labels),
    })

    start = time.perf_counter()
    ag = AgglomerativeClustering(n_clusters=3, linkage="ward")
    ag_labels = ag.fit_predict(arr)
    duration = time.perf_counter() - start
    rows.append({
        "stage": "hierarchical_ward_k3",
        "description": "Clustering ierarhic Ward cu k=3",
        "n_patients": n_patients,
        "n_features": X.shape[1],
        "duration_seconds": duration,
        "throughput_patients_per_second": n_patients / duration if duration > 0 else np.nan,
        "silhouette": safe_silhouette(arr, ag_labels),
    })

    start = time.perf_counter()
    db = DBSCAN(eps=3.0, min_samples=5)
    db_labels = db.fit_predict(arr)
    duration = time.perf_counter() - start
    non_noise = db_labels != -1
    sil = float("nan")
    if non_noise.sum() > 2:
        sil = safe_silhouette(arr[non_noise], db_labels[non_noise])
    rows.append({
        "stage": "dbscan_example_eps3",
        "description": "DBSCAN exemplificativ eps=3.0, min_samples=5",
        "n_patients": n_patients,
        "n_features": X.shape[1],
        "duration_seconds": duration,
        "throughput_patients_per_second": n_patients / duration if duration > 0 else np.nan,
        "silhouette": sil,
    })
    return pd.DataFrame(rows)


def replicate_matrix(X: pd.DataFrame, factor: int) -> np.ndarray:
    arr = X.to_numpy(dtype=float)
    if factor <= 1:
        return arr.copy()
    # Replicare determinista pentru test computational. Nu reprezinta pacienti reali suplimentari.
    return np.tile(arr, (factor, 1))


def benchmark_scalability(X: pd.DataFrame, factors: list[int]) -> pd.DataFrame:
    rows = []
    base_n = len(X)
    for factor in factors:
        arr = replicate_matrix(X, factor)
        n = len(arr)
        start = time.perf_counter()
        model = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10, algorithm="lloyd")
        model.fit(arr)
        duration = time.perf_counter() - start
        rows.append({
            "factor": int(factor),
            "n_rows_simulated": int(n),
            "base_n_patients": int(base_n),
            "n_features": int(X.shape[1]),
            "algorithm": "KMeans(k=3, n_init=10)",
            "duration_seconds": duration,
            "throughput_rows_per_second": n / duration if duration > 0 else np.nan,
            "note": "Replicare computationala a matricei; nu reprezinta pacienti reali noi.",
        })
    return pd.DataFrame(rows)


def benchmark_dashboard_resources(db_path: Path, outputs_ai: Path, outputs_desc: Path) -> pd.DataFrame:
    rows = []

    def measure_resource(name: str, fn, items: int | None = None):
        start = time.perf_counter()
        try:
            result = fn()
            status = "ok"
        except Exception as exc:
            result = None
            status = f"error: {exc}"
        duration = time.perf_counter() - start
        rows.append({
            "resource_group": name,
            "items_loaded": items if items is not None else (len(result) if hasattr(result, "__len__") and result is not None else np.nan),
            "duration_seconds": duration,
            "status": status,
        })
        return result

    measure_resource("sqlite_table_list", lambda: pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", sqlite3.connect(db_path)))
    for table in ["features_ai", "features_clustering"]:
        measure_resource(f"sqlite_preview_{table}", lambda t=table: read_table(db_path, t).head(100), items=100)

    ai_csvs = list(outputs_ai.glob("*.csv")) if outputs_ai.exists() else []
    measure_resource("outputs_ai_csv_load", lambda: [pd.read_csv(p) for p in ai_csvs], items=len(ai_csvs))

    tables_dir = outputs_desc / "tables_recommended"
    desc_csvs = list(tables_dir.glob("*.csv")) if tables_dir.exists() else []
    measure_resource("descriptive_tables_csv_load", lambda: [pd.read_csv(p) for p in desc_csvs], items=len(desc_csvs))

    figures_dir = outputs_desc / "figures_recommended"
    figures = list(figures_dir.glob("*.png")) if figures_dir.exists() else []
    measure_resource("dashboard_figures_scan", lambda: [p.stat().st_size for p in figures], items=len(figures))

    return pd.DataFrame(rows)


def create_pipeline_diagram(output_dir: Path) -> None:
    steps = [
        ("Excel brut\n12 foi", "import"),
        ("Curățare +\nnormalizare", "clean"),
        ("Anonimizare\nid_pacient", "privacy"),
        ("SQLite\n*_clean", "db"),
        ("Feature engineering\npe domenii", "features"),
        ("features_ai", "master"),
        ("features_clustering", "matrix"),
        ("Clustering +\nstatistici", "analysis"),
        ("CSV / PNG /\ndashboard", "export"),
    ]
    fig, ax = plt.subplots(figsize=(15, 3.6))
    ax.axis("off")
    xs = np.linspace(0.05, 0.95, len(steps))
    y = 0.55
    for i, ((label, _), x) in enumerate(zip(steps, xs)):
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.45", linewidth=1.0),
            transform=ax.transAxes,
        )
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.045, y),
                xytext=(x + 0.045, y),
                xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", linewidth=1.1),
            )
    ax.set_title("Diagrama pipeline-ului de date", fontsize=13, pad=16)
    fig.tight_layout()
    fig.savefig(output_dir / "data_pipeline_diagram.png", dpi=200)
    plt.close(fig)


def create_use_case_diagram(output_dir: Path) -> None:
    """Genereaza o diagrama simplificata de cazuri de utilizare pentru sistem."""
    use_cases = [
        "Import date Excel",
        "Curățare și\nnormalizare",
        "Anonimizare date",
        "Construire SQLite",
        "Feature engineering",
        "Clustering",
        "Statistici și\nvalidare",
        "Export rezultate",
        "Vizualizare dashboard",
    ]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")

    # Actor
    actor_x, actor_y = 0.12, 0.55
    ax.text(actor_x, actor_y, "Utilizator\n(student / cercetător)", ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.45", linewidth=1.0), transform=ax.transAxes)

    # System boundary
    boundary = plt.Rectangle((0.28, 0.10), 0.66, 0.78, fill=False, linewidth=1.2, transform=ax.transAxes)
    ax.add_patch(boundary)
    ax.text(0.61, 0.91, "Sistem de prelucrare și analiză exploratorie", ha="center", va="center", fontsize=12, transform=ax.transAxes)

    positions = [
        (0.43, 0.78), (0.68, 0.78),
        (0.43, 0.62), (0.68, 0.62),
        (0.43, 0.46), (0.68, 0.46),
        (0.43, 0.30), (0.68, 0.30),
        (0.56, 0.18),
    ]
    for label, (x, y) in zip(use_cases, positions):
        ax.text(x, y, label, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.35", linewidth=1.0), transform=ax.transAxes)
        ax.annotate("", xy=(x - 0.07, y), xytext=(actor_x + 0.08, actor_y), xycoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="-", linewidth=0.7))

    ax.set_title("Diagramă de cazuri de utilizare", fontsize=14, pad=16)
    fig.tight_layout()
    fig.savefig(output_dir / "use_case_diagram.png", dpi=200)
    plt.close(fig)


def plot_performance(scalability: pd.DataFrame, stage_summary: pd.DataFrame, output_dir: Path) -> None:
    if not scalability.empty:
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        ax.plot(scalability["n_rows_simulated"], scalability["duration_seconds"], marker="o")
        ax.set_title("Scalabilitate KMeans")
        ax.set_xlabel("Număr rânduri simulate")
        ax.set_ylabel("Timp rulare (secunde)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "kmeans_scalability_runtime.png", dpi=200)
        plt.close(fig)

    if not stage_summary.empty:
        plot_df = stage_summary.sort_values("duration_seconds", ascending=False)
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.barh(plot_df["stage"], plot_df["duration_seconds"])
        ax.invert_yaxis()
        ax.set_title("Cost computațional pe etape")
        ax.set_xlabel("Timp rulare (secunde)")
        ax.set_ylabel("")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "stage_runtime_summary.png", dpi=200)
        plt.close(fig)


def save_report(output_dir: Path, stage_summary: pd.DataFrame, scalability: pd.DataFrame, dashboard: pd.DataFrame, refresh: pd.DataFrame) -> None:
    lines = [
        "# Raport performanță și cost computațional",
        "",
        "Raportul măsoară costul operațiilor principale pe baza SQLite și pe output-urile generate local.",
        "Testul de scalabilitate este computational: matricea de caracteristici este replicată pentru a observa evoluția timpului de rulare, fără a reprezenta pacienți reali suplimentari.",
        "",
        "## Etape măsurate",
        dataframe_to_markdown(stage_summary),
        "",
        "## Refresh rate / regenerare locală",
        dataframe_to_markdown(refresh),
        "",
        "## Responsivitate dashboard",
        dataframe_to_markdown(dashboard),
        "",
        "## Scalabilitate KMeans",
        dataframe_to_markdown(scalability),
    ]
    (output_dir / "performance_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.db.exists():
        raise FileNotFoundError(f"Nu gasesc baza SQLite: {args.db}")

    X, read_duration = read_feature_matrix(args.db, args.table)
    algorithm_summary = benchmark_current_algorithms(X)
    read_row = pd.DataFrame([{
        "stage": "read_features_clustering",
        "description": "Citire matrice numerică din SQLite",
        "n_patients": len(X),
        "n_features": X.shape[1],
        "duration_seconds": read_duration,
        "throughput_patients_per_second": len(X) / read_duration if read_duration > 0 else np.nan,
        "silhouette": np.nan,
    }])
    stage_summary = pd.concat([read_row, algorithm_summary], ignore_index=True)

    factors = [int(x.strip()) for x in args.scalability_factors.split(",") if x.strip()]
    scalability = benchmark_scalability(X, factors)
    dashboard = benchmark_dashboard_resources(args.db, args.outputs_ai, args.outputs_desc)

    kmeans_time = float(stage_summary.loc[stage_summary["stage"].eq("kmeans_final_k3"), "duration_seconds"].iloc[0])
    dashboard_time = float(dashboard["duration_seconds"].sum()) if not dashboard.empty else np.nan
    downstream_refresh = read_duration + kmeans_time + dashboard_time
    refresh = pd.DataFrame([
        {
            "indicator": "dashboard_resource_refresh_seconds",
            "value_seconds": dashboard_time,
            "interpretation": "timp estimativ pentru încărcarea resurselor principale în dashboard",
        },
        {
            "indicator": "clustering_refresh_seconds",
            "value_seconds": read_duration + kmeans_time,
            "interpretation": "citire matrice + rerulare KMeans final k=3",
        },
        {
            "indicator": "downstream_refresh_seconds",
            "value_seconds": downstream_refresh,
            "interpretation": "estimare locală pentru citire date, clustering și încărcare resurse dashboard; nu include reconstruirea Excel→SQLite",
        },
    ])

    stage_summary.to_csv(args.output_dir / "performance_stage_summary.csv", index=False)
    scalability.to_csv(args.output_dir / "scalability_kmeans_runtime.csv", index=False)
    dashboard.to_csv(args.output_dir / "dashboard_responsiveness.csv", index=False)
    refresh.to_csv(args.output_dir / "refresh_rate_summary.csv", index=False)

    create_pipeline_diagram(args.output_dir)
    create_use_case_diagram(args.output_dir)
    plot_performance(scalability, stage_summary, args.output_dir)
    save_report(args.output_dir, stage_summary, scalability, dashboard, refresh)

    metadata = {
        "db_path": str(args.db),
        "table": args.table,
        "outputs_ai": str(args.outputs_ai),
        "outputs_desc": str(args.outputs_desc),
        "n_patients": int(len(X)),
        "n_features": int(X.shape[1]),
        "note": "Masuratori locale; rezultatele pot varia in functie de laptop si incarcarea sistemului.",
    }
    (args.output_dir / "performance_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nMasurarea performantei a fost finalizata.")
    print(f"Output: {args.output_dir}")
    print(stage_summary[["stage", "duration_seconds", "throughput_patients_per_second"]].to_string(index=False))


if __name__ == "__main__":
    main()
