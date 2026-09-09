#!/usr/bin/env python3
"""
Comparare metode de clustering pentru lucrarea de licenta.

Scop:
- foloseste aceeasi matrice features_clustering ca analiza KMeans finala;
- compara KMeans, Agglomerative/Hierarchical Clustering si DBSCAN;
- calculeaza scoruri interne: silhouette, Calinski-Harabasz, Davies-Bouldin;
- masoara timpul de rulare pentru fiecare configuratie;
- salveaza tabele si grafice pentru documentatie/dashboard.

Rulare:
    python compare_clustering_methods.py --db baza_date_licenta.db --output-dir outputs_engineering
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Iterable

# limitare thread-uri pentru rezultate mai stabile pe laptopuri
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

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
    parser = argparse.ArgumentParser(description="Comparare KMeans vs hierarchical vs DBSCAN")
    parser.add_argument("--db", type=Path, default=base_dir / "baza_date_licenta.db")
    parser.add_argument("--table", default="features_clustering")
    parser.add_argument("--output-dir", type=Path, default=base_dir / "outputs_engineering")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--dbscan-min-samples", type=int, default=5)
    return parser.parse_args()


def read_feature_matrix(db_path: Path, table_name: str) -> tuple[pd.Series, pd.DataFrame]:
    if not db_path.exists():
        raise FileNotFoundError(f"Nu gasesc baza SQLite: {db_path}")
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
    if "id_pacient" not in df.columns:
        raise ValueError(f"Tabela {table_name} trebuie sa contina id_pacient.")
    ids = df["id_pacient"].copy()
    X = df.drop(columns=["id_pacient"]).copy()
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Tabela {table_name} trebuie sa fie numerica. Coloane non-numerice: {non_numeric}")
    if X.isna().any().any():
        missing = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Tabela {table_name} contine NaN. Coloane cu lipsuri: {missing[:20]}")
    return ids, X


def valid_for_scores(labels: np.ndarray, ignore_noise: bool = False) -> tuple[bool, np.ndarray, str]:
    labels = np.asarray(labels)
    if ignore_noise:
        mask = labels != -1
        used = labels[mask]
        scope = "non_noise"
    else:
        mask = np.ones(len(labels), dtype=bool)
        used = labels
        scope = "all_points"
    unique = np.unique(used)
    if len(used) < 3 or len(unique) < 2 or len(unique) >= len(used):
        return False, mask, scope
    return True, mask, scope


def calculate_scores(X: pd.DataFrame, labels: np.ndarray, ignore_noise: bool = False) -> dict[str, float | str | None]:
    ok, mask, scope = valid_for_scores(labels, ignore_noise=ignore_noise)
    out: dict[str, float | str | None] = {
        "metric_scope": scope,
        "silhouette": np.nan,
        "calinski_harabasz": np.nan,
        "davies_bouldin": np.nan,
    }
    if not ok:
        return out
    X_used = X.to_numpy()[mask]
    labels_used = labels[mask]
    out["silhouette"] = float(silhouette_score(X_used, labels_used))
    out["calinski_harabasz"] = float(calinski_harabasz_score(X_used, labels_used))
    out["davies_bouldin"] = float(davies_bouldin_score(X_used, labels_used))
    return out


def label_summary(labels: np.ndarray) -> dict[str, int | float]:
    labels = np.asarray(labels)
    noise = int((labels == -1).sum())
    non_noise = labels[labels != -1]
    clusters = sorted([int(v) for v in np.unique(non_noise)])
    counts = pd.Series(non_noise).value_counts().sort_index() if len(non_noise) else pd.Series(dtype=int)
    return {
        "n_clusters": int(len(clusters)),
        "noise_points": noise,
        "noise_percent": round(100 * noise / len(labels), 3),
        "min_cluster_size": int(counts.min()) if not counts.empty else 0,
        "max_cluster_size": int(counts.max()) if not counts.empty else 0,
    }


def run_kmeans(X: pd.DataFrame, k_values: Iterable[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for k in k_values:
        start = time.perf_counter()
        model = KMeans(n_clusters=int(k), random_state=RANDOM_STATE, n_init=50, algorithm="lloyd")
        labels = model.fit_predict(X)
        runtime = time.perf_counter() - start
        row = {
            "method": "KMeans",
            "configuration": f"k={k}",
            "k": int(k),
            "eps": np.nan,
            "min_samples": np.nan,
            "runtime_seconds": runtime,
            "inertia": float(model.inertia_),
            **label_summary(labels),
            **calculate_scores(X, labels, ignore_noise=False),
        }
        rows.append(row)
    return rows


def run_hierarchical(X: pd.DataFrame, k_values: Iterable[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for k in k_values:
        start = time.perf_counter()
        model = AgglomerativeClustering(n_clusters=int(k), linkage="ward")
        labels = model.fit_predict(X)
        runtime = time.perf_counter() - start
        row = {
            "method": "Hierarchical/Ward",
            "configuration": f"k={k}",
            "k": int(k),
            "eps": np.nan,
            "min_samples": np.nan,
            "runtime_seconds": runtime,
            "inertia": np.nan,
            **label_summary(labels),
            **calculate_scores(X, labels, ignore_noise=False),
        }
        rows.append(row)
    return rows


def estimate_dbscan_eps_values(X: pd.DataFrame, min_samples: int) -> list[float]:
    n_neighbors = min(max(2, min_samples), len(X) - 1)
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    kth = np.sort(distances[:, -1])
    quantiles = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98]
    values = np.quantile(kth, quantiles)
    values = [float(round(v, 4)) for v in values if np.isfinite(v) and v > 0]
    # Adaugam cateva valori fixe utile in spatii standardizate/mixte.
    values.extend([1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
    return sorted(set(values))


def run_dbscan(X: pd.DataFrame, eps_values: Iterable[float], min_samples: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for eps in eps_values:
        start = time.perf_counter()
        model = DBSCAN(eps=float(eps), min_samples=int(min_samples))
        labels = model.fit_predict(X)
        runtime = time.perf_counter() - start
        row = {
            "method": "DBSCAN",
            "configuration": f"eps={eps}, min_samples={min_samples}",
            "k": np.nan,
            "eps": float(eps),
            "min_samples": int(min_samples),
            "runtime_seconds": runtime,
            "inertia": np.nan,
            **label_summary(labels),
            # Pentru DBSCAN, punctele marcate -1 sunt zgomot; scorurile sunt calculate pe punctele non-noise.
            **calculate_scores(X, labels, ignore_noise=True),
        }
        rows.append(row)
    return rows


def choose_best_per_method(results: pd.DataFrame) -> pd.DataFrame:
    best_rows = []
    for method, group in results.groupby("method", sort=False):
        valid = group.dropna(subset=["silhouette"]).copy()
        if valid.empty:
            row = group.iloc[0].copy()
            row["selection_reason"] = "nu exista configuratie valida pentru scoruri interne"
        else:
            # prioritate: silhouette mare, Davies-Bouldin mic, Calinski mare
            row = valid.sort_values(
                ["silhouette", "davies_bouldin", "calinski_harabasz"],
                ascending=[False, True, False],
            ).iloc[0].copy()
            row["selection_reason"] = "cel mai bun silhouette valid in cadrul metodei"
        best_rows.append(row)
    return pd.DataFrame(best_rows)


def plot_metric_comparison(best: pd.DataFrame, output_dir: Path) -> None:
    valid = best.dropna(subset=["silhouette"]).copy()
    if valid.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(valid["method"].astype(str), valid["silhouette"])
    ax.set_title("Comparare metode de clustering - silhouette")
    ax.set_ylabel("Silhouette score")
    ax.set_xlabel("Metodă")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "clustering_methods_silhouette_comparison.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(valid["method"].astype(str), valid["runtime_seconds"])
    ax.set_title("Comparare metode de clustering - timp de rulare")
    ax.set_ylabel("Timp rulare (secunde)")
    ax.set_xlabel("Metodă")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "clustering_methods_runtime_comparison.png", dpi=200)
    plt.close(fig)


def save_report(output_dir: Path, results: pd.DataFrame, best: pd.DataFrame, n_patients: int, n_features: int) -> None:
    lines = [
        "# Comparare metode de clustering",
        "",
        f"Pacienți analizați: **{n_patients}**",
        f"Caracteristici utilizate: **{n_features}**",
        "",
        "Metode evaluate: KMeans, hierarchical clustering cu legătură Ward și DBSCAN.",
        "Scorurile interne calculate sunt silhouette, Calinski-Harabasz și Davies-Bouldin.",
        "Pentru DBSCAN, punctele marcate ca zgomot (-1) nu sunt incluse în calculul scorurilor interne.",
        "",
        "## Configurații recomandate pe metodă",
        "",
        dataframe_to_markdown(best),
        "",
        "## Observație metodologică",
        "",
        "Compararea algoritmilor are rol exploratoriu și urmărește susținerea alegerii metodei finale din perspectivă computațională. Rezultatele nu reprezintă validare clinică a clusterelor.",
    ]
    (output_dir / "clustering_methods_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ids, X = read_feature_matrix(args.db, args.table)
    k_max = min(args.k_max, len(X) - 1)
    k_values = list(range(args.k_min, k_max + 1))

    rows: list[dict[str, object]] = []
    rows.extend(run_kmeans(X, k_values))
    rows.extend(run_hierarchical(X, k_values))
    eps_values = estimate_dbscan_eps_values(X, args.dbscan_min_samples)
    rows.extend(run_dbscan(X, eps_values, args.dbscan_min_samples))

    results = pd.DataFrame(rows)
    best = choose_best_per_method(results)

    results.to_csv(args.output_dir / "clustering_algorithm_comparison.csv", index=False)
    best.to_csv(args.output_dir / "clustering_algorithm_best_by_method.csv", index=False)
    results[results["method"].eq("DBSCAN")].to_csv(args.output_dir / "dbscan_parameter_grid.csv", index=False)

    plot_metric_comparison(best, args.output_dir)
    save_report(args.output_dir, results, best, len(ids), X.shape[1])

    metadata = {
        "db_path": str(args.db),
        "table": args.table,
        "n_patients": int(len(ids)),
        "n_features": int(X.shape[1]),
        "k_values": k_values,
        "dbscan_min_samples": int(args.dbscan_min_samples),
        "dbscan_eps_values": eps_values,
        "note": "Comparare exploratorie a metodelor de clustering; nu reprezinta validare clinica.",
    }
    (args.output_dir / "clustering_algorithm_comparison_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nCompararea algoritmilor a fost finalizata.")
    print(f"Output: {args.output_dir}")
    print(best[["method", "configuration", "n_clusters", "noise_points", "silhouette", "runtime_seconds"]].to_string(index=False))


if __name__ == "__main__":
    main()
