#!/usr/bin/env python3
"""Validare finala pentru pipeline-ul de licenta.

Scriptul verifica baza SQLite si fisierele generate dupa rularea completa:
    1. build_features_v4_final_audit.py
    2. train_ai_final.py --force-k 3
    3. generate_descriptive_analysis.py

Verificari incluse:
- existenta tabelelor asteptate in SQLite;
- validarea schemei minimale pentru tabelele principale;
- corespondenta numarului de pacienti intre pacienti, features_ai, features_clustering si ai_cluster_labels;
- lipsa identificatorilor directi in tabelele de features/clustering;
- lipsa valorilor NaN in matricea folosita de KMeans;
- verificari de sanity pentru clustere;
- verificarea metricilor KMeans;
- test de stabilitate prin rularea KMeans cu seed-uri diferite si calcularea ARI fata de etichetele finale.

Rulare:
    python validate_pipeline_outputs.py --db baza_date_licenta.db --outputs-ai outputs_ai --outputs-desc outputs_descriptive_analysis
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

FORBIDDEN_SUBSTRINGS = ("cnp", "nume", "prenume")
FORBIDDEN_CLUSTER_INPUT_SUBSTRINGS = ("cnp", "nume", "prenume", "faced", "bsi", "severitate", "cluster")

EXPECTED_TABLES = [
    "pacienti",
    "pacienti_clean",
    "simptome_features",
    "bacteriologie_features",
    "etiologie_features",
    "comorbiditati_features",
    "hrct_features",
    "tratament_features",
    "vizite_agg",
    "exacerbari_agg",
    "probe_ventilatorii_features",
    "biologic_features",
    "scoruri_features",
    "features_ai",
    "features_clustering",
    "features_clustering_raw",
    "features_clustering_descriptive",
    "features_clustering_descriptive_raw",
    "data_quality_report",
    "feature_quality_report",
    "clustering_feature_report",
    "ai_cluster_labels",
    "ai_cluster_metrics",
    "ai_cluster_profile",
    "ai_cluster_top_features",
]

EXPECTED_COLUMNS = {
    "pacienti": ["id_pacient", "sex", "varsta", "imc", "statut_fumator"],
    "features_ai": [
        "id_pacient", "sex", "varsta", "imc", "fev1_min_final", "fvc_min_final",
        "pseudomonas_final", "faced_score_partial", "bsi_score_partial",
    ],
    "features_clustering": [
        "id_pacient", "varsta", "imc", "mmrc_max_final", "exacerbari_an_final",
        "pseudomonas_final", "fev1_min_final", "fvc_min_final",
    ],
    "ai_cluster_labels": ["id_pacient", "cluster", "pca_1", "pca_2"],
    "ai_cluster_metrics": ["k", "inertia", "silhouette", "calinski_harabasz", "davies_bouldin"],
    "ai_cluster_top_features": ["cluster", "sens", "feature", "diferenta_standardizata"],
}


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validare pipeline licenta")
    parser.add_argument("--db", type=Path, default=Path("baza_date_licenta.db"), help="Calea catre baza SQLite")
    parser.add_argument("--outputs-ai", type=Path, default=Path("outputs_ai"), help="Folderul outputs_ai")
    parser.add_argument("--outputs-desc", type=Path, default=Path("outputs_descriptive_analysis"), help="Folderul outputs_descriptive_analysis")
    parser.add_argument("--outputs-eng", type=Path, default=Path("outputs_engineering"), help="Folderul outputs_engineering")
    parser.add_argument("--expected-patients", type=int, default=169, help="Numar asteptat de pacienti")
    parser.add_argument("--expected-k", type=int, default=3, help="Numar final de clustere asteptat")
    parser.add_argument("--stability-runs", type=int, default=30, help="Numar rulari pentru stabilitate")
    parser.add_argument("--min-cluster-size", type=int, default=5, help="Dimensiune minima acceptata pentru un cluster")
    parser.add_argument("--report", type=Path, default=Path("validation_report.md"), help="Raport Markdown generat")
    return parser.parse_args()


def list_tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {row[0] for row in rows}


def read_table(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    return pd.read_sql_query(f'SELECT * FROM "{table}"', conn)


def add(results: list[CheckResult], name: str, ok: bool, details: str) -> None:
    results.append(CheckResult(name=name, ok=ok, details=details))


def validate_tables(conn: sqlite3.Connection, results: list[CheckResult]) -> None:
    tables = list_tables(conn)
    missing = [table for table in EXPECTED_TABLES if table not in tables]
    add(results, "Tabele SQLite asteptate", not missing, "Lipsa: " + ", ".join(missing) if missing else f"Toate cele {len(EXPECTED_TABLES)} tabele asteptate exista.")

    for table, expected_cols in EXPECTED_COLUMNS.items():
        if table not in tables:
            add(results, f"Schema minima {table}", False, "Tabela lipseste.")
            continue
        cols = read_table(conn, table).columns.tolist()
        missing_cols = [col for col in expected_cols if col not in cols]
        add(results, f"Schema minima {table}", not missing_cols, "Lipsesc coloane: " + ", ".join(missing_cols) if missing_cols else "Coloanele minimale sunt prezente.")


def validate_patient_counts(conn: sqlite3.Connection, results: list[CheckResult], expected_patients: int) -> None:
    tables = list_tables(conn)
    patient_tables = ["pacienti", "features_ai", "features_clustering"]
    counts = {}
    ids = {}
    for table in patient_tables:
        df = read_table(conn, table)
        counts[table] = len(df)
        ids[table] = set(df["id_pacient"].astype(int))
        add(results, f"ID unic in {table}", not df["id_pacient"].duplicated().any(), f"Randuri: {len(df)}, pacienti unici: {df['id_pacient'].nunique()}")

    same_counts = len(set(counts.values())) == 1 and counts["features_clustering"] == expected_patients
    add(results, "Numar pacienti intre tabele", same_counts, f"{counts}; asteptat features_clustering={expected_patients}")

    same_ids = ids["pacienti"] == ids["features_ai"] == ids["features_clustering"]
    add(results, "Corespondenta ID pacienti", same_ids, "Seturile de id_pacient coincid." if same_ids else "Exista diferente intre seturile de id_pacient.")

    if "ai_cluster_labels" in tables:
        labels = read_table(conn, "ai_cluster_labels")
        label_ids = set(labels["id_pacient"].astype(int))
        add(results, "Corespondenta etichete-clustering", label_ids == ids["features_clustering"], f"labels={len(label_ids)}, features_clustering={len(ids['features_clustering'])}")


def validate_features(conn: sqlite3.Connection, results: list[CheckResult]) -> None:
    features_ai = read_table(conn, "features_ai")
    clustering = read_table(conn, "features_clustering")

    bad_ai_cols = [c for c in features_ai.columns if any(token in c.lower() for token in FORBIDDEN_SUBSTRINGS)]
    add(results, "Fara identificatori directi in features_ai", not bad_ai_cols, "Coloane problematice: " + ", ".join(bad_ai_cols) if bad_ai_cols else "Nu exista CNP/nume/prenume.")

    bad_clustering_cols = [c for c in clustering.columns if c != "id_pacient" and any(token in c.lower() for token in FORBIDDEN_CLUSTER_INPUT_SUBSTRINGS)]
    add(results, "Fara leakage in features_clustering", not bad_clustering_cols, "Coloane problematice: " + ", ".join(bad_clustering_cols) if bad_clustering_cols else "Nu exista identificatori, scoruri FACED/BSI, severitati sau cluster in input.")

    X = clustering.drop(columns=["id_pacient"])
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    add(results, "features_clustering numeric", not non_numeric, "Coloane non-numerice: " + ", ".join(non_numeric) if non_numeric else "Toate coloanele de input sunt numerice.")

    nan_count = int(X.isna().sum().sum())
    add(results, "Fara NaN in features_clustering", nan_count == 0, f"Numar valori NaN: {nan_count}")

    constant_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    add(results, "Fara coloane constante in features_clustering", not constant_cols, "Coloane constante: " + ", ".join(constant_cols[:20]) if constant_cols else "Nu exista coloane constante.")


def validate_clusters(conn: sqlite3.Connection, results: list[CheckResult], expected_k: int, min_cluster_size: int) -> None:
    labels = read_table(conn, "ai_cluster_labels")
    metrics = read_table(conn, "ai_cluster_metrics")
    top_features = read_table(conn, "ai_cluster_top_features")

    unique_clusters = sorted(labels["cluster"].astype(int).unique().tolist())
    add(results, "Numar clustere final", len(unique_clusters) == expected_k, f"Clustere gasite: {unique_clusters}; asteptat k={expected_k}")

    sizes = labels["cluster"].value_counts().sort_index()
    small = sizes[sizes < min_cluster_size]
    add(results, "Dimensiuni clustere", small.empty, "Dimensiuni: " + sizes.to_dict().__repr__())

    required_k = set(range(2, 9))
    present_k = set(metrics["k"].astype(int).tolist())
    add(results, "Metrici pentru k=2..8", required_k.issubset(present_k), f"k prezente: {sorted(present_k)}")

    finite_metrics = np.isfinite(metrics[["inertia", "silhouette", "calinski_harabasz", "davies_bouldin"]].to_numpy(dtype=float)).all()
    add(results, "Metrici finite", bool(finite_metrics), "Metricile KMeans nu contin NaN/inf." if finite_metrics else "Exista metrici NaN/inf.")

    enough_top = not top_features.empty and top_features.groupby("cluster").size().min() >= 5
    add(results, "Top features pe cluster", bool(enough_top), f"Randuri top_features: {len(top_features)}; distributie: {top_features.groupby('cluster').size().to_dict() if not top_features.empty else {}}")


def validate_output_files(results: list[CheckResult], outputs_ai: Path, outputs_desc: Path, outputs_eng: Path) -> None:
    expected_ai_files = [
        "ai_cluster_labels.csv", "ai_cluster_metrics.csv", "ai_cluster_profile.csv",
        "ai_cluster_top_features.csv", "training_metadata.json", "elbow_plot.png",
        "silhouette_plot.png", "pca_clusters.png", "cluster_sizes.png",
    ]
    missing_ai = [name for name in expected_ai_files if not (outputs_ai / name).exists()]
    add(results, "Fisiere outputs_ai", not missing_ai, "Lipsa: " + ", ".join(missing_ai) if missing_ai else "Fisierele principale outputs_ai exista.")

    recommended_dirs = [outputs_desc / "tables_recommended", outputs_desc / "figures_recommended"]
    missing_dirs = [str(path) for path in recommended_dirs if not path.exists()]
    add(results, "Foldere descriptive recomandate", not missing_dirs, "Lipsa: " + ", ".join(missing_dirs) if missing_dirs else "tables_recommended si figures_recommended exista.")

    expected_desc_files = [
        outputs_desc / "tables_recommended" / "table6_cluster_profile_with_variation_and_pvalues.csv",
        outputs_desc / "tables_recommended" / "stat_tests_continuous_by_cluster.csv",
        outputs_desc / "tables_recommended" / "stat_tests_categorical_chi_square_by_cluster.csv",
        outputs_desc / "tables_recommended" / "age_group_distribution_overall.csv",
        outputs_desc / "tables_recommended" / "age_group_distribution_by_sex.csv",
        outputs_desc / "tables_recommended" / "comorbidity_distribution_by_age_group.csv",
        outputs_desc / "tables_recommended" / "covariance_key_variables.csv",
        outputs_desc / "tables_recommended" / "faced_severity_distribution_calculable_only.csv",
        outputs_desc / "tables_recommended" / "bsi_severity_distribution_calculable_only.csv",
        outputs_desc / "figures_recommended" / "heatmap_cluster_profile.png",
        outputs_desc / "figures_recommended" / "sqlite_logical_er_diagram.png",
        outputs_desc / "figures_recommended" / "age_group_distribution_by_sex.png",
        outputs_desc / "figures_recommended" / "comorbidity_by_age_group_heatmap.png",
        outputs_desc / "figures_recommended" / "covariance_key_variables_heatmap.png",
    ]
    missing_desc = [str(path) for path in expected_desc_files if not path.exists()]
    add(results, "Fisiere descriptive cheie", not missing_desc, "Lipsa: " + ", ".join(missing_desc) if missing_desc else "Fisierele descriptive cheie exista.")



    expected_eng_files = [
        "clustering_algorithm_comparison.csv",
        "clustering_algorithm_best_by_method.csv",
        "dbscan_parameter_grid.csv",
        "performance_stage_summary.csv",
        "scalability_kmeans_runtime.csv",
        "dashboard_responsiveness.csv",
        "refresh_rate_summary.csv",
        "data_pipeline_diagram.png",
        "use_case_diagram.png",
        "stage_runtime_summary.png",
        "kmeans_scalability_runtime.png",
    ]
    missing_eng = [name for name in expected_eng_files if not (outputs_eng / name).exists()]
    add(results, "Fisiere engineering", not missing_eng, "Lipsa: " + ", ".join(missing_eng) if missing_eng else "Fisierele principale outputs_engineering exista.")

    meta_path = outputs_ai / "training_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ok = meta.get("best_k") == 3 and meta.get("selection_mode") == "manual_force_k"
        add(results, "Metadata clustering final", ok, f"best_k={meta.get('best_k')}, auto_k={meta.get('auto_k')}, selection_mode={meta.get('selection_mode')}")


def validate_stability(conn: sqlite3.Connection, results: list[CheckResult], expected_k: int, runs: int) -> pd.DataFrame:
    clustering = read_table(conn, "features_clustering")
    labels = read_table(conn, "ai_cluster_labels")

    X = clustering.drop(columns=["id_pacient"])
    ref = labels.sort_values("id_pacient")["cluster"].astype(int).to_numpy()
    ids = labels.sort_values("id_pacient")["id_pacient"].astype(int).tolist()
    X = clustering.set_index("id_pacient").loc[ids].reset_index(drop=True)

    rows = []
    for seed in range(runs):
        model = KMeans(n_clusters=expected_k, random_state=seed, n_init=50, algorithm="lloyd")
        pred = model.fit_predict(X)
        row = {
            "seed": seed,
            "adjusted_rand_index_vs_final": float(adjusted_rand_score(ref, pred)),
            "silhouette": float(silhouette_score(X, pred)),
            "inertia": float(model.inertia_),
        }
        counts = pd.Series(pred).value_counts().sort_index().to_dict()
        row["cluster_sizes"] = str(counts)
        rows.append(row)

    stability = pd.DataFrame(rows)
    min_ari = float(stability["adjusted_rand_index_vs_final"].min())
    mean_ari = float(stability["adjusted_rand_index_vs_final"].mean())
    sil_min = float(stability["silhouette"].min())
    sil_max = float(stability["silhouette"].max())

    # Pragul este intentionat moderat: silhouette-ul mic poate produce variatie intre rulari.
    # Raportam stabilitatea ca diagnostic metodologic, nu ca dovada absoluta.
    ok = mean_ari >= 0.50
    add(
        results,
        "Stabilitate clustering multi-seed",
        ok,
        f"ARI mediu={mean_ari:.3f}, ARI minim={min_ari:.3f}, silhouette interval=[{sil_min:.4f}, {sil_max:.4f}], rulari={runs}",
    )
    return stability


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


def write_report(path: Path, results: list[CheckResult], stability: pd.DataFrame | None) -> None:
    passed = sum(r.ok for r in results)
    failed = len(results) - passed
    lines = [
        "# Raport validare pipeline",
        "",
        f"Verificari trecute: **{passed}/{len(results)}**",
        f"Verificari esuate/avertismente: **{failed}**",
        "",
        "## Rezumat verificari",
        "",
        "| Verificare | Status | Detalii |",
        "|---|---:|---|",
    ]
    for result in results:
        status = "PASS" if result.ok else "CHECK"
        details = result.details.replace("|", "\\|").replace("\n", "<br>")
        lines.append(f"| {result.name} | {status} | {details} |")

    if stability is not None:
        lines.extend([
            "",
            "## Stabilitate KMeans pe seed-uri diferite",
            "",
            "Tabelul de mai jos compara etichetele obtinute la seed-uri diferite cu etichetele finale salvate, folosind Adjusted Rand Index. Valorile apropiate de 1 indica asignari foarte similare, iar valorile apropiate de 0 indica asemanare redusa peste nivelul aleator.",
            "",
            dataframe_to_markdown(stability),
        ])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.db.exists():
        raise FileNotFoundError(f"Nu gasesc baza SQLite: {args.db}")

    results: list[CheckResult] = []
    with sqlite3.connect(args.db) as conn:
        validate_tables(conn, results)
        validate_patient_counts(conn, results, args.expected_patients)
        validate_features(conn, results)
        validate_clusters(conn, results, args.expected_k, args.min_cluster_size)
        stability = validate_stability(conn, results, args.expected_k, args.stability_runs)

    validate_output_files(results, args.outputs_ai, args.outputs_desc, args.outputs_eng)
    write_report(args.report, results, stability)

    passed = sum(r.ok for r in results)
    total = len(results)
    print(f"Validare finalizata: {passed}/{total} verificari PASS")
    print(f"Raport scris la: {args.report}")
    for result in results:
        icon = "✅" if result.ok else "⚠️"
        print(f"{icon} {result.name}: {result.details}")

    # Nu iesim cu cod de eroare pentru CHECK, ca unele rezultate pot fi discutate metodologic.
    # Pentru CI strict, decomenteaza linia de mai jos.
    # raise SystemExit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
