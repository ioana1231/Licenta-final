#!/usr/bin/env python3
"""
train_ai_final.py - modul AI pentru licenta

Scop:
- citeste datasetul pregatit pentru clustering din baza SQLite;
- antreneaza modele KMeans pentru mai multe valori k;
- alege k pe baza silhouette score;
- salveaza modelul antrenat, etichetele de cluster, metricile, profilul clusterelor si graficele;
- NU foloseste FACED/BSI ca input pentru clustering.

Rulare uzuala:
    python3 train_ai_final.py

Rulare cu tabela descriptiva:
    python3 train_ai_final.py --table features_clustering_descriptive

Rulare cu alt DB:
    python3 train_ai_final.py --db alta_baza.db
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import Iterable

# Limitam thread-urile ca sa evitam blocaje pe laptopuri mai slabe.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


RANDOM_STATE = 42
DEFAULT_K_MIN = 2
DEFAULT_K_MAX = 8

# Coloane care nu trebuie folosite niciodata ca input direct pentru clustering.
FORBIDDEN_INPUT_SUBSTRINGS = [
    "cnp", "nume", "prenume", "pacient_2", "faced", "bsi", "severitate", "cluster"
]

# Coloane utile pentru profilare clinica. Scriptul le ignora automat pe cele inexistente.
PROFILE_PREFERRED_COLUMNS = [
    "varsta", "sex", "imc", "statut_fumator", "pachete_an_clasa", "expunere_noxe",
    "mmrc_max_final", "tuse_cronica", "expectoratie_cronica", "hemoptizie_vreodata",
    "exacerbari_an_final", "nr_exacerbari_inregistrate", "nr_exacerbari_severe",
    "nr_total_vizite", "nr_spitalizari_final", "pseudomonas_final",
    "bact_colonizare_pseudomonas", "bact_pseudomonas_vreodata", "bact_colonizare_mrsa",
    "hrct_lobi_sever", "hrct_bilateral", "hrct_tip_chistic", "hrct_fibroza",
    "pv_fev1_baseline_pct", "pv_fev1_final_pct", "fev1_min_final",
    "pv_fvc_baseline_pct", "pv_fvc_final_pct", "fvc_min_final",
    "bio_pcr_ex_max", "bio_leu_ex_max", "bio_vsh_stabil_med",
    "como_cardio", "como_diabet", "como_pulmonare", "como_gastro", "como_neoplazii",
    "trat_old", "trat_vni_cpap", "trat_terapie_tripla", "trat_atb_cronic",
    "faced_score_partial", "bsi_score_partial", "faced_score_calculat", "bsi_score_calculat",
]


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Antrenare clustering pentru licenta")
    parser.add_argument("--db", type=Path, default=base_dir / "baza_date_licenta.db", help="Calea catre baza SQLite")
    parser.add_argument("--table", default="features_clustering", help="Tabela de input pentru clustering")
    parser.add_argument("--k-min", type=int, default=DEFAULT_K_MIN, help="k minim testat")
    parser.add_argument("--k-max", type=int, default=DEFAULT_K_MAX, help="k maxim testat")
    parser.add_argument("--output-dir", type=Path, default=base_dir / "outputs_ai", help="Folder output")
    parser.add_argument(
        "--force-k",
        type=int,
        default=None,
        help="Forteaza un anumit numar de clustere pentru analiza finala; metricile pentru toate valorile k raman salvate.",
    )
    return parser.parse_args()


def read_sql_table(conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    try:
        return pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
    except Exception as exc:
        available = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn
        )["name"].tolist()
        raise RuntimeError(f"Nu pot citi tabela {table_name}. Tabele disponibile: {available}") from exc


def existing_table(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    ).fetchone()
    return row is not None


def validate_input_table(df: pd.DataFrame) -> pd.DataFrame:
    if "id_pacient" not in df.columns:
        raise ValueError("Tabela de clustering trebuie sa contina coloana id_pacient.")
    if df["id_pacient"].duplicated().any():
        raise ValueError("Tabela de clustering trebuie sa aiba o singura linie per pacient.")

    X = df.drop(columns=["id_pacient"]).copy()

    forbidden = [
        c for c in X.columns
        if any(bad in c.lower() for bad in FORBIDDEN_INPUT_SUBSTRINGS)
    ]
    if forbidden:
        # Nu oprim automat, dar le eliminam defensiv pentru a evita leakage.
        print(f"⚠ Elimin coloane interzise din input: {forbidden}")
        X = X.drop(columns=forbidden)

    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Inputul pentru KMeans trebuie sa fie numeric. Coloane non-numerice: {non_numeric}")

    if X.isna().any().any():
        missing_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(f"KMeans nu accepta NaN. Coloane cu lipsuri: {missing_cols[:30]}")

    constant_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if constant_cols:
        print(f"⚠ Elimin coloane constante: {constant_cols}")
        X = X.drop(columns=constant_cols)

    return X


def choose_best_k(X: pd.DataFrame, k_min: int, k_max: int) -> tuple[int, pd.DataFrame]:
    max_k = min(k_max, len(X) - 1)
    if k_min > max_k:
        raise ValueError(f"Interval k invalid: k_min={k_min}, k_max={k_max}, n={len(X)}")

    rows = []
    print("\n=== Evaluare KMeans ===")
    for k in range(k_min, max_k + 1):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=30, algorithm="lloyd")
        labels = model.fit_predict(X)
        row = {
            "k": k,
            "inertia": float(model.inertia_),
            "silhouette": float(silhouette_score(X, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
            "davies_bouldin": float(davies_bouldin_score(X, labels)),
        }
        rows.append(row)
        print(
            f"k={k}: inertia={row['inertia']:.2f}, "
            f"silhouette={row['silhouette']:.4f}, "
            f"CH={row['calinski_harabasz']:.2f}, DB={row['davies_bouldin']:.4f}"
        )

    metrics = pd.DataFrame(rows)
    best_k = int(metrics.sort_values(["silhouette", "calinski_harabasz"], ascending=[False, False]).iloc[0]["k"])
    return best_k, metrics


def build_profile(
    ids: pd.Series,
    labels: np.ndarray,
    conn: sqlite3.Connection,
    input_table: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    label_df = pd.DataFrame({"id_pacient": ids.astype(int), "cluster": labels.astype(int)})

    # Pentru profilare folosim, daca exista, features_ai sau tabela raw aferenta inputului.
    if existing_table(conn, "features_ai"):
        profile_source = read_sql_table(conn, "features_ai")
    elif existing_table(conn, f"{input_table}_raw"):
        profile_source = read_sql_table(conn, f"{input_table}_raw")
    else:
        profile_source = read_sql_table(conn, input_table)

    cols = ["id_pacient"] + [c for c in PROFILE_PREFERRED_COLUMNS if c in profile_source.columns]
    if len(cols) == 1:
        # fallback: toate numericile, daca nu avem coloane preferate.
        numeric_cols = profile_source.select_dtypes(include=[np.number]).columns.tolist()
        cols = ["id_pacient"] + [c for c in numeric_cols if c != "id_pacient"]

    profile_base = profile_source[cols].merge(label_df, on="id_pacient", how="inner")

    numeric_cols = [
        c for c in profile_base.select_dtypes(include=[np.number]).columns
        if c not in ["id_pacient", "cluster"]
    ]
    cluster_profile = profile_base.groupby("cluster")[numeric_cols].mean().round(3)
    cluster_profile.insert(0, "nr_pacienti", profile_base.groupby("cluster")["id_pacient"].count())
    cluster_profile = cluster_profile.reset_index()

    # Top features distinctive: diferenta medie cluster vs global, normalizata prin std global.
    global_mean = profile_base[numeric_cols].mean(numeric_only=True)
    global_std = profile_base[numeric_cols].std(ddof=0).replace(0, np.nan)
    top_rows = []
    for cluster_id, group in profile_base.groupby("cluster"):
        diff = ((group[numeric_cols].mean(numeric_only=True) - global_mean) / global_std).dropna()
        top_positive = diff.sort_values(ascending=False).head(10)
        top_negative = diff.sort_values(ascending=True).head(10)
        for feature, z_diff in top_positive.items():
            top_rows.append({
                "cluster": int(cluster_id),
                "sens": "mai mare decat media",
                "feature": feature,
                "diferenta_standardizata": round(float(z_diff), 3),
            })
        for feature, z_diff in top_negative.items():
            top_rows.append({
                "cluster": int(cluster_id),
                "sens": "mai mic decat media",
                "feature": feature,
                "diferenta_standardizata": round(float(z_diff), 3),
            })
    top_features = pd.DataFrame(top_rows)

    return label_df, cluster_profile, top_features


def save_plots(metrics: pd.DataFrame, results: pd.DataFrame, output_dir: Path, explained: np.ndarray) -> None:
    plt.figure()
    plt.plot(metrics["k"], metrics["inertia"], marker="o")
    plt.xlabel("Numar de clustere k")
    plt.ylabel("Inertia")
    plt.title("Metoda elbow pentru KMeans")
    plt.tight_layout()
    plt.savefig(output_dir / "elbow_plot.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(metrics["k"], metrics["silhouette"], marker="o")
    plt.xlabel("Numar de clustere k")
    plt.ylabel("Silhouette score")
    plt.title("Evaluarea silhouette pentru KMeans")
    plt.tight_layout()
    plt.savefig(output_dir / "silhouette_plot.png", dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    cluster_values = sorted(results["cluster"].dropna().unique())
    cmap = plt.get_cmap("tab10")
    for index, cluster_id in enumerate(cluster_values):
        points = results[results["cluster"].eq(cluster_id)]
        ax.scatter(
            points["pca_1"],
            points["pca_2"],
            color=cmap(index % 10),
            label=f"Cluster {cluster_id}",
            alpha=0.85,
            edgecolors="white",
            linewidths=0.35,
        )
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% varianta)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% varianta)")
    ax.set_title("Vizualizare PCA a clusterelor")
    ax.legend(title="Cluster", loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "pca_clusters.png", dpi=200)
    plt.close(fig)

    counts = results["cluster"].value_counts().sort_index()
    plt.figure()
    plt.bar(counts.index.astype(str), counts.values)
    plt.xlabel("Cluster")
    plt.ylabel("Numar pacienti")
    plt.title("Dimensiunea clusterelor")
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_sizes.png", dpi=200)
    plt.close()


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


def save_markdown_report(
    output_dir: Path,
    db_path: Path,
    input_table: str,
    best_k: int,
    metrics: pd.DataFrame,
    cluster_profile: pd.DataFrame,
    top_features: pd.DataFrame,
) -> None:
    best_row = metrics.loc[metrics["k"].eq(best_k)].iloc[0]
    lines = [
        "# Raport antrenare AI - clustering",
        "",
        f"Baza de date: `{db_path}`",
        f"Tabela input: `{input_table}`",
        f"Algoritm: `KMeans`",
        f"k utilizat: `{best_k}`",
        f"Silhouette score: `{best_row['silhouette']:.4f}`",
        f"Calinski-Harabasz: `{best_row['calinski_harabasz']:.2f}`",
        f"Davies-Bouldin: `{best_row['davies_bouldin']:.4f}`",
        "",
        "## Interpretare metodologica",
        "",
        "Clusterizarea este folosita exploratoriu/descriptiv. Ea nu reprezinta diagnostic automat si necesita validare clinica.",
        "FACED/BSI si severitatile derivate nu sunt folosite ca input al algoritmului, pentru a evita leakage-ul.",
        "",
        "## Profil clustere",
        "",
        dataframe_to_markdown(cluster_profile),
        "",
        "## Top caracteristici distinctive",
        "",
    ]
    if not top_features.empty:
        for cluster in sorted(top_features["cluster"].unique()):
            lines.append(f"### Cluster {cluster}")
            lines.append("")
            subset = top_features[top_features["cluster"] == cluster].head(12)
            lines.append(dataframe_to_markdown(subset))
            lines.append("")
    (output_dir / "ai_training_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.db.exists():
        raise FileNotFoundError(f"Nu gasesc baza de date: {args.db}")

    with sqlite3.connect(args.db) as conn:
        feature_table = read_sql_table(conn, args.table)
        X = validate_input_table(feature_table)
        ids = feature_table["id_pacient"]

        auto_k, metrics = choose_best_k(X, args.k_min, args.k_max)
        if args.force_k is not None:
            if args.force_k < args.k_min or args.force_k > args.k_max:
                raise ValueError(f"--force-k trebuie sa fie in intervalul [{args.k_min}, {args.k_max}].")
            best_k = args.force_k
            print(f"\n k ales automat dupa silhouette: {auto_k}")
            print(f" k folosit pentru analiza finala: {best_k} (ales manual pentru interpretabilitate)")
        else:
            best_k = auto_k
            print(f"\n k ales automat: {best_k}")

        model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=50, algorithm="lloyd")
        labels = model.fit_predict(X)

        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        coords = pca.fit_transform(X)
        explained = pca.explained_variance_ratio_

        results = pd.DataFrame({
            "id_pacient": ids.astype(int),
            "cluster": labels.astype(int),
            "pca_1": coords[:, 0],
            "pca_2": coords[:, 1],
        })

        label_df, cluster_profile, top_features = build_profile(ids, labels, conn, args.table)

        # Salvare in SQLite.
        results.to_sql("ai_cluster_labels", conn, if_exists="replace", index=False)
        cluster_profile.to_sql("ai_cluster_profile", conn, if_exists="replace", index=False)
        metrics.to_sql("ai_cluster_metrics", conn, if_exists="replace", index=False)
        top_features.to_sql("ai_cluster_top_features", conn, if_exists="replace", index=False)

    # Salvare fisiere.
    results.to_csv(args.output_dir / "ai_cluster_labels.csv", index=False)
    cluster_profile.to_csv(args.output_dir / "ai_cluster_profile.csv", index=False)
    metrics.to_csv(args.output_dir / "ai_cluster_metrics.csv", index=False)
    top_features.to_csv(args.output_dir / "ai_cluster_top_features.csv", index=False)

    model_payload = {
        "model": model,
        "feature_names": X.columns.tolist(),
        "input_table": args.table,
        "best_k": best_k,
        "random_state": RANDOM_STATE,
    }
    joblib.dump(model_payload, args.output_dir / "kmeans_model.joblib")

    metadata = {
        "db_path": str(args.db),
        "input_table": args.table,
        "n_patients": int(len(X)),
        "n_features": int(X.shape[1]),
        "auto_k": int(auto_k),
        "best_k": int(best_k),
        "selection_mode": "manual_force_k" if args.force_k is not None else "automatic_silhouette",
        "random_state": RANDOM_STATE,
        "note": "Clustering exploratoriu; nu diagnostic automat.",
    }
    (args.output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    save_plots(metrics, results, args.output_dir, explained)
    save_markdown_report(args.output_dir, args.db, args.table, best_k, metrics, cluster_profile, top_features)

    print("\n=== Output generat ===")
    print(f"Folder: {args.output_dir}")
    print("- kmeans_model.joblib")
    print("- ai_cluster_labels.csv")
    print("- ai_cluster_profile.csv")
    print("- ai_cluster_metrics.csv")
    print("- ai_cluster_top_features.csv")
    print("- ai_training_report.md")
    print("- elbow_plot.png, silhouette_plot.png, pca_clusters.png, cluster_sizes.png")
    print("\n=== Profil clustere ===")
    print(cluster_profile.to_string(index=False))


if __name__ == "__main__":
    main()
