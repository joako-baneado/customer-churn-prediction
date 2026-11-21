"""
Módulo de Segmentación No Supervisada y Reducción Dimensional (TF1 - CC209).
Compara K-Means, Gaussian Mixture Models (GMM) y Clustering Jerárquico (Ward)
ajustando todas las transformaciones exclusivamente sobre Train + Validation (Dev)
para preservar cero data leakage frente al conjunto Test.

Nota metodológica: los segmentos representan perfiles/proxies de valor y
vinculación contractual con base en variables disponibles (tenure, MonthlyCharges,
TotalCharges, TotalServicesSubscribed), no un RFM o CLV transaccional real.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer

from src.config import (
    CLUSTERING_FEATURES,
    FIGURES_TF1_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
)
from src.features import TelcoFeatureEngineer
from src.preprocessing import get_dev_and_test_splits


CLUSTER_LABELS_BY_K = {
    3: [
        "Bajo Valor / Entrada",
        "Valor Medio / Estándar",
        "Alto Valor / Consolidado",
    ],
    4: [
        "Bajo Valor / Entrada",
        "Valor Medio / Reciente",
        "Valor Medio-Alto / Estable",
        "Alto Valor / Premium",
    ],
}


class ValueClusterSegmenter:
    """
    Wrapper reproducible de segmentación de valor para inferencia online y batch.
    Encapsula feature engineering, escalado robusto, modelo de clustering,
    mapeo determinístico de etiquetas ordenadas por valor y proyección PCA 2D.
    """

    def __init__(
        self,
        feature_engineer: TelcoFeatureEngineer,
        scaler_pipeline: Pipeline,
        cluster_model,
        pca_model: PCA,
        cluster_mapping: Dict[int, int],
        label_mapping: Dict[int, str],
        algorithm_name: str,
        n_clusters: int,
    ):
        self.feature_engineer = feature_engineer
        self.scaler_pipeline = scaler_pipeline
        self.cluster_model = cluster_model
        self.pca_model = pca_model
        self.cluster_mapping = cluster_mapping
        self.label_mapping = label_mapping
        self.algorithm_name = algorithm_name
        self.n_clusters = n_clusters

    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        df_fe = self.feature_engineer.transform(df)
        X_clust = df_fe[CLUSTERING_FEATURES].copy()
        return self.scaler_pipeline.transform(X_clust)

    def predict_cluster_ids(self, df: pd.DataFrame) -> np.ndarray:
        X_scaled = self.transform_features(df)
        raw_preds = self.cluster_model.predict(X_scaled)
        return np.asarray([self.cluster_mapping[int(c)] for c in raw_preds], dtype=int)

    def predict_labels(self, df: pd.DataFrame) -> List[str]:
        tiers = self.predict_cluster_ids(df)
        return [self.label_mapping[int(t)] for t in tiers]

    def transform_pca(self, df: pd.DataFrame) -> np.ndarray:
        X_scaled = self.transform_features(df)
        return self.pca_model.transform(X_scaled)


def evaluate_clustering_candidates(
    X_scaled: np.ndarray,
    k_values: Tuple[int, ...] = (2, 3, 4, 5, 6),
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Compara K-Means, GMM y Agglomerative Clustering para distintos k."""
    rows = []

    for k in k_values:
        # 1. K-Means
        km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        km_labels = km.fit_predict(X_scaled)
        rows.append(
            {
                "Algoritmo": "K-Means",
                "k": k,
                "Soporta_Predict_Online": True,
                "Silhouette": silhouette_score(X_scaled, km_labels),
                "Davies_Bouldin": davies_bouldin_score(X_scaled, km_labels),
                "Calinski_Harabasz": calinski_harabasz_score(X_scaled, km_labels),
                "Inercia": float(km.inertia_),
                "BIC": np.nan,
            }
        )

        # 2. Gaussian Mixture Model (GMM)
        gmm = GaussianMixture(
            n_components=k, n_init=10, covariance_type="full", random_state=random_state
        )
        gmm_labels = gmm.fit_predict(X_scaled)
        rows.append(
            {
                "Algoritmo": "GMM",
                "k": k,
                "Soporta_Predict_Online": True,
                "Silhouette": silhouette_score(X_scaled, gmm_labels),
                "Davies_Bouldin": davies_bouldin_score(X_scaled, gmm_labels),
                "Calinski_Harabasz": calinski_harabasz_score(X_scaled, gmm_labels),
                "Inercia": np.nan,
                "BIC": float(gmm.bic(X_scaled)),
            }
        )

        # 3. Jerárquico (Ward) - referencia estructural sin .predict() nativo
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        agg_labels = agg.fit_predict(X_scaled)
        rows.append(
            {
                "Algoritmo": "Jerárquico (Ward)",
                "k": k,
                "Soporta_Predict_Online": False,
                "Silhouette": silhouette_score(X_scaled, agg_labels),
                "Davies_Bouldin": davies_bouldin_score(X_scaled, agg_labels),
                "Calinski_Harabasz": calinski_harabasz_score(X_scaled, agg_labels),
                "Inercia": np.nan,
                "BIC": np.nan,
            }
        )

    return pd.DataFrame(rows)


def select_best_online_clusterer(
    df_comparison: pd.DataFrame,
    X_scaled: np.ndarray,
    preferred_k_range: Tuple[int, ...] = (3, 4),
    random_state: int = RANDOM_STATE,
):
    """
    Selecciona el mejor algoritmo con soporte .predict() nativo dentro del rango
    accionable k in {3, 4} (evitando particiones binarias triviales k=2 para la
    matriz comercial Riesgo x Valor, aunque k=2 se reporta en la tabla completa).
    """
    candidates = df_comparison[
        (df_comparison["Soporta_Predict_Online"])
        & (df_comparison["k"].isin(preferred_k_range))
    ].copy()

    # Priorizar Silhouette más alto y menor Davies-Bouldin como desempate
    candidates = candidates.sort_values(
        by=["Silhouette", "Davies_Bouldin"],
        ascending=[False, True],
    )
    best_row = candidates.iloc[0]
    best_algo = str(best_row["Algoritmo"])
    best_k = int(best_row["k"])

    if best_algo == "K-Means":
        model = KMeans(n_clusters=best_k, n_init=20, random_state=random_state)
        raw_labels = model.fit_predict(X_scaled)
    else:
        model = GaussianMixture(
            n_components=best_k,
            n_init=10,
            covariance_type="full",
            random_state=random_state,
        )
        raw_labels = model.fit_predict(X_scaled)

    return best_algo, best_k, model, raw_labels


def build_value_ordering(
    df_fe: pd.DataFrame,
    raw_labels: np.ndarray,
    k: int,
) -> Tuple[Dict[int, int], Dict[int, str]]:
    """
    Ordena los clusters de forma determinística de menor a mayor valor promedio
    combinado (MonthlyCharges, TotalCharges, tenure) para garantizar estabilidad.
    """
    tmp = df_fe[["MonthlyCharges", "TotalCharges", "tenure"]].copy()
    tmp["raw_cluster"] = raw_labels
    summary = tmp.groupby("raw_cluster")[
        ["MonthlyCharges", "TotalCharges", "tenure"]
    ].mean()

    # Score normalizado simple para ordenar de menor a mayor perfil de valor
    rank_score = (
        summary["MonthlyCharges"].rank(method="dense") * 2.0
        + summary["TotalCharges"].rank(method="dense")
        + summary["tenure"].rank(method="dense") * 0.5
    )
    ordered_raw = rank_score.sort_values().index.tolist()

    cluster_mapping = {int(raw_id): idx for idx, raw_id in enumerate(ordered_raw)}
    default_labels = CLUSTER_LABELS_BY_K.get(
        k, [f"Segmento Valor {i + 1}" for i in range(k)]
    )
    label_mapping = {idx: default_labels[idx] for idx in range(k)}
    return cluster_mapping, label_mapping


def summarize_cluster_profiles(
    df_fe: pd.DataFrame,
    y_dev: pd.Series,
    cluster_tiers: np.ndarray,
    label_mapping: Dict[int, str],
) -> pd.DataFrame:
    """Genera tabla de perfiles descriptivos por segmento (solo sobre Dev)."""
    prof = df_fe.copy()
    prof["Cluster_ID"] = cluster_tiers
    prof["Cluster_Label"] = [label_mapping[int(c)] for c in cluster_tiers]
    prof["Churn"] = y_dev.values
    prof["Es_Mes_a_Mes"] = (prof["Contract"] == "Month-to-month").astype(int)
    prof["Es_Fibra"] = (prof["InternetService"] == "Fiber optic").astype(int)

    total_rows = len(prof)
    grouped = (
        prof.groupby(["Cluster_ID", "Cluster_Label"], as_index=False)
        .agg(
            Clientes=("Churn", "count"),
            Tenure_Medio=("tenure", "mean"),
            MonthlyCharges_Medio=("MonthlyCharges", "mean"),
            TotalCharges_Medio=("TotalCharges", "mean"),
            Servicios_Medios=("TotalServicesSubscribed", "mean"),
            Pct_Mes_a_Mes=("Es_Mes_a_Mes", "mean"),
            Pct_Fibra=("Es_Fibra", "mean"),
            Tasa_Churn_Observada=("Churn", "mean"),
        )
        .sort_values("Cluster_ID")
    )
    grouped["Pct_Cartera"] = grouped["Clientes"] / total_rows
    return grouped


def generate_clustering_figures(
    df_comp: pd.DataFrame,
    X_pca: np.ndarray,
    pca: PCA,
    cluster_tiers: np.ndarray,
    label_mapping: Dict[int, str],
    y_dev: pd.Series,
    df_profiles: pd.DataFrame,
    best_algo: str,
    best_k: int,
):
    FIGURES_TF1_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Comparación de métricas por k y algoritmo
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    km_df = df_comp[df_comp["Algoritmo"] == "K-Means"]
    axes[0].plot(km_df["k"], km_df["Inercia"], marker="o", linewidth=2, color="#1f77b4")
    axes[0].axvline(best_k, color="#d62728", linestyle="--", alpha=0.7, label=f"k={best_k}")
    axes[0].set_title("Método del Codo (Inercia K-Means)")
    axes[0].set_xlabel("Número de clusters (k)")
    axes[0].set_ylabel("Inercia intra-cluster")
    axes[0].legend()

    for algo, grp in df_comp.groupby("Algoritmo"):
        axes[1].plot(grp["k"], grp["Silhouette"], marker="o", linewidth=2, label=algo)
    axes[1].axvline(best_k, color="#d62728", linestyle="--", alpha=0.7)
    axes[1].set_title("Silhouette Score por Algoritmo (↑ mejor)")
    axes[1].set_xlabel("Número de clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].legend()

    for algo, grp in df_comp.groupby("Algoritmo"):
        axes[2].plot(grp["k"], grp["Davies_Bouldin"], marker="s", linewidth=2, label=algo)
    axes[2].axvline(best_k, color="#d62728", linestyle="--", alpha=0.7)
    axes[2].set_title("Índice Davies-Bouldin (↓ mejor)")
    axes[2].set_xlabel("Número de clusters (k)")
    axes[2].set_ylabel("Davies-Bouldin")
    axes[2].legend()

    fig.suptitle(
        f"Comparativa de Clustering en Train+Val (Seleccionado: {best_algo}, k={best_k})",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(FIGURES_TF1_DIR / "clustering_metricas_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Proyección PCA 2D
    var_exp = pca.explained_variance_ratio_
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    palette = sns.color_palette("viridis", n_colors=best_k)
    for tier_id in sorted(label_mapping.keys()):
        mask = cluster_tiers == tier_id
        axes[0].scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            s=14,
            alpha=0.55,
            color=palette[tier_id],
            label=f"C{tier_id}: {label_mapping[tier_id]}",
        )
    axes[0].set_title("Segmentos de Valor en Espacio PCA 2D")
    axes[0].set_xlabel(f"PC1 ({var_exp[0]:.1%} varianza)")
    axes[0].set_ylabel(f"PC2 ({var_exp[1]:.1%} varianza)")
    axes[0].legend(loc="best", fontsize=8.5)

    churn_colors = {0: "#2ca02c", 1: "#d62728"}
    churn_names = {0: "Retenido (0)", 1: "Fuga / Churn (1)"}
    for c_val in (0, 1):
        mask = y_dev.values == c_val
        axes[1].scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            s=14,
            alpha=0.45,
            color=churn_colors[c_val],
            label=churn_names[c_val],
        )
    axes[1].set_title("Distribución Real de Churn en PCA 2D")
    axes[1].set_xlabel(f"PC1 ({var_exp[0]:.1%} varianza)")
    axes[1].set_ylabel(f"PC2 ({var_exp[1]:.1%} varianza)")
    axes[1].legend(loc="best", fontsize=9)

    fig.suptitle(
        f"Visualización PCA 2D (Varianza acumulada: {var_exp.sum():.1%})",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(FIGURES_TF1_DIR / "clustering_pca_2d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Perfiles de valor y tasa de churn observada
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))
    x_labels = [f"C{r.Cluster_ID}\n{r.Cluster_Label}" for r in df_profiles.itertuples()]

    axes[0].bar(x_labels, df_profiles["MonthlyCharges_Medio"], color="#1f77b4", alpha=0.85)
    axes[0].set_title("Cargo Mensual Promedio ($) por Segmento")
    axes[0].set_ylabel("MonthlyCharges medio (USD)")
    for idx, val in enumerate(df_profiles["MonthlyCharges_Medio"]):
        axes[0].text(idx, val + 1.5, f"${val:.1f}", ha="center", fontweight="bold")

    axes[1].bar(
        x_labels,
        df_profiles["Tasa_Churn_Observada"] * 100.0,
        color="#d62728",
        alpha=0.85,
    )
    axes[1].axhline(y_dev.mean() * 100.0, color="#333333", linestyle="--", label=f"Media ({y_dev.mean():.1%})")
    axes[1].set_title("Tasa Observada de Churn (%) por Segmento")
    axes[1].set_ylabel("Tasa de Churn (%)")
    axes[1].legend()
    for idx, val in enumerate(df_profiles["Tasa_Churn_Observada"] * 100.0):
        axes[1].text(idx, val + 0.8, f"{val:.1f}%", ha="center", fontweight="bold")

    plt.tight_layout()
    fig.savefig(FIGURES_TF1_DIR / "clustering_perfiles_valor.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_clustering_pipeline() -> ValueClusterSegmenter:
    print("=" * 70)
    print("FASE 2 TF1 - SEGMENTACIÓN NO SUPERVISADA Y PCA 2D (SOLO TRAIN+VAL)")
    print("=" * 70)

    X_dev, _, y_dev, _ = get_dev_and_test_splits()
    feature_engineer = TelcoFeatureEngineer(enabled=True)
    df_dev_fe = feature_engineer.fit_transform(X_dev)

    scaler_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("power", PowerTransformer(method="yeo-johnson", standardize=True)),
        ]
    )
    X_scaled = scaler_pipeline.fit_transform(df_dev_fe[CLUSTERING_FEATURES])

    df_comp = evaluate_clustering_candidates(X_scaled)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    comp_path = REPORTS_DIR / "comparacion_clustering_tf1.csv"
    df_comp.to_csv(comp_path, index=False)

    best_algo, best_k, cluster_model, raw_labels = select_best_online_clusterer(
        df_comp, X_scaled
    )
    cluster_mapping, label_mapping = build_value_ordering(df_dev_fe, raw_labels, best_k)
    cluster_tiers = np.asarray(
        [cluster_mapping[int(r)] for r in raw_labels], dtype=int
    )

    pca_model = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca_model.fit_transform(X_scaled)

    df_profiles = summarize_cluster_profiles(
        df_dev_fe, y_dev, cluster_tiers, label_mapping
    )
    profiles_path = REPORTS_DIR / "perfiles_clusters_tf1.csv"
    df_profiles.to_csv(profiles_path, index=False)

    generate_clustering_figures(
        df_comp,
        X_pca,
        pca_model,
        cluster_tiers,
        label_mapping,
        y_dev,
        df_profiles,
        best_algo,
        best_k,
    )

    segmenter = ValueClusterSegmenter(
        feature_engineer=feature_engineer,
        scaler_pipeline=scaler_pipeline,
        cluster_model=cluster_model,
        pca_model=pca_model,
        cluster_mapping=cluster_mapping,
        label_mapping=label_mapping,
        algorithm_name=best_algo,
        n_clusters=best_k,
    )

    model_path = MODELS_DIR / "clustering_tf1.joblib"
    joblib.dump(segmenter, model_path)

    print(f"Modelo seleccionado: {best_algo} con k={best_k}")
    print(f"Varianza explicada PCA 2D: {pca_model.explained_variance_ratio_.sum():.2%}")
    print("\nPerfiles de valor (Train + Val):")
    print(df_profiles.to_string(index=False))
    print(f"\nArtefacto guardado en: {model_path}")
    return segmenter


if __name__ == "__main__":
    # Importar desde src.clustering garantiza que ValueClusterSegmenter se serialice
    # bajo el módulo 'src.clustering' y no bajo '__main__' al cargar con joblib.
    from src.clustering import run_clustering_pipeline as _run_clustering

    _run_clustering()
