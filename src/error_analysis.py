"""
Módulo de Análisis Exhaustivo de Errores (TF1 - CC209).
Audita sobre Test Holdout (1,057 clientes) los Falsos Negativos (FN) y
Falsos Positivos (FP), desagrega métricas por subgrupos de negocio y documenta
casos concretos distinguiendo explícitamente evidencia observada de hipótesis.
"""

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

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
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from src.clustering import ValueClusterSegmenter  # noqa: F401
from src.config import FIGURES_TF1_DIR, MODELS_DIR, REPORTS_DIR
from src.evaluation import value_weighted_recall
from src.features import TelcoFeatureEngineer
from src.preprocessing import load_raw_dataset, split_data


def _classify_outcome(y_true: int, y_pred: int) -> str:
    if y_true == 1 and y_pred == 1:
        return "TP"
    if y_true == 0 and y_pred == 0:
        return "TN"
    if y_true == 1 and y_pred == 0:
        return "FN"
    return "FP"


def _build_case_evidence_and_hypothesis(row: pd.Series, error_type: str) -> tuple[str, str]:
    evidencia = (
        f"Contract={row['Contract']}, tenure={int(row['tenure'])}m, "
        f"MonthlyCharges=${row['MonthlyCharges']:.2f}, Internet={row['InternetService']}, "
        f"TechSupport={row['TechSupport']}, Payment={row['PaymentMethod']}, "
        f"P(Churn)={row['y_proba']:.3f}"
    )
    if error_type == "FN":
        if row["Contract"] != "Month-to-month" or row["tenure"] >= 24:
            hipotesis = (
                "Hipótesis: el contrato de largo plazo y/o alta antigüedad reducen fuertemente la probabilidad estimada; "
                "la fuga pudo deberse a eventos no observados en el dataset transversal (incidencia técnica reciente, "
                "insatisfacción con soporte, ajuste tarifario puntual o contraoferta de competencia)."
            )
        elif row["InternetService"] != "Fiber optic":
            hipotesis = (
                "Hipótesis: ausencia de señales de alto riesgo típicas (sin fibra óptica o con pago automático/servicios adicionales), "
                "situando la probabilidad cerca pero por debajo del umbral de decisión."
            )
        else:
            hipotesis = (
                "Hipótesis: perfil mixto donde factores protectores (pago automático o servicios complementarios) "
                "compensaron señales de riesgo contractual."
            )
    else:
        hipotesis = (
            "Hipótesis: reúne múltiples factores estadísticamente asociados a fuga (contrato mes a mes, fibra óptica, "
            "cheque electrónico o baja antigüedad), pero permaneció retenido; puede reflejar alta lealtad no observada, "
            "barreras prácticas de cambio o satisfacción puntual con el servicio."
        )
    return evidencia, hipotesis


def compute_subgroup_metrics(df_audit: pd.DataFrame) -> pd.DataFrame:
    dimensions = ["Contract", "TenureBucket", "InternetService", "SeniorCitizen", "Cluster_Label"]
    rows = []

    for dim in dimensions:
        for val, grp in df_audit.groupby(dim, observed=False):
            y_t = grp["y_true"].values
            y_p = grp["y_pred"].values
            y_pr = grp["y_proba"].values
            mc = grp["MonthlyCharges"].values

            tp = int(np.sum((y_t == 1) & (y_p == 1)))
            fp = int(np.sum((y_t == 0) & (y_p == 1)))
            fn = int(np.sum((y_t == 1) & (y_p == 0)))
            tn = int(np.sum((y_t == 0) & (y_p == 0)))

            pr_auc = (
                float(average_precision_score(y_t, y_pr))
                if len(np.unique(y_t)) > 1
                else np.nan
            )

            rows.append(
                {
                    "Dimension": dim,
                    "Segmento": str(val),
                    "N": int(len(grp)),
                    "Churn_Real_Rate": float(y_t.mean()),
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "TN": tn,
                    "Recall": float(recall_score(y_t, y_p, zero_division=0)),
                    "Value_Weighted_Recall": float(value_weighted_recall(y_t, y_p, mc)),
                    "Precision": float(precision_score(y_t, y_p, zero_division=0)),
                    "F1_Score": float(f1_score(y_t, y_p, zero_division=0)),
                    "PR_AUC": pr_auc,
                    "MonthlyCharges_FN_Perdido": float(grp.loc[grp["Resultado"] == "FN", "MonthlyCharges"].sum()),
                }
            )

    return pd.DataFrame(rows)


def generate_error_figures(
    df_audit: pd.DataFrame,
    df_segments: pd.DataFrame,
    global_thr: float,
):
    FIGURES_TF1_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Desempeño por subgrupos clave
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    dims = [
        ("Contract", "Desempeño por Tipo de Contrato"),
        ("TenureBucket", "Desempeño por Tramo de Antigüedad"),
        ("InternetService", "Desempeño por Servicio de Internet"),
        ("Cluster_Label", "Desempeño por Segmento de Valor (Clustering)"),
    ]
    for ax, (dim, title) in zip(axes.ravel(), dims):
        sub = df_segments[df_segments["Dimension"] == dim].melt(
            id_vars="Segmento",
            value_vars=["Recall", "Precision", "Churn_Real_Rate"],
            var_name="Métrica",
            value_name="Valor",
        )
        sns.barplot(data=sub, x="Segmento", y="Valor", hue="Métrica", ax=ax, palette="Set2")
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=12)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(FIGURES_TF1_DIR / "error_analysis_segmentos.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Distribución de probabilidades y mapa espacial FN / FP
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    sns.histplot(
        data=df_audit,
        x="y_proba",
        hue="y_true",
        bins=25,
        stat="density",
        common_norm=False,
        alpha=0.45,
        palette={0: "#2ca02c", 1: "#d62728"},
        ax=axes[0],
    )
    axes[0].axvline(global_thr, color="#1f77b4", linestyle="-", linewidth=2.2, label=f"Umbral TF1 τ*={global_thr:.2f}")
    axes[0].axvline(0.50, color="#555555", linestyle="--", linewidth=1.8, label="Umbral TP1=0.50")
    axes[0].set_title("Distribución de Probabilidades Predichas en Test")
    axes[0].set_xlabel("Probabilidad estimada P(Churn)")
    axes[0].legend()

    palette_res = {"TP": "#1f77b4", "TN": "#cccccc", "FN": "#d62728", "FP": "#ff7f0e"}
    for res_type in ("TN", "TP", "FP", "FN"):
        sub = df_audit[df_audit["Resultado"] == res_type]
        alpha = 0.25 if res_type == "TN" else 0.78
        size = 16 if res_type == "TN" else 34
        axes[1].scatter(
            sub["tenure"],
            sub["MonthlyCharges"],
            s=size,
            alpha=alpha,
            color=palette_res[res_type],
            label=f"{res_type} (n={len(sub)})",
        )
    axes[1].set_title("Ubicación de Errores FN y FP (Antigüedad vs Cargo Mensual)")
    axes[1].set_xlabel("Antigüedad (tenure en meses)")
    axes[1].set_ylabel("Cargo Mensual ($)")
    axes[1].legend(fontsize=8.5)

    plt.tight_layout()
    fig.savefig(FIGURES_TF1_DIR / "error_analysis_distribucion_proba.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_error_analysis():
    print("=" * 72)
    print("FASE 5B TF1 - ANÁLISIS EXHAUSTIVO DE ERRORES EN TEST (1,057 CLIENTES)")
    print("=" * 72)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_TF1_DIR.mkdir(parents=True, exist_ok=True)

    model = joblib.load(MODELS_DIR / "pipeline_tf1_final.joblib")
    segmenter: ValueClusterSegmenter = joblib.load(MODELS_DIR / "clustering_tf1.joblib")
    with open(MODELS_DIR / "metadata_tf1.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    global_thr = float(metadata["global_threshold"])
    df_raw = load_raw_dataset()
    _, _, X_test_raw, _, _, y_test_raw = split_data(df_raw)
    test_ids = df_raw.loc[X_test_raw.index, "customerID"].values

    X_test = X_test_raw.reset_index(drop=True)
    y_test = y_test_raw.reset_index(drop=True)
    df_fe = TelcoFeatureEngineer(enabled=True).transform(X_test)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= global_thr).astype(int)
    cluster_ids = segmenter.predict_cluster_ids(X_test)
    cluster_labels = segmenter.predict_labels(X_test)

    df_audit = df_fe.copy()
    df_audit["customerID"] = test_ids
    df_audit["Test_Index"] = np.arange(len(df_audit))
    df_audit["y_true"] = y_test.values
    df_audit["y_proba"] = y_proba
    df_audit["y_pred"] = y_pred
    df_audit["Cluster_ID"] = cluster_ids
    df_audit["Cluster_Label"] = cluster_labels
    df_audit["Resultado"] = [
        _classify_outcome(int(yt), int(yp)) for yt, yp in zip(y_test.values, y_pred)
    ]

    # 1. Perfil comparado de los 4 cuadrantes
    df_audit["Es_Mes_a_Mes"] = (df_audit["Contract"] == "Month-to-month").astype(int)
    df_audit["Es_Fibra"] = (df_audit["InternetService"] == "Fiber optic").astype(int)
    df_audit["Sin_Soporte"] = (df_audit["TechSupport"] == "No").astype(int)
    df_audit["Es_Cheque_Elec"] = (df_audit["PaymentMethod"] == "Electronic check").astype(int)

    quad_summary = (
        df_audit.groupby("Resultado", as_index=False)
        .agg(
            Casos=("y_true", "count"),
            Prob_Media=("y_proba", "mean"),
            Tenure_Medio=("tenure", "mean"),
            MonthlyCharges_Medio=("MonthlyCharges", "mean"),
            Servicios_Medios=("TotalServicesSubscribed", "mean"),
            Pct_Mes_a_Mes=("Es_Mes_a_Mes", "mean"),
            Pct_Fibra=("Es_Fibra", "mean"),
            Pct_Sin_Soporte=("Sin_Soporte", "mean"),
            Pct_Cheque_Elec=("Es_Cheque_Elec", "mean"),
        )
    )
    quad_summary.to_csv(REPORTS_DIR / "error_analysis_cuadrantes_tf1.csv", index=False)
    print("\n[1/3] Perfil promedio por cuadrante (TP / FN / FP / TN):")
    print(quad_summary.to_string(index=False))

    # 2. Desempeño por subgrupos
    df_segments = compute_subgroup_metrics(df_audit)
    df_segments.to_csv(REPORTS_DIR / "error_analysis_segmentos_tf1.csv", index=False)
    print("\n[2/3] Desempeño por segmentos clave:")
    print(df_segments[["Dimension", "Segmento", "N", "Churn_Real_Rate", "FN", "FP", "Recall", "Precision"]].to_string(index=False))

    # 3. Casos concretos (Top 5 FN por MonthlyCharges + Top 5 FP por probabilidad)
    top_fn = df_audit[df_audit["Resultado"] == "FN"].sort_values(
        ["MonthlyCharges", "y_proba"], ascending=[False, True]
    ).head(5)
    top_fp = df_audit[df_audit["Resultado"] == "FP"].sort_values(
        ["y_proba", "MonthlyCharges"], ascending=[False, False]
    ).head(5)

    case_rows = []
    for err_type, subset in [("FN", top_fn), ("FP", top_fp)]:
        for _, r in subset.iterrows():
            evidencia, hipotesis = _build_case_evidence_and_hypothesis(r, err_type)
            case_rows.append(
                {
                    "Tipo_Error": err_type,
                    "customerID": r["customerID"],
                    "Test_Index": int(r["Test_Index"]),
                    "y_proba": round(float(r["y_proba"]), 4),
                    "Cluster_Label": r["Cluster_Label"],
                    "Contract": r["Contract"],
                    "tenure": int(r["tenure"]),
                    "MonthlyCharges": float(r["MonthlyCharges"]),
                    "TotalCharges": float(r["TotalCharges"]),
                    "InternetService": r["InternetService"],
                    "TechSupport": r["TechSupport"],
                    "PaymentMethod": r["PaymentMethod"],
                    "Evidencia_Observada": evidencia,
                    "Hipotesis_Explicativa": hipotesis,
                }
            )
    df_cases = pd.DataFrame(case_rows)
    df_cases.to_csv(REPORTS_DIR / "error_analysis_casos_tf1.csv", index=False)
    print(f"\n[3/3] Top 5 FN y Top 5 FP concretos exportados ({len(df_cases)} casos).")

    generate_error_figures(df_audit, df_segments, global_thr)
    print("Análisis de errores completado.")


if __name__ == "__main__":
    run_error_analysis()
