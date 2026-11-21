"""
Módulo de Calibración Probabilística, Optimización de Umbrales y Evaluación Única
en Test (TF1 - CC209).

Regla anti-leakage:
- Toda calibración y búsqueda de umbrales (global tau* y dinámicos por cluster)
  se realiza exclusivamente sobre predicciones Out-Of-Fold (OOF) de Train + Val.
- El conjunto Test (1,057 filas) se evalúa UNA ÚNICA VEZ al final del flujo.
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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold

from src.clustering import ValueClusterSegmenter  # noqa: F401 (necesario para joblib.load)
from src.config import (
    BENEFIT_TRUE_POSITIVE,
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    CV_FOLDS_EVAL,
    ECONOMIC_SCENARIOS,
    FIGURES_TF1_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    RECALL_TARGET,
    REPORTS_DIR,
)
from src.evaluation import (
    BusinessRuleBaseline,
    evaluate_at_threshold,
    evaluate_economic_scenarios,
    evaluate_predictions,
)
from src.preprocessing import get_dev_and_test_splits
from src.train_tf1 import build_model_pipeline


ACTION_MATRIX_RULES = {
    ("Alto Riesgo", 2): {
        "cuadrante": "Alto Riesgo + Alto Valor",
        "prioridad": "P1 - Crítica",
        "accion": "Retención VIP proactiva inmediata: llamada ejecutiva personalizada, revisión técnica prioritaria y oferta de fidelización anual.",
    },
    ("Alto Riesgo", 1): {
        "cuadrante": "Alto Riesgo + Valor Medio",
        "prioridad": "P2 - Alta",
        "accion": "Campaña multicanal dirigida: bono de soporte técnico gratuito o descuento temporal condicionado a migración contractual.",
    },
    ("Alto Riesgo", 0): {
        "cuadrante": "Alto Riesgo + Bajo Valor",
        "prioridad": "P3 - Media",
        "accion": "Retención digital automatizada de bajo costo: correo/SMS con beneficio leve o migración a débito automático.",
    },
    ("Bajo Riesgo", 2): {
        "cuadrante": "Bajo Riesgo + Alto Valor",
        "prioridad": "Fidelización / Upsell",
        "accion": "Cuidado de relación y venta cruzada (cross-selling) sin descuentos innecesarios.",
    },
    ("Bajo Riesgo", 1): {
        "cuadrante": "Bajo Riesgo + Valor Medio",
        "prioridad": "Monitoreo",
        "accion": "Seguimiento preventivo estándar y oferta gradual de paquetes adicionales.",
    },
    ("Bajo Riesgo", 0): {
        "cuadrante": "Bajo Riesgo + Bajo Valor",
        "prioridad": "Estándar",
        "accion": "Mantenimiento operativo estándar (sin intervención promocional).",
    },
}


def build_candidate_with_calibration(
    winner_family: str,
    best_params_by_family: dict,
    calibration_method: str,
):
    """Construye el pipeline ganador opcionalmente envuelto en CalibratedClassifierCV."""
    if winner_family == "stacking":
        base_pipe = build_model_pipeline(
            "stacking", stacking_base_params=best_params_by_family
        )
    else:
        base_pipe = build_model_pipeline(
            winner_family, params=best_params_by_family[winner_family]
        )

    if calibration_method == "none":
        return base_pipe
    return CalibratedClassifierCV(
        estimator=base_pipe,
        method=calibration_method,
        cv=5,
    )


def compute_oof_calibration_comparison(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    winner_family: str,
    best_params_by_family: dict,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], str]:
    """
    Evalúa en OOF (StratifiedKFold k=10) si la calibración Platt (sigmoid) o
    Isotónica mejora Brier Score / Log-Loss sin degradar PR-AUC.
    """
    cv10 = StratifiedKFold(n_splits=CV_FOLDS_EVAL, shuffle=True, random_state=RANDOM_STATE)
    oof_probas = {}
    rows = []

    for method in ("none", "sigmoid", "isotonic"):
        oof = np.zeros(len(X_dev), dtype=float)
        for tr_idx, val_idx in cv10.split(X_dev, y_dev):
            est = build_candidate_with_calibration(
                winner_family, best_params_by_family, method
            )
            est.fit(X_dev.iloc[tr_idx], y_dev.iloc[tr_idx])
            oof[val_idx] = est.predict_proba(X_dev.iloc[val_idx])[:, 1]

        oof_probas[method] = oof
        metrics = evaluate_at_threshold(
            y_dev,
            oof,
            threshold=0.50,
            monthly_charges=X_dev["MonthlyCharges"].values,
            model_name=f"Calibración: {method}",
        )
        rows.append(
            {
                "Metodo_Calibracion": method,
                "PR-AUC_OOF": metrics["PR-AUC"],
                "ROC-AUC_OOF": metrics["ROC-AUC"],
                "Brier_Score_OOF": metrics["Brier Score"],
                "Log_Loss_OOF": metrics["Log-Loss"],
            }
        )

    df_cal = pd.DataFrame(rows)
    df_cal.to_csv(REPORTS_DIR / "calibracion_oof_tf1.csv", index=False)

    base_row = df_cal[df_cal["Metodo_Calibracion"] == "none"].iloc[0]
    eligible = df_cal[df_cal["PR-AUC_OOF"] >= base_row["PR-AUC_OOF"] - 0.002]
    chosen_method = str(
        eligible.sort_values(["Brier_Score_OOF", "Log_Loss_OOF"]).iloc[0]["Metodo_Calibracion"]
    )
    return df_cal, oof_probas, chosen_method


def optimize_global_threshold(
    y_dev: pd.Series,
    oof_proba: np.ndarray,
    monthly_charges: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    """Barre umbrales en [0.05, 0.90] sobre OOF maximizando utilidad neta (con Recall >= RECALL_TARGET)."""
    sweep_rows = []
    for thr in np.round(np.arange(0.05, 0.905, 0.01), 2):
        res = evaluate_at_threshold(
            y_dev,
            oof_proba,
            threshold=float(thr),
            monthly_charges=monthly_charges,
            model_name="OOF Sweep",
        )
        res["Threshold"] = float(thr)
        sweep_rows.append(res)

    df_sweep = pd.DataFrame(sweep_rows)
    df_sweep.to_csv(REPORTS_DIR / "curva_umbrales_oof_tf1.csv", index=False)

    feasible = df_sweep[df_sweep["Recall (Clase 1)"] >= RECALL_TARGET]
    pool = feasible if not feasible.empty else df_sweep
    best_row = pool.sort_values(
        ["Valor Económico Neto ($)", "Value-Weighted Recall"],
        ascending=[False, False],
    ).iloc[0]
    return float(best_row["Threshold"]), df_sweep


def optimize_dynamic_thresholds_by_cluster(
    y_dev: pd.Series,
    oof_proba: np.ndarray,
    cluster_ids_dev: np.ndarray,
    monthly_charges_dev: np.ndarray,
    label_mapping: dict[int, str],
) -> dict[int, dict]:
    """
    Optimiza un umbral específico por perfil/cluster de valor en OOF.
    Pondera la pérdida por FN proporcionalmente al ingreso mensual promedio del
    cluster respecto a la media global (priorizando retener ingresos de alto valor).
    """
    global_mean_mc = float(np.mean(monthly_charges_dev))
    dynamic_cfg: dict[int, dict] = {}

    for cid in sorted(label_mapping.keys()):
        mask = cluster_ids_dev == cid
        y_c = y_dev.values[mask]
        p_c = oof_proba[mask]
        mc_c = monthly_charges_dev[mask]
        cluster_mean_mc = float(np.mean(mc_c))
        value_factor = cluster_mean_mc / global_mean_mc

        best_thr = 0.25
        best_weighted_utility = -np.inf
        best_base_utility = -np.inf

        for thr in np.round(np.arange(0.10, 0.71, 0.01), 2):
            y_pred_c = (p_c >= thr).astype(int)
            tp = int(np.sum((y_c == 1) & (y_pred_c == 1)))
            fn = int(np.sum((y_c == 1) & (y_pred_c == 0)))
            fp = int(np.sum((y_c == 0) & (y_pred_c == 1)))

            # Utilidad ponderada por valor del segmento (costo FN y beneficio TP escalan con ticket medio)
            weighted_util = (
                tp * (BENEFIT_TRUE_POSITIVE * value_factor)
                - fn * (COST_FALSE_NEGATIVE * value_factor)
                - fp * COST_FALSE_POSITIVE
            )
            base_util = (
                tp * BENEFIT_TRUE_POSITIVE
                - fn * COST_FALSE_NEGATIVE
                - fp * COST_FALSE_POSITIVE
            )
            if weighted_util > best_weighted_utility:
                best_weighted_utility = weighted_util
                best_base_utility = base_util
                best_thr = float(thr)

        dynamic_cfg[int(cid)] = {
            "label": label_mapping[int(cid)],
            "threshold": float(best_thr),
            "mean_monthly_charges": round(cluster_mean_mc, 2),
            "value_weight_factor": round(value_factor, 3),
            "oof_base_utility": float(best_base_utility),
            "n_dev": int(np.sum(mask)),
        }

    return dynamic_cfg


def build_threshold_vector(cluster_ids: np.ndarray, dynamic_cfg: dict[int, dict]) -> np.ndarray:
    return np.asarray([dynamic_cfg[int(c)]["threshold"] for c in cluster_ids], dtype=float)


def generate_threshold_figures(
    y_dev: pd.Series,
    oof_probas: dict[str, np.ndarray],
    chosen_cal: str,
    df_sweep: pd.DataFrame,
    global_thr: float,
    df_test_eval: pd.DataFrame,
    df_sens: pd.DataFrame,
):
    FIGURES_TF1_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Reliability Diagram (Calibración OOF)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Calibración perfecta")
    styles = {
        "none": ("Sin calibrar", "#1f77b4", "o"),
        "sigmoid": ("Platt (Sigmoid)", "#ff7f0e", "s"),
        "isotonic": ("Isotónica", "#2ca02c", "^"),
    }
    for m, (lbl, color, mk) in styles.items():
        prob_true, prob_pred = calibration_curve(y_dev, oof_probas[m], n_bins=10, strategy="quantile")
        mark = " ★" if m == chosen_cal else ""
        ax.plot(prob_pred, prob_true, marker=mk, linewidth=2, color=color, label=f"{lbl}{mark}")
    ax.set_title("Diagrama de Confiabilidad Probabilística (OOF Train+Val)")
    ax.set_xlabel("Probabilidad predicha media")
    ax.set_ylabel("Frecuencia empírica de Churn")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_TF1_DIR / "calibracion_probabilidades.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Curva Umbral vs Utilidad Económica + Recall/Precision
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    axes[0].plot(df_sweep["Threshold"], df_sweep["Valor Económico Neto ($)"], color="#1f77b4", linewidth=2.3)
    axes[0].axvline(0.50, color="#7f7f7f", linestyle="--", label="Umbral fijo TP1 (0.50)")
    axes[0].axvline(global_thr, color="#d62728", linestyle="-", linewidth=2, label=f"Umbral óptimo τ* ({global_thr:.2f})")
    axes[0].set_title("Utilidad Económica Simulada vs. Umbral (OOF)")
    axes[0].set_xlabel("Umbral de decisión (τ)")
    axes[0].set_ylabel("Utilidad Económica Neta ($)")
    axes[0].legend()

    axes[1].plot(df_sweep["Threshold"], df_sweep["Recall (Clase 1)"], label="Recall (Clase 1)", linewidth=2, color="#d62728")
    axes[1].plot(df_sweep["Threshold"], df_sweep["Value-Weighted Recall"], label="Value-Weighted Recall ($)", linewidth=2, color="#9467bd")
    axes[1].plot(df_sweep["Threshold"], df_sweep["Precision (Clase 1)"], label="Precision (Clase 1)", linewidth=2, color="#2ca02c")
    axes[1].axhline(RECALL_TARGET, color="#333333", linestyle=":", label=f"Meta Recall ({RECALL_TARGET:.0%})")
    axes[1].axvline(global_thr, color="#d62728", linestyle="--", alpha=0.7)
    axes[1].set_title("Trade-off Recall / Value-Weighted Recall / Precision (OOF)")
    axes[1].set_xlabel("Umbral de decisión (τ)")
    axes[1].set_ylabel("Métrica")
    axes[1].legend(fontsize=8.5)

    plt.tight_layout()
    fig.savefig(FIGURES_TF1_DIR / "curva_umbral_utilidad.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Comparación TP1 vs TF1 en Test
    focus = df_test_eval[
        ~df_test_eval["Modelo"].str.startswith("Baseline")
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))

    melted = focus.melt(
        id_vars="Modelo",
        value_vars=["PR-AUC", "Recall (Clase 1)", "Value-Weighted Recall", "F1-Score"],
        var_name="Métrica",
        value_name="Valor",
    )
    sns.barplot(data=melted, x="Métrica", y="Valor", hue="Modelo", ax=axes[0], palette="Set2")
    axes[0].set_title("Comparativa Técnica en Test Holdout (TP1 vs TF1)")
    axes[0].set_ylim(0, 1.0)
    axes[0].legend(fontsize=8, loc="lower right")

    sns.barplot(data=focus, y="Modelo", x="Valor Económico Neto ($)", hue="Modelo", legend=False, ax=axes[1], palette="viridis")
    axes[1].set_title("Utilidad Económica Neta Simulada en Test ($)")
    axes[1].set_ylabel("")
    for idx, val in enumerate(focus["Valor Económico Neto ($)"]):
        axes[1].text(val + 400, idx, f"${val:,.0f}", va="center", fontweight="bold", fontsize=9)

    plt.tight_layout()
    fig.savefig(FIGURES_TF1_DIR / "comparacion_tp1_vs_tf1_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Sensibilidad Económica (3 escenarios)
    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    sns.barplot(data=df_sens, x="Escenario", y="Valor Económico Neto ($)", hue="Modelo", ax=ax, palette="tab10")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Análisis de Sensibilidad Económica en Test (3 Escenarios)")
    ax.set_xlabel("")
    ax.set_ylabel("Utilidad Económica Neta ($)")
    ax.tick_params(axis="x", rotation=10)
    ax.legend(fontsize=8.5, loc="upper right")
    plt.tight_layout()
    fig.savefig(FIGURES_TF1_DIR / "sensibilidad_economica.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_calibration_thresholds_and_test():
    print("=" * 74)
    print("FASE 4 TF1 - CALIBRACIÓN, UMBRALES DINÁMICOS Y EVALUACIÓN ÚNICA EN TEST")
    print("=" * 74)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_TF1_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODELS_DIR / "best_params_tf1.json", "r", encoding="utf-8") as f:
        best_cfg = json.load(f)

    winner_name = best_cfg["winner_cv_name"]
    winner_family = best_cfg["winner_cv_family"]
    best_params_map = best_cfg["best_params_by_family"]

    segmenter: ValueClusterSegmenter = joblib.load(MODELS_DIR / "clustering_tf1.joblib")
    X_dev, X_test, y_dev, y_test = get_dev_and_test_splits()

    # 1. Calibración OOF
    print(f"[1/5] Evaluando calibración OOF sobre: {winner_name}")
    df_cal, oof_probas, chosen_cal = compute_oof_calibration_comparison(
        X_dev, y_dev, winner_family, best_params_map
    )
    print(df_cal.to_string(index=False))
    print(f"  -> Calibración elegida: {chosen_cal}")

    oof_chosen = oof_probas[chosen_cal]
    mc_dev = X_dev["MonthlyCharges"].values
    cluster_ids_dev = segmenter.predict_cluster_ids(X_dev)

    # 2. Optimización OOF de umbral global y dinámico por cluster
    global_thr, df_sweep = optimize_global_threshold(y_dev, oof_chosen, mc_dev)
    dynamic_cfg = optimize_dynamic_thresholds_by_cluster(
        y_dev, oof_chosen, cluster_ids_dev, mc_dev, segmenter.label_mapping
    )
    dyn_thr_dev = build_threshold_vector(cluster_ids_dev, dynamic_cfg)

    oof_summary = [
        evaluate_at_threshold(y_dev, oof_chosen, 0.50, mc_dev, "OOF Umbral Fijo 0.50"),
        evaluate_at_threshold(y_dev, oof_chosen, global_thr, mc_dev, f"OOF Umbral Global ({global_thr:.2f})"),
        evaluate_at_threshold(y_dev, oof_chosen, dyn_thr_dev, mc_dev, "OOF Umbrales Dinámicos por Cluster"),
    ]
    df_oof_summary = pd.DataFrame(oof_summary)
    df_oof_summary.to_csv(REPORTS_DIR / "comparacion_umbrales_oof_tf1.csv", index=False)

    print(f"\n[2/5] Umbrales optimizados exclusivamente en OOF (Train+Val):")
    print(f"  -> Umbral global óptimo tau*: {global_thr:.2f}")
    for cid, info in dynamic_cfg.items():
        print(
            f"  -> Cluster {cid} ({info['label']}): tau*={info['threshold']:.2f} "
            f"(MonthlyCharges medio=${info['mean_monthly_charges']:.2f})"
        )
    print("\nComparación políticas de umbral en OOF:")
    print(df_oof_summary[["Modelo", "Recall (Clase 1)", "Value-Weighted Recall", "Precision (Clase 1)", "Valor Económico Neto ($)"]].to_string(index=False))

    # 3. Reentrenamiento final sobre Train+Val completo
    print("\n[3/5] Reentrenando pipeline final TF1 con Train + Validation...")
    final_pipeline = build_candidate_with_calibration(
        winner_family, best_params_map, chosen_cal
    )
    final_pipeline.fit(X_dev, y_dev)
    final_model_path = MODELS_DIR / "pipeline_tf1_final.joblib"
    joblib.dump(final_pipeline, final_model_path)

    # 4. EVALUACIÓN FINAL ÚNICA EN TEST (1,057 filas)
    print("\n[4/5] Ejecutando evaluación final ÚNICA sobre Test Holdout (1,057 filas)...")
    mc_test = X_test["MonthlyCharges"].values
    cluster_ids_test = segmenter.predict_cluster_ids(X_test)
    dyn_thr_test = build_threshold_vector(cluster_ids_test, dynamic_cfg)

    # Baselines + TP1 histórico
    dummy = DummyClassifier(strategy="most_frequent").fit(X_dev, y_dev)
    heuristic = BusinessRuleBaseline(tenure_threshold=6).fit(X_dev, y_dev)
    tp1_pipeline = joblib.load(MODELS_DIR / "pipeline_tp1_seleccionado.joblib")

    y_proba_tf1_test = final_pipeline.predict_proba(X_test)[:, 1]

    test_rows = [
        evaluate_predictions(
            y_test,
            dummy.predict(X_test),
            dummy.predict_proba(X_test)[:, 1],
            mc_test,
            "Baseline Trivial (Dummy)",
        ),
        evaluate_predictions(
            y_test,
            heuristic.predict(X_test),
            heuristic.predict_proba(X_test)[:, 1],
            mc_test,
            "Baseline Heurístico (Mes-a-Mes & Tenure <= 6)",
        ),
        evaluate_predictions(
            y_test,
            tp1_pipeline.predict(X_test),
            tp1_pipeline.predict_proba(X_test)[:, 1],
            mc_test,
            "TP1: Random Forest (depth=10, umbral 0.50)",
        ),
        evaluate_at_threshold(
            y_test,
            y_proba_tf1_test,
            0.50,
            mc_test,
            f"TF1: {winner_name} (umbral 0.50)",
        ),
        evaluate_at_threshold(
            y_test,
            y_proba_tf1_test,
            global_thr,
            mc_test,
            f"TF1: {winner_name} (umbral global {global_thr:.2f})",
        ),
        evaluate_at_threshold(
            y_test,
            y_proba_tf1_test,
            dyn_thr_test,
            mc_test,
            f"TF1: {winner_name} (umbrales dinámicos por cluster)",
        ),
    ]
    df_test_eval = pd.DataFrame(test_rows)
    df_test_eval.to_csv(REPORTS_DIR / "evaluacion_final_test_tf1.csv", index=False)

    show_cols = [
        "Modelo", "PR-AUC", "ROC-AUC", "Recall (Clase 1)",
        "Value-Weighted Recall", "Precision (Clase 1)", "F1-Score", "Valor Económico Neto ($)"
    ]
    print(df_test_eval[show_cols].to_string(index=False))

    # 5. Sensibilidad Económica en Test (3 escenarios)
    print("\n[5/5] Análisis de sensibilidad económica en Test (3 escenarios)...")
    sens_models = {
        "Baseline Trivial (Dummy)": dummy.predict(X_test),
        "Baseline Heurístico": heuristic.predict(X_test),
        "TP1: Random Forest (0.50)": tp1_pipeline.predict(X_test),
        f"TF1 Umbral Global ({global_thr:.2f})": (y_proba_tf1_test >= global_thr).astype(int),
        "TF1 Umbrales Dinámicos": (y_proba_tf1_test >= dyn_thr_test).astype(int),
    }
    sens_rows = []
    for m_name, preds in sens_models.items():
        sens_rows.extend(evaluate_economic_scenarios(y_test, preds, model_name=m_name))
    df_sens = pd.DataFrame(sens_rows)
    df_sens.to_csv(REPORTS_DIR / "sensibilidad_economica_tf1.csv", index=False)

    # Elegir política recomendada según utilidad + Value-Weighted Recall OOF
    oof_global_val = float(df_oof_summary.iloc[1]["Valor Económico Neto ($)"])
    oof_dyn_val = float(df_oof_summary.iloc[2]["Valor Económico Neto ($)"])
    recommended_policy = "dynamic_cluster" if oof_dyn_val >= oof_global_val else "global"

    metadata = {
        "winner_name": winner_name,
        "winner_family": winner_family,
        "calibration_method": chosen_cal,
        "recommended_policy": recommended_policy,
        "global_threshold": global_thr,
        "dynamic_thresholds_by_cluster": {str(k): v for k, v in dynamic_cfg.items()},
        "action_matrix_rules": {f"{k[0]}|{k[1]}": v for k, v in ACTION_MATRIX_RULES.items()},
        "economic_scenarios": ECONOMIC_SCENARIOS,
        "oof_thresholds_summary": df_oof_summary.to_dict(orient="records"),
        "test_metrics_summary": df_test_eval.to_dict(orient="records"),
    }
    with open(MODELS_DIR / "metadata_tf1.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    generate_threshold_figures(
        y_dev, oof_probas, chosen_cal, df_sweep, global_thr, df_test_eval, df_sens
    )
    print(f"\nPolítica recomendada por evidencia OOF: {recommended_policy}")
    print(f"Artefactos guardados en: {final_model_path} y {MODELS_DIR / 'metadata_tf1.json'}")


if __name__ == "__main__":
    run_calibration_thresholds_and_test()
