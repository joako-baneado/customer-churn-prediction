"""
Módulo de Interpretabilidad Global y Local con SHAP (TF1 - CC209).
Calcula valores SHAP aditivos en el espacio preprocesado del pipeline ganador
(Stacking Heterogéneo) utilizando distribución background de Train + Val.

Precaución metodológica:
SHAP cuantifica la contribución marginal de cada variable a la probabilidad
predicha por el modelo; no implica causalidad directa.
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
import shap
from sklearn.model_selection import train_test_split

from src.clustering import ValueClusterSegmenter  # noqa: F401
from src.config import FIGURES_TF1_DIR, MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from src.preprocessing import get_dev_and_test_splits


def _unwrap_pipeline(model):
    """Obtiene (preprocessor, classifier_or_wrapper) compatible con wrapper calibrado."""
    if hasattr(model, "base_pipeline"):
        base = model.base_pipeline
        preprocessor = base.named_steps["preprocessor"]
        # En wrapper calibrado pasamos todo el paso post-preprocesado
        def predict_fn(X_arr: np.ndarray) -> np.ndarray:
            raw = base.named_steps["classifier"].predict_proba(X_arr)[:, 1]
            if model.method == "none" or model.calibrator is None:
                return raw
            if model.method == "sigmoid":
                return model.calibrator.predict_proba(model._to_logit(raw))[:, 1]
            return np.clip(model.calibrator.predict(raw), 1e-7, 1.0 - 1e-7)
        return preprocessor, predict_fn

    preprocessor = model.named_steps["preprocessor"]
    clf = model.named_steps["classifier"]
    return preprocessor, lambda X_arr: clf.predict_proba(X_arr)[:, 1]


def get_feature_names_from_preprocessor(preprocessor) -> list[str]:
    return preprocessor.named_steps["column_transformer"].get_feature_names_out().tolist()


def build_shap_explainer(model, X_background_transformed: np.ndarray, feature_names: list[str]):
    _, predict_fn = _unwrap_pipeline(model)
    return shap.Explainer(
        predict_fn,
        X_background_transformed,
        feature_names=feature_names,
        seed=RANDOM_STATE,
    )


def explain_single_customer(
    model,
    df_row: pd.DataFrame,
    background_path: Path | None = None,
) -> shap.Explanation:
    """
    Calcula el objeto shap.Explanation (1 fila) para visualización Waterfall
    en tiempo real dentro de la aplicación Streamlit.
    """
    preprocessor, _ = _unwrap_pipeline(model)
    bg_file = background_path or (MODELS_DIR / "shap_background_tf1.npz")
    data = np.load(bg_file, allow_pickle=True)
    X_bg = data["X_background"]
    feature_names = data["feature_names"].tolist()

    X_row_t = preprocessor.transform(df_row)
    explainer = build_shap_explainer(model, X_bg, feature_names)
    sv = explainer(X_row_t)
    return sv[0]


def _save_waterfall(explanation: shap.Explanation, title: str, out_path: Path):
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.title(title, fontsize=11, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")


def run_explainability():
    print("=" * 72)
    print("FASE 5A TF1 - INTERPRETABILIDAD GLOBAL Y LOCAL CON SHAP")
    print("=" * 72)

    FIGURES_TF1_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model = joblib.load(MODELS_DIR / "pipeline_tf1_final.joblib")
    segmenter: ValueClusterSegmenter = joblib.load(MODELS_DIR / "clustering_tf1.joblib")
    with open(MODELS_DIR / "metadata_tf1.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    global_thr = float(metadata["global_threshold"])
    X_dev, X_test, y_dev, y_test = get_dev_and_test_splits()
    preprocessor, _ = _unwrap_pipeline(model)
    feature_names = get_feature_names_from_preprocessor(preprocessor)

    # Background estratificado de 50 muestras de Train+Val (guardado para Streamlit)
    _, X_bg_raw = train_test_split(
        X_dev, test_size=50, stratify=y_dev, random_state=RANDOM_STATE
    )
    X_bg = preprocessor.transform(X_bg_raw)
    np.savez_compressed(
        MODELS_DIR / "shap_background_tf1.npz",
        X_background=X_bg,
        feature_names=np.asarray(feature_names, dtype=object),
    )

    # Muestra estratificada de 200 observaciones de Test para SHAP global
    _, X_sample_test, _, _ = train_test_split(
        X_test, y_test, test_size=200, stratify=y_test, random_state=RANDOM_STATE
    )
    X_sample_t = preprocessor.transform(X_sample_test)

    print("[1/3] Calculando valores SHAP globales sobre muestra estratificada (n=200)...", flush=True)
    explainer = build_shap_explainer(model, X_bg, feature_names)
    shap_values = explainer(X_sample_t)

    # Ranking tabular global
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    mean_signed = shap_values.values.mean(axis=0)
    df_imp = pd.DataFrame(
        {
            "Variable": feature_names,
            "Mean_Abs_SHAP": mean_abs,
            "Mean_Signed_SHAP": mean_signed,
        }
    ).sort_values("Mean_Abs_SHAP", ascending=False)
    df_imp.to_csv(REPORTS_DIR / "shap_importancia_global_tf1.csv", index=False)

    print("Top 12 variables por importancia media |SHAP|:")
    print(df_imp.head(12).to_string(index=False))

    # Gráficos globales
    print("\n[2/3] Generando gráficos globales (Beeswarm y Bar)...", flush=True)
    plt.figure(figsize=(10.5, 6.8))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("SHAP Global (Beeswarm) - Impacto sobre Probabilidad de Churn", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_TF1_DIR / "shap_global_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.title("Importancia Global Media |SHAP| (Top 15 Variables)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_TF1_DIR / "shap_global_bar.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # 3 Casos locales representativos en Test
    print("[3/3] Generando 3 gráficos Waterfall locales (TP, FN, FP críticos)...", flush=True)
    proba_test = model.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= global_thr).astype(int)
    clusters_test = segmenter.predict_cluster_ids(X_test)

    df_audit = X_test.copy()
    df_audit["y_true"] = y_test.values
    df_audit["y_proba"] = proba_test
    df_audit["y_pred"] = pred_test
    df_audit["cluster"] = clusters_test

    # Caso 1: Verdadero Positivo crítico (Alto Valor, Churn=1, Pred=1)
    tp_pool = df_audit[(df_audit["y_true"] == 1) & (df_audit["y_pred"] == 1)]
    tp_vip = tp_pool.sort_values(["cluster", "MonthlyCharges", "y_proba"], ascending=[False, False, False]).index[0]

    # Caso 2: Falso Negativo crítico (Churn=1 no detectado con mayor MonthlyCharges)
    fn_pool = df_audit[(df_audit["y_true"] == 1) & (df_audit["y_pred"] == 0)]
    fn_crit = fn_pool.sort_values(["MonthlyCharges", "y_proba"], ascending=[False, True]).index[0]

    # Caso 3: Falso Positivo crítico (Churn=0 con mayor probabilidad asignada)
    fp_pool = df_audit[(df_audit["y_true"] == 0) & (df_audit["y_pred"] == 1)]
    fp_crit = fp_pool.sort_values("y_proba", ascending=False).index[0]

    local_cases = [
        (tp_vip, f"SHAP Local - TP Crítico (Test idx={tp_vip}, P={proba_test[tp_vip]:.2f}, ${X_test.loc[tp_vip, 'MonthlyCharges']:.1f}/mes)", "shap_local_tp_critico.png"),
        (fn_crit, f"SHAP Local - FN Crítico (Test idx={fn_crit}, P={proba_test[fn_crit]:.2f}, ${X_test.loc[fn_crit, 'MonthlyCharges']:.1f}/mes)", "shap_local_fn_critico.png"),
        (fp_crit, f"SHAP Local - FP Crítico (Test idx={fp_crit}, P={proba_test[fp_crit]:.2f}, ${X_test.loc[fp_crit, 'MonthlyCharges']:.1f}/mes)", "shap_local_fp_critico.png"),
    ]
    for idx_case, title, fname in local_cases:
        X_case_t = preprocessor.transform(X_test.loc[[idx_case]])
        sv_case = explainer(X_case_t)[0]
        _save_waterfall(sv_case, title, FIGURES_TF1_DIR / fname)

    print("SHAP global y local generados en reports/figures_tf1/")


if __name__ == "__main__":
    run_explainability()
