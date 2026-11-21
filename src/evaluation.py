"""
Módulo de Evaluación de Modelos y Análisis de Negocio - TP1
Calcula métricas técnicas (PR-AUC, ROC-AUC, Recall) y la función de utilidad
económica basada en la matriz de confusión (Rúbrica Criterio 6).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    confusion_matrix
)
from src.config import (
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    BENEFIT_TRUE_POSITIVE,
    BENEFIT_TRUE_NEGATIVE,
    ECONOMIC_SCENARIOS,
)


def compute_business_impact(tn: int, fp: int, fn: int, tp: int) -> dict:
    """
    Calcula el impacto económico neto y los costos asociados según las decisiones del modelo.
    
    Supuestos de negocio:
    - FN (Cliente fugado no detectado): Pérdida de valor de vida (LTV) = -$500
    - FP (Cliente retenido contactado innecesariamente): Costo de incentivo = -$50
    - TP (Cliente en riesgo detectado y retenido con oferta): Beneficio neto = +$350
    - TN (Cliente satisfecho no intervenido): Costo neutral = $0
    """
    cost_fn = fn * COST_FALSE_NEGATIVE
    cost_fp = fp * COST_FALSE_POSITIVE
    benefit_tp = tp * BENEFIT_TRUE_POSITIVE
    net_economic_value = benefit_tp - (cost_fn + cost_fp)

    return {
        "costo_falsos_negativos": cost_fn,
        "costo_falsos_positivos": cost_fp,
        "beneficio_verdaderos_positivos": benefit_tp,
        "valor_economico_neto": net_economic_value
    }


def evaluate_classifier(model, X_test, y_test, model_name: str = "Modelo") -> dict:
    """
    Evalúa integralmente un clasificador reportando métricas técnicas
    y métricas económicas de negocio.
    """
    y_pred = model.predict(X_test)
    
    # Probabilidad de clase positiva si el estimador lo soporta
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = y_pred

    # Matriz de confusión
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    impact = compute_business_impact(tn, fp, fn, tp)

    # Métricas técnicas
    roc_auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_proba)) > 1 else 0.5
    pr_auc = average_precision_score(y_test, y_proba) if len(np.unique(y_proba)) > 1 else y_test.mean()

    return {
        "Modelo": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (Clase 1)": precision_score(y_test, y_pred, zero_division=0),
        "Recall (Clase 1)": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Valor Económico Neto ($)": impact["valor_economico_neto"]
    }


class BusinessRuleBaseline:
    """
    Baseline heurístico de negocio:
    Si un cliente tiene contrato mes a mes (Month-to-month) y antigüedad <= 6 meses,
    se asume que se fugará (Churn = 1); de lo contrario, se queda (0).
    """
    def __init__(self, tenure_threshold: int = 6):
        self.tenure_threshold = tenure_threshold

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        is_month_to_month = X["Contract"] == "Month-to-month"
        is_low_tenure = X["tenure"] <= self.tenure_threshold
        preds = (is_month_to_month & is_low_tenure).astype(int)
        return preds.values

    def predict_proba(self, X):
        preds = self.predict(X)
        proba_1 = preds.astype(float)
        proba_0 = 1.0 - proba_1
        return np.column_stack([proba_0, proba_1])


def value_weighted_recall(y_true, y_pred, monthly_charges) -> float:
    """
    Calcula el Recall ponderado por ingresos mensuales en riesgo:
    proporción de dólares mensuales de clientes que fugaron efectivamente
    que el modelo logró capturar (TP / (TP + FN) ponderado por MonthlyCharges).
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    charges = np.asarray(monthly_charges, dtype=float)

    positives_mask = y_t == 1
    total_at_risk_revenue = charges[positives_mask].sum()
    if total_at_risk_revenue <= 0:
        return 0.0
    captured_revenue = charges[positives_mask & (y_p == 1)].sum()
    return float(captured_revenue / total_at_risk_revenue)


def compute_business_impact_custom(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
    cost_fn: float = COST_FALSE_NEGATIVE,
    cost_fp: float = COST_FALSE_POSITIVE,
    benefit_tp: float = BENEFIT_TRUE_POSITIVE,
    benefit_tn: float = BENEFIT_TRUE_NEGATIVE,
) -> float:
    """Calcula utilidad económica neta parametrizada para análisis de sensibilidad."""
    return float(tp * benefit_tp + tn * benefit_tn - (fn * cost_fn + fp * cost_fp))


def evaluate_predictions(
    y_true,
    y_pred,
    y_proba,
    monthly_charges=None,
    model_name: str = "Modelo",
    cost_fn: float = COST_FALSE_NEGATIVE,
    cost_fp: float = COST_FALSE_POSITIVE,
    benefit_tp: float = BENEFIT_TRUE_POSITIVE,
    benefit_tn: float = BENEFIT_TRUE_NEGATIVE,
) -> dict:
    """Evalúa predicciones y probabilidades incluyendo Brier Score y Value-Weighted Recall."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    y_pr = np.clip(np.asarray(y_proba, dtype=float), 1e-7, 1.0 - 1e-7)

    tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
    net_value = compute_business_impact_custom(
        tn, fp, fn, tp, cost_fn=cost_fn, cost_fp=cost_fp, benefit_tp=benefit_tp, benefit_tn=benefit_tn
    )

    roc_auc = roc_auc_score(y_t, y_pr) if len(np.unique(y_pr)) > 1 else 0.5
    pr_auc = average_precision_score(y_t, y_pr) if len(np.unique(y_pr)) > 1 else float(y_t.mean())
    brier = float(brier_score_loss(y_t, y_pr))
    ll = float(log_loss(y_t, y_pr, labels=[0, 1]))
    vw_recall = (
        value_weighted_recall(y_t, y_p, monthly_charges)
        if monthly_charges is not None
        else np.nan
    )

    return {
        "Modelo": model_name,
        "Accuracy": float(accuracy_score(y_t, y_p)),
        "Precision (Clase 1)": float(precision_score(y_t, y_p, zero_division=0)),
        "Recall (Clase 1)": float(recall_score(y_t, y_p, zero_division=0)),
        "Value-Weighted Recall": vw_recall,
        "F1-Score": float(f1_score(y_t, y_p, zero_division=0)),
        "ROC-AUC": float(roc_auc),
        "PR-AUC": float(pr_auc),
        "Brier Score": brier,
        "Log-Loss": ll,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Valor Económico Neto ($)": net_value,
    }


def evaluate_at_threshold(
    y_true,
    y_proba,
    threshold,
    monthly_charges=None,
    model_name: str = "Modelo",
    costs: dict | None = None,
) -> dict:
    """Evalúa probabilidades aplicando un umbral escalar o un vector de umbrales por fila."""
    y_pr = np.asarray(y_proba, dtype=float)
    y_pred = (y_pr >= threshold).astype(int)
    c = costs or {}
    return evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_pr,
        monthly_charges=monthly_charges,
        model_name=model_name,
        cost_fn=c.get("cost_fn", COST_FALSE_NEGATIVE),
        cost_fp=c.get("cost_fp", COST_FALSE_POSITIVE),
        benefit_tp=c.get("benefit_tp", BENEFIT_TRUE_POSITIVE),
        benefit_tn=c.get("benefit_tn", BENEFIT_TRUE_NEGATIVE),
    )


def evaluate_economic_scenarios(
    y_true,
    y_pred,
    model_name: str = "Modelo",
    scenarios: dict | None = None,
) -> list[dict]:
    """Evalúa la utilidad neta bajo los escenarios de sensibilidad definidos."""
    sc_map = scenarios or ECONOMIC_SCENARIOS
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()

    rows = []
    for key, cfg in sc_map.items():
        net_val = compute_business_impact_custom(
            tn,
            fp,
            fn,
            tp,
            cost_fn=cfg["cost_fn"],
            cost_fp=cfg["cost_fp"],
            benefit_tp=cfg["benefit_tp"],
            benefit_tn=cfg.get("benefit_tn", 0.0),
        )
        rows.append(
            {
                "Modelo": model_name,
                "Escenario_Key": key,
                "Escenario": cfg["label"],
                "Costo_FN": cfg["cost_fn"],
                "Costo_FP": cfg["cost_fp"],
                "Beneficio_TP": cfg["benefit_tp"],
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
                "Valor Económico Neto ($)": net_val,
            }
        )
    return rows

