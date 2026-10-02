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
    confusion_matrix
)
from src.config import (
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    BENEFIT_TRUE_POSITIVE,
    BENEFIT_TRUE_NEGATIVE
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
