"""
Módulo de Ingeniería de Características de Negocio (TF1 - CC209).
Todas las transformaciones de TelcoFeatureEngineer son determinísticas
y calculadas fila por fila (stateless), garantizando CERO data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import (
    ENGINEERED_NUMERICAL_FEATURES,
    ENGINEERED_CATEGORICAL_FEATURES,
)


ADDON_SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

CONTRACT_RISK_MAP = {
    "Month-to-month": 2,
    "One year": 1,
    "Two year": 0,
}


class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Genera variables de dominio para retención de clientes Telco:
    - TotalServicesSubscribed: grado de empaquetamiento (0 a 8 servicios).
    - BillingRatio: razón entre facturación acumulada real y esperada.
    - AvgMonthlyCharge: cargo mensual promedio histórico.
    - ChargeDelta: diferencia entre cargo actual y promedio histórico.
    - HighRisk_Fiber_No_Support: interacción crítica Fibra Óptica sin soporte.
    - AutoPayFlag: indicador de pago automático.
    - ContractRiskTier: codificación ordinal de riesgo contractual (2, 1, 0).
    - TenureBucket: tramo de antigüedad comercial.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def fit(self, X: pd.DataFrame, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns.tolist(), dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if not self.enabled:
            return df

        tenure = pd.to_numeric(df["tenure"], errors="coerce").fillna(0.0).clip(lower=0.0)
        monthly = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0)
        total = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

        # 1. Conteo de servicios contratados (0 a 8)
        services_count = (df["PhoneService"].astype(str) == "Yes").astype(int)
        services_count = services_count + (df["InternetService"].astype(str) != "No").astype(int)
        for col in ADDON_SERVICE_COLUMNS:
            services_count = services_count + (df[col].astype(str) == "Yes").astype(int)
        df["TotalServicesSubscribed"] = services_count.astype(int)

        # 2. BillingRatio (si tenure == 0 o expected == 0 -> 1.0 para evitar divisiones por cero)
        expected_total = tenure * monthly
        billing_ratio = np.where(
            (tenure > 0) & (expected_total > 0),
            total / (expected_total + 1e-5),
            1.0,
        )
        df["BillingRatio"] = np.asarray(billing_ratio, dtype=float)

        # 3. AvgMonthlyCharge y ChargeDelta
        avg_monthly = total / np.maximum(tenure, 1.0)
        df["AvgMonthlyCharge"] = np.asarray(avg_monthly, dtype=float)
        df["ChargeDelta"] = np.asarray(monthly - avg_monthly, dtype=float)

        # 4. Indicador de vulnerabilidad crítica: Fibra óptica sin soporte técnico
        df["HighRisk_Fiber_No_Support"] = (
            (df["InternetService"].astype(str) == "Fiber optic")
            & (df["TechSupport"].astype(str) == "No")
        ).astype(int)

        # 5. Indicador de pago automático
        df["AutoPayFlag"] = (
            df["PaymentMethod"].astype(str).str.contains("(automatic)", regex=False)
        ).astype(int)

        # 6. Nivel ordinal de riesgo de contrato
        df["ContractRiskTier"] = (
            df["Contract"].astype(str).map(CONTRACT_RISK_MAP).fillna(1).astype(int)
        )

        # 7. Tramo de antigüedad (TenureBucket)
        conditions = [
            tenure <= 6,
            (tenure > 6) & (tenure <= 12),
            (tenure > 12) & (tenure <= 24),
            (tenure > 24) & (tenure <= 48),
            tenure > 48,
        ]
        choices = ["0-6m", "7-12m", "13-24m", "25-48m", "49m+"]
        df["TenureBucket"] = np.select(conditions, choices, default="0-6m").astype(str)

        return df

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            if hasattr(self, "feature_names_in_"):
                base = list(self.feature_names_in_)
            else:
                base = []
        else:
            base = list(input_features)
        if not self.enabled:
            return np.asarray(base, dtype=object)
        extra = [
            c
            for c in (ENGINEERED_NUMERICAL_FEATURES + ENGINEERED_CATEGORICAL_FEATURES)
            if c not in base
        ]
        return np.asarray(base + extra, dtype=object)
