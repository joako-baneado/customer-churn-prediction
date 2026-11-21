"""
Contratos de datos y validación de entrada para inferencia (TF1 - CC209).
Usa Pydantic v2 para garantizar integridad de esquema tanto en diagnóstico
individual como en scoring por lotes en Streamlit.
"""

from typing import Literal, Tuple, List
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES


REQUIRED_INPUT_COLUMNS: List[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


class CustomerInput(BaseModel):
    """Contrato de validación para un cliente individual."""

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal["0", "1"]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=120, description="Antigüedad en meses (0-120)")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(gt=0.0, le=1000.0, description="Cargo mensual en USD")
    TotalCharges: float = Field(ge=0.0, description="Cargos acumulados en USD")

    @field_validator("SeniorCitizen", mode="before")
    @classmethod
    def normalize_senior_citizen(cls, value):
        if isinstance(value, bool):
            raise ValueError("SeniorCitizen debe ser 0 o 1, no booleano")
        if isinstance(value, (int, float)) and value in (0, 1):
            return str(int(value))
        if isinstance(value, str) and value.strip() in {"0", "1"}:
            return value.strip()
        raise ValueError("SeniorCitizen debe ser '0' o '1'")

    @field_validator("TotalCharges", mode="before")
    @classmethod
    def normalize_total_charges(cls, value):
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return 0.0
        if pd.isna(value):
            return 0.0
        return float(value)

    @model_validator(mode="after")
    def validate_business_coherence(self):
        if self.tenure == 0 and self.TotalCharges > 0:
            raise ValueError(
                "Regla de coherencia: si tenure == 0, TotalCharges debe ser 0.0"
            )
        if self.PhoneService == "No" and self.MultipleLines != "No phone service":
            raise ValueError(
                "Regla de coherencia: sin PhoneService, MultipleLines debe ser 'No phone service'"
            )
        if self.InternetService == "No":
            internet_addons = [
                self.OnlineSecurity,
                self.OnlineBackup,
                self.DeviceProtection,
                self.TechSupport,
                self.StreamingTV,
                self.StreamingMovies,
            ]
            if any(val != "No internet service" for val in internet_addons):
                raise ValueError(
                    "Regla de coherencia: sin InternetService, los adicionales deben ser 'No internet service'"
                )
        return self

    def to_dataframe(self) -> pd.DataFrame:
        """Convierte el registro validado a DataFrame con el orden de columnas esperado."""
        data = self.model_dump()
        return pd.DataFrame([data], columns=CATEGORICAL_FEATURES + NUMERICAL_FEATURES)


def validate_batch_dataframe(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Valida un DataFrame de entrada por lotes fila por fila usando CustomerInput.

    Retorna:
    - df_valid: DataFrame limpio y normalizado con las filas válidas.
    - errors: Lista legible de errores por fila (si los hubiera).
    """
    missing_cols = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Faltan columnas requeridas en el archivo: {missing_cols}"
        )

    valid_records = []
    valid_customer_ids = []
    has_customer_id = "customerID" in df.columns
    errors: List[str] = []

    for idx, row in df.iterrows():
        payload = {col: row[col] for col in REQUIRED_INPUT_COLUMNS}
        try:
            validated = CustomerInput(**payload)
            valid_records.append(validated.model_dump())
            if has_customer_id:
                valid_customer_ids.append(row["customerID"])
        except Exception as exc:
            errors.append(f"Fila {idx}: {exc}")

    if not valid_records:
        return pd.DataFrame(columns=REQUIRED_INPUT_COLUMNS), errors

    df_valid = pd.DataFrame(valid_records)
    if has_customer_id:
        df_valid.insert(0, "customerID", valid_customer_ids)

    return df_valid, errors
