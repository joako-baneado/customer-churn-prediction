"""
Módulo de Preprocesamiento y Construcción de Pipelines - TP1
Diseñado para garantizar CERO data leakage ajustando transformadores
exclusivamente sobre el conjunto de entrenamiento.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.config import (
    DATA_RAW_PATH,
    TARGET_COLUMN,
    ID_COLUMN,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    RANDOM_STATE,
    TEST_SIZE
)


def load_raw_dataset(filepath=DATA_RAW_PATH) -> pd.DataFrame:
    """
    Carga el dataset bruto y realiza las correcciones de tipado iniciales
    sin involucrar transformaciones estadísticas (para evitar leakage).
    """
    df = pd.read_csv(filepath)

    # Tratamiento de anomalía conocida: TotalCharges contiene espacios en blanco ' '
    # en clientes con tenure = 0 (nuevos registros sin ciclo de facturación completado).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce")

    # Mapeo de la variable objetivo a binario formal (1 = Fuga / Churn, 0 = Retenido)
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    # Asegurar que SeniorCitizen sea tratado como categórico
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    return df


def split_data(df: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """
    Realiza la partición estratificada de datos en Train y Test.
    
    Justificación Metodológica (Rúbrica Criterio 4):
    - Al existir un desbalance moderado (26.5% de Churn), la estratificación garantiza
      que la prevalencia de clientes desertores sea idéntica en ambos conjuntos.
    - El ID del cliente se excluye para evitar memorización o leakage de identificadores.
    """
    feature_cols = [c for c in df.columns if c not in [TARGET_COLUMN, ID_COLUMN]]
    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def build_preprocessor() -> ColumnTransformer:
    """
    Construye el ColumnTransformer reproducible.
    
    Estrategia de transformación (Rúbrica Criterio 5):
    - Numéricas: Imputación por mediana (robusta a asimetría) + StandardScaler.
    - Categóricas: Imputación por moda + OneHotEncoder(drop='first', handle_unknown='ignore')
      para prevenir colinealidad exacta en modelos lineales.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    return preprocessor
