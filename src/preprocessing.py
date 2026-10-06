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
    VALIDATION_SIZE,
    TEST_SIZE
)


def load_raw_dataset(filepath=DATA_RAW_PATH) -> pd.DataFrame:
    """
    Carga el dataset bruto y realiza correcciones iniciales de tipado
    y reglas lógicas conocidas, sin aprender estadísticas del conjunto
    completo.
    """
    df = pd.read_csv(filepath)

    # TotalCharges contiene 11 registros vacíos.
    # El EDA mostró que corresponden a clientes con tenure = 0,
    # es decir, clientes nuevos que todavía no acumularon facturación.
    # Por regla lógica del problema, esos casos se representan
    # con TotalCharges = 0.
    total_charges_limpio = df["TotalCharges"].astype(str).str.strip()

    df["TotalCharges"] = pd.to_numeric(
        total_charges_limpio.replace("", np.nan),
        errors="coerce"
    )

    clientes_nuevos_sin_facturacion = (
        df["TotalCharges"].isna()
        & df["tenure"].eq(0)
    )

    df.loc[
        clientes_nuevos_sin_facturacion,
        "TotalCharges"
    ] = 0.0

    # Mapeo de la variable objetivo:
    # 1 = Fuga / Churn
    # 0 = Retenido
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({
            "Yes": 1,
            "No": 0
        })

    # SeniorCitizen representa una categoría binaria,
    # no una magnitud numérica continua.
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    return df


def split_data(
    df: pd.DataFrame,
    validation_size: float = VALIDATION_SIZE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
):
    """
    Divide los datos de forma estratificada en Train,
    Validation y Test.

    Estrategia:
    - Train: 70%
    - Validation: 15%
    - Test: 15%

    Validation se utiliza para comparar modelos y tomar
    decisiones preliminares.

    Test se reserva para la evaluación posterior del modelo
    seleccionado, evitando utilizarlo durante la selección.
    """

    feature_cols = [
        c for c in df.columns
        if c not in [TARGET_COLUMN, ID_COLUMN]
    ]

    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].copy()

    # Primer split:
    # 70% Train
    # 30% bloque temporal para Validation + Test
    holdout_size = validation_size + test_size

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=holdout_size,
        random_state=random_state,
        stratify=y
    )

    # Segundo split:
    # El 30% restante se divide en 15% Validation
    # y 15% Test.
    test_fraction_of_holdout = test_size / holdout_size

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=test_fraction_of_holdout,
        random_state=random_state,
        stratify=y_temp
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


def build_preprocessor() -> ColumnTransformer:
    """
    Construye el ColumnTransformer reproducible.

    Estrategia:
    - Numéricas:
      imputación por mediana + StandardScaler.

    - Categóricas:
      imputación por moda + OneHotEncoder.

    Estas transformaciones se ajustarán exclusivamente
    usando el conjunto de entrenamiento al formar parte
    del Pipeline del modelo.
    """

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                NUMERICAL_FEATURES
            ),
            (
                "cat",
                categorical_transformer,
                CATEGORICAL_FEATURES
            )
        ],
        remainder="drop"
    )

    return preprocessor