"""
Script de Entrenamiento y Comparación Preliminar de Modelos - TP1.

Flujo metodológico:
1. Cargar datos.
2. Separar Train / Validation / Test.
3. Entrenar y comparar modelos únicamente con Train y Validation.
4. Seleccionar el mejor candidato usando PR-AUC de Validation.
5. Reentrenar el modelo seleccionado con Train + Validation.
6. Evaluar una sola vez sobre Test.
"""

import sys
from pathlib import Path

# Agregar el directorio raíz al PYTHONPATH si es necesario.
BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.config import (
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    RANDOM_STATE
)

from src.preprocessing import (
    load_raw_dataset,
    split_data,
    build_preprocessor
)

from src.evaluation import (
    evaluate_classifier,
    BusinessRuleBaseline
)


def build_candidate_models():
    """
    Construye modelos nuevos e independientes.

    Cada modelo recibe su propio preprocesador para evitar compartir
    transformadores ajustados entre pipelines distintos.
    """

    logistic_regression = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor()
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE
                )
            )
        ]
    )

    random_forest = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor()
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            )
        ]
    )

    return {
        "Regresión Logística (Balanced)": logistic_regression,
        "Random Forest (Balanced, depth=10)": random_forest
    }


def export_partitions(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test
):
    """
    Guarda las tres particiones utilizadas en el TP1 para facilitar
    la reproducción y auditoría del experimento.
    """

    DATA_PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_export = X_train.copy()
    train_export["Churn"] = y_train

    validation_export = X_val.copy()
    validation_export["Churn"] = y_val

    test_export = X_test.copy()
    test_export["Churn"] = y_test

    train_export.to_csv(
        DATA_PROCESSED_DIR / "train.csv",
        index=False
    )

    validation_export.to_csv(
        DATA_PROCESSED_DIR / "validation.csv",
        index=False
    )

    test_export.to_csv(
        DATA_PROCESSED_DIR / "test.csv",
        index=False
    )


def run_pipeline():

    print("=" * 70)
    print("PROYECTO INTEGRADOR CC209 - MODELAMIENTO PRELIMINAR TP1")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Carga del dataset
    # ---------------------------------------------------------

    df = load_raw_dataset()

    print(
        f"[1/7] Dataset cargado: "
        f"{df.shape[0]} filas y {df.shape[1]} columnas."
    )

    # ---------------------------------------------------------
    # 2. Split Train / Validation / Test
    # ---------------------------------------------------------

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_data(df)

    print("[2/7] Partición estratificada realizada:")

    print(
        f"      Train:      {len(X_train):4d} "
        f"| Churn: {y_train.mean():.2%}"
    )

    print(
        f"      Validation: {len(X_val):4d} "
        f"| Churn: {y_val.mean():.2%}"
    )

    print(
        f"      Test:       {len(X_test):4d} "
        f"| Churn: {y_test.mean():.2%}"
    )

    # ---------------------------------------------------------
    # 3. Guardar particiones
    # ---------------------------------------------------------

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    export_partitions(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )

    print(
        "[3/7] Particiones guardadas en data/processed/."
    )

    # ---------------------------------------------------------
    # 4. Comparación de candidatos usando Validation
    # ---------------------------------------------------------

    candidate_models = build_candidate_models()

    validation_results = []

    print(
        "[4/7] Entrenando candidatos con Train "
        "y evaluando sobre Validation..."
    )

    for name, model in candidate_models.items():

        # Pipeline completo ajustado exclusivamente con Train.
        model.fit(
            X_train,
            y_train
        )

        result = evaluate_classifier(
            model,
            X_val,
            y_val,
            model_name=name
        )

        validation_results.append(result)

        print(
            f"      {name}"
        )

        print(
            f"        PR-AUC: "
            f"{result['PR-AUC']:.4f}"
        )

        print(
            f"        Recall: "
            f"{result['Recall (Clase 1)']:.4f}"
        )

        print(
            f"        Valor económico simulado: "
            f"${result['Valor Económico Neto ($)']:,.0f}"
        )

    df_validation = pd.DataFrame(
        validation_results
    )

    validation_path = (
        REPORTS_DIR
        / "comparacion_modelos_validacion_tp1.csv"
    )

    df_validation.to_csv(
        validation_path,
        index=False
    )

    # ---------------------------------------------------------
    # 5. Selección del candidato
    # ---------------------------------------------------------

    # PR-AUC es la métrica principal porque existe desbalance
    # de clases y el objetivo es identificar correctamente
    # clientes con riesgo de fuga.
    best_row = df_validation.loc[
        df_validation["PR-AUC"].idxmax()
    ]

    selected_model_name = best_row["Modelo"]

    print("[5/7] Modelo preliminar seleccionado:")
    print(
        f"      {selected_model_name}"
    )
    print(
        f"      PR-AUC Validation: "
        f"{best_row['PR-AUC']:.4f}"
    )

    # ---------------------------------------------------------
    # 6. Reentrenamiento con Train + Validation
    # ---------------------------------------------------------

    X_train_final = pd.concat(
        [
            X_train,
            X_val
        ],
        axis=0
    )

    y_train_final = pd.concat(
        [
            y_train,
            y_val
        ],
        axis=0
    )

    # Se crea un pipeline nuevo para evitar reutilizar
    # transformadores previamente ajustados.
    final_candidates = build_candidate_models()

    selected_model = final_candidates[
        selected_model_name
    ]

    selected_model.fit(
        X_train_final,
        y_train_final
    )

    print(
        "[6/7] Modelo seleccionado reentrenado "
        "con Train + Validation."
    )

    # ---------------------------------------------------------
    # 7. Evaluación final en Test
    # ---------------------------------------------------------

    final_test_results = []

    # Baseline trivial
    dummy = DummyClassifier(
        strategy="most_frequent"
    )

    dummy.fit(
        X_train_final,
        y_train_final
    )

    final_test_results.append(
        evaluate_classifier(
            dummy,
            X_test,
            y_test,
            model_name="Baseline Trivial (Dummy)"
        )
    )

    # Baseline heurístico
    business_baseline = BusinessRuleBaseline(
        tenure_threshold=6
    )

    business_baseline.fit(
        X_train_final,
        y_train_final
    )

    final_test_results.append(
        evaluate_classifier(
            business_baseline,
            X_test,
            y_test,
            model_name=(
                "Baseline Heurístico "
                "(Mes-a-Mes & Tenure <= 6)"
            )
        )
    )

    # Modelo seleccionado
    final_test_results.append(
        evaluate_classifier(
            selected_model,
            X_test,
            y_test,
            model_name=selected_model_name
        )
    )

    df_test = pd.DataFrame(
        final_test_results
    )

    test_results_path = (
        REPORTS_DIR
        / "evaluacion_final_test_tp1.csv"
    )

    df_test.to_csv(
        test_results_path,
        index=False
    )

    # Serialización del pipeline completo.
    model_path = (
        MODELS_DIR
        / "pipeline_tp1_seleccionado.joblib"
    )

    joblib.dump(
        selected_model,
        model_path
    )

    print("[7/7] Evaluación final sobre Test:")
    print()

    display_cols = [
        "Modelo",
        "Accuracy",
        "Precision (Clase 1)",
        "Recall (Clase 1)",
        "F1-Score",
        "ROC-AUC",
        "PR-AUC",
        "TN",
        "FP",
        "FN",
        "TP",
        "Valor Económico Neto ($)"
    ]

    print(
        df_test[
            display_cols
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Pipeline serializado en: "
        f"{model_path}"
    )

    print()
    print(
        "NOTA: El valor económico corresponde a una "
        "simulación basada en supuestos del grupo."
    )

    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()