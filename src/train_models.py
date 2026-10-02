"""
Script de Entrenamiento y Comparación Preliminar de Modelos - TP1
Ejecuta el flujo completo de modelamiento, genera métricas de evaluación
y serializa artefactos para asegurar reproducibilidad total.
"""
import sys
from pathlib import Path

# Agregar directorio raíz al PYTHONPATH si es necesario
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
from src.preprocessing import load_raw_dataset, split_data, build_preprocessor
from src.evaluation import evaluate_classifier, BusinessRuleBaseline


def run_pipeline():
    print("=" * 60)
    print("PROYECTO INTEGRADOR CC209 - ENTRENAMIENTO PRELIMINAR (TP1)")
    print("=" * 60)

    # 1. Carga y verificación de datos
    df = load_raw_dataset()
    print(f"[1/6] Dataset cargado: {df.shape[0]} filas y {df.shape[1]} columnas.")

    # 2. Split estratificado anti-leakage
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.20, random_state=RANDOM_STATE)
    print(f"[2/6] Split estratificado realizado:")
    print(f"      - Train: {X_train.shape[0]} muestras (Tasa Churn: {y_train.mean():.2%})")
    print(f"      - Test:  {X_test.shape[0]} muestras (Tasa Churn: {y_test.mean():.2%})")

    # Guardar particiones procesadas para reproducibilidad
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_export = X_train.copy()
    train_export["Churn"] = y_train
    train_export.to_csv(DATA_PROCESSED_DIR / "train.csv", index=False)

    test_export = X_test.copy()
    test_export["Churn"] = y_test
    test_export.to_csv(DATA_PROCESSED_DIR / "test.csv", index=False)
    print("      -> Particiones guardadas en 'data/processed/'.")

    # 3. Construcción del preprocesador reproducible
    preprocessor = build_preprocessor()
    print("[3/6] ColumnTransformer instanciado correctamente.")

    # 4. Definición de modelos a comparar
    models_to_evaluate = {
        "Baseline Trivial (Dummy Most Frequent)": DummyClassifier(strategy="most_frequent"),
        "Baseline Heurístico (Regla Mes-a-Mes & Tenure <= 6)": BusinessRuleBaseline(tenure_threshold=6),
        "Modelo 1: Regresión Logística (Balanced)": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE))
        ]),
        "Modelo 2: Random Forest (Balanced, depth=10)": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ])
    }

    # 5. Ajuste y evaluación
    print("[4/6] Entrenando y evaluando modelos...")
    results = []
    fitted_models = {}

    for name, model in models_to_evaluate.items():
        # Ajustar únicamente en entrenamiento
        if hasattr(model, "fit"):
            model.fit(X_train, y_train)
        
        # Evaluar sobre test
        res = evaluate_classifier(model, X_test, y_test, model_name=name)
        results.append(res)
        fitted_models[name] = model
        print(f"      [OK] {name}: PR-AUC = {res['PR-AUC']:.4f} | Recall = {res['Recall (Clase 1)']:.4f} | Utilidad = ${res['Valor Económico Neto ($)']:,.0f}")

    # Consolidar resultados
    df_results = pd.DataFrame(results)
    results_path = REPORTS_DIR / "tabla_comparativa_modelos_tp1.csv"
    df_results.to_csv(results_path, index=False)
    print(f"[5/6] Tabla comparativa exportada en '{results_path}'.")

    # 6. Serialización del mejor modelo del TP1
    # En este caso, Regresión Logística o Random Forest
    best_candidate_name = "Modelo 1: Regresión Logística (Balanced)"
    best_model = fitted_models[best_candidate_name]
    best_model_path = MODELS_DIR / "pipeline_tp1_logreg.joblib"
    joblib.dump(best_model, best_model_path)
    print(f"[6/6] Pipeline serializado con éxito en '{best_model_path}'.")

    print("\n" + "=" * 60)
    print("RESUMEN DE RENDIMIENTO PRELIMINAR (TP1):")
    print("=" * 60)
    display_cols = ["Modelo", "Accuracy", "Precision (Clase 1)", "Recall (Clase 1)", "ROC-AUC", "PR-AUC", "Valor Económico Neto ($)"]
    print(df_results[display_cols].to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
