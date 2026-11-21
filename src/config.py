"""
Módulo de Configuración Centralizada - Proyecto Integrador TP1 / TF1 (CC209)
Define rutas, constantes, semillas aleatorias, agrupación de variables
y escenarios de evaluación económica.
"""
from pathlib import Path

# Directorio raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Rutas de datos y artefactos
DATA_RAW_PATH = BASE_DIR / "data" / "raw" / "telco_customer_churn.csv"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_TF1_DIR = REPORTS_DIR / "figures_tf1"

# Tracking local de MLflow (archivo SQLite local ignorado por git; resultados exportados a CSV)
MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
MLFLOW_EXPERIMENT_NAME = "cc209_telco_churn_tf1"

# Semilla de reproducibilidad
RANDOM_STATE = 42

# Estrategia de partición del TP1 (mantenida intacta en TF1 para preservar el Test holdout):
# 70% entrenamiento, 15% validación y 15% prueba final.
# En TF1, Train + Validation (85% = 5,986 filas) se usan para CV, Optuna,
# clustering, calibración y optimización de umbral OOF; Test (1,057 filas) permanece reservado.
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

# Configuración de Validación Cruzada Estratificada (TF1)
CV_FOLDS_OPTUNA = 5
CV_FOLDS_EVAL = 10
OPTUNA_N_TRIALS = 25

# Definición de variables base del problema
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

NUMERICAL_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

# Variables derivadas por Feature Engineering de negocio (TF1)
ENGINEERED_NUMERICAL_FEATURES = [
    "TotalServicesSubscribed",
    "BillingRatio",
    "AvgMonthlyCharge",
    "ChargeDelta",
    "HighRisk_Fiber_No_Support",
    "AutoPayFlag",
    "ContractRiskTier"
]

ENGINEERED_CATEGORICAL_FEATURES = [
    "TenureBucket"
]

ALL_NUMERICAL_FEATURES_TF1 = NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_FEATURES
ALL_CATEGORICAL_FEATURES_TF1 = CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL_FEATURES

# Variables empleadas para segmentación no supervisada (perfiles/proxies de valor)
CLUSTERING_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "TotalServicesSubscribed"
]

# Parámetros de costo de negocio base (para matriz de utilidad económica)
COST_FALSE_NEGATIVE = 500.0   # Costo de perder al cliente (LTV estimado perdido)
COST_FALSE_POSITIVE = 50.0    # Costo de intervención preventiva inútil (bono/descuento)
BENEFIT_TRUE_POSITIVE = 350.0 # Beneficio neto estimado al retener al cliente con campaña
BENEFIT_TRUE_NEGATIVE = 0.0   # Cliente que no pensaba irse y no se gastó en él

# Escenarios para análisis de sensibilidad económica en TF1
ECONOMIC_SCENARIOS = {
    "base": {
        "label": "Escenario Base (Supuesto TP1)",
        "cost_fn": 500.0,
        "cost_fp": 50.0,
        "benefit_tp": 350.0,
        "benefit_tn": 0.0,
    },
    "pesimista_30pct": {
        "label": "Escenario Pesimista (30% efectividad de retención)",
        "cost_fn": 500.0,
        "cost_fp": 50.0,
        "benefit_tp": 105.0,
        "benefit_tn": 0.0,
    },
    "agresivo_fp100": {
        "label": "Escenario Agresivo (Costo intervención FP = $100)",
        "cost_fn": 500.0,
        "cost_fp": 100.0,
        "benefit_tp": 350.0,
        "benefit_tn": 0.0,
    },
}

# Meta interna de cobertura (Recall) definida por el equipo
RECALL_TARGET = 0.75

