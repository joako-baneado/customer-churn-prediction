"""
Módulo de Configuración Centralizada - Proyecto Integrador TP1 (CC209)
Define rutas, constantes, semillas aleatorias y agrupación de variables.
"""
from pathlib import Path

# Directorio raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Rutas de datos y artefactos
DATA_RAW_PATH = BASE_DIR / "data" / "raw" / "telco_customer_churn.csv"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Semilla de reproducibilidad
RANDOM_STATE = 42

# Estrategia de partición del TP1:
# 70% entrenamiento, 15% validación y 15% prueba final.
# Validation se utiliza para comparar modelos y tomar decisiones.
# Test se reserva hasta después de seleccionar el modelo preliminar.
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

# Definición de variables del problema
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

# Parámetros de costo de negocio (para matriz de utilidad económica)
COST_FALSE_NEGATIVE = 500.0   # Costo de perder al cliente (Customer Lifetime Value perdido)
COST_FALSE_POSITIVE = 50.0    # Costo de intervención preventiva inútil (bono/descuento)
BENEFIT_TRUE_POSITIVE = 350.0 # Beneficio neto estimado al retener al cliente con campaña
BENEFIT_TRUE_NEGATIVE = 0.0   # Cliente que no pensaba irse y no se gastó en él
