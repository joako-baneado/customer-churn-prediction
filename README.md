# Proyecto Integrador: Predicción de Fuga de Clientes (Customer Churn)

**Curso:** Data Mining Tools – CC209  
**Docente:** Carlos Fernando Montoya Cubas  
**Institución:** Universidad Peruana de Ciencias Aplicadas (UPC)  
**Hito:** Trabajo Parcial (TP1) – Semana 7  
**Modalidad:** Trabajo grupal

---

## 1. Definición del problema de Data Science

### 1.1 Contexto y necesidad

El proyecto aborda el problema de **customer churn** en una empresa de telecomunicaciones. El objetivo es identificar clientes con mayor riesgo de cancelar el servicio para priorizar acciones de retención y evitar campañas indiscriminadas sobre toda la cartera.

### 1.2 Unidad de análisis

**Un cliente/contrato individual registrado en el dataset.**

### 1.3 Pregunta principal

> ¿Qué clientes presentan mayor probabilidad de churn y qué variables contractuales, de servicio y facturación están asociadas con ese riesgo, de manera que puedan priorizarse intervenciones preventivas?

### 1.4 Tipo de problema

- **TP1:** clasificación binaria supervisada con desbalance moderado (~73.5% retención vs. ~26.5% fuga).
- **TF1 (proyección):** experimentación sistemática, interpretabilidad, análisis de errores y una técnica adicional solo si resulta pertinente para los datos y el problema.

### 1.5 Criterios de utilidad definidos por el grupo

1. **Superar los baselines:** PR-AUC superior a 0.60 frente a una prevalencia cercana a 0.265.
2. **Cobertura de riesgo:** meta interna de `Recall >= 0.75` para la clase churn.
3. **Utilidad económica simulada positiva:** bajo los supuestos de costo/beneficio definidos por el grupo.

Estos criterios son objetivos internos del proyecto, no requisitos impuestos por la rúbrica.

---

## 2. Dataset y procedencia

- **Dataset:** Telco Customer Churn.
- **Fuente declarada:** IBM Cognos Analytics sample / Kaggle (`blastchar`).
- **Dimensiones:** 7,043 observaciones y 21 atributos.
- **Variable objetivo:** `Churn` (`1 = fuga`, `0 = retención`).
- **Licencia:** **pendiente de verificación en la fuente original antes de la entrega final**.
- **Limitación principal:** conjunto transversal; no contiene un historial transaccional o temporal detallado que permita modelar la evolución del cliente en el tiempo.

### 2.1 Variables relevantes

| Variable | Tipo | Descripción |
| --- | --- | --- |
| `customerID` | ID | Identificador único; se excluye del modelamiento. |
| `tenure` | Numérica discreta | Meses del cliente con la compañía. |
| `MonthlyCharges` | Numérica continua | Cargo mensual. |
| `TotalCharges` | Numérica continua | Cargos acumulados. |
| `Contract` | Categórica | Month-to-month, One year, Two year. |
| `InternetService` | Categórica | DSL, Fiber optic, No. |
| `TechSupport` | Categórica | Estado del servicio de soporte técnico. |
| `PaymentMethod` | Categórica | Método de pago. |
| `Churn` | Binaria | Variable objetivo. |

El diccionario completo se conserva en el informe y en el EDA.

### 2.2 Calidad de datos

- Se identificaron **11 valores vacíos en `TotalCharges`**.
- Los 11 casos corresponden a clientes con `tenure = 0`.
- Por regla lógica del problema, esos registros se representan con `TotalCharges = 0`.
- Después de esta corrección no quedan faltantes en `TotalCharges`.
- Se verificaron **0 duplicados completos** y **0 `customerID` repetidos**. Existen 22 filas idénticas a otra al excluir el identificador; se conservan porque corresponden a clientes distintos con el mismo perfil.
- Las categorías no presentan variantes de escritura. `No internet service` (1,526 clientes) y `No phone service` (682) son redundantes con `InternetService` y `PhoneService`; se conservan en el TP1.
- Ninguna variable numérica (`tenure`, `MonthlyCharges`, `TotalCharges`) tiene valores fuera de los límites IQR, por lo que no se recorta ni se transforma ninguna.
- `customerID` se excluye del modelamiento por ser un identificador.
- `SeniorCitizen` se trata como variable categórica binaria.

El `SimpleImputer(strategy="median")` se mantiene dentro del pipeline como respaldo ante faltantes numéricos inesperados y aprende sus estadísticas únicamente a partir de los datos usados para entrenamiento.

---

## 3. Decisiones de herramientas

| Necesidad | Herramienta elegida | Alternativa considerada | Justificación |
| --- | --- | --- | --- |
| EDA | Pandas + Seaborn / Matplotlib | Herramientas de profiling automático | Permite formular preguntas específicas y acompañar cada visualización con interpretación. |
| Preparación | `Pipeline` + `ColumnTransformer` | Preprocesamiento manual / `get_dummies` | Encapsula imputación, escalado y One-Hot Encoding en un flujo reproducible y permite ajustar transformadores solo con los datos de entrenamiento. `get_dummies` no implica leakage por sí mismo; el riesgo aparece si se aprende información del conjunto reservado antes de evaluar. |
| Modelamiento | Logistic Regression + Random Forest | Árbol individual / boosting | Compara una familia lineal interpretable con un ensamble no lineal, ambas apropiadas para clasificación tabular. |
| Evaluación | PR-AUC, Recall, Precision, F1, ROC-AUC y matriz de confusión | Accuracy aislada | La clase churn es minoritaria; por ello Accuracy no es suficiente para decidir. |
| Experimentación TF1 | Validación cruzada estratificada + búsqueda de hiperparámetros | Un único split | Permitirá medir estabilidad y reducir dependencia de una sola partición. |
| Interpretabilidad TF1 | SHAP / importancia de variables según modelo | Solo métricas globales | Permitirá explicar el comportamiento del modelo a nivel global y local. |
| Despliegue TF1 | Streamlit | API/web app más compleja | Permite construir una demostración funcional de manera rápida y reproducible. |

---

## 4. Hallazgos principales del EDA

1. **Desbalance de clases:** 26.54% churn frente a 73.46% no churn.
2. **Contrato:** la tasa de churn es de 42.7% en `Month-to-month`, 11.3% en contratos de un año y 2.8% en contratos de dos años.
3. **Antigüedad:** la tasa de churn es de 54.3% entre 0 y 5 meses de `tenure`, 36.0% entre 6 y 12, 28.7% entre 13 y 24, 20.4% entre 25 y 48 y 9.5% entre 49 y 72.
4. **Fibra óptica y soporte técnico:** dentro de `Fiber optic`, la tasa de churn es de 49.4% sin `TechSupport` y de 22.6% con `TechSupport`.
5. **Método de pago:** `Electronic check` presenta 45.3% de churn, frente a 19.1% de `Mailed check`, 16.7% de transferencia bancaria automática y 15.2% de tarjeta de crédito automática.

Estos resultados representan **asociaciones observadas en la muestra y no relaciones causales**.

---

## 5. Separación de datos y prevención de leakage

Se utiliza una partición estratificada con `random_state=42`:

| Conjunto | Registros | Proporción | Churn |
| --- | ---: | ---: | ---: |
| Train | 4,930 | ~70% | 26.53% |
| Validation | 1,056 | ~15% | 26.52% |
| Test | 1,057 | ~15% | 26.58% |

Flujo de evaluación:

```text
Train
  -> entrenar Logistic Regression y Random Forest
  -> Validation para comparar y seleccionar por PR-AUC
  -> reentrenar el candidato elegido con Train + Validation
  -> Test para evaluación final
```

Las transformaciones estadísticas se encapsulan en el `Pipeline`. La imputación por mediana, el escalado y las categorías del `OneHotEncoder` se ajustan con los datos usados para entrenamiento en cada etapa; Validation y Test no se utilizan para seleccionar parámetros del preprocesamiento.

---

## 6. Modelos y evaluación

### 6.1 Modelos comparados en Validation

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Valor económico simulado |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Regresión Logística (balanced) | 0.7405 | 0.5067 | **0.8143** | 0.6247 | **0.8448** | 0.6313 | **+$42,700** |
| Random Forest (balanced, depth=10) | **0.7689** | **0.5464** | 0.7571 | **0.6347** | 0.8417 | **0.6327** | +$31,400 |

**Criterio principal de selección:** PR-AUC en Validation.  
Random Forest queda como candidato preliminar con `PR-AUC = 0.6327`, frente a `0.6313` de Regresión Logística. La diferencia es **mínima**, por lo que no se interpreta como superioridad concluyente.

En las métricas complementarias de Validation, la Regresión Logística obtiene mayor Recall (0.8143 frente a 0.7571) y mayor valor económico simulado (+$42,700 frente a +$31,400). La regla de selección por PR-AUC está definida en el código y se aplica de forma automática, por lo que se mantiene en el TP1; su coherencia con los criterios de utilidad del proyecto (sección 1.5) se revisará en el TF1.

### 6.2 Evaluación final en Test

Después de seleccionar el candidato, Random Forest se reentrena con Train + Validation y se evalúa en Test.

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Valor económico simulado |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline trivial (Dummy) | 0.7342 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2658 | -$140,500 |
| Baseline heurístico (Month-to-month & tenure <= 6) | 0.7635 | 0.5721 | 0.4377 | 0.4960 | 0.6596 | 0.3999 | -$40,550 |
| **Random Forest seleccionado** | **0.7833** | **0.5756** | **0.7046** | **0.6336** | **0.8400** | **0.6698** | **+$20,500** |

Matriz de confusión del Random Forest en Test:

- TN = 630
- FP = 146
- FN = 83
- TP = 198

### 6.3 Interpretación crítica

- El modelo seleccionado supera claramente a los dos baselines en PR-AUC.
- La meta interna de `PR-AUC > 0.60` sí se cumple en Test (`0.6698`).
- La utilidad económica simulada es positiva (`+$20,500`) bajo los supuestos definidos por el grupo.
- La meta interna de `Recall >= 0.75` **no se cumple en Test** (`0.7046`). Esto se reporta como una limitación del TP1 y no se utiliza Test para volver a seleccionar otro modelo.
- El valor económico es una **simulación**; no representa ganancias reales observadas.

### 6.4 Supuestos de la simulación económica

| Resultado | Impacto simulado |
| --- | ---: |
| Falso negativo (FN) | -$500 |
| Falso positivo (FP) | -$50 |
| Verdadero positivo (TP) | +$350 |
| Verdadero negativo (TN) | $0 |

Estos valores son supuestos de trabajo del grupo y requieren análisis de sensibilidad en el Trabajo Final.

---

## 7. Limitaciones del TP1 y plan hacia TF1

### 7.1 Limitaciones actuales

- Umbral de decisión fijo en 0.50.
- Hiperparámetros preliminares, sin búsqueda sistemática.
- Aún no se mide estabilidad con validación cruzada.
- La selección entre modelos depende de una diferencia de 0.0014 en PR-AUC y es sensible a la versión de scikit-learn: con la versión 1.9.1 el Random Forest produce otros valores y el script selecciona la Regresión Logística. Por ello `requirements.txt` fija versiones exactas.
- La meta interna de Recall no se sostiene en el Test final.
- Los costos económicos son supuestos del grupo.
- El baseline heurístico fue inspirado por el EDA; en una evaluación completamente ciega, las reglas de dominio deberían fijarse antes de observar un holdout reservado.
- El dataset no contiene suficiente información transaccional para afirmar que se dispone de un RFM o CLV real.

### 7.2 Próximos pasos

1. Aplicar `StratifiedKFold` y reportar promedio y dispersión de métricas.
2. Realizar búsqueda sistemática de hiperparámetros.
3. Evaluar una técnica adicional solo si es pertinente para los datos y el objetivo del proyecto.
4. Analizar el umbral de decisión y los falsos positivos/falsos negativos.
5. Incorporar interpretabilidad global y local.
6. Si se realiza clustering, utilizar variables realmente disponibles y describir los segmentos como perfiles/proxies, no como RFM o CLV real sin datos transaccionales suficientes.
7. Construir un despliegue funcional en Streamlit.

---

## 8. Estructura del repositorio

```text
customer-churn-prediction/
├── data/
│   ├── raw/
│   │   └── telco_customer_churn.csv
│   └── processed/
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
├── notebooks/
│   ├── 01_eda_exploratorio.ipynb
│   └── 02_preparacion_y_modelamiento.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── evaluation.py
│   └── train_models.py
├── models/
│   └── pipeline_tp1_seleccionado.joblib
├── reports/
│   ├── comparacion_modelos_validacion_tp1.csv
│   ├── evaluacion_final_test_tp1.csv
│   └── guion_y_diapositivas_tp1.md
├── README.md
└── requirements.txt
```

---

## 9. Reproducción del proyecto

### 9.1 Clonar y entrar al repositorio

```bash
git clone https://github.com/joako-baneado/customer-churn-prediction.git
cd customer-churn-prediction
```

### 9.2 Crear un entorno virtual

```bash
python -m venv venv
```

PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

CMD:

```cmd
venv\Scripts\activate.bat
```

Linux/macOS:

```bash
source venv/bin/activate
```

### 9.3 Instalar dependencias

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` fija versiones exactas (entorno verificado: Python 3.13, scikit-learn 1.8.0). Con esas versiones el pipeline reproduce las cifras de este README.

### 9.4 Ejecutar el pipeline

```bash
python -m src.train_models
```

El script genera/actualiza:

- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`
- `reports/comparacion_modelos_validacion_tp1.csv`
- `reports/evaluacion_final_test_tp1.csv`
- `models/pipeline_tp1_seleccionado.joblib`

### 9.5 Ejecutar notebooks

Abrir en VS Code/Jupyter y ejecutar en orden:

1. `notebooks/01_eda_exploratorio.ipynb`
2. `notebooks/02_preparacion_y_modelamiento.ipynb`

Los notebooks de entrega deben conservar los outputs, tablas y gráficos visibles.

---

## 10. Uso de IA generativa

Se utilizó ChatGPT como apoyo para **revisión de estructura, depuración, organización del código y mejora de redacción/documentación**. El equipo ejecutó el código, verificó los resultados y mantiene la responsabilidad sobre las decisiones metodológicas, interpretaciones y conclusiones del proyecto.
