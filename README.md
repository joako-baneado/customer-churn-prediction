# Proyecto Integrador: Sistema Predictivo de Fuga de Clientes con Segmentación de Valor (Customer Churn)

**Curso:** Data Mining Tools – CC209  
**Docente:** Carlos Fernando Montoya Cubas  
**Institución:** Universidad Peruana de Ciencias Aplicadas (UPC)  
**Hito Evaluado:** Trabajo Parcial (TP1) – Semana 7 (10% del curso)  
**Modalidad:** Trabajo Grupal  

---

## 1. Definición del Problema de Data Science (Rúbrica: 3 Pts)

### 1.1. Contexto de Negocio y Necesidad
En la industria de telecomunicaciones y servicios digitales por suscripción (*Telco / SaaS*), la tasa promedio de deserción mensual (*churn rate*) ronda el 20% al 30%. La adquisición de un nuevo cliente representa un costo entre 5 y 7 veces superior al de retener a un cliente actual. Sin embargo, aplicar campañas masivas e indiferenciadas de descuentos o beneficios resulta financieramente insostenible y erosiona los márgenes operativos.

### 1.2. Unidad de Análisis
**Un contrato / cliente individual activo durante un ciclo mensual de facturación.**

### 1.3. Pregunta Principal del Proyecto
> *¿Cuál es la probabilidad de que un cliente cancele su contrato de servicios en el próximo ciclo de facturación, y qué variables contractuales, técnicas y de consumo explican con mayor peso dicha decisión para priorizar intervenciones preventivas rentables?*

### 1.4. Tipo de Problema de Data Science
* **TP1:** Clasificación binaria supervisada con desbalance moderado (~73.5% retención vs. ~26.5% fuga).
* **TF1 (Proyección):** Aprendizaje semi-supervisado / no supervisado complementario mediante **Clustering (K-Means / RFM)** para segmentar clientes por *Customer Lifetime Value* (CLV) cruzado con riesgo de fuga.

### 1.5. Criterios de Utilidad y Éxito de la Solución
Para evitar formulaciones triviales, el proyecto no busca únicamente maximizar métricas genéricas, sino satisfacer criterios de decisión económica:
1. **Superación del Baseline:** Superar significativamente al baseline trivial (Dummy) y a una regla de negocio heurística en términos de discriminación en clases minoritarias (**PR-AUC superior a 0.60** vs 0.265 del azar).
2. **Cobertura de Riesgo (`Recall` $\ge 0.75$):** Detectar al menos al 75% de los clientes que efectivamente van a abandonar la empresa.
3. **Rentabilidad Neta Positiva:** Demostrar que el valor económico generado por clientes retenidos mediante alertas tempranas supera holgadamente el costo combinado de los incentivos comerciales (falsos positivos) y el valor de vida perdido (falsos negativos).

---

## 2. Dataset y Procedencia

* **Fuente:** Repositorio público oficial de *IBM Cognos Analytics / Kaggle (Telco Customer Churn)*.
* **Licencia:** Open Data Commons / Apache 2.0 (Uso libre para fines académicos y de investigación).
* **Dimensiones:** 7,043 observaciones y 21 atributos.
* **Período Temporal:** Registro transversal consolidado de clientes residenciales en California (Q3).
* **Variable Objetivo:** `Churn` (Codificada como `1 = Fuga / Yes` y `0 = Retención / No`).

### 2.1. Diccionario de Variables Relevante

| Variable | Tipo | Descripción y Dominio |
| :--- | :--- | :--- |
| `customerID` | Categórica (ID) | Identificador alfanumérico único (excluido del modelamiento para evitar leakage). |
| `gender` | Categórica binaria | Género del cliente (`Male`, `Female`). |
| `SeniorCitizen` | Categórica binaria | Indicador si el cliente es adulto mayor (`1`, `0`). |
| `Partner` / `Dependents` | Categórica binaria | Si el cliente tiene cónyuge / dependientes (`Yes`, `No`). |
| `tenure` | Numérica discreta | Número de meses que el cliente ha permanecido con la compañía (0 a 72). |
| `PhoneService` / `MultipleLines`| Categórica | Si cuenta con telefonía fija y líneas múltiples (`Yes`, `No`, `No phone service`). |
| `InternetService` | Categórica nominal | Proveedor de conexión (`DSL`, `Fiber optic`, `No`). |
| `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | Categórica | Servicios de valor agregado suscritos (`Yes`, `No`, `No internet service`). |
| `Contract` | Categórica ordinal | Duración del compromiso contractual (`Month-to-month`, `One year`, `Two year`). |
| `PaperlessBilling` | Categórica binaria | Modalidad de facturación electrónica sin papel (`Yes`, `No`). |
| `PaymentMethod` | Categórica nominal | Canal de pago (`Electronic check`, `Mailed check`, `Bank transfer`, `Credit card`). |
| `MonthlyCharges` | Numérica continua | Monto facturado mensual recurrente (\$18.25 a \$118.75). |
| `TotalCharges` | Numérica continua | Importe monetario acumulado facturado (\$18.80 a \$8684.80). |
| **`Churn`** | **Binaria (Objetivo)** | **Si el cliente canceló el servicio (`Yes` $\rightarrow 1$, `No` $\rightarrow 0$).** |

### 2.2. Problemas de Calidad Identificados y Limitaciones
* **Espacios en Blanco en `TotalCharges`:** 11 registros presentaban una cadena vacía `' '` en lugar de un número, provocando que la columna fuera leída erróneamente como texto (`object`). Al cruzar estos registros, se demostró que corresponden exactamente a clientes nuevos con `tenure = 0` (recién suscritos antes del primer cierre mensual) y ninguno había desertado. Se convirtió a tipo numérico imputando 0.
* **Limitación Temporal:** Al ser un corte transversal, no se dispone de registros de series temporales de llamadas diarias o logs de red, por lo que la modelación se enfoca en características contractuales y de perfil.

---

## 3. Matriz de Decisiones de Herramientas (Rúbrica Sección 6)

| Necesidad | Herramienta Elegida | Alternativa Considerada | Justificación Técnica |
| :--- | :--- | :--- | :--- |
| **EDA** | `Pandas` + `Seaborn` / `Matplotlib` | Sweetviz / Pandas Profiling | Generación de visualizaciones orientadas a hipótesis de negocio específicas en lugar de reportes automáticos ciegos; permite documentar explícitamente "lo que los datos muestran" vs. "lo que el equipo interpreta". |
| **Preparación** | `Scikit-Learn ColumnTransformer` | Pandas `get_dummies` directo | `get_dummies` induce data leakage al codificar categorías basadas en toda la muestra; `ColumnTransformer` encapsula `OneHotEncoder` y `StandardScaler` asegurando que los transformadores se ajusten **únicamente** sobre Train. |
| **Modelamiento Preliminar** | `LogisticRegression(class_weight='balanced')` y `RandomForestClassifier` | Redes Neuronales / XGBoost sin tunear | Cumple con evaluar dos familias distintas (lineal paramétrica vs ensamble no paramétrico de árboles) con bajo costo computacional y alta interpretabilidad inicial para el corte de Semana 7. |
| **Experimentación (TF1)** | `Optuna` + `MLflow` | GridSearchCV manual | Optuna ofrece optimización bayesiana eficiente del espacio de hiperparámetros; MLflow garantiza trazabilidad de experimentos y artefactos para el TF1. |
| **Interpretabilidad (TF1)**| `SHAP` (SHapley Additive exPlanations) | Feature Importance MDI básica | La importancia nativa de impureza en árboles sobreestima variables numéricas de alta cardinalidad; SHAP ofrece explicaciones locales individualizadas (Waterfall plots) matemáticamente fundamentadas. |
| **Despliegue (TF1)** | `Streamlit` | Flask / Django | Permite construir en Python puro una interfaz web interactiva con simulador en tiempo real y gráficos explicativos para usuarios de negocio sin sobrecarga de frontend. |

---

## 4. Hallazgos Clave del EDA

1. **Desbalance de Clases:** 26.54% de deserción frente a 73.46% de retención. El *Accuracy* queda descartado como métrica guía.
2. **Modalidad Contractual:** Los clientes con contrato mensual (*Month-to-month*) presentan una tasa de fuga del **42.7%**, contra solo **11.3%** en contratos anuales y **2.8%** en contratos bianuales.
3. **Ventana Crítica de Antigüedad:** El pico de abandono se concentra durante los **primeros 5 meses** (fenómeno de *onboarding friction*). Si el cliente supera el primer año, la retención se consolida.
4. **Vulnerabilidad en Fibra Óptica:** Clientes con Internet de Fibra Óptica sin Soporte Técnico (`TechSupport = No`) presentan una fuga del **49.4%**, debido al costo elevado (\$80+/mes) sumado a la falta de asistencia ante incidencias.
5. **Canal de Pago:** El pago mediante cheque electrónico (*Electronic check*) presenta una deserción del **45.3%**, duplicando a los métodos con débito automático bancario.

---

## 5. Prevención Rigurosa de Data Leakage (Rúbrica: 2 Pts)

Para garantizar la integridad del flujo experimental:
1. **Partición Previa a Cualquier Transformación:** Se realiza un `train_test_split` estratificado (80% Train, 20% Test) inmediatamente tras la carga de datos.
2. **Estratificación Justificada:** Dado el desbalance de clases (26.5%), la estratificación preserva idéntica prevalencia en entrenamiento (5,634 muestras) y prueba (1,409 muestras).
3. **Encapsulamiento en Pipeline:** Las medias, medianas, desviaciones estándar y categorías únicas para `OneHotEncoder` se calculan exclusivamente en `X_train` (`fit`) y se aplican sin re-ajuste a `X_test` (`transform`).

---

## 6. Resultados Comparativos de Modelos y Evaluación de Negocio (Rúbrica: 3 Pts)

### 6.1. Definición de la Función de Utilidad Económica
* **Falso Negativo (FN):** Cliente que cancela sin ser detectado. Pérdida del Customer Lifetime Value: **-\$500 USD**.
* **Falso Positivo (FP):** Cliente que no pensaba irse pero recibe un incentivo/descuento preventivo: **-\$50 USD**.
* **Verdadero Positivo (TP):** Cliente en riesgo detectado a tiempo y retenido con éxito mediante oferta: **+\$350 USD netos**.
* **Verdadero Negativo (TN):** Cliente retenido no contactado: **\$0 USD**.

### 6.2. Tabla de Resultados sobre Muestra de Prueba (1,409 clientes)

| Modelo | Accuracy | Recall (Clase 1) | Precision (Clase 1) | ROC-AUC | PR-AUC | Valor Económico Neto ($ USD) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Trivial (Dummy Most Frequent)** | 73.46% | 0.00% | 0.00% | 0.500 | 0.265 | **-\$187,000** |
| **Baseline Heurístico (Mes-a-Mes & Tenure $\le 6$)** | 76.93% | 45.45% | 58.42% | 0.669 | 0.410 | **-\$48,550** |
| **Modelo 1: Regresión Logística (Balanced)** | 73.88% | **78.34%** | 50.52% | **0.842** | 0.633 | **+\$47,700** |
| **Modelo 2: Random Forest (Balanced, depth=10)** | **75.59%** | **78.88%** | **52.68%** | 0.841 | **0.655** | **+\$50,500** |

### 6.3. Interpretación Crítica de Resultados
* **El engaño del Accuracy:** El baseline trivial obtiene un 73.5% de exactitud aparente pero causa una pérdida económica de **-\$187,000 USD** al no prevenir ni una sola fuga.
* **Aporte Demostrado del Machine Learning:** Ambos modelos de ML logran capturar casi el **79% de los clientes en fuga** (295 de 374), generando una utilidad neta superior a los **+\$50,000 USD** frente a las pérdidas del baseline heurístico.
* **Modelo Preliminar Seleccionado:** **Random Forest** obtiene el mejor balance global con un **PR-AUC de 0.655** y la mayor rentabilidad económica esperada.

---

## 7. Estado del Proyecto y Plan hacia el Trabajo Final (TF1 — Semana 15) (Rúbrica: 2 Pts)

### 7.1. Limitaciones Identificadas en el TP1
1. Los modelos preliminares utilizan un umbral fijo de probabilidad ($0.50$), sin calibración matemática específica para minimizar la función de costo financiero.
2. No se ha implementado búsqueda sistemática de hiperparámetros ni algoritmos de Gradient Boosting.
3. Se trata a la cartera como un grupo homogéneo, sin segmentación por valor del cliente (*Customer Lifetime Value*).

### 7.2. Hoja de Ruta para el TF1
* **Hito 1 (Semana 8–10) – Experimentación Avanzada:** Integración de **LightGBM / XGBoost**, optimización de hiperparámetros con **Optuna** y seguimiento de corridas en **MLflow**.
* **Hito 2 (Semana 11–12) – Técnica Adicional (Clustering):** Aplicación de **K-Means / RFM** para segmentar clientes en 3 niveles de valor comercial y cruzar con la probabilidad de fuga.
* **Hito 3 (Semana 12–13) – Interpretabilidad con SHAP:** Análisis global (Beeswarm) y explicaciones locales (Waterfall plots) para sustentar decisiones ante el área comercial.
* **Hito 4 (Semana 13–14) – Calibración de Umbral y Análisis de Casos de Error:** Ajuste del umbral de corte para maximizar la utilidad económica y auditoría detallada de falsos positivos y negativos.
* **Hito 5 (Semana 14–15) – Despliegue Funcional:** Construcción de un dashboard y simulador en **Streamlit** con el pipeline serializado en `.joblib`.

---

## 8. Estructura del Repositorio

```text
TP/
├── data/
│   ├── raw/
│   │   └── telco_customer_churn.csv      # Dataset inmutable original
│   └── processed/
│       ├── train.csv                      # Partición de entrenamiento (80%)
│       └── test.csv                       # Partición de prueba (20%)
├── notebooks/
│   ├── 01_eda_exploratorio.ipynb          # EDA ejecutado con gráficos e interpretaciones
│   └── 02_preparacion_y_modelamiento.ipynb# Flujo de modelado, baselines y evaluación
├── src/
│   ├── __init__.py
│   ├── config.py                          # Rutas, semillas (SEED=42) y costos de negocio
│   ├── preprocessing.py                   # ColumnTransformer reproducible
│   ├── evaluation.py                      # Métricas técnicas y matriz de costo económico
│   └── train_models.py                    # Script de ejecución end-to-end
├── models/
│   └── pipeline_tp1_logreg.joblib         # Artefacto serializado del pipeline preliminar
├── reports/
│   ├── tabla_comparativa_modelos_tp1.csv  # Métricas cuantitativas consolidadas
│   └── guion_y_diapositivas_tp1.md        # Estructura y guion para exposición de 10 min
├── README.md                              # Documentación técnica principal
└── requirements.txt                       # Dependencias reproducibles
```

---

## 9. Instrucciones para Reproducir el Proyecto

### 9.1. Clonar el repositorio y configurar el entorno
```bash
# 1. Clonar o posicionarse en la carpeta del proyecto
cd TP

# 2. Crear entorno virtual (opcional pero recomendado)
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate

# 3. Instalar dependencias exactas
pip install -r requirements.txt
```

### 9.2. Ejecutar el pipeline completo de entrenamiento
```bash
python -m src.train_models
```

### 9.3. Explorar los cuadernos Jupyter
```bash
jupyter notebook notebooks/01_eda_exploratorio.ipynb
jupyter notebook notebooks/02_preparacion_y_modelamiento.ipynb
```
*(Los cuadernos ya se encuentran completamente ejecutados con todos los gráficos y tablas renderizados).*
