# Estructura de Diapositivas y Guion de Exposición Oral (TP1)
**Proyecto:** Sistema Predictivo de Fuga de Clientes con Segmentación de Valor (Customer Churn)  
**Curso:** Data Mining Tools (CC209) — UPC  
**Tiempo Límite:** 10 minutos de exposición + 5 minutos de preguntas  
**Integrantes:** 2 a 4 estudiantes (incluye guía de reparto de turnos)  

---

## Esquema General de Tiempo (10 minutos)
* **Minutos 0:00 – 2:00:** Diapositivas 1 y 2 $\rightarrow$ Problema, contexto de negocio y unidad de análisis.
* **Minutos 2:00 – 4:30:** Diapositivas 3, 4 y 5 $\rightarrow$ Dataset, hallazgos críticos del EDA y calidad de datos.
* **Minutos 4:30 – 7:00:** Diapositivas 6 y 7 $\rightarrow$ Flujo anti-leakage, ColumnTransformer y Baselines.
* **Minutos 7:00 – 9:00:** Diapositivas 8 y 9 $\rightarrow$ Modelos preliminares, matriz de costos económicos.
* **Minutos 9:00 – 10:00:** Diapositiva 10 $\rightarrow$ Estado actual, limitaciones y plan hacia el TF1.

---

## Diapositiva 1: Portada y Definición del Problema (1 min)
* **Título:** Sistema Predictivo de Churn y Decisión de Retención de Clientes.
* **Subtítulo:** Proyecto Integrador de Data Science – Trabajo Parcial (TP1).
* **Contenido Visual:**
  * Contexto: Fuga del 26.5% mensual en servicios de telecomunicaciones.
  * Costo: Adquirir un nuevo cliente cuesta hasta 7 veces más que retenerlo.
  * Unidad de Análisis: **Un contrato / cliente individual activo por ciclo mensual**.
  * Pregunta Central: *¿Cuál es la probabilidad de que un cliente cancele su servicio en el próximo ciclo de facturación y qué factores explican la decisión para priorizar intervenciones rentables?*
* **Guion Oral:**
  > *"Buenos días profesor y compañeros. Nuestro proyecto aborda la problemática de la deserción de clientes en servicios por suscripción. En esta industria, perder clientes no solo significa perder facturación recurrente, sino que adquirir un cliente nuevo cuesta hasta siete veces más. Nuestra unidad de análisis es un contrato mensual activo. Nuestro objetivo no es simplemente predecir por predecir, sino proveer una herramienta que permita identificar oportunamente a los clientes en riesgo para desplegar acciones comerciales con retorno de inversión positivo."*

---

## Diapositiva 2: Dataset, Calidad y Limitaciones (1 min)
* **Contenido Visual:**
  * Dataset: *IBM Telco Customer Churn* (7,043 observaciones, 21 atributos).
  * Variable objetivo: `Churn` (Binaria: 26.5% Fuga / 73.5% Retención).
  * **Hallazgo Crítico de Calidad:** 11 clientes con espacios en blanco `' '` en `TotalCharges`.
  * **Explicación fundamentada:** Corresponden a `tenure = 0` (clientes nuevos sin facturación completada). Solución: Imputación controlada de \$0 y conversión a numérico.
* **Guion Oral:**
  > *"Analizamos una base de datos con 7,043 clientes. En la etapa de diagnóstico detectamos un problema de calidad clave: 11 registros con espacios en blanco en la variable TotalCharges, lo que provocaba que se leyera como texto. Al investigar a fondo, demostramos que todos tenían antigüedad cero; eran clientes recién dados de alta que aún no habían cerrado su primer ciclo. Por ello, se justificó imputar cero y tipificar la variable a numérico, reconociendo la lógica del negocio en lugar de eliminar filas arbitrariamente."*

---

## Diapositiva 3: EDA – Hallazgos Críticos de Negocio (1.5 min)
* **Contenido Visual:** 3 Gráficos clave del Cuaderno 01:
  1. **Tipo de Contrato:** Mes a mes (42.7% fuga) vs. 2 años (2.8% fuga).
  2. **Curva de Antigüedad (*Tenure*):** Concentración masiva de bajas en los primeros 5 meses.
  3. **Interacción Internet + Soporte:** Clientes con Fibra Óptica sin Soporte Técnico fugan un 49.4%.
* **Guion Oral:**
  > *"En el EDA distinguimos con claridad entre lo que los datos muestran y lo que interpretamos como equipo. Los datos muestran que el 42.7% de quienes tienen contrato mes a mes cancelan, frente a menos del 3% en contratos bianuales. Además, la curva de deserción se concentra en los primeros cinco meses de vida del cliente. Interpretamos esto como una fricción inicial de onboarding: si un cliente supera el primer semestre, su retención aumenta drásticamente. Asimismo, detectamos que la fibra óptica sin soporte técnico dispara la deserción al 50%, lo que nos da una palanca comercial inmediata: bonificar soporte técnico en planes de alta gama."*

---

## Diapositiva 4: Metodología Anti-Leakage y Pipeline (1.5 min)
* **Contenido Visual:**
  * Diagrama de flujo: `Carga` $\rightarrow$ `Split Estratificado (80/20)` $\rightarrow$ `ColumnTransformer` $\rightarrow$ `Modelos`.
  * Numéricas: `SimpleImputer(mediana)` + `StandardScaler`.
  * Categóricas: `SimpleImputer(moda)` + `OneHotEncoder(drop='first')`.
  * Regla estricta: Transformadores ajustados (`fit`) **solo** sobre Train.
* **Guion Oral:**
  > *"Para cumplir con el rigor técnico del curso y evitar data leakage, realizamos una partición estratificada del 80% entrenamiento y 20% prueba antes de calcular cualquier estadística. La estratificación fue indispensable para preservar la prevalencia de 26.5% en ambos conjuntos. Implementamos un ColumnTransformer modular donde todas las transformaciones —imputaciones, estandarización y One-Hot Encoding— se ajustan exclusivamente en el conjunto de entrenamiento, transformando la prueba sin contaminar el flujo."*

---

## Diapositiva 5: Baselines vs. Modelos Preliminares (1.5 min)
* **Contenido Visual:**
  * Baseline 1 (Dummy): Predice siempre no fuga $\rightarrow$ Accuracy 73.5%, pero Recall 0%.
  * Baseline 2 (Heurístico): Si contrato = mes-a-mes y antigüedad $\le 6$ meses $\rightarrow$ Fuga $\rightarrow$ Recall 45.5%, PR-AUC 0.410.
  * Modelo 1: Regresión Logística (`class_weight='balanced'`) $\rightarrow$ Recall 78.3%, PR-AUC 0.633.
  * Modelo 2: Random Forest (`class_weight='balanced'`) $\rightarrow$ Recall 78.9%, PR-AUC 0.655.
* **Guion Oral:**
  > *"Para responder con rigor si el Machine Learning aporta valor frente a soluciones simples, construimos dos baselines. El baseline trivial muestra la falacia del Accuracy: tiene 73.5% de exactitud aparente, pero no detecta a ningún desertor. Nuestro baseline heurístico de negocio captura el 45% de fugas. En contraste, nuestros modelos con ponderación balanceada logran capturar casi el 79% de los clientes en fuga y elevan el PR-AUC a 0.655 en Random Forest, superando con creces las referencias simples."*

---

## Diapositiva 6: Análisis de Matriz de Confusión y Retorno Económico (1.5 min)
* **Contenido Visual:**
  * Tabla económica y matrices de confusión:
    * Costo Falso Negativo (Cliente perdido): -\$500.
    * Costo Falso Positivo (Bono innecesario): -\$50.
    * Beneficio Verdadero Positivo (Cliente retenido): +\$350.
  * Comparativa de Utilidad Neta en muestra de prueba:
    * Dummy: **-\$187,000 USD**
    * Baseline Heurístico: **-\$48,550 USD**
    * Random Forest: **+\$50,500 USD**
* **Guion Oral:**
  > *"Más allá de reportar métricas técnicas, modelamos el impacto financiero del error. Un falso negativo cuesta 500 dólares en valor de vida perdido, mientras que un falso positivo cuesta solo 50 dólares en incentivos. Gracias a este enfoque, mientras el baseline trivial causaría una pérdida catastrófica de 187 mil dólares en la muestra de prueba, el modelo de Random Forest genera un valor económico neto positivo de más de 50 mil dólares, validando la utilidad real de la solución."*

---

## Diapositiva 7: Estado del Proyecto y Plan hacia el TF1 (1 min)
* **Contenido Visual:**
  * **Limitaciones actuales reconocidas:** Umbral fijo de 0.50, hiperparámetros estándar, sin segmentación por valor de cliente.
  * **Hoja de ruta al TF1 (Semana 15):**
    1. Optimización bayesiana con **Optuna** y tracking con **MLflow**.
    2. Segmentación no supervisada con **Clustering (K-Means)** por valor de cliente.
    3. Interpretabilidad local y global con **SHAP**.
    4. Calibración de umbral óptimo y análisis exhaustivo de falsos positivos/negativos.
    5. Despliegue de aplicación interactiva en **Streamlit**.
* **Guion Oral:**
  > *"Para concluir, reconocemos que el TP1 es nuestro primer corte. Identificamos como limitaciones el uso de hiperparámetros estándar y un umbral fijo en 0.50. De cara al Trabajo Final, implementaremos experimentación sistemática con Optuna y MLflow, aplicaremos Clustering para segmentar clientes por valor de vida, usaremos SHAP para explicar a los asesores las causas individuales del riesgo, y desplegaremos la solución en una aplicación web interactiva con Streamlit. Muchas gracias, quedamos atentos a sus preguntas."*

---

## Preguntas Frecuentes del Docente y Cómo Responderlas (Q&A)

1. **¿Por qué descartaron el Accuracy si supera el 73%?**
   * *Respuesta:* *"Porque la clase minoritaria representa el 26.5% de la base. Un modelo trivial que prediga que nadie se va obtiene 73.5% de Accuracy pero un Recall de 0%, lo que significaría perder al 100% de los clientes en riesgo con un impacto financiero desastroso. Por eso priorizamos PR-AUC, Recall de la clase 1 y el valor económico neto."*

2. **¿Cómo garantizaron que no haya Data Leakage en el preprocesamiento?**
   * *Respuesta:* *"El train_test_split estratificado se realizó de forma previa a cualquier cálculo. El ColumnTransformer calcula las medianas numéricas, la escala y los vocabularios del OneHotEncoder exclusivamente en el conjunto de entrenamiento mediante el método fit(), y luego simplemente transforma el conjunto de prueba."*

3. **¿Por qué consideraron dos baselines?**
   * *Respuesta:* *"El Dummy nos permite contrastar contra el azar o la prevalencia, mientras que el baseline heurístico basado en contratos mes a mes y baja antigüedad demuestra si las heurísticas tradicionales de negocio son suficientes o si el Machine Learning aporta un beneficio incremental mensurable."*

4. **¿Por qué imputaron cero en los espacios de `TotalCharges` en lugar de la mediana?**
   * *Respuesta:* *"Porque al cruzar esos 11 registros demostramos que todos tenían antigüedad cero (tenure = 0). Imputarles la mediana o media histórica de clientes consolidados habría falseado la realidad de un cliente nuevo que aún no ha generado cargos acumulados."*
