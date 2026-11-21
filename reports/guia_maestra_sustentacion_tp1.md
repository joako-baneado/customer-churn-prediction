# GUÍA MAESTRA DE SUSTENTACIÓN: TRABAJO PARCIAL (TP1)
## Data Mining Tools (CC209) — UPC | Ciclo 2026-20
**Proyecto:** Sistema Predictivo de Fuga de Clientes con Segmentación de Valor (Customer Churn)  
**Docente:** Carlos Fernando Montoya Cubas  
**Autores:** Joaquin Basas Requena, Fabian Marcelo Rojas Cuadros, Cielo Luwidka Chavez Merino, Rodrigo Alonso Gamero Vera  
**Repositorio Oficial:** `github.com/joako-baneado/customer-churn-prediction`  
**Presentación:** `reports/TP1_Customer_Churn_Data_Mining_Tools.pptx`  

---

## 📋 Estructura General y Estrategia de Evaluación

El Trabajo Parcial evalúa **rigor metodológico, trazabilidad técnica, justificación teórica y honestidad científica**. El profesor Montoya no busca modelos "mágicos" con 99% de métricas; califica:
1. La correcta formulación del problema de negocio y la unidad de análisis.
2. La comprensión teórica del desbalance y por qué descartar métricas ingenuas.
3. El blindaje absoluto contra el **Data Leakage**.
4. La comparación contra **Baselines** representativos (trivial y de negocio).
5. El análisis crítico de la matriz de confusión y su traducción a **utilidad financiera**.
6. La capacidad autocrítica de reconocer limitaciones del primer corte y trazar una hoja de ruta sólida al TF1.

---

# DIAPOSITIVA POR DIAPOSITIVA: TEORÍA, PRÁCTICA, DISCURSO Y DEFENSA

---

## DIAPOSITIVA 1: PORTADA

```text
DATA MINING TOOLS · CC209 · TRABAJO PARCIAL (TP1)
Predicción de fuga de clientes
Identificación de clientes en riesgo de cancelar su servicio de telecomunicaciones para priorizar la retención
Joaquin Basas Requena · Fabian Marcelo Rojas Cuadros · Cielo Luwidka Chavez Merino · Rodrigo Alonso Gamero Vera
Docente: Carlos Fernando Montoya Cubas
UPC · NRC 17496 · Ciclo 2026-20
```

### 1. Fundamento Teórico
* **Contexto de Minería de Datos en Negocios:** La retención de clientes es un problema clásico de valor de vida del cliente (*Customer Lifetime Value* - CLV). En industrias de suscripción recurrente (SaaS, telecomunicaciones), el coste de adquisición de clientes ($CAC$) suele superar entre 5 y 7 veces el coste de retención ($CRC$).
* **Fórmula clave:** 
  $$\text{CLV} = \frac{\text{Margen Bruto Mensual}}{\text{Tasa de Churn Mensual}}$$
  Una reducción marginal en el churn impacta exponencialmente el valor presente neto de la base de clientes.

### 2. Justificación en el Proyecto
* El proyecto no se concibe como un ejercicio académico aislado, sino como una herramienta analítica orientada a la toma de decisiones comerciales para reducir la pérdida de clientes en el sector telecomunicaciones.

### 3. Discurso Oral Sugerido (30 segundos)
> *"Buenos días profesor y compañeros. Nuestro proyecto aborda la problemática de la deserción de clientes en servicios por suscripción de telecomunicaciones. Perder clientes en esta industria erosiona el ingreso recurrente mensual y eleva los costos de adquisición, que son hasta siete veces más caros que retener. Nuestro propósito es diseñar un flujo de Data Mining riguroso y reproducible que identifique oportunamente qué usuarios están en riesgo de cancelación para habilitar intervenciones preventivas con retorno económico positivo."*

---

## DIAPOSITIVA 2: PROBLEMA Y PREGUNTA DEL PROYECTO

```text
Problema y pregunta del proyecto
¿Qué clientes presentan mayor probabilidad de fuga y qué variables contractuales, de servicio y de facturación se asocian con ese riesgo, de manera que puedan priorizarse intervenciones preventivas?
Unidad de análisis: un cliente con contrato individual registrado en el dataset.
Tipo de problema: clasificación binaria supervisada con desbalance moderado.
Criterios de utilidad definidos por el grupo:
  1. SUPERAR LOS BASELINES: PR-AUC > 0.60 (La prevalencia de fuga es 0.265)
  2. COBERTURA DE RIESGO: Recall ≥ 0.75 sobre la clase fuga
  3. UTILIDAD ECONÓMICA: Valor neto > 0 simulada con costos supuestos
```

### 1. Fundamento Teórico
* **Unidad de Análisis:** Entidad mínima sobre la cual se registran las mediciones y se infiere la predicción. No es un hogar, ni una factura, ni un reclamo: es un **contrato individual activo por cliente**.
* **Clasificación Binaria con Desbalance Moderado:** La variable dependiente $y \in \{0, 1\}$ presenta proporciones asimétricas ($P(Y=1) = 0.265$). En este régimen, optimizar Accuracy estándar induce al estimador a predecir la clase mayoritaria.
* **Justificación de Métricas Teóricas:**
  * **PR-AUC (Precision-Recall AUC):** Evalúa el compromiso entre precisión y exhaustividad exclusivamente sobre la clase minoritaria. Una predicción aleatoria tiene un valor base de $\text{PR-AUC} = \text{Prevalencia} = 0.265$. Superar $0.60$ garantiza una capacidad discriminante más de dos veces superior al azar.
  * **Recall (Exhaustividad de la clase positiva):** 
    $$\text{Recall} = \frac{TP}{TP + FN}$$
    Mide la fracción de fugas reales detectadas. Un falso negativo ($FN$) equivale a un cliente perdido sin intervención.
  * **Función de Utilidad Neta Esperada:**
    $$U = (TP \times V_{\text{retención}}) - (FN \times C_{\text{pérdida}}) - (FP \times C_{\text{campaña}})$$

### 2. Discurso Oral Sugerido (1 minuto)
> *"Delimitamos nuestra unidad de análisis de forma estricta: un cliente con contrato individual activo. El problema de Data Science es una clasificación binaria supervisada con desbalance moderado, ya que solo el 26.5% de los clientes cancelan el servicio. En vez de evaluar con Accuracy genérica, definimos tres criterios de utilidad técnica y de negocio a priori: primero, superar los baselines alcanzando un PR-AUC superior a 0.60; segundo, priorizar la cobertura del riesgo con un Recall mayor o igual al 75% sobre la clase fuga; y tercero, que la matriz de costos y beneficios arroje un valor económico neto positivo."*

### 3. Pregunta del Docente y Respuesta
* **Pregunta:** *¿Por qué dicen que el desbalance es 'moderado' y no severo?*
* **Respuesta:** *"Hablamos de desbalance moderado cuando la clase minoritaria se encuentra entre el 10% y el 30% (aquí es 26.5%, una relación aproximada de 1 a 3). Un desbalance severo o extremo ocurre cuando la clase positiva es inferior al 1%, como en detección de fraude o diagnóstico médico raro. En desbalance moderado, métodos como `class_weight='balanced'` son altamente efectivos sin requerir generación sintética agresiva como SMOTE."*

---

## DIAPOSITIVA 3: DATASET: TELCO CUSTOMER CHURN

```text
7,043 clientes, uno por fila
21 atributos: perfil, servicios, contrato y facturación
26.5 % de fuga (1,869 clientes): la accuracy no sirve como guía
Procedencia: Datos de muestra de IBM Cognos Analytics, publicados en Kaggle por blastchar.
Variable objetivo: Churn (1 = fuga, 0 = retención).
Limitación principal: Corte transversal: no contiene historial transaccional ni temporal que permita modelar la evolución del cliente.
```

### 1. Fundamento Teórico
* **Naturaleza Transversal (*Cross-Sectional Data*):** Las observaciones corresponden a una instantánea congelada en el tiempo ($t_0$). No se cuenta con series de tiempo ($t_{-1}, t_{-2}, \dots$) que permitan calcular deltas o tendencias en consumo de datos o minutos.
* **Falacia de la Exactitud (*Accuracy Paradox*):**
  $$\text{Accuracy}_{\text{Dummy}} = \frac{TN}{TN + FN} = 73.5\%$$
  Si un modelo predice que nadie se va, reporta un $73.5\%$ de exactitud mientras tiene un **Recall del 0%**, provocando la quiebra comercial del área de retención.

### 2. Discurso Oral Sugerido (45 segundos)
> *"El estudio se sustenta en el dataset de IBM Cognos Analytics, compuesto por 7,043 registros y 21 atributos sociodemográficos, de servicios contratados y de facturación. La variable objetivo es Churn, donde 1,869 clientes (26.5%) abandonaron el servicio. Esto confirma por qué descartamos la Accuracy como brújula: un clasificador ingenuo que diga que nadie se va tendría 73.5% de exactitud y 0% de utilidad. Como científicos de datos, reconocemos la limitación central del dataset: es un corte transversal estático; carece de eventos cronológicos históricos que permitan modelar la trayectoria temporal del cliente."*

---

## DIAPOSITIVA 4: CONTRATO Y ANTIGÜEDAD CONCENTRAN EL RIESGO

```text
Tasa de fuga por tipo de contrato:
  - Mensual: 42.7 %
  - Un año: 11.3 %
  - Dos años: 2.8 %
Interpretamos: el compromiso contractual actúa como barrera de salida; es la variable que más separa a los grupos.

Tasa de fuga por meses de antigüedad:
  - 0 a 5 meses: 54.3 %
  - 6 a 12 meses: 36.0 %
  - 13 a 24 meses: 28.7 %
  - 25 a 48 meses: 20.4 %
  - 49 a 72 meses: 9.5 %
Interpretamos: posible fricción en los primeros meses. De aquí sale el baseline heurístico: contrato mensual y antigüedad ≤ 6.
(Calculadas sobre los 7,043 clientes del dataset)
```

### 1. Fundamento Teórico
* **Teoría de Costes de Cambio y Barreras de Salida:** Los contratos a plazo forzoso imponen penalizaciones monetarias y fricción legal para rescindir, suprimiendo la fuga al $2.8\%$.
* **Curva de Mortalidad de Clientes (Análisis de Supervivencia):** En servicios por suscripción, la tasa de riesgo instantáneo (*hazard rate*) es máxima durante la fase de incorporación (*onboarding*). Si el cliente supera los primeros seis meses, se consolida el hábito de uso y la tasa cae monótonamente hasta el $9.5\%$.
* **Génesis del Baseline Heurístico:**
  $$\text{Regla:} \quad \hat{y} = 1 \iff (\text{Contract} = \text{'Month-to-month'}) \land (\text{tenure} \le 6)$$
  En Data Mining es mandatorio formular baselines con reglas de dominio previas antes de afirmar que un algoritmo complejo aporta valor.

### 2. Discurso Oral Sugerido (1 minuto)
> *"En el análisis exploratorio identificamos que el tipo de contrato y la antigüedad son los principales factores discriminantes. Los clientes con contratos mes a mes presentan una fuga del 42.7%, en contraste con apenas 2.8% en contratos bianuales; la cláusula contractual actúa como una barrera de salida natural. Asimismo, al analizar la variable tenure, más del 54% de los clientes en sus primeros 5 meses cancelan, reduciéndose a menos del 10% pasados los 4 años. Esta evidencia demuestra una fuerte fricción en el onboarding. De este cruce empírico construimos nuestro baseline heurístico: clientes mes a mes con 6 o menos meses de antigüedad."*

---

## DIAPOSITIVA 5: SERVICIO Y FORMA DE PAGO TAMBIÉN SEPARAN GRUPOS

```text
Fibra óptica: fuga según soporte técnico
  - Sin soporte: 49.4 %
  - Con soporte: 22.6 %
Interpretamos: un servicio caro sin asistencia ante incidencias puede generar insatisfacción. Quienes se van pagan una mediana de $79.65 al mes frente a $64.43.

Tasa de fuga por método de pago:
  - Cheque electrónico: 45.3 %
  - Cheque por correo: 19.1 %
  - Transferencia bancaria: 16.7 %
  - Tarjeta de crédito: 15.2 %
Interpretamos: el pago manual renueva cada mes la decisión de seguir; es una hipótesis, no una causa demostrada.
(Asociaciones observadas, no relaciones causales)
```

### 1. Fundamento Teórico
* **Interacción de Características (*Feature Interaction*):** El efecto de una variable sobre el target depende del valor de otra. La fibra óptica es de alta velocidad pero tiene un coste mayor ($\text{mediana} = \$79.65$). La ausencia de soporte técnico actúa como catalizador de abandono ($49.4\%$).
* **Economía del Comportamiento en Medios de Pago:** Los pagos manuales (cheque electrónico) imponen un costo de fricción cognitivo mensual: el usuario debe abrir un portal, ingresar datos y validar el cobro, reevaluando el valor percibido del servicio. En débito automático (tarjeta/transferencia $\approx 15\%$), el cobro pasa inadvertido (*pain of paying* reducido).
* **Causalidad vs. Correlación:** Es crucial en la rúbrica del profesor diferenciar asociación estadística de inferencia causal. No sabemos si el cheque electrónico causa la fuga o si los clientes insatisfechos cambian deliberadamente a cheque para evitar débitos no autorizados.

### 2. Discurso Oral Sugerido (1 minuto)
> *"Al cruzar servicios y transacciones detectamos dos patrones críticos. Primero, la fibra óptica es el servicio más caro de la cartera, y cuando el cliente no cuenta con soporte técnico, la deserción se dispara al 49.4%; no obstante, si cuenta con soporte técnico, la fuga cae a la mitad (22.6%). La insatisfacción técnica en planes costosos destruye valor. Segundo, en el método de pago, el cheque electrónico presenta una tasa de fuga del 45.3%, triplicando a los pagos automatizados por tarjeta de crédito o transferencia (15%). El pago manual obliga al cliente a reconsiderar mensualmente su gasto. Como equipo enfatizamos que son correlaciones observadas y no causalidad clínica."*

---

## DIAPOSITIVA 6: CALIDAD DE DATOS Y DECISIONES DE PREPARACIÓN

```text
VALORES FALTANTES: 11 TotalCharges en blanco.
Todos con antigüedad 0 y sin fuga: clientes sin primer cierre de facturación. Por regla lógica se representan con 0.
TIPOS DE DATOS: Dos columnas mal tipadas.
TotalCharges llegaba como texto y pasa a numérico. SeniorCitizen llegaba como 0/1 y se trata como categórica.
DUPLICADOS E IDENTIFICADOR: 0 filas duplicadas.
Tampoco hay identificadores repetidos. customerID se excluye del modelo porque no es una característica predictiva.
VALORES EXTREMOS: Ninguno fuera de los límites IQR.
Se revisaron tenure, MonthlyCharges y TotalCharges. No se elimina ni se recorta ningún registro.
Fuente: cuaderno 01_eda_exploratorio
```

### 1. Fundamento Teórico
* **Imputación por Conocimiento del Dominio (*Domain-Informed Imputation*):** Si se aplica ciegamente `SimpleImputer(strategy='median')` a los espacios en blanco de `TotalCharges`, se les asignaría una mediana de $\approx \$1,400$. Sin embargo, cruzar la variable con `tenure == 0` demuestra que son clientes que nunca han sido facturados:
  $$\text{TotalCharges}_{\text{teórico}} = \text{tenure} \times \text{MonthlyCharges} = 0 \times \$X = \$0$$
  Imputar \$0 preserva la consistencia lógica y evita distorsionar la distribución de clientes nuevos.
* **Tipificación Correcta:** `SeniorCitizen` está codificado numéricamente como $0$ y $1$, pero carece de propiedades métricas escalares (no existe "el doble de ciudadano senior"). Tratarlo como variable categórica evita que un modelo lineal asuma una pendiente forzada.
* **Tratamiento de Identificadores:** Variables como `customerID` poseen cardinalidad igual a $N$ ($7,043$ valores únicos). Incluirlas induciría memorización o sobreajuste espurio.
* **Análisis de Outliers (Rango Intercuartílico - IQR):**
  $$\text{IQR} = Q_3 - Q_1, \quad [\text{Límite Inferior}: Q_1 - 1.5 \times \text{IQR}, \quad \text{Límite Superior}: Q_3 + 1.5 \times \text{IQR}]$$
  Al verificar `tenure` ($[0, 72]$), `MonthlyCharges` ($[\$18.25, \$118.75]$) y `TotalCharges`, todos los puntos caen en límites físicos y comerciales válidos. No se borra información real.

### 2. Discurso Oral Sugerido (1 minuto)
> *"En la auditoría de calidad identificamos 11 registros donde TotalCharges figuraba como espacio en blanco, forzando a Pandas a interpretar la columna como tipo texto. En lugar de aplicar una imputación automática por mediana o borrar los registros, investigamos su naturaleza: todos tenían antigüedad cero y Churn cero. Eran altas nuevas que aún no habían completado su primer ciclo de facturación. Por lógica contable, imputamos cero dólares y reconvertimos la columna a numérico. Asimismo, reclasificamos SeniorCitizen como categórica, eliminamos customerID por ser un identificador no predictivo y verificamos mediante análisis IQR que no existen valores atípicos que requieran poda artificial."*

---

## DIAPOSITIVA 7: SEPARACIÓN DE DATOS Y FLUJO REPRODUCIBLE

```text
Train · 70 % (4,930 clientes): ajusta pipeline y modelos
Validation · 15 % (1,056 clientes): compara y selecciona
Reentrenamiento: Train + Validation; solo el modelo elegido
Test · 15 % (1,057 clientes): una sola evaluación final

3 variables numéricas: Imputación por mediana y StandardScaler sobre tenure, MonthlyCharges y TotalCharges.
16 variables categóricas: Imputación por moda y OneHotEncoder(drop='first'), dentro de un ColumnTransformer.

Contra el data leakage: partición estratificada con semilla 42; medianas, escalas y categorías se aprenden solo con los datos de entrenamiento. Validation decide; Test no participa en la selección.
Reproducible con: python -m src.train_models
```

### 1. Fundamento Teórico
* **Taxonomía de Particiones (Train / Validation / Test):**
  * **Train ($70\%$):** Optimiza los parámetros internos de los modelos ($\mathbf{w}, b$, umbrales de splits).
  * **Validation ($15\%$):** Conjunto independiente para contrastar hipótesis de modelado y seleccionar el mejor estimador.
  * **Reentrenamiento ($85\%$):** Una vez fijado el algoritmo, se entrena con $\text{Train} \cup \text{Validation}$ para maximizar la muestra antes del testeo ciego.
  * **Test ($15\%$):** Reserva inviolable para simular el comportamiento en producción.
* **Estratificación (*Stratified Split*):** Mantiene la proporción $P(Y=1) \approx 0.265$ en los tres bloques, evitando sesgo muestral.
* **Data Leakage y `ColumnTransformer`:**
  * Si se ajusta `StandardScaler` o `SimpleImputer` antes del split, se calculan $\mu_{\text{global}}$ y $\sigma_{\text{global}}$, contaminando Train con la distribución de Test.
  * En nuestro diseño modular:
    $$\text{Pipeline.fit}(X_{\text{train}}, y_{\text{train}}) \implies \mu_{\text{train}}, \sigma_{\text{train}} \text{ aprendidos exclusivamente de Train.}$$
    $$X_{\text{val\_proc}} = \text{Pipeline.transform}(X_{\text{val}})$$
    $$X_{\text{test\_proc}} = \text{Pipeline.transform}(X_{\text{test}})$$
* **Codificación Categórica:** `OneHotEncoder(drop='first')` previene la colinealidad perfecta (trampa de la variable ficticia) en modelos lineales.

### 2. Discurso Oral Sugerido (1 minuto)
> *"Para blindar el experimento contra Data Leakage, implementamos una partición estratificada con semilla 42 en tres conjuntos: 70% Train, 15% Validation y 15% Test. La estratificación fue indispensable para preservar la prevalencia de 26.5% en cada bloque. Diseñamos un ColumnTransformer modular donde las medianas de imputación, la estandarización y los vocabularios del One-Hot Encoding se ajustan con el método fit exclusivamente en Train. Validation se utilizó para seleccionar el modelo candidato, tras lo cual reentrenamos en Train más Validation para evaluar una sola vez en Test. Todo el flujo es 100% reproducible ejecutando `python -m src.train_models`."*

---

## DIAPOSITIVA 8: SELECCIÓN DEL MODELO EN VALIDATION

```text
Selección del modelo en Validation (1,056 clientes · umbral 0.50):
Modelo                                | PR-AUC | ROC-AUC | Recall | Precisión | Valor simulado
Regresión logística balanceada        | 0.6313 | 0.8448  | 81.4 % | 50.7 %    | +$42,700
Random Forest bal., profundidad 10    | 0.6327 | 0.8417  | 75.7 % | 54.6 %    | +$31,400

Criterio de selección: mayor PR-AUC en Validation, porque la clase de interés es minoritaria.
Candidato preliminar: Random Forest, por una ventaja de 0.0014 que no es concluyente.
Lectura crítica: la regresión logística logra mayor recall y mayor valor simulado.
```

### 1. Fundamento Teórico
* **Comparativa de Modelos:**
  * **Regresión Logística Regularizada:** Modelo lineal generalizado que estima log-odds:
    $$\log\left(\frac{p}{1-p}\right) = \mathbf{w}^T \mathbf{x} + b$$
    Con `class_weight='balanced'`, penaliza los errores de la clase positiva en función del inverso de su frecuencia.
  * **Random Forest (Ensemble por Bagging):** Construye $100$ árboles de decisión no correlacionados mediante muestreo bootstrap y selección aleatoria de características (*feature subspacing*). Limitar `max_depth=10` restringe la varianza y mitiga el sobreajuste.
* **Lectura Crítica y Honestidad Científica:**
  * La diferencia en PR-AUC es de apenas $\Delta = 0.0014$ ($0.6327 - 0.6313$).
  * La Regresión Logística superó a Random Forest en **Recall** ($81.4\%$ vs $75.7\%$) y en **Valor Económico Simulado** ($+\$42,700$ vs $+\$31,400$) bajo el umbral $0.50$.
  * Esto ocurre porque la Regresión Logística generó más positivos predichos, capturando $228$ verdaderos positivos frente a $212$ de Random Forest.

### 2. Discurso Oral Sugerido (1 minuto)
> *"En la etapa de validación contrastamos la Regresión Logística y Random Forest, ambos configurados con ponderación balanceada. Siguiendo la regla fijada a priori, seleccionamos Random Forest como candidato preliminar por alcanzar el mayor PR-AUC con 0.6327, superando nuestra meta de 0.60. No obstante, mantenemos una lectura crítica fundamental: la diferencia es de apenas 0.0014, estadísticamente indistinguible. De hecho, la Regresión Logística logró un mayor Recall de 81.4% y generó 42,700 dólares de valor simulado frente a 31,400 de Random Forest. Esto evidencia que un solo split y un umbral fijo en 0.50 no permiten dictaminar un ganador definitivo, justificando la validación cruzada y el ajuste de umbral para el TF1."*

---

## DIAPOSITIVA 9: EVALUACIÓN FINAL EN TEST

```text
Evaluación final en Test (1,057 clientes · 281 fugas reales):
Alternativa                         | Accuracy | Recall | Precisión | PR-AUC
Baseline trivial (clase mayoritaria)| 73.4 %   |  0.0 % |   0.0 %   | 0.266
Baseline heurístico (mensual y ≤6)  | 76.3 %   | 43.8 % |  57.2 %   | 0.400
Random Forest seleccionado          | 78.3 %   | 70.5 % |  57.6 %   | 0.670

Supera a ambos baselines: detecta 198 de las 281 fugas; la regla heurística detecta 123.
Meta de PR-AUC cumplida: 0.670 frente a 0.60, con ROC-AUC de 0.840.
Meta de recall no cumplida: 0.705 frente a 0.75; quedan 83 fugas sin detectar.
Matriz de confusión: TN 630, FP 146, FN 83, TP 198.
```

### 1. Fundamento Teórico
* **Desglose de la Matriz de Confusión en Test ($N=1,057$):**
  * Total de clientes en fuga reales: $FN + TP = 83 + 198 = 281$ ($26.58\%$).
  * Total de clientes retenidos reales: $TN + FP = 630 + 146 = 776$ ($73.42\%$).
* **Cálculo de Métricas:**
  $$\text{Accuracy} = \frac{630 + 198}{1,057} = \frac{828}{1,057} = 78.33\%$$
  $$\text{Recall} = \frac{198}{198 + 83} = \frac{198}{281} = 70.46\% \approx 70.5\%$$
  $$\text{Precision} = \frac{198}{198 + 146} = \frac{198}{344} = 57.56\% \approx 57.6\%$$
* **Cumplimiento vs. Incumplimiento de Metas:**
  * ✅ **PR-AUC:** Alcanzó $0.670$ (Meta $\ge 0.60$). Supera con creces el baseline trivial ($0.266$) y el heurístico ($0.400$).
  * ❌ **Recall:** Alcanzó $70.5\%$ (Meta $\ge 75.0\%$). No se cumplió debido a que el umbral de decisión se mantuvo rígidamente en $\tau = 0.50$.

### 2. Discurso Oral Sugerido (1.5 minutos)
> *"Evaluamos una sola vez en el conjunto de prueba independiente de 1,057 clientes. Los resultados ratifican el valor del Machine Learning: superamos contundentemente a ambos baselines. Mientras que el baseline trivial no detecta ninguna fuga y la heurística de negocio captura 123, Random Forest detecta 198 fugas, alcanzando un PR-AUC de 0.670, superando ampliamente la meta de 0.60. Sin embargo, asumimos con total honestidad que la meta de Recall del 75% no se cumplió en Test: obtuvimos 70.5%, dejando 83 clientes en fuga como falsos negativos. Esta brecha se debe al uso del umbral estándar de 0.50, que en el TF1 calibraremos según la función de costo para recuperar esas 83 fugas."*

---

## DIAPOSITIVA 10: DE LA PÉRDIDA A LA UTILIDAD SIMULADA

```text
Valor económico neto simulado sobre los 1,057 clientes de Test, en dólares:
Baseline trivial:   −$140,500
Baseline heurístico: −$40,550
Random Forest sel.:  +$20,500

Supuestos del grupo (no datos de la empresa):
  - Fuga no detectada (FN): −$500
  - Incentivo a quien no se iba (FP): −$50
  - Cliente en riesgo retenido (TP): +$350
  - Cliente no contactado (TN): $0
(Simulación; no representa ganancias reales observadas)
```

### 1. Fundamento Teórico
* **Matriz de Ganancias y Costos (*Cost-Benefit Matrix*):**
  | Real \ Pred | Predice 0 (No interviene) | Predice 1 (Interviene con oferta) |
  | :--- | :---: | :---: |
  | **Real 0 (Retenido)** | $TN \times \$0$ (Sin costo) | $FP \times (-\$50)$ (Costo de descuento/bono) |
  | **Real 1 (Fuga)** | $FN \times (-\$500)$ (Pérdida de valor de vida) | $TP \times (+\$350)$ (Margen recuperado neto) |

* **Cálculo Matemático Exacto:**
  * **Baseline Trivial:**
    $$U = 0(0) + 0(-50) + 281(-500) + 0(350) = -\$140,500$$
  * **Baseline Heurístico ($TN=684, FP=92, FN=158, TP=123$):**
    $$U = (123 \times 350) + (158 \times -500) + (92 \times -50) = 43,050 - 79,000 - 4,600 = -\$40,550$$
  * **Random Forest ($TN=630, FP=146, FN=83, TP=198$):**
    $$U = (198 \times 350) + (83 \times -500) + (146 \times -50) = 69,300 - 41,500 - 7,300 = +\$20,500$$
* **Diferencial de Negocio:** Frente a la inacción del baseline trivial, el modelo rescata **\$161,000 USD** de valor en la muestra de prueba ($+\$20,500 - (-\$140,500)$).

### 2. Discurso Oral Sugerido (1 minuto)
> *"Para traducir el rendimiento estadístico al lenguaje financiero, modelamos una matriz de costos y beneficios con supuestos razonados de negocio. En este escenario, perder a un cliente cuesta 500 dólares en valor de vida perdido, mientras que contactar a un cliente que no se iba a fugar cuesta solo 50 dólares en incentivos comerciales. Bajo este esquema, la inacción del baseline trivial genera una pérdida catastrófica de 140,500 dólares, y la regla heurística pierde 40,550. En contraste, nuestro Random Forest alcanza una utilidad neta positiva de más de 20,500 dólares, demostrando que la predicción genera viabilidad económica tangible."*

---

## DIAPOSITIVA 11: LIMITACIONES DEL TP1 Y HOJA DE RUTA AL TF1

```text
Lo que todavía no está resuelto:
  - El recall no alcanza la meta en Test (70.5% vs 75%).
  - La ventaja entre modelos es mínima y sin validación cruzada no sabemos si es estable.
  - Umbral de decisión fijo en 0.50.
  - Hiperparámetros sin búsqueda sistemática.
  - Costos económicos supuestos por el grupo.

Hoja de ruta:
  - Validación cruzada estratificada, con promedio y dispersión.
  - Búsqueda sistemática de hiperparámetros (Optuna / Random Search).
  - Ajuste del umbral y análisis de falsos positivos y negativos.
  - Interpretabilidad global y local (SHAP).
  - Técnica adicional solo si es pertinente para los datos.
  - Despliegue funcional en Streamlit.
Repositorio: github.com/joako-baneado/customer-churn-prediction
```

### 1. Fundamento Teórico
* **Autocrítica del Primer Corte (TP1):** Un error habitual en el TP1 es pretender que el proyecto está finalizado. El profesor evalúa la lucidez del grupo para identificar deuda técnica y metodológica.
* **Soluciones Metodológicas para el TF1:**
  1. **Validación Cruzada Estratificada ($k$-Fold):** Para comprobar si el $\Delta = 0.0014$ entre Random Forest y Regresión Logística es significativo o mero ruido muestral.
  2. **Curva de Costo vs. Umbral ($\tau^*$):** Dado que $\frac{C_{FP}}{C_{FN}} = \frac{50}{500} = 0.10$, el umbral Bayesiano óptimo no es $0.50$, sino aproximadamente:
     $$\tau^* \approx \frac{C_{FP}}{C_{FP} + C_{FN}} = \frac{50}{550} \approx 0.091 - 0.30$$
     Mover el umbral hacia la izquierda elevará el Recall de $70.5\%$ por encima del $80\%$, satisfaciendo la meta.
  3. **Interpretabilidad con SHAP (Shapley Additive Explanations):** Descompone la predicción individual de cada cliente para que el gestor comercial sepa *por qué* está en riesgo (e.g. contrato mensual + fibra óptica sin soporte).

### 2. Discurso Oral Sugerido (1 minuto)
> *"Para concluir, el TP1 representa nuestro primer hito. Reconocemos con total honestidad cinco limitaciones: no alcanzamos la meta de Recall por 4.5 puntos porcentuales, la ventaja entre modelos fue mínima, mantuvimos el umbral rígido en 0.50, no optimizamos hiperparámetros sistemáticamente y los costos son supuestos. Para el Trabajo Final en la semana 15, implementaremos validación cruzada estratificada con intervalos de confianza, calibración de umbral óptimo mediante curvas de costo, interpretabilidad local y global con SHAP y el despliegue de una aplicación web interactiva en Streamlit. Muchas gracias, quedamos a su disposición para las preguntas."*

---

# 🛡️ BANCO DE PREGUNTAS TRAMPA DEL DOCENTE (Q&A DE ALTO NIVEL)

### P1: *"¿Por qué eligieron Random Forest si en la Diapositiva 8 la Regresión Logística tenía mejor Recall (81.4% vs 75.7%) y mayor valor económico (+$42,700 vs +$31,400)?"*
* **Respuesta:**
  > *"Excelente pregunta, profesor. Fijamos a priori que el criterio de selección técnico sería el PR-AUC, porque evalúa la calidad del ordenamiento probabilístico a lo largo de todos los umbrales posibles sobre la clase minoritaria, donde Random Forest obtuvo 0.6327 frente a 0.6313. Sin embargo, como señalamos explícitamente en nuestra lectura crítica de la Diapositiva 8, esa diferencia de 0.0014 no es concluyente y la Regresión Logística superó a Random Forest bajo el umbral arbitrario de 0.50. Precisamente esa es una de las grandes lecciones del TP1: en el TF1 no decidiremos con un umbral fijo ni con un solo split, sino con validación cruzada estratificada y calibración de umbral de costo para determinar cuál modelo sostiene de forma robusta la mejor utilidad neta."*

### P2: *"¿Por qué imputaron $0 en TotalCharges en lugar de la mediana o de descartar las 11 filas?"*
* **Respuesta:**
  > *"Porque al cruzar esos 11 registros en el EDA demostramos que todos tenían antigüedad cero (`tenure == 0`). Eran clientes recién dados de alta que no habían completado su primer mes de facturación. Si hubiéramos imputado la mediana (que ronda los $1,400), habríamos introducido una inconsistencia lógica grave, haciendo creer al algoritmo que clientes novatos tenían acumulados miles de dólares. Descartar filas tampoco era necesario porque no representaba datos corruptos, sino una regla de negocio clara: sin antigüedad, los cargos acumulados son exactamente $0."*

### P3: *"¿Cómo garantizan que no hubo Data Leakage en su código?"*
* **Respuesta:**
  > *"Garantizamos cero data leakage mediante tres controles: primero, la partición estratificada en 70/15/15 se realizó al inicio con semilla fija 42; segundo, todas las transformaciones están encapsuladas en un `ColumnTransformer` de Scikit-Learn donde el método `fit()` se ejecuta única y exclusivamente sobre el conjunto de entrenamiento (Train); los conjuntos de validación y prueba solo reciben `transform()`; y tercero, el conjunto de prueba (Test) se reservó intacto hasta después de la selección del modelo preliminar, asegurando una evaluación totalmente ciega."*

### P4: *"¿Por qué no alcanzaron la meta de Recall en Test (sacaron 70.5% frente a 75%)?"*
* **Respuesta:**
  > *"Porque evaluamos el modelo con el umbral predeterminado de 0.50. Dado que en nuestro negocio no detectar una fuga cuesta $500 y un falso positivo cuesta solo $50, el costo del error es asimétrico en proporción 10 a 1. El umbral óptimo teórico se encuentra por debajo de 0.50 (entre 0.30 y 0.35). Al desplazar el umbral hacia la izquierda en el TF1, aumentaremos la sensibilidad del clasificador, capturando las 83 fugas no detectadas y superando con holgura el 75% de Recall con un costo marginal asumible."*

### P5: *"¿Por qué no aplicaron SMOTE para balancear las clases?"*
* **Respuesta:**
  > *"Porque el desbalance es moderado (26.5%, proporción aproximada de 1:3). En desbalances moderados, utilizar `class_weight='balanced'` ajusta la función de pérdida penalizando con mayor peso a la clase minoritaria sin alterar la distribución marginal ni fabricar datos artificiales. SMOTE en datasets con alta presencia de variables categóricas One-Hot tiende a crear puntos sintéticos fuera del espacio de soporte real del negocio, introduciendo ruido en la frontera de decisión."*

---

# 📊 RESUMEN EJECUTIVO DE MÉTRICAS (CHULETA PARA EXPOSITORES)

| Métrica / Concepto | Validación (Reg. Logística) | Validación (Random Forest) | Test (Baseline Trivial) | Test (Baseline Heurístico) | Test (Random Forest) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | 74.1 % | 76.9 % | 73.4 % | 76.3 % | **78.3 %** |
| **Recall (Clase 1)** | **81.4 %** | 75.7 % | 0.0 % | 43.8 % | **70.5 %** |
| **Precisión** | 50.7 % | 54.6 % | 0.0 % | 57.2 % | **57.6 %** |
| **ROC-AUC** | **0.845** | 0.842 | 0.500 | 0.660 | **0.840** |
| **PR-AUC** | 0.631 | **0.633** | 0.266 | 0.400 | **0.670** |
| **Valor Neto ($)** | **+$42,700** | +$31,400 | -$140,500 | -$40,550 | **+$20,500** |
| **Matriz Test** | — | — | FN: 281 | FN: 158, TP: 123 | **TN: 630, FP: 146, FN: 83, TP: 198** |
