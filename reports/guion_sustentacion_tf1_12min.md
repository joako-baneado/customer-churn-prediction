# Guion Cronometrado de Sustentación TF1 (12 Minutos + Demo en Vivo + Q&A)

**Curso:** Data Mining Tools (CC209) — UPC  
**Docente:** Carlos Fernando Montoya Cubas  
**Integrantes:** Joaquin Basas Requena, Fabian Marcelo Rojas Cuadros, Cielo Luwidka Chavez Merino, Rodrigo Alonso Gamero Vera  
**Regla de exposición (Sección 5.4):** 12 minutos totales (incluyendo 2.5 min de demostración funcional en Streamlit) + hasta 5 minutos de preguntas. Todos los integrantes intervienen y dominan el proyecto completo.

---

## 1. Distribución Cronometrada de Turnos (12:00 exactos)

| Tiempo | Duración | Diapositivas | Expositor sugerido | Eje temático |
| :---: | :---: | :---: | :--- | :--- |
| **0:00 – 2:30** | 2:30 | Slides 1 – 4 | **Joaquin Basas** | Contexto, brechas heredadas del TP1, evolución TP1 → TF1, arquitectura anti-leakage y matriz de herramientas. |
| **2:30 – 5:30** | 3:00 | Slides 5 – 7 | **Fabian Rojas** | Ablation Study, Optuna, CV 10-Fold, Wilcoxon, calibración OOF, paradoja `0.05` vs `0.20`, Test único y sensibilidad económica. |
| **5:30 – 8:30** | 3:00 | Slides 8 – 10 | **Cielo Chavez** | Clustering `K-Means (k=3)` + PCA 2D (`96.17%`), Matriz Riesgo × Valor 2×2, SHAP global/local y Error Analysis (`53` FN / `234` FP). |
| **8:30 – 11:00** | 2:30 | Slide 11 + App | **Rodrigo Gamero** | **Demo funcional en vivo** en Streamlit (Diagnóstico Individual + SHAP, Batch CSV y Simulador Financiero). |
| **11:00 – 12:00** | 1:00 | Slide 12 | **Rodrigo / Joaquin** | Conclusiones rigurosas (qué sí y qué no puede concluirse), limitaciones, riesgos y trabajo futuro. |

---

## 2. Guion Detallado por Bloque

### Bloque 1 — Joaquin Basas (0:00 – 2:30 | Slides 1–4)
- **Slide 1–2 (0:45):** *"Buenos días profesor y compañeros. Cerramos nuestro proyecto integrador de predicción de fuga en telecomunicaciones. En el TP1 logramos superar los baselines con Random Forest (`PR-AUC = 0.6698`, `+$20,500`), pero reconocimos con transparencia una brecha crítica: el Recall en Test quedó en `70.46%` frente a nuestra meta de `75%`, porque dependíamos de un umbral rígido en `0.50`, un único split sin validación cruzada y carecíamos de explicabilidad individual."*
- **Slide 3–4 (1:45):** *"En el TF1 evolucionamos en cinco frentes manteniendo bloqueado el mismo Test Holdout de `1,057` clientes desde el día uno: primero, creamos `TelcoFeatureEngineer` con 8 variables de negocio sin leakage y validación de esquemas con Pydantic; segundo, evaluamos 5 familias en `StratifiedKFold (k=10)` con Optuna y MLflow; tercero, calibramos y optimizamos umbrales en OOF; cuarto, añadimos segmentación no supervisada K-Means + PCA 2D y explicabilidad SHAP; y quinto, desplegamos la app funcional en Streamlit validada con `pytest`. Cada herramienta fue comparada frente a alternativas técnicas en nuestra matriz de decisiones."*

### Bloque 2 — Fabian Rojas (2:30 – 5:30 | Slides 5–7)
- **Slide 5 (1:00):** *"Sobre `Train + Val` (`5,986` filas), el ablation study confirmó que `TelcoFeatureEngineer` eleva el PR-AUC de Regresión Logística y reduce casi a la mitad la desviación estándar de LightGBM (`0.0164 -> 0.0088`). Tras 25 trials de Optuna por familia, el **Stacking Heterogéneo (`LR + LightGBM + XGBoost`)** obtuvo el mayor `PR-AUC = 0.6732 ± 0.0289` y `ROC-AUC = 0.8515` en CV 10-fold, superando de forma estadísticamente significativa por test de Wilcoxon a Random Forest (`p = 0.0020`) y Regresión Logística (`p = 0.0273`), y quedando estadísticamente empatado con XGBoost y LightGBM."*
- **Slide 6 (1:00):** *"En calibración OOF, el Stacking sin post-calibración logró el menor `Brier Score = 0.1333`. Al optimizar el umbral bajo nuestra matriz de costos (`FN=-$500`, `FP=-$50`, `TP=+$350`), descubrimos un hallazgo clave: el óptimo Bayesiano irrestricto cae en `0.05`, pero eso obligaría a contactar al `75.7%` de toda la cartera, contradiciendo el objetivo de evitar campañas indiscriminadas. Por ello impusimos en OOF la restricción operativa `Recall >= 75%` y `Precision >= 50%`, fijando **`tau* = 0.20`** y umbrales dinámicos diferenciados por segmento (`0.30 / 0.22 / 0.21`)."*
- **Slide 7 (1:00):** *"Al evaluar una sola vez sobre Test (`1,057` clientes), el TF1 sube a `PR-AUC = 0.6719`, dispara el `Recall` de `70.5%` a **`81.1%`**, rescata el **`84.1%` de los ingresos mensuales en riesgo (`Value-Weighted Recall`)** y duplica la utilidad neta simulada de `+$20,500` a **`+$41,600`**. En los tres escenarios de sensibilidad —incluyendo efectividad real del `30%` y doble costo FP— nuestra solución domina al TP1 y a ambos baselines."*

### Bloque 3 — Cielo Chavez (5:30 – 8:30 | Slides 8–10)
- **Slide 8 (1:00):** *"Como técnica de la segunda mitad del curso comparamos K-Means, GMM y Ward entre `k=2..6`, eligiendo **K-Means (`k=3`)** por mayor `Silhouette = 0.4394` en el rango accionable y soporte directo `.predict()` online, visualizado en PCA 2D con `96.17%` de varianza explicada. Aclaramos que son proxies de valor, no CLV transaccional real: identificamos un segmento **C1 Valor Medio (`31.5%` cartera)** con `50.1%` de fuga temprana, y **C2 Alto Valor (`42.7%` cartera)** con `$86.5/mes`, donde cruzar Riesgo × Valor define acciones VIP diferenciadas."*
- **Slide 9 (1:00):** *"Con **SHAP**, verificamos global y localmente que `tenure`, `ContractRiskTier` (variable de ingeniería), fibra óptica sin seguridad/soporte y cheque electrónico impulsan la fuga. Recordamos siempre que SHAP mide atribución dentro del modelo, no causalidad directa."*
- **Slide 10 (1:00):** *"Finalmente, en el análisis de errores sobre Test (`53` FN y `234` FP), la evidencia muestra un `Recall > 90%` en contratos mes a mes (`90.9%`), fibra óptica (`91.3%`) y segmento C1 (`94.1%`). En cambio, el `58.5%` de los `53` Falsos Negativos son clientes consolidados con contratos de 1 o 2 años y antigüedad media de `31` meses: proponemos como hipótesis que esas fugas responden a eventos externos no observados en datos estáticos, como fallas técnicas recientes o vencimiento de contrato."*

### Bloque 4 — Rodrigo Gamero (8:30 – 12:00 | Slide 11 + Demo Streamlit + Slide 12)
- **Checklist Demo en Vivo (2:30 min):**
  *(Nota: tener `streamlit run app/streamlit_app.py` abierto en el navegador antes de iniciar la exposición.)*
  1. *(0:45)* **Pestaña 1 — Diagnóstico Individual:** Clic en **`🔥 Cargar: Alto Valor en Riesgo`** → mostrar probabilidad de churn, semáforo rojo, segmento `Valor Medio / Alto`, acción P1/P2 de la Matriz 2×2 y el **gráfico SHAP Waterfall generado en vivo en <0.3s**. Cambiar brevemente en el sidebar a **Umbrales Dinámicos por Cluster** para mostrar cómo el sistema adapta la decisión al valor del cliente.
  2. *(0:45)* **Pestaña 2 — Batch Scoring CSV:** Clic en **`📥 Usar muestra offline (50 clientes de Test)`** → mostrar KPIs de facturación mensual alertada, gráfico interactivo Plotly y botón de descarga priorizada para call center.
  3. *(1:00)* **Pestaña 3 — Simulador Financiero:** Deslizar **`Efectividad Campaña (%)`** de `100%` a `30%` y luego `50%` para mostrar en vivo el punto de equilibrio económico y por qué el modelo TF1 sigue superando al TP1 y baselines.
- **Slide 12 (1:00):** *"Para cerrar: concluimos que el flujo TF1 es técnicamente reproducible, explicable y financieramente robusto bajo los supuestos evaluados; pero no afirmamos causalidad ni utilidad contable auditada. El principal riesgo fuera de contexto sería aplicar el umbral `0.20` en otra operadora con distinto mix tarifario sin recalibrar. Como trabajo futuro proponemos incorporar series temporales de uso y tickets, validar efectividad con un piloto A/B (uplift) y automatizar monitoreo de drift. Muchas gracias."*

---

## 3. Banco de Preguntas Difíciles del Docente (Q&A — hasta 5 min)

1. **¿Por qué eligieron Stacking (`0.6732`) si el test de Wilcoxon contra XGBoost (`0.6724`, `p=0.695`) y LightGBM (`0.6716`, `p=0.322`) no dio diferencia significativa?**
   - *Respuesta:* *"Excelente observación, profesor. Reportamos con total transparencia en la tabla de Wilcoxon que la diferencia frente a XGBoost y LightGBM es estadísticamente indistinguible en PR-AUC; sin embargo, Stacking combinó la calibración lineal de LR con la capacidad no lineal de Boosting logrando simultáneamente el mejor `PR-AUC` (`0.6732`), el mejor `ROC-AUC` (`0.8515`) y un `Brier Score` muy sólido (`0.1333`) sin requerir post-calibración. Si por latencia en producción se exigiera un único modelo más simple, XGBoost afinado sería el reemplazo directo con desempeño estadísticamente equivalente."*
2. **¿Por qué no usaron el umbral `0.05` si maximizaba la utilidad en OOF y en Test (`+$70,250`)?**
   - *Respuesta:* *"Porque bajo la asimetría `500:50`, el óptimo Bayesiano puro `0.05` clasifica como fuga al `75.7%` de toda la cartera (`Precision = 34.5%`). Eso equivale a hacer una campaña masiva indiscriminada, que contradice la definición del problema y colapsaría la capacidad operativa del call center. Al imponer la restricción `Precision >= 50%` y `Recall >= 75%` en OOF obtuvimos `tau* = 0.20`, que contacta de forma selectiva logrando `81.1%` de Recall y `+$41,600` netos."*
3. **¿Cómo garantizaron que no hubo data leakage al diseñar las 8 variables de Feature Engineering ni al ajustar K-Means o el umbral `0.20`?**
   - *Respuesta:* *"Todas las variables de `TelcoFeatureEngineer` (`BillingRatio`, `TotalServicesSubscribed`, etc.) se calculan fila por fila sin usar estadísticas agregadas de la muestra. La estandarización, `OneHotEncoder`, Optuna, `K-Means` y `PCA` se ajustaron exclusivamente dentro de `Train + Val` (`5,986` filas), y el umbral `0.20` se buscó sobre predicciones Out-Of-Fold de esos 10 folds. El bloque `Test` (`1,057` filas) se evaluó una única vez al final."*
4. **¿Qué garantiza que los clusters representan 'valor' si no tienen costos ni margen por cliente?**
   - *Respuesta:* *"Por rigor metodológico, nunca los presentamos como CLV o RFM contable real —lo documentamos como limitación desde el TP1—. Los definimos explícitamente como perfiles/proxies de facturación y vinculación contractual construidos con `MonthlyCharges`, `TotalCharges`, `tenure` y número de servicios contratados."*
5. **¿SHAP demuestra que ofrecer soporte técnico reducirá la fuga en un 15%?**
   - *Respuesta:* *"No. SHAP mide la contribución marginal dentro de la función aprendida por el modelo sobre datos observacionales transversales. Para estimar impacto causal real se requiere un experimento controlado A/B (uplift modeling) en producción."*

---

## 4. Plan de Contingencia Offline (Sección 5.4)

1. **Sin dependencia de internet:** Tanto `streamlit run app/streamlit_app.py` como el botón **`📥 Usar muestra offline (50 clientes de Test)`** operan 100% desde archivos locales (`.joblib`, `.npz`, `.json` y `data/processed/test.csv`).
2. **Fallo del entorno Python o proyector congelado:** Avanzar a la **Diapositiva 13** de `TF1_Customer_Churn_Data_Mining_Tools.pptx`, que resume las métricas verificadas de respaldo y los casos demostrativos pre-calculados, manteniendo disponible el repositorio y las figuras en `reports/figures_tf1/` para verificación inmediata del docente.
