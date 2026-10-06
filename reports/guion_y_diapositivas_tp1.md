# Estructura de Diapositivas y Guion de Exposición Oral (TP1)

**Proyecto:** Predicción de Fuga de Clientes (Customer Churn)  
**Curso:** Data Mining Tools (CC209) — UPC  
**Tiempo límite:** 10 minutos de exposición + 5 minutos de preguntas  
**Participación:** todos los integrantes deben intervenir

Este guion sigue, página por página, la presentación `reports/TP1_Customer_Churn_Data_Mining_Tools.pdf`. Todas las cifras provienen de `README.md`, de los cuadernos ejecutados y de los archivos `reports/comparacion_modelos_validacion_tp1.csv` y `reports/evaluacion_final_test_tp1.csv`.

---

## Esquema general de tiempo (10 minutos)

| Minutos | Página del PDF | Contenido |
| --- | --- | --- |
| 0:00 – 0:30 | 1 | Portada |
| 0:30 – 1:30 | 2 | Problema y criterio de éxito |
| 1:30 – 2:30 | 3 | Dataset y calidad de datos |
| 2:30 – 4:00 | 4 | EDA orientado a preguntas |
| 4:00 – 5:15 | 5 | Flujo reproducible y prevención de leakage |
| 5:15 – 6:30 | 6 | Selección del modelo en Validation |
| 6:30 – 8:00 | 7 | Evaluación final en Test |
| 8:00 – 9:00 | 8 | Lectura crítica del TP1 |
| 9:00 – 10:00 | 9 | Plan hacia el Trabajo Final |
| Preguntas | 10 | Apéndice de resultados reproducibles |

---

## Página 1: Portada (0.5 min)

> *"Buenos días. Nuestro proyecto busca identificar a los clientes con mayor riesgo de cancelar su servicio, para priorizar las acciones de retención en lugar de aplicarlas sobre toda la cartera. Lo que presentamos es un primer corte, no un producto concluido."*

---

## Página 2: Problema y criterio de éxito (1 min)

* **En pantalla:** pregunta del proyecto, unidad de análisis, 26.54 % de churn y los tres criterios del grupo (PR-AUC > 0.60, Recall ≥ 0.75, valor simulado > 0).

> *"Es un problema de clasificación binaria supervisada sobre clientes individuales. El 26.5 % canceló el servicio, de modo que un modelo que siempre prediga que nadie se va acierta el 73.5 % de las veces y no sirve. Por eso, antes de modelar, fijamos tres criterios para juzgar si el resultado es útil. Son metas internas nuestras, y más adelante veremos que una de ellas no se cumple."*

---

## Página 3: Dataset y calidad de datos (1 min)

* **En pantalla:** 7,043 clientes, 21 variables, 0 duplicados completos; 11 vacíos en `TotalCharges` con `tenure = 0`; procedencia y limitación de corte transversal.
* **Respaldo en el Cuaderno 01 (no está en la lámina):** 0 `customerID` repetidos; 22 perfiles idénticos al excluir el identificador, que se conservan; sin valores fuera de los límites IQR; `No internet service` y `No phone service` son categorías redundantes, no errores.

> *"Trabajamos con 7,043 clientes y 21 variables. Cada decisión de preparación tiene su razón: los 11 vacíos de TotalCharges corresponden a clientes con antigüedad cero, que aún no cierran su primer ciclo de facturación; por eso les asignamos cero y no la mediana, que habría falseado su situación. No encontramos duplicados ni valores extremos, de modo que no eliminamos filas ni recortamos variables. La limitación principal es que el dataset es un corte transversal."*

---

## Página 4: EDA orientado a preguntas (1.5 min)

* **En pantalla:** churn por contrato (42.7 %, 11.3 %, 2.8 %); fibra óptica sin soporte técnico 49.4 %; cheque electrónico 45.3 %; mayor riesgo al inicio.
* **Respaldo en el Cuaderno 01:** fibra óptica con soporte técnico 22.6 %; por tramos de antigüedad, 54.3 % (0 a 5 meses), 36.0 % (6 a 12), 28.7 % (13 a 24), 20.4 % (25 a 48) y 9.5 % (49 a 72); otros métodos de pago entre 15.2 % y 19.1 %.

> *"Distinguimos lo que los datos muestran de lo que interpretamos. Los datos muestran que el 42.7 % de los clientes con contrato mensual se va, frente al 2.8 % con contrato de dos años, y que más de la mitad de los clientes con menos de seis meses cancela. Interpretamos que el compromiso contractual actúa como barrera de salida y que existe una fricción en los primeros meses; de ahí sale nuestro baseline heurístico. En fibra óptica, la fuga es de 49.4 % sin soporte técnico y de 22.6 % con soporte. Son asociaciones: con estos datos no podemos afirmar que dar soporte reduzca la fuga."*

---

## Página 5: Flujo reproducible y prevención de leakage (1.25 min)

* **En pantalla:** Train 4,930 (70 %), Validation 1,056 (15 %), Test 1,057 (15 %); pipeline de preprocesamiento; "Validation decide; Test no participa en la selección".

> *"Separamos los datos en tres conjuntos estratificados antes de ajustar cualquier transformación. Validation sirve para elegir entre modelos y Test se reserva para una única evaluación final. La imputación, el escalado y la codificación viven dentro del Pipeline, de modo que sus estadísticas se aprenden solo con los datos de entrenamiento. Todo se reproduce con un comando, python -m src.train_models, con las versiones fijadas en requirements.txt."*

---

## Página 6: Selección del modelo en Validation (1.25 min)

* **En pantalla:** Regresión Logística PR-AUC 0.6313 y Recall 0.8143; Random Forest PR-AUC 0.6327 y Recall 0.7571; utilidad simulada +$42,700 y +$31,400.

> *"Comparamos una alternativa lineal y un ensamble de árboles. Nuestra regla, definida en el código, es elegir automáticamente por PR-AUC en Validation, y bajo esa regla queda el Random Forest por una diferencia de 0.0014. Lo decimos con franqueza: la Regresión Logística tiene mejor recall y mejor valor simulado, así que no afirmamos que el Random Forest sea superior. Es un candidato preliminar."*

---

## Página 7: Evaluación final en Test (1.5 min)

* **En pantalla:** Accuracy 0.7833, Precision 0.5756, Recall 0.7046, ROC-AUC 0.8400, PR-AUC 0.6698; matriz TN 630, FP 146, FN 83, TP 198; valor simulado −$140,500 (Dummy), −$40,550 (heurístico) y +$20,500 (Random Forest).

> *"¿El modelo mejora frente a una solución simple? Sí. El baseline trivial no detecta ninguna fuga y la regla heurística detecta el 44 %; el Random Forest detecta el 70 %, 198 de 281, con un PR-AUC de 0.67. Bajo nuestros supuestos de costo, se pasa de pérdidas a una utilidad simulada de 20,500 dólares. Dos advertencias: los montos son supuestos nuestros, no datos de la empresa, y la meta de recall de 0.75 no se cumple en Test. Lo reportamos como limitación y no volvemos a elegir modelo mirando Test."*

---

## Página 8: Lectura crítica del TP1 (1 min)

* **En pantalla:** lo que ya funciona, limitaciones y decisión del TP1.

> *"Lo que ya funciona: un flujo reproducible que supera a los baselines y deja a Test fuera de la selección. Lo que no está resuelto: el recall queda bajo la meta, la diferencia entre modelos es mínima, no hay validación cruzada ni búsqueda de hiperparámetros, y los costos son supuestos. Además comprobamos que la selección es sensible a la versión de scikit-learn, lo que confirma que la ventaja del Random Forest es frágil. Por eso lo mantenemos como candidato preliminar."*

---

## Página 9: Plan hacia el Trabajo Final (1 min)

* **En pantalla:** estabilidad, optimización, interpretabilidad y producto.

> *"Cada pendiente tiene una actividad que lo atiende: la validación cruzada medirá si la ventaja entre modelos es estable; el ajuste de umbral atacará el recall que hoy no alcanza la meta; la interpretabilidad y el análisis de errores explicarán en qué casos falla el modelo; y el despliegue en Streamlit lo hará utilizable fuera del cuaderno. Declaramos el uso de IA generativa como apoyo; las decisiones y conclusiones son del equipo. Quedamos atentos a sus preguntas y podemos abrir el repositorio para verificar cualquier cifra."*

---

## Página 10: Apéndice

Tabla de referencia con los valores de Validation y Test. Usarla solo si el docente pide una cifra exacta.

---

## Preguntas probables del docente y cómo responderlas

1. **¿Por qué descartaron Accuracy si supera el 73 %?**
   * *"Porque la clase de interés es el 26.5 % de la base. El baseline trivial obtiene 73.4 % de Accuracy en Test con un recall de 0 %. Por eso priorizamos PR-AUC y lo complementamos con recall, precision y la simulación económica."*

2. **¿Cómo evitaron el data leakage?**
   * *"La partición en Train, Validation y Test se hace antes de ajustar cualquier transformación. Las medianas, la escala y las categorías del OneHotEncoder se aprenden dentro del Pipeline solo con los datos de entrenamiento. La única corrección previa al split es asignar 0 a TotalCharges cuando tenure es 0, que es una regla lógica y no una estadística estimada de la muestra."*

3. **¿Por qué eligieron Random Forest si la Regresión Logística tiene mejor recall y mejor valor económico en Validation?**
   * *"Porque la regla de selección, PR-AUC en Validation, está definida en el código y se aplica de forma automática; no quisimos cambiarla después de ver los resultados. La diferencia es de 0.0014, así que no la consideramos concluyente. Reconocemos que esa regla no coincide del todo con nuestros criterios de utilidad, y en el TF1 la revisaremos con validación cruzada."*

4. **La meta de recall era 0.75 y obtuvieron 0.70. ¿Qué harán?**
   * *"No modificaremos el modelo mirando Test. En el TF1 ajustaremos el umbral de decisión, hoy fijo en 0.50, usando validación cruzada sobre los datos de entrenamiento."*

5. **¿Por qué imputaron 0 en `TotalCharges` y no la mediana?**
   * *"Porque los 11 registros tienen antigüedad cero: son clientes que aún no acumulan facturación. La mediana de clientes consolidados no representaría su situación."*

6. **¿El resultado es reproducible?**
   * *"Sí, con las versiones fijadas en requirements.txt. Lo señalamos porque comprobamos que con scikit-learn 1.9.1 el Random Forest arroja otros valores y la selección cambia, lo que confirma que la ventaja entre modelos es frágil."*

7. **¿De dónde salen los costos de la simulación?**
   * *"Son supuestos de trabajo del grupo, no datos de la empresa. La simulación asume además que todo cliente en riesgo detectado es retenido. En el TF1 haremos un análisis de sensibilidad."*
