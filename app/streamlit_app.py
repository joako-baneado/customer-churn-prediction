"""
Aplicación Web Interactiva en Streamlit - Proyecto Integrador TF1 (CC209).
Conecta directamente los artefactos serializados (pipeline Stacking + K-Means),
validación de contratos Pydantic v2 e interpretabilidad SHAP en vivo.

Ejecución desde la raíz del proyecto:
    streamlit run app/streamlit_app.py
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st
from pydantic import ValidationError
from sklearn.metrics import precision_score, recall_score

from src.clustering import ValueClusterSegmenter  # noqa: F401
from src.config import DATA_PROCESSED_DIR, MODELS_DIR
from src.evaluation import BusinessRuleBaseline, compute_business_impact_custom, value_weighted_recall
from src.explainability import explain_single_customer
from src.schemas import CustomerInput, validate_batch_dataframe


st.set_page_config(
    page_title="Telco Churn AI | TF1 CC209",
    page_icon="📡",
    layout="wide",
)


PRESETS = {
    "vip_riesgo": {
        "gender": "Female",
        "SeniorCitizen": "0",
        "Partner": "No",
        "Dependents": "No",
        "tenure": 4,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 98.50,
        "TotalCharges": 394.00,
    },
    "estable": {
        "gender": "Male",
        "SeniorCitizen": "0",
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 58,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 68.00,
        "TotalCharges": 3944.00,
    },
    "frontera": {
        "gender": "Female",
        "SeniorCitizen": "1",
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 16,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 79.25,
        "TotalCharges": 1268.00,
    },
}


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / "pipeline_tf1_final.joblib")
    tp1_model = joblib.load(MODELS_DIR / "pipeline_tp1_seleccionado.joblib")
    segmenter = joblib.load(MODELS_DIR / "clustering_tf1.joblib")
    with open(MODELS_DIR / "metadata_tf1.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, tp1_model, segmenter, metadata


@st.cache_data
def load_test_holdout() -> pd.DataFrame:
    df_test = pd.read_csv(DATA_PROCESSED_DIR / "test.csv")
    if "SeniorCitizen" in df_test.columns:
        df_test["SeniorCitizen"] = df_test["SeniorCitizen"].astype(str)
    return df_test


def _get_threshold_for_cluster(cluster_id: int, policy_mode: str, metadata: dict) -> float:
    if policy_mode == "dynamic":
        return float(metadata["dynamic_thresholds_by_cluster"][str(cluster_id)]["threshold"])
    return float(metadata["global_threshold"])


def _get_action_info(risk_label: str, cluster_id: int, metadata: dict) -> dict:
    key = f"{risk_label}|{cluster_id}"
    return metadata["action_matrix_rules"].get(
        key,
        {
            "cuadrante": f"{risk_label} + Segmento {cluster_id}",
            "prioridad": "Estándar",
            "accion": "Monitoreo estándar.",
        },
    )


def _apply_preset(preset_key: str):
    for k, v in PRESETS[preset_key].items():
        st.session_state[f"ind_{k}"] = v


def _init_form_state():
    if "ind_tenure" not in st.session_state:
        _apply_preset("vip_riesgo")


def main():
    model, tp1_model, segmenter, metadata = load_artifacts()
    df_test = load_test_holdout()

    st.title("📡 Sistema Predictivo de Fuga de Clientes (Telco Churn — TF1)")
    st.caption(
        "CC209 Data Mining Tools | Modelo: Stacking Heterogéneo (LR + LightGBM + XGBoost) "
        "+ Segmentación K-Means (k=3) + Explicabilidad SHAP"
    )

    with st.sidebar:
        st.header("⚙️ Política de Decisión")
        policy_choice = st.radio(
            "Estrategia de umbral",
            options=["global", "dynamic"],
            format_func=lambda x: (
                f"Umbral Global Operativo (τ* = {metadata['global_threshold']:.2f})"
                if x == "global"
                else "Umbrales Dinámicos por Cluster (0.30 / 0.22 / 0.21)"
            ),
            index=0,
        )

        st.divider()
        st.subheader("📊 Desempeño Verificado en Test (n=1,057)")
        st.metric("PR-AUC (Stacking TF1)", "0.6719", delta="+0.0021 vs TP1")
        st.metric("Recall (τ* = 0.20)", "81.1%", delta="+10.7 pp vs TP1")
        st.metric("Value-Weighted Recall", "84.1%", delta="+10.0 pp vs TP1")
        st.metric("Utilidad Neta Simulada", "$41,600", delta="+$21,100 vs TP1")
        st.caption(
            "Nota: los clusters representan perfiles/proxies de valor y la utilidad "
            "proviene de simulación bajo supuestos explícitos."
        )

    tab1, tab2, tab3 = st.tabs(
        [
            "🎯 Diagnóstico Individual (SHAP + Matriz 2×2)",
            "📂 Scoring por Lotes (Batch CSV)",
            "💰 Simulador Financiero Ejecutivo",
        ]
    )

    # =========================================================================
    # PESTAÑA 1: DIAGNÓSTICO INDIVIDUAL
    # =========================================================================
    with tab1:
        _init_form_state()
        st.subheader("Evaluación Individual de Riesgo y Retención")

        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1:
            if st.button("🔥 Cargar: Alto Valor en Riesgo", width="stretch"):
                _apply_preset("vip_riesgo")
        with c_p2:
            if st.button("🛡️ Cargar: Consolidado Estable", width="stretch"):
                _apply_preset("estable")
        with c_p3:
            if st.button("⚖️ Cargar: Caso Frontera", width="stretch"):
                _apply_preset("frontera")

        with st.form("individual_form"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                gender = st.selectbox("Género", ["Female", "Male"], key="ind_gender")
                senior = st.selectbox("Adulto Mayor (Senior)", ["0", "1"], key="ind_SeniorCitizen")
                partner = st.selectbox("Tiene Pareja", ["Yes", "No"], key="ind_Partner")
                dependents = st.selectbox("Dependientes", ["Yes", "No"], key="ind_Dependents")
                tenure = st.number_input("Antigüedad (meses)", min_value=0, max_value=120, step=1, key="ind_tenure")
            with col2:
                contract = st.selectbox("Contrato", ["Month-to-month", "One year", "Two year"], key="ind_Contract")
                paperless = st.selectbox("Facturación Digital", ["Yes", "No"], key="ind_PaperlessBilling")
                payment = st.selectbox(
                    "Método de Pago",
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                    key="ind_PaymentMethod",
                )
                monthly = st.number_input("Cargo Mensual ($)", min_value=1.0, max_value=1000.0, step=1.0, key="ind_MonthlyCharges")
                total = st.number_input("Cargos Acumulados ($)", min_value=0.0, max_value=50000.0, step=10.0, key="ind_TotalCharges")
            with col3:
                phone = st.selectbox("Servicio Telefónico", ["Yes", "No"], key="ind_PhoneService")
                multiple = st.selectbox("Múltiples Líneas", ["Yes", "No", "No phone service"], key="ind_MultipleLines")
                internet = st.selectbox("Internet", ["Fiber optic", "DSL", "No"], key="ind_InternetService")
                security = st.selectbox("Seguridad Online", ["Yes", "No", "No internet service"], key="ind_OnlineSecurity")
                backup = st.selectbox("Backup Online", ["Yes", "No", "No internet service"], key="ind_OnlineBackup")
            with col4:
                protection = st.selectbox("Protección Dispositivo", ["Yes", "No", "No internet service"], key="ind_DeviceProtection")
                support = st.selectbox("Soporte Técnico", ["Yes", "No", "No internet service"], key="ind_TechSupport")
                tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], key="ind_StreamingTV")
                movies = st.selectbox("Streaming Películas", ["Yes", "No", "No internet service"], key="ind_StreamingMovies")

            submitted = st.form_submit_button("🔍 Evaluar Cliente y Explicar con SHAP", type="primary")

        payload = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": int(tenure),
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": float(monthly),
            "TotalCharges": float(total),
        }

        try:
            validated_customer = CustomerInput(**payload)
            df_row = validated_customer.to_dataframe()
            prob = float(model.predict_proba(df_row)[:, 1][0])
            cid = int(segmenter.predict_cluster_ids(df_row)[0])
            clabel = segmenter.label_mapping[cid]
            thr_active = _get_threshold_for_cluster(cid, policy_choice, metadata)
            is_high_risk = prob >= thr_active
            risk_key = "Alto Riesgo" if is_high_risk else "Bajo Riesgo"
            action_info = _get_action_info(risk_key, cid, metadata)

            st.divider()
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Probabilidad de Churn", f"{prob:.1%}", delta=f"Umbral activo: {thr_active:.2f}")
            k2.metric("Semáforo de Riesgo", "🔴 ALTO RIESGO" if is_high_risk else "🟢 BAJO RIESGO")
            k3.metric("Segmento de Valor (Cluster)", f"C{cid}: {clabel}")
            k4.metric("Prioridad Comercial", action_info["prioridad"])

            if is_high_risk:
                st.error(f"**{action_info['cuadrante']}** — {action_info['accion']}")
            else:
                st.success(f"**{action_info['cuadrante']}** — {action_info['accion']}")

            # Explicabilidad SHAP en vivo
            with st.spinner("Calculando atribución SHAP local..."):
                sv_row = explain_single_customer(model, df_row)

            col_shap_plot, col_shap_tbl = st.columns([1.4, 1.0])
            with col_shap_plot:
                st.markdown("#### Explicación Local SHAP (Waterfall)")
                fig_wf = plt.figure(figsize=(8.5, 4.8))
                shap.plots.waterfall(sv_row, max_display=10, show=False)
                plt.tight_layout()
                st.pyplot(fig_wf, clear_figure=True)

            with col_shap_tbl:
                st.markdown("#### Top Factores Determinantes")
                df_contrib = pd.DataFrame(
                    {
                        "Variable": sv_row.feature_names,
                        "Impacto SHAP": sv_row.values,
                        "|Impacto|": np.abs(sv_row.values),
                    }
                ).sort_values("|Impacto|", ascending=False).head(8)
                st.dataframe(df_contrib[["Variable", "Impacto SHAP"]].round(4), width="stretch", hide_index=True)
                st.caption("SHAP refleja contribución aditiva en el modelo, no causalidad directa.")

        except ValidationError as err:
            st.error("⚠️ Error de validación de esquema (Pydantic): revisa la coherencia del perfil ingresado.")
            st.code(str(err))

    # =========================================================================
    # PESTAÑA 2: SCORING POR LOTES
    # =========================================================================
    with tab2:
        st.subheader("Priorización de Cartera por Lotes (Batch Scoring)")
        col_u1, col_u2 = st.columns([2, 1])
        with col_u1:
            uploaded = st.file_uploader("Sube un archivo CSV con clientes Telco", type=["csv"])
        with col_u2:
            st.write("")
            st.write("")
            use_demo_batch = st.button("📥 Usar muestra offline (50 clientes de Test)", width="stretch")

        df_batch_raw = None
        if uploaded is not None:
            df_batch_raw = pd.read_csv(uploaded)
        elif use_demo_batch or "batch_demo_loaded" in st.session_state:
            st.session_state["batch_demo_loaded"] = True
            df_batch_raw = df_test.drop(columns=["Churn"], errors="ignore").head(50).copy()
            df_batch_raw.insert(0, "customerID", [f"TEST-{i:04d}" for i in range(len(df_batch_raw))])

        if df_batch_raw is not None:
            try:
                df_valid, val_errors = validate_batch_dataframe(df_batch_raw)
                if val_errors:
                    st.warning(f"Se omitieron {len(val_errors)} filas con errores de esquema.")
                if not df_valid.empty:
                    feature_df = df_valid.drop(columns=["customerID"], errors="ignore")
                    probas = model.predict_proba(feature_df)[:, 1]
                    cids = segmenter.predict_cluster_ids(feature_df)
                    clabels = [segmenter.label_mapping[int(c)] for c in cids]
                    thresholds = np.asarray(
                        [_get_threshold_for_cluster(int(c), policy_choice, metadata) for c in cids]
                    )
                    alerts = probas >= thresholds

                    res_df = df_valid.copy()
                    res_df["Prob_Churn"] = np.round(probas, 4)
                    res_df["Umbral_Aplicado"] = thresholds
                    res_df["Alerta_Fuga"] = np.where(alerts, "Alto Riesgo", "Bajo Riesgo")
                    res_df["Cluster_Valor"] = clabels
                    res_df["Ingreso_Mensual_En_Riesgo"] = np.round(
                        res_df["MonthlyCharges"] * res_df["Prob_Churn"], 2
                    )
                    res_df["Accion_Recomendada"] = [
                        _get_action_info("Alto Riesgo" if al else "Bajo Riesgo", int(c), metadata)["accion"]
                        for al, c in zip(alerts, cids)
                    ]
                    res_df = res_df.sort_values("Ingreso_Mensual_En_Riesgo", ascending=False)

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Clientes Validados", len(res_df))
                    m2.metric("Clientes en Alto Riesgo", int(alerts.sum()), f"{alerts.mean():.1%} del lote")
                    m3.metric(
                        "Facturación Mensual Alertada",
                        f"${res_df.loc[alerts, 'MonthlyCharges'].sum():,.2f}",
                    )
                    m4.metric(
                        "Exposición Esperada (P × $)",
                        f"${res_df['Ingreso_Mensual_En_Riesgo'].sum():,.2f}",
                    )

                    fig_sc = px.scatter(
                        res_df,
                        x="MonthlyCharges",
                        y="Prob_Churn",
                        color="Cluster_Valor",
                        symbol="Alerta_Fuga",
                        hover_data=["tenure", "Contract", "Accion_Recomendada"],
                        title="Mapa de Priorización: Probabilidad de Fuga vs. Cargo Mensual ($)",
                    )
                    st.plotly_chart(fig_sc, width="stretch")

                    display_cols = [
                        c for c in [
                            "customerID", "Prob_Churn", "Alerta_Fuga", "Cluster_Valor",
                            "MonthlyCharges", "Ingreso_Mensual_En_Riesgo", "Contract", "tenure", "Accion_Recomendada"
                        ] if c in res_df.columns
                    ]
                    st.dataframe(res_df[display_cols], width="stretch", hide_index=True)

                    csv_bytes = res_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Descargar Lista Priorizada para Call Center (CSV)",
                        data=csv_bytes,
                        file_name="cartera_priorizada_tf1.csv",
                        mime="text/csv",
                    )
            except Exception as exc:
                st.error(f"Error procesando archivo: {exc}")
        else:
            st.info("Carga un CSV con las 19 columnas Telco o presiona el botón de muestra offline para explorar.")

    # =========================================================================
    # PESTAÑA 3: SIMULADOR FINANCIERO EJECUTIVO
    # =========================================================================
    with tab3:
        st.subheader("Simulador de Sensibilidad Económica sobre Test Holdout (n=1,057)")

        sc1, sc2, sc3, sc4 = st.columns(4)
        cost_fn = sc1.slider("Costo Falso Negativo ($)", min_value=100, max_value=1000, value=500, step=50)
        cost_fp = sc2.slider("Costo Falso Positivo ($)", min_value=10, max_value=200, value=50, step=10)
        gross_tp = sc3.slider("Beneficio Base TP ($)", min_value=100, max_value=700, value=350, step=25)
        effectiveness = sc4.slider("Efectividad Campaña (%)", min_value=20, max_value=100, value=100, step=5)

        effective_tp = gross_tp * (effectiveness / 100.0)
        use_dyn_sim = st.checkbox("Simular con Umbrales Dinámicos por Cluster (0.30 / 0.22 / 0.21)", value=False)
        manual_thr = st.slider(
            "Umbral de decisión τ (si no usas dinámicos)",
            min_value=0.05,
            max_value=0.80,
            value=float(metadata["global_threshold"]),
            step=0.01,
            disabled=use_dyn_sim,
        )

        X_t = df_test.drop(columns=["Churn"])
        y_t = df_test["Churn"].values
        mc_t = X_t["MonthlyCharges"].values

        p_tf1 = model.predict_proba(X_t)[:, 1]
        cids_t = segmenter.predict_cluster_ids(X_t)
        if use_dyn_sim:
            thr_vec = np.asarray(
                [metadata["dynamic_thresholds_by_cluster"][str(int(c))]["threshold"] for c in cids_t]
            )
            pred_tf1 = (p_tf1 >= thr_vec).astype(int)
            tf1_label = "TF1 Stacking (Umbrales Dinámicos)"
        else:
            pred_tf1 = (p_tf1 >= manual_thr).astype(int)
            tf1_label = f"TF1 Stacking (τ={manual_thr:.2f})"

        pred_tp1 = tp1_model.predict(X_t)
        pred_heur = BusinessRuleBaseline(tenure_threshold=6).predict(X_t)
        pred_dummy = np.zeros_like(y_t)

        sim_rows = []
        for label, preds in [
            ("Baseline Dummy", pred_dummy),
            ("Baseline Heurístico", pred_heur),
            ("TP1 Random Forest (0.50)", pred_tp1),
            (tf1_label, pred_tf1),
        ]:
            tp = int(np.sum((y_t == 1) & (preds == 1)))
            fp = int(np.sum((y_t == 0) & (preds == 1)))
            fn = int(np.sum((y_t == 1) & (preds == 0)))
            tn = int(np.sum((y_t == 0) & (preds == 0)))
            util = compute_business_impact_custom(
                tn, fp, fn, tp, cost_fn=cost_fn, cost_fp=cost_fp, benefit_tp=effective_tp
            )
            sim_rows.append(
                {
                    "Modelo": label,
                    "Utilidad Neta ($)": util,
                    "Recall": recall_score(y_t, preds, zero_division=0),
                    "Value-Weighted Recall": value_weighted_recall(y_t, preds, mc_t),
                    "Precision": precision_score(y_t, preds, zero_division=0),
                    "Contactados": tp + fp,
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                }
            )
        df_sim = pd.DataFrame(sim_rows)
        fig_bar = px.bar(
            df_sim,
            x="Modelo",
            y="Utilidad Neta ($)",
            color="Modelo",
            text_auto=",.0f",
            title=f"Impacto Económico Simulado (Efectividad={effectiveness}%, Beneficio TP efectivo=${effective_tp:.1f})",
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, width="stretch")
        st.dataframe(df_sim.round(4), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
