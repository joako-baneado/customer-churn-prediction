"""
Generador reproducible del Informe Técnico (PDF) y Presentación Ejecutiva (PPTX)
para el Trabajo Final (TF1 - CC209).
Lee directamente los CSVs y figuras reales generados en reports/.

Ejecución:
    python scripts/build_reports_tf1.py
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
from PIL import Image as PILImage
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.config import FIGURES_TF1_DIR, MODELS_DIR, REPORTS_DIR


NAVY = RGBColor(27, 54, 93)
TEAL = RGBColor(0, 128, 128)
DARK_TEXT = RGBColor(38, 45, 56)
MUTED = RGBColor(100, 110, 125)
LIGHT_BG = RGBColor(244, 247, 251)
WHITE = RGBColor(255, 255, 255)
ACCENT_RED = RGBColor(192, 57, 43)

RL_NAVY = colors.HexColor("#1B365D")
RL_TEAL = colors.HexColor("#008080")
RL_LIGHT = colors.HexColor("#F4F7FB")
RL_BORDER = colors.HexColor("#CFD8DC")


def _load_data():
    with open(MODELS_DIR / "metadata_tf1.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return {
        "metadata": metadata,
        "ablation": pd.read_csv(REPORTS_DIR / "ablation_feature_engineering_tf1.csv"),
        "cv10": pd.read_csv(REPORTS_DIR / "comparacion_modelos_cv_tf1.csv"),
        "wilcoxon": pd.read_csv(REPORTS_DIR / "wilcoxon_cv_tf1.csv"),
        "cal": pd.read_csv(REPORTS_DIR / "calibracion_oof_tf1.csv"),
        "oof_thr": pd.read_csv(REPORTS_DIR / "comparacion_umbrales_oof_tf1.csv"),
        "test": pd.read_csv(REPORTS_DIR / "evaluacion_final_test_tf1.csv"),
        "sens": pd.read_csv(REPORTS_DIR / "sensibilidad_economica_tf1.csv"),
        "clust_comp": pd.read_csv(REPORTS_DIR / "comparacion_clustering_tf1.csv"),
        "profiles": pd.read_csv(REPORTS_DIR / "perfiles_clusters_tf1.csv"),
        "shap_imp": pd.read_csv(REPORTS_DIR / "shap_importancia_global_tf1.csv"),
        "err_quad": pd.read_csv(REPORTS_DIR / "error_analysis_cuadrantes_tf1.csv"),
        "err_seg": pd.read_csv(REPORTS_DIR / "error_analysis_segmentos_tf1.csv"),
        "err_cases": pd.read_csv(REPORTS_DIR / "error_analysis_casos_tf1.csv"),
    }


def _fit_image_dims(img_path: Path, max_w: float, max_h: float) -> tuple[float, float]:
    with PILImage.open(img_path) as im:
        w, h = im.size
    ratio = min(max_w / w, max_h / h)
    return w * ratio, h * ratio


# =============================================================================
# GENERACIÓN DEL INFORME TÉCNICO EN PDF (REPORTLAB)
# =============================================================================

def _rl_table(data_rows: list[list], col_widths=None, font_size: int = 8) -> Table:
    tbl = Table(data_rows, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), RL_NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, RL_LIGHT]),
                ("GRID", (0, 0), (-1, -1), 0.4, RL_BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return tbl


def _rl_image(img_path: Path, max_w_cm: float = 16.5, max_h_cm: float = 9.0) -> RLImage:
    w_pt, h_pt = _fit_image_dims(img_path, max_w_cm * cm, max_h_cm * cm)
    im = RLImage(str(img_path), width=w_pt, height=h_pt)
    im.hAlign = "CENTER"
    return im


def build_pdf(data: dict, out_path: Path):
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title="Informe Trabajo Final TF1 - Telco Customer Churn (CC209)",
    )
    styles = getSampleStyleSheet()
    s_title = ParagraphStyle(
        "TitleCustom", parent=styles["Title"], fontSize=20, leading=24, textColor=RL_NAVY, spaceAfter=10
    )
    s_subtitle = ParagraphStyle(
        "SubCustom", parent=styles["Normal"], fontSize=12, leading=16, textColor=RL_TEAL, spaceAfter=14
    )
    s_h1 = ParagraphStyle(
        "H1Custom", parent=styles["Heading1"], fontSize=13, leading=17, textColor=RL_NAVY, spaceBefore=12, spaceAfter=6
    )
    s_h2 = ParagraphStyle(
        "H2Custom", parent=styles["Heading2"], fontSize=10.5, leading=14, textColor=RL_TEAL, spaceBefore=8, spaceAfter=4
    )
    s_body = ParagraphStyle(
        "BodyCustom", parent=styles["Normal"], fontSize=9.5, leading=13.5, spaceAfter=6
    )
    s_cell = ParagraphStyle("CellCustom", parent=styles["Normal"], fontSize=7.8, leading=10)

    def P(text, style=s_body):
        return Paragraph(str(text), style)

    def PC(text):
        return Paragraph(str(text), s_cell)

    story = []

    # Portada
    story.append(Spacer(1, 2.0 * cm))
    story.append(P("UNIVERSIDAD PERUANA DE CIENCIAS APLICADAS (UPC)", s_subtitle))
    story.append(P("Proyecto Integrador de Data Science — Trabajo Final (TF1)", s_title))
    story.append(P("Sistema Predictivo de Fuga de Clientes con Segmentación de Valor y Explicabilidad SHAP (Customer Churn)", s_h2))
    story.append(Spacer(1, 0.8 * cm))
    meta_rows = [
        ["Curso", "Data Mining Tools – CC209 (Ciclo 2026-20)"],
        ["Docente", "Carlos Fernando Montoya Cubas"],
        ["Integrantes", "Joaquin Basas Requena, Fabian Marcelo Rojas Cuadros, Cielo Luwidka Chavez Merino, Rodrigo Alonso Gamero Vera"],
        ["Dataset y Licencia", "IBM Telco Customer Churn (7,043 x 21) — IBM Sample Data Sets / Kaggle (uso académico y demostrativo)"],
        ["Modelo Ganador TF1", "Stacking Heterogéneo (LR + LightGBM + XGBoost) + K-Means (k=3) + SHAP + Streamlit"],
    ]
    story.append(_rl_table(meta_rows, col_widths=[4.2 * cm, 13.0 * cm], font_size=9))
    story.append(Spacer(1, 1.0 * cm))
    story.append(
        P(
            "<b>Resumen Ejecutivo:</b> Frente al Trabajo Parcial (TP1), donde Random Forest con umbral fijo 0.50 "
            "alcanzó PR-AUC = 0.6698, Recall = 70.46% (sin cumplir la meta interna >= 75%) y utilidad simulada de +$20,500, "
            "la solución integrada del TF1 eleva el desempeño en Test Holdout (n=1,057) a <b>PR-AUC = 0.6719</b>, "
            "<b>ROC-AUC = 0.8443</b>, <b>Recall = 81.14%</b>, <b>Value-Weighted Recall = 84.12%</b> y "
            "<b>utilidad económica neta simulada = +$41,600</b> (2.03× TP1), manteniéndose dominante en todos los escenarios de sensibilidad.",
            s_body,
        )
    )
    story.append(PageBreak())

    # 1. Evolución TP1 -> TF1
    story.append(P("1. Evolución Metodológica y Técnica (TP1 → TF1)", s_h1))
    evo_rows = [
        ["Elemento", "TP1 (Semana 7)", "TF1 (Semana 15)", "Motivo del cambio"],
        [PC("Preparación y Contratos"), PC("ColumnTransformer básico"), PC("TelcoFeatureEngineer + ColumnTransformer + Pydantic v2 + pytest"), PC("Evitar errores de esquema en producción y exponer nombres SHAP")],
        [PC("Variables"), PC("19 atributos crudos"), PC("+8 variables de dominio sin leakage"), PC("Capturar interacciones Fibra/Soporte, BillingRatio y tramos de tenure")],
        [PC("Validación"), PC("Split único 70/15/15"), PC("StratifiedKFold 5/10 folds en Train+Val; Test intacto"), PC("Medir dispersión (mu +- std) y significancia con Wilcoxon")],
        [PC("Modelamiento"), PC("LR y RF manuales"), PC("LR, RF, LightGBM, XGBoost (Optuna) + Stacking"), PC("Explotar Gradient Boosting y ensamble heterogéneo")],
        [PC("Umbral y Calibración"), PC("Fijo en 0.50 (Recall 70.5%)"), PC("Auditoría OOF + tau*=0.20 + dinámicos por cluster"), PC("Superar meta Recall >= 75% evitando campañas masivas (0.05)")],
        [PC("Técnica Adicional"), PC("Ninguna"), PC("K-Means (k=3) vs GMM/Ward + PCA 2D (96.17% var.)"), PC("Cruzar probabilidad de fuga con perfiles de valor (Matriz 2x2)")],
        [PC("Interpretabilidad y Uso"), PC("EDA global / scripts CLI"), PC("SHAP global/local + Error Analysis + App Streamlit"), PC("Explicar cada predicción al equipo comercial fuera del notebook")],
    ]
    story.append(_rl_table(evo_rows, col_widths=[3.2 * cm, 3.6 * cm, 5.2 * cm, 5.2 * cm], font_size=8))

    # 2. Matriz de herramientas
    story.append(P("2. Matriz de Decisiones de Herramientas", s_h1))
    tools_rows = [
        ["Necesidad", "Herramienta elegida", "Alternativa considerada", "Justificación técnica"],
        [PC("EDA"), PC("Pandas + Matplotlib/Seaborn"), PC("Profiling automático"), PC("Permite análisis guiado por hipótesis e interpretación directa")],
        [PC("Preparación"), PC("Pipeline + ColumnTransformer + Pydantic"), PC("get_dummies manual"), PC("Garantiza cero leakage en CV/inferencia y valida tipos/rangos")],
        [PC("Modelamiento"), PC("Stacking (LR + LGBM + XGB) + RF"), PC("Modelo lineal único"), PC("Mayor PR-AUC (0.6732) y ROC-AUC (0.8515) en CV 10-fold")],
        [PC("Experimentación"), PC("Optuna + StratifiedKFold + MLflow"), PC("GridSearch / split único"), PC("Búsqueda bayesiana eficiente y trazabilidad portable a CSV")],
        [PC("Clustering / Reducción"), PC("K-Means (k=3) + PCA 2D"), PC("GMM / Ward / cuantiles"), PC("Mayor Silhouette en k accionable (0.4394) y soporte .predict()")],
        [PC("Interpretabilidad"), PC("SHAP PermutationExplainer"), PC("Importancia Gini"), PC("Atribución aditiva global y local agnóstica al Stacking")],
        [PC("Despliegue y Pruebas"), PC("Streamlit + Plotly + pytest"), PC("API aislada sin UI"), PC("Demo ejecutiva en 3 pestañas verificada con AppTest y pytest")],
    ]
    story.append(_rl_table(tools_rows, col_widths=[2.9 * cm, 4.1 * cm, 3.7 * cm, 6.5 * cm], font_size=8))

    # 3. Experimentación CV y Wilcoxon
    story.append(P("3. Experimentación Sistemática (Solo Train + Val, n=5,986)", s_h1))
    story.append(
        P(
            "En el estudio de ablación (CV 5-fold), <i>TelcoFeatureEngineer</i> elevó el PR-AUC de Regresión Logística de 0.6566 a 0.6602 "
            "y redujo a la mitad la desviación estándar de LightGBM (0.0164 → 0.0088). Tras optimizar cada familia con Optuna (25 trials), "
            "se evaluaron 7 alternativas sobre los mismos 10 folds estratificados:",
            s_body,
        )
    )
    cv_rows = [["Modelo (CV 10-Fold)", "PR-AUC (mu +- std)", "ROC-AUC", "Brier Score", "Recall (0.50)"]]
    for _, r in data["cv10"].iterrows():
        cv_rows.append([
            PC(r["Modelo"]),
            f"{r['PR-AUC_Mean']:.4f} +- {r['PR-AUC_Std']:.4f}",
            f"{r['ROC-AUC_Mean']:.4f}",
            f"{r['Brier Score_Mean']:.4f}",
            f"{r['Recall (Clase 1)_Mean']:.4f}",
        ])
    story.append(_rl_table(cv_rows, col_widths=[6.4 * cm, 3.3 * cm, 2.3 * cm, 2.5 * cm, 2.7 * cm], font_size=8))
    story.append(Spacer(1, 0.3 * cm))
    story.append(_rl_image(FIGURES_TF1_DIR / "cv_comparacion_modelos_tf1.png", max_w_cm=16.5, max_h_cm=7.0))
    story.append(
        P(
            "<b>Prueba de Wilcoxon pareada (alpha=0.05):</b> Stacking Heterogéneo supera de forma estadísticamente significativa a "
            "Random Forest (p = 0.00195) y Regresión Logística (p = 0.02734), mientras que mantiene empate estadístico frente a "
            "XGBoost (p = 0.6953) y LightGBM (p = 0.3223).",
            s_body,
        )
    )
    story.append(PageBreak())

    # 4. Clustering y PCA
    story.append(P("4. Segmentación de Valor (K-Means + PCA 2D)", s_h1))
    story.append(
        P(
            "Se compararon K-Means, GMM y Jerárquico (Ward) para k en {2..6} sobre <i>tenure</i>, <i>MonthlyCharges</i>, "
            "<i>TotalCharges</i> y <i>TotalServicesSubscribed</i>. K-Means (k=3) obtuvo el mayor Silhouette (0.4394) en el rango accionable "
            "(k in {3,4}) y proyección PCA 2D con 96.17% de varianza explicada. Los segmentos representan <b>perfiles/proxies de valor</b> "
            "(no CLV/RFM transaccional real):",
            s_body,
        )
    )
    prof_rows = [["Cluster", "Clientes (%)", "Tenure", "Cargo Mes ($)", "Servicios", "% Mes-a-Mes", "Churn Obs."]]
    for _, r in data["profiles"].iterrows():
        prof_rows.append([
            PC(f"C{int(r['Cluster_ID'])}: {r['Cluster_Label']}"),
            f"{int(r['Clientes'])} ({r['Pct_Cartera']:.1%})",
            f"{r['Tenure_Medio']:.1f} m",
            f"${r['MonthlyCharges_Medio']:.2f}",
            f"{r['Servicios_Medios']:.2f}",
            f"{r['Pct_Mes_a_Mes']:.1%}",
            f"{r['Tasa_Churn_Observada']:.1%}",
        ])
    story.append(_rl_table(prof_rows, col_widths=[4.4 * cm, 2.5 * cm, 1.8 * cm, 2.3 * cm, 1.9 * cm, 2.2 * cm, 2.1 * cm], font_size=8))
    story.append(Spacer(1, 0.25 * cm))
    story.append(_rl_image(FIGURES_TF1_DIR / "clustering_pca_2d.png", max_w_cm=16.5, max_h_cm=7.2))

    # 5. Calibración, Umbrales y Evaluación Única en Test
    story.append(P("5. Calibración OOF, Umbrales y Evaluación Final Única en Test (n=1,057)", s_h1))
    story.append(
        P(
            "El Stacking sin post-calibración logró el menor Brier Score OOF (0.13331 vs 0.13346 Platt y 0.13357 Isotónica). "
            "Al analizar el umbral de decisión en OOF, el óptimo Bayesiano irrestricto bajo la relación 500:50 cae en tau=0.05 "
            "(contactaría al 75.7% de la cartera con Precision 34.5%). Imponiendo la restricción operativa Recall >= 75% y Precision >= 50%, "
            "se seleccionó <b>tau* global = 0.20</b> y umbrales dinámicos por cluster (C0=0.30, C1=0.22, C2=0.21).",
            s_body,
        )
    )
    test_rows = [["Modelo en Test Holdout", "PR-AUC", "ROC-AUC", "Recall", "VW-Recall ($)", "Precision", "Utilidad ($)"]]
    for _, r in data["test"].iterrows():
        test_rows.append([
            PC(r["Modelo"]),
            f"{r['PR-AUC']:.4f}",
            f"{r['ROC-AUC']:.4f}",
            f"{r['Recall (Clase 1)']:.4f}",
            f"{r['Value-Weighted Recall']:.4f}",
            f"{r['Precision (Clase 1)']:.4f}",
            f"${r['Valor Económico Neto ($)']:,.0f}",
        ])
    story.append(_rl_table(test_rows, col_widths=[6.2 * cm, 1.6 * cm, 1.7 * cm, 1.6 * cm, 2.1 * cm, 1.8 * cm, 2.2 * cm], font_size=7.6))
    story.append(Spacer(1, 0.25 * cm))
    story.append(_rl_image(FIGURES_TF1_DIR / "sensibilidad_economica.png", max_w_cm=16.2, max_h_cm=6.8))
    story.append(PageBreak())

    # 6. SHAP y Análisis de Errores
    story.append(P("6. Interpretabilidad SHAP y Auditoría de Errores", s_h1))
    story.append(
        P(
            "Los principales factores en SHAP (muestra estratificada n=200; mide atribución en el modelo, no causalidad) son: "
            "<i>tenure</i> (0.0753), <i>ContractRiskTier</i> (0.0656), <i>InternetService_Fiber optic</i> (0.0379), "
            "<i>OnlineSecurity_No</i> (0.0306), <i>PaymentMethod_Electronic check</i> (0.0219) y <i>HighRisk_Fiber_No_Support</i> (0.0149).",
            s_body,
        )
    )
    story.append(_rl_image(FIGURES_TF1_DIR / "shap_global_beeswarm.png", max_w_cm=15.0, max_h_cm=7.0))
    story.append(Spacer(1, 0.2 * cm))
    story.append(_rl_image(FIGURES_TF1_DIR / "error_analysis_segmentos.png", max_w_cm=16.2, max_h_cm=8.2))
    story.append(
        P(
            "<b>Auditoría de errores en Test (tau*=0.20):</b> Se registraron <b>53 Falsos Negativos</b> y <b>234 Falsos Positivos</b>. "
            "En segmentos vulnerables el Recall supera el 90% (<i>Month-to-month</i>: 90.9%, <i>Fiber optic</i>: 91.3%, "
            "<i>Cluster Valor Medio</i>: 94.1%). En contraste, el 58.5% de los FN corresponde a clientes con contratos de 1 o 2 años "
            "(antigüedad media de 31.0 meses), donde factores no medidos en el dataset transversal explicarían la baja.",
            s_body,
        )
    )

    # 7. Conclusiones y Limitaciones (5 preguntas obligatorias)
    story.append(P("7. Conclusiones, Limitaciones y Trabajo Futuro", s_h1))
    conclusions = [
        ("¿Qué puede concluirse?", "El flujo integrado TF1 supera al TP1 y a ambos baselines en discriminación, cobertura de fugas (81.1%), rescate de ingresos (84.1%) y utilidad simulada (+$41,600) bajo los tres escenarios evaluados."),
        ("¿Qué NO puede concluirse?", "No se demuestran relaciones causales ni ganancias contables reales auditadas; los segmentos son proxies descriptivos y no CLV transaccional."),
        ("Limitaciones principales", "Dataset transversal sin consumo longitudinal, tickets de soporte ni distinción de churn voluntario/involuntario; menor sensibilidad del modelo en contratos de 1-2 años."),
        ("Riesgos fuera de contexto", "Covariate shift y pérdida de validez del umbral tau*=0.20 al aplicar el modelo en otra operadora o sin monitoreo de deriva temporal."),
        ("Trabajo futuro razonable", "Incorporar variables temporales y análisis de supervivencia, validar mediante experimento A/B (uplift modeling) y automatizar monitoreo de drift."),
    ]
    for q, ans in conclusions:
        story.append(P(f"<b>{q}</b> {ans}", s_body))

    doc.build(story)


# =============================================================================
# GENERACIÓN DE LA PRESENTACIÓN PPTX (13 DIAPOSITIVAS)
# =============================================================================

def _add_header(slide, prs, title_text: str, subtitle_text: str, slide_num: int):
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.05))
    band.fill.solid()
    band.fill.fore_color.rgb = NAVY
    band.line.fill.background()

    accent = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(1.05), prs.slide_width, Inches(0.06))
    accent.fill.solid()
    accent.fill.fore_color.rgb = TEAL
    accent.line.fill.background()

    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.14), Inches(11.3), Inches(0.85))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = WHITE

    p2 = tf.add_paragraph()
    p2.text = subtitle_text
    p2.font.size = Pt(12)
    p2.font.color.rgb = RGBColor(205, 224, 240)

    # Footer con número de slide
    ft = slide.shapes.add_textbox(Inches(0.5), Inches(7.12), Inches(12.3), Inches(0.3))
    fp = ft.text_frame.paragraphs[0]
    fp.text = f"Data Mining Tools (CC209) | Trabajo Final (TF1) — Telco Customer Churn   |   Diapositiva {slide_num}/13"
    fp.font.size = Pt(9.5)
    fp.font.color.rgb = MUTED
    fp.alignment = PP_ALIGN.RIGHT


def _add_bullets(slide, left, top, width, height, items: list[tuple[str, str]], font_size: int = 12):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for idx, (lead, body) in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.space_after = Pt(6)
        r1 = p.add_run()
        r1.text = f"• {lead}: " if lead else "• "
        r1.font.bold = True
        r1.font.size = Pt(font_size)
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = body
        r2.font.size = Pt(font_size)
        r2.font.color.rgb = DARK_TEXT


def _add_pptx_table(slide, left, top, width, height, rows: list[list[str]], col_widths=None, font_size: int = 10):
    n_rows, n_cols = len(rows), len(rows[0])
    shape = slide.shapes.add_table(n_rows, n_cols, left, top, width, height)
    tbl = shape.table
    if col_widths:
        for i, cw in enumerate(col_widths):
            tbl.columns[i].width = cw
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            cell = tbl.cell(r_idx, c_idx)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if r_idx == 0:
                cell.fill.fore_color.rgb = NAVY
            elif r_idx % 2 == 1:
                cell.fill.fore_color.rgb = WHITE
            else:
                cell.fill.fore_color.rgb = LIGHT_BG
            p = cell.text_frame.paragraphs[0]
            p.text = str(val)
            p.font.size = Pt(font_size)
            p.font.bold = r_idx == 0
            p.font.color.rgb = WHITE if r_idx == 0 else DARK_TEXT


def _add_pptx_image(slide, img_path: Path, left, top, max_w, max_h):
    w_emu, h_emu = _fit_image_dims(img_path, max_w, max_h)
    slide.shapes.add_picture(str(img_path), left, top, width=int(w_emu), height=int(h_emu))


def _add_kpi_card(slide, left, top, width, height, label: str, value: str, delta: str):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = LIGHT_BG
    card.line.color.rgb = TEAL
    card.line.width = Pt(1.2)

    tf = card.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.text = label
    p1.font.size = Pt(10.5)
    p1.font.color.rgb = MUTED
    p1.alignment = PP_ALIGN.CENTER

    p2 = tf.add_paragraph()
    p2.text = value
    p2.font.size = Pt(22)
    p2.font.bold = True
    p2.font.color.rgb = NAVY
    p2.alignment = PP_ALIGN.CENTER

    p3 = tf.add_paragraph()
    p3.text = delta
    p3.font.size = Pt(10)
    p3.font.bold = True
    p3.font.color.rgb = TEAL
    p3.alignment = PP_ALIGN.CENTER


def build_pptx(data: dict, out_path: Path):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # Slide 1: Portada
    s1 = prs.slides.add_slide(blank)
    bg = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()

    bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(5.1), prs.slide_width, Inches(0.1))
    bar.fill.solid()
    bar.fill.fore_color.rgb = TEAL
    bar.line.fill.background()

    tb = s1.shapes.add_textbox(Inches(0.9), Inches(1.2), Inches(11.5), Inches(3.7))
    tf = tb.text_frame
    tf.word_wrap = True
    p0 = tf.paragraphs[0]
    p0.text = "UNIVERSIDAD PERUANA DE CIENCIAS APLICADAS (UPC)  |  DATA MINING TOOLS (CC209)"
    p0.font.size = Pt(13)
    p0.font.bold = True
    p0.font.color.rgb = RGBColor(127, 219, 218)

    p1 = tf.add_paragraph()
    p1.space_before = Pt(16)
    p1.text = "Sistema Predictivo de Fuga de Clientes con Segmentación de Valor y Explicabilidad SHAP"
    p1.font.size = Pt(31)
    p1.font.bold = True
    p1.font.color.rgb = WHITE

    p2 = tf.add_paragraph()
    p2.space_before = Pt(14)
    p2.text = "Trabajo Final (TF1 — Semana 15)  |  Evolución Integral desde el Trabajo Parcial (TP1)"
    p2.font.size = Pt(17)
    p2.font.color.rgb = RGBColor(215, 228, 242)

    tb2 = s1.shapes.add_textbox(Inches(0.9), Inches(5.4), Inches(11.5), Inches(1.5))
    tf2 = tb2.text_frame
    p3 = tf2.paragraphs[0]
    p3.text = "Integrantes: Joaquin Basas Requena · Fabian Marcelo Rojas Cuadros · Cielo Luwidka Chavez Merino · Rodrigo Alonso Gamero Vera"
    p3.font.size = Pt(13)
    p3.font.color.rgb = WHITE
    p4 = tf2.add_paragraph()
    p4.space_before = Pt(6)
    p4.text = "Docente: Carlos Fernando Montoya Cubas   |   Ciclo 2026-20"
    p4.font.size = Pt(12)
    p4.font.color.rgb = RGBColor(190, 210, 230)

    # Slide 2: Problema, Alcance y Brechas del TP1
    s2 = prs.slides.add_slide(blank)
    _add_header(s2, prs, "1. Problema de Negocio y Punto de Partida (TP1)", "De un modelo preliminar en 0.50 a una solución operativa rentable", 2)
    _add_bullets(
        s2,
        Inches(0.55),
        Inches(1.35),
        Inches(6.2),
        Inches(5.3),
        [
            ("Problema", "26.54% de deserción en cartera Telco (7,043 contratos; IBM Sample Data Sets)."),
            ("Unidad de análisis", "Un cliente/contrato individual; objetivo de priorizar retención sin campañas masivas."),
            ("Logros del TP1", "Split anti-leakage 70/15/15, Random Forest superó baselines (PR-AUC 0.6698 vs 0.266 Dummy / 0.400 Heurístico)."),
            ("Brechas heredadas del TP1", "Recall en Test = 70.46% (por debajo de la meta >= 75%), umbral fijo en 0.50, sin CV ni tuning, sin segmentación de valor ni explicabilidad local."),
        ],
        font_size=13,
    )
    _add_kpi_card(s2, Inches(7.1), Inches(1.45), Inches(2.8), Inches(1.6), "PR-AUC Test (TF1)", "0.6719", "+0.0021 vs TP1")
    _add_kpi_card(s2, Inches(10.1), Inches(1.45), Inches(2.8), Inches(1.6), "Recall Test (τ*=0.20)", "81.14%", "+10.68 pp vs TP1")
    _add_kpi_card(s2, Inches(7.1), Inches(3.4), Inches(2.8), Inches(1.6), "Value-Weighted Recall", "84.12%", "+10.01 pp vs TP1")
    _add_kpi_card(s2, Inches(10.1), Inches(3.4), Inches(2.8), Inches(1.6), "Utilidad Simulada", "+$41,600", "2.03× vs TP1 ($20,500)")

    # Slide 3: Evolución TP1 -> TF1 y Arquitectura
    s3 = prs.slides.add_slide(blank)
    _add_header(s3, prs, "2. Evolución TP1 → TF1 y Arquitectura Anti-Leakage", "Test Holdout (1,057 filas) bloqueado durante todo el desarrollo", 3)
    evo_tbl = [
        ["Dimensión", "TP1 (Semana 7)", "TF1 (Semana 15)"],
        ["Variables", "19 atributos crudos", "+8 variables negocio (TelcoFeatureEngineer)"],
        ["Validación", "Split único 70/15/15", "StratifiedKFold (5/10 folds) en Train+Val (5,986)"],
        ["Modelamiento", "LR + RF base", "LR, RF, LightGBM, XGBoost (Optuna) + Stacking"],
        ["Umbral", "Fijo 0.50 (Recall 70.5%)", "OOF tau*=0.20 + Dinámicos (0.30/0.22/0.21)"],
        ["Técnica Adicional", "Ninguna", "K-Means (k=3) vs GMM/Ward + PCA 2D"],
        ["Explicabilidad y Uso", "Scripts CLI", "SHAP + Error Analysis + Streamlit + Pydantic + pytest"],
    ]
    _add_pptx_table(
        s3, Inches(0.55), Inches(1.35), Inches(12.2), Inches(4.4), evo_tbl,
        col_widths=[Inches(2.5), Inches(3.8), Inches(5.9)], font_size=11.5
    )
    arch_box = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.55), Inches(6.0), Inches(12.2), Inches(0.85))
    arch_box.fill.solid()
    arch_box.fill.fore_color.rgb = LIGHT_BG
    arch_box.line.color.rgb = NAVY
    p = arch_box.text_frame.paragraphs[0]
    p.text = "Flujo Operativo: Usuario → Streamlit UI → Validación Pydantic → TelcoFeatureEngineer + ColumnTransformer → Stacking (.joblib) + K-Means (.joblib) → Riesgo + Cluster + SHAP + Matriz 2×2"
    p.font.size = Pt(11.5)
    p.font.bold = True
    p.font.color.rgb = NAVY
    p.alignment = PP_ALIGN.CENTER

    # Slide 4: Matriz de Decisiones de Herramientas
    s4 = prs.slides.add_slide(blank)
    _add_header(s4, prs, "3. Matriz de Decisiones de Herramientas", "Selección fundamentada frente a alternativas técnicas", 4)
    tools_tbl = [
        ["Necesidad", "Herramienta elegida", "Alternativa considerada", "Justificación técnica"],
        ["EDA", "Pandas + Seaborn/Matplotlib", "Profiling automático", "Análisis guiado por preguntas de negocio e interpretación crítica"],
        ["Preparación / Contratos", "Pipeline + ColumnTransformer + Pydantic", "get_dummies manual", "Cero leakage en CV y validación de esquemas antes de inferencia"],
        ["Modelamiento", "Stacking (LR + LGBM + XGB) + RF", "Modelo único lineal/árbol", "Mejor PR-AUC (0.6732) y ROC-AUC (0.8515) combinando familias"],
        ["Experimentación", "Optuna + StratifiedKFold + MLflow", "GridSearch / split único", "Tuning bayesiano trazable y exportado a CSV portable"],
        ["Técnica Adicional", "K-Means (k=3) + PCA 2D", "GMM / Ward / cuantiles", "Mayor Silhouette en k accionable (0.4394) y .predict() online"],
        ["Interpretabilidad", "SHAP PermutationExplainer", "Importancia Gini", "Atribución aditiva global y local agnóstica al Stacking"],
        ["Despliegue y QA", "Streamlit + Plotly + pytest", "API sin interfaz / CLI", "Demo interactiva en 3 pestañas verificada automáticamente"],
    ]
    _add_pptx_table(
        s4, Inches(0.5), Inches(1.35), Inches(12.3), Inches(5.35), tools_tbl,
        col_widths=[Inches(2.3), Inches(3.4), Inches(2.4), Inches(4.2)], font_size=11
    )

    # Slide 5: Experimentación Sistemática
    s5 = prs.slides.add_slide(blank)
    _add_header(s5, prs, "4. Experimentación Sistemática: Optuna, CV 10-Fold y Wilcoxon", "Evaluación rigurosa sobre Train + Validation (5,986 filas)", 5)
    cv_tbl = [["Modelo (CV 10-Fold)", "PR-AUC (μ ± σ)", "ROC-AUC", "Brier"]]
    for _, r in data["cv10"].head(5).iterrows():
        cv_tbl.append([
            r["Modelo"].replace(" (Optuna)", ""),
            f"{r['PR-AUC_Mean']:.4f} ± {r['PR-AUC_Std']:.4f}",
            f"{r['ROC-AUC_Mean']:.4f}",
            f"{r['Brier Score_Mean']:.4f}",
        ])
    _add_pptx_table(
        s5, Inches(0.45), Inches(1.35), Inches(5.8), Inches(3.1), cv_tbl,
        col_widths=[Inches(2.4), Inches(1.6), Inches(0.9), Inches(0.9)], font_size=10
    )
    _add_bullets(
        s5,
        Inches(0.45),
        Inches(4.65),
        Inches(5.8),
        Inches(2.3),
        [
            ("Ablation FE", "Elevó LR (0.6566 → 0.6602) y redujo 47% la varianza de LGBM."),
            ("Wilcoxon (α=0.05)", "Stacking supera significativamente a RF (p=0.0020) y LR (p=0.0273)."),
            ("Empate en cima", "Frente a XGB (p=0.695) y LGBM (p=0.322) la diferencia no es significativa, pero Stacking lidera PR-AUC y ROC-AUC."),
        ],
        font_size=11,
    )
    _add_pptx_image(s5, FIGURES_TF1_DIR / "cv_comparacion_modelos_tf1.png", Inches(6.4), Inches(1.35), Inches(6.5), Inches(5.4))

    # Slide 6: Calibración y Umbrales
    s6 = prs.slides.add_slide(blank)
    _add_header(s6, prs, "5. Calibración OOF, Paradoja Bayesiana y Umbrales", "Evitar campañas indiscriminadas manteniendo alta rentabilidad", 6)
    _add_pptx_image(s6, FIGURES_TF1_DIR / "curva_umbral_utilidad.png", Inches(0.45), Inches(1.35), Inches(7.7), Inches(4.6))
    _add_bullets(
        s6,
        Inches(8.3),
        Inches(1.4),
        Inches(4.6),
        Inches(5.4),
        [
            ("Calibración OOF", "Stacking sin post-calibración logró el menor Brier Score (0.1333)."),
            ("Paradoja 500:50", "El óptimo Bayesiano irrestricto cae en τ=0.05, pero alertaría al 75.7% de toda la cartera (campaña masiva)."),
            ("Umbral Global τ*=0.20", "Optimizado en OOF sujeto a Recall >= 75% y Precision >= 50% (OOF Recall 84.5%, Precision 50.2%)."),
            ("Umbrales Dinámicos", "Diferenciados por valor: C0=0.30, C1=0.22, C2=0.21."),
        ],
        font_size=12,
    )

    # Slide 7: Evaluación Final Única en Test + Sensibilidad
    s7 = prs.slides.add_slide(blank)
    _add_header(s7, prs, "6. Evaluación Única en Test (n=1,057) y Sensibilidad Económica", "Dominancia consistente en los 3 escenarios financieros", 7)
    t_tbl = [["Modelo en Test", "PR-AUC", "Recall", "VW-Recall", "Precision", "Utilidad ($)"]]
    for _, r in data["test"].iterrows():
        t_tbl.append([
            r["Modelo"].replace(" (LR+LGBM+XGB)", "").replace(" (depth=10, umbral 0.50)", " (0.50)"),
            f"{r['PR-AUC']:.4f}",
            f"{r['Recall (Clase 1)']:.1%}",
            f"{r['Value-Weighted Recall']:.1%}",
            f"{r['Precision (Clase 1)']:.1%}",
            f"${r['Valor Económico Neto ($)']:,.0f}",
        ])
    _add_pptx_table(
        s7, Inches(0.45), Inches(1.3), Inches(7.3), Inches(3.25), t_tbl,
        col_widths=[Inches(3.2), Inches(0.8), Inches(0.8), Inches(0.9), Inches(0.8), Inches(0.8)], font_size=9.5
    )
    _add_pptx_image(s7, FIGURES_TF1_DIR / "sensibilidad_economica.png", Inches(7.9), Inches(1.3), Inches(5.0), Inches(5.4))
    _add_bullets(
        s7,
        Inches(0.45),
        Inches(4.75),
        Inches(7.3),
        Inches(2.1),
        [
            ("Salto cualitativo", "Con τ*=0.20 se capturan 228 de 281 fugas (+30 clientes vs TP1) y el 84.1% del ingreso mensual en riesgo."),
            ("Sensibilidad robusta", "Incluso con 30% de efectividad real o FP=$100, TF1 supera ampliamente al TP1 y ambos baselines."),
        ],
        font_size=11.5,
    )

    # Slide 8: Clustering + PCA + Matriz 2x2
    s8 = prs.slides.add_slide(blank)
    _add_header(s8, prs, "7. Técnica Adicional: K-Means (k=3) + PCA 2D", "Perfiles/proxies de valor para priorización comercial 2×2", 8)
    _add_pptx_image(s8, FIGURES_TF1_DIR / "clustering_pca_2d.png", Inches(0.45), Inches(1.35), Inches(6.9), Inches(4.7))
    p_tbl = [["Segmento", "% Base", "Cargo Mes", "Churn", "Acción si Alto Riesgo"]]
    actions_short = ["Digital bajo costo", "Soporte / migración", "VIP P1 personalizada"]
    for i, (_, r) in enumerate(data["profiles"].iterrows()):
        p_tbl.append([
            f"C{int(r['Cluster_ID'])} {r['Cluster_Label'].split('/')[0].strip()}",
            f"{r['Pct_Cartera']:.1%}",
            f"${r['MonthlyCharges_Medio']:.1f}",
            f"{r['Tasa_Churn_Observada']:.1%}",
            actions_short[i],
        ])
    _add_pptx_table(
        s8, Inches(7.5), Inches(1.4), Inches(5.4), Inches(2.6), p_tbl,
        col_widths=[Inches(1.4), Inches(0.8), Inches(0.95), Inches(0.75), Inches(1.5)], font_size=9.5
    )
    _add_bullets(
        s8,
        Inches(7.5),
        Inches(4.2),
        Inches(5.4),
        Inches(2.6),
        [
            ("Comparativa", "K-Means (k=3) lideró Silhouette (0.4394) frente a GMM (0.4006) y Ward (0.4154)."),
            ("PCA 2D", "Captura 96.17% de la varianza de valor y servicio."),
            ("Foco crítico", "C1 (Valor Medio) concentra 50.1% de fuga temprana; C2 (Alto Valor) protege $86.5/mes por cliente."),
        ],
        font_size=11,
    )

    # Slide 9: Interpretabilidad SHAP
    s9 = prs.slides.add_slide(blank)
    _add_header(s9, prs, "8. Interpretabilidad Global y Local con SHAP", "Atribución transparente sobre el ensamble Stacking (no causalidad)", 9)
    _add_pptx_image(s9, FIGURES_TF1_DIR / "shap_global_beeswarm.png", Inches(0.45), Inches(1.35), Inches(6.2), Inches(4.9))
    _add_pptx_image(s9, FIGURES_TF1_DIR / "shap_local_tp_critico.png", Inches(6.75), Inches(1.35), Inches(6.1), Inches(4.5))
    _add_bullets(
        s9,
        Inches(0.5),
        Inches(6.2),
        Inches(12.3),
        Inches(0.8),
        [
            ("Consistencia de negocio", "tenure, ContractRiskTier, Fibra Óptica, falta de Seguridad/Soporte y Cheque Electrónico dominan tanto el SHAP global como cada explicación local Waterfall."),
        ],
        font_size=11,
    )

    # Slide 10: Error Analysis
    s10 = prs.slides.add_slide(blank)
    _add_header(s10, prs, "9. Análisis Exhaustivo de Errores en Test (53 FN / 234 FP)", "Diagnóstico segmentado y distinción entre evidencia e hipótesis", 10)
    _add_pptx_image(s10, FIGURES_TF1_DIR / "error_analysis_segmentos.png", Inches(0.45), Inches(1.3), Inches(7.4), Inches(5.6))
    _add_bullets(
        s10,
        Inches(8.05),
        Inches(1.4),
        Inches(4.85),
        Inches(5.4),
        [
            ("Cobertura en riesgo alto", "Recall > 90% en Month-to-month (90.9%), Fibra (91.3%), 0-6m (91.1%) y Cluster C1 (94.1%)."),
            ("Evidencia en 53 FN", "58.5% tienen contratos de 1 o 2 años y antigüedad media de 31.0 meses (Recall 21.2% en One year, 0% en Two year)."),
            ("Hipótesis de FN", "Fugas ligadas a eventos no observables en datos estáticos (averías, soporte, fin de contrato)."),
            ("Evidencia en 234 FP", "91.5% Month-to-month y 72.6% sin soporte: perfil idéntico a desertores pero retenidos."),
        ],
        font_size=11.5,
    )

    # Slide 11: Despliegue en Streamlit (Demo en vivo)
    s11 = prs.slides.add_slide(blank)
    _add_header(s11, prs, "10. Despliegue Funcional: Aplicación Web en Streamlit", "Demostración interactiva conectada al pipeline serializado (2.5 min)", 11)
    _add_bullets(
        s11,
        Inches(0.6),
        Inches(1.45),
        Inches(12.1),
        Inches(5.3),
        [
            ("Pestaña 1 — Diagnóstico Individual", "Formulario validado por Pydantic v2, 3 perfiles preconfigurados, semáforo con política Global (0.20) o Dinámica por Cluster, Matriz 2×2 y SHAP Waterfall en vivo (<0.3s)."),
            ("Pestaña 2 — Scoring por Lotes (Batch CSV)", "Carga de cartera o muestra offline integrada (50 clientes), cálculo de dólares mensuales en riesgo, gráfico interactivo Plotly y descarga de CSV priorizado para Call Center."),
            ("Pestaña 3 — Simulador Financiero", "Sliders ejecutivos (Costo FN, Costo FP, Beneficio TP, % Efectividad de campaña y política de umbral) recalculando utilidad neta en tiempo real frente a Dummy, Heurístico y TP1."),
            ("Ejecución reproducible", "Comando local único: streamlit run app/streamlit_app.py (validado automáticamente con AppTest y pytest)."),
        ],
        font_size=13.5,
    )

    # Slide 12: Conclusiones, Limitaciones y Trabajo Futuro
    s12 = prs.slides.add_slide(blank)
    _add_header(s12, prs, "11. Conclusiones, Limitaciones y Trabajo Futuro", "Respuestas rigurosas sobre alcance real y riesgos de uso", 12)
    _add_bullets(
        s12,
        Inches(0.55),
        Inches(1.35),
        Inches(12.2),
        Inches(5.5),
        [
            ("Qué sí se concluye", "El sistema TF1 supera al TP1 y baselines en PR-AUC (0.6719), Recall (81.1%), rescate de ingresos (84.1%) y utilidad simulada (+$41,600) en los 3 escenarios económicos."),
            ("Qué NO se concluye", "No prueba causalidad de intervenciones ni representa utilidad contable auditada o CLV/RFM transaccional completo."),
            ("Limitaciones", "Dataset transversal sin series temporales de uso, tickets de soporte ni separación entre churn voluntario e involuntario; baja detección en contratos largos."),
            ("Riesgos fuera de contexto", "Covariate shift e invalidez del umbral 0.20 en otras operadoras/mercados; riesgo de uso punitivo sin distinguir morosidad."),
            ("Trabajo futuro", "Variables longitudinales / supervivencia, piloto A/B con grupo de control (uplift real) y monitoreo continuo de drift."),
        ],
        font_size=12.5,
    )

    # Slide 13: Contingencia Offline
    s13 = prs.slides.add_slide(blank)
    _add_header(s13, prs, "12. Plan de Contingencia Offline (Respaldo de Demo)", "Verificación reproducible independiente de red externa", 13)
    _add_bullets(
        s13,
        Inches(0.55),
        Inches(1.4),
        Inches(6.2),
        Inches(5.2),
        [
            ("Ejecución 100% local", "La app Streamlit carga modelos .joblib, muestra offline y fondo SHAP desde disco sin acceso a internet."),
            ("Suite automatizada", "6/6 pruebas pytest aprobadas (split reproducible, Feature Engineering, Pydantic, inferencia y AppTest)."),
            ("Caso respaldo P1 (VIP Riesgo)", "Fibra sin soporte, Month-to-month, tenure=4m, $98.5/mes -> P(Churn) alta -> Retención VIP inmediata."),
            ("Caso respaldo Consolidado", "Two year, DSL, soporte=Yes, tenure=58m -> P(Churn) baja -> Cross-selling sin descuento."),
        ],
        font_size=12,
    )
    _add_pptx_image(s13, FIGURES_TF1_DIR / "comparacion_tp1_vs_tf1_test.png", Inches(6.9), Inches(1.45), Inches(6.0), Inches(5.0))

    prs.save(str(out_path))


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_data()
    pdf_path = REPORTS_DIR / "Informe_TF1_Customer_Churn_Data_Mining_Tools.pdf"
    pptx_path = REPORTS_DIR / "TF1_Customer_Churn_Data_Mining_Tools.pptx"

    build_pdf(data, pdf_path)
    build_pptx(data, pptx_path)
    print(f"PDF generado ({pdf_path.stat().st_size:,} bytes): {pdf_path}")
    print(f"PPTX generado ({pptx_path.stat().st_size:,} bytes): {pptx_path}")


if __name__ == "__main__":
    main()
