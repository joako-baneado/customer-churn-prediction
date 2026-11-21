"""
Suite de pruebas automatizadas para el Proyecto Integrador TF1 (CC209).
Verifica reproducibilidad de partición, ingeniería de características sin NaN/inf,
contratos Pydantic, métricas ponderadas por valor, inferencia end-to-end y Streamlit.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.clustering import ValueClusterSegmenter  # noqa: F401
from src.config import ENGINEERED_NUMERICAL_FEATURES, MODELS_DIR
from src.evaluation import evaluate_economic_scenarios, value_weighted_recall
from src.explainability import explain_single_customer
from src.features import TelcoFeatureEngineer
from src.preprocessing import get_dev_and_test_splits, load_raw_dataset, split_data
from src.schemas import CustomerInput, validate_batch_dataframe


def _sample_customer_dict(**overrides) -> dict:
    base = {
        "gender": "Female",
        "SeniorCitizen": "0",
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": 1026.0,
    }
    base.update(overrides)
    return base


def test_reproducible_split():
    df = load_raw_dataset()
    assert len(df) == 7043
    assert df["TotalCharges"].isna().sum() == 0

    X_tr1, X_val1, X_te1, y_tr1, y_val1, y_te1 = split_data(df)
    X_tr2, X_val2, X_te2, y_tr2, y_val2, y_te2 = split_data(df)

    assert (len(X_tr1), len(X_val1), len(X_te1)) == (4930, 1056, 1057)
    assert np.isclose(y_tr1.mean(), 0.2653, atol=1e-3)
    assert np.isclose(y_val1.mean(), 0.2652, atol=1e-3)
    assert np.isclose(y_te1.mean(), 0.2658, atol=1e-3)
    assert X_te1.index.equals(X_te2.index)
    assert y_tr1.equals(y_tr2)


def test_feature_engineer_no_nans_and_bounds():
    df_sample = pd.DataFrame(
        [
            _sample_customer_dict(tenure=0, TotalCharges=0.0),
            _sample_customer_dict(tenure=45, MonthlyCharges=99.0, TotalCharges=4455.0),
        ]
    )
    fe = TelcoFeatureEngineer(enabled=True)
    out = fe.fit_transform(df_sample)

    num_mat = out[ENGINEERED_NUMERICAL_FEATURES].to_numpy(dtype=float)
    assert not np.isnan(num_mat).any()
    assert not np.isinf(num_mat).any()
    assert out.loc[0, "BillingRatio"] == pytest.approx(1.0)
    assert out.loc[0, "TenureBucket"] == "0-6m"
    assert out.loc[1, "TenureBucket"] == "25-48m"
    assert (out["TotalServicesSubscribed"] >= 0).all()
    assert (out["TotalServicesSubscribed"] <= 8).all()


def test_pydantic_schema_validation():
    valid = CustomerInput(**_sample_customer_dict(SeniorCitizen=1))
    assert valid.SeniorCitizen == "1"

    new_zero = CustomerInput(**_sample_customer_dict(tenure=0, TotalCharges=None))
    assert new_zero.TotalCharges == 0.0

    with pytest.raises(ValueError):
        CustomerInput(**_sample_customer_dict(tenure=-1))

    with pytest.raises(ValueError):
        CustomerInput(**_sample_customer_dict(MonthlyCharges=0.0))

    with pytest.raises(ValueError):
        CustomerInput(**_sample_customer_dict(tenure=0, TotalCharges=150.0))

    with pytest.raises(ValueError):
        CustomerInput(**_sample_customer_dict(InternetService="No", TechSupport="Yes"))

    df_batch = pd.DataFrame(
        [
            {"customerID": "C-001", **_sample_customer_dict()},
            {"customerID": "C-002", **_sample_customer_dict(tenure=-5)},
            {"customerID": "C-003", **_sample_customer_dict(Contract="Two year")},
        ]
    )
    df_valid, errors = validate_batch_dataframe(df_batch)
    assert len(df_valid) == 2
    assert len(errors) == 1
    assert df_valid["customerID"].tolist() == ["C-001", "C-003"]


def test_evaluation_value_weighted_recall_and_scenarios():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    charges = np.array([100.0, 20.0, 50.0, 30.0])

    vwr = value_weighted_recall(y_true, y_pred, charges)
    assert vwr == pytest.approx(100.0 / 120.0)

    scenarios = evaluate_economic_scenarios(y_true, y_pred, model_name="TestModel")
    assert len(scenarios) == 3
    assert {s["Escenario_Key"] for s in scenarios} == {
        "base",
        "pesimista_30pct",
        "agresivo_fp100",
    }


def test_inference_artifacts_end_to_end():
    model = joblib.load(MODELS_DIR / "pipeline_tf1_final.joblib")
    segmenter = joblib.load(MODELS_DIR / "clustering_tf1.joblib")
    with open(MODELS_DIR / "metadata_tf1.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # 1. Inferencia individual validada
    row_df = CustomerInput(**_sample_customer_dict()).to_dataframe()
    proba_1 = model.predict_proba(row_df)[:, 1]
    assert proba_1.shape == (1,)
    assert 0.0 <= float(proba_1[0]) <= 1.0

    cluster_id = segmenter.predict_cluster_ids(row_df)
    assert int(cluster_id[0]) in {0, 1, 2}
    assert str(int(cluster_id[0])) in metadata["dynamic_thresholds_by_cluster"]

    # 2. Explicación SHAP individual
    sv = explain_single_customer(model, row_df)
    assert sv.values.shape == (52,)
    assert np.isfinite(sv.values).all()

    # 3. Inferencia en lote sobre muestra de Test
    _, X_test, _, _ = get_dev_and_test_splits()
    batch = X_test.head(20)
    probas_batch = model.predict_proba(batch)[:, 1]
    assert probas_batch.shape == (20,)
    assert np.all((probas_batch >= 0.0) & (probas_batch <= 1.0))
    assert len(segmenter.predict_labels(batch)) == 20


def test_streamlit_app_smoke():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(BASE_DIR / "app" / "streamlit_app.py"), default_timeout=30)
    at.run()
    assert not at.exception
