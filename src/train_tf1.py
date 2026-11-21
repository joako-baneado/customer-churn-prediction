"""
Módulo de Experimentación Sistemática para el TF1 (CC209).
Ejecuta exclusivamente sobre Train + Validation (5,986 filas):
1. Ablation Study de Feature Engineering (CV 5-fold).
2. Optimización Bayesiana de hiperparámetros con Optuna (4 familias).
3. Ensamble por Stacking Heterogéneo (LR + LGBM + XGB).
4. Comparación justa en StratifiedKFold (k=10) con mu +- sigma y test de Wilcoxon.
5. Registro trazable en MLflow local y exportación portable a CSV.

El conjunto Test (1,057 filas) NO se toca en este módulo.
"""

import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from scipy.stats import ttest_rel, wilcoxon
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import (
    CV_FOLDS_EVAL,
    CV_FOLDS_OPTUNA,
    FIGURES_TF1_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    OPTUNA_N_TRIALS,
    RANDOM_STATE,
    REPORTS_DIR,
)
from src.evaluation import BusinessRuleBaseline, evaluate_predictions
from src.preprocessing import build_preprocessor_tf1, get_dev_and_test_splits

optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_estimator(family: str, params: dict):
    """Instancia el clasificador base según familia e hiperparámetros."""
    if family == "logistic_regression":
        return LogisticRegression(
            C=params["C"],
            l1_ratio=params["l1_ratio"],
            solver="saga",
            class_weight=params.get("class_weight", "balanced"),
            max_iter=2500,
            random_state=RANDOM_STATE,
        )
    if family == "random_forest":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            class_weight=params.get("class_weight", "balanced"),
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if family == "lightgbm":
        return LGBMClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"],
            max_depth=params["max_depth"],
            min_child_samples=params["min_child_samples"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            scale_pos_weight=params["scale_pos_weight"],
            subsample_freq=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
    if family == "xgboost":
        return XGBClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            min_child_weight=params["min_child_weight"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            scale_pos_weight=params["scale_pos_weight"],
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
    raise ValueError(f"Familia desconocida: {family}")


def build_model_pipeline(
    family: str,
    params: dict | None = None,
    use_feature_engineering: bool = True,
    stacking_base_params: dict | None = None,
) -> Pipeline:
    """Construye un Pipeline end-to-end limpio para entrenamiento CV o final."""
    preprocessor = build_preprocessor_tf1(use_feature_engineering=use_feature_engineering)

    if family == "stacking":
        if not stacking_base_params:
            raise ValueError("stacking_base_params requerido para StackingClassifier")
        estimators = [
            ("lr", make_estimator("logistic_regression", stacking_base_params["logistic_regression"])),
            ("lgbm", make_estimator("lightgbm", stacking_base_params["lightgbm"])),
            ("xgb", make_estimator("xgboost", stacking_base_params["xgboost"])),
        ]
        meta = LogisticRegression(C=1.0, max_iter=1500, random_state=RANDOM_STATE)
        clf = StackingClassifier(
            estimators=estimators,
            final_estimator=meta,
            cv=5,
            stack_method="predict_proba",
            n_jobs=1,
        )
        return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])

    clf = make_estimator(family, params or {})
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])


def _cross_val_pr_auc(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> float:
    scores = []
    for tr_idx, val_idx in cv.split(X, y):
        pipe = clone(pipeline)
        X_tr, X_va = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[val_idx]
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_va)[:, 1]
        scores.append(average_precision_score(y_va, proba))
    return float(np.mean(scores))


def run_ablation_study(X_dev: pd.DataFrame, y_dev: pd.Series) -> pd.DataFrame:
    """Compara modelos base con y sin Feature Engineering en CV 5-fold."""
    cv5 = StratifiedKFold(n_splits=CV_FOLDS_OPTUNA, shuffle=True, random_state=RANDOM_STATE)
    configs = [
        ("LogisticRegression", "logistic_regression", {"C": 1.0, "l1_ratio": 0.0, "class_weight": "balanced"}),
        ("LightGBM", "lightgbm", {
            "n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31,
            "max_depth": -1, "min_child_samples": 20, "subsample": 0.8,
            "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0,
            "scale_pos_weight": 1.0,
        }),
    ]
    rows = []
    for label, fam, params in configs:
        for use_fe in (False, True):
            pipe = build_model_pipeline(fam, params=params, use_feature_engineering=use_fe)
            fold_scores = []
            for tr_idx, val_idx in cv5.split(X_dev, y_dev):
                p = clone(pipe)
                p.fit(X_dev.iloc[tr_idx], y_dev.iloc[tr_idx])
                y_pr = p.predict_proba(X_dev.iloc[val_idx])[:, 1]
                fold_scores.append(average_precision_score(y_dev.iloc[val_idx], y_pr))
            rows.append(
                {
                    "Modelo_Base": label,
                    "Feature_Engineering_TF1": use_fe,
                    "PR_AUC_Media": float(np.mean(fold_scores)),
                    "PR_AUC_Std": float(np.std(fold_scores, ddof=1)),
                }
            )
    df_ablation = pd.DataFrame(rows)
    df_ablation.to_csv(REPORTS_DIR / "ablation_feature_engineering_tf1.csv", index=False)
    return df_ablation


def tune_family_optuna(
    family: str,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    n_trials: int = OPTUNA_N_TRIALS,
) -> tuple[dict, float, pd.DataFrame]:
    """Ejecuta optimización bayesiana con Optuna (CV 5-fold, métrica PR-AUC)."""
    cv5 = StratifiedKFold(n_splits=CV_FOLDS_OPTUNA, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        if family == "logistic_regression":
            params = {
                "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
                "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
            }
        elif family == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 120, 400, step=40),
                "max_depth": trial.suggest_int("max_depth", 4, 16),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
                "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None]),
            }
        elif family == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 450, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
                "subsample": trial.suggest_float("subsample", 0.65, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", [1.0, 2.77]),
            }
        elif family == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 450, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 7),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
                "subsample": trial.suggest_float("subsample", 0.65, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", [1.0, 2.77]),
            }
        else:
            raise ValueError(family)

        pipe = build_model_pipeline(family, params=params, use_feature_engineering=True)
        return _cross_val_pr_auc(pipe, X_dev, y_dev, cv5)

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=f"optuna_{family}")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    trials_df = study.trials_dataframe()[["number", "value", "state"]].copy()
    trials_df["familia"] = family
    trials_df["best_so_far"] = trials_df["value"].cummax()
    return study.best_params, float(study.best_value), trials_df


def evaluate_candidates_cv10(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    best_params_map: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evalúa los 2 baselines + 5 modelos TF1 en los mismos 10 folds estratificados."""
    cv10 = StratifiedKFold(n_splits=CV_FOLDS_EVAL, shuffle=True, random_state=RANDOM_STATE)

    model_specs = {
        "Baseline Trivial (Dummy)": lambda: DummyClassifier(strategy="most_frequent"),
        "Baseline Heurístico (Mes-a-Mes & Tenure <= 6)": lambda: BusinessRuleBaseline(tenure_threshold=6),
        "Regresión Logística (Optuna)": lambda: build_model_pipeline(
            "logistic_regression", params=best_params_map["logistic_regression"]
        ),
        "Random Forest (Optuna)": lambda: build_model_pipeline(
            "random_forest", params=best_params_map["random_forest"]
        ),
        "LightGBM (Optuna)": lambda: build_model_pipeline(
            "lightgbm", params=best_params_map["lightgbm"]
        ),
        "XGBoost (Optuna)": lambda: build_model_pipeline(
            "xgboost", params=best_params_map["xgboost"]
        ),
        "Stacking Heterogéneo (LR+LGBM+XGB)": lambda: build_model_pipeline(
            "stacking", stacking_base_params=best_params_map
        ),
    }

    fold_records = []
    for fold_idx, (tr_idx, val_idx) in enumerate(cv10.split(X_dev, y_dev), start=1):
        X_tr, X_va = X_dev.iloc[tr_idx], X_dev.iloc[val_idx]
        y_tr, y_va = y_dev.iloc[tr_idx], y_dev.iloc[val_idx]
        mc_va = X_va["MonthlyCharges"].values

        for model_name, builder in model_specs.items():
            est = builder()
            est.fit(X_tr, y_tr)
            y_pred = est.predict(X_va)
            y_proba = est.predict_proba(X_va)[:, 1]
            metrics = evaluate_predictions(
                y_true=y_va,
                y_pred=y_pred,
                y_proba=y_proba,
                monthly_charges=mc_va,
                model_name=model_name,
            )
            metrics["Fold"] = fold_idx
            fold_records.append(metrics)

    df_folds = pd.DataFrame(fold_records)
    df_folds.to_csv(REPORTS_DIR / "cv_fold_scores_tf1.csv", index=False)

    metric_cols = [
        "PR-AUC",
        "ROC-AUC",
        "Recall (Clase 1)",
        "Value-Weighted Recall",
        "Precision (Clase 1)",
        "F1-Score",
        "Brier Score",
        "Log-Loss",
        "Valor Económico Neto ($)",
    ]
    summary_rows = []
    for model_name in model_specs.keys():
        sub = df_folds[df_folds["Modelo"] == model_name]
        row = {"Modelo": model_name}
        for col in metric_cols:
            row[f"{col}_Mean"] = float(sub[col].mean())
            row[f"{col}_Std"] = float(sub[col].std(ddof=1))
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows).sort_values("PR-AUC_Mean", ascending=False)
    df_summary.to_csv(REPORTS_DIR / "comparacion_modelos_cv_tf1.csv", index=False)
    return df_summary, df_folds


def run_statistical_tests(df_summary: pd.DataFrame, df_folds: pd.DataFrame) -> pd.DataFrame:
    """Compara mediante Wilcoxon pareado y t-test pareado el Top-1 contra otros candidatos."""
    ml_models = df_summary[~df_summary["Modelo"].str.startswith("Baseline")]["Modelo"].tolist()
    top1 = ml_models[0]
    top1_scores = df_folds[df_folds["Modelo"] == top1].sort_values("Fold")["PR-AUC"].values

    test_rows = []
    for competitor in ml_models[1:]:
        comp_scores = df_folds[df_folds["Modelo"] == competitor].sort_values("Fold")["PR-AUC"].values
        diff = top1_scores - comp_scores
        w_stat, w_pval = wilcoxon(diff, zero_method="wilcox", alternative="two-sided") if np.any(diff != 0) else (0.0, 1.0)
        t_stat, t_pval = ttest_rel(top1_scores, comp_scores)
        test_rows.append(
            {
                "Modelo_Referencia": top1,
                "Modelo_Comparado": competitor,
                "Delta_PR_AUC_Media": float(np.mean(diff)),
                "Wilcoxon_Stat": float(w_stat),
                "Wilcoxon_p_value": float(w_pval),
                "TTest_Stat": float(t_stat),
                "TTest_p_value": float(t_pval),
                "Significativo_alpha_0.05": bool(w_pval < 0.05),
            }
        )

    df_tests = pd.DataFrame(test_rows)
    df_tests.to_csv(REPORTS_DIR / "wilcoxon_cv_tf1.csv", index=False)
    return df_tests


def log_experiments_mlflow(
    df_ablation: pd.DataFrame,
    best_params_map: dict,
    df_summary: pd.DataFrame,
):
    """Registra todos los candidatos de CV en MLflow local y exporta CSV portable."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    family_lookup = {
        "Regresión Logística (Optuna)": "logistic_regression",
        "Random Forest (Optuna)": "random_forest",
        "LightGBM (Optuna)": "lightgbm",
        "XGBoost (Optuna)": "xgboost",
    }

    for _, row in df_ablation.iterrows():
        run_name = f"ablation_{row['Modelo_Base']}_FE_{row['Feature_Engineering_TF1']}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("stage", "ablation_5fold")
            mlflow.log_param("modelo_base", row["Modelo_Base"])
            mlflow.log_param("feature_engineering", bool(row["Feature_Engineering_TF1"]))
            mlflow.log_metric("cv5_pr_auc_mean", float(row["PR_AUC_Media"]))
            mlflow.log_metric("cv5_pr_auc_std", float(row["PR_AUC_Std"]))

    for _, row in df_summary.iterrows():
        name = row["Modelo"]
        with mlflow.start_run(run_name=f"cv10_{name}"):
            mlflow.set_tag("stage", "cv10_comparative")
            mlflow.log_param("modelo", name)
            mlflow.log_param("cv_folds", CV_FOLDS_EVAL)
            if name in family_lookup:
                for p_k, p_v in best_params_map[family_lookup[name]].items():
                    mlflow.log_param(p_k, p_v)
            mlflow.log_metric("pr_auc_mean", float(row["PR-AUC_Mean"]))
            mlflow.log_metric("pr_auc_std", float(row["PR-AUC_Std"]))
            mlflow.log_metric("roc_auc_mean", float(row["ROC-AUC_Mean"]))
            mlflow.log_metric("recall_mean", float(row["Recall (Clase 1)_Mean"]))
            mlflow.log_metric("value_weighted_recall_mean", float(row["Value-Weighted Recall_Mean"]))
            mlflow.log_metric("brier_score_mean", float(row["Brier Score_Mean"]))
            mlflow.log_metric("valor_economico_mean", float(row["Valor Económico Neto ($)_Mean"]))

    exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    runs_df = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    runs_df.to_csv(REPORTS_DIR / "mlflow_experimentos_tf1.csv", index=False)


def generate_modeling_figures(df_summary: pd.DataFrame, df_folds: pd.DataFrame, df_trials: pd.DataFrame):
    FIGURES_TF1_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Comparación CV 10-fold (PR-AUC y ROC-AUC)
    order = df_summary["Modelo"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    sns.boxplot(data=df_folds, y="Modelo", x="PR-AUC", order=order, ax=axes[0], palette="viridis")
    axes[0].set_title("Distribución PR-AUC en StratifiedKFold (k=10)")
    axes[0].set_xlabel("PR-AUC (↑ mejor)")
    axes[0].set_ylabel("")

    sns.boxplot(data=df_folds, y="Modelo", x="ROC-AUC", order=order, ax=axes[1], palette="magma")
    axes[1].set_title("Distribución ROC-AUC en StratifiedKFold (k=10)")
    axes[1].set_xlabel("ROC-AUC (↑ mejor)")
    axes[1].set_ylabel("")

    plt.tight_layout()
    fig.savefig(FIGURES_TF1_DIR / "cv_comparacion_modelos_tf1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Convergencia de Optuna
    fig, ax = plt.subplots(figsize=(10, 5))
    labels_map = {
        "logistic_regression": "Regresión Logística",
        "random_forest": "Random Forest",
        "lightgbm": "LightGBM",
        "xgboost": "XGBoost",
    }
    for fam, grp in df_trials.groupby("familia"):
        ax.plot(grp["number"] + 1, grp["best_so_far"], marker="o", markersize=4, linewidth=2, label=labels_map.get(fam, fam))
    ax.set_title("Convergencia de Optimización Bayesiana (Optuna, CV 5-fold)")
    ax.set_xlabel("Número de Trial")
    ax.set_ylabel("Mejor PR-AUC acumulado (CV 5-fold)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_TF1_DIR / "optuna_convergencia_tf1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_training_experiments():
    print("=" * 72)
    print("FASE 3 TF1 - EXPERIMENTACIÓN SISTEMÁTICA (SOLO TRAIN + VALIDATION)")
    print("=" * 72)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_TF1_DIR.mkdir(parents=True, exist_ok=True)

    X_dev, _, y_dev, _ = get_dev_and_test_splits()
    print(f"[1/5] Conjunto de desarrollo (Train+Val): {len(X_dev)} filas | Churn: {y_dev.mean():.2%}")

    # 1. Ablation Feature Engineering
    df_ablation = run_ablation_study(X_dev, y_dev)
    print("\n[2/5] Ablation Feature Engineering (CV 5-fold):")
    print(df_ablation.to_string(index=False))

    # 2. Optuna tuning por familia
    print(f"\n[3/5] Optimización Optuna ({OPTUNA_N_TRIALS} trials/familia, CV 5-fold)...")
    best_params_map = {}
    optuna_scores = {}
    trial_dfs = []
    for fam in ["logistic_regression", "random_forest", "lightgbm", "xgboost"]:
        params, score, t_df = tune_family_optuna(fam, X_dev, y_dev, n_trials=OPTUNA_N_TRIALS)
        best_params_map[fam] = params
        optuna_scores[fam] = score
        trial_dfs.append(t_df)
        print(f"  - {fam:20s}: mejor PR-AUC (CV5) = {score:.4f}")

    df_trials = pd.concat(trial_dfs, ignore_index=True)
    df_trials.to_csv(REPORTS_DIR / "optuna_trials_tf1.csv", index=False)

    # 3. Evaluación justa en StratifiedKFold k=10
    print(f"\n[4/5] Comparativa en StratifiedKFold (k={CV_FOLDS_EVAL}) + Stacking...")
    df_summary, df_folds = evaluate_candidates_cv10(X_dev, y_dev, best_params_map)
    cols_show = [
        "Modelo", "PR-AUC_Mean", "PR-AUC_Std", "ROC-AUC_Mean",
        "Recall (Clase 1)_Mean", "Brier Score_Mean", "Valor Económico Neto ($)_Mean"
    ]
    print(df_summary[cols_show].to_string(index=False))

    # 4. Pruebas estadísticas pareadas
    df_tests = run_statistical_tests(df_summary, df_folds)
    print("\n[5/5] Pruebas de significancia estadística frente al Top-1:")
    print(df_tests.to_string(index=False))

    # Guardar configuración ganadora e hiperparámetros
    top_ml_name = df_summary[~df_summary["Modelo"].str.startswith("Baseline")].iloc[0]["Modelo"]
    top_ml_pr = float(df_summary[~df_summary["Modelo"].str.startswith("Baseline")].iloc[0]["PR-AUC_Mean"])

    model_name_to_family = {
        "Regresión Logística (Optuna)": "logistic_regression",
        "Random Forest (Optuna)": "random_forest",
        "LightGBM (Optuna)": "lightgbm",
        "XGBoost (Optuna)": "xgboost",
        "Stacking Heterogéneo (LR+LGBM+XGB)": "stacking",
    }

    best_config = {
        "winner_cv_name": top_ml_name,
        "winner_cv_family": model_name_to_family[top_ml_name],
        "winner_cv_pr_auc_mean": top_ml_pr,
        "optuna_cv5_scores": optuna_scores,
        "best_params_by_family": best_params_map,
    }
    with open(MODELS_DIR / "best_params_tf1.json", "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)

    # 5. Tracking MLflow + gráficos
    log_experiments_mlflow(df_ablation, best_params_map, df_summary)
    generate_modeling_figures(df_summary, df_folds, df_trials)
    print(f"\nCandidato ganador en CV 10-fold: {top_ml_name} (PR-AUC = {top_ml_pr:.4f})")
    print("Resultados exportados en reports/ y mlflow_experimentos_tf1.csv")


if __name__ == "__main__":
    run_training_experiments()
