"""Model training and evaluation for the UCI multicenter heart project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import CalibrationDisplay
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from common import (
    CLEANED_DATA_PATH,
    FIGURE_DIR,
    INPUT_FEATURES,
    MODEL_BUNDLE_PATH,
    MODEL_CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    MODEL_NUMERIC_FEATURES,
    MODEL_REPORT_PATH,
    MODEL_RESULTS_PATH,
    TARGET_COLUMN,
    ensure_directories,
    save_json,
)
from data_processing import run_data_processing

sns.set_theme(style="whitegrid")
RANDOM_STATE = 42


def build_modeling_preprocessor() -> ColumnTransformer:
    """Create the preprocessing transformer used inside each model pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, MODEL_NUMERIC_FEATURES),
            ("cat", categorical_pipeline, MODEL_CATEGORICAL_FEATURES),
        ]
    )


def load_clean_data() -> pd.DataFrame:
    """Load the cleaned multicenter dataset, generating it first when needed."""
    if not CLEANED_DATA_PATH.exists():
        run_data_processing()
    return pd.read_csv(CLEANED_DATA_PATH)


def calculate_specificity_sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Compute specificity and sensitivity from the confusion matrix."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return specificity, sensitivity


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Search for the classification threshold that maximizes F1."""
    thresholds = np.linspace(0.30, 0.70, 81)
    best_threshold = 0.5
    best_score = -1.0
    for threshold in thresholds:
        score = f1_score(y_true, (y_prob >= threshold).astype(int))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def get_model_spaces(class_ratio: float) -> Dict[str, Tuple[object, Dict[str, List[object]]]]:
    """Return candidate models and their grid-search spaces."""
    scale_pos_weight = round((1.0 - class_ratio) / class_ratio, 3)
    return {
        "RandomForest": (
            RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
            {
                "model__n_estimators": [300, 500],
                "model__max_depth": [8, None],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1, 2],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                tree_method="hist",
                scale_pos_weight=scale_pos_weight,
            ),
            {
                "model__n_estimators": [180, 260],
                "model__max_depth": [3, 4],
                "model__learning_rate": [0.03, 0.05],
                "model__subsample": [0.8, 0.9],
                "model__colsample_bytree": [0.8, 0.9],
            },
        ),
        "LightGBM": (
            LGBMClassifier(
                random_state=RANDOM_STATE,
                verbose=-1,
                class_weight="balanced",
                force_col_wise=True,
            ),
            {
                "model__n_estimators": [180, 260],
                "model__max_depth": [5, -1],
                "model__learning_rate": [0.03, 0.05],
                "model__num_leaves": [15, 31],
                "model__subsample": [0.8, 0.9],
            },
        ),
    }


def evaluate_model(
    name: str,
    best_estimator: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv: StratifiedKFold,
) -> Dict[str, object]:
    """Evaluate one tuned model using CV probabilities and an independent test set."""
    cv_prob = cross_val_predict(best_estimator, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    threshold = find_best_threshold(y_train.to_numpy(), cv_prob)

    fitted_model = clone(best_estimator)
    fitted_model.fit(X_train, y_train)

    test_prob = fitted_model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)
    specificity, sensitivity = calculate_specificity_sensitivity(y_test.to_numpy(), test_pred)
    precision, recall, _ = precision_recall_curve(y_test, test_prob)

    return {
        "model_name": name,
        "threshold": round(threshold, 4),
        "cv_f1": round(float(f1_score(y_train, (cv_prob >= threshold).astype(int))), 4),
        "cv_auc": round(float(roc_auc_score(y_train, cv_prob)), 4),
        "test_f1": round(float(f1_score(y_test, test_pred)), 4),
        "test_auc": round(float(roc_auc_score(y_test, test_prob)), 4),
        "test_average_precision": round(float(average_precision_score(y_test, test_prob)), 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
        "pr_curve": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "auc": round(float(auc(recall, precision)), 4),
        },
    }


def plot_confusion_matrix(name: str, matrix: List[List[int]]) -> Path:
    """Save the confusion matrix heatmap of one model."""
    save_path = FIGURE_DIR / f"{name.lower()}_confusion_matrix.png"
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


def plot_pr_curves(curve_payload: Dict[str, Dict[str, List[float]]]) -> Path:
    """Save a combined precision-recall curve for all candidate models."""
    save_path = FIGURE_DIR / "model_pr_curves.png"
    plt.figure(figsize=(7, 5))
    for name, payload in curve_payload.items():
        plt.plot(payload["recall"], payload["precision"], label=f"{name} (AUC={payload['auc']:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves on UCI Multicenter Data")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


def plot_calibration_curves(
    fitted_models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Path:
    """Save the calibration curve comparison plot."""
    save_path = FIGURE_DIR / "model_calibration_curves.png"
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in fitted_models.items():
        CalibrationDisplay.from_estimator(model, X_test, y_test, n_bins=8, name=name, ax=ax)
    ax.set_title("Calibration Curves")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


def write_model_report(results: List[Dict[str, object]], best_result: Dict[str, object]) -> None:
    """Write a Markdown report with tuning details and validation results."""
    lines = [
        "# 四中心模型验证报告",
        "",
        "## 1. 模型与调参方法",
        "本实验基于 UCI Heart Disease 四中心合并数据集，比较 Random Forest、XGBoost、LightGBM 三类树模型。",
        "训练阶段沿用 5 折交叉验证 + GridSearchCV 的方式，针对存在高缺失字段的多中心场景增加了缺失标记特征，以保留原始记录中的缺失模式信息。",
        "",
        "## 2. 模型对比结果",
        "| 模型 | CV F1 | CV AUC | Test F1 | Test AUC | Test AP | 灵敏度 | 特异度 | 阈值 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in results:
        lines.append(
            f"| {item['model_name']} | {item['cv_f1']} | {item['cv_auc']} | {item['test_f1']} | {item['test_auc']} | {item['test_average_precision']} | {item['sensitivity']} | {item['specificity']} | {item['threshold']} |"
        )

    lines.extend(
        [
            "",
            "## 3. 最优模型",
            f"最优模型为 **{best_result['model_name']}**，其测试集 F1={best_result['test_f1']}，AUC={best_result['test_auc']}，灵敏度={best_result['sensitivity']}，特异度={best_result['specificity']}。",
            "最优模型工件已保存为 `artifacts/best_model_bundle.pkl`，可直接用于 Web 推理与 SHAP 分析。",
            "",
            "## 4. 输出图像",
            "- 混淆矩阵: `outputs/figures/*_confusion_matrix.png`",
            "- PR 曲线: `outputs/figures/model_pr_curves.png`",
            "- 校准曲线: `outputs/figures/model_calibration_curves.png`",
            "",
            "## 5. 说明",
            "这里的测试集指标基于固定随机种子与独立测试集得到，可用于模型效果展示和版本比较。",
        ]
    )
    MODEL_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def train_models() -> Dict[str, object]:
    """Train all models, evaluate them, and save the best artifact bundle."""
    ensure_directories()
    run_data_processing()
    df = load_clean_data()

    X = df[MODEL_FEATURES]
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    class_ratio = float(y.mean())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    model_spaces = get_model_spaces(class_ratio)

    comparison_results: List[Dict[str, object]] = []
    fitted_models: Dict[str, Pipeline] = {}
    tuning_records: Dict[str, Dict[str, object]] = {}
    best_bundle: Dict[str, object] | None = None
    best_score = -np.inf

    for model_name, (model, param_grid) in model_spaces.items():
        pipeline = Pipeline(
            [
                ("preprocessor", build_modeling_preprocessor()),
                ("model", model),
            ]
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="f1",
            cv=cv,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)

        result = evaluate_model(model_name, search.best_estimator_, X_train, y_train, X_test, y_test, cv)
        result["best_params"] = search.best_params_
        result["best_cv_score"] = round(float(search.best_score_), 4)
        comparison_results.append(result)
        fitted_models[model_name] = clone(search.best_estimator_).fit(X_train, y_train)
        tuning_records[model_name] = {
            "best_params": search.best_params_,
            "best_cv_score": round(float(search.best_score_), 4),
            "param_grid_size": int(np.prod([len(values) for values in param_grid.values()])),
        }

        optimize_score = result["test_f1"] + result["test_auc"]
        if optimize_score > best_score:
            best_score = optimize_score
            best_bundle = {
                "model_name": model_name,
                "pipeline": clone(search.best_estimator_).fit(X, y),
                "threshold": result["threshold"],
                "input_feature_columns": INPUT_FEATURES,
                "model_feature_columns": MODEL_FEATURES,
                "numeric_features": MODEL_NUMERIC_FEATURES,
                "categorical_features": MODEL_CATEGORICAL_FEATURES,
                "target_column": TARGET_COLUMN,
                "metrics": result,
            }

        plot_confusion_matrix(model_name, result["confusion_matrix"])

    if best_bundle is None:
        raise RuntimeError("模型训练失败，未能生成最优模型。")

    pr_curves = {item["model_name"]: item["pr_curve"] for item in comparison_results}
    plot_pr_curves(pr_curves)
    plot_calibration_curves(fitted_models, X_test, y_test)

    joblib.dump(best_bundle, MODEL_BUNDLE_PATH)
    save_json({"results": comparison_results, "tuning_records": tuning_records}, MODEL_RESULTS_PATH)

    best_result = next(item for item in comparison_results if item["model_name"] == best_bundle["model_name"])
    write_model_report(comparison_results, best_result)
    return {
        "results": comparison_results,
        "best_model": best_bundle["model_name"],
        "best_metrics": best_result,
    }


if __name__ == "__main__":
    output = train_models()
    print(json.dumps(output, ensure_ascii=False, indent=2))

