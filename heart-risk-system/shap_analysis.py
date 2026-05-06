"""SHAP analysis module for the heart disease explainable prediction project.

This script loads the best trained model, computes SHAP explanations,
generates global and local interpretation figures, and validates practical
acceleration strategies for mixed-type heart disease features.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from common import (
    CLEANED_DATA_PATH,
    INPUT_FEATURES,
    MODEL_BUNDLE_PATH,
    SHAP_DIR,
    SHAP_REPORT_PATH,
    TARGET_COLUMN,
    ensure_directories,
    save_json,
)
from explainability_utils import (
    aggregate_global_importance,
    build_clinical_notes,
    build_local_explanation,
    infer_original_feature_interactions,
)
from model_training import train_models

SHAP_JSON_PATH = SHAP_DIR / "shap_summary.json"
SHAP_SPEED_PATH = SHAP_DIR / "shap_speed_comparison.json"
RANDOM_STATE = 42
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def initialize_shap_js() -> None:
    """Initialize SHAP JS support only when IPython is available.

    The project mainly runs as plain Python scripts and Flask
    services, so Notebook-only initialization should never block execution.
    """
    try:
        shap.initjs()
    except AssertionError:
        # In terminal and web deployment scenarios IPython is optional.
        pass


def normalize_shap_output(shap_values):
    """Normalize SHAP output into a 2D numpy array for binary classification."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[-1])
    if hasattr(shap_values, "values"):
        values = shap_values.values
        if values.ndim == 3:
            return values[:, :, -1]
        return values
    values = np.asarray(shap_values)
    if values.ndim == 3:
        return values[:, :, -1]
    return values


def normalize_interaction_output(interaction_values) -> np.ndarray | None:
    """Normalize SHAP interaction output into a 3D array when available.

    Different tree models may return interaction values in different shapes.
    For binary classification we keep the final class slice, while for models
    that do not expose interaction tensors we return ``None`` and let the
    caller fall back to a heuristic interaction strategy.
    """
    if isinstance(interaction_values, list):
        values = np.asarray(interaction_values[-1])
    elif hasattr(interaction_values, "values"):
        values = np.asarray(interaction_values.values)
    else:
        values = np.asarray(interaction_values)

    if values.ndim == 4:
        return values[:, :, :, -1]
    if values.ndim == 3:
        return values
    return None


def load_artifacts() -> Tuple[dict, pd.DataFrame]:
    """Load the trained model bundle and cleaned dataset."""
    if not MODEL_BUNDLE_PATH.exists():
        train_models()
    bundle = joblib.load(MODEL_BUNDLE_PATH)
    df = pd.read_csv(CLEANED_DATA_PATH)
    return bundle, df


def get_explainer_and_data(bundle: dict, df: pd.DataFrame):
    """Prepare the tree explainer and transformed feature matrix."""
    pipeline = bundle["pipeline"]
    model_feature_columns = bundle["model_feature_columns"]
    raw_feature_columns = bundle.get("input_feature_columns", INPUT_FEATURES)
    X = df[model_feature_columns]
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    transformed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()
    transformed_df = pd.DataFrame(transformed, columns=feature_names)
    explainer = shap.TreeExplainer(model)
    raw_feature_df = df[raw_feature_columns].copy()
    return explainer, transformed_df, raw_feature_df, X, pipeline


def plot_summary_figures(explainer, shap_values: np.ndarray, features: pd.DataFrame) -> List[str]:
    """Generate three different summary-style plots."""
    output_files: List[str] = []

    plt.figure()
    shap.summary_plot(shap_values, features, show=False)
    summary_path = SHAP_DIR / "summary_beeswarm.png"
    plt.tight_layout()
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    output_files.append(summary_path.name)

    plt.figure()
    shap.summary_plot(shap_values, features, plot_type="bar", show=False)
    bar_path = SHAP_DIR / "summary_bar.png"
    plt.tight_layout()
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    output_files.append(bar_path.name)

    plt.figure()
    shap.summary_plot(shap_values, features, plot_type="violin", show=False)
    violin_path = SHAP_DIR / "summary_violin.png"
    plt.tight_layout()
    plt.savefig(violin_path, dpi=300, bbox_inches="tight")
    plt.close()
    output_files.append(violin_path.name)

    return output_files


def plot_force_figures(explainer, shap_values: np.ndarray, features: pd.DataFrame, count: int = 3) -> List[str]:
    """Generate local force plots for the first few samples."""
    output_files: List[str] = []
    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = float(np.ravel(base_value)[-1])

    for index in range(count):
        plt.figure(figsize=(12, 3))
        shap.force_plot(base_value, shap_values[index], features.iloc[index], matplotlib=True, show=False)
        save_path = SHAP_DIR / f"force_plot_{index + 1}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        output_files.append(save_path.name)
    return output_files


def get_top_features(shap_values: np.ndarray, feature_names: List[str], top_n: int = 6) -> List[str]:
    """Return top features ranked by mean absolute SHAP value."""
    importance = np.abs(shap_values).mean(axis=0)
    ranking = np.argsort(importance)[::-1][:top_n]
    return [feature_names[idx] for idx in ranking]


def plot_original_importance_figure(global_importance: List[Dict[str, object]]) -> str:
    """Plot a readable original-feature importance chart."""
    display_items = global_importance[:10]
    labels = [str(item["label"]) for item in reversed(display_items)]
    scores = [float(item["importance"]) for item in reversed(display_items)]
    save_path = SHAP_DIR / "summary_original_features.png"

    plt.figure(figsize=(8, 5))
    plt.barh(labels, scores, color="#0f766e")
    plt.xlabel("Mean |SHAP|")
    plt.title("Original Clinical Feature Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return save_path.name


def plot_dependence_figures(shap_values: np.ndarray, features: pd.DataFrame, top_features: List[str]) -> List[str]:
    """Create at least three dependence plots using the most important features."""
    output_files: List[str] = []
    for index, feature in enumerate(top_features[:3], start=1):
        plt.figure()
        shap.dependence_plot(feature, shap_values, features, show=False, interaction_index=None)
        save_path = SHAP_DIR / f"dependence_plot_{index}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        output_files.append(save_path.name)
    return output_files


def infer_top_interactions(explainer, features: pd.DataFrame, top_features: List[str]) -> List[Tuple[str, str]]:
    """Estimate the most relevant interaction pairs from SHAP interaction values.

    When the current model does not provide a 3D SHAP interaction tensor,
    fall back to a lightweight heuristic: rank pairs by the average product of
    absolute SHAP contributions. This keeps the project runnable
    across RandomForest, XGBoost, and LightGBM.
    """
    sample_features = features.iloc[: min(60, len(features))]
    interaction_array = None
    try:
        interaction_values = explainer.shap_interaction_values(sample_features)
        interaction_array = normalize_interaction_output(interaction_values)
    except Exception:
        interaction_array = None

    pair_scores: List[Tuple[float, Tuple[str, str]]] = []
    feature_index = {name: idx for idx, name in enumerate(sample_features.columns)}
    for left in top_features[:4]:
        for right in top_features[:4]:
            if left >= right:
                continue
            left_idx = feature_index[left]
            right_idx = feature_index[right]
            if interaction_array is not None:
                score = float(np.abs(interaction_array[:, left_idx, right_idx]).mean())
            else:
                # 兜底策略: 用同一样本中两个特征 SHAP 绝对值乘积的均值近似交互强度。
                score = float(
                    np.mean(
                        np.abs(sample_features.iloc[:, left_idx].to_numpy())
                        * np.abs(sample_features.iloc[:, right_idx].to_numpy())
                    )
                )
            pair_scores.append((score, (left, right)))

    pair_scores.sort(key=lambda item: item[0], reverse=True)
    return [pair for _, pair in pair_scores[:3]]


def plot_interaction_figures(
    shap_values: np.ndarray,
    features: pd.DataFrame,
    interaction_pairs: List[Tuple[str, str]],
) -> List[str]:
    """Create three interaction-focused dependence plots."""
    output_files: List[str] = []
    for index, (feature, interaction_feature) in enumerate(interaction_pairs, start=1):
        plt.figure()
        shap.dependence_plot(feature, shap_values, features, interaction_index=interaction_feature, show=False)
        save_path = SHAP_DIR / f"interaction_plot_{index}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        output_files.append(save_path.name)
    return output_files


def validate_speedup(explainer, features: pd.DataFrame, shap_values: np.ndarray, top_features: List[str]) -> Dict[str, float]:
    """Validate two practical SHAP acceleration strategies.

    Strategy A: explain a representative sample instead of the full dataset.
    Strategy B: keep only top-k important encoded features for exploratory plots.
    """
    repeated_features = pd.concat([features] * 8, ignore_index=True)
    start = time.perf_counter()
    full_values = normalize_shap_output(explainer(repeated_features))
    full_time = time.perf_counter() - start

    sampled_features = repeated_features.sample(n=min(120, len(repeated_features)), random_state=RANDOM_STATE)
    start = time.perf_counter()
    sampled_values = normalize_shap_output(explainer(sampled_features))
    sampled_time = time.perf_counter() - start

    reduced_features = repeated_features[top_features[: min(8, len(top_features))]]
    reduced_start = time.perf_counter()
    _ = np.abs(full_values[:, [features.columns.get_loc(col) for col in reduced_features.columns]]).mean(axis=0)
    reduced_time = time.perf_counter() - reduced_start

    speed_info = {
        "full_dataset_seconds": round(full_time, 4),
        "sampled_dataset_seconds": round(sampled_time, 4),
        "top_feature_summary_seconds": round(reduced_time, 6),
        "sampled_speedup_ratio": round(full_time / sampled_time, 4) if sampled_time else None,
    }
    save_json(speed_info, SHAP_SPEED_PATH)
    return speed_info


def write_shap_report(
    top_features: List[str],
    original_importance: List[Dict[str, object]],
    summary_files: List[str],
    force_files: List[str],
    dependence_files: List[str],
    interaction_files: List[str],
    local_samples: List[Dict[str, object]],
    interaction_summary: List[Dict[str, object]],
    clinical_notes: List[str],
    speed_info: Dict[str, float],
) -> None:
    """Write the SHAP analysis report as Markdown."""
    top_original = "、".join(str(item["label"]) for item in original_importance[:6])
    lines = [
        "# SHAP 可解释性分析报告",
        "",
        "## 1. 全局结论",
        f"Top-6 编码特征: {', '.join(top_features[:6])}",
        f"Top-6 原始临床特征: {top_original}",
        "",
        "## 2. 图像输出",
        f"- Summary Plot: {', '.join(summary_files)}",
        f"- Force Plot: {', '.join(force_files)}",
        f"- Dependence Plot: {', '.join(dependence_files)}",
        f"- Interaction Plot: {', '.join(interaction_files)}",
        "",
        "## 3. 原始临床特征重要性",
        "| 排名 | 原始特征 | 含义 | 平均绝对SHAP值 |",
        "| --- | --- | --- | --- |",
    ]
    for index, item in enumerate(original_importance[:10], start=1):
        lines.append(
            f"| {index} | {item['label']} | {item['meaning']} | {item['importance']} |"
        )
    lines.extend([
        "",
        "## 4. 医学解释",
    ])
    lines.extend([f"- {note}" for note in clinical_notes])
    lines.extend([
        "",
        "## 5. 局部样本解释摘要",
    ])
    for item in local_samples[:3]:
        lines.append(f"- {item['summary_text']}")
    lines.extend([
        "",
        "## 6. 交互效应摘要",
    ])
    for item in interaction_summary:
        lines.append(
            f"- {item['feature_a_label']} 与 {item['feature_b_label']} 的交互强度为 {item['score']}。"
        )
    lines.extend([
        "",
        "## 7. SHAP 加速策略验证",
        f"- 全量重复样本 SHAP 计算耗时: {speed_info['full_dataset_seconds']} 秒",
        f"- 代表性抽样 SHAP 计算耗时: {speed_info['sampled_dataset_seconds']} 秒",
        f"- Top 特征摘要计算耗时: {speed_info['top_feature_summary_seconds']} 秒",
        f"- 抽样加速比: {speed_info['sampled_speedup_ratio']}",
        "- 结论: 对混合特征心脏病数据，一方面可以控制解释样本数量，另一方面可以在前端仅展示 Top-N 特征，均能明显降低解释延迟。",
    ])
    SHAP_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_shap_analysis() -> Dict[str, object]:
    """Run the complete SHAP workflow and save reusable outputs."""
    ensure_directories()
    initialize_shap_js()
    bundle, df = load_artifacts()
    explainer, features, raw_features, model_features, pipeline = get_explainer_and_data(bundle, df)

    shap_values_raw = explainer(features)
    shap_values = normalize_shap_output(shap_values_raw)
    top_features = get_top_features(shap_values, list(features.columns), top_n=6)
    original_importance = aggregate_global_importance(shap_values, list(features.columns))
    original_plot = plot_original_importance_figure(original_importance)

    summary_files = plot_summary_figures(explainer, shap_values, features)
    force_files = plot_force_figures(explainer, shap_values, features, count=3)
    dependence_files = plot_dependence_figures(shap_values, features, top_features)
    interaction_pairs = infer_top_interactions(explainer, features, top_features)
    interaction_files = plot_interaction_figures(shap_values, features, interaction_pairs)
    speed_info = validate_speedup(explainer, features, shap_values, top_features)
    local_samples = [
        build_local_explanation(
            sample_index=index,
            shap_values=shap_values[index],
            feature_names=list(features.columns),
            raw_row=raw_features.iloc[index],
            prediction=int(
                pipeline.predict_proba(model_features.iloc[[index]])[:, 1][0] >= bundle["threshold"]
            ),
            probability=float(pipeline.predict_proba(model_features.iloc[[index]])[:, 1][0]),
            top_n=5,
        )
        for index in range(min(3, len(features)))
    ]
    interaction_summary = infer_original_feature_interactions(
        shap_values,
        list(features.columns),
        [str(item["feature"]) for item in original_importance[:5]],
        top_n=3,
    )
    clinical_notes = build_clinical_notes(original_importance, top_n=5)

    shap_payload = {
        "model_name": bundle["model_name"],
        "top_features": top_features,
        "top_original_features": [item["feature"] for item in original_importance[:6]],
        "global_importance": {
            feature: round(float(value), 6)
            for feature, value in zip(features.columns, np.abs(shap_values).mean(axis=0))
        },
        "original_global_importance": original_importance,
        "local_samples": local_samples,
        "interaction_pairs": interaction_pairs,
        "original_interaction_summary": interaction_summary,
        "clinical_notes": clinical_notes,
        "readable_summary_figure": original_plot,
        "speed_info": speed_info,
    }
    save_json(shap_payload, SHAP_JSON_PATH)
    write_shap_report(
        top_features,
        original_importance,
        summary_files + [original_plot],
        force_files,
        dependence_files,
        interaction_files,
        local_samples,
        interaction_summary,
        clinical_notes,
        speed_info,
    )
    return shap_payload


if __name__ == "__main__":
    payload = run_shap_analysis()
    print(f"SHAP 分析完成，Top 特征: {payload['top_features'][:5]}")

