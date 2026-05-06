"""Shared backend services for the Flask web application.

The service layer keeps model loading, file validation, prediction,
and SHAP payload generation separate from the HTTP routes.
"""

from __future__ import annotations

import copy
import threading
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
import shap
from werkzeug.utils import secure_filename

from common import (
    FEATURE_MEANINGS,
    INPUT_FEATURES,
    MODEL_BUNDLE_PATH,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    prepare_feature_frame,
)
from explainability_utils import (
    aggregate_global_importance,
    build_clinical_notes,
    build_local_explanation,
    build_original_feature_matrix,
    infer_original_feature_interactions,
)
from model_training import train_models

UPLOAD_DIR = Path("web/uploads")
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
MAX_PREVIEW_ROWS = 8
MAX_EXPLAIN_ROWS = 20
MAX_UPLOAD_ROWS = 500
_PREDICTION_CACHE: Dict[tuple[str, int], Dict[str, object]] = {}
_PREDICTION_INFLIGHT: Dict[tuple[str, int], threading.Event] = {}
_CACHE_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def get_model_bundle() -> dict:
    """Load and cache the best model bundle for repeated web requests."""
    if not MODEL_BUNDLE_PATH.exists():
        train_models()
    return joblib.load(MODEL_BUNDLE_PATH)


@lru_cache(maxsize=1)
def get_explainer() -> shap.TreeExplainer:
    """Create and cache a tree explainer from the best model."""
    bundle = get_model_bundle()
    model = bundle["pipeline"].named_steps["model"]
    return shap.TreeExplainer(model)


def allowed_file(filename: str) -> bool:
    """Check whether the uploaded file has a supported extension."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def get_project_metadata() -> Dict[str, object]:
    """Return frontend metadata used to render field descriptions."""
    bundle = get_model_bundle()
    return {
        "required_columns": bundle.get("input_feature_columns", INPUT_FEATURES),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_meanings": {
            key: FEATURE_MEANINGS[key]
            for key in bundle.get("input_feature_columns", INPUT_FEATURES)
            if key in FEATURE_MEANINGS
        },
        "allowed_extensions": sorted(ALLOWED_EXTENSIONS),
        "max_upload_rows": MAX_UPLOAD_ROWS,
    }


def save_uploaded_file(file_storage) -> Path:
    """Validate and persist an uploaded file using a safe generated name."""
    original_name = secure_filename(file_storage.filename or "")
    if not original_name or not allowed_file(original_name):
        raise ValueError("仅支持上传 CSV、XLSX、XLS 格式文件。")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    unique_name = f"{uuid.uuid4().hex}_{original_name}"
    save_path = UPLOAD_DIR / unique_name
    file_storage.save(save_path)
    return save_path


def read_tabular_file(path: Path) -> pd.DataFrame:
    """Read CSV or Excel data into a DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def validate_columns(df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
    """Ensure the uploaded file contains the columns required for prediction."""
    normalized_df = df.copy()
    normalized_df.columns = [column.strip() for column in normalized_df.columns]
    missing = [column for column in required_columns if column not in normalized_df.columns]
    if missing:
        raise ValueError(f"上传文件缺少必要字段: {missing}")
    return normalized_df[required_columns + ([TARGET_COLUMN] if TARGET_COLUMN in normalized_df.columns else [])]


def validate_row_count(df: pd.DataFrame) -> pd.DataFrame:
    """Keep uploads within a stable size for interactive SHAP inference."""
    if df.empty:
        raise ValueError("上传文件为空，请提供至少 1 行样本。")
    if len(df) > MAX_UPLOAD_ROWS:
        raise ValueError(f"单次在线预测最多支持 {MAX_UPLOAD_ROWS} 行样本，请拆分文件后重试。")
    return df


def preview_file(path: Path) -> Dict[str, object]:
    """Return a quick summary after upload so the frontend can show feedback."""
    bundle = get_model_bundle()
    df = validate_row_count(validate_columns(read_tabular_file(path), bundle.get("input_feature_columns", INPUT_FEATURES)))
    return {
        "file_id": path.name,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "preview": df.head(MAX_PREVIEW_ROWS).to_dict(orient="records"),
    }


def _prediction_cache_key(path: Path) -> tuple[str, int]:
    """Build a stable cache key from file name and mtime."""
    return path.name, path.stat().st_mtime_ns


def _compute_prediction_payload_uncached(path: Path) -> Dict[str, object]:
    """Run the full prediction pipeline without using the in-memory cache."""
    bundle = get_model_bundle()
    explainer = get_explainer()
    df = validate_row_count(validate_columns(read_tabular_file(path), bundle.get("input_feature_columns", INPUT_FEATURES)))
    raw_feature_df = df[bundle.get("input_feature_columns", INPUT_FEATURES)].copy()
    feature_df = prepare_feature_frame(raw_feature_df)[bundle["model_feature_columns"]].copy()
    pipeline = bundle["pipeline"]
    probs = pipeline.predict_proba(feature_df)[:, 1]
    preds = (probs >= bundle["threshold"]).astype(int)

    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(feature_df)
    feature_names = list(preprocessor.get_feature_names_out())
    transformed_df = pd.DataFrame(transformed, columns=feature_names)
    shap_values = explainer(transformed_df)
    shap_array = shap_values.values
    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, -1]
    original_shap_df = build_original_feature_matrix(shap_array, feature_names)

    original_importance = aggregate_global_importance(shap_array, feature_names)
    top_features = original_importance[:12]
    clinical_notes = build_clinical_notes(original_importance, top_n=5)
    interaction_summary = infer_original_feature_interactions(
        shap_array,
        feature_names,
        [str(item["feature"]) for item in original_importance[:5]],
        top_n=3,
    )

    local_explanations = []
    for row_index in range(min(MAX_EXPLAIN_ROWS, len(feature_df))):
        local_item = build_local_explanation(
            sample_index=row_index,
            shap_values=shap_array[row_index],
            feature_names=feature_names,
            raw_row=raw_feature_df.iloc[row_index],
            prediction=int(preds[row_index]),
            probability=float(probs[row_index]),
            top_n=5,
        )
        local_explanations.append(
            {
                "row_index": row_index,
                "prediction": int(preds[row_index]),
                "probability": round(float(probs[row_index]), 4),
                "contributions": local_item["top_contributions"],
                "summary_text": local_item["summary_text"],
                "prediction_label": local_item["prediction_label"],
            }
        )

    dependence_source = []
    for item in top_features[:3]:
        feature_name = str(item["feature"])
        if feature_name in original_shap_df.columns:
            dependence_source.append((feature_name, str(item["label"]), str(item["meaning"])))

    dependence_data = []
    for original_feature, label, meaning in dependence_source:
        dependence_data.append(
            {
                "feature": original_feature,
                "label": label,
                "meaning": meaning,
                "x": raw_feature_df[original_feature].round(4).tolist(),
                "y": original_shap_df[original_feature].round(6).tolist(),
            }
        )

    report_path = generate_report_file(
        path.name,
        probs,
        preds,
        local_explanations,
        top_features,
        interaction_summary,
        clinical_notes,
    )
    return {
        "file_id": path.name,
        "summary": {
            "total_rows": int(len(feature_df)),
            "positive_predictions": int(preds.sum()),
            "negative_predictions": int((preds == 0).sum()),
            "mean_probability": round(float(probs.mean()), 4),
            "model_name": bundle["model_name"],
            "threshold": bundle["threshold"],
        },
        "predictions": [
            {
                "row_index": int(index),
                "prediction": int(pred),
                "probability": round(float(prob), 4),
            }
            for index, (pred, prob) in enumerate(zip(preds, probs))
        ],
        "global_importance": [
            {
                "feature": item["feature"],
                "label": item["label"],
                "meaning": item["meaning"],
                "importance": item["importance"],
            }
            for item in top_features
        ],
        "local_explanations": local_explanations,
        "dependence_data": dependence_data,
        "interaction_summary": interaction_summary,
        "clinical_notes": clinical_notes,
        "report_url": f"/api/report/{report_path.name}",
    }


def compute_prediction_payload(file_id: str) -> Dict[str, object]:
    """Run prediction and SHAP analysis for an uploaded file."""
    path = UPLOAD_DIR / secure_filename(file_id)
    if not path.exists():
        raise FileNotFoundError("找不到上传文件，请重新上传。")

    cache_key = _prediction_cache_key(path)

    with _CACHE_LOCK:
        cached = _PREDICTION_CACHE.get(cache_key)
        if cached is not None:
            return copy.deepcopy(cached)

        inflight_event = _PREDICTION_INFLIGHT.get(cache_key)
        if inflight_event is None:
            inflight_event = threading.Event()
            _PREDICTION_INFLIGHT[cache_key] = inflight_event
            is_owner = True
        else:
            is_owner = False

    if not is_owner:
        inflight_event.wait(timeout=60)
        with _CACHE_LOCK:
            cached = _PREDICTION_CACHE.get(cache_key)
        if cached is None:
            raise RuntimeError("预测缓存同步失败，请重试。")
        return copy.deepcopy(cached)

    try:
        payload = _compute_prediction_payload_uncached(path)
        with _CACHE_LOCK:
            _PREDICTION_CACHE[cache_key] = payload
        return copy.deepcopy(payload)
    finally:
        with _CACHE_LOCK:
            event = _PREDICTION_INFLIGHT.pop(cache_key, None)
            if event is not None:
                event.set()


def generate_report_file(file_id: str, probs, preds, local_explanations, top_features, interaction_summary, clinical_notes) -> Path:
    """Create a Markdown report that can be downloaded from the frontend."""
    report_path = UPLOAD_DIR / f"{Path(file_id).stem}_prediction_report.md"
    lines = [
        "# 在线预测 SHAP 报告",
        "",
        f"源文件: {file_id}",
        f"总样本数: {len(preds)}",
        f"阳性预测数: {int(preds.sum())}",
        f"平均预测概率: {round(float(probs.mean()), 4)}",
        "",
        "## Top 全局解释特征",
    ]
    for item in top_features[:8]:
        lines.append(f"- {item['label']} ({item['feature']}): {item['importance']}")
    lines.append("")
    lines.append("## 医学一致性提示")
    for note in clinical_notes:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## 前3个样本局部解释")
    for item in local_explanations[:3]:
        lines.append(f"- {item['summary_text']}")
        for contribution in item["contributions"][:5]:
            lines.append(
                f"  - {contribution['label']}({contribution['raw_value']}): SHAP={contribution['shap_value']}, 方向={contribution['direction']}"
            )
    lines.append("")
    lines.append("## 主要交互效应")
    for item in interaction_summary:
        lines.append(
            f"- {item['feature_a_label']} 与 {item['feature_b_label']} 的交互强度为 {item['score']}"
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path

