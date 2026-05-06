"""Shared configuration and helpers for the UCI multicenter heart project."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_ZIP_DIR = RAW_DIR / "uci_zip"
ARTIFACT_DIR = BASE_DIR / "artifacts"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"
SHAP_DIR = OUTPUT_DIR / "shap"
UPLOAD_DIR = BASE_DIR / "web" / "uploads"

RAW_ZIP_PATH = RAW_DIR / "heart_disease_uci.zip"
RAW_MERGED_DATA_PATH = DATA_DIR / "heart_multicenter_merged.csv"
RAW_SAMPLE_DATA_PATH = BASE_DIR / "sample_upload.csv"
CLEANED_DATA_PATH = DATA_DIR / "heart_cleaned.csv"
PROCESSED_DATA_PATH = DATA_DIR / "heart_processed.csv"
PROCESSED_EXCEL_PATH = DATA_DIR / "heart_processed.xlsx"
PREPROCESSOR_PATH = ARTIFACT_DIR / "preprocessor.pkl"
MODEL_BUNDLE_PATH = ARTIFACT_DIR / "best_model_bundle.pkl"
MODEL_RESULTS_PATH = ARTIFACT_DIR / "model_results.json"
DATA_SUMMARY_PATH = REPORT_DIR / "data_processing_report.md"
MODEL_REPORT_PATH = REPORT_DIR / "model_validation_report.md"
SHAP_REPORT_PATH = REPORT_DIR / "shap_analysis_report.md"
RUNTIME_REPORT_PATH = REPORT_DIR / "web_runtime_report.md"
PROCESSING_METADATA_PATH = DATA_DIR / "processing_metadata.json"

RAW_COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]
RAW_CENTER_FILES = {
    "cleveland": RAW_ZIP_DIR / "processed.cleveland.data",
    "hungarian": RAW_ZIP_DIR / "processed.hungarian.data",
    "switzerland": RAW_ZIP_DIR / "processed.switzerland.data",
    "va": RAW_ZIP_DIR / "processed.va.data",
}

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
INPUT_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MISSING_INDICATOR_BASE_FEATURES = ["trestbps", "thalach", "oldpeak", "fbs", "exang", "slope", "ca", "thal"]
MISSING_INDICATOR_FEATURES = [f"{column}_missing" for column in MISSING_INDICATOR_BASE_FEATURES]
MODEL_NUMERIC_FEATURES = NUMERIC_FEATURES + MISSING_INDICATOR_FEATURES
MODEL_CATEGORICAL_FEATURES = CATEGORICAL_FEATURES
MODEL_FEATURES = MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES
TARGET_COLUMN = "target"
ORIGINAL_TARGET_COLUMN = "num"
CENTER_COLUMN = "center"

FEATURE_MEANINGS: Dict[str, str] = {
    "age": "年龄，单位为岁。",
    "sex": "性别，1 表示男性，0 表示女性。",
    "cp": "胸痛类型，1-4 表示不同胸痛模式。",
    "trestbps": "静息血压，单位为 mm Hg。",
    "chol": "血清胆固醇，单位为 mg/dl。",
    "fbs": "空腹血糖是否大于 120 mg/dl。",
    "restecg": "静息心电图结果。",
    "thalach": "达到的最大心率。",
    "exang": "运动诱发性心绞痛。",
    "oldpeak": "运动后 ST 压低程度。",
    "slope": "运动峰值 ST 段斜率。",
    "ca": "主要血管数目，来源于荧光透视检查。",
    "thal": "地中海贫血相关检查结果。",
    "trestbps_missing": "静息血压缺失标记，1 表示原始记录缺失。",
    "thalach_missing": "最大心率缺失标记，1 表示原始记录缺失。",
    "oldpeak_missing": "ST 压低缺失标记，1 表示原始记录缺失。",
    "fbs_missing": "空腹血糖缺失标记，1 表示原始记录缺失。",
    "exang_missing": "运动诱发性心绞痛缺失标记，1 表示原始记录缺失。",
    "slope_missing": "ST 斜率缺失标记，1 表示原始记录缺失。",
    "ca_missing": "主要血管数目缺失标记，1 表示原始记录缺失。",
    "thal_missing": "地中海贫血检查缺失标记，1 表示原始记录缺失。",
    "center": "样本所属数据中心，用于说明多中心来源。",
    "num": "UCI 原始标签，0 表示无心脏病，1-4 表示不同程度风险。",
    "target": "二分类目标变量，1 表示存在心脏病风险，0 表示无明显风险。",
}

DATA_SOURCE_DESCRIPTION = (
    "数据集来源于 UCI Machine Learning Repository 官方 Heart Disease 数据包，"
    "合并 Cleveland、Hungarian、Switzerland 和 VA Long Beach 四个中心的 processed 子集，"
    "共 920 条记录。项目将原始多分类标签 num 转换为二分类目标：num=0 记为 0，num>0 记为 1。"
)


@dataclass
class DatasetInfo:
    rows: int
    columns: int
    positive_count: int
    negative_count: int


def ensure_directories() -> None:
    """Create all required project directories if they do not exist."""
    for path in [DATA_DIR, RAW_DIR, ARTIFACT_DIR, OUTPUT_DIR, FIGURE_DIR, REPORT_DIR, SHAP_DIR, UPLOAD_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict, path: Path) -> None:
    """Save a Python dictionary as UTF-8 JSON with indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Dict:
    """Read UTF-8 JSON content from disk and return it as a dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))


def feature_description_markdown() -> str:
    """Render the original dataset feature dictionary as Markdown table rows."""
    display_keys = INPUT_FEATURES + [CENTER_COLUMN, ORIGINAL_TARGET_COLUMN, TARGET_COLUMN]
    lines: List[str] = ["| 特征 | 含义 |", "| --- | --- |"]
    for name in display_keys:
        lines.append(f"| {name} | {FEATURE_MEANINGS[name]} |")
    return "\n".join(lines)


def missing_indicator_description_markdown() -> str:
    """Render the internally added missingness indicators as Markdown rows."""
    lines: List[str] = ["| 衍生特征 | 含义 |", "| --- | --- |"]
    for name in MISSING_INDICATOR_FEATURES:
        lines.append(f"| {name} | {FEATURE_MEANINGS[name]} |")
    return "\n".join(lines)


def prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize schema and add internal missingness indicator features."""
    normalized = df.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]

    for column in INPUT_FEATURES:
        if column not in normalized.columns:
            raise KeyError(f"缺少必要字段: {column}")
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    if ORIGINAL_TARGET_COLUMN in normalized.columns:
        normalized[ORIGINAL_TARGET_COLUMN] = pd.to_numeric(normalized[ORIGINAL_TARGET_COLUMN], errors="coerce")
    if TARGET_COLUMN in normalized.columns:
        normalized[TARGET_COLUMN] = pd.to_numeric(normalized[TARGET_COLUMN], errors="coerce")

    for column in MISSING_INDICATOR_BASE_FEATURES:
        normalized[f"{column}_missing"] = normalized[column].isna().astype(int)

    return normalized
