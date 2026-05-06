"""Data processing module for the UCI multicenter heart disease project.

The script merges the four official processed center files from UCI,
converts the original multi-class target into a binary diagnosis task,
handles missing values and outliers, creates internal missingness flags,
and exports reproducible cleaned and encoded datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (
    CATEGORICAL_FEATURES,
    CENTER_COLUMN,
    CLEANED_DATA_PATH,
    DATA_SOURCE_DESCRIPTION,
    DATA_SUMMARY_PATH,
    FEATURE_MEANINGS,
    INPUT_FEATURES,
    MODEL_CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    MODEL_NUMERIC_FEATURES,
    MISSING_INDICATOR_BASE_FEATURES,
    ORIGINAL_TARGET_COLUMN,
    PREPROCESSOR_PATH,
    PROCESSING_METADATA_PATH,
    PROCESSED_DATA_PATH,
    PROCESSED_EXCEL_PATH,
    RAW_CENTER_FILES,
    RAW_COLUMN_NAMES,
    RAW_MERGED_DATA_PATH,
    RAW_SAMPLE_DATA_PATH,
    TARGET_COLUMN,
    ensure_directories,
    feature_description_markdown,
    missing_indicator_description_markdown,
    prepare_feature_frame,
    save_json,
)


def load_multicenter_dataset() -> pd.DataFrame:
    """Read and merge the four official UCI processed center files."""
    frames: List[pd.DataFrame] = []
    for center_name, path in RAW_CENTER_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"未找到四中心原始文件: {path}")
        center_df = pd.read_csv(
            path,
            header=None,
            names=RAW_COLUMN_NAMES,
            na_values="?",
        )
        center_df[CENTER_COLUMN] = center_name
        frames.append(center_df)

    merged_df = pd.concat(frames, ignore_index=True)
    merged_df[ORIGINAL_TARGET_COLUMN] = pd.to_numeric(merged_df[ORIGINAL_TARGET_COLUMN], errors="coerce")
    merged_df[TARGET_COLUMN] = (merged_df[ORIGINAL_TARGET_COLUMN] > 0).astype(int)
    merged_df = prepare_feature_frame(merged_df)
    return merged_df


def create_missing_value_report(df: pd.DataFrame) -> Dict[str, int]:
    """Return the count of missing values for every original input field."""
    return {column: int(df[column].isna().sum()) for column in INPUT_FEATURES}


def impute_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Impute original clinical fields while keeping missing flags unchanged."""
    imputed_df = df.copy()
    strategies: Dict[str, str] = {}

    for column in MODEL_NUMERIC_FEATURES:
        if column.endswith("_missing"):
            strategies[column] = "缺失标记保留"
            continue
        imputed_df[column] = imputed_df[column].fillna(imputed_df[column].median())
        strategies[column] = "中位数填充"

    for column in MODEL_CATEGORICAL_FEATURES:
        mode_value = imputed_df[column].mode(dropna=True)
        fill_value = mode_value.iloc[0] if not mode_value.empty else 0
        imputed_df[column] = imputed_df[column].fillna(fill_value)
        strategies[column] = "众数填充"

    strategies[ORIGINAL_TARGET_COLUMN] = "原始标签保留"
    strategies[TARGET_COLUMN] = "二分类标签保留"
    strategies[CENTER_COLUMN] = "中心来源保留"
    return imputed_df, strategies


def remove_outliers_iqr(df: pd.DataFrame, iqr_multiplier: float = 3.0) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Remove numeric outliers on original continuous variables only.

    For the multicenter UCI data, a slightly wider IQR range is more stable
    than the default 1.5*IQR because the four centers have明显不同的取值分布。
    """
    filtered_df = df.copy()
    valid_mask = pd.Series(True, index=filtered_df.index)
    outlier_summary: Dict[str, int] = {}

    for column in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
        q1 = filtered_df[column].quantile(0.25)
        q3 = filtered_df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        column_mask = filtered_df[column].between(lower, upper, inclusive="both")
        outlier_summary[column] = int((~column_mask).sum())
        valid_mask &= column_mask

    filtered_df = filtered_df.loc[valid_mask].reset_index(drop=True)
    return filtered_df, outlier_summary


def build_preprocessor() -> ColumnTransformer:
    """Create the training and inference preprocessor for model features."""
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


def encode_features(df: pd.DataFrame, preprocessor: ColumnTransformer) -> pd.DataFrame:
    """Apply scaling and one-hot encoding to the cleaned model input."""
    encoded = preprocessor.fit_transform(df[MODEL_FEATURES])
    encoded_columns = preprocessor.get_feature_names_out()
    encoded_df = pd.DataFrame(encoded, columns=encoded_columns, index=df.index)
    encoded_df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return encoded_df


def create_sample_upload_file(df: pd.DataFrame, sample_size: int = 200) -> pd.DataFrame:
    """Create a compact sample file used by tests and web upload examples."""
    feature_df = df[INPUT_FEATURES + [TARGET_COLUMN]].copy()
    if len(feature_df) <= sample_size:
        return feature_df

    sampled_df, _ = train_test_split(
        feature_df,
        train_size=sample_size,
        random_state=42,
        stratify=feature_df[TARGET_COLUMN],
    )
    return sampled_df.reset_index(drop=True)


def write_processing_report(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    missing_report: Dict[str, int],
    impute_strategy: Dict[str, str],
    outlier_summary: Dict[str, int],
    encoded_df: pd.DataFrame,
) -> None:
    """Generate a Markdown report that explains the multicenter processing flow."""
    center_counts = raw_df[CENTER_COLUMN].value_counts().sort_index().to_dict()
    original_target_counts = raw_df[ORIGINAL_TARGET_COLUMN].value_counts().sort_index().to_dict()
    binary_target_counts = cleaned_df[TARGET_COLUMN].value_counts().sort_index().to_dict()
    lines: List[str] = [
        "# UCI 四中心心脏病数据处理说明文档",
        "",
        "## 1. 数据集来源",
        DATA_SOURCE_DESCRIPTION,
        "",
        "原始文件列表：",
    ]
    for center_name, path in RAW_CENTER_FILES.items():
        lines.append(f"- {center_name}: `{path.name}`")

    lines.extend(
        [
            "",
            f"四中心合并样本数: {len(raw_df)}",
            f"清洗后样本数: {len(cleaned_df)}",
            f"编码后特征数: {encoded_df.shape[1] - 1}",
            "",
            "## 2. 四中心样本分布",
            "| 中心 | 样本数 |",
            "| --- | --- |",
        ]
    )
    for center_name, count in center_counts.items():
        lines.append(f"| {center_name} | {count} |")

    lines.extend(
        [
            "",
            "## 3. 字段含义",
            feature_description_markdown(),
            "",
            "## 4. 内部缺失标记特征",
            missing_indicator_description_markdown(),
            "",
            "## 5. 标签转换规则",
            "UCI 原始标签 `num` 为 0-4 的多分类编码。本文统一转换为二分类：`num=0 -> target=0`，`num>0 -> target=1`。",
            "",
            f"原始多分类分布: {original_target_counts}",
            f"二分类分布(清洗后): {binary_target_counts}",
            "",
            "## 6. 数据处理步骤",
            "1. 读取 UCI 官方四个 processed 中心文件并按字段顺序合并。",
            "2. 记录中心来源 `center`，同时将原始多分类标签 `num` 转换为二分类目标 `target`。",
            "3. 针对高缺失字段生成缺失标记特征，如 `ca_missing`、`thal_missing`，保留缺失模式信息。",
            "4. 对连续变量使用中位数填充，对分类型变量使用众数填充。",
            "5. 对连续变量采用 3.0×IQR 方法剔除异常值，兼顾多中心差异与极端值控制。",
            "6. 对分类型变量执行 One-Hot 编码，对连续变量和缺失标记变量执行标准化。",
            "7. 输出清洗后的宽表、编码后的训练表以及 Web 应用 使用的示例样本。",
            "",
            "## 7. 缺失值统计",
            "| 字段 | 缺失值数量 | 处理策略 |",
            "| --- | --- | --- |",
        ]
    )
    for column in INPUT_FEATURES:
        lines.append(f"| {column} | {missing_report[column]} | {impute_strategy[column]} |")

    lines.extend(
        [
            "",
            "## 8. 异常值处理统计",
            "| 连续特征 | IQR识别到的异常值数量 |",
            "| --- | --- |",
        ]
    )
    for column, count in outlier_summary.items():
        lines.append(f"| {column} | {count} |")

    lines.extend(
        [
            "",
            "## 9. 输出文件",
            "- ??????: `data/heart_multicenter_merged.csv`",
            "- ?????: `data/heart_cleaned.csv`",
            "- ??? CSV: `data/heart_processed.csv`",
            "- ??? Excel: `data/heart_processed.xlsx`",
            "- Web ????: `sample_upload.csv`",
            "- ????: `artifacts/preprocessor.pkl`",
        ]
    )
    DATA_SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_data_processing() -> Dict[str, object]:
    """Execute the full UCI multicenter preprocessing workflow."""
    ensure_directories()

    raw_df = load_multicenter_dataset()
    raw_df.to_csv(RAW_MERGED_DATA_PATH, index=False, encoding="utf-8-sig")

    missing_report = create_missing_value_report(raw_df)
    imputed_df, impute_strategy = impute_missing_values(raw_df)
    cleaned_df, outlier_summary = remove_outliers_iqr(imputed_df, iqr_multiplier=3.0)

    preprocessor = build_preprocessor()
    encoded_df = encode_features(cleaned_df, preprocessor)
    sample_df = create_sample_upload_file(raw_df)

    cleaned_df.to_csv(CLEANED_DATA_PATH, index=False, encoding="utf-8-sig")
    encoded_df.to_csv(PROCESSED_DATA_PATH, index=False, encoding="utf-8-sig")
    encoded_df.to_excel(PROCESSED_EXCEL_PATH, index=False)
    sample_df.to_csv(RAW_SAMPLE_DATA_PATH, index=False, encoding="utf-8-sig")
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    metadata = {
        "raw_shape": list(raw_df.shape),
        "cleaned_shape": list(cleaned_df.shape),
        "encoded_shape": list(encoded_df.shape),
        "center_distribution": raw_df[CENTER_COLUMN].value_counts().sort_index().to_dict(),
        "original_target_distribution": raw_df[ORIGINAL_TARGET_COLUMN].value_counts().sort_index().to_dict(),
        "class_distribution": cleaned_df[TARGET_COLUMN].value_counts().sort_index().to_dict(),
        "missing_report": missing_report,
        "impute_strategy": impute_strategy,
        "outlier_summary": outlier_summary,
        "feature_meanings": FEATURE_MEANINGS,
        "missing_indicator_features": MISSING_INDICATOR_BASE_FEATURES,
        "sample_rows": int(len(sample_df)),
    }
    save_json(metadata, PROCESSING_METADATA_PATH)
    write_processing_report(raw_df, cleaned_df, missing_report, impute_strategy, outlier_summary, encoded_df)
    return metadata


if __name__ == "__main__":
    metadata = run_data_processing()
    print("四中心数据处理完成。")
    print(f"清洗后数据维度: {metadata['cleaned_shape']}")
    print(f"编码后数据维度: {metadata['encoded_shape']}")


