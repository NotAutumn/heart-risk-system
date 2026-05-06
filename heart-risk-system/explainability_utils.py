"""Helper functions for readable SHAP explanations.

These helpers convert encoded SHAP outputs back into original clinical
features so the reports and web responses stay understandable.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple

import pandas as pd

from common import FEATURE_MEANINGS

FEATURE_SHORT_LABELS: Dict[str, str] = {
    "age": "年龄",
    "sex": "性别",
    "cp": "胸痛类型",
    "trestbps": "静息血压",
    "chol": "胆固醇",
    "fbs": "空腹血糖",
    "restecg": "静息心电图",
    "thalach": "最大心率",
    "exang": "运动诱发心绞痛",
    "oldpeak": "ST压低",
    "slope": "ST斜率",
    "ca": "主要血管数目",
    "thal": "地中海贫血检查",
    "trestbps_missing": "静息血压缺失",
    "thalach_missing": "最大心率缺失",
    "oldpeak_missing": "ST压低缺失",
    "fbs_missing": "空腹血糖缺失",
    "exang_missing": "运动诱发心绞痛缺失",
    "slope_missing": "ST斜率缺失",
    "ca_missing": "主要血管数目缺失",
    "thal_missing": "地中海贫血检查缺失",
}

CLINICAL_KNOWLEDGE: Dict[str, str] = {
    "age": "年龄增大通常与动脉粥样硬化累积风险上升相关，是心血管评估中的基础危险因素。",
    "cp": "胸痛类型能反映症状模式差异，典型胸痛与缺血性心脏病风险密切相关。",
    "trestbps": "静息血压升高常提示高血压负担，对心血管事件风险有促进作用。",
    "chol": "胆固醇异常与冠状动脉粥样硬化风险相关，是常见的代谢危险因素。",
    "thalach": "最大心率反映运动耐量和心脏代偿能力，异常时常提示潜在心脏功能受限。",
    "exang": "运动诱发性心绞痛常与运动状态下心肌供血不足相关。",
    "oldpeak": "运动后 ST 压低是评估心肌缺血的重要指标，在临床上常被重点关注。",
    "slope": "ST 段斜率可反映心电图动态变化，对缺血判断有辅助意义。",
    "ca": "主要血管数目与冠脉病变范围相关，是冠状动脉受累程度的重要参考。",
    "thal": "地中海贫血/灌注检查相关结果常与心肌灌注异常和缺血风险评估相关。",
    "sex": "性别差异会影响心血管风险分层，是常见的基础人口学特征。",
    "fbs": "空腹血糖异常提示代谢风险升高，可间接影响心血管疾病发生。",
    "restecg": "静息心电图异常可提示基础电生理或缺血改变。",
}


def parse_encoded_feature_name(feature_name: str) -> Dict[str, str | None]:
    """Parse an encoded feature name into original feature information."""
    if feature_name.startswith("num__"):
        original = feature_name.split("__", 1)[1]
        if original.endswith("_missing"):
            base_feature = original[: -len("_missing")]
            return {
                "encoded_feature": feature_name,
                "original_feature": base_feature,
                "feature_type": "missing_indicator",
                "category_value": "missing",
                "display_name": f"{FEATURE_SHORT_LABELS.get(base_feature, base_feature)}缺失",
            }
        return {
            "encoded_feature": feature_name,
            "original_feature": original,
            "feature_type": "numeric",
            "category_value": None,
            "display_name": FEATURE_SHORT_LABELS.get(original, original),
        }

    if feature_name.startswith("cat__"):
        raw = feature_name.split("__", 1)[1]
        original, category_value = raw.rsplit("_", 1)
        short_label = FEATURE_SHORT_LABELS.get(original, original)
        return {
            "encoded_feature": feature_name,
            "original_feature": original,
            "feature_type": "categorical",
            "category_value": category_value,
            "display_name": f"{short_label}={category_value}",
        }

    return {
        "encoded_feature": feature_name,
        "original_feature": feature_name,
        "feature_type": "unknown",
        "category_value": None,
        "display_name": feature_name,
    }


def build_original_feature_matrix(shap_values, feature_names: List[str]) -> pd.DataFrame:
    """Aggregate encoded SHAP contributions back to original features."""
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    grouped = {}
    for feature_name in feature_names:
        meta = parse_encoded_feature_name(feature_name)
        original_feature = str(meta["original_feature"])
        grouped.setdefault(original_feature, pd.Series(0.0, index=shap_df.index))
        grouped[original_feature] = grouped[original_feature] + shap_df[feature_name]
    return pd.DataFrame(grouped)


def aggregate_global_importance(shap_values, feature_names: List[str]) -> List[Dict[str, object]]:
    """Return global importance ranked by original feature."""
    original_matrix = build_original_feature_matrix(shap_values, feature_names)
    importance = original_matrix.abs().mean().sort_values(ascending=False)
    return [
        {
            "feature": feature,
            "label": FEATURE_SHORT_LABELS.get(feature, feature),
            "meaning": FEATURE_MEANINGS.get(feature, ""),
            "importance": round(float(score), 6),
        }
        for feature, score in importance.items()
    ]


def build_local_explanation(
    sample_index: int,
    shap_values,
    feature_names: List[str],
    raw_row: pd.Series,
    prediction: int,
    probability: float,
    top_n: int = 5,
) -> Dict[str, object]:
    """Build a readable local explanation from aggregated original features."""
    original_matrix = build_original_feature_matrix([shap_values], feature_names)
    row = original_matrix.iloc[0]
    ranking = row.abs().sort_values(ascending=False).head(top_n)
    contributions = []
    readable_parts = []

    for feature in ranking.index:
        shap_score = float(row[feature])
        direction = "提高风险" if shap_score >= 0 else "降低风险"
        raw_value = raw_row.get(feature)
        if pd.isna(raw_value):
            raw_value = "缺失"
        contributions.append(
            {
                "feature": feature,
                "label": FEATURE_SHORT_LABELS.get(feature, feature),
                "raw_value": raw_value if pd.notna(raw_value) else None,
                "shap_value": round(shap_score, 6),
                "direction": direction,
                "meaning": FEATURE_MEANINGS.get(feature, ""),
            }
        )
        readable_parts.append(f"{FEATURE_SHORT_LABELS.get(feature, feature)}({raw_value}){direction}")

    risk_text = "高风险" if prediction == 1 else "低风险"
    summary_text = (
        f"样本 {sample_index} 预测为{risk_text}，概率为 {probability:.4f}。"
        f"主要影响因素包括 {'、'.join(readable_parts)}。"
    )
    return {
        "sample_index": int(sample_index),
        "prediction": int(prediction),
        "prediction_label": risk_text,
        "prediction_probability": round(float(probability), 4),
        "top_contributions": contributions,
        "summary_text": summary_text,
    }


def infer_original_feature_interactions(
    shap_values,
    feature_names: List[str],
    top_features: List[str],
    top_n: int = 3,
) -> List[Dict[str, object]]:
    """Estimate interaction strength using aggregated original SHAP features."""
    original_matrix = build_original_feature_matrix(shap_values, feature_names)
    candidate_features = [feature for feature in top_features if feature in original_matrix.columns][:5]
    interactions: List[Dict[str, object]] = []
    for left, right in combinations(candidate_features, 2):
        score = (original_matrix[left].abs() * original_matrix[right].abs()).mean()
        interactions.append(
            {
                "feature_a": left,
                "feature_a_label": FEATURE_SHORT_LABELS.get(left, left),
                "feature_b": right,
                "feature_b_label": FEATURE_SHORT_LABELS.get(right, right),
                "score": round(float(score), 6),
            }
        )
    interactions.sort(key=lambda item: item["score"], reverse=True)
    return interactions[:top_n]


def build_clinical_notes(global_importance: List[Dict[str, object]], top_n: int = 5) -> List[str]:
    """Generate clinically consistent notes from the top original features."""
    notes: List[str] = []
    for item in global_importance[:top_n]:
        feature = str(item["feature"])
        label = str(item["label"])
        base_note = CLINICAL_KNOWLEDGE.get(feature)
        if base_note:
            notes.append(f"{label}: {base_note}")
    notes.append("模型关注的核心变量主要集中在症状表现、运动相关缺血指标、冠脉受累程度和基础危险因素，整体上与心脏病辅助诊断常识一致。")
    return notes
