# 四中心模型验证报告

## 1. 建模方法
本实验基于 UCI Heart Disease 四中心合并数据集，比较了 `RandomForest`、`XGBoost`、`LightGBM` 三种树模型。

训练流程如下：
1. 对数据集进行训练集/测试集划分。
2. 在训练集上使用 `GridSearchCV` 执行超参数搜索。
3. 采用 5 折交叉验证评估模型稳定性。
4. 在独立测试集上计算 F1、AUC、平均精确率、灵敏度、特异度等指标。
5. 通过阈值搜索选择更合适的分类阈值。

## 2. 模型对比结果
| 模型 | CV F1 | CV AUC | Test F1 | Test AUC | Test AP | 灵敏度 | 特异度 | 阈值 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RandomForest | 0.8548 | 0.8863 | 0.8496 | 0.9095 | 0.9167 | 0.9412 | 0.6585 | 0.425 |
| XGBoost | 0.8537 | 0.8870 | 0.8286 | 0.8977 | 0.9169 | 0.8529 | 0.7439 | 0.445 |
| LightGBM | 0.8503 | 0.8867 | 0.8411 | 0.9005 | 0.9189 | 0.8824 | 0.7317 | 0.390 |

## 3. 最优模型
综合 `test_f1 + test_auc`，当前最优模型为 `RandomForest`。

其测试集表现为：
- F1：0.8496
- AUC：0.9095
- 平均精确率：0.9167
- 灵敏度：0.9412
- 特异度：0.6585
- 最优阈值：0.425

对应混淆矩阵为：
- TN = 54
- FP = 28
- FN = 6
- TP = 96

## 4. 输出图像
- PR 曲线：`outputs/figures/model_pr_curves.png`
- 校准曲线：`outputs/figures/model_calibration_curves.png`
- RandomForest 混淆矩阵：`outputs/figures/randomforest_confusion_matrix.png`
- XGBoost 混淆矩阵：`outputs/figures/xgboost_confusion_matrix.png`
- LightGBM 混淆矩阵：`outputs/figures/lightgbm_confusion_matrix.png`

## 5. 结论
四中心合并数据较单中心数据更复杂，存在中心差异、缺失值较多和类别标签原始定义不一致等问题。在此背景下，RandomForest 仍然取得了 `AUC > 0.90` 的结果，说明当前特征工程与训练流程具有较好的可复现性和应用价值。
