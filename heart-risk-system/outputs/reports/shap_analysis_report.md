# SHAP 可解释性分析报告

## 1. 分析对象
本报告基于最优模型 `RandomForest` 生成 SHAP 解释结果，用于说明模型整体决策规律、单个样本预测原因以及重要特征间的交互效应。

## 2. 全局解释结果
编码特征层面的 Top 特征包括：
- `cat__cp_4.0`
- `num__oldpeak`
- `cat__exang_0.0`
- `num__chol`
- `cat__exang_1.0`
- `cat__cp_2.0`

将编码后特征聚合回原始临床特征后，重要性前列特征为：
1. `cp` 胸痛类型，重要性 0.132704
2. `exang` 运动诱发性心绞痛，重要性 0.087354
3. `oldpeak` ST 压低，重要性 0.052189
4. `sex` 性别，重要性 0.050181
5. `chol` 胆固醇，重要性 0.045032
6. `thal` 地中海贫血检查，重要性 0.040607
7. `thalach` 最大心率，重要性 0.038985
8. `age` 年龄，重要性 0.032845

这些结果与心血管临床经验基本一致，说明模型更关注胸痛表现、运动相关缺血指标、代谢风险与基础人口学因素。

## 3. 局部样本解释
代表性局部样本解释如下：
- 样本 0：预测为低风险，概率 0.2329。主要是 `cp=1.0`、`exang=0.0` 将风险拉低，而 `oldpeak=2.3`、`age=63` 对风险有一定提升作用。
- 样本 1：预测为高风险，概率 0.9728。`cp=4.0`、`exang=1.0`、`thalach=108`、`oldpeak=1.5` 和年龄偏高共同推动预测为阳性。
- 样本 2：预测为高风险，概率 0.9796。`cp=4.0`、`oldpeak=2.6`、`exang=1.0`、`thal=7.0`、`ca=2.0` 均明显推动高风险判断。

## 4. 交互效应结果
当前交互分析显示，较强的特征对包括：
- `cp` 与 `exang`，交互强度 0.011523
- `cp` 与 `oldpeak`，交互强度 0.006849
- `cp` 与 `sex`，交互强度 0.006268

这说明模型在判断风险时，不仅关注单一变量，还会结合胸痛类型与运动诱发症状、ST 段变化等因素的联合作用。

## 5. 图像输出
- Summary 图：`summary_beeswarm.png`、`summary_bar.png`、`summary_violin.png`
- 原始特征重要性图：`summary_original_features.png`
- Force Plot：`force_plot_1.png`、`force_plot_2.png`、`force_plot_3.png`
- Dependence Plot：`dependence_plot_1.png`、`dependence_plot_2.png`、`dependence_plot_3.png`
- Interaction Plot：`interaction_plot_1.png`、`interaction_plot_2.png`、`interaction_plot_3.png`

## 6. SHAP 加速验证
为验证混合特征下 SHAP 计算的效率问题，项目比较了全量计算与抽样计算：
- 全量样本 SHAP 耗时：93.7771 秒
- 代表性抽样 SHAP 耗时：1.5037 秒
- Top 特征摘要计算耗时：0.000481 秒
- 抽样加速比：62.3636

结论：对多中心混合特征心脏病数据，使用代表性抽样与 Top-N 摘要策略，可以显著降低解释延迟，适合 Web 在线场景。

## 7. 结论
当前 SHAP 解释模块已实现：
- 全局特征重要性分析
- 原始临床特征重要性聚合
- 局部样本解释
- 特征交互可视化
- 解释效率加速验证

解释结果与临床常识保持一致，可用于结果分析和页面展示。



