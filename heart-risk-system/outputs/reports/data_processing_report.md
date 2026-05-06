# 数据处理报告

## 数据来源

数据来自 UCI Heart Disease 四中心 processed 数据文件，合并 Cleveland、Hungarian、Switzerland、VA Long Beach 四个中心，共 920 条原始记录。

## 处理流程

1. 统一字段名并合并四中心数据。
2. 将原始 `num` 标签转换为二分类 `target`。
3. 对连续变量使用中位数填充，对分类变量使用众数填充。
4. 对连续变量采用 IQR 方法处理异常值。
5. 对分类变量进行 One-Hot 编码，对连续变量进行标准化。
6. 保存清洗数据、编码数据、预处理器和示例上传文件。

## 输出文件

- `data/heart_multicenter_merged.csv`
- `data/heart_cleaned.csv`
- `data/heart_processed.csv`
- `data/heart_processed.xlsx`
- `artifacts/preprocessor.pkl`
- `sample_upload.csv`
