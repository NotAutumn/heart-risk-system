# Heart Risk Explainability System

基于四中心 UCI Heart Disease 数据集的心脏病风险预测与 SHAP 可解释性 Web 系统。

## 目录说明

- `app.py` / `routes.py` / `web_service.py`：Flask Web 服务。
- `web/`：前端页面、样式和 JavaScript。
- `data_processing.py`：数据清洗、异常值处理和特征编码。
- `model_training.py`：Random Forest、XGBoost、LightGBM 训练与评估。
- `shap_analysis.py`：SHAP 全局、局部和交互解释图生成。
- `artifacts/`：已训练模型和预处理器。
- `data/`：原始数据和处理后数据。
- `outputs/`：模型评估图、SHAP 图和报告。
- `sample_upload.csv`：网页上传测试样例。

## 本地运行

```powershell
conda create -n heart-risk python=3.11 -y
conda activate heart-risk
pip install -r requirements.txt
python app.py
```
环境配置参照requirements.txt

浏览器打开：`http://127.0.0.1:5000`

## 重新生成全部结果

```powershell
python run_pipeline.py
```

## Docker 运行

```powershell
docker build -t heart-risk-system .
docker run -p 5000:5000 heart-risk-system
```

## 使用流程

1. 打开网页。
2. 上传 `sample_upload.csv` 或同字段格式的 CSV/Excel 文件。
3. 点击“开始预测”。
4. 查看预测结果、SHAP 图表，并下载报告。

## 输入字段

必须包含：`age`、`sex`、`cp`、`trestbps`、`chol`、`fbs`、`restecg`、`thalach`、`exang`、`oldpeak`、`slope`、`ca`、`thal`。
