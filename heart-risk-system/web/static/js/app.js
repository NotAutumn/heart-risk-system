const state = {
  fileId: null,
  metadata: null,
  resultData: null,
};

const statusBox = document.getElementById('statusBox');
const uploadBtn = document.getElementById('uploadBtn');
const predictBtn = document.getElementById('predictBtn');
const dataFile = document.getElementById('dataFile');
const previewTableWrap = document.getElementById('previewTableWrap');
const predictionTableWrap = document.getElementById('predictionTableWrap');
const summaryCards = document.getElementById('summaryCards');
const reportLink = document.getElementById('reportLink');
const sampleSelector = document.getElementById('sampleSelector');
const localText = document.getElementById('localText');
const clinicalNotes = document.getElementById('clinicalNotes');
const schemaWrap = document.getElementById('schemaWrap');

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function setStatus(message, isError = false) {
  statusBox.textContent = message;
  statusBox.style.borderLeftColor = isError ? '#b42318' : '#d97706';
  statusBox.style.background = isError ? 'rgba(180,35,24,0.10)' : 'rgba(217,119,6,0.12)';
}

function renderTable(container, rows) {
  if (!rows || rows.length === 0) {
    container.innerHTML = '<p>暂无数据。</p>';
    return;
  }

  const headers = Object.keys(rows[0]);
  const thead = headers.map(header => `<th>${escapeHtml(header)}</th>`).join('');
  const tbody = rows.map(row => {
    const cells = headers.map(header => `<td>${escapeHtml(row[header])}</td>`).join('');
    return `<tr>${cells}</tr>`;
  }).join('');
  container.innerHTML = `<table><thead><tr>${thead}</tr></thead><tbody>${tbody}</tbody></table>`;
}

function renderSchema(metadata) {
  if (!metadata) {
    schemaWrap.innerHTML = '';
    return;
  }

  const rows = metadata.required_columns.map(feature => ({
    字段名: feature,
    类型: metadata.numeric_features.includes(feature) ? '连续型' : '分类型',
    含义: metadata.feature_meanings[feature] || '',
  }));
  const headers = Object.keys(rows[0]);
  const thead = headers.map(header => `<th>${escapeHtml(header)}</th>`).join('');
  const tbody = rows.map(row => {
    const cells = headers.map(header => `<td>${escapeHtml(row[header])}</td>`).join('');
    return `<tr>${cells}</tr>`;
  }).join('');
  schemaWrap.innerHTML = `
    <p>在线预测支持 ${escapeHtml(metadata.allowed_extensions.join(' / '))}，单次最多 ${escapeHtml(metadata.max_upload_rows)} 行样本。</p>
    <table>
      <thead><tr>${thead}</tr></thead>
      <tbody>${tbody}</tbody>
    </table>
  `;
}

function renderSummaryCards(summary) {
  const items = [
    ['模型', summary.model_name],
    ['样本数', summary.total_rows],
    ['阳性预测', summary.positive_predictions],
    ['阴性预测', summary.negative_predictions],
    ['平均概率', summary.mean_probability],
    ['阈值', summary.threshold],
  ];
  summaryCards.innerHTML = items.map(item => (
    `<div class="summary-card"><span>${escapeHtml(item[0])}</span><strong>${escapeHtml(item[1])}</strong></div>`
  )).join('');
}

function renderImportanceChart(globalImportance) {
  const features = globalImportance.map(item => item.label || item.feature).reverse();
  const values = globalImportance.map(item => item.importance).reverse();
  Plotly.newPlot('importanceChart', [{
    type: 'bar',
    orientation: 'h',
    y: features,
    x: values,
    marker: { color: '#0f766e' },
    hovertemplate: '特征: %{y}<br>重要性: %{x}<extra></extra>',
  }], {
    title: '原始临床特征重要性',
    margin: { l: 180, r: 30, t: 60, b: 50 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
  }, { responsive: true, displaylogo: false });
}

function renderInteractionChart(interactionSummary) {
  if (!interactionSummary || interactionSummary.length === 0) {
    document.getElementById('interactionChart').innerHTML = '';
    return;
  }

  const labels = interactionSummary.map(item => `${item.feature_a_label} × ${item.feature_b_label}`).reverse();
  const scores = interactionSummary.map(item => item.score).reverse();
  Plotly.newPlot('interactionChart', [{
    type: 'bar',
    orientation: 'h',
    y: labels,
    x: scores,
    marker: { color: '#d97706' },
    hovertemplate: '交互对: %{y}<br>强度: %{x}<extra></extra>',
  }], {
    title: '主要交互效应强度',
    margin: { l: 200, r: 30, t: 60, b: 50 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
  }, { responsive: true, displaylogo: false });
}

function renderDependenceCharts(dependenceData) {
  const container = document.getElementById('dependenceCharts');
  container.innerHTML = dependenceData.map((_, index) => `<div id="dependence_${index}" class="mini-chart"></div>`).join('');

  dependenceData.forEach((item, index) => {
    Plotly.newPlot(`dependence_${index}`, [{
      type: 'scatter',
      mode: 'markers',
      x: item.x,
      y: item.y,
      marker: { color: '#d97706', size: 8, opacity: 0.75 },
      hovertemplate: `${item.label || item.feature}<br>原始值: %{x}<br>SHAP: %{y}<extra></extra>`,
    }], {
      title: `Dependence: ${item.label || item.feature}`,
      margin: { l: 50, r: 20, t: 50, b: 50 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      xaxis: { title: 'Original Feature Value' },
      yaxis: { title: 'Aggregated SHAP Value' },
    }, { responsive: true, displaylogo: false });
  });
}

function renderClinicalNotes(notes) {
  if (!notes || notes.length === 0) {
    clinicalNotes.innerHTML = '暂无医学解释提示。';
    return;
  }
  clinicalNotes.innerHTML = `<strong>医学一致性提示</strong><ul>${notes.map(item => `<li>${escapeHtml(item)}</li>`).join('')}</ul>`;
}

function renderSampleOptions(localExplanations) {
  sampleSelector.innerHTML = localExplanations.map(item => (
    `<option value="${escapeHtml(item.row_index)}">样本 ${escapeHtml(item.row_index)} | 概率 ${escapeHtml(item.probability)}</option>`
  )).join('');
  if (localExplanations.length > 0) {
    renderLocalExplanation(localExplanations[0].row_index);
  }
}

function renderLocalExplanation(rowIndex) {
  const explanation = state.resultData.local_explanations.find(item => item.row_index === Number(rowIndex));
  if (!explanation) {
    return;
  }

  const x = explanation.contributions.map(item => item.shap_value).reverse();
  const y = explanation.contributions.map(item => item.label || item.feature).reverse();
  const colors = x.map(value => value >= 0 ? '#b45309' : '#0f766e');

  Plotly.newPlot('localChart', [{
    type: 'bar',
    orientation: 'h',
    x,
    y,
    marker: { color: colors },
    hovertemplate: '特征: %{y}<br>SHAP: %{x}<extra></extra>',
  }], {
    title: `Force-like Local Explanation | 样本 ${explanation.row_index}`,
    margin: { l: 180, r: 30, t: 60, b: 50 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
  }, { responsive: true, displaylogo: false });

  localText.textContent = explanation.summary_text || `样本 ${explanation.row_index} 的局部解释已生成。`;
}

async function fetchMetadata() {
  try {
    const response = await fetch('/api/metadata');
    const result = await response.json();
    if (!result.success) {
      throw new Error('元数据加载失败。');
    }
    state.metadata = result.data;
    renderSchema(result.data);
  } catch (_error) {
    schemaWrap.innerHTML = '<p>字段说明加载失败，请直接参考页面提示的必填字段。</p>';
  }
}

uploadBtn.addEventListener('click', async () => {
  if (!dataFile.files.length) {
    setStatus('请先选择待上传文件。', true);
    return;
  }

  const formData = new FormData();
  formData.append('file', dataFile.files[0]);
  setStatus('正在上传并校验文件，请稍候...');

  try {
    const response = await fetch('/api/upload', { method: 'POST', body: formData });
    const result = await response.json();
    if (!result.success) {
      throw new Error(result.message);
    }
    state.fileId = result.data.file_id;
    predictBtn.disabled = false;
    renderTable(previewTableWrap, result.data.preview);
    setStatus(`上传成功，共检测到 ${result.data.rows} 行样本，可以开始预测。`);
  } catch (error) {
    setStatus(error.message || '上传失败。', true);
  }
});

predictBtn.addEventListener('click', async () => {
  if (!state.fileId) {
    setStatus('请先上传文件。', true);
    return;
  }

  setStatus('正在调用模型并生成 SHAP 报告，请稍候...');
  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file_id: state.fileId }),
    });
    const result = await response.json();
    if (!result.success) {
      throw new Error(result.message);
    }

    state.resultData = result.data;
    renderSummaryCards(result.data.summary);
    renderTable(predictionTableWrap, result.data.predictions.slice(0, 20));
    renderImportanceChart(result.data.global_importance);
    renderInteractionChart(result.data.interaction_summary);
    renderDependenceCharts(result.data.dependence_data);
    renderClinicalNotes(result.data.clinical_notes);
    renderSampleOptions(result.data.local_explanations);
    reportLink.href = result.data.report_url;
    reportLink.classList.remove('disabled');
    setStatus('预测完成，结果和 SHAP 图表已更新。');
  } catch (error) {
    setStatus(error.message || '预测失败。', true);
  }
});

sampleSelector.addEventListener('change', event => {
  renderLocalExplanation(event.target.value);
});

fetchMetadata();
