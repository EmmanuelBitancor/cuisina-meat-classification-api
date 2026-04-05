const apiBaseInput = document.getElementById("apiBase");
const topKInput = document.getElementById("topK");
const imageInput = document.getElementById("imageInput");
const checkApiBtn = document.getElementById("checkApiBtn");
const predictBtn = document.getElementById("predictBtn");
const statusEl = document.getElementById("status");
const previewWrap = document.getElementById("previewWrap");
const previewImage = document.getElementById("previewImage");
const resultsEl = document.getElementById("results");
const filenameEl = document.getElementById("filename");
const inferenceGraphEl = document.getElementById("inferenceGraph");
const inferenceStatsEl = document.getElementById("inferenceStats");
const toggleAdvancedBtn = document.getElementById("toggleAdvancedBtn");
const hideAdvancedBtn = document.getElementById("hideAdvancedBtn");
const evalPanelEl = document.getElementById("evalPanel");
const datasetInput = document.getElementById("datasetInput");
const evaluateBtn = document.getElementById("evaluateBtn");
const evalStatusEl = document.getElementById("evalStatus");
const evalSummaryEl = document.getElementById("evalSummary");
const confusionMatrixEl = document.getElementById("confusionMatrix");
const reportGraphEl = document.getElementById("reportGraph");
const probabilityGraphEl = document.getElementById("probabilityGraph");

let selectedFile = null;
let isEvaluating = false;
let isAdvancedVisible = false;

const IMAGE_RE = /\.(jpg|jpeg|png|bmp|webp)$/i;

function normLabel(s) {
  return String(s || "").trim().toLowerCase();
}

function setEvalStatus(text, tone = "") {
  evalStatusEl.textContent = text;
  evalStatusEl.classList.remove("ok", "err");
  if (tone) evalStatusEl.classList.add(tone);
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function clearEvaluationVisuals() {
  confusionMatrixEl.innerHTML = "";
  reportGraphEl.innerHTML = "";
  probabilityGraphEl.innerHTML = "";
  evalSummaryEl.textContent = "";
}

function setAdvancedVisible(visible) {
  isAdvancedVisible = Boolean(visible);

  if (isAdvancedVisible) {
    evalPanelEl.classList.remove("is-hidden");
    toggleAdvancedBtn.textContent = "Hide Advanced Evaluation";
  } else {
    evalPanelEl.classList.add("is-hidden");
    toggleAdvancedBtn.textContent = "Show Advanced Evaluation";
  }
}

function makeLegend(items) {
  const legend = document.createElement("div");
  legend.className = "legend";

  items.forEach((item) => {
    const x = document.createElement("span");
    x.className = "legend-item";
    x.innerHTML = `<span class="legend-dot ${item.cls}"></span>${escapeHtml(item.label)}`;
    legend.appendChild(x);
  });

  return legend;
}

async function fetchHealth() {
  const base = apiBaseInput.value.trim().replace(/\/$/, "");
  if (!base) throw new Error("Set API Base URL first.");

  const res = await fetch(`${base}/health`);
  if (!res.ok) throw new Error(`Health check failed: HTTP ${res.status}`);
  return res.json();
}

async function loadLabels() {
  const labelCandidates = [];

  try {
    const res = await fetch("/labels.txt", { cache: "no-store" });
    if (res.ok) {
      const txt = await res.text();
      const parsed = txt
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter((line) => line && !line.startsWith("#"));
      if (parsed.length > 0) {
        return parsed;
      }
    }
  } catch {
    // fallback to /health below
  }

  try {
    const health = await fetchHealth();
    const count = Number(health?.output?.shape?.[1] || 0);
    for (let i = 0; i < count; i += 1) {
      labelCandidates.push(`class_${i}`);
    }
  } catch {
    // ignored; handled by caller if empty
  }

  return labelCandidates;
}

function setStatus(text, tone = "") {
  statusEl.textContent = text;
  statusEl.classList.remove("ok", "err");
  if (tone) statusEl.classList.add(tone);
}

function clearResults() {
  resultsEl.innerHTML = "";
  filenameEl.textContent = "";
  inferenceGraphEl.innerHTML = "";
  inferenceStatsEl.innerHTML = "";
}

function renderPredictions(predictions) {
  clearResults();

  predictions.forEach((item, i) => {
    const li = document.createElement("li");
    li.className = "prediction";
    li.style.animationDelay = `${i * 70}ms`;

    const label = item.label || `class_${item.index}`;
    const pct = Math.max(0, Math.min(100, item.score * 100));

    li.innerHTML = `
      <div class="prediction-head">
        <strong>${label}</strong>
        <span>${pct.toFixed(2)}%</span>
      </div>
      <div class="bar"><span style="--w:${pct}%"></span></div>
    `;

    resultsEl.appendChild(li);
  });
}

function renderInferenceVisuals(predictions, numClasses) {
  inferenceGraphEl.innerHTML = "";
  inferenceStatsEl.innerHTML = "";

  const safe = (Array.isArray(predictions) ? predictions : [])
    .map((p) => ({
      label: p.label || `class_${p.index}`,
      score: Number(p.score || 0),
    }))
    .sort((a, b) => b.score - a.score);

  if (!safe.length) {
    inferenceGraphEl.innerHTML = "<p class=\"subtitle tiny\">No probability data available.</p>";
    return;
  }

  const chart = document.createElement("div");
  chart.className = "prob-chart";

  safe.forEach((item) => {
    const pct = Math.max(0, Math.min(100, item.score * 100));
    const col = document.createElement("div");
    col.className = "prob-col";
    col.innerHTML = `
      <div class="prob-value">${pct.toFixed(1)}%</div>
      <div class="prob-bar-track">
        <span class="prob-bar-fill" style="height:${pct}%"></span>
      </div>
      <div class="prob-label" title="${escapeHtml(item.label)}">${escapeHtml(item.label)}</div>
    `;
    chart.appendChild(col);
  });

  inferenceGraphEl.appendChild(chart);

  const top1 = safe[0]?.score || 0;
  const top2 = safe[1]?.score || 0;
  const margin = Math.max(0, top1 - top2);
  const sum = safe.reduce((s, x) => s + x.score, 0) || 1;
  const norm = safe.map((x) => x.score / sum);
  const entropy = -norm.reduce((acc, p) => (p > 0 ? acc + p * Math.log(p) : acc), 0);
  const denom = Math.log(Math.max(2, Number(numClasses || safe.length || 2)));
  const entropyNorm = Math.min(1, Math.max(0, entropy / denom));
  const certainty = 1 - entropyNorm;

  inferenceStatsEl.appendChild(
    makeLegend([
      { cls: "c-prob-top", label: `Top-1 confidence: ${(top1 * 100).toFixed(2)}%` },
      { cls: "c-prob-true", label: `Top-1 vs Top-2 margin: ${(margin * 100).toFixed(2)}%` },
      { cls: "c-f1", label: `Prediction certainty: ${(certainty * 100).toFixed(2)}%` },
    ])
  );
}

async function checkApi() {
  setStatus("Checking API...");

  try {
    const data = await fetchHealth();
    const shape = JSON.stringify(data.input?.shape || []);
    setStatus(`API connected. Input shape: ${shape}. Labels loaded: ${data.labels_loaded ?? 0}`, "ok");
  } catch (err) {
    setStatus(`API check failed: ${err.message}`, "err");
  }
}

async function predictFile(file, topK) {
  const base = apiBaseInput.value.trim().replace(/\/$/, "");
  if (!base) {
    throw new Error("Set API Base URL first.");
  }

  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${base}/predict?top_k=${encodeURIComponent(topK)}`, {
    method: "POST",
    body: form,
  });

  const data = await res.json();
  if (!res.ok) {
    const detail = data?.detail || `HTTP ${res.status}`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }

  return data;
}

async function runPrediction() {
  const topK = Number(topKInput.value || 5);

  if (!selectedFile) {
    setStatus("Choose an image before predicting.", "err");
    return;
  }

  const form = new FormData();
  form.append("file", selectedFile);

  predictBtn.disabled = true;
  setStatus("Running prediction...");

  try {
    const data = await predictFile(selectedFile, 50);
    const allPreds = Array.isArray(data.predictions) ? data.predictions : [];
    const shownPreds = allPreds.slice(0, Math.max(1, topK));

    filenameEl.textContent = data.filename || "";
    renderPredictions(shownPreds);
    renderInferenceVisuals(allPreds, Number(data.num_classes || allPreds.length || 0));
    setStatus("Prediction complete.", "ok");
  } catch (err) {
    setStatus(`Prediction failed: ${err.message}`, "err");
  } finally {
    predictBtn.disabled = false;
  }
}

function renderConfusionMatrix(matrix, labels) {
  const n = labels.length;
  const maxCell = Math.max(1, ...matrix.flat());

  let html = "<table class=\"cm-table\"><thead><tr><th class=\"left\">True \\ Pred</th>";
  labels.forEach((name) => {
    html += `<th>${escapeHtml(name)}</th>`;
  });
  html += "</tr></thead><tbody>";

  for (let i = 0; i < n; i += 1) {
    const rowSum = matrix[i].reduce((a, b) => a + b, 0);
    html += `<tr><th class=\"left\">${escapeHtml(labels[i])}</th>`;

    for (let j = 0; j < n; j += 1) {
      const count = matrix[i][j];
      const pct = rowSum > 0 ? (count / rowSum) * 100 : 0;
      const intensity = count / maxCell;
      const color = i === j
        ? `rgba(46, 125, 50, ${0.08 + 0.72 * intensity})`
        : `rgba(198, 40, 40, ${0.05 + 0.55 * intensity})`;

      html += `<td style=\"background:${color}\"><div>${count}</div><div>${pct.toFixed(1)}%</div></td>`;
    }

    html += "</tr>";
  }

  html += "</tbody></table>";
  confusionMatrixEl.innerHTML = html;
}

function barHtml(name, value, colorClass) {
  const pct = Math.max(0, Math.min(100, value * 100));
  return `
    <div class=\"mini-bar\">
      <span>${escapeHtml(name)}</span>
      <span class=\"mini-bar-track\"><span class=\"mini-bar-fill ${colorClass}\" style=\"width:${pct}%\"></span></span>
      <span>${pct.toFixed(1)}%</span>
    </div>
  `;
}

function renderClassificationReportGraph(rows) {
  reportGraphEl.innerHTML = "";
  reportGraphEl.appendChild(
    makeLegend([
      { cls: "c-precision", label: "Precision" },
      { cls: "c-recall", label: "Recall" },
      { cls: "c-f1", label: "F1-score" },
    ])
  );

  rows.forEach((row) => {
    const item = document.createElement("div");
    item.className = "metric-row";
    item.innerHTML = `
      <div class=\"metric-head\">
        <strong>${escapeHtml(row.label)}</strong>
        <span>support: ${row.support}</span>
      </div>
      <div class=\"metric-bars\">
        ${barHtml("Precision", row.precision, "c-precision")}
        ${barHtml("Recall", row.recall, "c-recall")}
        ${barHtml("F1-score", row.f1, "c-f1")}
      </div>
    `;
    reportGraphEl.appendChild(item);
  });
}

function renderProbabilityStatsGraph(rows) {
  probabilityGraphEl.innerHTML = "";
  probabilityGraphEl.appendChild(
    makeLegend([
      { cls: "c-prob-true", label: "Mean True-Class Probability" },
      { cls: "c-prob-top", label: "Mean Top-1 Confidence" },
    ])
  );

  rows.forEach((row) => {
    const item = document.createElement("div");
    item.className = "metric-row";
    item.innerHTML = `
      <div class=\"metric-head\">
        <strong>${escapeHtml(row.label)}</strong>
        <span>samples: ${row.samples}</span>
      </div>
      <div class=\"metric-bars\">
        ${barHtml("True-class prob", row.meanTrueProb, "c-prob-true")}
        ${barHtml("Top-1 confidence", row.meanTopConf, "c-prob-top")}
      </div>
    `;
    probabilityGraphEl.appendChild(item);
  });
}

function buildReportAndStats(matrix, labels, sampleRows) {
  const n = labels.length;
  const colSums = Array.from({ length: n }, () => 0);
  const rowSums = Array.from({ length: n }, () => 0);

  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      rowSums[i] += matrix[i][j];
      colSums[j] += matrix[i][j];
    }
  }

  const reportRows = [];
  const probRows = [];

  for (let i = 0; i < n; i += 1) {
    const tp = matrix[i][i];
    const fp = colSums[i] - tp;
    const fn = rowSums[i] - tp;
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

    reportRows.push({
      label: labels[i],
      support: rowSums[i],
      precision,
      recall,
      f1,
    });

    const classSamples = sampleRows.filter((x) => x.trueIdx === i);
    const meanTrueProb = classSamples.length
      ? classSamples.reduce((s, x) => s + x.trueProb, 0) / classSamples.length
      : 0;
    const meanTopConf = classSamples.length
      ? classSamples.reduce((s, x) => s + x.topConf, 0) / classSamples.length
      : 0;

    probRows.push({
      label: labels[i],
      samples: classSamples.length,
      meanTrueProb,
      meanTopConf,
    });
  }

  return { reportRows, probRows };
}

async function runDatasetEvaluation() {
  if (isEvaluating) {
    return;
  }

  const allFiles = Array.from(datasetInput.files || []);
  const files = allFiles.filter(
    (file) => file.type.startsWith("image/") || IMAGE_RE.test(file.name)
  );

  if (!files.length) {
    setEvalStatus("Select a dataset folder with image files.", "err");
    return;
  }

  isEvaluating = true;
  evaluateBtn.disabled = true;
  clearEvaluationVisuals();
  setEvalStatus("Preparing labels and checking API...");

  try {
    const labels = await loadLabels();
    if (!labels.length) {
      throw new Error("No labels available. Check labels.txt.");
    }

    const labelIndex = new Map(labels.map((x, i) => [normLabel(x), i]));
    const n = labels.length;
    const matrix = Array.from({ length: n }, () => Array.from({ length: n }, () => 0));
    const sampleRows = [];

    let processed = 0;
    let skipped = 0;
    let correct = 0;
    const evalTopK = Math.max(n, Number(topKInput.value || n));

    for (let i = 0; i < files.length; i += 1) {
      const file = files[i];
      const rel = file.webkitRelativePath || file.name;
      const parts = rel.split(/[\\/]/).filter(Boolean);
      const trueLabel = parts.length >= 2 ? parts[parts.length - 2] : "";
      const trueIdx = labelIndex.get(normLabel(trueLabel));

      if (typeof trueIdx !== "number") {
        skipped += 1;
        continue;
      }

      setEvalStatus(`Evaluating ${i + 1}/${files.length}: ${file.name}`);

      const data = await predictFile(file, evalTopK);
      const predictions = Array.isArray(data.predictions) ? data.predictions : [];

      const probs = Array.from({ length: n }, () => 0);
      predictions.forEach((pred) => {
        let idx = -1;
        if (typeof pred.index === "number" && pred.index >= 0 && pred.index < n) {
          idx = pred.index;
        } else if (pred.label && labelIndex.has(normLabel(pred.label))) {
          idx = labelIndex.get(normLabel(pred.label));
        }

        if (idx >= 0) {
          probs[idx] = Number(pred.score || 0);
        }
      });

      let predIdx = 0;
      let topConf = probs[0] || 0;
      for (let j = 1; j < probs.length; j += 1) {
        if (probs[j] > topConf) {
          topConf = probs[j];
          predIdx = j;
        }
      }

      matrix[trueIdx][predIdx] += 1;
      processed += 1;

      if (predIdx === trueIdx) {
        correct += 1;
      }

      sampleRows.push({
        trueIdx,
        predIdx,
        trueProb: probs[trueIdx] || 0,
        topConf,
      });
    }

    if (processed === 0) {
      throw new Error("No valid labeled images processed. Ensure folder names match labels.txt.");
    }

    const { reportRows, probRows } = buildReportAndStats(matrix, labels, sampleRows);
    renderConfusionMatrix(matrix, labels);
    renderClassificationReportGraph(reportRows);
    renderProbabilityStatsGraph(probRows);

    const acc = (correct / processed) * 100;
    evalSummaryEl.textContent = `accuracy ${acc.toFixed(2)}% | processed ${processed} | skipped ${skipped}`;
    setEvalStatus("Evaluation visuals generated in frontend.", "ok");
  } catch (err) {
    setEvalStatus(`Evaluation failed: ${err.message}`, "err");
  } finally {
    isEvaluating = false;
    evaluateBtn.disabled = false;
  }
}

imageInput.addEventListener("change", () => {
  const file = imageInput.files?.[0];
  selectedFile = file || null;

  clearResults();

  if (!file) {
    previewWrap.classList.add("hidden");
    predictBtn.disabled = true;
    setStatus("Select an image to begin.");
    return;
  }

  const url = URL.createObjectURL(file);
  previewImage.src = url;
  previewWrap.classList.remove("hidden");
  predictBtn.disabled = false;
  setStatus("Image ready. Click Run Prediction.");
});

datasetInput.addEventListener("change", () => {
  const files = Array.from(datasetInput.files || []);
  const imageCount = files.filter(
    (file) => file.type.startsWith("image/") || IMAGE_RE.test(file.name)
  ).length;
  clearEvaluationVisuals();
  setEvalStatus(`Dataset selected: ${imageCount} image files detected.`);
});

toggleAdvancedBtn.addEventListener("click", () => {
  setAdvancedVisible(!isAdvancedVisible);
});

hideAdvancedBtn.addEventListener("click", () => {
  setAdvancedVisible(false);
});

checkApiBtn.addEventListener("click", checkApi);
predictBtn.addEventListener("click", runPrediction);
evaluateBtn.addEventListener("click", runDatasetEvaluation);

setAdvancedVisible(false);
