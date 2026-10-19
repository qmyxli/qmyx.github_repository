const state = {
  catalog: null,
  uploadedFiles: [],
  lastTrace: null,
};

const els = {
  question: document.querySelector("#questionInput"),
  drugName: document.querySelector("#drugNameInput"),
  genericName: document.querySelector("#genericNameInput"),
  category: document.querySelector("#categoryInput"),
  uploadCategory: document.querySelector("#uploadCategoryInput"),
  useCatalog: document.querySelector("#useCatalogInput"),
  showTrace: document.querySelector("#showTraceInput"),
  chatForm: document.querySelector("#chatForm"),
  askButton: document.querySelector("#askButton"),
  retrieveButton: document.querySelector("#retrieveButton"),
  settingsButton: document.querySelector("#settingsButton"),
  settingsDrawer: document.querySelector("#settingsDrawer"),
  fileInput: document.querySelector("#fileInput"),
  dropZone: document.querySelector("#dropZone"),
  chooseFileButton: document.querySelector("#chooseFileButton"),
  uploadStatus: document.querySelector("#uploadStatus"),
  refreshCatalogButton: document.querySelector("#refreshCatalogButton"),
  uploadedFiles: document.querySelector("#uploadedFiles"),
  chatMessages: document.querySelector("#chatMessages"),
  riskBadge: document.querySelector("#riskBadge"),
  systemStatus: document.querySelector("#systemStatus"),
};

function requestBody() {
  return {
    question: els.question.value.trim(),
    drug_name: emptyToNull(els.drugName.value),
    generic_name: emptyToNull(els.genericName.value),
    category: emptyToNull(els.category.value),
    use_catalog: els.useCatalog.checked,
  };
}

function emptyToNull(value) {
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
}

async function api(path, options = {}) {
  const headers = options.body instanceof FormData ? {} : { "Content-Type": "application/json" };
  const response = await fetch(path, {
    headers,
    ...options,
  });
  const text = await response.text();
  let payload;
  try {
    payload = text ? JSON.parse(text) : {};
  } catch {
    payload = { detail: text };
  }
  if (!response.ok) {
    throw new Error(payload.detail || `HTTP ${response.status}`);
  }
  return payload;
}

async function loadHealth() {
  try {
    const health = await api("/api/health");
    els.systemStatus.textContent = health.catalog_exists
      ? `Chroma ${health.collection} · Catalog ready`
      : `Chroma ${health.collection} · Catalog missing`;
  } catch (error) {
    els.systemStatus.textContent = `服务异常：${error.message}`;
  }
}

async function loadCatalog() {
  try {
    state.catalog = await api("/api/catalog");
    renderUploadedFiles();
  } catch (error) {
    els.uploadedFiles.innerHTML = `<div class="empty-state">目录加载失败：${escapeHtml(error.message)}</div>`;
  }
}

async function ask(event) {
  event?.preventDefault();
  const body = requestBody();
  if (!body.question) {
    els.question.focus();
    return;
  }
  appendMessage("user", body.question);
  els.question.value = "";
  const pending = appendMessage("assistant", "正在检索说明书并生成回答...");
  setBusy(true);
  try {
    const payload = await api("/api/ask", {
      method: "POST",
      body: JSON.stringify(body),
    });
    renderSafety(payload.safety);
    renderTrace(payload.trace, payload.catalog_resolution, payload.safety);
    updateAssistantMessage(pending, payload.answer || "", payload.trace, payload.catalog_resolution, payload.safety);
  } catch (error) {
    renderSafety(null);
    updateAssistantMessage(pending, `请求失败：${error.message}`);
  } finally {
    setBusy(false);
  }
}

async function retrieveOnly() {
  const body = requestBody();
  if (!body.question) {
    els.question.focus();
    closeSettings();
    return;
  }
  const pending = appendMessage("assistant", "正在执行只检索流程...");
  setBusy(true);
  try {
    const payload = await api("/api/retrieve", {
      method: "POST",
      body: JSON.stringify(body),
    });
    renderSafety(null);
    renderTrace(payload.trace, payload.catalog_resolution);
    updateAssistantMessage(pending, "已完成检索，下面是命中的说明书片段。", payload.trace, payload.catalog_resolution);
    closeSettings();
  } catch (error) {
    updateAssistantMessage(pending, `检索失败：${error.message}`);
  } finally {
    setBusy(false);
  }
}

async function uploadFiles(files) {
  const selected = Array.from(files || []);
  if (!selected.length) return;
  for (const file of selected) {
    if (file.size > 10 * 1024 * 1024) {
      setUploadStatus(`${file.name} 超过 10MB，已跳过。`, "error");
      continue;
    }
    const formData = new FormData();
    formData.append("file", file);
    formData.append("category", els.uploadCategory.value.trim() || "上传文档");
    if (emptyToNull(els.drugName.value)) formData.append("drug_name", emptyToNull(els.drugName.value));
    if (emptyToNull(els.genericName.value)) formData.append("generic_name", emptyToNull(els.genericName.value));

    setUploadStatus(`正在上传并入库：${file.name}`, "active");
    setBusy(true);
    try {
      const payload = await api("/api/upload", {
        method: "POST",
        body: formData,
      });
      if (payload.action !== "skipped") {
        state.uploadedFiles.unshift(payload);
      }
      setUploadStatus(
        `${payload.filename} ${payload.message || "已处理。"}${payload.action === "skipped" ? "" : ` 生成 ${payload.chunk_count} 个片段。`}`,
        "success"
      );
      await loadCatalog();
    } catch (error) {
      setUploadStatus(`${file.name} 上传失败：${error.message}`, "error");
    } finally {
      setBusy(false);
    }
  }
  els.fileInput.value = "";
}

function appendMessage(role, text) {
  const article = document.createElement("article");
  article.className = `message ${role === "user" ? "user-message" : "assistant-message"}`;
  article.innerHTML = `
    <div class="message-role">${role === "user" ? "您" : "AI助手"}</div>
    <p>${escapeHtml(text)}</p>
  `;
  els.chatMessages.appendChild(article);
  scrollChatToBottom();
  return article;
}

function updateAssistantMessage(article, answer, trace = null, catalogResolution = null, safety = null) {
  const traceHtml = trace && els.showTrace.checked ? renderTraceBlock(trace, catalogResolution, safety) : "";
  const displayAnswer = safety ? stripSafetySummary(answer) : answer;
  article.innerHTML = `
    <div class="message-role">AI助手</div>
    <pre class="message-answer">${escapeHtml(displayAnswer)}</pre>
    ${traceHtml}
  `;
  scrollChatToBottom();
}

function renderTraceBlock(trace, catalogResolution, safety = null) {
  const route = trace.route || {};
  const counts = trace.stage_counts || {};
  const decision = trace.decision || {};
  const results = trace.results || [];
  const safetyHtml = renderSourceSafety(safety);
  const resolutionText = catalogResolution
    ? `${catalogResolution.drug_name || "-"} / ${catalogResolution.generic_name || "-"} / ${catalogResolution.category || "-"}`
    : "-";
  const sourceItems = results
    .slice(0, 5)
    .map((item) => {
      const sourceFile = item.source_file || "未知文件";
      const section = item.section || "未知章节";
      return `
        <li>
          <strong>${escapeHtml(sourceFile)} · ${escapeHtml(section)}</strong>
          <span>${escapeHtml(item.retrieval_stage || "-")} · page ${escapeHtml(String(item.page_start))}-${escapeHtml(String(item.page_end))}</span>
        </li>
      `;
    })
    .join("");
  return `
    <details class="trace-card" open>
      <summary>数据来源与检索 Trace</summary>
      ${safetyHtml}
      <div class="trace-line">strategy: ${escapeHtml(decision.strategy || "auto")} · ${escapeHtml(decision.reason || "")}</div>
      <div class="trace-line">intent: ${escapeHtml(route.intent || "-")} · confidence: ${route.confidence ?? "-"}</div>
      <div class="trace-line">mode: ${escapeHtml(trace.mode || "-")} · rerank: ${escapeHtml(trace.rerank || "-")}</div>
      <div class="trace-line">filter: ${escapeHtml(JSON.stringify(trace.metadata_filter || {}))}</div>
      <div class="trace-line">stages: ${escapeHtml(JSON.stringify(counts))}</div>
      <div class="trace-line">catalog: ${escapeHtml(resolutionText)}</div>
      <ul>${sourceItems || "<li>无检索结果</li>"}</ul>
    </details>
  `;
}

function renderSourceSafety(safety) {
  if (!safety) return "";
  const level = safety.risk_level || "unknown";
  const findings = safety.findings || [];
  const labels = findings.map((item) => item.label).filter(Boolean);
  const labelText = labels.length ? labels.join(", ") : "none";
  return `
    <div class="source-safety">
      <span class="source-safety-pill risk-${escapeHtml(level)}">安全等级：${escapeHtml(level)}</span>
      <span class="source-safety-pill">风险标签：${escapeHtml(labelText)}</span>
    </div>
  `;
}

function stripSafetySummary(answer) {
  return String(answer)
    .replace(/\n\n安全等级：\n[\s\S]*?(?=\n\n风险标签：)/, "")
    .replace(/\n\n风险标签：\n[\s\S]*?(?=\n\n需要咨询医生或药师的情况：)/, "")
    .trim();
}

function renderTrace(trace, catalogResolution, safety = null) {
  state.lastTrace = trace ? { trace, catalogResolution, safety } : null;
}

function renderSafety(safety) {
  const badge = els.riskBadge;
  if (!badge) return;
  badge.className = "risk-badge risk-empty";
  badge.textContent = "未评估";
  if (!safety) return;
  const level = safety.risk_level || "unknown";
  badge.textContent = level;
  badge.className = `risk-badge risk-${level}`;
}

function renderUploadedFiles() {
  const catalogDrugs = (state.catalog?.drugs || []).map((drug) => ({
    filename: drug.source_file || "-",
    drug_name: drug.drug_name || "-",
    generic_name: drug.generic_name || "-",
    category: drug.category || "-",
    chunk_count: drug.chunk_count || 0,
  }));
  const merged = [...state.uploadedFiles, ...catalogDrugs].filter(
    (item, index, arr) => arr.findIndex((candidate) => candidate.filename === item.filename) === index
  );
  if (!merged.length) {
    els.uploadedFiles.innerHTML = `<div class="empty-state">暂无已入库文件</div>`;
    return;
  }
  els.uploadedFiles.innerHTML = merged
    .map((item) => {
      return `
        <article class="file-item">
          <div>
            <strong>${escapeHtml(item.drug_name || item.filename || "-")}</strong>
            <span>${escapeHtml(item.generic_name || "-")} · ${item.chunk_count || 0} chunks</span>
          </div>
          <em>${escapeHtml(item.category || "未分类")}</em>
        </article>
      `;
    })
    .join("");
}

function setUploadStatus(message, status = "active") {
  els.uploadStatus.textContent = message;
  els.uploadStatus.className = `upload-status upload-${status}`;
}

function setBusy(isBusy) {
  document.body.classList.toggle("busy", isBusy);
  els.askButton.disabled = isBusy;
  els.retrieveButton.disabled = isBusy;
  els.chooseFileButton.disabled = isBusy;
}

function openSettings() {
  els.settingsDrawer.classList.add("open");
  els.settingsDrawer.setAttribute("aria-hidden", "false");
}

function closeSettings() {
  els.settingsDrawer.classList.remove("open");
  els.settingsDrawer.setAttribute("aria-hidden", "true");
}

function scrollChatToBottom() {
  els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => {
    els.question.value = button.dataset.question;
    els.question.focus();
  });
});

els.chatForm.addEventListener("submit", ask);
els.retrieveButton.addEventListener("click", retrieveOnly);
els.settingsButton.addEventListener("click", openSettings);
document.querySelectorAll("[data-close-settings]").forEach((button) => {
  button.addEventListener("click", closeSettings);
});

els.chooseFileButton.addEventListener("click", () => els.fileInput.click());
els.fileInput.addEventListener("change", (event) => uploadFiles(event.target.files));
els.dropZone.addEventListener("click", (event) => {
  if (event.target !== els.chooseFileButton) els.fileInput.click();
});
els.dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    els.fileInput.click();
  }
});
["dragenter", "dragover"].forEach((eventName) => {
  els.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.dropZone.classList.add("dragging");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  els.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.dropZone.classList.remove("dragging");
  });
});
els.dropZone.addEventListener("drop", (event) => uploadFiles(event.dataTransfer.files));

els.refreshCatalogButton.addEventListener("click", async () => {
  await loadHealth();
  await loadCatalog();
  setUploadStatus("目录已刷新。", "success");
});

loadHealth();
loadCatalog();
