const API = "/api/v1/financial";
const MAX_FILE_BYTES = 10 * 1024 * 1024;
const MAX_BATCH_BYTES = 90 * 1024 * 1024;
const ALLOWED_EXTENSIONS = new Set(["txt", "md", "markdown"]);
const CONVERSATION_STORAGE_KEY = "financial-rag.conversations.v1";
const LLM_PROFILES_STORAGE_KEY = "financial-rag.llm-profiles.v1";
const LLM_SELECTION_STORAGE_KEY = "financial-rag.llm-selection.v1";
const MAX_SAVED_CONVERSATIONS = 24;
const MAX_TURNS_PER_CONVERSATION = 40;

const elements = {
  status: document.querySelector("#service-status"),
  statusText: document.querySelector("#service-status-text"),
  documentCount: document.querySelector("#document-count"),
  chunkCount: document.querySelector("#chunk-count"),
  listCount: document.querySelector("#document-list-count"),
  documentList: document.querySelector("#document-list"),
  refresh: document.querySelector("#refresh-documents"),
  uploadForm: document.querySelector("#upload-form"),
  fileInput: document.querySelector("#file-input"),
  dropzone: document.querySelector("#dropzone"),
  chooseFiles: document.querySelector("#choose-files"),
  selectedFiles: document.querySelector("#selected-files"),
  uploadButton: document.querySelector("#upload-button"),
  uploadFeedback: document.querySelector("#upload-feedback"),
  companyCode: document.querySelector("#company-code"),
  conversation: document.querySelector("#conversation"),
  newConversation: document.querySelector("#new-conversation"),
  conversationHistoryToggle: document.querySelector("#conversation-history-toggle"),
  conversationHistoryList: document.querySelector("#conversation-history-list"),
  conversationHistoryItems: document.querySelector("#conversation-history-items"),
  conversationHistoryCount: document.querySelector("#conversation-history-count"),
  historyBar: document.querySelector("#history-bar"),
  historyTrack: document.querySelector("#history-track"),
  historyCard: document.querySelector(".history-card"),
  historyTitle: document.querySelector("#history-title"),
  historyDescription: document.querySelector("#history-description"),
  historyBookmark: document.querySelector("#history-bookmark"),
  queryForm: document.querySelector("#query-form"),
  queryInput: document.querySelector("#query-input"),
  topK: document.querySelector("#top-k"),
  llmChoice: document.querySelector("#llm-choice"),
  askButton: document.querySelector("#ask-button"),
};

const welcomeTemplate = document.querySelector("#welcome")?.cloneNode(true);
let selectedFiles = [];
let currentDocuments = [];
let scopedRepository = false;
let isUploading = false;
let isAsking = false;
let conversations = [];
let activeConversationId = null;
let historyEntries = [];
let activeHistoryId = null;
let previewHistoryId = null;
let historySequence = 0;
let historyScrollFrame = 0;
let conversationSyncChain = Promise.resolve();

function formatNumber(value) {
  return new Intl.NumberFormat("vi-VN").format(Number(value) || 0);
}

function formatBytes(value) {
  if (value < 1024 * 1024) return `${Math.max(1, Math.round(value / 1024))} KB`;
  return `${(value / (1024 * 1024)).toLocaleString("vi-VN", { maximumFractionDigits: 1 })} MB`;
}

function readLlmProfiles() {
  try {
    const profiles = JSON.parse(localStorage.getItem(LLM_PROFILES_STORAGE_KEY) || "[]");
    return Array.isArray(profiles) ? profiles.filter((profile) => profile && profile.id && profile.model) : [];
  } catch {
    return [];
  }
}

function renderLlmChoice(defaultModel = "qwen3.5:4b") {
  if (!elements.llmChoice) return;
  const profiles = readLlmProfiles();
  let selectedId = "default";
  try {
    selectedId = localStorage.getItem(LLM_SELECTION_STORAGE_KEY) || "default";
  } catch {
    // Continue with the current service model if browser storage is unavailable.
  }

  const options = [new Option(`Mặc định · ${defaultModel}`, "default")];
  for (const profile of profiles) {
    const provider = profile.provider === "ollama" ? "Ollama" : (profile.provider_label || "API");
    options.push(new Option(profile.label || `${provider} · ${profile.model}`, String(profile.id)));
  }
  elements.llmChoice.replaceChildren(...options);
  elements.llmChoice.value = options.some((option) => option.value === selectedId) ? selectedId : "default";
}

function saveLlmChoice() {
  try {
    localStorage.setItem(LLM_SELECTION_STORAGE_KEY, elements.llmChoice.value || "default");
  } catch {
    // The selection still applies for this page even if storage is disabled.
  }
}

function setServiceState(state, message) {
  elements.status.classList.remove("is-loading", "is-ready", "is-error");
  elements.status.classList.add(state);
  elements.statusText.textContent = message;
}

async function requestJson(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API}${path}`, {
      ...options,
      headers: {
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...options.headers,
      },
    });
  } catch (error) {
    throw new Error("Không kết nối được Financial RAG. Hãy kiểm tra dịch vụ đang chạy.");
  }

  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = payload?.detail;
    throw new Error(typeof detail === "string" ? detail : `Yêu cầu thất bại (HTTP ${response.status}).`);
  }
  return payload;
}

async function refreshWorkspace() {
  elements.refresh.disabled = true;
  try {
    const [health, documents] = await Promise.all([
      requestJson("/health"),
      requestJson("/documents"),
    ]);
    setServiceState("is-ready", "Dịch vụ sẵn sàng");
    scopedRepository = health.runtime_mode === "scoped_repository";
  renderLlmChoice(health.ollama_model || "qwen3.5:4b");
    const modelLabel = document.querySelector("#current-model");
    if (modelLabel) modelLabel.textContent = health.ollama_model || "Mô hình của dịch vụ";
    elements.documentCount.textContent = formatNumber(health.stored_documents ?? documents.length);
    elements.chunkCount.textContent = formatNumber(health.indexed_chunks);
    elements.listCount.textContent = formatNumber(documents.length);
    renderDocuments(documents);
    window.notebookWorkspace?.renderCatalog();
    await window.notebookWorkspace?.refreshActive();
  } catch (error) {
    setServiceState("is-error", "Không kết nối được");
    renderLlmChoice();
    elements.documentCount.textContent = "—";
    elements.chunkCount.textContent = "—";
    elements.documentList.replaceChildren(placeholder(error.message));
  } finally {
    elements.refresh.disabled = false;
  }
}

function placeholder(message) {
  const node = document.createElement("div");
  node.className = "list-placeholder";
  node.textContent = message;
  return node;
}

function renderDocuments(documents) {
  currentDocuments = documents;
  const search = document.querySelector("#document-search");
  const companySelect = document.querySelector("#document-company");
  const selectedCompany = companySelect?.value || "";
  if (companySelect) {
    const codes = [...new Set(documents.map((item) => item.company_code).filter(Boolean))].sort();
    companySelect.replaceChildren(new Option("Mọi doanh nghiệp", ""), ...codes.map((code) => new Option(code, code)));
    companySelect.value = codes.includes(selectedCompany) ? selectedCompany : "";
  }
  const normalized = (value) => String(value || "").normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/đ/gi, "d").toLocaleLowerCase();
  const terms = normalized(search?.value).trim().split(/\s+/).filter(Boolean);
  const filtered = documents.filter((item) => (!companySelect?.value || item.company_code === companySelect.value)
    && terms.every((term) => normalized([item.source_file, item.company_code, item.fiscal_year, documentDisplayName(item)].join(" ")).includes(term)))
    .sort((a, b) => (a.company_code || "").localeCompare(b.company_code || "") || Number(b.fiscal_year || 0) - Number(a.fiscal_year || 0));
  elements.listCount.textContent = formatNumber(filtered.length);
  const resultLabel = document.querySelector("#catalog-result-label");
  if (resultLabel) resultLabel.textContent = terms.length || companySelect?.value ? `Kết quả · ${documents.length} tài liệu trong kho` : "Tài liệu trong kho";
  elements.documentList.replaceChildren();
  if (!documents.length) {
    elements.documentList.append(placeholder("Kho chưa có tài liệu. Mở Nạp tệp từ thiết bị để thêm TXT hoặc Markdown."));
    return;
  }
  if (!filtered.length) {
    elements.documentList.append(placeholder("Không tìm thấy tài liệu. Thử mã doanh nghiệp, năm khác hoặc bỏ bộ lọc."));
    return;
  }

  for (const item of filtered) {
    const card = document.createElement("article");
    card.className = "document-card";

    const badge = document.createElement("span");
    badge.className = "file-badge";
    badge.textContent = extensionLabel(item.source_file);

    const copy = document.createElement("div");
    copy.className = "document-copy";
    const name = document.createElement("strong");
    name.textContent = documentDisplayName(item);
    name.title = item.source_file || "";
    const details = document.createElement("span");
    details.textContent = item.source_file || "Tài liệu không tên";
    details.title = item.source_file || "";
    copy.append(name, details);

    const actions = document.createElement("div");
    actions.className = "document-actions";
    const add = document.createElement("button");
    add.type = "button";
    add.className = "button";
    add.dataset.attachSource = item.logical_key || "";
    const source = window.notebookWorkspace?.active?.sources.find((source) => source.logical_key === item.logical_key);
    const attached = Boolean(source && (source.catalog_projection_id || source.active_projection_id) === item.active_projection_id);
    add.dataset.attached = String(attached);
    add.disabled = attached || isAsking || Boolean(window.notebookWorkspace?.busy);
    add.textContent = attached ? "✓ Đã thêm" : source ? "Nhận bản mới" : "+ Thêm vào notebook";
    add.setAttribute("aria-label", `${add.textContent} · ${documentDisplayName(item)}`);
    add.addEventListener("click", async () => {
      if (!window.notebookWorkspace?.active) {
        document.querySelector("#open-create-notebook")?.click();
        return;
      }
      add.disabled = true;
      try { await window.notebookWorkspace.attach(item.logical_key); }
      catch (error) { window.workspaceUI?.notify(error.message, true); }
      finally { renderDocuments(currentDocuments); }
    });

    const remove = document.createElement("button");
    remove.className = "delete-document";
    remove.type = "button";
    remove.textContent = "Xóa khỏi kho";
    remove.title = `Xóa ${item.source_file}`;
    remove.setAttribute("aria-label", `Xóa ${item.source_file}`);
    remove.addEventListener("click", () => deleteDocument(item, remove));

    const menu = document.createElement("details");
    menu.className = "document-menu";
    const menuTitle = document.createElement("summary");
    menuTitle.textContent = "⋯";
    menuTitle.setAttribute("aria-label", `Tùy chọn tài liệu ${documentDisplayName(item)}`);
    menu.append(menuTitle, remove);
    actions.append(add, menu);
    card.append(badge, copy, actions);
    elements.documentList.append(card);
  }
}

function documentDisplayName(item) {
  const file = item.source_file || "";
  const report = /hopnhat|hợp nhất|hop nhat/i.test(file) || item.report_type === "consolidated" ? "Hợp nhất"
    : /congtyme|riêng|rieng/i.test(file) || ["separate", "standalone"].includes(item.report_type) ? "Riêng lẻ" : "";
  return item.company_code ? [item.company_code, item.fiscal_year, report].filter(Boolean).join(" · ") : file || "Tài liệu không tên";
}

function extensionLabel(filename = "") {
  const ext = filename.split(".").pop()?.toUpperCase();
  return ext === "MARKDOWN" ? "MD" : (ext || "TXT").slice(0, 4);
}

async function deleteDocument(document, button) {
  const label = document.source_file || "tài liệu này";
  if (!window.confirm(`Xóa ${label} khỏi kho riêng?`)) return;
  button.disabled = true;
  try {
    await requestJson(`/documents/${encodeURIComponent(document.logical_key)}`, { method: "DELETE" });
    await refreshWorkspace();
  } catch (error) {
    window.alert(error.message);
    button.disabled = false;
  }
}

function addFiles(fileList) {
  const rejected = [];
  for (const file of fileList) {
    const extension = file.name.split(".").pop()?.toLowerCase() || "";
    if (!ALLOWED_EXTENSIONS.has(extension)) {
      rejected.push(`${file.name}: chỉ nhận TXT hoặc Markdown`);
      continue;
    }
    if (file.size === 0) {
      rejected.push(`${file.name}: tệp đang trống`);
      continue;
    }
    if (file.size > MAX_FILE_BYTES) {
      rejected.push(`${file.name}: vượt quá 10 MB`);
      continue;
    }
    const duplicate = selectedFiles.some((item) => item.name === file.name && item.lastModified === file.lastModified && item.size === file.size);
    if (!duplicate) selectedFiles.push(file);
  }
  renderSelectedFiles();
  if (rejected.length) setUploadFeedback(rejected.join(". "), "is-error");
}

function renderSelectedFiles() {
  elements.selectedFiles.replaceChildren();
  for (const [index, file] of selectedFiles.entries()) {
    const row = document.createElement("li");
    row.className = "selected-file";
    const copy = document.createElement("span");
    copy.className = "selected-file-copy";
    const name = document.createElement("strong");
    name.textContent = file.name;
    const size = document.createElement("small");
    size.textContent = formatBytes(file.size);
    copy.append(name, size);
    const remove = document.createElement("button");
    remove.className = "remove-file";
    remove.type = "button";
    remove.textContent = "×";
    remove.setAttribute("aria-label", `Bỏ ${file.name} khỏi danh sách nạp`);
    remove.addEventListener("click", () => {
      selectedFiles.splice(index, 1);
      renderSelectedFiles();
    });
    row.append(copy, remove);
    elements.selectedFiles.append(row);
  }
  elements.uploadButton.disabled = !selectedFiles.length || isUploading;
}

function setUploadFeedback(message, state = "") {
  elements.uploadFeedback.textContent = message;
  elements.uploadFeedback.className = `feedback ${state}`.trim();
}

async function uploadSelectedFiles(event) {
  event.preventDefault();
  if (!selectedFiles.length || isUploading) return;

  const companyCode = elements.companyCode.value.trim().toUpperCase();
  const replacingFiles = selectedFiles.filter((file) => currentDocuments.some((item) => {
    if ((item.source_file || "").toLocaleLowerCase() !== file.name.toLocaleLowerCase()) return false;
    return !companyCode || (item.company_code || "").toUpperCase() === companyCode;
  }));
  if (replacingFiles.length && !window.confirm(
    `Tài liệu cùng tên/mã có thể bị thay thế: ${replacingFiles.map((file) => file.name).join(", ")}. Tiếp tục nạp?`,
  )) {
    setUploadFeedback("Đã hủy nạp; các tài liệu hiện có được giữ nguyên.");
    return;
  }

  isUploading = true;
  renderSelectedFiles();
  setUploadFeedback(`Đang đọc và nạp ${selectedFiles.length} tệp…`);
  try {
    const filesToUpload = [...selectedFiles];
    const documents = await Promise.all(filesToUpload.map(async (file) => ({
      content: await file.text(),
      source_file: file.name,
      ...(companyCode ? { company_code: companyCode } : {}),
    })));
    const textBytes = documents.reduce((sum, document) => sum + new Blob([document.content]).size, 0);
    if (textBytes > MAX_BATCH_BYTES) throw new Error("Tổng dung lượng một lượt nạp không được vượt quá 90 MB.");

    const job = await requestJson(scopedRepository ? "/ingest/jobs" : "/ingest/batch", {
      method: "POST",
      body: JSON.stringify({ documents }),
    });
    let result = job;
    const deadline = Date.now() + 10 * 60 * 1000;
    while (["queued", "running"].includes(result.status) && Date.now() < deadline) {
      setUploadFeedback(`Đang xử lý ${filesToUpload.length} tệp · ${result.status === "queued" ? "đợi worker" : "lập chỉ mục"} · lượt ${result.attempts || 0}`);
      await new Promise((resolve) => setTimeout(resolve, 2000));
      result = await requestJson(`/ingest/jobs/${encodeURIComponent(job.job_id)}`);
    }
    if (result.status === "failed") throw new Error(`Nạp thất bại: ${result.error || job.job_id}`);
    if (!["succeeded", "success"].includes(result.status)) throw new Error(`Job ${job.job_id} vẫn đang chạy; có thể kiểm tra lại bằng mã này.`);
    selectedFiles = [];
    elements.fileInput.value = "";
    renderSelectedFiles();
    setUploadFeedback(
      `Đã hoàn tất ${filesToUpload.length} tài liệu${job.job_id ? ` · mã job ${job.job_id.slice(0, 12)}` : ""}.`,
      "is-success",
    );
    await refreshWorkspace();
  } catch (error) {
    setUploadFeedback(error.message, "is-error");
  } finally {
    isUploading = false;
    renderSelectedFiles();
  }
}

function removeWelcome() {
  document.querySelector("#welcome")?.remove();
}

function scrollConversationToEnd() {
  elements.conversation.scrollTop = elements.conversation.scrollHeight;
}

function makeConversationId() {
  if (window.crypto?.randomUUID) return window.crypto.randomUUID();
  return "conversation-" + Date.now() + "-" + Math.random().toString(36).slice(2, 10);
}

function storedResponse(response) {
  if (!response || typeof response !== "object") return null;
  const validation = response.validation && typeof response.validation === "object"
    ? response.validation
    : {};
  const normalizedCitations = Array.isArray(response.citations)
    ? response.citations.map((citation) => ({
      citation_id: String(citation.citation_id || ""),
      file_name: String(citation.file_name || ""),
      page: citation.page ?? null,
      unit: String(citation.unit || ""),
      section: String(citation.section || ""),
      snippet: String(citation.snippet || "").slice(0, 400),
    }))
    : [];
  const calculations = Array.isArray(response.calculations)
    ? response.calculations.slice(0, 100).map((calculation) => ({
      value: String(calculation.value ?? ""),
      unit: String(calculation.unit || ""),
      formula_id: String(calculation.formula_id || ""),
      formula_version: String(calculation.formula_version || ""),
      expression: String(calculation.expression || ""),
      input_citation_ids: Array.isArray(calculation.input_citation_ids)
        ? calculation.input_citation_ids.slice(0, 20).map((citationId) => String(citationId))
        : [],
      inputs: Array.isArray(calculation.inputs)
        ? calculation.inputs.slice(0, 200).map((input) => ({
          role: String(input.role || ""), label: String(input.label || ""),
          year: input.year ?? null, value: String(input.value ?? ""),
          unit: String(input.unit || ""), citation_id: String(input.citation_id || ""),
        }))
        : [],
    }))
    : [];
  const calculationCitationIds = new Set(
    calculations.flatMap((calculation) => calculation.input_citation_ids),
  );
  const citations = normalizedCitations
    .filter((citation, index) => index < 8 || calculationCitationIds.has(citation.citation_id))
    .slice(0, 100);
  return {
    answer: String(response.answer || "").slice(0, 30000),
    validation: {
      is_valid: Boolean(validation.is_valid),
      details: String(validation.details || "").slice(0, 2000),
    },
    citations,
    calculations,
    answer_status: ["complete", "partial", "needs_clarification", "insufficient_evidence"].includes(response.answer_status)
      ? response.answer_status : null,
    analysis_coverage: response.analysis_coverage || null,
    source_snapshot: response.source_snapshot || null,
    claim_checks: response.claim_checks || [],
  };
}

function normalizeConversationRecords(records) {
  if (!Array.isArray(records)) return [];
  return records
      .filter((conversation) => conversation && typeof conversation.id === "string" && Array.isArray(conversation.turns))
      .map((conversation) => ({
        id: conversation.id,
        title: String(conversation.title || "Cuộc trò chuyện"),
        createdAt: String(conversation.createdAt || new Date().toISOString()),
        updatedAt: String(conversation.updatedAt || conversation.createdAt || new Date().toISOString()),
        turns: conversation.turns
          .filter((turn) => turn && typeof turn.query === "string")
          .slice(-MAX_TURNS_PER_CONVERSATION)
          .map((turn) => ({
            id: String(turn.id || makeConversationId()),
            query: turn.query,
            summary: String(turn.summary || "Chưa có câu trả lời.").slice(0, 2_000),
            bookmarked: Boolean(turn.bookmarked),
            response: storedResponse(turn.response),
            error: turn.error ? String(turn.error).slice(0, 4_000) : "",
          })),
      }))
      .filter((conversation) => conversation.turns.length)
      .sort((left, right) => right.updatedAt.localeCompare(left.updatedAt))
      .slice(0, MAX_SAVED_CONVERSATIONS);
}

function loadConversationHistory() {
  try {
    const saved = JSON.parse(localStorage.getItem(CONVERSATION_STORAGE_KEY) || "{}");
    if (!Array.isArray(saved.conversations)) return;
    conversations = normalizeConversationRecords(saved.conversations);
    const storedActiveId = saved.activeConversationId;
    activeConversationId = conversations.some((conversation) => conversation.id === storedActiveId)
      ? storedActiveId
      : null;
  } catch {
    conversations = [];
    activeConversationId = null;
  }
}

function persistConversationHistory(syncConversationId = null) {
  conversations.sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));
  const savedConversations = conversations
    .filter((conversation) => conversation.turns.length)
    .slice(0, MAX_SAVED_CONVERSATIONS)
    .map((conversation) => ({
      id: conversation.id,
      title: conversation.title,
      createdAt: conversation.createdAt,
      updatedAt: conversation.updatedAt,
      turns: conversation.turns.slice(-MAX_TURNS_PER_CONVERSATION).map((turn) => ({
        id: turn.id,
        query: turn.query,
        summary: String(turn.summary || "").slice(0, 2_000),
        bookmarked: Boolean(turn.bookmarked),
        response: storedResponse(turn.response),
        error: String(turn.error || "").slice(0, 4_000),
      })),
    }));
  const payload = {
    version: 1,
    activeConversationId: savedConversations.some((item) => item.id === activeConversationId)
      ? activeConversationId
      : null,
    conversations: savedConversations,
  };

  try {
    localStorage.setItem(CONVERSATION_STORAGE_KEY, JSON.stringify(payload));
  } catch {
    const active = savedConversations.find((item) => item.id === activeConversationId);
    const others = savedConversations.filter((item) => item.id !== activeConversationId);
    while (others.length) {
      others.pop();
      payload.conversations = active ? [active, ...others] : others;
      try {
        localStorage.setItem(CONVERSATION_STORAGE_KEY, JSON.stringify(payload));
        break;
      } catch {
        // Keep trimming older saved conversations until browser storage accepts the current chat.
      }
    }
  }
  if (syncConversationId) {
    const conversation = payload.conversations.find((item) => item.id === syncConversationId);
    if (conversation) queueConversationSave(conversation);
  }
  renderConversationHistoryList();
}

function queueConversationSave(conversation) {
  const snapshot = JSON.parse(JSON.stringify(conversation));
  conversationSyncChain = conversationSyncChain
    .catch(() => {})
    .then(() => requestJson(`/conversations/${encodeURIComponent(snapshot.id)}`, {
      method: "PUT",
      body: JSON.stringify(snapshot),
    }))
    .catch((error) => console.warn("Could not save conversation history to MongoDB:", error));
}

function queueConversationDelete(conversationId) {
  conversationSyncChain = conversationSyncChain
    .catch(() => {})
    .then(() => requestJson(`/conversations/${encodeURIComponent(conversationId)}`, {
      method: "DELETE",
    }))
    .catch((error) => console.warn("Could not delete conversation history from MongoDB:", error));
}

async function restoreConversationHistoryFromMongo() {
  try {
    const remoteRecords = await requestJson("/conversations");
    const activeIdBeforeRestore = activeConversationId;
    const localRecords = [...conversations];
    const merged = new Map(normalizeConversationRecords(remoteRecords).map((item) => [item.id, item]));
    const localRecordsToUpload = [];

    for (const local of localRecords) {
      const remote = merged.get(local.id);
      if (!remote || local.updatedAt > remote.updatedAt) {
        merged.set(local.id, local);
        if (!remote || local.updatedAt > remote.updatedAt) localRecordsToUpload.push(local);
      }
    }

    conversations = [...merged.values()]
      .sort((left, right) => right.updatedAt.localeCompare(left.updatedAt))
      .slice(0, MAX_SAVED_CONVERSATIONS);
    activeConversationId = conversations.some((item) => item.id === activeIdBeforeRestore)
      ? activeIdBeforeRestore
      : null;
    persistConversationHistory();
    renderConversationHistoryList();

    const retainedIds = new Set(conversations.map((item) => item.id));
    for (const local of localRecordsToUpload) {
      if (retainedIds.has(local.id)) queueConversationSave(local);
    }
    if (activeConversationId && !isAsking && !window.notebookWorkspace?.active) openConversation(activeConversationId);
  } catch (error) {
    console.warn("Could not load conversation history from MongoDB; using the browser cache:", error);
  }
}

function formatConversationDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return new Intl.DateTimeFormat("vi-VN", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" }).format(date);
}

function renderConversationHistoryList() {
  const savedConversations = conversations
    .filter((conversation) => conversation.turns.length)
    .sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));
  elements.conversationHistoryCount.textContent = String(savedConversations.length);
  elements.conversationHistoryItems.replaceChildren();
  if (!savedConversations.length) {
    const empty = document.createElement("p");
    empty.className = "conversation-history-empty";
    empty.textContent = "Các cuộc trò chuyện của bạn sẽ được lưu ở đây.";
    elements.conversationHistoryItems.append(empty);
    return;
  }

  for (const conversation of savedConversations) {
    const row = document.createElement("div");
    row.className = "conversation-history-item" + (conversation.id === activeConversationId ? " is-active" : "");
    const open = document.createElement("button");
    open.className = "conversation-history-open";
    open.type = "button";
    open.dataset.conversationId = conversation.id;
    open.title = conversation.title;
    const title = document.createElement("strong");
    title.textContent = conversation.title;
    const details = document.createElement("span");
    details.textContent = formatConversationDate(conversation.updatedAt) + " · " + conversation.turns.length + " câu hỏi";
    open.append(title, details);

    const remove = document.createElement("button");
    remove.className = "conversation-history-delete";
    remove.type = "button";
    remove.dataset.deleteConversationId = conversation.id;
    remove.textContent = "×";
    remove.title = "Xóa cuộc trò chuyện";
    remove.setAttribute("aria-label", "Xóa cuộc trò chuyện: " + conversation.title);
    row.append(open, remove);
    elements.conversationHistoryItems.append(row);
  }
}

function setConversationHistoryOpen(isOpen) {
  elements.conversationHistoryList.hidden = !isOpen;
  elements.conversationHistoryToggle.setAttribute("aria-expanded", String(isOpen));
}

function showWelcome() {
  if (!welcomeTemplate) return;
  elements.conversation.replaceChildren(welcomeTemplate.cloneNode(true));
  window.workspaceUI?.renderWelcome();
}

function startNewConversation() {
  if (isAsking) return;
  if (window.notebookWorkspace?.active) {
    window.notebookWorkspace.newConversation();
    return;
  }
  activeConversationId = null;
  historyEntries = [];
  activeHistoryId = null;
  showWelcome();
  renderHistoryBar();
  setConversationHistoryOpen(false);
  persistConversationHistory();
  elements.queryInput.focus();
}

function openConversation(id) {
  if (isAsking) return;
  const conversation = conversations.find((item) => item.id === id);
  if (!conversation) return;
  activeConversationId = conversation.id;
  historyEntries = conversation.turns;
  activeHistoryId = historyEntries.at(-1)?.id || null;
  elements.conversation.replaceChildren();
  if (!historyEntries.length) showWelcome();
  for (const turn of historyEntries) {
    appendUserMessage(turn.query, turn.id, false);
    if (turn.response) appendAnswer(turn.response, turn.id, false);
    else if (turn.error) appendErrorMessage(turn.error, turn.id, false);
    else appendErrorMessage("Tra cứu chưa hoàn tất. Bạn có thể gửi lại câu hỏi.", turn.id, false);
  }
  renderHistoryBar();
  setConversationHistoryOpen(false);
  persistConversationHistory();
  scrollConversationToEnd();
}

function deleteConversation(id) {
  const conversation = conversations.find((item) => item.id === id);
  if (!conversation || !window.confirm("Xóa cuộc trò chuyện “" + conversation.title + "”?")) return;
  conversations = conversations.filter((item) => item.id !== id);
  queueConversationDelete(id);
  if (activeConversationId === id) {
    activeConversationId = null;
    const nextConversation = conversations
      .filter((item) => item.turns.length)
      .sort((left, right) => right.updatedAt.localeCompare(left.updatedAt))[0];
    if (nextConversation) openConversation(nextConversation.id);
    else startNewConversation();
    return;
  }
  persistConversationHistory();
}

function renderHistoryBar() {
  const visibleEntries = historyEntries.slice(-16);
  elements.historyTrack.replaceChildren();
  previewHistoryId = null;
  if (!visibleEntries.length) {
    elements.historyBar.hidden = true;
    elements.historyBar.classList.remove("is-preview-open");
    return;
  }

  elements.historyBar.hidden = false;
  if (!visibleEntries.some((entry) => entry.id === activeHistoryId)) activeHistoryId = visibleEntries.at(-1).id;
  for (const entry of visibleEntries) {
    const marker = document.createElement("button");
    marker.className = `history-marker${entry.id === activeHistoryId ? " is-active" : ""}`;
    marker.type = "button";
    marker.dataset.historyId = entry.id;
    marker.title = entry.query;
    marker.setAttribute("aria-label", `Xem câu hỏi và câu trả lời: ${entry.query}`);
    if (entry.id === activeHistoryId) marker.setAttribute("aria-current", "true");
    marker.addEventListener("pointerenter", () => openHistoryPreview(entry.id));
    marker.addEventListener("focus", () => openHistoryPreview(entry.id));
    marker.addEventListener("click", () => selectHistoryEntry(entry.id, marker));
    elements.historyTrack.append(marker);
  }
  elements.historyTrack.scrollTop = elements.historyTrack.scrollHeight;
  elements.historyBar.classList.remove("is-preview-open");
  renderActiveHistoryCard();
}

function renderActiveHistoryCard(entryId = activeHistoryId) {
  const activeEntry = historyEntries.find((entry) => entry.id === entryId);
  if (!activeEntry) {
    elements.historyTitle.textContent = "Lịch sử tra cứu";
    elements.historyDescription.textContent = "Câu hỏi gần đây và tóm tắt câu trả lời sẽ xuất hiện ở đây.";
    elements.historyBookmark.disabled = true;
    elements.historyBookmark.classList.remove("is-bookmarked");
    elements.historyBookmark.setAttribute("aria-pressed", "false");
    elements.historyBookmark.setAttribute("aria-label", "Chưa có câu hỏi để ghim");
    elements.historyBookmark.title = "Chưa có câu hỏi để ghim";
    return;
  }

  elements.historyTitle.textContent = activeEntry.query;
  elements.historyTitle.title = activeEntry.query;
  elements.historyDescription.textContent = activeEntry.summary;
  elements.historyBookmark.disabled = false;
  elements.historyBookmark.classList.toggle("is-bookmarked", activeEntry.bookmarked);
  elements.historyBookmark.setAttribute("aria-pressed", String(activeEntry.bookmarked));
  elements.historyBookmark.setAttribute("aria-label", activeEntry.bookmarked ? "Bỏ ghim câu hỏi" : "Ghim câu hỏi trong phiên này");
  elements.historyBookmark.title = activeEntry.bookmarked ? "Bỏ ghim câu hỏi" : "Ghim câu hỏi trong phiên này";
}

function positionHistoryPreview() {
  if (elements.historyBar.hidden) return;
  elements.historyCard.style.top = "50%";
}

function setActiveHistoryEntry(id) {
  if (!id) return;
  activeHistoryId = id;
  elements.historyTrack.querySelectorAll(".history-marker").forEach((item) => {
    const isActive = item.dataset.historyId === id;
    item.classList.toggle("is-active", isActive);
    if (isActive) item.setAttribute("aria-current", "true");
    else item.removeAttribute("aria-current");
  });
  renderActiveHistoryCard(previewHistoryId || activeHistoryId);
}

function openHistoryPreview(id) {
  previewHistoryId = id;
  renderActiveHistoryCard(id);
  elements.historyBar.classList.add("is-preview-open");
  positionHistoryPreview();
}

function closeHistoryPreview(event) {
  if (event?.relatedTarget && elements.historyBar.contains(event.relatedTarget)) return;
  elements.historyBar.classList.remove("is-preview-open");
  previewHistoryId = null;
  renderActiveHistoryCard();
}

function addHistoryEntry(query) {
  let conversation = conversations.find((item) => item.id === activeConversationId);
  if (!conversation) {
    const now = new Date().toISOString();
    conversation = {
      id: makeConversationId(),
      title: query.slice(0, 120),
      createdAt: now,
      updatedAt: now,
      turns: [],
    };
    conversations.push(conversation);
    activeConversationId = conversation.id;
    historyEntries = conversation.turns;
  }
  const entry = {
    id: "turn-" + Date.now() + "-" + (++historySequence),
    query,
    summary: "Đang tìm trong tài liệu…",
    bookmarked: false,
    response: null,
    error: "",
  };
  historyEntries.push(entry);
  conversation.updatedAt = new Date().toISOString();
  activeHistoryId = entry.id;
  persistConversationHistory(conversation.id);
  renderHistoryBar();
  return entry;
}

function updateHistorySummary(id, summary) {
  const entry = historyEntries.find((item) => item.id === id);
  if (!entry) return;
  entry.summary = String(summary || "Không có nội dung tóm tắt.").slice(0, 2_000);
  if (id === activeHistoryId || id === previewHistoryId) {
    renderActiveHistoryCard(previewHistoryId || activeHistoryId);
    positionHistoryPreview();
  }
  const conversation = conversations.find((item) => item.id === activeConversationId);
  if (conversation) {
    conversation.updatedAt = new Date().toISOString();
    persistConversationHistory(conversation.id);
  }
}

function selectHistoryEntry(id, marker) {
  elements.historyBar.classList.remove("is-preview-open");
  previewHistoryId = null;
  marker?.blur();
  setActiveHistoryEntry(id);
  const message = [...elements.conversation.querySelectorAll(".message-user")]
    .find((item) => item.dataset.historyId === id);
  message?.scrollIntoView({ behavior: "smooth", block: "center" });
}

function updateHistorySelectionFromScroll() {
  const visibleIds = new Set([...elements.historyTrack.querySelectorAll(".history-marker")]
    .map((marker) => marker.dataset.historyId));
  const messages = [...elements.conversation.querySelectorAll(".message-user")]
    .filter((message) => visibleIds.has(message.dataset.historyId));
  if (!messages.length) return;

  const conversationTop = elements.conversation.getBoundingClientRect().top;
  const selectionLine = conversationTop + elements.conversation.clientHeight * 0.5;
  let currentMessage = messages[0];
  for (const message of messages) {
    if (message.getBoundingClientRect().top > selectionLine) break;
    currentMessage = message;
  }
  if (currentMessage.dataset.historyId !== activeHistoryId) {
    setActiveHistoryEntry(currentMessage.dataset.historyId);
  }
}

function scheduleHistorySelectionUpdate() {
  if (historyScrollFrame) return;
  historyScrollFrame = requestAnimationFrame(() => {
    historyScrollFrame = 0;
    updateHistorySelectionFromScroll();
  });
}

function toggleHistoryBookmark() {
  const entry = historyEntries.find((item) => item.id === (previewHistoryId || activeHistoryId));
  if (!entry) return;
  entry.bookmarked = !entry.bookmarked;
  const conversation = conversations.find((item) => item.id === activeConversationId);
  if (conversation) conversation.updatedAt = new Date().toISOString();
  renderActiveHistoryCard(previewHistoryId || activeHistoryId);
  persistConversationHistory(activeConversationId);
}

function appendUserMessage(query, historyId, shouldScroll = true) {
  removeWelcome();
  const message = document.createElement("article");
  message.className = "message message-user";
  message.dataset.historyId = historyId;
  const label = document.createElement("span");
  label.className = "message-label";
  label.textContent = "Bạn";
  const bubble = document.createElement("div");
  bubble.className = "user-bubble";
  bubble.textContent = query;
  message.append(label, bubble);
  elements.conversation.append(message);
  if (shouldScroll) scrollConversationToEnd();
}

function appendTypingIndicator() {
  const node = document.createElement("article");
  node.className = "message message-assistant";
  node.id = "typing-indicator";
  const label = document.createElement("span");
  label.className = "message-label";
  label.textContent = "Financial RAG";
  const indicator = document.createElement("span");
  indicator.className = "typing-indicator";
  indicator.append(document.createTextNode("Đang tìm trong tài liệu "));
  const dots = document.createElement("span");
  dots.className = "typing-dots";
  dots.setAttribute("aria-hidden", "true");
  for (let index = 0; index < 3; index += 1) dots.append(document.createElement("i"));
  indicator.append(dots);
  node.append(label, indicator);
  elements.conversation.append(node);
  scrollConversationToEnd();
}

function appendErrorMessage(errorMessage, historyId, shouldScroll = true) {
  const message = document.createElement("article");
  message.className = "message message-assistant";
  message.dataset.historyId = historyId;
  const label = document.createElement("span");
  label.className = "message-label";
  label.textContent = "Financial RAG";
  const card = document.createElement("div");
  card.className = "error-card";
  card.textContent = errorMessage;
  message.append(label, card);
  elements.conversation.append(message);
  const entry = historyEntries.find((item) => item.id === historyId);
  if (entry) {
    entry.response = null;
    entry.error = errorMessage;
    updateHistorySummary(historyId, errorMessage);
  }
  if (shouldScroll) scrollConversationToEnd();
}

function appendAnswer(response, historyId, shouldScroll = true) {
  const historyEntry = historyEntries.find((item) => item.id === historyId);
  if (historyEntry) {
    historyEntry.response = storedResponse(response);
    historyEntry.error = "";
  }
  const message = document.createElement("article");
  message.className = "message message-assistant";
  message.dataset.historyId = historyId;
  const label = document.createElement("span");
  label.className = "message-label";
  label.textContent = "Financial RAG";
  const card = document.createElement("div");
  card.className = "answer-card";
  const answer = response.reader_mode && window.notebookWorkspace
    ? window.notebookWorkspace.renderExtractive(response.answer || "Không có nội dung trả lời.")
    : renderAnswerContent(response.answer || "Không có nội dung trả lời.");
  card.append(answer);

  const validation = response.validation || {};
  const badge = document.createElement("span");
  badge.className = `validation-badge${validation.is_valid ? "" : " is-invalid"}`;
  badge.textContent = validation.is_valid ? "Số liệu và trích dẫn đã qua kiểm tra" : "Cần đối chiếu câu trả lời với nguồn";
  if (response.reader_mode === "source_extract") badge.textContent = "Trích nguyên văn · mở nguồn để đối chiếu";
  if (validation.is_valid && response.answer_status === "partial") {
    badge.classList.add("is-partial");
    badge.textContent = response.reader_mode ? "Trích nguồn · phạm vi chưa đầy đủ" : "Số liệu đã kiểm tra · Bằng chứng thành phần chưa đầy đủ";
  } else if (response.answer_status === "insufficient_evidence") {
    badge.classList.add("is-partial");
    badge.textContent = "Chưa đủ bằng chứng để trả lời";
  }
  card.append(badge);
  const coverage = response.analysis_coverage?.missing_components || [];
  if (coverage.length) {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = `Thiếu bằng chứng: ${coverage.length} mục`;
    details.append(summary);
    for (const item of coverage) {
      const line = document.createElement("p");
      line.textContent = item;
      details.append(line);
    }
    card.append(details);
  }
  if (response.source_snapshot?.policy) {
    const convention = document.createElement("p");
    convention.className = "validation-details";
    convention.textContent = `Chính sách số liệu: ${response.source_snapshot.policy === "latest_restated" ? "bản trình bày mới nhất cho kỳ được hỏi" : "báo cáo của kỳ được hỏi"}. Kiểm tra máy không thay thế rà soát độc lập.`;
    if (response.notebook_id) {
      convention.textContent = response.source_snapshot.policy === "latest_restated"
        ? "Dùng bản trình bày mới nhất trong các nguồn đã ghim của notebook."
        : "Đối chiếu các báo cáo đã ghim trong notebook cho kỳ được hỏi.";
      const sourceIds = new Set((response.citations || []).map((citation) => citation.notebook_source_id));
      const sourceYears = [...new Set((response.source_snapshot.documents || [])
        .filter((source) => sourceIds.has(source.id) && source.fiscal_year)
        .map((source) => Number(source.fiscal_year)))];
      const askedYears = response.parsed_query?.fiscal_years?.length ? response.parsed_query.fiscal_years
        : (response.parsed_query?.fiscal_year ? [response.parsed_query.fiscal_year] : []);
      const otherYears = askedYears.filter((year) => !sourceYears.includes(Number(year)));
      if (otherYears.length && sourceYears.length) convention.textContent +=
        ` Số năm ${otherYears.join(", ")} được trích từ báo cáo ${sourceYears.join(", ")} đã chọn.`;
    }
    card.append(convention);
  }
  if (!validation.is_valid && validation.details) {
    const detail = document.createElement("p");
    detail.className = "validation-details";
    detail.textContent = validation.details;
    card.append(detail);
  }

  if (Array.isArray(response.calculations) && response.calculations.length) {
    card.append(renderCalculations(response.calculations));
  }

  if (Array.isArray(response.citations) && response.citations.length) {
    const evidence = document.createElement("section");
    evidence.className = "citation-evidence";
    const citations = document.createElement("div");
    citations.className = "citation-list";
    citations.id = `citation-list-${makeConversationId()}`;

    const citationCount = response.citations.length;
    if (citationCount > 2) {
      const header = document.createElement("div");
      header.className = "citation-evidence-header";
      const count = document.createElement("span");
      count.className = "citation-evidence-count";
      count.textContent = `Dẫn chứng · 2/${citationCount}`;
      const toggle = document.createElement("button");
      toggle.className = "citation-toggle";
      toggle.type = "button";
      toggle.textContent = "Xem tất cả";
      toggle.setAttribute("aria-expanded", "false");
      toggle.setAttribute("aria-controls", citations.id);
      citations.classList.add("is-collapsed");
      toggle.addEventListener("click", () => {
        const expanded = toggle.getAttribute("aria-expanded") === "true";
        toggle.setAttribute("aria-expanded", String(!expanded));
        citations.classList.toggle("is-collapsed", expanded);
        count.textContent = `Dẫn chứng · ${expanded ? 2 : citationCount}/${citationCount}`;
        toggle.textContent = expanded ? "Xem tất cả" : "Thu gọn";
      });
      header.append(count, toggle);
      evidence.append(header);
    }

    for (const citation of response.citations) citations.append(renderCitation(citation, response));
    evidence.append(citations);
    card.append(evidence);
  }

  window.notebookWorkspace?.decorateAnswer(card, response);
  const copyAnswer = document.createElement("button");
  copyAnswer.className = "button copy-answer";
  copyAnswer.type = "button";
  copyAnswer.textContent = "Sao chép";
  copyAnswer.setAttribute("aria-label", "Sao chép câu trả lời");
  copyAnswer.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(response.answer || ""); window.workspaceUI?.notify("Đã sao chép câu trả lời."); }
    catch { window.workspaceUI?.notify("Không sao chép được tự động. Bạn có thể chọn và sao chép nội dung câu trả lời.", true); }
  });
  card.append(copyAnswer);

  message.append(label, card);
  elements.conversation.append(message);
  updateHistorySummary(historyId, response.answer || "Không có nội dung trả lời.");
  if (shouldScroll) scrollConversationToEnd();
}

function renderCalculations(calculations) {
  const labels = {
    absolute_change: "Chênh lệch tuyệt đối",
    growth_rate: "Tỷ lệ thay đổi",
    percentage_point_change: "Chênh lệch điểm phần trăm",
    balance_sheet_reconciliation: "Đối chiếu phương trình kế toán",
    contribution_to_change: "Đóng góp vào biến động",
    component_sum_reconciliation: "Phần dư so với tổng báo cáo",
    contribution_residual: "Biến động chưa đối chiếu",
    share_of_total: "Tỷ trọng trong tổng",
    liabilities_to_assets_percent: "Nợ phải trả / Tổng tài sản",
    equity_to_assets_percent: "Vốn chủ sở hữu / Tổng tài sản",
    gross_margin_percent: "Biên lợi nhuận gộp",
    operating_margin_percent: "Biên lợi nhuận hoạt động",
    net_profit_margin_percent: "Biên lợi nhuận ròng",
  };
  const section = document.createElement("details");
  section.className = "calculation-evidence";
  const summary = document.createElement("summary");
  summary.textContent = `Phép tính đã kiểm tra · ${calculations.length}`;
  section.append(summary);
  const list = document.createElement("ul");
  list.className = "calculation-list";
  for (const calculation of calculations) {
    const item = document.createElement("li");
    const heading = document.createElement("strong");
    heading.textContent = `${labels[calculation.formula_id] || calculation.formula_id}: ${calculation.value} ${calculation.unit}`.trim();
    item.append(heading);
    if (calculation.expression) {
      const expression = document.createElement("span");
      expression.textContent = calculation.expression;
      item.append(expression);
    }
    const roleLabels = {
      prior_value: "Kỳ gốc", current_value: "Kỳ hiện tại",
      numerator: "Tử số", denominator: "Mẫu số",
      prior_numerator: "Tử số kỳ gốc", prior_denominator: "Mẫu số kỳ gốc",
      current_numerator: "Tử số kỳ hiện tại", current_denominator: "Mẫu số kỳ hiện tại",
      prior_component: "Thành phần kỳ gốc", current_component: "Thành phần kỳ hiện tại",
      prior_total: "Tổng kỳ gốc", current_total: "Tổng kỳ hiện tại",
      total_value: "Tổng báo cáo", total_assets: "Tổng tài sản",
      total_liabilities: "Tổng nợ", total_equity: "Tổng vốn chủ sở hữu",
    };
    for (const input of Array.isArray(calculation.inputs) ? calculation.inputs : []) {
      const fact = document.createElement("span");
      const role = roleLabels[input.role] || (input.role?.startsWith("prior_component_")
        ? "Thành phần kỳ gốc" : input.role?.startsWith("current_component_")
          ? "Thành phần kỳ hiện tại" : "Thành phần");
      fact.textContent = `${role} · ${input.label} · ${input.year}: ${input.value} ${input.unit} [${input.citation_id}]`;
      item.append(fact);
    }
    const sourceIds = Array.isArray(calculation.input_citation_ids)
      ? [...new Set(calculation.input_citation_ids)]
      : [];
    if (sourceIds.length) {
      const sources = document.createElement("span");
      sources.textContent = `Nguồn đầu vào: ${sourceIds.map((id) => `[${id}]`).join(" · ")}`;
      item.append(sources);
    }
    list.append(item);
  }
  section.append(list);
  return section;
}

function markdownCells(line) {
  return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());
}

function isMarkdownTableSeparator(line) {
  const cells = markdownCells(line);
  return cells.length > 1 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function renderAnswerContent(markdown) {
  const content = document.createElement("div");
  content.className = "answer-text answer-content";
  const lines = String(markdown).replace(/\r\n?/g, "\n").split("\n");
  let paragraphLines = [];
  let list = null;

  const flushParagraph = () => {
    if (!paragraphLines.length) return;
    const paragraph = document.createElement("p");
    paragraph.className = "answer-paragraph";
    paragraphLines.forEach((line, index) => {
      if (index) paragraph.append(document.createElement("br"));
      paragraph.append(document.createTextNode(line));
    });
    content.append(paragraph);
    paragraphLines = [];
  };

  const flushList = () => {
    if (!list) return;
    content.append(list);
    list = null;
  };

  for (let index = 0; index < lines.length;) {
    const line = lines[index];
    const nextLine = lines[index + 1] || "";
    if (line.trim().startsWith("|") && nextLine.trim().startsWith("|") && isMarkdownTableSeparator(nextLine)) {
      flushParagraph();
      flushList();
      const table = document.createElement("table");
      table.className = "answer-table";
      const header = document.createElement("thead");
      const headerRow = document.createElement("tr");
      for (const value of markdownCells(line)) {
        const cell = document.createElement("th");
        cell.scope = "col";
        cell.textContent = value;
        headerRow.append(cell);
      }
      header.append(headerRow);
      table.append(header);
      const body = document.createElement("tbody");
      index += 2;
      while (index < lines.length && lines[index].trim().startsWith("|")) {
        if (!isMarkdownTableSeparator(lines[index])) {
          const row = document.createElement("tr");
          for (const value of markdownCells(lines[index])) {
            const cell = document.createElement("td");
            cell.textContent = value;
            row.append(cell);
          }
          body.append(row);
        }
        index += 1;
      }
      table.append(body);
      const wrapper = document.createElement("div");
      wrapper.className = "answer-table-wrap";
      wrapper.append(table);
      content.append(wrapper);
      continue;
    }

    const bullet = line.match(/^\s*[-*]\s+(.+)$/);
    if (bullet) {
      flushParagraph();
      if (!list) {
        list = document.createElement("ul");
        list.className = "answer-list";
      }
      const item = document.createElement("li");
      item.textContent = bullet[1];
      list.append(item);
      index += 1;
      continue;
    }

    const heading = line.match(/^#{1,4}\s+(.+)$/);
    if (heading) {
      flushParagraph();
      flushList();
      const title = document.createElement("h3");
      title.className = "answer-heading";
      title.textContent = heading[1];
      content.append(title);
      index += 1;
      continue;
    }

    if (!line.trim()) {
      flushParagraph();
      flushList();
      index += 1;
      continue;
    }

    flushList();
    paragraphLines.push(line);
    index += 1;
  }
  flushParagraph();
  flushList();
  return content;
}

function renderCitation(citation, response = {}) {
  const details = document.createElement("details");
  details.className = "citation-card";
  const summary = document.createElement("summary");
  summary.className = "citation-summary";
  const id = document.createElement("span");
  id.className = "citation-id";
  id.textContent = `[${citation.citation_id || "?"}]`;
  const source = document.createElement("span");
  source.className = "citation-source";
  const file = document.createElement("strong");
  file.textContent = citation.file_name || "Nguồn tài liệu";
  file.title = citation.file_name || "";
  const metadata = [citation.page ? `Trang ${citation.page}` : null, citation.unit, citation.section].filter(Boolean);
  const meta = document.createElement("span");
  meta.textContent = metadata.join(" · ") || "Đoạn nguồn được truy xuất";
  source.append(file, meta);
  const open = document.createElement("span");
  open.className = "citation-open";
  open.setAttribute("aria-hidden", "true");
  open.textContent = "⌄";
  summary.append(id, source, open);
  details.append(summary);
  if (citation.snippet) {
    const snippet = document.createElement("div");
    snippet.className = "citation-snippet";
    snippet.textContent = citation.snippet;
    details.append(snippet);
  }
  if (citation.notebook_source_id && response.notebook_id && response.run_id) {
    const button = document.createElement("button");
    button.className = "button notebook-source-open";
    button.type = "button";
    button.textContent = citation.block_id ? "Mở đúng đoạn nguồn" : "Mở nguồn · vị trí chưa xác định";
    button.addEventListener("click", () => window.notebookWorkspace.openSource(response.notebook_id,
      citation.notebook_source_id, { runId: response.run_id, blockId: citation.block_id }));
    details.append(button);
  }
  return details;
}

async function askQuestion(event) {
  event.preventDefault();
  const query = elements.queryInput.value.trim();
  if (!query || isAsking || window.notebookWorkspace?.busy) return;
  if (window.notebookWorkspace?.active) {
    await window.notebookWorkspace.ask(query);
    return;
  }

  isAsking = true;
  elements.askButton.disabled = true;
  elements.queryInput.disabled = true;
  elements.newConversation.disabled = true;
  elements.conversationHistoryToggle.disabled = true;
  window.workspaceUI?.syncAskState();
  setConversationHistoryOpen(false);
  const historyEntry = addHistoryEntry(query);
  appendUserMessage(query, historyEntry.id);
  elements.queryInput.value = "";
  appendTypingIndicator();

  try {
    const response = await requestJson("/answer", {
      method: "POST",
      body: JSON.stringify({ query, top_k: Number(elements.topK.value),
        source_policy: document.querySelector("#source-policy")?.value || "source_report" }),
    });
    document.querySelector("#typing-indicator")?.remove();
    appendAnswer(response, historyEntry.id);
  } catch (error) {
    document.querySelector("#typing-indicator")?.remove();
    appendErrorMessage(error.message, historyEntry.id);
  } finally {
    isAsking = false;
    elements.askButton.disabled = false;
    elements.queryInput.disabled = false;
    elements.newConversation.disabled = false;
    elements.conversationHistoryToggle.disabled = false;
    elements.queryInput.focus();
    window.workspaceUI?.syncAskState();
  }
}

elements.refresh.addEventListener("click", refreshWorkspace);
elements.llmChoice.addEventListener("change", saveLlmChoice);
window.addEventListener("storage", (event) => {
  if (event.key === LLM_PROFILES_STORAGE_KEY || event.key === LLM_SELECTION_STORAGE_KEY) {
    renderLlmChoice();
  }
});
elements.newConversation.addEventListener("click", startNewConversation);
elements.conversationHistoryToggle.addEventListener("click", () => {
  setConversationHistoryOpen(elements.conversationHistoryList.hidden);
  if (!elements.conversationHistoryList.hidden) renderConversationHistoryList();
});
elements.conversationHistoryItems.addEventListener("click", (event) => {
  const deleteButton = event.target.closest("[data-delete-conversation-id]");
  if (deleteButton) {
    deleteConversation(deleteButton.dataset.deleteConversationId);
    return;
  }
  const openButton = event.target.closest("[data-conversation-id]");
  if (openButton) openConversation(openButton.dataset.conversationId);
});
document.addEventListener("pointerdown", (event) => {
  if (!elements.conversationHistoryToggle.parentElement.contains(event.target)) setConversationHistoryOpen(false);
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !elements.conversationHistoryList.hidden) {
    setConversationHistoryOpen(false);
    elements.conversationHistoryToggle.focus();
  }
});
elements.historyBookmark.addEventListener("click", toggleHistoryBookmark);
elements.historyBar.addEventListener("pointerleave", closeHistoryPreview);
elements.historyBar.addEventListener("focusout", closeHistoryPreview);
elements.conversation.addEventListener("scroll", scheduleHistorySelectionUpdate, { passive: true });
elements.chooseFiles.addEventListener("click", () => elements.fileInput.click());
elements.fileInput.addEventListener("change", (event) => addFiles(event.target.files));
elements.uploadForm.addEventListener("submit", uploadSelectedFiles);
elements.dropzone.addEventListener("click", (event) => {
  if (event.target.closest("button")) return;
  elements.fileInput.click();
});
elements.dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  elements.dropzone.classList.add("is-dragging");
});
elements.dropzone.addEventListener("dragleave", (event) => {
  if (!elements.dropzone.contains(event.relatedTarget)) elements.dropzone.classList.remove("is-dragging");
});
elements.dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  elements.dropzone.classList.remove("is-dragging");
  addFiles(event.dataTransfer.files);
});
elements.queryForm.addEventListener("submit", askQuestion);
elements.queryInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    elements.queryForm.requestSubmit();
  }
});
elements.conversation.addEventListener("click", (event) => {
  const suggestion = event.target.closest(".suggestion");
  if (!suggestion) return;
  elements.queryInput.value = suggestion.dataset.query || "";
  elements.queryInput.focus();
  window.workspaceUI?.syncAskState();
});

loadConversationHistory();
renderLlmChoice();
renderConversationHistoryList();
if (activeConversationId) openConversation(activeConversationId);
else renderHistoryBar();
restoreConversationHistoryFromMongo();
refreshWorkspace();
