const LLM_TEST_ENDPOINT = "/api/v1/financial/llm/test-connection";
const LLM_PROFILES_STORAGE_KEY = "financial-rag.llm-profiles.v1";
const LLM_SELECTION_STORAGE_KEY = "financial-rag.llm-selection.v1";
const LLM_DRAFT_STORAGE_KEY = "financial-rag.llm-draft.v1";

const providerPresets = {
  openai: { label: "OpenAI", baseUrl: "https://api.openai.com/v1" },
  openrouter: { label: "OpenRouter", baseUrl: "https://openrouter.ai/api/v1" },
  deepseek: { label: "DeepSeek", baseUrl: "https://api.deepseek.com" },
  custom: { label: "API tương thích OpenAI", baseUrl: "" },
};

const fields = {
  providerButtons: [...document.querySelectorAll("[data-provider]")],
  ollamaPanel: document.querySelector("#ollama-panel"),
  apiPanel: document.querySelector("#api-panel"),
  ollamaUrl: document.querySelector("#ollama-url"),
  ollamaModel: document.querySelector("#ollama-model"),
  apiPreset: document.querySelector("#api-preset"),
  apiUrl: document.querySelector("#api-url"),
  apiModel: document.querySelector("#api-model"),
  apiKey: document.querySelector("#api-key"),
  apiModelOptions: document.querySelector("#api-model-options"),
  status: document.querySelector("#connection-status"),
  statusText: document.querySelector("#connection-status span:last-child"),
  testButton: document.querySelector("#test-connection"),
  saveButton: document.querySelector("#save-profile"),
  toggleApiKey: document.querySelector("#toggle-api-key"),
  profiles: document.querySelector("#saved-profiles"),
  profileCount: document.querySelector("#profile-count"),
};

let selectedProvider = "ollama";
let lastSuccessfulFingerprint = "";

function readJson(key, fallback) {
  try {
    const value = JSON.parse(localStorage.getItem(key) || "null");
    return value ?? fallback;
  } catch {
    return fallback;
  }
}

function readProfiles() {
  const saved = readJson(LLM_PROFILES_STORAGE_KEY, []);
  return Array.isArray(saved) ? saved.filter((profile) => profile && profile.id && profile.model) : [];
}

function currentConfig() {
  if (selectedProvider === "ollama") {
    return {
      provider: "ollama",
      provider_label: "Ollama",
      base_url: fields.ollamaUrl.value.trim(),
      model: fields.ollamaModel.value.trim(),
      api_key: "",
    };
  }
  const preset = fields.apiPreset.value;
  return {
    provider: "openai_compatible",
    provider_label: providerPresets[preset]?.label || providerPresets.custom.label,
    preset,
    base_url: fields.apiUrl.value.trim(),
    model: fields.apiModel.value.trim(),
    api_key: fields.apiKey.value,
  };
}

function configFingerprint(config = currentConfig()) {
  return JSON.stringify({
    provider: config.provider,
    preset: config.preset || "",
    base_url: config.base_url,
    model: config.model,
    api_key: config.api_key || "",
  });
}

function saveDraft() {
  const draft = {
    provider: selectedProvider,
    ollama_url: fields.ollamaUrl.value.trim(),
    ollama_model: fields.ollamaModel.value,
    api_preset: fields.apiPreset.value,
    api_url: fields.apiUrl.value.trim(),
    api_model: fields.apiModel.value.trim(),
  };
  try {
    localStorage.setItem(LLM_DRAFT_STORAGE_KEY, JSON.stringify(draft));
  } catch {
    // Draft persistence is optional; the current form remains usable.
  }
}

function setStatus(state, message) {
  fields.status.classList.remove("is-checking", "is-success", "is-warning", "is-error");
  if (state) fields.status.classList.add(`is-${state}`);
  fields.statusText.textContent = message;
}

function invalidateConnectionCheck() {
  lastSuccessfulFingerprint = "";
  setStatus("", "Cấu hình đã đổi · hãy kiểm tra lại kết nối.");
}

function setProvider(provider) {
  selectedProvider = provider === "openai_compatible" ? "openai_compatible" : "ollama";
  fields.providerButtons.forEach((button) => {
    const isSelected = button.dataset.provider === selectedProvider;
    button.classList.toggle("is-selected", isSelected);
    button.setAttribute("aria-pressed", String(isSelected));
  });
  fields.ollamaPanel.hidden = selectedProvider !== "ollama";
  fields.apiPanel.hidden = selectedProvider !== "openai_compatible";
  saveDraft();
  invalidateConnectionCheck();
}

function applyPreset(preset, force = false) {
  const selected = providerPresets[preset] ? preset : "custom";
  if (force || selected !== "custom") fields.apiUrl.value = providerPresets[selected].baseUrl;
  fields.apiPreset.value = selected;
  saveDraft();
  invalidateConnectionCheck();
}

function restoreDraft() {
  const draft = readJson(LLM_DRAFT_STORAGE_KEY, {});
  if (typeof draft.ollama_url === "string" && draft.ollama_url) fields.ollamaUrl.value = draft.ollama_url;
  if (typeof draft.ollama_model === "string" && draft.ollama_model) fields.ollamaModel.value = draft.ollama_model;
  if (typeof draft.api_model === "string") fields.apiModel.value = draft.api_model;

  const preset = providerPresets[draft.api_preset] ? draft.api_preset : "openai";
  fields.apiPreset.value = preset;
  fields.apiUrl.value = typeof draft.api_url === "string" && draft.api_url
    ? draft.api_url
    : providerPresets[preset].baseUrl;
  setProvider(draft.provider);
  setStatus("", "Chưa kiểm tra kết nối.");
}

function addOllamaModels(models) {
  if (!models.length) return;
  const selected = fields.ollamaModel.value;
  const uniqueModels = [...new Set(models)];
  if (selected && !uniqueModels.includes(selected)) uniqueModels.unshift(selected);
  const options = uniqueModels.map((model) => {
    const option = new Option(model, model);
    if (!models.includes(model)) option.textContent = `${model} · chưa thấy trên Ollama`;
    return option;
  });
  fields.ollamaModel.replaceChildren(...options);
  fields.ollamaModel.value = selected;
}

function addApiModels(models) {
  fields.apiModelOptions.replaceChildren(...models.map((model) => new Option(model, model)));
}

async function testConnection() {
  const config = currentConfig();
  if (!config.base_url) {
    setStatus("error", "Nhập địa chỉ kết nối trước khi kiểm tra.");
    return;
  }

  const fingerprint = configFingerprint(config);
  fields.testButton.disabled = true;
  setStatus("checking", "Đang kiểm tra endpoint và danh sách mô hình…");
  try {
    const response = await fetch(LLM_TEST_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
    const result = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = result?.detail;
      throw new Error(typeof detail === "string" ? detail : `Yêu cầu thất bại (HTTP ${response.status}).`);
    }

    if (selectedProvider === "ollama") addOllamaModels(Array.isArray(result.models) ? result.models : []);
    else addApiModels(Array.isArray(result.models) ? result.models : []);

    if (!result.connected) {
      lastSuccessfulFingerprint = "";
      setStatus("error", result.message || "Không thể kết nối đến mô hình.");
    } else if (result.selected_model_available === false) {
      lastSuccessfulFingerprint = fingerprint;
      setStatus("warning", `${result.message} Chọn model khả dụng trong danh sách rồi lưu.`);
    } else {
      lastSuccessfulFingerprint = fingerprint;
      setStatus("success", result.message || "Kết nối thành công.");
    }
  } catch (error) {
    lastSuccessfulFingerprint = "";
    setStatus("error", error.message || "Không thể kiểm tra kết nối.");
  } finally {
    fields.testButton.disabled = false;
  }
}

function formatTestDate(value) {
  if (!value) return "Chưa kiểm tra";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Chưa kiểm tra";
  return `Đã kiểm tra ${new Intl.DateTimeFormat("vi-VN", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" }).format(date)}`;
}

function renderProfiles() {
  const profiles = readProfiles().sort((left, right) => (right.updated_at || "").localeCompare(left.updated_at || ""));
  fields.profileCount.textContent = `${profiles.length} cấu hình`;
  fields.profiles.replaceChildren();
  if (!profiles.length) {
    const empty = document.createElement("p");
    empty.className = "saved-empty";
    empty.textContent = "Cấu hình đã lưu sẽ xuất hiện tại đây và trong ô chọn ở thanh chat.";
    fields.profiles.append(empty);
    return;
  }

  for (const profile of profiles) {
    const row = document.createElement("div");
    row.className = "saved-profile";
    const copy = document.createElement("div");
    copy.className = "saved-profile-copy";
    const name = document.createElement("strong");
    name.textContent = profile.label || `${profile.provider_label || "LLM"} · ${profile.model}`;
    name.title = name.textContent;
    const detail = document.createElement("span");
    detail.textContent = `${profile.model} · ${formatTestDate(profile.last_test_at)}`;
    copy.append(name, detail);
    const remove = document.createElement("button");
    remove.className = "profile-delete";
    remove.type = "button";
    remove.dataset.deleteProfile = String(profile.id);
    remove.textContent = "×";
    remove.title = "Xóa cấu hình";
    remove.setAttribute("aria-label", `Xóa cấu hình ${name.textContent}`);
    row.append(copy, remove);
    fields.profiles.append(row);
  }
}

function saveProfile() {
  const config = currentConfig();
  if (!config.base_url || !config.model) {
    setStatus("error", "Cần nhập địa chỉ kết nối và tên mô hình trước khi lưu.");
    return;
  }

  const existingProfiles = readProfiles();
  const existing = existingProfiles.find((profile) =>
    profile.provider === config.provider
    && profile.base_url === config.base_url
    && profile.model === config.model,
  );
  const profile = {
    id: existing?.id || (window.crypto?.randomUUID ? window.crypto.randomUUID() : `llm-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`),
    provider: config.provider,
    provider_label: config.provider_label,
    preset: config.preset || "",
    base_url: config.base_url,
    model: config.model,
    label: `${config.provider_label} · ${config.model}`,
    updated_at: new Date().toISOString(),
    last_test_at: lastSuccessfulFingerprint === configFingerprint(config) ? new Date().toISOString() : "",
  };
  const profiles = [profile, ...existingProfiles.filter((item) => item.id !== profile.id)];
  try {
    localStorage.setItem(LLM_PROFILES_STORAGE_KEY, JSON.stringify(profiles));
    saveDraft();
    renderProfiles();
    setStatus("success", "Đã lưu cấu hình trên trình duyệt này; chat chưa chuyển sang mô hình này.");
  } catch {
    setStatus("error", "Không lưu được cấu hình vào bộ nhớ trình duyệt.");
  }
}

function deleteProfile(id) {
  const profiles = readProfiles().filter((profile) => String(profile.id) !== id);
  try {
    localStorage.setItem(LLM_PROFILES_STORAGE_KEY, JSON.stringify(profiles));
    if (localStorage.getItem(LLM_SELECTION_STORAGE_KEY) === id) {
      localStorage.setItem(LLM_SELECTION_STORAGE_KEY, "default");
    }
  } catch {
    setStatus("error", "Không xóa được cấu hình khỏi bộ nhớ trình duyệt.");
  }
  renderProfiles();
}

fields.providerButtons.forEach((button) => {
  button.addEventListener("click", () => setProvider(button.dataset.provider));
});
fields.apiPreset.addEventListener("change", () => applyPreset(fields.apiPreset.value));
fields.apiUrl.addEventListener("input", () => {
  if (fields.apiPreset.value !== "custom") fields.apiPreset.value = "custom";
  saveDraft();
  invalidateConnectionCheck();
});
for (const input of [fields.ollamaUrl, fields.ollamaModel, fields.apiModel, fields.apiKey]) {
  input.addEventListener("input", () => {
    saveDraft();
    invalidateConnectionCheck();
  });
  input.addEventListener("change", () => {
    saveDraft();
    invalidateConnectionCheck();
  });
}
fields.testButton.addEventListener("click", testConnection);
fields.saveButton.addEventListener("click", saveProfile);
fields.toggleApiKey.addEventListener("click", () => {
  const isVisible = fields.apiKey.type === "text";
  fields.apiKey.type = isVisible ? "password" : "text";
  fields.toggleApiKey.textContent = isVisible ? "Hiện" : "Ẩn";
  fields.toggleApiKey.setAttribute("aria-pressed", String(!isVisible));
});
fields.profiles.addEventListener("click", (event) => {
  const removeButton = event.target.closest("[data-delete-profile]");
  if (removeButton) deleteProfile(removeButton.dataset.deleteProfile);
});
window.addEventListener("storage", (event) => {
  if (event.key === LLM_PROFILES_STORAGE_KEY) renderProfiles();
});

restoreDraft();
renderProfiles();
