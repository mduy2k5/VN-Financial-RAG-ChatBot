/* Presentation state only; notebook records and source scope stay on the service. */
(() => {
  const $ = (selector) => document.querySelector(selector);
  let notebook = null;
  let selectedIds = [];
  let toastTimer;

  function selectSourceTab(library) {
    document.body.dataset.sourceView = library ? "library" : "selected";
    $("#source-selected-panel").hidden = library;
    $("#source-library-panel").hidden = !library;
    for (const [id, selected] of [["source-tab-selected", !library], ["source-tab-library", library]]) {
      $("#" + id).setAttribute("aria-selected", String(selected));
      $("#" + id).tabIndex = selected ? 0 : -1;
    }
  }

  function selectWorkspaceTab(view) {
    document.body.dataset.workspaceView = view;
    document.querySelectorAll("[data-workspace-tab]").forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.workspaceTab === view));
    });
  }

  function showSources(library = false) {
    selectWorkspaceTab("sources");
    selectSourceTab(library);
  }

  function showChat() { selectWorkspaceTab("chat"); }

  function openCreate() {
    if (isAsking || window.notebookWorkspace?.busy) return;
    $("#create-notebook-dialog").showModal();
    $("#notebook-name").focus();
  }

  function openNotes() {
    if (!notebook) { notify("Tạo hoặc chọn notebook để lưu ghi chú."); return; }
    const dialog = $("#notes-dialog");
    if (!dialog.open) dialog.showModal();
    document.querySelector('[data-workspace-tab="notes"]').setAttribute("aria-pressed", "true");
    document.querySelector('[data-workspace-tab="' + document.body.dataset.workspaceView + '"]').setAttribute("aria-pressed", "false");
  }

  function notify(text, error = false) {
    clearTimeout(toastTimer);
    const toast = $("#workspace-toast");
    toast.textContent = text;
    toast.classList.toggle("is-error", error);
    toast.hidden = false;
    toastTimer = setTimeout(() => { toast.hidden = true; }, error ? 8500 : 5500);
  }

  function syncAskState() {
    const locked = isAsking || Boolean(window.notebookWorkspace?.busy);
    const emptyScope = notebook && !selectedIds.length;
    elements.askButton.disabled = locked || Boolean(emptyScope) || !elements.queryInput.value.trim();
    elements.askButton.querySelector("span").textContent = isAsking ? "Đang tìm…" : "Gửi";
    $("#composer-hint").textContent = emptyScope ? "Chọn ít nhất một nguồn để đặt câu hỏi" : "Enter gửi · Shift + Enter xuống dòng";
    $("#open-create-notebook").disabled = locked;
    document.querySelectorAll("[data-attach-source]").forEach((button) => {
      button.disabled = locked || button.dataset.attached === "true";
    });
  }

  function renderWelcome() {
    const welcome = $("#welcome");
    if (!welcome) return;
    const sources = (notebook?.sources || []).filter((source) => selectedIds.includes(source.id));
    const list = welcome.querySelector(".suggestion-list");
    const action = welcome.querySelector(".welcome-source-action");
    let suggestions;
    if (notebook && !sources.length) {
      welcome.querySelector("h3").textContent = "Thêm nguồn, bắt đầu khám phá";
      welcome.querySelector(".welcome-description").textContent = "Chọn báo cáo từ kho hoặc nạp tài liệu mới. Câu trả lời sẽ chỉ dựa trên nguồn bạn chọn.";
      suggestions = [];
      action.textContent = "Thêm nguồn từ kho →";
      action.dataset.action = "browse-library";
    } else if (notebook) {
      welcome.querySelector("h3").textContent = "Bạn muốn biết điều gì?";
      welcome.querySelector(".welcome-description").textContent = `Khám phá ${sources.length} nguồn đã chọn. Mỗi câu trả lời giữ bằng chứng để bạn đọc và đối chiếu.`;
      const company = [...new Set(sources.map((source) => source.company_code).filter(Boolean))];
      const years = [...new Set(sources.map((source) => source.fiscal_year).filter(Boolean))].sort((a, b) => b - a);
      const scope = [company.length === 1 ? company[0] : "", years.length === 1 ? `năm ${years[0]}` : ""].filter(Boolean).join(" ");
      suggestions = [{label: "Tổng quan nguồn đã chọn", query: "Tóm tắt tài liệu"}, {label: "Tìm một nội dung trong nguồn", query: "Nguồn đã chọn đề cập những nội dung chính nào?"}];
      if (scope && company.length === 1) suggestions = [suggestions[0], {label: `Tổng tài sản ${scope}`, query: `Tổng tài sản ${scope} theo nguồn đã chọn là bao nhiêu?`}, {label: `Doanh thu ${scope}`, query: `Doanh thu ${scope} theo nguồn đã chọn là bao nhiêu?`}, {label: "Dòng tiền từ kinh doanh", query: `Lưu chuyển tiền thuần từ hoạt động kinh doanh ${scope} theo nguồn đã chọn là bao nhiêu?`}];
      action.textContent = "Quản lý nguồn đang chọn ↗";
      action.dataset.action = "manage-sources";
    } else {
      welcome.querySelector("h3").textContent = "Khám phá báo cáo của bạn";
      welcome.querySelector(".welcome-description").textContent = "Tra cứu trong kho chung, hoặc tạo notebook để tập trung vào một nhóm nguồn và lưu ghi chú.";
      suggestions = [{label: "Tra cứu tài sản FPT 2025", query: "Tổng tài sản FPT năm 2025 theo báo cáo hợp nhất là bao nhiêu?"}, {label: "So sánh doanh thu hai năm", query: "So sánh doanh thu năm 2024 và 2025 của FPT theo báo cáo hợp nhất"}];
      action.textContent = "Chọn nguồn cho notebook ↗";
      action.dataset.action = "browse-library";
    }
    list.replaceChildren(...suggestions.map(({label, query}) => {
      const button = document.createElement("button");
      button.className = "suggestion";
      button.type = "button";
      button.dataset.query = query;
      button.textContent = label;
      return button;
    }));
  }

  function syncNotebook(value, sources = []) {
    const changed = notebook?.id !== value?.id;
    notebook = value;
    selectedIds = sources;
    document.body.classList.toggle("has-notebook", Boolean(value));
    document.body.dataset.emptyScope = String(Boolean(value) && !sources.length);
    $("#workspace-title").textContent = value?.title || "Notebook tài chính";
    $("#workspace-description").textContent = value ? "Nguồn của bạn. Câu hỏi của bạn. Bằng chứng luôn đi cùng." : "Tập trung vào nguồn. Theo sát bằng chứng.";
    const scope = value ? (sources.length ? `Đang dùng ${sources.length} / ${value.sources.length} nguồn đã ghim` : "Chưa chọn nguồn để trả lời") : "Tra cứu trong kho chung";
    $("#chat-scope-label").textContent = scope;
    $("#composer-scope").textContent = scope;
    $("#selected-source-count").textContent = value ? `${sources.length}/${value.sources.length}` : "0";
    $("#mobile-source-count").textContent = String(sources.length);
    $("#source-general-state").hidden = Boolean(value);
    $("#notebook-thread-wrap").hidden = !value;
    $("#open-notes").disabled = !value;
    document.querySelector('[data-workspace-tab="notes"]').disabled = !value;
    const check = $("#notebook-select-all");
    check.checked = Boolean(value?.sources.length) && sources.length === value.sources.length;
    check.indeterminate = Boolean(sources.length) && sources.length < (value?.sources.length || 0);
    check.disabled = !value?.sources.length || Boolean(window.notebookWorkspace?.busy);
    if (changed) {
      if ($("#create-notebook-dialog").open) $("#create-notebook-dialog").close();
      selectSourceTab(!value || !value.sources.length);
      if (!value) $("#notes-dialog").close();
    }
    renderWelcome();
    renderDocuments(currentDocuments);
    syncAskState();
  }

  window.workspaceUI = { syncNotebook, syncAskState, renderWelcome, showSources, showChat, openNotes, notify };
  $("#source-tab-selected").addEventListener("click", () => selectSourceTab(false));
  $("#source-tab-library").addEventListener("click", () => selectSourceTab(true));
  document.querySelector(".source-tabs").addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const library = event.key === "End" || (!["Home"].includes(event.key) && document.body.dataset.sourceView !== "library");
    selectSourceTab(library);
    $(library ? "#source-tab-library" : "#source-tab-selected").focus();
  });
  document.querySelectorAll("[data-workspace-tab]").forEach((button) => button.addEventListener("click", () => {
    if (button.dataset.workspaceTab === "notes") openNotes();
    else selectWorkspaceTab(button.dataset.workspaceTab);
  }));
  $("#browse-source-library").addEventListener("click", () => showSources(true));
  $("#open-create-notebook").addEventListener("click", openCreate);
  $("#create-notebook-close").addEventListener("click", () => $("#create-notebook-dialog").close());
  $("#open-notes").addEventListener("click", openNotes);
  $("#notes-close").addEventListener("click", () => $("#notes-dialog").close());
  $("#notes-dialog").addEventListener("close", () => selectWorkspaceTab(document.body.dataset.workspaceView));
  document.addEventListener("click", (event) => {
    const action = event.target.closest("[data-action]")?.dataset.action;
    if (action === "create-notebook") openCreate();
    if (action === "browse-library") showSources(true);
    if (action === "manage-sources") showSources(false);
    if (!event.target.closest("#answer-options")) $("#answer-options").open = false;
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && $("#answer-options").open) {
      $("#answer-options").open = false;
      $("#answer-options summary").focus();
    }
  });
  $("#document-search").addEventListener("input", () => renderDocuments(currentDocuments));
  $("#document-company").addEventListener("change", () => renderDocuments(currentDocuments));
  elements.queryInput.addEventListener("input", syncAskState);
  elements.conversation.addEventListener("click", (event) => { if (event.target.closest(".suggestion")) syncAskState(); });
  elements.queryForm.addEventListener("submit", showChat);
  $(".skip-link").addEventListener("click", () => { showChat(); elements.queryInput.focus(); });
  $("#open-create-notebook").setAttribute("aria-label", "Tạo notebook mới");
  $("#open-notes").setAttribute("aria-label", "Mở ghi chú");
  $(".settings-link").setAttribute("aria-label", "Mở cấu hình AI");
  selectSourceTab(false);
  renderWelcome();
  syncAskState();
})();
