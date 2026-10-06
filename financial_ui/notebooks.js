/* Notebook records and evidence live on the private service, not in local chat history. */
(() => {
  const $ = (selector) => document.querySelector(selector);
  const base = "/api/v1/notebooks";
  const selectedSources = new Set();
  let active = null;
  let conversation = null;
  let viewer = null;
  let previousConversation = activeConversationId;
  let isMutating = false;
  let isChoosing = false;
  let isLoadingThread = false;
  let choosing = 0;
  let threadRequest = 0;
  let notesRequest = 0;
  let sourceRequest = 0;
  let savedRunsNotebookId = null;
  let savedRunIds = new Set();

  function isBusy() { return isAsking || isMutating || isChoosing || isLoadingThread; }

  function isRunSaved(notebookId, runId) {
    return savedRunsNotebookId === notebookId && savedRunIds.has(runId);
  }

  function syncSavedAnswerButtons() {
    document.querySelectorAll("button[data-note-run-id]").forEach((save) => {
      if (!isRunSaved(save.dataset.noteNotebookId, save.dataset.noteRunId)) return;
      save.disabled = true;
      save.textContent = "Đã lưu vào ghi chú";
    });
  }

  function feedback(text, error = false, notify = false) {
    $("#notebook-feedback").textContent = text;
    $("#notebook-feedback").classList.toggle("is-error", error);
    if (error || notify) window.workspaceUI?.notify(text, error);
  }

  async function api(path = "", options = {}) {
    let response;
    try {
      response = await fetch(base + path, { ...options,
        headers: { ...(options.body ? { "Content-Type": "application/json" } : {}), ...options.headers } });
    } catch {
      throw new Error("Không kết nối được dịch vụ. Kiểm tra kết nối rồi thử lại.");
    }
    let value;
    try { value = await response.json(); }
    catch { throw new Error("Dịch vụ chưa trả về dữ liệu hợp lệ. Vui lòng thử lại."); }
    if (!response.ok) {
      if (response.status === 409) throw new Error("Notebook đã được cập nhật ở nơi khác. Làm mới nguồn trước khi tiếp tục.");
      if (response.status === 503) throw new Error("Kho notebook chưa sẵn sàng. Vui lòng thử lại sau.");
      throw new Error(typeof value.detail === "string" ? value.detail : "Không thực hiện được yêu cầu. Kiểm tra thông tin rồi thử lại.");
    }
    return value;
  }

  function button(text, action) {
    const node = document.createElement("button");
    node.type = "button";
    node.className = "button";
    node.textContent = text;
    node.addEventListener("click", () => Promise.resolve().then(action).catch((error) => feedback(error.message, true)));
    return node;
  }

  function resetChat() {
    conversation = null;
    historyEntries = [];
    activeHistoryId = null;
    $("#notebook-thread").value = "";
    showWelcome();
    renderHistoryBar();
    window.workspaceUI?.renderWelcome();
    window.workspaceUI?.syncAskState();
  }

  function syncBusyControls() {
    const locked = isBusy();
    elements.queryInput.disabled = locked;
    elements.newConversation.disabled = locked;
    $("#notebook-select").disabled = locked;
    document.querySelectorAll("#notebook-create button, #notebook-create input, #notebook-sources button, #notebook-sources input, #notebook-thread, #notebook-mode, #notebook-add-source, #notebook-attach")
      .forEach((node) => { node.disabled = locked; });
    const selectAll = $("#notebook-select-all");
    if (selectAll) selectAll.disabled = locked || !active?.sources.length;
    window.workspaceUI?.syncAskState();
    if (locked) elements.askButton.disabled = true;
  }

  function busy(value) {
    isAsking = value;
    syncBusyControls();
  }

  async function mutate(action) {
    if (isBusy()) return;
    isMutating = true;
    syncBusyControls();
    try { return await action(); }
    finally { isMutating = false; syncBusyControls(); }
  }

  function renderCatalog() {
    const select = $("#notebook-add-source");
    select.replaceChildren(new Option("Chọn nguồn từ kho…", ""), ...currentDocuments.map((document) => {
      const source = active?.sources.find((item) => item.logical_key === document.logical_key);
      const attached = source && (source.catalog_projection_id || source.active_projection_id) === document.active_projection_id;
      const option = new Option(`${document.source_file || document.document_id}${attached ? " · đã thêm" : ""}`, document.logical_key || "");
      option.disabled = Boolean(attached);
      return option;
    }));
  }

  function syncNotebook() {
    const count = $("#selected-source-count");
    if (count) count.textContent = `${selectedSources.size}/${active?.sources.length || 0}`;
    const selectAll = $("#notebook-select-all");
    if (selectAll) {
      selectAll.checked = Boolean(active?.sources.length) && selectedSources.size === active.sources.length;
      selectAll.indeterminate = selectedSources.size > 0 && selectedSources.size < (active?.sources.length || 0);
      selectAll.disabled = isBusy() || !active?.sources.length;
    }
    window.workspaceUI?.syncNotebook(active, [...selectedSources]);
    window.workspaceUI?.syncAskState();
  }

  function sourceLabel(source) {
    const company = source.company_code;
    const year = source.fiscal_year;
    const reportType = { consolidated: "Hợp nhất", separate: "Riêng lẻ", standalone: "Riêng lẻ" }[source.report_type];
    if (company) return [company, year, reportType].filter(Boolean).join(" · ");
    return (source.source_file || "Tài liệu").replace(/\.(txt|md|markdown)$/i, "").replace(/[_-]+/g, " ");
  }

  function renderSources() {
    const container = $("#notebook-sources");
    container.replaceChildren();
    if (!active?.sources.length) container.append(placeholder("Notebook chưa có nguồn. Mở Kho tài liệu để thêm, hoặc tải một tệp mới."));
    const notebookId = active?.id;
    for (const source of active?.sources || []) {
      const row = document.createElement("article");
      row.className = "notebook-source";
      row.classList.toggle("is-selected", selectedSources.has(source.id));
      const label = document.createElement("label");
      const check = document.createElement("input");
      check.type = "checkbox";
      check.checked = selectedSources.has(source.id);
      check.setAttribute("aria-label", `Dùng nguồn ${sourceLabel(source)} để trả lời`);
      check.addEventListener("change", () => {
        if (check.checked) selectedSources.add(source.id); else selectedSources.delete(source.id);
        row.classList.toggle("is-selected", check.checked);
        resetChat();
        syncNotebook();
        feedback(`Đang dùng ${selectedSources.size} nguồn. Đổi phạm vi sẽ bắt đầu hội thoại mới; hội thoại cũ vẫn được lưu.`);
      });
      const title = document.createElement("span");
      title.className = "notebook-source-title";
      title.textContent = sourceLabel(source);
      title.title = source.source_file;
      label.append(check, title);
      const filename = document.createElement("p");
      filename.className = "notebook-source-filename";
      filename.textContent = source.source_file;
      filename.title = source.source_file;
      const version = document.createElement("small");
      version.textContent = "Bản nguồn đã ghim";
      version.title = `Phiên bản ${source.active_version_id || ""}`;
      const actions = document.createElement("div");
      actions.className = "notebook-source-actions";
      actions.append(button("Đọc nguồn", () => openSource(notebookId, source.id)),
                     button("Các phần", () => showGuide(source.id)),
                     button("Gỡ", () => mutate(async () => {
                       active = await api(`/${notebookId}/sources/${source.id}?expected_revision=${active.revision}`, { method: "DELETE" });
                       selectedSources.delete(source.id);
                       resetChat(); renderSources(); renderCatalog(); await renderNotes();
                       feedback("Đã gỡ nguồn khỏi notebook. Câu trả lời và ghi chú đã lưu vẫn giữ bằng chứng cũ.", false, true);
                     })));
      const current = currentDocuments.find((item) => item.logical_key === source.logical_key);
      if (current?.active_projection_id && current.active_projection_id !== (source.catalog_projection_id || source.active_projection_id)) {
        actions.append(button("Nhận bản mới", () => attach(source.logical_key)));
      }
      row.append(label, filename, version, actions);
      container.append(row);
    }
    syncNotebook();
    syncBusyControls();
  }

  async function threads() {
    if (!active) return;
    const notebookId = active.id;
    const result = await api(`/${notebookId}/conversations`);
    if (active?.id !== notebookId) return;
    $("#notebook-thread").replaceChildren(new Option("Hội thoại mới", ""), ...result.conversations.map((item) =>
      new Option(item.title, item.id)));
    $("#notebook-thread").value = conversation?.id || "";
  }

  async function choose(id) {
    // Creation calls choose() while its mutation is still active. The separate
    // loading flag locks chat without showing answer-generation progress.
    if (isAsking || isLoadingThread) return;
    const request = ++choosing;
    isChoosing = true;
    syncBusyControls();
    try {
      if (!active && id) previousConversation = activeConversationId;
      const value = id ? await api(`/${id}`) : null;
      if (request !== choosing) return;
      active = value;
      if (savedRunsNotebookId !== active?.id) {
        savedRunsNotebookId = active?.id || null;
        savedRunIds = new Set();
      }
      document.body.classList.toggle("has-notebook", Boolean(active));
      $("#notebook-select").value = id;
      $("#notebook-scope").hidden = !active;
      $("#notebook-notes").hidden = !active;
      elements.conversationHistoryToggle.hidden = Boolean(active);
      setConversationHistoryOpen(false);
      selectedSources.clear();
      if (active) {
        activeConversationId = null;
        for (const source of active.sources) selectedSources.add(source.id);
        resetChat(); renderSources(); renderCatalog();
        await Promise.all([threads(), renderNotes()]);
        if (request !== choosing) return;
        feedback(`Notebook “${active.title}” · ${active.sources.length} nguồn. Mọi câu trả lời giữ phiên bản nguồn đã chọn.`);
      } else {
        activeConversationId = previousConversation;
        if (activeConversationId) openConversation(activeConversationId); else startNewConversation();
        feedback("Kho chung đang dùng luồng hỏi đáp và hội thoại cũ.");
        const count = $("#note-count");
        if (count) count.textContent = "0";
        $("#notes-dialog")?.close();
        renderSources(); renderCatalog();
      }
      syncNotebook();
      try { localStorage.setItem("financial-rag.active-notebook.v1", id); } catch { /* optional preference */ }
    } finally {
      if (request === choosing) {
        isChoosing = false;
        $("#notebook-select").value = active?.id || "";
        syncBusyControls();
      }
    }
  }

  async function reload() {
    const result = await api();
    const select = $("#notebook-select");
    select.replaceChildren(new Option("Kho chung · hội thoại cũ", ""), ...result.notebooks.map((item) => new Option(item.title, item.id)));
    select.value = active?.id || "";
    renderCatalog();
  }

  async function attach(key) {
    if (!active || !key || isBusy()) return;
    return mutate(async () => {
      active = await api(`/${active.id}/sources`, { method: "POST",
        body: JSON.stringify({ logical_key: key, expected_revision: active.revision }) });
      const source = active.sources.find((item) => item.logical_key === key);
      selectedSources.add(source.id);
      resetChat(); renderSources(); renderCatalog(); await renderNotes();
      feedback("Đã thêm nguồn và ghim phiên bản hiện tại. Bạn có thể bắt đầu hỏi đáp.", false, true);
    });
  }

  async function renderNotes() {
    if (!active) return;
    const notebookId = active.id;
    const request = ++notesRequest;
    const result = await api(`/${notebookId}/notes`);
    if (active?.id !== notebookId || request !== notesRequest) return;
    // Rebuild saved state from the service after page reload or scope changes.
    // Keep locally confirmed saves if an older notes response is still in flight.
    if (savedRunsNotebookId !== notebookId) {
      savedRunsNotebookId = notebookId;
      savedRunIds = new Set();
    }
    for (const note of result.notes) if (note.run_id) savedRunIds.add(note.run_id);
    syncSavedAnswerButtons();
    const count = $("#note-count");
    if (count) count.textContent = String(result.notes.length);
    const list = $("#notebook-note-list");
    list.replaceChildren();
    for (const note of result.notes) {
      const node = document.createElement("details");
      const heading = document.createElement("summary");
      heading.textContent = `${note.title}${note.stale ? " · nguồn đã đổi hoặc đã gỡ" : ""}`;
      const text = document.createElement("div");
      text.className = "notebook-note-content";
      text.textContent = note.content;
      node.append(heading, text);
      if (note.run_id) node.append(button("Xem câu trả lời và bằng chứng đã lưu", async () => {
        const run = await api(`/${notebookId}/runs/${note.run_id}`);
        $("#notes-dialog")?.close();
        window.workspaceUI?.showChat();
        appendAnswer(run.response, `saved-${note.id}`);
      }));
      list.append(node);
    }
    if (!result.notes.length) list.append(placeholder("Chưa có ghi chú. Có thể lưu một câu trả lời bên trên."));
  }

  function decorateAnswer(card, response) {
    if (!response.notebook_id || !response.run_id) return;
    const detail = document.createElement("p");
    detail.className = "validation-details";
    detail.textContent = `Notebook · phiên bản ${response.notebook_revision}.`;
    if (response.resolved_query && response.resolved_query !== response.query) detail.textContent += ` Câu hỏi đã xác định: ${response.resolved_query}`;
    if (response.context_budget?.omitted?.length) detail.textContent += ` ${response.context_budget.omitted.length} khối chưa được đưa vào do ngân sách.`;
    const save = button("Lưu vào ghi chú", async () => {
      if (save.disabled || isRunSaved(response.notebook_id, response.run_id)) return;
      save.disabled = true;
      save.textContent = "Đang lưu…";
      let confirmed = false;
      try {
        await api(`/${response.notebook_id}/notes`, { method: "POST",
          body: JSON.stringify({ title: response.query.slice(0, 200) || "Câu trả lời", run_id: response.run_id }) });
        confirmed = true;
        if (savedRunsNotebookId === response.notebook_id) savedRunIds.add(response.run_id);
        save.textContent = "Đã lưu vào ghi chú";
        syncSavedAnswerButtons();
        await renderNotes();
        feedback("Đã lưu câu trả lời cùng bằng chứng nguồn.", false, true);
        if (active?.id === response.notebook_id) window.workspaceUI?.openNotes();
      } catch (error) {
        if (confirmed) {
          feedback("Đã lưu câu trả lời cùng bằng chứng. Danh sách ghi chú chưa tải lại được; làm mới để xem mục vừa lưu.", true);
          return;
        }
        save.disabled = false;
        save.textContent = "Lưu vào ghi chú";
        throw error;
      }
    });
    save.dataset.noteNotebookId = response.notebook_id;
    save.dataset.noteRunId = response.run_id;
    if (isRunSaved(response.notebook_id, response.run_id)) {
      save.disabled = true;
      save.textContent = "Đã lưu vào ghi chú";
    }
    card.append(detail, save);
  }

  async function ask(query) {
    if (!active || isBusy()) return;
    if (!selectedSources.size) {
      feedback("Hãy chọn ít nhất một nguồn để bắt đầu hỏi đáp.", true);
      window.workspaceUI?.showSources();
      return;
    }
    window.workspaceUI?.showChat();
    busy(true);
    const historyId = `notebook-${Date.now()}`;
    appendUserMessage(query, historyId);
    elements.queryInput.value = "";
    appendTypingIndicator();
    try {
      const response = await api(`/${active.id}/answer`, { method: "POST", body: JSON.stringify({
        query, notebook_revision: active.revision, source_ids: [...selectedSources],
        conversation_id: conversation?.id || null, conversation_revision: conversation?.revision || null,
        top_k: Number(elements.topK.value), mode: $("#notebook-mode").value,
        source_policy: $("#source-policy").value,
      }) });
      conversation = { id: response.conversation_id, revision: response.conversation_revision };
      $("#typing-indicator")?.remove();
      appendAnswer(response, historyId);
      await threads();
    } catch (error) {
      $("#typing-indicator")?.remove();
      appendErrorMessage(error.message, historyId);
      feedback(error.message, true);
    } finally {
      busy(false); elements.queryInput.focus();
    }
  }

  function displaySource(value, highlight = false) {
    const title = $("#notebook-viewer-title");
    title.textContent = value.source.source_file;
    const pages = [...new Set((value.structure.blocks || [])
      .filter((block) => block.start < value.end && block.end > value.start && block.page != null)
      .map((block) => block.page))];
    const pageLabel = pages.length ? `Trang ${pages.length > 1 ? `${pages[0]}–${pages.at(-1)}` : pages[0]} · ` : "";
    $("#notebook-viewer-version").textContent = `${pageLabel}Bản nguồn đã ghim${highlight ? " · Đoạn được trích dẫn" : ""}`;
    $("#notebook-viewer-version").title = `Phiên bản ${value.source.active_version_id} · ký tự ${value.start}–${value.end}/${value.total_chars}`;
    $("#notebook-viewer").setAttribute("aria-labelledby", "notebook-viewer-title");
    const text = $("#notebook-viewer-text");
    text.replaceChildren();
    text.classList.toggle("is-evidence", highlight);
    renderSourceText(text, value.text);
    $("#notebook-viewer-section").replaceChildren(new Option("Đi tới phần…", ""),
      ...value.structure.sections.slice(1).map((section) => new Option(section.title, String(section.start))));
    viewer.end = value.end;
    $("#notebook-viewer-more").hidden = !value.has_more;
    if (!$("#notebook-viewer").open) $("#notebook-viewer").showModal();
  }

  function renderSourceText(container, sourceText) {
    let previous = 0;
    for (const match of sourceText.matchAll(/<table\b[^>]*>[\s\S]*?<\/table>/gi)) {
      const before = document.createElement("span");
      before.textContent = sourceText.slice(previous, match.index);
      container.append(before);
      // Parse into an inert template. Only fresh table cells with textContent
      // are attached; source scripts, links, styles and URL attributes are discarded.
      const inert = document.createElement("template");
      inert.innerHTML = match[0];
      const table = document.createElement("table");
      for (const sourceRow of inert.content.querySelectorAll("tr")) {
        const row = document.createElement("tr");
        for (const sourceCell of [...sourceRow.children].filter((node) => ["TD", "TH"].includes(node.tagName))) {
          const cell = document.createElement(sourceCell.tagName.toLowerCase());
          cell.textContent = sourceCell.textContent;
          for (const attribute of ["rowspan", "colspan"]) {
            const span = Number(sourceCell.getAttribute(attribute));
            if (Number.isInteger(span) && span > 1 && span <= 100) cell.setAttribute(attribute, String(span));
          }
          row.append(cell);
        }
        table.append(row);
      }
      const scroll = document.createElement("div");
      scroll.className = "source-table-scroll";
      scroll.append(table);
      container.append(scroll);
      previous = match.index + match[0].length;
    }
    const after = document.createElement("span");
    after.textContent = sourceText.slice(previous);
    container.append(after);
    if (previous) {
      const details = document.createElement("details");
      const heading = document.createElement("summary");
      heading.textContent = "Xem văn bản gốc của đoạn này";
      const raw = document.createElement("pre");
      raw.textContent = sourceText;
      details.append(heading, raw); container.append(details);
    }
  }

  async function openSource(notebookId, sourceId, options = {}) {
    try {
      const request = ++sourceRequest;
      viewer = { notebookId, sourceId, runId: options.runId || null, end: 0 };
      const params = new URLSearchParams();
      if (options.runId) params.set("run_id", options.runId);
      if (options.blockId) params.set("block_id", options.blockId);
      const value = await api(`/${notebookId}/sources/${sourceId}/content?${params}`);
      if (request !== sourceRequest) return false;
      $("#notebook-viewer-guide").replaceChildren();
      displaySource(value, Boolean(options.blockId));
      return true;
    } catch (error) { feedback(error.message, true); return false; }
  }

  async function navigate(start) {
    if (!viewer) return;
    const current = viewer;
    const request = ++sourceRequest;
    const params = new URLSearchParams({ start: String(start) });
    if (current.runId) params.set("run_id", current.runId);
    const value = await api(`/${current.notebookId}/sources/${current.sourceId}/content?${params}`);
    if (request !== sourceRequest || viewer !== current) return;
    displaySource(value);
  }

  async function showGuide(sourceId) {
    const notebookId = active.id;
    const request = ++sourceRequest;
    const result = await api(`/${notebookId}/sources/${sourceId}/guide`);
    if (active?.id !== notebookId || request !== sourceRequest) return;
    if (!await openSource(notebookId, sourceId)) return;
    const guide = $("#notebook-viewer-guide");
    const label = document.createElement("p");
    label.textContent = `${result.guide.section_count} phần · ${result.guide.block_count} khối nguồn. Các mục gợi ý lấy từ tiêu đề tài liệu.`;
    guide.append(label);
    for (const extract of result.guide.extracts) {
      const quote = document.createElement("blockquote");
      quote.textContent = extract.text;
      guide.append(quote);
    }
    for (const question of result.guide.suggested_questions) guide.append(button(question, () => {
      $("#notebook-viewer").close(); window.workspaceUI?.showChat();
      elements.queryInput.value = question; elements.queryInput.focus(); window.workspaceUI?.syncAskState();
    }));
  }

  window.notebookWorkspace = {
    get active() { return active; },
    get busy() { return isBusy(); },
    get selectedSources() { return [...selectedSources]; },
    ask, attach, decorateAnswer, openSource, renderCatalog,
    newConversation() {
      if (isBusy()) return;
      resetChat(); window.workspaceUI?.showChat(); elements.queryInput.focus();
    },
    renderExtractive(text) {
      const node = document.createElement("div");
      node.className = "source-reader-text";
      renderSourceText(node, text);
      return node;
    },
    async refreshActive() {
      if (!active || isBusy()) return;
      const notebookId = active.id;
      const value = await api(`/${notebookId}`);
      if (active?.id !== notebookId) return;
      active = value;
      for (const id of [...selectedSources]) if (!active.sources.some((source) => source.id === id)) selectedSources.delete(id);
      renderSources(); renderCatalog(); await renderNotes();
    },
  };

  $("#notebook-select").addEventListener("change", (event) => choose(event.target.value).catch((error) => feedback(error.message, true)));
  $("#notebook-create").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (isBusy()) return;
    const title = $("#notebook-name").value.trim();
    if (!title) { feedback("Nhập tên notebook để tạo không gian mới.", true); return; }
    try {
      await mutate(async () => {
        const notebook = await api("", { method: "POST", body: JSON.stringify({ title }) });
        $("#notebook-name").value = "";
        await reload(); await choose(notebook.id);
        window.workspaceUI?.showSources(true);
        feedback("Đã tạo notebook. Thêm nguồn từ kho tài liệu để bắt đầu.", false, true);
      });
    } catch (error) { feedback(error.message, true); }
  });
  $("#notebook-attach").addEventListener("click", () => attach($("#notebook-add-source").value).catch((error) => feedback(error.message, true)));
  $("#notebook-select-all")?.addEventListener("click", () => {
    if (!active || isBusy()) return;
    const allSelected = selectedSources.size === active.sources.length;
    selectedSources.clear();
    if (!allSelected) for (const source of active.sources) selectedSources.add(source.id);
    resetChat(); renderSources();
    feedback(`Đang dùng ${selectedSources.size} nguồn. Phạm vi mới bắt đầu hội thoại mới; hội thoại cũ vẫn được lưu.`);
  });
  $("#notebook-thread").addEventListener("change", async (event) => {
    if (!active || isBusy()) return;
    if (!event.target.value) { resetChat(); return; }
    const notebookId = active.id;
    const threadId = event.target.value;
    const request = ++threadRequest;
    isLoadingThread = true;
    syncBusyControls();
    try {
      const record = await api(`/${notebookId}/conversations/${threadId}`);
      if (active?.id !== notebookId || $("#notebook-thread").value !== threadId) return;
      conversation = { id: record.id, revision: record.revision };
      elements.conversation.replaceChildren(); historyEntries = []; renderHistoryBar();
      for (const turn of record.turns) {
        appendUserMessage(turn.query, turn.run_id);
        appendAnswer(turn.response, turn.run_id);
      }
      window.workspaceUI?.showChat();
    } catch (error) { feedback(error.message, true); }
    finally {
      if (request === threadRequest) {
        isLoadingThread = false;
        $("#notebook-thread").value = conversation?.id || "";
        syncBusyControls();
      }
    }
  });
  $("#notebook-note-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!active) return;
    const form = event.target;
    const submit = form.querySelector('button[type="submit"], button:not([type])');
    if (submit?.disabled) return;
    const notebookId = active.id;
    const title = $("#notebook-note-title").value.trim();
    if (!title) { feedback("Nhập tiêu đề trước khi lưu ghi chú.", true); return; }
    if (submit) { submit.disabled = true; submit.textContent = "Đang lưu…"; }
    try {
      await api(`/${notebookId}/notes`, { method: "POST", body: JSON.stringify({
        title, content: $("#notebook-note-content").value }) });
      if (active?.id === notebookId) { form.reset(); await renderNotes(); }
      feedback("Đã lưu ghi chú riêng.", false, true);
    } catch (error) { feedback(error.message, true); }
    finally { if (submit) { submit.disabled = false; submit.textContent = "Lưu ghi chú"; } }
  });
  $("#notebook-viewer-close").addEventListener("click", () => $("#notebook-viewer").close());
  $("#notebook-viewer-more").addEventListener("click", () => navigate(viewer.end).catch((error) => feedback(error.message, true)));
  $("#notebook-viewer-section").addEventListener("change", (event) => {
    if (event.target.value !== "") navigate(Number(event.target.value)).catch((error) => feedback(error.message, true));
  });
  reload().then(async () => {
    let saved = "";
    try { saved = localStorage.getItem("financial-rag.active-notebook.v1") || ""; } catch { /* optional preference */ }
    if (saved && [...$("#notebook-select").options].some((option) => option.value === saved)) await choose(saved);
    else syncNotebook();
  }).catch((error) => feedback(error.message, true));
})();
