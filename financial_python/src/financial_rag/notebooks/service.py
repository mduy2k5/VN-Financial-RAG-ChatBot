"""Independent, source-scoped workspaces on the existing financial runtime.

The reader intentionally extracts source blocks. It never labels an unverified
language-model summary as evidence, and never silently widens the source scope.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4
import os
import re
import html
from fastapi import HTTPException

from ..context.budget import TokenBudget, estimate_tokens
from ..generation.chat import generation_budget_trace
from ..query.directives import normalize_directive_text
from ..retrieval.runtime import RequestSnapshot, ScopeCache, request_snapshot
from ..structure.normalized import fingerprint
from ..validation.claims import check_claims
from ..schemas import Citation
from .records import NotebookError, RecordRepository
from .structure import build_structure, extract_guide, locate_chunk


def now():
    return datetime.now(timezone.utc).isoformat()


def identity():
    return str(uuid4())


class NotebookService:
    def __init__(self, store, runtime):
        self.store = store
        self.runtime = runtime
        self.records = RecordRepository(store)
        self.cache = ScopeCache(max_scopes=4, max_chunks=30000)

    def require(self, kind, record_id, notebook_id=None):
        value = self.records.get(kind, record_id, notebook_id)
        if value is None:
            raise NotebookError("Không tìm thấy bản ghi trong notebook này.", 404)
        return value

    def create(self, title, instructions=""):
        return self.records.put("notebook", {"id": identity(), "title": title,
            "instructions": instructions, "sources": [], "created_at": now(), "updated_at": now()})

    def notebook(self, notebook_id):
        return self.require("notebook", notebook_id)

    def update(self, notebook_id, expected, **changes):
        notebook = self.notebook(notebook_id)
        return self.records.put("notebook", {**notebook, **changes, "updated_at": now()}, expected)

    def attach(self, notebook_id, logical_key, expected):
        notebook = self.notebook(notebook_id)
        if notebook["revision"] != expected:
            raise NotebookError("Notebook đã thay đổi; hãy tải lại.", 409)
        catalog = next((item for item in self.store.list_documents() if item["logical_key"] == logical_key), None)
        if not catalog or not catalog.get("active_projection_id"):
            raise NotebookError("Nguồn chưa có projection hoạt động.", 404)
        if hasattr(self.store, "pin_active_document"):
            try:
                document = self.store.pin_active_document(logical_key, catalog["active_projection_id"])
            except ValueError as exc:
                raise NotebookError(str(exc), 409) from exc
        else:
            document = self.read_document(logical_key, catalog["active_projection_id"])
        structure = self.pinned_structure(document)
        previous = next((item for item in notebook["sources"] if item["logical_key"] == logical_key), None)
        source = {key: document.get(key) for key in ("logical_key", "source_file", "company_code",
            "fiscal_year", "report_type", "source_sha256", "active_version_id", "active_projection_id")}
        source["id"] = previous["id"] if previous else identity()
        source["structure_id"] = structure["id"]
        source["structure_sha256"] = fingerprint(structure)
        source["catalog_projection_id"] = document.get("catalog_projection_id", source["active_projection_id"])
        source["projection_origin"] = document.get("projection_origin", "active_projection")
        sources = [item for item in notebook["sources"] if item["logical_key"] != logical_key] + [source]
        if len(sources) > 100:
            raise NotebookError("Một notebook nhận tối đa 100 nguồn.")
        return self.update(notebook_id, expected, sources=sources)

    def remove(self, notebook_id, source_id, expected):
        notebook = self.notebook(notebook_id)
        if not any(item["id"] == source_id for item in notebook["sources"]):
            raise NotebookError("Nguồn không thuộc notebook.", 404)
        return self.update(notebook_id, expected,
            sources=[item for item in notebook["sources"] if item["id"] != source_id])

    def read_document(self, logical_key, projection):
        try:
            return self.store.get_projection_document(logical_key, projection)
        except ValueError as exc:
            raise NotebookError(str(exc), 409) from exc

    def pinned_structure(self, document, source=None):
        structure_id = (source or {}).get("structure_id") or fingerprint(
            ["notebook_structure_v1", document["active_projection_id"]])
        record = self.records.get("structure", structure_id)
        if record is None:
            if source and source.get("structure_id"):
                raise NotebookError("Cấu trúc nguồn đã ghim không còn khả dụng.", 409)
            value = {"id": structure_id, "schema_version": "notebook_structure_v1",
                     **build_structure(document)}
            try:
                record = self.records.put("structure", value)
            except NotebookError as exc:
                if exc.status != 409:
                    raise
                record = self.require("structure", structure_id)
        if (record["projection_id"] != document["active_projection_id"]
                or record["version_id"] != document["active_version_id"]
                or record["source_sha256"] != document["source_sha256"]
                or (source and source.get("structure_sha256") and fingerprint(record) != source["structure_sha256"])):
            raise NotebookError("Cấu trúc không khớp phiên bản nguồn đã ghim.", 409)
        return record

    def source(self, notebook_id, source_id, run_id=None):
        notebook = self.notebook(notebook_id)
        sources = notebook["sources"]
        if run_id:
            sources = self.require("run", run_id, notebook_id)["sources"]
        source = next((item for item in sources if item["id"] == source_id), None)
        if not source:
            raise NotebookError("Nguồn không thuộc phạm vi yêu cầu.", 404)
        document = self.read_document(source["logical_key"], source["active_projection_id"])
        structure = self.pinned_structure(document, source)
        return source, document, structure

    def guide(self, notebook_id, source_id):
        source, document, structure = self.source(notebook_id, source_id)
        return {"source": source, "structure": structure, "guide": extract_guide(document, structure)}

    def content(self, notebook_id, source_id, *, run_id=None, block_id=None, start=0, limit=16000):
        source, document, structure = self.source(notebook_id, source_id, run_id)
        raw = document["raw_content"]
        end = min(len(raw), start + limit)
        if block_id:
            block = next((item for item in structure["blocks"] if item["id"] == block_id), None)
            if not block:
                raise NotebookError("Block không thuộc phiên bản nguồn này.", 404)
            start, end = block["start"], min(block["end"], block["start"] + limit)
        if start > len(raw):
            raise NotebookError("Vị trí nằm ngoài nguồn.")
        return {"source": source, "start": start, "end": end, "total_chars": len(raw),
                "text": raw[start:end], "has_more": end < len(raw), "structure": structure}

    def selected(self, notebook, source_ids):
        if source_ids is None:
            return notebook["sources"]
        allowed = {item["id"] for item in notebook["sources"]}
        if not set(source_ids) <= allowed:
            raise NotebookError("Có nguồn không thuộc notebook.", 404)
        return [item for item in notebook["sources"] if item["id"] in set(source_ids)]

    def snapshot(self, notebook, sources):
        key = fingerprint(["notebook-v1", notebook["id"], notebook["revision"], sources])
        return self.cache.get_or_build(key, lambda: self._build_snapshot(sources, key))

    def _build_snapshot(self, sources, key):
        chunks, documents = [], []
        raw_bytes = 0
        for source in sources:
            document = self.read_document(source["logical_key"], source["active_projection_id"])
            raw_bytes += len(document["raw_content"].encode())
            if raw_bytes > 50_000_000 or len(chunks) + len(document["chunks"]) > 30000:
                raise NotebookError("Phạm vi quá lớn; hãy chọn ít nguồn hơn.")
            structure = self.pinned_structure(document, source)
            # Namespace IDs because parser IDs can repeat across report versions.
            ids = {item.chunk_id: fingerprint([source["id"], source["active_projection_id"], item.chunk_id])
                   for item in document["chunks"]}
            for original in document["chunks"]:
                chunk = original.model_copy(deep=True)
                block = locate_chunk(chunk, structure, document["raw_content"])
                chunk.chunk_id = ids[chunk.chunk_id]
                chunk.parent_chunk_id = ids.get(chunk.parent_chunk_id)
                chunk.metadata.update({"notebook_source_id": source["id"],
                    "version_id": source["active_version_id"], "projection_id": source["active_projection_id"],
                    "source_sha256": structure["source_sha256"], "block_id": block["id"] if block else None,
                    "source_span": {"start": block["start"], "end": block["end"]} if block else None})
                chunks.append(chunk)
            documents.append({key: value for key, value in document.items() if key not in {"raw_content", "chunks"}})
        index = self.runtime.build_retriever(chunks, isolated=True) if chunks else None
        result = RequestSnapshot(tuple(index.chunks) if index else (), index, tuple(documents), key)
        return result

    def resolve_followup(self, query, conversation, sources):
        parsed = self.runtime.query_engine.parse(query)
        if not conversation or not conversation.get("last_scope"):
            return query
        previous = conversation["last_scope"]
        text = normalize_directive_text(query)
        followup = bool(re.search(r"\b(con|nam truoc|nam sau|cong ty do|chi tieu do|khoan do|the thi)\b", text))
        if not followup:
            return query
        # History from a different source selection must not silently determine scope.
        signature = fingerprint(sources)
        if previous.get("source_signature") != signature:
            raise NotebookError("Nguồn đã đổi. Hãy nêu rõ doanh nghiệp, kỳ và chỉ tiêu cho câu hỏi tiếp theo.")
        additions = []
        companies = previous.get("company_codes") or ([previous.get("company_code")] if previous.get("company_code") else [])
        if not parsed.company_codes and not parsed.company_code:
            if len(companies) != 1:
                raise NotebookError("Hãy nêu doanh nghiệp cho câu hỏi tiếp theo.")
            additions.append(companies[0])
        if not parsed.fiscal_years and parsed.fiscal_year is None:
            years = previous.get("fiscal_years") or ([previous.get("fiscal_year")] if previous.get("fiscal_year") else [])
            if len(years) != 1:
                raise NotebookError("Hãy nêu kỳ báo cáo cho câu hỏi tiếp theo.")
            delta = -1 if "nam truoc" in text else (1 if "nam sau" in text else 0)
            # Relative-year wording must be replaced to avoid the parser treating it
            # as a comparative column of the resolved year.
            if delta:
                query = re.sub(r"năm\s+(trước|sau)", f"năm {int(years[0]) + delta}", query, flags=re.I)
            else:
                additions.append(f"năm {years[0]}")
        if not parsed.canonical_metric and re.search(r"chi tieu do|khoan do|^con (?:nam|20)", text):
            metric = previous.get("primary_metric") or previous.get("canonical_metric")
            if not metric:
                raise NotebookError("Hãy nêu chỉ tiêu cho câu hỏi tiếp theo.")
            additions.append(metric)
        if not parsed.report_type and previous.get("report_type"):
            additions.append("hợp nhất" if previous["report_type"] == "consolidated" else "riêng lẻ")
        return " ".join([query, *additions])

    def reader(self, query, sources, mode, instructions=""):
        summary = mode == "summary" or any(term in normalize_directive_text(query)
                                           for term in ("tom tat", "tong quan", "gioi thieu"))
        terms = set(normalize_directive_text(query).split()) - {
            "doc", "phan", "tai", "lieu", "nguon", "noi", "dung", "cua", "trong", "la", "gi", "hay", "ve"}
        parsed = self.runtime.query_engine.parse(query)
        requested_companies = {str(value).casefold() for value in
            (parsed.company_codes or ([parsed.company_code] if parsed.company_code else []))}
        blocks = []
        source_block_counts = {}
        raw_bytes = 0
        block_count = 0
        for source in sources:
            _, document, structure = self.source_from_attachment(source)
            raw = document["raw_content"]
            raw_bytes += len(raw.encode())
            block_count += len(structure["blocks"])
            if raw_bytes > 50_000_000 or block_count > 30000:
                raise NotebookError("Phạm vi quá lớn; hãy chọn ít nguồn hơn để đọc.")
            source_block_counts[source["id"]] = len(structure["blocks"])
            if requested_companies:
                company = str(source.get("company_code") or "").casefold()
                if ((company and company not in requested_companies) or
                        (not company and not any(re.search(rf"\b{re.escape(code)}\b", raw, re.I)
                                                 for code in requested_companies))):
                    blocks.append([])
                    continue
            candidates = []
            for block in structure["blocks"]:
                text = raw[block["start"]:block["end"]]
                words = set(normalize_directive_text(text).split())
                score = len(terms & words) + len(terms & set(normalize_directive_text(block["title"]).split()))
                if summary or score > 0:
                    candidates.append({**block, "text": text, "source": source, "score": score})
            if not summary:
                candidates.sort(key=lambda value: (-value["score"], value["start"]))
            else:
                # Cover the document instead of filling the budget with cover
                # pages, postal addresses, repeated headers and tiny OCR fragments.
                substantive = [item for item in candidates if item["type"] == "table"
                               or len(item["text"].strip()) >= 80]
                if substantive:
                    candidates = substantive

                def priority(item):
                    text = normalize_directive_text(html.unescape(re.sub(r"<[^>]+>", " ", item["text"])))
                    main_statement = item["type"] == "table" and any(term in text for term in (
                        "tong tai san", "tong cong tai san", "loi nhuan sau thue", "luu chuyen tien"))
                    return (3 if main_statement else 2 if item["type"] == "table" else 1,
                            min(len(text), 1200))

                groups = {}
                for item in candidates:
                    group = item["page"] if item["page"] is not None else item["section_id"]
                    groups.setdefault(group, []).append(item)
                for items in groups.values():
                    items.sort(key=lambda item: (-priority(item)[0], -priority(item)[1], item["start"]))
                leads = sorted((items[0] for items in groups.values()),
                               key=lambda item: (-priority(item)[0], item["start"]))
                remaining = [item for items in groups.values() for item in items[1:]]
                candidates = [*leads, *remaining]
            blocks.append(candidates)
        # Round robin ensures one long source cannot displace all other sources.
        ordered = [items[index] for index in range(max((len(items) for items in blocks), default=0))
                   for items in blocks if index < len(items)]
        budget = TokenBudget(context_window=int(os.getenv("NOTEBOOK_READER_TOKEN_BUDGET", "16000")),
                             output_reserve=0, instruction_tokens=estimate_tokens(instructions),
                             query_tokens=estimate_tokens(query))
        # Headers and citation markup are budgeted together with the whole block.
        packed, trace = budget.pack([{**block, "text": f"{block['source']['source_file']}\n{block['text']} [c99999]"}
                                    for block in ordered])
        chosen = {item["id"] for item in packed}
        selected = [item for item in ordered if item["id"] in chosen]
        if summary and len(selected) > 24:
            trace["omitted"].extend({"id": item["id"], "reason": "summary_block_limit"}
                                    for item in selected[24:])
            selected = selected[:24]
            kept = {item["id"] for item in selected}
            trace["estimated_tokens"] = (sum(estimate_tokens(item["text"]) for item in packed if item["id"] in kept)
                                         + max(0, len(selected) - 1) * estimate_tokens("\n---\n"))
        trace["selection_method"] = "page_section_diversity" if summary else "lexical_section_match"
        citations, parts = [], []
        for index, block in enumerate(selected, 1):
            source, c_id = block["source"], f"c{index}"
            citations.append({"citation_id": c_id, "document_id": source["logical_key"],
                "chunk_id": None, "file_name": source["source_file"], "page": block["page"],
                "section": block["title"], "unit": None, "statement_type": None,
                "snippet": block["text"][:500], "notebook_source_id": source["id"],
                "version_id": source["active_version_id"], "projection_id": source["active_projection_id"],
                "source_sha256": source["source_sha256"], "block_id": block["id"],
                "source_span": {"start": block["start"], "end": block["end"]}})
            parts.append(f"{source['source_file']}\n\n{block['text']} [{c_id}]")
        represented = {item["notebook_source_id"] for item in citations}
        missing = [item["id"] for item in sources if item["id"] not in represented]
        partial = bool(missing or trace["omitted"] or (summary and len(selected) < sum(source_block_counts.values())))
        return {"query": query,
            "answer": ("Trích xuất nguyên văn theo nguồn đã chọn:\n\n" + "\n\n---\n\n".join(parts)) if parts
                      else "Chưa tìm thấy đoạn nguồn phù hợp trong phạm vi đã chọn.",
            "citations": citations, "calculations": [], "reader_mode": "source_extract",
            "answer_status": "insufficient_evidence" if not parts else ("partial" if partial else "complete"),
            "validation": {"is_valid": bool(parts), "citation_valid": bool(parts), "unit_match": True,
                "details": "Trích nguyên văn; chưa phải bản tổng hợp diễn giải bằng mô hình."},
            "evidence_coverage": {"selected_sources": len(sources), "represented_sources": len(represented),
                "missing_source_ids": missing, "selected_blocks": len(selected),
                "source_block_counts": source_block_counts, "scope": "source_blocks"},
            "context_budget": {**trace, "purpose": "extractive_reader", "model_called": False}}

    def source_from_attachment(self, source):
        document = self.read_document(source["logical_key"], source["active_projection_id"])
        return source, document, self.pinned_structure(document, source)

    def answer(self, notebook_id, request):
        notebook = self.notebook(notebook_id)
        if notebook["revision"] != request.notebook_revision:
            raise NotebookError("Notebook đã thay đổi; hãy tải lại trước khi hỏi.", 409)
        sources = self.selected(notebook, request.source_ids)
        conversation = self.require("conversation", request.conversation_id, notebook_id) if request.conversation_id else None
        if conversation and request.conversation_revision != conversation["revision"]:
            raise NotebookError("Hội thoại đã thay đổi; hãy tải lại.", 409)
        if not sources:
            raise NotebookError("Hãy chọn ít nhất một nguồn trong notebook.")
        resolved = self.resolve_followup(request.query, conversation, sources)
        parsed = self.runtime.query_engine.parse(resolved)
        finance = request.mode == "financial" or (request.mode == "auto" and
            (parsed.canonical_metric or parsed.metric_id or parsed.structured_operation not in {None, "summary"}))
        if finance:
            snapshot = self.snapshot(notebook, sources)
            token = request_snapshot.set(snapshot)
            budget_token = generation_budget_trace.set(None)
            try:
                if not snapshot.chunks:
                    raise NotebookError("Nguồn không có dữ liệu tài chính đã phân tích; hãy dùng chế độ đọc nguồn.")
                try:
                    response = self.runtime.answer_question(self.runtime.QueryRequest(
                        query=resolved, top_k=request.top_k, source_policy=request.source_policy)).model_dump(mode="json")
                except HTTPException as exc:
                    if exc.status_code != 404:
                        raise
                    response = {"query": resolved, "parsed_query": parsed.model_dump(mode="json"),
                        "answer": "Chưa tìm thấy bằng chứng cho doanh nghiệp, kỳ và chỉ tiêu trong nguồn đã chọn.",
                        "answer_status": "insufficient_evidence", "citations": [], "calculations": [],
                        "validation": {"is_valid": False, "citation_valid": False, "unit_match": True,
                                       "details": "Không mở rộng ra ngoài phạm vi notebook."}}
                if generation_budget_trace.get() is not None:
                    response["context_budget"] = generation_budget_trace.get()
            finally:
                request_snapshot.reset(token)
                generation_budget_trace.reset(budget_token)
            allowed = {(item["id"], item["active_version_id"], item["active_projection_id"]) for item in sources}
            if any((item.get("notebook_source_id"), item.get("version_id"), item.get("projection_id")) not in allowed
                   for item in response["citations"]):
                raise NotebookError("Citation nằm ngoài phạm vi nguồn; câu trả lời bị chặn.", 500)
            if response.get("context_budget", {}).get("model_called"):
                # A metric-lookup intent does not make generated prose an
                # executor result. Until an entailment verifier is calibrated,
                # allow only extractive claims from the actually cited chunks.
                response["claim_checks"] = check_claims(response["answer"],
                    [Citation.model_validate(item) for item in response["citations"]],
                    snapshot.retriever.chunks_by_id, deterministic=False)
                if any(item["status"] == "unverified" for item in response["claim_checks"]):
                    response["answer"] = "Diễn giải của mô hình chưa được xác minh với đoạn nguồn; hãy mở citation để đối chiếu."
                    response["answer_status"] = "insufficient_evidence"
                    response["validation"]["is_valid"] = False
                    response["validation"]["details"] = "Diễn giải chưa khớp nguyên văn nguồn; không gắn nhãn executor_checked."
                    response["calculations"] = []
            if response.get("answer_status") is None:
                resolved_scope = response.get("parsed_query") or {}
                one_company = len(resolved_scope.get("company_codes") or
                                  ([resolved_scope.get("company_code")] if resolved_scope.get("company_code") else [])) == 1
                one_year = len(resolved_scope.get("fiscal_years") or
                               ([resolved_scope.get("fiscal_year")] if resolved_scope.get("fiscal_year") else [])) == 1
                simple_lookup = (resolved_scope.get("intent") == "metric_lookup" and
                                 not resolved_scope.get("structured_operation") and one_company and one_year
                                 and not re.search(r"\b(va|cac|nhung)\b", normalize_directive_text(resolved)))
                verified = response.get("validation", {}).get("is_valid") and bool(response["citations"])
                response["answer_status"] = ("complete" if simple_lookup and verified else
                                             ("partial" if verified else "insufficient_evidence"))
        else:
            response = self.reader(resolved, sources, request.mode, notebook["instructions"])
        run_id = identity()
        response.update({"query": request.query, "resolved_query": resolved, "notebook_id": notebook_id,
            "run_id": run_id, "notebook_revision": notebook["revision"],
            "source_snapshot": {"id": fingerprint(sources), "notebook_revision": notebook["revision"],
                                "policy": request.source_policy, "documents": sources}})
        conversation = conversation or {"id": identity(), "notebook_id": notebook_id, "turns": [],
                                        "created_at": now(), "title": request.query[:100]}
        scope = response.get("parsed_query") or parsed.model_dump(mode="json")
        scope["source_signature"] = fingerprint(sources)
        self.records.put("run", {"id": run_id, "notebook_id": notebook_id, "sources": sources,
            "response": response, "created_at": now(), "updated_at": now()})
        conversation = self.records.put("conversation", {**conversation,
            "last_scope": scope, "updated_at": now(), "turns": [*conversation["turns"],
            {"query": request.query, "resolved_query": resolved, "run_id": run_id}][-40:]},
            conversation.get("revision"))
        response.update({"conversation_id": conversation["id"], "conversation_revision": conversation["revision"]})
        return response

    def conversation(self, notebook_id, conversation_id):
        self.notebook(notebook_id)
        conversation = self.require("conversation", conversation_id, notebook_id)
        return {**conversation, "turns": [{**turn,
            "response": self.require("run", turn["run_id"], notebook_id)["response"]}
            for turn in conversation["turns"]]}

    def save_note(self, notebook_id, title, content, run_id=None):
        self.notebook(notebook_id)
        if not run_id and not content.strip():
            raise NotebookError("Ghi chú cần nội dung hoặc câu trả lời đã lưu.")
        run = self.require("run", run_id, notebook_id) if run_id else None
        return self.records.put("note", {"id": identity(), "notebook_id": notebook_id, "title": title,
            "content": run["response"]["answer"] if run else content, "run_id": run_id,
            "sources": run["sources"] if run else [], "created_at": now(), "updated_at": now()})

    def notes(self, notebook_id):
        current = self.notebook(notebook_id)
        active = {(item["id"], item["active_projection_id"]) for item in current["sources"]}
        return [{**note, "stale": any((item["id"], item["active_projection_id"]) not in active
                                     for item in note["sources"])}
                for note in self.records.list("note", notebook_id)]
