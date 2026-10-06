import importlib
import json

import pytest
from fastapi.testclient import TestClient

from financial_rag.context.assembler import ContextAssembler
from financial_rag.context.budget import TokenBudget, estimate_tokens
from financial_rag.generation.chat import FinancialChatEngine
from financial_rag.generation.chat import generation_budget_trace
from financial_rag.context.citation_mapper import CitationMapper
from financial_rag.notebooks.api import NotebookQuery
from financial_rag.notebooks.records import NotebookError
from financial_rag.notebooks.service import NotebookService
from financial_rag.notebooks.structure import build_structure, locate_chunk
from financial_rag.schemas import ChunkType, FinancialChunk, RetrievalResult, StatementType
from financial_rag.schemas import ParsedQuery, QueryIntent
from financial_rag.schemas import FinancialAnswerResponse, ValidationResult
from financial_rag.retrieval.runtime import request_snapshot
from financial_rag.storage.sqlite_store import SQLiteFinancialStore

runtime = importlib.import_module("financial_rag.api.app")


def report(key="fpt", amount="100", year=2025):
    raw = (f"# Công ty FPT {year}\n\n===== PAGE 2 =====\n\n## Bảng cân đối kế toán\n\n"
           f"Đơn vị: Triệu đồng. Báo cáo hợp nhất.\n\n"
           f"| Chỉ tiêu | {year} | {year - 1} |\n| --- | --- | --- |\n"
           f"| Tổng tài sản | {amount} | 80 |\n\n## Chiến lược\n\n"
           "Doanh nghiệp tập trung phát triển sản phẩm, đào tạo nhân lực và mở rộng thị trường quốc tế.")
    chunk = FinancialChunk(chunk_id=f"{key}:same-row", source_file=f"{key}.md", document_id=key,
        company_code="FPT", fiscal_year=year, comparative_fiscal_year=year - 1,
        page=2, report_type="consolidated", statement_type=StatementType.BALANCE_SHEET,
        chunk_type=ChunkType.FINANCIAL_ITEM, canonical_metric="TỔNG TÀI SẢN", unit="Triệu đồng",
        content=f"Tổng tài sản\nNăm nay ({year}): {amount} Triệu đồng\nNăm trước ({year - 1}): 80 Triệu đồng",
        metadata={"metric_name": "Tổng tài sản", "metric": "Tổng tài sản", "end_value": amount,
                  "start_value": "80", "period_values": {str(year): amount, str(year - 1): "80"}})
    return {"logical_key": key, "document_id": key, "source_file": f"{key}.md",
            "company_code": "FPT", "fiscal_year": year, "report_type": "consolidated",
            "raw_content": raw, "chunks": [chunk]}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("FINANCIAL_ENABLE_DENSE", "false")
    store = SQLiteFinancialStore(tmp_path / "data.sqlite")
    store.replace_documents([report()])
    monkeypatch.setattr(runtime, "document_store", store)
    service = NotebookService(store, runtime)
    notebook = service.create("Kiểm thử")
    notebook = service.attach(notebook["id"], "fpt", notebook["revision"])
    return store, service, notebook


def ask(service, notebook, query="Tổng tài sản FPT năm 2025 hợp nhất là bao nhiêu?", **values):
    return service.answer(notebook["id"], NotebookQuery(query=query,
        notebook_revision=notebook["revision"], **values))


def test_notebooks_use_pinned_values_without_global_corpus(workspace, monkeypatch):
    store, service, first = workspace
    monkeypatch.setattr(runtime, "CHUNKS_DB", [report("outside", "9999")["chunks"][0]])
    store.replace_documents([report(amount="200")])
    second = service.create("Bản mới")
    second = service.attach(second["id"], "fpt", second["revision"])
    one, two = ask(service, first), ask(service, second)
    assert "100" in one["answer"] and "200" not in one["answer"], one
    assert "200" in two["answer"], two
    assert one["citations"] and two["citations"]
    assert one["citations"][0]["version_id"] != two["citations"][0]["version_id"]
    assert "9999" not in one["answer"] + two["answer"]
    citation = one["citations"][0]
    content = service.content(first["id"], citation["notebook_source_id"],
                              run_id=one["run_id"], block_id=citation["block_id"])
    assert "100" in content["text"] and "200" not in content["text"]


def test_empty_or_foreign_source_selection_does_not_widen(workspace):
    _, service, notebook = workspace
    for sources in ([], ["foreign-source"]):
        with pytest.raises(NotebookError):
            ask(service, notebook, source_ids=sources)
    answer = ask(service, notebook, query="Tổng tài sản VPB năm 2025 là bao nhiêu?")
    assert answer["answer_status"] in {"insufficient_evidence", "needs_clarification"}
    assert answer["citations"] == []


def test_revision_checks_and_cross_notebook_conversation(workspace):
    _, service, notebook = workspace
    result = ask(service, notebook)
    other = service.create("Khác")
    other = service.attach(other["id"], "fpt", other["revision"])
    with pytest.raises(NotebookError, match="Không tìm thấy"):
        ask(service, other, conversation_id=result["conversation_id"], conversation_revision=1)
    service.update(notebook["id"], notebook["revision"], title="Đổi tên")
    with pytest.raises(NotebookError) as conflict:
        ask(service, notebook)
    assert conflict.value.status == 409


def test_saved_answer_survives_source_removal_and_restart(workspace):
    store, service, notebook = workspace
    result = ask(service, notebook, query="Tóm tắt tài liệu", mode="summary")
    note = service.save_note(notebook["id"], "Tóm tắt", "Client cannot replace saved evidence", result["run_id"])
    assert note["content"] == result["answer"]
    service.remove(notebook["id"], notebook["sources"][0]["id"], notebook["revision"])
    reopened = NotebookService(SQLiteFinancialStore(store.database_path), runtime)
    assert reopened.notes(notebook["id"])[0]["stale"]
    content = reopened.content(notebook["id"], notebook["sources"][0]["id"], run_id=result["run_id"])
    assert "Tổng tài sản" in content["text"]
    with pytest.raises(NotebookError):
        reopened.content(notebook["id"], notebook["sources"][0]["id"])
    restored = reopened.conversation(notebook["id"], result["conversation_id"])
    assert restored["turns"][0]["response"]["run_id"] == result["run_id"]


def test_historical_storage_checks_ownership_and_checksum(workspace):
    store, _, notebook = workspace
    projection = notebook["sources"][0]["active_projection_id"]
    with pytest.raises(ValueError):
        store.get_projection_document("wrong-document", projection)
    with store._connect() as connection:
        connection.execute("UPDATE projection_chunk_snapshots SET payload_json = ? WHERE projection_id = ?",
                           (json.dumps([]), projection))
    with pytest.raises(ValueError, match="checksum"):
        store.get_projection_document("fpt", projection)


def test_legacy_active_snapshot_upgrade_is_verified(workspace):
    store, _, notebook = workspace
    projection = notebook["sources"][0]["active_projection_id"]
    with store._connect() as connection:
        connection.execute("DELETE FROM projection_chunk_snapshots")
    assert store.get_projection_document("fpt", projection)["chunks"]
    store.replace_documents([report(amount="200")])
    with store._connect() as connection:
        connection.execute("DELETE FROM projection_chunk_snapshots WHERE projection_id = ?", (projection,))
    with pytest.raises(ValueError, match="Historical"):
        store.get_projection_document("fpt", projection)


def test_followup_resolves_year_but_explicit_company_wins(workspace):
    _, service, notebook = workspace
    first = ask(service, notebook)
    followup = ask(service, notebook, query="Còn năm trước?", conversation_id=first["conversation_id"],
                   conversation_revision=first["conversation_revision"])
    assert "2024" in followup["resolved_query"] and "FPT" in followup["resolved_query"]
    assert "80" in followup["answer"], followup
    next_turn = ask(service, notebook, query="Còn tổng tài sản VPB năm 2025?",
                    conversation_id=followup["conversation_id"], conversation_revision=followup["conversation_revision"])
    assert "FPT" not in next_turn["resolved_query"]
    assert not next_turn["citations"]


def test_changed_sources_require_explicit_followup(workspace):
    _, service, notebook = workspace
    first = ask(service, notebook)
    updated = service.update(notebook["id"], notebook["revision"], sources=[])
    with pytest.raises(NotebookError):
        ask(service, updated, query="Còn năm trước?", conversation_id=first["conversation_id"],
            conversation_revision=first["conversation_revision"])


def test_structure_offsets_atomic_tables_and_unique_locations(workspace):
    store, _, notebook = workspace
    document = store.get_projection_document("fpt", notebook["sources"][0]["active_projection_id"])
    structure = build_structure(document)
    tables = [block for block in structure["blocks"] if block["type"] == "table"]
    assert len(tables) == 1 and tables[0]["page"] == 2
    text = document["raw_content"][tables[0]["start"]:tables[0]["end"]]
    assert "2025" in text and "100" in text and "80" in text
    located = locate_chunk(document["chunks"][0], structure, document["raw_content"])
    assert located == tables[0]
    assert all(document["raw_content"][b["start"]:b["end"]].strip() for b in structure["blocks"])
    changed = {**document, "raw_content": document["raw_content"] + "\n\nTổng tài sản 100 và 80 cũng xuất hiện ở đây."}
    assert locate_chunk(document["chunks"][0], build_structure(changed), changed["raw_content"]) is None


def test_extractive_reader_budget_reports_incomplete_coverage(workspace, monkeypatch):
    _, service, notebook = workspace
    monkeypatch.setenv("NOTEBOOK_READER_TOKEN_BUDGET", "220")
    result = ask(service, notebook, query="Tóm tắt", mode="summary")
    assert result["context_budget"]["estimated_tokens"] <= result["context_budget"]["budget_tokens"]
    assert result["context_budget"]["omitted"]
    assert result["answer_status"] in {"partial", "insufficient_evidence"}


def test_context_budget_keeps_child_whole_and_omits_oversized_parent():
    parent = FinancialChunk(chunk_id="parent", source_file="test", content="X" * 10000)
    child = FinancialChunk(chunk_id="child", source_file="test", content="Số tiền: 123 Triệu đồng",
                           parent_chunk_id="parent")
    context, citations, evidence = ContextAssembler().assemble_with_sources(
        [RetrievalResult(chunk=child, rrf_score=1)], {"parent": parent}, max_tokens=200)
    assert child.content in context and "XXXX" not in context and "XXXX" not in str(evidence)
    assert citations[0].chunk_id == "child" and estimate_tokens(context) <= 200
    assert TokenBudget(100, 90, query_tokens=20).available == 0


def test_notebook_api_persists_scoped_run_and_rejects_foreign_reads(workspace):
    _, _, notebook = workspace
    client = TestClient(runtime.app)
    base = f"/api/v1/notebooks/{notebook['id']}"
    response = client.post(base + "/answer", json={"query": "Đọc phần Chiến lược", "mode": "reader",
                                                  "notebook_revision": notebook["revision"]})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["citations"] and result["reader_mode"] == "source_extract"
    assert "đào tạo nhân lực" in result["answer"]
    other = client.post("/api/v1/notebooks", json={"title": "Other"}).json()
    assert client.get(f"/api/v1/notebooks/{other['id']}/runs/{result['run_id']}").status_code == 404
    assert client.get(base + f"/sources/unknown/content?run_id={result['run_id']}").status_code == 404
    assert client.delete(base + f"/sources/{notebook['sources'][0]['id']}?expected_revision=1").status_code == 409


def test_llm_request_counts_instructions_query_context_and_output(monkeypatch):
    engine = FinancialChatEngine()
    engine.num_ctx = 8192
    engine.num_predict = 256
    chunk = FinancialChunk(chunk_id="narrative", source_file="policy.md",
                           content="Doanh nghiệp áp dụng chính sách ghi nhận doanh thu theo hợp đồng.")
    results = [RetrievalResult(chunk=chunk, rrf_score=1)]
    parsed = ParsedQuery(original_query="Giải thích chính sách này", intent=QueryIntent.EXPLANATION)
    captured = []

    def post(url, *, json, timeout):
        prompt = json["messages"][0]["content"]
        assert estimate_tokens(prompt) + json["options"]["num_predict"] + 128 <= json["options"]["num_ctx"]
        assert chunk.content in prompt
        captured.append(prompt)

        class Response:
            status_code = 200

            def json(self):
                return {"message": {"content": chunk.content + " [c1]"}}

        return Response()

    monkeypatch.setattr("financial_rag.generation.chat.requests.post", post)
    _, citations, _ = engine.generate_answer(parsed.original_query, parsed, results)
    assert len(captured) == 1 and citations[0].chunk_id == chunk.chunk_id
    engine.num_ctx = 1024
    answer, citations, _ = engine.generate_answer("Câu hỏi " * 1000, parsed, results)
    assert "ngân sách" in answer and not citations and len(captured) == 1


def test_structures_are_pinned_and_corrupt_structure_is_rejected(workspace, monkeypatch):
    _, service, notebook = workspace
    source = notebook["sources"][0]
    original = service.guide(notebook["id"], source["id"])["structure"]

    def wrong_structure(_):
        raise AssertionError("Pinned structures must not be reparsed")

    monkeypatch.setattr("financial_rag.notebooks.service.build_structure", wrong_structure)
    assert service.guide(notebook["id"], source["id"])["structure"] == original
    service.records.put("structure", {**original, "blocks": []}, expected=original["revision"])
    with pytest.raises(NotebookError, match="không khớp"):
        service.guide(notebook["id"], source["id"])


def test_generic_document_ingest_and_notebook_reading(workspace):
    store, service, _ = workspace
    client = TestClient(runtime.app)
    raw = "# Hướng dẫn học tập\n\n## Ôn tập\n\nÔn tập cách quãng giúp người học củng cố kiến thức qua các lần luyện tập."
    response = client.post("/api/v1/financial/ingest", json={"content": raw, "source_file": "study-guide.md"})
    assert response.status_code == 200, response.text
    document = next(item for item in store.list_documents() if item["source_file"] == "study-guide.md")
    notebook = service.create("Tài liệu tổng quát")
    notebook = service.attach(notebook["id"], document["logical_key"], notebook["revision"])
    result = ask(service, notebook, query="Đọc phần Ôn tập", mode="reader")
    assert "củng cố kiến thức" in result["answer"] and result["citations"]
    assert result["reader_mode"] == "source_extract"
    assert client.post(f"/api/v1/notebooks/{notebook['id']}/answer", json={
        "query": "   ", "notebook_revision": notebook["revision"]}).status_code == 422


def test_model_prose_is_not_promoted_to_executor_checked(workspace, monkeypatch):
    _, service, notebook = workspace

    def generated(request):
        snapshot = request_snapshot.get()
        citations, _ = CitationMapper.map_citations([RetrievalResult(chunk=snapshot.chunks[0], rrf_score=1)])
        generation_budget_trace.set({"model_called": True})
        return FinancialAnswerResponse(query=request.query,
            parsed_query=runtime.query_engine.parse(request.query), citations=citations,
            validation=ValidationResult(is_valid=True),
            answer="Công ty này sẽ tăng trưởng mạnh và có chất lượng tài sản vượt trội [c1].")

    monkeypatch.setattr(runtime, "answer_question", generated)
    result = ask(service, notebook)
    assert result["answer_status"] == "insufficient_evidence" and not result["validation"]["is_valid"]
    assert "tăng trưởng mạnh" not in result["answer"]
    assert result["claim_checks"][0]["status"] == "unverified"


def test_summary_reaches_later_pages_instead_of_cover_boilerplate(workspace, monkeypatch):
    store, service, _ = workspace
    document = report("long_report")
    front = "===== PAGE 1 =====\n\n" + "\n\n".join(f"Dòng bìa {n}" for n in range(200))
    document["raw_content"] = front + "\n\n===== PAGE 9 =====\n\n" + document["raw_content"].replace("===== PAGE 2 =====", "")
    store.replace_documents([document])
    notebook = service.create("Long")
    notebook = service.attach(notebook["id"], "long_report", notebook["revision"])
    monkeypatch.setenv("NOTEBOOK_READER_TOKEN_BUDGET", "1200")
    result = ask(service, notebook, query="Tóm tắt tài liệu", mode="summary")
    assert "| Tổng tài sản | 100 | 80 |" in result["answer"]
    assert any(citation["page"] == 9 for citation in result["citations"])
    assert "Dòng bìa" not in result["answer"]
    assert result["context_budget"]["selection_method"] == "page_section_diversity"
    assert result["answer_status"] == "partial"


def test_reader_does_not_answer_explicit_outside_company(workspace):
    _, service, notebook = workspace
    result = ask(service, notebook, query="Đọc chính sách của VPB", mode="reader")
    assert result["answer_status"] == "insufficient_evidence"
    assert result["citations"] == []
