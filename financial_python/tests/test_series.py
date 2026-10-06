import importlib
from types import SimpleNamespace

from fastapi.testclient import TestClient

from financial_rag.analysis.series import build_series_answer, series_within_budget
from financial_rag.analysis.overview import _select_period_fact
from financial_rag.context.citation_mapper import CitationMapper
from financial_rag.query.understanding import QueryUnderstandingEngine
from financial_rag.retrieval.hybrid import FinancialHybridRetriever
from financial_rag.schemas import ChunkType, FinancialChunk, ParsedQuery, RetrievalResult, StatementType
from financial_rag.storage.sqlite_store import SQLiteFinancialStore

api = importlib.import_module("financial_rag.api.app")


def row(company, year, value, *, unit="Triệu đồng"):
    return FinancialChunk(
        chunk_id=f"{company}-{year}-{value}", document_id=f"{company}-{year}",
        source_file=f"{company}_{year}_Hopnhat.txt", company_code=company,
        fiscal_year=year, report_type="consolidated", page=1, unit=unit,
        statement_type=StatementType.BALANCE_SHEET, chunk_type=ChunkType.FINANCIAL_ITEM,
        content=f"Chỉ tiêu gốc trong BCTC: TỔNG TÀI SẢN\nSố cuối năm ({year}): {value} {unit}",
        metadata={"metric_name": "TỔNG TÀI SẢN", "period_values": {str(year): str(value)}},
    )


def test_multi_period_api_uses_cited_arithmetic_without_llm(monkeypatch):
    chunks = [row("VPB", year, value) for year, value in ((2023, 100), (2024, 120), (2025, 150))]
    monkeypatch.setattr(api, "CHUNKS_DB", chunks)
    monkeypatch.setattr(api, "retriever", FinancialHybridRetriever(chunks))
    monkeypatch.setattr("financial_rag.generation.chat.requests.post", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("multi-period arithmetic must not call the LLM")))
    response = TestClient(api.app).post("/api/v1/financial/answer", json={
        "query": "So sánh tổng tài sản VPB năm 2023, 2024 và 2025 theo báo cáo hợp nhất"})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["answer_status"] == "complete", data
    assert data["validation"]["is_valid"], data
    assert {item["value"] for item in data["calculations"] if item["formula_id"] == "growth_rate"} == {"20.00", "25.00"}
    assert len(data["citations"]) == 3


def test_three_issuer_api_retains_each_requested_company(monkeypatch):
    chunks = [row(company, 2025, value) for company, value in (("VPB", 100), ("ACB", 120), ("MBB", 150))]
    monkeypatch.setattr(api, "CHUNKS_DB", chunks)
    monkeypatch.setattr(api, "retriever", FinancialHybridRetriever(chunks))
    monkeypatch.setattr("financial_rag.generation.chat.requests.post", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("comparison must not call the LLM")))
    data = TestClient(api.app).post("/api/v1/financial/answer", json={
        "query": "So sánh tổng tài sản VPB, ACB và MBB năm 2025 theo báo cáo hợp nhất"}).json()
    assert data["answer_status"] == "complete", data
    assert data["validation"]["is_valid"], data
    assert len(data["citations"]) == 3


def test_missing_middle_year_does_not_interpolate_or_create_pair():
    chunks = [row("VPB", 2023, 100), row("VPB", 2025, 150)]
    results = [RetrievalResult(chunk=chunk, rrf_score=1) for chunk in chunks]
    citations, _ = CitationMapper.map_citations(results)
    metric = next(metric for metric in QueryUnderstandingEngine().metric_mapper.metrics_list
                  if metric.metric_id == "tong_cong_tai_san")
    answer = build_series_answer(SimpleNamespace(planned_metric_ids=[metric.metric_id]),
        ParsedQuery(original_query="compare", company_codes=["VPB"], fiscal_years=[2023, 2024, 2025]),
        {metric.metric_id: metric}, {metric.metric_id: results}, citations)
    assert answer.answer_status == "partial" and not answer.calculations
    assert "kỳ 2024" in " ".join(answer.missing_components)


def test_work_budget_is_bounded():
    assert series_within_budget(["VPB"], [2023, 2024, 2025], ["assets"])
    assert not series_within_budget(["VPB"] * 6, [2025], ["assets"])
    assert not series_within_budget(["VPB"], list(range(2010, 2025)), ["assets"])
    assert not series_within_budget(["VPB"] * 5, list(range(2010, 2020)), ["a", "b", "c"])


def test_statement_total_is_not_confused_with_currency_risk_total():
    primary = row("MBB", 2025, 1473741402)
    primary.metadata["metric_name"] = "TỔNG NỢ PHẢI TRẢ"
    note = row("MBB", 2025, 533578174)
    note.metadata.update({"metric_name": "Tổng nợ phải trả (2)",
                          "table_title": "51. RÙI RO TIÊN TỆ (tiếp theo)"})
    results = [RetrievalResult(chunk=chunk, rrf_score=score)
               for chunk, score in ((primary, 1), (note, 10))]
    citations, _ = CitationMapper.map_citations(results)
    metric = next(metric for metric in QueryUnderstandingEngine().metric_mapper.metrics_list
                  if metric.metric_id == "tong_no_phai_tra")
    fact, error = _select_period_fact(metric.metric_id, metric, 2025, results,
                                      {item.chunk_id: item for item in citations}, "MBB", "consolidated")
    assert not error and fact.chunk.chunk_id == primary.chunk_id


def test_fact_endpoints_validate_scope_and_pagination(monkeypatch, tmp_path):
    store = SQLiteFinancialStore(tmp_path / "reports.sqlite3")
    store.replace_documents([{"logical_key": "doc", "document_id": "doc", "source_file": "VPB.txt",
                              "raw_content": "source", "chunks": [row("VPB", 2025, 100)]}])
    monkeypatch.setattr(api, "document_store", store)
    client = TestClient(api.app)
    data = client.get("/api/v1/financial/documents/doc/facts?period_year=2025").json()
    assert data["total"] == 1 and data["facts"][0]["value"] == "100"
    assert client.get("/api/v1/financial/documents/doc/facts?limit=1001").status_code == 422
    assert client.get("/api/v1/financial/documents/doc/facts?offset=-1").status_code == 422
    assert client.get("/api/v1/financial/documents/other/facts", params={
        "projection_id": data["projection_id"]}).status_code == 404
    assert len(client.get("/api/v1/financial/documents/doc/versions").json()) == 1
