import importlib
from concurrent.futures import ThreadPoolExecutor
from threading import Event

from fastapi.testclient import TestClient

from financial_rag.retrieval.hybrid import FinancialHybridRetriever, _scope_reports
from financial_rag.retrieval.runtime import RequestSnapshot, ScopeCache
from financial_rag.schemas import FinancialChunk, ChunkType, StatementType
from financial_rag.structure.semantics import period_supported
from financial_rag.validation.claims import check_claims

api = importlib.import_module("financial_rag.api.app")


def row(company="FPT", year=2025, value="123", header=None):
    return FinancialChunk(chunk_id=f"{company}:{year}", source_file=f"{company}_{year}.txt",
        company_code=company, fiscal_year=year, report_type="consolidated", content=f"TỔNG TÀI SẢN {value} Triệu đồng",
        unit="Triệu đồng", chunk_type=ChunkType.FINANCIAL_ITEM, statement_type=StatementType.BALANCE_SHEET,
        metadata={"metric_name": "TỔNG TÀI SẢN", "period_values": {str(year): value},
                  "source_period_columns": {str(year): 1}, "table_headers": ["Chỉ tiêu", header or str(year)]})


def test_scope_cache_is_bounded_and_version_identity_changes():
    cache = ScopeCache(max_scopes=2, max_chunks=3)
    cache.put(RequestSnapshot((1, 2), None, snapshot_id="old"))
    cache.put(RequestSnapshot((3, 4), None, snapshot_id="new"))
    assert cache.get("old") is None and cache.stats()["chunks"] == 2
    first = [{"logical_key": "doc", "active_projection_id": "p1"}]
    assert cache.key(first) != cache.key([{**first[0], "active_projection_id": "p2"}])


def test_source_policy_uses_latest_comparative_only_when_explicit():
    original = row(year=2024, value="100")
    restated = row(year=2025, value="120")
    restated.metadata["period_values"]["2024"] = "110"
    assert _scope_reports([original, restated], [2024], ["FPT"], "consolidated") == [original]
    assert _scope_reports([original, restated], [2024], ["FPT"], "consolidated", "latest_restated") == [restated]


def test_annual_report_is_not_a_quarterly_note():
    from financial_rag.metadata.period_extractor import PeriodExtractor
    assert PeriodExtractor.extract_document_quarter("Báo cáo tài chính hợp nhất năm 2025 đã kiểm toán. Chi phí quý 1 biến động.") is None
    assert PeriodExtractor.extract_document_quarter("BÁO CÁO TÀI CHÍNH HỢP NHẤT QUÝ II NĂM 2025") == 2
    assert PeriodExtractor.extract_document_quarter("", "FPT_BCTC_Q3_2025.txt") == 3


def test_contradicted_maturity_and_currency_headers_are_not_periods():
    assert period_supported(row(header="Không kỳ hạn"), 2025) is False
    assert period_supported(row(header="Dưới 3 tháng"), 2025) is False
    assert period_supported(row(header="USD"), 2025) is False
    assert period_supported(row(header="2024"), 2025) is False
    assert period_supported(row(header="31/12/2025"), 2025) is True
    assert period_supported(row(header="31/12/2025VND"), 2025) is True
    assert period_supported(row(header="Tại ngày 31 tháng 12 năm"), 2025) is None
    opening = row(year=2024, header="01/01/2025")
    opening.metadata["source_statement_type"] = "balance_sheet"
    assert period_supported(opening, 2024) is True


def test_retriever_build_does_not_mutate_published_chunks(monkeypatch):
    monkeypatch.setenv("FINANCIAL_ENABLE_DENSE", "false")
    source = row()
    before = source.model_dump()
    built = api.build_retriever([source], isolated=True)
    assert source.model_dump() == before and built.chunks[0] is not source


def test_request_context_keeps_old_retriever_during_publication(monkeypatch):
    old = row(value="100")
    index = FinancialHybridRetriever([old])
    monkeypatch.setattr(api, "retriever", index)
    monkeypatch.setattr(api, "CHUNKS_DB", [old])
    monkeypatch.setattr(api, "document_store", None)
    started, proceed = Event(), Event()

    @api._with_request_snapshot
    def request(req):
        started.set()
        assert proceed.wait(5)
        return api._runtime_chunks()[0].metadata["period_values"]["2025"], api._runtime_retriever()

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(request, api.QueryRequest(query="Tổng tài sản FPT năm 2025"))
        assert started.wait(5)
        new = row(value="200")
        monkeypatch.setattr(api, "CHUNKS_DB", [new])
        monkeypatch.setattr(api, "retriever", FinancialHybridRetriever([new]))
        proceed.set()
        assert pending.result() == ("100", index)


def test_scoped_startup_does_not_load_corpus(monkeypatch):
    class Store:
        def read_scope(self, *args, **kwargs):
            pass
        def load_company_aliases(self):
            return {}
        def load_chunks(self):
            raise AssertionError("Scoped startup must not load all chunks")
    monkeypatch.setattr(api, "document_store", Store())
    monkeypatch.setenv("FINANCIAL_SCOPED_RUNTIME", "true")
    api.restore_persisted_documents()
    assert api.CHUNKS_DB == [] and api.retriever is None


def test_multiissuer_ranking_retains_missing_issuer(monkeypatch):
    chunks = [row(company="FPT", value="200"), row(company="BMP", value="100")]
    monkeypatch.setattr(api, "CHUNKS_DB", chunks)
    monkeypatch.setattr(api, "retriever", FinancialHybridRetriever(chunks))
    monkeypatch.setattr(api, "document_store", None)
    data = TestClient(api.app).post("/api/v1/financial/answer", json={
        "query": "Xếp hạng tổng tài sản FPT, BMP và VIC năm 2025 theo báo cáo hợp nhất"}).json()
    assert data["validation"]["is_valid"], data
    assert data["answer_status"] == "partial" and "VIC" in str(data["analysis_coverage"])
    assert "| 1 | FPT |" in data["answer"] and "| 2 | BMP |" in data["answer"]


def test_valid_citation_does_not_verify_causal_paraphrase():
    from financial_rag.schemas import Citation
    source = row()
    citation = Citation(citation_id="c1", chunk_id=source.chunk_id, document_id="doc", file_name=source.source_file, snippet=source.content)
    checks = check_claims("Tài sản tăng do quản trị tốt [c1]", [citation], {source.chunk_id: source}, deterministic=False)
    assert checks[0]["status"] == "unverified" and not checks[0]["human_reviewed"]


def test_sparse_snapshot_survives_rebuild_and_rejects_corruption(monkeypatch, tmp_path):
    from financial_rag.retrieval.bm25_retriever import BM25Retriever
    monkeypatch.setenv("FINANCIAL_SPARSE_INDEX_DIR", str(tmp_path))
    chunks = [row(), row(company="BMP")]
    original = BM25Retriever(chunks).search("TỔNG TÀI SẢN FPT")
    restored = BM25Retriever(chunks)
    assert restored.persisted_index_hit and restored.search("TỔNG TÀI SẢN FPT") == original
    for path in tmp_path.glob("*.json"):
        path.write_text("corrupt", encoding="utf-8")
    rebuilt = BM25Retriever(chunks)
    assert not rebuilt.persisted_index_hit and rebuilt.search("TỔNG TÀI SẢN FPT") == original


def test_roa_roe_are_computed_for_every_requested_year(monkeypatch):
    chunks = []
    for year, profit, opening_assets, assets, opening_equity, equity in (
        (2024, "30", "1000", "1500", "200", "400"),
        (2025, "50", "1500", "2000", "400", "500"),
    ):
        for label, current, prior, statement in (
            ("LỢI NHUẬN SAU THUẾ", profit, "20", StatementType.INCOME_STATEMENT),
            ("TỔNG TÀI SẢN", assets, opening_assets, StatementType.BALANCE_SHEET),
            ("TỔNG VỐN CHỦ SỞ HỮU", equity, opening_equity, StatementType.BALANCE_SHEET),
            ("Tổng thu nhập hoạt động", "200", "100", StatementType.INCOME_STATEMENT),
        ):
            chunks.append(FinancialChunk(chunk_id=f"{year}:{label}", document_id=f"report:{year}",
                source_file=f"VPB_{year}.txt", company_code="VPB", fiscal_year=year,
                comparative_fiscal_year=year-1, report_type="consolidated", unit="Triệu đồng",
                chunk_type=ChunkType.FINANCIAL_ITEM, statement_type=statement,
                content=f"Chỉ tiêu: {label}. Năm {year}: {current} Triệu đồng. Năm {year-1}: {prior} Triệu đồng.",
                metadata={"metric_name": label, "period_values": {str(year): current, str(year-1): prior},
                          "schema_node_id": "INCOME.BANK.TOTAL_OPERATING_INCOME" if label == "Tổng thu nhập hoạt động" else ""}))
    monkeypatch.setattr(api, "CHUNKS_DB", chunks)
    monkeypatch.setattr(api, "retriever", FinancialHybridRetriever(chunks))
    monkeypatch.setattr(api, "document_store", None)
    result = TestClient(api.app).post("/api/v1/financial/answer", json={
        "query": "So sánh ROA và ROE VPB năm 2024 và 2025 theo báo cáo hợp nhất"}).json()
    returns = [calc for calc in result["calculations"] if calc["formula_id"].startswith("return_on_average")]
    assert result["validation"]["is_valid"], result
    assert sorted(calc["value"] for calc in returns) == ["10.00", "11.11", "2.40", "2.86"]
    for calc in returns:
        assert {fact["year"] for fact in calc["inputs"]} in ({2023, 2024}, {2024, 2025})
