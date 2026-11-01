import importlib
from pathlib import Path
import pytest

from fastapi.testclient import TestClient
from financial_rag.chunking.financial_chunker import FinancialMarkdownChunker
from financial_rag.api.app import app
from financial_rag.context.assembler import ContextAssembler
from financial_rag.generation.chat import FinancialGenerationError
from financial_rag.retrieval.hybrid import FinancialHybridRetriever
from financial_rag.schemas import ChunkType, FinancialChunk, QueryIntent, RetrievalResult, StatementType
from financial_rag.storage.sqlite_store import SQLiteFinancialStore

client = TestClient(app)
api_module = importlib.import_module("financial_rag.api.app")


def _synthetic_mbb_balance_sheet_chunks(years=(2024, 2025)):
    """Return fictional statement rows for offline regression tests; values are not real."""
    fixtures = {
        2024: {
            "assets": "1.500.000",
            "liabilities": "1.000.000",
            "deposits": "600.000",
            "borrowings": "250.000",
            "other_liabilities": "150.000",
            "equity": "500.000",
            "capital": "450.000",
            "retained_earnings": "50.000",
        },
        2025: {
            "assets": "2.162.000",
            "liabilities": "1.462.000",
            "deposits": "807.214",
            "borrowings": "500.000",
            "other_liabilities": "154.786",
            "equity": "700.000",
            "capital": "630.000",
            "retained_earnings": "70.000",
        },
    }
    chunks = []
    prior_keys = {
        "total-assets": "assets",
        "total-liabilities": "liabilities",
        "customer-deposits": "deposits",
        "total-equity": "equity",
        "capital": "capital",
        "retained-earnings": "retained_earnings",
    }
    prior_keys["borrowings"] = "borrowings"
    prior_keys["other-liabilities"] = "other_liabilities"

    def add(year, page, key, label, value, *, node=None, parent=None, children=None):
        chunk_id = f"synthetic-mbb-{year}-{key}"
        prior_key = prior_keys.get(key)
        metadata = {
            "metric_name": label,
            "canonical_metric": label,
            "period_values": (
                {"2024": fixtures[2024][prior_key], "2025": value}
                if year == 2025 and prior_key
                else {str(year): value}
            ),
            "row_index": len(chunks),
        }
        if node:
            metadata["schema_node_id"] = node
        if parent:
            metadata["schema_parent_id"] = parent
            metadata["schema_node_kind"] = "line"
        if children:
            metadata["schema_children"] = children
            metadata["schema_node_kind"] = "total"
        if key == "total-assets":
            metadata["metric_aliases"] = ["Tổng cộng tài sản", "TỔNG CỘNG TÀI SẢN"]
        chunks.append(FinancialChunk(
            chunk_id=chunk_id,
            source_file=f"MBB_synthetic_test_fixture_{year}.txt",
            document_id=f"mbb-synthetic-{year}",
            company_code="MBB",
            fiscal_year=year,
            comparative_fiscal_year=year - 1,
            report_type="consolidated",
            page=page,
            statement_type=StatementType.BALANCE_SHEET,
            canonical_metric=label,
            chunk_type=ChunkType.FINANCIAL_ITEM,
            unit="Triệu đồng",
            content=(
                f"Chỉ tiêu gốc trong fixture kiểm thử: {label}"
                + "\n".join(
                    f"Năm {period}: {period_value} Triệu đồng"
                    for period, period_value in metadata["period_values"].items()
                )
            ),
            metadata=metadata,
        ))

    asset_children = [
        "ASSET.CASH_EQUIV.CASH_ON_HAND",
        "ASSET.INTERBANK.DEPOSITS",
        "ASSET.FIN_INV.TRADING_SECURITIES",
        "ASSET.LOANS.CUSTOMERS",
        "ASSET.FIXED_ASSETS",
        *(f"ASSET.SYNTHETIC_COMPONENT_{index:02d}" for index in range(1, 31)),
    ]
    synthetic_asset_labels = (
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
        "hotel", "india", "juliet", "kilo", "lima", "mike", "november",
        "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform",
        "victor", "whiskey", "xray", "yankee", "zulu", "amber", "birch",
    )
    liability_children = [
        "LIABILITY.CUSTOMER_DEPOSITS",
        "LIABILITY.BANK_BORROWINGS",
        "LIABILITY.OTHER",
    ]
    equity_children = ["EQUITY.OWNER_CAPITAL", "EQUITY.RETAINED_EARNINGS"]

    for year in years:
        fixture = fixtures[year]
        if year == 2024:
            asset_rows = [
                ("cash", "Tiền mặt, vàng bạc, đá quý", "100.000", asset_children[0], "ASSET"),
                ("deposits", "Tiền gửi tại Ngân hàng Nhà nước", "100.000", asset_children[1], "ASSET"),
                ("securities", "Chứng khoán kinh doanh", "200.000", asset_children[2], "ASSET"),
                ("loans", "Cho vay khách hàng", "500.000", asset_children[3], "ASSET"),
                ("fixed-assets", "Tài sản cố định", "50.000", asset_children[4], "ASSET"),
            ]
            asset_rows.extend(
                (f"component-{label}", f"Khoản mục tài sản kiểm thử {label}", "20.000", node, "ASSET")
                for label, node in zip(synthetic_asset_labels, asset_children[7:])
            )
            asset_rows[-1] = (*asset_rows[-1][:2], "10.000", *asset_rows[-1][3:])
            for key, label, value, node, parent in asset_rows:
                add(year, 10, key, label, value, node=node, parent=parent)
            # Report totals follow the component rows in statement order.
            add(
                year, 10, "total-assets", "TỔNG TÀI SẢN", fixture["assets"],
                node="TOTAL_ASSETS", parent="ASSET", children=asset_children,
            )

        # Deliberately leave customer deposits unmapped to exercise source-label fallback.
        add(year, 11, "customer-deposits", "Tiền gửi của khách hàng", fixture["deposits"])
        add(year, 11, "borrowings", "Tiền vay từ các tổ chức tín dụng", fixture["borrowings"], node=liability_children[1], parent="TOTAL_LIABILITIES")
        add(year, 11, "other-liabilities", "Các khoản nợ khác", fixture["other_liabilities"], node=liability_children[2], parent="TOTAL_LIABILITIES")
        add(
            year, 11, "total-liabilities", "TỔNG NỢ PHẢI TRẢ", fixture["liabilities"],
            node="TOTAL_LIABILITIES", parent="LIABILITY", children=liability_children,
        )
        add(year, 11, "retained-earnings", "Lợi nhuận giữ lại", fixture["retained_earnings"], node=equity_children[1], parent="TOTAL_EQUITY")
        add(
            year, 11, "capital", "Vốn điều lệ", fixture["capital"], node=equity_children[0], parent="TOTAL_EQUITY"
        )
        add(
            year, 11, "total-equity", "TỔNG VỐN CHỦ SỞ HỮU", fixture["equity"],
            node="TOTAL_EQUITY", parent="EQUITY", children=equity_children,
        )
        if year == 2025:
            # Simulate a legacy fuzzy mapping. The label must keep this combined
            # total from being mistaken for the liabilities root.
            add(year, 11, "combined-total", "TỔNG NỢ PHẢI TRẢ VÀ VỐN CHỦ SỞ HỮU", "2.162.000")
            chunks[-1].metadata["metric_aliases"] = ["Nợ phải trả", "Tổng nợ phải trả"]
            chunks[-1].metadata["metric_id"] = "tong_no_phai_tra"

    return chunks


@pytest.mark.parametrize("operation", ["composition", "contribution"])
@pytest.mark.parametrize("remove_schema", [False, True])
@pytest.mark.parametrize("include_prior_filing", [False, True])
def test_liabilities_source_bounds_and_contribution_include_unmapped_rows(monkeypatch, operation, remove_schema, include_prior_filing):
    chunks = _synthetic_mbb_balance_sheet_chunks((2024, 2025) if include_prior_filing else (2025,))
    for chunk in chunks:
        if remove_schema:
            for key in list(chunk.metadata):
                if key.startswith("schema_"):
                    chunk.metadata.pop(key)
        # Simulate legacy advisory mappings that confuse a child/combined
        # total with the statement root. Original labels remain intact.
        if chunk.page == 11:
            chunk.metadata["metric_aliases"] = ["Nợ phải trả", "Tổng nợ phải trả"]
            chunk.metadata["metric_id"] = "tong_no_phai_tra"
    monkeypatch.setattr(api_module, "CHUNKS_DB", chunks)
    monkeypatch.setattr(api_module, "retriever", FinancialHybridRetriever(chunks))
    monkeypatch.setattr(
        "financial_rag.generation.chat.requests.post",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("source analysis must not call the LLM")),
    )
    question = (
        "Cơ cấu nợ phải trả MBB năm 2025" if operation == "composition"
        else "Khoản nào đóng góp vào biến động nợ phải trả MBB năm 2024 và 2025"
    )
    response = client.post("/api/v1/financial/answer", json={"query": question})
    assert response.status_code == 200, response.text
    payload = response.json()
    answer = payload["answer"]
    assert payload["validation"]["is_valid"], payload["validation"]
    assert "807.214" in answer  # Synthetic customer deposits have no schema mapping.
    assert "1.462.000" in answer
    assert "Vốn điều lệ" not in answer
    assert "TỔNG VỐN CHỦ SỞ HỮU" not in answer
    assert "TỔNG NỢ PHẢI TRẢ VÀ VỐN CHỦ SỞ HỮU" not in answer
    assert all(citation["page"] == 11 for citation in payload["citations"])
    if operation == "composition":
        assert "tổng các dòng thành phần khớp" in answer
    else:
        assert "Đóng góp vào biến động" in answer
        assert "462.000" in answer
        assert "207.214" in answer
        assert "44,85%" in answer
        assert "khớp tổng báo cáo ở cả hai kỳ" in answer
        assert any(c["formula_id"] == "contribution_to_change" for c in payload["calculations"])


def test_source_root_recognition_ignores_fuzzy_metric_ids():
    root = FinancialChunk(
        chunk_id="root", source_file="source.txt", content="", chunk_type=ChunkType.FINANCIAL_ITEM,
        statement_type=StatementType.BALANCE_SHEET,
        metadata={"metric_name": "TỔNG NỢ PHẢI TRẢ", "metric_id": "other"},
    )
    assert api_module._balance_sheet_root_metric_id(root) == "tong_no_phai_tra"
    for label in ["Các khoản phải trả và công nợ khác", "Các khoản nợ khác", "TỔNG NỢ PHẢI TRẢ VÀ VỐN CHỦ SỞ HỮU"]:
        child = root.model_copy(update={"metadata": {"metric_name": label, "metric_id": "tong_no_phai_tra", "metric_aliases": ["Tổng nợ phải trả"]}})
        assert api_module._balance_sheet_root_metric_id(child) == ""


def test_period_comparison_uses_complete_separate_basis_when_consolidated_is_partial(monkeypatch):
    label = "Cộng doanh thu hoạt động"
    filings = [
        (2023, "consolidated", {2023: "1.800.000.000"}),
        (2023, "separate", {2023: "2.000.000.000"}),
        (2024, "separate", {2023: "2.000.000.000", 2024: "3.000.000.000"}),
    ]
    chunks = []
    for year, basis, values in filings:
        chunks.append(FinancialChunk(
            chunk_id=f"acbs-{basis}-{year}", source_file=f"ACBS_{basis}_{year}.txt",
            document_id=f"acbs-{basis}-{year}", company_code="ACBS", fiscal_year=year,
            comparative_fiscal_year=year - 1 if len(values) > 1 else None,
            report_type=basis, page=15, unit="VND",
            chunk_type=ChunkType.FINANCIAL_ITEM, statement_type=StatementType.INCOME_STATEMENT,
            content=f"Chỉ tiêu gốc trong BCTC: {label} | " + " | ".join(
                f"Năm {period}: {value} VND" for period, value in values.items()
            ),
            metadata={"metric_name": label, "period_values": {str(k): v for k, v in values.items()}},
        ))
    monkeypatch.setattr(api_module, "CHUNKS_DB", chunks)
    monkeypatch.setattr(api_module, "retriever", FinancialHybridRetriever(chunks))
    query = "So sánh tổng doanh thu hoạt động ACBS năm 2023 và 2024"
    response = client.post("/api/v1/financial/answer", json={"query": query})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["validation"]["is_valid"], data["validation"]
    assert "Cơ sở báo cáo: riêng lẻ" in data["answer"]
    assert any(
        item["formula_id"] == "growth_rate" and item["value"] == "50.00"
        for item in data["calculations"]
    )
    assert "1.000.000.000" in data["answer"]
    assert "1.800.000.000" not in data["answer"]
    assert {c["file_name"] for c in data["citations"]} == {"ACBS_separate_2024.txt"}
    evidence = client.post("/api/v1/financial/retrieve", json={"query": query}).json()
    assert evidence
    assert all(item["chunk"]["report_type"] == "separate" for item in evidence)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "financial-rag-python"


def test_liveness_and_readiness_probes_report_core_state():
    live = client.get("/health/live")
    ready = client.get("/health/ready")

    assert live.status_code == 200
    assert live.json()["status"] == "alive"
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert ready.json()["components"]["answer_model"] == "external_unchecked"


def test_query_endpoint_rejects_blank_and_oversized_queries():
    blank = client.post("/api/v1/financial/query/understand", json={"query": "   "})
    oversized = client.post("/api/v1/financial/query/understand", json={"query": "x" * 4_097})

    assert blank.status_code == 422
    assert oversized.status_code == 422


def _answer_test_retriever():
    source = (
        "# Page 9\n"
        "**BÁO CÁO TÌNH HÌNH TÀI CHÍNH HỢP NHẤT NĂM 2025**\n"
        "Đơn vị: Triệu đồng\n"
        "**TỔNG NỢ PHẢI TRẢ**\n"
        "*   Số cuối năm: 750.000\n"
        "*   Số đầu năm: 620.000\n"
    )
    chunks = FinancialMarkdownChunker().chunk_text(
        source,
        source_file="DEMO_SYNTHETIC_2025.md",
        source_pdf="DEMO_SYNTHETIC_2025.pdf",
        company_code="DEMO",
        document_id="demo-2025-test",
    )
    return FinancialHybridRetriever(chunks)


def test_financial_analysis_returns_formula_and_citation_lineage(monkeypatch):
    mapper = api_module.query_engine.metric_mapper
    rows = [
        ("tong_cong_tai_san", "TỔNG CỘNG TÀI SẢN", 1_000, 1_200),
        ("tong_no_phai_tra", "TỔNG NỢ PHẢI TRẢ", 800, 900),
        ("tong_von_chu_so_huu", "TỔNG VỐN CHỦ SỞ HỮU", 200, 300),
    ]
    chunks = []
    for metric_id, label, prior, current in rows:
        metric = next(item for item in mapper.metrics_list if item.metric_id == metric_id)
        chunks.append(FinancialChunk(
            chunk_id=f"mbb-2025-{metric_id}",
            source_file="MBB_BCTC_2025.md",
            document_id="mbb-2025-consolidated",
            company_code="MBB",
            fiscal_year=2025,
            comparative_fiscal_year=2024,
            report_type="consolidated",
            page=11,
            statement_type=metric.statement_type,
            canonical_metric=label,
            chunk_type=ChunkType.FINANCIAL_ITEM,
            unit="Triệu đồng",
            content=(
                f"Chỉ tiêu gốc trong BCTC: {label}\n"
                f"Số cuối năm (2025): {current:,} Triệu đồng\n"
                f"Số đầu năm (2024): {prior:,} Triệu đồng"
            ),
            metadata={
                "metric_name": label,
                "period_values": {"2024": str(prior), "2025": str(current)},
                "start_value": str(prior),
                "end_value": str(current),
            },
        ))
    monkeypatch.setattr(api_module, "CHUNKS_DB", chunks)
    monkeypatch.setattr(api_module, "retriever", FinancialHybridRetriever(chunks))

    def unexpected_model_call(*args, **kwargs):
        raise AssertionError("deterministic financial analysis must not require a model")

    monkeypatch.setattr("financial_rag.generation.chat.requests.post", unexpected_model_call)
    response = client.post(
        "/api/v1/financial/answer",
        json={"query": "Đánh giá sức khỏe tài chính MBB năm 2025"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["validation"]["is_valid"] is True, payload["validation"]
    calculations = payload["calculations"]
    assert calculations
    reconciliation = next(
        item for item in calculations
        if item["formula_id"] == "balance_sheet_reconciliation"
    )
    assert reconciliation["value"] == "0"
    assert reconciliation["expression"] == "total_assets - total_liabilities - total_equity"
    citation_ids = {citation["citation_id"] for citation in payload["citations"]}
    assert set(reconciliation["input_citation_ids"]).issubset(citation_ids)
    assert set(reconciliation["input_chunk_ids"]) == {
        "mbb-2025-tong_cong_tai_san",
        "mbb-2025-tong_no_phai_tra",
        "mbb-2025-tong_von_chu_so_huu",
    }


@pytest.mark.parametrize("company", ["VIC", "FPT"])
def test_health_overview_resolves_numbered_totals_without_using_child_rows(monkeypatch, company):
    mapper = api_module.query_engine.metric_mapper
    metrics = {metric.metric_id: metric for metric in mapper.metrics_list}
    # Mirror legacy persisted OCR rows: outline prefixes, inconsistent accents,
    # and child rows carrying the same advisory canonical mapping as the total.
    rows = [
        ("tong_cong_tai_san", "TỔNG CỘNG TÀI SẢN", "1.200"),
        ("tong_no_phai_tra", "C. NỘ PHẢI TRẢ", "900"),
        ("tong_von_chu_so_huu", "D. VỐN CHỦ SỞ HỮU", "300"),
        ("tai_san_ngan_han", "A. TÀI SẢN NGĂN HẠN", "600"),
        ("no_ngan_han", "I. Nợ ngắn hạn", "500"),
        ("tien_va_tuong_duong_tien", "I. Tiền và các khoản tương đương tiền", "100"),
        ("tien_va_tuong_duong_tien", "Tiền mặt", "1"),
        ("tien_va_tuong_duong_tien", "Tiền gửi ngân hàng", "99"),
        ("tai_san_ngan_han", "V. Tài sản ngắn hạn khác", "40"),
        ("tong_von_chu_so_huu", "6. Quỹ khác thuộc vốn chủ sở hữu", "8"),
        ("tong_no_phai_tra", "TỔNG NỢ PHẢI TRẢ VÀ VỐN CHỦ SỞ HỮU", "1.200"),
    ]
    chunks = []
    for index, (metric_id, label, current) in enumerate(rows):
        metric = metrics[metric_id]
        chunks.append(FinancialChunk(
            chunk_id=f"{company}-health-{index}", document_id=f"{company}-2025",
            source_file=f"{company}_BCTC_2025.txt", company_code=company,
            fiscal_year=2025, comparative_fiscal_year=2024, report_type="consolidated",
            page=9 if index < 6 else 44, unit="Triệu đồng",
            chunk_type=ChunkType.FINANCIAL_ITEM, statement_type=metric.statement_type,
            canonical_metric=metric.canonical_name,
            content=f"Chỉ tiêu gốc trong BCTC: {label} | 2025: {current} Triệu đồng",
            metadata={"metric_name": label, "period_values": {"2025": current, "2024": "50"}},
        ))
    monkeypatch.setattr(api_module, "CHUNKS_DB", chunks)
    monkeypatch.setattr(api_module, "retriever", FinancialHybridRetriever(chunks))
    monkeypatch.setattr(
        "financial_rag.generation.chat.requests.post",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("overview must not call the LLM")),
    )
    response = client.post("/api/v1/financial/answer", json={
        "query": f"Đánh giá sức khỏe tài chính {company} năm 2025 theo báo cáo hợp nhất",
    })
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["validation"]["is_valid"], payload["validation"]
    assert payload["parsed_query"]["fiscal_years"] == [2025]
    answer = payload["answer"]
    assert f"{company} năm 2025" in answer
    assert "2024–2025" not in answer
    assert "Tiền và các khoản tương đương tiền | 100 Triệu đồng" in answer
    assert "TÀI SẢN NGĂN HẠN | 600 Triệu đồng" in answer
    assert "Nợ ngắn hạn | 500 Triệu đồng" in answer
    assert "75,00%" in answer
    assert "25,00%" in answer
    assert "khớp, 0 Triệu đồng" in answer
    assert "Tiền mặt" not in answer
    assert "Tiền gửi ngân hàng" not in answer
    assert {citation["chunk_id"] for citation in payload["citations"]} == {
        chunk.chunk_id for chunk in chunks[:6]
    }


def test_mixed_balance_sheet_analysis_routes_to_overview_and_includes_reconciliation(monkeypatch):
    mapper = api_module.query_engine.metric_mapper
    rows = [
        ("tong_cong_tai_san", "TỔNG CỘNG TÀI SẢN", "900", "1.000"),
        ("tong_no_phai_tra", "TỔNG NỢ PHẢI TRẢ", "720", "800"),
        ("tong_von_chu_so_huu", "TỔNG VỐN CHỦ SỞ HỮU", "180", "200"),
    ]
    chunks = []
    for metric_id, label, prior, current in rows:
        metric = next(item for item in mapper.metrics_list if item.metric_id == metric_id)
        chunks.append(FinancialChunk(
            chunk_id=f"mbb-2024-{metric_id}",
            source_file="MBB_BCTC_2024.md",
            document_id="mbb-2024-consolidated",
            company_code="MBB",
            fiscal_year=2024,
            comparative_fiscal_year=2023,
            report_type="consolidated",
            page=11,
            statement_type=metric.statement_type,
            canonical_metric=metric.canonical_name,
            chunk_type=ChunkType.FINANCIAL_ITEM,
            unit="Triệu đồng",
            content=(
                f"Chỉ tiêu gốc trong BCTC: {label}\n"
                f"Số cuối năm (2024): {current} Triệu đồng\n"
                f"Số đầu năm (2023): {prior} Triệu đồng"
            ),
            metadata={
                "metric_name": label,
                "period_values": {"2023": prior, "2024": current},
                "start_value": prior,
                "end_value": current,
            },
        ))
    monkeypatch.setattr(api_module, "CHUNKS_DB", chunks)
    monkeypatch.setattr(api_module, "retriever", FinancialHybridRetriever(chunks))
    monkeypatch.setattr(
        "financial_rag.generation.chat.requests.post",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("structured balance-sheet analysis should not call the model")
        ),
    )

    response = client.post(
        "/api/v1/financial/answer",
        json={
            "query": (
                "Phân tích cơ cấu tài sản, nợ phải trả và vốn chủ sở hữu của MBB năm 2024; "
                "tính tỷ lệ nợ trên tài sản và đối chiếu phương trình kế toán."
            )
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["validation"]["is_valid"] is True, payload["validation"]
    assert "TỔNG CỘNG TÀI SẢN" in payload["answer"]
    assert "80,00%" in payload["answer"]
    assert "Đối chiếu phương trình bảng cân đối kế toán" in payload["answer"]
    assert any(
        calculation["formula_id"] == "balance_sheet_reconciliation"
        and calculation["value"] == "0"
        for calculation in payload["calculations"]
    )


def test_answer_endpoint_computes_profit_margin_from_same_statement_and_exposes_formula(monkeypatch):
    mapper = api_module.query_engine.metric_mapper
    rows = [
        ("loi_nhuan_gop", "LỢI NHUẬN GỘP", "40"),
        ("doanh_thu_thuan", "DOANH THU THUẦN", "160"),
    ]
    chunks = []
    for metric_id, label, value in rows:
        metric = next(item for item in mapper.metrics_list if item.metric_id == metric_id)
        chunks.append(FinancialChunk(
            chunk_id=f"fpt-2025-{metric_id}",
            source_file="FPT_BCTC_2025.md",
            document_id="fpt-2025-consolidated",
            company_code="FPT",
            fiscal_year=2025,
            comparative_fiscal_year=2024,
            report_type="consolidated",
            page=55,
            statement_type=metric.statement_type,
            canonical_metric=metric.canonical_name,
            chunk_type=ChunkType.FINANCIAL_ITEM,
            unit="Tỷ đồng",
            content=(
                f"Chỉ tiêu gốc trong BCTC: {label}\n"
                f"Số cuối năm (2025): {value} Tỷ đồng"
            ),
            metadata={
                "metric_name": label,
                "period_values": {"2025": value},
            },
        ))
    monkeypatch.setattr(api_module, "CHUNKS_DB", chunks)
    monkeypatch.setattr(api_module, "retriever", FinancialHybridRetriever(chunks))
    monkeypatch.setattr(
        "financial_rag.generation.chat.requests.post",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("a registered financial ratio must not require model generation")
        ),
    )

    response = client.post(
        "/api/v1/financial/answer",
        json={"query": "Biên lợi nhuận gộp của FPT năm 2025"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["validation"]["is_valid"] is True, payload["validation"]
    assert "25,00%" in payload["answer"]
    assert payload["validation"]["is_valid"] is True, payload["validation"]
    assert payload["calculations"][0]["formula_id"] == "gross_margin_percent"
    assert payload["calculations"][0]["expression"] == "gross_profit / net_revenue * 100"
    assert set(payload["calculations"][0]["input_citation_ids"]) == {
        citation["citation_id"] for citation in payload["citations"]
    }


def test_ratio_endpoint_retrieves_each_balance_sheet_input_independently(monkeypatch):
    mapper = api_module.query_engine.metric_mapper
    rows = [
        ("tong_no_phai_tra", "TỔNG NỢ PHẢI TRẢ", "800"),
        ("tong_cong_tai_san", "TỔNG CỘNG TÀI SẢN", "1.000"),
    ]
    chunks = []
    for metric_id, label, value in rows:
        metric = next(item for item in mapper.metrics_list if item.metric_id == metric_id)
        chunks.append(FinancialChunk(
            chunk_id=f"mbb-2024-{metric_id}",
            source_file="MBB_BCTC_2024.md",
            document_id="mbb-2024-consolidated",
            company_code="MBB",
            fiscal_year=2024,
            report_type="consolidated",
            page=11,
            statement_type=metric.statement_type,
            canonical_metric=metric.canonical_name,
            chunk_type=ChunkType.FINANCIAL_ITEM,
            unit="Triệu đồng",
            content=(
                f"Chỉ tiêu gốc trong BCTC: {label}\n"
                f"Số cuối năm (2024): {value} Triệu đồng"
            ),
            metadata={
                "metric_name": label,
                "period_values": {"2024": value},
            },
        ))
    monkeypatch.setattr(api_module, "CHUNKS_DB", chunks)
    monkeypatch.setattr(api_module, "retriever", FinancialHybridRetriever(chunks))
    monkeypatch.setattr(
        "financial_rag.generation.chat.requests.post",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("registered ratio calculation should not call the model")
        ),
    )

    response = client.post(
        "/api/v1/financial/answer",
        json={"query": "Tỷ lệ nợ phải trả trên tổng tài sản MBB năm 2024 là bao nhiêu?"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert "80,00%" in payload["answer"]
    assert payload["validation"]["is_valid"] is True, payload["validation"]
    assert len(payload["calculations"]) == 1
    assert payload["calculations"][0]["formula_id"] == "liabilities_to_assets_percent"
    assert set(payload["calculations"][0]["input_chunk_ids"]) == {
        "mbb-2024-tong_no_phai_tra",
        "mbb-2024-tong_cong_tai_san",
    }
    ratio_citation_ids = {
        citation["citation_id"] for citation in payload["citations"]
    }
    assert set(payload["calculations"][0]["input_citation_ids"]) == ratio_citation_ids
    assert all(f"[{citation_id}]" in payload["answer"] for citation_id in ratio_citation_ids)

    retrieval_response = client.post(
        "/api/v1/financial/retrieve",
        json={"query": "Tỷ lệ nợ phải trả trên tổng tài sản MBB năm 2024 là bao nhiêu?", "top_k": 5},
    )
    assert retrieval_response.status_code == 200, retrieval_response.text
    assert {
        item["chunk"]["chunk_id"] for item in retrieval_response.json()
    } == {
        "mbb-2024-tong_no_phai_tra",
        "mbb-2024-tong_cong_tai_san",
    }


def test_composition_endpoint_returns_verified_share_calculations(monkeypatch):
    rows = [
        ("mbb-total-liabilities", "TỔNG NỢ PHẢI TRẢ", "1.000"),
        ("mbb-short-liabilities", "NỢ NGẮN HẠN", "600"),
    ]
    chunks = [
        FinancialChunk(
            chunk_id=chunk_id,
            source_file="MBB_BCTC_2024.md",
            document_id="mbb-2024-consolidated",
            company_code="MBB",
            fiscal_year=2024,
            report_type="consolidated",
            page=11,
            statement_type=StatementType.BALANCE_SHEET,
            chunk_type=ChunkType.FINANCIAL_ITEM,
            unit="Triệu đồng",
            content=f"Chỉ tiêu gốc trong BCTC: {label}\nNăm 2024: {value} Triệu đồng",
            metadata={
                "metric_name": label,
                "period_values": {"2024": value},
                **(
                    {
                        "schema_node_id": "liabilities.total",
                        "schema_node_kind": "total",
                        "schema_children": ["liabilities.current"],
                    }
                    if chunk_id == "mbb-total-liabilities"
                    else {
                        "schema_node_id": "liabilities.current",
                        "schema_parent_id": "liabilities.total",
                    }
                ),
            },
        )
        for chunk_id, label, value in rows
    ]
    monkeypatch.setattr(api_module, "CHUNKS_DB", chunks)
    monkeypatch.setattr(api_module, "retriever", FinancialHybridRetriever(chunks))
    monkeypatch.setattr(
        "financial_rag.generation.chat.requests.post",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("source-grounded composition should not require model generation")
        ),
    )

    response = client.post(
        "/api/v1/financial/answer",
        json={"query": "Cơ cấu nợ phải trả MBB năm 2024"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert "60,00%" in payload["answer"]
    assert payload["validation"]["is_valid"] is True, payload["validation"]
    component_share = next(
        item for item in payload["calculations"]
        if item["formula_id"] == "share_of_total"
        and item["input_chunk_ids"] == ["mbb-short-liabilities", "mbb-total-liabilities"]
    )
    assert component_share["value"] == "60.00"
    assert component_share["expression"] == "component_value / total_value * 100"


def test_composition_matches_canonical_alias_and_expands_the_selected_root_page():
    chunks = _synthetic_mbb_balance_sheet_chunks((2024,))
    anchor = next(
        chunk for chunk in chunks
        if chunk.metadata.get("schema_node_id") == "TOTAL_ASSETS"
    )
    for chunk in chunks:
        for key in (
            "schema_node_id",
            "schema_parent_id",
            "schema_node_kind",
            "schema_children",
            "schema_component_variants",
        ):
            chunk.metadata.pop(key, None)

    def key_for(chunk):
        return (
            (chunk.company_code or chunk.source_file).casefold(),
            chunk.document_id or chunk.source_file,
            chunk.report_type or "",
            str(chunk.fiscal_year or ""),
            chunk.statement_type.value,
            api_module._normalized_table_title(chunk)
            or f"page:{chunk.page or 0}",
        )

    selected = api_module._composition_source_rows(
        "Cơ cấu tổng cộng tài sản của MBB năm 2024",
        ["MBB"],
        {key_for(anchor)},
        chunks,
        key_for,
    )

    assert anchor in selected
    assert len(selected) > 30
    assert any(
        chunk.metadata.get("metric_name") == "Tiền mặt, vàng bạc, đá quý"
        for chunk in selected
    )
    assert any(
        chunk.metadata.get("metric_name") == "TỔNG TÀI SẢN"
        for chunk in selected
    )


def test_synthetic_asset_composition_reconciles_to_total(monkeypatch):
    chunks = _synthetic_mbb_balance_sheet_chunks((2024,))
    monkeypatch.setattr(api_module, "CHUNKS_DB", chunks)
    monkeypatch.setattr(api_module, "retriever", FinancialHybridRetriever(chunks))
    monkeypatch.setattr(
        "financial_rag.generation.chat.requests.post",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("source-grounded composition should not require model generation")
        ),
    )

    response = client.post(
        "/api/v1/financial/answer",
        json={"query": "Cơ cấu tài sản MBB năm 2024"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    answer = payload["answer"]
    assert payload["validation"]["is_valid"] is True, payload["validation"]
    assert "Cơ cấu Tài sản của MBB năm 2024" in answer
    assert "Đối chiếu: năm 2024, tổng các dòng thành phần khớp với tổng cơ sở." in answer
    assert "| Chứng khoán kinh doanh | 200.000 Triệu đồng |" in answer
    assert "| Cho vay khách hàng | 500.000 Triệu đồng |" in answer
    assert "Tiền mặt, vàng bạc, đá quý" in answer
    assert "Lợi thế thương mại" not in answer

    from decimal import Decimal
    import re

    table_rows = [
        line for line in answer.splitlines()
        if line.startswith("|")
        and not line.startswith("| ---")
        and not line.startswith("| Chỉ tiêu gốc")
    ]
    component_amounts = []
    for line in table_rows:
        amount_text = line.strip("|").split("|")[1].strip().split()[0]
        negative = amount_text.startswith("(") and amount_text.endswith(")")
        digits = re.sub(r"[^0-9]", "", amount_text)
        component_amounts.append(Decimal(digits) * (-1 if negative else 1))
    assert sum(component_amounts) == Decimal("1500000")


def test_combined_balance_sheet_analysis_includes_all_requested_compositions(monkeypatch):
    chunks = _synthetic_mbb_balance_sheet_chunks((2024,))
    monkeypatch.setattr(api_module, "CHUNKS_DB", chunks)
    monkeypatch.setattr(api_module, "retriever", FinancialHybridRetriever(chunks))
    monkeypatch.setattr(
        "financial_rag.generation.chat.requests.post",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("deterministic overview analysis should not call a model")
        ),
    )

    response = client.post(
        "/api/v1/financial/answer",
        json={
            "query": (
                "Phân tích cơ cấu tài sản, nợ phải trả và vốn chủ sở hữu của MBB năm 2024; "
                "tính tỷ lệ nợ trên tài sản và đối chiếu phương trình kế toán."
            )
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["validation"]["is_valid"] is True, payload["validation"]
    answer = payload["answer"]
    assert "Cơ cấu TỔNG CỘNG TÀI SẢN" in answer
    assert "Cơ cấu TỔNG NỢ PHẢI TRẢ" in answer
    assert "Cơ cấu TỔNG VỐN CHỦ SỞ HỮU" in answer
    assert answer.count("tổng các dòng thành phần khớp với tổng cơ sở") == 3
    assert "66,67%" in answer
    assert "Đối chiếu phương trình bảng cân đối kế toán" in answer


def test_answer_endpoint_answers_structured_financial_item_without_model(monkeypatch):
    retriever = _answer_test_retriever()
    monkeypatch.setattr(api_module, "retriever", retriever)

    def unexpected_model_call(*args, **kwargs):
        raise AssertionError("structured financial facts should not require model generation")

    monkeypatch.setattr("financial_rag.generation.chat.requests.post", unexpected_model_call)
    response = client.post(
        "/api/v1/financial/answer",
        json={"query": "Tổng nợ phải trả DEMO đầu năm 2025 là bao nhiêu?"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "620.000" in data["answer"]
    assert "[c1]" in data["answer"]
    assert data["validation"]["is_valid"] is True, data["validation"]
    assert data["validation"]["citation_valid"] is True
    assert data["citations"][0]["page"] == 9
    assert data["citations"][0]["file_name"] == "DEMO_SYNTHETIC_2025.pdf"


def test_exact_row_selection_prefers_requested_child_over_higher_ranked_parent():
    from financial_rag.api.app import _select_preferred_exact_source_rows

    parent = FinancialChunk(
        chunk_id="demo-investment-parent",
        source_file="DEMO_synthetic.txt",
        document_id="demo-investment",
        company_code="DEMO",
        fiscal_year=2025,
        report_type="consolidated",
        page=7,
        statement_type=StatementType.BALANCE_SHEET,
        chunk_type=ChunkType.FINANCIAL_ITEM,
        unit="Triệu đồng",
        content="Chứng khoán đầu tư: 144.164 triệu đồng",
        metadata={
            "metric_name": "Chứng khoán đầu tư",
            "schema_node_id": "ASSET.FIN_INV",
            "period_values": {"2025": "144.164"},
        },
    )
    child = FinancialChunk(
        chunk_id="demo-investment-child",
        source_file="DEMO_synthetic.txt",
        document_id="demo-investment",
        company_code="DEMO",
        fiscal_year=2025,
        report_type="consolidated",
        page=7,
        statement_type=StatementType.BALANCE_SHEET,
        chunk_type=ChunkType.FINANCIAL_ITEM,
        unit="Triệu đồng",
        content="Chứng khoán đầu tư sẵn sàng để bán: 132.778 triệu đồng",
        metadata={
            "metric_name": "Chứng khoán đầu tư sẵn sàng để bán",
            "schema_node_id": "ASSET.FIN_INV.AFS",
            "period_values": {"2025": "132.778"},
        },
    )
    parsed = api_module.query_engine.parse(
        "Chứng khoán đầu tư sẵn sàng để bán của DEMO năm 2025"
    )
    results = [RetrievalResult(chunk=parent, rrf_score=0.95)]

    selected = _select_preferred_exact_source_rows(
        parsed.original_query,
        parsed,
        results,
        candidate_chunks=[
            parent,
            child,
            child.model_copy(update={"chunk_id": "other-company-child", "company_code": "OTHER"}),
            child.model_copy(update={"chunk_id": "prior-year-child", "fiscal_year": 2024}),
            child.model_copy(update={"chunk_id": "separate-basis-child", "report_type": "separate"}),
        ],
    )

    assert [result.chunk.chunk_id for result in selected] == [child.chunk_id]

    # The same exact source-label recovery still works for legacy chunks that
    # predate schema annotations, using label specificity as the fallback.
    parent.metadata.pop("schema_node_id")
    child.metadata.pop("schema_node_id")
    legacy_selected = _select_preferred_exact_source_rows(
        parsed.original_query,
        parsed,
        results,
        candidate_chunks=[parent, child],
    )
    assert [result.chunk.chunk_id for result in legacy_selected] == [child.chunk_id]


def test_answer_endpoint_falls_back_to_retrieval_when_schema_has_no_mapping(monkeypatch):
    retriever = _answer_test_retriever()
    monkeypatch.setattr(api_module, "retriever", retriever)

    def fallback_answer(_query, _parsed, results, _chunks_by_id):
        chunk = results[0].chunk
        citation = api_module.Citation(
            citation_id="c1",
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            file_name=chunk.source_pdf or chunk.source_file,
            page=chunk.page,
            statement_type=chunk.statement_type.value,
            section=chunk.section,
            unit=chunk.unit,
            snippet=chunk.content,
        )
        return (
            "Tổng nợ phải trả DEMO năm 2025 là 750.000 Triệu đồng [c1].",
            [citation],
            [chunk.content],
        )

    monkeypatch.setattr(api_module.chat_engine, "generate_answer", fallback_answer)
    response = client.post(
        "/api/v1/financial/answer",
        json={"query": "Cơ cấu nợ phải trả DEMO năm 2025"},
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert "750.000" in data["answer"]
    assert data["validation"]["is_valid"] is True


def test_answer_endpoint_compares_year_end_values_across_companies(monkeypatch):
    chunks = [
        FinancialChunk(
            chunk_id="abb-2024-assets",
            source_file="ABB_BCTC_2024.md",
            document_id="abb-2024",
            company_code="ABB",
            fiscal_year=2024,
            comparative_fiscal_year=2023,
            report_type="consolidated",
            page=7,
            statement_type=StatementType.BALANCE_SHEET,
            canonical_metric="TỔNG CỘNG TÀI SẢN",
            chunk_type=ChunkType.FINANCIAL_ITEM,
            unit="Triệu VND",
            content="Số cuối năm (2024): 2.400 Triệu VND\nSố đầu năm (2023): 2.000 Triệu VND",
            metadata={
                "canonical_metric": "TỔNG CỘNG TÀI SẢN",
                "original_label": "TỔNG CỘNG TÀI SẢN",
                "period_values": {"2024": "2.400", "2023": "2.000"},
                "end_value": "2.400",
                "start_value": "2.000",
            },
        ),
        FinancialChunk(
            chunk_id="acb-2025-assets",
            source_file="ACB_BCTC_2025.md",
            document_id="acb-2025",
            company_code="ACB",
            fiscal_year=2025,
            comparative_fiscal_year=2024,
            report_type="consolidated",
            page=7,
            statement_type=StatementType.BALANCE_SHEET,
            canonical_metric="TỔNG CỘNG TÀI SẢN",
            chunk_type=ChunkType.FINANCIAL_ITEM,
            unit="Triệu đồng",
            content="Số cuối năm (2025): 5.000 Triệu đồng\nSố đầu năm (2024): 4.000 Triệu đồng",
            metadata={
                "canonical_metric": "TỔNG CỘNG TÀI SẢN",
                "original_label": "TỔNG CỘNG TÀI SẢN",
                "period_values": {"2025": "5.000", "2024": "4.000"},
                "end_value": "5.000",
                "start_value": "4.000",
            },
        ),
    ]
    monkeypatch.setattr(api_module, "retriever", FinancialHybridRetriever(chunks))

    def unexpected_model_call(*args, **kwargs):
        raise AssertionError("period comparisons should not require model generation")

    monkeypatch.setattr("financial_rag.generation.chat.requests.post", unexpected_model_call)

    response = client.post(
        "/api/v1/financial/answer",
        json={
            "query": "So sánh tổng tài sản cuối năm 2024 của ABB và ACB theo báo cáo hợp nhất.",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "| ABB |" in data["answer"] and "2.400 Triệu VND [c1]" in data["answer"]
    assert "| ACB |" in data["answer"] and "4.000 Triệu đồng [c2]" in data["answer"]
    assert data["validation"]["is_valid"] is True, data["validation"]
    assert data["validation"]["citation_valid"] is True
    assert [citation["page"] for citation in data["citations"]] == [7, 7]


def test_answer_endpoint_computes_growth_from_labeled_period_values(monkeypatch):
    chunk = FinancialChunk(
        chunk_id="demo-2025-net-profit",
        source_file="DEMO_SYNTHETIC_2025.md",
        document_id="demo-2025",
        company_code="DEMO",
        fiscal_year=2025,
        comparative_fiscal_year=2024,
        report_type="consolidated",
        page=11,
        statement_type=StatementType.INCOME_STATEMENT,
        canonical_metric="LỢI NHUẬN SAU THUẾ",
        chunk_type=ChunkType.FINANCIAL_ITEM,
        unit="Triệu đồng",
        content=(
            "LỢI NHUẬN SAU THUẾ\n"
            "Số cuối năm (2025): 96 Triệu đồng\n"
            "Số đầu năm (2024): 64 Triệu đồng"
        ),
        metadata={
            "canonical_metric": "LỢI NHUẬN SAU THUẾ",
            "metric_name": "LỢI NHUẬN SAU THUẾ",
            "period_values": {"2025": "96", "2024": "64"},
        },
    )
    monkeypatch.setattr(api_module, "retriever", FinancialHybridRetriever([chunk]))

    def unexpected_model_call(*args, **kwargs):
        raise AssertionError("period arithmetic should not require model generation")

    monkeypatch.setattr("financial_rag.generation.chat.requests.post", unexpected_model_call)

    response = client.post(
        "/api/v1/financial/answer",
        json={"query": "Lợi nhuận sau thuế DEMO năm 2025 tăng bao nhiêu so với năm 2024?"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "32 Triệu đồng" in data["answer"]
    assert any(
        item["formula_id"] == "absolute_change" and item["value"] == "32"
        for item in data["calculations"]
    )
    assert any(
        item["formula_id"] == "growth_rate" and item["value"] == "50.00"
        for item in data["calculations"]
    )
    assert data["validation"]["is_valid"] is True, data["validation"]
    assert data["validation"]["citation_valid"] is True


def test_answer_endpoint_does_not_reuse_growth_from_the_wrong_years(monkeypatch):
    chunk = FinancialChunk(
        chunk_id="demo-2024-net-profit",
        source_file="DEMO_SYNTHETIC_2024.md",
        document_id="demo-2024",
        company_code="DEMO",
        fiscal_year=2024,
        comparative_fiscal_year=2023,
        report_type="consolidated",
        page=10,
        statement_type=StatementType.INCOME_STATEMENT,
        canonical_metric="LỢI NHUẬN SAU THUẾ",
        chunk_type=ChunkType.FINANCIAL_ITEM,
        unit="Triệu VND",
        content=(
            "Năm nay (2024): 700 Triệu VND\n"
            "Năm trước (2023): 400 Triệu VND\n"
            "So sánh biến động: Tăng 300 Triệu VND"
        ),
        metadata={
            "canonical_metric": "LỢI NHUẬN SAU THUẾ",
            "metric_name": "LỢI NHUẬN SAU THUẾ",
            "period_values": {"2024": "700", "2023": "400"},
            "difference_formatted": "300",
            "trend": "Tăng",
        },
    )
    monkeypatch.setattr(api_module, "retriever", FinancialHybridRetriever([chunk]))

    def unexpected_model_call(*args, **kwargs):
        raise AssertionError("unavailable periods should be reported without model generation")

    monkeypatch.setattr("financial_rag.generation.chat.requests.post", unexpected_model_call)

    response = client.post(
        "/api/v1/financial/answer",
        json={"query": "Lợi nhuận sau thuế DEMO năm 2025 tăng bao nhiêu so với năm 2024?"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "300" not in data["answer"]
    assert data["validation"]["is_valid"] is True, data["validation"]
    assert "Chưa tìm thấy giá trị biến động" in data["answer"]


def test_answer_endpoint_respects_row_unit_for_per_share_values(monkeypatch):
    chunk = FinancialChunk(
        chunk_id="abb-2025-eps-table",
        source_file="DEMO_SYNTHETIC_2025.md",
        document_id="abb-2025",
        company_code="ABB",
        fiscal_year=2025,
        comparative_fiscal_year=2024,
        report_type="consolidated",
        page=11,
        statement_type=StatementType.INCOME_STATEMENT,
        chunk_type=ChunkType.TABLE,
        unit="Triệu đồng",
        content=(
            "(Đơn vị: Triệu đồng)\n"
            "**Thuyết minh** | **Năm nay (Triệu đồng)** | **Năm trước (Triệu đồng)**\n"
            "---|---|---\n"
            "**Lãi cơ bản trên cổ phiếu (đồng/cổ phiếu)** | 2.714 | 535"
        ),
    )
    monkeypatch.setattr(api_module, "retriever", FinancialHybridRetriever([chunk]))

    def unexpected_model_call(*args, **kwargs):
        raise AssertionError("an exact table row should not require model generation")

    monkeypatch.setattr("financial_rag.generation.chat.requests.post", unexpected_model_call)

    response = client.post(
        "/api/v1/financial/answer",
        json={"query": "Lãi cơ bản trên cổ phiếu ABB năm 2025 là bao nhiêu?"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "2.714 VNĐ/cổ phiếu" in data["answer"]
    assert data["validation"]["is_valid"] is True, data["validation"]


def test_answer_endpoint_keeps_a_report_named_metric_as_a_metric_lookup(monkeypatch):
    sample_file = Path(__file__).resolve().parents[1] / "sample_data" / "DEMO_SYNTHETIC_2025.md"
    chunks = FinancialMarkdownChunker().chunk_file(sample_file, company_code="DEMO")
    monkeypatch.setattr(api_module, "CHUNKS_DB", chunks)
    monkeypatch.setattr(api_module, "retriever", FinancialHybridRetriever(chunks))

    def unexpected_model_call(*args, **kwargs):
        raise AssertionError("a labeled cash-flow row should not require model generation")

    monkeypatch.setattr("financial_rag.generation.chat.requests.post", unexpected_model_call)
    response = client.post(
        "/api/v1/financial/answer",
        json={
            "query": (
                "Lưu chuyển tiền thuần từ hoạt động kinh doanh DEMO năm 2024 "
                "trên báo cáo lưu chuyển tiền tệ hợp nhất là bao nhiêu?"
            )
        },
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["parsed_query"]["structured_operation"] is None
    assert "110 Triệu đồng" in data["answer"]
    assert data["citations"][0]["page"] == 12
    assert data["validation"]["is_valid"] is True, data["validation"]


def _company_scope_asset_chunks():
    facts = [
        ("MBB", 2024, "100000", "MBB_BCTC_2024.md"),
        ("MBB", 2025, "120000", "MBB_BCTC_2025.md"),
        ("VPB", 2025, "900000", "VPB_BCTC_2025.md"),
    ]
    chunks = [
        FinancialChunk(
            chunk_id=f"{company}-{year}-assets",
            source_file=source_file,
            document_id=f"{company}-{year}",
            company_code=company,
            fiscal_year=year,
            report_type="consolidated",
            page=11,
            statement_type=StatementType.BALANCE_SHEET,
            canonical_metric="TỔNG CỘNG TÀI SẢN",
            chunk_type=ChunkType.FINANCIAL_ITEM,
            unit="tỷ đồng",
            content=(
                "Chỉ tiêu gốc trong BCTC: TỔNG CỘNG TÀI SẢN\n"
                f"Năm {year}: {value} tỷ đồng"
            ),
            metadata={
                "metric_name": "TỔNG CỘNG TÀI SẢN",
                "period_values": {str(year): value},
            },
        )
        for company, year, value, source_file in facts
    ]
    return chunks


def test_answer_endpoint_does_not_mix_company_or_year_for_total_assets(monkeypatch):
    chunks = _company_scope_asset_chunks()
    monkeypatch.setattr(api_module, "CHUNKS_DB", chunks)
    monkeypatch.setattr(api_module, "retriever", FinancialHybridRetriever(chunks))

    def unexpected_model_call(*args, **kwargs):
        raise AssertionError("a scoped exact-row lookup must not need model generation")

    monkeypatch.setattr("financial_rag.generation.chat.requests.post", unexpected_model_call)
    response = client.post(
        "/api/v1/financial/answer",
        json={"query": "Tổng tài sản MBB hợp nhất năm 2025 là bao nhiêu?"},
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert "120000 tỷ đồng" in data["answer"]
    assert "100000" not in data["answer"]
    assert "900000" not in data["answer"]
    assert [citation["file_name"] for citation in data["citations"]] == ["MBB_BCTC_2025.md"]
    assert data["validation"]["is_valid"] is True, data["validation"]


def test_exact_metric_answer_selects_matching_fact_from_mixed_company_context(monkeypatch):
    chunks = _company_scope_asset_chunks()
    query = "Tổng tài sản MBB hợp nhất năm 2025 là bao nhiêu?"
    parsed = api_module.query_engine.parse(query)
    results = [RetrievalResult(chunk=chunk, rrf_score=1.0) for chunk in chunks]

    answer, citations, source_contexts = api_module.chat_engine.generate_answer(
        query, parsed, results, {chunk.chunk_id: chunk for chunk in chunks}
    )

    assert "MBB" in answer
    assert "2025" in answer
    assert "120000 tỷ đồng" in answer
    assert "900000" not in answer
    assert [citation.file_name for citation in citations] == ["MBB_BCTC_2025.md"]
    assert len(source_contexts) == 1
    assert "120000" in source_contexts[0]
    assert "900000" not in source_contexts[0]


def test_context_assembler_marks_conflicting_source_scopes():
    chunks = [
        FinancialChunk(
            chunk_id=f"{company}-{year}",
            source_file=f"{company}_{year}.md",
            company_code=company,
            fiscal_year=year,
            report_type="consolidated",
            statement_type=StatementType.BALANCE_SHEET,
            content=f"Tổng tài sản năm {year}: {value}",
        )
        for company, year, value in (
            ("MBB", 2024, "100000"),
            ("MBB", 2025, "120000"),
            ("VPB", 2025, "900000"),
        )
    ]
    results = [RetrievalResult(chunk=chunk, rrf_score=1.0) for chunk in chunks]

    context, _citations, _source_contexts = ContextAssembler().assemble_with_sources(results)

    assert "source_scope: company=MBB; report_basis=consolidated; filing_year=2024" in context
    assert "source_scope: company=MBB; report_basis=consolidated; filing_year=2025" in context
    assert "source_scope: company=VPB; report_basis=consolidated; filing_year=2025" in context


def test_answer_endpoint_flags_unsupported_numbers(monkeypatch):
    retriever = _answer_test_retriever()
    monkeypatch.setattr(api_module, "retriever", retriever)

    def unsupported_answer(_query, _parsed, results, chunks_by_id):
        _context, citations, source_contexts = ContextAssembler().assemble_with_sources(results, chunks_by_id)
        return "Nợ phải trả đầu năm là 999.999.999 triệu đồng [c1].", citations, source_contexts

    monkeypatch.setattr(api_module.chat_engine, "generate_answer", unsupported_answer)
    response = client.post(
        "/api/v1/financial/answer",
        json={"query": "Tổng nợ phải trả DEMO đầu năm 2025 là bao nhiêu?"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["validation"]["is_valid"] is False
    assert data["validation"]["citation_valid"] is True
    assert "999.999.999" not in data["answer"]
    assert data["answer"].startswith("Không thể xác minh câu trả lời")


def test_answer_endpoint_returns_gateway_error_when_model_is_unavailable(monkeypatch):
    monkeypatch.setattr(api_module, "retriever", _answer_test_retriever())

    def unavailable(*args, **kwargs):
        raise FinancialGenerationError("Ollama unavailable")

    monkeypatch.setattr(api_module.chat_engine, "generate_answer", unavailable)
    response = client.post(
        "/api/v1/financial/answer",
        json={"query": "Tổng nợ phải trả DEMO đầu năm 2025 là bao nhiêu?"},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == "Financial answer model is unavailable"


def test_chat_fallback_uses_bounded_nonthinking_chat_api(monkeypatch):
    retriever = _answer_test_retriever()
    parsed = api_module.query_engine.parse("Giải thích cách trình bày nợ phải trả DEMO trong báo cáo 2025.")
    parsed.intent = QueryIntent.EXPLANATION
    parsed.canonical_metric = None
    results = retriever.search(parsed, top_k=5)
    assert results

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"message": {"role": "assistant", "content": "Bảng trình bày các khoản nợ phải trả [c1]."}}

        @staticmethod
        def raise_for_status():
            return None

    def fake_post(url, *, json, timeout):
        assert url.endswith("/api/chat")
        assert json["model"] == api_module.chat_engine.model_name
        assert json["stream"] is False
        assert json["think"] is False
        assert json["options"]["num_predict"] <= 256
        assert json["options"]["num_ctx"] >= 1024
        assert timeout > 0
        prompt = json["messages"][0]["content"]
        assert "620.000" in prompt
        assert "[QUY TẮC KHỚP PHẠM VI NGUỒN]" in prompt
        assert "company=DEMO; fiscal_year=2025" in prompt
        assert "[c1] [DEMO -" in prompt
        return FakeResponse()

    monkeypatch.setattr("financial_rag.generation.chat.requests.post", fake_post)
    answer, citations, _source_contexts = api_module.chat_engine.generate_answer(
        parsed.original_query, parsed, results, retriever.chunks_by_id
    )

    assert answer == "Bảng trình bày các khoản nợ phải trả [c1]."
    assert citations


def test_answer_endpoint_does_not_generate_without_matching_company_evidence(monkeypatch):
    monkeypatch.setattr(api_module, "retriever", _answer_test_retriever())

    def unexpected_generation(*args, **kwargs):
        raise AssertionError("answer generation must not run without matching evidence")

    monkeypatch.setattr(api_module.chat_engine, "generate_answer", unexpected_generation)
    unanswered_queries = (
        "Doanh thu thuần FPT năm 2025 là bao nhiêu?",
        "Tổng nợ phải trả DEMO cuối năm 2030 là bao nhiêu?",
        "Doanh thu thuần DEMO năm 2025 là bao nhiêu?",
    )
    for query in unanswered_queries:
        response = client.post("/api/v1/financial/answer", json={"query": query})
        assert response.status_code == 404, query
        assert response.json()["detail"] == "Không tìm thấy thông tin phù hợp trong tài liệu đã nạp."

def test_query_understand_endpoint():
    payload = {"query": "Tổng nợ phải trả của ABB đầu năm 2025 là bao nhiêu?"}
    response = client.post("/api/v1/financial/query/understand", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["company_code"] == "ABB"
    assert data["primary_metric"] == "TỔNG NỢ PHẢI TRẢ"


def test_query_understand_endpoint_returns_both_comparison_years():
    response = client.post(
        "/api/v1/financial/query/understand",
        json={"query": "Doanh thu thuần FPT năm 2025 tăng bao nhiêu so với năm 2024?"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["fiscal_years"] == [2025, 2024]
    assert data["metadata_filter"]["fiscal_years"] == [2025, 2024]
    assert "doanh thu thuần" in [alias.casefold() for alias in data["metric_aliases"]]


def test_retrieve_endpoint():
    # Load the sample explicitly so service startup stays empty by default.
    sample_file = Path(__file__).resolve().parents[1] / "sample_data" / "DEMO_SYNTHETIC_2025.md"
    ingested = client.post(
        "/api/v1/financial/ingest",
        json={
            "source_file": sample_file.name,
            "company_code": "DEMO",
            "document_id": "test-synthetic-demo-2025",
            "content": sample_file.read_text(encoding="utf-8"),
        },
    )
    assert ingested.status_code == 200
    payload = {"query": "Tổng nợ phải trả của DEMO đầu năm 2025 là bao nhiêu?", "top_k": 2}
    response = client.post("/api/v1/financial/retrieve", json=payload)
    assert response.status_code == 200
    results = response.json()
    assert len(results) > 0
    # Top 1 phải chứa TỔNG NỢ PHẢI TRẢ
    top_chunk = results[0]["chunk"]
    assert "TỔNG NỢ PHẢI TRẢ" in top_chunk["content"].upper()


def test_ingest_replaces_existing_document_without_duplicate_chunks(monkeypatch):
    monkeypatch.setattr(api_module, "CHUNKS_DB", [])
    monkeypatch.setattr(api_module, "retriever", None)
    content = (
        "# Page 9\n"
        "**BÁO CÁO TÌNH HÌNH TÀI CHÍNH HỢP NHẤT NĂM 2025**\n"
        "Đơn vị: VND\n"
        "**TỔNG TÀI SẢN**\n"
        "*   Số cuối năm: 250\n"
        "*   Số đầu năm: 200\n"
    )
    payload = {
        "source_file": "FPT_BCTC_2025.md",
        "source_pdf": "FPT_BCTC_2025.pdf",
        "company_code": "FPT",
        "document_id": "fpt-2025-idempotent-test",
        "content": content,
    }

    first = client.post("/api/v1/financial/ingest", json=payload)
    assert first.status_code == 200
    first_data = first.json()
    assert first_data["new_chunks_count"] > 0
    assert first_data["total_chunks_indexed"] == first_data["new_chunks_count"]

    payload["content"] = content.replace("250", "300")
    second = client.post("/api/v1/financial/ingest", json=payload)
    assert second.status_code == 200
    second_data = second.json()
    assert second_data["new_chunks_count"] == first_data["new_chunks_count"]
    assert second_data["total_chunks_indexed"] == first_data["total_chunks_indexed"]

    retrieved = client.post(
        "/api/v1/financial/retrieve",
        json={"query": "Tổng tài sản FPT cuối năm 2025 là bao nhiêu?", "top_k": 5},
    )
    assert retrieved.status_code == 200
    retrieved_content = [item["chunk"]["content"] for item in retrieved.json()]
    assert any("300" in chunk for chunk in retrieved_content)
    assert not any("250" in chunk for chunk in retrieved_content)
    assert len(api_module.CHUNKS_DB) == first_data["new_chunks_count"]

    answered = client.post(
        "/api/v1/financial/answer",
        json={"query": "Tổng tài sản FPT cuối năm 2025 là bao nhiêu?"},
    )
    assert answered.status_code == 200
    assert "300" in answered.json()["answer"]
    assert answered.json()["validation"]["is_valid"] is True


def test_standalone_sqlite_batch_ingest_survives_restore_and_supports_delete(
    tmp_path, monkeypatch
):
    store = SQLiteFinancialStore(tmp_path / "financial-rag.sqlite3")
    monkeypatch.setattr(api_module, "document_store", store)
    monkeypatch.setattr(api_module, "CHUNKS_DB", [])
    monkeypatch.setattr(api_module, "retriever", None)
    documents = [
        {
            "source_file": "ABB_BCTC_2024.md",
            "company_code": "ABB",
            "content": (
                "# Page 7\n"
                "BÁO CÁO TÌNH HÌNH TÀI CHÍNH HỢP NHẤT NĂM 2024\n"
                "Đơn vị: Triệu đồng\n"
                "**TỔNG CỘNG TÀI SẢN**\n"
                "* Số cuối năm: 100\n"
                "* Số đầu năm: 80\n"
            ),
        },
        {
            "source_file": "ACB_BCTC_2024.md",
            "company_code": "ACB",
            "content": (
                "# Page 7\n"
                "BÁO CÁO TÌNH HÌNH TÀI CHÍNH HỢP NHẤT NĂM 2024\n"
                "Đơn vị: Triệu đồng\n"
                "**TỔNG CỘNG TÀI SẢN**\n"
                "* Số cuối năm: 200\n"
                "* Số đầu năm: 180\n"
            ),
        },
    ]

    ingested = client.post("/api/v1/financial/ingest/batch", json={"documents": documents})
    assert ingested.status_code == 200
    assert ingested.json()["documents_with_chunks"] == 2
    assert len(api_module.CHUNKS_DB) > 0
    assert store.document_count() == 2

    listed = client.get("/api/v1/financial/documents")
    assert listed.status_code == 200
    stored_documents = listed.json()
    assert {item["source_file"] for item in stored_documents} == {
        "ABB_BCTC_2024.md",
        "ACB_BCTC_2024.md",
    }

    monkeypatch.setattr(api_module, "CHUNKS_DB", [])
    monkeypatch.setattr(api_module, "retriever", None)
    with TestClient(api_module.app) as persisted_client:
        assert len(api_module.CHUNKS_DB) == ingested.json()["new_chunks_count"]
        assert persisted_client.get("/health").json()["persistence"] == "sqlite"

        retrieved = persisted_client.post(
            "/api/v1/financial/retrieve",
            json={
                "query": "Tổng tài sản ABB cuối năm 2024 theo báo cáo hợp nhất là bao nhiêu?",
                "top_k": 5,
            },
        )
        assert retrieved.status_code == 200
        assert any("100" in item["chunk"]["content"] for item in retrieved.json())

        deleted = persisted_client.delete(
            f"/api/v1/financial/documents/{stored_documents[0]['logical_key']}"
        )
        assert deleted.status_code == 200
    assert store.document_count() == 1
