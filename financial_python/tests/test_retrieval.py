import pytest

from financial_rag.schemas import FinancialChunk, StatementType, ChunkType, ParsedQuery, PeriodType
from financial_rag.retrieval.hybrid import FinancialHybridRetriever
import financial_rag.retrieval.hybrid as hybrid_module
from financial_rag.retrieval.rrf import reciprocal_rank_fusion
from financial_rag.chunking.financial_chunker import FinancialMarkdownChunker
from financial_rag.query.understanding import QueryUnderstandingEngine


def _filing_row(year, basis, values):
    label = "TỔNG NỢ PHẢI TRẢ"
    return FinancialChunk(
        chunk_id=f"AAA-{basis}-{year}", document_id=f"AAA-{basis}-{year}",
        source_file=f"AAA_{basis}_{year}.txt", company_code="AAA",
        fiscal_year=year, report_type=basis, page=11,
        comparative_fiscal_year=min(values) if len(values) > 1 else None,
        statement_type=StatementType.BALANCE_SHEET,
        chunk_type=ChunkType.FINANCIAL_ITEM, unit="Triệu đồng",
        content=f"Chỉ tiêu gốc trong BCTC: {label} | {values}",
        metadata={"metric_name": label, "period_values": {str(k): str(v) for k, v in values.items()}},
    )


@pytest.mark.parametrize("explicit_basis,expected", [
    (None, ["AAA-separate-2024"]),
    ("separate", ["AAA-separate-2024"]),
    ("consolidated", ["AAA-consolidated-2023"]),
])
def test_comparison_selects_one_basis_with_period_coverage(explicit_basis, expected):
    chunks = [
        _filing_row(2023, "consolidated", {2022: 50, 2023: 60}),
        _filing_row(2024, "separate", {2023: 40, 2024: 70}),
    ]
    query = ParsedQuery(
        original_query="So sánh tổng nợ phải trả AAA năm 2023 và 2024",
        company_code="AAA", fiscal_years=[2023, 2024], report_type=explicit_basis,
        primary_metric="TỔNG NỢ PHẢI TRẢ", statement_type=StatementType.BALANCE_SHEET,
    )
    results = FinancialHybridRetriever(chunks).search(query)
    assert [result.chunk.chunk_id for result in results] == expected


def test_missing_period_uses_comparative_evidence_even_when_another_year_is_exact():
    chunks = [
        _filing_row(2023, "consolidated", {2022: 50, 2023: 60}),
        _filing_row(2025, "consolidated", {2024: 70, 2025: 90}),
    ]
    query = ParsedQuery(
        original_query="So sánh tổng nợ phải trả AAA năm 2023 và 2024",
        company_code="AAA", fiscal_years=[2023, 2024], primary_metric="TỔNG NỢ PHẢI TRẢ",
    )
    results = FinancialHybridRetriever(chunks).search(query)
    assert {result.chunk.chunk_id for result in results} == {chunk.chunk_id for chunk in chunks}


def test_incomplete_comparison_never_mixes_separate_and_consolidated():
    chunks = [
        _filing_row(2023, "consolidated", {2023: 60}),
        _filing_row(2024, "separate", {2024: 70}),
    ]
    query = ParsedQuery(
        original_query="So sánh tổng nợ phải trả AAA năm 2023 và 2024",
        company_code="AAA", fiscal_years=[2023, 2024], primary_metric="TỔNG NỢ PHẢI TRẢ",
    )
    results = FinancialHybridRetriever(chunks).search(query)
    assert [result.chunk.chunk_id for result in results] == ["AAA-consolidated-2023"]


def test_equally_covering_bases_keep_default_consolidated_preference():
    chunks = [
        _filing_row(2024, "consolidated", {2023: 60, 2024: 90}),
        _filing_row(2024, "separate", {2023: 40, 2024: 70}),
    ]
    query = ParsedQuery(
        original_query="So sánh tổng nợ phải trả AAA năm 2023 và 2024",
        company_code="AAA", fiscal_years=[2023, 2024], primary_metric="TỔNG NỢ PHẢI TRẢ",
    )
    results = FinancialHybridRetriever(chunks).search(query)
    assert [result.chunk.chunk_id for result in results] == ["AAA-consolidated-2024"]

def test_rrf():
    ranking1 = ["doc1", "doc2", "doc3"]
    ranking2 = ["doc2", "doc1", "doc4"]
    fused = reciprocal_rank_fusion([ranking1, ranking2], k=60)
    assert len(fused) == 4
    # doc1 và doc2 phải đứng đầu vì xuất hiện ở cả 2 bảng
    top_docs = [doc_id for doc_id, _ in fused[:2]]
    assert "doc1" in top_docs and "doc2" in top_docs

def test_hybrid_search_boost():
    chunks = [
        FinancialChunk(
            chunk_id="c_noise",
            source_file="bctc.md",
            page=20,
            statement_type=StatementType.NOTES,
            chunk_type=ChunkType.NARRATIVE,
            content="Chính sách kế toán về nợ và tài sản tài chính của ngân hàng..."
        ),
        FinancialChunk(
            chunk_id="c_target",
            source_file="bctc.md",
            company_code="ABB",
            fiscal_year=2025,
            page=9,
            statement_type=StatementType.BALANCE_SHEET,
            chunk_type=ChunkType.FINANCIAL_ITEM,
            primary_metrics=["TỔNG NỢ PHẢI TRẢ"],
            content="Chỉ tiêu: TỔNG NỢ PHẢI TRẢ | Số đầu năm: 620.000 | Số cuối năm: 750.000"
        )
    ]

    retriever = FinancialHybridRetriever(chunks)
    pq = ParsedQuery(
        original_query="Tổng nợ phải trả đầu năm là bao nhiêu?",
        company_code="ABB",
        statement_type=StatementType.BALANCE_SHEET,
        primary_metric="TỔNG NỢ PHẢI TRẢ",
        period_type=PeriodType.BEGINNING_OF_YEAR,
        rewritten_query_bm25="TỔNG NỢ PHẢI TRẢ Số đầu năm"
    )

    results = retriever.search(pq, top_k=1)
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "c_target"


def test_hybrid_retriever_rejects_duplicate_chunk_ids():
    duplicate_chunks = [
        FinancialChunk(chunk_id="duplicate", source_file="one.txt", content="one"),
        FinancialChunk(chunk_id="duplicate", source_file="two.txt", content="two"),
    ]

    with pytest.raises(ValueError, match="unique chunk IDs"):
        FinancialHybridRetriever(duplicate_chunks)


def test_metadata_filters_do_not_fall_back_to_other_company_or_year():
    chunk = FinancialChunk(
        chunk_id="abb_2025",
        source_file="abb_2025.md",
        company_code="ABB",
        fiscal_year=2025,
        statement_type=StatementType.BALANCE_SHEET,
        chunk_type=ChunkType.FINANCIAL_ITEM,
        content="TỔNG NỢ PHẢI TRẢ: 620.000",
    )
    query = ParsedQuery(
        original_query="Tổng nợ phải trả của VCB năm 2024",
        company_code="VCB",
        fiscal_year=2024,
        statement_type=StatementType.BALANCE_SHEET,
        primary_metric="TỔNG NỢ PHẢI TRẢ",
        rewritten_query_bm25="TỔNG NỢ PHẢI TRẢ",
    )

    assert FinancialHybridRetriever([chunk]).search(query) == []


def test_query_for_comparative_year_retrieves_the_prior_column():
    source = (
        "# Page 167\n"
        "**BÁO CÁO TÌNH HÌNH TÀI CHÍNH HỢP NHẤT**\n"
        "Đơn vị: VND\n"
        "**TỔNG TÀI SẢN**\n"
        "*   Số cuối năm: 250\n"
        "*   Số đầu năm: 200\n"
    )
    chunks = FinancialMarkdownChunker().chunk_text(
        source, source_file="FPT_BCTC_2025.md", company_code="FPT"
    )
    query = QueryUnderstandingEngine().parse("Tổng tài sản FPT cuối năm 2024")

    results = FinancialHybridRetriever(chunks).search(query, top_k=3)

    assert results
    assert results[0].chunk.metadata["period_values"]["2024"] == "200"
    assert "Giá trị cuối năm 2024 (cột so sánh): 200" in results[0].chunk.content


def test_comparison_query_retrieves_evidence_for_both_requested_years():
    source = (
        "# Page 170\n"
        "**BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH HỢP NHẤT**\n"
        "Đơn vị: VND\n"
        "**DOANH THU THUẦN**\n"
        "*   Năm nay: 70,112,825,100,710\n"
        "*   Năm trước: 62,848,794,351,367\n"
    )
    chunks = FinancialMarkdownChunker().chunk_text(
        source, source_file="FPT_Annual_Report_2025_benchmark.md", company_code="FPT"
    )
    query = QueryUnderstandingEngine().parse(
        "Doanh thu thuần FPT năm 2025 tăng bao nhiêu so với năm 2024?"
    )

    results = FinancialHybridRetriever(chunks).search(query, top_k=3)

    assert results
    assert results[0].chunk.metadata["period_values"] == {
        "2025": "70,112,825,100,710",
        "2024": "62,848,794,351,367",
    }


def test_multi_company_comparison_retrieves_both_synthetic_values():
    chunks = [
        FinancialChunk(
            chunk_id="demo-assets-2025",
            source_file="DEMO_SYNTHETIC_2025.md",
            company_code="DEMO",
            fiscal_year=2025,
            page=8,
            statement_type=StatementType.BALANCE_SHEET,
            chunk_type=ChunkType.FINANCIAL_ITEM,
            primary_metrics=["TỔNG CỘNG TÀI SẢN"],
            canonical_metric="TỔNG CỘNG TÀI SẢN",
            content="TỔNG CỘNG TÀI SẢN: 1.250 Triệu đồng",
            metadata={"period_values": {"2025": "1.250"}},
        ),
        FinancialChunk(
            chunk_id="mock-assets-2025",
            source_file="MOCK_SYNTHETIC_2025.md",
            company_code="MOCK",
            fiscal_year=2025,
            page=8,
            statement_type=StatementType.BALANCE_SHEET,
            chunk_type=ChunkType.FINANCIAL_ITEM,
            primary_metrics=["TỔNG CỘNG TÀI SẢN"],
            canonical_metric="TỔNG CỘNG TÀI SẢN",
            content="TỔNG CỘNG TÀI SẢN: 2.400 Triệu đồng",
            metadata={"period_values": {"2025": "2.400"}},
        ),
    ]
    query = QueryUnderstandingEngine().parse(
        "So sánh tổng tài sản cuối năm 2025 của DEMO và MOCK, nêu giá trị từng bên."
    )

    results = FinancialHybridRetriever(chunks).search(query, top_k=5)

    retrieved_values = {
        result.chunk.company_code: result.chunk.content
        for result in results
        if result.chunk.company_code in {"DEMO", "MOCK"}
    }
    assert query.company_codes == ["DEMO", "MOCK"]
    assert [result.chunk.company_code for result in results] == ["DEMO", "MOCK"]
    assert "1.250" in retrieved_values["DEMO"]
    assert "2.400" in retrieved_values["MOCK"]
    retrieved_company_pages = {
        (result.chunk.company_code, result.chunk.page)
        for result in results
        if result.chunk.company_code in {"DEMO", "MOCK"}
    }
    assert {("DEMO", 8), ("MOCK", 8)} <= retrieved_company_pages


def test_known_metric_without_evidence_returns_no_results():
    source = (
        "# Page 167\n"
        "**BÁO CÁO TÌNH HÌNH TÀI CHÍNH HỢP NHẤT**\n"
        "Đơn vị: VND\n"
        "**TỔNG TÀI SẢN**\n"
        "*   Số cuối năm: 3,000\n"
        "*   Số đầu năm: 2,500\n"
    )
    chunks = FinancialMarkdownChunker().chunk_text(
        source, source_file="DEMO_SYNTHETIC_2025.md", company_code="DEMO"
    )
    query = QueryUnderstandingEngine().parse(
        "Tiền gửi của khách hàng DEMO cuối năm 2025 là bao nhiêu?"
    )

    assert query.primary_metric == "TIỀN GỬI CỦA KHÁCH HÀNG"
    assert FinancialHybridRetriever(chunks).search(query, top_k=3) == []


def test_exact_financial_item_lookup_keeps_context_on_the_matching_source_page():
    target = FinancialChunk(
        chunk_id="fpt-liabilities",
        source_file="fpt_2025.md",
        document_id="fpt-2025",
        company_code="FPT",
        fiscal_year=2025,
        page=9,
        statement_type=StatementType.BALANCE_SHEET,
        chunk_type=ChunkType.FINANCIAL_ITEM,
        primary_metrics=["TỔNG NỢ PHẢI TRẢ"],
        content="Chỉ tiêu: TỔNG NỢ PHẢI TRẢ | Số cuối năm: 100",
    )
    related_page = FinancialChunk(
        chunk_id="fpt-table-page-9",
        source_file="fpt_2025.md",
        document_id="fpt-2025",
        company_code="FPT",
        fiscal_year=2025,
        page=9,
        statement_type=StatementType.BALANCE_SHEET,
        chunk_type=ChunkType.TABLE,
        content="TỔNG NỢ PHẢI TRẢ và các khoản mục liên quan",
    )
    noisy_page = FinancialChunk(
        chunk_id="fpt-notes-page-20",
        source_file="fpt_2025.md",
        document_id="fpt-2025",
        company_code="FPT",
        fiscal_year=2025,
        page=20,
        statement_type=StatementType.BALANCE_SHEET,
        chunk_type=ChunkType.NARRATIVE,
        content="Chính sách quản lý tổng nợ phải trả và rủi ro thanh khoản",
    )
    query = ParsedQuery(
        original_query="Tổng nợ phải trả của FPT năm 2025",
        company_code="FPT",
        fiscal_year=2025,
        statement_type=StatementType.BALANCE_SHEET,
        primary_metric="TỔNG NỢ PHẢI TRẢ",
        rewritten_query_bm25="TỔNG NỢ PHẢI TRẢ",
    )

    results = FinancialHybridRetriever([target, related_page, noisy_page]).search(query, top_k=5)

    assert results
    assert {result.chunk.page for result in results} == {9}


def test_hybrid_reuses_one_bm25_scope_and_rebuilds_when_scope_changes(monkeypatch):
    original_retriever = hybrid_module.BM25Retriever
    indexed_contents = []

    class CountingBM25Retriever(original_retriever):
        def __init__(self, chunks):
            indexed_contents.append([chunk.content for chunk in chunks])
            super().__init__(chunks)

    monkeypatch.setattr(hybrid_module, "BM25Retriever", CountingBM25Retriever)
    chunks = [
        FinancialChunk(
            chunk_id="abb-asset",
            source_file="abb.md",
            company_code="ABB",
            content="Tổng tài sản ABB",
        ),
        FinancialChunk(
            chunk_id="fpt-asset",
            source_file="fpt.md",
            company_code="FPT",
            content="Tổng tài sản FPT",
        ),
    ]
    retriever = FinancialHybridRetriever(chunks)
    query = ParsedQuery(original_query="Tổng tài sản", rewritten_query_bm25="Tổng tài sản")

    assert indexed_contents == []  # No unused full-corpus index at startup.
    retriever.search(query)
    retriever.search(query)
    assert len(indexed_contents) == 1

    chunks[1].content = "Doanh thu thuần FPT"
    retriever.search(query)
    assert len(indexed_contents) == 2
    assert indexed_contents[-1][1] == "Doanh thu thuần FPT"

    abb_query = ParsedQuery(
        original_query="Tổng tài sản ABB",
        company_code="ABB",
        rewritten_query_bm25="Tổng tài sản",
    )
    retriever.search(abb_query)
    assert len(indexed_contents) == 3
    assert retriever._bm25_cache is not None
    assert [chunk.chunk_id for chunk in retriever._bm25_cache.chunks] == ["abb-asset"]


def test_statement_routing_and_canonical_metric_boost():
    chunks = [
        FinancialChunk(
            chunk_id="c_bs",
            source_file="bctc.txt",
            company_code="ACV",
            statement_type=StatementType.BALANCE_SHEET,
            statement_name_vi="Bảng cân đối kế toán",
            canonical_metric="TỔNG NỢ PHẢI TRẢ",
            content="Phân hệ báo cáo: Bảng cân đối kế toán | Chỉ tiêu chuẩn hóa: TỔNG NỢ PHẢI TRẢ | Số cuối năm: 750.000"
        ),
        FinancialChunk(
            chunk_id="c_is",
            source_file="bctc.txt",
            company_code="ACV",
            statement_type=StatementType.INCOME_STATEMENT,
            statement_name_vi="Báo cáo kết quả hoạt động kinh doanh",
            canonical_metric="DOANH THU THUẦN",
            content="Phân hệ báo cáo: Báo cáo kết quả hoạt động kinh doanh | Chỉ tiêu chuẩn hóa: DOANH THU THUẦN | Năm nay: 900"
        ),
        FinancialChunk(
            chunk_id="c_cf",
            source_file="bctc.txt",
            company_code="ACV",
            statement_type=StatementType.CASH_FLOW,
            statement_name_vi="Báo cáo lưu chuyển tiền tệ",
            canonical_metric="LƯU CHUYỂN TIỀN TỪ HOẠT ĐỘNG KINH DOANH",
            metric_aliases=["dòng tiền kinh doanh", "tiền từ hđkd"],
            content="Phân hệ báo cáo: Báo cáo lưu chuyển tiền tệ | Chỉ tiêu chuẩn hóa: LƯU CHUYỂN TIỀN TỪ HOẠT ĐỘNG KINH DOANH | Năm nay: 8.500.000"
        ),
        FinancialChunk(
            chunk_id="c_notes",
            source_file="bctc.txt",
            company_code="ACV",
            statement_type=StatementType.NOTES,
            statement_name_vi="Thuyết minh báo cáo tài chính",
            canonical_metric="THUYẾT MINH HÀNG TỒN KHO",
            content="Phân hệ báo cáo: Thuyết minh báo cáo tài chính | Chỉ tiêu chuẩn hóa: THUYẾT MINH HÀNG TỒN KHO"
        ),
    ]

    retriever = FinancialHybridRetriever(chunks)
    engine = QueryUnderstandingEngine()

    # Query 1: Tìm dòng tiền kinh doanh -> Phải ra chunk Cash Flow c_cf
    pq_cf = engine.parse("Dòng tiền từ hoạt động kinh doanh của ACV")
    res_cf = retriever.search(pq_cf, top_k=1)
    assert len(res_cf) == 1
    assert res_cf[0].chunk.chunk_id == "c_cf"
    assert res_cf[0].chunk.statement_name_vi == "Báo cáo lưu chuyển tiền tệ"

    # Query 2: Tìm nợ phải trả -> Phải ra chunk Balance Sheet c_bs
    pq_bs = engine.parse("Tổng nợ phải trả của ACV")
    res_bs = retriever.search(pq_bs, top_k=1)
    assert len(res_bs) == 1
    assert res_bs[0].chunk.chunk_id == "c_bs"
    assert res_bs[0].chunk.statement_name_vi == "Bảng cân đối kế toán"


def test_exact_metric_prefers_requested_statement_over_duplicate_note_row():
    chunks = [
        FinancialChunk(
            chunk_id="duplicate-note",
            source_file="MBB_2025.txt",
            document_id="mbb-2025",
            company_code="MBB",
            fiscal_year=2025,
            report_type="consolidated",
            page=55,
            statement_type=StatementType.NOTES,
            chunk_type=ChunkType.FINANCIAL_ITEM,
            canonical_metric="LỢI NHUẬN SAU THUẾ",
            unit="Triệu đồng",
            metadata={
                "metric_name": "Lợi nhuận sau thuế",
                "period_values": {"2025": "944.059"},
            },
            content="Chỉ tiêu gốc trong BCTC: Lợi nhuận sau thuế | 2025: 944.059 Triệu đồng",
        ),
        FinancialChunk(
            chunk_id="income-statement-row",
            source_file="MBB_2025.txt",
            document_id="mbb-2025",
            company_code="MBB",
            fiscal_year=2025,
            report_type="consolidated",
            page=12,
            statement_type=StatementType.INCOME_STATEMENT,
            chunk_type=ChunkType.FINANCIAL_ITEM,
            canonical_metric="LỢI NHUẬN SAU THUẾ",
            unit="Triệu đồng",
            metadata={
                "metric_name": "Lợi nhuận sau thuế",
                "period_values": {"2025": "944.059"},
            },
            content="Chỉ tiêu gốc trong BCTC: Lợi nhuận sau thuế | 2025: 944.059 Triệu đồng",
        ),
    ]
    query = ParsedQuery(
        original_query="Lợi nhuận sau thuế MBB năm 2025 là bao nhiêu?",
        company_code="MBB",
        fiscal_year=2025,
        report_type="consolidated",
        statement_type=StatementType.INCOME_STATEMENT,
        primary_metric="LỢI NHUẬN SAU THUẾ",
        canonical_metric="LỢI NHUẬN SAU THUẾ",
        metric_aliases=["lợi nhuận sau thuế"],
    )

    results = FinancialHybridRetriever(chunks).search(query, top_k=5)

    assert [result.chunk.chunk_id for result in results] == ["income-statement-row"]


def test_shorter_statement_label_beats_longer_duplicate_note_label():
    metric = "TỔNG LỢI NHUẬN KẾ TOÁN TRƯỚC THUẾ"
    chunks = [
        FinancialChunk(
            chunk_id="full-label-note",
            source_file="MBB_2025.txt",
            company_code="MBB",
            fiscal_year=2025,
            report_type="consolidated",
            page=55,
            statement_type=StatementType.NOTES,
            chunk_type=ChunkType.FINANCIAL_ITEM,
            canonical_metric=metric,
            metadata={"metric_name": metric, "period_values": {"2025": "1.200"}},
            content=f"Chỉ tiêu gốc trong BCTC: {metric} | 2025: 1.200 Triệu đồng",
        ),
        FinancialChunk(
            chunk_id="short-label-income-row",
            source_file="MBB_2025.txt",
            company_code="MBB",
            fiscal_year=2025,
            report_type="consolidated",
            page=12,
            statement_type=StatementType.INCOME_STATEMENT,
            chunk_type=ChunkType.FINANCIAL_ITEM,
            canonical_metric=metric,
            metadata={
                "metric_name": "TỔNG LỢI NHUẬN TRƯỚC THUẾ",
                "period_values": {"2025": "1.200"},
            },
            content=(
                "Chỉ tiêu gốc trong BCTC: TỔNG LỢI NHUẬN TRƯỚC THUẾ "
                "| 2025: 1.200 Triệu đồng"
            ),
        ),
    ]
    query = ParsedQuery(
        original_query=f"{metric} MBB năm 2025 là bao nhiêu?",
        company_code="MBB",
        fiscal_year=2025,
        report_type="consolidated",
        statement_type=StatementType.INCOME_STATEMENT,
        primary_metric=metric,
        canonical_metric=metric,
        metric_aliases=[metric],
    )

    results = FinancialHybridRetriever(chunks).search(query, top_k=5)

    assert [result.chunk.chunk_id for result in results] == ["short-label-income-row"]


def test_statement_preference_keeps_duplicate_note_as_fallback_when_no_statement_row():
    note = FinancialChunk(
        chunk_id="duplicate-note",
        source_file="MBB_2025.txt",
        company_code="MBB",
        fiscal_year=2025,
        report_type="consolidated",
        page=55,
        statement_type=StatementType.NOTES,
        chunk_type=ChunkType.FINANCIAL_ITEM,
        canonical_metric="LỢI NHUẬN SAU THUẾ",
        unit="Triệu đồng",
        metadata={
            "metric_name": "Lợi nhuận sau thuế",
            "period_values": {"2025": "944.059"},
        },
        content="Chỉ tiêu gốc trong BCTC: Lợi nhuận sau thuế | 2025: 944.059 Triệu đồng",
    )
    query = ParsedQuery(
        original_query="Lợi nhuận sau thuế MBB năm 2025 là bao nhiêu?",
        company_code="MBB",
        fiscal_year=2025,
        report_type="consolidated",
        statement_type=StatementType.INCOME_STATEMENT,
        primary_metric="LỢI NHUẬN SAU THUẾ",
        canonical_metric="LỢI NHUẬN SAU THUẾ",
        metric_aliases=["lợi nhuận sau thuế"],
    )

    results = FinancialHybridRetriever([note]).search(query, top_k=5)

    assert [result.chunk.chunk_id for result in results] == ["duplicate-note"]


def test_schema_match_does_not_fill_missing_source_label_in_multi_company_query():
    metric = "VỐN TÀI TRỢ ỦY THÁC ĐẦU TƯ CHO VAY MÀ TỔ CHỨC TÍN DỤNG CHỊU RỦI RO"
    shared_schema_node = "balance_sheet.entrusted_funding"
    chunks = [
        FinancialChunk(
            chunk_id="aaa-target-row",
            source_file="AAA_2025.txt",
            company_code="AAA",
            fiscal_year=2025,
            report_type="consolidated",
            page=11,
            statement_type=StatementType.BALANCE_SHEET,
            chunk_type=ChunkType.FINANCIAL_ITEM,
            canonical_metric=metric,
            metadata={"metric_name": metric, "schema_node_id": shared_schema_node},
            content=f"Chỉ tiêu gốc trong BCTC: {metric} | 2025: 100 Triệu đồng",
        ),
        FinancialChunk(
            chunk_id="bbb-different-source-label",
            source_file="BBB_2025.txt",
            company_code="BBB",
            fiscal_year=2025,
            report_type="consolidated",
            page=11,
            statement_type=StatementType.BALANCE_SHEET,
            chunk_type=ChunkType.FINANCIAL_ITEM,
            canonical_metric=metric,
            metadata={"metric_name": "Các nguồn vốn khác", "schema_node_id": shared_schema_node},
            content="Chỉ tiêu gốc trong BCTC: Các nguồn vốn khác | 2025: 200 Triệu đồng",
        ),
    ]
    query = ParsedQuery(
        original_query=f"So sánh {metric} của AAA và BBB năm 2025",
        company_code="AAA",
        company_codes=["AAA", "BBB"],
        fiscal_year=2025,
        report_type="consolidated",
        statement_type=StatementType.BALANCE_SHEET,
        primary_metric=metric,
        canonical_metric=metric,
        metric_aliases=[metric],
        schema_node_id=shared_schema_node,
    )

    results = FinancialHybridRetriever(chunks).search(query, top_k=5)

    # The second row carries the requested schema node, but its extracted
    # source label does not support the requested metric. Schema annotations
    # may help chunking; they must not supply a missing comparison value.
    assert results == []


def test_exact_source_alias_beats_broad_alias_that_matches_a_child_row():
    """A broad revenue alias must not mask an exact total-row alias."""
    chunks = [
        FinancialChunk(
            chunk_id="operating-revenue-child",
            source_file="ACBS_2024.txt",
            document_id="acbs-2024-separate",
            company_code="ACBS",
            fiscal_year=2024,
            comparative_fiscal_year=2023,
            report_type="separate",
            page=15,
            statement_type=StatementType.INCOME_STATEMENT,
            chunk_type=ChunkType.FINANCIAL_ITEM,
            metadata={
                "metric_name": "Doanh thu hoạt động tư vấn tài chính",
                "period_values": {"2023": "1.900.000.000", "2024": "1.100.000.000"},
            },
            content=(
                "Chỉ tiêu gốc trong BCTC: Doanh thu hoạt động tư vấn tài chính "
                "| Năm nay (2024): 1.100.000.000 VND "
                "| Năm trước (2023): 1.900.000.000 VND"
            ),
        ),
        FinancialChunk(
            chunk_id="operating-revenue-total",
            source_file="ACBS_2024.txt",
            document_id="acbs-2024-separate",
            company_code="ACBS",
            fiscal_year=2024,
            comparative_fiscal_year=2023,
            report_type="separate",
            page=15,
            statement_type=StatementType.INCOME_STATEMENT,
            chunk_type=ChunkType.FINANCIAL_ITEM,
            metadata={
                "metric_name": "Cộng doanh thu hoạt động",
                "period_values": {
                    "2023": "2.000.000.000",
                    "2024": "3.000.000.000",
                },
            },
            content=(
                "Chỉ tiêu gốc trong BCTC: Cộng doanh thu hoạt động "
                "| Năm nay (2024): 3.000.000.000 VND "
                "| Năm trước (2023): 2.000.000.000 VND"
            ),
        ),
    ]
    query = ParsedQuery(
        original_query="Tổng doanh thu hoạt động ACBS năm 2024",
        company_code="ACBS",
        fiscal_year=2024,
        report_type="separate",
        statement_type=StatementType.INCOME_STATEMENT,
        primary_metric="Tổng doanh thu hoạt động",
        canonical_metric="Tổng doanh thu hoạt động",
        metric_aliases=[
            "Cộng doanh thu hoạt động",
            "Doanh thu hoạt động",
        ],
    )

    results = FinancialHybridRetriever(chunks).search(query, top_k=5)

    assert [item.chunk.chunk_id for item in results] == ["operating-revenue-total"]


def test_multi_company_metric_does_not_return_a_partial_comparison():
    metric = "VỐN TÀI TRỢ ỦY THÁC ĐẦU TƯ CHO VAY MÀ TỔ CHỨC TÍN DỤNG CHỊU RỦI RO"
    only_company_row = FinancialChunk(
        chunk_id="aaa-target-row",
        source_file="AAA_2025.txt",
        company_code="AAA",
        fiscal_year=2025,
        report_type="consolidated",
        page=11,
        statement_type=StatementType.BALANCE_SHEET,
        chunk_type=ChunkType.FINANCIAL_ITEM,
        canonical_metric=metric,
        metadata={"metric_name": metric},
        content=f"Chỉ tiêu gốc trong BCTC: {metric} | 2025: 100 Triệu đồng",
    )
    query = ParsedQuery(
        original_query=f"So sánh {metric} của AAA và BBB năm 2025",
        company_code="AAA",
        company_codes=["AAA", "BBB"],
        fiscal_year=2025,
        report_type="consolidated",
        statement_type=StatementType.BALANCE_SHEET,
        primary_metric=metric,
        canonical_metric=metric,
        metric_aliases=[metric],
    )

    results = FinancialHybridRetriever([only_company_row]).search(query, top_k=5)

    assert results == []
