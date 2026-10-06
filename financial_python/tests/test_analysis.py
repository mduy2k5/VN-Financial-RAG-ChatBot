from financial_rag.analysis.ratio import build_ratio_answer
from financial_rag.analysis.overview import build_financial_overview_answer
from financial_rag.generation.chat import FinancialChatEngine
from financial_rag.metadata.metric_mapper import FinancialMetricMapper
from financial_rag.query.understanding import QueryUnderstandingEngine
from financial_rag.query_guides import build_query_plan
from financial_rag.schemas import (
    Citation,
    ChunkType,
    FinancialChunk,
    ParsedQuery,
    RetrievalResult,
    StatementType,
)


def test_overview_never_calculates_change_across_reporting_bases():
    engine = QueryUnderstandingEngine()
    question = "So sánh doanh thu thuần FPT năm 2024 và 2025"
    parsed = engine.parse(question)
    plan = build_query_plan(
        question, access_filter={"corpus": "local_financial_index"},
        parsed_query=parsed, engine=engine,
    )
    metrics = {item.metric_id: item for item in engine.metric_mapper.metrics_list}
    results, citations = [], []
    for index, (year, basis, value) in enumerate([(2024, "consolidated", "100"), (2025, "separate", "160")], 1):
        chunk = FinancialChunk(
            chunk_id=f"fpt-{year}", document_id=f"fpt-{year}", source_file=f"FPT_{year}.txt",
            company_code="FPT", fiscal_year=year, report_type=basis, page=11, unit="Triệu đồng",
            statement_type=StatementType.INCOME_STATEMENT, chunk_type=ChunkType.FINANCIAL_ITEM,
            content=f"Chỉ tiêu gốc trong BCTC: DOANH THU THUẦN | {year}: {value} Triệu đồng",
            metadata={"metric_name": "DOANH THU THUẦN", "period_values": {str(year): value}},
        )
        results.append(RetrievalResult(chunk=chunk, rrf_score=1))
        citations.append(Citation(
            citation_id=f"c{index}", document_id=chunk.document_id, chunk_id=chunk.chunk_id,
            file_name=chunk.source_file, snippet=chunk.content, unit=chunk.unit,
        ))
    answer = build_financial_overview_answer(
        plan, parsed, metrics, {"doanh_thu_thuan": results}, citations, [2024, 2025],
    )
    assert answer.calculations == []
    assert "Không tính biến động" in answer.answer
    assert "không cùng cơ sở báo cáo" in answer.answer
    assert "60,00%" not in answer.answer


def _margin_evidence(*, revenue_document: str = "fpt-2025"):
    metrics = {
        metric.metric_id: metric
        for metric in FinancialMetricMapper().metrics_list
    }
    rows = [
        ("gross-profit", "loi_nhuan_gop", "LỢI NHUẬN GỘP", "40", StatementType.INCOME_STATEMENT),
        ("net-revenue", "doanh_thu_thuan", "DOANH THU THUẦN", "160", StatementType.INCOME_STATEMENT),
    ]
    chunks = []
    citations = []
    results = []
    for index, (suffix, metric_id, label, value_2025, statement) in enumerate(rows, start=1):
        document_id = revenue_document if metric_id == "doanh_thu_thuan" else "fpt-2025"
        value_2024 = "30" if metric_id == "loi_nhuan_gop" else "100"
        chunk = FinancialChunk(
            chunk_id=f"fpt-{suffix}",
            source_file="FPT_BCTC_2025.md",
            document_id=document_id,
            company_code="FPT",
            fiscal_year=2025,
            comparative_fiscal_year=2024,
            report_type="consolidated",
            page=50,
            statement_type=statement,
            canonical_metric=metrics[metric_id].canonical_name,
            chunk_type=ChunkType.FINANCIAL_ITEM,
            unit="Triệu đồng",
            content=(
                f"Chỉ tiêu gốc trong BCTC: {label}\n"
                f"Số cuối năm (2025): {value_2025} Triệu đồng\n"
                f"Số đầu năm (2024): {value_2024} Triệu đồng"
            ),
            metadata={
                "metric_name": label,
                "period_values": {"2024": value_2024, "2025": value_2025},
            },
        )
        citation = Citation(
            citation_id=f"c{index}",
            document_id=document_id,
            chunk_id=chunk.chunk_id,
            file_name=chunk.source_file,
            page=chunk.page,
            statement_type=statement.value,
            unit=chunk.unit,
            snippet=chunk.content,
        )
        chunks.append(chunk)
        citations.append(citation)
        results.append(RetrievalResult(chunk=chunk, rrf_score=1.0))
    return chunks, citations, results


def test_profit_margin_uses_source_labels_without_schema_node_ids():
    _chunks, citations, results = _margin_evidence()
    parsed = ParsedQuery(
        original_query="Biên lợi nhuận gộp của FPT năm 2025",
        company_code="FPT",
        company_codes=["FPT"],
        fiscal_year=2025,
        fiscal_years=[2024, 2025],
        report_type="consolidated",
        statement_type=StatementType.INCOME_STATEMENT,
    )

    answer = build_ratio_answer(
        parsed,
        results,
        citations,
        [citation.snippet for citation in citations],
    )

    assert "30,00%" in answer.answer
    assert "25,00%" in answer.answer
    assert len(answer.calculations) == 2
    assert {item.formula_id for item in answer.calculations} == {"gross_margin_percent"}
    assert all(set(item.input_citation_ids) == {"c1", "c2"} for item in answer.calculations)


def test_ratio_inputs_from_different_filings_are_not_combined():
    _chunks, citations, results = _margin_evidence(revenue_document="fpt-separate-2025")
    parsed = ParsedQuery(
        original_query="Biên lợi nhuận gộp của FPT năm 2025",
        company_code="FPT",
        company_codes=["FPT"],
        fiscal_year=2025,
        fiscal_years=[2025],
        report_type="consolidated",
        statement_type=StatementType.INCOME_STATEMENT,
    )

    answer = build_ratio_answer(
        parsed,
        results,
        citations,
        [citation.snippet for citation in citations],
    )

    assert "Không đủ bằng chứng để tính tỷ số" in answer.answer
    assert answer.calculations == []


def test_unqualified_profit_margin_requests_definition_clarification():
    _chunks, citations, results = _margin_evidence()
    parsed = ParsedQuery(
        original_query="Biên lợi nhuận của FPT năm 2025",
        company_code="FPT",
        company_codes=["FPT"],
        fiscal_year=2025,
        fiscal_years=[2025],
    )

    answer = build_ratio_answer(
        parsed,
        results,
        citations,
        [citation.snippet for citation in citations],
    )

    assert "cần chỉ rõ biên lợi nhuận gộp, hoạt động hay sau thuế" in answer.answer
    assert answer.calculations == []


def test_query_understanding_routes_explicit_profit_margin_to_ratio_analysis():
    parsed = QueryUnderstandingEngine().parse("Biên lợi nhuận gộp FPT năm 2025")

    assert parsed.structured_operation == "ratio_analysis"


def test_analytical_plan_ignores_schema_node_id_as_a_financial_metric():
    question = (
        "Phân tích cơ cấu tài sản, nợ phải trả và vốn chủ sở hữu của MBB năm 2024; "
        "tính tỷ lệ nợ trên tài sản và đối chiếu phương trình kế toán."
    )
    engine = QueryUnderstandingEngine()
    parsed = engine.parse(question)
    plan = build_query_plan(
        question,
        access_filter={"corpus": "local_financial_index"},
        parsed_query=parsed,
        engine=engine,
    )

    assert plan.task == "analysis"
    assert plan.planned_metric_ids == [
        "tong_no_phai_tra",
        "tong_cong_tai_san",
        "tong_von_chu_so_huu",
    ]
    assert "LIABILITY" not in plan.planned_metric_ids


def test_composition_share_evidence_requires_same_source_table():
    total = FinancialChunk(
        chunk_id="fpt-total-assets",
        source_file="FPT_BCTC_2025.md",
        document_id="fpt-2025",
        company_code="FPT",
        fiscal_year=2025,
        report_type="consolidated",
        page=7,
        statement_type=StatementType.BALANCE_SHEET,
        chunk_type=ChunkType.FINANCIAL_ITEM,
        unit="Triệu đồng",
        content="Chỉ tiêu gốc trong BCTC: TỔNG CỘNG TÀI SẢN",
        metadata={
            "metric_name": "TỔNG CỘNG TÀI SẢN",
            "period_values": {"2025": "1.000"},
        },
    )
    cash = FinancialChunk(
        chunk_id="fpt-cash",
        source_file="FPT_BCTC_2025.md",
        document_id="fpt-2025",
        company_code="FPT",
        fiscal_year=2025,
        report_type="consolidated",
        page=7,
        statement_type=StatementType.BALANCE_SHEET,
        chunk_type=ChunkType.FINANCIAL_ITEM,
        unit="Triệu đồng",
        content="Chỉ tiêu gốc trong BCTC: TIỀN VÀ TƯƠNG ĐƯƠNG TIỀN",
        metadata={
            "metric_name": "TIỀN VÀ TƯƠNG ĐƯƠNG TIỀN",
            "period_values": {"2025": "100"},
        },
    )
    citations = [
        Citation(
            citation_id=f"c{index}",
            document_id="fpt-2025",
            chunk_id=chunk.chunk_id,
            file_name=chunk.source_file,
            page=chunk.page,
            statement_type=chunk.statement_type.value,
            unit=chunk.unit,
            snippet=chunk.content,
        )
        for index, chunk in enumerate((total, cash), start=1)
    ]
    parsed = ParsedQuery(
        original_query="Cơ cấu tài sản FPT năm 2025",
        company_code="FPT",
        company_codes=["FPT"],
        fiscal_year=2025,
        fiscal_years=[2025],
        structured_operation="composition",
    )
    engine = FinancialChatEngine()

    same_table = engine.source_composition_calculations(
        parsed,
        [RetrievalResult(chunk=total, rrf_score=1), RetrievalResult(chunk=cash, rrf_score=0.9)],
        citations,
    )
    assert any(item.value == "10.00" and item.formula_id == "share_of_total" for item in same_table)

    cash.page = 8
    different_page = engine.source_composition_calculations(
        parsed,
        [RetrievalResult(chunk=total, rrf_score=1), RetrievalResult(chunk=cash, rrf_score=0.9)],
        citations,
    )
    assert all(
        item.input_chunk_ids[0] != "fpt-cash"
        for item in different_page
    )
