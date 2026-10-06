"""Company-scoped lookup smoke gate using fictional source rows."""

from __future__ import annotations

import pytest

from financial_rag.generation.chat import FinancialChatEngine
from financial_rag.query.understanding import QueryUnderstandingEngine
from financial_rag.retrieval.hybrid import FinancialHybridRetriever
from financial_rag.schemas import (
    ChunkType,
    Citation,
    FinancialChunk,
    StatementType,
)
from financial_rag.validation.citation import CitationValidator
from financial_rag.validation.numeric import NumericValidator


CASES: list[dict[str, object]] = [
    {
        "id": "demo_assets_2025",
        "company": "DEMO",
        "source_file": "DEMO_SYNTHETIC_2025.md",
        "source_label": "TỔNG TÀI SẢN",
        "value": "1.250",
        "unit": "Triệu đồng",
        "report_type": "consolidated",
        "page": 8,
        "query": "Tổng tài sản của DEMO cuối năm 2025 là bao nhiêu?",
    },
    {
        "id": "mock_assets_2025",
        "company": "MOCK",
        "source_file": "MOCK_SYNTHETIC_2025.md",
        "source_label": "TỔNG TÀI SẢN",
        "value": "2.400",
        "unit": "Triệu đồng",
        "report_type": "consolidated",
        "page": 8,
        "query": "Tổng tài sản của MOCK cuối năm 2025 là bao nhiêu?",
    },
]


def _source_chunk(case: dict[str, object]) -> FinancialChunk:
    company = str(case["company"])
    source_file = str(case["source_file"])
    label = str(case["source_label"])
    value = str(case["value"])
    unit = str(case["unit"])
    page = int(case["page"])
    content = (
        f"Doanh nghiệp: {company}\n"
        "Phân hệ báo cáo: Bảng cân đối kế toán (balance_sheet)\n"
        f"Chỉ tiêu gốc trong BCTC: {label}\n"
        f"Số cuối năm (2025): {value} {unit}"
    )
    return FinancialChunk(
        chunk_id=f"{company.casefold()}-total-assets-2024",
        source_file=source_file,
        document_id=f"{company.casefold()}-2024-{case['report_type']}",
        company_code=company,
        fiscal_year=2025,
        comparative_fiscal_year=2024,
        report_type=str(case["report_type"]),
        page=page,
        statement_type=StatementType.BALANCE_SHEET,
        canonical_metric="TỔNG TÀI SẢN",
        chunk_type=ChunkType.FINANCIAL_ITEM,
        unit=unit,
        content=content,
        primary_metrics=[label],
        metadata={
            "original_label": label,
            "canonical_metric": "TỔNG TÀI SẢN",
            "period_values": {"2025": value},
            "end_value": value,
        },
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_metric_lookup_is_grounded_for_each_corpus_issuer(case: dict[str, object]) -> None:
    chunk = _source_chunk(case)
    retriever = FinancialHybridRetriever([chunk])
    query = QueryUnderstandingEngine().parse(str(case["query"]))

    assert query.company_codes == [case["company"]]
    results = retriever.search(query, top_k=5)
    assert [result.chunk.chunk_id for result in results] == [chunk.chunk_id]

    citation = Citation(
        citation_id="c1",
        document_id=chunk.document_id or chunk.source_file,
        chunk_id=chunk.chunk_id,
        file_name=chunk.source_file,
        page=chunk.page,
        statement_type=chunk.statement_type.value,
        section=chunk.section,
        unit=chunk.unit,
        snippet=chunk.content,
    )
    answer = FinancialChatEngine._source_exact_metric_answer(query, results, [citation])

    assert answer is not None
    assert str(case["value"]) in answer
    assert str(case["unit"]) in answer
    assert CitationValidator.validate(answer, [citation])
    assert NumericValidator.validate(answer, [chunk.content]).is_valid
