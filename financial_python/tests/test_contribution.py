from dataclasses import replace
from decimal import Decimal

import pytest

from financial_rag.analysis.contribution import (
    AnalysisInputError, PeriodAmount, build_contribution_answer, calculate_contributions,
)
from financial_rag.analysis.evidence import validate_calculation
from financial_rag.context.citation_mapper import CitationMapper
from financial_rag.schemas import ChunkType, FinancialChunk, ParsedQuery, RetrievalResult, StatementType


def _evidence(rows, *, label="TỔNG NỢ PHẢI TRẢ", company="FPT", total_first=False):
    chunks = []
    for index, (name, before, after) in enumerate(rows):
        periods = {str(year): str(value) for year, value in ((2024, before), (2025, after)) if value is not None}
        content = f"Chỉ tiêu gốc trong BCTC: {name} | " + " | ".join(
            f"Năm {year}: {value} Triệu đồng" for year, value in periods.items()
        )
        chunks.append(FinancialChunk(
            chunk_id=f"source_item_{index}", document_id="source-2025", source_file=f"{company}_2025.md",
            company_code=company, fiscal_year=2025, comparative_fiscal_year=2024,
            statement_type=StatementType.BALANCE_SHEET, chunk_type=ChunkType.FINANCIAL_ITEM,
            page=7, report_type="consolidated", unit="Triệu đồng", table_title="Bảng cân đối kế toán",
            content=content, metadata={
                "metric_name": name, "period_values": periods, "row_index": index,
                "section_group": label,
                "code": "250" if name == label else str(250 + index) if total_first else "",
            },
        ))
    results = [RetrievalResult(chunk=chunk, rrf_score=1) for chunk in chunks]
    citations, _ = CitationMapper.map_citations(results)
    parsed = ParsedQuery(
        original_query=f"Khoản nào đóng góp vào biến động {label} {company} năm 2024 và 2025",
        company_code=company, company_codes=[company], fiscal_years=[2024, 2025], fiscal_year=2025,
        report_type="consolidated", statement_type=StatementType.BALANCE_SHEET,
        primary_metric=label, structured_operation="contribution_analysis",
    )
    answer = build_contribution_answer(parsed, results, citations, [c.content for c in chunks],
                                       chunks_by_id={c.chunk_id: c for c in chunks})
    return chunks, citations, answer


def _valid_calculations(chunks, citations, answer):
    assert answer.calculations
    assert all(validate_calculation(item, {c.citation_id: c for c in citations},
                                   {c.chunk_id: c for c in chunks}) for item in answer.calculations)


def test_partial_components_keep_values_and_publish_residuals():
    chunks, citations, answer = _evidence([
        ("Vay ngắn hạn", 60, 80), ("Phải trả người bán", 40, None),
        ("TỔNG NỢ PHẢI TRẢ", 100, 130),
    ])
    assert answer.answer_status == "partial"
    assert answer.missing_components == ("Phải trả người bán",)
    assert answer.residuals == {2024: "40", 2025: "50"}
    assert "66,67%" in answer.answer
    assert "| Phải trả người bán | 40 Triệu đồng | — | — | —" in answer.answer
    assert "chưa kết luận khoản đóng góp lớn nhất" in answer.answer
    assert "| Biến động chưa đối chiếu | 10 Triệu đồng" in answer.answer
    assert "khớp tổng báo cáo ở cả hai kỳ" not in answer.answer
    _valid_calculations(chunks, citations, answer)


def test_unreconciled_rows_are_partial_even_when_all_visible_rows_have_two_years():
    chunks, citations, answer = _evidence([
        ("Vay ngắn hạn", 60, 80), ("TỔNG NỢ PHẢI TRẢ", 100, 130),
    ])
    assert answer.answer_status == "partial"
    assert answer.missing_components == ()
    assert answer.residuals == {2024: "40", 2025: "50"}
    assert "không được gán thành một khoản nợ" in answer.answer
    _valid_calculations(chunks, citations, answer)


def test_zero_total_change_keeps_component_changes_without_dividing_by_zero():
    chunks, citations, answer = _evidence([
        ("Vay ngắn hạn", 41, 70), ("Phải trả người bán", 59, 30),
        ("TỔNG NỢ PHẢI TRẢ", 100, 100),
    ])
    assert answer.answer_status == "complete"
    assert "| 29 Triệu đồng | — |" in answer.answer
    assert "| -29 Triệu đồng | — |" in answer.answer
    assert "Tổng không biến động nên không tính tỷ lệ đóng góp" in answer.answer
    assert not any(item.formula_id == "contribution_to_change" for item in answer.calculations)
    _valid_calculations(chunks, citations, answer)


def test_missing_total_period_returns_only_source_total():
    chunks, citations, answer = _evidence([
        ("Vay ngắn hạn", 60, 80), ("TỔNG NỢ PHẢI TRẢ", None, 130),
    ])
    assert answer.answer_status == "partial"
    assert answer.missing_periods == (2024,)
    assert answer.calculations == []
    assert "| TỔNG NỢ PHẢI TRẢ | 2025 | 130 Triệu đồng |" in answer.answer
    assert len(answer.citations) == 1
    assert "Không suy ra thành phần" in answer.answer


def test_five_unmapped_investment_components_are_kept():
    label = "Đầu tư tài chính dài hạn"
    chunks, citations, answer = _evidence([
        (label, 177, 197), ("Đầu tư vào công ty con", 11, 13),
        ("Đầu tư vào công ty liên kết", 23, 29), ("Đầu tư góp vốn khác", 37, 41),
        ("Dự án hợp tác đầu tư", 47, 53), ("Đầu tư nắm giữ đến ngày đáo hạn", 59, 61),
    ], label=label, total_first=True)
    assert answer.answer_status == "complete"
    assert all(name in answer.answer for name, _, _ in [
        ("Đầu tư vào công ty con", 11, 13), ("Đầu tư vào công ty liên kết", 23, 29),
        ("Đầu tư góp vốn khác", 37, 41), ("Dự án hợp tác đầu tư", 47, 53),
        ("Đầu tư nắm giữ đến ngày đáo hạn", 59, 61),
    ])
    assert len([item for item in answer.calculations if item.formula_id == "contribution_to_change"]) == 5
    assert not any("schema_node_id" in chunk.metadata for chunk in chunks)
    _valid_calculations(chunks, citations, answer)


def test_negative_total_delta_preserves_signed_contribution_and_offsets():
    chunks, citations, answer = _evidence([
        ("Vay ngắn hạn", 70, 30), ("Phải trả người bán", 30, 40),
        ("TỔNG NỢ PHẢI TRẢ", 100, 70),
    ])
    assert "133,33%" in answer.answer and "-33,33%" in answer.answer
    assert answer.calculations[0].value == "-30"
    _valid_calculations(chunks, citations, answer)


def test_totals_only_never_claim_complete_components():
    chunks, citations, answer = _evidence([("TỔNG NỢ PHẢI TRẢ", 100, 130)])
    assert answer.answer_status == "partial"
    assert answer.residuals == {2024: "100", 2025: "130"}
    assert "khớp tổng báo cáo ở cả hai kỳ" not in answer.answer
    _valid_calculations(chunks, citations, answer)


@pytest.mark.parametrize("mutation", ["year", "value", "role", "citation", "output", "unit", "missing_inputs"])
def test_calculation_validation_rejects_tampered_lineage(mutation):
    chunks, citations, answer = _evidence([
        ("Vay ngắn hạn", 60, 80), ("Phải trả người bán", 40, 50),
        ("TỔNG NỢ PHẢI TRẢ", 100, 130),
    ])
    item = answer.calculations[0]
    first = item.inputs[0]
    if mutation == "year":
        item = replace(item, inputs=(replace(first, year=2025), item.inputs[1]))
    elif mutation == "value":
        item = replace(item, inputs=(replace(first, value="130"), item.inputs[1]))
    elif mutation == "role":
        item = replace(item, inputs=(replace(first, role="current_value"), item.inputs[1]))
    elif mutation == "citation":
        item = replace(item, inputs=(replace(first, citation_id=citations[0].citation_id), item.inputs[1]))
    elif mutation == "output":
        item = replace(item, value="300")
    elif mutation == "unit":
        item = replace(item, unit="Tỷ đồng")
    else:
        item = replace(item, inputs=())
    assert not validate_calculation(item, {c.citation_id: c for c in citations}, {c.chunk_id: c for c in chunks})


@pytest.mark.parametrize("field,value", [
    ("unit", "Tỷ đồng"), ("company_code", "VPB"), ("report_type", "separate"),
    ("year", 2023), ("label", "Chỉ tiêu khác"), ("value", Decimal("NaN")),
])
def test_partial_calculator_still_rejects_unsafe_inputs(field, value):
    before = PeriodAmount("FPT", "consolidated", 2024, Decimal(100), "Triệu đồng", "Tổng", "total", "c1")
    after = replace(before, year=2025, value=Decimal(130))
    component_before = replace(before, value=Decimal(60), label="Khoản A", chunk_id="component", citation_id="c2")
    component_after = replace(component_before, year=2025, value=Decimal(80))
    with pytest.raises(AnalysisInputError):
        calculate_contributions(before, after, [(component_before, replace(component_after, **{field: value}))])


@pytest.mark.parametrize("kind", ["partial", "zero_change", "missing_total"])
def test_api_preserves_contribution_status_and_typed_calculation_inputs(monkeypatch, kind):
    import importlib
    from fastapi.testclient import TestClient
    from financial_rag.retrieval.hybrid import FinancialHybridRetriever

    module = importlib.import_module("financial_rag.api.app")
    if kind == "partial":
        rows = [("Vay ngắn hạn", 60, 80), ("Phải trả người bán", 40, None), ("TỔNG NỢ PHẢI TRẢ", 100, 130)]
    elif kind == "zero_change":
        rows = [("Vay ngắn hạn", 41, 70), ("Phải trả người bán", 59, 30), ("TỔNG NỢ PHẢI TRẢ", 100, 100)]
    else:
        rows = [("Vay ngắn hạn", 60, 80), ("TỔNG NỢ PHẢI TRẢ", None, 130)]
    chunks, _, _ = _evidence(rows)
    monkeypatch.setattr(module, "CHUNKS_DB", chunks)
    monkeypatch.setattr(module, "retriever", FinancialHybridRetriever(chunks))
    monkeypatch.setattr("financial_rag.generation.chat.requests.post", lambda *args, **kwargs: (
        _ for _ in ()
    ).throw(AssertionError("source analysis must not call the LLM")))
    response = TestClient(module.app).post("/api/v1/financial/answer", json={
        "query": "Khoản nào đóng góp vào biến động nợ phải trả FPT năm 2024 và 2025",
    })
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["validation"]["is_valid"], payload
    assert payload["answer_status"] == ("complete" if kind == "zero_change" else "partial")
    if kind == "partial":
        assert payload["analysis_coverage"]["residuals"] == {"2024": "40", "2025": "50"}
        assert payload["analysis_coverage"]["missing_components"] == ["Phải trả người bán"]
    if kind == "missing_total":
        assert payload["calculations"] == []
        assert payload["analysis_coverage"]["missing_periods"] == [2024]
    else:
        assert all(item["inputs"] for item in payload["calculations"])
        total_change = payload["calculations"][0]
        assert [fact["role"] for fact in total_change["inputs"]] == ["prior_value", "current_value"]
        assert [fact["year"] for fact in total_change["inputs"]] == [2024, 2025]


def test_api_validator_rejects_wrong_result_even_if_that_number_exists_in_source():
    import importlib
    module = importlib.import_module("financial_rag.api.app")
    chunks, citations, answer = _evidence([
        ("Vay ngắn hạn", 60, 80), ("Phải trả người bán", 40, 50), ("TỔNG NỢ PHẢI TRẢ", 100, 130),
    ])
    item = replace(answer.calculations[0], value="130")
    validation = module._validate_answer(
        "Mức tăng tổng là 130 Triệu đồng [c3].", citations, [chunk.content for chunk in chunks],
        [item], chunks_by_id={chunk.chunk_id: chunk for chunk in chunks},
    )
    assert not validation.is_valid
    assert "vai trò/kỳ/giá trị nguồn" in validation.details
