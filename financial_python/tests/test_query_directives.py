from financial_rag.query.directives import QueryDirectivePlanner
from financial_rag.query.understanding import QueryUnderstandingEngine
from financial_rag.retrieval.operators import (
    FilterOperator,
    StructuralOperator,
    ValidateOperator,
)
from financial_rag.schemas import ChunkType, FinancialChunk, RetrievalResult


def _result(
    chunk_id: str,
    content: str,
    chunk_type: ChunkType = ChunkType.TABLE,
    section: str | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        chunk=FinancialChunk(
            chunk_id=chunk_id,
            source_file="FPT_BCTC_2025.md",
            company_code="FPT",
            fiscal_year=2025,
            chunk_type=chunk_type,
            section=section,
            content=content,
        ),
        rrf_score=1.0,
    )


def test_planner_extracts_structure_explicit_exclusions_and_disambiguation():
    query = (
        "Tổng tài sản VPB năm 2025, xem bảng; bỏ qua mục lục, "
        "đừng nhầm với tài sản thuế thu nhập hoãn lại."
    )
    parsed = QueryUnderstandingEngine().parse(query)
    plan = QueryDirectivePlanner().plan(query, parsed)

    assert plan.structural.preferred_chunk_types == ["table", "financial_item"]
    assert plan.filters.exclude_phrases == ["muc luc"]
    assert plan.validation.negative_constraints == ["tai san thue thu nhap hoan lai"]
    assert plan.validation.positive_constraints == []

    explicit_positive = QueryDirectivePlanner().plan(
        "Tổng tài sản VPB năm 2025, chỉ sử dụng số liệu hợp nhất", parsed
    )
    assert explicit_positive.validation.positive_constraints == ["so lieu hop nhat"]


def test_structural_operator_selects_matching_chunks_and_falls_back_if_no_match():
    parsed = QueryUnderstandingEngine().parse("Doanh thu thuần FPT năm 2025, trong bảng")
    plan = QueryDirectivePlanner().plan(parsed.original_query, parsed)
    table = _result("table", "Doanh thu thuần 500", ChunkType.TABLE)
    narrative = _result("narrative", "Giải thích doanh thu", ChunkType.NARRATIVE)

    selected, applied = StructuralOperator.select([narrative.chunk, table.chunk], plan)
    assert applied
    assert [chunk.chunk_id for chunk in selected] == ["table"]

    no_match_plan = plan.model_copy(
        update={
            "structural": plan.structural.model_copy(
                update={"preferred_chunk_types": ["parent_child"]}
            )
        }
    )
    fallback, applied = StructuralOperator.select([narrative.chunk, table.chunk], no_match_plan)
    assert not applied
    assert fallback == [narrative.chunk, table.chunk]


def test_structural_operator_uses_complete_rows_for_component_questions():
    query = "Thành phần tài sản FPT năm 2025"
    parsed = QueryUnderstandingEngine().parse(query)
    plan = QueryDirectivePlanner().plan(query, parsed)
    narrative = _result("narrative", "Diễn giải tình hình tài sản", ChunkType.NARRATIVE)
    item = _result("item", "Tiền và các khoản tương đương tiền: 500", ChunkType.FINANCIAL_ITEM)

    selected, applied = StructuralOperator.select([narrative.chunk, item.chunk], plan)
    assert plan.structural.require_complete_rows
    assert applied
    assert [chunk.chunk_id for chunk in selected] == ["item"]


def test_filter_operator_removes_explicit_noise_but_preserves_metric_evidence():
    query = "Doanh thu thuần FPT năm 2025, bỏ qua mục lục"
    parsed = QueryUnderstandingEngine().parse(query)
    plan = QueryDirectivePlanner().plan(query, parsed)
    toc = _result("toc", "Mục lục báo cáo và danh sách các bảng")
    evidence = _result("evidence", "Mục lục thuyết minh: Doanh thu thuần 500")

    filtered, applied = FilterOperator.apply([toc, evidence], plan, parsed)
    assert applied
    assert [result.chunk.chunk_id for result in filtered] == ["evidence"]


def test_filter_operator_restores_original_candidates_if_everything_would_be_removed():
    query = "Doanh thu thuần FPT năm 2025, bỏ qua mục lục"
    parsed = QueryUnderstandingEngine().parse(query)
    plan = QueryDirectivePlanner().plan(query, parsed)
    candidates = [_result("toc-1", "Mục lục báo cáo"), _result("toc-2", "Mục lục phụ lục")]

    filtered, applied = FilterOperator.apply(candidates, plan, parsed)
    assert not applied
    assert filtered == candidates


def test_filter_operator_uses_batch_classifier_when_configured():
    class FakeClassifier:
        def classify(self, query, exclusions, chunks):
            assert "muc luc" in exclusions
            return [False, True]

    query = "Doanh thu thuần FPT năm 2025, bỏ qua mục lục"
    parsed = QueryUnderstandingEngine().parse(query)
    plan = QueryDirectivePlanner().plan(query, parsed)
    candidates = [
        _result("noise", "Đoạn giới thiệu rủi ro chung"),
        _result("evidence", "Doanh thu thuần: 500"),
    ]

    filtered, applied = FilterOperator.apply(
        candidates, plan, parsed, classifier=FakeClassifier()
    )
    assert applied
    assert [result.chunk.chunk_id for result in filtered] == ["evidence"]


def test_validate_operator_ranks_away_explicitly_confusable_metric():
    query = "Doanh thu thuần FPT năm 2025, đừng nhầm với doanh thu tài chính"
    parsed = QueryUnderstandingEngine().parse(query)
    plan = QueryDirectivePlanner().plan(query, parsed)
    confusable = _result("financial", "Doanh thu tài chính: 80")
    correct = _result("net", "Doanh thu thuần: 500")

    ranked, applied = ValidateOperator.rerank([confusable, correct], plan)
    assert applied
    assert [result.chunk.chunk_id for result in ranked] == ["net", "financial"]


def test_validate_operator_keeps_order_when_constraints_are_not_discriminative():
    query = "Doanh thu thuần FPT năm 2025, đừng nhầm với khoản không tồn tại"
    parsed = QueryUnderstandingEngine().parse(query)
    plan = QueryDirectivePlanner().plan(query, parsed)
    candidates = [_result("first", "Doanh thu thuần 500"), _result("second", "Doanh thu thuần 450")]

    ranked, applied = ValidateOperator.rerank(candidates, plan)
    assert not applied
    assert ranked == candidates


def test_validate_operator_ignores_generic_financial_context_constraints():
    query = "Tổng nợ phải trả ABB năm 2025, chỉ sử dụng báo cáo tài chính"
    parsed = QueryUnderstandingEngine().parse(query)
    plan = QueryDirectivePlanner().plan(query, parsed)
    candidates = [
        _result("first", "Tổng nợ phải trả: 500"),
        _result("second", "Báo cáo tài chính hợp nhất năm 2025"),
    ]

    ranked, applied = ValidateOperator.rerank(candidates, plan)
    assert plan.validation.positive_constraints == ["bao cao tai chinh"]
    assert not applied
    assert ranked == candidates
