import json

import pytest

from evaluation.run_financial_analysis_benchmark import (
    BenchmarkDataError,
    load_cases,
    run_benchmark,
    score_case,
)


def _case(*, answerable=True, review_status="verified"):
    case = {
        "id": "fpt_revenue_compare_2024_2025",
        "split": "dev",
        "category": "period_comparison",
        "review_status": review_status,
        "query": "So sánh doanh thu FPT năm 2024 và 2025",
        "answerable": answerable,
        "expected_scope": {"company_codes": ["FPT"], "fiscal_years": [2024, 2025]},
    }
    if answerable:
        case.update({
            "expected_facts": [
                {
                    "label": "Doanh thu năm 2024",
                    "year": 2024,
                    "value": "100",
                    "unit": "Triệu đồng",
                    "sources": [{"file_name": "FPT_BCTC_2025.md", "page": 50}],
                },
                {
                    "label": "Doanh thu năm 2025",
                    "year": 2025,
                    "value": "120",
                    "unit": "Triệu đồng",
                    "sources": [{"file_name": "FPT_BCTC_2025.md", "page": 50}],
                },
            ],
            "expected_calculations": [
                {"formula_id": "absolute_change", "value": "20", "unit": "Triệu đồng"}
            ],
            "evidence_groups": [
                {"id": "revenue_2024", "sources": [{"file_name": "FPT_BCTC_2025.md", "page": 50}]},
                {"id": "revenue_2025", "sources": [{"file_name": "FPT_BCTC_2025.md", "page": 50}]},
            ],
        })
    else:
        case["expected_scope"] = {"company_codes": ["FPT"], "fiscal_years": [2030]}
        case["abstention_markers"] = ["không tìm thấy"]
    return case


def _response(answer="Doanh thu năm 2024 là 100 Triệu đồng [c1]. Năm 2025 là 120 Triệu đồng [c2]. Chênh lệch 20 Triệu đồng [c1][c2]."):
    return {
        "answer": answer,
        "parsed_query": {
            "company_code": "FPT",
            "company_codes": ["FPT"],
            "fiscal_year": 2025,
            "fiscal_years": [2024, 2025],
        },
        "validation": {"is_valid": True, "unit_match": True, "citation_valid": True},
        "citations": [
            {
                "citation_id": "c1", "file_name": "FPT_BCTC_2025.md", "page": 50,
                "unit": "Triệu đồng", "chunk_id": "fpt-2024",
                "snippet": "Chỉ tiêu gốc trong BCTC: Doanh thu | Năm 2024: 100 Triệu đồng",
            },
            {
                "citation_id": "c2", "file_name": "FPT_BCTC_2025.md", "page": 50,
                "unit": "Triệu đồng", "chunk_id": "fpt-2025",
                "snippet": "Chỉ tiêu gốc trong BCTC: Doanh thu | Năm 2025: 120 Triệu đồng",
            },
        ],
        "calculations": [
            {
                "formula_id": "absolute_change", "value": "20", "unit": "Triệu đồng",
                "input_citation_ids": ["c1", "c2"],
                "input_chunk_ids": ["fpt-2024", "fpt-2025"],
            }
        ],
    }


def test_score_case_checks_scope_facts_units_formula_lineage_and_retrieval_groups():
    retrieved = [
        {"chunk": {"source_file": "FPT_BCTC_2025.md", "page": 50}},
    ]
    scored = score_case(_case(), _response(), retrieved)

    assert scored["case_pass"] is True
    assert scored["scope"]["passed"] is True
    assert all(fact["passed"] for fact in scored["expected_facts"])
    assert scored["expected_calculations"][0]["passed"] is True
    assert scored["calculation_lineage_valid"] is True
    assert scored["retrieval_evidence_group_recall_at_k"] == 1.0
    assert scored["answer_evidence_group_coverage"] == 1.0
    assert scored["citation_precision"] == 1.0

    response = _response(answer=_response()["answer"] + " Biên lợi nhuận là 25%.")
    scored_with_extra_claim = score_case(_case(), response, retrieved)
    assert scored_with_extra_claim["case_pass"] is False
    assert scored_with_extra_claim["unexpected_numeric_values"] == ["25"]


@pytest.mark.parametrize("swapped", [False, True])
def test_two_period_prose_with_one_source_is_bound_to_explicit_year_values(swapped):
    case = _case()
    case["expected_calculations"] = []
    before, after = (120, 100) if swapped else (100, 120)
    response = _response(f"Doanh thu: 2024 {before} Triệu đồng; 2025 {after} Triệu đồng [c1].")
    response["citations"] = response["citations"][:1]
    response["citations"][0]["snippet"] += " | Năm 2025: 120 Triệu đồng"
    response["calculations"] = []
    result = score_case(case, response)
    assert all(fact["passed"] for fact in result["expected_facts"]) is (not swapped)


def test_score_case_fails_abstention_with_amount_or_missing_refusal_marker():
    case = _case(answerable=False)
    response = _response("Không tìm thấy dữ liệu của FPT năm 2030.")
    response["parsed_query"]["fiscal_year"] = 2030
    response["parsed_query"]["fiscal_years"] = [2030]
    response["calculations"] = []
    assert score_case(case, response)["case_pass"] is True

    response["answer"] = "Không tìm thấy, nhưng số dư là 1.500 Triệu đồng."
    assert score_case(case, response)["abstention_pass"] is False
    response["answer"] = "Năm 2030 chưa có báo cáo."
    assert score_case(case, response)["abstention_pass"] is False


@pytest.mark.parametrize("answer", [
    "Doanh thu năm 2024 là 120 Triệu đồng [c2]. Năm 2025 là 100 Triệu đồng [c1]. Chênh lệch 20 Triệu đồng [c1][c2].",
    "Doanh thu năm 2024 là 100 Triệu đồng [c2]. Năm 2025 là 120 Triệu đồng [c1]. Chênh lệch 20 Triệu đồng [c1][c2].",
    "| Chỉ tiêu | 2024 | 2025 | Nguồn |\n| --- | ---: | ---: | --- |\n| Doanh thu | 120 Triệu đồng | 100 Triệu đồng | [c1] [c2] |\nChênh lệch 20 Triệu đồng [c1][c2].",
])
def test_fact_scoring_rejects_swapped_years_or_citations(answer):
    scored = score_case(_case(), _response(answer))
    assert not scored["case_pass"]
    assert any(not fact["passed"] for fact in scored["expected_facts"])


def test_fact_scoring_accepts_correct_comparison_columns_with_shared_row_citations():
    answer = "| Chỉ tiêu | 2024 | 2025 | Nguồn |\n| --- | ---: | ---: | --- |\n| Doanh thu | 100 Triệu đồng | 120 Triệu đồng | [c1] [c2] |\nChênh lệch 20 Triệu đồng [c1][c2]."
    scored = score_case(_case(), _response(answer), [{"chunk": {"source_file": "FPT_BCTC_2025.md", "page": 50}}])
    assert scored["case_pass"]


def test_holdout_cases_must_be_marked_verified(tmp_path):
    case = _case(review_status="pending")
    case["split"] = "holdout"
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(case), encoding="utf-8")

    with pytest.raises(BenchmarkDataError, match="reviewer-verified"):
        load_cases(path, "holdout")


def test_verified_holdout_requires_reviewer_and_timestamp(tmp_path):
    case = _case(review_status="verified")
    case["split"] = "holdout"
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(case), encoding="utf-8")
    with pytest.raises(BenchmarkDataError, match="reviewer_id"):
        load_cases(path, "holdout")

    case["reviewer_id"] = "reviewer-1"
    case["reviewed_at"] = "2026-09-29T10:30:00+07:00"
    path.write_text(json.dumps(case), encoding="utf-8")
    assert len(load_cases(path, "holdout")) == 1


@pytest.mark.parametrize("leak", ["document", "issuer", "reviewer"])
def test_holdout_selection_checks_the_entire_dataset_for_leakage(tmp_path, leak):
    dev = _case()
    dev["reviewer_id"] = "reviewer-1"
    dev["reviewed_at"] = "2026-10-02T09:00:00+07:00"
    holdout = json.loads(json.dumps(dev))
    holdout.update({"id": "holdout-case", "split": "holdout"})
    holdout["expected_scope"]["company_codes"] = ["MBB"]
    if leak != "document":
        for group in (holdout["expected_facts"], holdout["evidence_groups"]):
            for record in group:
                record["sources"][0]["file_name"] = "MBB_2025.txt"
    if leak == "issuer":
        holdout["expected_scope"]["company_codes"] = ["FPT"]
    if leak == "reviewer":
        holdout["author_id"] = "reviewer-1"
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(json.dumps(case) for case in (dev, holdout)), encoding="utf-8")
    with pytest.raises(BenchmarkDataError, match={"document": "document appears", "issuer": "issuer appears",
                                               "reviewer": "independent reviewer"}[leak]):
        load_cases(path, "holdout")


def test_run_benchmark_calls_retrieval_and_answer_and_records_hybrid_readiness():
    case = _case()
    calls = []

    def fake_get(url, timeout):
        assert url.endswith("/health/ready")
        return {"status": "ready", "components": {"dense_retrieval": "ready"}}

    def fake_post(url, payload, timeout):
        calls.append((url, payload["query"], payload["top_k"]))
        if url.endswith("/retrieve"):
            return [{"chunk": {"source_file": "FPT_BCTC_2025.md", "page": 50}}]
        return _response()

    result = run_benchmark(
        [case], base_url="http://financial-rag", top_k=5, timeout=1,
        poster=fake_post, getter=fake_get,
    )

    assert result["run_metadata"]["retrieval_mode_inferred"] == "hybrid"
    assert result["run_metadata"]["holdout_verified"] is False
    assert result["summary"]["case_pass_rate"] == 1.0
    assert len(calls) == 2
    assert all(call[1:] == (case["query"], 5) for call in calls)


def test_run_benchmark_marks_mode_transition_instead_of_claiming_pure_hybrid():
    statuses = iter(["indexing", "ready"])

    def fake_get(url, timeout):
        return {"status": "ready", "components": {"dense_retrieval": next(statuses)}}

    def fake_post(url, payload, timeout):
        if url.endswith("/retrieve"):
            return [{"chunk": {"source_file": "FPT_BCTC_2025.md", "page": 50}}]
        return _response()

    result = run_benchmark(
        [_case(review_status="pending")], base_url="http://financial-rag", top_k=5, timeout=1,
        poster=fake_post, getter=fake_get,
    )
    assert result["run_metadata"]["retrieval_mode_inferred"] == "transitioned_during_run"
    assert result["run_metadata"]["readiness_after"]["components"]["dense_retrieval"] == "ready"
    assert result["run_metadata"]["verified_case_count"] == 0
