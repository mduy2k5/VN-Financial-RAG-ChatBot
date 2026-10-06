from pathlib import Path

from financial_rag.evaluation.benchmark import _without_metadata_scope, run_benchmark
from financial_rag.evaluation.metrics import CitationEvaluator, RetrievalEvaluator
from financial_rag.schemas import ParsedQuery, PeriodType, StatementType


def test_metadata_filter_ablation_preserves_query_intent_and_metric():
    query = ParsedQuery(
        original_query="So sánh doanh thu FPT 2025 với 2024",
        company_code="FPT",
        fiscal_year=2025,
        fiscal_years=[2025, 2024],
        statement_type=StatementType.INCOME_STATEMENT,
        primary_metric="DOANH THU THUẦN",
        period_type=PeriodType.FULL_YEAR,
    )

    unscoped = _without_metadata_scope(query)

    assert query.company_code == "FPT"
    assert query.fiscal_years == [2025, 2024]
    assert unscoped.company_code is None
    assert unscoped.fiscal_year is None
    assert unscoped.fiscal_years == []
    assert unscoped.statement_type is None
    assert unscoped.primary_metric == query.primary_metric
    assert unscoped.period_type == query.period_type


def test_retrieval_metrics_measure_page_hits():
    retrieved = [4, 7, 7, 9]
    relevant = [7, 9]

    assert RetrievalEvaluator.recall_at_k(retrieved, relevant, k=3) == 0.5
    assert RetrievalEvaluator.precision_at_k(retrieved, relevant, k=3) == 2 / 3
    assert RetrievalEvaluator.precision_of_returned_results(retrieved, relevant) == 0.75
    assert RetrievalEvaluator.reciprocal_rank(retrieved, relevant) == 0.5
    assert RetrievalEvaluator.precision_at_k([7], [7], k=5) == 0.2
    assert RetrievalEvaluator.precision_of_returned_results([7], [7]) == 1.0
    assert RetrievalEvaluator.precision_of_returned_results([], [7]) == 0.0


def test_citation_metrics_ignore_missing_page_numbers():
    assert CitationEvaluator.page_recall([None, 7], [7, 9]) == 0.5
    assert CitationEvaluator.page_precision([None, 7], [7, 9]) == 1.0


def test_labeled_sample_benchmark_has_retrieval_citations_and_numeric_evidence():
    root = Path(__file__).resolve().parents[1]
    result = run_benchmark(
        root / "evaluation" / "demo_cases.jsonl",
        root / "sample_data" / "DEMO_SYNTHETIC_2025.md",
        top_k=5,
        strict_metric_scope=True,
    )

    assert result["summary"]["cases"] == 2
    assert result["summary"]["recall_at_k"] == 1.0
    assert result["summary"]["mrr"] == 1.0
    assert result["summary"]["citation_page_recall"] == 1.0
    assert result["summary"]["evidence_value_hit_rate"] == 1.0
    assert len(result["summary"]["dataset_sha256"]) == 64
    assert all(len(source["sha256"]) == 64 for source in result["summary"]["source_sha256"])
    assert result["summary"]["companies"] == ["DEMO"]
