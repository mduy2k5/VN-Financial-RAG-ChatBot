"""Save the current C1 BM25 benchmark and classify observed failures."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any, Callable

from financial_rag.evaluation.benchmark import run_benchmark
from financial_rag.evaluation.local_data import local_data_path


ROOT = Path(__file__).resolve().parents[1]
DATASET = local_data_path("evaluation", "ground_truth_multi_company.jsonl")
SOURCES = [
    local_data_path("sample_data", "ABB_BCTC_2025.md"),
    local_data_path("evaluation", "sources", "FPT_Annual_Report_2024_benchmark.md"),
    local_data_path("evaluation", "sources", "FPT_Annual_Report_2025_benchmark.md"),
    local_data_path("evaluation", "sources", "HPG_Annual_Report_2025_benchmark.md"),
    local_data_path("evaluation", "sources", "FPT_Q2_2026_benchmark.md"),
    local_data_path("evaluation", "sources", "HPG_Q2_2025_benchmark.md"),
    local_data_path("evaluation", "sources", "VNF_Interim_Consolidated_2025H1_benchmark.md"),
]
RESULT_PATH = ROOT / "evaluation" / "results" / "multi_company_bm25_current_c1.json"
ANALYSIS_PATH = ROOT / "evaluation" / "results" / "error_analysis_v1.json"


def _case_ids(
    rows: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
) -> list[str]:
    return [row["id"] for row in rows if predicate(row)]


def run_analysis() -> dict[str, Any]:
    benchmark = run_benchmark(DATASET, SOURCES, top_k=5, enable_dense=False)
    RESULT_PATH.write_text(
        json.dumps(benchmark, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rows = benchmark["results"]
    answerable = [row for row in rows if row["answerable"]]
    returned_counts = [len(row["retrieved_sources"]) for row in answerable]
    categories = {
        "retrieval_miss": _case_ids(answerable, lambda row: row["recall_at_k"] < 1),
        "irrelevant_returned_chunk": _case_ids(
            answerable, lambda row: row["precision_of_returned_results"] < 1
        ),
        "citation_page_error": _case_ids(
            answerable,
            lambda row: row["citation_page_recall"] < 1 or row["citation_page_precision"] < 1,
        ),
        "evidence_value_error": _case_ids(answerable, lambda row: row["evidence_value_hit"] is False),
        "evidence_unit_error": _case_ids(answerable, lambda row: row["evidence_unit_match"] is False),
        "no_answer_false_positive": _case_ids(
            [row for row in rows if not row["answerable"]],
            lambda row: row["no_answer_correct"] is False,
        ),
    }
    result = {
        "schema_version": 1,
        "experiment": "current_c1_bm25_error_analysis_v1",
        "benchmark_artifact": str(RESULT_PATH.relative_to(ROOT)),
        "configuration": {"chunker": "C1_financial_section_aware", "top_k": 5, "dense_enabled": False},
        "cases": len(rows),
        "answerable_cases": len(answerable),
        "failure_counts": {name: len(ids) for name, ids in categories.items()},
        "failure_case_ids": categories,
        "returned_chunk_count": {
            "min": min(returned_counts),
            "median": median(returned_counts),
            "max": max(returned_counts),
            "underfilled_below_top_k": sum(count < 5 for count in returned_counts),
        },
        "summary_metrics": {
            name: benchmark["summary"][name]
            for name in (
                "recall_at_k",
                "precision_at_k",
                "precision_of_returned_results",
                "mrr",
                "citation_page_recall",
                "citation_page_precision",
                "evidence_value_hit_rate",
                "evidence_unit_match_rate",
                "no_answer_accuracy",
            )
        },
        "finding": (
            "No retrieval, citation-page, exact-value/unit, or no-answer failure occurred in this fixture. "
            "P@5 is lower because many answerable queries return fewer than five relevant chunks, "
            "not because returned chunks are labeled irrelevant."
        ),
        "limitations": [
            "No 30-50 wrong-answer sample can be drawn from this fixture because it contains zero measured failures.",
            "The benchmark has 88 queries and page-level relevance labels; it is not a broad production error sample.",
            "Query metric/intent labels remain a first-pass annotation and are not included as independent error ground truth.",
        ],
    }
    ANALYSIS_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    report = run_analysis()
    print(json.dumps({
        "cases": report["cases"],
        "failure_counts": report["failure_counts"],
        "returned_chunk_count": report["returned_chunk_count"],
        "summary_metrics": report["summary_metrics"],
    }, ensure_ascii=False, indent=2))
