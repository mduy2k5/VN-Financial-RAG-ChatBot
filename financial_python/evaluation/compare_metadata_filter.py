"""Run one controlled metadata-filter ablation over the Markdown fixture."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

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
OUTPUT = ROOT / "evaluation" / "results" / "metadata_filter_ablation_c1_v1.json"
METRICS = (
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


def _timed_run(*, apply_metadata_filters: bool) -> tuple[dict[str, Any], float]:
    start = perf_counter()
    result = run_benchmark(
        DATASET,
        SOURCES,
        top_k=5,
        enable_dense=False,
        apply_metadata_filters=apply_metadata_filters,
    )
    return result, perf_counter() - start


def run_comparison() -> dict[str, Any]:
    filtered, filtered_seconds = _timed_run(apply_metadata_filters=True)
    unfiltered, unfiltered_seconds = _timed_run(apply_metadata_filters=False)
    filtered_rows = {row["id"]: row for row in filtered["results"]}
    unfiltered_rows = {row["id"]: row for row in unfiltered["results"]}
    changed_cases = [
        {
            "id": case_id,
            "with_filters_first_source": (
                filtered_rows[case_id]["retrieved_sources"][:1] or None
            ),
            "without_filters_first_source": (
                unfiltered_rows[case_id]["retrieved_sources"][:1] or None
            ),
            "with_filters_result_count": len(filtered_rows[case_id]["retrieved_sources"]),
            "without_filters_result_count": len(unfiltered_rows[case_id]["retrieved_sources"]),
        }
        for case_id in filtered_rows
        if filtered_rows[case_id]["retrieved_sources"]
        != unfiltered_rows[case_id]["retrieved_sources"]
    ]
    metric_deltas = {
        metric: round(
            unfiltered["summary"][metric] - filtered["summary"][metric],
            4,
        )
        for metric in METRICS
        if metric in filtered["summary"] and metric in unfiltered["summary"]
    }
    comparison = {
        "schema_version": 1,
        "experiment": "metadata_filter_ablation",
        "scope": "Markdown fixture; Python retriever company/year/quarter/report/statement filters only",
        "fixed_configuration": {
            "chunker": "C1_financial_section_aware",
            "top_k": 5,
            "dense_enabled": False,
        },
        "runs": {
            "with_metadata_filters": {
                "elapsed_seconds_single_run": round(filtered_seconds, 4),
                "summary": filtered["summary"],
            },
            "without_metadata_filters": {
                "elapsed_seconds_single_run": round(unfiltered_seconds, 4),
                "summary": unfiltered["summary"],
            },
        },
        "delta_without_minus_with": metric_deltas,
        "changed_case_count": len(changed_cases),
        "changed_cases": changed_cases,
        "timing_note": "One run per configuration; elapsed time is descriptive only, not a stable performance benchmark.",
        "limitations": [
            "This ablation measures the Python retriever on the existing seven-file Markdown fixture.",
            "This ablation measures the standalone Python retrieval filter; it does not cover durable storage or interactive chat.",
            "Page-level relevance labels and small fixture limit generalization.",
        ],
    }
    OUTPUT.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return comparison


if __name__ == "__main__":
    report = run_comparison()

    def compact_metrics(summary: dict[str, Any]) -> dict[str, float]:
        return {metric: summary[metric] for metric in METRICS if metric in summary}

    print(json.dumps({
        "with_filters_seconds": report["runs"]["with_metadata_filters"]["elapsed_seconds_single_run"],
        "without_filters_seconds": report["runs"]["without_metadata_filters"]["elapsed_seconds_single_run"],
        "with_filters": compact_metrics(report["runs"]["with_metadata_filters"]["summary"]),
        "without_filters": compact_metrics(report["runs"]["without_metadata_filters"]["summary"]),
        "delta_without_minus_with": report["delta_without_minus_with"],
        "changed_case_count": report["changed_case_count"],
    }, ensure_ascii=False, indent=2))
