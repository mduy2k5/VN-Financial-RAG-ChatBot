"""Run and save a top-k sensitivity report for the multi-company benchmark."""

import json
from pathlib import Path
from typing import Any

from financial_rag.evaluation.benchmark import run_benchmark
from financial_rag.evaluation.local_data import local_data_path


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = local_data_path("evaluation", "ground_truth_multi_company.jsonl")
SOURCE_PATHS = [
    local_data_path("sample_data", "ABB_BCTC_2025.md"),
    local_data_path("evaluation", "sources", "FPT_Annual_Report_2024_benchmark.md"),
    local_data_path("evaluation", "sources", "FPT_Annual_Report_2025_benchmark.md"),
    local_data_path("evaluation", "sources", "HPG_Annual_Report_2025_benchmark.md"),
    local_data_path("evaluation", "sources", "FPT_Q2_2026_benchmark.md"),
    local_data_path("evaluation", "sources", "HPG_Q2_2025_benchmark.md"),
    local_data_path("evaluation", "sources", "VNF_Interim_Consolidated_2025H1_benchmark.md"),
]
TOP_K_VALUES = (1, 2, 3, 5, 8)
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


def run_top_k_sensitivity() -> dict[str, Any]:
    dataset_path = DATASET_PATH
    source_paths = SOURCE_PATHS
    runs = []
    provenance = None

    for top_k in TOP_K_VALUES:
        summary = run_benchmark(dataset_path, source_paths, top_k=top_k)["summary"]
        if provenance is None:
            provenance = {
                "dataset_sha256": summary["dataset_sha256"],
                "source_sha256": [
                    {"source": str(path), "sha256": item["sha256"]}
                    for path, item in zip(source_paths, summary["source_sha256"])
                ],
            }
        runs.append({
            "top_k": top_k,
            "dev": {metric: summary["by_split"]["dev"].get(metric) for metric in METRICS},
            "holdout": {metric: summary["by_split"]["holdout"].get(metric) for metric in METRICS},
            "document_disjoint": {
                metric: summary["by_source_partition"]["document_disjoint"].get(metric)
                for metric in METRICS
            },
        })

    if provenance is None:
        raise RuntimeError("No top-k values are configured")

    return {
        "dataset": str(DATASET_PATH),
        "dataset_sha256": provenance["dataset_sha256"],
        "sources": [str(path) for path in SOURCE_PATHS],
        "source_sha256": provenance["source_sha256"],
        "dense_enabled": False,
        "top_k_values": list(TOP_K_VALUES),
        "runs": runs,
    }


def main() -> None:
    result = run_top_k_sensitivity()
    output_path = ROOT / "evaluation" / "results" / "multi_company_top_k_sensitivity.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
