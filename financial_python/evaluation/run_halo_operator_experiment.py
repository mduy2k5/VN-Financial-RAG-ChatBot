"""Ablate Halo-inspired query operators on the labeled TXT/Markdown corpus."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from financial_rag.chunking.financial_chunker import FinancialMarkdownChunker
from financial_rag.evaluation.benchmark import run_benchmark
from financial_rag.evaluation.local_data import local_data_path


MODES = ("baseline", "structural", "filter", "validate", "full")
SCENARIOS = {
    "natural_query": "",
    "explicit_structure_and_exclusion": "; ưu tiên bảng tài chính, bỏ qua mục lục",
    "explicit_validation": "; chỉ sử dụng báo cáo tài chính, đừng nhầm với mục lục",
}
COMPARE_METRICS = (
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


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _delta(summary: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for metric in COMPARE_METRICS:
        current = summary.get(metric)
        base = baseline.get(metric)
        result[metric] = (
            round(float(current) - float(base), 4)
            if isinstance(current, (int, float)) and isinstance(base, (int, float))
            else None
        )
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Compare baseline retrieval with Structural, Filter, and Validate ablations"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "evaluation" / "results" / "halo_operator_ablation.json",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense", action="store_true", help="Enable local E5 dense retrieval")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    sources = [
        local_data_path("sample_data", "ABB_BCTC_2025.md"),
        local_data_path("evaluation", "sources", "FPT_Annual_Report_2024_benchmark.md"),
        local_data_path("evaluation", "sources", "FPT_Annual_Report_2025_benchmark.md"),
        local_data_path("evaluation", "sources", "HPG_Annual_Report_2025_benchmark.md"),
        local_data_path("evaluation", "sources", "FPT_Q2_2026_benchmark.md"),
        local_data_path("evaluation", "sources", "HPG_Q2_2025_benchmark.md"),
        local_data_path("evaluation", "sources", "VNF_Interim_Consolidated_2025H1_benchmark.md"),
    ]
    missing = [path for path in sources if not path.exists()]
    if missing:
        parser.error("Missing benchmark sources: " + ", ".join(str(path) for path in missing))
    dataset = local_data_path("evaluation", "ground_truth_multi_company.jsonl")
    if not dataset.exists():
        parser.error(f"Missing benchmark dataset: {dataset}")

    chunker = FinancialMarkdownChunker()
    chunks = []
    for path in sources:
        chunks.extend(
            chunker.chunk_file(
                path,
                company_code=path.name.split("_", maxsplit=1)[0].upper(),
                source_pdf=path.name.replace("_benchmark.md", ".pdf"),
            )
        )

    runs: dict[str, dict[str, dict[str, Any]]] = {}
    for scenario, suffix in SCENARIOS.items():
        scenario_runs: dict[str, dict[str, Any]] = {}
        for mode in MODES:
            started = time.perf_counter()
            result = run_benchmark(
                dataset,
                sources,
                top_k=args.top_k,
                enable_dense=args.dense,
                apply_metadata_filters=True,
                directive_mode=mode,
                strict_metric_scope=True,
                directive_suffix=suffix,
                financial_chunker=chunker,
                chunks_override=chunks,
            )
            result["run_seconds"] = round(time.perf_counter() - started, 4)
            scenario_runs[mode] = result

        baseline_summary = scenario_runs["baseline"]["summary"]
        for result in scenario_runs.values():
            result["delta_vs_baseline"] = _delta(result["summary"], baseline_summary)
        runs[scenario] = scenario_runs

    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "experiment": "halo_query_operator_ablation",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "dataset_sha256": _hash(dataset),
        "sources": [
            {"path": str(path), "sha256": _hash(path)}
            for path in sources
        ],
        "top_k": args.top_k,
        "dense_enabled": args.dense,
        "chunk_count": len(chunks),
        "llm_calls": 0,
        "notes": [
            "Filter and Validate operate on an over-fetched candidate set, then return top_k.",
            "Full-corpus runtime fallback is evaluated by API-level validation, not simulated by this retrieval-only benchmark.",
            "Source-grounded metrics inherit the page-level labels and limitations of the input benchmark.",
            "The explicit_structure_and_exclusion scenario appends only 'prefer financial tables, ignore table of contents' to each query; answer labels and retrieval query stay unchanged.",
            "The explicit_validation scenario adds positive and negative evidence constraints to measure Validate reranking; answer labels and retrieval query stay unchanged.",
        ],
        "runs": runs,
    }
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "summaries": {
            scenario: {
                mode: {
                    "recall_at_k": result["summary"].get("recall_at_k"),
                    "mrr": result["summary"].get("mrr"),
                    "evidence_value_hit_rate": result["summary"].get("evidence_value_hit_rate"),
                    "no_answer_accuracy": result["summary"].get("no_answer_accuracy"),
                    "directive_activity": result["summary"].get("directive_activity"),
                    "delta_vs_baseline": result["delta_vs_baseline"],
                    "run_seconds": result["run_seconds"],
                }
                for mode, result in scenario_runs.items()
            }
            for scenario, scenario_runs in runs.items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
