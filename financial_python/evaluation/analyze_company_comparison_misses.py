"""Break down company-comparison stress misses by evidence match level.

This is descriptive analysis of parser-derived labels, not independent gold
evaluation. It separates exact-value retrieval from metadata/source coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "evaluation" / "results" / "ocr_txt_corpus_comprehensive_final_2026-09-27.json"
DEFAULT_OUTPUT = ROOT / "evaluation" / "review" / "company_comparison_miss_analysis_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_match_levels(target: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, bool]:
    company = target.get("company_code")
    metric = target.get("canonical_metric")
    statement = target.get("statement_type")
    report_type = target.get("report_type")
    year = target.get("fiscal_year")
    expected_value = str(target.get("expected_value", "")).strip()
    source_file = target.get("source_file")

    def same_company(result: dict[str, Any]) -> bool:
        return result.get("company_code") == company

    def same_metric(result: dict[str, Any]) -> bool:
        return same_company(result) and result.get("canonical_metric") == metric

    def has_year(result: dict[str, Any]) -> bool:
        return year in {
            result.get("fiscal_year"),
            result.get("comparative_fiscal_year"),
        }

    def has_target_value(result: dict[str, Any]) -> bool:
        values = result.get("period_values") or {}
        return str(values.get(str(year), "")).strip() == expected_value

    def metadata(result: dict[str, Any]) -> bool:
        return (
            same_metric(result)
            and result.get("statement_type") == statement
            and result.get("report_type") == report_type
            and has_year(result)
        )
    return {
        "company_in_top5": any(same_company(result) for result in results),
        "company_metric_in_top5": any(same_metric(result) for result in results),
        "company_metric_year_report_in_top5": any(metadata(result) for result in results),
        "expected_source_file_in_top5": any(result.get("source_file") == source_file for result in results),
        "exact_expected_value_in_top5": any(metadata(result) and has_target_value(result) for result in results),
    }


def analyze(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    artifact = json.loads(input_path.read_text(encoding="utf-8"))
    misses = [
        case
        for case in artifact["bm25_full_corpus"]["cases"]
        if case.get("kind") == "company_comparison" and not case.get("hit_at_k", False)
    ]
    level_counts: Counter[str] = Counter()
    sides_total = 0
    case_strata: Counter[str] = Counter()
    by_metric: dict[str, Counter[str]] = {}
    examples: dict[str, list[dict[str, Any]]] = {}

    for case in misses:
        matched_count = len(case.get("matched_targets", []))
        case_strata["one_target_in_top5" if matched_count == 1 else "no_targets_in_top5"] += 1
        metric = str(case.get("expected_metric") or "unknown")
        by_metric.setdefault(metric, Counter())[
            "one_target_in_top5" if matched_count == 1 else "no_targets_in_top5"
        ] += 1
        examples.setdefault(metric, [])

        for target in case.get("expected_targets", []):
            sides_total += 1
            levels = _target_match_levels(target, case.get("top_results", []))
            level_counts.update({name: int(value) for name, value in levels.items()})
            if not levels["exact_expected_value_in_top5"] and len(examples[metric]) < 4:
                examples[metric].append(
                    {
                        "case_id": case["case_id"],
                        "query": case["query"],
                        "expected_target": target,
                        "match_levels": levels,
                        "retrieved_top5_evidence": case.get("top_results", []),
                    }
                )

    summary = {
        "schema_version": 1,
        "artifact_type": "descriptive_parser_derived_miss_analysis_not_gold_evaluation",
        "input_artifact": str(input_path.relative_to(ROOT.parent.resolve())),
        "input_sha256": _sha256(input_path),
        "company_comparison_miss_count": len(misses),
        "expected_target_sides_across_misses": sides_total,
        "case_strata": dict(case_strata),
        "case_strata_by_metric": {
            metric: dict(counts) for metric, counts in sorted(by_metric.items())
        },
        "target_side_match_levels": {
            name: {"count": count, "rate": round(count / sides_total, 4) if sides_total else None}
            for name, count in level_counts.items()
        },
        "interpretation_limit": (
            "Expected values and target labels were generated from the same parser/chunker. "
            "These counts locate where exact parser-derived evidence is missing from top-5; "
            "they do not establish whether the parser label or the retriever is wrong."
        ),
        "examples_for_source_review": examples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = analyze(args.input, args.output)
    print(json.dumps({key: value for key, value in summary.items() if key != "examples_for_source_review"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
