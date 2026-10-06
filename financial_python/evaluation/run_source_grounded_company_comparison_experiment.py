"""Re-score the fixed full-corpus comparison benchmark against source totals.

Expected values in the original stress benchmark were parser-derived. This
experiment preserves its queries and retrieved top-5 lists, but re-anchors each
target to a direct statement total or a tax total derived from current + deferred
tax lines. It does not claim independent human gold labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FINANCIAL_PYTHON = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FINANCIAL_PYTHON / "evaluation"))

from ground_company_comparison_miss_targets import (  # noqa: E402
    DATA_DIR,
    _parse_rows,
    _side_candidates,
    _source_sha,
)


DEFAULT_INPUT = FINANCIAL_PYTHON / "evaluation" / "results" / "ocr_txt_corpus_comprehensive_final_2026-09-27.json"
DEFAULT_JSON = FINANCIAL_PYTHON / "evaluation" / "results" / "company_comparison_source_grounded_bm25_v1.json"
DEFAULT_CSV = FINANCIAL_PYTHON / "evaluation" / "results" / "company_comparison_source_grounded_bm25_v1.csv"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_metric[str(case.get("expected_metric") or "unknown")].append(case)

    result: dict[str, Any] = {}
    for metric, metric_cases in sorted(by_metric.items()):
        groundable = [case for case in metric_cases if case["both_targets_groundable"]]
        original_hits_on_groundable = sum(bool(case["original_hit_at_k"]) for case in groundable)
        signed_hits = sum(bool(case["source_grounded_hit_at_k_signed"]) for case in groundable)
        magnitude_hits = sum(bool(case["source_grounded_hit_at_k_magnitude"]) for case in groundable)
        source_page_hits = sum(bool(case["source_grounded_source_page_hit_at_k_signed"]) for case in groundable)
        result[metric] = {
            "case_count": len(metric_cases),
            "parser_derived_baseline_hit_count": sum(bool(case["original_hit_at_k"]) for case in metric_cases),
            "both_targets_groundable_count": len(groundable),
            "parser_derived_baseline_hit_count_on_groundable_subset": original_hits_on_groundable,
            "parser_derived_baseline_hit_rate_on_groundable_subset": round(original_hits_on_groundable / len(groundable), 4) if groundable else None,
            "source_reanchored_signed_hit_count": signed_hits,
            "source_reanchored_magnitude_hit_count": magnitude_hits,
            "source_file_page_value_hit_count": source_page_hits,
            "source_file_page_value_hit_rate": round(source_page_hits / len(groundable), 4) if groundable else None,
            "source_reanchored_signed_hit_rate": round(
                signed_hits / len(groundable), 4
            ) if groundable else None,
            "source_reanchored_magnitude_hit_rate": round(
                magnitude_hits / len(groundable), 4
            ) if groundable else None,
            "signed_delta_vs_parser_baseline_on_groundable_subset": round((signed_hits - original_hits_on_groundable) / len(groundable), 4) if groundable else None,
            "case_counts_by_target_source_status": dict(Counter(
                status for case in metric_cases for status in case["target_candidate_statuses"]
            )),
        }
    return result


def run(input_path: Path = DEFAULT_INPUT, output_json: Path = DEFAULT_JSON, output_csv: Path = DEFAULT_CSV) -> dict[str, Any]:
    input_path = input_path.resolve()
    artifact = json.loads(input_path.read_text(encoding="utf-8"))
    corpus_cases = artifact["bm25_full_corpus"]["cases"]
    comparison_cases = [case for case in corpus_cases if case.get("kind") == "company_comparison"]
    row_cache: dict[str, list[dict[str, Any]]] = {}
    source_hashes: dict[str, str] = {}
    evaluated: list[dict[str, Any]] = []
    target_sides: list[dict[str, Any]] = []

    for case in comparison_cases:
        top5 = case.get("top_results", [])
        side_results: list[dict[str, Any]] = []
        for target in case.get("expected_targets", []):
            source_file = str(target.get("source_file") or "")
            source_path = DATA_DIR / source_file
            if not source_path.is_file():
                result = {
                    "candidate_status": "source_file_missing",
                    "grounded_candidate_value": None,
                    "top5_has_grounded_value": False,
                    "top5_has_grounded_magnitude": False,
                    "selected_candidate": None,
                    "candidate_rows_with_period_values": [],
                }
            else:
                row_cache.setdefault(source_file, _parse_rows(source_path))
                source_hashes.setdefault(source_file, _source_sha(source_path))
                result = _side_candidates(
                    target,
                    source_file,
                    str(target.get("expected_value") or ""),
                    int(target["page"]) if target.get("page") is not None else None,
                    top5,
                    row_cache[source_file],
                )

            selected = result.get("selected_candidate") or {}
            side = {
                "case_id": case["case_id"],
                "query": case["query"],
                "expected_metric": case.get("expected_metric"),
                "company_code": target.get("company_code"),
                "fiscal_year": target.get("fiscal_year"),
                "report_type": target.get("report_type"),
                "statement_type": target.get("statement_type"),
                "parser_expected_value": target.get("expected_value"),
                "grounded_candidate_value": result.get("grounded_candidate_value"),
                "candidate_status": result.get("candidate_status"),
                "source_file": source_file,
                "original_target_page": target.get("page"),
                "grounded_source_page": selected.get("page"),
                "grounded_row_label": selected.get("label"),
                "grounded_row_text": selected.get("row_text"),
                "grounded_row_code": selected.get("code"),
                "source_sha256": source_hashes.get(source_file),
                "tax_component_derivation": result.get("tax_component_derivation"),
                "top5_has_grounded_value_signed": bool(result.get("top5_has_grounded_value")),
                "top5_has_grounded_value_magnitude": bool(result.get("top5_has_grounded_magnitude")),
                "top5_has_grounded_source_page_value": bool(result.get("top5_has_grounded_source_page_value")),
                "top5_grounded_value_match_count": result.get("top5_grounded_value_match_count", 0),
                "top5_grounded_magnitude_match_count": result.get("top5_grounded_magnitude_match_count", 0),
                "top5_grounded_source_page_match_count": result.get("top5_grounded_source_page_match_count", 0),
            }
            side_results.append(side)
            target_sides.append(side)

        both_groundable = len(side_results) == 2 and all(item.get("grounded_candidate_value") for item in side_results)
        evaluated.append({
            "case_id": case["case_id"],
            "kind": case.get("kind"),
            "query": case["query"],
            "expected_metric": case.get("expected_metric"),
            "expected_companies": case.get("expected_companies", []),
            "expected_years": case.get("expected_years", []),
            "expected_report_type": case.get("expected_report_type"),
            "query_parse_exact": case.get("query_parse_exact"),
            "original_hit_at_k": bool(case.get("hit_at_k")),
            "both_targets_groundable": both_groundable,
            "source_grounded_hit_at_k_signed": both_groundable and all(item["top5_has_grounded_value_signed"] for item in side_results),
            "source_grounded_hit_at_k_magnitude": both_groundable and all(item["top5_has_grounded_value_magnitude"] for item in side_results),
            "source_grounded_source_page_hit_at_k_signed": both_groundable and all(item["top5_has_grounded_source_page_value"] for item in side_results),
            "target_candidate_statuses": [item["candidate_status"] for item in side_results],
            "target_parser_values": [item["parser_expected_value"] for item in side_results],
            "target_source_grounded_values": [item["grounded_candidate_value"] for item in side_results],
            "target_source_files": [item["source_file"] for item in side_results],
            "target_source_pages": [item["grounded_source_page"] for item in side_results],
        })

    groundable = [case for case in evaluated if case["both_targets_groundable"]]
    parser_hits_on_groundable = sum(bool(case["original_hit_at_k"]) for case in groundable)
    signed_hits = sum(bool(case["source_grounded_hit_at_k_signed"]) for case in groundable)
    magnitude_hits = sum(bool(case["source_grounded_hit_at_k_magnitude"]) for case in groundable)
    source_page_hits = sum(bool(case["source_grounded_source_page_hit_at_k_signed"]) for case in groundable)
    summary = {
        "schema_version": 1,
        "artifact_type": "fixed_query_source_row_reanchored_retrieval_evaluation_not_human_gold",
        "input_artifact": str(input_path),
        "input_sha256": _sha256(input_path),
        "source_document_count": len(source_hashes),
        "source_document_hashes": source_hashes,
        "company_comparison_case_count": len(evaluated),
        "target_side_count": len(target_sides),
        "groundable_case_count": len(groundable),
        "unresolved_case_count": len(evaluated) - len(groundable),
        "parser_derived_original_hit_count": sum(case["original_hit_at_k"] for case in evaluated),
        "parser_derived_original_hit_rate": round(
            sum(case["original_hit_at_k"] for case in evaluated) / len(evaluated), 4
        ) if evaluated else None,
        "parser_derived_baseline_hit_count_on_groundable_subset": parser_hits_on_groundable,
        "parser_derived_baseline_hit_rate_on_groundable_subset": round(parser_hits_on_groundable / len(groundable), 4) if groundable else None,
        "source_reanchored_signed_hit_count": signed_hits,
        "source_reanchored_signed_hit_rate": round(
            signed_hits / len(groundable), 4
        ) if groundable else None,
        "source_reanchored_magnitude_hit_count": magnitude_hits,
        "source_reanchored_magnitude_hit_rate": round(
            magnitude_hits / len(groundable), 4
        ) if groundable else None,
        "source_file_page_value_hit_count": source_page_hits,
        "source_file_page_value_hit_rate": round(source_page_hits / len(groundable), 4) if groundable else None,
        "signed_delta_vs_parser_baseline_on_groundable_subset": round((signed_hits - parser_hits_on_groundable) / len(groundable), 4) if groundable else None,
        "query_parse_exact_rate": round(sum(bool(case["query_parse_exact"]) for case in evaluated) / len(evaluated), 4) if evaluated else None,
        "by_metric": _metric_summary(evaluated),
        "method": (
            "Keep every original comparison query and its original top-5 BM25 results. Re-anchor each expected value "
            "to source statement labels/codes: capital 411, receivables 130, interest expense 23, operating/financing "
            "cash flows 20/40, retained earnings 421, net income 60, balance-sheet cash 110, cash-flow beginning/end "
            "cash 60/70, and pre-tax profit 50/90. Tax uses explicit total rows/code 100 or derives current + deferred "
            "tax from income-statement codes 51 + 52. Select current/comparative amount from report year in source filename."
        ),
        "interpretation_limit": (
            "This corrects parser-derived value labels using source statement rows but does not replace expert/human "
            "adjudication. Tax values preserve statement sign and are also evaluated by magnitude because source notes "
            "sometimes present expense magnitude without parentheses. Unresolved sources are excluded from the grounded denominator."
        ),
        "cases": evaluated,
        "target_sides": target_sides,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with output_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = [
            "case_id", "query", "expected_metric", "expected_companies", "expected_years",
            "original_hit_at_k", "both_targets_groundable", "source_grounded_hit_at_k_signed",
            "source_grounded_hit_at_k_magnitude", "source_grounded_source_page_hit_at_k_signed",
            "target_candidate_statuses", "target_parser_values",
            "target_source_grounded_values", "target_source_files", "target_source_pages",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in evaluated:
            writer.writerow({key: json.dumps(case[key], ensure_ascii=False) if isinstance(case[key], (list, dict)) else case[key] for key in fields})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--json", dest="output_json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv", dest="output_csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    result = run(args.input, args.output_json, args.output_csv)
    print(json.dumps({key: value for key, value in result.items() if key not in {"cases", "target_sides", "source_document_hashes", "by_metric"}}, ensure_ascii=False, indent=2))
    print(json.dumps(result["by_metric"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
