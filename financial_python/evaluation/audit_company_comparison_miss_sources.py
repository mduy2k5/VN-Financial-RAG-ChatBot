"""Create a source-backed automated pre-audit for the comparison miss sheet.

This tool checks whether parser-derived expected values appear in source rows,
maps the row labels with the current ontology, and checks top-5 evidence. It is
an aid for review, not human-confirmed ground truth; it never edits the review
sheet or changes a case from ``pending``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


FINANCIAL_PYTHON = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FINANCIAL_PYTHON / "src"))

from financial_rag.metadata.metric_mapper import FinancialMetricMapper  # noqa: E402


DEFAULT_INPUT = FINANCIAL_PYTHON / "evaluation" / "review" / "company_comparison_miss_review_v2.csv"
DEFAULT_JSON = FINANCIAL_PYTHON / "evaluation" / "review" / "company_comparison_miss_source_audit_v2.json"
DEFAULT_CSV = FINANCIAL_PYTHON / "evaluation" / "review" / "company_comparison_miss_source_audit_v2.csv"

OUTPUT_FIELDS = [
    "sample_id",
    "case_id",
    "company_code",
    "fiscal_year",
    "report_type",
    "expected_metric",
    "expected_value",
    "source_file",
    "source_page",
    "source_exists",
    "source_sha256",
    "source_row_labels",
    "row_label_metric_mappings",
    "source_label_status",
    "top5_company_present",
    "top5_expected_metric_year_present",
    "top5_exact_value_present",
    "automated_pre_review_hint",
    "review_status_in_original_sheet",
    "source_row_hashes",
]


def _json_load(value: str) -> Any:
    return json.loads(value) if value else []


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _extract_label(row: str) -> str:
    """Return the first plausible description cell in a flattened HTML row."""
    cells = [re.sub(r"\s+", " ", part).strip() for part in row.split("|")]
    for cell in cells:
        if not cell or len(cell) < 4:
            continue
        if re.fullmatch(r"[A-Za-z]{1,3}[\w.-]*", cell):
            continue
        if re.fullmatch(r"[\d.,()\-+\s]+", cell):
            continue
        if re.search(r"[A-Za-zÀ-ỹ]{2,}", cell):
            return cell
    return ""


def _year_matches(target: dict[str, Any], evidence: dict[str, Any]) -> bool:
    year = target.get("fiscal_year")
    return year in {evidence.get("fiscal_year"), evidence.get("comparative_fiscal_year")}


def _audit_side(
    *,
    sample_id: str,
    case_id: str,
    expected_metric: str,
    target: dict[str, Any],
    raw_source: dict[str, Any],
    top5: list[dict[str, Any]],
    review_status: str,
    mapper: FinancialMetricMapper,
) -> dict[str, Any]:
    rows = raw_source.get("rows_containing_expected_value", [])
    row_details: list[dict[str, Any]] = []
    for row in rows:
        label = _extract_label(str(row))
        mapped = mapper.lookup_metric(label) if label else None
        row_details.append({
            "label": label,
            "canonical_metric": mapped.canonical_name if mapped else None,
            "row_sha256": _sha256_text(str(row)),
        })

    expected_value = str(target.get("value") or "").strip()
    expected_year = target.get("fiscal_year")
    company = str(target.get("company_code") or "")
    target_rows = [item for item in row_details if item["canonical_metric"] == expected_metric]
    different_metric_rows = [
        item for item in row_details
        if item["canonical_metric"] and item["canonical_metric"] != expected_metric
    ]
    if not rows:
        source_status = "expected_value_row_not_found"
    elif target_rows and different_metric_rows:
        source_status = "expected_metric_row_present_with_competing_mapped_rows"
    elif target_rows:
        source_status = "expected_metric_row_present"
    elif different_metric_rows:
        source_status = "value_rows_map_to_other_metrics"
    else:
        source_status = "value_row_label_unmapped_or_ambiguous"

    matching_top5 = [
        item for item in top5
        if item.get("company_code") == company
        and item.get("canonical_metric") == expected_metric
        and item.get("report_type") == target.get("report_type")
        and _year_matches(target, item)
    ]
    exact_top5 = any(
        str((item.get("period_values") or {}).get(str(expected_year), "")).strip() == expected_value
        for item in matching_top5
    )
    company_in_top5 = any(item.get("company_code") == company for item in top5)

    if source_status not in {"expected_metric_row_present", "expected_metric_row_present_with_competing_mapped_rows"}:
        hint = "review_source_label_or_parser_target"
    elif exact_top5:
        hint = "expected_evidence_already_in_top5_check_benchmark_match_logic"
    elif not company_in_top5:
        hint = "likely_company_filter_or_ranking_miss"
    elif not matching_top5:
        hint = "likely_metric_period_report_filter_or_ranking_miss"
    else:
        hint = "likely_value_or_chunk_representation_miss"

    return {
        "sample_id": sample_id,
        "case_id": case_id,
        "company_code": company,
        "fiscal_year": expected_year,
        "report_type": target.get("report_type"),
        "expected_metric": expected_metric,
        "expected_value": expected_value,
        "source_file": raw_source.get("source_file"),
        "source_page": raw_source.get("page"),
        "source_exists": raw_source.get("source_exists"),
        "source_sha256": raw_source.get("source_sha256"),
        "source_row_labels": [item["label"] for item in row_details],
        "row_label_metric_mappings": row_details,
        "source_label_status": source_status,
        "top5_company_present": company_in_top5,
        "top5_expected_metric_year_present": bool(matching_top5),
        "top5_exact_value_present": exact_top5,
        "automated_pre_review_hint": hint,
        "review_status_in_original_sheet": review_status,
    }


def audit(input_csv: Path = DEFAULT_INPUT, output_json: Path = DEFAULT_JSON, output_csv: Path = DEFAULT_CSV) -> dict[str, Any]:
    mapper = FinancialMetricMapper()
    sides: list[dict[str, Any]] = []
    case_summary: list[dict[str, Any]] = []

    with input_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            targets = _json_load(row.get("parser_expected_evidence", ""))
            raw_sources = _json_load(row.get("raw_source_evidence_candidates", ""))
            top5 = _json_load(row.get("retrieved_top5_evidence", ""))
            raw_by_file = {item.get("source_file"): item for item in raw_sources}
            case_sides: list[dict[str, Any]] = []
            for target in targets:
                source = raw_by_file.get(target.get("source_file"), {})
                item = _audit_side(
                    sample_id=row.get("sample_id", ""),
                    case_id=row.get("case_id", ""),
                    expected_metric=str(target.get("canonical_metric") or row.get("expected_metric") or ""),
                    target=target,
                    raw_source=source,
                    top5=top5,
                    review_status=row.get("review_status", ""),
                    mapper=mapper,
                )
                sides.append(item)
                case_sides.append(item)

            source_clear = all(
                item["source_label_status"] in {
                    "expected_metric_row_present",
                    "expected_metric_row_present_with_competing_mapped_rows",
                }
                for item in case_sides
            ) and len(case_sides) == 2
            case_summary.append({
                "sample_id": row.get("sample_id", ""),
                "case_id": row.get("case_id", ""),
                "expected_metric": row.get("expected_metric", ""),
                "source_metric_rows_supported_both_sides": source_clear,
                "top5_exact_expected_sides": sum(bool(item["top5_exact_value_present"]) for item in case_sides),
                "automated_pre_review_hint": (
                    "likely_retrieval_miss_after_source_support"
                    if source_clear and any(not item["top5_exact_value_present"] for item in case_sides)
                    else "source_or_label_adjudication_needed"
                    if not source_clear
                    else "benchmark_match_logic_needs_review"
                ),
                "review_status_in_original_sheet": row.get("review_status", ""),
            })

    source_counts = Counter(item["source_label_status"] for item in sides)
    hint_counts = Counter(item["automated_pre_review_hint"] for item in sides)
    mapping_counts = Counter(
        item["canonical_metric"] or "UNMAPPED"
        for side in sides for item in side["row_label_metric_mappings"]
    )
    summary = {
        "schema_version": 1,
        "artifact_type": "automated_source_pre_audit_not_human_verified_gold",
        "input_csv": str(input_csv.resolve()),
        "input_sha256": hashlib.sha256(input_csv.read_bytes()).hexdigest(),
        "case_count": len(case_summary),
        "target_side_count": len(sides),
        "review_status_counts_unchanged": dict(Counter(item["review_status_in_original_sheet"] for item in sides)),
        "source_label_status_counts": dict(source_counts),
        "pre_review_hint_counts": dict(hint_counts),
        "row_label_metric_mapping_counts": dict(mapping_counts),
        "cases_with_source_metric_rows_supported_both_sides": sum(
            item["source_metric_rows_supported_both_sides"] for item in case_summary
        ),
        "cases_with_both_exact_expected_values_in_top5": sum(
            item["top5_exact_expected_sides"] == 2 for item in case_summary
        ),
        "interpretation_limit": (
            "Source rows are matched using parser-derived expected values and the current metric ontology. "
            "Repeated values, nested subtotals, OCR errors, statement semantics, and label correctness require "
            "human confirmation. No review status or benchmark label was changed."
        ),
        "cases": case_summary,
        "target_sides": sides,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with output_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for item in sides:
            row = {key: item.get(key) for key in OUTPUT_FIELDS}
            for field in ("source_row_labels", "row_label_metric_mappings", "source_row_hashes"):
                if field == "source_row_hashes":
                    row[field] = json.dumps(
                        [mapping["row_sha256"] for mapping in item["row_label_metric_mappings"]],
                        ensure_ascii=False,
                    )
                else:
                    row[field] = json.dumps(row[field], ensure_ascii=False)
            writer.writerow(row)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--json", dest="output_json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv", dest="output_csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    result = audit(args.input, args.output_json, args.output_csv)
    print(json.dumps({key: value for key, value in result.items() if key not in {"cases", "target_sides"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
