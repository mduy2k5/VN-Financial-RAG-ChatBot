"""Create a deterministic, stratified review sheet from TXT stress-test misses.

The TXT stress test is parser-derived, so the generated expected values are
review candidates only. A reviewer must check them against the source files
before treating any row as ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from html import unescape
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "evaluation" / "results" / "ocr_txt_corpus_comprehensive_final_2026-09-27.json"
DEFAULT_OUTPUT = ROOT / "evaluation" / "review" / "company_comparison_miss_review_v1.csv"
DEFAULT_SUMMARY = ROOT / "evaluation" / "review" / "company_comparison_miss_review_v1.json"
DEFAULT_SEED = "company-comparison-miss-review-v1"
TOP_METRIC_COUNT = 3
CASES_PER_METRIC_AND_STRATUM = 20

FIELDS = [
    "sample_id",
    "case_id",
    "stratum_metric",
    "stratum_retrieval_status",
    "query",
    "expected_companies",
    "expected_years",
    "expected_report_type",
    "expected_metric",
    "parser_expected_evidence",
    "raw_source_evidence_candidates",
    "retrieved_top5_evidence",
    "matched_target_count",
    "review_status",
    "reviewer",
    "verified_metric",
    "verified_company_year_report_values",
    "verified_pages",
    "root_cause",
    "review_notes",
    "reviewed_at",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_key(case: dict[str, Any], seed: str) -> str:
    return hashlib.sha256(f"{seed}:{case['case_id']}".encode("utf-8")).hexdigest()


def _pair_key(case: dict[str, Any]) -> str:
    companies = sorted(str(value) for value in case.get("expected_companies", []))
    return "-".join(companies) or "unknown-company-pair"


def _stratified_round_robin(
    cases: list[dict[str, Any]], limit: int, seed: str
) -> list[dict[str, Any]]:
    """Balance the sample across company pairs before taking more from a pair."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        key = "|".join(
            (
                _pair_key(case),
                str(case.get("expected_report_type") or "unknown-report-type"),
            )
        )
        groups[key].append(case)
    for group in groups.values():
        group.sort(key=lambda case: _stable_key(case, seed))

    selected: list[dict[str, Any]] = []
    group_keys = sorted(groups, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())
    depth = 0
    while len(selected) < limit:
        appended = False
        for group_key in group_keys:
            group = groups[group_key]
            if depth < len(group):
                selected.append(group[depth])
                appended = True
                if len(selected) == limit:
                    break
        if not appended:
            break
        depth += 1
    return selected


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _source_evidence_candidates(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    data_dir = ROOT.parent / "dataset" / "ocr_annual_financials"
    evidence: list[dict[str, Any]] = []
    for target in targets:
        source_file = str(target.get("source_file") or "")
        source_path = data_dir / source_file
        item: dict[str, Any] = {
            "source_file": source_file,
            "page": target.get("page"),
            "expected_value_to_check": target.get("expected_value"),
            "source_exists": source_path.is_file(),
            "source_sha256": _sha256(source_path) if source_path.is_file() else None,
            "rows_containing_expected_value": [],
        }
        if not source_path.is_file():
            evidence.append(item)
            continue

        text = source_path.read_text(encoding="utf-8", errors="replace")
        page_number = target.get("page")
        page_match = re.search(
            rf"(?m)^\s*=+\s*PAGE\s+{int(page_number)}\s*=+\s*$",
            text,
            flags=re.IGNORECASE,
        ) if page_number is not None else None
        if page_match:
            next_page = re.search(r"(?m)^\s*=+\s*PAGE\s+\d+\s*=+\s*$", text[page_match.end():], flags=re.IGNORECASE)
            page_text = text[page_match.end():page_match.end() + next_page.start()] if next_page else text[page_match.end():]
            expected_value = str(target.get("expected_value") or "")
            rows = re.findall(r"<tr\b[^>]*>.*?</tr>", page_text, flags=re.IGNORECASE | re.DOTALL)
            for row in rows:
                if expected_value and expected_value in unescape(row):
                    rendered = re.sub(r"</t[dh]\s*>", " | ", row, flags=re.IGNORECASE)
                    rendered = re.sub(r"<[^>]+>", " ", rendered)
                    rendered = re.sub(r"\s+", " ", unescape(rendered)).strip(" |")
                    item["rows_containing_expected_value"].append(rendered[:1200])
                    if len(item["rows_containing_expected_value"]) >= 4:
                        break
        evidence.append(item)
    return evidence


def _row(case: dict[str, Any], metric: str, stratum: str) -> dict[str, str]:
    expected_evidence = [
        {
            "company_code": target.get("company_code"),
            "fiscal_year": target.get("fiscal_year"),
            "report_type": target.get("report_type"),
            "canonical_metric": target.get("canonical_metric"),
            "value": target.get("expected_value"),
            "period_values": target.get("expected_period_values", {}),
            "unit": target.get("unit"),
            "source_file": target.get("source_file"),
            "page": target.get("page"),
        }
        for target in case.get("expected_targets", [])
    ]
    retrieved = [
        {
            "rank": rank,
            "company_code": result.get("company_code"),
            "fiscal_year": result.get("fiscal_year"),
            "report_type": result.get("report_type"),
            "canonical_metric": result.get("canonical_metric"),
            "period_values": result.get("period_values", {}),
            "unit": result.get("unit"),
            "source_file": result.get("source_file"),
            "page": result.get("page"),
            "chunk_id": result.get("chunk_id"),
            "chunk_type": result.get("chunk_type"),
        }
        for rank, result in enumerate(case.get("top_results", []), start=1)
    ]
    return {
        "sample_id": "",
        "case_id": str(case["case_id"]),
        "stratum_metric": metric,
        "stratum_retrieval_status": stratum,
        "query": str(case.get("query", "")),
        "expected_companies": _json(case.get("expected_companies", [])),
        "expected_years": _json(case.get("expected_years", [])),
        "expected_report_type": str(case.get("expected_report_type") or ""),
        "expected_metric": str(case.get("expected_metric") or ""),
        "parser_expected_evidence": _json(expected_evidence),
        "raw_source_evidence_candidates": _json(
            _source_evidence_candidates(case.get("expected_targets", []))
        ),
        "retrieved_top5_evidence": _json(retrieved),
        "matched_target_count": str(len(case.get("matched_targets", []))),
        "review_status": "pending",
        "reviewer": "",
        "verified_metric": "",
        "verified_company_year_report_values": "",
        "verified_pages": "",
        "root_cause": "",
        "review_notes": "",
        "reviewed_at": "",
    }


def prepare_review(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY,
    seed: str = DEFAULT_SEED,
) -> dict[str, Any]:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    summary_path = summary_path.resolve()
    artifact = json.loads(input_path.read_text(encoding="utf-8"))
    cases = artifact["bm25_full_corpus"]["cases"]
    misses = [
        case
        for case in cases
        if case.get("kind") == "company_comparison" and not case.get("hit_at_k", False)
    ]
    metric_counts = Counter(case.get("expected_metric", "") for case in misses)
    top_metrics = [metric for metric, _ in metric_counts.most_common(TOP_METRIC_COUNT)]

    sampled: list[tuple[dict[str, Any], str, str]] = []
    strata_counts: dict[str, dict[str, int]] = {}
    for metric in top_metrics:
        metric_cases = [case for case in misses if case.get("expected_metric") == metric]
        outcomes = {
            "one_target_in_top5": [case for case in metric_cases if len(case.get("matched_targets", [])) == 1],
            "no_targets_in_top5": [case for case in metric_cases if len(case.get("matched_targets", [])) == 0],
        }
        strata_counts[metric] = {}
        for outcome, outcome_cases in outcomes.items():
            if len(outcome_cases) < CASES_PER_METRIC_AND_STRATUM:
                raise ValueError(
                    f"Insufficient cases for {metric}/{outcome}: {len(outcome_cases)}"
                )
            selected = _stratified_round_robin(
                outcome_cases,
                CASES_PER_METRIC_AND_STRATUM,
                f"{seed}:{metric}:{outcome}",
            )
            strata_counts[metric][outcome] = len(selected)
            sampled.extend((case, metric, outcome) for case in selected)

    rows = [_row(case, metric, outcome) for case, metric, outcome in sampled]
    for index, row in enumerate(rows, start=1):
        row["sample_id"] = f"CR-{index:03d}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema_version": 1,
        "artifact_type": "parser_derived_miss_review_candidates_not_gold_labels",
        "input_artifact": str(input_path.relative_to(ROOT.parent.resolve())),
        "input_sha256": _sha256(input_path),
        "source_file_count": artifact.get("scope", {}).get("source_files"),
        "input_case_count": len(cases),
        "company_comparison_miss_count": len(misses),
        "company_comparison_miss_count_by_metric": dict(metric_counts.most_common()),
        "sample_size": len(rows),
        "sample_size_per_metric_per_stratum": CASES_PER_METRIC_AND_STRATUM,
        "selected_metrics": top_metrics,
        "selected_strata": strata_counts,
        "sampling_seed": seed,
        "sampled_case_ids": [row["case_id"] for row in rows],
        "review_csv": str(output_path.relative_to(ROOT.parent.resolve())),
        "review_csv_sha256": _sha256(output_path),
        "annotation_state": "all rows pending source verification; generated labels are not independent ground truth",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args()
    print(json.dumps(prepare_review(args.input, args.output, args.summary, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
