"""Stress-test TXT financial chunking and retrieval on a local report corpus.

The generated cases are parser-derived retrieval stress cases, not independent
human ground truth. They are useful for scale/coverage/regression checks and
must be read alongside the manually labeled Markdown benchmark.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import statistics
import sys
import time
import unicodedata
from typing import Any

from financial_rag.chunking.txt_financial_chunker import TxtFinancialChunker
from financial_rag.embedding.e5_adapter import E5EmbeddingAdapter
from financial_rag.metadata.metric_mapper import FinancialMetricMapper
from financial_rag.query.understanding import QueryUnderstandingEngine
from financial_rag.retrieval.hybrid import FinancialHybridRetriever
from financial_rag.schemas import ChunkType, FinancialChunk, StatementType


STATEMENTS_TO_COVER = (
    StatementType.BALANCE_SHEET,
    StatementType.INCOME_STATEMENT,
    StatementType.CASH_FLOW,
)
MAX_METRICS_PER_STATEMENT = 4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _filename_metadata(path: Path) -> dict[str, Any]:
    name = path.name
    company_match = re.match(r"(?P<company>[A-Z0-9]{3,5})(?=[_.\s])", name.upper())
    year_match = re.search(r"(?<!\d)(?:19|20)\d{2}(?!\d)", name)
    folded = unicodedata.normalize("NFKD", name.casefold())
    unaccented = "".join(char for char in folded if not unicodedata.combining(char))
    normalized = re.sub(r"[^a-z0-9]+", "", unaccented)
    report_type = None
    if "hopnhat" in normalized or "consolidated" in normalized:
        report_type = "consolidated"
    elif any(marker in normalized for marker in ("congtyme", "rieng", "separate")):
        report_type = "separate"
    return {
        "company_code": company_match.group("company") if company_match else None,
        "fiscal_year": int(year_match.group(0)) if year_match else None,
        "report_type": report_type,
    }


def _period_value(chunk: FinancialChunk, year: int) -> str | None:
    values = chunk.metadata.get("period_values")
    if not isinstance(values, dict):
        return None
    value = values.get(str(year))
    if value is None or str(value).strip() in {"", "—", "-", "None"}:
        return None
    return str(value).strip()


def _choose_metrics(
    chunks_by_source: dict[str, list[FinancialChunk]], mapper: FinancialMetricMapper
) -> dict[str, list[dict[str, Any]]]:
    docs_by_metric: dict[tuple[str, str], set[str]] = defaultdict(set)
    for source, chunks in chunks_by_source.items():
        for chunk in chunks:
            if chunk.chunk_type != ChunkType.FINANCIAL_ITEM or not chunk.canonical_metric:
                continue
            docs_by_metric[(chunk.statement_type.value, chunk.canonical_metric)].add(source)

    selected: dict[str, list[dict[str, Any]]] = {}
    for statement in STATEMENTS_TO_COVER:
        candidates = [
            (metric, sources)
            for (statement_name, metric), sources in docs_by_metric.items()
            if statement_name == statement.value
        ]
        candidates.sort(key=lambda item: (-len(item[1]), item[0]))
        selected[statement.value] = []
        for metric, sources in candidates[:MAX_METRICS_PER_STATEMENT]:
            definition = next(
                (entry for entry in mapper.metrics_list if entry.canonical_name == metric), None
            )
            if definition is None:
                continue
            aliases = [alias.strip() for alias in definition.aliases if alias.strip()]
            phrase = next(
                (alias.casefold() for alias in aliases if not re.search(r"[A-Za-z]{4,}", alias)),
                metric.casefold(),
            )
            selected[statement.value].append(
                {
                    "canonical_metric": metric,
                    "metric_id": definition.metric_id,
                    "phrase": phrase,
                    "documents": len(sources),
                    "statement_type": statement.value,
                }
            )
    return selected


def _report_phrase(report_type: str | None) -> str:
    if report_type == "consolidated":
        return "báo cáo hợp nhất"
    if report_type == "separate":
        return "báo cáo tài chính riêng"
    return "báo cáo tài chính"


def _period_phrase(statement_type: str, year: int) -> str:
    if statement_type == StatementType.BALANCE_SHEET.value:
        return f"cuối năm {year}"
    return f"trong năm {year}"


def _build_targets(
    chunks_by_source: dict[str, list[FinancialChunk]],
    source_metadata: dict[str, dict[str, Any]],
    selected_metrics: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    metric_lookup = {
        entry["canonical_metric"]: entry
        for entries in selected_metrics.values()
        for entry in entries
    }
    targets: list[dict[str, Any]] = []
    for source, chunks in sorted(chunks_by_source.items()):
        metadata = source_metadata[source]
        year = metadata["fiscal_year"]
        company = metadata["company_code"]
        if year is None or company is None:
            continue
        for metric, metric_info in metric_lookup.items():
            candidates = [
                chunk
                for chunk in chunks
                if chunk.chunk_type == ChunkType.FINANCIAL_ITEM
                and chunk.canonical_metric == metric
                and _period_value(chunk, year) is not None
            ]
            if not candidates:
                continue
            candidates.sort(key=lambda chunk: (chunk.page or 0, chunk.chunk_id))
            chunk = candidates[0]
            report_type = metadata["report_type"]
            report_phrase = _report_phrase(report_type)
            phrase = metric_info["phrase"]
            period = _period_phrase(chunk.statement_type.value, year)
            wording = (len(targets) + year) % 2
            if wording:
                query = (
                    f"Trong {report_phrase} năm {year}, {company} ghi nhận "
                    f"{phrase} bao nhiêu?"
                )
            else:
                query = (
                    f"{phrase} của {company} {period} theo {report_phrase} là bao nhiêu?"
                )
            targets.append(
                {
                    "source_key": source,
                    "source_file": Path(source).name,
                    "company_code": company,
                    "fiscal_year": year,
                    "report_type": report_type,
                    "statement_type": chunk.statement_type.value,
                    "canonical_metric": metric,
                    "metric_id": metric_info["metric_id"],
                    "metric_phrase": phrase,
                    "page": chunk.page,
                    "chunk_id": chunk.chunk_id,
                    "expected_value": _period_value(chunk, year),
                    "period_values": chunk.metadata.get("period_values", {}),
                    "query": query,
                }
            )
    return targets


def _make_cases(targets: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    direct = [
        {
            "case_id": f"lookup:{target['source_key']}:{target['canonical_metric']}",
            "kind": "single_metric",
            "query": target["query"],
            "targets": [target],
            "expected_companies": [target["company_code"]],
            "expected_years": [target["fiscal_year"]],
            "expected_report_type": target["report_type"],
            "expected_metric": target["canonical_metric"],
            "expected_statement_type": target["statement_type"],
        }
        for target in targets
    ]

    temporal: list[dict[str, Any]] = []
    for target in targets:
        values = target["period_values"]
        if not isinstance(values, dict):
            continue
        year = target["fiscal_year"]
        comparative_years = [
            int(key)
            for key, value in values.items()
            if str(key).isdigit()
            and int(key) < year
            and value is not None
            and str(value).strip() not in {"", "—", "-", "None"}
        ]
        if not comparative_years:
            continue
        comparative_year = max(comparative_years)
        report_phrase = _report_phrase(target["report_type"])
        temporal.append(
            {
                "case_id": f"year_compare:{target['source_key']}:{target['canonical_metric']}",
                "kind": "year_comparison",
                "query": (
                    f"So sánh {target['metric_phrase']} của {target['company_code']} "
                    f"năm {year} với năm {comparative_year} theo {report_phrase}."
                ),
                "targets": [target],
                "expected_companies": [target["company_code"]],
                "expected_years": [year, comparative_year],
                "expected_report_type": target["report_type"],
                "expected_metric": target["canonical_metric"],
                "expected_statement_type": target["statement_type"],
                "expected_values": {
                    str(year): target["period_values"].get(str(year)),
                    str(comparative_year): target["period_values"].get(str(comparative_year)),
                },
            }
        )

    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for target in targets:
        key = (
            target["fiscal_year"],
            target["report_type"],
            target["canonical_metric"],
        )
        grouped[key].setdefault(target["company_code"], target)
    cross_company: list[dict[str, Any]] = []
    for (year, report_type, metric), company_targets in sorted(
        grouped.items(), key=lambda item: str(item[0])
    ):
        companies = sorted(company_targets)
        for left_index, left_company in enumerate(companies):
            for right_company in companies[left_index + 1 :]:
                left = company_targets[left_company]
                right = company_targets[right_company]
                report_phrase = _report_phrase(report_type)
                period = _period_phrase(left["statement_type"], year)
                cross_company.append(
                    {
                        "case_id": f"company_compare:{year}:{report_type}:{metric}:{left_company}:{right_company}",
                        "kind": "company_comparison",
                        "query": (
                            f"So sánh {left['metric_phrase']} của {left_company} và {right_company} "
                            f"{period} theo {report_phrase}."
                        ),
                        "targets": [left, right],
                        "expected_companies": [left_company, right_company],
                        "expected_years": [year],
                        "expected_report_type": report_type,
                        "expected_metric": metric,
                        "expected_statement_type": left["statement_type"],
                    }
                )

    no_answer: list[dict[str, Any]] = []
    company_type_target: dict[tuple[str, str | None], dict[str, Any]] = {}
    for target in targets:
        company_type_target.setdefault(
            (target["company_code"], target["report_type"]), target
        )
    for (company, report_type), target in sorted(
        company_type_target.items(), key=lambda item: str(item[0])
    ):
        report_phrase = _report_phrase(report_type)
        year = 2099
        no_answer.append(
            {
                "case_id": f"no_answer:{company}:{report_type}:2099",
                "kind": "no_answer",
                "query": (
                    f"{target['metric_phrase']} của {company} năm {year} theo "
                    f"{report_phrase} là bao nhiêu?"
                ),
                "targets": [],
                "expected_companies": [company],
                "expected_years": [year],
                "expected_report_type": report_type,
                "expected_metric": target["canonical_metric"],
                "expected_statement_type": target["statement_type"],
            }
        )
    return {
        "single_metric": direct,
        "year_comparison": temporal,
        "company_comparison": cross_company,
        "no_answer": no_answer,
    }


def _parse_ok(query: Any, case: dict[str, Any]) -> bool:
    companies = list(query.company_codes or ([query.company_code] if query.company_code else []))
    return (
        companies == case["expected_companies"]
        and query.fiscal_years == case["expected_years"]
        and query.report_type == case["expected_report_type"]
        and query.canonical_metric == case["expected_metric"]
        and query.statement_type is not None
        and query.statement_type.value == case["expected_statement_type"]
    )


def _result_matches_target(
    chunk: FinancialChunk, target: dict[str, Any], year: int | None = None
) -> bool:
    if (
        chunk.company_code != target["company_code"]
        or chunk.canonical_metric != target["canonical_metric"]
        or chunk.statement_type.value != target["statement_type"]
        or chunk.report_type != target["report_type"]
        or chunk.page is None
    ):
        return False
    target_year = year if year is not None else target["fiscal_year"]
    if target_year not in {
        chunk.fiscal_year,
        chunk.comparative_fiscal_year,
        chunk.metadata.get("comparative_fiscal_year"),
    }:
        return False
    expected_value = (
        target["period_values"].get(str(target_year))
        if year is not None
        else target["expected_value"]
    )
    if expected_value is None:
        return False
    return _period_value(chunk, target_year) == str(expected_value).strip()


def _metadata_violations(results: list[Any], query: Any) -> list[dict[str, Any]]:
    expected_companies = {
        code.casefold()
        for code in (query.company_codes or ([query.company_code] if query.company_code else []))
    }
    years = query.fiscal_years or ([query.fiscal_year] if query.fiscal_year else [])
    violations = []
    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        failed_fields = []
        if expected_companies and (chunk.company_code or "").casefold() not in expected_companies:
            failed_fields.append("company_code")
        if years and not any(
            year in {
                chunk.fiscal_year,
                chunk.comparative_fiscal_year,
                chunk.metadata.get("comparative_fiscal_year"),
            }
            for year in years
        ):
            failed_fields.append("fiscal_year")
        if query.report_type and chunk.report_type != query.report_type:
            failed_fields.append("report_type")
        if query.statement_type and query.statement_type != StatementType.GENERAL:
            if chunk.statement_type != query.statement_type:
                failed_fields.append("statement_type")
        if failed_fields:
            violations.append(
                {
                    "rank": rank,
                    "chunk_id": chunk.chunk_id,
                    "source_file": chunk.source_file,
                    "failed_fields": failed_fields,
                }
            )
    return violations


def _evaluate_cases(
    cases: list[dict[str, Any]],
    retriever: FinancialHybridRetriever,
    query_engine: QueryUnderstandingEngine,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for case in cases:
        query = query_engine.parse(case["query"])
        parse_ok = _parse_ok(query, case)
        started = time.perf_counter()
        results = retriever.search(query, top_k=top_k)
        elapsed = time.perf_counter() - started

        found_targets: list[dict[str, Any]] = []
        if case["kind"] == "year_comparison":
            target = case["targets"][0]
            years = case["expected_values"]
            year_ranks = []
            for year, expected_value in years.items():
                rank_for_value = next(
                    (
                        index
                        for index, result in enumerate(results, start=1)
                        if result.chunk.company_code == target["company_code"]
                        and result.chunk.canonical_metric == target["canonical_metric"]
                        and result.chunk.statement_type.value == target["statement_type"]
                        and result.chunk.report_type == target["report_type"]
                        and int(year) in {
                            result.chunk.fiscal_year,
                            result.chunk.comparative_fiscal_year,
                            result.chunk.metadata.get("comparative_fiscal_year"),
                        }
                        and _period_value(result.chunk, int(year)) == str(expected_value).strip()
                    ),
                    None,
                )
                if rank_for_value is not None:
                    year_ranks.append(rank_for_value)
            hit = len(year_ranks) == len(years)
            rank = max(year_ranks) if hit else None
            if hit:
                found_targets.append(
                    {
                        "company_code": target["company_code"],
                        "year_values": years,
                        "retrieved_sources": sorted(
                            {
                                result.chunk.source_file
                                for result in results
                                if result.chunk.company_code == target["company_code"]
                                and result.chunk.canonical_metric == target["canonical_metric"]
                            }
                        ),
                    }
                )
        elif case["kind"] == "no_answer":
            hit = not results
            rank = None
        else:
            ranks = []
            for target in case["targets"]:
                rank_for_target = next(
                    (
                        index
                        for index, result in enumerate(results, start=1)
                        if _result_matches_target(result.chunk, target)
                    ),
                    None,
                )
                if rank_for_target is not None:
                    ranks.append(rank_for_target)
                    matched_result = results[rank_for_target - 1]
                    found_targets.append(
                        {
                            "expected_source_file": target["source_file"],
                            "retrieved_source_file": matched_result.chunk.source_file,
                            "source_file_exact_match": matched_result.chunk.source_file == target["source_file"],
                            "company_code": target["company_code"],
                            "value": target["expected_value"],
                            "page": matched_result.chunk.page,
                            "rank": rank_for_target,
                        }
                    )
            hit = len(ranks) == len(case["targets"])
            rank = max(ranks) if ranks else None

        metadata_violation_details = _metadata_violations(results, query)
        top_results = [
            {
                "company_code": result.chunk.company_code,
                "source_file": result.chunk.source_file,
                "fiscal_year": result.chunk.fiscal_year,
                "comparative_fiscal_year": result.chunk.comparative_fiscal_year,
                "report_type": result.chunk.report_type,
                "page": result.chunk.page,
                "chunk_id": result.chunk.chunk_id,
                "canonical_metric": result.chunk.canonical_metric,
                "period_values": result.chunk.metadata.get("period_values", {}),
                "unit": result.chunk.unit,
                "statement_type": result.chunk.statement_type.value,
                "chunk_type": result.chunk.chunk_type.value,
            }
            for result in results
        ]
        output.append(
            {
                "case_id": case["case_id"],
                "kind": case["kind"],
                "query": case["query"],
                "query_parse_exact": parse_ok,
                "expected_companies": case["expected_companies"],
                "parsed_companies": query.company_codes,
                "expected_years": case["expected_years"],
                "parsed_years": query.fiscal_years,
                "expected_report_type": case["expected_report_type"],
                "parsed_report_type": query.report_type,
                "expected_metric": case["expected_metric"],
                "parsed_metric": query.canonical_metric,
                "expected_targets": [
                    {
                        "source_file": target["source_file"],
                        "company_code": target["company_code"],
                        "fiscal_year": target["fiscal_year"],
                        "report_type": target["report_type"],
                        "statement_type": target["statement_type"],
                        "canonical_metric": target["canonical_metric"],
                        "expected_value": target["expected_value"],
                        "expected_period_values": target["period_values"],
                        "page": target["page"],
                    }
                    for target in case["targets"]
                ],
                "hit_at_k": hit,
                "best_relevant_rank": rank,
                "metadata_constraint_violations": len(metadata_violation_details),
                "metadata_violation_details": metadata_violation_details,
                "matched_targets": found_targets,
                "top_results": top_results,
                "retrieval_seconds": round(elapsed, 6),
            }
        )
    times = [row["retrieval_seconds"] for row in output]
    sorted_times = sorted(times)
    p95 = sorted_times[min(len(sorted_times) - 1, int(len(sorted_times) * 0.95))] if sorted_times else 0.0
    summary = {
        "cases": len(output),
        "query_parse_exact_rate": round(
            sum(row["query_parse_exact"] for row in output) / len(output), 4
        ) if output else None,
        "retrieval_hit_at_k": round(
            sum(row["hit_at_k"] for row in output) / len(output), 4
        ) if output else None,
        "mean_reciprocal_rank": round(
            statistics.fmean(
                (1 / row["best_relevant_rank"]) if row["best_relevant_rank"] else 0.0
                for row in output
                if row["kind"] != "no_answer"
            ) if any(row["kind"] != "no_answer" for row in output) else 0.0,
            4,
        ),
        "mean_search_seconds": round(statistics.fmean(times), 6) if times else 0.0,
        "p95_search_seconds": round(p95, 6),
        "metadata_constraint_violations": sum(
            row["metadata_constraint_violations"] for row in output
        ),
        "failure_count": sum(not row["hit_at_k"] for row in output),
    }
    return output, summary


def _subset_cases(
    cases_by_kind: dict[str, list[dict[str, Any]]], selected_sources: set[str]
) -> dict[str, list[dict[str, Any]]]:
    subset: dict[str, list[dict[str, Any]]] = {}
    for kind, cases in cases_by_kind.items():
        selected = []
        for case in cases:
            if kind == "no_answer":
                selected_company_reports = {
                    (target["targets"][0]["company_code"], target["expected_report_type"])
                    for target in cases_by_kind["single_metric"]
                    if target["targets"][0]["source_key"] in selected_sources
                }
                if (case["expected_companies"][0], case["expected_report_type"]) in selected_company_reports:
                    selected.append(case)
            elif case["targets"] and all(
                target["source_key"] in selected_sources for target in case["targets"]
            ):
                selected.append(case)
        subset[kind] = selected
    return subset


def _choose_dense_sources(
    source_metadata: dict[str, dict[str, Any]], chunks_by_source: dict[str, list[FinancialChunk]], limit: int
) -> list[str]:
    candidates: dict[tuple[str, str | None], list[tuple[int, str]]] = defaultdict(list)
    for source, chunks in chunks_by_source.items():
        if not chunks:
            continue
        metadata = source_metadata[source]
        company = metadata.get("company_code") or chunks[0].company_code
        report_type = metadata.get("report_type") or chunks[0].report_type
        year = metadata.get("fiscal_year") or chunks[0].fiscal_year or 0
        if company:
            candidates[(company, report_type)].append((year, source))
    selected = []
    for key in sorted(candidates, key=str):
        entries = sorted(candidates[key], key=lambda item: (-item[0], item[1]))
        selected.append(entries[0][1])
    return selected[:limit]


def _document_metadata_quality(
    chunks_by_source: dict[str, list[FinancialChunk]],
    expected_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    checks = {key: {"checked": 0, "matched": 0} for key in ("company_code", "fiscal_year", "report_type")}
    doc_rows = []
    for source, chunks in sorted(chunks_by_source.items()):
        expected = expected_metadata[source]
        actual = {
            "company_code": sorted({chunk.company_code for chunk in chunks if chunk.company_code}),
            "fiscal_year": sorted({chunk.fiscal_year for chunk in chunks if chunk.fiscal_year is not None}),
            "report_type": sorted({chunk.report_type for chunk in chunks if chunk.report_type}),
        }
        matches = {}
        for key, expected_value in expected.items():
            if expected_value is None:
                continue
            checks[key]["checked"] += 1
            matched = expected_value in actual[key]
            checks[key]["matched"] += int(matched)
            matches[key] = matched
        doc_rows.append(
            {
                "source_file": source,
                "expected": expected,
                "actual": actual,
                "metadata_matches": matches,
                "chunk_count": len(chunks),
            }
        )
    for value in checks.values():
        value["match_rate"] = round(value["matched"] / value["checked"], 4) if value["checked"] else None
    return {"field_checks": checks, "documents": doc_rows}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset" / "ocr_annual_financials",
        help="Directory of local OCR TXT annual reports; only direct *.txt files are included.",
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON result path")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-files", type=int, default=0, help="Optional small smoke limit; 0 means all files")
    parser.add_argument("--dense", action="store_true", help="Compare E5 on a stratified report subset")
    parser.add_argument("--dense-doc-limit", type=int, default=12)
    parser.add_argument(
        "--reuse-bm25",
        type=Path,
        help="Reuse a compatible full-corpus BM25 result and compute the dense comparison only",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    data_dir = args.data_dir.resolve()
    source_paths = sorted(path for path in data_dir.glob("*.txt") if path.is_file())
    if args.max_files > 0:
        source_paths = source_paths[: args.max_files]
    if not source_paths:
        raise SystemExit(f"No direct .txt reports found in {data_dir}")

    started_at = datetime.now(timezone.utc).isoformat()
    chunker = TxtFinancialChunker()
    query_engine = QueryUnderstandingEngine()
    metric_mapper = FinancialMetricMapper()
    source_metadata: dict[str, dict[str, Any]] = {}
    chunks_by_source: dict[str, list[FinancialChunk]] = {}
    source_hashes = []
    parse_failures = []
    chunking_seconds = 0.0
    for path in source_paths:
        relative_source = path.relative_to(repo_root).as_posix() if path.is_relative_to(repo_root) else path.name
        source_metadata[relative_source] = _filename_metadata(path)
        source_hashes.append(
            {
                "source": relative_source,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
        started = time.perf_counter()
        try:
            chunks = chunker.chunk_file(path)
            chunks_by_source[relative_source] = chunks
        except Exception as exc:  # Record data/encoding defects and continue the corpus audit.
            chunks_by_source[relative_source] = []
            parse_failures.append(
                {
                    "source": relative_source,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                }
            )
        chunking_seconds += time.perf_counter() - started

    chunks = [chunk for source_chunks in chunks_by_source.values() for chunk in source_chunks]
    chunk_ids = Counter(chunk.chunk_id for chunk in chunks)
    duplicate_chunk_ids = {key: value for key, value in chunk_ids.items() if value > 1}
    chunk_id_manifest_sha256 = hashlib.sha256(
        "\n".join(sorted(chunk.chunk_id for chunk in chunks)).encode("utf-8")
    ).hexdigest()
    if duplicate_chunk_ids:
        samples = list(duplicate_chunk_ids.items())[:5]
        raise SystemExit(
            f"Refusing retrieval evaluation: {len(duplicate_chunk_ids)} duplicate chunk IDs; "
            f"examples: {samples}"
        )
    metric_docs = defaultdict(set)
    metric_chunk_counts = Counter()
    chunks_by_type = Counter()
    chunks_by_statement = Counter()
    chunks_by_company = Counter()
    chunks_by_year = Counter()
    chunks_by_report_type = Counter()
    page_counts = {}
    for chunk in chunks:
        chunks_by_type[chunk.chunk_type.value] += 1
        chunks_by_statement[chunk.statement_type.value] += 1
        chunks_by_company[chunk.company_code or "missing"] += 1
        chunks_by_year[str(chunk.fiscal_year) if chunk.fiscal_year else "missing"] += 1
        chunks_by_report_type[chunk.report_type or "missing"] += 1
        if chunk.chunk_type == ChunkType.FINANCIAL_ITEM and chunk.canonical_metric:
            metric_docs[(chunk.statement_type.value, chunk.canonical_metric)].add(chunk.source_file)
            metric_chunk_counts[(chunk.statement_type.value, chunk.canonical_metric)] += 1
    for source, source_chunks in chunks_by_source.items():
        page_counts[source] = len({chunk.page for chunk in source_chunks if chunk.page is not None})

    selected_metrics = _choose_metrics(chunks_by_source, metric_mapper)
    targets = _build_targets(chunks_by_source, source_metadata, selected_metrics)
    cases_by_kind = _make_cases(targets)
    all_cases = [case for group in cases_by_kind.values() for case in group]
    all_cases.sort(key=lambda case: (
        tuple(case["expected_companies"]),
        tuple(case["expected_years"]),
        case["expected_report_type"] or "",
        case["expected_statement_type"],
        case["expected_metric"],
        case["kind"],
        case["case_id"],
    ))

    metadata_quality = _document_metadata_quality(chunks_by_source, source_metadata)
    duplicate_hashes: dict[str, list[str]] = defaultdict(list)
    for item in source_hashes:
        duplicate_hashes[item["sha256"]].append(item["source"])
    duplicate_files = [sources for sources in duplicate_hashes.values() if len(sources) > 1]

    output_path = args.output if args.output.is_absolute() else Path.cwd() / args.output
    payload: dict[str, Any] = {
        "experiment": "txt_annual_corpus_retrieval_stress",
        "started_at_utc": started_at,
        "scope": {
            "llm_generation_calls": 0,
            "embedding_http_calls": 0,
            "embedding_texts": 0,
            "data_directory": str(data_dir),
            "file_selection": "direct *.txt files only; nested .cache directory excluded",
            "source_files": len(source_paths),
            "source_bytes": sum(item["bytes"] for item in source_hashes),
            "source_sha256_manifest": source_hashes,
            "top_k": args.top_k,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu_count": __import__("os").cpu_count(),
        },
        "ingestion": {
            "chunking_seconds": round(chunking_seconds, 4),
            "documents_with_chunks": sum(bool(items) for items in chunks_by_source.values()),
            "documents_without_chunks": [source for source, items in chunks_by_source.items() if not items],
            "parse_failures": parse_failures,
            "total_chunks": len(chunks),
            "duplicate_chunk_ids": duplicate_chunk_ids,
            "chunk_id_manifest_sha256": chunk_id_manifest_sha256,
            "chunk_type_counts": dict(chunks_by_type),
            "statement_type_counts": dict(chunks_by_statement),
            "chunk_company_counts": dict(chunks_by_company),
            "chunk_year_counts": dict(chunks_by_year),
            "chunk_report_type_counts": dict(chunks_by_report_type),
            "pages_per_source": page_counts,
            "financial_item_chunks_without_metric": sum(
                chunk.chunk_type == ChunkType.FINANCIAL_ITEM and not chunk.canonical_metric
                for chunk in chunks
            ),
        },
        "metadata_quality": metadata_quality,
        "source_duplicates_by_sha256": duplicate_files,
        "selected_metrics": selected_metrics,
        "query_set_counts": {key: len(value) for key, value in cases_by_kind.items()},
        "evaluation_note": (
            "The TXT retrieval cases are generated from the same parser's financial-item chunks. "
            "They are broad self-labeled stress/regression cases, not independent human ground truth. "
            "Interpret them alongside the 88-case manually labeled Markdown benchmark."
        ),
        "bm25_full_corpus": {},
        "dense_stratified": None,
    }

    if args.reuse_bm25:
        cached_path = args.reuse_bm25.resolve()
        cached_payload = json.loads(cached_path.read_text(encoding="utf-8"))
        cached_scope = cached_payload.get("scope", {})
        if cached_scope.get("source_sha256_manifest") != source_hashes:
            raise SystemExit("BM25 cache does not match the current source-file hashes")
        if cached_scope.get("top_k") != args.top_k:
            raise SystemExit("BM25 cache uses a different top-k value")
        if cached_payload.get("query_set_counts") != payload["query_set_counts"]:
            raise SystemExit("BM25 cache uses a different generated query set")
        if cached_payload.get("ingestion", {}).get("chunk_id_manifest_sha256") != chunk_id_manifest_sha256:
            raise SystemExit("BM25 cache uses a different chunk ID manifest")
        cached_rows = cached_payload.get("bm25_full_corpus", {}).get("cases", [])
        current_manifest = [
            (case["case_id"], case["kind"], case["query"]) for case in all_cases
        ]
        cached_manifest = [
            (row.get("case_id"), row.get("kind"), row.get("query")) for row in cached_rows
        ]
        if not cached_rows:
            raise SystemExit("BM25 cache has no full-corpus case results")
        if cached_manifest != current_manifest:
            raise SystemExit("BM25 cache does not match the current generated case queries")
        payload["bm25_full_corpus"] = cached_payload["bm25_full_corpus"]
        payload["bm25_reused_from"] = str(cached_path)
    else:
        bm25_retriever = FinancialHybridRetriever(chunks, enable_dense=False)
        retrieval_started = time.perf_counter()
        bm25_rows, bm25_summary = _evaluate_cases(
            all_cases, bm25_retriever, query_engine, args.top_k
        )
        bm25_total_seconds = time.perf_counter() - retrieval_started
        bm25_groups = {
            kind: {
                "summary": _evaluate_cases_summary(rows)
            }
            for kind, rows in _group_rows(bm25_rows).items()
        }
        payload["bm25_full_corpus"] = {
            "total_retrieval_seconds": round(bm25_total_seconds, 4),
            "overall_summary": bm25_summary,
            "by_query_type": bm25_groups,
            "cases": bm25_rows,
        }
    _write_json(output_path, payload)

    if args.dense:
        selected_dense_sources = _choose_dense_sources(
            source_metadata, chunks_by_source, max(1, args.dense_doc_limit)
        )
        dense_source_set = set(selected_dense_sources)
        dense_chunks = [
            chunk
            for source in selected_dense_sources
            for chunk in chunks_by_source.get(source, [])
        ]
        dense_cases_by_kind = _subset_cases(cases_by_kind, dense_source_set)
        dense_cases = [case for group in dense_cases_by_kind.values() for case in group]
        dense_cases.sort(key=lambda case: (
            tuple(case["expected_companies"]),
            tuple(case["expected_years"]),
            case["expected_report_type"] or "",
            case["expected_statement_type"],
            case["expected_metric"],
            case["kind"],
            case["case_id"],
        ))
        dense_payload: dict[str, Any] = {
            "selected_sources": selected_dense_sources,
            "documents": len(selected_dense_sources),
            "chunks": len(dense_chunks),
            "query_set_counts": {key: len(value) for key, value in dense_cases_by_kind.items()},
            "embedding_model": None,
            "embedding_endpoint": None,
            "embedding_index_seconds": None,
            "bm25_on_same_subset": None,
            "bm25_e5_hybrid_on_same_subset": None,
        }
        embedding_stats = {"http_calls": 0, "texts": 0, "seconds": 0.0}
        original_embed = E5EmbeddingAdapter._call_embed

        def timed_embed(adapter: E5EmbeddingAdapter, texts: Any) -> Any:
            embed_started = time.perf_counter()
            vectors = original_embed(adapter, texts)
            embedding_stats["http_calls"] += 1
            embedding_stats["texts"] += len(texts)
            embedding_stats["seconds"] += time.perf_counter() - embed_started
            return vectors

        try:
            subset_bm25 = FinancialHybridRetriever(dense_chunks, enable_dense=False)
            E5EmbeddingAdapter._call_embed = timed_embed
            index_started = time.perf_counter()
            subset_dense = FinancialHybridRetriever(dense_chunks, enable_dense=True)
            dense_payload["embedding_index_seconds"] = round(
                time.perf_counter() - index_started, 4
            )
            embedding_adapter = E5EmbeddingAdapter()
            dense_payload["embedding_model"] = embedding_adapter.model
            dense_payload["embedding_endpoint"] = embedding_adapter._endpoint()
            bm25_subset_rows, bm25_subset_summary = _evaluate_cases(
                dense_cases, subset_bm25, query_engine, args.top_k
            )
            hybrid_rows, hybrid_summary = _evaluate_cases(
                dense_cases, subset_dense, query_engine, args.top_k
            )
            dense_payload["bm25_on_same_subset"] = {
                "summary": bm25_subset_summary,
                "by_query_type": {
                    kind: _evaluate_cases_summary(rows)
                    for kind, rows in _group_rows(bm25_subset_rows).items()
                },
                "cases": bm25_subset_rows,
            }
            dense_payload["bm25_e5_hybrid_on_same_subset"] = {
                "summary": hybrid_summary,
                "by_query_type": {
                    kind: _evaluate_cases_summary(rows)
                    for kind, rows in _group_rows(hybrid_rows).items()
                },
                "cases": hybrid_rows,
            }
            dense_payload["embedding_vectors"] = len(subset_dense.dense_retriever.chunk_vectors)
        except Exception as exc:  # Preserve completed BM25 results if local E5 is unavailable.
            dense_payload["error"] = f"{type(exc).__name__}: {str(exc)[:1000]}"
        finally:
            E5EmbeddingAdapter._call_embed = original_embed
        dense_payload["embedding_http_calls"] = embedding_stats["http_calls"]
        dense_payload["embedding_texts"] = embedding_stats["texts"]
        dense_payload["embedding_seconds"] = round(embedding_stats["seconds"], 4)
        payload["scope"]["embedding_http_calls"] = embedding_stats["http_calls"]
        payload["scope"]["embedding_texts"] = embedding_stats["texts"]
        payload["dense_stratified"] = dense_payload
        _write_json(output_path, payload)

    print(
        json.dumps(
            {
                "output": str(output_path),
                "documents": len(source_paths),
                "chunks": len(chunks),
                "query_set_counts": payload["query_set_counts"],
                "bm25_summary": payload["bm25_full_corpus"].get("overall_summary"),
                "dense": (
                    {
                        "documents": payload["dense_stratified"].get("documents"),
                        "chunks": payload["dense_stratified"].get("chunks"),
                        "error": payload["dense_stratified"].get("error"),
                    }
                    if payload["dense_stratified"]
                    else None
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["kind"]].append(row)
    return dict(grouped)


def _evaluate_cases_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"cases": 0, "retrieval_hit_at_k": None, "query_parse_exact_rate": None}
    times = [row["retrieval_seconds"] for row in rows]
    sorted_times = sorted(times)
    target_matches = [
        target
        for row in rows
        for target in row["matched_targets"]
        if "source_file_exact_match" in target
    ]
    retrieval_rows = [row for row in rows if row["kind"] != "no_answer"]
    return {
        "cases": len(rows),
        "query_parse_exact_rate": round(
            sum(row["query_parse_exact"] for row in rows) / len(rows), 4
        ),
        "retrieval_hit_at_k": round(
            sum(row["hit_at_k"] for row in rows) / len(rows), 4
        ),
        "mean_reciprocal_rank": round(
            statistics.fmean(
                (1 / row["best_relevant_rank"]) if row["best_relevant_rank"] else 0.0
                for row in retrieval_rows
            ) if retrieval_rows else 0.0,
            4,
        ),
        "exact_source_match_rate": round(
            sum(target["source_file_exact_match"] for target in target_matches) / len(target_matches),
            4,
        ) if target_matches else None,
        "mean_search_seconds": round(statistics.fmean(times), 6),
        "p95_search_seconds": round(
            sorted_times[min(len(sorted_times) - 1, int(len(sorted_times) * 0.95))], 6
        ),
        "metadata_constraint_violations": sum(
            row["metadata_constraint_violations"] for row in rows
        ),
        "failure_count": sum(not row["hit_at_k"] for row in rows),
    }


if __name__ == "__main__":
    main()
