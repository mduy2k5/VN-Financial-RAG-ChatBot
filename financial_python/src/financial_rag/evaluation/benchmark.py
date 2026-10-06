"""Run the labeled financial retrieval and citation benchmark."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from ..chunking.financial_chunker import FinancialMarkdownChunker
from ..context.assembler import ContextAssembler
from ..query.directives import QueryDirectivePlanner
from ..query.understanding import QueryUnderstandingEngine
from ..retrieval.hybrid import FinancialHybridRetriever
from ..retrieval.operators import FilterOperator, ValidateOperator
from ..schemas import FinancialChunk, ParsedQuery, PeriodType
from .metrics import CitationEvaluator, RetrievalEvaluator


def run_benchmark(
    dataset_path: Path,
    source_path: Path | Iterable[Path],
    top_k: int = 5,
    enable_dense: bool = False,
    apply_metadata_filters: bool = True,
    directive_mode: str = "baseline",
    strict_metric_scope: bool | None = None,
    directive_suffix: str = "",
    financial_chunker: FinancialMarkdownChunker | None = None,
    chunks_override: Iterable[FinancialChunk] | None = None,
) -> dict[str, Any]:
    if directive_mode not in {"baseline", "structural", "filter", "validate", "full"}:
        raise ValueError(
            "directive_mode must be one of baseline, structural, filter, validate, or full"
        )
    examples = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not examples:
        raise ValueError(f"Benchmark dataset is empty: {dataset_path}")

    source_paths = [source_path] if isinstance(source_path, Path) else list(source_path)
    if not source_paths:
        raise ValueError("At least one benchmark source is required")
    chunker = financial_chunker or FinancialMarkdownChunker()
    chunks = list(chunks_override) if chunks_override is not None else []
    source_names = {path.name for path in source_paths}
    if chunks_override is None:
        for path in source_paths:
            company_code = path.name.split("_", maxsplit=1)[0].upper()
            source_pdf = path.name.replace("_benchmark.md", ".pdf")
            chunks.extend(chunker.chunk_file(path, company_code=company_code, source_pdf=source_pdf))
    if not chunks:
        raise ValueError(f"No chunks were produced from benchmark sources: {source_paths}")
    document_disjoint_sources = {
        Path(item["source"]).name
        for example in examples
        if example.get("source_partition") == "document_disjoint"
        for item in example.get("relevant_sources", [])
    }
    missing_disjoint_sources = document_disjoint_sources - source_names
    if missing_disjoint_sources:
        raise ValueError(
            "Document-disjoint benchmark sources were not provided: "
            + ", ".join(sorted(missing_disjoint_sources))
        )
    retrievers: dict[str, FinancialHybridRetriever] = {}
    source_names_by_partition: dict[str, list[str]] = {}
    query_engine = QueryUnderstandingEngine()
    directive_planner = QueryDirectivePlanner()
    assembler = ContextAssembler()
    directive_activity = {
        "queries_with_structural_hints": 0,
        "queries_with_exclusion_hints": 0,
        "queries_with_validation_constraints": 0,
        "filter_removed_candidates": 0,
        "validation_reranked_queries": 0,
    }
    rows = []

    for example in examples:
        query = query_engine.parse(example["query"])
        answerable = example.get("answerable", True)
        split = example.get("split", "dev")
        source_partition = example.get(
            "source_partition",
            "development" if split == "dev" else "shared_documents",
        )
        if source_partition not in retrievers:
            if source_partition == "document_disjoint":
                partition_chunks = chunks
                partition_source_names = sorted(source_names)
            else:
                partition_chunks = [
                    chunk
                    for chunk in chunks
                    if Path(chunk.source_file).name not in document_disjoint_sources
                ]
                partition_source_names = sorted(source_names - document_disjoint_sources)
            if not partition_chunks:
                raise ValueError(f"No source chunks are available for partition {source_partition}")
            retrievers[source_partition] = FinancialHybridRetriever(
                partition_chunks,
                enable_dense=enable_dense,
            )
            source_names_by_partition[source_partition] = partition_source_names
        retriever = retrievers[source_partition]
        relevant_evidence = [
            (Path(item["source"]).name, int(item["page"]))
            for item in example.get("relevant_sources", [])
        ]
        if answerable and not relevant_evidence:
            relevant_evidence = [
                (source_paths[0].name, int(page))
                for page in example.get("relevant_pages", [])
            ]
            if not relevant_evidence:
                raise ValueError(f"Answerable benchmark case {example['id']} has no relevant source page")
        if not answerable and relevant_evidence:
            raise ValueError(f"Unanswerable benchmark case {example['id']} has relevant source pages")

        retrieval_query = query if apply_metadata_filters else _without_metadata_scope(query)
        directives = directive_planner.plan(
            example["query"] + directive_suffix, query
        )
        directive_activity["queries_with_structural_hints"] += int(
            bool(
                directives.structural.preferred_chunk_types
                or directives.structural.preferred_sections
            )
        )
        directive_activity["queries_with_exclusion_hints"] += int(
            bool(directives.filters.exclude_phrases)
        )
        directive_activity["queries_with_validation_constraints"] += int(
            bool(
                directives.validation.positive_constraints
                or directives.validation.negative_constraints
            )
        )
        search_directives = (
            directives if directive_mode in {"structural", "full"} else None
        )
        strict_scope = (
            directive_mode != "baseline"
            if strict_metric_scope is None
            else strict_metric_scope
        )
        candidate_k = (
            min(100, max(top_k, top_k * 3))
            if directive_mode != "baseline" or strict_scope
            else top_k
        )
        results = retriever.search(
            retrieval_query,
            top_k=candidate_k,
            directive_plan=search_directives,
            strict_metric_scope=strict_scope,
        )
        if directive_mode in {"validate", "full"}:
            results, ranking_applied = ValidateOperator.rerank(results, directives)
            directive_activity["validation_reranked_queries"] += int(ranking_applied)
        if directive_mode in {"filter", "full"}:
            before_filter = len(results)
            results, filter_applied = FilterOperator.apply(results, directives, query)
            if filter_applied:
                directive_activity["filter_removed_candidates"] += before_filter - len(results)
        results = results[:top_k]
        retrieved_evidence = [
            (Path(result.chunk.source_file).name, result.chunk.page)
            for result in results
            if result.chunk.page is not None
        ]
        _, citations = assembler.assemble(results, retriever.chunks_by_id)
        cited_evidence = [
            (Path(retriever.chunks_by_id[citation.chunk_id].source_file).name, citation.page)
            for citation in citations
            if citation.page is not None and citation.chunk_id in retriever.chunks_by_id
        ]
        expected_values = example.get("expected_values")
        value_hit = None
        if expected_values:
            value_hit = all(
                any(
                    _value_matches(result.chunk, str(expected_value), str(year))
                    for result in results
                )
                for year, expected_value in expected_values.items()
            )
        elif example.get("expected_value"):
            expected_value = str(example["expected_value"])
            target_year = (
                str(query.fiscal_year - 1)
                if query.fiscal_year is not None
                and query.period_type == PeriodType.BEGINNING_OF_YEAR
                else str(query.fiscal_year)
                if query.fiscal_year is not None
                else None
            )
            value_hit = any(
                _value_matches(result.chunk, expected_value, target_year)
                for result in results
            )
        expected_unit = example.get("expected_unit")
        unit_match = None
        if expected_unit:
            unit_match = any(
                _unit_matches(result.chunk.unit, expected_unit)
                for result in results
            )

        row = {
            "id": example["id"],
            "company_code": example.get("company_code", query.company_code),
            "fiscal_year": example.get("fiscal_year", query.fiscal_year),
            "fiscal_years": example.get("fiscal_years", query.fiscal_years),
            "fiscal_quarter": query.fiscal_quarter,
            "intent": query.intent.value,
            "split": split,
            "source_partition": source_partition,
            "answerable": answerable,
            "query": example["query"],
            "retrieved_pages": [page for _, page in retrieved_evidence],
            "retrieved_sources": [{"source": source, "page": page} for source, page in retrieved_evidence],
            "cited_pages": [page for _, page in cited_evidence],
            "cited_sources": [{"source": source, "page": page} for source, page in cited_evidence],
            "evidence_value_hit": value_hit,
            "evidence_unit_match": unit_match,
        }
        if answerable:
            row.update({
                "recall_at_k": RetrievalEvaluator.recall_at_k(retrieved_evidence, relevant_evidence, top_k),
                "precision_at_k": RetrievalEvaluator.precision_at_k(retrieved_evidence, relevant_evidence, top_k),
                "precision_of_returned_results": RetrievalEvaluator.precision_of_returned_results(
                    retrieved_evidence, relevant_evidence
                ),
                "mrr": RetrievalEvaluator.reciprocal_rank(retrieved_evidence, relevant_evidence),
                "citation_page_recall": CitationEvaluator.page_recall(cited_evidence, relevant_evidence),
                "citation_page_precision": CitationEvaluator.page_precision(cited_evidence, relevant_evidence),
                "no_answer_correct": None,
            })
        else:
            row.update({
                "recall_at_k": None,
                "precision_at_k": None,
                "precision_of_returned_results": None,
                "mrr": None,
                "citation_page_recall": None,
                "citation_page_precision": None,
                "no_answer_correct": not results,
            })
        rows.append(row)

    summary: dict[str, Any] = {
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256_file(dataset_path),
        "sources": [str(path) for path in source_paths],
        "source_sha256": [
            {"source": str(path), "sha256": _sha256_file(path)}
            for path in source_paths
        ],
        "sources_by_partition": source_names_by_partition,
        "cases": len(rows),
        "answerable_cases": sum(row["answerable"] for row in rows),
        "no_answer_cases": sum(not row["answerable"] for row in rows),
        "companies": sorted({row["company_code"] for row in rows if row["company_code"]}),
        "answerable_companies": sorted({
            row["company_code"] for row in rows if row["answerable"] and row["company_code"]
        }),
        "fiscal_years": sorted({row["fiscal_year"] for row in rows if row["fiscal_year"]}),
        "answerable_fiscal_years": sorted({
            row["fiscal_year"] for row in rows if row["answerable"] and row["fiscal_year"]
        }),
        "fiscal_quarters": sorted({
            row["fiscal_quarter"] for row in rows if row["fiscal_quarter"]
        }),
        "answerable_fiscal_quarters": sorted({
            row["fiscal_quarter"]
            for row in rows
            if row["answerable"] and row["fiscal_quarter"]
        }),
        "top_k": top_k,
        "dense_enabled": enable_dense,
        "metadata_filters_enabled": apply_metadata_filters,
        "directive_mode": directive_mode,
        "directive_suffix": directive_suffix,
        "directive_activity": directive_activity,
        "strict_metric_scope": (
            directive_mode != "baseline"
            if strict_metric_scope is None
            else strict_metric_scope
        ),
    }
    retrieval_metrics = (
        "recall_at_k",
        "precision_at_k",
        "precision_of_returned_results",
        "mrr",
        "citation_page_recall",
        "citation_page_precision",
    )
    _add_split_metrics(summary, rows, retrieval_metrics)
    _add_group_metrics(
        summary,
        rows,
        retrieval_metrics,
        group_field="source_partition",
        output_key="by_source_partition",
    )
    no_answer_rows = [row for row in rows if not row["answerable"]]
    if no_answer_rows:
        summary["no_answer_accuracy"] = round(
            fmean(row["no_answer_correct"] for row in no_answer_rows), 4
        )
    value_rows = [
        row["evidence_value_hit"]
        for row in rows
        if row["answerable"] and row["evidence_value_hit"] is not None
    ]
    if value_rows:
        summary["evidence_value_hit_rate"] = round(sum(value_rows) / len(value_rows), 4)
    unit_rows = [
        row["evidence_unit_match"]
        for row in rows
        if row["answerable"] and row["evidence_unit_match"] is not None
    ]
    if unit_rows:
        summary["evidence_unit_match_rate"] = round(sum(unit_rows) / len(unit_rows), 4)
    return {"summary": summary, "results": rows}


def _without_metadata_scope(query: ParsedQuery) -> ParsedQuery:
    """Keep query understanding intact while disabling only corpus metadata filters."""
    return query.model_copy(update={
        "company_code": None,
        "fiscal_year": None,
        "fiscal_years": [],
        "fiscal_quarter": None,
        "report_type": None,
        "statement_type": None,
    })


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _add_split_metrics(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    metrics: tuple[str, ...],
) -> None:
    answerable_rows = [row for row in rows if row["answerable"]]
    for metric in metrics:
        values = [row[metric] for row in answerable_rows if row[metric] is not None]
        if values:
            summary[metric] = round(fmean(values), 4)

    _add_group_metrics(summary, rows, metrics, group_field="split", output_key="by_split")


def _add_group_metrics(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    metrics: tuple[str, ...],
    *,
    group_field: str,
    output_key: str,
) -> None:
    summary[output_key] = {}
    for group in sorted({row[group_field] for row in rows}):
        group_rows = [row for row in rows if row[group_field] == group]
        answerable_rows = [row for row in group_rows if row["answerable"]]
        no_answer_rows = [row for row in group_rows if not row["answerable"]]
        group_summary: dict[str, Any] = {
            "cases": len(group_rows),
            "answerable_cases": len(answerable_rows),
            "no_answer_cases": len(no_answer_rows),
        }
        for metric in metrics:
            values = [row[metric] for row in answerable_rows if row[metric] is not None]
            if values:
                group_summary[metric] = round(fmean(values), 4)
        if no_answer_rows:
            group_summary["no_answer_accuracy"] = round(
                fmean(row["no_answer_correct"] for row in no_answer_rows), 4
            )
        for row_metric, summary_metric in (
            ("evidence_value_hit", "evidence_value_hit_rate"),
            ("evidence_unit_match", "evidence_unit_match_rate"),
        ):
            values = [row[row_metric] for row in answerable_rows if row[row_metric] is not None]
            if values:
                group_summary[summary_metric] = round(sum(values) / len(values), 4)
        summary[output_key][group] = group_summary


def _value_matches(chunk, expected_value: str, target_year: str | None) -> bool:
    values = chunk.metadata.get("period_values", {})
    if target_year and isinstance(values, dict) and target_year in values:
        return str(values[target_year]) == expected_value
    return expected_value in chunk.content


def _unit_matches(actual: str | None, expected: str) -> bool:
    def normalize(value: str) -> str:
        return "".join(char for char in value.casefold() if char.isalnum())

    aliases = {
        "vnd": "vnd",
        "vnđ": "vnd",
        "đồng": "vnd",
        "triệuđồng": "triệuvnd",
        "vnđcổphiếu": "vndshare",
        "vndcophieu": "vndshare",
        "vndshare": "vndshare",
        "vndpershare": "vndshare",
    }
    return aliases.get(normalize(actual or ""), normalize(actual or "")) == aliases.get(
        normalize(expected), normalize(expected)
    )


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Evaluate financial retrieval and citation quality")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=root / "evaluation" / "demo_cases.jsonl",
        help="JSONL labels; the default contains only fictional demo cases",
    )
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        default=None,
        help="Source Markdown file; pass multiple times for a multi-document benchmark",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense", action="store_true", help="Enable TEI-backed E5 embeddings")
    parser.add_argument(
        "--disable-metadata-filters",
        action="store_true",
        help="Ablation mode: retrieve without company/period/report/statement scoping",
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    print(json.dumps(
        run_benchmark(
            args.dataset,
            args.source or [root / "sample_data" / "DEMO_SYNTHETIC_2025.md"],
            args.top_k,
            args.dense,
            not args.disable_metadata_filters,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
