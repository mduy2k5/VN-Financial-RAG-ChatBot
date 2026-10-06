"""Evaluate deterministic financial answers over the manually labeled 88-query set.

This measures structured numeric answers and citation wiring only. It makes no
Ollama calls and must be read alongside the retrieval benchmark and the pending
human review of the parser-derived corpus stress cases.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from statistics import fmean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from financial_rag.chunking.financial_chunker import FinancialMarkdownChunker
from financial_rag.context.assembler import ContextAssembler
from financial_rag.generation.chat import FinancialChatEngine
from financial_rag.evaluation.local_data import local_data_path
from financial_rag.query.understanding import QueryUnderstandingEngine
from financial_rag.retrieval.hybrid import FinancialHybridRetriever
from financial_rag.validation.citation import CitationValidator
from financial_rag.validation.numeric import NumericValidator


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _value_matches(answer: str, expected: str) -> bool:
    wanted = NumericValidator.normalize_number(expected)
    return bool(wanted) and any(
        NumericValidator.normalize_number(value) == wanted
        for value in NumericValidator.extract_numbers(answer)
    )


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row["answerable"]]
    scored = [row for row in answerable if row["answer"] is not None]
    no_answer = [row for row in rows if not row["answerable"]]
    expected_pages = sum(row["expected_page_count"] for row in answerable)
    hit_pages = sum(row["expected_page_hits"] for row in answerable)
    cited_pages = sum(row["cited_page_count"] for row in answerable)
    return {
        "cases": len(rows),
        "answerable_cases": len(answerable),
        "direct_answer_cases": len(scored),
        "direct_answer_coverage": round(len(scored) / len(answerable), 4) if answerable else None,
        "answer_validation_rate": round(
            fmean(row["answer_valid"] for row in scored), 4
        ) if scored else None,
        "exact_value_hit_rate": round(
            fmean(row["exact_value_hit"] for row in scored), 4
        ) if scored else None,
        "expected_unit_match_rate": round(
            fmean(row["expected_unit_match"] for row in scored if row["expected_unit_match"] is not None), 4
        ) if any(row["expected_unit_match"] is not None for row in scored) else None,
        "citation_id_valid_rate": round(
            fmean(row["citation_valid"] for row in scored), 4
        ) if scored else None,
        "expected_page_recall": round(hit_pages / expected_pages, 4) if expected_pages else None,
        "cited_page_precision": round(hit_pages / cited_pages, 4) if cited_pages else None,
        "no_answer_cases": len(no_answer),
        "no_answer_accuracy": round(
            fmean(row["no_answer_correct"] for row in no_answer), 4
        ) if no_answer else None,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Evaluate direct numeric answers, value/unit validation, and citation pages"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "evaluation" / "results" / "structured_answer_88_2026-09-27.json",
        help="Output JSON path",
    )
    # The API expands an ordinary direct-metric request to four times its
    # requested top_k (the UI default is 5). Use the same candidate budget here.
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    source_paths = [
        local_data_path("sample_data", "ABB_BCTC_2025.md"),
        local_data_path("evaluation", "sources", "FPT_Annual_Report_2024_benchmark.md"),
        local_data_path("evaluation", "sources", "FPT_Annual_Report_2025_benchmark.md"),
        local_data_path("evaluation", "sources", "HPG_Annual_Report_2025_benchmark.md"),
        local_data_path("evaluation", "sources", "FPT_Q2_2026_benchmark.md"),
        local_data_path("evaluation", "sources", "HPG_Q2_2025_benchmark.md"),
        local_data_path("evaluation", "sources", "VNF_Interim_Consolidated_2025H1_benchmark.md"),
    ]
    dataset_path = local_data_path("evaluation", "ground_truth_multi_company.jsonl")
    examples = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    source_names = {path.name for path in source_paths}
    chunker = FinancialMarkdownChunker()
    chunks = []
    for path in source_paths:
        chunks.extend(
            chunker.chunk_file(
                path,
                company_code=path.name.split("_", maxsplit=1)[0].upper(),
                source_pdf=path.name.replace("_benchmark.md", ".pdf"),
            )
        )

    document_disjoint_sources = {
        Path(item["source"]).name
        for example in examples
        if example.get("source_partition") == "document_disjoint"
        for item in example.get("relevant_sources", [])
    }
    missing_sources = document_disjoint_sources - source_names
    if missing_sources:
        raise ValueError("Missing document-disjoint sources: " + ", ".join(sorted(missing_sources)))

    parser_engine = QueryUnderstandingEngine()
    answer_engine = FinancialChatEngine()
    assembler = ContextAssembler()
    retrievers: dict[str, FinancialHybridRetriever] = {}
    output_rows = []

    for example in examples:
        query = parser_engine.parse(example["query"])
        split = example.get("split", "dev")
        partition = example.get(
            "source_partition", "development" if split == "dev" else "shared_documents"
        )
        if partition not in retrievers:
            partition_chunks = (
                chunks
                if partition == "document_disjoint"
                else [chunk for chunk in chunks if Path(chunk.source_file).name not in document_disjoint_sources]
            )
            retrievers[partition] = FinancialHybridRetriever(partition_chunks)
        retriever = retrievers[partition]
        results = retriever.search(query, top_k=args.top_k)
        answerable = example.get("answerable", True)
        row: dict[str, Any] = {
            "id": example["id"],
            "query": example["query"],
            "split": split,
            "source_partition": partition,
            "intent": query.intent.value,
            "answerable": answerable,
            "retrieved_count": len(results),
            "answer": None,
            "answer_valid": False,
            "numeric_valid": False,
            "unit_valid": False,
            "citation_valid": False,
            "exact_value_hit": False,
            "expected_unit_match": None,
            "expected_page_count": 0,
            "expected_page_hits": 0,
            "cited_page_count": 0,
            "no_answer_correct": None,
        }

        if not answerable:
            # This offline experiment measures whether the deterministic
            # financial answer path can produce a grounded numeric answer.
            # Retrieval may still return nearby context for an unsupported
            # query; that is not an answer unless the formatter emits a value.
            no_answer = None
            if results:
                _context, no_answer_citations, _source_contexts = assembler.assemble_with_sources(
                    results, retriever.chunks_by_id
                )
                no_answer = answer_engine._structured_financial_answer(
                    query, results, no_answer_citations
                )
            row["no_answer_answer"] = no_answer
            row["no_answer_correct"] = not (
                no_answer
                and NumericValidator.extract_numbers(no_answer)
            )
            output_rows.append(row)
            continue
        if not results:
            output_rows.append(row)
            continue

        _context, citations, source_contexts = assembler.assemble_with_sources(
            results, retriever.chunks_by_id
        )
        answer = answer_engine._structured_financial_answer(query, results, citations)
        if answer is None:
            row["fallback_required"] = True
            output_rows.append(row)
            continue

        numeric_result = NumericValidator.validate(answer, source_contexts)
        citation_valid = CitationValidator.validate(answer, citations)
        row["answer"] = answer
        row["numeric_valid"] = numeric_result.is_valid
        row["unit_valid"] = numeric_result.unit_match
        row["citation_valid"] = citation_valid
        row["answer_valid"] = numeric_result.is_valid and numeric_result.unit_match and citation_valid

        expected_value = example.get("expected_value")
        expected_values = example.get("expected_values") or {}
        if expected_value:
            expected_values = {**expected_values, str(query.fiscal_year or ""): expected_value}
        expected_number_checks = [
            _value_matches(answer, str(value)) for value in expected_values.values() if value is not None
        ]
        row["expected_value_checks"] = expected_number_checks
        row["exact_value_hit"] = bool(expected_number_checks) and all(expected_number_checks)

        expected_unit = example.get("expected_unit")
        if expected_unit:
            expected = NumericValidator._normalize_unit(expected_unit)
            cited_units = [
                NumericValidator._normalize_unit(citation.unit)
                for citation in citations
                if citation.unit
            ]
            row["expected_unit_match"] = expected in cited_units

        expected_sources = {
            (Path(item["source"]).name, int(item["page"]))
            for item in example.get("relevant_sources", [])
        }
        if not expected_sources:
            expected_sources = {
                (source_paths[0].name, int(page))
                for page in example.get("relevant_pages", [])
            }
        used_ids = {
            citation_id.casefold()
            for citation_id in re.findall(r"\[(c?\d+)\]", answer, flags=re.IGNORECASE)
        }
        cited_sources = {
            (Path(retriever.chunks_by_id[citation.chunk_id].source_file).name, citation.page)
            for citation in citations
            if citation.citation_id.casefold() in used_ids
            and citation.page is not None
            and citation.chunk_id in retriever.chunks_by_id
        }
        row["expected_page_count"] = len(expected_sources)
        row["expected_page_hits"] = len(expected_sources & cited_sources)
        row["cited_page_count"] = len(cited_sources)
        output_rows.append(row)

    summary = _aggregate(output_rows)
    summary["by_split"] = {
        split: _aggregate([row for row in output_rows if row["split"] == split])
        for split in sorted({row["split"] for row in output_rows})
    }
    summary["by_intent"] = {
        intent: _aggregate([row for row in output_rows if row["intent"] == intent])
        for intent in sorted({row["intent"] for row in output_rows})
    }
    output_path = args.output if args.output.is_absolute() else root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "run_metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": str(dataset_path),
            "dataset_sha256": _sha256(dataset_path),
            "sources": [
                {"path": str(path), "sha256": _sha256(path)} for path in source_paths
            ],
            "top_k": args.top_k,
            "answer_mode": "deterministic financial rows only; explanatory LLM path not called",
            "llm_calls": 0,
            "note": (
                "top_k matches the API's direct-metric candidate budget. No-answer accuracy means "
                "the deterministic formatter did not emit a number; it does not grade a generated "
                "LLM refusal. Read with retrieval metrics and the separate FAQ regression gate."
            ),
        },
        "summary": summary,
        "results": output_rows,
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
