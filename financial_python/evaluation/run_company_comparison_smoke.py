"""Check whether one company comparison retrieves both values and their source pages."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from financial_rag.chunking.txt_financial_chunker import TxtFinancialChunker
from financial_rag.embedding.e5_adapter import E5EmbeddingAdapter
from financial_rag.query.understanding import QueryUnderstandingEngine
from financial_rag.retrieval.hybrid import FinancialHybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense", action="store_true", help="Enable local E5 dense retrieval")
    parser.add_argument("--output", type=Path, required=True, help="JSON output path")
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="CODE=PATH",
        help="Input TXT report and company code; repeat once per company",
    )
    parser.add_argument(
        "--expected",
        action="append",
        required=True,
        metavar="CODE=VALUE:PAGE",
        help="Expected fictional or local value and page; repeat once per company",
    )
    parser.add_argument("--query", required=True, help="Comparison query to run")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sources = []
    for item in args.source:
        company, separator, path_text = item.partition("=")
        if not separator or not company or not path_text:
            parser.error(f"Invalid --source {item!r}; expected CODE=PATH")
        path = Path(path_text)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            parser.error(f"Source file does not exist: {path}")
        sources.append((path, company.upper()))

    expected = {}
    for item in args.expected:
        company, separator, details = item.partition("=")
        value, value_separator, page = details.rpartition(":")
        if not separator or not value_separator or not company or not value:
            parser.error(f"Invalid --expected {item!r}; expected CODE=VALUE:PAGE")
        try:
            page_number = int(page)
        except ValueError:
            parser.error(f"Invalid page number in --expected {item!r}")
        expected[company.upper()] = {"value": value, "page": page_number}

    source_companies = {company for _, company in sources}
    if source_companies != set(expected):
        parser.error("Provide exactly one --expected value for every --source company")
    chunker = TxtFinancialChunker()
    chunks = []
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    for path, company in sources:
        chunks.extend(chunker.chunk_file(path, company_code=company))
    chunking_seconds = time.perf_counter() - started

    query = QueryUnderstandingEngine().parse(args.query)
    search_started = time.perf_counter()
    retriever = FinancialHybridRetriever(chunks, enable_dense=args.dense)
    index_seconds = time.perf_counter() - search_started
    search_started = time.perf_counter()
    results = retriever.search(query, top_k=5)
    search_seconds = time.perf_counter() - search_started

    result_rows = []
    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        result_rows.append(
            {
                "rank": rank,
                "company_code": chunk.company_code,
                "source_file": chunk.source_file,
                "page": chunk.page,
                "canonical_metric": chunk.canonical_metric,
                "chunk_type": chunk.chunk_type.value,
                "unit": chunk.unit,
                "period_values": chunk.metadata.get("period_values"),
                "content_excerpt": chunk.content[:400],
            }
        )

    company_hits = {}
    for company, expected_row in expected.items():
        company_rows = [row for row in result_rows if row["company_code"] == company]
        value_hit = any(
            expected_row["value"] in row["content_excerpt"]
            or expected_row["value"] in (row["period_values"].values() if isinstance(row["period_values"], dict) else [])
            for row in company_rows
        )
        page_hit = any(row["page"] == expected_row["page"] for row in company_rows)
        value_and_page_hit = any(
            row["page"] == expected_row["page"]
            and (
                expected_row["value"] in row["content_excerpt"]
                or expected_row["value"] in (row["period_values"].values() if isinstance(row["period_values"], dict) else [])
            )
            for row in company_rows
        )
        company_hits[company] = {
            "expected_value": expected_row["value"],
            "expected_page": expected_row["page"],
            "value_hit_in_top5": value_hit,
            "page_hit_in_top5": page_hit,
            "value_and_page_hit_in_top5": value_and_page_hit,
        }

    embedding_adapter = E5EmbeddingAdapter()
    total_wall_seconds = time.perf_counter() - started
    output_data = {
        "started_at_utc": started_at,
        "mode": "BM25 + E5 dense + RRF" if args.dense else "BM25 only",
        "llm_calls": 0,
        "query": args.query,
        "parsed_query": query.model_dump(mode="json"),
        "top_k": 5,
        "success": all(hit["value_and_page_hit_in_top5"] for hit in company_hits.values()),
        "company_hits": company_hits,
        "chunks": {"count": len(chunks), "by_company": {code: sum(c.company_code == code for c in chunks) for code in expected}},
        "timing_seconds": {
            "total_wall": round(total_wall_seconds, 4),
            "chunking": round(chunking_seconds, 4),
            "embedding_index": round(index_seconds, 4) if args.dense else 0.0,
            "query_retrieval_including_query_embedding": round(search_seconds, 6),
        },
        "embedding": {
            "enabled": args.dense,
            "endpoint": embedding_adapter._endpoint() if args.dense else None,
            "model": embedding_adapter.model if args.dense else None,
        },
        "source_files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "company_code": company,
            }
            for path, company in sources
        ],
        "results": result_rows,
        "limitations": [
            "Single query over two extracted TXT documents; this is a targeted retrieval smoke, not a broad accuracy estimate.",
            "The result measures only the supplied local source files, expected values, pages, and query.",
        ],
    }
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "success": output_data["success"], "company_hits": company_hits, "timing_seconds": output_data["timing_seconds"], "embedding": output_data["embedding"], "results": result_rows}, ensure_ascii=False, indent=2))
    if not output_data["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
