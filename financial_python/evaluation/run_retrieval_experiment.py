"""Run the labeled financial retrieval benchmark and capture stage timings."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
from statistics import mean
import time

from financial_rag.chunking.financial_chunker import FinancialMarkdownChunker
from financial_rag.embedding.e5_adapter import E5EmbeddingAdapter
from financial_rag.evaluation.benchmark import run_benchmark
from financial_rag.evaluation.local_data import local_data_path
from financial_rag.query.understanding import QueryUnderstandingEngine
from financial_rag.retrieval.hybrid import FinancialHybridRetriever


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Evaluate chunking, optional E5 embeddings, and retrieval without chat generation"
    )
    parser.add_argument("--dense", action="store_true", help="Enable local E5 dense retrieval")
    parser.add_argument(
        "--strict-metric-scope",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Match the API's source-row scoping for financial metric queries "
            "(disable only for a broad-retrieval ablation)"
        ),
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON output path")
    args = parser.parse_args()

    stats = {
        "chunking_seconds": 0.0,
        "chunk_file_calls": 0,
        "chunk_count": 0,
        "query_parse_seconds": 0.0,
        "query_parse_calls": 0,
        "search_seconds": [],
        "search_calls": 0,
        "embedding_seconds": 0.0,
        "embedding_calls": 0,
        "embedding_texts": 0,
        "passage_embedding_texts": 0,
        "query_embedding_texts": 0,
    }

    original_chunk_file = FinancialMarkdownChunker.chunk_file
    original_parse = QueryUnderstandingEngine.parse
    original_search = FinancialHybridRetriever.search
    original_embed = E5EmbeddingAdapter._call_embed

    def timed_chunk_file(self, *call_args, **call_kwargs):
        started = time.perf_counter()
        chunks = original_chunk_file(self, *call_args, **call_kwargs)
        stats["chunking_seconds"] += time.perf_counter() - started
        stats["chunk_file_calls"] += 1
        stats["chunk_count"] += len(chunks)
        return chunks

    def timed_parse(self, *call_args, **call_kwargs):
        started = time.perf_counter()
        parsed = original_parse(self, *call_args, **call_kwargs)
        stats["query_parse_seconds"] += time.perf_counter() - started
        stats["query_parse_calls"] += 1
        return parsed

    def timed_search(self, *call_args, **call_kwargs):
        started = time.perf_counter()
        results = original_search(self, *call_args, **call_kwargs)
        stats["search_seconds"].append(time.perf_counter() - started)
        stats["search_calls"] += 1
        return results

    def timed_embed(self, texts):
        started = time.perf_counter()
        vectors = original_embed(self, texts)
        stats["embedding_seconds"] += time.perf_counter() - started
        stats["embedding_calls"] += 1
        stats["embedding_texts"] += len(texts)
        if texts and all(str(text).lower().startswith("passage:") for text in texts):
            stats["passage_embedding_texts"] += len(texts)
        elif texts and all(str(text).lower().startswith("query:") for text in texts):
            stats["query_embedding_texts"] += len(texts)
        return vectors

    FinancialMarkdownChunker.chunk_file = timed_chunk_file
    QueryUnderstandingEngine.parse = timed_parse
    FinancialHybridRetriever.search = timed_search
    E5EmbeddingAdapter._call_embed = timed_embed

    sources = [
        local_data_path("sample_data", "ABB_BCTC_2025.md"),
        local_data_path("evaluation", "sources", "FPT_Annual_Report_2024_benchmark.md"),
        local_data_path("evaluation", "sources", "FPT_Annual_Report_2025_benchmark.md"),
        local_data_path("evaluation", "sources", "HPG_Annual_Report_2025_benchmark.md"),
        local_data_path("evaluation", "sources", "FPT_Q2_2026_benchmark.md"),
        local_data_path("evaluation", "sources", "HPG_Q2_2025_benchmark.md"),
        local_data_path("evaluation", "sources", "VNF_Interim_Consolidated_2025H1_benchmark.md"),
    ]
    dataset = local_data_path("evaluation", "ground_truth_multi_company.jsonl")
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    result = run_benchmark(
        dataset,
        sources,
        top_k=5,
        enable_dense=args.dense,
        apply_metadata_filters=True,
        strict_metric_scope=args.strict_metric_scope,
    )
    wall_seconds = time.perf_counter() - started

    search_times = sorted(stats["search_seconds"])
    p95_index = max(0, min(len(search_times) - 1, int(len(search_times) * 0.95)))
    model = os.getenv("FINANCIAL_EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
    embedding_base_url = os.getenv("FINANCIAL_EMBEDDING_URL", "http://e5:80")
    result["run_metadata"] = {
        "started_at_utc": started_at,
        "mode": "BM25 + multilingual E5 dense + RRF" if args.dense else "BM25 only",
        "metadata_filters_enabled": True,
        "strict_metric_scope": args.strict_metric_scope,
        "top_k": 5,
        "llm_calls": 0,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "embedding_endpoint": E5EmbeddingAdapter(base_url=embedding_base_url)._endpoint()
        if args.dense
        else None,
        "embedding_model": model if args.dense else None,
        "total_wall_seconds": round(wall_seconds, 4),
        "chunking_seconds": round(stats["chunking_seconds"], 4),
        "chunk_file_calls": stats["chunk_file_calls"],
        "chunk_count_all_sources": stats["chunk_count"],
        "query_parse_seconds": round(stats["query_parse_seconds"], 4),
        "query_parse_calls": stats["query_parse_calls"],
        "embedding_seconds_total": round(stats["embedding_seconds"], 4),
        "embedding_http_calls": stats["embedding_calls"],
        "embedding_texts_total": stats["embedding_texts"],
        "passage_embedding_texts": stats["passage_embedding_texts"],
        "query_embedding_texts": stats["query_embedding_texts"],
        "search_calls": stats["search_calls"],
        "search_seconds_total_including_query_embedding": round(sum(search_times), 4),
        "search_seconds_mean": round(mean(search_times), 6) if search_times else 0.0,
        "search_seconds_p95": round(search_times[p95_index], 6) if search_times else 0.0,
        "timing_note": (
            "Single local run; chunking, parsing, embedding, and search timings are stage timings. "
            "Query embedding is included in search timing, so those fields overlap."
        ),
    }

    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": result["summary"], "run_metadata": result["run_metadata"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
