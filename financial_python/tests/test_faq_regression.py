"""Grounded answer regressions over a small, fully fictional report."""

from __future__ import annotations

from pathlib import Path

import pytest

from financial_rag.chunking.financial_chunker import FinancialMarkdownChunker
from financial_rag.context.assembler import ContextAssembler
from financial_rag.generation.chat import FinancialChatEngine
from financial_rag.query.understanding import QueryUnderstandingEngine
from financial_rag.retrieval.hybrid import FinancialHybridRetriever
from financial_rag.validation.citation import CitationValidator
from financial_rag.validation.numeric import NumericValidator


DEMO_PATH = Path(__file__).resolve().parents[1] / "sample_data" / "DEMO_SYNTHETIC_2025.md"


@pytest.fixture(scope="module")
def faq_runtime():
    chunks = FinancialMarkdownChunker().chunk_file(DEMO_PATH, company_code="DEMO")
    return {
        "retriever": FinancialHybridRetriever(chunks, enable_dense=False),
        "chunks_by_id": {chunk.chunk_id: chunk for chunk in chunks},
        "query_engine": QueryUnderstandingEngine(),
        "answer_engine": FinancialChatEngine(),
        "assembler": ContextAssembler(),
    }


def test_exact_lookup_uses_synthetic_source_value_and_citation(faq_runtime):
    query = faq_runtime["query_engine"].parse("Doanh thu thuần của DEMO năm 2025 là bao nhiêu?")
    retriever = faq_runtime["retriever"]
    results = retriever.search(query, top_k=5)
    _context, citations, source_contexts = faq_runtime["assembler"].assemble_with_sources(
        results,
        faq_runtime["chunks_by_id"],
        max_chars=4_000,
    )

    answer = faq_runtime["answer_engine"]._source_exact_metric_answer(query, results, citations)

    assert answer is not None
    assert "900 Triệu đồng" in answer
    assert "[c1]" in answer
    assert NumericValidator.validate(answer, source_contexts).is_valid
    assert CitationValidator.validate(answer, citations)
    assert {citation.page for citation in citations if "[c1]" in answer} == {11}


def test_unknown_metric_does_not_reuse_an_unrelated_demo_value(faq_runtime):
    query = faq_runtime["query_engine"].parse(
        "Tiền gửi của khách hàng DEMO cuối năm 2025 là bao nhiêu?"
    )

    assert faq_runtime["retriever"].search(query, top_k=5) == []
