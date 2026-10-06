from financial_rag.retrieval.hybrid import FinancialHybridRetriever
from financial_rag.schemas import FinancialChunk, ParsedQuery


def chunks():
    return [FinancialChunk(chunk_id=company, source_file=company + ".txt", company_code=company,
                           fiscal_year=2025, content="Tổng tài sản: 123 Triệu đồng")
            for company in ("VPB", "FPT", "MBB")]


def test_alternating_issuers_reuse_exact_bm25_scope():
    retriever = FinancialHybridRetriever(chunks())
    for company in ("VPB", "FPT", "VPB"):
        results = retriever.search(ParsedQuery(original_query="Tổng tài sản", company_code=company,
                                               company_codes=[company], fiscal_year=2025))
        assert [result.chunk.company_code for result in results] == [company]
    assert retriever.bm25_cache_hits == 1 and retriever.bm25_cache_misses == 2


def test_chunk_budget_evicts_least_recent_scope(monkeypatch):
    monkeypatch.setenv("FINANCIAL_BM25_CACHE_CHUNKS", "2")
    retriever = FinancialHybridRetriever(chunks())
    first, second, third = [[chunk] for chunk in retriever.chunks]
    index = retriever._get_bm25_retriever(first)
    retriever._get_bm25_retriever(second)
    assert retriever._get_bm25_retriever(first) is index
    retriever._get_bm25_retriever(third)
    assert retriever._bm25_cached_chunks == 2 and len(retriever._bm25_scopes) == 2
    assert retriever._get_bm25_retriever(first) is index
    retriever._get_bm25_retriever(retriever.chunks)
    assert retriever._bm25_cached_chunks == 2
    assert len(retriever._bm25_scopes) == 2 and retriever._bm25_cache is None
