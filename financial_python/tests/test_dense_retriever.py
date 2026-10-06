from math import isclose, sqrt

from financial_rag.retrieval.dense_retriever import DenseRetriever
from financial_rag.schemas import FinancialChunk


class StubEmbeddingAdapter:
    def __init__(self, passage_vectors, query_vector):
        self.passage_vectors = passage_vectors
        self.query_vector = query_vector
        self.next_vector = 0

    def embed_passages(self, passages):
        start = self.next_vector
        self.next_vector += len(passages)
        return self.passage_vectors[start:self.next_vector]

    def embed_query(self, query):
        return self.query_vector


def test_dense_retriever_normalizes_once_and_filters_before_top_k():
    chunks = [
        FinancialChunk(chunk_id="x", source_file="x.md", content="X"),
        FinancialChunk(chunk_id="y", source_file="y.md", content="Y"),
        FinancialChunk(chunk_id="z", source_file="z.md", content="Z"),
        FinancialChunk(chunk_id="zero", source_file="zero.md", content="Zero vector"),
    ]
    adapter = StubEmbeddingAdapter(
        passage_vectors=[[10, 0], [0, 5], [1, 1], [0, 0]],
        query_vector=[0, 3],
    )
    retriever = DenseRetriever(chunks, embed_adapter=adapter)
    retriever.index(batch_size=2)
    indexed_matrix = retriever._normalized_vectors

    assert indexed_matrix is not None
    assert retriever.search("asset", top_k=1)[0][0] == "y"
    scoped_results = retriever.search("asset", top_k=1, allowed_ids={"x", "z", "zero"})
    assert scoped_results[0][0] == "z"
    # Vectors are stored as float32; allow the expected conversion rounding.
    assert isclose(scoped_results[0][1], sqrt(0.5), rel_tol=1e-7)
    assert retriever._normalized_vectors is indexed_matrix


def test_dense_retriever_returns_no_results_for_empty_candidate_scope():
    adapter = StubEmbeddingAdapter(passage_vectors=[[1, 0]], query_vector=[1, 0])
    retriever = DenseRetriever(
        [FinancialChunk(chunk_id="only", source_file="only.md", content="Only")],
        embed_adapter=adapter,
    )
    retriever.index()

    assert retriever.search("query", allowed_ids=set()) == []
