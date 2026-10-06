"""Qdrant integration fixture; checks storage/filtering, not embedding quality."""
from __future__ import annotations

import json
import os
from threading import Event
import uuid

from financial_rag.retrieval.qdrant_dense import QdrantDenseRetriever
from financial_rag.schemas import FinancialChunk


class FixtureEmbedding:
    calls = 0

    def embed_passages(self, texts):
        self.calls += len(texts)
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, query):
        return [1.0, 0.0, 0.0]


def main():
    from qdrant_client import QdrantClient
    client = QdrantClient(url=os.environ["QDRANT_URL"], timeout=60)
    name = "financial_search_test_" + uuid.uuid4().hex
    embedding = FixtureEmbedding()
    chunks = [FinancialChunk(chunk_id=company, source_file=company + ".txt", company_code=company,
                              fiscal_year=2025, report_type="consolidated", content="identical text")
              for company in ("VPB", "ACB", "MBB")]
    try:
        first = QdrantDenseRetriever(chunks, embedding, client=client,
                                     namespace="fixture", collection=name)
        first.index(batch_size=2)
        assert first.index_status == "ready" and first._normalized_vectors is None
        assert client.count(name, exact=True).count == 3  # Same text, three citation identities.
        assert first.search("assets", allowed_ids={"ACB"}) == [("ACB", 1.0)]
        second = QdrantDenseRetriever(chunks[:2], embedding, client=client,
                                      namespace="fixture", collection=name)
        second.index()
        assert embedding.calls == 3 and second.cached_vector_count == 2
        assert {row[0] for row in second.search("assets")} == {"VPB", "ACB"}
        assert {row[0] for row in first.search("assets")} == {"VPB", "ACB", "MBB"}
        changed = chunks[0].model_copy(update={"content": "revised evidence"})
        third = QdrantDenseRetriever([changed], embedding, client=client,
                                     namespace="fixture", collection=name)
        third.index()
        assert embedding.calls == 4 and client.count(name, exact=True).count == 4
        assert third.search("assets") == [("VPB", 1.0)]
        cancelled = QdrantDenseRetriever(chunks, embedding, client=client,
                                         namespace="fixture", collection=name)
        event = Event()
        event.set()
        cancelled.index(cancel_event=event)
        assert cancelled.index_status == "cancelled" and cancelled.search("assets") == []
        print(json.dumps({"passed": True, "embedding_quality_evaluated": False,
                          "checks": ["same_text_distinct_issuers", "server_filter", "incremental_reuse",
                                     "snapshot_isolation", "source_revision", "cancelled_not_searchable",
                                     "no_local_vector_matrix"]}))
    finally:
        assert name.startswith("financial_search_test_")
        if client.collection_exists(name):
            client.delete_collection(name)
        client.close()


if __name__ == "__main__":
    main()
