"""Compare real E5 server/local dense scores on a bounded source sample."""
from __future__ import annotations

import json
import os
import uuid

from financial_rag.retrieval.dense_retriever import DenseRetriever
from financial_rag.retrieval.qdrant_dense import QdrantDenseRetriever
from financial_rag.schemas import ChunkType, FinancialChunk


def main():
    from pymongo import MongoClient
    from qdrant_client import QdrantClient
    with MongoClient(os.environ["MONGODB_URI"]) as mongo:
        database = mongo[os.getenv("FINANCIAL_MONGO_DATABASE", "financial_rag")]
        chunks = []
        for company in ("VPB", "ACB", "MBB"):
            documents = database.financial_documents.find(
                {"company_code": company, "fiscal_year": 2025, "report_type": "consolidated"})
            doc = next(documents)
            candidates = (FinancialChunk.model_validate(row["payload"]) for row in
                          database.financial_chunks.find({"logical_key": doc["_id"]}).sort("ordinal", 1))
            count = 0
            for chunk in candidates:
                if chunk.chunk_type == ChunkType.FINANCIAL_ITEM:
                    chunks.append(chunk)
                    count += 1
                    if count == 8:
                        break
    client = QdrantClient(url=os.environ["QDRANT_URL"], timeout=60)
    name = "financial_search_e5_test_" + uuid.uuid4().hex
    try:
        local = DenseRetriever(chunks)
        local.index(batch_size=8)
        remote = QdrantDenseRetriever(chunks, client=client, collection=name)
        remote.index(batch_size=8)
        assert local.index_status == remote.index_status == "ready"
        rows = []
        for company in ("VPB", "ACB", "MBB"):
            allowed = {chunk.chunk_id for chunk in chunks if chunk.company_code == company}
            query = f"Tiền và tài sản của {company} năm 2025"
            expected = dict(local.search(query, top_k=8, allowed_ids=allowed))
            actual = dict(remote.search(query, top_k=8, allowed_ids=allowed))
            assert set(actual) == set(expected) == allowed
            error = max(abs(actual[key] - expected[key]) for key in actual)
            assert error < 0.00001, error
            rows.append({"company": company, "result_count": len(actual), "max_cosine_error": error})
        print(json.dumps({"passed": True, "real_model": local.embed_adapter.model,
                          "sample_chunk_count": len(chunks), "local_cached_vectors": local.cached_vector_count,
                          "remote_local_matrix": remote._normalized_vectors is not None,
                          "financial_semantic_accuracy_measured": False, "comparisons": rows}))
    finally:
        assert name.startswith("financial_search_e5_test_")
        if client.collection_exists(name):
            client.delete_collection(name)
        client.close()


if __name__ == "__main__":
    main()
