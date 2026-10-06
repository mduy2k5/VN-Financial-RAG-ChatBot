"""Exercise real Mongo transactions in a temporary, explicitly isolated database."""
from __future__ import annotations

import json
import os
import uuid

from financial_rag.schemas import ChunkType, FinancialChunk
from financial_rag.storage.mongo_store import MongoFinancialStore


def document(source, value):
    return {"logical_key": "integration_test", "document_id": "fixture", "source_file": "fixture.txt",
            "company_code": "FIXTURE", "fiscal_year": 2025, "raw_content": source,
            "chunks": [FinancialChunk(chunk_id="fixture_row", source_file="fixture.txt", fiscal_year=2025,
                                       content="fixture evidence", unit="Triệu đồng",
                                       chunk_type=ChunkType.FINANCIAL_ITEM,
                                       metadata={"metric_name": "Test metric", "period_values": {"2025": value}})]}


def main():
    name = "financial_normalized_test_" + uuid.uuid4().hex
    store = MongoFinancialStore(os.environ["MONGODB_URI"], name)
    try:
        store.replace_documents([document("v1", "100")])
        first = store.list_documents()[0]["active_projection_id"]
        store.replace_documents([document("v2", "120")])
        assert len(store.list_versions("integration_test")) == 2
        assert store.list_facts("integration_test")["facts"][0]["value"] == "120"
        assert store.list_facts("integration_test", projection_id=first)["facts"][0]["value"] == "100"
        store.replace_documents([document("v2", "120")])
        assert store.projections.count_documents({}) == 2
        try:
            store.list_facts("wrong_scope", projection_id=first)
        except ValueError:
            pass
        else:
            raise AssertionError("A foreign projection was accepted")
        invalid = document("v3", "999")
        invalid["chunks"] *= 2
        try:
            store.replace_documents([invalid])
        except Exception:
            pass
        else:
            raise AssertionError("Duplicate chunks must fail atomically")
        assert len(store.list_versions("integration_test")) == 2
        assert store.list_facts("integration_test")["facts"][0]["value"] == "120"
        assert store.delete_document("integration_test")
        assert store.list_facts("integration_test")["total"] == 0
        assert len(store.list_versions("integration_test")) == 2
        print(json.dumps({"passed": True, "backend": "MongoDB replica set",
                          "checks": ["version_retention", "active_projection", "idempotency",
                                     "scope_isolation", "transaction_rollback", "delete_keeps_history"]}))
    finally:
        assert name.startswith("financial_normalized_test_") and name != "financial_rag"
        store.client.drop_database(name)
        store.client.close()


if __name__ == "__main__":
    main()
