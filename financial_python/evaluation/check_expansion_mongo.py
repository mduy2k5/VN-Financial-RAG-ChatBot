"""Real concurrent snapshot, fenced job recovery and rollback in an isolated Mongo DB."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import os
from threading import Event
import uuid

from financial_rag.schemas import FinancialChunk, ChunkType, StatementType
from financial_rag.storage.mongo_store import MongoFinancialStore
from financial_rag.retrieval.runtime import ScopeCache, RequestSnapshot


def document(value):
    return {"logical_key": "fixture", "document_id": "fixture", "source_file": "fixture.txt",
            "company_code": "FIXTURE", "fiscal_year": 2025, "report_type": "consolidated",
            "raw_content": f"source {value}", "chunks": [FinancialChunk(
                chunk_id="fixture:row", document_id="fixture", source_file="fixture.txt", company_code="FIXTURE",
                fiscal_year=2025, report_type="consolidated", unit="Triệu đồng",
                statement_type=StatementType.BALANCE_SHEET, chunk_type=ChunkType.FINANCIAL_ITEM,
                content=f"TỔNG TÀI SẢN năm 2025: {value} Triệu đồng",
                metadata={"metric_name": "TỔNG TÀI SẢN", "period_values": {"2025": value}})]}


def main():
    name = "financial_expansion_test_" + uuid.uuid4().hex
    store = MongoFinancialStore(os.environ["MONGODB_URI"], name)
    checks = []
    try:
        store.replace_documents([document("100")])
        original = store.list_documents()[0]["active_projection_id"]
        catalog_read, resume = Event(), Event()

        def callback(docs):
            catalog_read.set()
            assert resume.wait(10)
            return None

        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(store.read_scope, ["FIXTURE"], [2025], "consolidated", cache_lookup=callback)
            assert catalog_read.wait(10)
            store.replace_documents([document("200")])
            resume.set()
            old = pending.result(timeout=10)
        assert old["documents"][0]["active_projection_id"] == original
        assert old["chunks"][0].metadata["period_values"]["2025"] == "100"
        new = store.read_scope(["FIXTURE"], [2025], "consolidated")
        assert new["chunks"][0].metadata["period_values"]["2025"] == "200"
        assert store.read_scope(["OTHER"], [2025], "consolidated")["chunks"] == []
        checks.extend(["cross_process_snapshot_during_replace", "issuer_scope_isolation"])
        cache = ScopeCache()
        snapshot = RequestSnapshot(tuple(new["chunks"]), None, tuple(new["documents"]), cache.key(new["documents"]))
        cache.put(snapshot)
        cached = store.read_scope(["FIXTURE"], [2025], "consolidated", cache_lookup=lambda docs: cache.get(cache.key(docs)))
        assert cached["cached"] is snapshot and not cached["chunks"]
        checks.append("exact_projection_cache_reuse")

        queued = store.enqueue_ingest({"documents": [{"content": "fixture job", "source_file": "fixture.txt"}]})
        assert store.enqueue_ingest({"documents": [{"content": "fixture job", "source_file": "fixture.txt"}]})["job_id"] == queued["job_id"]
        first = store.claim_ingest("stopped-worker", lease_seconds=1)
        store.jobs.update_one({"_id": first["_id"]}, {"$set": {"lease_until": datetime.now(timezone.utc) - timedelta(seconds=1)}})
        recovered = store.claim_ingest("restarted-worker")
        assert recovered["attempts"] == 2 and recovered["lease_token"] != first["lease_token"]
        try:
            store.replace_documents([document("300")], job=(first["_id"], first["lease_token"]))
        except RuntimeError:
            pass
        else:
            raise AssertionError("Stale worker published a document")
        assert store.read_scope(["FIXTURE"], [2025], "consolidated")["chunks"][0].metadata["period_values"]["2025"] == "200"
        invalid = document("300")
        invalid["chunks"] *= 2
        try:
            store.replace_documents([invalid], job=(recovered["_id"], recovered["lease_token"]))
        except Exception:
            pass
        else:
            raise AssertionError("Invalid publication accepted")
        assert store.get_ingest_job(queued["job_id"])["status"] == "running"
        store.replace_documents([document("300")], job=(recovered["_id"], recovered["lease_token"]))
        assert store.get_ingest_job(queued["job_id"])["status"] == "succeeded"
        assert "payload" not in store.jobs.find_one({"_id": queued["job_id"]})
        checks.extend(["enqueue_idempotency", "expired_job_recovery", "stale_worker_fencing",
                       "job_and_document_transaction_rollback", "job_success_payload_cleanup"])
        store.activate_projection("fixture", original)
        assert store.read_scope(["FIXTURE"], [2025], "consolidated")["chunks"][0].metadata["period_values"]["2025"] == "100"
        store.delete_document("fixture")
        assert store.read_scope(["FIXTURE"], [2025], "consolidated")["chunks"] == []
        store.activate_projection("fixture", original)
        assert store.chunk_count() == 1
        checks.extend(["checksum_verified_projection_rollback", "delete_and_restore_keeps_source_history"])
        print(json.dumps({"passed": True, "checks": checks, "production_documents_changed": False}))
    finally:
        assert name.startswith("financial_expansion_test_") and name != "financial_rag"
        store.client.drop_database(name)
        store.close()


if __name__ == "__main__":
    main()
