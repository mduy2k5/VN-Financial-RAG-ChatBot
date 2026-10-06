"""Failure injection against an isolated Mongo database, never the live corpus."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile
import uuid

from financial_rag.ingestion.stages import StageCache
from financial_rag.schemas import FinancialChunk
from financial_rag.storage.ingest_worker import run_once
from financial_rag.storage.mongo_store import MongoFinancialStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    name = "financial_ingestion_test_" + uuid.uuid4().hex
    store = MongoFinancialStore(os.environ["MONGODB_URI"], name)
    calls = []
    checks = []

    def document(raw, key="fixture"):
        chunk = FinancialChunk(chunk_id=key + ":row", source_file=key + ".txt", content=raw)
        return {"logical_key": key, "document_id": key, "source_file": key + ".txt",
                "company_code": "FIXTURE", "raw_content": raw, "chunks": [chunk]}

    def prepare(req):
        calls.append(req.content)
        doc = document(req.content, req.document_id)
        return doc["chunks"], doc

    def active(key="fixture"):
        return next(d["active_projection_id"] for d in store.list_documents() if d["logical_key"] == key)

    def publish(docs, **kwargs):
        kwargs.pop("assert_lease")()
        return store.replace_documents(docs, **kwargs)

    def expire(job_id):
        store.jobs.update_one({"_id": job_id}, {"$set": {"lease_until": datetime.now(timezone.utc) - timedelta(seconds=1)}})

    try:
        with tempfile.TemporaryDirectory() as temp:
            cache = StageCache(Path(temp) / "stages.sqlite3")
            store.replace_documents([document("Original source")])
            original = active()
            payload = {"documents": [{"document_id": "fixture", "source_file": "fixture.txt", "content": "Queued old update"}]}
            job_id = store.enqueue_ingest(payload)["job_id"]

            def fail_publish(*args, **kwargs):
                raise RuntimeError("Injected index failure")

            run_once(store, prepare=prepare, publish=fail_publish, cache=cache)
            assert active() == original
            store.replace_documents([document("Newer direct update")])
            newer = active()
            for _ in range(2):
                expire(job_id)
                run_once(store, prepare=prepare, publish=publish, cache=cache)
                assert active() == newer
                assert store.jobs.find_one({"_id": job_id})["expected_projections"] == {"fixture": original}
            assert store.get_ingest_job(job_id)["status"] == "failed"
            assert calls.count("Queued old update") == 1
            checks.extend(["retry_replays_valid_stage", "retry_keeps_original_CAS", "stale_publish_transaction_rollback"])

            store.replace_documents([document("Before parse", "parse")])
            before_parse = active("parse")
            parse_id = store.enqueue_ingest({"documents": [
                {"document_id": "parse", "content": "Queued before parse failure"}]})["job_id"]
            def fail_parse(req):
                raise RuntimeError("Injected parse failure")
            run_once(store, prepare=fail_parse, publish=publish, cache=cache)
            store.replace_documents([document("Newer source after parse failure", "parse")])
            after_parse = active("parse")
            for _ in range(2):
                expire(parse_id)
                run_once(store, prepare=prepare, publish=publish, cache=cache)
                assert active("parse") == after_parse
                assert store.jobs.find_one({"_id": parse_id})["expected_projections"] == {"parse": before_parse}
            assert store.get_ingest_job(parse_id)["status"] == "failed"
            checks.append("early_parse_failure_preserves_original_CAS")

            fresh = {"documents": [{"document_id": "new", "source_file": "new.txt", "content": "Fresh source"}]}
            fresh_id = store.enqueue_ingest(fresh)["job_id"]
            run_once(store, prepare=prepare, publish=publish, cache=cache)
            assert store.get_ingest_job(fresh_id)["status"] == "succeeded"
            projection = active("new")
            assert store.enqueue_ingest(fresh)["status"] == "succeeded"
            store.replace_documents([document("Changed new source", "new")])
            assert store.enqueue_ingest(fresh)["status"] == "queued"
            run_once(store, prepare=prepare, publish=publish, cache=cache)
            assert active("new") == projection
            checks.extend(["successful_publish", "enqueue_idempotent", "explicit_reenqueue_restores_version"])

            lost = store.enqueue_ingest({"documents": [{"document_id": "lost", "content": "Lost source"}]})["job_id"]
            lease = store.claim_ingest("fixture-worker")
            assert lease["_id"] == lost
            expire(lost)
            try:
                store.replace_documents([document("Must not publish", "lost")], job=(lost, lease["lease_token"]))
            except RuntimeError as exc:
                assert "lease lost" in str(exc)
            else:
                raise AssertionError("Expired lease was accepted")
            assert not any(d["logical_key"] == "lost" for d in store.list_documents())
            checks.append("expired_lease_blocks_atomic_publish")
            # Projection is limited server-side, so the worker never transfers full reports for aliases.
            assert isinstance(store.load_company_aliases(), dict)
            checks.append("bounded_alias_projection")
            report = {"passed": True, "checks": checks, "stage_cache": cache.stats(),
                      "live_corpus_modified": False, "database": name}
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report))
    finally:
        assert name.startswith("financial_ingestion_test_") and name != "financial_rag"
        store.client.drop_database(name)
        store.client.close()


if __name__ == "__main__":
    main()
