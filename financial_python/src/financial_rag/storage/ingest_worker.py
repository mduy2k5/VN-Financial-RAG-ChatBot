"""Durable ingest worker sharing the API's preparation/publication contract."""
import logging
import os
import socket
from threading import Event, Thread
import time

from ..application.ingestion import BatchIngestRequest, prepare_document, publish_documents
from ..ingestion.stages import IngestionPipeline, StageCache
from .factory import create_document_store

logger = logging.getLogger(__name__)


def run_once(store, *, prepare, publish, lease_seconds=120, cache=None, cache_context=None) -> bool:
    job = store.claim_ingest(f"{socket.gethostname()}:{os.getpid()}", lease_seconds)
    if not job:
        return False
    stop = Event()
    lost = Event()

    def assert_lease():
        if lost.is_set() or not store.renew_ingest(job["_id"], job["lease_token"], lease_seconds):
            lost.set()
            raise RuntimeError("Ingest lease lost; processing stopped")

    def heartbeat():
        while not stop.wait(max(1, lease_seconds / 3)):
            if not store.renew_ingest(job["_id"], job["lease_token"], lease_seconds):
                lost.set()
                return

    thread = Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        assert_lease()
        request = BatchIngestRequest.model_validate(job["payload"])
        active = {item["logical_key"]: item.get("active_projection_id") for item in store.list_documents()}
        # Persist before alias loading/parsing can fail; a retry must not adopt newer pointers.
        active = store.set_ingest_start_projections(job["_id"], job["lease_token"], active)
        context = cache_context() if callable(cache_context) else cache_context

        def progress(event):
            assert_lease()
            store.update_ingest_progress(job["_id"], job["lease_token"], event)

        pipeline = IngestionPipeline(prepare, cache=cache, progress=progress, assert_lease=assert_lease,
                                     context_fingerprint=context)
        documents = pipeline.prepare_batch(request)
        assert_lease()
        expected = store.set_ingest_expected_projections(job["_id"], job["lease_token"],
            {doc["logical_key"]: active.get(doc["logical_key"]) for doc in documents})
        publish(documents, job=(job["_id"], job["lease_token"]), expected_projections=expected,
                assert_lease=assert_lease)
    except Exception as exc:
        logger.exception("ingest_job_failed job_id=%s", job["_id"])
        store.fail_ingest(job["_id"], job["lease_token"], str(exc))
    finally:
        stop.set()
        thread.join(timeout=5)
    return True


def main():
    logging.basicConfig(level=logging.INFO)
    document_store = create_document_store()
    if not hasattr(document_store, "claim_ingest"):
        raise RuntimeError("Durable worker requires MongoDB")
    from ..chunking import UniversalFinancialChunker
    from ..retrieval.hybrid import FinancialHybridRetriever
    from ..embedding.text_builder import TextBuilder
    from ..schemas import ChunkType

    chunker = UniversalFinancialChunker()

    def prepare(request):
        return prepare_document(request, chunker)

    def build_index(chunks):
        chunks = [item.model_copy(deep=True) for item in chunks]
        for item in chunks:
            if item.chunk_type == ChunkType.FINANCIAL_ITEM:
                item.content = TextBuilder.build_source_content(item)
        chunker.schema_catalog.annotate_chunks(chunks)
        return FinancialHybridRetriever(chunks, enable_dense=True, background_dense_index=True)

    cache = StageCache(os.getenv("FINANCIAL_INGEST_STAGE_DB", "/ingest-stages/stages.sqlite3"))

    def publish(documents, **kwargs):
        return publish_documents(document_store, documents, build_index, **kwargs)

    def cache_context():
        from ..structure.normalized import fingerprint
        aliases = document_store.load_company_aliases()
        chunker.md_chunker.company_extractor.set_dynamic_aliases(aliases)
        chunker.txt_chunker.company_extractor.set_dynamic_aliases(aliases)
        return fingerprint(aliases)

    while True:
        try:
            if not run_once(document_store, prepare=prepare, publish=publish, cache=cache,
                            cache_context=cache_context):
                time.sleep(2)
        except Exception:
            logger.exception("ingest_worker_iteration_failed")
            time.sleep(5)


if __name__ == "__main__":
    main()
