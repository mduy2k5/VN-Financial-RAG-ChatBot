"""Source preparation and publication without importing the HTTP application."""
from __future__ import annotations

import hashlib
import os
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints, model_validator

from ..schemas import FinancialChunk
from ..retrieval.bm25_retriever import BM25Retriever

DocumentContent = Annotated[str, StringConstraints(min_length=1, max_length=10_000_000)]
SourceName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)]


class IngestRequest(BaseModel):
    content: DocumentContent
    source_file: SourceName = "bctc.md"
    company_code: Annotated[str, StringConstraints(strip_whitespace=True, max_length=32)] | None = None
    document_id: Annotated[str, StringConstraints(max_length=512)] | None = None
    source_pdf: SourceName | None = None

    @model_validator(mode="after")
    def normalize_company_code(self):
        if self.company_code:
            self.company_code = self.company_code.upper()
        return self


class BatchIngestRequest(BaseModel):
    documents: list[IngestRequest] = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def enforce_total_content_limit(self):
        if sum(len(item.content.encode("utf-8")) for item in self.documents) > 100_000_000:
            raise ValueError("Batch content exceeds 100 MB")
        return self


def prepare_document(req: IngestRequest, chunker):
    chunks = chunker.chunk_text(req.content, source_file=req.source_file,
        company_code=req.company_code, source_pdf=req.source_pdf, document_id=req.document_id)
    first = chunks[0] if chunks else None
    company = (first.company_code if first else req.company_code) or ""
    logical_key = hashlib.sha256((req.document_id or
        f"{req.source_file.casefold()}|{company.casefold()}").encode()).hexdigest()
    if first is not None:
        prefix = req.document_id or f"{req.source_file}:{company or 'unknown'}:{first.fiscal_year or 'unknown'}"
        ids = {chunk.chunk_id: f"{prefix}:{chunk.chunk_id}" for chunk in chunks}
        for chunk in chunks:
            chunk.chunk_id = ids[chunk.chunk_id]
            if chunk.parent_chunk_id:
                chunk.parent_chunk_id = ids.get(chunk.parent_chunk_id, chunk.parent_chunk_id)
            if chunk.metadata.get("parent_id"):
                parent = chunk.metadata["parent_id"]
                chunk.metadata["parent_id"] = ids.get(parent, parent)
    document = {"logical_key": logical_key, "document_id":
        (first.document_id if first else None) or req.document_id or req.source_file,
        "source_file": req.source_file, "source_pdf": req.source_pdf,
        "company_code": company or None, "fiscal_year": first.fiscal_year if first else None,
        "fiscal_quarter": first.fiscal_quarter if first else None,
        "report_type": first.report_type if first else None, "raw_content": req.content,
        "chunks": chunks}
    return chunks, document


def publish_documents(store, documents, build_index, *, job=None, expected_projections=None,
                      assert_lease=lambda: None):
    for document in documents:
        assert_lease()
        BM25Retriever(document["chunks"])
        document["search_index"] = {"sparse": "ready", "dense": "disabled"}
    if os.getenv("FINANCIAL_ENABLE_DENSE", "false").casefold() in {"1", "true", "yes"}:
        if os.getenv("FINANCIAL_DENSE_SEARCH_BACKEND", "qdrant") != "qdrant":
            raise RuntimeError("Scoped ingest requires Qdrant dense search")
        for document in documents:
            assert_lease()
            index = build_index(document["chunks"])
            if index._dense_index_thread:
                index._dense_index_thread.join(timeout=float(os.getenv("FINANCIAL_INGEST_INDEX_TIMEOUT", "120")))
            if index.dense_index_status != "ready":
                index.cancel_dense_indexing()
                raise RuntimeError("Dense index is not ready; document remains unpublished")
            document["search_index"].update(dense="ready", collection=index.dense_retriever.collection_name,
                embedding_revision=index.dense_retriever.namespace,
                point_count=len(index.dense_retriever._points_by_chunk))
    assert_lease()
    kwargs = {"job": job}
    if expected_projections is not None:
        kwargs["expected_projections"] = expected_projections
    return store.replace_documents(documents, **kwargs)


def serialize_document(document):
    return {**document, "chunks": [item.model_dump(mode="json") for item in document["chunks"]]}


def deserialize_document(document):
    return {**document, "chunks": [FinancialChunk.model_validate(item) for item in document["chunks"]]}
