"""Server-side dense search with immutable chunk points and bounded batch RAM.

Embedding cache points remain content-addressed. Search points additionally
carry chunk identity and report scope, so identical text from different issuers
does not collapse into one citation. An in-process snapshot becomes searchable
only after every point has been acknowledged; old snapshots keep their own IDs.
"""
from __future__ import annotations

import hashlib
import os
from threading import Event
from typing import cast
import uuid

import numpy as np

from ..structure.normalized import fingerprint
from .dense_retriever import DenseRetriever


class QdrantDenseRetriever(DenseRetriever):
    def __init__(self, chunks, embed_adapter=None, *, client=None, namespace=None, collection=None):
        super().__init__(chunks, embed_adapter)
        if client is None:
            from qdrant_client import QdrantClient
            url = os.getenv("QDRANT_URL", "").strip()
            if not url:
                raise ValueError("QDRANT_URL is required for server-side dense search")
            client = QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY") or None, timeout=120)
        self.client = client
        self._chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.namespace = namespace
        self.collection_name = collection
        self._points_by_chunk: dict[str, str] = {}
        self._dimension = None
        self.indexed_vector_count = 0

    def _ensure_collection(self, dimension: int) -> None:
        from qdrant_client.models import Distance, PayloadSchemaType, VectorParams
        if not self.client.collection_exists(self.collection_name):
            try:
                self.client.create_collection(self.collection_name,
                                              vectors_config=VectorParams(size=dimension, distance=Distance.COSINE))
            except Exception:
                if not self.client.collection_exists(self.collection_name):
                    raise
        info = self.client.get_collection(self.collection_name)
        if getattr(info.config.params.vectors, "size", None) != dimension:
            raise ValueError("Qdrant search vector dimension changed within the model namespace")
        for field, kind in (("company_code", PayloadSchemaType.KEYWORD),
                            ("report_type", PayloadSchemaType.KEYWORD),
                            ("fiscal_year", PayloadSchemaType.INTEGER)):
            if field not in info.payload_schema:
                self.client.create_payload_index(self.collection_name, field, kind, wait=True)
        self._dimension = dimension

    def index(self, max_chunks=None, batch_size=None, *, cancel_event: Event | None = None,
              seed_retriever=None):
        from qdrant_client.models import PointStruct
        self.index_status = "indexing"
        self.last_error = None
        self.cached_vector_count = 0
        self.indexed_vector_count = 0
        self._points_by_chunk = {}
        chunks = self.chunks if max_chunks is None else self.chunks[:max_chunks]
        cache = None
        try:
            if not chunks:
                self.index_status = "empty"
                return
            cache = self._cache()
            namespace = self.namespace or (getattr(cache, "namespace", None) if cache else None)
            if not namespace:
                raise ValueError("Server-side dense search requires an immutable embedding revision")
            self.namespace = namespace
            self.embedding_identity = namespace
            self.collection_name = self.collection_name or f"financial_search_{namespace}"
            exists = self.client.collection_exists(self.collection_name)
            if exists:
                info = self.client.get_collection(self.collection_name)
                self._ensure_collection(int(info.config.params.vectors.size))
            batch_size = max(1, int(batch_size or os.getenv("FINANCIAL_EMBEDDING_BATCH_SIZE", "32")))
            pending_ids: dict[str, str] = {}
            for start in range(0, len(chunks), batch_size):
                if cancel_event and cancel_event.is_set():
                    self.index_status = "cancelled"
                    return
                batch = chunks[start:start + batch_size]
                texts = [self._passage_text(chunk) for chunk in batch]
                payloads = [{"chunk_id": chunk.chunk_id, "document_id": chunk.document_id or chunk.source_file,
                             "company_code": (chunk.company_code or "").casefold(),
                             "report_type": chunk.report_type, "fiscal_year": chunk.fiscal_year,
                             "fiscal_quarter": chunk.fiscal_quarter,
                             "text_hash": hashlib.sha256(text.encode()).hexdigest()}
                            for chunk, text in zip(batch, texts)]
                point_ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"financial-search:{namespace}:{fingerprint(payload)}"))
                             for payload in payloads]
                existing = {str(point.id) for point in self.client.retrieve(
                    self.collection_name, point_ids, with_vectors=False, with_payload=False)} if exists else set()
                missing = [i for i, point_id in enumerate(point_ids) if point_id not in existing]
                self.cached_vector_count += len(existing)
                if missing:
                    missing_texts = [texts[i] for i in missing]
                    cached = cache.load(missing_texts) if cache else {}
                    vectors = {i: cached[cache.text_hash(texts[i])] for i in missing
                               if cache and cache.text_hash(texts[i]) in cached}
                    needs_embedding = [i for i in missing if i not in vectors]
                    embedded = self._embed_batch_resilient([texts[i] for i in needs_embedding]) if needs_embedding else []
                    vectors.update(zip(needs_embedding, embedded))
                    points = []
                    normalized = []
                    for i in missing:
                        vector = np.asarray(vectors[i], dtype=np.float32)
                        norm = np.linalg.norm(vector)
                        if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all() or norm <= 0:
                            raise ValueError("Embedding service returned an invalid search vector")
                        if self._dimension is None:
                            self._ensure_collection(int(vector.size))
                            exists = True
                        if vector.size != self._dimension:
                            raise ValueError("Embedding dimensions disagree")
                        vector = vector / norm
                        normalized.append(vector)
                        points.append(PointStruct(id=point_ids[i], vector=vector.tolist(), payload=payloads[i]))
                    self.client.upsert(self.collection_name, points, wait=True)
                    if cache:
                        cache.save(missing_texts, np.asarray(normalized, dtype=np.float32))
                pending_ids.update(zip((chunk.chunk_id for chunk in batch), point_ids))
                self.indexed_vector_count += len(batch)
            if cancel_event and cancel_event.is_set():
                self.index_status = "cancelled"
                return
            self._points_by_chunk = pending_ids
            self.index_status = "ready"
        except Exception as exc:
            self._points_by_chunk = {}
            self.index_status = "error"
            self.last_error = f"{type(exc).__name__}: {exc}"[:500]
            raise
        finally:
            if cache and hasattr(cache, "close"):
                cache.close()

    def search(self, query: str, top_k: int = 20, allowed_ids: set[str] | None = None):
        from qdrant_client.models import Condition, FieldCondition, Filter, HasIdCondition, MatchAny
        if self.index_status != "ready":
            return []
        selected = [self._chunks_by_id[chunk_id] for chunk_id in
                    (allowed_ids if allowed_ids is not None else self._points_by_chunk)
                    if chunk_id in self._points_by_chunk]
        if not selected:
            return []
        vector = np.asarray(self.embed_adapter.embed_query(query), dtype=np.float32)
        if vector.ndim != 1 or len(vector) != self._dimension or not np.isfinite(vector).all():
            raise ValueError("Query embedding is malformed or has the wrong dimension")
        norm = np.linalg.norm(vector)
        if norm <= 0:
            return []
        point_ids = cast(
            list[int | str | uuid.UUID],
            [self._points_by_chunk[chunk.chunk_id] for chunk in selected],
        )
        companies = sorted({(chunk.company_code or "").casefold() for chunk in selected})
        must: list[Condition] = [
            HasIdCondition(has_id=point_ids),
            FieldCondition(key="company_code", match=MatchAny(any=companies)),
        ]
        # HasId enforces the exact active snapshot and period/basis candidate set,
        # including comparative columns. Do not replace it with fiscal_year alone.
        result = self.client.query_points(self.collection_name, query=(vector / norm).tolist(),
                                          query_filter=Filter(must=must), limit=top_k,
                                          with_payload=True, with_vectors=False)
        permitted = set(point_ids)
        return [(point.payload["chunk_id"], float(point.score)) for point in result.points
                if str(point.id) in permitted]
