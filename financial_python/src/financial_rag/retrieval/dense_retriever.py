from __future__ import annotations

import numpy as np
import os
import hashlib
import json
import logging
from threading import Event
from typing import Any, List, Tuple, Optional, Sequence
from ..schemas import FinancialChunk
from ..embedding.e5_adapter import E5EmbeddingAdapter
from ..embedding.text_builder import TextBuilder
from ..storage.embedding_cache import EmbeddingCache

logger = logging.getLogger(__name__)

class DenseRetriever:
    """
    Bộ truy xuất vector ngữ nghĩa dùng E5EmbeddingAdapter.
    Hỗ trợ tính Cosine Similarity giữa câu truy vấn và các vector chunk.
    """

    def __init__(
        self,
        chunks: List[FinancialChunk],
        embed_adapter: Optional[E5EmbeddingAdapter] = None,
    ):
        self.chunks = chunks
        self.chunk_ids = [c.chunk_id for c in self.chunks]
        self.embed_adapter = embed_adapter or E5EmbeddingAdapter()
        self.chunk_vectors: List[List[float]] = []
        self._normalized_vectors: Optional[np.ndarray] = None
        self._valid_vector_mask: Optional[np.ndarray] = None
        self.index_status = "not_started"
        self.last_error: Optional[str] = None
        self.cached_vector_count = 0
        self.embedding_identity: tuple[Any | None, Any | None, Any | None] | str = (
            getattr(self.embed_adapter, "base_url", None),
            getattr(self.embed_adapter, "model", None),
            getattr(self.embed_adapter, "passage_prefix", None),
        )

    def _cache(self) -> Optional[Any]:
        if not isinstance(self.embed_adapter, E5EmbeddingAdapter):
            return None
        revision = self.embed_adapter.model_revision()
        if not revision:
            logger.warning("Dense cache disabled: embedding model has no immutable revision")
            return None
        identity = {
            "format": "normalized_float32_v1", "model": self.embed_adapter.model,
            "revision": revision, "provider": self.embed_adapter.base_url,
            "passage_prefix": self.embed_adapter.passage_prefix,
        }
        namespace = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.embedding_identity = namespace
        backend = os.getenv("FINANCIAL_STORAGE_BACKEND", "sqlite").strip().casefold()
        if backend in {"mongo_qdrant", "mongodb_qdrant"}:
            from ..storage.qdrant_embedding_cache import QdrantEmbeddingCache

            url = os.getenv("QDRANT_URL", "").strip()
            if not url:
                logger.warning("Dense cache disabled: QDRANT_URL is not configured")
                return None
            return QdrantEmbeddingCache(
                url,
                namespace,
                api_key=os.getenv("QDRANT_API_KEY"),
            )
        path = os.getenv("FINANCIAL_DENSE_CACHE_PATH", "").strip()
        if not path:
            return None
        return EmbeddingCache(path, namespace)

    @staticmethod
    def _passage_text(chunk: FinancialChunk) -> str:
        text = TextBuilder.build_chunk_text(chunk)
        # The configured E5 model has a finite context window. Atomic financial
        # row chunks retain their complete label/value at the beginning; long
        # page/table chunks are bounded so one oversized page cannot fail the
        # entire corpus index.
        max_chars = max(256, int(os.getenv("FINANCIAL_EMBEDDING_MAX_CHARS", "1800")))
        if len(text) <= max_chars:
            return text
        bounded = text[:max_chars]
        return bounded.rsplit(" ", 1)[0] if " " in bounded else bounded

    def _embed_batch_resilient(
        self,
        texts: Sequence[str],
        *,
        split_depth: int = 0,
    ) -> List[List[float]]:
        """Boundedly split batches on size/context errors, never on outages."""
        try:
            vectors = self.embed_adapter.embed_passages(texts)
            if len(vectors) != len(texts):
                raise ValueError(
                    "Embedding service returned a vector count that does not match the batch"
                )
            return vectors
        except Exception as exc:
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
            error_text = str(exc).casefold()
            if response is not None:
                error_text += " " + str(getattr(response, "text", "")).casefold()
            size_markers = (
                "too many tokens", "context length", "maximum context", "batch too large",
                "payload too large", "input is too long", "out of memory", "cuda oom",
            )
            splittable = status_code == 413 or any(marker in error_text for marker in size_markers)
            if (
                len(texts) <= 1
                or split_depth >= 2
                or not splittable
            ):
                raise
            midpoint = len(texts) // 2
            return (
                self._embed_batch_resilient(texts[:midpoint], split_depth=split_depth + 1)
                + self._embed_batch_resilient(texts[midpoint:], split_depth=split_depth + 1)
            )

    def index(
        self,
        max_chunks: Optional[int] = None,
        batch_size: Optional[int] = None,
        *,
        cancel_event: Optional[Event] = None,
        seed_retriever: Optional["DenseRetriever"] = None,
    ):
        """Build a compact float32 index, reusing unchanged vectors when available."""
        self.chunk_vectors = []
        self._normalized_vectors = None
        self._valid_vector_mask = None
        self.last_error = None
        self.index_status = "indexing"
        self.cached_vector_count = 0
        chunks = self.chunks if max_chunks is None else self.chunks[:max_chunks]
        configured_batch_size = (
            batch_size
            if batch_size is not None
            else int(os.getenv("FINANCIAL_EMBEDDING_BATCH_SIZE", "32"))
        )
        batch_size = max(1, configured_batch_size)
        try:
            if not chunks:
                self.index_status = "empty"
                return

            try:
                cache = self._cache()
            except Exception as exc:
                logger.warning("Dense cache unavailable; indexing from source: %s", type(exc).__name__)
                cache = None

            seed_matrix = (
                seed_retriever._normalized_vectors
                if seed_retriever is not None and seed_retriever.index_status == "ready"
                and seed_retriever.embedding_identity == self.embedding_identity
                else None
            )
            vector_size = seed_matrix.shape[1] if seed_matrix is not None else None
            matrix = (
                np.empty((len(chunks), vector_size), dtype=np.float32)
                if vector_size is not None
                else None
            )
            valid_mask = np.zeros(len(chunks), dtype=bool)
            seeded_mask = np.zeros(len(chunks), dtype=bool)

            if seed_retriever is not None and seed_matrix is not None:
                seed_valid_mask = seed_retriever._valid_vector_mask
                old_positions = {
                    chunk_id: index
                    for index, chunk_id in enumerate(seed_retriever.chunk_ids)
                    if index < len(seed_matrix)
                }
                old_chunks = {chunk.chunk_id: chunk for chunk in seed_retriever.chunks}
                for index, chunk in enumerate(chunks):
                    old_index = old_positions.get(chunk.chunk_id)
                    old_chunk = old_chunks.get(chunk.chunk_id)
                    if (
                        old_index is None
                        or old_chunk is None
                        or seed_valid_mask is None
                        or self._passage_text(old_chunk) != self._passage_text(chunk)
                    ):
                        continue
                    if matrix is None:
                        raise RuntimeError("Dense index matrix was not allocated for the seeded vectors")
                    matrix[index] = seed_matrix[old_index]
                    valid_mask[index] = bool(seed_valid_mask[old_index])
                    seeded_mask[index] = True

            missing_indices = [index for index, seeded in enumerate(seeded_mask) if not seeded]
            try:
                cached = cache.load([self._passage_text(chunks[i]) for i in missing_indices]) if cache else {}
                dimensions = {len(vector) for vector in cached.values()}
                if cache is not None and len(dimensions) == 1:
                    cache_dimension = next(iter(dimensions))
                    if vector_size is None:
                        vector_size = cache_dimension
                        matrix = np.zeros((len(chunks), vector_size), dtype=np.float32)
                    if cache_dimension == vector_size:
                        for index in missing_indices:
                            vector = cached.get(cache.text_hash(self._passage_text(chunks[index])))
                            if vector is not None:
                                if matrix is None:
                                    raise RuntimeError("Dense index matrix was not allocated for cached vectors")
                                matrix[index] = vector
                                valid_mask[index] = True
                                seeded_mask[index] = True
                                self.cached_vector_count += 1
                missing_indices = [index for index, seeded in enumerate(seeded_mask) if not seeded]
            except Exception as exc:
                logger.warning("Dense cache unavailable; indexing from source: %s", type(exc).__name__)
                cache = None
            for batch_start in range(0, len(missing_indices), batch_size):
                if cancel_event is not None and cancel_event.is_set():
                    self.chunk_vectors = []
                    self._normalized_vectors = None
                    self._valid_vector_mask = None
                    self.index_status = "cancelled"
                    return
                indices = missing_indices[batch_start:batch_start + batch_size]
                passage_texts = [self._passage_text(chunks[index]) for index in indices]
                vectors = self._embed_batch_resilient(passage_texts)
                if cancel_event is not None and cancel_event.is_set():
                    self.chunk_vectors = []
                    self._normalized_vectors = None
                    self._valid_vector_mask = None
                    self.index_status = "cancelled"
                    return

                batch_matrix = np.asarray(vectors, dtype=np.float32)
                if batch_matrix.ndim != 2 or batch_matrix.shape[0] != len(indices):
                    raise ValueError("Embedding service returned malformed vectors")
                if vector_size is None:
                    vector_size = int(batch_matrix.shape[1])
                    if vector_size == 0:
                        self.index_status = "empty"
                        return
                    matrix = np.empty((len(chunks), vector_size), dtype=np.float32)
                elif batch_matrix.shape[1] != vector_size:
                    raise ValueError("Embedding service returned vectors with inconsistent dimensions")

                norms = np.linalg.norm(batch_matrix, axis=1)
                if not np.isfinite(batch_matrix).all():
                    raise ValueError("Embedding service returned non-finite vector values")
                batch_valid = norms > 0
                np.divide(
                    batch_matrix,
                    norms[:, np.newaxis],
                    out=batch_matrix,
                    where=batch_valid[:, np.newaxis],
                )
                if matrix is None:
                    raise RuntimeError("Dense index matrix was not allocated for embedded vectors")
                matrix[indices] = batch_matrix
                valid_mask[indices] = batch_valid
                if cache is not None:
                    try:
                        cache.save(passage_texts, batch_matrix)
                    except Exception as exc:
                        logger.warning("Dense cache write failed: %s", type(exc).__name__)
                        cache = None

            if matrix is None or vector_size is None:
                self.index_status = "empty"
                return
            if cancel_event is not None and cancel_event.is_set():
                self.index_status = "cancelled"
                return
            self._normalized_vectors = matrix
            self._valid_vector_mask = valid_mask
            self.chunk_vectors = []
            self.index_status = "ready"
        except Exception as exc:
            self.chunk_vectors = []
            self._normalized_vectors = None
            self._valid_vector_mask = None
            self.last_error = f"{type(exc).__name__}: {exc}"[:500]
            self.index_status = "error"
            raise

    def search(
        self,
        query: str,
        top_k: int = 20,
        allowed_ids: Optional[set[str]] = None,
    ) -> List[Tuple[str, float]]:
        normalized_vectors = self._normalized_vectors
        valid_vector_mask = self._valid_vector_mask
        if normalized_vectors is None or valid_vector_mask is None:
            return []

        query_vec = self.embed_adapter.embed_query(query)
        if not query_vec:
            return []

        q = np.asarray(query_vec, dtype=np.float32)
        if q.ndim != 1 or q.shape[0] != normalized_vectors.shape[1]:
            raise ValueError("Query embedding dimension does not match the indexed vectors")
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []

        candidate_indices = [
            index for index, chunk_id in enumerate(self.chunk_ids[:len(normalized_vectors)])
            if (allowed_ids is None or chunk_id in allowed_ids)
            and valid_vector_mask[index]
        ]
        if not candidate_indices:
            return []

        similarities = normalized_vectors[candidate_indices] @ (q / q_norm)
        ranked_indices = np.argsort(-similarities, kind="stable")
        return [
            (self.chunk_ids[candidate_indices[rank]], float(similarities[rank]))
            for rank in ranked_indices[:top_k]
        ]
