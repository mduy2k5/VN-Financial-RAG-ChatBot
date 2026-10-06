"""Versioned, content-addressed persistence for passage embeddings in Qdrant."""
from __future__ import annotations

import hashlib
import uuid
from typing import Iterable

import numpy as np


class QdrantEmbeddingCache:
    """Persist one normalized vector per unique passage and model revision."""

    def __init__(self, url: str, namespace: str, api_key: str | None = None):
        from qdrant_client import QdrantClient

        if not url.strip():
            raise ValueError("QDRANT_URL must be set for Qdrant vector persistence")
        self.namespace = namespace
        self.collection_name = f"financial_embeddings_{namespace}"
        self.client = QdrantClient(
            url=url,
            api_key=api_key or None,
            timeout=120,
            prefer_grpc=True,
        )

    @staticmethod
    def text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _point_id(self, text_hash: str) -> str:
        return str(uuid.uuid5(
            uuid.NAMESPACE_URL, f"financial-rag:{self.namespace}:{text_hash}"
        ))

    def _collection_exists(self) -> bool:
        return bool(self.client.collection_exists(self.collection_name))

    def _ensure_collection(self, dimension: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        if not self._collection_exists():
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
                )
            except Exception:
                # Another indexing worker may have created it concurrently.
                if not self._collection_exists():
                    raise
        info = self.client.get_collection(self.collection_name)
        vector_config = info.config.params.vectors
        configured_dimension = getattr(vector_config, "size", None)
        if configured_dimension is not None and int(configured_dimension) != dimension:
            raise ValueError(
                f"Qdrant collection {self.collection_name} has dimension "
                f"{configured_dimension}, expected {dimension}"
            )

    def load(self, texts: list[str]) -> dict[str, np.ndarray]:
        hashes = list(dict.fromkeys(self.text_hash(text) for text in texts))
        if not hashes or not self._collection_exists():
            return {}

        found: dict[str, np.ndarray] = {}
        for start in range(0, len(hashes), 512):
            batch_hashes = hashes[start:start + 512]
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[self._point_id(item) for item in batch_hashes],
                with_payload=True,
                with_vectors=True,
            )
            requested = set(batch_hashes)
            for point in points:
                payload = point.payload or {}
                text_hash = payload.get("text_hash")
                if text_hash not in requested or payload.get("namespace") != self.namespace:
                    continue
                raw_vector = point.vector
                if isinstance(raw_vector, dict):
                    dense_vector = raw_vector.get("")
                elif isinstance(raw_vector, list):
                    dense_vector = raw_vector
                else:
                    continue
                if not isinstance(dense_vector, list):
                    continue
                vector = np.asarray(dense_vector, dtype=np.float32)
                norm = float(np.linalg.norm(vector))
                if (
                    vector.ndim == 1
                    and vector.size > 0
                    and np.isfinite(vector).all()
                    and abs(norm - 1.0) < 0.0001
                ):
                    found[str(text_hash)] = vector
        return found

    def save(self, texts: list[str], vectors: np.ndarray) -> None:
        hashes = [self.text_hash(text) for text in texts]
        self.save_hashes(hashes, vectors)

    def save_hashes(self, hashes: Iterable[str], vectors: np.ndarray) -> int:
        from qdrant_client.models import PointStruct

        vector_matrix = np.asarray(vectors, dtype=np.float32)
        hash_list = list(hashes)
        if vector_matrix.ndim != 2 or vector_matrix.shape[0] != len(hash_list):
            raise ValueError("Hash count must match the embedding matrix rows")
        if not hash_list:
            return 0
        self._ensure_collection(int(vector_matrix.shape[1]))

        # Collapse repeated passage hashes before upsert; one exact passage and
        # one model namespace always resolve to the same vector point.
        unique: dict[str, np.ndarray] = {}
        for text_hash, vector in zip(hash_list, vector_matrix, strict=False):
            norm = float(np.linalg.norm(vector))
            if np.isfinite(vector).all() and norm > 0:
                unique[str(text_hash)] = vector / norm

        hashes_to_write = list(unique)
        for start in range(0, len(hashes_to_write), 256):
            batch_hashes = hashes_to_write[start:start + 256]
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=self._point_id(text_hash),
                        vector=unique[text_hash].tolist(),
                        payload={"namespace": self.namespace, "text_hash": text_hash},
                    )
                    for text_hash in batch_hashes
                ],
                wait=True,
            )
        return len(unique)

    def count(self) -> int:
        if not self._collection_exists():
            return 0
        return int(self.client.count(self.collection_name, exact=True).count)

    def close(self) -> None:
        self.client.close()
