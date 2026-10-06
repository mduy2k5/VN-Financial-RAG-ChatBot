"""Content addressed, versioned cache for normalized passage embeddings."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sqlite3

import numpy as np


class EmbeddingCache:
    def __init__(self, path: str | Path, namespace: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS vectors (namespace TEXT, text_hash TEXT, "
                "dimension INTEGER, vector BLOB, PRIMARY KEY(namespace,text_hash))"
            )

    def _connect(self):
        return sqlite3.connect(self.path, timeout=30)

    @staticmethod
    def text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def load(self, texts: list[str]) -> dict[str, np.ndarray]:
        hashes = list(dict.fromkeys(self.text_hash(text) for text in texts))
        found: dict[str, np.ndarray] = {}
        with self._connect() as connection:
            for start in range(0, len(hashes), 400):
                batch = hashes[start:start + 400]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    f"SELECT text_hash,dimension,vector FROM vectors WHERE namespace=? AND text_hash IN ({placeholders})",
                    [self.namespace, *batch],
                )
                for text_hash, dimension, blob in rows:
                    if not isinstance(blob, bytes) or len(blob) != dimension * 4 or dimension <= 0:
                        continue
                    vector = np.frombuffer(blob, dtype="<f4").copy()
                    norm = np.linalg.norm(vector)
                    if np.isfinite(vector).all() and abs(float(norm) - 1.0) < 0.0001:
                        found[text_hash] = vector
        return found

    def save(self, texts: list[str], vectors: np.ndarray) -> None:
        records = []
        for text, vector in zip(texts, vectors):
            if np.isfinite(vector).all() and np.linalg.norm(vector) > 0:
                records.append((self.namespace, self.text_hash(text), len(vector), vector.astype("<f4").tobytes()))
        with self._connect() as connection:
            connection.executemany("INSERT OR REPLACE INTO vectors VALUES (?,?,?,?)", records)
