"""One-time, idempotent migration from the standalone SQLite volume.

Run with the MongoDB replica set and Qdrant containers available. The original
SQLite volume is read-only input and is never deleted or rewritten.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any

import numpy as np

from ..schemas import FinancialChunk
from .mongo_store import MongoFinancialStore
from .qdrant_embedding_cache import QdrantEmbeddingCache


def _read_only_connection(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"Legacy SQLite database not found: {path}")
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def migrate_documents(source: Path, store: MongoFinancialStore) -> tuple[int, int]:
    with closing(_read_only_connection(source / "financial_rag.sqlite3")) as connection:
        source_document_count = int(
            connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        )
        source_chunk_count = int(
            connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        )
        documents = connection.execute(
            """SELECT logical_key, document_id, source_file, source_pdf, company_code,
                      fiscal_year, fiscal_quarter, report_type, raw_content, chunk_count,
                      ingested_at
               FROM documents ORDER BY ingested_at, source_file"""
        ).fetchall()

        migrated_chunks = 0
        for position, row in enumerate(documents, start=1):
            document = dict(row)
            chunk_rows = connection.execute(
                "SELECT payload_json FROM chunks WHERE logical_key = ? ORDER BY rowid",
                (document["logical_key"],),
            ).fetchall()
            document["chunks"] = [
                FinancialChunk.model_validate_json(item["payload_json"])
                for item in chunk_rows
            ]
            if len(document["chunks"]) != int(document["chunk_count"]):
                raise RuntimeError(
                    f"Chunk count mismatch for {document['source_file']}: "
                    f"metadata={document['chunk_count']}, rows={len(document['chunks'])}"
                )
            store.replace_documents([document])
            migrated_chunks += len(document["chunks"])
            if position % 20 == 0 or position == source_document_count:
                print(f"Migrated {position}/{source_document_count} documents")

    if store.document_count() < source_document_count:
        raise RuntimeError(
            f"Mongo document count is lower than the SQLite corpus: expected at least {source_document_count}, "
            f"found {store.document_count()}"
        )
    if store.chunk_count() < source_chunk_count or migrated_chunks != source_chunk_count:
        raise RuntimeError(
            f"Mongo chunk count is lower than the SQLite corpus: expected at least {source_chunk_count}, "
            f"found {store.chunk_count()}"
        )
    return source_document_count, source_chunk_count


def migrate_vectors(source: Path, qdrant_url: str, api_key: str | None) -> dict[str, int]:
    vector_path = source / "financial_embeddings.sqlite3"
    if not vector_path.is_file():
        return {}

    migrated_by_namespace: dict[str, int] = {}
    with closing(_read_only_connection(vector_path)) as connection:
        namespaces = [
            str(row[0]) for row in connection.execute(
                "SELECT DISTINCT namespace FROM vectors ORDER BY namespace"
            ).fetchall()
        ]
        for namespace in namespaces:
            cache = QdrantEmbeddingCache(qdrant_url, namespace, api_key)
            cursor = connection.execute(
                """SELECT text_hash, dimension, vector FROM vectors
                   WHERE namespace = ? ORDER BY text_hash""",
                (namespace,),
            )
            migrated = 0
            while rows := cursor.fetchmany(256):
                hashes: list[str] = []
                vectors: list[np.ndarray] = []
                for row in rows:
                    dimension = int(row["dimension"])
                    blob: Any = row["vector"]
                    if dimension <= 0 or not isinstance(blob, bytes) or len(blob) != dimension * 4:
                        continue
                    vector = np.frombuffer(blob, dtype="<f4").copy()
                    norm = float(np.linalg.norm(vector))
                    if not np.isfinite(vector).all() or abs(norm - 1.0) >= 0.0001:
                        continue
                    hashes.append(str(row["text_hash"]))
                    vectors.append(vector)
                if not hashes:
                    continue
                migrated += cache.save_hashes(hashes, np.asarray(vectors, dtype=np.float32))
            point_count = cache.count()
            if point_count < migrated:
                cache.close()
                raise RuntimeError(
                    f"Qdrant point count is lower than migrated vectors for namespace {namespace}"
                )
            migrated_by_namespace[namespace] = migrated
            print(f"Migrated {migrated} vectors into {cache.collection_name}")
            cache.close()
    return migrated_by_namespace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-data-dir", type=Path, required=True)
    args = parser.parse_args()

    mongo_uri = os.getenv("MONGODB_URI", "").strip()
    mongo_database = os.getenv("FINANCIAL_MONGO_DATABASE", "financial_rag")
    qdrant_url = os.getenv("QDRANT_URL", "").strip()
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if not mongo_uri or not qdrant_url:
        parser.error("MONGODB_URI and QDRANT_URL must be configured")

    store = MongoFinancialStore(mongo_uri, mongo_database)
    try:
        document_count, chunk_count = migrate_documents(args.legacy_data_dir, store)
        vectors = migrate_vectors(args.legacy_data_dir, qdrant_url, qdrant_api_key)
    finally:
        store.close()

    print(json.dumps({
        "status": "migrated",
        "documents": document_count,
        "chunks": chunk_count,
        "qdrant_vectors_by_namespace": vectors,
        "legacy_data_preserved": True,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
