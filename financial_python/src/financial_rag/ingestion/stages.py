"""SQLite stage cache with checksums, atomic writes and a cross-process byte quota."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import time

from ..structure.normalized import fingerprint, normalize_report, parser_revision
from ..application.ingestion import deserialize_document, serialize_document


class StageCache:
    def __init__(self, path, max_bytes=64 * 1024 * 1024):
        self.path, self.max_bytes = Path(path), max_bytes
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS stages (id TEXT PRIMARY KEY, payload TEXT, "
                         "checksum TEXT NOT NULL, accessed REAL NOT NULL)")

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def get(self, key):
        with self.connect() as conn:
            row = conn.execute("SELECT payload,checksum FROM stages WHERE id=?", (key,)).fetchone()
            if not row:
                return None
            if hashlib.sha256(row[0].encode()).hexdigest() != row[1]:
                conn.execute("DELETE FROM stages WHERE id=?", (key,))
                return None
            conn.execute("UPDATE stages SET accessed=? WHERE id=?", (time.time(), key))
        try:
            return json.loads(row[0])
        except ValueError:
            return None

    def put(self, key, payload):
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        size = len(text.encode())
        if size > min(self.max_bytes, 8_000_000):
            return False
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("INSERT OR REPLACE INTO stages VALUES (?,?,?,?)",
                         (key, text, hashlib.sha256(text.encode()).hexdigest(), time.time()))
            while conn.execute("SELECT COALESCE(SUM(length(CAST(payload AS BLOB))),0) FROM stages").fetchone()[0] > self.max_bytes:
                conn.execute("DELETE FROM stages WHERE id=(SELECT id FROM stages ORDER BY accessed LIMIT 1)")
        return True

    def stats(self):
        with self.connect() as conn:
            count, size = conn.execute("SELECT COUNT(*),COALESCE(SUM(length(CAST(payload AS BLOB))),0) FROM stages").fetchone()
        return {"entries": count, "bytes": size, "max_bytes": self.max_bytes}


class IngestionPipeline:
    def __init__(self, prepare, *, cache=None, revision=None, progress=lambda event: None,
                 assert_lease=lambda: None, context_fingerprint=None):
        self.prepare, self.cache = prepare, cache
        self.revision = revision or fingerprint([parser_revision(),
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            hashlib.sha256(Path(__file__).parents[1].joinpath("application/ingestion.py").read_bytes()).hexdigest()])
        self.revision = fingerprint([self.revision, context_fingerprint])
        self.progress, self.assert_lease = progress, assert_lease

    def prepare_batch(self, request):
        documents, keys = [], set()
        for source in request.documents:
            self.assert_lease()
            key = fingerprint(["prepare_v1", self.revision, source.model_dump()])
            cached = self.cache.get(key) if self.cache else None
            if cached is None:
                _, document = self.prepare(source)
                if not document["chunks"]:
                    raise ValueError(f"No chunks extracted: {source.source_file}")
                self.assert_lease()
                # Validate normalized structure before caching, never claim human verification.
                normalize_report(document)
                if self.cache:
                    self.cache.put(key, serialize_document(document))
            else:
                document = deserialize_document(cached)
                normalize_report(document)
            if document["logical_key"] in keys:
                raise ValueError("Batch contains duplicate logical documents")
            keys.add(document["logical_key"])
            documents.append(document)
            self.progress({"stage": "prepare_and_normalize", "key": key,
                           "cache_hit": cached is not None, "chunks": len(document["chunks"])})
        self.assert_lease()
        return documents
