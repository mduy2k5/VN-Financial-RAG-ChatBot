"""Bounded immutable request scopes; cached entries never change after publication."""
from collections import OrderedDict
from contextvars import ContextVar
from concurrent.futures import Future
from dataclasses import dataclass
from threading import RLock
from typing import Any

from ..structure.normalized import fingerprint


@dataclass(frozen=True)
class RequestSnapshot:
    chunks: tuple
    retriever: Any
    documents: tuple = ()
    snapshot_id: str = "legacy"
    source_policy: str = "source_report"


request_snapshot: ContextVar[RequestSnapshot | None] = ContextVar("financial_snapshot", default=None)


class ScopeCache:
    def __init__(self, max_scopes: int = 8, max_chunks: int = 20000,
                 max_bytes: int = 64 * 1024 * 1024):
        self.max_scopes = max_scopes
        self.max_chunks = max_chunks
        self.entries: OrderedDict[str, RequestSnapshot] = OrderedDict()
        self.max_bytes = max_bytes
        self.sizes: dict[str, int] = {}
        self.pending: dict[str, Future[RequestSnapshot]] = {}
        self.lock = RLock()

    @staticmethod
    def key(documents: list[dict]) -> str:
        return fingerprint(sorted((item["logical_key"], item.get("active_projection_id"),
                                   item.get("ingested_at")) for item in documents))

    def get(self, key: str):
        with self.lock:
            value = self.entries.get(key)
            if value is not None:
                self.entries.move_to_end(key)
            return value

    def put(self, snapshot: RequestSnapshot):
        import json
        # A conservative serialized-payload budget, not a promise about process RSS.
        size = sum(len(json.dumps(chunk.model_dump(mode="json") if hasattr(chunk, "model_dump")
                                  else chunk, ensure_ascii=False).encode()) * 4 + 512
                   for chunk in snapshot.chunks)
        with self.lock:
            if self.max_scopes < 1 or len(snapshot.chunks) > self.max_chunks or size > self.max_bytes:
                return
            self.entries[snapshot.snapshot_id] = snapshot
            self.sizes[snapshot.snapshot_id] = size
            self.entries.move_to_end(snapshot.snapshot_id)
            while (len(self.entries) > self.max_scopes
                   or sum(len(item.chunks) for item in self.entries.values()) > self.max_chunks
                   or sum(self.sizes.values()) > self.max_bytes):
                key, _ = self.entries.popitem(last=False)
                self.sizes.pop(key, None)

    def get_or_build(self, key, build):
        """One concurrent build per immutable identity; waiters share failures too."""
        with self.lock:
            cached = self.get(key)
            if cached is not None:
                return cached
            future = self.pending.get(key)
            owner = future is None
            if owner:
                future = self.pending[key] = Future()
        if not owner:
            return future.result()
        try:
            snapshot = build()
            if snapshot.snapshot_id != key:
                raise ValueError("Built snapshot does not match requested identity")
            self.put(snapshot)
            future.set_result(snapshot)
            return snapshot
        except BaseException as exc:
            future.set_exception(exc)
            raise
        finally:
            with self.lock:
                self.pending.pop(key, None)

    def clear(self):
        with self.lock:
            self.entries.clear()
            self.sizes.clear()

    def stats(self):
        with self.lock:
            return {"scopes": len(self.entries),
                    "chunks": sum(len(item.chunks) for item in self.entries.values()),
                    "max_chunks": self.max_chunks, "estimated_bytes": sum(self.sizes.values()),
                    "max_bytes": self.max_bytes, "pending_builds": len(self.pending)}
