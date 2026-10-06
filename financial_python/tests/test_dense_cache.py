import sqlite3
from threading import Event

import numpy as np
import pytest

from financial_rag.embedding.e5_adapter import E5EmbeddingAdapter
from financial_rag.retrieval.dense_retriever import DenseRetriever
from financial_rag.schemas import FinancialChunk
from financial_rag.storage.embedding_cache import EmbeddingCache


def _chunk(identifier, text):
    return FinancialChunk(chunk_id=identifier, source_file="new.md", content=text)


def _adapter(monkeypatch, revision="digest-one"):
    adapter = E5EmbeddingAdapter(base_url="http://embedding", model="e5")
    calls = []
    monkeypatch.setattr(adapter, "model_revision", lambda: revision)
    def embed(texts):
        calls.extend(texts)
        return [[1.0, 2.0] for text in texts]
    monkeypatch.setattr(adapter, "embed_passages", embed)
    return adapter, calls


def test_restart_reuses_cache_and_changed_passage_is_reembedded(tmp_path, monkeypatch):
    monkeypatch.setenv("FINANCIAL_DENSE_CACHE_PATH", str(tmp_path / "cache.sqlite"))
    adapter, calls = _adapter(monkeypatch)
    chunks = [_chunk("a", "Alpha"), _chunk("b", "Beta")]
    first = DenseRetriever(chunks, adapter)
    first.index()
    assert len(calls) == 2
    calls.clear()
    restored = DenseRetriever(chunks, adapter)
    restored.index()
    assert calls == []
    assert restored.cached_vector_count == 2
    np.testing.assert_allclose(first._normalized_vectors, restored._normalized_vectors)
    changed = DenseRetriever([chunks[0], _chunk("b", "New Beta")], adapter)
    changed.index()
    assert len(calls) == 1
    assert changed.cached_vector_count == 1
    deleted = DenseRetriever([chunks[0]], adapter)
    calls.clear()
    deleted.index()
    assert calls == []
    assert deleted.chunk_ids == ["a"]


def test_changed_model_digest_never_reuses_cache_or_memory_seed(tmp_path, monkeypatch):
    monkeypatch.setenv("FINANCIAL_DENSE_CACHE_PATH", str(tmp_path / "cache.sqlite"))
    adapter, _ = _adapter(monkeypatch)
    chunks = [_chunk("a", "Alpha")]
    first = DenseRetriever(chunks, adapter)
    first.index()
    new_adapter, calls = _adapter(monkeypatch, "digest-two")
    changed = DenseRetriever(chunks, new_adapter)
    changed.index(seed_retriever=first)
    assert len(calls) == 1
    assert changed.cached_vector_count == 0
    assert changed.embedding_identity != first.embedding_identity


def test_corrupt_cache_record_is_ignored_and_recomputed(tmp_path, monkeypatch):
    path = tmp_path / "cache.sqlite"
    monkeypatch.setenv("FINANCIAL_DENSE_CACHE_PATH", str(path))
    adapter, calls = _adapter(monkeypatch)
    chunks = [_chunk("a", "Alpha")]
    first = DenseRetriever(chunks, adapter)
    first.index()
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE vectors SET vector=?", (b"corrupt",))
    calls.clear()
    restored = DenseRetriever(chunks, adapter)
    restored.index()
    assert len(calls) == 1
    assert restored.index_status == "ready"


def test_no_immutable_revision_disables_durable_reuse(tmp_path, monkeypatch):
    monkeypatch.setenv("FINANCIAL_DENSE_CACHE_PATH", str(tmp_path / "cache.sqlite"))
    adapter, calls = _adapter(monkeypatch, None)
    chunks = [_chunk("a", "Alpha")]
    DenseRetriever(chunks, adapter).index()
    DenseRetriever(chunks, adapter).index()
    assert len(calls) == 2
    assert not (tmp_path / "cache.sqlite").exists()


def test_cancelled_index_does_not_publish_partly_embedded_matrix(tmp_path, monkeypatch):
    monkeypatch.setenv("FINANCIAL_DENSE_CACHE_PATH", str(tmp_path / "cache.sqlite"))
    adapter, calls = _adapter(monkeypatch)
    event = Event()
    event.set()
    retriever = DenseRetriever([_chunk("a", "Alpha")], adapter)
    retriever.index(cancel_event=event)
    assert retriever.index_status == "cancelled"
    assert retriever._normalized_vectors is None
    assert calls == []


def test_non_finite_embedding_is_not_published_or_cached(tmp_path, monkeypatch):
    path = tmp_path / "cache.sqlite"
    monkeypatch.setenv("FINANCIAL_DENSE_CACHE_PATH", str(path))
    adapter, _ = _adapter(monkeypatch)
    monkeypatch.setattr(adapter, "embed_passages", lambda texts: [[float("nan"), 1]])
    retriever = DenseRetriever([_chunk("a", "Alpha")], adapter)
    with pytest.raises(ValueError, match="non-finite"):
        retriever.index()
    assert retriever.index_status == "error"
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM vectors").fetchone()[0] == 0


def test_cache_namespace_isolated(tmp_path):
    one = EmbeddingCache(tmp_path / "cache.sqlite", "one")
    one.save(["Alpha"], np.array([[1.0, 0.0]], dtype=np.float32))
    two = EmbeddingCache(tmp_path / "cache.sqlite", "two")
    assert len(one.load(["Alpha"])) == 1
    assert two.load(["Alpha"]) == {}
