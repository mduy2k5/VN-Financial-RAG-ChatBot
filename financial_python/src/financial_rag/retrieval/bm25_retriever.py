import re
import json
import os
from pathlib import Path
import tempfile
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from ..schemas import FinancialChunk
from ..embedding.text_builder import TextBuilder
from ..structure.normalized import fingerprint

class BM25Retriever:
    """
    Bộ truy xuất từ khóa BM25 chuyên biệt cho tài chính:
    - Tokenizer chuẩn hóa tiếng Việt cơ bản.
    - Chấm điểm và xếp hạng danh sách chunk IDs.
    """

    def __init__(self, chunks: List[FinancialChunk]):
        self.chunks = chunks
        self.chunk_ids = [c.chunk_id for c in self.chunks]
        # Use the same source-grounded representation as dense retrieval. In
        # particular, exclude canonical aliases inserted by older chunkers;
        # those labels can be wrong and should not dominate lexical search.
        texts = [TextBuilder.build_chunk_text(c) for c in self.chunks]
        key = fingerprint(["bm25_vi_tokenizer_v1", list(zip(self.chunk_ids, texts))])
        directory = os.getenv("FINANCIAL_SPARSE_INDEX_DIR", "").strip()
        path = Path(directory) / f"{key}.json" if directory else None
        tokens = None
        self.persisted_index_hit = False
        if path and path.exists():
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                candidate = record["tokens"]
                if (record.get("key") == key and isinstance(candidate, list) and len(candidate) == len(chunks)
                        and all(isinstance(row, list) and all(isinstance(token, str) for token in row) for row in candidate)):
                    tokens = candidate
                    self.persisted_index_hit = True
            except (OSError, ValueError, KeyError, TypeError):
                pass
        self.corpus_tokens = tokens if tokens is not None else [self._tokenize(text) for text in texts]
        if path and tokens is None:
            temporary = None
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = json.dumps({"key": key, "tokens": self.corpus_tokens}, ensure_ascii=False)
                budget = max(0, int(os.getenv("FINANCIAL_SPARSE_DISK_BYTES", "67108864")))
                if len(payload.encode()) <= budget:
                    files = sorted(path.parent.glob("*.json"), key=lambda item: item.stat().st_mtime)
                    size = sum(item.stat().st_size for item in files)
                    for old in files:
                        if size + len(payload.encode()) <= budget:
                            break
                        size -= old.stat().st_size
                        old.unlink(missing_ok=True)
                    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                                     prefix="sparse-", suffix=".tmp", delete=False) as stream:
                        temporary = Path(stream.name)
                        stream.write(payload)
                    os.replace(temporary, path)
            except OSError:
                pass  # Cache failure must not make source data unavailable.
            finally:
                if temporary:
                    temporary.unlink(missing_ok=True)
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def _tokenize(self, text: str) -> List[str]:
        text_clean = re.sub(r"[^\w\s]", " ", text.lower())
        return [t for t in text_clean.split() if len(t) > 0]

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        if not self.bm25 or not self.corpus_tokens:
            return []
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        scored_pairs = list(zip(self.chunk_ids, [float(s) for s in scores]))
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        return scored_pairs[:top_k]
