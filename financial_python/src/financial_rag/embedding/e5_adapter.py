import os
import re
from typing import List, Optional, Sequence
from urllib.parse import urljoin

import requests


class E5EmbeddingAdapter:
    """Client cho endpoint OpenAI-compatible của Text Embeddings Inference."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        timeout_seconds: Optional[float] = None,
    ):
        self.base_url = (
            base_url
            or os.getenv("FINANCIAL_EMBEDDING_URL")
            or os.getenv("OLLAMA_URL")
            or "http://localhost:11434"
        ).rstrip("/")
        self.model = (
            model
            or os.getenv("FINANCIAL_EMBEDDING_MODEL")
            or "qllama/multilingual-e5-large:q4_k_m"
        )
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.timeout_seconds = timeout_seconds or float(
            os.getenv("FINANCIAL_EMBEDDING_TIMEOUT_SECONDS", "90")
        )

    def embed_query(self, query: str) -> List[float]:
        return self._call_embed([self._with_prefix(query, self.query_prefix)])[0]

    def model_revision(self) -> Optional[str]:
        """Require an immutable revision before reusing durable embeddings."""
        configured = os.getenv("FINANCIAL_EMBEDDING_REVISION", "").strip()
        if configured:
            return configured
        try:
            response = requests.get(
                urljoin(f"{self.base_url.removesuffix('/v1')}/", "api/tags"), timeout=10,
            )
            response.raise_for_status()
            for model in response.json().get("models", []):
                if self.model in {model.get("name"), model.get("model")} and model.get("digest"):
                    return str(model["digest"])
        except (requests.RequestException, ValueError, AttributeError, TypeError):
            pass
        return None

    def embed_passage(self, passage: str) -> List[float]:
        return self._call_embed([self._with_prefix(passage, self.passage_prefix)])[0]

    def embed_passages(self, passages: Sequence[str]) -> List[List[float]]:
        return self._call_embed([self._with_prefix(text, self.passage_prefix) for text in passages])

    @staticmethod
    def _with_prefix(text: str, prefix: str) -> str:
        # QueryUnderstandingEngine may return a prefixed query; strip known
        # prefixes first so the model receives exactly one E5 task prefix.
        body = re.sub(r"^(?:query|passage|search_query|search_document):\s*", "", text.strip(), flags=re.IGNORECASE)
        return f"{prefix}{body}"

    def _endpoint(self) -> str:
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/embeddings"
        return urljoin(f"{self.base_url}/", "v1/embeddings")

    def _call_embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        response = requests.post(
            self._endpoint(),
            json={"model": self.model, "input": list(texts)},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise ValueError("Embedding service returned an unexpected number of vectors")

        ordered = sorted(data, key=lambda item: item.get("index", 0))
        vectors = [item.get("embedding") for item in ordered]
        if any(not isinstance(vector, list) or not vector for vector in vectors):
            raise ValueError("Embedding service returned an empty or invalid vector")
        return vectors
