"""Conservative Structural, Filter, and Validate query operators."""

from __future__ import annotations

import json
from typing import Protocol

import requests

from ..query.directives import QueryDirectivePlan, normalize_directive_text
from ..schemas import FinancialChunk, ParsedQuery, RetrievalResult


_STOP_WORDS = {
    "cac", "cho", "cua", "la", "mot", "nhung", "phan", "noi", "dung", "khong",
    "theo", "va", "voi", "bao", "cao", "hay", "duoc", "trong", "tai", "ve",
    "the", "of", "and", "the", "to", "ignore", "exclude", "skip", "bo", "qua",
}
_LOW_SIGNAL_CONSTRAINTS = {
    "bao cao",
    "bao cao tai chinh",
    "du lieu",
    "so lieu",
    "tai chinh",
    "tai lieu",
    "thong tin",
}


def _tokens(value: str) -> list[str]:
    return [token for token in normalize_directive_text(value).split() if token not in _STOP_WORDS]


def _matches_phrase(phrase: str, text: str) -> bool:
    normalized_phrase = normalize_directive_text(phrase)
    normalized_text = normalize_directive_text(text)
    if not normalized_phrase or not normalized_text:
        return False
    if f" {normalized_phrase} " in f" {normalized_text} ":
        return True
    phrase_tokens = _tokens(phrase)
    text_tokens = set(_tokens(text))
    if len(phrase_tokens) < 2:
        return bool(phrase_tokens and phrase_tokens[0] in text_tokens)
    overlap = sum(token in text_tokens for token in phrase_tokens)
    return overlap >= 2 and overlap / len(set(phrase_tokens)) >= 0.8


class ChunkFilterClassifier(Protocol):
    def classify(
        self,
        query: str,
        exclusions: list[str],
        chunks: list[FinancialChunk],
    ) -> list[bool | None]: ...


class OllamaChunkFilter:
    """Optional batched SLM filter; malformed/unavailable output fails open."""

    def __init__(self, ollama_url: str, model_name: str, timeout_seconds: float = 45.0):
        self.ollama_url = ollama_url.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds

    def classify(
        self,
        query: str,
        exclusions: list[str],
        chunks: list[FinancialChunk],
    ) -> list[bool | None]:
        if not chunks:
            return []
        payload = [
            {"id": index, "text": chunk.content[:1_200]}
            for index, chunk in enumerate(chunks[:10])
        ]
        prompt = (
            "Bạn lọc đoạn bằng chứng cho hỏi đáp báo cáo tài chính. "
            "Với từng đoạn, trả lời KEEP nếu nó có thể chứa thông tin hữu ích để trả lời câu hỏi "
            "hoặc không chắc chắn; chỉ trả lời DROP nếu chắc chắn đoạn chỉ chứa nội dung bị loại trừ. "
            "Không được loại đoạn chỉ vì nó có nhắc tới nội dung nhiễu nếu vẫn có số liệu liên quan.\n"
            f"Câu hỏi: {query}\n"
            f"Nội dung cần bỏ qua: {json.dumps(exclusions, ensure_ascii=False)}\n"
            "Trả về JSON theo dạng {\"decisions\":[{\"id\":0,\"decision\":\"KEEP\"}]} "
            "cho từng id.\n"
            f"Các đoạn: {json.dumps(payload, ensure_ascii=False)}"
        )
        try:
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "think": False,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0, "num_predict": 256, "num_ctx": 4096},
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")
            parsed = json.loads(content)
            decisions = parsed.get("decisions", []) if isinstance(parsed, dict) else []
            by_id = {
                int(item["id"]): str(item["decision"]).strip().upper()
                for item in decisions
                if isinstance(item, dict)
                and str(item.get("decision", "")).strip().upper() in {"KEEP", "DROP"}
            }
            output: list[bool | None] = [None] * len(chunks)
            for index in range(min(len(chunks), 10)):
                decision = by_id.get(index)
                if decision:
                    output[index] = decision == "KEEP"
            return output
        except (requests.RequestException, ValueError, TypeError, KeyError):
            return [None] * len(chunks)


class StructuralOperator:
    """Apply explicit structure hints to the already metadata-scoped corpus.

    A missing structural match is treated as an imprecise hint and falls back
    to the complete metadata-scoped candidate set.
    """

    @classmethod
    def select(
        cls,
        chunks: list[FinancialChunk],
        directives: QueryDirectivePlan | None,
    ) -> tuple[list[FinancialChunk], bool]:
        if not chunks or directives is None:
            return chunks, False
        preferred_types = set(directives.structural.preferred_chunk_types)
        if directives.structural.require_complete_rows:
            preferred_types.update({"table", "financial_item", "parent_child"})
        preferred_sections = [
            normalize_directive_text(section)
            for section in directives.structural.preferred_sections
            if normalize_directive_text(section)
        ]
        if not preferred_types and not preferred_sections:
            return chunks, False

        selected: list[FinancialChunk] = []
        for chunk in chunks:
            chunk_type = getattr(chunk.chunk_type, "value", str(chunk.chunk_type))
            type_match = chunk_type in preferred_types
            section_values = [
                chunk.section or "",
                chunk.table_title or "",
                str(chunk.metadata.get("section_title", "")),
                str(chunk.metadata.get("table_title", "")),
            ]
            section_match = any(
                _matches_phrase(preferred, value)
                for preferred in preferred_sections
                for value in section_values
                if value
            )
            if type_match or section_match:
                selected.append(chunk)

        if not selected:
            return chunks, False
        return selected, len(selected) != len(chunks)


class FilterOperator:
    """Conservatively remove chunks matching explicit user exclusions only."""

    @classmethod
    def apply(
        cls,
        results: list[RetrievalResult],
        directives: QueryDirectivePlan | None,
        parsed_query: ParsedQuery,
        classifier: ChunkFilterClassifier | None = None,
    ) -> tuple[list[RetrievalResult], bool]:
        if not results or directives is None or not directives.filters.exclude_phrases:
            return results, False

        anchors = [parsed_query.canonical_metric or parsed_query.primary_metric or ""]
        anchors.extend(parsed_query.metric_aliases)
        anchors = [anchor for anchor in anchors if anchor.strip()]
        decisions = (
            classifier.classify(
                parsed_query.original_query,
                directives.filters.exclude_phrases,
                [result.chunk for result in results],
            )
            if classifier is not None
            else []
        )
        kept: list[RetrievalResult] = []
        removed = 0
        for index, result in enumerate(results):
            content = result.chunk.content
            semantic_decision = decisions[index] if index < len(decisions) else None
            excluded = (
                not semantic_decision
                if semantic_decision is not None
                else any(
                    _matches_phrase(phrase, content)
                    for phrase in directives.filters.exclude_phrases
                )
            )
            # Keep a chunk that directly carries the requested metric even if
            # an exclusion phrase appears elsewhere in the same table/page.
            contains_anchor = any(_matches_phrase(anchor, content) for anchor in anchors)
            if excluded and not contains_anchor:
                removed += 1
            else:
                kept.append(result)

        # Never turn an explicit filter into an empty-context query.
        if not kept:
            return results, False
        return kept, removed > 0


class ValidateOperator:
    """Rank candidate evidence against positive and negative query constraints.

    It never drops evidence. If constraints provide no ranking signal, the
    original retrieval order is retained (the Halo ranking fallback).
    """

    @classmethod
    def rerank(
        cls,
        results: list[RetrievalResult],
        directives: QueryDirectivePlan | None,
    ) -> tuple[list[RetrievalResult], bool]:
        if not results or directives is None:
            return results, False
        positives = [
            term
            for term in directives.validation.positive_constraints
            if normalize_directive_text(term) not in _LOW_SIGNAL_CONSTRAINTS
        ]
        negatives = [
            term
            for term in directives.validation.negative_constraints
            if normalize_directive_text(term) not in _LOW_SIGNAL_CONSTRAINTS
        ]
        if not positives and not negatives:
            return results, False

        scored: list[tuple[int, int, RetrievalResult]] = []
        for index, result in enumerate(results):
            text = result.chunk.content
            score = 0
            score += 2 * sum(_matches_phrase(term, text) for term in positives)
            score -= 3 * sum(_matches_phrase(term, text) for term in negatives)
            scored.append((score, index, result))

        scores = {score for score, _, _ in scored}
        if len(scores) < 2:
            return results, False
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [result for _, _, result in scored], True


def directive_operator_state(plan: QueryDirectivePlan | None) -> bool:
    """Whether the plan contains any operator capable of changing execution."""
    if plan is None:
        return False
    return bool(
        plan.structural.preferred_chunk_types
        or plan.structural.preferred_sections
        or plan.structural.require_complete_rows
        or plan.filters.exclude_phrases
        or plan.validation.positive_constraints
        or plan.validation.negative_constraints
    )
