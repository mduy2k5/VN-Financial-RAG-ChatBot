from __future__ import annotations

import re
from typing import Any, Protocol

from pydantic import BaseModel, Field

from ..schemas import ParsedQuery, QueryIntent
from .catalog import GuideCatalog


class QueryEmbedder(Protocol):
    def embed_query(self, query: str) -> list[float]: ...


class GuideMatch(BaseModel):
    guide_id: str | None = None
    domain: str | None = None
    task: str | None = None
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    accepted: bool = False
    method: str
    reason: str


class GuideMatcher(Protocol):
    def match(
        self,
        question: str,
        *,
        parsed: ParsedQuery | None = None,
    ) -> GuideMatch: ...


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", text.casefold())).strip()


def _intent_value(parsed: ParsedQuery | None) -> str | None:
    if parsed is None:
        return None
    intent = parsed.intent
    return intent.value if isinstance(intent, QueryIntent) else str(intent)


class RuleBasedGuideMatcher:
    """Offline route prototype; this is not a semantic-similarity substitute."""

    METHOD = "offline_rules_v1"

    def __init__(
        self,
        catalog: GuideCatalog,
        *,
        confidence_threshold: float = 0.75,
    ):
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.catalog = catalog
        self.confidence_threshold = confidence_threshold
        self._routes = catalog.by_route()

    def _domain(
        self,
        question: str,
        parsed: ParsedQuery | None,
    ) -> tuple[str | None, float, str]:
        text = _normalize(question)
        if parsed and (parsed.metric_id or parsed.primary_metric):
            return "financial", 0.99, "financial metric resolved by existing parser"

        candidates: list[tuple[int, int, str, float, list[str]]] = []
        priority = {"esg": 3, "financial": 2, "annual_report": 1}
        for domain, keywords in self.catalog.domains.items():
            hits = [keyword for keyword in keywords if _normalize(keyword) in text]
            if not hits:
                continue
            confidence = min(0.98, 0.84 + 0.04 * (len(hits) - 1))
            candidates.append((len(hits), priority.get(domain, 0), domain, confidence, hits))
        if not candidates:
            return None, 0.0, "no domain signal"
        _, _, domain, confidence, hits = max(candidates)
        return domain, confidence, f"domain keywords: {', '.join(hits[:3])}"

    def _task(
        self,
        question: str,
        parsed: ParsedQuery | None,
    ) -> tuple[str | None, float, str]:
        text = _normalize(question)
        intent = _intent_value(parsed)

        # Analytical requests need a multi-evidence recipe even when they also
        # mention a trend, a comparison, or an explanation.
        for task in ("driver_analysis", "evaluation", "analysis"):
            keywords = self.catalog.tasks.get(task, [])
            hits = [keyword for keyword in keywords if _normalize(keyword) in text]
            if hits:
                confidence = min(0.98, 0.91 + 0.02 * (len(hits) - 1))
                return task, confidence, f"analytical task keywords: {', '.join(hits[:3])}"

        if intent == QueryIntent.GROWTH_RATE.value:
            return "trend", 0.97, "existing parser intent: growth_rate"
        if intent == QueryIntent.COMPARISON.value:
            return "comparison", 0.97, "existing parser intent: comparison"
        if intent == QueryIntent.EXPLANATION.value:
            return "explanation", 0.96, "existing parser intent: explanation"

        # More specific temporal/relational tasks precede generic summary and lookup.
        task_order = ("trend", "comparison", "explanation", "summary", "lookup")
        for task in task_order:
            keywords = self.catalog.tasks.get(task, [])
            hits = [keyword for keyword in keywords if _normalize(keyword) in text]
            if hits:
                confidence = min(0.98, 0.88 + 0.03 * (len(hits) - 1))
                return task, confidence, f"task keywords: {', '.join(hits[:3])}"

        if parsed and parsed.metric_id:
            return "lookup", 0.95, "financial metric resolved by existing parser"
        return None, 0.0, "no task signal"

    def match(
        self,
        question: str,
        *,
        parsed: ParsedQuery | None = None,
    ) -> GuideMatch:
        domain, domain_score, domain_reason = self._domain(question, parsed)
        task, task_score, task_reason = self._task(question, parsed)
        if domain is None or task is None:
            return GuideMatch(
                domain=domain,
                task=task,
                method=self.METHOD,
                reason=f"fallback: {domain_reason}; {task_reason}",
            )

        guide = self._routes.get((domain, task))
        if guide is None:
            return GuideMatch(
                domain=domain,
                task=task,
                score=min(domain_score, task_score),
                method=self.METHOD,
                reason=f"no guide for route {domain}/{task}; general fallback",
            )

        score = min(domain_score, task_score)
        accepted = score >= self.confidence_threshold
        return GuideMatch(
            guide_id=guide.guide_id if accepted else None,
            domain=domain,
            task=task,
            score=score,
            accepted=accepted,
            method=self.METHOD,
            reason=(
                f"{domain_reason}; {task_reason}"
                if accepted
                else f"below local prototype threshold {self.confidence_threshold:.2f}"
            ),
        )


class RedisVLGuideMatcher:
    """Read-only RedisVL query adapter for an already-created isolated index.

    The index and embedding provider are injected. This class never creates,
    loads, clears, or deletes Redis data. A cosine-distance cutoff must be
    calibrated on an independently labeled guide-matching set before use.
    """

    METHOD = "redisvl_vector_query"

    def __init__(
        self,
        *,
        index: Any,
        embedder: QueryEmbedder,
        catalog: GuideCatalog,
        score_threshold: float,
        vector_field_name: str = "embedding",
        top_k: int = 5,
        route_classifier: RuleBasedGuideMatcher | None = None,
    ):
        if not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold must be between 0 and 1")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.index = index
        self.embedder = embedder
        self.catalog = catalog
        self.score_threshold = score_threshold
        self.vector_field_name = vector_field_name
        self.top_k = top_k
        self._guides = catalog.by_id()
        self.route_classifier = route_classifier or RuleBasedGuideMatcher(catalog)

    def match(
        self,
        question: str,
        *,
        parsed: ParsedQuery | None = None,
        domain: str | None = None,
        task: str | None = None,
    ) -> GuideMatch:
        # Extract route signals before vector matching. They become metadata
        # filters, while company/period slots remain in the QueryPlan.
        if not domain or not task:
            route = self.route_classifier.match(question, parsed=parsed)
            if route.accepted:
                domain = domain or route.domain
                task = task or route.task
        try:
            from redisvl.query import VectorQuery
            from redisvl.query.filter import Tag
        except ImportError as exc:
            raise RuntimeError(
                "RedisVLGuideMatcher requires the optional redisvl package"
            ) from exc

        filter_expression = Tag("active") == "true"
        if domain:
            filter_expression = filter_expression & (Tag("domain") == domain)
        if task:
            filter_expression = filter_expression & (Tag("task") == task)

        vector = self.embedder.embed_query(question)
        if not vector:
            return GuideMatch(method=self.METHOD, reason="empty query embedding")
        query = VectorQuery(
            vector=vector,
            vector_field_name=self.vector_field_name,
            filter_expression=filter_expression,
            num_results=self.top_k,
            return_fields=["guide_id", "domain", "task", "guide_version", "active"],
        )
        results = self.index.query(query)
        candidates: list[tuple[float, dict[str, Any], float]] = []
        for row in results:
            guide_id = str(row.get("guide_id", ""))
            guide = self._guides.get(guide_id)
            if guide is None:
                continue
            if str(row.get("guide_version", "")) != str(self.catalog.version):
                continue
            if str(row.get("active", "true")).casefold() not in {"true", "1", "yes"}:
                continue
            if domain and guide.domain != domain:
                continue
            if task and guide.task != task:
                continue
            try:
                distance = float(row["vector_distance"])
            except (KeyError, TypeError, ValueError):
                continue
            # RedisVL returns cosine distance; map [0, 2] to a provisional [1, 0] score.
            score = max(0.0, min(1.0, 1.0 - distance / 2.0))
            candidates.append((score, row, distance))

        if not candidates:
            return GuideMatch(method=self.METHOD, reason="no active guide result for current version")
        score, row, distance = max(candidates, key=lambda candidate: candidate[0])
        accepted = score >= self.score_threshold
        return GuideMatch(
            guide_id=str(row["guide_id"]) if accepted else None,
            domain=str(row.get("domain") or domain or "") or None,
            task=str(row.get("task") or task or "") or None,
            score=score,
            accepted=accepted,
            method=self.METHOD,
            reason=(
                f"cosine distance {distance:.4f}"
                if accepted
                else f"cosine-derived score below configured threshold {self.score_threshold:.2f}"
            ),
        )
