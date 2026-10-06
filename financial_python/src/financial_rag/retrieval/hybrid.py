from __future__ import annotations

import logging
from collections import OrderedDict
import math
import os
import re
from threading import Event, Lock, Thread
import unicodedata
from typing import TYPE_CHECKING, List

from ..schemas import (
    FinancialChunk,
    ParsedQuery,
    QueryIntent,
    RetrievalResult,
    ChunkType,
)
from ..query.directives import QueryDirectivePlan
from ..structure.source_groups import display_source_item_label, source_item_label
from .rrf import reciprocal_rank_fusion
from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever
from .operators import StructuralOperator

if TYPE_CHECKING:
    from .qdrant_dense import QdrantDenseRetriever
from .runtime import request_snapshot


_METRIC_STOP_WORDS = {
    "cac", "cua", "va", "tu", "theo", "trong", "la", "mot", "tai", "nam",
    "ky", "bao", "cao", "tai", "chinh", "so", "ve", "duoc", "co", "nhu",
}
_QUERY_OPERATION_WORDS = {
    "cau", "thanh", "phan", "tong", "hop", "nhat", "gia", "tri", "bao",
    "nhieu", "nao", "la", "khong", "duoc", "hay", "cho", "biet",
}
_DENSE_INDEX_LOCK = Lock()
logger = logging.getLogger(__name__)


def _metric_tokens(value: str) -> set[str]:
    decomposed = unicodedata.normalize("NFD", display_source_item_label(value).casefold()).replace("đ", "d")
    plain = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return {
        token for token in re.findall(r"[a-z0-9]+", plain)
        if token not in _METRIC_STOP_WORDS
    }


def _normalized_metric_label(value: str) -> str:
    """Normalize a source label for comparison without trusting mapped aliases."""
    decomposed = unicodedata.normalize(
        "NFD", display_source_item_label(value).casefold()
    ).replace("đ", "d")
    plain = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return " ".join(re.findall(r"[a-z0-9]+", plain))


def _raw_metric_label(chunk: FinancialChunk) -> str:
    """Return the label extracted from the source row, not a mapped metric name."""
    return source_item_label(chunk)


def _raw_label_matches(raw_label: str, expected_labels: set[str]) -> bool:
    raw = _normalized_metric_label(raw_label)
    if not raw:
        return False
    for expected in expected_labels:
        label = _normalized_metric_label(expected)
        if not label:
            continue
        if raw == label:
            return True
        # Labels often carry a leading "Tổng", a table index, or an OCR
        # clarification. Require a meaningful phrase to avoid matching broad
        # words such as "tài sản" against every row in a statement.
        if len(label) >= 10 and len(label.split()) >= 2 and label in raw:
            return True
        raw_tokens = _metric_tokens(raw)
        label_tokens = _metric_tokens(label)
        # Official statements and notes can describe the same row with a
        # shorter label (for example, "Tổng lợi nhuận trước thuế" versus
        # "Tổng lợi nhuận kế toán trước thuế"). Accept only a meaningful
        # source phrase whose tokens are all present in the requested label;
        # very short labels remain too broad to match this way.
        if len(raw_tokens) >= 3 and raw_tokens.issubset(label_tokens):
            return True
    return False


def _source_label_affinity(query: str, source_label: str) -> float:
    """Measure lexical agreement with the extracted row label, without schema aliases."""
    query = re.sub(r"\b(?:19|20)\d{2}\b", " ", query)
    query_tokens = _metric_tokens(query) - _QUERY_OPERATION_WORDS
    label_tokens = _metric_tokens(source_label) - _QUERY_OPERATION_WORDS
    if not query_tokens or not label_tokens:
        return 0.0
    overlap = len(query_tokens & label_tokens)
    return overlap / len(query_tokens | label_tokens)


def _covered_years(chunk: FinancialChunk) -> set[int]:
    values = chunk.metadata.get("period_values") or {}
    years = {int(year) for year in values if str(year).isdigit()} if isinstance(values, dict) else set()
    for year in (
        chunk.fiscal_year, chunk.comparative_fiscal_year,
        chunk.metadata.get("comparative_fiscal_year"),
    ):
        if year is not None and str(year).isdigit():
            years.add(int(year))
    return years


def _scope_reports(
    chunks: List[FinancialChunk], requested_years: list[int],
    company_order: list[str], requested_report_type: str | None,
    source_policy: str = "source_report",
) -> List[FinancialChunk]:
    """Choose one filing basis per issuer, then scope every requested period.

    Period coverage precedes the default preference for consolidated reports.
    An explicit basis remains a hard constraint even when it is incomplete.
    Exact-year reports are preferred independently for each requested period.
    """
    companies = company_order or list(dict.fromkeys(chunk.company_code or "" for chunk in chunks))
    selected: list[FinancialChunk] = []
    requested = set(requested_years)
    for company in companies:
        by_basis: dict[str | None, list[FinancialChunk]] = {}
        for chunk in chunks:
            if (chunk.company_code or "").casefold() != company.casefold():
                continue
            if requested_report_type and chunk.report_type != requested_report_type:
                continue
            by_basis.setdefault(chunk.report_type, []).append(chunk)
        if not by_basis:
            continue

        def basis_preference(basis: str | None) -> tuple[int, int]:
            periods = set().union(*(_covered_years(chunk) for chunk in by_basis[basis]))
            coverage = len(requested & periods) if requested else 0
            default_preference = {"consolidated": 2, "separate": 1}.get(basis or "", 0)
            return coverage, default_preference

        basis_chunks = by_basis[max(by_basis, key=basis_preference)]
        if not requested:
            selected.extend(basis_chunks)
            continue
        selected_ids: set[str] = set()
        for year in requested_years:
            exact = [chunk for chunk in basis_chunks if chunk.fiscal_year == year]
            candidates = exact or [chunk for chunk in basis_chunks if year in _covered_years(chunk)]
            snapshot = request_snapshot.get()
            if source_policy == "latest_restated" or (snapshot and snapshot.source_policy == "latest_restated"):
                candidates = [chunk for chunk in basis_chunks if year in _covered_years(chunk)]
                latest = max((chunk.fiscal_year or 0 for chunk in candidates), default=0)
                candidates = [chunk for chunk in candidates if chunk.fiscal_year == latest]
            for chunk in candidates:
                if chunk.chunk_id not in selected_ids:
                    selected.append(chunk)
                    selected_ids.add(chunk.chunk_id)
    return selected

class FinancialHybridRetriever:
    """
    Vector-first retrieval over source-grounded chunk text, with BM25 and
    report metadata as supporting signals. Schema mappings are not candidate
    gates; they help the ingest path create and annotate accurate row chunks.
    """

    def __init__(
        self,
        chunks: List[FinancialChunk],
        enable_dense: bool = False,
        background_dense_index: bool = False,
        dense_seed: DenseRetriever | None = None,
    ):
        self.chunks = chunks
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("FinancialHybridRetriever requires unique chunk IDs")
        self.chunks_by_id = {c.chunk_id: c for c in self.chunks}
        self._chunks_by_company: dict[str, list[FinancialChunk]] = {}
        for chunk in chunks:
            self._chunks_by_company.setdefault((chunk.company_code or "").casefold(), []).append(chunk)
        self._bm25_cache_key: tuple[tuple[str, str], ...] | None = None
        self._bm25_cache: BM25Retriever | None = None
        self._bm25_scopes: OrderedDict[tuple, BM25Retriever] = OrderedDict()
        self._bm25_max_scopes = max(0, int(os.getenv("FINANCIAL_BM25_CACHE_SCOPES", "8")))
        self._bm25_chunk_budget = max(0, int(os.getenv("FINANCIAL_BM25_CACHE_CHUNKS", "20000")))
        self._bm25_cached_chunks = 0
        self.bm25_cache_hits = 0
        self.bm25_cache_misses = 0
        self._bm25_cache_lock = Lock()
        self._dense_index_cancel = Event()
        self._dense_index_thread: Thread | None = None
        self._dense_seed = dense_seed if enable_dense else None
        self.dense_retriever: DenseRetriever | QdrantDenseRetriever | None = None
        dense_backend = os.getenv("FINANCIAL_DENSE_SEARCH_BACKEND", "numpy").strip().casefold()
        if enable_dense and dense_backend == "qdrant":
            from .qdrant_dense import QdrantDenseRetriever
            self.dense_retriever = QdrantDenseRetriever(self.chunks)
        elif enable_dense and dense_backend != "numpy":
            raise ValueError("FINANCIAL_DENSE_SEARCH_BACKEND must be numpy or qdrant")
        else:
            self.dense_retriever = DenseRetriever(self.chunks) if enable_dense else None
        if self.dense_retriever:
            if background_dense_index:
                self._dense_index_thread = Thread(
                    target=self._index_dense_in_background,
                    name="financial-e5-index",
                    daemon=True,
                )
                self._dense_index_thread.start()
            else:
                self.dense_retriever.index(seed_retriever=dense_seed)
                self._dense_seed = None

    @property
    def dense_index_status(self) -> str:
        if self.dense_retriever is None:
            return "disabled"
        return self.dense_retriever.index_status

    def cancel_dense_indexing(self) -> None:
        self._dense_index_cancel.set()

    def _index_dense_in_background(self) -> None:
        dense_retriever = self.dense_retriever
        if dense_retriever is None:
            return
        with _DENSE_INDEX_LOCK:
            if self._dense_index_cancel.is_set():
                return
            logger.info("dense_index_started chunks=%s", len(self.chunks))
            try:
                dense_retriever.index(
                    cancel_event=self._dense_index_cancel,
                    seed_retriever=self._dense_seed,
                )
            except Exception:
                logger.exception("dense_index_failed chunks=%s", len(self.chunks))
                return
            finally:
                self._dense_seed = None
        logger.info(
            "dense_index_finished status=%s chunks=%s cached=%s",
            dense_retriever.index_status,
            len(self.chunks),
            dense_retriever.cached_vector_count,
        )

    def search(
        self,
        parsed_query: ParsedQuery,
        top_k: int = 5,
        directive_plan: QueryDirectivePlan | None = None,
        strict_metric_scope: bool = True,
    ) -> List[RetrievalResult]:
        """Retrieve by vector similarity first, with lexical and metadata support.

        Company/year/report constraints scope the candidate corpus. Metric
        aliases and schema IDs never remove candidates; they are ingestion
        annotations, not retrieval gates.
        """
        if not self.chunks:
            return []

        requested_years = parsed_query.fiscal_years or (
            [parsed_query.fiscal_year] if parsed_query.fiscal_year is not None else []
        )
        company_order = parsed_query.company_codes or (
            [parsed_query.company_code] if parsed_query.company_code else []
        )
        requested_companies = {
            company.casefold() for company in company_order if company
        }

        # Keep identity and requested-period constraints hard. Statement type
        # and metric labels can be wrong on newly ingested or unmapped rows, so
        # they remain searchable and are never used as hard retrieval gates.
        filtered_chunks = (
            [chunk for company in dict.fromkeys(company.casefold() for company in company_order)
             for chunk in self._chunks_by_company.get(company, [])]
            if requested_companies else self.chunks
        )
        filtered_chunks = [
            chunk for chunk in filtered_chunks
            if (
                parsed_query.fiscal_quarter is None
                or chunk.fiscal_quarter == parsed_query.fiscal_quarter
                or chunk.metadata.get("fiscal_quarter") == parsed_query.fiscal_quarter
            )
            and (
                not parsed_query.report_type
                or chunk.report_type == parsed_query.report_type
            )
        ]
        filtered_chunks = _scope_reports(
            filtered_chunks, requested_years, company_order, parsed_query.report_type,
            parsed_query.source_policy,
        )
        if not filtered_chunks:
            return []

        # Retain structural matches as a weak preference, not a filter. A
        # missing or stale section tag must not hide relevant source text.
        structural_chunks, _ = StructuralOperator.select(filtered_chunks, directive_plan)
        structural_ids = {chunk.chunk_id for chunk in structural_chunks}

        # BM25 is a supporting ranking over the same source-scoped corpus.
        bm25_retriever = self._get_bm25_retriever(filtered_chunks)
        # Use the user's words here: a canonical rewrite may encode an alias
        # collision and then make both retrieval channels search for the wrong
        # financial line.
        bm25_query = parsed_query.original_query
        raw_bm25_results = bm25_retriever.search(bm25_query, top_k=top_k * 4)

        # Apply only small source/period preferences to the sparse ranking.
        # Canonical metrics and aliases are deliberately excluded from scoring.
        boosted_scores: List[tuple[str, float]] = []
        for chunk_id, base_score in raw_bm25_results:
            chunk = self.chunks_by_id[chunk_id]
            score = base_score

            # Keep vector/BM25 as the retrieval foundation, then prefer a
            # source row whose extracted label literally matches the user's
            # words. This reduces alias collisions without consulting schema
            # IDs or removing other candidates.
            raw_label = _raw_metric_label(chunk)
            if raw_label:
                score += 20.0 * _source_label_affinity(
                    parsed_query.original_query,
                    raw_label,
                )

            if chunk_id in structural_ids:
                score += 1.0
            if parsed_query.statement_type and chunk.statement_type == parsed_query.statement_type:
                score += 1.0

            is_comparative_year = (
                len(requested_years) == 1
                and requested_years[0] == chunk.comparative_fiscal_year
            )
            if len(requested_years) <= 1 and (is_comparative_year or parsed_query.period_type.value == "beginning_of_year") and (
                "Số đầu năm" in chunk.content or "start_value" in chunk.metadata
            ):
                score += 2.0
            elif len(requested_years) <= 1 and (not is_comparative_year and parsed_query.period_type.value == "end_of_year") and (
                "Số cuối năm" in chunk.content or "end_value" in chunk.metadata
            ):
                score += 2.0

            boosted_scores.append((chunk_id, score))

        # Sắp xếp danh sách BM25 sau boost
        boosted_scores.sort(key=lambda x: x[1], reverse=True)
        bm25_ranking = [cid for cid, _ in boosted_scores]

        # Dense semantic retrieval is primary. Use the original query so a
        # weak schema/registry interpretation cannot overwrite its meaning.
        rankings: list[list[str]] = []
        ranking_weights: list[float] = []
        dense_ranking: list[str] = []
        if self.dense_retriever and self.dense_retriever.index_status == "ready":
            allowed_ids = {chunk.chunk_id for chunk in filtered_chunks}
            try:
                dense_results = self.dense_retriever.search(
                    parsed_query.original_query,
                    top_k=max(top_k * 8, 40),
                    allowed_ids=allowed_ids,
                )
            except Exception:
                logger.exception("dense_query_failed; falling back to BM25")
                dense_results = []
            dense_ranking = [cid for cid, _ in dense_results]
            if dense_ranking:
                rankings.append(dense_ranking)
                ranking_weights.append(2.0)
        elif self.dense_retriever:
            logger.debug(
                "dense_retrieval_unavailable status=%s error=%s",
                self.dense_retriever.index_status,
                self.dense_retriever.last_error,
            )
        if bm25_ranking:
            rankings.append(bm25_ranking)
            ranking_weights.append(1.0)
        if structural_ids:
            supporting_ranking = [
                chunk_id for chunk_id in bm25_ranking if chunk_id in structural_ids
            ]
            if supporting_ranking:
                rankings.append(supporting_ranking)
                ranking_weights.append(0.25)
        if not rankings:
            return []

        # 3. Weighted RRF: vector > BM25 > explicit structure hints.
        fused = reciprocal_rank_fusion(rankings, k=60, weights=ranking_weights)

        # A metric lookup must be grounded in a source row (or a table that
        # contains the requested label). The broad vector/BM25 ranking finds
        # likely report locations; this source-label pass removes nearby but
        # unrelated rows/pages before they can be cited. For structured table
        # questions, the scope helper expands a matched row only to its source
        # page so the caller can assemble the requested table.
        should_scope_metric = bool(parsed_query.primary_metric) and (
            parsed_query.structured_operation is not None
            or parsed_query.intent in {
                QueryIntent.METRIC_LOOKUP,
                QueryIntent.COMPARISON,
                QueryIntent.GROWTH_RATE,
            }
        )
        metric_scope_chunks: list[FinancialChunk] = []
        if strict_metric_scope and should_scope_metric and parsed_query.structured_operation is None:
            metric_scope_chunks = self._scope_exact_metric_pages(
                filtered_chunks,
                parsed_query,
                allow_semantic_fallback=False,
            )
            if not metric_scope_chunks:
                # A nearby financial chunk is not evidence for a metric that
                # is absent from the selected issuer/year/report scope.
                return []

            # Keep only grounded evidence. The helper returns one selected
            # source row per requested company/period for scalar queries, and
            # source-page context for structured table queries. Its selection
            # order is deterministic and follows the requested company order.
            ranked_scores = dict(fused)
            sparse_scores = dict(boosted_scores)
            return [
                RetrievalResult(
                    chunk=chunk,
                    rrf_score=ranked_scores.get(chunk.chunk_id, 0.0),
                    sparse_score=sparse_scores.get(chunk.chunk_id),
                )
                for chunk in metric_scope_chunks[:top_k]
            ]

        # A company-to-company comparison needs at least one vector-ranked
        # evidence chunk for each requested issuer near the top of the results.
        if len(requested_companies) > 1 and top_k >= len(requested_companies):
            company_anchors: list[tuple[str, float]] = []
            selected_ids: set[str] = set()
            for company in company_order:
                if not company:
                    continue
                anchor = next(
                    (
                        (chunk_id, score)
                        for chunk_id, score in fused
                        if (self.chunks_by_id[chunk_id].company_code or "").casefold()
                        == company.casefold()
                    ),
                    None,
                )
                if anchor and anchor[0] not in selected_ids:
                    company_anchors.append(anchor)
                    selected_ids.add(anchor[0])
            fused = company_anchors + [
                (chunk_id, score)
                for chunk_id, score in fused
                if chunk_id not in selected_ids
            ]

        # 6. Gói kết quả vào RetrievalResult
        results: List[RetrievalResult] = []
        for chunk_id, rrf_score in fused[:top_k]:
            chunk = self.chunks_by_id[chunk_id]
            # Điểm sparse chuẩn hóa
            sparse_score = next((score for cid, score in boosted_scores if cid == chunk_id), None)
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    rrf_score=rrf_score,
                    sparse_score=sparse_score
                )
            )

        return results

    def _get_bm25_retriever(self, chunks: List[FinancialChunk]) -> BM25Retriever:
        # Reuse exact scopes when requests alternate issuers. Both entry count
        # and total chunk count are bounded; a whole-corpus query is not cached
        # when it exceeds the budget. Keys reference immutable snapshot content.
        cache_key = tuple((chunk.chunk_id, chunk.content) for chunk in chunks)
        with self._bm25_cache_lock:
            retriever = self._bm25_scopes.get(cache_key)
            if retriever is not None:
                self.bm25_cache_hits += 1
                self._bm25_scopes.move_to_end(cache_key)
            else:
                self.bm25_cache_misses += 1
                retriever = BM25Retriever(chunks)
                if self._bm25_max_scopes and len(chunks) <= self._bm25_chunk_budget:
                    while self._bm25_scopes and (
                        len(self._bm25_scopes) >= self._bm25_max_scopes
                        or self._bm25_cached_chunks + len(chunks) > self._bm25_chunk_budget
                    ):
                        _, evicted = self._bm25_scopes.popitem(last=False)
                        self._bm25_cached_chunks -= len(evicted.chunks)
                    self._bm25_scopes[cache_key] = retriever
                    self._bm25_cached_chunks += len(chunks)
            # Retain the last cached scope for existing diagnostic callers.
            self._bm25_cache_key = cache_key if cache_key in self._bm25_scopes else None
            self._bm25_cache = retriever if cache_key in self._bm25_scopes else None
            return retriever

    @staticmethod
    def _scope_exact_metric_pages(
        chunks: List[FinancialChunk],
        parsed_query: ParsedQuery,
        *,
        allow_semantic_fallback: bool = True,
    ) -> List[FinancialChunk]:
        if not parsed_query.primary_metric:
            return chunks

        requested_company_codes = {
            str(company).casefold()
            for company in (
                parsed_query.company_codes
                or ([parsed_query.company_code] if parsed_query.company_code else [])
            )
            if company
        }

        def covers_requested_companies(candidates: List[FinancialChunk]) -> bool:
            if not requested_company_codes:
                return True
            candidate_companies = {
                (chunk.company_code or "").casefold() for chunk in candidates
            }
            return requested_company_codes.issubset(candidate_companies)

        target_metric = parsed_query.canonical_metric or parsed_query.primary_metric
        expected_labels = {
            metric.strip()
            for metric in [target_metric, parsed_query.primary_metric, *parsed_query.metric_aliases]
            if metric and metric.strip()
        }
        expected_metrics = {metric.casefold() for metric in expected_labels}

        # Prefer labels that match the requested metric or one of its explicit
        # aliases. A merely similar row (for example, "other operating income"
        # for "total operating income") must not replace an absent total.
        if not parsed_query.structured_operation:
            source_financial_items = [
                chunk
                for chunk in chunks
                if chunk.chunk_type == ChunkType.FINANCIAL_ITEM
                and _raw_metric_label(chunk)
            ]

            # The canonical label must win before alias matching. Some legacy
            # aliases are intentionally broad (for example, "các khoản nợ")
            # and can also match a child row. Searching the combined alias set
            # first can therefore hide an exact total that is present in this
            # same company/year/report scope.
            primary_labels = {
                metric.strip()
                for metric in (target_metric, parsed_query.primary_metric)
                if metric and metric.strip()
            }
            statement_primary_items = [
                chunk
                for chunk in source_financial_items
                if parsed_query.statement_type is not None
                and chunk.statement_type == parsed_query.statement_type
                and _raw_label_matches(_raw_metric_label(chunk), primary_labels)
            ]
            if statement_primary_items and covers_requested_companies(statement_primary_items):
                return FinancialHybridRetriever._select_metric_items(
                    statement_primary_items, parsed_query
                )

            primary_exact_items = [
                chunk
                for chunk in source_financial_items
                if _normalized_metric_label(_raw_metric_label(chunk))
                in {_normalized_metric_label(label) for label in primary_labels}
            ]
            if primary_exact_items and covers_requested_companies(primary_exact_items):
                return FinancialHybridRetriever._select_metric_items(
                    primary_exact_items, parsed_query
                )

            # A report can label an explicitly requested total differently
            # from the canonical metric (for example, "Cộng doanh thu hoạt
            # động" for "Tổng doanh thu hoạt động"). Check exact source-label
            # matches across the complete alias set before any phrase/substring
            # fallback: a broad alias such as "doanh thu hoạt động" also occurs
            # inside child rows and must not mask the exact total.
            expected_exact_items = [
                chunk
                for chunk in source_financial_items
                if _normalized_metric_label(_raw_metric_label(chunk))
                in {_normalized_metric_label(label) for label in expected_labels}
            ]
            if expected_exact_items and covers_requested_companies(expected_exact_items):
                return FinancialHybridRetriever._select_metric_items(
                    expected_exact_items, parsed_query
                )

            primary_phrase_items = [
                chunk
                for chunk in source_financial_items
                if _raw_label_matches(_raw_metric_label(chunk), primary_labels)
            ]
            if primary_phrase_items and covers_requested_companies(primary_phrase_items):
                return FinancialHybridRetriever._select_metric_items(
                    primary_phrase_items, parsed_query
                )

            source_label_items = [
                chunk
                for chunk in source_financial_items
                if _raw_label_matches(_raw_metric_label(chunk), expected_labels)
            ]
            if source_label_items and covers_requested_companies(source_label_items):
                return FinancialHybridRetriever._select_metric_items(
                    source_label_items, parsed_query
                )

            # Preserve a narrow lexical fallback for report-specific labels
            # that add a harmless clarifier, but do not accept rows that only
            # share a broad phrase with the requested metric.
            phrase_items = [
                chunk
                for chunk in chunks
                if chunk.chunk_type == ChunkType.FINANCIAL_ITEM
                and _raw_metric_label(chunk)
                and len(_metric_tokens(_raw_metric_label(chunk))) >= 2
                and _source_label_affinity(
                    parsed_query.original_query,
                    _raw_metric_label(chunk),
                ) >= 0.80
            ]
            if phrase_items and covers_requested_companies(phrase_items):
                return FinancialHybridRetriever._select_metric_items(
                    phrase_items, parsed_query
                )

        def matches_metric(chunk: FinancialChunk) -> bool:
            if not parsed_query.structured_operation and chunk.chunk_type == ChunkType.FINANCIAL_ITEM:
                source_label = _raw_metric_label(chunk)
                if source_label:
                    return _raw_label_matches(source_label, expected_labels)
            return (
                any(metric in chunk.content.casefold() for metric in expected_metrics)
                or any(metric.strip().casefold() in expected_metrics for metric in chunk.primary_metrics)
                or bool(chunk.canonical_metric and chunk.canonical_metric.casefold() in expected_metrics)
            )

        metric_chunks = [
            chunk
            for chunk in chunks
            if matches_metric(chunk)
        ]
        if not metric_chunks:
            # OCR and report-specific wording often inserts parenthetical
            # clarifiers into a label (for example, "tiền thuần từ (sử dụng
            # vào) hoạt động kinh doanh"). Permit a strong token overlap as
            # the lexical fallback, while rejecting unrelated chunks that
            # merely score against common question words.
            metric_token_sets = [
                _metric_tokens(metric) for metric in expected_metrics if _metric_tokens(metric)
            ]
            overlapping_chunks: list[FinancialChunk] = []
            if metric_token_sets:
                for chunk in chunks:
                    source_label = (
                        _raw_metric_label(chunk)
                        if not parsed_query.structured_operation
                        and chunk.chunk_type == ChunkType.FINANCIAL_ITEM
                        else ""
                    )
                    content_tokens = _metric_tokens(source_label or chunk.content)
                    if any(
                        len(metric_tokens & content_tokens)
                        >= FinancialHybridRetriever._minimum_metric_overlap(metric_tokens)
                        for metric_tokens in metric_token_sets
                    ):
                        overlapping_chunks.append(chunk)
            if overlapping_chunks:
                raw_labeled_items = [
                    chunk
                    for chunk in overlapping_chunks
                    if chunk.chunk_type == ChunkType.FINANCIAL_ITEM
                    and _raw_metric_label(chunk)
                ]
                if raw_labeled_items and not parsed_query.structured_operation:
                    return FinancialHybridRetriever._select_metric_items(
                        raw_labeled_items, parsed_query
                    )
                metric_chunks = overlapping_chunks
            else:
                # Signal that strict metric scoping failed. The caller retries
                # BM25 on the broader company/year/report-scoped candidates.
                if (
                    parsed_query.intent in {QueryIntent.METRIC_LOOKUP, QueryIntent.COMPARISON}
                    and not allow_semantic_fallback
                ):
                    return []
                return chunks

        exact_items = [
            chunk
            for chunk in chunks
            if chunk.chunk_type == ChunkType.FINANCIAL_ITEM
            and matches_metric(chunk)
        ]

        if exact_items and not parsed_query.structured_operation:
            selected_items = FinancialHybridRetriever._select_metric_items(
                exact_items, parsed_query
            )
            if covers_requested_companies(selected_items):
                return selected_items
            if not allow_semantic_fallback and len(requested_company_codes) > 1:
                # Never present one side of a multi-company comparison as if it
                # answered the full question. The caller can abstain or retry
                # with its broader vector/BM25 fallback path.
                return []
            return selected_items

        if not parsed_query.structured_operation:
            # A page may contain many unrelated financial lines. For a direct
            # metric lookup, do not expand a lexical/narrative hit back to every
            # financial item on that page.
            if not allow_semantic_fallback:
                # In strict mode, a narrative that merely mentions the metric
                # is not numeric evidence. A table hit is usable because it
                # retains the source row/column values for citation validation.
                has_financial_value = re.compile(
                    r"(?<!\d)(?:\d{1,3}(?:[.,]\d{3})+|\d{5,})(?!\d)"
                )
                table_hits = [
                    chunk for chunk in metric_chunks
                    if chunk.chunk_type == ChunkType.TABLE
                    or (
                        has_financial_value.search(chunk.content)
                        and _raw_label_matches(
                            _raw_metric_label(chunk) or chunk.canonical_metric or chunk.content,
                            expected_labels,
                        )
                    )
                ]
                if table_hits:
                    return table_hits
                return []
            return metric_chunks or exact_items

        source_pages = {
            (chunk.document_id or chunk.source_file, chunk.page)
            for chunk in exact_items or metric_chunks
            if chunk.page is not None
        }
        if not source_pages:
            return exact_items or metric_chunks
        scoped = [
            chunk
            for chunk in chunks
            if (chunk.document_id or chunk.source_file, chunk.page) in source_pages
        ]
        return scoped or exact_items or metric_chunks

    @staticmethod
    def _minimum_metric_overlap(metric_tokens: set[str]) -> int:
        if len(metric_tokens) == 1:
            # A short remaining token (for example, "sản" from "tài sản") is
            # too common to identify a financial row on its own.
            return 1 if len(next(iter(metric_tokens))) >= 6 else 2
        return max(2, math.ceil(len(metric_tokens) * 0.75))

    @staticmethod
    def _remove_conflicting_metric_items(
        chunks: List[FinancialChunk],
        parsed_query: ParsedQuery,
    ) -> List[FinancialChunk]:
        if not parsed_query.primary_metric or parsed_query.structured_operation:
            return chunks
        expected_labels = {
            metric.strip()
            for metric in [
                parsed_query.canonical_metric,
                parsed_query.primary_metric,
                *parsed_query.metric_aliases,
            ]
            if metric and metric.strip()
        }
        return [
            chunk
            for chunk in chunks
            if chunk.chunk_type != ChunkType.FINANCIAL_ITEM
            or not _raw_metric_label(chunk)
            or _raw_label_matches(_raw_metric_label(chunk), expected_labels)
        ]

    @staticmethod
    def _select_metric_items(
        chunks: List[FinancialChunk],
        parsed_query: ParsedQuery,
    ) -> List[FinancialChunk]:
        requested_years = parsed_query.fiscal_years or (
            [parsed_query.fiscal_year]
            if parsed_query.fiscal_year is not None
            else []
        )
        requested_companies = parsed_query.company_codes or (
            [parsed_query.company_code] if parsed_query.company_code else []
        )

        def preference(chunk: FinancialChunk) -> tuple[int, int, int, int, int]:
            # Mặc định dùng BCTC hợp nhất khi câu hỏi không yêu cầu báo cáo
            # riêng; nếu không có bản hợp nhất thì vẫn lấy tài liệu sẵn có.
            # Within the selected filing, prefer the statement named or inferred
            # from the query when an exact source row is duplicated in notes.
            # This remains a preference: if OCR metadata misses the intended
            # statement, the source row is still eligible as a fallback.
            statement_match = int(
                parsed_query.statement_type is not None
                and chunk.statement_type == parsed_query.statement_type
            )
            consolidated = int(
                parsed_query.report_type is None
                and chunk.report_type == "consolidated"
            )
            coverage = len(set(requested_years).intersection(_covered_years(chunk)))
            return (
                statement_match,
                consolidated,
                coverage,
                int(chunk.chunk_type == ChunkType.FINANCIAL_ITEM),
                int(chunk.fiscal_year or 0),
            )

        company_order = requested_companies or list(
            dict.fromkeys(chunk.company_code or "" for chunk in chunks)
        )
        selected: list[FinancialChunk] = []
        for company in company_order:
            company_items = [
                chunk for chunk in chunks
                if (chunk.company_code or "").casefold() == company.casefold()
            ]
            if not company_items:
                continue
            if not requested_years:
                selected.append(max(company_items, key=preference))
                continue

            fully_covering = [
                chunk for chunk in company_items
                if set(requested_years).issubset(_covered_years(chunk))
            ]
            if fully_covering:
                selected.append(max(fully_covering, key=preference))
                continue

            for year in requested_years:
                covering = [
                    chunk for chunk in company_items
                    if year in _covered_years(chunk)
                ]
                if covering:
                    best = max(covering, key=preference)
                    if best.chunk_id not in {item.chunk_id for item in selected}:
                        selected.append(best)
        return selected
