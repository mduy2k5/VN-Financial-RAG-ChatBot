import logging
import json
import os
from pathlib import Path
import re
import time
import sys
from functools import wraps
from dataclasses import replace
from threading import RLock
from contextlib import asynccontextmanager, contextmanager
from typing import Annotated, Any, Dict, Iterator, List, Literal, Optional
from urllib.parse import urlsplit

import requests
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, StringConstraints

from ..schemas import (
    ChunkType, FinancialChunk, ParsedQuery, RetrievalResult,
    FinancialAnswerResponse, CalculationEvidenceResponse, Citation, ValidationResult,
    QueryIntent, StatementType, CalculationInputResponse, AnalysisCoverageResponse,
)
from ..analysis.contribution import CalculationEvidence
from ..analysis.evidence import validate_calculation
from ..analysis.overview import build_financial_overview_answer
from ..analysis.series import build_series_answer, series_within_budget
from ..analysis.matrix import build_matrix_answer
from ..analysis.formula_catalog import FORMULAS
from ..analysis.ratio import matches_metric_source, requested_ratio_inputs
from ..metadata.metric_mapper import MetricDefinition, STATEMENT_NAMES_VI
from ..embedding.text_builder import TextBuilder
from ..chunking import UniversalFinancialChunker
from ..query.understanding import QueryUnderstandingEngine
from ..query.directives import QueryDirectivePlanner, normalize_directive_text
from ..retrieval.hybrid import FinancialHybridRetriever, _scope_reports
from ..retrieval.runtime import RequestSnapshot, ScopeCache, request_snapshot
from ..retrieval.operators import (
    FilterOperator,
    OllamaChunkFilter,
    ValidateOperator,
    directive_operator_state,
)
from ..generation.chat import FinancialChatEngine, FinancialGenerationError
from ..context.citation_mapper import CitationMapper
from ..validation.numeric import NumericValidator
from ..validation.citation import CitationValidator
from ..validation.claims import check_claims
from ..storage.factory import create_document_store
from ..application.ingestion import (IngestRequest, BatchIngestRequest, prepare_document, publish_documents)
from ..notebooks.api import create_router as create_notebook_router
from ..structure.source_groups import (
    analysis_target_phrase,
    is_direct_code_child,
    is_total_source_label,
    root_source_segment,
    source_root_metric_id,
    source_phrase_tokens,
    source_label_matches_query,
    source_row_code,
    source_item_label,
)
from ..query_guides import QueryPlan, build_query_plan

app = FastAPI(
    title="Financial RAG Service (Vietnamese Financial Understanding)",
    description="Dịch vụ hỏi đáp báo cáo tài chính tiếng Việt chạy độc lập",
    version="1.0.0"
)
# Uvicorn configures this logger with its console handler and INFO level, so
# operation timing is visible in the normal service container logs.
logger = logging.getLogger("uvicorn.error")

QueryText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=4_096),
]
# Global in-memory storage for prototype
CHUNKS_DB: List[FinancialChunk] = []
retriever: Optional[FinancialHybridRetriever] = None
runtime_lock = RLock()
scope_cache = ScopeCache(int(os.getenv("FINANCIAL_SCOPE_CACHE_SCOPES", "8")),
                         int(os.getenv("FINANCIAL_SCOPE_CACHE_CHUNKS", "20000")))
database_path = os.getenv("FINANCIAL_DB_PATH", "").strip()
storage_backend = os.getenv("FINANCIAL_STORAGE_BACKEND", "sqlite").strip().casefold()


def _create_document_store() -> Any:
    return create_document_store()

document_store = _create_document_store()
query_engine = QueryUnderstandingEngine()
directive_planner = QueryDirectivePlanner()
chat_engine = FinancialChatEngine(
    ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
    model_name=os.getenv("OLLAMA_MODEL", "qwen3.5:4b"),
)
filter_model_name = os.getenv("FINANCIAL_FILTER_MODEL", "").strip()
filter_classifier = (
    OllamaChunkFilter(
        ollama_url=chat_engine.ollama_url,
        model_name=filter_model_name,
        timeout_seconds=chat_engine.timeout_seconds,
    )
    if filter_model_name
    else None
)
chunker = UniversalFinancialChunker()


def build_retriever(chunks: List[FinancialChunk], *, isolated: bool = False) -> FinancialHybridRetriever:
    chunks = [chunk.model_copy(deep=True) for chunk in chunks]
    for chunk in chunks:
        if chunk.chunk_type == ChunkType.FINANCIAL_ITEM:
            # Older persisted chunks include mapper-generated canonical aliases
            # in their body. Strip those synthetic lines before indexing and
            # citation assembly while preserving the extracted source row.
            chunk.content = TextBuilder.build_source_content(chunk)
    chunker.schema_catalog.annotate_chunks(chunks)
    enable_dense = os.getenv("FINANCIAL_ENABLE_DENSE", "false").casefold() in {"1", "true", "yes"}
    previous_retriever = None if isolated else retriever
    dense_seed = (
        previous_retriever.dense_retriever
        if previous_retriever
        and previous_retriever.dense_retriever
        and previous_retriever.dense_retriever.index_status == "ready"
        else None
    )
    if retriever is not None and not isolated:
        retriever.cancel_dense_indexing()
    return FinancialHybridRetriever(
        chunks,
        enable_dense=enable_dense,
        background_dense_index=True,
        dense_seed=dense_seed,
    )


def _scoped_runtime() -> bool:
    return bool(document_store is not None and hasattr(document_store, "read_scope")
                and os.getenv("FINANCIAL_SCOPED_RUNTIME", "true").casefold() in {"1", "true", "yes"})


def _runtime_chunks():
    snapshot = request_snapshot.get()
    return snapshot.chunks if snapshot else CHUNKS_DB


def _runtime_retriever():
    snapshot = request_snapshot.get()
    return snapshot.retriever if snapshot else retriever


def _catalog_chunks():
    snapshot = request_snapshot.get()
    return [FinancialChunk(chunk_id=item["logical_key"], source_file=item["source_file"],
                           company_code=item.get("company_code"), fiscal_year=item.get("fiscal_year"),
                           report_type=item.get("report_type"), content="")
            for item in (snapshot.documents if snapshot else document_store.list_documents())]


def _with_request_snapshot(function):
    @wraps(function)
    def wrapped(req):
        existing = request_snapshot.get()
        if existing is not None:
            snapshot = existing
        elif _scoped_runtime():
            parsed = query_engine.resolve_company_context(query_engine.parse(req.query), _catalog_chunks())
            companies = parsed.company_codes or ([parsed.company_code] if parsed.company_code else [])
            years = parsed.fiscal_years or ([parsed.fiscal_year] if parsed.fiscal_year else [])
            if len(companies) > 5 or len(years) > 10:
                if function.__name__ == "answer_question":
                    return FinancialAnswerResponse(query=req.query, parsed_query=parsed, citations=[],
                        validation=ValidationResult(is_valid=True, unit_match=True, citation_valid=True),
                        answer="Phạm vi quá lớn: hãy chọn tối đa 5 doanh nghiệp, 10 kỳ và 100 ô chỉ tiêu.",
                        answer_status="needs_clarification")
                raise HTTPException(status_code=422, detail="Chọn tối đa 5 doanh nghiệp và 10 kỳ.")
            try:
                scope = document_store.read_scope(
                    companies, years, parsed.report_type, source_policy=req.source_policy,
                    cache_lookup=lambda docs: scope_cache.get(ScopeCache.key(docs)),
                    max_chunks=int(os.getenv("FINANCIAL_REQUEST_MAX_CHUNKS", "30000")),
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            snapshot = scope["cached"]
            if snapshot is None:
                key = ScopeCache.key(scope["documents"])
                def build_scope():
                    index = build_retriever(scope["chunks"], isolated=True) if scope["chunks"] else None
                    return RequestSnapshot(tuple(index.chunks) if index else (), index,
                                           tuple(scope["documents"]), key)
                snapshot = scope_cache.get_or_build(key, build_scope)
        else:
            with runtime_lock:
                snapshot = RequestSnapshot(tuple(CHUNKS_DB), retriever)
        snapshot = replace(snapshot, source_policy=req.source_policy)
        token = request_snapshot.set(snapshot)
        try:
            response = function(req)
            if isinstance(response, FinancialAnswerResponse):
                used_ids = set(re.findall(r"\[(c\d+)\]", response.answer, flags=re.I))
                used_ids.update(key for calc in response.calculations for key in calc.input_citation_ids)
                response.citations = [item for item in response.citations if item.citation_id in used_ids]
                deterministic = bool(response.calculations or response.parsed_query.structured_operation
                                     or response.parsed_query.intent not in {QueryIntent.EXPLANATION, QueryIntent.GENERAL}
                                     or "Phân tích các chỉ tiêu truy xuất" in response.answer)
                response.claim_checks = check_claims(response.answer, response.citations,
                    snapshot.retriever.chunks_by_id if snapshot.retriever else {}, deterministic=deterministic)
                if (not deterministic and any(item["status"] == "unverified" for item in response.claim_checks)
                        and os.getenv("FINANCIAL_EXTRACTIVE_NARRATIVE", "true").casefold() in {"1", "true", "yes"}):
                    response.answer = "Chưa đủ bằng chứng để xác minh diễn giải; hãy đối chiếu trực tiếp các nguồn đã truy xuất."
                    response.answer_status = "insufficient_evidence"
                    response.validation.is_valid = False
                    response.validation.details += " Diễn giải không khớp trích đoạn nguồn; kiểm định ngữ nghĩa theo chế độ trích xuất."
                    response.calculations = []
                response.source_snapshot = {"id": snapshot.snapshot_id, "policy": req.source_policy,
                    "documents": [{key: item.get(key) for key in
                                   ("logical_key", "active_version_id", "active_projection_id", "source_file")}
                                  for item in snapshot.documents]}
            return response
        finally:
            request_snapshot.reset(token)
    return wrapped


def _with_runtime_write_lock(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with runtime_lock:
            return function(*args, **kwargs)
    return wrapped


def _publish_scoped_documents(documents, *, job=None):
    count = publish_documents(document_store, documents,
                              lambda chunks: build_retriever(chunks, isolated=True), job=job)
    scope_cache.clear()
    refresh_company_aliases()
    return count

def refresh_company_aliases() -> None:
    if document_store is None:
        return
    aliases = document_store.load_company_aliases()
    query_engine.set_dynamic_company_aliases(aliases)
    for extractor in (
        chunker.md_chunker.company_extractor,
        chunker.txt_chunker.company_extractor,
    ):
        extractor.set_dynamic_aliases(aliases)


def restore_persisted_documents() -> None:
    global CHUNKS_DB, retriever
    if document_store is None:
        return
    refresh_company_aliases()
    if _scoped_runtime():
        CHUNKS_DB = []
        retriever = None
        return
    stored_chunks = document_store.load_chunks()
    if stored_chunks:
        updated_retriever = build_retriever(stored_chunks)
        CHUNKS_DB = stored_chunks
        retriever = updated_retriever


@asynccontextmanager
async def lifespan(_app: FastAPI):
    restore_persisted_documents()
    yield


app.router.lifespan_context = lifespan

# Chỉ nạp báo cáo minh họa tổng hợp khi bật rõ ràng cho demo cục bộ.
sample_md = Path(__file__).resolve().parent.parent.parent.parent / "sample_data" / "DEMO_SYNTHETIC_2025.md"
load_sample_data = os.getenv("FINANCIAL_LOAD_SAMPLE_DATA", "false").casefold() in {"1", "true", "yes"}
if load_sample_data and sample_md.exists():
    try:
        CHUNKS_DB = chunker.chunk_file(sample_md, company_code="DEMO")
        retriever = build_retriever(CHUNKS_DB)
        print(f"[API STARTUP] Đã nạp tự động {len(CHUNKS_DB)} chunks từ {sample_md.name}")
    except Exception as e:
        print(f"[API STARTUP WARN] Không thể nạp sample data: {e}")

class ConversationHistoryTurn(BaseModel):
    id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
    query: Annotated[str, StringConstraints(min_length=1, max_length=4_096)]
    summary: Annotated[str, StringConstraints(max_length=2_000)] = "Chưa có câu trả lời."
    bookmarked: bool = False
    response: Optional[Dict[str, Any]] = None
    error: Annotated[str, StringConstraints(max_length=4_000)] = ""


class ConversationHistoryPayload(BaseModel):
    id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
    title: Annotated[str, StringConstraints(min_length=1, max_length=120)]
    createdAt: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    updatedAt: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    turns: List[ConversationHistoryTurn] = Field(min_length=1, max_length=40)


@contextmanager
def _log_document_operation(operation: str, request: IngestRequest) -> Iterator[None]:
    started_at = time.perf_counter()
    identity_type = "document_id" if request.document_id else "source_file"
    identity = request.document_id or request.source_file
    try:
        yield
    except Exception:
        duration_ms = (time.perf_counter() - started_at) * 1_000
        logger.exception(
            "financial_document_%s_failed identity_type=%s identity=%r duration_ms=%.2f",
            operation,
            identity_type,
            identity,
            duration_ms,
        )
        raise
    else:
        duration_ms = (time.perf_counter() - started_at) * 1_000
        logger.info(
            "financial_document_%s_completed identity_type=%s identity=%r duration_ms=%.2f",
            operation,
            identity_type,
            identity,
            duration_ms,
        )

class QueryRequest(BaseModel):
    query: QueryText
    top_k: int = Field(default=5, ge=1, le=100)
    source_policy: Literal["source_report", "latest_restated"] = "source_report"


def _matrix_answer(req, parsed):
    companies = parsed.company_codes or [parsed.company_code]
    years = parsed.fiscal_years or [parsed.fiscal_year]
    plans, grouped, definitions, results = {}, {}, {}, {}
    for company in companies:
        issuer = parsed.model_copy(update={"company_codes": [company], "company_code": company})
        plan = build_query_plan(req.query, access_filter={"corpus": "local_financial_index"},
                                parsed_query=issuer, engine=query_engine, source_chunks=_runtime_chunks())
        inputs = [spec.metric_id for spec in requested_ratio_inputs(req.query)]
        if any(term in req.query.casefold() for term in ("roa", "roe")):
            inputs.append("loi_nhuan_sau_thue")
            if "roa" in req.query.casefold():
                inputs.append("tong_cong_tai_san")
            if "roe" in req.query.casefold():
                inputs.append("tong_von_chu_so_huu")
        if inputs:
            plan.planned_metric_ids = list(dict.fromkeys(inputs))
            plan.task = "analysis"
        elif parsed.structured_operation == "ranking" and parsed.metric_id:
            plan.planned_metric_ids = [parsed.metric_id]
        elif not plan.planned_metric_ids and parsed.metric_id:
            plan.planned_metric_ids = [parsed.metric_id]
        if not series_within_budget(companies, years, plan.planned_metric_ids):
            return FinancialAnswerResponse(query=req.query, parsed_query=parsed, citations=[],
                validation=ValidationResult(is_valid=True, unit_match=True, citation_valid=True),
                answer="Hãy chọn chỉ tiêu cụ thể, tối đa 5 doanh nghiệp, 10 kỳ và 100 ô chỉ tiêu.",
                answer_status="needs_clarification")
        plans[company] = plan
        rows, selected, found = _retrieve_series_metrics(plan, issuer, req.top_k)
        definitions.update(found)
        for metric, values in rows.items():
            grouped.setdefault(metric, {}).update({row.chunk.chunk_id: row for row in values})
        results.update({row.chunk.chunk_id: row for row in selected})
    citations, _ = CitationMapper.map_citations(list(results.values()))
    answer = build_matrix_answer(parsed, plans, definitions,
                                 {metric: list(rows.values()) for metric, rows in grouped.items()},
                                 citations, _runtime_retriever().chunks_by_id)
    validation_text = (re.sub(r"(?m)^\| \d+ \|", "| — |", answer.answer)
                       if parsed.structured_operation == "ranking" else answer.answer)
    validation = _validate_answer(validation_text, answer.citations, answer.source_contexts,
                                  answer.calculations, _runtime_retriever().chunks_by_id)
    if answer.answer_status == "needs_clarification" and not answer.citations and not answer.calculations:
        validation = ValidationResult(is_valid=True, unit_match=True, citation_valid=True)
    # Share the standard validation and lineage serialization with the API.
    calculations = [CalculationEvidenceResponse(
        **{key: getattr(item, key) for key in ("value", "unit", "formula_id", "formula_version", "input_chunk_ids", "input_citation_ids")},
        expression=FORMULAS[item.formula_id].expression,
        inputs=[CalculationInputResponse(**vars(fact)) for fact in item.inputs]) for item in answer.calculations]
    return FinancialAnswerResponse(query=req.query, parsed_query=parsed,
        answer=answer.answer if validation.is_valid else "Không thể xác minh câu trả lời từ các nguồn đã truy xuất.",
        citations=answer.citations, validation=validation, calculations=calculations if validation.is_valid else [],
        answer_status=answer.answer_status if validation.is_valid else "insufficient_evidence",
        analysis_coverage=AnalysisCoverageResponse(missing_components=list(answer.missing_components)))


class LLMConnectionTestRequest(BaseModel):
    provider: Literal["ollama", "openai_compatible"]
    base_url: Annotated[str, StringConstraints(strip_whitespace=True, min_length=8, max_length=500)]
    model: Annotated[str, StringConstraints(strip_whitespace=True, max_length=160)] = ""
    api_key: Optional[Annotated[str, StringConstraints(max_length=2_000)]] = None


def _validate_answer(
    answer_text: str,
    citations: List[Citation],
    source_contexts: List[str],
    calculation_evidence: List[CalculationEvidence] | None = None,
    chunks_by_id: dict[str, FinancialChunk] | None = None,
) -> ValidationResult:
    allowed_formula_ids = set(FORMULAS)
    citation_by_chunk = {
        citation.chunk_id: citation
        for citation in citations
        if citation.chunk_id
    }
    citation_by_id = {
        citation.citation_id: citation
        for citation in citations
    }
    trusted_calculations: list[CalculationEvidence] = []
    invalid_calculation = False
    if calculation_evidence and chunks_by_id is None:
        chunks_by_id = {chunk.chunk_id: chunk for chunk in _runtime_chunks()}
    for item in calculation_evidence or []:
        citation_ids = set(item.input_citation_ids)
        chunk_ids = set(item.input_chunk_ids)
        citation_lineage_matches = all(
            citation_id in citation_by_id
            and citation_by_id[citation_id].chunk_id in chunk_ids
            for citation_id in citation_ids
        )
        if (
            item.formula_id not in allowed_formula_ids
            or item.formula_version != FORMULAS[item.formula_id].version
            or not NumericValidator.normalize_number(item.value)
            or not NumericValidator._normalize_unit(item.unit)
            or not citation_ids
            or not chunk_ids
            or not citation_lineage_matches
            or not citation_ids.issubset({citation.citation_id for citation in citations})
            or not chunk_ids.issubset(citation_by_chunk)
            or not validate_calculation(item, citation_by_id, chunks_by_id or {})
        ):
            invalid_calculation = True
            continue
        trusted_calculations.append(item)
    validation = NumericValidator.validate(
        answer_text,
        source_contexts,
        derived_values=[(item.value, item.unit) for item in trusted_calculations],
    )
    if invalid_calculation:
        validation.is_valid = False
        validation.details += " Kết quả tính không khớp công thức, vai trò/kỳ/giá trị nguồn hoặc citation đầu vào."
    validation.citation_valid = CitationValidator.validate_ids(
        answer_text,
        [citation.citation_id for citation in citations],
        require_citation=not _answer_is_abstention(answer_text),
    )
    validation.is_valid = (
        validation.is_valid and validation.citation_valid and validation.unit_match
    )
    return validation


def _answer_is_abstention(answer_text: str) -> bool:
    normalized = normalize_directive_text(answer_text)
    return any(
        phrase in normalized
        for phrase in (
            "khong tim thay",
            "chua tim thay",
            "khong co trong tai lieu",
            "khong du thong tin",
            "khong du ngu canh",
            "khong du bang chung",
            "khong the xac dinh",
            "khong the tra loi",
        )
    )


def _normalized_table_title(chunk: FinancialChunk) -> str:
    title = str(chunk.table_title or chunk.metadata.get("table_title") or "")
    title = re.sub(r"\s*[（(]?\s*(?:tiếp theo|continued)\s*[）)]?\s*$", "", title, flags=re.IGNORECASE)
    return " ".join(title.casefold().split())


def _source_group_matches_query(
    query: str,
    source_group: str,
    company_codes: List[str],
) -> bool:
    query_signature = source_phrase_tokens(query, company_codes)
    group_signature = source_phrase_tokens(source_group, company_codes)
    return bool(query_signature and query_signature == group_signature)


def _source_label(chunk: FinancialChunk) -> str:
    return source_item_label(chunk)


def _source_label_variants(chunk: FinancialChunk) -> list[str]:
    """Return extracted and normalized aliases usable for source-row matching."""
    metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
    values: list[object] = [
        _source_label(chunk),
        metadata.get("canonical_metric"),
        metadata.get("schema_original_label"),
    ]
    aliases = metadata.get("metric_aliases")
    if isinstance(aliases, (list, tuple, set)):
        values.extend(aliases)
    elif isinstance(aliases, str):
        values.append(aliases)

    return list(dict.fromkeys(
        str(value).strip().strip("*_ `")
        for value in values
        if value is not None and str(value).strip()
    ))


def _balance_sheet_root_metric_id(chunk: FinancialChunk) -> str:
    return source_root_metric_id(chunk)


def _composition_source_rows(
    query: str,
    company_codes: List[str],
    selected_keys: set[tuple[str, str, str, str, str, str]],
    chunks: List[FinancialChunk],
    key_for: Any,
) -> List[FinancialChunk]:
    """Resolve a vector-selected source group across the same report statement."""
    query_tokens = set(source_phrase_tokens(query, company_codes))
    if not query_tokens:
        return []

    selected_scopes = {key[:5] for key in selected_keys}
    candidates = [
        chunk for chunk in chunks
        if key_for(chunk)[:5] in selected_scopes
        and chunk.chunk_type == ChunkType.FINANCIAL_ITEM
    ]
    selected_table_chunks = [
        chunk for chunk in candidates if key_for(chunk) in selected_keys
    ]
    label_token_variants = {
        chunk.chunk_id: [
            set(source_phrase_tokens(label, company_codes))
            for label in _source_label_variants(chunk)
            if source_phrase_tokens(label, company_codes)
        ]
        for chunk in candidates
    }
    label_tokens = {
        chunk.chunk_id: set(source_phrase_tokens(_source_label(chunk), company_codes))
        for chunk in candidates
    }
    # Original labels win over advisory aliases; otherwise a combined source
    # total can be admitted under the same broad canonical liability label.
    exact_anchors = [
        chunk for chunk in selected_table_chunks
        if label_tokens[chunk.chunk_id] == query_tokens
    ]
    exact_anchors = exact_anchors or [
        chunk for chunk in selected_table_chunks
        if any(
            tokens == query_tokens
            for tokens in label_token_variants[chunk.chunk_id]
        )
    ]
    if not exact_anchors:
        containing_anchors = [
            chunk for chunk in selected_table_chunks
            if any(
                query_tokens.issubset(tokens)
                for tokens in label_token_variants[chunk.chunk_id]
            )
        ]
        if containing_anchors:
            fewest_extra_tokens = min(
                min(
                    len(tokens - query_tokens)
                    for tokens in label_token_variants[chunk.chunk_id]
                    if query_tokens.issubset(tokens)
                )
                for chunk in containing_anchors
            )
            exact_anchors = [
                chunk for chunk in containing_anchors
                if min(
                    len(tokens - query_tokens)
                    for tokens in label_token_variants[chunk.chunk_id]
                    if query_tokens.issubset(tokens)
                ) == fewest_extra_tokens
            ]

    selected: dict[str, FinancialChunk] = {}
    if exact_anchors:
        for anchor in exact_anchors:
            anchor_label_tokens = label_tokens[anchor.chunk_id]
            source_scope = key_for(anchor)[:5]
            same_scope = [
                chunk for chunk in candidates
                if key_for(chunk)[:5] == source_scope
                and (not anchor.metadata.get("table_id")
                     or chunk.metadata.get("table_id") == anchor.metadata["table_id"])
            ]
            anchor_table_title = _normalized_table_title(anchor)
            same_table_family = [
                chunk for chunk in same_scope
                if _normalized_table_title(chunk) == anchor_table_title
            ]
            same_page = [chunk for chunk in same_scope if chunk.page == anchor.page]
            balance_sheet_root_id = _balance_sheet_root_metric_id(anchor)

            if balance_sheet_root_id and is_total_source_label(_source_label(anchor)):
                # A terminal statement total closes its source section. Bounds
                # use raw roots and row order, including untagged components;
                # schema edges cannot extend it into the following section.
                segment = root_source_segment(anchor, same_scope)
                if segment:
                    selected.update({row.chunk_id: row for row in segment})
                    continue

            # Schema annotations are chunking hints only: the query and vector
            # ranking still choose the anchor, and every returned amount comes
            # from an existing source row in the selected report statement.
            anchor_node_id = str(anchor.metadata.get("schema_node_id") or "")
            schema_child_ids = {
                str(node_id)
                for node_id in anchor.metadata.get("schema_children", [])
                if node_id
            }
            component_variants = anchor.metadata.get("schema_component_variants")
            if isinstance(component_variants, dict) and component_variants:
                present_node_ids = {
                    str(chunk.metadata.get("schema_node_id") or "")
                    for chunk in same_table_family
                }
                available_variants: list[tuple[int, list[str]]] = []
                for variant in component_variants.values():
                    if not isinstance(variant, list):
                        continue
                    variant_ids = [str(node_id) for node_id in variant if node_id]
                    match_count = len(present_node_ids.intersection(variant_ids))
                    if match_count:
                        available_variants.append((match_count, variant_ids))
                if available_variants:
                    best_variant_score = max(score for score, _ in available_variants)
                    schema_child_ids = set(next(
                        ids for score, ids in available_variants
                        if score == best_variant_score
                    ))

            schema_children: list[FinancialChunk] = []
            direct_schema_children = [
                chunk for chunk in same_table_family
                if anchor_node_id
                and str(chunk.metadata.get("schema_parent_id") or "") == anchor_node_id
                and chunk.chunk_id != anchor.chunk_id
            ]
            schema_children.extend(direct_schema_children)
            pending_schema_ids = list(schema_child_ids)
            pending_schema_ids.extend(
                str(chunk.metadata.get("schema_node_id") or "")
                for chunk in direct_schema_children
                if chunk.metadata.get("schema_node_id")
            )
            visited_schema_ids: set[str] = set()
            while pending_schema_ids:
                node_id = pending_schema_ids.pop()
                if node_id in visited_schema_ids:
                    continue
                visited_schema_ids.add(node_id)
                matching_nodes = [
                    chunk for chunk in same_table_family
                    if str(chunk.metadata.get("schema_node_id") or "") == node_id
                    or str(chunk.metadata.get("schema_parent_id") or "") == node_id
                ]
                for chunk in matching_nodes:
                    schema_children.append(chunk)
                    pending_schema_ids.extend(
                        str(child_id)
                        for child_id in chunk.metadata.get("schema_children", [])
                        if child_id and str(child_id) not in visited_schema_ids
                    )

            anchor_code = source_row_code(anchor.metadata.get("code"))
            direct_children = [
                chunk for chunk in same_table_family
                if is_direct_code_child(anchor_code, chunk.metadata.get("code"))
            ] if anchor_code else []

            group_signature = source_phrase_tokens(
                str(anchor.metadata.get("section_group") or ""), company_codes
            )
            section_siblings = [
                chunk for chunk in same_table_family
                if group_signature and group_signature == anchor_label_tokens
                and source_phrase_tokens(
                    str(chunk.metadata.get("section_group") or ""), company_codes
                ) == group_signature
            ]
            group_children = []
            for chunk in same_table_family:
                if chunk.chunk_id == anchor.chunk_id:
                    continue
                child_group_signature = source_phrase_tokens(
                    str(chunk.metadata.get("section_group") or ""), company_codes
                )
                child_tokens = label_tokens[chunk.chunk_id]
                minimum_overlap = min(2, len(anchor_label_tokens), len(child_tokens))
                if (
                    child_group_signature == anchor_label_tokens
                    and minimum_overlap > 0
                    and len(anchor_label_tokens & child_tokens) >= minimum_overlap
                ):
                    group_children.append(chunk)

            related_rows = {
                chunk.chunk_id: chunk
                for chunk in [*schema_children, *direct_children, *section_siblings, *group_children]
                if chunk.chunk_id != anchor.chunk_id
            }
            schema_kind = str(anchor.metadata.get("schema_node_kind") or "")

            # OCR occasionally leaves a valid financial row without a schema
            # node (for example, a bank's "Tiền gửi của khách hàng" row). For
            # a vector-selected statement total, use the first source row that
            # the schema recognizes as one of its children as a boundary, then
            # restore every original financial row up to the selected total.
            # The amount and label always remain the extracted source values;
            # schema IDs only help locate the start of the source section.
            if (schema_kind in {"total", "root"} and schema_child_ids) or balance_sheet_root_id:
                ordered_page_items = [
                    chunk for chunk in same_page
                    if chunk.chunk_type == ChunkType.FINANCIAL_ITEM
                ]
                anchor_position = next(
                    (
                        index for index, chunk in enumerate(ordered_page_items)
                        if chunk.chunk_id == anchor.chunk_id
                    ),
                    None,
                )
                child_positions = [
                    index for index, chunk in enumerate(ordered_page_items)
                    if (
                        str(chunk.metadata.get("schema_node_id") or "") in schema_child_ids
                        or str(chunk.metadata.get("schema_parent_id") or "") == anchor_node_id
                    )
                ]
                if anchor_position is not None and child_positions:
                    first_child_position = min(child_positions)
                    if first_child_position <= anchor_position:
                        for chunk in ordered_page_items[first_child_position:anchor_position + 1]:
                            if chunk.chunk_id != anchor.chunk_id:
                                related_rows[chunk.chunk_id] = chunk

                present_declared_children = {
                    child_id
                    for child_id in schema_child_ids
                    if any(
                        str(chunk.metadata.get("schema_node_id") or "") == child_id
                        for chunk in same_table_family
                    )
                }
                schema_children_incomplete = (
                    bool(schema_child_ids)
                    and len(present_declared_children) < len(schema_child_ids)
                )

                # A schema may declare children that this filing's OCR rows do
                # not carry, or only some of the declared child rows may exist.
                # In that case the vector-selected total and the source page
                # are stronger boundaries than an incomplete schema edge.
                if (
                    schema_children_incomplete
                    or (schema_child_ids and not child_positions)
                    or (balance_sheet_root_id and not related_rows)
                ):
                    ordered_page_items = [
                        chunk for chunk in same_page
                        if chunk.chunk_type == ChunkType.FINANCIAL_ITEM
                    ]
                    anchor_position = next(
                        (
                            index for index, chunk in enumerate(ordered_page_items)
                            if chunk.chunk_id == anchor.chunk_id
                        ),
                        None,
                    )
                    page_segment = ordered_page_items
                    if (
                        anchor_position is not None
                        and balance_sheet_root_id
                    ):
                        previous_root_positions = [
                            index
                            for index, chunk in enumerate(ordered_page_items[:anchor_position])
                            if _balance_sheet_root_metric_id(chunk)
                        ]
                        segment_start = (
                            max(previous_root_positions) + 1
                            if previous_root_positions
                            else 0
                        )
                        page_segment = ordered_page_items[segment_start:anchor_position + 1]
                    related_rows.update({
                        chunk.chunk_id: chunk
                        for chunk in page_segment
                        if chunk.chunk_id != anchor.chunk_id
                    })

            if not related_rows and not schema_child_ids and (
                group_signature == anchor_label_tokens
                or (anchor_code and schema_kind in {"group", "section", "root", "total"})
            ):
                # Older Markdown chunks may have neither source codes nor
                # persisted hierarchy. The original source page is the narrowest
                # available boundary for recovering its adjacent table rows.
                related_rows = {
                    chunk.chunk_id: chunk
                    for chunk in same_page
                    if chunk.chunk_id != anchor.chunk_id
                }
            selected[anchor.chunk_id] = anchor
            for chunk in related_rows.values():
                selected[chunk.chunk_id] = chunk
        return list(selected.values())

    exact_groups = [
        chunk for chunk in candidates
        if source_phrase_tokens(
            str(chunk.metadata.get("section_group") or ""), company_codes
        ) == query_tokens
    ]
    if exact_groups:
        return exact_groups
    return []


def _raw_source_label_in_query(query: str, chunk: FinancialChunk) -> bool:
    label = source_item_label(chunk)
    label = re.sub(r"\s*\([^)]*\)\s*$", "", label)
    return source_label_matches_query(
        query,
        label,
        [chunk.company_code] if chunk.company_code else [],
    )


def _select_preferred_exact_source_rows(
    query: str,
    parsed_query: ParsedQuery,
    results: List[RetrievalResult],
    candidate_chunks: List[FinancialChunk] | None = None,
) -> List[RetrievalResult]:
    """Prefer the primary statement row over duplicate mentions in note tables."""
    exact_rows = [
        result for result in results
        if result.chunk.chunk_type == ChunkType.FINANCIAL_ITEM
        and _raw_source_label_in_query(query, result.chunk)
        and result.chunk.metadata.get("report_eligible") is not False
    ]

    requested_companies = parsed_query.company_codes or (
        [parsed_query.company_code] if parsed_query.company_code else []
    )
    requested_years = list(parsed_query.fiscal_years)
    if not requested_years and parsed_query.fiscal_year is not None:
        requested_years = [parsed_query.fiscal_year]

    # Sparse retrieval can omit a requested row while returning its broader
    # parent. Recover an exact source-label match from the already request-
    # scoped corpus, constrained to issuer, period, basis and statement.
    existing_ids = {result.chunk.chunk_id for result in exact_rows}
    requested_company_set = {
        str(company).casefold() for company in requested_companies
    }
    for chunk in (candidate_chunks if requested_company_set and candidate_chunks else []):
        if (
            chunk.chunk_id in existing_ids
            or chunk.chunk_type != ChunkType.FINANCIAL_ITEM
            or chunk.metadata.get("report_eligible") is False
            or not _raw_source_label_in_query(query, chunk)
        ):
            continue
        if requested_company_set and (chunk.company_code or "").casefold() not in requested_company_set:
            continue
        if parsed_query.statement_type and chunk.statement_type != parsed_query.statement_type:
            continue
        if parsed_query.report_type and chunk.report_type != parsed_query.report_type:
            continue
        if requested_years:
            period_values = chunk.metadata.get("period_values") or {}
            years_on_chunk = {chunk.fiscal_year, chunk.comparative_fiscal_year}
            if isinstance(period_values, dict):
                years_on_chunk.update(
                    int(year) for year in period_values if str(year).isdigit()
                )
            if not set(requested_years).intersection(years_on_chunk):
                continue
        exact_rows.append(RetrievalResult(chunk=chunk, rrf_score=0.0))
        existing_ids.add(chunk.chunk_id)

    if not exact_rows:
        return []

    # If the query parser resolved an exact schema line, trust that line over
    # a more highly ranked ancestor whose label is also contained in the
    # question (for example, "Chứng khoán đầu tư" versus its AFS component).
    target_node_id = parsed_query.schema_node_id
    def source_label_specificity(result: RetrievalResult) -> int:
        label = source_item_label(result.chunk)
        label = re.sub(r"\s*\([^)]*\)\s*$", "", label)
        label = re.sub(r"^\s*(?:\d+(?:\.\d+)*|[A-Z]|[IVXLCDM]+)[.)-]\s*", "", label)
        return len(re.sub(r"[^\w]+", " ", label, flags=re.UNICODE).strip())

    target_node = chunker.schema_catalog.nodes.get(str(target_node_id or ""))

    def is_target_lineage_row(result: RetrievalResult) -> bool:
        node = chunker.schema_catalog.nodes.get(
            str(result.chunk.metadata.get("schema_node_id") or "")
        )
        while target_node is not None and node is not None:
            if node.node_id == target_node.node_id:
                return True
            node = chunker.schema_catalog.nodes.get(node.parent_id or "")
        return False

    schema_exact_rows = [result for result in exact_rows if is_target_lineage_row(result)]
    if schema_exact_rows:
        longest_schema_label = max(
            source_label_specificity(result) for result in schema_exact_rows
        )
        schema_exact_rows = [
            result for result in schema_exact_rows
            if source_label_specificity(result) == longest_schema_label
        ]
        exact_rows = schema_exact_rows
    else:
        # Older chunks may not have a schema node ID. In that case, retain the
        # longest source label explicitly present in the question and discard
        # its shorter parent label. Keep equal labels from different rows so
        # existing period/document selection can resolve them below.
        longest_label = max(source_label_specificity(result) for result in exact_rows)
        if longest_label:
            exact_rows = [
                result for result in exact_rows
                if source_label_specificity(result) == longest_label
            ]

    if not requested_companies:
        requested_companies = list(dict.fromkeys(
            result.chunk.company_code for result in exact_rows
            if result.chunk.company_code
        ))

    if not requested_years:
        requested_years = sorted({
            result.chunk.fiscal_year for result in exact_rows
            if result.chunk.fiscal_year is not None
        }, reverse=True)[:1]

    def has_period_value(result: RetrievalResult, year: int) -> bool:
        chunk = result.chunk
        values = chunk.metadata.get("period_values") or {}
        if isinstance(values, dict) and values.get(str(year)) not in (None, "", "N/A"):
            return True
        if chunk.fiscal_year == year:
            return chunk.metadata.get("end_value") not in (None, "", "N/A")
        if chunk.comparative_fiscal_year == year:
            return chunk.metadata.get("start_value") not in (None, "", "N/A")
        return False

    selected: list[RetrievalResult] = []
    for company in requested_companies:
        company_rows = [
            result for result in exact_rows
            if (result.chunk.company_code or "").casefold() == str(company).casefold()
        ]
        if parsed_query.statement_type:
            statement_rows = [
                result for result in company_rows
                if result.chunk.statement_type == parsed_query.statement_type
            ]
            if statement_rows:
                company_rows = statement_rows
        if not company_rows:
            continue

        for year in requested_years:
            year_rows = [result for result in company_rows if has_period_value(result, year)]
            if not year_rows:
                continue

            if parsed_query.report_type:
                report_rows = [
                    result for result in year_rows
                    if result.chunk.report_type == parsed_query.report_type
                ]
                if report_rows:
                    year_rows = report_rows
            else:
                consolidated_rows = [
                    result for result in year_rows
                    if (result.chunk.report_type or "").casefold() == "consolidated"
                ]
                if consolidated_rows:
                    year_rows = consolidated_rows

            current_year_rows = [
                result for result in year_rows
                if result.chunk.fiscal_year == year
            ]
            if current_year_rows:
                year_rows = current_year_rows

            rows_by_document: dict[str, list[RetrievalResult]] = {}
            for result in year_rows:
                document_id = result.chunk.document_id or result.chunk.source_file
                rows_by_document.setdefault(document_id, []).append(result)
            best_document_rows = max(
                rows_by_document.values(),
                key=lambda rows: (
                    max((row.rrf_score for row in rows), default=0.0),
                    -min((row.chunk.page or 10**9 for row in rows), default=10**9),
                ),
            )

            # In financial reports the balance sheet precedes the notes. Repeated
            # labels later in the same file often describe maturity or sensitivity
            # tables, so resolve ties to the first cited source row with a unit.
            unit_rows = [
                result for result in best_document_rows
                if result.chunk.unit or result.chunk.metadata.get("unit")
            ]
            if unit_rows:
                best_document_rows = unit_rows
            earliest_page = min(
                (result.chunk.page or 10**9 for result in best_document_rows),
                default=10**9,
            )
            earliest_rows = [
                result for result in best_document_rows
                if (result.chunk.page or 10**9) == earliest_page
            ]
            selected.append(max(earliest_rows, key=lambda result: result.rrf_score))

    return selected or exact_rows


def _attach_source_row_citations(
    answer: str,
    citations: List[Citation],
    chunks_by_id: Dict[str, FinancialChunk],
) -> str:
    """Attach the actual row citation when generated text repeats its label and value."""
    updated_lines: list[str] = []
    citation_pattern = re.compile(r"\[(?:c?\d+)\]", re.IGNORECASE)
    for line in answer.splitlines():
        line_without_citations = citation_pattern.sub(" ", line)
        line_text = normalize_directive_text(line_without_citations)
        line_values = {
            NumericValidator.normalize_number(number)
            for number in NumericValidator.extract_numbers(line_without_citations)
        }
        line_values.discard("")
        matching_ids: list[str] = []
        if line_text and line_values:
            for citation in citations:
                chunk = chunks_by_id.get(citation.chunk_id or "")
                if chunk is None or chunk.chunk_type != ChunkType.FINANCIAL_ITEM:
                    continue
                raw_label = source_item_label(chunk)
                raw_label = re.sub(r"\s*\(\s*\d+(?:\s*[=.:].*)?\s*\)\s*$", "", raw_label)
                normalized_label = normalize_directive_text(raw_label)
                if not normalized_label or normalized_label not in line_text:
                    continue

                source_values = chunk.metadata.get("period_values") or {}
                normalized_source_values = {
                    NumericValidator.normalize_number(str(value))
                    for value in source_values.values()
                } if isinstance(source_values, dict) else set()
                for key in ("end_value", "start_value"):
                    value = chunk.metadata.get(key)
                    if value not in (None, "", "N/A"):
                        normalized_source_values.add(
                            NumericValidator.normalize_number(str(value))
                        )
                normalized_source_values.discard("")
                if line_values & normalized_source_values:
                    matching_ids.append(citation.citation_id)

        if matching_ids:
            references = " ".join(
                f"[{citation_id}]" for citation_id in dict.fromkeys(matching_ids)
            )
            replacement_index = 0

            def replace_existing_reference(_match: re.Match[str]) -> str:
                nonlocal replacement_index
                if replacement_index:
                    return ""
                replacement_index += 1
                return references

            # If the model supplied a citation for the wrong row, replace it
            # in place with the source row(s) whose label and amount match.
            corrected_line = citation_pattern.sub(replace_existing_reference, line)
            if replacement_index == 0:
                corrected_line = corrected_line.rstrip()
                terminal_punctuation = re.search(r"([.!?])\s*$", corrected_line)
                if terminal_punctuation:
                    insert_at = terminal_punctuation.start(1)
                    corrected_line = (
                        corrected_line[:insert_at].rstrip()
                        + " " + references
                        + corrected_line[insert_at:]
                    )
                elif re.search(r"\|\s*$", corrected_line):
                    pipe_at = corrected_line.rfind("|")
                    corrected_line = (
                        corrected_line[:pipe_at].rstrip()
                        + " " + references + " "
                        + corrected_line[pipe_at:]
                    )
                else:
                    corrected_line += " " + references
            updated_lines.append(corrected_line)
        else:
            updated_lines.append(line)
    return "\n".join(updated_lines)


def _expand_vector_selected_tables(
    results: List[RetrievalResult],
    chunks: List[FinancialChunk],
    parsed_query: ParsedQuery | None = None,
) -> List[RetrievalResult]:
    """Expand vector-selected table groups with their neighboring source rows.

    The vector ranking chooses which source table answers the question. The
    table title/page metadata then restores sibling rows so composition queries
    have enough context; no schema node is used to select or exclude rows.
    """
    if not results:
        return results

    # A structured request such as "cơ cấu nợ phải trả" can match identically
    # named rows in statement pages, segment notes, liquidity notes, and risk
    # notes. If retrieval found evidence in the statement explicitly implied
    # by the query, choose among those vector-ranked results first. Keep the
    # broader evidence as a fallback for OCR chunks with missing/wrong section
    # metadata; the schema does not select component rows.
    if parsed_query and parsed_query.statement_type is not None:
        statement_results = [
            result for result in results
            if result.chunk.statement_type == parsed_query.statement_type
        ]
        if statement_results:
            results = statement_results

    def key_for(chunk: FinancialChunk) -> tuple[str, str, str, str, str, str]:
        company = (chunk.company_code or chunk.source_file).casefold()
        document = chunk.document_id or chunk.source_file
        report_type = chunk.report_type or ""
        statement_type = chunk.statement_type.value
        title = _normalized_table_title(chunk)
        if not title:
            title = f"page:{chunk.page or 0}"
        return company, document, report_type, str(chunk.fiscal_year or ""), statement_type, title

    group_scores: dict[tuple[str, str, str, str, str, str], float] = {}
    group_hits: dict[tuple[str, str, str, str, str, str], int] = {}
    group_years: dict[tuple[str, str, str, str, str, str], set[int]] = {}
    seed_keys: list[tuple[str, str, str, str, str, str]] = []
    for rank, result in enumerate(results, start=1):
        key = key_for(result.chunk)
        if key not in group_scores:
            seed_keys.append(key)
            group_scores[key] = 0.0
            group_hits[key] = 0
            group_years[key] = set()
        group_scores[key] += max(result.rrf_score, 0.0) / (1.0 + rank * 0.05)
        group_hits[key] += 1
        chunk = result.chunk
        years = {int(chunk.fiscal_year)} if chunk.fiscal_year is not None else set()
        comparative_year = chunk.comparative_fiscal_year or chunk.metadata.get(
            "comparative_fiscal_year"
        )
        try:
            if comparative_year is not None:
                years.add(int(comparative_year))
        except (TypeError, ValueError):
            pass
        period_values = chunk.metadata.get("period_values") or {}
        if isinstance(period_values, dict):
            years.update(int(year) for year in period_values if str(year).isdigit())
        group_years[key].update(years)

    requested_years = (
        parsed_query.fiscal_years
        or ([parsed_query.fiscal_year] if parsed_query and parsed_query.fiscal_year else [])
    ) if parsed_query else []
    best_by_company: dict[str, tuple[str, str, str, str, str, str]] = {}
    selected_keys: set[tuple[str, str, str, str, str, str]] = set()
    companies = list(dict.fromkeys(key[0] for key in seed_keys))
    if requested_years:
        for company in companies:
            company_keys = [key for key in seed_keys if key[0] == company]
            company_selected = False
            for year in requested_years:
                covering = [key for key in company_keys if year in group_years[key]]
                if covering:
                    selected_keys.add(max(
                        covering,
                        key=lambda key: (group_scores[key], group_hits[key]),
                    ))
                    company_selected = True
            if company_keys and not company_selected:
                selected_keys.add(max(
                    company_keys,
                    key=lambda key: (group_scores[key], group_hits[key]),
                ))
    if not selected_keys:
        for key in seed_keys:
            company = key[0]
            current = best_by_company.get(company)
            rank_value = (group_scores[key], group_hits[key])
            if current is None or rank_value > (group_scores[current], group_hits[current]):
                best_by_company[company] = key
        selected_keys = set(best_by_company.values())
    if not selected_keys:
        return results

    selected_seed_results = [result for result in results if key_for(result.chunk) in selected_keys]
    # Resolve composition requests from an exact raw row/group phrase. If the
    # raw hierarchy is absent, keep only vector-selected evidence instead of
    # expanding the whole statement table into an unrelated answer.
    if parsed_query and parsed_query.structured_operation in {"components", "composition", "contribution_analysis"}:
        company_codes = parsed_query.company_codes or (
            [parsed_query.company_code] if parsed_query.company_code else []
        )
        composition_rows = _composition_source_rows(
            (
                analysis_target_phrase(parsed_query.original_query)
                if parsed_query.structured_operation == "contribution_analysis"
                else parsed_query.original_query
            ),
            company_codes,
            selected_keys,
            chunks,
            key_for,
        )
        if composition_rows:
            result_by_id = {result.chunk.chunk_id: result for result in selected_seed_results}
            def composition_row_position(chunk: FinancialChunk) -> tuple[int, int, str]:
                item_match = re.search(r"_item_(\d+)(?:_|$)", chunk.chunk_id)
                return (
                    chunk.page or 0,
                    int(item_match.group(1)) if item_match is not None else 0,
                    chunk.chunk_id,
                )

            composition_rows.sort(key=composition_row_position)
            return [
                result_by_id.get(chunk.chunk_id)
                or RetrievalResult(chunk=chunk, rrf_score=0.0)
                for chunk in composition_rows
            ]
        vector_selected_items = [
            result for result in selected_seed_results
            if result.chunk.chunk_type == ChunkType.FINANCIAL_ITEM
        ]
        return vector_selected_items or selected_seed_results
    present_ids = {result.chunk.chunk_id for result in selected_seed_results}
    expansions: list[FinancialChunk] = []
    for chunk in chunks:
        if key_for(chunk) not in selected_keys or chunk.chunk_id in present_ids:
            continue
        if chunk.chunk_type not in {ChunkType.FINANCIAL_ITEM, ChunkType.TABLE}:
            continue
        expansions.append(chunk)

    expansions.sort(key=lambda chunk: (
        0 if chunk.chunk_type == ChunkType.FINANCIAL_ITEM else 1,
        chunk.page or 0,
        int(chunk.metadata.get("row_start", 0) or 0),
        chunk.chunk_id,
    ))
    selected_seed_results.extend(
        RetrievalResult(chunk=chunk, rrf_score=0.0) for chunk in expansions
    )
    return selected_seed_results


def _prepare_ingested_document(req: IngestRequest) -> tuple[List[FinancialChunk], dict[str, Any]]:
    return prepare_document(req, chunker)

def _replaces_request(chunk: FinancialChunk, req: IngestRequest, company_code: str | None) -> bool:
    if req.document_id:
        return chunk.document_id == req.document_id
    return (
        chunk.source_file.casefold() == req.source_file.casefold()
        and (chunk.company_code or "").casefold() == (company_code or "").casefold()
    )

@app.get("/health")
@app.get("/api/v1/financial/health")
def health_check():
    return {
        "status": "healthy",
        "service": "financial-rag-python",
        "indexed_chunks": document_store.chunk_count() if _scoped_runtime() else len(CHUNKS_DB),
        "runtime_mode": "scoped_repository" if _scoped_runtime() else "legacy_memory",
        "scope_cache": scope_cache.stats(),
        "persistence": storage_backend if document_store is not None else "in_memory",
        "stored_documents": document_store.document_count() if document_store is not None else None,
        "vector_store": "qdrant" if storage_backend in {"mongo_qdrant", "mongodb_qdrant"} else "sqlite_cache",
        "ollama_model": chat_engine.model_name,
        "dense_retrieval": retriever.dense_index_status if retriever else "disabled",
        "dense_cached_vectors": (
            retriever.dense_retriever.cached_vector_count
            if retriever and retriever.dense_retriever else 0
        ),
        "dense_retrieval_error": (
            retriever.dense_retriever.last_error
            if retriever and retriever.dense_retriever
            else None
        ),
    }


@app.post("/api/v1/financial/llm/test-connection")
def test_llm_connection(req: LLMConnectionTestRequest):
    """Check an Ollama or OpenAI-compatible endpoint without changing chat runtime."""
    try:
        parsed_url = urlsplit(req.base_url)
        hostname = parsed_url.hostname
        parsed_url.port  # Validate malformed ports before passing the URL to requests.
    except ValueError:
        parsed_url = None
        hostname = None
    if (
        parsed_url is None
        or parsed_url.scheme not in {"http", "https"}
        or not hostname
        or parsed_url.username
        or parsed_url.password
        or parsed_url.query
        or parsed_url.fragment
    ):
        return {
            "connected": False,
            "message": "Địa chỉ kết nối cần bắt đầu bằng http:// hoặc https:// và không chứa thông tin đăng nhập.",
            "models": [],
            "selected_model_available": None,
        }

    base_url = req.base_url.rstrip("/")
    if req.provider == "ollama":
        endpoint = f"{base_url}/api/tags"
        headers: Dict[str, str] = {}
    else:
        endpoint = f"{base_url}/models"
        headers = {"Authorization": f"Bearer {req.api_key}"} if req.api_key else {}

    try:
        response = requests.get(endpoint, headers=headers, timeout=12)
    except requests.Timeout:
        return {
            "connected": False,
            "message": "Hết thời gian chờ. Kiểm tra dịch vụ và địa chỉ kết nối.",
            "models": [],
            "selected_model_available": None,
        }
    except requests.RequestException:
        return {
            "connected": False,
            "message": "Không kết nối được. Kiểm tra địa chỉ và trạng thái dịch vụ.",
            "models": [],
            "selected_model_available": None,
        }

    if response.status_code in {401, 403}:
        return {
            "connected": False,
            "message": "Dịch vụ từ chối xác thực. Kiểm tra API key.",
            "models": [],
            "selected_model_available": None,
        }
    if response.status_code == 404 and req.provider == "openai_compatible":
        return {
            "connected": False,
            "message": "Không tìm thấy /models. Cần địa chỉ API tương thích OpenAI có endpoint /models.",
            "models": [],
            "selected_model_available": None,
        }
    if not response.ok:
        return {
            "connected": False,
            "message": f"Dịch vụ trả về HTTP {response.status_code}.",
            "models": [],
            "selected_model_available": None,
        }

    try:
        payload = response.json()
    except ValueError:
        return {
            "connected": False,
            "message": "Dịch vụ phản hồi không đúng định dạng JSON.",
            "models": [],
            "selected_model_available": None,
        }

    if req.provider == "ollama":
        model_rows = payload.get("models", []) if isinstance(payload, dict) else []
        model_names = [
            str(item.get("name") or item.get("model") or "").strip()
            for item in model_rows
            if isinstance(item, dict)
        ]
    else:
        model_rows = payload.get("data", []) if isinstance(payload, dict) else []
        model_names = [
            str(item.get("id") or "").strip()
            for item in model_rows
            if isinstance(item, dict)
        ]

    models = sorted({name for name in model_names if name})[:250]
    selected_model_available = req.model in models if req.model and models else None
    if selected_model_available is False:
        message = "Đã kết nối, nhưng chưa thấy mô hình đã chọn trong danh sách dịch vụ."
    elif not models:
        message = "Đã kết nối. Dịch vụ không trả danh sách mô hình."
    else:
        message = f"Kết nối thành công · {len(models)} mô hình khả dụng."

    return {
        "connected": True,
        "message": message,
        "models": models,
        "selected_model_available": selected_model_available,
    }


@app.get("/health/live")
@app.get("/api/v1/financial/health/live")
def liveness_check():
    return {"status": "alive", "service": "financial-rag-python"}


@app.get("/api/v1/financial/capabilities")
def financial_capabilities():
    return {"profiles": ["bank", "enterprise", "securities"],
            "operations": ["lookup", "comparison", "trend", "composition", "ratio",
                           "contribution", "assessment", "source_explanation", "issuer_ranking"],
            "budgets": {"companies": 5, "periods": 10, "metric_cells": 100,
                        "request_chunks": int(os.getenv("FINANCIAL_REQUEST_MAX_CHUNKS", "30000"))},
            "source_policies": ["source_report", "latest_restated"],
            "interpretation_policy": "extractive_or_deterministic; no implicit normative thresholds",
            "independent_accuracy_gate": "pending_source_and_interpretation_review",
            "ingestion": {"worker_initializes_api": False, "stage_replay": "checksum_and_revision",
                          "stage_cache_bytes": 67108864, "publication": "lease_fenced_and_projection_cas"},
            "retention": {"source_versions": "retain", "retrieval_snapshots": "retain",
                          "qdrant_history": "retain; excluded by active request point IDs",
                          "terminal_jobs_days": 30, "sparse_cache_bytes": 67108864}}


@app.get("/health/ready")
@app.get("/api/v1/financial/health/ready")
def readiness_check():
    if chunker is None or query_engine is None:
        raise HTTPException(status_code=503, detail="Financial service is not initialized")
    return {
        "status": "ready",
        "service": "financial-rag-python",
        "components": {
            "markdown_chunker": "ready",
            "query_understanding": "ready",
            "document_store": "ready" if document_store is not None else "in_memory",
            "vector_store": (
                "qdrant"
                if storage_backend in {"mongo_qdrant", "mongodb_qdrant"}
                else "sqlite_cache"
            ),
            "answer_model": "external_unchecked",
            "dense_retrieval": retriever.dense_index_status if retriever else "disabled",
            "dense_retrieval_error": (
                retriever.dense_retriever.last_error
                if retriever and retriever.dense_retriever
                else None
            ),
        },
    }

@app.post("/api/v1/financial/ingest", response_model=Dict[str, Any])
@_with_runtime_write_lock
def ingest_document(req: IngestRequest):
    global CHUNKS_DB, retriever
    with _log_document_operation("ingest", req):
        new_chunks, document = _prepare_ingested_document(req)
        if not new_chunks:
            raise HTTPException(
                status_code=422,
                detail=f"Không trích xuất được dữ liệu để lập chỉ mục từ {req.source_file}.",
            )
        if _scoped_runtime():
            _publish_scoped_documents([document])
            return {"status": "success", "new_chunks_count": len(new_chunks),
                    "total_chunks_indexed": document_store.chunk_count(),
                    "document_id": document["document_id"], "logical_key": document["logical_key"]}

        # Gửi lại cùng một file sẽ thay dữ liệu cũ thay vì tạo bản sao trong index.
        updated_chunks = [
            chunk for chunk in CHUNKS_DB
            if not _replaces_request(chunk, req, document.get("company_code"))
        ]
        updated_chunks.extend(new_chunks)
        updated_retriever = build_retriever(updated_chunks)
        if document_store is not None:
            document_store.replace_documents([document])
            refresh_company_aliases()
        CHUNKS_DB = updated_chunks
        retriever = updated_retriever
        return {
            "status": "success",
            "new_chunks_count": len(new_chunks),
            "total_chunks_indexed": len(CHUNKS_DB),
            "document_id": document["document_id"],
            "logical_key": document["logical_key"],
        }


@app.post("/api/v1/financial/ingest/batch", response_model=Dict[str, Any])
@_with_runtime_write_lock
def ingest_documents_batch(req: BatchIngestRequest):
    global CHUNKS_DB, retriever
    prepared = [
        (source, *_prepare_ingested_document(source))
        for source in req.documents
    ]
    nonempty = [item for item in prepared if item[1]]
    documents_without_chunks = [
        source.source_file for source, chunks, _ in prepared if not chunks
    ]
    if not nonempty:
        file_list = ", ".join(documents_without_chunks)
        raise HTTPException(
            status_code=422,
            detail=f"Không tạo được chunk để lập chỉ mục từ các tệp: {file_list}.",
        )

    documents = [item[2] for item in nonempty]
    logical_keys = [document["logical_key"] for document in documents]
    if len(logical_keys) != len(set(logical_keys)):
        raise HTTPException(status_code=422, detail="Batch contains duplicate logical documents")

    if _scoped_runtime():
        _publish_scoped_documents(documents)
        return {"status": "success", "documents_processed": len(req.documents),
                "documents_with_chunks": len(documents), "documents_without_chunks": documents_without_chunks,
                "new_chunks_count": sum(len(item[1]) for item in nonempty),
                "total_chunks_indexed": document_store.chunk_count(), "logical_keys": logical_keys}

    updated_chunks = [
        chunk
        for chunk in CHUNKS_DB
        if not any(_replaces_request(chunk, source, document.get("company_code"))
                   for source, _, document in nonempty)
    ]
    for _, chunks, _ in nonempty:
        updated_chunks.extend(chunks)
    updated_retriever = build_retriever(updated_chunks)
    if document_store is not None:
        document_store.replace_documents(documents)
        refresh_company_aliases()
    CHUNKS_DB = updated_chunks
    retriever = updated_retriever
    return {
        "status": "success",
        "documents_processed": len(req.documents),
        "documents_with_chunks": len(documents),
        "documents_without_chunks": documents_without_chunks,
        "new_chunks_count": sum(len(item[1]) for item in nonempty),
        "total_chunks_indexed": len(CHUNKS_DB),
        "logical_keys": logical_keys,
    }


@app.get("/api/v1/financial/documents", response_model=List[Dict[str, Any]])
def list_ingested_documents():
    if document_store is None:
        raise HTTPException(status_code=503, detail="Durable document storage is not configured")
    return document_store.list_documents()


@app.post("/api/v1/financial/ingest/jobs", status_code=202)
def enqueue_ingest_job(req: BatchIngestRequest):
    if not _scoped_runtime():
        raise HTTPException(status_code=503, detail="Durable ingest jobs require scoped MongoDB storage")
    return document_store.enqueue_ingest(req.model_dump(mode="json"))


@app.get("/api/v1/financial/ingest/jobs/{job_id}")
def get_ingest_job(job_id: str):
    if not _scoped_runtime():
        raise HTTPException(status_code=503, detail="Durable ingest jobs require scoped MongoDB storage")
    job = document_store.get_ingest_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingest job not found")
    return job


@app.post("/api/v1/financial/ingest/jobs/{job_id}/retry")
def retry_ingest_job(job_id: str):
    if not _scoped_runtime():
        raise HTTPException(status_code=503, detail="Durable ingest jobs require scoped MongoDB storage")
    job = document_store.retry_ingest(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingest job not found")
    return job


@app.get("/api/v1/financial/documents/{logical_key}/versions")
def list_document_versions(logical_key: str):
    if document_store is None:
        raise HTTPException(status_code=503, detail="Durable document storage is not configured")
    return document_store.list_versions(logical_key)


class ActivateProjectionRequest(BaseModel):
    projection_id: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]


@app.post("/api/v1/financial/documents/{logical_key}/activate")
@_with_runtime_write_lock
def activate_document_projection(logical_key: str, req: ActivateProjectionRequest):
    if not _scoped_runtime():
        raise HTTPException(status_code=503, detail="Projection activation requires scoped MongoDB")
    try:
        chunks = document_store.projection_chunks(logical_key, req.projection_id)
        if os.getenv("FINANCIAL_ENABLE_DENSE", "false").casefold() in {"1", "true", "yes"}:
            index = build_retriever(chunks, isolated=True)
            if index._dense_index_thread:
                index._dense_index_thread.join()
            if index.dense_index_status != "ready":
                raise RuntimeError("Dense snapshot is not ready")
        result = document_store.activate_projection(logical_key, req.projection_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    scope_cache.clear()
    refresh_company_aliases()
    return result


@app.get("/api/v1/financial/documents/{logical_key}/facts")
def list_document_facts(
    logical_key: str, period_year: int | None = Query(default=None, ge=1900, le=2200),
    projection_id: str | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0, le=1_000_000),
):
    if document_store is None:
        raise HTTPException(status_code=503, detail="Durable document storage is not configured")
    try:
        return document_store.list_facts(logical_key, period_year=period_year,
                                         projection_id=projection_id, limit=limit, offset=offset)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _conversation_history_store() -> Any:
    if document_store is None or not hasattr(document_store, "list_conversations"):
        raise HTTPException(
            status_code=503,
            detail="MongoDB conversation history is not configured",
        )
    return document_store


@app.get("/api/v1/financial/conversations", response_model=List[Dict[str, Any]])
def list_conversation_history():
    return _conversation_history_store().list_conversations(limit=24)


@app.put("/api/v1/financial/conversations/{conversation_id}", response_model=Dict[str, Any])
def save_conversation_history(
    conversation_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)],
    req: ConversationHistoryPayload,
):
    if conversation_id != req.id:
        raise HTTPException(status_code=422, detail="Conversation id does not match the request body")
    payload = req.model_dump(mode="json")
    payload_size = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    if payload_size > 8_000_000:
        raise HTTPException(status_code=413, detail="Conversation history exceeds the 8 MB limit")
    _conversation_history_store().save_conversation(payload)
    return {"status": "saved", "conversation_id": conversation_id}


@app.delete("/api/v1/financial/conversations/{conversation_id}", response_model=Dict[str, Any])
def delete_conversation_history(
    conversation_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)],
):
    if not _conversation_history_store().delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "deleted", "conversation_id": conversation_id}


@app.delete("/api/v1/financial/documents/{logical_key}", response_model=Dict[str, Any])
@_with_runtime_write_lock
def delete_ingested_document(logical_key: str):
    global CHUNKS_DB, retriever
    if document_store is None:
        raise HTTPException(status_code=503, detail="Durable document storage is not configured")
    document = next(
        (item for item in document_store.list_documents() if item["logical_key"] == logical_key),
        None,
    )
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if not document_store.delete_document(logical_key):
        raise HTTPException(status_code=404, detail="Document not found")
    refresh_company_aliases()
    retained_chunks = [
        chunk for chunk in CHUNKS_DB
        if chunk.document_id != document["document_id"]
    ]
    updated_retriever = build_retriever(retained_chunks) if retained_chunks else None
    CHUNKS_DB = retained_chunks
    retriever = updated_retriever
    return {
        "status": "deleted",
        "source_file": document["source_file"],
        "remaining_chunks": document_store.chunk_count() if _scoped_runtime() else len(CHUNKS_DB),
    }


@app.post("/api/v1/financial/query/understand", response_model=ParsedQuery)
def understand_query(req: QueryRequest):
    parsed = query_engine.parse(req.query)
    parsed.source_policy = req.source_policy
    return query_engine.resolve_company_context(parsed, _catalog_chunks() if _scoped_runtime() else _runtime_chunks())


def _retrieve_financial_overview_metrics(
    plan: QueryPlan,
    parsed: ParsedQuery,
    top_k: int,
) -> tuple[dict[str, list[RetrievalResult]], list[RetrievalResult], dict[str, Any]]:
    """Vector-retrieve each planned metric while preserving the request scope."""
    metrics_by_id = {
        metric.metric_id: metric
        for metric in query_engine.metric_mapper.metrics_list
    }
    for metric_id in plan.planned_metric_ids:
        if metric_id in metrics_by_id:
            continue
        node = query_engine.schema_catalog.nodes.get(metric_id)
        if node and node.kind in {"line", "total", "group", "note_row"}:
            metrics_by_id[metric_id] = MetricDefinition(
                canonical_name=node.canonical_name,
                code=None,
                statement_type=node.statement_type,
                statement_name_vi=STATEMENT_NAMES_VI.get(
                    node.statement_type, "Báo cáo tài chính"
                ),
                section=node.parent_id or "schema_source_metric",
                aliases=list(node.aliases),
                metric_id=node.node_id,
            )
    companies = parsed.company_codes or ([parsed.company_code] if parsed.company_code else [])
    years = parsed.fiscal_years or (
        [parsed.fiscal_year] if parsed.fiscal_year is not None else []
    )
    if len(companies) != 1 or not years:
        return {}, [], metrics_by_id

    company = companies[0]
    year_text = " ".join(str(year) for year in years)
    report_phrase = {
        "consolidated": "báo cáo hợp nhất",
        "separate": "báo cáo riêng",
    }.get(parsed.report_type or "", "")
    results_by_metric: dict[str, list[RetrievalResult]] = {}
    unique_results: dict[str, RetrievalResult] = {}
    metric_top_k = min(32, max(16, top_k * 3))

    for metric_id in plan.planned_metric_ids:
        metric = metrics_by_id.get(metric_id)
        if metric is None:
            results_by_metric[metric_id] = []
            continue
        metric_results_by_id: dict[str, RetrievalResult] = {}
        query_terms = [metric.canonical_name]
        normalized_terms = {
            query_engine.schema_catalog.normalize_label(metric.canonical_name)
        }
        for alias in sorted(metric.aliases, key=len, reverse=True):
            normalized_alias = query_engine.schema_catalog.normalize_label(alias)
            if len(normalized_alias) < 8 or normalized_alias in normalized_terms:
                continue
            normalized_terms.add(normalized_alias)
            query_terms.append(alias)
            if len(query_terms) >= 4:
                break

        def has_requested_period(chunk: FinancialChunk, year: int) -> bool:
            period_values = chunk.metadata.get("period_values")
            raw = (
                period_values.get(str(year), period_values.get(year))
                if isinstance(period_values, dict)
                else None
            )
            if raw not in (None, "", "-", "N/A"):
                return True
            if chunk.fiscal_year == year:
                return chunk.metadata.get("end_value") not in (None, "", "-", "N/A")
            if chunk.comparative_fiscal_year == year:
                return chunk.metadata.get("start_value") not in (None, "", "-", "N/A")
            return False

        for query_term in query_terms:
            synthetic_query = " ".join(
                part for part in (
                    query_term,
                    "của",
                    company,
                    "năm",
                    year_text,
                    report_phrase,
                ) if part
            )
            metric_query = query_engine.parse(synthetic_query).model_copy(update={
                "original_query": synthetic_query,
                "company_code": company,
                "company_codes": [company],
                "statement_type": metric.statement_type,
                "statement_name_vi": metric.statement_name_vi,
                "canonical_metric": metric.canonical_name,
                "metric_id": metric.metric_id,
                "primary_metric": metric.canonical_name,
                "metric_aliases": list(metric.aliases),
                "fiscal_year": years[-1],
                "fiscal_years": list(years),
                "fiscal_quarter": parsed.fiscal_quarter,
                "report_type": parsed.report_type,
                "period_type": parsed.period_type,
                "intent": QueryIntent.METRIC_LOOKUP,
                "structured_operation": None,
                # Schema annotations support chunking, but the overview path
                # verifies the retrieved source row against its label.
                "schema_node_id": None,
                "schema_node_name": None,
            })
            for result in _runtime_retriever().search(
                metric_query, top_k=metric_top_k,
                strict_metric_scope=plan.analysis_requirements.get("financial_profile") != "bank",
            ):
                chunk = result.chunk
                if (
                    chunk.chunk_type != ChunkType.FINANCIAL_ITEM
                    or (chunk.company_code or "").casefold() != company.casefold()
                    or chunk.metadata.get("report_eligible") is False
                    or (parsed.report_type and chunk.report_type != parsed.report_type)
                    or not matches_metric_source(chunk, metric_id, company, metrics_by_id)
                ):
                    continue
                previous = metric_results_by_id.get(chunk.chunk_id)
                if previous is None or result.rrf_score > previous.rrf_score:
                    metric_results_by_id[chunk.chunk_id] = result
            if all(
                any(has_requested_period(result.chunk, year) for result in metric_results_by_id.values())
                for year in years
            ):
                break

        metric_results = list(metric_results_by_id.values())
        results_by_metric[metric_id] = metric_results
        for result in metric_results:
            prior = unique_results.get(result.chunk.chunk_id)
            if prior is None or result.rrf_score > prior.rrf_score:
                unique_results[result.chunk.chunk_id] = result

    return results_by_metric, list(unique_results.values()), metrics_by_id


def _retrieve_series_metrics(plan: QueryPlan, parsed: ParsedQuery, top_k: int):
    """Retrieve each issuer-period cell independently to avoid top-k starvation."""
    grouped: dict[str, dict[str, RetrievalResult]] = {}
    unique: dict[str, RetrievalResult] = {}
    definitions: dict[str, MetricDefinition] = {}
    years = sorted({year for year in parsed.fiscal_years if year is not None})
    if not years and parsed.fiscal_year is not None:
        years = [parsed.fiscal_year]
    companies = [company for company in parsed.company_codes if company]
    if not companies and parsed.company_code is not None:
        companies = [parsed.company_code]
    for company in dict.fromkeys(companies):
        scope = _runtime_retriever()._chunks_by_company.get(company.casefold(), [])
        if parsed.fiscal_quarter is None:
            scope = [chunk for chunk in scope if chunk.fiscal_quarter is None]
        scope = _scope_reports(scope, years, [company], parsed.report_type)
        bases = {chunk.report_type for chunk in scope}
        basis = next(iter(bases)) if len(bases) == 1 else parsed.report_type
        for year in years:
            cell_query = parsed.model_copy(update={"company_code": company, "company_codes": [company],
                                                    "fiscal_year": year, "fiscal_years": [year],
                                                    "report_type": basis})
            rows, selected, definitions = _retrieve_financial_overview_metrics(plan, cell_query, top_k)
            for metric_id, values in rows.items():
                for result in values:
                    grouped.setdefault(metric_id, {})[result.chunk.chunk_id] = result
            unique.update({result.chunk.chunk_id: result for result in selected})
    return {metric: list(rows.values()) for metric, rows in grouped.items()}, list(unique.values()), definitions


def _retrieve_financial_overview_narrative(
    plan: QueryPlan,
    parsed: ParsedQuery,
    top_k: int,
) -> list[RetrievalResult]:
    """Find narrative evidence for interpretation without claiming it is causal."""
    if not any(group in plan.evidence_groups for group in (
        "supporting_narrative", "management_explanation", "candidate_driver_values"
    )):
        return []
    companies = parsed.company_codes or ([parsed.company_code] if parsed.company_code else [])
    years = parsed.fiscal_years or (
        [parsed.fiscal_year] if parsed.fiscal_year is not None else []
    )
    if len(companies) != 1 or not years:
        return []
    company = companies[0]
    query_text = (
        f"{parsed.original_query[:1200]} {company} {' '.join(map(str, years))} "
        "thuyết minh giải trình diễn biến kết quả tài chính và rủi ro"
    )
    narrative_query = query_engine.parse(query_text).model_copy(update={
        "original_query": query_text,
        "company_code": company,
        "company_codes": [company],
        "statement_type": StatementType.NOTES,
        "statement_name_vi": "Thuyết minh báo cáo tài chính",
        "canonical_metric": None,
        "metric_id": None,
        "primary_metric": None,
        "metric_aliases": [],
        "fiscal_year": years[-1],
        "fiscal_years": list(years),
        "fiscal_quarter": parsed.fiscal_quarter,
        "report_type": parsed.report_type,
        "period_type": parsed.period_type,
        "intent": QueryIntent.GENERAL,
        "structured_operation": None,
        "schema_node_id": None,
        "schema_node_name": None,
    })
    candidate_results = _runtime_retriever().search(narrative_query, top_k=min(12, max(6, top_k)))
    narrative_results = [
        result for result in candidate_results
        if result.chunk.statement_type == StatementType.NOTES
        and result.chunk.chunk_type in {ChunkType.NARRATIVE, ChunkType.PARENT_CHILD}
    ]
    return narrative_results[:4]


def _retrieve_ratio_input_rows(
    parsed: ParsedQuery,
    top_k: int,
) -> list[RetrievalResult]:
    """Retrieve each ratio operand independently, retaining only source-label matches."""
    companies = parsed.company_codes or ([parsed.company_code] if parsed.company_code else [])
    years = parsed.fiscal_years or (
        [parsed.fiscal_year] if parsed.fiscal_year is not None else []
    )
    if len(companies) != 1 or not years:
        return []
    requested_inputs = requested_ratio_inputs(parsed.original_query)
    if not requested_inputs:
        return []

    metrics_by_id = {
        metric.metric_id: metric
        for metric in query_engine.metric_mapper.metrics_list
    }
    company = companies[0]
    year_text = " ".join(str(year) for year in years)
    report_phrase = {
        "consolidated": "báo cáo hợp nhất",
        "separate": "báo cáo riêng",
    }.get(parsed.report_type or "", "")
    metric_top_k = min(24, max(12, top_k * 3))
    selected_results: dict[str, RetrievalResult] = {}

    for spec in requested_inputs:
        metric = metrics_by_id.get(spec.metric_id)
        if metric is None:
            continue
        synthetic_query = " ".join(part for part in (
            metric.canonical_name,
            "của",
            company,
            "năm",
            year_text,
            report_phrase,
        ) if part)
        metric_query = query_engine.parse(synthetic_query).model_copy(update={
            "original_query": synthetic_query,
            "company_code": company,
            "company_codes": [company],
            "statement_type": metric.statement_type,
            "statement_name_vi": metric.statement_name_vi,
            "canonical_metric": metric.canonical_name,
            "metric_id": metric.metric_id,
            "primary_metric": metric.canonical_name,
            "metric_aliases": list(metric.aliases),
            "fiscal_year": years[-1],
            "fiscal_years": list(years),
            "fiscal_quarter": parsed.fiscal_quarter,
            "report_type": parsed.report_type,
            "period_type": parsed.period_type,
            "intent": QueryIntent.METRIC_LOOKUP,
            "structured_operation": None,
            "schema_node_id": None,
            "schema_node_name": None,
        })
        candidates = _runtime_retriever().search(metric_query, top_k=metric_top_k)
        best_by_document: dict[tuple[str, str], RetrievalResult] = {}
        for result in candidates:
            chunk = result.chunk
            if (
                chunk.chunk_type != ChunkType.FINANCIAL_ITEM
                or (chunk.company_code or "").casefold() != company.casefold()
                or chunk.metadata.get("report_eligible") is False
                or not matches_metric_source(
                    chunk, spec.metric_id, company, metrics_by_id
                )
            ):
                continue
            key = (chunk.document_id or chunk.source_file, chunk.report_type or "")
            previous = best_by_document.get(key)
            if previous is None or result.rrf_score > previous.rrf_score:
                best_by_document[key] = result
        for result in best_by_document.values():
            previous = selected_results.get(result.chunk.chunk_id)
            if previous is None or result.rrf_score > previous.rrf_score:
                selected_results[result.chunk.chunk_id] = result

    return list(selected_results.values())


def _requested_overview_composition_metrics(
    question: str,
    plan: QueryPlan,
) -> list[str]:
    """Find balance-sheet roots explicitly requested after a composition cue."""
    normalized = normalize_directive_text(question)
    composition_start = normalized.find("co cau")
    if composition_start < 0:
        composition_start = normalized.find("thanh phan")
    if composition_start < 0:
        return []
    clause = normalized[composition_start:]
    clause = re.split(
        r"[;.!?]|\b(?:tinh|ty le|ti le|doi chieu|so sanh|bien dong)\b",
        clause,
        maxsplit=1,
    )[0]
    concepts = {
        "tong_cong_tai_san": ("tai san",),
        "tong_no_phai_tra": ("no phai tra",),
        "tong_von_chu_so_huu": ("von chu so huu",),
    }
    planned = set(plan.planned_metric_ids)
    return [
        metric_id
        for metric_id, terms in concepts.items()
        if metric_id in planned and any(term in clause for term in terms)
    ]


def _chunk_contains_period(chunk: FinancialChunk, years: list[int]) -> bool:
    period_values = chunk.metadata.get("period_values")
    for year in years:
        raw = None
        if isinstance(period_values, dict):
            raw = period_values.get(str(year), period_values.get(year))
        if raw in (None, "", "N/A", "-"):
            if chunk.fiscal_year == year:
                raw = chunk.metadata.get("end_value")
            elif chunk.comparative_fiscal_year == year:
                raw = chunk.metadata.get("start_value")
        if not NumericValidator.normalize_number(str(raw or "")):
            return False
    return bool(years)


def _retrieve_overview_composition_rows(
    question: str,
    plan: QueryPlan,
    parsed: ParsedQuery,
    results_by_metric: dict[str, list[RetrievalResult]],
    overview_citations: list[Citation],
) -> dict[str, list[RetrievalResult]]:
    """Expand only cited, scope-matched total rows for requested balance-sheet groups."""
    metric_ids = _requested_overview_composition_metrics(question, plan)
    companies = parsed.company_codes or ([parsed.company_code] if parsed.company_code else [])
    years = parsed.fiscal_years or (
        [parsed.fiscal_year] if parsed.fiscal_year is not None else []
    )
    if not metric_ids or len(companies) != 1 or not years:
        return {}

    cited_chunk_ids = {
        citation.chunk_id
        for citation in overview_citations
        if citation.chunk_id
    }
    metrics_by_id = {
        metric.metric_id: metric
        for metric in query_engine.metric_mapper.metrics_list
    }
    company = companies[0]
    expanded_by_metric: dict[str, list[RetrievalResult]] = {}
    for metric_id in metric_ids:
        metric = metrics_by_id.get(metric_id)
        if metric is None:
            continue
        anchors = [
            result
            for result in results_by_metric.get(metric_id, [])
            if result.chunk.chunk_id in cited_chunk_ids
            and result.chunk.chunk_type == ChunkType.FINANCIAL_ITEM
            and (result.chunk.company_code or "").casefold() == company.casefold()
            and result.chunk.statement_type == metric.statement_type
            and result.chunk.metadata.get("report_eligible") is not False
            and (not parsed.report_type or result.chunk.report_type == parsed.report_type)
            and matches_metric_source(result.chunk, metric_id, company, metrics_by_id)
            and _chunk_contains_period(result.chunk, years)
        ]
        if not anchors:
            continue
        # Use the cited row from the latest requested statement period. Its
        # comparative column supplies the earlier year when a two-year view
        # was requested; all expanded rows remain in this exact filing.
        anchor = max(
            anchors,
            key=lambda result: (
                result.chunk.fiscal_year == max(years),
                result.rrf_score,
            ),
        )
        composition_query = parsed.model_copy(update={
            "original_query": f"Cơ cấu {metric.canonical_name} của {company} năm {' '.join(map(str, years))}",
            "company_code": company,
            "company_codes": [company],
            "statement_type": metric.statement_type,
            "statement_name_vi": metric.statement_name_vi,
            "canonical_metric": metric.canonical_name,
            "metric_id": metric.metric_id,
            "primary_metric": metric.canonical_name,
            "metric_aliases": list(metric.aliases),
            "fiscal_year": max(years),
            "fiscal_years": list(years),
            "intent": QueryIntent.METRIC_LOOKUP,
            "structured_operation": "composition",
            "schema_node_id": None,
            "schema_node_name": None,
        })
        expanded = _expand_vector_selected_tables([anchor], _runtime_chunks(), composition_query)
        document_key = anchor.chunk.document_id or anchor.chunk.source_file
        expanded = [
            result
            for result in expanded
            if result.chunk.chunk_type == ChunkType.FINANCIAL_ITEM
            and (result.chunk.company_code or "").casefold() == company.casefold()
            and (result.chunk.document_id or result.chunk.source_file) == document_key
            and result.chunk.report_type == anchor.chunk.report_type
            and result.chunk.statement_type == metric.statement_type
        ]
        if anchor.chunk.chunk_id not in {result.chunk.chunk_id for result in expanded}:
            expanded.insert(0, anchor)
        expanded_by_metric[metric_id] = expanded
    return expanded_by_metric

@app.post("/api/v1/financial/retrieve", response_model=List[RetrievalResult])
@_with_request_snapshot
def retrieve_chunks(req: QueryRequest):
    if not _runtime_retriever():
        raise HTTPException(status_code=400, detail="Chưa có dữ liệu nào được lập chỉ mục.")
    parsed = query_engine.resolve_company_context(
        query_engine.parse(req.query), _runtime_chunks()
    )
    if parsed.structured_operation == "ratio_analysis":
        # A ratio retrieval request represents the complete formula evidence
        # bundle, not only the one balance-sheet root that won query parsing.
        # Retrieve every operand independently so callers and benchmarks can
        # inspect both numerator and denominator citations.
        return _retrieve_ratio_input_rows(parsed, req.top_k)[:req.top_k]
    directives = directive_planner.plan(req.query, parsed)
    candidate_k = min(100, max(req.top_k, req.top_k * 3))
    results = _runtime_retriever().search(parsed, top_k=candidate_k, directive_plan=directives)
    results, _ = ValidateOperator.rerank(results, directives)
    results, _ = FilterOperator.apply(
        results, directives, parsed, classifier=filter_classifier
    )
    return results[:req.top_k]

@app.post("/api/v1/financial/answer", response_model=FinancialAnswerResponse)
@_with_request_snapshot
def answer_question(req: QueryRequest):
    if not _runtime_retriever():
        if _scoped_runtime():
            parsed = query_engine.resolve_company_context(query_engine.parse(req.query), _catalog_chunks())
            return FinancialAnswerResponse(query=req.query, parsed_query=parsed, citations=[],
                validation=ValidationResult(is_valid=True, unit_match=True, citation_valid=True),
                answer="Chưa có báo cáo phù hợp với doanh nghiệp, kỳ và cơ sở được hỏi. Hãy xác định phạm vi hoặc nạp báo cáo tương ứng.",
                answer_status="needs_clarification" if not parsed.company_code else "insufficient_evidence")
        raise HTTPException(status_code=400, detail="Chưa có dữ liệu nào được lập chỉ mục.")

    # 1. Query Understanding
    parsed = query_engine.resolve_company_context(
        query_engine.parse(req.query), _runtime_chunks()
    )
    parsed.source_policy = req.source_policy
    directives = directive_planner.plan(req.query, parsed)

    query_plan = None
    try:
        # This API indexes one local corpus and currently has no per-tenant ACL;
        # company, year, quarter, and report basis remain hard retrieval scope.
        query_plan = build_query_plan(
            req.query,
            access_filter={"corpus": "local_financial_index"},
            parsed_query=parsed,
            engine=query_engine,
            source_chunks=_runtime_chunks(),
        )
    except Exception:
        logger.exception("financial_query_plan_failed; using the existing answer route")

    ratio_input_ids = {spec.metric_id for spec in requested_ratio_inputs(req.query)}
    balance_sheet_ratio_ids = {
        "tong_no_phai_tra",
        "tong_cong_tai_san",
        "tong_von_chu_so_huu",
    }
    analytical_ratio_request = bool(
        parsed.structured_operation == "ratio_analysis"
        and ratio_input_ids
        and ratio_input_ids.issubset(balance_sheet_ratio_ids)
    )
    parsed_companies = parsed.company_codes or (
        [parsed.company_code] if parsed.company_code else []
    )
    parsed_years = parsed.fiscal_years or (
        [parsed.fiscal_year] if parsed.fiscal_year is not None else []
    )
    matrix_request = bool(parsed_companies and parsed_years and (
        len(parsed_companies) > 1 or len(parsed_years) > 2
        or (len(parsed_years) > 1 and any(term in req.query.casefold() for term in ("roa", "roe")))
    ) and parsed.structured_operation not in {"composition", "components", "contribution_analysis", "table"})
    if matrix_request:
        return _matrix_answer(req, parsed)
    single_company_period_comparison = bool(
        query_plan
        and query_plan.task in {"comparison", "trend"}
        and len(parsed_companies) == 1
        and len(parsed_years) == 2
        and query_plan.planned_metric_ids
    )
    series_request = bool(
        query_plan and query_plan.domain == "financial"
        and query_plan.task in {"comparison", "trend"} and query_plan.planned_metric_ids
        and (len(parsed_years) > 2 or len(parsed_companies) > 2)
        and parsed.structured_operation not in {"ratio_analysis", "ranking", "contribution_analysis"}
    )
    financial_analysis = bool(
        query_plan
        and query_plan.domain == "financial"
        and (
            query_plan.task in {"analysis", "evaluation", "driver_analysis"}
            or single_company_period_comparison
        )
        and (
            parsed.structured_operation is None or analytical_ratio_request
            or (
                query_plan.task == "evaluation"
                and parsed.structured_operation == "composition"
                and not any(cue in normalize_directive_text(req.query) for cue in (
                    "co cau", "cau thanh", "cau truc", "thanh phan", "ty trong", "liet ke", "phan bo",
                ))
            )
        )
    )
    financial_analysis = financial_analysis or series_request
    if series_request and query_plan is not None and not series_within_budget(
        parsed_companies, parsed_years, query_plan.planned_metric_ids
    ):
        return FinancialAnswerResponse(
            query=req.query, parsed_query=parsed, citations=[], validation=ValidationResult(),
            answer="Phạm vi quá lớn: hãy chọn tối đa 5 doanh nghiệp, 10 kỳ và 100 ô chỉ tiêu.",
            answer_status="needs_clarification",
        )
    if financial_analysis and query_plan is not None and query_plan.next_action == "clarify":
        missing_labels = {
            "company_codes": "mã hoặc tên doanh nghiệp",
            "fiscal_years": "năm tài chính (một hoặc hai năm)",
        }
        missing = [missing_labels.get(slot, slot) for slot in query_plan.missing_slots]
        answer = "Để lập phân tích có căn cứ, vui lòng nêu " + " và ".join(missing or [
            "phạm vi doanh nghiệp, kỳ và nhóm chỉ tiêu cần phân tích"
        ]) + "."
        return FinancialAnswerResponse(
            query=req.query,
            answer=answer,
            parsed_query=parsed,
            citations=[],
            validation=ValidationResult(
                is_valid=True,
                unit_match=True,
                citation_valid=True,
                details="Đang cần làm rõ phạm vi câu hỏi; chưa thực hiện truy xuất.",
            ),
        )

    analytical_years = parsed.fiscal_years or (
        [parsed.fiscal_year] if parsed.fiscal_year is not None else []
    )
    if financial_analysis and not series_request and len(analytical_years) > 2:
        return FinancialAnswerResponse(
            query=req.query,
            answer="MVP hiện hỗ trợ phân tích tối đa hai kỳ trong một lần hỏi. Hãy chọn một năm hoặc hai năm cụ thể.",
            parsed_query=parsed,
            citations=[],
            validation=ValidationResult(
                is_valid=True,
                unit_match=True,
                citation_valid=True,
                details="Câu hỏi vượt quá phạm vi số kỳ được hỗ trợ; chưa thực hiện truy xuất.",
            ),
            answer_status="needs_clarification",
        )

    # Retrieval always chooses the evidence before any deterministic formatter
    # runs. Composition/table requests need more candidates so the answer model
    # can see neighboring rows from the vector-selected source pages.
    operator_pipeline_used = not financial_analysis
    structured_request = parsed.structured_operation in {
        "table", "components", "composition", "ranking", "summary",
        "contribution_analysis", "ratio_analysis",
    } or financial_analysis
    candidate_k = min(
        100,
        max(req.top_k, req.top_k * (12 if structured_request else 4)),
    )
    results_by_metric: dict[str, list[RetrievalResult]] = {}
    metric_definitions: dict[str, Any] = {}
    narrative_results: list[RetrievalResult] = []
    if financial_analysis and query_plan is not None:
        metric_retrieval = _retrieve_series_metrics if series_request else _retrieve_financial_overview_metrics
        results_by_metric, results, metric_definitions = metric_retrieval(
            query_plan, parsed, req.top_k
        )
        narrative_results = [] if series_request else _retrieve_financial_overview_narrative(
            query_plan, parsed, req.top_k
        )
        known_ids = {result.chunk.chunk_id for result in results}
        results.extend(
            result for result in narrative_results
            if result.chunk.chunk_id not in known_ids
        )
        answer_results = results
    else:
        results = _runtime_retriever().search(
            parsed, top_k=candidate_k, directive_plan=directives
        )
        results, _ranking_applied = ValidateOperator.rerank(results, directives)
        results, _filter_applied = FilterOperator.apply(
            results, directives, parsed, classifier=filter_classifier
        )
        if not results and directive_operator_state(directives):
            results = _runtime_retriever().search(parsed, top_k=candidate_k)
        if not results:
            raise HTTPException(
                status_code=404,
                detail="Không tìm thấy thông tin phù hợp trong tài liệu đã nạp.",
            )
        # Preserve the broader vector-selected context for composition questions;
        # the context assembler enforces its own prompt-size budget.
        answer_results = (
            _expand_vector_selected_tables(results, _runtime_chunks(), parsed)
            if structured_request
            else results[:max(req.top_k, 10)]
        )
        if (
            not structured_request
            and parsed.intent.value == "metric_lookup"
        ):
            exact_source_rows = _select_preferred_exact_source_rows(
                req.query, parsed, results, _runtime_chunks()
            )
            if exact_source_rows:
                answer_results = exact_source_rows
    if parsed.structured_operation == "ratio_analysis":
        ratio_input_rows = _retrieve_ratio_input_rows(parsed, req.top_k)
        if ratio_input_rows:
            answer_results = ratio_input_rows
    calculation_evidence: List[CalculationEvidence] = []
    answer_status = None
    analysis_coverage = None
    try:
        if financial_analysis and query_plan is not None:
            # This route formats a deterministic analysis from retrieval results;
            # it does not send these facts to the LLM. Map citations for every
            # retrieved candidate so a prompt-size budget cannot hide valid facts.
            retrieved_citations, _chunk_to_citation = CitationMapper.map_citations(
                answer_results
            )
            analysis_answer = build_series_answer(
                query_plan, parsed, metric_definitions, results_by_metric, retrieved_citations,
            ) if series_request else build_financial_overview_answer(
                query_plan,
                parsed,
                metric_definitions,
                results_by_metric,
                retrieved_citations,
                analytical_years,
                narrative_results,
            )
            answer_text = analysis_answer.answer
            citations = analysis_answer.citations
            source_contexts = analysis_answer.source_contexts
            calculation_evidence = analysis_answer.calculations
            answer_status = analysis_answer.answer_status
            if answer_status:
                analysis_coverage = AnalysisCoverageResponse(
                    missing_components=list(analysis_answer.missing_components),
                )
            composition_results = {} if series_request else _retrieve_overview_composition_rows(
                req.query,
                query_plan,
                parsed,
                results_by_metric,
                analysis_answer.citations,
            )
            if composition_results:
                existing_result_ids = {result.chunk.chunk_id for result in answer_results}
                for rows in composition_results.values():
                    for result in rows:
                        if result.chunk.chunk_id not in existing_result_ids:
                            answer_results.append(result)
                            existing_result_ids.add(result.chunk.chunk_id)
                all_citations, _ = CitationMapper.map_citations(answer_results)
                composition_sections: list[str] = []
                composition_citation_ids: set[str] = set()
                composition_calculations: list[CalculationEvidence] = []
                composition_contexts: list[str] = []
                for metric_id, rows in composition_results.items():
                    metric = metric_definitions.get(metric_id)
                    if metric is None:
                        continue
                    composition_query = parsed.model_copy(update={
                        "original_query": (
                            f"Cơ cấu {metric.canonical_name} của "
                            f"{parsed.company_code or parsed.company_codes[0]} "
                            f"năm {' '.join(map(str, analytical_years))}"
                        ),
                        "primary_metric": metric.canonical_name,
                        "metric_id": metric.metric_id,
                        "metric_aliases": list(metric.aliases),
                        "statement_type": metric.statement_type,
                        "structured_operation": "composition",
                    })
                    section = chat_engine._source_composition_answer(
                        composition_query, rows, all_citations
                    )
                    if not section:
                        continue
                    composition_sections.append(section)
                    used_ids = {
                        citation_id.casefold()
                        for citation_id in re.findall(
                            r"\[(c\d+)\]", section, flags=re.IGNORECASE
                        )
                    }
                    used_citations = [
                        citation for citation in all_citations
                        if citation.citation_id.casefold() in used_ids
                    ]
                    composition_citation_ids.update(
                        citation.citation_id for citation in used_citations
                    )
                    composition_contexts.extend(
                        _runtime_retriever().chunks_by_id[citation.chunk_id].content
                        for citation in used_citations
                        if citation.chunk_id in _runtime_retriever().chunks_by_id
                    )
                    composition_calculations.extend(
                        chat_engine.source_composition_calculations(
                            composition_query, rows, all_citations
                        )
                    )
                if composition_sections:
                    answer_text = "\n\n".join([
                        analysis_answer.answer,
                        *composition_sections,
                    ])
                    citations_by_id = {
                        citation.citation_id: citation
                        for citation in analysis_answer.citations
                    }
                    citations_by_id.update({
                        citation.citation_id: citation
                        for citation in all_citations
                        if citation.citation_id in composition_citation_ids
                    })
                    citations = list(citations_by_id.values())
                    source_contexts = list(dict.fromkeys([
                        *analysis_answer.source_contexts,
                        *composition_contexts,
                    ]))
                    calculation_evidence = [
                        *analysis_answer.calculations,
                        *composition_calculations,
                    ]
        elif parsed.structured_operation == "contribution_analysis":
            contribution_answer = chat_engine.generate_contribution_answer(
                parsed, answer_results, _runtime_retriever().chunks_by_id
            )
            answer_text = contribution_answer.answer
            citations = contribution_answer.citations
            source_contexts = contribution_answer.source_contexts
            calculation_evidence = contribution_answer.calculations
            answer_status = contribution_answer.answer_status
            analysis_coverage = AnalysisCoverageResponse(
                missing_components=list(contribution_answer.missing_components),
                residuals=contribution_answer.residuals or {},
                missing_periods=list(contribution_answer.missing_periods),
            )
        elif parsed.structured_operation == "ratio_analysis":
            ratio_answer = chat_engine.generate_ratio_answer(
                parsed, answer_results, _runtime_retriever().chunks_by_id
            )
            answer_text = ratio_answer.answer
            citations = ratio_answer.citations
            source_contexts = ratio_answer.source_contexts
            calculation_evidence = ratio_answer.calculations
        else:
            answer_text, citations, source_contexts = chat_engine.generate_answer(
                req.query, parsed, answer_results, _runtime_retriever().chunks_by_id
            )
            if parsed.structured_operation in {"composition", "components"}:
                calculation_evidence = chat_engine.source_composition_calculations(
                    parsed, answer_results, citations
                )
    except FinancialGenerationError as exc:
        logger.exception("financial_answer_generation_failed model=%s", chat_engine.model_name)
        raise HTTPException(status_code=502, detail="Financial answer model is unavailable") from exc
    if (
        structured_request
        and not financial_analysis
        and parsed.structured_operation not in {"ratio_analysis", "contribution_analysis"}
    ):
        answer_text = _attach_source_row_citations(
            answer_text, citations, _runtime_retriever().chunks_by_id
        )

    # 4. Numeric & Citation Validation
    validation = _validate_answer(
        answer_text,
        citations,
        source_contexts,
        calculation_evidence=calculation_evidence,
        chunks_by_id=_runtime_retriever().chunks_by_id,
    )

    # Halo-style context fallback: if operator-shaped retrieval produced an
    # answer that fails the existing hard validators, retry with the original
    # hybrid path and no domain operators. Keep the fallback bounded to once.
    if (
        (not validation.is_valid or _answer_is_abstention(answer_text))
        and operator_pipeline_used
        and not financial_analysis
        and parsed.structured_operation != "contribution_analysis"
        and directive_operator_state(directives)
    ):
        fallback_results = _runtime_retriever().search(parsed, top_k=candidate_k)
        fallback_results = (
            _expand_vector_selected_tables(fallback_results, _runtime_chunks(), parsed)
            if structured_request
            else fallback_results[:max(req.top_k, 10)]
        )
        if not structured_request and parsed.intent.value == "metric_lookup":
            exact_source_rows = _select_preferred_exact_source_rows(
                req.query, parsed, fallback_results, _runtime_chunks()
            )
            if exact_source_rows:
                fallback_results = exact_source_rows
        if fallback_results:
            try:
                fallback_answer, fallback_citations, fallback_contexts = chat_engine.generate_answer(
                    req.query, parsed, fallback_results, _runtime_retriever().chunks_by_id
                )
            except FinancialGenerationError:
                logger.exception("financial_answer_fallback_generation_failed model=%s", chat_engine.model_name)
            else:
                if structured_request:
                    fallback_answer = _attach_source_row_citations(
                        fallback_answer,
                        fallback_citations,
                        _runtime_retriever().chunks_by_id,
                    )
                fallback_validation = _validate_answer(
                    fallback_answer, fallback_citations, fallback_contexts
                )
                if fallback_validation.is_valid:
                    answer_text = fallback_answer
                    citations = fallback_citations
                    source_contexts = fallback_contexts
                    validation = fallback_validation
                    logger.info("financial_answer_operator_fallback_succeeded")

    if not validation.is_valid:
        answer_status = "insufficient_evidence"
        analysis_coverage = None
        numeric_valid = validation.is_valid
        logger.warning(
            "financial_answer_rejected model=%s numeric_valid=%s unit_valid=%s citation_valid=%s",
            chat_engine.model_name,
            numeric_valid,
            validation.unit_match,
            validation.citation_valid,
        )
        citation_refs = ", ".join(f"[{citation.citation_id}]" for citation in citations)
        answer_text = "Không thể xác minh câu trả lời từ các nguồn đã truy xuất."
        if citation_refs:
            answer_text += f" Có thể đối chiếu nguồn tại {citation_refs}."
        validation.citation_valid = CitationValidator.validate(answer_text, citations)
        validation.details = (
            "Nội dung sinh không vượt qua kiểm định và đã được thay bằng câu từ chối. "
            + validation.details
        )

    calculation_responses = [
        CalculationEvidenceResponse(
            value=item.value,
            unit=item.unit,
            formula_id=item.formula_id,
            formula_version=item.formula_version,
            expression=FORMULAS[item.formula_id].expression,
            input_chunk_ids=list(item.input_chunk_ids),
            input_citation_ids=list(item.input_citation_ids),
            inputs=[CalculationInputResponse(
                role=fact.role, value=fact.value, unit=fact.unit, year=fact.year,
                label=fact.label, chunk_id=fact.chunk_id, citation_id=fact.citation_id,
            ) for fact in item.inputs],
        )
        for item in calculation_evidence
        if validation.is_valid
        and item.formula_id in FORMULAS
        and item.formula_version == FORMULAS[item.formula_id].version
    ]

    return FinancialAnswerResponse(
        query=req.query,
        answer=answer_text,
        parsed_query=parsed,
        citations=citations,
        validation=validation,
        calculations=calculation_responses,
        answer_status=answer_status,
        analysis_coverage=analysis_coverage,
    )


app.include_router(create_notebook_router(sys.modules[__name__]))
