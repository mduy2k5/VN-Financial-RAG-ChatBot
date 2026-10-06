from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..query.understanding import QueryUnderstandingEngine
from ..schemas import FinancialChunk, ParsedQuery, PeriodType, StatementType
from ..structure.source_groups import display_source_item_label, source_item_label, source_label_matches_query
from .catalog import GuideCatalog, GuideDefinition, load_guide_catalog
from .matcher import GuideMatch, GuideMatcher, RuleBasedGuideMatcher


class ResolvedSlot(BaseModel):
    value: Any
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_basis: str
    provenance: str


class QueryPlan(BaseModel):
    original_question: str
    normalized_question: str
    catalog_version: int
    domain: str | None = None
    task: str | None = None
    guide_id: str | None = None
    guide_score: float = Field(ge=0.0, le=1.0)
    guide_match_method: str
    guide_match_reason: str
    slots: dict[str, ResolvedSlot] = Field(default_factory=dict)
    hard_filters: dict[str, Any] = Field(default_factory=dict)
    analysis_dimensions: list[str] = Field(default_factory=list)
    metric_groups_by_dimension: dict[str, list[str]] = Field(default_factory=dict)
    planned_metric_ids: list[str] = Field(default_factory=list)
    unsupported_analysis_dimensions: list[str] = Field(default_factory=list)
    analysis_requirements: dict[str, Any] = Field(default_factory=dict)
    clarification_reasons: list[str] = Field(default_factory=list)
    evidence_groups: list[str] = Field(default_factory=list)
    retrieval_policy: dict[str, Any] = Field(default_factory=dict)
    evidence_policy: dict[str, Any] = Field(default_factory=dict)
    answer_format: str = "grounded_response"
    missing_slots: list[str] = Field(default_factory=list)
    next_action: str
    source_policy: str = "source_report"
    execution_budget: dict[str, int] = Field(default_factory=lambda: {
        "companies": 5, "periods": 10, "cells": 100, "candidate_chunks": 30000})


class QueryPlanError(ValueError):
    pass


def _hard_filters(parsed: ParsedQuery, access_filter: dict[str, Any]) -> dict[str, Any]:
    filters: dict[str, Any] = {"access": access_filter.copy()}
    companies = parsed.company_codes or ([parsed.company_code] if parsed.company_code else [])
    years = parsed.fiscal_years or ([parsed.fiscal_year] if parsed.fiscal_year is not None else [])
    if companies:
        filters["company_codes"] = list(companies)
    if years:
        filters["fiscal_years"] = list(years)
    if parsed.fiscal_quarter is not None:
        filters["fiscal_quarter"] = parsed.fiscal_quarter
    if parsed.report_type is not None:
        filters["report_type"] = parsed.report_type
    if parsed.period_type != PeriodType.UNSPECIFIED:
        filters["period_type"] = parsed.period_type.value
    return filters


def _remove_domain_acronyms_from_company_slots(
    parsed: ParsedQuery,
    catalog: GuideCatalog,
    engine: QueryUnderstandingEngine,
) -> ParsedQuery:
    """Avoid treating three-letter guide terms such as ESG as company tickers."""
    domain_acronyms = {
        keyword.upper()
        for keywords in catalog.domains.values()
        for keyword in keywords
        if re.fullmatch(r"[a-zA-Z]{3}", keyword.strip())
    }
    known_company_codes = {code.upper() for code in engine.company_extractor.aliases}
    companies = [
        code
        for code in (parsed.company_codes or ([parsed.company_code] if parsed.company_code else []))
        if code.upper() not in domain_acronyms or code.upper() in known_company_codes
    ]
    if companies == parsed.company_codes and (companies[0] if companies else None) == parsed.company_code:
        return parsed
    return parsed.model_copy(
        update={"company_codes": companies, "company_code": companies[0] if companies else None}
    )


def _slots(parsed: ParsedQuery) -> dict[str, ResolvedSlot]:
    values: dict[str, Any] = {
        "company_codes": parsed.company_codes or ([parsed.company_code] if parsed.company_code else None),
        "fiscal_years": parsed.fiscal_years or ([parsed.fiscal_year] if parsed.fiscal_year is not None else None),
        "fiscal_quarter": parsed.fiscal_quarter,
        "report_type": parsed.report_type,
        "metric_id": parsed.metric_id,
        "topic": parsed.primary_metric,
        "statement_type": parsed.statement_type.value if parsed.statement_type else None,
        "period_type": (
            parsed.period_type.value
            if parsed.period_type != PeriodType.UNSPECIFIED
            else None
        ),
    }
    return {
        name: ResolvedSlot(
            value=value,
            confidence=0.85,
            confidence_basis="heuristic_not_calibrated",
            provenance="existing_financial_query_parser",
        )
        for name, value in values.items()
        if value is not None
    }


def _missing_slots(guide: GuideDefinition | None, slots: dict[str, ResolvedSlot]) -> list[str]:
    if guide is None:
        return []
    return [slot for slot in guide.required_slots if slot not in slots]


def _analysis_details(
    question: str,
    parsed: ParsedQuery,
    guide: GuideDefinition | None,
    catalog: GuideCatalog,
    registered_metric_ids: set[str],
    schema_metric_ids: set[str],
    financial_profile: str | None = None,
) -> tuple[list[str], dict[str, list[str]], list[str], dict[str, Any]]:
    supported_tasks = {"analysis", "evaluation", "driver_analysis", "comparison", "trend"}
    if guide is None or guide.task not in supported_tasks:
        return [], {}, [], {}

    normalized = question.casefold()
    dimensions = catalog.analysis_dimensions.get(guide.domain, {})
    period_task = guide.task in {"comparison", "trend"}
    selected = [] if period_task else [
        dimension
        for dimension, keywords in dimensions.items()
        if any(keyword.casefold() in normalized for keyword in keywords)
    ]
    policy = guide.retrieval_policy
    default_dimensions = not selected and not period_task
    if default_dimensions:
        selected = list(policy.get("default_dimensions", []))
        if financial_profile == "bank" and guide.task == "evaluation":
            selected.append("asset_quality")

    metric_groups: dict[str, list[str]] = {}
    unsupported_dimensions: list[str] = []
    if guide.domain == "financial":
        registered_groups = catalog.analysis_metric_groups.get("financial", {})
        if financial_profile in catalog.analysis_metric_profiles:
            registered_groups = catalog.analysis_metric_profiles[financial_profile]
        for dimension in selected:
            metric_ids = list(registered_groups.get(dimension, []))
            if metric_ids:
                metric_groups[dimension] = metric_ids
            else:
                unsupported_dimensions.append(dimension)
        # Period comparisons retrieve only the explicitly named metric. A
        # source-schema line may identify the exact requested row when it has
        # no entry in the general metric registry; the source label/value still
        # has to validate against retrieved evidence downstream.
        if parsed.metric_id and parsed.metric_id in registered_metric_ids:
            metric_groups["requested_metric"] = [parsed.metric_id]
        elif period_task and parsed.metric_id in schema_metric_ids:
            metric_groups["requested_metric"] = [parsed.metric_id]
        if not period_task:
            average_inputs = []
            if re.search(r"\broa\b", normalized):
                average_inputs.extend(["loi_nhuan_sau_thue", "tong_cong_tai_san"])
            if re.search(r"\broe\b", normalized):
                average_inputs.extend(["loi_nhuan_sau_thue", "tong_von_chu_so_huu"])
            if average_inputs:
                metric_groups["average_return_inputs"] = list(dict.fromkeys(average_inputs))

    requested_years = parsed.fiscal_years or (
        [parsed.fiscal_year] if parsed.fiscal_year is not None else []
    )
    analytical_operations = []
    if len(requested_years) > 1:
        analytical_operations.append("period_comparison")
    if any(term in normalized for term in ("xu hướng", "qua các năm", "qua các kỳ", "theo thời gian")):
        analytical_operations.append("trend_analysis")
    if guide.task == "driver_analysis":
        analytical_operations.append("driver_analysis")

    max_dimensions = policy.get("max_dimension_groups")
    if financial_profile == "bank" and default_dimensions and max_dimensions:
        max_dimensions = max(max_dimensions, len(selected))
    return selected, metric_groups, unsupported_dimensions, {
        "financial_profile": financial_profile,
        "use_report_comparative_column": guide.task == "evaluation",
        "max_dimension_groups": max_dimensions,
        "dimensions_exceed_limit": bool(max_dimensions and len(selected) > max_dimensions),
        "operations": list(dict.fromkeys(analytical_operations)),
        "period_basis": requested_years or None,
        "metric_ids_grouped_by_dimension": bool(metric_groups),
        "unsupported_dimensions_must_be_reported": bool(unsupported_dimensions),
        "derive_only_registered_formulas": policy.get("derive_only_registered_formulas", True),
        "require_same_company_and_period_for_formula_inputs": True,
        "require_cited_inputs_for_derived_values": True,
        "require_disclosed_benchmark_before_normative_rating": True,
        "do_not_infer_unreported_drivers": True,
    }


def _financial_profile(
    parsed: ParsedQuery, source_chunks: list[FinancialChunk],
) -> str | None:
    """Resolve the reporting profile from the requested issuer's statements."""
    companies = parsed.company_codes or ([parsed.company_code] if parsed.company_code else [])
    if len(companies) != 1:
        return None
    years = set(parsed.fiscal_years or ([parsed.fiscal_year] if parsed.fiscal_year else []))
    profiles: set[str] = set()
    for chunk in source_chunks:
        if (
            (chunk.company_code or "").casefold() != companies[0].casefold()
            or (parsed.report_type and chunk.report_type != parsed.report_type)
            or (years and chunk.fiscal_year not in years)
            or chunk.metadata.get("report_eligible") is False
            or chunk.statement_type != StatementType.INCOME_STATEMENT
        ):
            continue
        node = str(chunk.metadata.get("schema_node_id") or "")
        # EPS/tax rows can map to another sector's shared aliases. Resolve
        # only from sector-specific operating rows and their source labels.
        signatures = {
            "INCOME.BANK.NET_INTEREST_INCOME": ("bank", "Thu nhập lãi thuần"),
            "INCOME.BANK.TOTAL_OPERATING_INCOME": ("bank", "Tổng thu nhập hoạt động"),
            "INCOME.SECURITIES.TOTAL_OPERATING_REVENUE": ("securities", "Tổng doanh thu hoạt động"),
            "INCOME.ENTERPRISE.REVENUE_NET": ("enterprise", "Doanh thu thuần"),
        }
        if node in signatures:
            profile, label = signatures[node]
            if source_label_matches_query(label, display_source_item_label(source_item_label(chunk))):
                profiles.add(profile)
    return next(iter(profiles)) if len(profiles) == 1 else None


def build_query_plan(
    question: str,
    *,
    access_filter: dict[str, Any],
    parsed_query: ParsedQuery | None = None,
    matcher: GuideMatcher | None = None,
    engine: QueryUnderstandingEngine | None = None,
    catalog: GuideCatalog | None = None,
    catalog_path: Path | None = None,
    source_chunks: list[FinancialChunk] | None = None,
) -> QueryPlan:
    """Build a plan without executing retrieval or contacting any data store."""
    if not question or not question.strip():
        raise QueryPlanError("question must not be empty")
    if not isinstance(access_filter, dict) or not access_filter:
        raise QueryPlanError("a non-empty caller-authorized access_filter is required")

    active_catalog = catalog or load_guide_catalog(catalog_path)
    query_engine = engine or QueryUnderstandingEngine()
    if parsed_query is not None and parsed_query.original_query != question:
        raise QueryPlanError("parsed_query must belong to the supplied question")
    parsed = parsed_query.model_copy(deep=True) if parsed_query is not None else query_engine.parse(question)
    parsed = _remove_domain_acronyms_from_company_slots(parsed, active_catalog, query_engine)
    financial_profile = _financial_profile(parsed, source_chunks or [])
    active_matcher = matcher or RuleBasedGuideMatcher(active_catalog)
    match: GuideMatch = active_matcher.match(question, parsed=parsed)
    guide = active_catalog.by_id().get(match.guide_id or "") if match.accepted else None
    slots = _slots(parsed)
    missing_slots = _missing_slots(guide, slots)
    (
        analysis_dimensions,
        metric_groups_by_dimension,
        unsupported_analysis_dimensions,
        analysis_requirements,
    ) = _analysis_details(
        question,
        parsed,
        guide,
        active_catalog,
        {metric.metric_id for metric in query_engine.metric_mapper.metrics_list},
        {
            node.node_id
            for node in query_engine.schema_catalog.nodes.values()
            if node.kind in {"line", "total", "group", "note_row"}
        },
        financial_profile,
    )
    clarification_reasons = [f"missing required slot: {slot}" for slot in missing_slots]
    if analysis_requirements.get("dimensions_exceed_limit"):
        clarification_reasons.append("requested analysis spans more dimensions than this guide supports")
    next_action = (
        "clarify"
        if clarification_reasons
        else "retrieve"
        if guide is not None
        else "general_fallback"
    )
    plan = QueryPlan(
        original_question=question,
        normalized_question=" ".join(question.split()),
        catalog_version=active_catalog.version,
        domain=match.domain,
        task=match.task,
        guide_id=guide.guide_id if guide else None,
        guide_score=match.score,
        guide_match_method=match.method,
        guide_match_reason=match.reason,
        slots=slots,
        hard_filters=_hard_filters(parsed, access_filter),
        analysis_dimensions=analysis_dimensions,
        metric_groups_by_dimension=metric_groups_by_dimension,
        planned_metric_ids=list(dict.fromkeys(
            metric_id
            for group in metric_groups_by_dimension.values()
            for metric_id in group
        )),
        unsupported_analysis_dimensions=unsupported_analysis_dimensions,
        analysis_requirements=analysis_requirements,
        clarification_reasons=clarification_reasons,
        evidence_groups=guide.evidence_groups if guide else ["relevant_source"],
        retrieval_policy=guide.retrieval_policy if guide else {},
        evidence_policy=guide.evidence_policy if guide else {"require_citations": True},
        answer_format=guide.answer_format if guide else "grounded_response",
        missing_slots=missing_slots,
        next_action=next_action,
        source_policy=parsed.source_policy,
    )
    validate_query_scope(question, plan, access_filter=access_filter, parsed=parsed)
    return plan


def validate_query_scope(
    question: str,
    plan: QueryPlan,
    *,
    access_filter: dict[str, Any],
    parsed: ParsedQuery | None = None,
    engine: QueryUnderstandingEngine | None = None,
) -> None:
    """Fail closed if access or any parser-resolved scope was omitted or broadened."""
    if not access_filter or plan.hard_filters.get("access") != access_filter:
        raise QueryPlanError("hard access filter is missing or changed")
    active_parsed = parsed or (engine or QueryUnderstandingEngine()).parse(question)
    active_engine = engine or QueryUnderstandingEngine()
    if parsed is None:
        active_parsed = _remove_domain_acronyms_from_company_slots(
            active_parsed,
            load_guide_catalog(),
            active_engine,
        )
    expected = _hard_filters(active_parsed, access_filter)
    for name, value in expected.items():
        if plan.hard_filters.get(name) != value:
            raise QueryPlanError(f"hard filter {name!r} is missing or changed")
    for name in ("company_codes", "fiscal_years", "fiscal_quarter", "report_type", "period_type"):
        if name in plan.hard_filters and name not in expected:
            raise QueryPlanError(f"plan silently expanded scope with {name!r}")
