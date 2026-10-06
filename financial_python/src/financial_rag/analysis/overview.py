from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import re
from typing import Any, Literal

from ..metadata.metric_mapper import MetricDefinition
from ..schemas import Citation, ChunkType, FinancialChunk, ParsedQuery, RetrievalResult
from ..structure.source_groups import (
    display_source_item_label,
    source_item_label,
)
from ..validation.numeric import NumericValidator
from ..query.directives import normalize_directive_text
from .contribution import CalculationEvidence
from .evidence import calculation, source_input
from .ratio import matches_metric_source
from .banking import CREDIT_GROUP_IDS, GROSS_LOANS_ID, MARGIN_LOANS_ID, reconcile_credit_groups, reconcile_gross_loans
from .bank_ratios import build_bank_ratios
from ..structure.semantics import period_supported


@dataclass(frozen=True)
class MetricPeriodFact:
    metric_id: str
    label: str
    year: int
    value: Decimal
    unit: str
    chunk: FinancialChunk
    citation: Citation


@dataclass(frozen=True)
class OverviewAnswer:
    answer: str
    citations: list[Citation]
    source_contexts: list[str]
    calculations: list[CalculationEvidence]
    answer_status: Literal["complete", "partial", "needs_clarification", "insufficient_evidence"] | None = None
    missing_components: tuple[str, ...] = ()


def _period_value(chunk: FinancialChunk, year: int) -> Decimal | None:
    if period_supported(chunk, year) is False:
        return None
    values = chunk.metadata.get("period_values")
    raw: Any = None
    if isinstance(values, dict):
        raw = values.get(str(year), values.get(year))
    if raw in (None, "", "N/A", "-"):
        if chunk.fiscal_year == year:
            raw = chunk.metadata.get("end_value")
        elif chunk.comparative_fiscal_year == year:
            raw = chunk.metadata.get("start_value")
    normalized = NumericValidator.normalize_number(str(raw if raw is not None else ""))
    if not normalized:
        return None
    try:
        value = Decimal(normalized)
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def _format_decimal(value: Decimal, *, places: int | None = None) -> str:
    if places is not None:
        value = value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    text = format(value, "f")
    integer, separator, fraction = text.partition(".")
    sign = "-" if integer.startswith("-") else ""
    digits = integer[1:] if sign else integer
    grouped = f"{int(digits):,}".replace(",", ".")
    if separator:
        fraction = fraction.rstrip("0") if places is None else fraction
        return f"{sign}{grouped},{fraction}" if fraction else f"{sign}{grouped}"
    return f"{sign}{grouped}"


def _has_comparative_source_value(chunk: FinancialChunk, year: int) -> bool:
    """Require the comparative value and its year in the cited source text."""
    value = _period_value(chunk, year)
    if value is None or str(year) not in chunk.content:
        return False
    return any(
        Decimal(normalized) == value
        for raw in NumericValidator.extract_numbers(chunk.content)
        if (normalized := NumericValidator.normalize_number(raw))
    )


def _same_value_signature(fact: MetricPeriodFact) -> tuple[str, str]:
    normalized_unit = NumericValidator._normalize_unit(fact.unit) or ""
    return format(fact.value.normalize(), "f"), normalized_unit


def _same_reporting_basis(before: MetricPeriodFact, after: MetricPeriodFact) -> bool:
    if before.chunk.report_type or after.chunk.report_type:
        return bool(before.chunk.report_type) and before.chunk.report_type == after.chunk.report_type
    return (before.chunk.document_id or before.chunk.source_file) == (
        after.chunk.document_id or after.chunk.source_file
    )


def _select_period_fact(
    metric_id: str,
    metric: MetricDefinition,
    year: int,
    results: list[RetrievalResult],
    citation_by_chunk: dict[str, Citation],
    company_code: str,
    requested_report_type: str | None,
) -> tuple[MetricPeriodFact | None, str | None]:
    candidates: list[tuple[RetrievalResult, MetricPeriodFact]] = []
    for result in results:
        chunk = result.chunk
        if chunk.chunk_type != ChunkType.FINANCIAL_ITEM:
            continue
        if chunk.metadata.get("report_eligible") is False:
            continue
        if (chunk.company_code or "").casefold() != company_code.casefold():
            continue
        if requested_report_type and chunk.report_type != requested_report_type:
            continue
        if metric_id in {"tong_cong_tai_san", "tong_no_phai_tra", "tong_von_chu_so_huu"}:
            table_title = normalize_directive_text(str(chunk.table_title or chunk.metadata.get("table_title") or ""))
            if any(scope in table_title for scope in (
                "thoi gian dao han", "thoi han dao han", "thoi han dinh lai lai suat",
                "bao cao bo phan", "do nhay", "phan tich ky han", "rui ro thanh khoan",
                "rui ro tien te", "rui ro lai suat", "trang thai ngoai te",
            )):
                continue
        label = source_item_label(chunk)
        if not matches_metric_source(chunk, metric_id, company_code, {metric_id: metric}):
            continue
        citation = citation_by_chunk.get(chunk.chunk_id)
        value = _period_value(chunk, year)
        if citation is None or value is None:
            continue
        unit = str(chunk.unit or chunk.metadata.get("unit") or citation.unit or "").strip()
        candidates.append((result, MetricPeriodFact(
            metric_id=metric_id,
            label=label,
            year=year,
            value=value,
            unit=unit,
            chunk=chunk,
            citation=citation,
        )))

    if not candidates:
        return None, None

    # A short alias such as "Vốn" may match both share capital and total
    # equity. Prefer the most specific verified source label before checking
    # for conflicting values; never use an advisory schema ID to decide.
    def label_specificity(fact: MetricPeriodFact) -> int:
        # A footnote marker does not make a maturity-table total more specific.
        label = re.sub(r"\s*\(\d+\)\s*$", "", display_source_item_label(fact.label))
        return len(label.split())

    specificity = max(label_specificity(fact) for _, fact in candidates)
    candidates = [
        pair for pair in candidates
        if label_specificity(pair[1]) == specificity
    ]

    # Prefer the source statement associated with the registered metric and
    # the report whose fiscal year exactly matches the requested period. These
    # are preferences after vector retrieval; they do not create candidates.
    statement_matches = [
        pair for pair in candidates
        if pair[1].chunk.statement_type == metric.statement_type
    ]
    if statement_matches:
        candidates = statement_matches
    from ..retrieval.runtime import request_snapshot
    snapshot = request_snapshot.get()
    if snapshot and snapshot.source_policy == "latest_restated":
        latest = max(pair[1].chunk.fiscal_year or 0 for pair in candidates)
        candidates = [pair for pair in candidates if pair[1].chunk.fiscal_year == latest]
    exact_period = [pair for pair in candidates if pair[1].chunk.fiscal_year == year]
    if exact_period:
        candidates = exact_period

    signatures = {_same_value_signature(fact) for _result, fact in candidates}
    if len(signatures) > 1:
        return None, "có nhiều giá trị nguồn khác nhau cho cùng kỳ"
    result, fact = max(candidates, key=lambda pair: pair[0].rrf_score)
    return fact, None


def _calculation(
    value: Decimal,
    unit: str,
    formula_id: str,
    inputs: tuple[MetricPeriodFact, ...],
) -> CalculationEvidence:
    roles: tuple[str, ...]
    if formula_id in {"absolute_change", "growth_rate"}:
        roles = ("prior_value", "current_value")
    elif formula_id == "percentage_point_change":
        roles = ("prior_numerator", "prior_denominator", "current_numerator", "current_denominator")
    elif formula_id == "balance_sheet_reconciliation":
        roles = ("total_assets", "total_liabilities", "total_equity")
    elif formula_id == "nonperforming_loans_ratio_percent":
        roles = ("standard_loans", "special_mention_loans", "substandard_loans", "doubtful_loans", "loss_loans")
    else:
        roles = ("numerator", "denominator")
    return calculation(value, unit, formula_id, (
        source_input(role, fact.chunk, fact.citation, fact.year, fact.value)
        for role, fact in zip(roles, inputs, strict=True)
    ))


def build_financial_overview_answer(
    plan: Any,
    parsed: ParsedQuery,
    metric_definitions: dict[str, MetricDefinition],
    results_by_metric: dict[str, list[RetrievalResult]],
    citations: list[Citation],
    requested_years: list[int],
    narrative_results: list[RetrievalResult] | None = None,
) -> OverviewAnswer:
    """Build a source-backed descriptive analysis from vector-selected metric rows.

    This MVP reports observed values and registered arithmetic only. It does
    not assign good/bad ratings or infer causes from correlated changes.
    """
    companies = parsed.company_codes or ([parsed.company_code] if parsed.company_code else [])
    citation_by_chunk = {
        citation.chunk_id: citation
        for citation in citations
        if citation.chunk_id
    }
    if len(companies) != 1 or not requested_years:
        return OverviewAnswer(
            "Không đủ bằng chứng để lập phân tích: cần xác định một doanh nghiệp và kỳ tài chính.",
            [], [], [],
        )

    years = list(dict.fromkeys(int(year) for year in requested_years))
    years = sorted(years)
    if len(years) not in {1, 2}:
        return OverviewAnswer(
            "MVP hiện hỗ trợ phân tích tối đa hai kỳ trong một lần hỏi. Hãy chọn một năm hoặc hai năm cụ thể.",
            [], [], [],
        )
    evaluation = plan.task == "evaluation"
    bank_profile = plan.analysis_requirements.get("financial_profile") == "bank"
    implicit_comparison = False
    wants_average_returns = bool(re.search(r"\b(?:roa|roe)\b", parsed.original_query.casefold()))
    if (evaluation or wants_average_returns) and len(years) == 1:
        target = years[0]
        # Read a comparative column from retrieved target-year filings only;
        # do not silently widen the company's report/year retrieval scope.
        implicit_comparison = any(
            result.chunk.fiscal_year == target
            and _period_value(result.chunk, target) is not None
            and _has_comparative_source_value(result.chunk, target - 1)
            for rows in results_by_metric.values() for result in rows
        )
        if implicit_comparison:
            years = [target - 1, target]
    comparison_mode = len(years) == 2
    if bank_profile and any(metric_id in plan.planned_metric_ids for metric_id in CREDIT_GROUP_IDS):
        # Reconciliation uses requested years. Missing comparative credit data
        # will remain missing rather than being borrowed from another table.
        results_by_metric = reconcile_credit_groups(results_by_metric, requested_years)
    if bank_profile and GROSS_LOANS_ID in plan.planned_metric_ids:
        results_by_metric = reconcile_gross_loans(results_by_metric, requested_years)

    facts: dict[str, dict[int, MetricPeriodFact]] = {}
    missing_metrics: list[str] = []
    ambiguous_metrics: list[str] = []
    missing_periods: dict[str, list[int]] = {}
    for metric_id in plan.planned_metric_ids:
        metric = metric_definitions.get(metric_id)
        if metric is None:
            missing_metrics.append(metric_id)
            continue
        selected_metric_facts: dict[int, MetricPeriodFact] = {}
        metric_ambiguous = False
        for year in years:
            metric_results = results_by_metric.get(metric_id, [])
            if implicit_comparison and year == years[0]:
                metric_results = [
                    result for result in metric_results
                    if result.chunk.fiscal_year == years[-1]
                    and _period_value(result.chunk, years[-1]) is not None
                    and _has_comparative_source_value(result.chunk, year)
                ]
            fact, error = _select_period_fact(
                metric_id,
                metric,
                year,
                metric_results,
                citation_by_chunk,
                companies[0],
                parsed.report_type,
            )
            if error:
                metric_ambiguous = True
            elif fact is not None:
                selected_metric_facts[year] = fact
        if metric_ambiguous:
            ambiguous_metrics.append(metric.canonical_name)
            continue
        if selected_metric_facts:
            facts[metric_id] = selected_metric_facts
            absent_years = [year for year in years if year not in selected_metric_facts]
            if absent_years:
                missing_periods[metric.canonical_name] = absent_years
        else:
            if not (bank_profile and metric_id == MARGIN_LOANS_ID):
                missing_metrics.append(metric.canonical_name)

    if not facts:
        refs = " ".join(f"[{citation.citation_id}]" for citation in citations[:8])
        answer = "Không đủ bằng chứng để lập phân tích tài chính từ các dòng đã truy xuất."
        if plan.unsupported_analysis_dimensions:
            answer += (
                " MVP chưa có nhóm chỉ tiêu được đăng ký cho: "
                + ", ".join(plan.unsupported_analysis_dimensions)
                + "."
            )
        if refs:
            answer += f" Có thể đối chiếu nguồn tại {refs}."
        return OverviewAnswer(
            answer, citations[:8], [citation.snippet for citation in citations[:8]], [],
            answer_status="insufficient_evidence",
            missing_components=tuple(missing_metrics),
        )

    calculations: list[CalculationEvidence] = []
    used_facts: dict[str, MetricPeriodFact] = {}
    narrative_citations: list[Citation] = []
    narrative_contexts: list[str] = []
    for result in narrative_results or []:
        chunk = result.chunk
        citation = citation_by_chunk.get(chunk.chunk_id)
        if citation is None or chunk.chunk_type not in {ChunkType.NARRATIVE, ChunkType.PARENT_CHILD}:
            continue
        narrative_citations.append(citation)
        narrative_contexts.append(chunk.content)
        if len(narrative_citations) >= 4:
            break
    rows: list[tuple[str, MetricDefinition, dict[int, MetricPeriodFact], Decimal | None, Decimal | None]] = []
    incompatible_bases: list[str] = []
    for metric_id in plan.planned_metric_ids:
        metric = metric_definitions.get(metric_id)
        metric_facts = facts.get(metric_id)
        if metric is None or not metric_facts:
            continue
        before = metric_facts.get(years[0]) if comparison_mode else None
        after = metric_facts.get(years[-1])
        change: Decimal | None = None
        growth: Decimal | None = None
        if before and after:
            same_unit = (
                NumericValidator._normalize_unit(before.unit)
                and NumericValidator._normalize_unit(before.unit)
                == NumericValidator._normalize_unit(after.unit)
            )
            same_basis = _same_reporting_basis(before, after)
            if not same_basis:
                incompatible_bases.append(metric.canonical_name)
            if same_unit and same_basis:
                change = after.value - before.value
                calculations.append(_calculation(
                    change, before.unit, "absolute_change", (before, after)
                ))
                if before.value > 0:
                    growth = (change * Decimal("100") / before.value).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )
                    calculations.append(_calculation(
                        growth, "%", "growth_rate", (before, after)
                    ))
            used_facts[before.chunk.chunk_id] = before
            used_facts[after.chunk.chunk_id] = after
        else:
            fact = after or before
            if fact:
                used_facts[fact.chunk.chunk_id] = fact
        rows.append((metric_id, metric, metric_facts, change, growth))

    if comparison_mode:
        output_lines = [
            f"Phân tích các chỉ tiêu truy xuất được của {companies[0]} giai đoạn {years[0]}–{years[1]}:",
            "",
            f"| Chỉ tiêu gốc trong BCTC | {years[0]} | {years[1]} | Thay đổi | Thay đổi % | Nguồn |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    else:
        output_lines = [
            f"Phân tích các chỉ tiêu truy xuất được của {companies[0]} năm {years[0]}:",
            "",
            f"| Chỉ tiêu gốc trong BCTC | {years[0]} | Nguồn |",
            "| --- | ---: | --- |",
        ]
    if evaluation:
        output_lines[0] = f"Đánh giá sức khỏe tài chính {companies[0]} năm {years[-1]} từ bằng chứng trong BCTC:"
        if implicit_comparison:
            output_lines[1:1] = [
                "", f"Đối chiếu năm {years[0]} bằng cột so sánh trong báo cáo năm {years[-1]}; giữ nguyên số liệu trình bày lại nếu có.",
            ]
    increases: list[str] = []
    decreases: list[str] = []
    same_values: list[str] = []
    for _metric_id, metric, metric_facts, change, growth in rows:
        before = metric_facts.get(years[0]) if comparison_mode else None
        after = metric_facts.get(years[-1])

        def amount(fact: MetricPeriodFact | None) -> str:
            return f"{_format_decimal(fact.value)} {fact.unit}".strip() if fact else "—"

        source_ids = list(dict.fromkeys(
            fact.citation.citation_id for fact in (before, after) if fact
        ))
        source_text = " ".join(f"[{citation_id}]" for citation_id in source_ids) or "—"
        change_text = "—"
        growth_text = "—"
        if change is not None:
            unit = before.unit if before else after.unit if after else ""
            change_text = f"{_format_decimal(change)} {unit}".strip()
            growth_text = f"{_format_decimal(growth, places=2)}%" if growth is not None else "—"
            change_source_fact = after or before
            source_label = display_source_item_label(change_source_fact.label) if change_source_fact else metric.canonical_name
            if _metric_id in {"ASSET.CUSTOMER_LOANS.ALLOWANCE", "INCOME.BANK.OPERATING_EXPENSE"}:
                pass  # Negative presentation is not a falling provision or expense amount.
            elif change > 0:
                increases.append(f"{source_label} [{source_ids[-1]}]")
            elif change < 0:
                decreases.append(f"{source_label} [{source_ids[-1]}]")
            else:
                same_values.append(f"{source_label} [{source_ids[-1]}]")
        label = (
            display_source_item_label(source_item_label(after.chunk))
            if after
            else display_source_item_label(source_item_label(before.chunk))
            if before
            else metric.canonical_name
        )
        if comparison_mode:
            output_lines.append(
                f"| {label} | {amount(before)} | {amount(after)} | {change_text} | {growth_text} | {source_text} |"
            )
        else:
            output_lines.append(f"| {label} | {amount(after)} | {source_text} |")

    if comparison_mode:
        output_lines.extend(["", "Diễn biến quan sát được:"])
        if increases:
            output_lines.append("- Tăng: " + ", ".join(increases) + ".")
        if decreases:
            output_lines.append("- Giảm: " + ", ".join(decreases) + ".")
        if same_values:
            output_lines.append("- Không đổi: " + ", ".join(same_values) + ".")
        if not any((increases, decreases, same_values)):
            output_lines.append("- Chưa có đủ cặp giá trị cùng đơn vị và cơ sở báo cáo để tính biến động.")

    # Ratios in the MVP are restricted to two registered balance-sheet
    # formulas and require both source rows to come from the same filing.
    ratio_specs = (
        ("liabilities_to_assets_percent", "tong_no_phai_tra", "tong_cong_tai_san", "Nợ phải trả / Tổng tài sản"),
        ("equity_to_assets_percent", "tong_von_chu_so_huu", "tong_cong_tai_san", "Vốn chủ sở hữu / Tổng tài sản"),
    )
    ratio_values: dict[str, dict[int, tuple[Decimal, MetricPeriodFact, MetricPeriodFact]]] = {}
    for formula_id, numerator_id, denominator_id, label in ratio_specs:
        ratio_values[formula_id] = {}
        for year in years:
            numerator = facts.get(numerator_id, {}).get(year)
            denominator = facts.get(denominator_id, {}).get(year)
            if not numerator or not denominator:
                continue
            same_basis = (
                numerator.chunk.company_code == denominator.chunk.company_code
                and numerator.chunk.report_type == denominator.chunk.report_type
                and (numerator.chunk.document_id or numerator.chunk.source_file)
                == (denominator.chunk.document_id or denominator.chunk.source_file)
                and bool(NumericValidator._normalize_unit(numerator.unit))
                and NumericValidator._normalize_unit(numerator.unit)
                == NumericValidator._normalize_unit(denominator.unit)
                and denominator.value > 0
            )
            if not same_basis:
                continue
            value = (numerator.value * Decimal("100") / denominator.value).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            ratio_values[formula_id][year] = (value, numerator, denominator)
            calculations.append(_calculation(
                value, "%", formula_id, (numerator, denominator)
            ))
            used_facts[numerator.chunk.chunk_id] = numerator
            used_facts[denominator.chunk.chunk_id] = denominator

    ratio_change_rows: list[tuple[str, Decimal, tuple[MetricPeriodFact, ...]]] = []
    if comparison_mode:
        for formula_id, _numerator_id, _denominator_id, label in ratio_specs:
            prior = ratio_values[formula_id].get(years[0]) if comparison_mode else None
            current = ratio_values[formula_id].get(years[-1])
            if not prior or not current or not _same_reporting_basis(prior[1], current[1]):
                continue
            point_change = current[0] - prior[0]
            ratio_change_inputs = (prior[1], prior[2], current[1], current[2])
            calculations.append(_calculation(
                point_change, "điểm phần trăm", "percentage_point_change", ratio_change_inputs
            ))
            ratio_change_rows.append((label, point_change, ratio_change_inputs))

    if any(ratio_values.values()):
        if comparison_mode:
            output_lines.extend([
                "",
                f"Tỷ số cấu trúc vốn theo các kỳ {years[0]}–{years[-1]}:",
                "",
                f"| Chỉ số | {years[0]} (%) | {years[1]} (%) | Thay đổi (điểm phần trăm) | Nguồn đầu vào |",
                "| --- | ---: | ---: | ---: | --- |",
            ])
        else:
            output_lines.extend([
                "",
                f"Tỷ số cấu trúc vốn năm {years[0]}:",
                "",
                f"| Chỉ số | {years[0]} (%) | Nguồn đầu vào |",
                "| --- | ---: | --- |",
            ])
        for formula_id, _numerator_id, _denominator_id, label in ratio_specs:
            prior = ratio_values[formula_id].get(years[0])
            current = ratio_values[formula_id].get(years[-1])
            if not prior and not current:
                continue
            prior_text = f"{_format_decimal(prior[0], places=2)}%" if prior else "—"
            current_text = f"{_format_decimal(current[0], places=2)}%" if current else "—"
            display_point_change = (
                current[0] - prior[0]
                if comparison_mode and prior and current and _same_reporting_basis(prior[1], current[1])
                else None
            )
            change_text = f"{_format_decimal(display_point_change, places=2)} điểm phần trăm" if display_point_change is not None else "—"
            ratio_display_inputs: list[MetricPeriodFact] = []
            for ratio_fact in (prior, current):
                if ratio_fact:
                    ratio_display_inputs.extend((ratio_fact[1], ratio_fact[2]))
            unique_inputs = {fact.chunk.chunk_id: fact for fact in ratio_display_inputs}
            refs = " ".join(
                f"[{fact.citation.citation_id}]" for fact in unique_inputs.values()
            )
            if comparison_mode:
                output_lines.append(
                    f"| {label} | {prior_text} | {current_text} | {change_text} | {refs} |"
                )
            else:
                output_lines.append(f"| {label} | {current_text} | {refs} |")
        for label, point_change, inputs in ratio_change_rows:
            refs = " ".join(dict.fromkeys(
                f"[{fact.citation.citation_id}]" for fact in inputs
            ))
            direction = "tăng" if point_change > 0 else "giảm" if point_change < 0 else "không đổi"
            output_lines.extend([
                "",
                f"{label} {direction}; mức thay đổi {_format_decimal(point_change, places=2)} điểm phần trăm; nguồn: {refs}.",
            ])

    additional_missing: list[str] = []
    if bank_profile or wants_average_returns:
        bank_lines, bank_calculations, bank_inputs, additional_missing = build_bank_ratios(
            parsed, facts, years, plan.analysis_dimensions if bank_profile else [], _format_decimal,
        )
        if bank_lines:
            output_lines.extend(["", "Các tỷ số tính từ BCTC:", *bank_lines])
        calculations.extend(bank_calculations)
        for fact in bank_inputs:
            used_facts[fact.chunk.chunk_id] = fact
        if additional_missing:
            output_lines.extend(["", "Chưa đủ dữ liệu tính tỷ số: " + "; ".join(additional_missing) + "."])
    credit_ratios: dict[int, Decimal] = {}
    if bank_profile:
        credit_lines = []
        for year in years:
            classified = [facts.get(metric_id, {}).get(year) for metric_id in CREDIT_GROUP_IDS]
            gross = facts.get(GROSS_LOANS_ID, {}).get(year)
            if not gross or any(fact is None for fact in classified):
                continue
            inputs = tuple(fact for fact in classified if fact is not None)
            margin = facts.get(MARGIN_LOANS_ID, {}).get(year)
            reconciliation_inputs = (*inputs, margin) if margin else inputs
            if (
                gross.value <= 0
                or any(fact.value < 0 for fact in inputs)
                or sum((fact.value for fact in reconciliation_inputs), Decimal(0)) != gross.value
                or any(
                    (fact.chunk.document_id or fact.chunk.source_file)
                    != (gross.chunk.document_id or gross.chunk.source_file)
                    or fact.chunk.report_type != gross.chunk.report_type
                    or NumericValidator._normalize_unit(fact.unit)
                    != NumericValidator._normalize_unit(gross.unit)
                    for fact in reconciliation_inputs
                )
            ):
                continue
            classified_total = sum((fact.value for fact in inputs), Decimal(0))
            if classified_total <= 0:
                continue
            risk_inputs = inputs
            value = (sum((fact.value for fact in inputs[-3:]), Decimal(0)) * 100 / classified_total).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP,
            )
            credit_ratios[year] = value
            calculations.append(_calculation(value, "%", "nonperforming_loans_ratio_percent", risk_inputs))
            calculations.append(calculation(
                Decimal(0), gross.unit, "component_sum_reconciliation",
                [source_input("total_value", gross.chunk, gross.citation, year, gross.value)]
                + [source_input(f"component_{index}", fact.chunk, fact.citation, year, fact.value)
                   for index, fact in enumerate(reconciliation_inputs)],
            ))
            for fact in (*reconciliation_inputs, gross):
                used_facts[fact.chunk.chunk_id] = fact
            refs = " ".join(dict.fromkeys(f"[{fact.citation.citation_id}]" for fact in risk_inputs))
            credit_lines.append(f"| {year} | {_format_decimal(value, places=2)}% | {refs} |")
        if credit_lines:
            output_lines.extend([
                "", "Chất lượng tín dụng từ bảng phân loại cho vay khách hàng:",
                "Tỷ lệ nợ xấu được tính trên dư nợ thuộc năm nhóm phân loại, không gồm khoản cho vay giao dịch ký quỹ trình bày riêng. Tổng các nhóm nợ và khoản ký quỹ (nếu có) đã được đối chiếu với cho vay khách hàng trước dự phòng trong cùng báo cáo. Đây là tỷ lệ tính từ BCTC, không mặc định đồng nhất với tỷ lệ ngân hàng công bố.",
                "", "| Năm | Tỷ lệ nợ xấu trên dư nợ phân loại | Nguồn đầu vào |", "| --- | ---: | --- |",
                *credit_lines,
            ])

    # Reconcile the accounting equation when all three totals are retrieved
    # from one report, period, unit, and document.
    reconciliation_rows: list[tuple[int, Decimal, tuple[MetricPeriodFact, ...]]] = []
    for year in years:
        assets = facts.get("tong_cong_tai_san", {}).get(year)
        liabilities = facts.get("tong_no_phai_tra", {}).get(year)
        equity = facts.get("tong_von_chu_so_huu", {}).get(year)
        if not assets or not liabilities or not equity:
            continue
        inputs = (assets, liabilities, equity)
        same_basis = all(
            fact.chunk.company_code == assets.chunk.company_code
            and fact.chunk.report_type == assets.chunk.report_type
            and (fact.chunk.document_id or fact.chunk.source_file)
            == (assets.chunk.document_id or assets.chunk.source_file)
            and NumericValidator._normalize_unit(fact.unit)
            == NumericValidator._normalize_unit(assets.unit)
            for fact in inputs
        )
        if not same_basis or not NumericValidator._normalize_unit(assets.unit):
            continue
        difference = assets.value - liabilities.value - equity.value
        reconciliation_rows.append((year, difference, inputs))
        calculations.append(_calculation(
            difference, assets.unit, "balance_sheet_reconciliation", inputs
        ))
    if reconciliation_rows:
        output_lines.extend([
            "",
            "Đối chiếu phương trình bảng cân đối kế toán (Tổng tài sản − Nợ phải trả − Vốn chủ sở hữu):",
        ])
        for year, difference, inputs in reconciliation_rows:
            refs = " ".join(dict.fromkeys(
                f"[{fact.citation.citation_id}]" for fact in inputs
            ))
            status = "khớp" if difference == 0 else "còn chênh lệch"
            output_lines.append(
                f"- Năm {year}: {status}, {_format_decimal(difference)} {inputs[0].unit} [nguồn: {refs}]."
            )

    if narrative_citations:
        refs = " ".join(f"[{citation.citation_id}]" for citation in narrative_citations)
        output_lines.extend([
            "",
            "Thuyết minh liên quan được truy xuất để đối chiếu diễn giải: " + refs + ".",
            "Các đoạn này là bằng chứng văn bản bổ trợ; chỉ xem là nguyên nhân khi báo cáo nêu trực tiếp mối quan hệ đó.",
        ])
    elif comparison_mode and (
        "supporting_narrative" in plan.evidence_groups or plan.task == "driver_analysis"
    ):
        output_lines.extend([
            "",
            "Chưa truy xuất được thuyết minh phù hợp; phần trả lời chỉ mô tả biến động số liệu, không kết luận nguyên nhân.",
        ])

    if missing_metrics:
        output_lines.extend(["", "Chưa truy xuất được số liệu cho: " + ", ".join(dict.fromkeys(missing_metrics)) + "."])
    if missing_periods:
        output_lines.extend(["", "Thiếu dữ liệu theo kỳ:"])
        for label, absent_years in missing_periods.items():
            if comparison_mode:
                output_lines.append(
                    f"- Chưa tìm thấy giá trị biến động cho {label} ở năm "
                    f"{', '.join(map(str, absent_years))} trong nguồn đã nạp; "
                    "chưa tính chênh lệch cho cặp kỳ."
                )
            else:
                output_lines.append(
                    f"- Chưa tìm thấy {label} cho năm "
                    f"{', '.join(map(str, absent_years))} trong nguồn đã nạp."
                )
    if ambiguous_metrics:
        output_lines.extend(["", "Không đưa vào phép tính do nguồn có giá trị trùng kỳ nhưng khác nhau: " + ", ".join(dict.fromkeys(ambiguous_metrics)) + "."])
    if incompatible_bases:
        output_lines.extend([
            "", "Không tính biến động do các kỳ không cùng cơ sở báo cáo: "
            + ", ".join(dict.fromkeys(incompatible_bases)) + ".",
        ])
    report_bases = {fact.chunk.report_type for fact in used_facts.values()}
    if len(report_bases) == 1:
        basis_name = {"consolidated": "hợp nhất", "separate": "riêng lẻ"}.get(next(iter(report_bases)) or "")
        if basis_name:
            output_lines[1:1] = ["", f"Cơ sở báo cáo: {basis_name}."]
    if plan.unsupported_analysis_dimensions:
        output_lines.extend([
            "",
            "Chưa có nhóm chỉ tiêu được đăng ký trong MVP cho: "
            + ", ".join(plan.unsupported_analysis_dimensions)
            + ".",
        ])
    missing_coverage = list(dict.fromkeys([
        *missing_metrics, *ambiguous_metrics, *plan.unsupported_analysis_dimensions, *additional_missing,
        *(f"{label} (thiếu kỳ {', '.join(map(str, absent))})" for label, absent in missing_periods.items()),
    ]))
    if evaluation:
        output_lines.extend(["", "Nhận định từ các bằng chứng đã kiểm chứng:"])
        profit = facts.get("loi_nhuan_sau_thue", {}).get(years[-1])
        if profit:
            state = "có lãi" if profit.value > 0 else "bị lỗ" if profit.value < 0 else "hòa vốn"
            output_lines.append(f"- Sinh lời: doanh nghiệp {state} sau thuế [{profit.citation.citation_id}].")
        assessment_metrics = {
            "loi_nhuan_sau_thue": "Lợi nhuận sau thuế",
            "tong_thu_nhap_hoat_dong": "Tổng thu nhập hoạt động",
            "doanh_thu_thuan": "Doanh thu thuần",
            "tong_cong_tai_san": "Tổng tài sản",
            "tong_von_chu_so_huu": "Vốn chủ sở hữu",
            "luu_chuyen_tien_tu_hoat_dong_kinh_doanh": "Dòng tiền thuần từ hoạt động kinh doanh",
            "BANK.NOTE.8.LOSS": "Nợ có khả năng mất vốn",
            "ASSET.CUSTOMER_LOANS.GROSS": "Cho vay khách hàng trước dự phòng",
            "tien_gui_cua_khach_hang": "Tiền gửi khách hàng",
        }
        for metric_id, _metric, metric_facts, change, growth in rows:
            if metric_id not in assessment_metrics or growth is None or change is None:
                continue
            before, after = metric_facts[years[0]], metric_facts[years[-1]]
            direction = "tăng" if change > 0 else "giảm" if change < 0 else "không đổi"
            refs = " ".join(dict.fromkeys(
                f"[{fact.citation.citation_id}]" for fact in (before, after)
            ))
            # The signed rate is retained exactly as in calculation evidence.
            output_lines.append(
                f"- {assessment_metrics[metric_id]} {direction}; biến động {_format_decimal(growth, places=2)}% "
                f"so với năm {years[0]} {refs}."
            )
        growth_rates = {metric_id: growth for metric_id, _, _, _, growth in rows if growth is not None}
        if bank_profile and GROSS_LOANS_ID in growth_rates and "tien_gui_cua_khach_hang" in growth_rates:
            loan_growth = growth_rates[GROSS_LOANS_ID]
            deposit_growth = growth_rates["tien_gui_cua_khach_hang"]
            relation = "nhanh hơn" if loan_growth > deposit_growth else "chậm hơn" if loan_growth < deposit_growth else "bằng"
            refs = " ".join(dict.fromkeys(
                f"[{fact.citation.citation_id}]"
                for metric_id in (GROSS_LOANS_ID, "tien_gui_cua_khach_hang")
                for fact in facts[metric_id].values()
            ))
            output_lines.append(f"- Cho vay khách hàng trước dự phòng tăng trưởng {relation} tiền gửi khách hàng {refs}. Chênh lệch tốc độ tăng trưởng chưa đủ để kết luận thiếu nguồn vốn hoặc mất an toàn thanh khoản.")
        if bank_profile:
            if comparison_mode and years[0] in credit_ratios and years[-1] in credit_ratios:
                direction = (
                    "giảm" if credit_ratios[years[-1]] < credit_ratios[years[0]]
                    else "tăng" if credit_ratios[years[-1]] > credit_ratios[years[0]] else "không đổi"
                )
                refs = " ".join(dict.fromkeys(
                    f"[{facts[metric_id][year].citation.citation_id}]"
                    for year in years for metric_id in CREDIT_GROUP_IDS
                ))
                output_lines.append(
                    f"- Chất lượng tín dụng: tỷ lệ nợ xấu tính từ BCTC {direction}, "
                    f"từ {_format_decimal(credit_ratios[years[0]], places=2)}% "
                    f"đến {_format_decimal(credit_ratios[years[-1]], places=2)}%; "
                    f"cần xem đồng thời biến động nợ có khả năng mất vốn và dự phòng {refs}."
                )
                loss_before = facts.get("BANK.NOTE.8.LOSS", {}).get(years[0])
                loss_after = facts.get("BANK.NOTE.8.LOSS", {}).get(years[-1])
                if direction == "giảm" and loss_before and loss_after and loss_after.value > loss_before.value:
                    output_lines.append(
                        "- Điểm cần theo dõi: tỷ lệ nợ xấu trên dư nợ phân loại giảm nhưng số dư nợ có khả năng mất vốn tăng; "
                        f"chưa thể kết luận chất lượng tín dụng cải thiện đồng đều {refs}."
                    )
            equity_ratios = ratio_values.get("equity_to_assets_percent", {})
            if comparison_mode and years[0] in equity_ratios and years[-1] in equity_ratios:
                prior_ratio, current_ratio = equity_ratios[years[0]], equity_ratios[years[-1]]
                if current_ratio[0] < prior_ratio[0]:
                    refs = " ".join(dict.fromkeys(
                        f"[{fact.citation.citation_id}]" for fact in (*prior_ratio[1:], *current_ratio[1:])
                    ))
                    output_lines.append(
                        "- Điểm cần theo dõi về vốn: tỷ trọng vốn chủ sở hữu trong tổng tài sản giảm; "
                        f"mức đệm vốn kế toán trên tài sản thấp hơn kỳ so sánh {refs}."
                    )
            if "asset_quality" in plan.analysis_dimensions and years[-1] not in credit_ratios:
                missing_coverage.append("bảng phân loại nợ đối chiếu được với dư nợ cho vay khách hàng")
            if "leverage_and_capital" in plan.analysis_dimensions:
                output_lines.append("- Vốn: tỷ lệ vốn chủ sở hữu trên tài sản là tỷ số kế toán, không thay thế hệ số an toàn vốn CAR.")
                missing_coverage.append("CAR được công bố cùng phạm vi báo cáo")
            if "liquidity" in plan.analysis_dimensions:
                output_lines.append("- Thanh khoản: số dư tiền và dòng tiền kinh doanh chỉ là bằng chứng bổ trợ; cần phân tích kỳ hạn tài sản và nguồn vốn trước khi kết luận khả năng đáp ứng nghĩa vụ.")
                missing_coverage.append("đánh giá thanh khoản theo kỳ hạn")
        if not comparison_mode:
            missing_coverage.append("kỳ so sánh để đánh giá diễn biến")
        output_lines.extend([
            "", "Đây là đánh giá có giới hạn theo các chỉ tiêu và biến động đã dẫn nguồn. "
            "Chưa đủ cơ sở để xếp hạng sức khỏe tài chính tổng thể hoặc kết luận nguyên nhân.",
        ])
        if missing_coverage:
            output_lines.append("Phần đánh giá chưa đủ bằng chứng: " + "; ".join(dict.fromkeys(missing_coverage)) + ".")
    else:
        output_lines.extend([
            "",
            "Các kết luận trên mô tả số liệu và phép tính đã dẫn nguồn; chưa đánh giá tốt/xấu "
            "và không suy diễn nguyên nhân khi báo cáo không nêu bằng chứng.",
        ])

    used_citation_ids = {
        fact.citation.citation_id for fact in used_facts.values()
    }
    used_citation_ids.update(citation.citation_id for citation in narrative_citations)
    used_citations = [citation for citation in citations if citation.citation_id in used_citation_ids]
    source_contexts = list(dict.fromkeys(
        [fact.chunk.content for fact in used_facts.values()] + narrative_contexts
    ))
    return OverviewAnswer(
        "\n".join(output_lines), used_citations, source_contexts, calculations,
        answer_status="partial" if evaluation and missing_coverage else "complete" if evaluation else None,
        missing_components=tuple(dict.fromkeys(missing_coverage)),
    )
