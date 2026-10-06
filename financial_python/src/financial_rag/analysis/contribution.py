from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import re
from typing import Iterable, Literal

from ..schemas import Citation, ChunkType, FinancialChunk, ParsedQuery, RetrievalResult
from ..structure.report_schema_catalog import ReportSchemaCatalog
from ..structure.source_groups import (
    analysis_target_phrase, is_direct_code_child, is_total_source_label,
    root_source_segment, source_item_label, source_phrase_tokens, source_row_position,
)
from ..structure.source_rows import SourceCompositionRows
from ..validation.numeric import NumericValidator
from .evidence import CalculationEvidence, CalculationInput, calculation


@dataclass(frozen=True)
class ContributionAnswer:
    answer: str
    citations: list[Citation]
    source_contexts: list[str]
    calculations: list[CalculationEvidence]
    answer_status: Literal[
        "complete", "partial", "needs_clarification", "insufficient_evidence"
    ] = "insufficient_evidence"
    missing_components: tuple[str, ...] = ()
    residuals: dict[int, str] | None = None
    missing_periods: tuple[int, ...] = ()


@dataclass(frozen=True)
class PeriodAmount:
    company_code: str
    report_type: str
    year: int
    value: Decimal
    unit: str
    label: str
    chunk_id: str
    citation_id: str
    document_id: str = ""


@dataclass(frozen=True)
class ContributionLine:
    label: str
    before: PeriodAmount
    after: PeriodAmount
    change: Decimal
    contribution_percent: Decimal | None


@dataclass(frozen=True)
class IncompleteLine:
    label: str
    before: PeriodAmount | None
    after: PeriodAmount | None
    chunk_id: str


@dataclass(frozen=True)
class ContributionComputation:
    total_before: PeriodAmount
    total_after: PeriodAmount
    lines: list[ContributionLine]
    residual_before: Decimal
    residual_after: Decimal
    incomplete_lines: tuple[IncompleteLine, ...] = ()

    @property
    def total_change(self) -> Decimal:
        return self.total_after.value - self.total_before.value

    @property
    def complete(self) -> bool:
        return bool(self.lines) and not self.incomplete_lines and self.residual_before == 0 and self.residual_after == 0


class AnalysisInputError(ValueError):
    pass


def _period_value(chunk: FinancialChunk, year: int) -> object | None:
    values = chunk.metadata.get("period_values")
    if isinstance(values, dict):
        value = values.get(str(year), values.get(year))
        if value not in (None, "", "N/A", "-"):
            return value
    if chunk.fiscal_year == year:
        value = chunk.metadata.get("end_value")
    elif chunk.comparative_fiscal_year == year:
        value = chunk.metadata.get("start_value")
    else:
        value = None
    return value if value not in (None, "", "N/A", "-") else None


def _decimal(value: object | None) -> Decimal | None:
    if value is None:
        return None
    normalized = NumericValidator.normalize_number(str(value))
    if not normalized:
        return None
    try:
        parsed = Decimal(normalized)
    except InvalidOperation:
        return None
    return parsed if parsed.is_finite() else None


def _number_text(value: Decimal) -> str:
    normalized = format(value, "f")
    integer, separator, fraction = normalized.partition(".")
    integer = f"{int(integer):,}".replace(",", ".")
    return f"{integer},{fraction}" if separator else integer


def _normalized_table_title(chunk: FinancialChunk) -> str:
    title = str(chunk.table_title or chunk.metadata.get("table_title") or "")
    title = re.sub(r"\s*[（(]?\s*(?:tiếp theo|continued)\s*[）)]?\s*$", "", title, flags=re.I)
    return " ".join(title.casefold().split())


def _scope_key(chunk: FinancialChunk) -> tuple[str, str, str, str, int, str]:
    return (
        (chunk.company_code or "").casefold(),
        chunk.document_id or chunk.source_file,
        chunk.report_type or "",
        chunk.statement_type.value,
        int(chunk.page or 0),
        str(chunk.metadata.get("table_id") or _normalized_table_title(chunk)),
    )


def _target_signatures(parsed: ParsedQuery, company_codes: list[str]) -> set[tuple[str, ...]]:
    labels = [
        analysis_target_phrase(parsed.original_query),
        parsed.primary_metric, parsed.canonical_metric, parsed.schema_node_name,
    ]
    signatures = {
        tuple(source_phrase_tokens(str(label), company_codes))
        for label in labels
        if label and source_phrase_tokens(str(label), company_codes)
    }
    return signatures






def _amount(
    chunk: FinancialChunk,
    citation: Citation,
    year: int,
) -> PeriodAmount:
    value = _decimal(_period_value(chunk, year))
    if value is None:
        raise AnalysisInputError(f"thiếu giá trị nguồn cho năm {year}")
    unit = str(chunk.unit or chunk.metadata.get("unit") or citation.unit or "").strip()
    if not NumericValidator._normalize_unit(unit):
        raise AnalysisInputError("nguồn thiếu đơn vị cho một hoặc nhiều dòng")
    return PeriodAmount(
        company_code=str(chunk.company_code or ""),
        report_type=str(chunk.report_type or ""),
        year=year,
        value=value,
        unit=unit,
        label=source_item_label(chunk),
        chunk_id=chunk.chunk_id,
        citation_id=citation.citation_id,
        document_id=chunk.document_id or chunk.source_file,
    )


def calculate_contributions(
    total_before: PeriodAmount,
    total_after: PeriodAmount,
    component_pairs: Iterable[tuple[PeriodAmount, PeriodAmount]],
) -> ContributionComputation:
    """Keep valid paired rows and report residuals; never fabricate missing amounts."""
    if total_before.year >= total_after.year:
        raise AnalysisInputError("kỳ gốc phải đứng trước kỳ hiện tại")
    if total_before.company_code.casefold() != total_after.company_code.casefold():
        raise AnalysisInputError("hai giá trị tổng thuộc hai công ty khác nhau")
    if total_before.report_type != total_after.report_type:
        raise AnalysisInputError("hai giá trị tổng thuộc hai cơ sở báo cáo khác nhau")
    if not total_before.report_type and total_before.document_id != total_after.document_id:
        raise AnalysisInputError("nguồn chưa xác định cơ sở báo cáo và thuộc hai tài liệu khác nhau")
    total_unit = NumericValidator._normalize_unit(total_before.unit)
    if not total_unit or total_unit != NumericValidator._normalize_unit(total_after.unit):
        raise AnalysisInputError("hai giá trị tổng không cùng đơn vị")

    pairs = list(component_pairs)
    if any(not fact.value.is_finite() for fact in (
        total_before, total_after, *(fact for pair in pairs for fact in pair),
    )):
        raise AnalysisInputError("giá trị nguồn không phải số hữu hạn")
    if len({before.label.casefold() for before, _after in pairs}) != len(pairs):
        raise AnalysisInputError("nhãn dòng thành phần bị trùng, không thể đối chiếu duy nhất")

    before_sum = Decimal(0)
    after_sum = Decimal(0)
    for before, after in pairs:
        if before.label.casefold() != after.label.casefold():
            raise AnalysisInputError("hai kỳ thuộc hai dòng thành phần khác nhau")
        if before.year != total_before.year or after.year != total_after.year:
            raise AnalysisInputError("các dòng thành phần không cùng cặp kỳ")
        if before.company_code.casefold() != total_before.company_code.casefold() or after.company_code.casefold() != total_after.company_code.casefold():
            raise AnalysisInputError("các dòng thành phần không cùng công ty với tổng")
        if before.report_type != total_before.report_type or after.report_type != total_after.report_type:
            raise AnalysisInputError("các dòng thành phần không cùng cơ sở báo cáo với tổng")
        if not total_before.report_type and (
            before.document_id != total_before.document_id or after.document_id != total_after.document_id
        ):
            raise AnalysisInputError("thành phần chưa xác định cơ sở báo cáo và thuộc tài liệu khác tổng")
        if NumericValidator._normalize_unit(before.unit) != total_unit or NumericValidator._normalize_unit(after.unit) != total_unit:
            raise AnalysisInputError("các dòng thành phần không cùng đơn vị với tổng")
        before_sum += before.value
        after_sum += after.value

    total_change = total_after.value - total_before.value

    lines = []
    for before, after in pairs:
        change = after.value - before.value
        contribution = (change * Decimal("100") / total_change).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        ) if total_change else None
        lines.append(ContributionLine(
            label=before.label,
            before=before,
            after=after,
            change=change,
            contribution_percent=contribution,
        ))
    lines.sort(key=lambda line: (
        line.contribution_percent if line.contribution_percent is not None else abs(line.change),
        line.change,
    ), reverse=True)
    return ContributionComputation(
        total_before, total_after, lines,
        total_before.value - before_sum, total_after.value - after_sum,
    )


def _analysis_years(parsed: ParsedQuery) -> tuple[int, int]:
    years = list(dict.fromkeys(int(year) for year in parsed.fiscal_years))
    if not years and parsed.fiscal_year is not None:
        years = [int(parsed.fiscal_year)]
    if len(years) == 1:
        # A named year plus a request for contribution to its increase/decrease
        # uses the directly preceding comparison period, never the latest indexed year.
        return years[0] - 1, years[0]
    if len(years) != 2:
        raise AnalysisInputError("cần nêu một năm cụ thể hoặc đúng hai năm để đối chiếu")
    return min(years), max(years)


def _build_analysis(
    parsed: ParsedQuery,
    results: list[RetrievalResult],
    citation_by_chunk: dict[str, Citation],
) -> tuple[str, ContributionComputation, list[str]]:
    companies = parsed.company_codes or ([parsed.company_code] if parsed.company_code else [])
    if len(companies) != 1:
        raise AnalysisInputError("cần xác định duy nhất một doanh nghiệp")
    company = companies[0]
    before_year, after_year = _analysis_years(parsed)
    target_signatures = _target_signatures(parsed, companies)
    if not target_signatures:
        raise AnalysisInputError("chưa xác định được chỉ tiêu tổng cần phân tích")

    groups: dict[tuple[str, str, str, str, int, str], list[FinancialChunk]] = {}
    result_score: dict[str, float] = {}
    for result in results:
        chunk = result.chunk
        if chunk.chunk_type != ChunkType.FINANCIAL_ITEM or chunk.chunk_id not in citation_by_chunk:
            continue
        if (chunk.company_code or "").casefold() != company.casefold():
            continue
        if parsed.statement_type and chunk.statement_type != parsed.statement_type:
            continue
        if parsed.report_type and chunk.report_type != parsed.report_type:
            continue
        if chunk.metadata.get("report_eligible") is False:
            continue
        groups.setdefault(_scope_key(chunk), []).append(chunk)
        result_score[chunk.chunk_id] = max(result_score.get(chunk.chunk_id, 0.0), result.rrf_score)

    candidates = []
    last_error = "không tìm thấy dòng tổng có đủ hai kỳ trong cùng bảng"
    for scoped_rows in groups.values():
        anchors = [
            chunk for chunk in scoped_rows
            if tuple(source_phrase_tokens(source_item_label(chunk), companies)) in target_signatures
            and _period_value(chunk, before_year) is not None
            and _period_value(chunk, after_year) is not None
        ]
        for anchor in anchors:
            try:
                label = source_item_label(anchor)
                source_rows = root_source_segment(anchor, scoped_rows) if is_total_source_label(source_item_label(anchor)) else []
                if not source_rows:
                    label_signature = source_phrase_tokens(label, companies)
                    source_rows = [
                        row for row in scoped_rows
                        if row.chunk_id == anchor.chunk_id
                        or source_phrase_tokens(str(row.metadata.get("section_group") or ""), companies) == label_signature
                        or is_direct_code_child(anchor.metadata.get("code"), row.metadata.get("code"))
                    ]
                row_records = [
                    {
                        "chunk": row, "label": source_item_label(row),
                        "values": {year: _period_value(row, year) for year in (before_year, after_year)},
                        "unit": row.unit or citation_by_chunk[row.chunk_id].unit or "",
                        "citation": citation_by_chunk[row.chunk_id],
                    }
                    for row in sorted(source_rows, key=source_row_position)
                ]
                total_record = next((row for row in row_records if row["chunk"] is anchor), None)
                component_records = SourceCompositionRows._non_overlapping_composition_rows(row_records, total_record)
                component_pairs: list[tuple[PeriodAmount, PeriodAmount]] = []
                incomplete_lines: list[IncompleteLine] = []
                used_chunks = [anchor]
                for record in component_records:
                    row_value = record.get("chunk")
                    if not isinstance(row_value, FinancialChunk):
                        continue
                    row = row_value
                    citation = citation_by_chunk.get(row.chunk_id)
                    if citation is None:
                        continue
                    before = _amount(row, citation, before_year) if _period_value(row, before_year) is not None else None
                    after = _amount(row, citation, after_year) if _period_value(row, after_year) is not None else None
                    if before and after:
                        component_pairs.append((before, after))
                    else:
                        incomplete_lines.append(IncompleteLine(source_item_label(row), before, after, row.chunk_id))
                    used_chunks.append(row)

                total_citation = citation_by_chunk[anchor.chunk_id]
                total_before = _amount(anchor, total_citation, before_year)
                total_after = _amount(anchor, total_citation, after_year)
                if any(
                    fact is not None and NumericValidator._normalize_unit(fact.unit)
                    != NumericValidator._normalize_unit(total_before.unit)
                    for line in incomplete_lines for fact in (line.before, line.after)
                ):
                    raise AnalysisInputError("dòng thiếu kỳ có đơn vị khác tổng; không thể ghép vào bảng")
                computed = calculate_contributions(
                    total_before, total_after, component_pairs
                )
                computed = ContributionComputation(
                    computed.total_before, computed.total_after, computed.lines,
                    computed.residual_before, computed.residual_after, tuple(incomplete_lines),
                )
                used_ids = {chunk.chunk_id for chunk in used_chunks}
                score = sum(result_score.get(chunk_id, 0.0) for chunk_id in used_ids)
                current_report = int(anchor.fiscal_year == after_year)
                candidates.append((int(computed.complete), current_report, score, label, computed, used_chunks))
            except AnalysisInputError as exc:
                last_error = str(exc)

    if not candidates:
        raise AnalysisInputError(last_error)
    candidates.sort(key=lambda item: item[:3], reverse=True)
    best = candidates[0]

    def fingerprint(computed: ContributionComputation) -> tuple:
        return (
            computed.total_before.value, computed.total_after.value,
            tuple(sorted((line.label, line.before.value, line.after.value) for line in computed.lines)),
            tuple(sorted((line.label, line.before.value if line.before else None,
                          line.after.value if line.after else None) for line in computed.incomplete_lines)),
        )

    if len(candidates) > 1:
        next_candidate = candidates[1]
        if best[:3] == next_candidate[:3] and fingerprint(best[4]) != fingerprint(next_candidate[4]):
            raise AnalysisInputError("có nhiều bảng nguồn cùng phù hợp nhưng số liệu khác nhau")
    _complete, _current_report, _score, total_label, computed, used_chunks = best
    return total_label, computed, [
        chunk.chunk_id for chunk in used_chunks
    ]


def build_contribution_answer(
    parsed: ParsedQuery,
    results: list[RetrievalResult],
    citations: list[Citation],
    source_contexts: list[str],
    *,
    chunks_by_id: dict[str, FinancialChunk] | None = None,
    schema: ReportSchemaCatalog | None = None,
) -> ContributionAnswer:
    """Build complete or explicitly partial source-row contribution analysis."""
    citation_by_chunk = {
        citation.chunk_id: citation
        for citation in citations
        if citation.chunk_id
    }
    available_citations = citations
    available_contexts = source_contexts
    try:
        total_label, computed, used_chunk_ids = _build_analysis(
            parsed,
            results,
            citation_by_chunk,
        )
    except AnalysisInputError as exc:
        partial_totals = _available_total_answer(parsed, results, citation_by_chunk, str(exc))
        if partial_totals is not None:
            return partial_totals
        refs = " ".join(f"[{citation.citation_id}]" for citation in available_citations[:8])
        answer = f"Không đủ bằng chứng để tính mức đóng góp: {exc}."
        if refs:
            answer += f" Có thể đối chiếu nguồn đã truy xuất tại {refs}."
        return ContributionAnswer(answer, available_citations, available_contexts, [])

    total_before, total_after = computed.total_before, computed.total_after
    total_change, lines = computed.total_change, computed.lines
    total_citation = citation_by_chunk[total_before.chunk_id]
    unit = total_after.unit
    company = total_after.company_code
    before_year = total_before.year
    after_year = total_after.year
    change_word = "tăng" if total_change > 0 else "giảm" if total_change < 0 else "không đổi, chênh lệch"
    lines_out = [
        f"Đóng góp vào biến động {total_label} của {company} từ năm {before_year} đến {after_year}:",
        "",
        (
            f"{total_label}: {before_year} {_number_text(total_before.value)} {unit}; "
            f"{after_year} {_number_text(total_after.value)} {unit}; "
            f"{change_word} {_number_text(abs(total_change))} {unit} "
            f"[{total_citation.citation_id}]."
        ),
        "",
        f"| Thành phần | {before_year} | {after_year} | Thay đổi | Đóng góp vào biến động | Nguồn |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    def fact_input(role: str, fact: PeriodAmount) -> CalculationInput:
        return CalculationInput(role, format(fact.value, "f"), fact.unit, fact.year, fact.label,
                                fact.chunk_id, fact.citation_id)

    calculations = [calculation(total_change, unit, "absolute_change", (
        fact_input("prior_value", total_before), fact_input("current_value", total_after),
    ))]
    for line in lines:
        citation = citation_by_chunk[line.before.chunk_id]
        label = line.label.replace("|", "\\|")
        lines_out.append(
            f"| {label} | {_number_text(line.before.value)} {unit} "
            f"| {_number_text(line.after.value)} {unit} "
            f"| {_number_text(line.change)} {unit} "
            f"| {(_number_text(line.contribution_percent) + '%') if line.contribution_percent is not None else '—'} "
            f"| [{citation.citation_id}] |"
        )
        calculations.append(calculation(line.change, unit, "absolute_change", (
            fact_input("prior_value", line.before), fact_input("current_value", line.after),
        )))
        if line.contribution_percent is not None:
            calculations.append(calculation(line.contribution_percent, "%", "contribution_to_change", (
                fact_input("prior_component", line.before), fact_input("current_component", line.after),
                fact_input("prior_total", total_before), fact_input("current_total", total_after),
            )))
    for incomplete in computed.incomplete_lines:
        before_text = f"{_number_text(incomplete.before.value)} {unit}" if incomplete.before else "—"
        after_text = f"{_number_text(incomplete.after.value)} {unit}" if incomplete.after else "—"
        lines_out.append(
            f"| {incomplete.label.replace('|', chr(92) + '|')} | {before_text} | {after_text} | — | — "
            f"| [{citation_by_chunk[incomplete.chunk_id].citation_id}] |"
        )
    lines_out.extend([
        "",
        "Công thức: mức thay đổi thành phần = giá trị kỳ hiện tại − kỳ gốc; "
        "đóng góp = mức thay đổi thành phần / mức thay đổi tổng, quy đổi sang phần trăm. "
    ])
    if computed.complete:
        lines_out.append("Các dòng thành phần trực tiếp đã khớp tổng báo cáo ở cả hai kỳ.")
    else:
        lines_out.extend([
            "",
            "Bằng chứng thành phần chưa đầy đủ hoặc chưa đối chiếu khớp tổng. "
            "Thứ tự trong bảng chỉ áp dụng cho các khoản có đủ hai kỳ đã xác minh; "
            "chưa kết luận khoản đóng góp lớn nhất trong toàn bộ cơ cấu.",
            "Phần dư dưới đây bằng tổng báo cáo trừ các thành phần có đủ hai kỳ trong bảng; "
            "không được gán thành một khoản nợ hoặc một nguyên nhân cụ thể.",
            "",
            "| Đối chiếu | Giá trị | Nguồn |",
            "| --- | ---: | --- |",
        ])
        for total, residual, period_role in (
            (total_before, computed.residual_before, "before"),
            (total_after, computed.residual_after, "after"),
        ):
            inputs = [fact_input("total_value", total)] + [
                fact_input(f"component_{index}", getattr(line, period_role))
                for index, line in enumerate(lines)
            ]
            calculations.append(calculation(residual, unit, "component_sum_reconciliation", inputs))
            refs = " ".join(f"[{ref}]" for ref in dict.fromkeys(fact.citation_id for fact in inputs))
            lines_out.append(f"| Phần dư năm {total.year} | {_number_text(residual)} {unit} | {refs} |")
        residual_change = computed.residual_after - computed.residual_before
        residual_inputs = [fact_input("prior_total", total_before), fact_input("current_total", total_after)]
        for index, line in enumerate(lines):
            residual_inputs.extend((fact_input(f"prior_component_{index}", line.before),
                                    fact_input(f"current_component_{index}", line.after)))
        calculations.append(calculation(residual_change, unit, "contribution_residual", residual_inputs))
        refs = " ".join(f"[{ref}]" for ref in dict.fromkeys(fact.citation_id for fact in residual_inputs))
        lines_out.append(f"| Biến động chưa đối chiếu | {_number_text(residual_change)} {unit} | {refs} |")
        if computed.incomplete_lines:
            lines_out.append("Thiếu số liệu cho một hoặc cả hai kỳ ở: " + "; ".join(
                line.label for line in computed.incomplete_lines
            ) + ". Các ô thiếu giữ trống, không được suy ra giá trị cho ô trống.")
    if total_change == 0:
        lines_out.append("Tổng không biến động nên không tính tỷ lệ đóng góp; bảng vẫn trình bày mức tăng/giảm từng thành phần.")

    used_citations = [
        citation for citation in available_citations
        if citation.chunk_id in set(used_chunk_ids)
    ]
    if chunks_by_id:
        used_contexts = [
            chunks_by_id[chunk_id].content
            for chunk_id in used_chunk_ids
            if chunk_id in chunks_by_id
        ]
    else:
        used_contexts = available_contexts
    return ContributionAnswer(
        "\n".join(lines_out), used_citations, used_contexts, calculations,
        answer_status="complete" if computed.complete else "partial",
        missing_components=tuple(line.label for line in computed.incomplete_lines),
        residuals={before_year: format(computed.residual_before, "f"), after_year: format(computed.residual_after, "f")},
    )


def _available_total_answer(
    parsed: ParsedQuery, results: list[RetrievalResult], citations: dict[str, Citation], reason: str,
) -> ContributionAnswer | None:
    """Preserve source totals when one comparison period is absent; never substitute a year."""
    companies = parsed.company_codes or ([parsed.company_code] if parsed.company_code else [])
    if len(companies) != 1:
        return None
    try:
        years = _analysis_years(parsed)
    except AnalysisInputError:
        return None
    signatures = _target_signatures(parsed, companies)
    rows = [result for result in results if (
        result.chunk.chunk_type == ChunkType.FINANCIAL_ITEM
        and (result.chunk.company_code or "").casefold() == companies[0].casefold()
        and (not parsed.report_type or result.chunk.report_type == parsed.report_type)
        and (not parsed.statement_type or result.chunk.statement_type == parsed.statement_type)
        and result.chunk.metadata.get("report_eligible") is not False
        and result.chunk.chunk_id in citations
        and tuple(source_phrase_tokens(source_item_label(result.chunk), companies)) in signatures
    )]
    # Do not mask a rejected scope/unit/group with a fallback value from the
    # same complete anchor; that failure must remain visible.
    if any(all(_period_value(row.chunk, year) is not None for year in years) for row in rows):
        return None
    bases = {row.chunk.report_type or row.chunk.document_id or row.chunk.source_file for row in rows}
    if len(bases) != 1:
        return None
    facts = []
    used_chunks = {}
    for year in years:
        candidates = []
        for result in rows:
            try:
                fact = _amount(result.chunk, citations[result.chunk.chunk_id], year)
            except AnalysisInputError:
                continue
            candidates.append((result, fact))
        if not candidates or len({(fact.value, NumericValidator._normalize_unit(fact.unit)) for _, fact in candidates}) != 1:
            continue
        result, fact = max(candidates, key=lambda pair: (pair[0].chunk.fiscal_year == year, pair[0].rrf_score))
        facts.append(fact)
        used_chunks[fact.chunk_id] = result.chunk
    if not facts:
        return None
    text = [
        "Chỉ xác minh được các giá trị tổng dưới đây; chưa đủ bằng chứng cùng bảng/cặp kỳ để phân rã biến động.",
        "", "| Chỉ tiêu gốc | Năm | Giá trị | Nguồn |", "| --- | ---: | ---: | --- |",
    ]
    for fact in facts:
        text.append(f"| {fact.label.replace('|', chr(92) + '|')} | {fact.year} | {_number_text(fact.value)} {fact.unit} | [{fact.citation_id}] |")
    text.extend(["", f"Giới hạn: {reason}. Không suy ra thành phần, tỷ lệ đóng góp hoặc nguyên nhân."])
    return ContributionAnswer(
        "\n".join(text), [citations[chunk_id] for chunk_id in used_chunks],
        [chunk.content for chunk in used_chunks.values()], [], answer_status="partial",
        missing_periods=tuple(year for year in years if year not in {fact.year for fact in facts}),
    )
