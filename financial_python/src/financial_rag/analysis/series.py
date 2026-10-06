"""Bounded multi-issuer/period arithmetic over retrieved, source-checked rows."""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from ..metadata.metric_mapper import MetricDefinition
from ..schemas import Citation, ParsedQuery, RetrievalResult
from ..validation.numeric import NumericValidator
from .overview import (OverviewAnswer, _calculation, _format_decimal, _same_reporting_basis,
                       _select_period_fact)

MAX_SERIES_YEARS = 10
MAX_SERIES_COMPANIES = 5
MAX_SERIES_CELLS = 100


def series_within_budget(companies: list[str], years: list[int], metrics: list[str]) -> bool:
    return bool(companies and years and metrics
                and len(companies) <= MAX_SERIES_COMPANIES and len(years) <= MAX_SERIES_YEARS
                and len(companies) * len(years) * len(metrics) <= MAX_SERIES_CELLS)


def build_series_answer(plan: Any, parsed: ParsedQuery,
                        metrics: dict[str, MetricDefinition],
                        results_by_metric: dict[str, list[RetrievalResult]],
                        citations: list[Citation]) -> OverviewAnswer:
    raw_companies = parsed.company_codes or (
        [parsed.company_code] if parsed.company_code is not None else []
    )
    companies = list(dict.fromkeys(raw_companies))
    years = sorted(set(parsed.fiscal_years or (
        [parsed.fiscal_year] if parsed.fiscal_year is not None else []
    )))
    if not series_within_budget(companies, years, plan.planned_metric_ids):
        return OverviewAnswer("Phạm vi quá lớn: hãy chọn tối đa 5 doanh nghiệp, 10 kỳ và 100 ô chỉ tiêu.",
                              [], [], [], answer_status="needs_clarification")
    by_chunk = {
        citation.chunk_id: citation for citation in citations if citation.chunk_id is not None
    }
    lines = ["So sánh số liệu theo doanh nghiệp và kỳ từ các báo cáo đã truy xuất:", "",
             "| Doanh nghiệp | Chỉ tiêu gốc trong BCTC | " + " | ".join(map(str, years)) + " |",
             "| --- | --- | " + " | ".join("---:" for _ in years) + " |"]
    missing = []
    used = {}
    calculations = []
    change_lines = []
    for company in companies:
        for metric_id in plan.planned_metric_ids:
            metric = metrics.get(metric_id)
            if metric is None:
                missing.append(f"{company}: {metric_id} chưa có định nghĩa chỉ tiêu")
                continue
            values = {}
            for year in years:
                # A yearly request must not substitute a quarterly flow.
                results = [result for result in results_by_metric.get(metric_id, [])
                           if parsed.fiscal_quarter is not None or result.chunk.fiscal_quarter is None]
                fact, error = _select_period_fact(metric_id, metric, year, results, by_chunk,
                                                 company, parsed.report_type)
                if error or fact is None:
                    missing.append(f"{company}: {metric.canonical_name}, kỳ {year}"
                                   + (f" ({error})" if error else ""))
                else:
                    values[year] = fact
                    used[fact.chunk.chunk_id] = fact
            label = next((fact.label for fact in values.values()), metric.canonical_name)
            cells = [f"{_format_decimal(values[year].value)} {values[year].unit} "
                     f"[{values[year].citation.citation_id}]" if year in values else "—"
                     for year in years]
            lines.append(f"| {company} | {label} | " + " | ".join(cells) + " |")
            for prior_year, current_year in zip(years, years[1:]):
                before, after = values.get(prior_year), values.get(current_year)
                if before is None or after is None:
                    continue
                unit = NumericValidator._normalize_unit(before.unit)
                if (not unit or unit != NumericValidator._normalize_unit(after.unit)
                        or not _same_reporting_basis(before, after)
                        or before.chunk.fiscal_quarter != after.chunk.fiscal_quarter):
                    missing.append(f"{company}: {label}, {prior_year}→{current_year} khác đơn vị hoặc cơ sở kỳ/báo cáo")
                    continue
                change = after.value - before.value
                calculations.append(_calculation(change, before.unit, "absolute_change", (before, after)))
                text = f"- {company}, {label}, {prior_year}→{current_year}: chênh lệch {_format_decimal(change)} {before.unit}"
                if before.value > 0:
                    growth = (change * 100 / before.value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                    calculations.append(_calculation(growth, "%", "growth_rate", (before, after)))
                    text += f" ({_format_decimal(growth, places=2)}%)"
                text += f" [{before.citation.citation_id}] [{after.citation.citation_id}]."
                change_lines.append(text)
    if change_lines:
        lines.extend(["", "Biến động giữa các kỳ được hỏi:", *change_lines])
    if missing:
        lines.extend(["", "Phần chưa đủ bằng chứng:", *["- " + item + "." for item in missing]])
    lines.extend(["", "Mỗi kỳ ưu tiên báo cáo của chính kỳ đó; không tự thay bằng số trình bày lại trong báo cáo mới hơn. Không nội suy kỳ thiếu hoặc kết luận nguyên nhân từ biến động số liệu."])
    if not used:
        return OverviewAnswer("Không đủ bằng chứng cho các doanh nghiệp và kỳ được hỏi.", [], [], [],
                              answer_status="insufficient_evidence", missing_components=tuple(missing))
    return OverviewAnswer("\n".join(lines), [fact.citation for fact in used.values()],
                          [fact.chunk.content for fact in used.values()], calculations,
                          answer_status="partial" if missing else "complete",
                          missing_components=tuple(missing))
