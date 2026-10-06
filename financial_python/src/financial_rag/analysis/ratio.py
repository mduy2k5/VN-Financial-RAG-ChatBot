from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import re
import unicodedata
from typing import TypedDict

from ..metadata.metric_mapper import FinancialMetricMapper, MetricDefinition
from ..schemas import Citation, ChunkType, FinancialChunk, ParsedQuery, RetrievalResult, StatementType
from ..structure.source_groups import (
    display_source_item_label, source_item_label, source_label_matches_query,
)
from ..validation.numeric import NumericValidator
from .contribution import CalculationEvidence
from .evidence import calculation, source_input
from ..structure.semantics import period_supported


@dataclass(frozen=True)
class RatioAnswer:
    answer: str
    citations: list[Citation]
    source_contexts: list[str]
    calculations: list[CalculationEvidence]


@dataclass(frozen=True)
class RatioInput:
    metric_id: str
    label: str
    statement_type: StatementType


class RatioDefinition(TypedDict):
    label: str
    numerator: RatioInput
    denominator: RatioInput
    query_terms: tuple[str, ...]


RATIO_DEFINITIONS: dict[str, RatioDefinition] = {
    "liabilities_to_assets_percent": {
        "label": "Nợ phải trả / Tổng tài sản",
        "numerator": RatioInput("tong_no_phai_tra", "Tổng nợ phải trả", StatementType.BALANCE_SHEET),
        "denominator": RatioInput("tong_cong_tai_san", "Tổng tài sản", StatementType.BALANCE_SHEET),
        "query_terms": ("no phai tra", "no tren", "he so no"),
    },
    "equity_to_assets_percent": {
        "label": "Vốn chủ sở hữu / Tổng tài sản",
        "numerator": RatioInput("tong_von_chu_so_huu", "Tổng vốn chủ sở hữu", StatementType.BALANCE_SHEET),
        "denominator": RatioInput("tong_cong_tai_san", "Tổng tài sản", StatementType.BALANCE_SHEET),
        "query_terms": ("von chu so huu",),
    },
    "gross_margin_percent": {
        "label": "Biên lợi nhuận gộp",
        "numerator": RatioInput("loi_nhuan_gop", "Lợi nhuận gộp", StatementType.INCOME_STATEMENT),
        "denominator": RatioInput("doanh_thu_thuan", "Doanh thu thuần", StatementType.INCOME_STATEMENT),
        "query_terms": ("bien loi nhuan gop", "ty suat loi nhuan gop", "gross margin"),
    },
    "operating_margin_percent": {
        "label": "Biên lợi nhuận hoạt động",
        "numerator": RatioInput(
            "loi_nhuan_thuan_tu_hoat_dong_kinh_doanh",
            "Lợi nhuận thuần từ hoạt động kinh doanh",
            StatementType.INCOME_STATEMENT,
        ),
        "denominator": RatioInput("doanh_thu_thuan", "Doanh thu thuần", StatementType.INCOME_STATEMENT),
        "query_terms": ("bien loi nhuan hoat dong", "bien loi nhuan kinh doanh", "operating margin"),
    },
    "net_profit_margin_percent": {
        "label": "Biên lợi nhuận sau thuế",
        "numerator": RatioInput("loi_nhuan_sau_thue", "Lợi nhuận sau thuế", StatementType.INCOME_STATEMENT),
        "denominator": RatioInput("doanh_thu_thuan", "Doanh thu thuần", StatementType.INCOME_STATEMENT),
        "query_terms": ("bien loi nhuan sau thue", "bien loi nhuan rong", "net profit margin"),
    },
}


def _normalize_query(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.casefold()).replace("đ", "d")
    plain = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return " ".join(re.findall(r"[a-z0-9]+", plain))


def _requested_formulas(question: str) -> list[str]:
    normalized = _normalize_query(question)
    selected = [
        formula_id
        for formula_id, definition in RATIO_DEFINITIONS.items()
        if any(term in normalized for term in definition["query_terms"])
        and (
            not formula_id.endswith("_to_assets_percent")
            or "tai san" in normalized
        )
    ]
    # The unqualified phrase “biên lợi nhuận” has several valid definitions.
    # Do not silently choose gross, operating, or net margin.
    if not selected and "bien loi nhuan" in normalized:
        return []
    has_ratio_language = any(
        term in normalized
        for term in (
            "ty le", "ti le", "ty trong", "he so", " tren ",
            "bien loi nhuan", "margin", "ty suat",
        )
    )
    if not has_ratio_language:
        return []
    return selected


def requested_ratio_inputs(question: str) -> list[RatioInput]:
    """Return the unique source metrics needed by the explicitly named ratios."""
    inputs: dict[str, RatioInput] = {}
    for formula_id in _requested_formulas(question):
        definition = RATIO_DEFINITIONS[formula_id]
        for key in ("numerator", "denominator"):
            spec: RatioInput = definition[key]
            inputs.setdefault(spec.metric_id, spec)
    return list(inputs.values())


def _source_value(chunk: FinancialChunk, year: int) -> Decimal | None:
    if period_supported(chunk, year) is False:
        return None
    period_values = chunk.metadata.get("period_values")
    raw = None
    if isinstance(period_values, dict):
        raw = period_values.get(str(year), period_values.get(year))
    if raw in (None, "", "N/A", "-"):
        if chunk.fiscal_year == year:
            raw = chunk.metadata.get("end_value")
        elif chunk.comparative_fiscal_year == year:
            raw = chunk.metadata.get("start_value")
    normalized = NumericValidator.normalize_number(str(raw or ""))
    if not normalized:
        return None
    try:
        value = Decimal(normalized)
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def _collapse_duplicate_fact_rows(
    rows: list[FinancialChunk],
    years: list[int],
    scores_by_chunk: dict[str, float],
) -> list[FinancialChunk]:
    """Collapse duplicate parser views of the same row, but retain conflicts.

    Some TXT reports yield both a table-row chunk and a second atomic chunk for
    the same fact. Identical values from the same filing are one fact; distinct
    values remain separate so the ratio path can abstain on ambiguity.
    """
    by_signature: dict[tuple[str, tuple[str, ...]], FinancialChunk] = {}
    for chunk in rows:
        values = [_source_value(chunk, year) for year in years]
        if not values or any(value is None for value in values):
            continue
        unit = NumericValidator._normalize_unit(
            str(chunk.unit or chunk.metadata.get("unit") or "")
        ) or ""
        signature = (
            unit,
            tuple(format(value.normalize(), "f") for value in values if value is not None),
        )
        previous = by_signature.get(signature)
        if previous is None:
            by_signature[signature] = chunk
            continue

        def preference(candidate: FinancialChunk) -> tuple[bool, bool, float]:
            metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
            return (
                bool(candidate.table_title or metadata.get("table_title")),
                bool(metadata.get("schema_node_id")),
                scores_by_chunk.get(candidate.chunk_id, 0.0),
            )

        if preference(chunk) > preference(previous):
            by_signature[signature] = chunk
    return list(by_signature.values())


def matches_metric_source(
    chunk: FinancialChunk,
    metric_id: str,
    company_code: str,
    metrics_by_id: dict[str, MetricDefinition],
) -> bool:
    metric = metrics_by_id.get(metric_id)
    label = display_source_item_label(source_item_label(chunk))
    if metric is None or not label:
        return False
    if chunk.statement_type != metric.statement_type:
        return False
    return any(
        source_label_matches_query(expected, label, (company_code,))
        for expected in (metric.canonical_name, *metric.aliases)
        if expected
    )


def _format_number(value: Decimal) -> str:
    text = format(value, "f")
    integer, separator, fraction = text.partition(".")
    grouped = f"{int(integer):,}".replace(",", ".")
    return f"{grouped},{fraction}" if separator else grouped


def build_ratio_answer(
    parsed: ParsedQuery,
    results: list[RetrievalResult],
    citations: list[Citation],
    source_contexts: list[str],
    *,
    chunks_by_id: dict[str, FinancialChunk] | None = None,
) -> RatioAnswer:
    """Calculate only explicitly requested ratios from cited source labels.

    The metric registry provides known source-label variants; schema node IDs
    do not select a numerator or denominator.
    """
    companies = parsed.company_codes or ([parsed.company_code] if parsed.company_code else [])
    requested_years = list(dict.fromkeys(int(year) for year in parsed.fiscal_years))
    if not requested_years and parsed.fiscal_year is not None:
        requested_years = [int(parsed.fiscal_year)]
    citation_by_chunk = {citation.chunk_id: citation for citation in citations if citation.chunk_id}
    chunks_by_result_id = {
        result.chunk.chunk_id: result.chunk
        for result in results
        if result.chunk.chunk_id in citation_by_chunk
    }
    scores_by_chunk = {
        result.chunk.chunk_id: result.rrf_score
        for result in results
        if result.chunk.chunk_id in citation_by_chunk
    }

    try:
        if len(companies) != 1:
            raise ValueError("cần xác định duy nhất một doanh nghiệp")
        if not requested_years:
            raise ValueError("cần nêu năm tài chính; hệ thống không tự chọn năm gần nhất")
        if len(requested_years) > 4:
            raise ValueError("một yêu cầu chỉ hỗ trợ tối đa bốn kỳ")
        formula_ids = _requested_formulas(parsed.original_query)
        if not formula_ids:
            if "bien loi nhuan" in _normalize_query(parsed.original_query):
                raise ValueError("cần chỉ rõ biên lợi nhuận gộp, hoạt động hay sau thuế")
            raise ValueError("chưa nhận diện được tỷ số có công thức được đăng ký")

        metrics_by_id = {
            metric.metric_id: metric for metric in FinancialMetricMapper().metrics_list
        }
        candidates = [
            chunk for chunk in chunks_by_result_id.values()
            if chunk.chunk_type == ChunkType.FINANCIAL_ITEM
            and (chunk.company_code or "").casefold() == companies[0].casefold()
            and (not parsed.report_type or chunk.report_type == parsed.report_type)
            and chunk.metadata.get("report_eligible") is not False
        ]
        groups: dict[tuple[str, str], list[FinancialChunk]] = {}
        for chunk in candidates:
            key = (chunk.document_id or chunk.source_file, chunk.report_type or "")
            groups.setdefault(key, []).append(chunk)

        computed_rows: list[
            tuple[str, int, Decimal, RatioInput, RatioInput, FinancialChunk, FinancialChunk]
        ] = []
        calculations: list[CalculationEvidence] = []
        used_chunk_ids: set[str] = set()
        for formula_id in formula_ids:
            definition = RATIO_DEFINITIONS[formula_id]
            numerator_spec: RatioInput = definition["numerator"]
            denominator_spec: RatioInput = definition["denominator"]
            valid_groups = []
            for (document_id, report_type), rows in groups.items():
                numerators = [
                    row for row in rows
                    if matches_metric_source(
                        row, numerator_spec.metric_id, companies[0], metrics_by_id
                    )
                ]
                numerators = _collapse_duplicate_fact_rows(
                    numerators, requested_years, scores_by_chunk
                )
                denominators = [
                    row for row in rows
                    if matches_metric_source(
                        row, denominator_spec.metric_id, companies[0], metrics_by_id
                    )
                ]
                denominators = _collapse_duplicate_fact_rows(
                    denominators, requested_years, scores_by_chunk
                )
                if len(numerators) == 1 and len(denominators) == 1:
                    valid_groups.append((document_id, report_type, numerators[0], denominators[0]))
            if not valid_groups:
                raise ValueError(
                    f"không truy xuất được đồng thời ‘{numerator_spec.label}’ và ‘{denominator_spec.label}’ "
                    "trong cùng báo cáo"
                )
            if len(valid_groups) > 1:
                latest_year = max(requested_years)
                latest_matches = [
                    group for group in valid_groups
                    if any(row.fiscal_year == latest_year for row in (group[2], group[3]))
                ]
                if len(latest_matches) == 1:
                    valid_groups = latest_matches
                elif len({(group[0], group[1]) for group in valid_groups}) > 1:
                    raise ValueError("có nhiều báo cáo cùng phù hợp; cần chỉ rõ báo cáo hợp nhất hay riêng")
            _document_id, _report_type, numerator_chunk, denominator_chunk = valid_groups[0]

            for year in requested_years:
                numerator = _source_value(numerator_chunk, year)
                denominator = _source_value(denominator_chunk, year)
                if numerator is None or denominator is None:
                    raise ValueError(f"thiếu một đầu vào của công thức cho năm {year}")
                numerator_unit = str(numerator_chunk.unit or numerator_chunk.metadata.get("unit") or "").strip()
                denominator_unit = str(denominator_chunk.unit or denominator_chunk.metadata.get("unit") or "").strip()
                if (
                    not NumericValidator._normalize_unit(numerator_unit)
                    or NumericValidator._normalize_unit(numerator_unit)
                    != NumericValidator._normalize_unit(denominator_unit)
                ):
                    raise ValueError("tử số và mẫu số không cùng đơn vị")
                if denominator <= 0:
                    raise ValueError(f"mẫu số không dương ở năm {year}; không tính tỷ số")
                percentage = (numerator * Decimal("100") / denominator).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
                numerator_citation = citation_by_chunk[numerator_chunk.chunk_id]
                denominator_citation = citation_by_chunk[denominator_chunk.chunk_id]
                computed_rows.append((
                    str(definition["label"]), year, percentage, numerator_spec,
                    denominator_spec, numerator_chunk, denominator_chunk,
                ))
                used_chunk_ids.update((numerator_chunk.chunk_id, denominator_chunk.chunk_id))
                calculations.append(calculation(percentage, "%", formula_id, (
                    source_input("numerator", numerator_chunk, numerator_citation, year, numerator),
                    source_input("denominator", denominator_chunk, denominator_citation, year, denominator),
                )))

        lines = [
            f"Các tỷ số tài chính của {companies[0]} theo dữ liệu đã truy xuất:",
            "",
            "| Chỉ số | Năm | Tử số | Mẫu số | Kết quả | Nguồn |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for (
            label, year, percentage, _numerator_spec, _denominator_spec,
            numerator_chunk, denominator_chunk,
        ) in computed_rows:
            numerator_citation = citation_by_chunk[numerator_chunk.chunk_id]
            denominator_citation = citation_by_chunk[denominator_chunk.chunk_id]
            numerator_value = _source_value(numerator_chunk, year)
            denominator_value = _source_value(denominator_chunk, year)
            if numerator_value is None or denominator_value is None:
                raise ValueError(f"thiếu một đầu vào của công thức cho năm {year}")
            unit = str(numerator_chunk.unit or numerator_chunk.metadata.get("unit") or "").strip()
            lines.append(
                f"| {label} | {year} | {_format_number(numerator_value)} {unit} "
                f"| {_format_number(denominator_value)} {unit} | {_format_number(percentage)}% "
                f"| [{numerator_citation.citation_id}] [{denominator_citation.citation_id}] |"
            )
        lines.extend([
            "",
            "Tỷ số được tính từ các giá trị cùng công ty, kỳ, đơn vị và một báo cáo. "
            "Đây là chỉ số mô tả; chưa áp dụng ngưỡng tốt/xấu hoặc xếp hạng.",
        ])
        used_citations = [citation for citation in citations if citation.chunk_id in used_chunk_ids]
        used_contexts = (
            [
                chunks_by_id[chunk_id].content
                for chunk_id in used_chunk_ids
                if chunks_by_id and chunk_id in chunks_by_id
            ]
            if chunks_by_id
            else source_contexts
        )
        return RatioAnswer("\n".join(lines), used_citations, used_contexts, calculations)
    except ValueError as exc:
        refs = " ".join(f"[{citation.citation_id}]" for citation in citations[:8])
        answer = f"Không đủ bằng chứng để tính tỷ số: {exc}."
        if refs:
            answer += f" Có thể đối chiếu nguồn đã truy xuất tại {refs}."
        return RatioAnswer(answer, citations, source_contexts, [])
