"""Ratios with explicit conventions and source-period requirements."""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import re
from typing import Any, Callable

from ..schemas import ParsedQuery, PeriodType
from ..validation.numeric import NumericValidator
from .evidence import CalculationEvidence, calculation, source_input


def build_bank_ratios(
    parsed: ParsedQuery, facts: dict[str, dict[int, Any]], years: list[int],
    dimensions: list[str], format_number: Callable[..., str],
) -> tuple[list[str], list[CalculationEvidence], list[Any], list[str]]:
    lines: list[str] = []
    calculations: list[CalculationEvidence] = []
    used: list[Any] = []
    missing: list[str] = []
    question = parsed.original_query.casefold()
    target = years[-1]

    def compatible(inputs: tuple[Any, ...]) -> bool:
        return (
            len({(fact.chunk.document_id or fact.chunk.source_file, fact.chunk.report_type, fact.chunk.company_code) for fact in inputs}) == 1
            and len({NumericValidator._normalize_unit(fact.unit) for fact in inputs}) == 1
            and bool(NumericValidator._normalize_unit(inputs[0].unit))
        )

    def record(value: Decimal, formula: str, inputs: tuple[Any, ...], roles: tuple[str, ...]) -> str:
        calculations.append(calculation(value, "%", formula, (
            source_input(role, fact.chunk, fact.citation, fact.year, fact.value)
            for role, fact in zip(roles, inputs, strict=True)
        )))
        used.extend(inputs)
        return " ".join(dict.fromkeys(f"[{fact.citation.citation_id}]" for fact in inputs))

    for acronym, metric, formula in (
        ("ROA", "tong_cong_tai_san", "return_on_average_assets_after_tax_percent"),
        ("ROE", "tong_von_chu_so_huu", "return_on_average_equity_after_tax_percent"),
    ):
        if not re.search(rf"\b{acronym.casefold()}\b", question):
            continue
        for requested_year in parsed.fiscal_years:
            if requested_year != target:
                missing.append(f"{acronym} năm {requested_year}: cần số dư đầu năm trước kỳ này; hiện chỉ tính tỷ số cho năm {target}")
        profit = facts.get("loi_nhuan_sau_thue", {}).get(target)
        opening = facts.get(metric, {}).get(target - 1)
        closing = facts.get(metric, {}).get(target)
        if (
            parsed.fiscal_quarter is not None or parsed.period_type == PeriodType.HALF_YEAR
            or not profit or not opening or not closing
            or opening.value <= 0 or closing.value <= 0
            or not compatible((profit, opening, closing))
        ):
            missing.append(f"{acronym}: lợi nhuận cả năm và số dư đầu/cuối năm cùng báo cáo")
            continue
        value = (profit.value * 100 / ((opening.value + closing.value) / 2)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        refs = record(value, formula, (profit, opening, closing), ("net_profit_after_tax", "opening_balance", "closing_balance"))
        lines.append(f"- {acronym} sau thuế năm {target}, bình quân đầu–cuối năm: {format_number(value, places=2)}% {refs}.")
    if any(re.search(rf"\b{term}\b", question) for term in ("roa", "roe")):
        lines.append("Quy ước ROA/ROE: dùng tổng lợi nhuận sau thuế và tổng tài sản/vốn chủ sở hữu cùng phạm vi báo cáo; bình quân là trung bình hai số dư đầu và cuối năm. Không thay bằng mẫu số cuối năm khi thiếu đầu năm; không mặc định trùng tỷ lệ ngân hàng công bố.")

    if "operating_efficiency" in dimensions:
        values = {}
        for year in years:
            expense = facts.get("INCOME.BANK.OPERATING_EXPENSE", {}).get(year)
            income = facts.get("tong_thu_nhap_hoat_dong", {}).get(year)
            if not expense or not income or income.value <= 0 or not compatible((expense, income)):
                if year == target:
                    missing.append("CIR: tổng chi phí hoạt động và tổng thu nhập hoạt động cùng kỳ, cùng báo cáo")
                continue
            value = (abs(expense.value) * 100 / income.value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            refs = record(value, "cost_to_income_percent", (expense, income), ("operating_expense", "operating_income"))
            values[year] = value
            lines.append(f"- CIR năm {year}: {format_number(value, places=2)}% {refs}.")
        lines.append("Quy ước CIR: trị tuyệt đối tổng chi phí hoạt động / tổng thu nhập hoạt động; không cộng chi phí dự phòng rủi ro tín dụng vào tử số.")
        if len(values) == 2:
            direction = "tăng" if values[target] > values[years[0]] else "giảm" if values[target] < values[years[0]] else "không đổi"
            lines.append(f"CIR {direction} so với kỳ so sánh; tỷ lệ chi phí hoạt động trên thu nhập {direction}. Chưa kết luận nguyên nhân hoặc xếp loại hiệu quả khi thiếu bằng chứng bổ sung.")
    if re.search(r"\bnim\b", question):
        missing.append("NIM: thu nhập lãi thuần và tài sản sinh lãi bình quân đúng phạm vi")
    return lines, calculations, used, missing
