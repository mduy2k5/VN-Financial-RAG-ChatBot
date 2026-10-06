"""Typed calculation inputs with period and citation lineage."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Iterable

from ..schemas import Citation, FinancialChunk
from ..structure.source_groups import source_item_label
from ..validation.numeric import NumericValidator
from .formula_catalog import FORMULAS
from ..structure.semantics import period_supported


@dataclass(frozen=True)
class CalculationInput:
    role: str
    value: str
    unit: str
    year: int
    label: str
    chunk_id: str
    citation_id: str


@dataclass(frozen=True)
class CalculationEvidence:
    value: str
    unit: str
    formula_id: str
    input_chunk_ids: tuple[str, ...]
    input_citation_ids: tuple[str, ...]
    formula_version: str
    inputs: tuple[CalculationInput, ...] = ()


def source_input(
    role: str, chunk: FinancialChunk, citation: Citation, year: int, value: Decimal,
) -> CalculationInput:
    return CalculationInput(
        role=role, value=format(value, "f"), year=year,
        unit=str(chunk.unit or chunk.metadata.get("unit") or citation.unit or "").strip(),
        label=source_item_label(chunk), chunk_id=chunk.chunk_id,
        citation_id=citation.citation_id,
    )


def calculation(
    value: Decimal, unit: str, formula_id: str, inputs: Iterable[CalculationInput],
) -> CalculationEvidence:
    inputs = tuple(inputs)
    return CalculationEvidence(
        value=format(value, "f"), unit=unit, formula_id=formula_id,
        formula_version=FORMULAS[formula_id].version, inputs=inputs,
        input_chunk_ids=tuple(dict.fromkeys(item.chunk_id for item in inputs)),
        input_citation_ids=tuple(dict.fromkeys(item.citation_id for item in inputs)),
    )


def _source_amount(chunk: FinancialChunk, year: int) -> Decimal | None:
    if period_supported(chunk, year) is False:
        return None
    periods = chunk.metadata.get("period_values")
    raw = periods.get(str(year), periods.get(year)) if isinstance(periods, dict) else None
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


def _evaluate(formula_id: str, inputs: dict[str, CalculationInput]) -> Decimal:
    values = {role: Decimal(item.value) for role, item in inputs.items()}

    def require(*roles: str) -> None:
        if set(roles) != set(values):
            raise ValueError("vai trò đầu vào không đúng công thức")

    def ratio(numerator: Decimal, denominator: Decimal) -> Decimal:
        if denominator <= 0:
            raise ValueError("mẫu số không dương")
        return (numerator * 100 / denominator).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    if formula_id in {"absolute_change", "growth_rate"}:
        require("prior_value", "current_value")
        if inputs["prior_value"].year >= inputs["current_value"].year:
            raise ValueError("kỳ gốc phải trước kỳ hiện tại")
        delta = values["current_value"] - values["prior_value"]
        return delta if formula_id == "absolute_change" else ratio(delta, values["prior_value"])
    if formula_id == "contribution_to_change":
        require("prior_component", "current_component", "prior_total", "current_total")
        _paired_periods(inputs)
        delta = values["current_total"] - values["prior_total"]
        if delta == 0:
            raise ValueError("tổng không biến động")
        return ((values["current_component"] - values["prior_component"]) * 100 / delta).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP,
        )
    if formula_id == "percentage_point_change":
        require("prior_numerator", "prior_denominator", "current_numerator", "current_denominator")
        _paired_periods(inputs)
        return ratio(values["current_numerator"], values["current_denominator"]) - ratio(
            values["prior_numerator"], values["prior_denominator"],
        )
    if formula_id == "balance_sheet_reconciliation":
        require("total_assets", "total_liabilities", "total_equity")
        _same_period(inputs.values())
        return values["total_assets"] - values["total_liabilities"] - values["total_equity"]
    if formula_id in {"return_on_average_assets_after_tax_percent", "return_on_average_equity_after_tax_percent"}:
        require("net_profit_after_tax", "opening_balance", "closing_balance")
        target_year = inputs["net_profit_after_tax"].year
        if inputs["opening_balance"].year != target_year - 1 or inputs["closing_balance"].year != target_year:
            raise ValueError("số dư đầu/cuối năm không khớp kỳ lợi nhuận")
        if values["opening_balance"] <= 0 or values["closing_balance"] <= 0:
            raise ValueError("số dư bình quân phải dương")
        return ratio(values["net_profit_after_tax"], (values["opening_balance"] + values["closing_balance"]) / 2)
    if formula_id == "cost_to_income_percent":
        require("operating_expense", "operating_income")
        _same_period(inputs.values())
        return ratio(abs(values["operating_expense"]), values["operating_income"])
    if formula_id == "nonperforming_loans_ratio_percent":
        require("standard_loans", "special_mention_loans", "substandard_loans", "doubtful_loans", "loss_loans")
        _same_period(inputs.values())
        if any(value < 0 for value in values.values()):
            raise ValueError("số dư nợ âm")
        return ratio(
            values["substandard_loans"] + values["doubtful_loans"] + values["loss_loans"],
            sum(values.values(), Decimal(0)),
        )
    if formula_id == "component_sum_reconciliation":
        if "total_value" not in inputs or any(
            role != "total_value" and not role.startswith("component_") for role in inputs
        ):
            raise ValueError("vai trò đối chiếu thành phần không hợp lệ")
        _same_period(inputs.values())
        return values["total_value"] - sum(
            (value for role, value in values.items() if role != "total_value"), Decimal(0),
        )
    if formula_id == "contribution_residual":
        if {"prior_total", "current_total"} - inputs.keys() or any(
            role not in {"prior_total", "current_total"}
            and not role.startswith(("prior_component_", "current_component_")) for role in inputs
        ):
            raise ValueError("vai trò phần dư biến động không hợp lệ")
        _paired_periods(inputs)
        prior_keys = {role.removeprefix("prior_component_") for role in inputs if role.startswith("prior_component_")}
        current_keys = {role.removeprefix("current_component_") for role in inputs if role.startswith("current_component_")}
        if prior_keys != current_keys:
            raise ValueError("thành phần thiếu một kỳ")
        return values["current_total"] - values["prior_total"] - sum(
            (values[f"current_component_{key}"] - values[f"prior_component_{key}"] for key in prior_keys), Decimal(0),
        )
    if formula_id in {
        "share_of_total", "liabilities_to_assets_percent", "equity_to_assets_percent",
        "gross_margin_percent", "operating_margin_percent", "net_profit_margin_percent",
    }:
        require("numerator", "denominator")
        _same_period(inputs.values())
        return ratio(values["numerator"], values["denominator"])
    raise ValueError("công thức không hỗ trợ kiểm định")


def _same_period(inputs: Iterable[CalculationInput]) -> None:
    if len({item.year for item in inputs}) != 1:
        raise ValueError("đầu vào không cùng kỳ")


def _paired_periods(inputs: dict[str, CalculationInput]) -> None:
    prior = [item for role, item in inputs.items() if role.startswith("prior_")]
    current = [item for role, item in inputs.items() if role.startswith("current_")]
    _same_period(prior)
    _same_period(current)
    if prior[0].year >= current[0].year:
        raise ValueError("kỳ gốc phải trước kỳ hiện tại")


def validate_calculation(
    item: CalculationEvidence, citations: dict[str, Citation], chunks: dict[str, FinancialChunk],
) -> bool:
    """Recheck raw input values by period, roles, source scope and arithmetic."""
    try:
        definition = FORMULAS[item.formula_id]
        if item.formula_version != definition.version or not item.inputs:
            return False
        inputs = {fact.role: fact for fact in item.inputs}
        if len(inputs) != len(item.inputs):
            return False
        if set(item.input_chunk_ids) != {fact.chunk_id for fact in item.inputs} or set(
            item.input_citation_ids
        ) != {fact.citation_id for fact in item.inputs}:
            return False
        scopes = set()
        units = set()
        for fact in item.inputs:
            chunk = chunks[fact.chunk_id]
            citation = citations[fact.citation_id]
            if citation.chunk_id != fact.chunk_id or fact.label != source_item_label(chunk):
                return False
            value = Decimal(fact.value)
            if not value.is_finite() or value != _source_amount(chunk, fact.year):
                return False
            unit = NumericValidator._normalize_unit(fact.unit)
            source_unit = str(chunk.unit or chunk.metadata.get("unit") or citation.unit or "")
            if not unit or unit != NumericValidator._normalize_unit(source_unit):
                return False
            units.add(unit)
            # Unknown reporting bases may only combine facts from one source.
            scopes.add(((chunk.company_code or "").casefold(), chunk.report_type or (
                chunk.document_id or chunk.source_file
            )))
        if len(scopes) != 1 or len(units) != 1:
            return False
        if item.formula_id in {
            "return_on_average_assets_after_tax_percent", "return_on_average_equity_after_tax_percent", "cost_to_income_percent",
        } and len({chunks[fact.chunk_id].document_id or chunks[fact.chunk_id].source_file for fact in item.inputs}) != 1:
            return False
        output_unit = NumericValidator._normalize_unit(item.unit)
        expected_unit = next(iter(units)) if definition.output_unit == "same_as_inputs" else NumericValidator._normalize_unit(
            "điểm phần trăm" if definition.output_unit == "percentage_points" else definition.output_unit,
        )
        return output_unit == expected_unit and Decimal(item.value) == _evaluate(item.formula_id, inputs)
    except (KeyError, ValueError, InvalidOperation, ZeroDivisionError):
        return False
