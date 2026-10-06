from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FormulaDefinition:
    formula_id: str
    version: str
    expression: str
    output_unit: str


FORMULAS = {
    "return_on_average_assets_after_tax_percent": FormulaDefinition(
        formula_id="return_on_average_assets_after_tax_percent", version="1.0",
        expression="net_profit_after_tax / ((opening_balance + closing_balance) / 2) * 100",
        output_unit="%",
    ),
    "return_on_average_equity_after_tax_percent": FormulaDefinition(
        formula_id="return_on_average_equity_after_tax_percent", version="1.0",
        expression="net_profit_after_tax / ((opening_balance + closing_balance) / 2) * 100",
        output_unit="%",
    ),
    "cost_to_income_percent": FormulaDefinition(
        formula_id="cost_to_income_percent", version="1.0",
        expression="abs(operating_expense) / operating_income * 100", output_unit="%",
    ),
    "nonperforming_loans_ratio_percent": FormulaDefinition(
        formula_id="nonperforming_loans_ratio_percent", version="1.0",
        expression="(substandard_loans + doubtful_loans + loss_loans) / (standard_loans + special_mention_loans + substandard_loans + doubtful_loans + loss_loans) * 100",
        output_unit="%",
    ),
    "absolute_change": FormulaDefinition(
        formula_id="absolute_change",
        version="1.0",
        expression="current_value - prior_value",
        output_unit="same_as_inputs",
    ),
    "growth_rate": FormulaDefinition(
        formula_id="growth_rate",
        version="1.0",
        expression="(current_value - prior_value) / prior_value * 100",
        output_unit="%",
    ),
    "percentage_point_change": FormulaDefinition(
        formula_id="percentage_point_change",
        version="1.0",
        expression="current_ratio_percent - prior_ratio_percent",
        output_unit="percentage_points",
    ),
    "share_of_total": FormulaDefinition(
        formula_id="share_of_total",
        version="1.0",
        expression="component_value / total_value * 100",
        output_unit="%",
    ),
    "balance_sheet_reconciliation": FormulaDefinition(
        formula_id="balance_sheet_reconciliation",
        version="1.0",
        expression="total_assets - total_liabilities - total_equity",
        output_unit="same_as_inputs",
    ),
    "contribution_to_change": FormulaDefinition(
        formula_id="contribution_to_change",
        version="1.0",
        expression="component_change / total_change * 100",
        output_unit="%",
    ),
    "component_sum_reconciliation": FormulaDefinition(
        formula_id="component_sum_reconciliation", version="1.0",
        expression="total_value - sum(component_values)", output_unit="same_as_inputs",
    ),
    "contribution_residual": FormulaDefinition(
        formula_id="contribution_residual", version="1.0",
        expression="(current_total - prior_total) - sum(current_component_values - prior_component_values)",
        output_unit="same_as_inputs",
    ),
    "liabilities_to_assets_percent": FormulaDefinition(
        formula_id="liabilities_to_assets_percent",
        version="1.0",
        expression="total_liabilities / total_assets * 100",
        output_unit="%",
    ),
    "equity_to_assets_percent": FormulaDefinition(
        formula_id="equity_to_assets_percent",
        version="1.0",
        expression="total_equity / total_assets * 100",
        output_unit="%",
    ),
    "gross_margin_percent": FormulaDefinition(
        formula_id="gross_margin_percent",
        version="1.0",
        expression="gross_profit / net_revenue * 100",
        output_unit="%",
    ),
    "operating_margin_percent": FormulaDefinition(
        formula_id="operating_margin_percent",
        version="1.0",
        expression="operating_profit / net_revenue * 100",
        output_unit="%",
    ),
    "net_profit_margin_percent": FormulaDefinition(
        formula_id="net_profit_margin_percent",
        version="1.0",
        expression="net_profit_after_tax / net_revenue * 100",
        output_unit="%",
    ),
}
