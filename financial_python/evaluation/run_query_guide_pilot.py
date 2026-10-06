from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

from financial_rag.evaluation.query_understanding_benchmark import (
    run_query_understanding_benchmark,
)
from financial_rag.metadata.metric_mapper import FinancialMetricMapper
from financial_rag.query_guides.catalog import load_guide_catalog
from financial_rag.query_guides.planner import (
    QueryPlanError,
    build_query_plan,
    validate_query_scope,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = PROJECT_ROOT / "evaluation" / "query_guide_pilot_cases.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "results" / "query_guide_pilot_v1.json"
FIXTURE_ACCESS = {"tenant_id": "offline-fixture"}


def _load_cases(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _evaluate_routes(cases_path: Path) -> dict[str, Any]:
    cases = _load_cases(cases_path)
    errors: list[dict[str, Any]] = []
    correct = 0
    forced_negative_routes = 0
    positive_routes_needing_clarification = 0
    for case in cases:
        plan = build_query_plan(case["query"], access_filter=FIXTURE_ACCESS)
        predicted = plan.guide_id
        expected = case["expected_guide_id"]
        is_correct = predicted == expected
        correct += int(is_correct)
        if expected is None and predicted is not None:
            forced_negative_routes += 1
        if expected is not None and plan.next_action == "clarify":
            positive_routes_needing_clarification += 1
        if not is_correct:
            errors.append({
                "id": case["id"],
                "expected_guide_id": expected,
                "predicted_guide_id": predicted,
                "domain": plan.domain,
                "task": plan.task,
                "score": plan.guide_score,
                "reason": plan.guide_match_reason,
            })
    return {
        "cases": len(cases),
        "correct": correct,
        "exact_route_accuracy": round(correct / len(cases), 4) if cases else None,
        "forced_routes_on_negative_cases": forced_negative_routes,
        "positive_routes_needing_clarification": positive_routes_needing_clarification,
        "errors": errors,
        "annotation_note": "Small synthetic pilot set; labels are not independent human review.",
    }


def _evaluate_scope() -> dict[str, Any]:
    outcomes: dict[str, bool] = {}

    compare = build_query_plan(
        "So sánh tổng tài sản cuối năm 2024 của ABB và ACB theo BCTC hợp nhất.",
        access_filter=FIXTURE_ACCESS,
    )
    outcomes["access_filter_retained"] = compare.hard_filters.get("access") == FIXTURE_ACCESS
    outcomes["both_companies_retained"] = compare.hard_filters.get("company_codes") == ["ABB", "ACB"]
    outcomes["explicit_year_retained"] = compare.hard_filters.get("fiscal_years") == [2024]
    outcomes["explicit_report_type_retained"] = compare.hard_filters.get("report_type") == "consolidated"

    two_years = build_query_plan(
        "So sánh doanh thu thuần FPT năm 2025 so với năm 2024.",
        access_filter=FIXTURE_ACCESS,
    )
    outcomes["both_comparison_years_retained"] = (
        two_years.hard_filters.get("fiscal_years") == [2025, 2024]
    )

    no_period = build_query_plan(
        "Tổng tài sản ABB là bao nhiêu?",
        access_filter=FIXTURE_ACCESS,
    )
    outcomes["missing_period_not_silently_defaulted"] = "fiscal_years" not in no_period.hard_filters

    analysis = build_query_plan(
        "Đánh giá sức khỏe tài chính của ABB trong năm 2024.",
        access_filter=FIXTURE_ACCESS,
    )
    outcomes["evaluation_uses_multi_dimension_recipe"] = (
        analysis.guide_id == "financial_evaluation"
        and len(analysis.analysis_dimensions) >= 3
        and analysis.retrieval_policy.get("retrieve_structured_values_and_related_explanations") is True
    )
    outcomes["derived_ratios_require_registered_formula_and_cited_inputs"] = (
        analysis.analysis_requirements.get("derive_only_registered_formulas") is True
        and analysis.analysis_requirements.get("require_cited_inputs_for_derived_values") is True
    )
    outcomes["normative_rating_requires_disclosed_benchmark"] = (
        analysis.analysis_requirements.get("require_disclosed_benchmark_before_normative_rating") is True
    )
    outcomes["evaluation_plan_contains_concrete_metric_ids"] = {
        "loi_nhuan_sau_thue",
        "doanh_thu_thuan",
        "tien_va_tuong_duong_tien",
        "tong_no_phai_tra",
    }.issubset(set(analysis.planned_metric_ids))

    unsupported_dimension = build_query_plan(
        "Đánh giá chất lượng tài sản ABB năm 2024.",
        access_filter=FIXTURE_ACCESS,
    )
    outcomes["unsupported_quality_metrics_are_disclosed"] = (
        "asset_quality" in unsupported_dimension.unsupported_analysis_dimensions
    )

    drivers = build_query_plan(
        "Phân tích các yếu tố thúc đẩy tăng trưởng doanh thu FPT năm 2025 so với năm 2024.",
        access_filter=FIXTURE_ACCESS,
    )
    outcomes["driver_analysis_keeps_both_periods"] = (
        drivers.guide_id == "financial_driver_analysis"
        and drivers.hard_filters.get("fiscal_years") == [2025, 2024]
        and "period_comparison" in drivers.analysis_requirements.get("operations", [])
    )

    unscoped_evaluation = build_query_plan(
        "Đánh giá sức khỏe tài chính của ABB.",
        access_filter=FIXTURE_ACCESS,
    )
    outcomes["evaluation_without_period_requests_clarification"] = (
        unscoped_evaluation.next_action == "clarify"
        and "fiscal_years" in unscoped_evaluation.missing_slots
    )

    mutated = compare.model_copy(deep=True)
    mutated.hard_filters["fiscal_years"] = [2023]
    try:
        validate_query_scope(
            compare.original_question,
            mutated,
            access_filter=FIXTURE_ACCESS,
        )
        outcomes["scope_validator_rejects_changed_year"] = False
    except QueryPlanError:
        outcomes["scope_validator_rejects_changed_year"] = True

    clarify = build_query_plan(
        "Tóm tắt ESG trong báo cáo thường niên.",
        access_filter=FIXTURE_ACCESS,
    )
    outcomes["missing_required_company_requests_clarification"] = (
        clarify.next_action == "clarify" and "company_codes" in clarify.missing_slots
    )

    try:
        build_query_plan("Doanh thu FPT năm 2024", access_filter={})
        outcomes["empty_access_scope_rejected"] = False
    except QueryPlanError:
        outcomes["empty_access_scope_rejected"] = True

    catalog = load_guide_catalog()
    configured_metric_ids = {
        metric_id
        for dimension_groups in catalog.analysis_metric_groups.values()
        for metric_ids in dimension_groups.values()
        for metric_id in metric_ids
    }
    registered_metric_ids = {metric.metric_id for metric in FinancialMetricMapper().metrics_list}
    outcomes["analysis_metric_ids_exist_in_registry"] = configured_metric_ids.issubset(
        registered_metric_ids
    )

    return {
        "checks": outcomes,
        "passed": sum(outcomes.values()),
        "total": len(outcomes),
        "all_passed": all(outcomes.values()),
    }


def run_pilot(cases_path: Path = DEFAULT_CASES) -> dict[str, Any]:
    baseline = run_query_understanding_benchmark(write_result=False)
    return {
        "pilot": "query_guides_v1_offline",
        "route_evaluation": _evaluate_routes(cases_path),
        "scope_evaluation": _evaluate_scope(),
        "existing_financial_parser_baseline": {
            "cases": baseline["cases"],
            "accuracies": baseline["accuracies"],
            "thresholds_met_on_this_fixture": baseline["thresholds_met_on_this_fixture"],
            "annotation_limitations": baseline["annotation_limitations"],
        },
        "redisvl_installed": importlib.util.find_spec("redisvl") is not None,
        "external_services_contacted": False,
        "limitations": [
            "The pilot route matcher uses transparent keyword/parser rules, not vector similarity.",
            "The existing 88-question financial labels were assigned after seeing resolver output and need independent review.",
            "The synthetic route cases do not establish ESG or annual-report retrieval quality.",
            "Analysis recipes describe evidence to collect; the current pilot does not calculate ratios or generate analytical answers.",
            "No retrieval, citation validation, answer generation, Redis index, or production store was exercised.",
            "RedisVL cosine score threshold must be calibrated on independently labeled positive and negative paraphrases.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline query-guide pilot.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = run_pilot(args.cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
