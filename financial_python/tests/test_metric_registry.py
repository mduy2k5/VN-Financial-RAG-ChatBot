import json
import re
from pathlib import Path

from financial_rag.metadata.metric_mapper import FinancialMetricMapper
from financial_rag.schemas import StatementType


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPOSITORY_ROOT / "financial_metric_registry.json"


def test_exported_registry_matches_runtime_mapper():
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    mapper = FinancialMetricMapper()

    assert registry["schema_version"] == 1
    assert registry["metric_count"] == len(registry["metrics"]) == len(mapper.metrics_list)
    assert len({metric["metric_id"] for metric in registry["metrics"]}) == registry["metric_count"]
    assert all(re.fullmatch(r"[a-z0-9_]+", metric["metric_id"]) for metric in registry["metrics"])

    expected = [
        {
            "metric_id": metric.metric_id,
            "canonical_name": metric.canonical_name,
            "statement_type": metric.statement_type.value,
            "statement_name_vi": metric.statement_name_vi,
            "code": metric.code,
            "section": metric.section,
            "aliases": metric.aliases,
        }
        for metric in mapper.metrics_list
    ]
    assert registry["metrics"] == expected


def test_added_registry_metrics_resolve_by_alias_and_statement_code():
    mapper = FinancialMetricMapper()

    deferred_tax = mapper.lookup_metric("Tài sản thuế hoãn lại")
    assert deferred_tax is not None
    assert deferred_tax.metric_id == "tai_san_thue_thu_nhap_hoan_lai"
    assert deferred_tax.code == "262"

    taxes = mapper.lookup_metric(
        "",
        code="313",
        statement_hint=StatementType.BALANCE_SHEET,
    )
    assert taxes is not None
    assert taxes.metric_id == "thue_va_cac_khoan_phai_nop_nha_nuoc"

    employees = mapper.lookup_metric("phải trả người lao động")
    assert employees is not None
    assert employees.metric_id == "phai_tra_nguoi_lao_dong"


def test_tax_expense_components_remain_distinct_when_form_code_collides():
    mapper = FinancialMetricMapper()

    current = mapper.lookup_metric(
        "Chi phí thuế thu nhập doanh nghiệp hiện hành",
        code="51",
        statement_hint=StatementType.INCOME_STATEMENT,
    )
    deferred = mapper.lookup_metric(
        "Chi phí thuế thu nhập doanh nghiệp hoãn lại",
        code="51",
        statement_hint=StatementType.INCOME_STATEMENT,
    )
    total = mapper.lookup_metric(
        "Chi phí thuế thu nhập doanh nghiệp",
        code="51",
        statement_hint=StatementType.INCOME_STATEMENT,
    )
    generic_query = mapper.lookup_metric("chi phí thuế")
    code_only = mapper.lookup_metric(
        "",
        code="51",
        statement_hint=StatementType.INCOME_STATEMENT,
    )

    assert current is not None and current.metric_id == "chi_phi_thue_tndn_hien_hanh"
    assert deferred is not None and deferred.metric_id == "chi_phi_thue_tndn_hoan_lai"
    assert total is not None and total.metric_id == "chi_phi_thue_tndn"
    assert generic_query is not None and generic_query.metric_id == "chi_phi_thue_tndn"
    assert code_only is not None and code_only.metric_id == "chi_phi_thue_tndn"


def test_exact_metric_alias_respects_statement_hint():
    mapper = FinancialMetricMapper()

    note_row = mapper.lookup_metric(
        "Thuế thu nhập doanh nghiệp",
        code="51",
        statement_hint=StatementType.NOTES,
    )

    assert note_row is None
