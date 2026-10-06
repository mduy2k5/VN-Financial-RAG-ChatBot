from decimal import Decimal

import pytest

from financial_rag.chunking import UniversalFinancialChunker
from financial_rag.schemas import ChunkType, FinancialChunk, StatementType
from financial_rag.storage.migrate_normalized import backfill_coordinates
from financial_rag.storage.sqlite_store import SQLiteFinancialStore
from financial_rag.structure.normalized import normalize_report


def document(raw="source v1", *, value="(1.234,56)"):
    return {
        "logical_key": "report", "document_id": "doc", "source_file": "FPT_2025.txt",
        "company_code": "FPT", "fiscal_year": 2025, "report_type": "consolidated",
        "raw_content": raw,
        "chunks": [FinancialChunk(
            chunk_id="row", source_file="FPT_2025.txt", fiscal_year=2025,
            comparative_fiscal_year=2024, page=2, statement_type=StatementType.INCOME_STATEMENT,
            chunk_type=ChunkType.FINANCIAL_ITEM, unit="Triệu đồng", content="evidence",
            metadata={"metric_name": "Lợi nhuận", "table_id": "table", "source_row_index": 0,
                      "source_cells": ["Lợi nhuận", value, "0"],
                      "table_headers": ["Chỉ tiêu", "2025", "2024"],
                      "source_period_columns": {"2025": 1, "2024": 2},
                      "period_values": {"2025": value, "2024": "0"}},
        )],
    }


def test_decimal_values_units_and_source_coordinates_are_preserved():
    bundle = normalize_report(document(), revision="parser1")
    current, prior = bundle.facts
    assert current.value == "-1234.56"
    assert Decimal(current.value) * Decimal(current.scale) == Decimal("-1234560000")
    assert current.raw_value == "(1.234,56)"
    assert current.currency == "VND" and current.period_kind == "duration"
    assert current.coordinate_status == "located" and current.column_header == "2025"
    assert prior.value == "0" and prior.period_role == "comparative"
    assert bundle.warnings == {}


def test_parser_revision_does_not_create_a_new_source_version():
    first = normalize_report(document(), revision="p1")
    second = normalize_report(document(), revision="p2")
    assert first.version.version_id == second.version.version_id
    assert first.projection_id != second.projection_id
    changed = normalize_report(document("source v2"), revision="p1")
    assert changed.version.version_id != first.version.version_id
    assert normalize_report(document(), revision="p1") == first


def test_legacy_missing_coordinates_are_not_invented():
    source = document()
    source["chunks"][0].metadata = {"metric": "Lợi nhuận", "end_value": "123",
                                    "start_value": "-", "unit": "shares"}
    source["chunks"][0].unit = "shares"
    bundle = normalize_report(source, revision="p1")
    assert len(bundle.facts) == 1 and bundle.facts[0].coordinate_status == "unlocated"
    assert bundle.facts[0].measurement == "unknown" and bundle.facts[0].scale is None
    assert bundle.warnings == {"unlocated_cell": 1, "unknown_unit": 1, "non_numeric_cell": 1}


def test_mismatched_source_cell_is_not_marked_located():
    source = document()
    source["chunks"][0].metadata["source_cells"][1] = "999"
    assert normalize_report(source).facts[0].coordinate_status == "unlocated"


@pytest.mark.parametrize("unit,measurement,scale", [
    ("VNĐ/cổ phiếu", "per_share", "1"), ("%", "percent", "1"),
    ("Tỷ đồng", "money", "1000000000"), (None, "unknown", None),
])
def test_dimensions_do_not_mix_money_percent_and_per_share(unit, measurement, scale):
    source = document()
    source["chunks"][0].unit = unit
    fact = normalize_report(source).facts[0]
    assert (fact.measurement, fact.scale) == (measurement, scale)


def test_sqlite_restated_source_and_old_values_remain_queryable(tmp_path):
    store = SQLiteFinancialStore(tmp_path / "reports.sqlite3")
    store.replace_documents([document()])
    first = store.list_documents()[0]["active_projection_id"]
    store.replace_documents([document("source v2", value="200")])
    assert len(store.list_versions("report")) == 2
    assert store.list_facts("report", period_year=2025)["facts"][0]["value"] == "200"
    old = store.list_facts("report", period_year=2025, projection_id=first)
    assert not old["active"] and old["facts"][0]["value"] == "-1234.56"
    store.replace_documents([document("source v2", value="200")])
    assert len(store.list_versions("report")) == 2
    with pytest.raises(ValueError):
        store.list_facts("another_report", projection_id=first)
    assert store.delete_document("report")
    assert store.list_facts("report")["total"] == 0
    assert len(store.list_versions("report")) == 2


def test_invalid_replace_rolls_back_active_and_history(tmp_path):
    store = SQLiteFinancialStore(tmp_path / "reports.sqlite3")
    store.replace_documents([document()])
    invalid = document("source v2")
    invalid["chunks"] *= 2  # Unique chunk constraint must reject the whole transaction.
    with pytest.raises(Exception):
        store.replace_documents([invalid])
    assert len(store.list_versions("report")) == 1
    assert store.load_chunks()[0].chunk_id == "row"


def test_coordinate_backfill_preserves_chunk_identity_and_content():
    raw = """===== PAGE 2 =====
FPT - Báo cáo tài chính hợp nhất năm 2025
BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH
Đơn vị: Triệu đồng
| Chỉ tiêu | 2025 | 2024 |
| --- | --- | --- |
| Lợi nhuận sau thuế | 123 | 100 |
"""
    parsed = UniversalFinancialChunker().chunk_text(raw, "FPT_2025.txt", company_code="FPT")
    row = next(chunk for chunk in parsed if chunk.chunk_type == ChunkType.FINANCIAL_ITEM)
    row.chunk_id = "legacy_id"
    row.content = "legacy evidence"
    for key in ("table_id", "source_cells", "source_row_index", "source_period_columns"):
        row.metadata.pop(key, None)
    source = document(raw)
    source["chunks"] = [row]
    source["document_id"] = row.document_id
    updated, stats = backfill_coordinates(source)
    assert stats == {"matched_rows": 1}
    assert updated[0].chunk_id == "legacy_id" and updated[0].content == "legacy evidence"
    assert "source_cells" not in row.metadata
    assert all(fact.coordinate_status == "located" for fact in normalize_report({**source, "chunks": updated}).facts)
