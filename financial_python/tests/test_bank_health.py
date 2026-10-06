"""Regressions for a fictional bank-health question, including misleading note rows."""
import importlib
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from financial_rag.retrieval.hybrid import FinancialHybridRetriever
from financial_rag.schemas import ChunkType, FinancialChunk, StatementType, Citation
from financial_rag.analysis.evidence import CalculationEvidence, CalculationInput, validate_calculation


api = importlib.import_module("financial_rag.api.app")
QUESTION = "Đánh giá sức khỏe tài chính DEMO năm 2025 theo báo cáo hợp nhất"


def bank_rows():
    # Amounts and rows form a fictional bank fixture for offline regressions.
    records = [
        ("profit", "LỢI NHUẬN SAU THUẾ", "2.000", "1.500", "income_statement", "INCOME.BANK.PROFIT_AFTER_TAX", 12),
        ("interest", "Thu nhập lãi thuần", "8.000", "7.500", "income_statement", "INCOME.BANK.NET_INTEREST_INCOME", 12),
        ("income", "TỔNG THU NHẬP HOẠT ĐỘNG", "10.000", "9.000", "income_statement", "INCOME.BANK.TOTAL_OPERATING_INCOME", 12),
        ("expense", "TỔNG CHI PHÍ HOẠT ĐỘNG", "(2.500)", "(1.800)", "income_statement", "INCOME.BANK.OPERATING_EXPENSE", 12),
        ("eps", "Lãi cơ bản trên cổ phiếu", "10", "8", "income_statement", "INCOME.SECURITIES.BASIC_EPS", 12),
        ("assets", "TỔNG TÀI SẢN", "120.000", "100.000", "balance_sheet", "", 9),
        ("loans-net", "Cho vay khách hàng", "70.000", "60.000", "balance_sheet", "", 9),
        ("loans-gross", "Cho vay khách hàng", "72.000", "61.000", "balance_sheet", "", 9),
        ("deposits", "Tiền gửi của khách hàng", "60.000", "58.000", "balance_sheet", "", 10),
        ("liabilities", "TỔNG NỢ PHẢI TRẢ", "95.000", "80.000", "balance_sheet", "", 10),
        ("equity", "TỔNG VỐN CHỦ SỞ HỮU", "25.000", "20.000", "balance_sheet", "", 10),
        ("share-capital", "Vốn", "12.000", "12.000", "balance_sheet", "", 10),
        ("allowance", "Dự phòng rủi ro cho vay khách hàng", "(2.000)", "(1.000)", "balance_sheet", "", 9),
        ("cash-beginning", "Tiền và các khoản tương đương tiền đầu năm", "1.500", "1.200", "cash_flow", "", 14),
        ("cash-ending", "Tiền và các khoản tương đương tiền cuối năm", "2.000", "1.500", "cash_flow", "", 14),
        ("cfo", "Lưu chuyển tiền thuần từ hoạt động kinh doanh", "500", "400", "cash_flow", "", 13),
        ("standard", "Nợ đủ tiêu chuẩn", "67.880", "57.105", "notes", "BANK.NOTE.8.STANDARD_DEBT", 45),
        ("special", "Nợ cần chú ý", "720", "610", "notes", "BANK.NOTE.8.SPECIAL_MENTION", 45),
        ("substandard", "Nợ dưới tiêu chuẩn", "700", "900", "notes", "BANK.NOTE.8.SUBSTANDARD", 45),
        ("doubtful", "Nợ nghi ngờ", "350", "600", "notes", "BANK.NOTE.8.DOUBTFUL", 45),
        ("loss", "Nợ có khả năng mất vốn", "350", "285", "notes", "BANK.NOTE.8.LOSS", 45),
        ("margin", "Cho vay giao dịch ký quỹ và ứng trước cho khách hàng", "2.000", "1.500", "notes", "", 45),
        ("bond-loss", "Nợ có khả năng mất vốn", "5", "5", "notes", "", 55),
        ("loan-term", "Nợ ngắn hạn", "30.000", "25.000", "notes", "", 45),
    ]
    rows = []
    for identifier, label, current, prior, statement, node, page in records:
        rows.append(FinancialChunk(
            chunk_id=identifier, document_id="demo-consolidated-2025",
            source_file="DEMO_SYNTHETIC_2025.txt", company_code="DEMO",
            fiscal_year=2025, comparative_fiscal_year=2024, report_type="consolidated",
            page=page, statement_type=StatementType(statement), chunk_type=ChunkType.FINANCIAL_ITEM,
            unit="Triệu đồng", primary_metrics=[label],
            content=f"Chỉ tiêu gốc trong BCTC: {label}\nNăm nay (2025): {current} Triệu đồng\nNăm trước (2024): {prior} Triệu đồng",
            metadata={
                "metric_name": label, "schema_node_id": node,
                "table_id": f"demo:p{page}:classification" if page == 45 and identifier != "loan-term" else f"demo:p{page}:other",
                "period_values": {"2025": current, "2024": prior},
                "end_value": current, "start_value": prior,
            },
        ))
    return rows


def ask(monkeypatch, rows, question=QUESTION):
    monkeypatch.setattr(api, "CHUNKS_DB", rows)
    monkeypatch.setattr(api, "retriever", FinancialHybridRetriever(rows))
    response = TestClient(api.app).post("/api/v1/financial/answer", json={"query": question})
    assert response.status_code == 200, response.text
    return response.json()


def test_demo_health_uses_bank_evidence_and_scoped_comparatives(monkeypatch):
    rows = bank_rows()
    separate = [row.model_copy(deep=True) for row in rows]
    for row in separate:
        row.chunk_id += "-separate"
        row.document_id = "demo-separate-2025"
        row.source_file = "DEMO_2025_Congtyme.txt"
        row.report_type = "separate"
    data = ask(monkeypatch, rows + separate)
    assert data["validation"]["is_valid"], data["validation"]
    assert data["parsed_query"]["fiscal_years"] == [2025]
    assert data["parsed_query"]["report_type"] == "consolidated"
    assert "Chưa truy xuất được số liệu" not in data["answer"]
    assert "2.000" in data["answer"]
    assert "chất lượng tín dụng cải thiện đồng đều" in data["answer"]
    cited = {item["chunk_id"] for item in data["citations"]}
    assert "cash-ending" in cited and "cash-beginning" not in cited
    assert "loans-gross" in cited and "loans-net" not in cited
    assert "equity" in cited and "share-capital" not in cited
    assert "bond-loss" not in cited and "loan-term" not in cited
    assert all(item["file_name"] == "DEMO_SYNTHETIC_2025.txt" for item in data["citations"])
    npl = [item for item in data["calculations"] if item["formula_id"] == "nonperforming_loans_ratio_percent"]
    assert {item["value"] for item in npl} == {"2.00", "3.00"}
    assert all(set(item["input_chunk_ids"]) == {"standard", "special", "substandard", "doubtful", "loss"} for item in npl)
    assert data["answer_status"] == "partial"
    assert any("CAR" in item for item in data["analysis_coverage"]["missing_components"])


@pytest.mark.parametrize("broken", ["standard", "margin"])
def test_credit_ratio_is_withheld_when_classification_does_not_reconcile(monkeypatch, broken):
    rows = bank_rows()
    for row in rows:
        if row.chunk_id == broken:
            row.metadata["period_values"]["2025"] = "1"
            row.metadata["end_value"] = "1"
            row.content = row.content.split("\n")[0] + "\nNăm nay (2025): 1 Triệu đồng"
    data = ask(monkeypatch, rows)
    assert data["validation"]["is_valid"], data["validation"]
    assert not any(item["formula_id"] == "nonperforming_loans_ratio_percent" for item in data["calculations"])
    assert "bảng phân loại nợ đối chiếu được" in data["answer"]


def test_metadata_only_prior_values_do_not_create_comparison(monkeypatch):
    rows = bank_rows()
    for row in rows:
        row.content = row.content.split("\nNăm trước")[0]
    data = ask(monkeypatch, rows)
    assert data["validation"]["is_valid"], data["validation"]
    assert "Đối chiếu năm 2024" not in data["answer"]
    assert not any(item["formula_id"] == "growth_rate" for item in data["calculations"])


def test_credit_evaluation_with_ratio_language_uses_customer_loans(monkeypatch):
    data = ask(monkeypatch, bank_rows(),
        "Đánh giá chất lượng tín dụng DEMO năm 2025 theo báo cáo hợp nhất: tỷ lệ nợ xấu giảm có đi cùng giảm nợ có khả năng mất vốn không?")
    assert data["validation"]["is_valid"], data["validation"]
    assert "2,00%" in data["answer"]
    assert "chưa thể kết luận chất lượng tín dụng cải thiện đồng đều" in data["answer"]
    assert "bond-loss" not in {row["chunk_id"] for row in data["citations"]}
    assert "Cơ cấu các khoản mục" not in data["answer"]


def test_average_roa_roe_have_three_cited_inputs_and_correct_years(monkeypatch):
    data = ask(monkeypatch, bank_rows(),
        "Đánh giá khả năng sinh lời DEMO năm 2025 theo báo cáo hợp nhất, tính ROA và ROE trên số dư bình quân đầu và cuối năm.")
    assert data["validation"]["is_valid"], data["validation"]
    assert data["parsed_query"]["company_codes"] == ["DEMO"]
    ratios = {row["formula_id"]: row for row in data["calculations"] if row["formula_id"].startswith("return_on_average")}
    assert ratios["return_on_average_assets_after_tax_percent"]["value"] == "1.82"
    assert ratios["return_on_average_equity_after_tax_percent"]["value"] == "8.89"
    for row in ratios.values():
        inputs = {item["role"]: item for item in row["inputs"]}
        assert inputs["opening_balance"]["year"] == 2024
        assert inputs["closing_balance"]["year"] == inputs["net_profit_after_tax"]["year"] == 2025


def test_average_returns_do_not_fall_back_to_closing_balances(monkeypatch):
    rows = bank_rows()
    for row in rows:
        row.content = row.content.split("\nNăm trước")[0]
    data = ask(monkeypatch, rows, "Đánh giá ROA và ROE của DEMO năm 2025 theo báo cáo hợp nhất.")
    assert data["validation"]["is_valid"], data["validation"]
    assert data["answer_status"] == "partial"
    assert not any(row["formula_id"].startswith("return_on_average") for row in data["calculations"])
    assert any("ROA" in item for item in data["analysis_coverage"]["missing_components"])


def test_cir_uses_total_expense_absolute_value(monkeypatch):
    data = ask(monkeypatch, bank_rows(), "Đánh giá hiệu quả hoạt động DEMO năm 2025 theo báo cáo hợp nhất, tính CIR.")
    assert data["validation"]["is_valid"], data["validation"]
    assert data["parsed_query"]["company_codes"] == ["DEMO"]
    ratios = [row for row in data["calculations"] if row["formula_id"] == "cost_to_income_percent"]
    assert {row["value"] for row in ratios} == {"20.00", "25.00"}
    assert all(set(row["input_chunk_ids"]) == {"expense", "income"} for row in ratios)
    assert "CIR tăng" in data["answer"]
    assert "Giảm: TỔNG CHI PHÍ HOẠT ĐỘNG" not in data["answer"]


def test_loan_growth_reconciles_net_and_gross_without_credit_question(monkeypatch):
    data = ask(monkeypatch, bank_rows(), "Đánh giá tăng trưởng tín dụng và tiền gửi khách hàng DEMO năm 2025 theo báo cáo hợp nhất.")
    assert data["validation"]["is_valid"], data["validation"]
    assert "18,03%" in data["answer"] and "3,45%" in data["answer"]
    assert "loans-gross" in {row["chunk_id"] for row in data["citations"]}
    assert "loans-net" not in {row["chunk_id"] for row in data["citations"]}
    assert "Không đưa vào phép tính" not in data["answer"]
    assert "tăng trưởng nhanh hơn tiền gửi khách hàng" in data["answer"]


@pytest.mark.parametrize("invalid", ["zero", "negative", "unit"])
def test_average_roa_withholds_invalid_balance_or_unit(monkeypatch, invalid):
    rows = bank_rows()
    assets = next(row for row in rows if row.chunk_id == "assets")
    if invalid == "unit":
        assets.unit = "Tỷ đồng"
    else:
        amount = "0" if invalid == "zero" else "-1"
        assets.metadata["period_values"]["2024"] = amount
        assets.metadata["start_value"] = amount
        assets.content = assets.content.split("\nNăm trước")[0] + f"\nNăm trước (2024): {amount} Triệu đồng"
    data = ask(monkeypatch, rows, "Đánh giá ROA của DEMO năm 2025 theo báo cáo hợp nhất.")
    assert data["validation"]["is_valid"], data["validation"]
    assert data["answer_status"] == "partial"
    assert not any(row["formula_id"].startswith("return_on_average") for row in data["calculations"])


def test_average_return_validator_rejects_mixed_filings_and_wrong_period(monkeypatch):
    rows = bank_rows()
    data = ask(monkeypatch, rows, "Đánh giá ROA của DEMO năm 2025 theo báo cáo hợp nhất.")
    raw = next(row for row in data["calculations"] if row["formula_id"] == "return_on_average_assets_after_tax_percent")
    item = CalculationEvidence(**{
        **{key: value for key, value in raw.items() if key != "expression"},
        "inputs": tuple(CalculationInput(**fact) for fact in raw["inputs"]),
    })
    citations = {row["citation_id"]: Citation.model_validate(row) for row in data["citations"]}
    chunks = {row.chunk_id: row for row in rows}
    assert validate_calculation(item, citations, chunks)
    altered_chunks = {identifier: chunk.model_copy(deep=True) for identifier, chunk in chunks.items()}
    altered_chunks["assets"].document_id = "another-filing"
    assert not validate_calculation(item, citations, altered_chunks)
    profit, opening, closing = item.inputs
    wrong_period = replace(item, inputs=(profit, replace(opening, year=2025, value=closing.value), closing))
    assert not validate_calculation(wrong_period, citations, chunks)


def test_roa_does_not_use_repeated_total_in_maturity_note(monkeypatch):
    rows = bank_rows()
    decoy = next(row for row in rows if row.chunk_id == "assets").model_copy(deep=True)
    decoy.chunk_id = "assets-maturity-note"
    decoy.page = 100
    decoy.metadata.update({
        "metric_name": "Tổng tài sản (1)", "table_id": "demo:p100:table_1",
        "table_title": "Bảng dưới đây trình bày thời gian đáo hạn của tài sản và nợ phải trả tại ngày 31 tháng 12 năm 2025:",
        "period_values": {"2025": "550.000", "2024": "700.000"},
        "end_value": "550.000", "start_value": "700.000",
    })
    decoy.content = "Chỉ tiêu gốc trong BCTC: Tổng tài sản (1)\nNăm nay (2025): 550.000 Triệu đồng\nNăm trước (2024): 700.000 Triệu đồng"
    data = ask(monkeypatch, rows + [decoy], "Đánh giá ROA của DEMO năm 2025 theo báo cáo hợp nhất.")
    assert data["validation"]["is_valid"], data["validation"]
    assert "1,82%" in data["answer"]
    assert decoy.chunk_id not in {row["chunk_id"] for row in data["citations"]}
