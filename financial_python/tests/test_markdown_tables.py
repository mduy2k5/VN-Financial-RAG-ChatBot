import importlib

from fastapi.testclient import TestClient
import pytest

from evaluation.run_analysis_runtime_lifecycle import source
from financial_rag.chunking.universal_chunker import UniversalFinancialChunker
from financial_rag.parsing.txt_table_parser import TxtFinancialParser, markdown_table_spans
from financial_rag.retrieval.hybrid import FinancialHybridRetriever
from financial_rag.schemas import ChunkType


@pytest.mark.parametrize("extension", ["md", "txt"])
def test_new_pipe_or_html_table_is_answerable_immediately_without_schema(monkeypatch, extension):
    module = importlib.import_module("financial_rag.api.app")
    chunks = UniversalFinancialChunker().chunk_text(
        source("FPT", extension, 130), source_file=f"FPT_TEST_2029_Hopnhat.{extension}", company_code="FPT",
    )
    rows = [chunk for chunk in chunks if chunk.chunk_type == ChunkType.FINANCIAL_ITEM]
    assert len(rows) == 3
    assert [row.metadata["period_values"] for row in rows] == [
        {"2029": "80", "2028": "60"}, {"2029": "50", "2028": "40"}, {"2029": "130", "2028": "100"},
    ]
    assert all(row.comparative_fiscal_year == 2028 and row.page == 7 for row in rows)
    assert all(row.metadata["table_id"] for row in rows)
    for row in rows:
        for key in list(row.metadata):
            if key.startswith("schema_"):
                row.metadata.pop(key)
    monkeypatch.setattr(module, "CHUNKS_DB", chunks)
    monkeypatch.setattr(module, "retriever", FinancialHybridRetriever(chunks))
    payload = TestClient(module.app).post("/api/v1/financial/answer", json={
        "query": "Khoản nào đóng góp vào biến động nợ phải trả FPT năm 2028 và 2029",
    }).json()
    assert payload["validation"]["is_valid"], payload
    assert payload["answer_status"] == "complete", payload
    assert payload["calculations"][0]["value"] == "30"
    assert all(citation["file_name"] == f"FPT_TEST_2029_Hopnhat.{extension}" for citation in payload["citations"])


def test_pipe_parser_preserves_blank_and_escaped_cells_and_skips_fenced_examples():
    content = (
        "```markdown\n| Tên | 2025 |\n| --- | --- |\n| Minh họa | 999 |\n```\n"
        "# BẢNG CÂN ĐỐI KẾ TOÁN\nĐơn vị: Triệu đồng\n"
        "Chỉ tiêu | 2025 | 2024\n--- | ---: | ---:\nA\\|B | 10 | \nC | (5) | 2\n"
    )
    spans = markdown_table_spans(content)
    assert len(spans) == 1
    assert spans[0][2] == [["Chỉ tiêu", "2025", "2024"], ["A|B", "10", ""], ["C", "(5)", "2"]]
    pages = TxtFinancialParser.parse_text(content)
    assert len(pages[0].tables) == 1
    assert pages[0].tables[0].raw_html == content[spans[0][0]:spans[0][1]].rstrip()
    assert r"A\|B" in pages[0].tables[0].to_markdown()


def test_malformed_pipe_row_is_not_truncated_into_a_false_amount():
    content = "| Chỉ tiêu | 2025 | 2024 |\n| --- | --- | --- |\n| Sai | 10 | 20 | 30 |\n"
    assert markdown_table_spans(content) == []


def test_two_source_tables_on_same_page_keep_distinct_ids_and_bounds():
    content = source("FPT", "md", 130) + "\n\n## BẢNG CÂN ĐỐI KẾ TOÁN\nĐơn vị: Triệu đồng\n" + (
        "| Chỉ tiêu | Mã số | 31/12/2029 | 01/01/2029 |\n| --- | --- | --- | --- |\n"
        "| Vốn điều lệ | 411 | 300 | 200 |\n| TỔNG VỐN CHỦ SỞ HỮU | 400 | 300 | 200 |\n"
    )
    chunks = UniversalFinancialChunker().chunk_text(content, source_file="FPT_2029_Hopnhat.md", company_code="FPT")
    rows = [row for row in chunks if row.chunk_type == ChunkType.FINANCIAL_ITEM]
    assert len(rows) == 5
    assert len({row.chunk_id for row in rows}) == 5
    assert len({row.metadata["table_id"] for row in rows}) == 2
    module = importlib.import_module("financial_rag.api.app")
    parsed = module.query_engine.parse("Cơ cấu nợ phải trả FPT năm 2029")
    retriever = FinancialHybridRetriever(chunks)
    expanded = module._expand_vector_selected_tables(retriever.search(parsed, top_k=15), chunks, parsed)
    assert not any(row.chunk.metadata.get("metric_name") == "Vốn điều lệ" for row in expanded)


def test_opening_balance_date_is_not_confused_with_income_period_range():
    text = source("FPT", "md", 130).replace("BẢNG CÂN ĐỐI KẾ TOÁN", "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH")
    text = text.replace("31/12/2029", "01/01/2029 đến 31/12/2029").replace("01/01/2029 |", "Năm trước 2028 |")
    chunks = UniversalFinancialChunker().chunk_text(text, source_file="FPT_2029_Hopnhat.md", company_code="FPT")
    rows = [row for row in chunks if row.chunk_type == ChunkType.FINANCIAL_ITEM]
    assert rows[0].metadata["period_values"]["2029"] == "80"
    assert rows[0].metadata["period_values"]["2028"] == "60"
