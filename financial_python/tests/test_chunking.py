from financial_rag.chunking.financial_chunker import FinancialMarkdownChunker
from financial_rag.schemas import ChunkType, StatementType
from financial_rag.chunking.financial_item_chunker import FinancialItemChunker
from financial_rag.metadata.period_extractor import PeriodExtractor
from financial_rag.parsing.markdown_parser import MarkdownFinancialParser
from financial_rag.structure.statement_detector import StatementDetector

def test_chunking_types():
    sample_text = """# Page 9
## Content
**BÁO CÁO TÌNH HÌNH TÀI CHÍNH HỢP NHẤT**
tại ngày 31 tháng 12 năm 2025

*   **TỔNG NỢ PHẢI TRẢ**
    *   Thuyết minh: N/A
    *   Số cuối năm (Triệu đồng): 750.000
    *   Số đầu năm (Triệu đồng): 620.000

# Page 20
## Content
**THUYẾT MINH BÁO CÁO TÀI CHÍNH**
### 17. Các khoản nợ Chính phủ và NHNN
*   **Vay chiết khấu giấy tờ có giá**
    *   Số cuối năm: 45.000
    *   Số đầu năm: 30.000
"""
    chunker = FinancialMarkdownChunker()
    chunks = chunker.chunk_text(sample_text, source_file="test.md", company_code="DEMO")

    types_found = {c.chunk_type for c in chunks}
    assert ChunkType.FINANCIAL_ITEM in types_found or ChunkType.TABLE in types_found

    # Kiểm tra metadata
    for c in chunks:
        assert c.company_code == "DEMO"
        assert c.fiscal_year == 2025
        assert c.page in [9, 20]


def test_docreader_html_page_markers_preserve_pdf_citation_pages():
    sample_text = """<!-- page: 9 -->
**BÁO CÁO TÌNH HÌNH TÀI CHÍNH HỢP NHẤT**
**TỔNG TÀI SẢN**
*   Số cuối năm: 250
*   Số đầu năm: 200

<!-- page: 14 -->
**THUYẾT MINH BÁO CÁO TÀI CHÍNH**
### 1. Cơ sở lập báo cáo
Thông tin được trình bày theo chuẩn mực kế toán.
"""

    pages = MarkdownFinancialParser.parse_text(sample_text)
    assert [page.page_number for page in pages] == [9, 14]
    assert all("<!-- page:" not in page.content for page in pages)

    chunks = FinancialMarkdownChunker().chunk_text(
        sample_text,
        source_file="FPT_BCTC_2025.pdf",
        company_code="FPT",
    )
    item = next(chunk for chunk in chunks if chunk.primary_metrics == ["TỔNG TÀI SẢN"])
    assert item.page == 9


def test_lf_and_crlf_markdown_produce_identical_pages_and_chunks():
    lf_text = (
        "<!-- page: 9 -->\n"
        "**BÁO CÁO TÌNH HÌNH TÀI CHÍNH HỢP NHẤT NĂM 2025**\n"
        "Đơn vị: Triệu đồng\n"
        "**TỔNG TÀI SẢN**\n"
        "*   Số cuối năm: 250\n"
        "*   Số đầu năm: 200\n"
        "<!-- page: 14 -->\n"
        "**THUYẾT MINH BÁO CÁO TÀI CHÍNH**\n"
        "### 1. Cơ sở lập báo cáo\n"
        "Thông tin được trình bày theo chuẩn mực kế toán.\n"
    )
    crlf_text = lf_text.replace("\n", "\r\n")

    lf_pages = MarkdownFinancialParser.parse_text(lf_text)
    crlf_pages = MarkdownFinancialParser.parse_text(crlf_text)
    assert [(page.page_number, page.content) for page in crlf_pages] == [
        (page.page_number, page.content) for page in lf_pages
    ]

    chunker = FinancialMarkdownChunker()
    lf_chunks = chunker.chunk_text(
        lf_text,
        source_file="FPT_BCTC_2025.md",
        source_pdf="FPT_BCTC_2025.pdf",
        company_code="FPT",
    )
    crlf_chunks = chunker.chunk_text(
        crlf_text,
        source_file="FPT_BCTC_2025.md",
        source_pdf="FPT_BCTC_2025.pdf",
        company_code="FPT",
    )
    assert crlf_chunks == lf_chunks


def test_parent_child_ablation_changes_note_chunks_only_when_enabled():
    text = (
        "# Page 20\n"
        "**THUYẾT MINH BÁO CÁO TÀI CHÍNH**\n"
        "### Thuyết minh 17: Các khoản nợ Chính phủ\n"
        "*   **Vay chiết khấu giấy tờ có giá**\n"
        "    *   Số cuối năm: 45.000\n"
        "    *   Số đầu năm: 30.000\n"
    )

    with_parent_child = FinancialMarkdownChunker(include_parent_child=True).chunk_text(
        text, source_file="DEMO_SYNTHETIC_2025.md", company_code="DEMO"
    )
    without_parent_child = FinancialMarkdownChunker(include_parent_child=False).chunk_text(
        text, source_file="DEMO_SYNTHETIC_2025.md", company_code="DEMO"
    )

    assert FinancialMarkdownChunker().include_parent_child is False
    assert any(chunk.chunk_type == ChunkType.PARENT_CHILD for chunk in with_parent_child)
    assert all(chunk.chunk_type != ChunkType.PARENT_CHILD for chunk in without_parent_child)


def test_filename_year_and_parenthesized_negative_value():
    assert PeriodExtractor.extract_fiscal_year("", "DEMO_SYNTHETIC_2024.md") == 2024

    chunks = FinancialItemChunker().chunk(
        "* **LỖ LŨY KẾ**\n    * Số cuối năm: (1.234)\n",
        source_file="DEMO_SYNTHETIC_2025.md",
        page=3,
        company_code="DEMO",
        fiscal_year=2025,
    )

    assert len(chunks) == 1
    assert chunks[0].metadata["end_value"] == "(1.234)"


def test_standalone_total_asset_is_extracted_as_financial_item():
    chunks = FinancialItemChunker().chunk(
        "**TỔNG TÀI SẢN**\n"
        "*   Số cuối năm: 1.250 triệu đồng\n"
        "*   Số đầu năm: 1.000 triệu đồng\n",
        source_file="DEMO_SYNTHETIC_2025.md",
        page=8,
        company_code="DEMO",
        fiscal_year=2025,
        unit="Triệu đồng",
    )

    assert len(chunks) == 1
    assert chunks[0].primary_metrics == ["TỔNG TÀI SẢN"]
    assert chunks[0].metadata["end_value"] == "1.250"
    assert chunks[0].metadata["start_value"] == "1.000"


def test_legacy_fiscal_years_are_extracted_from_filename_and_explicit_text():
    assert PeriodExtractor.extract_fiscal_year("", "VCB_BCTC_1999.pdf") == 1999
    assert PeriodExtractor.extract_fiscal_year("Báo cáo cho năm tài chính 2009") == 2009


def test_comparative_year_is_extracted_from_annual_report_markers():
    assert PeriodExtractor.extract_comparative_fiscal_year(
        "Báo cáo 2025, có số đầu năm và năm trước", 2025
    ) == 2024


def test_report_type_filename_overrides_incidental_ocr_reference():
    text = "Thuyết minh có nhắc đến báo cáo hợp nhất của tập đoàn."

    assert PeriodExtractor.extract_report_type(text, "DEMO_2024_Congtyme.txt") == "separate"
    assert PeriodExtractor.extract_report_type(text, "DEMO_2024_Hopnhat.txt") == "consolidated"
    assert PeriodExtractor.extract_report_type(text, "DEMO BCTC rieng le kiem toan 2022.txt") == "separate"


def test_markdown_chunker_uses_separate_report_type_from_filename():
    text = (
        "# Page 7\n"
        "**BÁO CÁO TÌNH HÌNH TÀI CHÍNH HỢP NHẤT**\n"
        "**TỔNG TÀI SẢN**\n"
        "*   Số cuối năm: 250\n"
        "*   Số đầu năm: 200\n"
    )

    chunks = FinancialMarkdownChunker().chunk_text(
        text, source_file="DEMO_SYNTHETIC_2024_Congtyme.md", company_code="DEMO"
    )

    assert chunks
    assert all(chunk.report_type == "separate" for chunk in chunks)


def test_income_statement_detector_ignores_a_note_that_mentions_the_statement():
    note_page = (
        "THUYẾT MINH BÁO CÁO TÀI CHÍNH\n"
        "Thu nhập thuế hoãn lại ảnh hưởng báo cáo kết quả hoạt động."
    )

    assert StatementDetector.detect(note_page) == StatementType.NOTES


def test_explicit_note_form_overrides_a_referenced_income_statement():
    note_page = (
        "THUYẾT MINH BẢO CẢO TÀI CHÍNH HỢP NHẤT (tiếp theo)\n"
        "B05/TCTD-HN\n"
        "Báo cáo kết quả hoạt động kinh doanh hợp nhất được trình bày ở phần trước."
    )

    assert StatementDetector.detect(note_page) == StatementType.NOTES


def test_financial_item_labels_comparative_value_with_its_fiscal_year():
    chunks = FinancialItemChunker().chunk(
        "**TỔNG TÀI SẢN**\n"
        "*   Số cuối năm: 250\n"
        "*   Số đầu năm: 200\n",
        source_file="FPT_BCTC_2025.md",
        page=8,
        company_code="FPT",
        fiscal_year=2025,
        comparative_fiscal_year=2024,
        unit="VND",
    )

    assert len(chunks) == 1
    assert chunks[0].comparative_fiscal_year == 2024
    assert chunks[0].metadata["period_values"] == {"2025": "250", "2024": "200"}
    assert "Số đầu năm (kỳ 2025): 200 (VND)" in chunks[0].content
    assert "Giá trị cuối năm 2024 (cột so sánh): 200 (VND)" in chunks[0].content


def test_per_share_metric_uses_per_share_unit():
    chunks = FinancialItemChunker().chunk(
        "**LÃI CƠ BẢN TRÊN CỔ PHIẾU**\n*   Năm nay: 5,216\n*   Năm trước: 4,292\n",
        source_file="FPT_BCTC_2025.md",
        page=171,
        statement_type=StatementType.INCOME_STATEMENT,
        fiscal_year=2025,
        unit="VND",
    )

    assert len(chunks) == 1
    assert chunks[0].unit == "VNĐ/cổ phiếu"
