from pathlib import Path

from financial_rag.parsing.txt_table_parser import (
    TxtFinancialParser,
    HTMLTableParser,
    raw_table_to_matrix,
    TableBlock,
)
from financial_rag.chunking.txt_financial_chunker import TxtFinancialChunker
from financial_rag.chunking.universal_chunker import UniversalFinancialChunker
from financial_rag.schemas import StatementType, ChunkType


def test_txt_parser_page_split():
    raw_txt = """
===== PAGE 1 =====
BÁO CÁO TÀI CHÍNH NĂM 2019
TỔNG CÔNG TY CẢNG HÀNG KHÔNG VIỆT NAM - CTCP

===== PAGE 2 =====
MỤC LỤC
<table><tr><td>NỘI DUNG</td><td>TRANG</td></tr><tr><td>BẢNG CÂN ĐỐI KẾ TOÁN</td><td>5 - 6</td></tr></table>
"""
    pages = TxtFinancialParser.parse_text(raw_txt)
    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert "BÁO CÁO TÀI CHÍNH" in pages[0].content
    assert pages[1].page_number == 2
    assert len(pages[1].tables) == 1
    assert pages[1].tables[0].headers == ["NỘI DUNG", "TRANG"]


def test_html_table_parser_colspan_rowspan():
    table_html = """
<table>
  <tr><td>TÀI SẢN</td><td colspan="2">Mã Thuyết số minh</td><td>Số cuối năm</td><td>Số đầu năm</td></tr>
  <tr><td>A. TÀI SẢN NGẮN HẠN</td><td>100</td><td></td><td>987.654.321.000</td><td>876.543.210.000</td></tr>
  <tr><td>I. Tiền &amp; tương đương tiền</td><td>110</td><td>5</td><td>123.456.789.000</td><td>98.765.432.100</td></tr>
</table>
"""
    parser = HTMLTableParser()
    parser.feed(table_html)
    assert len(parser.tables) == 1
    raw_rows = parser.tables[0]
    matrix = raw_table_to_matrix(raw_rows)

    assert len(matrix) == 3
    # Check that colspan=2 on 'Mã Thuyết số minh' expanded cleanly to 5 columns
    assert matrix[0] == ["TÀI SẢN", "Mã số", "Thuyết minh", "Số cuối năm", "Số đầu năm"]
    assert matrix[1] == ["A. TÀI SẢN NGẮN HẠN", "100", "", "987.654.321.000", "876.543.210.000"]
    assert matrix[2] == ["I. Tiền & tương đương tiền", "110", "5", "123.456.789.000", "98.765.432.100"]


def test_table_block_to_markdown():
    matrix = [
        ["CHỈ TIÊU", "Mã số", "Năm nay", "Năm trước"],
        ["Doanh thu thuần", "10", "900", "800"],
    ]
    tb = TableBlock(
        raw_html="<table>...</table>",
        matrix=matrix,
        page_number=9,
        title="BÁO CÁO KẾT QUẢ KINH DOANH",
        unit="Triệu đồng",
    )
    md = tb.to_markdown()
    assert "### BÁO CÁO KẾT QUẢ KINH DOANH" in md
    assert "(Đơn vị: Triệu đồng)" in md
    assert "| CHỈ TIÊU | Mã số | Năm nay | Năm trước |" in md
    assert "| Doanh thu thuần | 10 | 900 | 800 |" in md


def test_txt_chunker_atomic_financial_items():
    sample_txt = """
===== PAGE 8 =====
CÔNG TY HƯ CẤU MINH HỌA
BẢNG CÂN ĐỐI KẾ TOÁN HỢP NHẤT
Tại ngày 31 tháng 12 năm 2025
Đơn vị: Triệu đồng

<table>
  <tr><td>NGUỒN VỐN</td><td>Mã số</td><td>Thuyết minh</td><td>Số cuối năm</td><td>Số đầu năm</td></tr>
  <tr><td>C. NỢ PHẢI TRẢ</td><td>300</td><td></td><td>750.000</td><td>620.000</td></tr>
  <tr><td>I. Nợ ngắn hạn</td><td>310</td><td></td><td>500.000</td><td>400.000</td></tr>
  <tr><td>1. Phải trả người bán ngắn hạn</td><td>311</td><td>19</td><td>80.000</td><td>65.000</td></tr>
</table>
"""
    chunker = TxtFinancialChunker()
    chunks = chunker.chunk_text(sample_txt, source_file="DEMO_SYNTHETIC_2025.txt", company_code="DEMO")

    item_chunks = [c for c in chunks if c.chunk_type == ChunkType.FINANCIAL_ITEM]
    assert len(item_chunks) == 3

    # Check Nợ phải trả
    debt_chunk = next(c for c in item_chunks if "C. NỢ PHẢI TRẢ" in c.content)
    assert debt_chunk.statement_type == StatementType.BALANCE_SHEET
    assert debt_chunk.page == 8
    assert debt_chunk.metadata["metric_name"] == "C. NỢ PHẢI TRẢ"
    assert debt_chunk.metadata["code"] == "300"
    assert debt_chunk.metadata["end_value"] == "750.000"
    assert debt_chunk.metadata["start_value"] == "620.000"
    assert "Số cuối năm (2025): 750.000" in debt_chunk.content


def test_txt_chunker_finds_metric_column_after_blank_ocr_headers():
    sample_txt = """===== PAGE 7 =====
BÁO CÁO TÀI CHÍNH DOANH NGHIỆP MINH HỌA NĂM 2024
BẢNG CÂN ĐỐI KẾ TOÁN

<table>
  <tr><td></td><td></td><td>Thuyết minh</td><td>31/12/2024 Triệu VND</td><td>31/12/2023 Triệu VND</td></tr>
  <tr><td></td><td>TỔNG TÀI SẢN</td><td></td><td>1.250</td><td>1.000</td></tr>
  <tr><td></td><td>TỔNG NỢ PHẢI TRẢ</td><td></td><td>750</td><td>620</td></tr>
</table>
"""
    chunker = TxtFinancialChunker()
    chunks = chunker.chunk_text(
        sample_txt,
        source_file="DEMO_SYNTHETIC_2024.txt",
        company_code="DEMO",
    )
    items = [chunk for chunk in chunks if chunk.chunk_type == ChunkType.FINANCIAL_ITEM]

    assets = next(chunk for chunk in items if "TỔNG TÀI SẢN" in chunk.content)
    liabilities = next(chunk for chunk in items if "TỔNG NỢ PHẢI TRẢ" in chunk.content)
    assert assets.metadata["end_value"] == "1.250"
    assert assets.metadata["start_value"] == "1.000"
    assert assets.metadata["unit"] == "Triệu VND"
    assert "Số cuối năm (2024): 1.250 Triệu VND" in assets.content
    assert liabilities.metadata["end_value"] == "750"
    assert liabilities.metadata["start_value"] == "620"


def test_txt_chunker_uses_separate_report_type_from_filename():
    sample_txt = """===== PAGE 7 =====
BÁO CÁO TÀI CHÍNH MBB NĂM 2022
BÁO CÁO TÌNH HÌNH TÀI CHÍNH HỢP NHẤT
Đơn vị: Triệu đồng

<table>
  <tr><td>CHỈ TIÊU</td><td>Mã số</td><td>Thuyết minh</td><td>Số cuối năm</td><td>Số đầu năm</td></tr>
  <tr><td>TỔNG TÀI SẢN</td><td>100</td><td></td><td>250</td><td>200</td></tr>
</table>
"""

    chunks = TxtFinancialChunker().chunk_text(
        sample_txt,
        source_file="MBB_BCTC_rieng_le_kiem_toan_2022.txt",
        company_code="MBB",
    )

    assert chunks
    assert all(chunk.report_type == "separate" for chunk in chunks)


def test_txt_chunker_generates_unique_stable_ids_for_multiple_tables_on_one_page():
    sample_txt = """===== PAGE 8 =====
BÁO CÁO TÀI CHÍNH ABB NĂM 2024
BẢNG CÂN ĐỐI KẾ TOÁN
<table>
  <tr><td>CHỈ TIÊU</td><td>Mã số</td><td>Thuyết minh</td><td>Số cuối năm</td><td>Số đầu năm</td></tr>
  <tr><td>TỔNG TÀI SẢN</td><td>100</td><td></td><td>250</td><td>200</td></tr>
</table>
<table>
  <tr><td>CHỈ TIÊU</td><td>Mã số</td><td>Thuyết minh</td><td>Số cuối năm</td><td>Số đầu năm</td></tr>
  <tr><td>TỔNG NỢ PHẢI TRẢ</td><td>300</td><td></td><td>150</td><td>120</td></tr>
</table>
"""
    chunker = TxtFinancialChunker()
    first = chunker.chunk_text(sample_txt, source_file="ABB_BCTC_2024.txt")
    second = chunker.chunk_text(sample_txt, source_file="ABB_BCTC_2024.txt")

    first_ids = [chunk.chunk_id for chunk in first]
    assert first_ids
    assert len(set(first_ids)) == len(first_ids)
    assert [chunk.chunk_id for chunk in second] == first_ids


def test_txt_chunker_finds_period_columns_when_headers_only_say_tai_ngay():
    sample_txt = """===== PAGE 7 =====
BÁO CÁO TÀI CHÍNH ACB NĂM 2024
BÁO CÁO TÌNH HÌNH TÀI CHÍNH HỢP NHẤT
Đơn vị: Triệu đồng

<table>
  <tr><td></td><td></td><td>Thuyết minh</td><td>Tại ngày</td><td>Tại ngày</td></tr>
  <tr><td></td><td>TỔNG TÀI SẢN</td><td></td><td>3.000</td><td>2.500</td></tr>
</table>
"""
    chunks = TxtFinancialChunker().chunk_text(
        sample_txt,
        source_file="DEMO_SYNTHETIC_2025.txt",
        company_code="DEMO",
    )
    assets = next(
        chunk
        for chunk in chunks
        if chunk.chunk_type == ChunkType.FINANCIAL_ITEM
        and chunk.metadata["metric_name"] == "TỔNG TÀI SẢN"
    )

    assert assets.metadata["end_value"] == "3.000"
    assert assets.metadata["start_value"] == "2.500"


def test_txt_chunker_does_not_treat_footnote_text_as_financial_values():
    sample_txt = """===== PAGE 79 =====
BÁO CÁO TÀI CHÍNH ABB NĂM 2024
BẢNG THUYẾT MINH BÁO CÁO TÀI CHÍNH

<table>
  <tr><td>Chỉ tiêu</td><td>Mã số</td><td>Thuyết minh</td><td>Số cuối năm</td><td>Số đầu năm</td></tr>
  <tr><td>Tổng tài sản (1)</td><td>100</td><td>1</td><td>Tổng tài sản (1)</td><td>Tổng tài sản (2)</td></tr>
<tr><td>Tổng nợ phải trả</td><td>300</td><td></td><td>350.000</td><td>Tổng nợ phải trả (1)</td></tr>
</table>
"""
    chunks = TxtFinancialChunker().chunk_text(
        sample_txt,
        source_file="DEMO_SYNTHETIC_2025.txt",
        company_code="DEMO",
    )

    assert any(chunk.chunk_type == ChunkType.TABLE for chunk in chunks)
    items = [chunk for chunk in chunks if chunk.chunk_type == ChunkType.FINANCIAL_ITEM]
    assert len(items) == 1
    assert items[0].metadata["end_value"] == "350.000"
    assert items[0].metadata["start_value"] == ""


def test_markdown_chunker_synthetic_demo_file():
    demo_path = Path(__file__).resolve().parent.parent / "sample_data" / "DEMO_SYNTHETIC_2025.md"
    chunker = UniversalFinancialChunker()
    chunks = chunker.chunk_file(demo_path, company_code="DEMO")
    item_chunks = [c for c in chunks if c.chunk_type == ChunkType.FINANCIAL_ITEM]
    assert len(item_chunks) >= 7
    assert any("TỔNG NỢ PHẢI TRẢ" in c.content and "750" in c.content for c in item_chunks)
    assert any("DOANH THU THUẦN" in c.content and "900" in c.content for c in item_chunks)


def test_universal_chunker_auto_detect():
    universal = UniversalFinancialChunker()

    md_text = """# Page 1
## BÁO CÁO TÀI CHÍNH MINH HỌA 2025
* **TỔNG NỢ PHẢI TRẢ**
  * Số cuối năm: 750.000
  * Số đầu năm: 620.000
"""
    txt_text = """===== PAGE 1 =====
<table><tr><td>CHỈ TIÊU</td><td>Số cuối năm</td></tr><tr><td>Doanh thu</td><td>1000</td></tr></table>
"""

    md_chunks = universal.chunk_text(md_text, source_file="test.md")
    txt_chunks = universal.chunk_text(txt_text, source_file="test.txt")

    assert len(md_chunks) > 0
    assert len(txt_chunks) > 0
    assert any(c.chunk_type == ChunkType.FINANCIAL_ITEM for c in txt_chunks)


def test_txt_chunker_comparison_and_variance():
    sample_txt = """===== PAGE 9 =====
BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH
Năm 2020
Đơn vị: Triệu đồng

Đoạn văn thuyết minh dài về ban giám đốc và tình hình kinh doanh không được tạo chunk narrative.

<table>
  <tr><td>CHỈ TIÊU</td><td>Mã số</td><td>Thuyết minh</td><td>Năm nay</td><td>Năm trước</td></tr>
  <tr><td>Doanh thu thuần</td><td>10</td><td>25</td><td>120.000</td><td>100.000</td></tr>
  <tr><td>Chi phí tài chính</td><td>22</td><td>26</td><td>15.000</td><td>20.000</td></tr>
</table>
"""
    chunker = TxtFinancialChunker()
    chunks = chunker.chunk_text(sample_txt, source_file="BCTC_2020.txt", company_code="TEST")

    # Both atomic financial facts and surrounding explanatory prose should be
    # retained so narrative questions remain answerable from TXT reports.
    chunk_types = {c.chunk_type for c in chunks}
    assert ChunkType.FINANCIAL_ITEM in chunk_types
    assert ChunkType.NARRATIVE in chunk_types
    assert any("Đoạn văn thuyết minh dài" in c.content for c in chunks)

    items = [c for c in chunks if c.chunk_type == ChunkType.FINANCIAL_ITEM]
    rev_item = next(c for c in items if "Doanh thu thuần" in c.content)

    # 120.000 so với 100.000 -> Tăng 20.000 (+20.00%)
    assert rev_item.metadata["difference"] == 20000.0
    assert rev_item.metadata["pct_change"] == 20.0
    assert rev_item.metadata["trend"] == "Tăng"
    assert "So sánh biến động: Tăng 20.000 Triệu đồng (+20.00%) so với năm trước" in rev_item.content

    # 15.000 so với 20.000 -> Giảm 5.000 (-25.00%)
    exp_item = next(c for c in items if "Chi phí tài chính" in c.content)
    assert exp_item.metadata["difference"] == -5000.0
    assert exp_item.metadata["pct_change"] == -25.0
    assert exp_item.metadata["trend"] == "Giảm"
    assert "So sánh biến động: Giảm -5.000 Triệu đồng (-25.00%) so với năm trước" in exp_item.content


def test_txt_chunker_section_group_tracking():
    sample_txt = """===== PAGE 5 =====
BẢNG CÂN ĐỐI KẾ TOÁN
Năm 2020
Đơn vị: VND

<table>
  <tr><td>TÀI SẢN</td><td>Mã số</td><td>Số cuối năm</td><td>Số đầu năm</td></tr>
  <tr><td>A. TÀI SẢN NGẮN HẠN</td><td></td><td></td><td></td></tr>
  <tr><td>I. Tiền và các khoản tương đương tiền</td><td>110</td><td>500.000.000</td><td>300.000.000</td></tr>
  <tr><td>B. TÀI SẢN DÀI HẠN</td><td></td><td></td><td></td></tr>
  <tr><td>I. Tài sản cố định</td><td>220</td><td>800.000.000</td><td>750.000.000</td></tr>
</table>
"""
    chunker = TxtFinancialChunker()
    chunks = chunker.chunk_text(sample_txt, source_file="BCTC_2020.txt", company_code="TEST")
    items = [c for c in chunks if c.chunk_type == ChunkType.FINANCIAL_ITEM]
    assert len(items) == 2

    cash_item = next(c for c in items if "Tiền và các khoản tương đương tiền" in c.content)
    assert cash_item.metadata["section_group"] == "A. TÀI SẢN NGẮN HẠN"
    assert "Phân nhóm: A. TÀI SẢN NGẮN HẠN" in cash_item.content

    fa_item = next(c for c in items if "Tài sản cố định" in c.content)
    assert fa_item.metadata["section_group"] == "B. TÀI SẢN DÀI HẠN"
    assert "Phân nhóm: B. TÀI SẢN DÀI HẠN" in fa_item.content


def test_txt_chunker_filters_non_financial_tables():
    sample_txt = """===== PAGE 2 =====
MỤC LỤC
<table>
  <tr><td>NỘI DUNG</td><td>TRANG</td></tr>
  <tr><td>BÁO CÁO CỦA BAN TỔNG GIÁM ĐỐC</td><td>1 - 3</td></tr>
</table>

===== PAGE 3 =====
HỘI ĐỒNG QUẢN TRỊ
<table>
  <tr><td>Họ và tên</td><td>Chức vụ</td><td>Ngày bổ nhiệm</td></tr>
  <tr><td>Nguyễn Văn A</td><td>Chủ tịch</td><td>01/01/2018</td></tr>
</table>
"""
    chunker = TxtFinancialChunker()
    chunks = chunker.chunk_text(sample_txt, source_file="TOC.txt")
    # The table blocks are filtered, while their nearby headings remain
    # available as narrative text chunks.
    assert chunks
    assert all(chunk.chunk_type == ChunkType.NARRATIVE for chunk in chunks)
