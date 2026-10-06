from pathlib import Path
from hashlib import sha256
from typing import List, Optional

from ..schemas import FinancialChunk, StatementType
from ..structure.statement_detector import StatementDetector
from ..structure.section_detector import SectionDetector
from ..metadata.company_extractor import CompanyExtractor
from ..metadata.period_extractor import PeriodExtractor
from ..parsing.markdown_parser import MarkdownFinancialParser, normalize_newlines
from ..parsing.txt_table_parser import TxtFinancialParser, markdown_table_spans
from .txt_financial_chunker import TxtFinancialChunker

from .narrative_chunker import NarrativeChunker
from .table_chunker import TableChunker
from .financial_item_chunker import FinancialItemChunker
from .parent_child_chunker import ParentChildChunker

class FinancialMarkdownChunker:
    """
    Orchestrator chunking tổng hợp cho Báo cáo tài chính Việt Nam.
    Tích hợp 4 chiến lược chunking:
    1. TableChunker: Bảo toàn bảng số liệu.
    2. FinancialItemChunker: Trích xuất chỉ tiêu tài chính nguyên tử (Atomic Facts).
    3. NarrativeChunker: Xử lý thuyết minh văn bản, chính sách kế toán.
    4. ParentChildChunker: Xử lý cấu trúc phân cấp Thuyết minh -> Tiểu mục.
    """

    def __init__(self, *, include_parent_child: bool = False):
        self.narrative_chunker = NarrativeChunker()
        self.table_chunker = TableChunker()
        self.item_chunker = FinancialItemChunker()
        self.source_table_chunker = TxtFinancialChunker()
        self.parent_child_chunker = ParentChildChunker()
        self.company_extractor = CompanyExtractor()
        # Parent-child note chunks remain available as an explicit option;
        # the current 88-question fixture found no gain for the extra chunks.
        self.include_parent_child = include_parent_child

    def chunk_file(
        self,
        file_path: Path,
        company_code: Optional[str] = None,
        source_pdf: Optional[str] = None,
    ) -> List[FinancialChunk]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return self.chunk_text(
            content,
            source_file=file_path.name,
            company_code=company_code,
            source_pdf=source_pdf,
        )

    def chunk_text(
        self,
        text: str,
        source_file: str = "bctc.md",
        company_code: Optional[str] = None,
        source_pdf: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> List[FinancialChunk]:
        text = normalize_newlines(text)
        # Header OCR có thể kéo dài qua nhiều trang đầu; quét đủ rộng để gắn
        # metadata cho tài liệu mới có bìa/mục lục dài.
        metadata_text = text[:20_000]
        inferred_company = company_code or self.company_extractor.extract(
            text[:4_000], fallback_file_name=source_file
        ) or self.company_extractor.extract(metadata_text, fallback_file_name=source_file)
        fiscal_year = PeriodExtractor.extract_fiscal_year(metadata_text, fallback_file_name=source_file)
        comparative_fiscal_year = PeriodExtractor.extract_comparative_fiscal_year(text, fiscal_year)
        fiscal_quarter = PeriodExtractor.extract_document_quarter(metadata_text, source_file)
        report_type = PeriodExtractor.extract_report_type(
            metadata_text, fallback_file_name=source_file
        )
        stable_document_id = document_id or sha256(
            f"{source_file}|{inferred_company}|{fiscal_year}|{fiscal_quarter}|{report_type}".encode("utf-8")
        ).hexdigest()

        # 2. Parse theo từng trang
        pages = MarkdownFinancialParser.parse_text(text)
        all_chunks: List[FinancialChunk] = []

        current_statement = StatementType.GENERAL

        for page_obj in pages:
            page_num = page_obj.page_number
            page_text = page_obj.content

            # Cập nhật loại báo cáo nếu phát hiện tiêu đề mới
            detected_stmt = StatementDetector.detect(page_text)
            if detected_stmt != StatementType.GENERAL:
                current_statement = detected_stmt

            unit = SectionDetector.extract_unit(page_text)
            note_num = SectionDetector.extract_note_number(page_text)

            # Pipe tables need the same row/period extraction as HTML tables.
            # Keep the original Markdown and the existing bullet/narrative path.
            if markdown_table_spans(page_text):
                _, source_tables, _ = TxtFinancialParser._parse_page_content(page_text, page_num)
                for table_index, table in enumerate(source_tables):
                    if not self.source_table_chunker._is_financial_table(table):
                        continue
                    table_statement = StatementDetector.detect(table.title or "")
                    if table_statement == StatementType.GENERAL:
                        table_statement = current_statement
                    table_items = self.source_table_chunker._extract_atomic_items_from_table(
                        table=table, page_num=page_num, statement_type=table_statement,
                        company_code=inferred_company, fiscal_year=fiscal_year,
                        comparative_fiscal_year=comparative_fiscal_year, default_unit=unit,
                        source_file=source_file,
                    )
                    table_id = f"{stable_document_id}:p{page_num}:markdown_table_{table_index}"
                    for row in table_items:
                        row.chunk_id = f"{source_file}_markdown_table_{table_index}_{row.chunk_id}"
                        row.metadata.update({
                            "table_id": table_id, "source_start": table.source_start,
                            "source_end": table.source_end,
                        })
                    all_chunks.extend(table_items)

            # 3. Phân bổ chiến lược chunking dựa trên loại báo cáo và nội dung
            if current_statement in [StatementType.BALANCE_SHEET, StatementType.INCOME_STATEMENT, StatementType.CASH_FLOW]:
                # A. Tạo các Atomic Financial Item chunks để tra cứu chính xác
                item_chunks = self.item_chunker.chunk(
                    page_text,
                    source_file=source_file,
                    page=page_num,
                    statement_type=current_statement,
                    unit=unit,
                    company_code=inferred_company,
                    fiscal_year=fiscal_year,
                    comparative_fiscal_year=comparative_fiscal_year,
                )
                all_chunks.extend(item_chunks)

                # B. Tạo Table chunks để giữ toàn bộ bối cảnh bảng
                table_chunks = self.table_chunker.chunk(
                    page_text,
                    source_file=source_file,
                    page=page_num,
                    statement_type=current_statement,
                    unit=unit,
                    company_code=inferred_company,
                    fiscal_year=fiscal_year
                )
                all_chunks.extend(table_chunks)

            elif current_statement == StatementType.NOTES:
                if self.include_parent_child and note_num and ("*" in page_text or "|" in page_text):
                    # Thuyết minh có số hiệu cụ thể và danh sách mục con -> Parent-Child
                    pc_chunks = self.parent_child_chunker.chunk(
                        page_text,
                        source_file=source_file,
                        page=page_num,
                        statement_type=current_statement,
                        note_number=note_num,
                        company_code=inferred_company,
                        fiscal_year=fiscal_year
                    )
                    all_chunks.extend(pc_chunks)
                else:
                    # Thuyết minh văn bản mô tả -> Narrative
                    narrative_chunks = self.narrative_chunker.chunk(
                        page_text,
                        source_file=source_file,
                        page=page_num,
                        statement_type=current_statement,
                        company_code=inferred_company,
                        fiscal_year=fiscal_year
                    )
                    all_chunks.extend(narrative_chunks)
            else:
                # Khái quát, ý kiến kiểm toán, chính sách kế toán chung -> Narrative
                narrative_chunks = self.narrative_chunker.chunk(
                    page_text,
                    source_file=source_file,
                    page=page_num,
                    statement_type=current_statement,
                    company_code=inferred_company,
                    fiscal_year=fiscal_year
                )
                all_chunks.extend(narrative_chunks)

        for chunk in all_chunks:
            chunk.document_id = stable_document_id
            chunk.source_pdf = source_pdf
            chunk.fiscal_quarter = fiscal_quarter
            chunk.report_type = report_type
            chunk.section = chunk.statement_type.value
            chunk_comparative_year = chunk.metadata.get("comparative_fiscal_year", comparative_fiscal_year)
            if (
                chunk.chunk_type.value == "financial_item"
                and chunk_comparative_year is not None
                and str(chunk_comparative_year) not in chunk.metadata.get("period_values", {})
            ):
                chunk_comparative_year = None
            chunk.comparative_fiscal_year = chunk_comparative_year
            if fiscal_quarter is not None:
                chunk.metadata["fiscal_quarter"] = fiscal_quarter
            if chunk_comparative_year is not None:
                chunk.metadata["comparative_fiscal_year"] = chunk_comparative_year
            if report_type:
                chunk.metadata["report_type"] = report_type

        return all_chunks
