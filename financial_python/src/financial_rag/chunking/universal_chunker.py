from pathlib import Path
from typing import List, Optional
from ..schemas import FinancialChunk
from .financial_chunker import FinancialMarkdownChunker
from .txt_financial_chunker import TxtFinancialChunker
from ..structure.report_schema_catalog import get_report_schema_catalog
from ..structure.production_report_adapter import ProductionReportAdapter


class UniversalFinancialChunker:
    """
    Bộ chunker vạn năng tự động nhận diện định dạng:
    - Nếu là JSON theo hợp đồng production_report v2: Dùng adapter schema.
    - Nếu file là .txt hoặc nội dung chứa <table>: Dùng TxtFinancialChunker.
    - Nếu file là .md hoặc văn bản Markdown thuần: Dùng FinancialMarkdownChunker.
    """

    def __init__(self):
        self.md_chunker = FinancialMarkdownChunker()
        self.txt_chunker = TxtFinancialChunker()
        self.schema_catalog = get_report_schema_catalog()
        self.production_report_adapter = ProductionReportAdapter(self.schema_catalog)

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
        source_file: str = "doc.txt",
        company_code: Optional[str] = None,
        source_pdf: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> List[FinancialChunk]:
        is_json_candidate = source_file.lower().endswith(".json") or text.lstrip().startswith("{")
        if is_json_candidate:
            production_chunks = self.production_report_adapter.chunk_text(
                text=text,
                source_file=source_file,
                company_code=company_code,
                source_pdf=source_pdf,
                document_id=document_id,
            )
            if production_chunks is not None:
                return production_chunks
            if source_file.lower().endswith(".json"):
                return []

        is_txt_table = "<table>" in text.lower() or source_file.lower().endswith(".txt")
        if is_txt_table:
            chunks = self.txt_chunker.chunk_text(
                text=text,
                source_file=source_file,
                company_code=company_code,
                source_pdf=source_pdf,
                document_id=document_id,
            )
        else:
            chunks = self.md_chunker.chunk_text(
                text=text,
                source_file=source_file,
                company_code=company_code,
                source_pdf=source_pdf,
                document_id=document_id,
            )
        self.schema_catalog.annotate_chunks(chunks)
        return chunks
