import re
from typing import List, Optional
from ..schemas import FinancialChunk, StatementType, ChunkType
from .base import BaseChunker

class TableChunker(BaseChunker):
    """
    Chunker chuyên dụng cho bảng số liệu tài chính hoặc bảng bullet dạng cây.
    Bảo toàn:
    - Tiêu đề bảng / Section header
    - Đơn vị tính (Triệu đồng, Tỷ đồng)
    - Các cột kỳ kế toán (Số cuối năm, Số đầu năm, Năm nay, Năm trước)
    - Nhóm hàng dữ liệu kế toán liên quan.
    """

    def __init__(self, max_items_per_chunk: int = 8):
        self.max_items_per_chunk = max_items_per_chunk

    def chunk(
        self,
        text: str,
        source_file: str = "doc.md",
        page: Optional[int] = None,
        statement_type: StatementType = StatementType.BALANCE_SHEET,
        table_title: Optional[str] = None,
        unit: Optional[str] = None,
        company_code: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        **kwargs
    ) -> List[FinancialChunk]:
        chunks: List[FinancialChunk] = []

        # Tách theo các mục bullet chính (*   **)
        items = re.split(r"(?=\n\*\s+\*\*)", text)
        if len(items) <= 1:
            items = [text]

        chunk_idx = 0
        current_group: List[str] = []

        for item in items:
            item_clean = item.strip()
            if not item_clean:
                continue

            current_group.append(item_clean)
            if len(current_group) >= self.max_items_per_chunk:
                content_block = "\n\n".join(current_group)
                unit_header = f"(Đơn vị: {unit})\n" if unit else ""
                title_header = f"### {table_title}\n{unit_header}\n" if table_title else f"{unit_header}\n"
                chunks.append(
                    FinancialChunk(
                        chunk_id=f"{source_file}_table_p{page}_{chunk_idx}",
                        source_file=source_file,
                        company_code=company_code,
                        fiscal_year=fiscal_year,
                        page=page,
                        statement_type=statement_type,
                        chunk_type=ChunkType.TABLE,
                        is_table=True,
                        table_title=table_title,
                        unit=unit,
                        content=(title_header + content_block).strip()
                    )
                )
                chunk_idx += 1
                current_group = []

        if current_group:
            content_block = "\n\n".join(current_group)
            unit_header = f"(Đơn vị: {unit})\n" if unit else ""
            title_header = f"### {table_title}\n{unit_header}\n" if table_title else f"{unit_header}\n"
            chunks.append(
                FinancialChunk(
                    chunk_id=f"{source_file}_table_p{page}_{chunk_idx}",
                    source_file=source_file,
                    company_code=company_code,
                    fiscal_year=fiscal_year,
                    page=page,
                    statement_type=statement_type,
                    chunk_type=ChunkType.TABLE,
                    is_table=True,
                    table_title=table_title,
                    unit=unit,
                    content=(title_header + content_block).strip()
                )
            )

        return chunks
