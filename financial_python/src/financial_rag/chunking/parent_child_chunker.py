import re
from typing import List, Optional
from ..schemas import FinancialChunk, StatementType, ChunkType
from .base import BaseChunker

class ParentChildChunker(BaseChunker):
    """
    Chunker cho cấu trúc Phụ lục / Thuyết minh phân tầng:
    - Parent Chunk: Tiêu đề Thuyết minh và mô tả ngữ cảnh chung (VD: Thuyết minh 17: Các khoản nợ Chính phủ và NHNN).
    - Child Chunks: Từng khoản mục chi tiết bên trong (VD: Vay chiết khấu, Tiền gửi Kho bạc).
    Khi truy vấn, tìm kiếm trên Child Chunk, sau đó khôi phục lại Parent Chunk để cung cấp đầy đủ ngữ cảnh cho LLM.
    """

    def chunk(
        self,
        text: str,
        source_file: str = "doc.md",
        page: Optional[int] = None,
        statement_type: StatementType = StatementType.NOTES,
        note_title: Optional[str] = None,
        note_number: Optional[str] = None,
        company_code: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        **kwargs
    ) -> List[FinancialChunk]:
        chunks: List[FinancialChunk] = []

        parent_id = f"{source_file}_parent_p{page}_tm_{note_number or 'sec'}"
        parent_header = f"### THUYẾT MINH {note_number or ''}: {note_title or ''}\n(Trang {page})\n"

        # 1. Tạo Parent Chunk đại diện cho toàn bộ thuyết minh
        parent_chunk = FinancialChunk(
            chunk_id=parent_id,
            source_file=source_file,
            company_code=company_code,
            fiscal_year=fiscal_year,
            page=page,
            statement_type=statement_type,
            chunk_type=ChunkType.PARENT_CHILD,
            is_table=False,
            content=f"{parent_header}\n{text[:600]}...",
            metadata={"is_parent": True, "note_number": note_number}
        )
        chunks.append(parent_chunk)

        # 2. Tách các Child Chunks theo bullet mục con (*   **)
        items = re.split(r"(?=\n\*\s+\*\*)", text)
        if len(items) <= 1:
            items = [text]

        for idx, item in enumerate(items):
            item_clean = item.strip()
            if not item_clean or len(item_clean) < 30:
                continue

            child_content = f"[{company_code or 'Không xác định'} - THUYẾT MINH {note_number or ''} - Trang {page}]\n{item_clean}"
            chunks.append(
                FinancialChunk(
                    chunk_id=f"{parent_id}_child_{idx}",
                    parent_chunk_id=parent_id,
                    source_file=source_file,
                    company_code=company_code,
                    fiscal_year=fiscal_year,
                    page=page,
                    statement_type=statement_type,
                    chunk_type=ChunkType.PARENT_CHILD,
                    is_table=True,
                    content=child_content,
                    metadata={"is_parent": False, "parent_id": parent_id, "note_number": note_number}
                )
            )

        return chunks
