import re
from typing import List, Optional
from ..schemas import FinancialChunk, StatementType, ChunkType
from .base import BaseChunker

class NarrativeChunker(BaseChunker):
    """
    Chunker cho phần diễn giải bằng văn bản: Thuyết minh, Chính sách kế toán, Ý kiến kiểm toán.
    Chia theo đoạn văn bản tự nhiên, bảo toàn câu hoàn chỉnh và duy trì overlap.
    """

    def __init__(self, target_chars: int = 1200, overlap_chars: int = 200):
        self.target_chars = target_chars
        self.overlap_chars = overlap_chars

    def chunk(
        self,
        text: str,
        source_file: str = "doc.md",
        page: Optional[int] = None,
        statement_type: StatementType = StatementType.NOTES,
        company_code: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        **kwargs
    ) -> List[FinancialChunk]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks: List[FinancialChunk] = []
        current_chunk = ""
        chunk_idx = 0

        for p in paragraphs:
            if len(current_chunk) + len(p) > self.target_chars and current_chunk:
                # Tạo chunk
                chunks.append(
                    FinancialChunk(
                        chunk_id=f"{source_file}_narrative_p{page}_{chunk_idx}",
                        source_file=source_file,
                        company_code=company_code,
                        fiscal_year=fiscal_year,
                        page=page,
                        statement_type=statement_type,
                        chunk_type=ChunkType.NARRATIVE,
                        is_table=False,
                        content=current_chunk.strip()
                    )
                )
                chunk_idx += 1
                # Overlap: lấy phần đuôi của chunk trước
                current_chunk = current_chunk[-self.overlap_chars:] + "\n\n" + p
            else:
                current_chunk = (current_chunk + "\n\n" + p).strip()

        if current_chunk.strip():
            chunks.append(
                FinancialChunk(
                    chunk_id=f"{source_file}_narrative_p{page}_{chunk_idx}",
                    source_file=source_file,
                    company_code=company_code,
                    fiscal_year=fiscal_year,
                    page=page,
                    statement_type=statement_type,
                    chunk_type=ChunkType.NARRATIVE,
                    is_table=False,
                    content=current_chunk.strip()
                )
            )

        return chunks
