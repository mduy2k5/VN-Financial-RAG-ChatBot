import re
from typing import List, Dict, Tuple
from ..schemas import Citation, FinancialChunk, RetrievalResult
from ..structure.source_groups import source_item_label

class CitationMapper:
    """
    Tạo ánh xạ trích dẫn chuẩn hóa [c1], [c2], ... cho từng chunk dữ liệu.
    Bảo toàn: Tên file nguồn, số trang, loại báo cáo tài chính, và trích đoạn bằng chứng.
    """

    @classmethod
    def map_citations(cls, results: List[RetrievalResult]) -> Tuple[List[Citation], Dict[str, str]]:
        citations: List[Citation] = []
        chunk_to_citation_id: Dict[str, str] = {}

        for idx, res in enumerate(results, 1):
            c = res.chunk
            c_id = f"c{idx}"
            chunk_to_citation_id[c.chunk_id] = c_id

            snippet = cls._evidence_snippet(c)

            citation = Citation(
                citation_id=c_id,
                document_id=c.document_id or c.source_file,
                chunk_id=c.chunk_id,
                file_name=c.source_pdf or c.source_file,
                page=c.page,
                statement_type=c.statement_type.value,
                section=c.section,
                unit=c.unit,
                snippet=snippet[:500],
                **{key: c.metadata.get(key) for key in ("notebook_source_id", "version_id", "projection_id",
                                                       "block_id", "source_span", "source_sha256")}
            )
            citations.append(citation)

        return citations, chunk_to_citation_id

    @staticmethod
    def _evidence_snippet(chunk: FinancialChunk) -> str:
        """Prefer the source row and its reported values over document headers."""
        lines = [line.strip() for line in chunk.content.splitlines() if line.strip()]
        label = source_item_label(chunk)
        source_label = f"Chỉ tiêu gốc trong BCTC: {label}" if label else next(
            (line for line in lines if line.casefold().startswith("chỉ tiêu gốc trong bctc:")),
            "",
        )
        value_lines = [
            line
            for line in lines
            if re.match(
                r"^(?:Năm nay|Năm trước|Số cuối năm|Số đầu năm|Số cuối kỳ|Số đầu kỳ)\s*\(",
                line,
                re.IGNORECASE,
            )
        ]
        if not value_lines:
            period_values = chunk.metadata.get("period_values") or {}
            if isinstance(period_values, dict):
                value_lines = [
                    f"Năm {year}: {value} {chunk.unit or ''}".strip()
                    for year, value in sorted(period_values.items(), key=lambda item: str(item[0]))
                ]

        evidence_lines = [line for line in [source_label, *value_lines] if line]
        if evidence_lines:
            return " | ".join(evidence_lines)

        # Retain useful context for narrative/table chunks that do not expose a
        # structured source label and period-value pair.
        return " ".join(lines[:2])
