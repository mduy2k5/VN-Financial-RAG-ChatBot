import re
from typing import List
from ..schemas import Citation

class CitationValidator:
    """
    Xác thực các thẻ trích dẫn [c1], [c2] trong câu trả lời của LLM.
    Đảm bảo:
    - Mọi thẻ trích dẫn xuất hiện trong câu trả lời đều nằm trong danh sách Citation đã cấp.
    - Không có trích dẫn ma / bịa đặt.
    """

    @classmethod
    def validate(cls, answer: str, citations: List[Citation]) -> bool:
        return cls.validate_ids(answer, [citation.citation_id for citation in citations])

    @classmethod
    def validate_ids(
        cls,
        answer: str,
        citation_ids: List[str],
        require_citation: bool = True,
    ) -> bool:
        valid_ids = {citation_id.strip().casefold() for citation_id in citation_ids if citation_id.strip()}
        cited_ids = re.findall(r"\[(c?\d+)\]", answer, flags=re.IGNORECASE)
        if not cited_ids:
            return not require_citation

        for citation_id in cited_ids:
            normalized_id = citation_id.strip().casefold()
            if normalized_id.isdigit():
                normalized_id = f"c{normalized_id}"
            if normalized_id not in valid_ids:
                return False
        return True
