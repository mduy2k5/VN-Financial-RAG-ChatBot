import re
from typing import Optional

class SectionDetector:
    """
    Phát hiện các phần mục, số thuyết minh (Note number), tiêu đề bảng,
    và các tiểu mục trong báo cáo tài chính.
    """

    NOTE_PATTERNS = [
        r"(?:Thuyết minh|Thuyết minh số|TM|Note)\s*:?\s*(\d+(?:\.\d+)?)",
        r"(?:^|\n)#+\s*(\d+(?:\.\d+)?)\s*\.\s+([^\n]+)",
        r"(?:^|\n)\*\s+\*\*(\d+(?:\.\d+)?)\s*\.\s+([^\*]+)\*\*"
    ]

    @classmethod
    def extract_note_number(cls, text: str) -> Optional[str]:
        for pattern in cls.NOTE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    @classmethod
    def is_table_block(cls, text: str) -> bool:
        """Kiểm tra một đoạn text có phải dạng bảng hoặc biểu mẫu kế toán phân cột hay không."""
        if "|" in text and "-|-" in text:
            return True
        if "Số cuối năm" in text and "Số đầu năm" in text:
            return True
        if "Năm nay" in text and "Năm trước" in text:
            return True
        if "Triệu đồng" in text and ("Thuyết minh" in text or "Số tiền" in text):
            return True
        return False

    @classmethod
    def extract_unit(cls, text: str) -> Optional[str]:
        """Trích xuất đơn vị tính (Triệu đồng, Tỷ đồng, Đồng/VND)."""
        text_lower = text.lower()
        if re.search(r"(?:vnd|vnđ)\s+millions?\b|millions?\s+(?:of\s+)?(?:vnd|vnđ)\b", text_lower):
            return "Triệu VND"
        if "triệu đồng" in text_lower or "trđ" in text_lower:
            return "Triệu đồng"
        elif "triệu vnd" in text_lower:
            return "Triệu VND"
        elif (
            "tỷ đồng" in text_lower
            or "tỷ vnd" in text_lower
        ):
            return "Tỷ đồng"
        elif re.search(r"(?:vnd|vnđ)\s+billions?\b|billions?\s+(?:of\s+)?(?:vnd|vnđ)\b", text_lower):
            return "Tỷ VND"
        elif (
            "nghìn đồng" in text_lower
            or "ngàn đồng" in text_lower
        ):
            return "Nghìn đồng"
        elif re.search(r"(?:vnd|vnđ)\s+thousands?\b|thousands?\s+(?:of\s+)?(?:vnd|vnđ)\b", text_lower):
            return "Nghìn VND"
        elif "đồng" in text_lower or "vnd" in text_lower:
            return "VNĐ"
        return None
