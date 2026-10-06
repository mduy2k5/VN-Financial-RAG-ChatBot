import re
import unicodedata
from typing import Optional
from ..schemas import PeriodType

class PeriodExtractor:
    """
    Trích xuất năm tài chính, quý, và phân loại kỳ (Số đầu năm vs Số cuối năm).
    """

    @classmethod
    def extract_fiscal_year(cls, text: str, fallback_file_name: str = "") -> Optional[int]:
        # 1. Ưu tiên số năm nằm ngay trong tên file (VD: DEMO_SYNTHETIC_2025.md)
        if fallback_file_name:
            file_match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", fallback_file_name)
            if file_match:
                return int(file_match.group(1))

        # 2. Tìm theo mẫu năm tài chính trong văn bản
        text_match = re.search(r"(?:năm tài chính|năm|year:)\s*((?:19|20)\d{2})", text, re.IGNORECASE)
        if text_match:
            return int(text_match.group(1))

        matches = re.findall(r"\b(20[0-3]\d)\b", text)
        if matches:
            return int(matches[0])
        return None

    @classmethod
    def extract_comparative_fiscal_year(cls, text: str, fiscal_year: Optional[int]) -> Optional[int]:
        """Return the prior period year when the report contains comparison markers."""
        if fiscal_year is None:
            return None

        explicit = re.search(
            r"(?:năm so sánh|năm trước|previous year|comparative year)\s*[:：]?\s*((?:19|20)\d{2})",
            text,
            re.IGNORECASE,
        )
        if explicit and int(explicit.group(1)) < fiscal_year:
            return int(explicit.group(1))

        if re.search(r"(?:số đầu năm|năm trước|previous year|comparative)", text, re.IGNORECASE):
            return fiscal_year - 1

        years = [int(value) for value in re.findall(r"\b((?:19|20)\d{2})\b", text)]
        prior_years = [year for year in years if year < fiscal_year]
        return max(prior_years) if prior_years else None

    @classmethod
    def extract_fiscal_quarter(cls, text: str) -> Optional[int]:
        match = re.search(r"(?:quý|q)\s*([1-4])\b", text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        roman_match = re.search(r"\bquý\s*(i{1,3}|iv)\b", text, re.IGNORECASE)
        if roman_match:
            return {"i": 1, "ii": 2, "iii": 3, "iv": 4}[roman_match.group(1).casefold()]
        return None

    @classmethod
    def extract_document_quarter(cls, text: str, file_name: str = "") -> Optional[int]:
        name = re.sub(r"[_-]+", " ", file_name)
        explicit = cls.extract_fiscal_quarter(name)
        if explicit:
            return explicit
        front = " ".join(text[:5000].split())
        quarterly = re.search(r"(?:báo cáo tài chính|cho)[^.;]{0,80}?quý\s*(?:[1-4]|i{1,3}|iv)\b", front, re.I)
        if quarterly and not re.search(r"năm|year", quarterly.group().rsplit("quý", 1)[0], re.I):
            return cls.extract_fiscal_quarter(quarterly.group())
        # Incidental quarter references in a note/variance explanation cannot
        # turn an annual audited report into a quarterly report.
        annual = re.search(r"(?:báo cáo tài chính|cho năm).{0,100}?(?:năm tài chính|năm\s*20\d{2}\s*(?:đã\s*)?kiểm toán)", front, re.I)
        if annual:
            return None
        return None  # Missing document-level evidence remains unspecified.

    @staticmethod
    def extract_report_type(text: str, fallback_file_name: str = "") -> Optional[str]:
        # A curated source filename is a document-level label and should win
        # over incidental OCR references to other report types in the body.
        normalized_name = unicodedata.normalize("NFKD", fallback_file_name.casefold())
        normalized_name = "".join(
            char for char in normalized_name if not unicodedata.combining(char)
        )
        compact_name = re.sub(r"[^a-z0-9]", "", normalized_name)
        if "hopnhat" in compact_name or "consolidated" in compact_name:
            return "consolidated"
        if (
            "congtyme" in compact_name
            or "baocaorieng" in compact_name
            or "rieng" in compact_name
            or "separate" in compact_name
        ):
            return "separate"

        text_lower = text.casefold()
        if "hợp nhất" in text_lower:
            return "consolidated"
        if "báo cáo tài chính riêng" in text_lower or "báo cáo riêng" in text_lower:
            return "separate"
        return None

    @classmethod
    def extract_period_type(cls, text: str) -> PeriodType:
        text_lower = text.lower()
        if any(marker in text_lower for marker in ("sáu tháng", "6 tháng", "bán niên", "nửa đầu năm")):
            return PeriodType.HALF_YEAR
        if "đầu năm" in text_lower or "đầu kỳ" in text_lower or "1/1/" in text_lower or "01/01/" in text_lower:
            return PeriodType.BEGINNING_OF_YEAR
        elif "cuối năm" in text_lower or "cuối kỳ" in text_lower or "31/12/" in text_lower:
            return PeriodType.END_OF_YEAR
        elif "quý 1" in text_lower or "q1" in text_lower or "quý 2" in text_lower or "q2" in text_lower or "quý 3" in text_lower or "q3" in text_lower or "quý 4" in text_lower or "q4" in text_lower:
            return PeriodType.QUARTERLY
        elif "cả năm" in text_lower or "lũy kế" in text_lower:
            return PeriodType.FULL_YEAR
        return PeriodType.UNSPECIFIED
