import re
from ..schemas import StatementType

class StatementDetector:
    """
    Nhận diện loại báo cáo tài chính từ văn bản, tiêu đề hoặc ngữ cảnh:
    - Báo cáo tình hình tài chính / Bảng cân đối kế toán (balance_sheet)
    - Báo cáo kết quả hoạt động kinh doanh (income_statement)
    - Báo cáo lưu chuyển tiền tệ (cash_flow)
    - Thuyết minh báo cáo tài chính (notes)
    - Báo cáo kiểm toán độc lập (audit_opinion)
    - Chính sách kế toán chủ yếu (accounting_policy)
    """

    PATTERNS = [
        (StatementType.BALANCE_SHEET, [
            r"(?:CONSOLIDATED\s+)?STATEMENT\s+OF\s+FINANCIAL\s+POSITION",
            r"(?:CONSOLIDATED\s+)?BALANCE\s+SHEET",
            r"BÁO CÁO TÌNH HÌNH TÀI CHÍNH",
            r"BẢNG CÂN Đ[ỐỒỔỘO]I KẾ TOÁN",
            r"B01/TCTD",
            r"B\s*02\s*/\s*TCTD",
            r"B\s*01\s*[-–—]\s*DN",
            r"B01a-DN",
        ]),
        (StatementType.CASH_FLOW, [
            r"(?:CONSOLIDATED\s+)?STATEMENT\s+OF\s+CASH\s+FLOWS?",
            r"(?:CONSOLIDATED\s+)?CASH\s+FLOW\s+STATEMENT",
            r"BÁO CÁO LƯU CHUYỂN TIỀN TỆ",
            r"L[UƯ\']{1,2}\s*CHUY[EÊỂẾ\']{1,2}N\s*TI[EÊỀ\']{1,2}N\s*T[EỆ]",
            r"LƯU CHUYỂN TIỀN TỆ",
            r"B\s*03\s*[-–—]\s*DN",
            r"B03a-DN",
            r"B\s*04\s*/\s*TCTD",
            r"B\s*04\s*-\s*DN",
        ]),
        (StatementType.INCOME_STATEMENT, [
            r"(?:CONSOLIDATED\s+)?STATEMENT\s+OF\s+PROFIT\s+(?:OR|AND)\s+LOSS",
            r"(?:CONSOLIDATED\s+)?(?:INCOME|PROFIT\s+AND\s+LOSS)\s+STATEMENT",
            r"STATEMENT\s+OF\s+COMPREHENSIVE\s+INCOME",
            r"BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH",
            r"(?m)^\s*\*{0,2}BÁO CÁO KẾT QUẢ HOẠT ĐỘNG\b",
            r"KẾT QUẢ KINH DOANH",
            r"B\s*02\s*[-–—]\s*DN",
            r"B02a-DN",
            r"B\s*03\s*/\s*TCTD",
            r"B\s*03\s*-\s*DN",
        ]),
        (StatementType.AUDIT_OPINION, [
            r"BÁO CÁO CỦA CÔNG TY KIỂM TOÁN ĐỘC LẬP",
            r"BÁO CÁO KI[EÊ]M TOÁN ĐỘC LẬP",
            r"Ý KIẾN CỦA KIỂM TOÁN VIÊN",
        ]),
        (StatementType.ACCOUNTING_POLICY, [
            r"CHÍNH SÁCH KẾ TOÁN CHỦ YẾU",
            r"TÓM TẮT CÁC CHÍNH SÁCH KẾ TOÁN",
        ]),
        (StatementType.NOTES, [
            r"THUYẾT MINH BÁO CÁO TÀI CHÍNH",
            r"BẢN THUYẾT MINH BÁO CÁO TÀI CHÍNH",
            r"B05/TCTD",
            r"B\s*09\s*-\s*DN",
        ]),
    ]

    EXPLICIT_NOTE_MARKERS = [
        r"\bB\s*05\s*/\s*TCTD(?:-[A-Z0-9]+)?\b",
        r"\bB\s*09\s*-\s*DN\b",
        r"THUYẾT MINH BÁO CÁO TÀI CHÍNH",
        r"BẢN THUYẾT MINH BÁO CÁO TÀI CHÍNH",
    ]

    @classmethod
    def detect(cls, text: str) -> StatementType:
        text_upper = text.upper()
        if re.search(r"^\s*MỤC\s*LỤC\b", text_upper, re.MULTILINE):
            return StatementType.GENERAL

        # Note pages often repeat the names of the primary statements in their
        # explanatory text. An explicit note form/header must take precedence
        # over those references so note rows do not masquerade as statement rows.
        if any(re.search(pattern, text_upper) for pattern in cls.EXPLICIT_NOTE_MARKERS):
            return StatementType.NOTES

        for stmt_type, patterns in cls.PATTERNS:
            for p in patterns:
                if re.search(p, text_upper):
                    return stmt_type
        return StatementType.GENERAL
