import re
from typing import List, Optional
from ..schemas import FinancialChunk, StatementType, ChunkType
from .base import BaseChunker

class FinancialItemChunker(BaseChunker):
    """
    Tạo các chunk vi mô (Atomic Item Chunks) cho từng chỉ tiêu tài chính quan trọng.
    Ví dụ:
    Doanh nghiệp: ABB
    Năm tài chính: 2025
    Báo cáo: BẢNG CÂN ĐỐI KẾ TOÁN / BÁO CÁO TÌNH HÌNH TÀI CHÍNH
    Chỉ tiêu: TỔNG NỢ PHẢI TRẢ
    Số cuối năm: 203.694.036
    Số đầu năm: 162.688.881
    Đơn vị: Triệu đồng
    Trang: 9
    """

    def chunk(
        self,
        text: str,
        source_file: str = "doc.md",
        page: Optional[int] = None,
        statement_type: StatementType = StatementType.BALANCE_SHEET,
        unit: Optional[str] = None,
        company_code: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        comparative_fiscal_year: Optional[int] = None,
        **kwargs
    ) -> List[FinancialChunk]:
        chunks: List[FinancialChunk] = []

        # Bắt các bullet chứa chỉ tiêu và số liệu:
        # * **Chỉ tiêu**
        #   * Thuyết minh: X
        #   * Số cuối năm: Y
        #   * Số đầu năm: Z
        item_pattern = re.compile(
            r"\*\s+\*\*([^\*]+)\*\*\s*\n((?:\s+\*\s+[^\n]+\n?)+)",
            re.MULTILINE
        )
        # Some reports render final totals as a standalone bold line instead
        # of a bullet, followed by two unindented value bullets.
        standalone_pattern = re.compile(
            r"^[ \t]*\*\*([^\*]+)\*\*[ \t]*\r?\n"
            r"((?:^[ \t]*\*[ \t]+(?!\*\*)[^\n]*(?:\r?\n|$))+)",
            re.MULTILINE
        )

        matches = [
            (match.start(), match.group(1), match.group(2))
            for pattern in (item_pattern, standalone_pattern)
            for match in pattern.finditer(text)
        ]
        matches.sort(key=lambda match: match[0])
        chunk_idx = 0

        for _, metric_name, details in matches:
            metric_clean = metric_name.strip()
            # Trích xuất số cuối năm và số đầu năm
            value_pattern = (
                r"[ \t]*(?:\([^\)\n]*\)[ \t]*)?:?[ \t]*"
                r"((?:\(\s*[+-]?\s*\d[\d.,]*\s*\)|[+-]?\s*\d[\d.,]*))"
            )
            end_match = re.search(r"(?:Số cuối năm|Năm nay|Cuối kỳ)" + value_pattern, details, re.IGNORECASE)
            start_match = re.search(r"(?:Số đầu năm|Năm trước|Đầu kỳ)" + value_pattern, details, re.IGNORECASE)
            note_match = re.search(r"(?:Thuyết minh|TM)[^\d]*(\d+(?:\.\d+)?)", details, re.IGNORECASE)

            if end_match or start_match:
                end_val = end_match.group(1).strip() if end_match else "N/A"
                start_val = start_match.group(1).strip() if start_match else "N/A"
                note_val = note_match.group(1).strip() if note_match else "N/A"

                item_unit = "VNĐ/cổ phiếu" if "cổ phiếu" in metric_clean.casefold() else unit
                unit_suffix = f" ({item_unit})" if item_unit else ""
                prior_year = comparative_fiscal_year
                if prior_year is None and fiscal_year is not None and start_match:
                    prior_year = fiscal_year - 1
                current_period_label = "Số cuối năm" if statement_type == StatementType.BALANCE_SHEET else "Năm nay"
                prior_marker = re.search(r"Số đầu năm|Năm trước|Đầu kỳ", details, re.IGNORECASE)
                if prior_marker and "năm trước" in prior_marker.group(0).casefold():
                    prior_period_label = f"Năm trước ({prior_year})" if prior_year is not None else "Năm trước"
                elif statement_type == StatementType.BALANCE_SHEET:
                    prior_period_label = f"Số đầu năm (kỳ {fiscal_year})" if fiscal_year is not None else "Số đầu năm"
                else:
                    prior_period_label = f"Giá trị so sánh ({prior_year})" if prior_year is not None else "Giá trị so sánh"
                item_lines = [
                    f"Doanh nghiệp: {company_code or 'Không xác định'}",
                    f"Năm tài chính: {fiscal_year if fiscal_year is not None else 'Không xác định'}",
                ]
                if prior_year is not None and start_match:
                    item_lines.append(f"Năm so sánh: {prior_year}")
                item_lines.extend([
                    f"Báo cáo: {statement_type.value.upper()}",
                    f"Chỉ tiêu: {metric_clean}",
                    f"Thuyết minh: {note_val}",
                ])
                if end_match:
                    current_year_text = f" ({fiscal_year})" if fiscal_year is not None else ""
                    item_lines.append(f"{current_period_label}{current_year_text}: {end_val}{unit_suffix}")
                    if fiscal_year is not None:
                        item_lines.append(f"Giá trị năm {fiscal_year}: {end_val}{unit_suffix}")
                if start_match:
                    item_lines.append(f"{prior_period_label}: {start_val}{unit_suffix}")
                    if prior_year is not None:
                        prior_value_label = (
                            f"Giá trị cuối năm {prior_year} (cột so sánh)"
                            if statement_type == StatementType.BALANCE_SHEET
                            else f"Giá trị năm {prior_year}"
                        )
                        item_lines.append(f"{prior_value_label}: {start_val}{unit_suffix}")
                item_lines.extend([
                    f"Đơn vị: {item_unit or 'Không xác định'}",
                    f"Trang: {page}",
                ])
                item_content = "\n".join(item_lines)

                chunks.append(
                    FinancialChunk(
                        chunk_id=f"{source_file}_item_p{page}_{chunk_idx}",
                        source_file=source_file,
                        company_code=company_code,
                        fiscal_year=fiscal_year,
                        comparative_fiscal_year=prior_year if start_match else None,
                        page=page,
                        statement_type=statement_type,
                        chunk_type=ChunkType.FINANCIAL_ITEM,
                        is_table=True,
                        unit=item_unit,
                        content=item_content,
                        primary_metrics=[metric_clean.upper()],
                        metadata={
                            "metric": metric_clean,
                            "end_value": end_val,
                            "start_value": start_val,
                            "period_values": {
                                **({str(fiscal_year): end_val} if fiscal_year is not None and end_match else {}),
                                **({str(prior_year): start_val} if prior_year is not None and start_match else {}),
                            },
                            "comparative_fiscal_year": prior_year if start_match else None,
                            "note": note_val,
                            "unit": item_unit
                        }
                    )
                )
                chunk_idx += 1

        return chunks
