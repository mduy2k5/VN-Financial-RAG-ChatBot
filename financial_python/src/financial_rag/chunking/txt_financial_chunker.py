from hashlib import sha256
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union

from ..schemas import FinancialChunk, StatementType, ChunkType
from ..structure.statement_detector import StatementDetector
from ..structure.section_detector import SectionDetector
from ..metadata.company_extractor import CompanyExtractor
from ..metadata.period_extractor import PeriodExtractor
from ..parsing.txt_table_parser import (
    TxtFinancialParser,
    TableBlock,
    ParsedTxtPage,
    TextBlock,
    normalize_newlines,
)
from ..metadata.metric_mapper import FinancialMetricMapper
from ..structure.report_schema_catalog import get_report_schema_catalog
from ..structure.source_groups import is_direct_code_child, source_parent_code, source_row_code
from .narrative_chunker import NarrativeChunker


_NUMERIC_RE = re.compile(r"\d[\d.,]*")
_NUMERIC_VALUE_RE = re.compile(r"^\s*(?:[+-]?\s*\d[\d.,\s]*|\(\s*\d[\d.,\s]*\))\s*%?\s*$")


def parse_financial_number(s: str) -> Optional[float]:
    """Phân tích chuỗi số tài chính (hỗ trợ định dạng Việt/Anh, số âm ngoặc đơn)."""
    if not s:
        return None
    val = s.strip()
    is_neg = False
    if val.startswith("(") and val.endswith(")"):
        is_neg = True
        val = val[1:-1].strip()
    elif val.startswith("-"):
        is_neg = True
        val = val[1:].strip()

    val = re.sub(r"[^\d.,]", "", val)
    if not val:
        return None

    if "." in val and "," in val:
        if val.rfind(".") > val.rfind(","):
            val = val.replace(",", "")
        else:
            val = val.replace(".", "").replace(",", ".")
    elif "." in val:
        parts = val.split(".")
        if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3):
            val = val.replace(".", "")
    elif "," in val:
        parts = val.split(",")
        if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3):
            val = val.replace(",", "")
        else:
            val = val.replace(",", ".")

    try:
        f = float(val)
        return -f if is_neg else f
    except ValueError:
        return None


def format_financial_number(val: float) -> str:
    """Định dạng số theo quy chuẩn báo cáo tài chính Việt Nam (dấu chấm phân cách hàng nghìn)."""
    sign = "-" if val < 0 else ""
    abs_val = abs(val)
    if float(abs_val).is_integer():
        formatted = f"{int(abs_val):,}"
    else:
        formatted = f"{abs_val:,.2f}"
    formatted = formatted.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{sign}{formatted}"


class TxtFinancialChunker:
    """
    Bộ chunker chuyên sâu cho tài liệu BCTC định dạng .txt có cấu trúc thẻ <table>:
    1. Bóc tách chỉ tiêu và bảng số liệu từ các bảng <table>.
    2. Bóc tách từng chỉ tiêu tài chính nguyên tử (Atomic Financial Item Chunks) kèm:
       - Số cuối năm / Số đầu năm (hoặc Năm nay / Năm trước).
       - Tự động tính toán chênh lệch tuyệt đối, tỷ lệ tăng/giảm (%) và xu hướng phục vụ so sánh số liệu.
       - Gắn kết phân nhóm chỉ tiêu (ví dụ: TÀI SẢN NGẮN HẠN, NỢ PHẢI TRẢ) theo thứ bậc phân cấp.
    3. Tạo các Table Chunks phân nhóm theo hàng giữ trọn vẹn ngữ cảnh bảng phục vụ tra cứu toàn diện.
    4. Giữ lại văn bản ngoài bảng dưới dạng narrative chunks để tìm được bài viết và thuyết minh.
    5. Gắn metadata cấp tài liệu và trang (company_code, fiscal_year, report_type, page,...).
    """

    def __init__(self, max_table_rows_per_chunk: int = 10):
        self.max_table_rows_per_chunk = max_table_rows_per_chunk
        self.company_extractor = CompanyExtractor()
        self.metric_mapper = FinancialMetricMapper()
        self.schema_catalog = get_report_schema_catalog()
        self.narrative_chunker = NarrativeChunker()


    def chunk_file(
        self,
        file_path: Union[Path, str],
        company_code: Optional[str] = None,
        source_pdf: Optional[str] = None,
    ) -> List[FinancialChunk]:
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return self.chunk_text(
            content,
            source_file=path.name,
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
        text = normalize_newlines(text)

        # 1. Header OCR có thể dài hơn vài trang, nên quét vùng đầu rộng hơn
        # để tránh tạo chunk không có mã công ty/năm báo cáo.
        metadata_text = text[:20_000]
        inferred_company = company_code or self.company_extractor.extract(
            text[:4_000], fallback_file_name=source_file
        ) or self.company_extractor.extract(metadata_text, fallback_file_name=source_file)
        fiscal_year = PeriodExtractor.extract_fiscal_year(
            metadata_text, fallback_file_name=source_file
        )
        comparative_fiscal_year = PeriodExtractor.extract_comparative_fiscal_year(
            text, fiscal_year
        )
        fiscal_quarter = PeriodExtractor.extract_document_quarter(metadata_text, source_file)
        report_type = PeriodExtractor.extract_report_type(
            metadata_text, fallback_file_name=source_file
        )
        content_sha256 = sha256(text.encode("utf-8")).hexdigest()
        chunk_namespace = sha256(
            f"{document_id or ''}|{source_file}|{content_sha256}".encode("utf-8")
        ).hexdigest()[:16]
        stable_document_id = document_id or sha256(
            f"{source_file}|{inferred_company}|{fiscal_year}|{fiscal_quarter}|{report_type}|{content_sha256}".encode("utf-8")
        ).hexdigest()

        # 2. Parse theo từng trang
        pages: List[ParsedTxtPage] = TxtFinancialParser.parse_text(text)
        all_chunks: List[FinancialChunk] = []
        current_statement = StatementType.GENERAL

        for page_obj in pages:
            page_num = page_obj.page_number
            page_content = page_obj.content

            # Nhận diện loại báo cáo từ nội dung trang
            detected_stmt = StatementDetector.detect(page_content)
            if detected_stmt != StatementType.GENERAL:
                current_statement = detected_stmt

            page_unit = SectionDetector.extract_unit(page_content)

            # 3. Giữ lại cả phần văn bản ngoài bảng. Bản TXT mới có thể là bài
            # viết hoặc phần giải trình, không có thẻ <table> để tạo table chunk.
            for block in page_obj.blocks:
                if not isinstance(block, TextBlock) or not block.text.strip():
                    continue
                narrative_statement = StatementDetector.detect(block.text)
                if narrative_statement == StatementType.GENERAL:
                    narrative_statement = detected_stmt
                narrative_chunks = self.narrative_chunker.chunk(
                    block.text,
                    source_file=source_file,
                    page=page_num,
                    statement_type=narrative_statement,
                    company_code=inferred_company,
                    fiscal_year=fiscal_year,
                )
                all_chunks.extend(narrative_chunks)

            # 4. Bóc tách các bảng tài chính và từng hàng số liệu.
            for table_index, table in enumerate(page_obj.tables):
                # Lọc bỏ các bảng không phải dữ liệu tài chính (Mục lục, Danh sách nhân sự,...)
                if not self._is_financial_table(table):
                    continue

                # Xác định loại báo cáo riêng cho bảng nếu tiêu đề bảng cụ thể hơn
                tbl_statement = current_statement
                if table.title:
                    tbl_stmt_detect = StatementDetector.detect(table.title)
                    if tbl_stmt_detect != StatementType.GENERAL:
                        tbl_statement = tbl_stmt_detect

                # A. Tạo Atomic Financial Item Chunks (Từng hàng chỉ tiêu tài chính + tính toán so sánh)
                item_chunks = self._extract_atomic_items_from_table(
                    table=table,
                    page_num=page_num,
                    statement_type=tbl_statement,
                    company_code=inferred_company,
                    fiscal_year=fiscal_year,
                    comparative_fiscal_year=comparative_fiscal_year,
                    default_unit=page_unit,
                    source_file=source_file,
                )
                all_chunks.extend(item_chunks)

                # B. Tạo Table Chunks cho toàn bộ bảng (chia nhóm hàng bảo toàn headers & title)
                tbl_chunks = self._create_table_chunks(
                    table=table,
                    page_num=page_num,
                    statement_type=tbl_statement,
                    company_code=inferred_company,
                    fiscal_year=fiscal_year,
                    comparative_fiscal_year=comparative_fiscal_year,
                    default_unit=page_unit,
                    source_file=source_file,
                )
                table_id = f"{stable_document_id}:p{page_num}:table_{table_index}"
                for source_chunk in [*item_chunks, *tbl_chunks]:
                    source_chunk.metadata.update({
                        "table_id": table_id, "source_start": table.source_start,
                        "source_end": table.source_end,
                    })
                all_chunks.extend(tbl_chunks)

        # 5. Gán metadata chuẩn cho toàn bộ chunk
        for chunk_index, chunk in enumerate(all_chunks):
            # Table and item indices restart for each table. Namespacing by the
            # document fingerprint and appending the document order keeps IDs
            # unique even when a page contains several tables with the same
            # local item/table index.
            chunk.chunk_id = f"{chunk_namespace}_{chunk.chunk_id}_{chunk_index}"
            chunk.document_id = stable_document_id
            chunk.source_pdf = source_pdf
            chunk.fiscal_quarter = fiscal_quarter
            chunk.report_type = report_type
            chunk.section = chunk.statement_type.value
            chunk_comparative_year = chunk.metadata.get("comparative_fiscal_year", comparative_fiscal_year)
            if (
                chunk.chunk_type.value == "financial_item"
                and chunk_comparative_year is not None
                and str(chunk_comparative_year)
                not in chunk.metadata.get("period_values", {})
            ):
                chunk_comparative_year = None
            chunk.comparative_fiscal_year = chunk_comparative_year
            if fiscal_quarter is not None:
                chunk.metadata["fiscal_quarter"] = fiscal_quarter
            if chunk_comparative_year is not None:
                chunk.metadata["comparative_fiscal_year"] = chunk_comparative_year
            if report_type:
                chunk.metadata["report_type"] = report_type

        return all_chunks

    def _is_financial_table(self, table: TableBlock) -> bool:
        """Kiểm tra bảng có chứa dữ liệu số liệu tài chính hay không."""
        if not table.matrix or not table.data_rows:
            return False

        # Kiểm tra tiêu đề bảng
        if table.title:
            t_upper = table.title.upper()
            if "MỤC LỤC" in t_upper or "DANH SÁCH BAN TỔNG GIÁM ĐỐC" in t_upper:
                return False

        # Kiểm tra headers
        header_text = " ".join(table.headers).lower()
        if re.search(r"\bhọ\s+và\s+tên\b|\bchức\s+vụ\b|\bngày\s+bổ\s+nhiệm\b", header_text):
            return False
        if re.search(r"\bnội\s+dung\b", header_text) and re.search(r"\btrang\b", header_text):
            return False

        # Phải có ít nhất 1 ô chứa ký tự số trong các hàng dữ liệu
        has_any_number = any(
            bool(_NUMERIC_RE.search(cell))
            for row in table.data_rows
            for cell in row
        )
        return has_any_number

    def _extract_atomic_items_from_table(
        self,
        table: TableBlock,
        page_num: int,
        statement_type: StatementType,
        company_code: Optional[str],
        fiscal_year: Optional[int],
        comparative_fiscal_year: Optional[int],
        default_unit: Optional[str],
        source_file: str,
    ) -> List[FinancialChunk]:
        """Bóc tách từng chỉ tiêu tài chính nguyên tử kèm tính toán so sánh số liệu."""
        if not table.headers or not table.data_rows:
            return []

        # Opening balance on 1 January of the report year is the prior year
        # closing balance. A date range beginning on 1 January is not a balance.
        if statement_type == StatementType.BALANCE_SHEET and fiscal_year and any(
            re.fullmatch(rf"\s*0?1[./-]0?1[./-]{fiscal_year}\s*", header)
            for header in table.headers
        ):
            comparative_fiscal_year = fiscal_year - 1

        col_roles = self._detect_table_columns(
            table.headers,
            fiscal_year=fiscal_year,
            comparative_fiscal_year=comparative_fiscal_year,
            data_rows=table.data_rows,
        )
        m_c = col_roles["metric_col"]
        c_c = col_roles["code_col"]
        n_c = col_roles["note_col"]
        e_c = col_roles["end_col"]
        s_c = col_roles["start_col"]
        detected_col_unit = col_roles.get("unit")

        # Nếu không tìm thấy cột kỳ nào thì không tạo atomic items
        if e_c is None and s_c is None:
            return []

        item_unit = table.unit or detected_col_unit or default_unit
        chunks: List[FinancialChunk] = []
        current_group: Optional[str] = None
        row_codes = {
            source_row_code(row[c_c])
            for row in table.data_rows
            if c_c is not None and c_c < len(row)
        }
        row_codes.discard("")
        # BCTC numeric codes often carry an explicit parent/child hierarchy
        # (for example 120 -> 123). Use it only when this table contains
        # multiple distinct codes; several bank statements repeat code 300
        # for unrelated rows and must keep their textual section headings.
        has_code_hierarchy = len(row_codes) > 1
        hierarchical_parent_codes = {
            parent_code
            for parent_code in row_codes
            if any(
                is_direct_code_child(parent_code, child_code)
                for child_code in row_codes
            )
        } if has_code_hierarchy else set()
        code_group_labels: Dict[str, str] = {}

        for idx, row in enumerate(table.data_rows):
            metric_raw = row[m_c].strip() if m_c < len(row) else ""
            if not metric_raw or len(metric_raw) < 2:
                continue
            code_val = row[c_c].strip() if c_c is not None and c_c < len(row) else ""
            parent_code = source_parent_code(code_val) if has_code_hierarchy else ""
            row_group = code_group_labels.get(parent_code) or current_group
            schema_row = self.schema_catalog.resolve_row_label(
                metric_raw,
                statement_type,
                parent_context=tuple(
                    context for context in (row_group, table.title) if context is not None
                ),
            )

            end_val = row[e_c].strip() if e_c is not None and e_c < len(row) else ""
            start_val = row[s_c].strip() if s_c is not None and s_c < len(row) else ""
            end_is_numeric = bool(_NUMERIC_VALUE_RE.fullmatch(end_val))
            start_is_numeric = bool(_NUMERIC_VALUE_RE.fullmatch(start_val))
            if not end_is_numeric:
                end_val = ""
            if not start_is_numeric:
                start_val = ""

            # Kiểm tra xem hàng này có số liệu không
            has_num = end_is_numeric or start_is_numeric
            if not has_num:
                # Nếu không có số nhưng có tiêu đề nhóm (ví dụ: A. TÀI SẢN NGẮN HẠN, C. NỢ PHẢI TRẢ)
                # Cập nhật current_group để các chỉ tiêu con kế tiếp kế thừa ngữ cảnh phân nhóm
                if (
                    schema_row and schema_row.kind in {"group", "section"}
                ) or (len(metric_raw) > 2 and not any(_NUMERIC_RE.search(c) for c in row)):
                    current_group = metric_raw
                continue

            note_val = row[n_c].strip() if n_c is not None and n_c < len(row) else ""

            # Xử lý đơn vị tính
            unit_str = "VNĐ/cổ phiếu" if "cổ phiếu" in metric_raw.casefold() else item_unit
            unit_suffix = f" {unit_str}" if unit_str else ""

            # Xử lý map giá trị theo kỳ
            period_values: Dict[str, str] = {}
            if fiscal_year is not None and end_val:
                period_values[str(fiscal_year)] = end_val
            if comparative_fiscal_year is not None and start_val:
                period_values[str(comparative_fiscal_year)] = start_val

            # TÍNH TOÁN SO SÁNH SỐ LIỆU (Chênh lệch tuyệt đối, tỷ lệ tăng/giảm, xu hướng)
            end_num = parse_financial_number(end_val)
            start_num = parse_financial_number(start_val)
            diff_num: Optional[float] = None
            pct_change: Optional[float] = None
            trend: Optional[str] = None
            comparison_line: Optional[str] = None

            if end_num is not None and start_num is not None:
                diff_num = end_num - start_num
                if start_num != 0:
                    pct_change = round(((end_num - start_num) / abs(start_num)) * 100.0, 2)

                if diff_num > 0:
                    trend = "Tăng"
                elif diff_num < 0:
                    trend = "Giảm"
                else:
                    trend = "Không đổi"

                diff_str = format_financial_number(diff_num)
                pct_str = f" ({pct_change:+.2f}%)" if pct_change is not None else ""
                prev_label = "đầu năm" if statement_type == StatementType.BALANCE_SHEET else "năm trước"
                comparison_line = f"So sánh biến động: {trend} {diff_str}{unit_suffix}{pct_str} so với {prev_label}"

            # The row-level schema match is based on the extracted source label
            # and table context, so prefer it over registry substring aliases.
            # This improves atomic chunk metadata without making the schema a
            # retrieval requirement; the raw row label remains the evidence.
            metric_def = self.metric_mapper.lookup_metric(
                raw_text=metric_raw,
                code=code_val,
                statement_hint=statement_type,
            )
            canonical_metric: Optional[str] = None
            statement_section: Optional[str] = None
            if schema_row:
                canonical_metric = schema_row.canonical_name
                item_statement_type = schema_row.statement_type
                statement_name_vi = self.metric_mapper.get_statement_name(item_statement_type)
                parent_node = self.schema_catalog.nodes.get(schema_row.parent_id or "")
                statement_section = row_group or (
                    parent_node.canonical_name if parent_node else None
                )
                metric_aliases = list(schema_row.aliases)
                effective_code = code_val
            elif metric_def:
                canonical_metric = metric_def.canonical_name
                item_statement_type = metric_def.statement_type
                statement_name_vi = metric_def.statement_name_vi
                statement_section = metric_def.section
                metric_aliases = metric_def.aliases
                effective_code = metric_def.code or code_val
            else:
                canonical_metric = None
                item_statement_type = statement_type
                statement_name_vi = self.metric_mapper.get_statement_name(statement_type)
                statement_section = row_group
                metric_aliases = []
                effective_code = code_val

            # XÂY DỰNG NỘI DUNG CHUNK CHUYÊN DỤNG CHO SEARCH VÀ SO SÁNH
            content_lines = [
                f"Doanh nghiệp: {company_code or 'Không xác định'}",
                f"Phân hệ báo cáo: {statement_name_vi} ({item_statement_type.value})",
                f"Năm tài chính: {fiscal_year if fiscal_year is not None else 'Không xác định'}",
            ]
            if comparative_fiscal_year is not None:
                content_lines.append(f"Năm so sánh: {comparative_fiscal_year}")

            if table.title:
                content_lines.append(f"Bảng: {table.title}")
            if row_group:
                content_lines.append(f"Phân nhóm: {row_group}")

            content_lines.append(f"Chỉ tiêu gốc trong BCTC: {metric_raw}")
            if effective_code:
                content_lines.append(f"Mã số: {effective_code}")
            if note_val:
                content_lines.append(f"Thuyết minh: {note_val}")

            if end_val:
                label = "Số cuối năm" if item_statement_type == StatementType.BALANCE_SHEET else "Năm nay"
                year_tag = f" ({fiscal_year})" if fiscal_year is not None else ""
                content_lines.append(f"{label}{year_tag}: {end_val}{unit_suffix}")

            if start_val:
                label = "Số đầu năm" if item_statement_type == StatementType.BALANCE_SHEET else "Năm trước"
                comp_tag = f" ({comparative_fiscal_year})" if comparative_fiscal_year is not None else ""
                content_lines.append(f"{label}{comp_tag}: {start_val}{unit_suffix}")

            if comparison_line:
                content_lines.append(comparison_line)

            content_lines.append(f"Trang: {page_num}")
            content_str = "\n".join(content_lines)

            metadata_dict: Dict[str, Any] = {
                "source_statement_type": statement_type.value,
                "metric_name": metric_raw,
                "canonical_metric": canonical_metric,
                "statement_name_vi": statement_name_vi,
                "statement_section": statement_section,
                "metric_aliases": metric_aliases,
                "code": effective_code,
                "note": note_val,
                "end_value": end_val,
                "start_value": start_val,
                "period_values": period_values,
                "comparative_fiscal_year": comparative_fiscal_year if start_val else None,
                "table_title": table.title,
                "section_group": row_group,
                "unit": unit_str,
                "source_row_index": idx,
                "source_cells": list(row),
                "table_headers": list(table.headers),
                "source_period_columns": {
                    **({str(fiscal_year): e_c} if fiscal_year is not None and end_val else {}),
                    **({str(comparative_fiscal_year): s_c}
                       if comparative_fiscal_year is not None and start_val else {}),
                },
            }
            if diff_num is not None:
                metadata_dict["difference"] = diff_num
                metadata_dict["difference_formatted"] = format_financial_number(diff_num)
            if pct_change is not None:
                metadata_dict["pct_change"] = pct_change
            if trend:
                metadata_dict["trend"] = trend

            chunks.append(
                FinancialChunk(
                    chunk_id=f"{source_file}_p{page_num}_item_{idx}",
                    source_file=source_file,
                    company_code=company_code,
                    fiscal_year=fiscal_year,
                    page=page_num,
                    statement_type=item_statement_type,
                    statement_name_vi=statement_name_vi,
                    canonical_metric=canonical_metric,
                    metric_aliases=metric_aliases,
                    section=statement_section,
                    chunk_type=ChunkType.FINANCIAL_ITEM,
                    is_table=True,
                    unit=unit_str,
                    content=content_str,
                    primary_metrics=[canonical_metric] if canonical_metric else [metric_raw],
                    metadata=metadata_dict,
                )
            )
            is_code_parent = code_val in hierarchical_parent_codes
            is_schema_group = bool(
                schema_row and schema_row.kind in {"group", "section"}
            )
            if is_code_parent or is_schema_group:
                current_group = metric_raw
                if has_code_hierarchy and source_row_code(code_val):
                    code_group_labels[source_row_code(code_val)] = metric_raw

        return chunks

    def _create_table_chunks(
        self,
        table: TableBlock,
        page_num: int,
        statement_type: StatementType,
        company_code: Optional[str],
        fiscal_year: Optional[int],
        comparative_fiscal_year: Optional[int],
        default_unit: Optional[str],
        source_file: str,
    ) -> List[FinancialChunk]:
        """Tạo các Table Chunks phân nhóm theo hàng giữ trọn vẹn ngữ cảnh cột và đơn vị."""
        if not table.matrix:
            return []

        unit_str = table.unit or default_unit
        chunks: List[FinancialChunk] = []
        rows = table.data_rows

        table_stmt_name_vi = self.metric_mapper.get_statement_name(statement_type)

        # Tạo metadata header cho bảng
        meta_lines = []
        if table.title:
            meta_lines.append(f"### {table.title}")
        context_tags = [
            f"Doanh nghiệp: {company_code or 'N/A'}",
            f"Báo cáo: {table_stmt_name_vi} ({statement_type.value})",
            f"Năm: {fiscal_year or 'N/A'}",
        ]
        if comparative_fiscal_year:
            context_tags.append(f"Năm trước: {comparative_fiscal_year}")
        if unit_str:
            context_tags.append(f"Đơn vị: {unit_str}")
        context_tags.append(f"Trang: {page_num}")
        meta_lines.append(" | ".join(context_tags))
        header_prefix = "\n".join(meta_lines) + "\n\n"

        if len(rows) <= self.max_table_rows_per_chunk:
            # Bảng nhỏ: giữ trọn vẹn trong 1 chunk
            md_content = table.to_markdown()
            full_content = (header_prefix + md_content).strip()
            chunks.append(
                FinancialChunk(
                    chunk_id=f"{source_file}_p{page_num}_table_0",
                    source_file=source_file,
                    company_code=company_code,
                    fiscal_year=fiscal_year,
                    page=page_num,
                    statement_type=statement_type,
                    statement_name_vi=table_stmt_name_vi,
                    chunk_type=ChunkType.TABLE,
                    is_table=True,
                    table_title=table.title,
                    unit=unit_str,
                    content=full_content,
                    metadata={
                        "table_title": table.title,
                        "statement_name_vi": table_stmt_name_vi,
                        "unit": unit_str,
                        "total_rows": len(rows),
                    },
                )
            )
        else:
            # Bảng lớn: chia thành nhiều chunk, mỗi chunk lặp lại Header
            header = table.headers
            sep = ["---"] * len(header)

            for i in range(0, len(rows), self.max_table_rows_per_chunk):
                batch_rows = rows[i : i + self.max_table_rows_per_chunk]
                lines = [
                    "| " + " | ".join(header) + " |",
                    "| " + " | ".join(sep) + " |",
                ]
                for r in batch_rows:
                    padded_r = r + [""] * (len(header) - len(r))
                    lines.append("| " + " | ".join(padded_r[: len(header)]) + " |")

                part_content = header_prefix + "\n".join(lines)
                chunks.append(
                    FinancialChunk(
                        chunk_id=f"{source_file}_p{page_num}_table_{i // self.max_table_rows_per_chunk}",
                        source_file=source_file,
                        company_code=company_code,
                        fiscal_year=fiscal_year,
                        page=page_num,
                        statement_type=statement_type,
                        statement_name_vi=table_stmt_name_vi,
                        chunk_type=ChunkType.TABLE,
                        is_table=True,
                        table_title=table.title,
                        unit=unit_str,
                        content=part_content.strip(),
                        metadata={
                            "table_title": table.title,
                            "statement_name_vi": table_stmt_name_vi,
                            "unit": unit_str,
                            "row_start": i,
                            "row_end": min(i + self.max_table_rows_per_chunk, len(rows)),
                            "total_rows": len(rows),
                        },
                    )
                )

        return chunks


    def _detect_table_columns(
        self,
        headers: List[str],
        fiscal_year: Optional[int] = None,
        comparative_fiscal_year: Optional[int] = None,
        data_rows: Optional[List[List[str]]] = None,
    ) -> Dict[str, Any]:
        """Dò tìm vai trò các cột trong bảng tài chính với độ linh hoạt cao."""
        metric_col = 0
        code_col: Optional[int] = None
        note_col: Optional[int] = None
        end_col: Optional[int] = None
        start_col: Optional[int] = None
        detected_unit: Optional[str] = None

        # Kiểm tra nếu cột 0 là STT hoặc Mã số và cột 1 là tên chỉ tiêu
        if len(headers) > 1:
            h0_norm = headers[0].strip().lower()
            if re.fullmatch(r"stt|tt|no\.?", h0_norm) or re.search(r"^mã(\s*số)?$", h0_norm):
                metric_col = 1
                if re.search(r"^mã(\s*số)?$", h0_norm):
                    code_col = 0

        for idx, h in enumerate(headers):
            h_norm = h.strip().lower()

            # Trích xuất đơn vị tính nhúng trong header (ví dụ: Triệu đồng, Triệu VND)
            if re.search(r"triệu\s*(?:vnd|vnđ|đồng)", h_norm):
                detected_unit = "Triệu VND" if re.search(r"triệu\s*(?:vnd|vnđ)", h_norm) else "Triệu đồng"
            elif re.search(r"tỷ\s*(?:vnd|vnđ|đồng)", h_norm):
                detected_unit = "Tỷ VND" if re.search(r"tỷ\s*(?:vnd|vnđ)", h_norm) else "Tỷ đồng"
            elif re.search(r"\bvnd\b|\bvnđ\b|\bđồng\b", h_norm) and not detected_unit:
                detected_unit = "VND"

            # Nhận diện cột Mã số
            if idx != metric_col and re.search(r"\bmã(\s*số)?\b|\bms\b|\bcode\b", h_norm):
                code_col = idx

            # Nhận diện cột Thuyết minh
            elif idx != metric_col and re.search(r"\bthuyết\s*minh\b|\btm\b|\bnote\b", h_norm):
                note_col = idx

            # Nhận diện cột Kỳ Hiện tại (Cuối năm / Năm nay / 31/12/YYYY / YYYY)
            elif re.fullmatch(r"0?1[./-]0?1[./-](?:19|20)\d{2}", h_norm):
                if start_col is None:
                    start_col = idx

            elif (
                re.search(r"cuối\s*năm|năm\s*nay|cuối\s*kỳ|quý\s*này", h_norm)
                or (fiscal_year and str(fiscal_year) in h_norm and not re.search(r"trước|đầu", h_norm))
            ):
                if end_col is None:
                    end_col = idx

            # Nhận diện cột Kỳ So sánh (Đầu năm / Năm trước / 01/01/YYYY / YYYY-1)
            elif (
                re.search(r"đầu\s*năm|năm\s*trước|đầu\s*kỳ|quý\s*trước", h_norm)
                or (comparative_fiscal_year and str(comparative_fiscal_year) in h_norm)
            ):
                if start_col is None:
                    start_col = idx

        # Một số OCR HTML thêm các ô trống trước tên chỉ tiêu mà không có header.
        # Xác định cột tên chỉ tiêu trước fallback cột số; nếu làm ngược lại,
        # cột tên có thể bị chọn nhầm làm số cuối kỳ khi header không có năm.
        if data_rows and (metric_col >= len(headers) or not headers[metric_col].strip()):
            excluded_cols = {code_col, note_col, end_col, start_col}
            metric_candidates = [
                idx for idx in range(len(headers)) if idx not in excluded_cols
            ]

            def label_count(col_idx: int) -> int:
                return sum(
                    1
                    for row in data_rows
                    if col_idx < len(row)
                    and len(row[col_idx].strip()) >= 2
                    and any(char.isalpha() for char in row[col_idx])
                )

            if metric_candidates:
                best_metric_col = max(metric_candidates, key=label_count)
                if label_count(best_metric_col) > label_count(metric_col):
                    metric_col = best_metric_col

        # Fallback: nếu chưa tìm thấy cột kỳ nhưng bảng có 3-5 cột, chọn các cột sau thuyết minh/mã số
        if end_col is None and start_col is None:
            candidate_cols = [
                i for i in range(len(headers))
                if i != metric_col and i != code_col and i != note_col
            ]
            if data_rows:
                numeric_coverage = {
                    col_idx: sum(
                        1
                        for row in data_rows
                        if col_idx < len(row)
                        and _NUMERIC_VALUE_RE.fullmatch(row[col_idx].strip())
                    )
                    for col_idx in candidate_cols
                }
                numeric_candidate_cols = sorted(
                    (col_idx for col_idx, count in numeric_coverage.items() if count),
                    key=lambda col_idx: (-numeric_coverage[col_idx], col_idx),
                )
                if numeric_candidate_cols:
                    candidate_cols = sorted(numeric_candidate_cols[:2])
            if len(candidate_cols) >= 2:
                end_col = candidate_cols[0]
                start_col = candidate_cols[1]
            elif len(candidate_cols) == 1:
                end_col = candidate_cols[0]

        return {
            "metric_col": metric_col,
            "code_col": code_col,
            "note_col": note_col,
            "end_col": end_col,
            "start_col": start_col,
            "unit": detected_unit,
        }
