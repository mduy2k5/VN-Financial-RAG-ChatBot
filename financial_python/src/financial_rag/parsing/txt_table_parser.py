import re
import html
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Hỗ trợ phân trang: ===== PAGE X =====, === PAGE X ===, # Page X, <!-- page: X -->
_TXT_PAGE_SPLIT_RE = re.compile(
    r"(?=(?:^|\n)={3,}\s*PAGE\s*\d+\s*={3,}|(?:\n|^)# Page \d+|<!--\s*page\s*:\s*\d+\s*-->)",
    re.IGNORECASE,
)
_TXT_PAGE_NUMBER_RE = re.compile(
    r"={3,}\s*PAGE\s*(\d+)\s*={3,}|# Page (\d+)|<!--\s*page\s*:\s*(\d+)\s*-->",
    re.IGNORECASE,
)

_TABLE_TAG_RE = re.compile(r"<table>.*?</table>", re.IGNORECASE | re.DOTALL)
_UNIT_RE = re.compile(
    r"(?:Đơn vị(?:\s*tính)?|Don vi(?:\s*tinh)?)\s*:\s*([^\n\r,;]+)",
    re.IGNORECASE,
)


def markdown_table_cells(line: str) -> List[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|") and not line.endswith(r"\|"):
        line = line[:-1]
    return [cell.strip().replace(r"\|", "|") for cell in re.split(r"(?<!\\)\|", line)]


def markdown_table_spans(text: str) -> List[Tuple[int, int, List[List[str]]]]:
    """Read rectangular pipe tables outside fenced examples; preserve blank cells."""
    lines = text.splitlines(keepends=True)
    offsets = []
    position = 0
    for line in lines:
        offsets.append(position)
        position += len(line)
    spans = []
    index = 0
    fence = None
    while index < len(lines):
        stripped = lines[index].strip()
        marker = re.match(r"^(`{3,}|~{3,})", stripped)
        if marker:
            current = marker.group(1)
            if fence is None:
                fence = current
            elif current[0] == fence[0] and len(current) >= len(fence):
                fence = None
            index += 1
            continue
        if fence or index + 1 >= len(lines) or "|" not in stripped:
            index += 1
            continue
        header = markdown_table_cells(lines[index])
        separator = markdown_table_cells(lines[index + 1])
        if len(header) < 2 or len(separator) != len(header) or not all(
            re.fullmatch(r":?-{3,}:?", cell) for cell in separator
        ):
            index += 1
            continue
        end_index = index + 2
        rows = []
        malformed = False
        while end_index < len(lines) and "|" in lines[end_index] and lines[end_index].strip():
            row = markdown_table_cells(lines[end_index])
            if len(row) > len(header):
                malformed = True
            rows.append(row + [""] * max(0, len(header) - len(row)))
            end_index += 1
        if rows and not malformed:
            end = offsets[end_index] if end_index < len(offsets) else len(text)
            spans.append((offsets[index], end, [header, *rows]))
        index = end_index
    return spans


def normalize_newlines(text: str) -> str:
    """Normalize line endings so parsing is OS-independent."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


class HTMLTableParser(HTMLParser):
    """
    Parser HTML Table chuyên dụng cho Báo cáo tài chính.
    Bảo toàn cấu trúc ô, mở rộng colspan và rowspan thành ma trận lưới 2 chiều,
    xử lý decode entities và chuẩn hóa khoảng trắng.
    """

    def __init__(self):
        super().__init__()
        self.tables: List[List[List[Dict[str, Any]]]] = []
        self.current_table: Optional[List[List[Dict[str, Any]]]] = None
        self.current_row: Optional[List[Dict[str, Any]]] = None
        self.current_cell: Optional[List[str]] = None
        self.cell_attrs: Dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]):
        tag_lower = tag.lower()
        attrs_dict = {k.lower(): (v or "") for k, v in attrs}
        if tag_lower == "table":
            self.current_table = []
        elif tag_lower == "tr":
            self.current_row = []
        elif tag_lower in ("td", "th"):
            self.current_cell = []
            self.cell_attrs = attrs_dict

    def handle_endtag(self, tag: str):
        tag_lower = tag.lower()
        if tag_lower in ("td", "th"):
            raw_val = "".join(self.current_cell) if self.current_cell else ""
            text = html.unescape(raw_val).strip()
            # Chuẩn hóa khoảng trắng bên trong ô
            text = re.sub(r"\s+", " ", text)
            # Chuẩn hóa lỗi chính tả OCR phổ biến trong BCTC
            text = re.sub(r"\bNỘ\s+PHẢI\s+TRẢ\b", "NỢ PHẢI TRẢ", text, flags=re.IGNORECASE)
            text = re.sub(r"\bBẢO\s+CẢO\b", "BÁO CÁO", text, flags=re.IGNORECASE)
            text = re.sub(r"\bCÂN\s+ĐỒI\b", "CÂN ĐỐI", text, flags=re.IGNORECASE)
            text = re.sub(r"\bLU[UƯ\']{1,2}\s+CHUY[EÊỂẾ\']{1,2}N\b", "LƯU CHUYỂN", text, flags=re.IGNORECASE)
            try:
                colspan = int(self.cell_attrs.get("colspan", "1"))
            except ValueError:
                colspan = 1
            try:
                rowspan = int(self.cell_attrs.get("rowspan", "1"))
            except ValueError:
                rowspan = 1
            if self.current_row is not None:
                self.current_row.append({
                    "text": text,
                    "colspan": max(1, colspan),
                    "rowspan": max(1, rowspan),
                    "is_th": tag_lower == "th",
                })
            self.current_cell = None
        elif tag_lower == "tr":
            if self.current_table is not None and self.current_row is not None:
                self.current_table.append(self.current_row)
            self.current_row = None
        elif tag_lower == "table":
            if self.current_table is not None:
                self.tables.append(self.current_table)
            self.current_table = None

    def handle_data(self, data: str):
        if self.current_cell is not None:
            self.current_cell.append(data)


def raw_table_to_matrix(raw_rows: List[List[Dict[str, Any]]]) -> List[List[str]]:
    """
    Biến đổi danh sách hàng chứa colspan/rowspan thành ma trận chữ nhật 2D hoàn chỉnh.
    """
    grid: Dict[Tuple[int, int], str] = {}
    for r_idx, row in enumerate(raw_rows):
        c_idx = 0
        for cell in row:
            while (r_idx, c_idx) in grid:
                c_idx += 1
            text = cell["text"]
            colspan = cell["colspan"]
            rowspan = cell["rowspan"]

            # Xử lý trường hợp đặc thù của OCR BCTC Việt Nam:
            # Ví dụ: <td colspan="2">Mã Thuyết số minh</td>
            if colspan == 2 and re.search(r"mã.*thuyết.*minh", text, re.IGNORECASE):
                texts = ["Mã số", "Thuyết minh"]
            else:
                texts = [text] * colspan

            for dr in range(rowspan):
                for dc in range(colspan):
                    grid[(r_idx + dr, c_idx + dc)] = texts[dc]
            c_idx += colspan

    if not grid:
        return []

    max_r = max(r for r, c in grid.keys()) + 1
    max_c = max(c for r, c in grid.keys()) + 1
    matrix = [[grid.get((r, c), "") for c in range(max_c)] for r in range(max_r)]
    return matrix


class TableBlock:
    """
    Đại diện cho một khối bảng HTML trong file TXT tài chính.
    Cung cấp các phương thức chuyển đổi sang Markdown, Dictionary và Atomic items.
    """

    def __init__(
        self,
        raw_html: str,
        matrix: List[List[str]],
        page_number: int,
        title: Optional[str] = None,
        unit: Optional[str] = None,
        source_start: int = 0,
        source_end: int = 0,
    ):
        self.raw_html = raw_html
        self.matrix = matrix
        self.page_number = page_number
        self.title = title
        self.unit = unit
        self.source_start = source_start
        self.source_end = source_end

        self.headers: List[str] = matrix[0] if matrix else []
        self.data_rows: List[List[str]] = matrix[1:] if len(matrix) > 1 else []

    def to_markdown(self) -> str:
        """Chuyển đổi bảng sang Markdown Table chuẩn."""
        if not self.matrix:
            return ""
        header = [cell.replace("|", r"\|") for cell in self.headers]
        sep = ["---"] * len(header)
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(sep) + " |",
        ]
        for row in self.data_rows:
            # Pad row nếu thiếu cột
            padded_row = [cell.replace("|", r"\|") for cell in row] + [""] * (len(header) - len(row))
            lines.append("| " + " | ".join(padded_row[: len(header)]) + " |")

        md_content = "\n".join(lines)
        prefix = ""
        if self.title:
            prefix += f"### {self.title}\n"
        if self.unit:
            prefix += f"(Đơn vị: {self.unit})\n"
        return (prefix + md_content).strip()

    def to_dict_rows(self) -> List[Dict[str, str]]:
        """Trả về danh sách các hàng dạng key-value map theo tên cột."""
        if not self.headers or not self.data_rows:
            return []
        rows = []
        for d_row in self.data_rows:
            row_dict = {}
            for idx, h in enumerate(self.headers):
                val = d_row[idx] if idx < len(d_row) else ""
                row_dict[h] = val
            rows.append(row_dict)
        return rows


class TextBlock:
    """Khối văn bản tự do (không phải bảng)."""

    def __init__(self, text: str, page_number: int):
        self.text = text
        self.page_number = page_number


class ParsedTxtPage:
    """
    Đại diện cho một trang được bóc tách từ file TXT.
    Tương thích hoàn toàn với PageContent trong MarkdownFinancialParser.
    """

    def __init__(
        self,
        page_number: int,
        blocks: List[Union[TextBlock, TableBlock]],
        raw_text: str,
        content: str,
        tables: List[TableBlock],
        raw_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.page_number = page_number
        self.blocks = blocks
        self.raw_text = raw_text
        self.content = content
        self.tables = tables
        self.raw_metadata = raw_metadata or {}


class TxtFinancialParser:
    """
    Bộ lõi đọc và phân tích cấu trúc file .txt báo cáo tài chính chứa bảng HTML <table>.
    Cơ chế tương tự MarkdownFinancialParser nhưng tối ưu cho cú pháp bảng HTML đa cột/đa dòng.
    """

    @classmethod
    def parse_file(cls, file_path: Path) -> List[ParsedTxtPage]:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls.parse_text(text)

    @classmethod
    def parse_text(cls, text: str) -> List[ParsedTxtPage]:
        text = normalize_newlines(text)
        page_splits = _TXT_PAGE_SPLIT_RE.split(text)
        pages: List[ParsedTxtPage] = []

        for p_str in page_splits:
            p_str = p_str.strip()
            if not p_str:
                continue

            page_num_match = _TXT_PAGE_NUMBER_RE.search(p_str)
            page_number = (
                next((g for g in page_num_match.groups() if g), None)
                if page_num_match
                else None
            )
            page_num = int(page_number) if page_number else len(pages) + 1
            page_text = _TXT_PAGE_NUMBER_RE.sub("", p_str, count=1).strip()

            blocks, tables, markdown_content = cls._parse_page_content(
                page_text, page_num
            )

            pages.append(
                ParsedTxtPage(
                    page_number=page_num,
                    blocks=blocks,
                    raw_text=page_text,
                    content=markdown_content,
                    tables=tables,
                )
            )

        return pages

    @classmethod
    def _parse_page_content(
        cls, page_text: str, page_num: int
    ) -> Tuple[List[Union[TextBlock, TableBlock]], List[TableBlock], str]:
        blocks: List[Union[TextBlock, TableBlock]] = []
        tables: List[TableBlock] = []
        md_segments: List[str] = []

        last_idx = 0
        spans: List[Tuple[int, int, Optional[List[List[str]]]]] = [
            (match.start(), match.end(), None) for match in _TABLE_TAG_RE.finditer(page_text)
        ]
        html_spans = [(start, end) for start, end, _ in spans]
        spans.extend((start, end, matrix) for start, end, matrix in markdown_table_spans(page_text)
                     if not any(html_start <= start < html_end for html_start, html_end in html_spans))
        for start, end, matrix in sorted(spans, key=lambda span: span[0]):

            # Phần văn bản nằm trước bảng
            before_text = page_text[last_idx:start].strip()
            if before_text:
                blocks.append(TextBlock(text=before_text, page_number=page_num))
                md_segments.append(before_text)

            # Trích xuất tiêu đề bảng & đơn vị tính từ đoạn văn bản ngay trước bảng
            table_title = cls._extract_table_title(before_text)
            unit = cls._extract_unit(before_text)

            # Parse HTML Table
            table_html = page_text[start:end]
            if matrix is None:
                html_parser = HTMLTableParser()
                html_parser.feed(table_html)
                raw_rows = html_parser.tables[0] if html_parser.tables else []
                matrix = raw_table_to_matrix(raw_rows)

            table_block = TableBlock(
                raw_html=table_html,
                matrix=matrix,
                page_number=page_num,
                title=table_title,
                unit=unit,
                source_start=start,
                source_end=end,
            )
            blocks.append(table_block)
            tables.append(table_block)

            # Chuyển đổi bảng sang Markdown Table để tích hợp vào `content`
            md_table = table_block.to_markdown()
            if md_table:
                md_segments.append(md_table)

            last_idx = end

        # Phần văn bản còn lại sau bảng cuối cùng
        remaining_text = page_text[last_idx:].strip()
        if remaining_text:
            blocks.append(TextBlock(text=remaining_text, page_number=page_num))
            md_segments.append(remaining_text)

        full_markdown_content = "\n\n".join(md_segments).strip()
        return blocks, tables, full_markdown_content

    @classmethod
    def _extract_table_title(cls, text: str) -> Optional[str]:
        """Trích xuất tiêu đề bảng từ các dòng văn bản ngay trước bảng."""
        if not text:
            return None
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None
        # Quét ngược từ dòng dưới cùng lên tìm dòng có khả năng là tiêu đề
        for line in reversed(lines[-3:]):
            # Bỏ qua các dòng đơn vị tính hoặc mẫu số
            if _UNIT_RE.search(line) or re.search(r"mẫu\s*số", line, re.IGNORECASE):
                continue
            if len(line) < 120 and (
                line.isupper()
                or re.match(r"^\d+\.?\s+[A-ZÀ-Ỹ]", line)
                or "BẢNG" in line.upper()
                or "BÁO CÁO" in line.upper()
                or "THUYẾT MINH" in line.upper()
                or "MỤC LỤC" in line.upper()
            ):
                cleaned_title = re.sub(r"\bNỘ\s+PHẢI\s+TRẢ\b", "NỢ PHẢI TRẢ", line, flags=re.IGNORECASE)
                cleaned_title = re.sub(r"\bBẢO\s+CẢO\b", "BÁO CÁO", cleaned_title, flags=re.IGNORECASE)
                cleaned_title = re.sub(r"\bCÂN\s+ĐỒI\b", "CÂN ĐỐI", cleaned_title, flags=re.IGNORECASE)
                cleaned_title = re.sub(r"\bLU[UƯ\']{1,2}\s+CHUY[EÊỂẾ\']{1,2}N\b", "LƯU CHUYỂN", cleaned_title, flags=re.IGNORECASE)
                return cleaned_title
        return None

    @classmethod
    def _extract_unit(cls, text: str) -> Optional[str]:
        """Trích xuất đơn vị tính (VND, triệu đồng, tỷ đồng)."""
        if not text:
            return None
        match = _UNIT_RE.search(text)
        if match:
            return match.group(1).strip()
        return None
