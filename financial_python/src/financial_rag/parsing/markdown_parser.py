import re
from pathlib import Path
from typing import Any, Dict, List, Optional

_PAGE_SPLIT_RE = re.compile(r"(?=# Page \d+|<!--\s*page\s*:\s*\d+\s*-->)", re.IGNORECASE)
_PAGE_NUMBER_RE = re.compile(r"# Page (\d+)|<!--\s*page\s*:\s*(\d+)\s*-->", re.IGNORECASE)


def normalize_newlines(text: str) -> str:
    """Normalize Markdown line endings so chunking is independent of the source OS."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


class PageContent:
    def __init__(
        self,
        page_number: int,
        content: str,
        raw_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.page_number = page_number
        self.content = content
        self.raw_metadata = raw_metadata or {}

class MarkdownFinancialParser:
    """
    Parser chuyên trách đọc Markdown BCTC trích xuất từ OCR/DocReader.
    Nhận diện số trang (# Page X), khối Metadata đầu trang và nội dung báo cáo.
    """

    @classmethod
    def parse_file(cls, file_path: Path) -> List[PageContent]:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls.parse_text(text)

    @classmethod
    def parse_text(cls, text: str) -> List[PageContent]:
        text = normalize_newlines(text)
        # Accept legacy fixture headers and invisible markers emitted by
        # DocReader for each PDF page.
        page_splits = _PAGE_SPLIT_RE.split(text)
        pages: List[PageContent] = []

        for p_str in page_splits:
            p_str = p_str.strip()
            if not p_str:
                continue

            page_num_match = _PAGE_NUMBER_RE.search(p_str)
            page_number = (
                next((group for group in page_num_match.groups() if group), None)
                if page_num_match
                else None
            )
            page_num = int(page_number) if page_number else len(pages) + 1
            page_text = _PAGE_NUMBER_RE.sub("", p_str, count=1).strip()

            # Tách phần metadata header nếu có (thường nằm giữa ## Metadata và ## Content)
            meta_dict: Dict[str, Any] = {}
            if "## Metadata" in page_text and "## Content" in page_text:
                parts = page_text.split("## Content", 1)
                meta_section = parts[0]
                content_section = parts[1] if len(parts) > 1 else ""

                # Trích xuất metadata lines
                for line in meta_section.splitlines():
                    kv = re.search(r"-\s+\*\*([^\*:]+)\*\*:\s*(.*)", line)
                    if kv:
                        key = kv.group(1).strip()
                        val = kv.group(2).strip()
                        meta_dict[key] = val
                clean_content = content_section.strip()
            else:
                clean_content = page_text

            pages.append(PageContent(page_number=page_num, content=clean_content, raw_metadata=meta_dict))

        return pages
