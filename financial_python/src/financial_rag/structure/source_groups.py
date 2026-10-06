"""Source-only helpers for resolving a requested row group in a financial table."""

import re
import unicodedata
from typing import Any, Iterable


def _normalize_source_phrase(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.casefold())
    without_marks = "".join(
        char for char in decomposed if unicodedata.category(char) != "Mn"
    ).replace("đ", "d")
    return re.sub(r"[^a-z0-9]+", " ", without_marks).strip()


_STRUCTURE_FILLER = {
    "co", "cau", "thanh", "phan", "tong", "hop", "nhat", "cua", "cac",
    "khoan", "cho", "biet", "bao", "cao", "nam", "ky",
}
_QUESTION_FILLER = {"ma", "la", "nhieu"}
_SOURCE_LABEL_KEYS = (
    "metric_name",
    "metric",
    "original_label",
    "schema_original_label",
)


def source_item_label(chunk: Any) -> str:
    """Return the extracted source label across TXT and Markdown chunk formats."""
    metadata = getattr(chunk, "metadata", {})
    if isinstance(metadata, dict):
        for key in _SOURCE_LABEL_KEYS:
            value = metadata.get(key)
            if value is not None and str(value).strip():
                return str(value).strip().strip("*_ `")

    content = str(getattr(chunk, "content", "") or "")
    match = re.search(
        r"(?im)^\s*(?:chỉ\s+tiêu\s+gốc\s+trong\s+bctc|chỉ\s+tiêu|nhãn\s+chỉ\s+tiêu\s+gốc|nhãn\s+gốc|metric)\s*:\s*(?P<label>[^\n|]+)",
        content,
    )
    return match.group("label").strip().strip("*_ `") if match else ""


def display_source_item_label(label: str) -> str:
    """Remove report row numbering and arithmetic codes from a display label.

    The unmodified label remains in the source chunk/citation. This display
    form prevents OCR form codes (for example ``(10 = 01 - 02)``) from being
    presented as financial amounts or failing numeric validation.
    """
    cleaned = str(label or "").strip().strip("*_ `")
    cleaned = re.sub(r"^\s*(?:[A-Z]|[IVXLCDM]+)[.)]\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*\d+(?:\.\d+)*(?:[a-z])?[.):\s-]+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\(\s*\d+\s*=\s*[^)]*\)\s*$", "", cleaned)
    return cleaned.strip().strip("*_ `")


def source_phrase_tokens(value: str, company_codes: Iterable[str] = ()) -> tuple[str, ...]:
    """Normalize a requested/source phrase, dropping query boilerplate and periods."""
    excluded = set(_STRUCTURE_FILLER)
    for code in company_codes:
        excluded.update(_normalize_source_phrase(str(code)).split())
    return tuple(
        token
        for token in _normalize_source_phrase(value).split()
        if not token.isdigit() and token not in excluded
    )


def source_label_matches_query(
    query: str,
    label: str,
    company_codes: Iterable[str] = (),
) -> bool:
    """Match an exact source label even when the query omits a boilerplate word."""
    normalized_query = _normalize_source_phrase(query)
    normalized_label = _normalize_source_phrase(label)
    if not normalized_query or not normalized_label:
        return False
    if normalized_label in normalized_query:
        return True
    query_tokens = tuple(
        token for token in source_phrase_tokens(query, company_codes)
        if token not in _QUESTION_FILLER
    )
    label_tokens = tuple(
        token for token in source_phrase_tokens(label, company_codes)
        if token not in _QUESTION_FILLER
    )
    return bool(query_tokens and query_tokens == label_tokens)


def source_row_code(value: object) -> str:
    """Keep a simple numeric BCTC row code; reject note references and other IDs."""
    raw = str(value or "").strip()
    if not re.fullmatch(r"\d{3,4}", raw):
        return ""
    return raw


def source_parent_code(value: object) -> str:
    """Infer the immediate parent for common three-digit BCTC codes."""
    code = source_row_code(value)
    if len(code) != 3:
        return ""
    if code.endswith("00"):
        return ""
    if code.endswith("0"):
        return code[0] + "00"
    return code[:2] + "0"


def is_direct_code_child(parent_code: object, child_code: object) -> bool:
    """Recognize direct children in common three-digit BCTC row-code layouts."""
    parent = source_row_code(parent_code)
    child = source_row_code(child_code)
    if len(parent) != 3 or len(child) != 3 or parent == child:
        return False
    if parent.endswith("00"):
        return child[0] == parent[0] and (
            (child[1] != "0" and child[2] == "0")
            or (child[:2] == parent[:2] and child[2] != "0")
        )
    if parent.endswith("0"):
        return child[:2] == parent[:2] and child[2] != "0"
    return False


def is_total_source_label(value: str) -> bool:
    normalized = _normalize_source_phrase(value)
    return normalized.startswith(("tong ", "total ", "cong "))


def source_root_metric_id(chunk: Any) -> str:
    """Recognize statement roots from original labels, never fuzzy aliases."""
    statement = getattr(getattr(chunk, "statement_type", None), "value", None)
    if statement != "balance_sheet":
        return ""
    tokens = source_phrase_tokens(display_source_item_label(source_item_label(chunk)))
    roots = {
        ("tai", "san"): "tong_cong_tai_san",
        ("cong", "tai", "san"): "tong_cong_tai_san",
        ("no", "phai", "tra"): "tong_no_phai_tra",
        ("von", "chu", "so", "huu"): "tong_von_chu_so_huu",
        ("cong", "no", "phai", "tra"): "tong_no_phai_tra",
        ("cong", "von", "chu", "so", "huu"): "tong_von_chu_so_huu",
    }
    return roots.get(tokens, "")


def source_row_position(chunk: Any) -> tuple[int, int, str]:
    metadata = getattr(chunk, "metadata", {})
    match = re.search(r"_item_(\d+)(?:_|$)", str(chunk.chunk_id))
    position = metadata.get("row_start", metadata.get("row_index"))
    if position is None:
        position = match.group(1) if match else 0
    return int(chunk.page or 0), int(position or 0), str(chunk.chunk_id)


def root_source_segment(anchor: Any, rows: Iterable[Any]) -> list[Any]:
    """Bound a root by source order and the preceding genuine statement root."""
    if not source_root_metric_id(anchor):
        return []
    ordered = sorted(rows, key=source_row_position)
    anchor_index = next((i for i, row in enumerate(ordered) if row.chunk_id == anchor.chunk_id), None)
    if anchor_index is None:
        return []
    previous_roots = [i for i, row in enumerate(ordered[:anchor_index]) if source_root_metric_id(row)]
    start = previous_roots[-1] + 1 if previous_roots else 0
    return ordered[start:anchor_index + 1]


def analysis_target_phrase(query: str) -> str:
    """Remove question operation phrases without changing original source labels."""
    normalized = _normalize_source_phrase(query)
    patterns = (
        r"\b(?:khoan|thanh phan|chi tieu) nao\b",
        r"\b(?:dong gop|anh huong) (?:nhieu nhat |lon nhat )?(?:vao |den )?\b",
        r"\b(?:muc |su )?bien dong\b",
        r"\b(?:muc |su )?(?:tang|giam) (?:cua )?\b",
        r"\b(?:tu|den|va)\b",
    )
    for pattern in patterns:
        normalized = re.sub(pattern, " ", normalized)
    return " ".join(normalized.split())
