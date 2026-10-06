"""Conservative source-column checks independent of fuzzy metric mappings."""
import re

from ..query.directives import normalize_directive_text


DIMENSION_TERMS = ("khong ky han", "dao han", "duoi 3 thang", "tren 3 thang", "ngoai te",
                   "bo phan", "trong han", "qua han")


def table_scope(chunk) -> str:
    title = normalize_directive_text(str(chunk.table_title or chunk.metadata.get("table_title") or ""))
    if any(term in title for term in ("dao han", "dinh lai", "ky han", "thanh khoan")):
        return "maturity"
    if any(term in title for term in ("rui ro tien te", "trang thai ngoai te", "rui ro lai suat", "do nhay")):
        return "risk_dimension"
    if "bo phan" in title:
        return "segment"
    physical = chunk.metadata.get("source_statement_type")
    if physical in {"balance_sheet", "income_statement", "cash_flow"}:
        return "main_statement"
    return "notes" if physical == "notes" else "unknown"


def period_supported(chunk, year: int) -> bool | None:
    """False for contradicted headers; None means legacy/ambiguous evidence."""
    roles = chunk.metadata.get("source_period_columns") or {}
    column = roles.get(str(year))
    headers = chunk.metadata.get("table_headers") or []
    if not isinstance(column, int) or not 0 <= column < len(headers):
        return None
    raw_header = str(headers[column]).strip()
    header = normalize_directive_text(raw_header)
    explicit_years = {int(value) for value in re.findall(r"(?<!\d)(?:19|20|21)\d{2}(?!\d)", header)}
    if explicit_years:
        # A January opening balance belongs to the preceding year's closing.
        opening = re.fullmatch(r"\s*0?1[./-]0?1[./-](\d{4})(?:\s*(?:VND|VNĐ|triệu đồng|nghìn đồng))?\s*", raw_header, re.I)
        if opening and chunk.statement_type.value == "balance_sheet":
            return year == int(opening.group(1)) - 1
        return year in explicit_years
    if (any(term in header for term in DIMENSION_TERMS) or header in {"vnd", "usd", "eur"}
            or re.fullmatch(r"(?:duoi |tren |tu )?\d+\s*(thang|ngay)(?: tro len)?", header)):
        return False
    if "cuoi nam" in header or "nam nay" in header:
        return year == chunk.fiscal_year
    if "dau nam" in header or "nam truoc" in header:
        return chunk.fiscal_year is not None and year == chunk.fiscal_year - 1
    if table_scope(chunk) in {"maturity", "risk_dimension", "segment"}:
        return False
    return None
