"""Deterministic sections/blocks with offsets into unchanged source text."""
import hashlib
import re

from ..query.directives import normalize_directive_text
from ..structure.normalized import fingerprint

PAGE = re.compile(r"(?im)^(?:={3,}\s*PAGE\s*(\d+)\s*={3,}|# Page (\d+))[^\n]*"
                  r"|<!--\s*page\s*:\s*(\d+)\s*-->")
HEADINGS = re.compile(r"(?m)^(#{1,6})[ \t]+([^\n]+)|^((?:[IVX]+\.|\d+\.)?\s*"
                      r"(?:THUYẾT MINH|BẢNG CÂN ĐỐI|BÁO CÁO KẾT QUẢ|BÁO CÁO LƯU CHUYỂN)"
                      r"[^\n]{0,200})$")
TABLES = re.compile(r"<table\b[^>]*>.*?</table>|(?m:^[ \t]*\|[^\n]+\|[ \t]*$"
                    r"(?:\r?\n[ \t]*\|[^\n]+\|[ \t]*$)+)", re.DOTALL | re.IGNORECASE)


def build_structure(document: dict) -> dict:
    raw = document["raw_content"]
    version = document["active_version_id"]
    projection = document["active_projection_id"]
    root_id = fingerprint([projection, "root"])
    sections = [{"id": root_id, "parent_id": None, "title": document["source_file"],
                 "level": 0, "start": 0, "end": len(raw)}]
    stack = [sections[0]]
    for match in HEADINGS.finditer(raw):
        if PAGE.match(match.group(0)):
            continue
        level = len(match.group(1)) if match.group(1) else 2
        while len(stack) > 1 and stack[-1]["level"] >= level:
            stack.pop()["end"] = match.start()
        section = {"id": fingerprint([projection, "section", match.start()]),
                   "parent_id": stack[-1]["id"], "title": (match.group(2) or match.group(3)).strip(),
                   "level": level, "start": match.start(), "end": len(raw)}
        sections.append(section)
        stack.append(section)
    pages = [(match.start(), int(next(v for v in match.groups() if v)))
             for match in PAGE.finditer(raw)]

    def page_at(offset):
        return next((number for start, number in reversed(pages) if start <= offset), None)

    spans: list[tuple[int, int, str]] = []
    previous = 0
    for table in TABLES.finditer(raw):
        spans.extend((previous + m.start(), previous + m.end(), "paragraph") for m in
                     re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)",
                                 raw[previous:table.start()], re.DOTALL))
        start = table.start()
        if spans and table.start() - spans[-1][0] < 2000:
            preceding = raw[spans[-1][0]:spans[-1][1]]
            if re.search(r"đơn\s*vị|unit\s*:", preceding, re.I):
                start = spans.pop()[0]
        spans.append((start, table.end(), "table"))
        previous = table.end()
    spans.extend((previous + m.start(), previous + m.end(), "paragraph") for m in
                 re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", raw[previous:], re.DOTALL))
    blocks = []
    for start, end, kind in spans:
        section = next((s for s in reversed(sections) if s["start"] <= start < s["end"]), sections[0])
        # A table remains atomic. Long narrative uses paragraph boundaries;
        # never shorten a source span while claiming full evidence coverage.
        blocks.append({"id": fingerprint([projection, "block", start, end]),
                       "section_id": section["id"], "type": kind, "start": start, "end": end,
                       "page": page_at(start), "title": section["title"],
                       "estimated_tokens": len(raw[start:end].encode())})
    return {"version_id": version, "projection_id": projection,
            "source_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "sections": sections, "blocks": blocks}


def locate_chunk(chunk, structure: dict, raw: str) -> dict | None:
    label = chunk.metadata.get("metric_name") or chunk.metadata.get("metric")
    needle = label or chunk.content.strip()
    if not needle:
        return None
    values = chunk.metadata.get("period_values") or {}
    expected = [str(value).strip() for value in values.values() if str(value).strip() not in {"", "N/A", "-"}]
    candidates = [block for block in structure["blocks"] if
                  (chunk.page is None or block["page"] == chunk.page)
                  and needle in raw[block["start"]:block["end"]]
                  and all(value in raw[block["start"]:block["end"]] for value in expected)]
    return candidates[0] if len(candidates) == 1 else None


def extract_guide(document: dict, structure: dict) -> dict:
    raw = document["raw_content"]
    topics = list(dict.fromkeys(s["title"] for s in structure["sections"][1:]))[:20]
    introductory = [b for b in structure["blocks"] if b["type"] == "paragraph"
                    and 60 <= b["end"] - b["start"] <= 1600
                    and "metadata" not in normalize_directive_text(b["title"])]
    extracts = [{**b, "text": raw[b["start"]:b["end"]]} for b in introductory[:3]]
    return {"method": "source_extract", "topics": topics, "extracts": extracts,
            "suggested_questions": [f"Đọc phần {topic}" for topic in topics[:4]],
            "section_count": len(structure["sections"]) - 1,
            "block_count": len(structure["blocks"])}
