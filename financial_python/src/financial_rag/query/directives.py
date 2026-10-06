"""Typed query directives inspired by Halo's domain-aware operators.

The planner is deliberately deterministic in the first version. It reuses the
existing query-understanding result and only extracts extra structure when a
user explicitly asks to focus on or exclude content.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from pydantic import BaseModel, Field

from ..schemas import ParsedQuery


class StructuralDirectives(BaseModel):
    preferred_chunk_types: list[str] = Field(default_factory=list)
    preferred_sections: list[str] = Field(default_factory=list)
    require_complete_rows: bool = False


class FilterDirectives(BaseModel):
    exclude_phrases: list[str] = Field(default_factory=list)


class ValidationDirectives(BaseModel):
    positive_constraints: list[str] = Field(default_factory=list)
    negative_constraints: list[str] = Field(default_factory=list)


class QueryDirectivePlan(BaseModel):
    """Small, serializable execution hints for retrieval operators."""

    version: str = "halo-adapter-v1"
    structural: StructuralDirectives = Field(default_factory=StructuralDirectives)
    filters: FilterDirectives = Field(default_factory=FilterDirectives)
    validation: ValidationDirectives = Field(default_factory=ValidationDirectives)


def normalize_directive_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.casefold())
    without_marks = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    without_marks = without_marks.replace("đ", "d")
    return re.sub(r"[^a-z0-9]+", " ", without_marks).strip()


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = " ".join(value.split()).strip(" ,:;-.")
        key = normalize_directive_text(clean)
        if clean and key and key not in seen:
            result.append(clean)
            seen.add(key)
    return result


class QueryDirectivePlanner:
    """Turn query understanding plus explicit prompt hints into typed directives."""

    _EXCLUDE_PATTERN = re.compile(
        r"^(?:bo qua|loai tru|khong xet|khong lay|ignore|exclude|skip)\s+(.+)$",
        re.IGNORECASE,
    )
    _NEGATIVE_PATTERN = re.compile(
        r"^(?:dung nham voi|khong nham voi|khong phai la|khong phai|"
        r"do not confuse with|not to be confused with|avoid)\s+(.+)$",
        re.IGNORECASE,
    )
    _POSITIVE_PATTERN = re.compile(
        r"^(?:chi su dung|dap an phai la|dap an phai bao gom|"
        r"phai bao gom|must include|use only)\s+(.+)$",
        re.IGNORECASE,
    )

    def plan(self, query: str, parsed: ParsedQuery) -> QueryDirectivePlan:
        normalized = normalize_directive_text(query)
        clauses = [
            normalize_directive_text(clause)
            for clause in re.split(r"[,;.!?\n]+", query)
            if clause.strip()
        ]
        preferred_types: list[str] = []
        preferred_sections: list[str] = []

        if re.search(r"\b(bang|table|bieu)\b", normalized):
            preferred_types.extend(("table", "financial_item"))
        if re.search(r"\b(thuyet minh|ghi chu|notes?)\b", normalized):
            preferred_sections.extend(("thuyet minh", "ghi chu", "notes"))

        complete_rows = parsed.structured_operation in {
            "table", "components", "composition", "ranking", "summary"
        }
        positive = [
            match.group(1)
            for clause in clauses
            if (match := self._POSITIVE_PATTERN.match(clause))
        ]
        negative = [
            match.group(1)
            for clause in clauses
            if (match := self._NEGATIVE_PATTERN.match(clause))
        ]
        excluded = [
            match.group(1)
            for clause in clauses
            if (match := self._EXCLUDE_PATTERN.match(clause))
        ]

        return QueryDirectivePlan(
            structural=StructuralDirectives(
                preferred_chunk_types=list(dict.fromkeys(preferred_types)),
                preferred_sections=list(dict.fromkeys(preferred_sections)),
                require_complete_rows=complete_rows,
            ),
            filters=FilterDirectives(exclude_phrases=_unique(excluded)),
            validation=ValidationDirectives(
                positive_constraints=_unique(positive),
                negative_constraints=_unique(negative),
            ),
        )
