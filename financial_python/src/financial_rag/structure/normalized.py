"""Versioned, source-preserving projections of financial row chunks.

This module never embeds attributes or resolves competing filings. A cell from
an old chunk without coordinates remains explicitly unlocated. Mapping labels
and assigning period roles are parser annotations, not independently reviewed
financial facts.
"""
from __future__ import annotations

from collections import Counter
from decimal import Decimal
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..schemas import ChunkType
from ..validation.numeric import NumericValidator
from .semantics import period_supported, table_scope

SCHEMA_VERSION = "financial_cells_v2"


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


@lru_cache(maxsize=1)
def parser_revision() -> str:
    root = Path(__file__).resolve().parents[1]
    files = sorted(path for folder in ("chunking", "parsing", "metadata", "structure")
                   for path in (root / folder).glob("*.py"))
    hashes = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    config = root.parents[1] / "configs"
    for path in sorted(config.rglob("*.json")):
        hashes["configs/" + path.relative_to(config).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    schemas = Path(os.getenv("FINANCIAL_SCHEMA_DIR", str(root.parents[2] / "schemas")))
    for path in sorted(schemas.rglob("*.json")):
        hashes["schemas/" + path.relative_to(schemas).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return fingerprint(hashes)


class ReportVersion(BaseModel):
    version_id: str
    logical_key: str
    source_sha256: str
    source_file: str
    source_pdf: str | None = None
    company_code: str | None = None
    report_year: int | None = None
    fiscal_quarter: int | None = None
    report_type: str | None = None


class SourceTable(BaseModel):
    table_id: str
    source_table_id: str
    projection_id: str
    page: int | None = None
    title: str | None = None
    headers: list[str] = Field(default_factory=list)
    source_start: int | None = None
    source_end: int | None = None
    source_scope: str = "unknown"


class SourceRow(BaseModel):
    row_id: str
    projection_id: str
    table_id: str | None = None
    chunk_id: str
    label: str
    page: int | None = None
    row_index: int | None = None
    raw_cells: list[str] = Field(default_factory=list)
    canonical_metric: str | None = None
    section_group: str | None = None
    statement_type: str
    metric_validation: str = "advisory"


class FinancialCell(BaseModel):
    fact_id: str
    projection_id: str
    version_id: str
    row_id: str
    chunk_id: str
    period_year: int
    column_index: int | None = None
    column_header: str | None = None
    raw_value: str
    value: str  # Canonical Decimal text, never float or BSON double.
    unit: str | None = None
    currency: str | None = None
    scale: str | None = None
    measurement: Literal["money", "per_share", "percent", "unknown"] = "unknown"
    period_role: Literal["closing", "opening", "current", "comparative", "unknown"]
    period_kind: Literal["instant", "duration", "unknown"]
    coordinate_status: Literal["located", "unlocated"]
    interpretation_status: Literal["parser_inferred"] = "parser_inferred"
    period_validation: Literal["supported", "contradicted", "unknown"] = "unknown"
    formula_eligible: bool = False


class NormalizedReport(BaseModel):
    version: ReportVersion
    projection_id: str
    schema_version: str = SCHEMA_VERSION
    parser_revision: str
    chunk_sha256: str
    tables: list[SourceTable]
    rows: list[SourceRow]
    facts: list[FinancialCell]
    warnings: dict[str, int]

    def manifest(self) -> dict[str, Any]:
        return {"projection_id": self.projection_id, "version_id": self.version.version_id,
                "schema_version": self.schema_version, "parser_revision": self.parser_revision,
                "chunk_sha256": self.chunk_sha256, "table_count": len(self.tables),
                "row_count": len(self.rows), "fact_count": len(self.facts),
                "warnings": self.warnings}


def unit_dimensions(unit: str | None) -> dict[str, Any]:
    normalized = NumericValidator._normalize_unit(unit)
    if normalized == "%":
        return {"measurement": "percent", "scale": "1"}
    if normalized == "vndshare":
        return {"measurement": "per_share", "currency": "VND", "scale": "1"}
    scale = {"đồng": "1", "triệu đồng": "1000000", "tỷ đồng": "1000000000",
             "nghìn đồng": "1000"}.get(normalized or "")
    return {"measurement": "money", "currency": "VND", "scale": scale} if scale else {}


def normalize_report(document: dict[str, Any], *, revision: str | None = None) -> NormalizedReport:
    chunks = list(document["chunks"])
    source_hash = hashlib.sha256(document["raw_content"].encode()).hexdigest()
    version = ReportVersion(
        version_id=fingerprint([document["logical_key"], source_hash]),
        logical_key=document["logical_key"], source_sha256=source_hash,
        source_file=document["source_file"], source_pdf=document.get("source_pdf"),
        company_code=document.get("company_code"), report_year=document.get("fiscal_year"),
        fiscal_quarter=document.get("fiscal_quarter"), report_type=document.get("report_type"),
    )
    chunk_hash = fingerprint([chunk.model_dump(mode="json") for chunk in chunks])
    revision = revision or parser_revision()
    projection = fingerprint([version.version_id, SCHEMA_VERSION, revision, chunk_hash])
    tables: dict[str, SourceTable] = {}
    rows = []
    facts = []
    warnings: Counter[str] = Counter()
    for chunk in chunks:
        metadata = chunk.metadata
        source_table = metadata.get("table_id")
        table_id = fingerprint([projection, source_table]) if source_table else None
        if table_id and table_id not in tables:
            tables[table_id] = SourceTable(
                table_id=table_id, source_table_id=source_table, projection_id=projection,
                page=chunk.page, title=chunk.table_title or metadata.get("table_title"),
                headers=metadata.get("table_headers") or [],
                source_start=metadata.get("source_start"), source_end=metadata.get("source_end"),
                source_scope=table_scope(chunk),
            )
        if chunk.chunk_type != ChunkType.FINANCIAL_ITEM:
            continue
        row = SourceRow(
            row_id=fingerprint([projection, chunk.chunk_id]), projection_id=projection,
            table_id=table_id, chunk_id=chunk.chunk_id,
            label=str(metadata.get("metric_name") or metadata.get("metric") or ""),
            page=chunk.page,
            row_index=metadata.get("source_row_index"), raw_cells=metadata.get("source_cells") or [],
            canonical_metric=chunk.canonical_metric, statement_type=chunk.statement_type.value,
            section_group=metadata.get("section_group"),
        )
        rows.append(row)
        from ..query.directives import normalize_directive_text
        row.metric_validation = ("source_label_exact" if row.canonical_metric and
                                 normalize_directive_text(row.label) == normalize_directive_text(row.canonical_metric)
                                 else "advisory" if row.canonical_metric else "unmapped")
        values = dict(metadata.get("period_values") or {})
        # Support persisted rows from before period_values existed.
        for year, key in ((chunk.fiscal_year, "end_value"),
                          (chunk.comparative_fiscal_year, "start_value")):
            if year is not None and str(year) not in values and metadata.get(key) is not None:
                values[str(year)] = metadata[key]
        for year, raw in values.items():
            if not str(year).isdigit() or not 1900 <= int(year) <= 2200:
                warnings["invalid_period"] += 1
                continue
            numeric = NumericValidator.normalize_number(str(raw))
            if not numeric:
                warnings["non_numeric_cell"] += 1
                continue
            period_year = int(year)
            is_current = period_year == chunk.fiscal_year
            roles = metadata.get("source_period_columns") or {}
            column = roles.get(str(year))
            headers = metadata.get("table_headers") or []
            located = (table_id is not None and row.row_index is not None
                       and isinstance(column, int) and 0 <= column < len(row.raw_cells)
                       and str(row.raw_cells[column]).strip() == str(raw).strip())
            if not located:
                warnings["unlocated_cell"] += 1
            statement = chunk.statement_type.value if chunk.statement_type is not None else ""
            kind: Literal["instant", "duration", "unknown"] = "instant" if statement == "balance_sheet" else (
                "duration" if statement in {"income_statement", "cash_flow"} else "unknown")
            role: Literal["closing", "opening", "current", "comparative", "unknown"] = (
                ("closing" if is_current else "opening") if kind == "instant" else (
                    ("current" if is_current else "comparative") if kind == "duration" else "unknown")
            )
            unit = chunk.unit or metadata.get("unit")
            dimensions = unit_dimensions(unit)
            supported = period_supported(chunk, period_year)
            if supported is False:
                kind, role = "unknown", "unknown"
                warnings["contradicted_period_header"] += 1
            if not dimensions:
                warnings["unknown_unit"] += 1
            facts.append(FinancialCell(
                fact_id=fingerprint([row.row_id, period_year]), projection_id=projection,
                version_id=version.version_id, row_id=row.row_id, chunk_id=chunk.chunk_id,
                period_year=period_year, column_index=column,
                column_header=headers[column] if isinstance(column, int) and 0 <= column < len(headers) else None,
                raw_value=str(raw), value=format(Decimal(numeric), "f"), unit=unit,
                period_kind=kind, period_role=role,
                period_validation="supported" if supported is True else "contradicted" if supported is False else "unknown",
                formula_eligible=bool(located and supported is True and dimensions and row.label
                                      and row.metric_validation == "source_label_exact"
                                      and table_scope(chunk) not in {"maturity", "risk_dimension", "segment"}
                                      and kind != "unknown"),
                coordinate_status="located" if located else "unlocated", **dimensions,
            ))
    return NormalizedReport(version=version, projection_id=projection, parser_revision=revision,
                            chunk_sha256=chunk_hash, tables=list(tables.values()), rows=rows,
                            facts=facts, warnings=dict(warnings))
