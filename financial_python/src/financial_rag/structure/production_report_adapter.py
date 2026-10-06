"""Adapt the production_report v2 JSON contract to financial fact chunks."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Optional

from ..metadata.company_extractor import CompanyExtractor
from ..schemas import ChunkType, FinancialChunk, StatementType
from .report_schema_catalog import ReportSchemaCatalog, get_report_schema_catalog


_STATEMENT_TYPES = {
    "financial_position": StatementType.BALANCE_SHEET,
    "income_statement": StatementType.INCOME_STATEMENT,
    "cash_flow": StatementType.CASH_FLOW,
}


class ProductionReportAdapter:
    """Convert schema-normalized reports into the same fact model as TXT tables."""

    def __init__(self, catalog: Optional[ReportSchemaCatalog] = None):
        self.catalog = catalog or get_report_schema_catalog()
        self.company_extractor = CompanyExtractor()

    def chunk_text(
        self,
        text: str,
        source_file: str,
        company_code: Optional[str] = None,
        source_pdf: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> Optional[list[FinancialChunk]]:
        """Return None for non-contract JSON; otherwise return normalized chunks."""
        try:
            report_data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(report_data, dict):
            return None
        contract = report_data.get("contract")
        if not isinstance(contract, dict) or contract.get("name") != "vn_financial_report_output":
            return None
        if str(contract.get("version")) != "2.0.0":
            return None

        report = report_data.get("report") or {}
        periods = report_data.get("periods") or {}
        if not isinstance(report, dict) or not isinstance(periods, dict):
            return []
        current_period = periods.get("current") or {}
        previous_period = periods.get("previous") or {}
        if not isinstance(current_period, dict) or not isinstance(previous_period, dict):
            return []
        current_year = self._year(current_period.get("year"))
        previous_year = self._year(previous_period.get("year"))
        fiscal_quarter = self._quarter(periods.get("type"), current_period)
        report_id = str(report.get("id") or document_id or source_file)
        stable_document_id = document_id or report_id
        resolved_company = (
            company_code
            or report.get("company_code")
            or self.company_extractor.extract(source_file)
        )
        report_source = report.get("source") or {}
        if not isinstance(report_source, dict):
            report_source = {}
        source_name = str(report_source.get("name") or source_file)
        source_hash = report_source.get("sha256")
        quality = report_data.get("quality") or {}
        if not isinstance(quality, dict):
            quality = {}
        quality_gate = quality.get("gate") or {}
        if not isinstance(quality_gate, dict):
            quality_gate = {}
        report_eligible = quality_gate.get("eligible_for_use")

        chunks: list[FinancialChunk] = []
        statements = report_data.get("statements") or {}
        if not isinstance(statements, dict):
            return []
        for statement_key, statement_type in _STATEMENT_TYPES.items():
            statement = statements.get(statement_key) or {}
            if not isinstance(statement, dict):
                continue
            context = statement.get("context") or {}
            if not isinstance(context, dict):
                context = {}
            unit = context.get("unit")
            sector = context.get("sector")
            schema = statement.get("schema") or {}
            if not isinstance(schema, dict):
                schema = {}
            statement_version = schema.get("version")
            records = statement.get("items") or []
            if not isinstance(records, list):
                continue
            for index, record in enumerate(records):
                chunk = self._record_chunk(
                    record=record,
                    index=index,
                    report_id=report_id,
                    source_file=source_file,
                    source_name=source_name,
                    source_hash=source_hash,
                    source_pdf=source_pdf,
                    document_id=stable_document_id,
                    company_code=str(resolved_company) if resolved_company else None,
                    statement_type=statement_type,
                    current_year=current_year,
                    previous_year=previous_year,
                    default_unit=str(unit) if unit else None,
                    report_sector=str(sector) if sector else None,
                    statement_version=str(statement_version) if statement_version else None,
                    report_eligible=report_eligible,
                    periods_type=periods.get("type"),
                    fiscal_quarter=fiscal_quarter,
                )
                if chunk is not None:
                    chunks.append(chunk)
        self.catalog.annotate_chunks(chunks)
        return chunks

    def _record_chunk(
        self,
        *,
        record: Any,
        index: int,
        report_id: str,
        source_file: str,
        source_name: str,
        source_hash: Any,
        source_pdf: Optional[str],
        document_id: str,
        company_code: Optional[str],
        statement_type: StatementType,
        current_year: Optional[int],
        previous_year: Optional[int],
        default_unit: Optional[str],
        report_sector: Optional[str],
        statement_version: Optional[str],
        report_eligible: Any,
        periods_type: Any,
        fiscal_quarter: Optional[int],
    ) -> Optional[FinancialChunk]:
        if not isinstance(record, dict):
            return None
        value = record.get("value") or {}
        if not isinstance(value, dict):
            return None
        raw_values = value.get("raw_values") or {}
        if not isinstance(raw_values, dict):
            raw_values = {}
        current = self._value_text(raw_values.get("current"), value.get("current"))
        previous = self._value_text(raw_values.get("previous"), value.get("previous"))
        if not current and not previous:
            return None

        label = str(record.get("label") or "").strip()
        if not label:
            return None
        attributes = record.get("attributes") or {}
        provenance = record.get("provenance") or {}
        if not isinstance(attributes, dict):
            attributes = {}
        if not isinstance(provenance, dict):
            provenance = {}
        mapping = provenance.get("mapping") or {}
        if not isinstance(mapping, dict):
            mapping = {}
        evidence = provenance.get("evidence") or []
        if not isinstance(evidence, list):
            evidence = []
        target_id = mapping.get("target_id")
        node = self.catalog.nodes.get(str(target_id)) if target_id else None
        canonical_name = node.canonical_name if node else label
        row_unit = value.get("unit") or default_unit
        period_values: dict[str, str] = {}
        if current and current_year:
            period_values[str(current_year)] = current
        if previous and previous_year:
            period_values[str(previous_year)] = previous

        source_context = attributes.get("source_context")
        page = self._page_number(source_context)
        note = value.get("note")
        content_lines = [
            f"Chỉ tiêu gốc: {label}",
            f"Chỉ tiêu chuẩn hóa: {canonical_name}",
        ]
        if current:
            period_label = (
                "Số cuối năm" if statement_type == StatementType.BALANCE_SHEET else "Năm nay"
            )
            year_tag = f" ({current_year})" if current_year else ""
            unit_suffix = f" {row_unit}" if row_unit else ""
            content_lines.append(f"{period_label}{year_tag}: {current}{unit_suffix}")
        if previous:
            period_label = (
                "Số đầu năm" if statement_type == StatementType.BALANCE_SHEET else "Năm trước"
            )
            year_tag = f" ({previous_year})" if previous_year else ""
            unit_suffix = f" {row_unit}" if row_unit else ""
            content_lines.append(f"{period_label}{year_tag}: {previous}{unit_suffix}")
        content_lines.extend([
            f"Doanh nghiệp: {company_code or 'Không xác định'}",
            f"Phân hệ báo cáo: {statement_type.value}",
            f"Báo cáo: {report_id}",
        ])
        if note is not None:
            content_lines.append(f"Thuyết minh: {note}")
        source_label = self._first_evidence(evidence, "source_label")
        source_line = self._first_evidence(evidence, "source_line")
        if source_label:
            content_lines.append(f"Nhãn nguồn: {source_label}")
        if source_line is not None:
            content_lines.append(f"Dòng nguồn: {source_line}")

        record_key = str(record.get("id") or record.get("source_id") or index)
        digest = hashlib.sha1(f"{report_id}|{statement_type.value}|{record_key}".encode("utf-8")).hexdigest()[:12]
        metadata: dict[str, Any] = {
            "metric_name": label,
            "canonical_metric": canonical_name,
            "schema_node_id": str(target_id) if node else None,
            "schema_parent_id": attributes.get("parent_id") or (node.parent_id if node else None),
            "schema_context_node_id": attributes.get("parent_id"),
            "schema_children": attributes.get("children") or [],
            "schema_node_kind": attributes.get("kind") or (node.kind if node else None),
            "schema_sector": attributes.get("sector") or report_sector or (node.sector if node else None),
            "schema_version": statement_version or (node.schema_version if node else None),
            "schema_mapping_status": mapping.get("status"),
            "schema_mapping_method": mapping.get("method"),
            "schema_mapping_confidence": mapping.get("confidence"),
            "period_values": period_values,
            "end_value": current,
            "start_value": previous,
            "unit": row_unit,
            "source_name": source_name,
            "source_sha256": source_hash,
            "evidence": evidence,
            "raw_text": value.get("raw_text"),
            "report_eligible": report_eligible,
            "periods_type": periods_type,
            "fiscal_quarter": fiscal_quarter,
            "source_context": source_context,
        }
        return FinancialChunk(
            chunk_id=f"{source_file}_{statement_type.value}_{digest}",
            source_file=source_file,
            document_id=document_id,
            source_pdf=source_pdf,
            company_code=company_code,
            fiscal_year=current_year,
            comparative_fiscal_year=previous_year,
            fiscal_quarter=fiscal_quarter,
            page=page,
            statement_type=statement_type,
            statement_name_vi=statement_type.value,
            canonical_metric=canonical_name,
            chunk_type=ChunkType.FINANCIAL_ITEM,
            is_table=True,
            table_title=source_name,
            unit=str(row_unit) if row_unit else None,
            content="\n".join(content_lines),
            primary_metrics=[canonical_name],
            metadata=metadata,
        )

    @staticmethod
    def _value_text(raw_value: Any, numeric_value: Any) -> str:
        if isinstance(raw_value, str) and raw_value.strip():
            return raw_value.strip()
        if numeric_value is None or isinstance(numeric_value, bool):
            return ""
        if isinstance(numeric_value, float):
            return format(numeric_value, ".15g")
        return str(numeric_value)

    @staticmethod
    def _year(value: Any) -> Optional[int]:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if 1900 <= parsed <= 2200 else None

    @staticmethod
    def _quarter(period_type: Any, current_period: Any) -> Optional[int]:
        if not isinstance(current_period, dict):
            return None
        normalized_type = str(period_type or "").casefold()
        match = re.search(r"(?:quarter|quý|q)\s*([1-4])\b", normalized_type)
        if match:
            return int(match.group(1))
        if not any(marker in normalized_type for marker in ("quarter", "quý", "q1", "q2", "q3", "q4")):
            return None
        end_date = str(current_period.get("end_date") or "")[:10]
        try:
            month = int(end_date[5:7])
        except (TypeError, ValueError):
            return None
        return (month - 1) // 3 + 1 if 1 <= month <= 12 else None

    @staticmethod
    def _page_number(source_context: Any) -> Optional[int]:
        if not isinstance(source_context, dict):
            return None
        for key in ("page", "page_number", "page_no"):
            raw_value = source_context.get(key)
            if raw_value is None:
                continue
            try:
                value = int(raw_value)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
        return None

    @staticmethod
    def _first_evidence(evidence: Any, key: str) -> Any:
        if not isinstance(evidence, list):
            return None
        for item in evidence:
            if isinstance(item, dict) and item.get(key) not in {None, ""}:
                return item[key]
        return None
