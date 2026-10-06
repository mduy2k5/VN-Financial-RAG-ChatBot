"""Deterministic answers for schema-backed report composition questions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from difflib import SequenceMatcher
from typing import Any, Iterable, Optional

from ..context.citation_mapper import CitationMapper
from ..schemas import (
    ChunkType,
    Citation,
    FinancialChunk,
    ParsedQuery,
    PeriodType,
    RetrievalResult,
    StatementType,
)
from ..validation.numeric import NumericValidator
from .report_schema_catalog import ReportSchemaCatalog, ReportSchemaNode, get_report_schema_catalog
from .source_groups import source_item_label


@dataclass(frozen=True)
class _ComponentValue:
    node: ReportSchemaNode
    chunk: FinancialChunk
    value: str
    number: Optional[Decimal]


class SchemaDataAnswerer:
    """List and compare the mapped components of a report node without an LLM."""

    def __init__(
        self,
        catalog: Optional[ReportSchemaCatalog] = None,
    ):
        self.catalog = catalog or get_report_schema_catalog()

    def answer(
        self,
        parsed_query: ParsedQuery,
        chunks: Iterable[FinancialChunk],
    ) -> Optional[tuple[str, list[Citation], list[str]]]:
        if not parsed_query.structured_operation:
            return None
        all_chunks = list(chunks)
        if parsed_query.structured_operation == "table":
            return self._answer_statement_table(parsed_query, all_chunks)
        if (
            parsed_query.schema_node_id == "SPECIAL.CASH_FLOW_STRUCTURE"
            and parsed_query.structured_operation in {"components", "composition", "ranking"}
        ):
            return self._answer_cash_flow_structure(parsed_query, all_chunks)
        if not parsed_query.schema_node_id:
            if parsed_query.structured_operation == "summary":
                return self._answer_report_summary(parsed_query, all_chunks)
            return None
        target = self.catalog.nodes.get(parsed_query.schema_node_id)
        group_config = self.catalog.component_groups.get(parsed_query.schema_node_id, {})
        allowed_node_ids: set[str] = set()
        configured_component_ids: Optional[list[str]] = None
        configured_total_ids: Optional[list[str]] = None
        if target is None and group_config:
            resolved_group = self._resolve_component_group(
                group_config, parsed_query, all_chunks
            )
            if resolved_group is None:
                return None
            group_config = resolved_group
            component_ids = [
                str(node_id) for node_id in group_config.get("component_node_ids", [])
                if str(node_id) in self.catalog.nodes
            ]
            total_ids = [
                str(node_id) for node_id in group_config.get("total_node_ids", [])
                if str(node_id) in self.catalog.nodes
            ]
            if group_config.get("total_node_id") in self.catalog.nodes:
                total_ids.append(str(group_config["total_node_id"]))
            allowed_node_ids = set([*component_ids, *total_ids])
            configured_component_ids = component_ids
            configured_total_ids = total_ids
            target = ReportSchemaNode(
                node_id=parsed_query.schema_node_id,
                canonical_name=str(group_config.get("display_name") or parsed_query.schema_node_name or "Các thành phần"),
                statement_type=StatementType(str(group_config.get("statement_type") or (
                    parsed_query.statement_type.value if parsed_query.statement_type else "general"
                ))),
                parent_id=None,
                children=tuple(dict.fromkeys([*component_ids, *total_ids])),
                kind="section",
                sector=None,
                aliases=(),
                parent_contexts=(),
                schema_version=self.catalog.nodes[next(iter(allowed_node_ids))].schema_version if allowed_node_ids else "configured",
            )
        if target is None:
            return None

        if (
            target.node_id in {"INCOME.ENTERPRISE.REVENUE_GROSS", "INCOME.SECURITIES.TOTAL_OPERATING_REVENUE"}
            and parsed_query.structured_operation in {"components", "composition", "ranking"}
            and "doanh thu" in self.catalog.normalize_label(parsed_query.original_query)
        ):
            if self._has_company_sector_chunks(parsed_query, all_chunks, "INCOME.BANK."):
                sector_target_id = "INCOME.BANK.TOTAL_OPERATING_INCOME"
            elif self._has_company_sector_chunks(parsed_query, all_chunks, "INCOME.SECURITIES."):
                sector_target_id = "INCOME.SECURITIES.TOTAL_OPERATING_REVENUE"
            else:
                sector_target_id = ""
            sector_target = self.catalog.nodes.get(sector_target_id) if sector_target_id else None
            if sector_target is not None:
                target = sector_target
                group_config = self.catalog.component_groups.get(target.node_id, {})

        if target.node_id in self.catalog.component_groups and configured_component_ids is None:
            configured_component_ids = [
                str(node_id) for node_id in group_config.get("component_node_ids", [])
                if str(node_id) in self.catalog.nodes
            ] or None
            configured_total_ids = [
                str(node_id) for node_id in group_config.get("total_node_ids", [])
                if str(node_id) in self.catalog.nodes
            ]
            if group_config.get("total_node_id") in self.catalog.nodes:
                configured_total_ids.append(str(group_config["total_node_id"]))
            if not configured_total_ids:
                configured_total_ids = None
            allowed_node_ids.update(configured_component_ids or [])
            allowed_node_ids.update(configured_total_ids or [])

        if (
            target.node_id == "ASSET"
            and parsed_query.structured_operation in {"components", "composition", "ranking"}
        ):
            table_chunks = [
                chunk
                for chunk in all_chunks
                if chunk.chunk_type == ChunkType.TABLE
                and chunk.metadata.get("report_eligible") is not False
                and (parsed_query.fiscal_quarter is None or chunk.fiscal_quarter == parsed_query.fiscal_quarter)
                and (parsed_query.report_type is None or chunk.report_type == parsed_query.report_type)
                and self._chunk_matches_requested_year(parsed_query, chunk)
                and self._chunk_matches_requested_companies(parsed_query, chunk)
            ]
            document_groups = self._choose_documents(parsed_query, target, table_chunks)
            if not document_groups:
                return None
            return self._answer_asset_table(
                parsed_query, target, document_groups, all_chunks
            )

        if not allowed_node_ids:
            allowed_node_ids = set(self.catalog.children.get(target.node_id, target.children))
        scoped = self._scope_chunks(
            parsed_query, all_chunks, allowed_node_ids=allowed_node_ids
        )
        if not scoped:
            return None
        document_groups = self._choose_documents(parsed_query, target, scoped)
        if not document_groups:
            return None

        chunks_by_document: dict[tuple[str, str], list[FinancialChunk]] = {}
        for chunk in all_chunks:
            key = (
                (chunk.company_code or "").casefold(),
                chunk.document_id or chunk.source_file,
            )
            chunks_by_document.setdefault(key, []).append(chunk)

        facts_by_group: list[tuple[str, int, list[_ComponentValue], Optional[_ComponentValue]]] = []
        for company_code, report_year, report_chunks in document_groups:
            period_year = self._value_year(parsed_query, report_year)
            selected_doc_keys = {
                (
                    (chunk.company_code or "").casefold(),
                    chunk.document_id or chunk.source_file,
                )
                for chunk in report_chunks
            }
            report_evidence = [
                chunk
                for key in selected_doc_keys
                for chunk in chunks_by_document.get(key, [])
                if chunk.chunk_type == ChunkType.FINANCIAL_ITEM
                and (
                    chunk.statement_type == target.statement_type
                    or str(chunk.metadata.get("schema_node_id", "")) in allowed_node_ids
                )
                and chunk.metadata.get("report_eligible") is not False
            ]
            components = self._collect_components(
                target, report_evidence, period_year,
                component_node_ids=configured_component_ids,
            )
            unmapped_components = self._collect_unmapped_components(
                target,
                report_evidence,
                period_year,
                component_node_ids=configured_component_ids,
            )
            existing_node_ids = {item.node.node_id for item in components}
            components.extend(
                item for item in unmapped_components
                if item.node.node_id not in existing_node_ids
            )
            if configured_component_ids:
                table_components = self._collect_table_components(
                    parsed_query,
                    target,
                    report_chunks,
                    all_chunks,
                    configured_component_ids,
                    period_year,
                )
                existing_node_ids = {item.node.node_id for item in components}
                components.extend(
                    item for item in table_components
                    if item.node.node_id not in existing_node_ids
                )
            if not components:
                continue
            total = self._find_total(
                target, report_evidence, period_year,
                total_node_ids=configured_total_ids,
            )
            facts_by_group.append((company_code, report_year, components, total))
        if not facts_by_group:
            if parsed_query.structured_operation in {"components", "composition"}:
                return self._answer_aggregate_without_breakdown(
                    parsed_query, target, document_groups
                )
            return None

        selected_chunks: list[FinancialChunk] = []
        for _, _, components, total in facts_by_group:
            selected_chunks.extend(component.chunk for component in components)
            if total:
                selected_chunks.append(total.chunk)
        selected_chunks = list({chunk.chunk_id: chunk for chunk in selected_chunks}.values())
        retrieval_results = [RetrievalResult(chunk=chunk, rrf_score=0.0) for chunk in selected_chunks]
        citations, _ = CitationMapper.map_citations(retrieval_results)
        source_contexts = [chunk.content for chunk in selected_chunks]
        citation_by_chunk = {citation.chunk_id: citation for citation in citations if citation.chunk_id}
        derivation_contexts: list[str] = []
        answer_lines: list[str] = []

        for company_code, report_year, components, total in facts_by_group:
            available_components = [
                item for item in components if item.chunk.chunk_id in citation_by_chunk
            ]
            available_total = (
                total if total and total.chunk.chunk_id in citation_by_chunk else None
            )
            if not available_components:
                continue
            if parsed_query.structured_operation == "ranking":
                available_components.sort(
                    key=lambda item: item.number if item.number is not None else Decimal("-Infinity"),
                    reverse=True,
                )

            period_label = self._period_label(parsed_query, report_year)
            answer_lines.append(f"{target.canonical_name} — {company_code or 'doanh nghiệp'} — {period_label}:")
            if available_total:
                total_citation = citation_by_chunk[available_total.chunk.chunk_id]
                total_unit = available_total.chunk.unit or ""
                unit_suffix = f" {total_unit}" if total_unit else ""
                answer_lines.append(
                    f"Tổng được báo cáo: {available_total.value}{unit_suffix} "
                    f"[{total_citation.citation_id}]."
                )

            ratio_total_number = available_total.number if available_total else None
            ratio_total_unit = available_total.chunk.unit if available_total else None
            ratio_total_citations: list[str] = []
            if available_total:
                citation = citation_by_chunk.get(available_total.chunk.chunk_id)
                if citation:
                    ratio_total_citations.append(citation.citation_id)
            elif (
                parsed_query.structured_operation == "composition"
                and group_config.get("complete")
                and group_config.get("sum_components_if_total_missing")
                and available_components
                and all(item.number is not None for item in available_components)
                and len({item.chunk.unit for item in available_components}) == 1
            ):
                ratio_total_number = sum(
                    (item.number or Decimal(0) for item in available_components), Decimal(0)
                )
                ratio_total_unit = available_components[0].chunk.unit
                ratio_total_citations = list(dict.fromkeys(
                    citation_by_chunk[item.chunk.chunk_id].citation_id
                    for item in available_components
                    if item.chunk.chunk_id in citation_by_chunk
                ))
                answer_lines.append(
                    "Không có dòng tổng riêng trong bảng; tỷ trọng được tính trên tổng các thành phần đã liệt kê."
                )
                derivation_contexts.append(
                    "Tổng thành phần tính toán = " + " + ".join(item.value for item in available_components)
                    + f" = {ratio_total_number} {ratio_total_unit or ''}"
                )

            answer_lines.extend([
                "| Thành phần | Giá trị | Tỷ trọng | Nguồn |",
                "| --- | ---: | ---: | --- |",
            ])

            for item in available_components:
                citation = citation_by_chunk[item.chunk.chunk_id]
                unit = item.chunk.unit or ""
                unit_suffix = f" {unit}" if unit else ""
                ratio_cell = "—"
                if (
                    parsed_query.structured_operation == "composition"
                    and ratio_total_number is not None
                    and item.number is not None
                    and NumericValidator._normalize_unit(item.chunk.unit)
                    == NumericValidator._normalize_unit(ratio_total_unit)
                    and ratio_total_number != 0
                ):
                    ratio = (item.number / ratio_total_number * Decimal(100)).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )
                    ratio_text = format(ratio, "f").replace(".", ",")
                    citation_refs = " ".join(
                        f"[{citation_id}]" for citation_id in ratio_total_citations
                    )
                    ratio_label = "tổng được báo cáo" if available_total else "tổng cấu phần"
                    ratio_cell = f"{ratio_text}% {ratio_label} {citation_refs}".strip()
                    total_value = available_total.value if available_total else str(ratio_total_number)
                    derivation_contexts.append(
                        f"Tính toán có kiểm soát: {item.value} / {total_value} * 100 = {ratio_text}%"
                    )
                answer_lines.append(
                    f"| {item.node.canonical_name} | {item.value}{unit_suffix} | "
                    f"{ratio_cell} | [{citation.citation_id}] |"
                )
                source_contexts.append(
                    f"| {item.node.canonical_name} | {item.value}{unit_suffix} |"
                )

            if (
                parsed_query.structured_operation == "composition"
                and group_config.get("complete")
                and available_total
                and all(item.number is not None for item in available_components)
                and available_total.number is not None
                and sum(
                    (item.number or Decimal(0) for item in available_components), Decimal(0)
                ) != available_total.number
            ):
                answer_lines.append(
                    "Các dòng thành phần tìm thấy chưa cộng khớp tổng được báo cáo; "
                    "các tỷ lệ bên trên là từng dòng chia cho tổng, chưa phải cơ cấu đầy đủ."
                )

        if not answer_lines:
            return None
        answer_lines.append(
            "Chỉ liệt kê các dòng có số liệu đã ánh xạ được vào schema; các dòng chưa ánh xạ không bị suy đoán."
        )
        if parsed_query.structured_operation == "composition" and any(
            re.search(r"\|\s*—\s*\|", line) for line in answer_lines
        ):
            answer_lines.append(
                "Tỷ trọng được để trống khi báo cáo không cung cấp tổng cùng đơn vị hoặc dữ liệu thành phần chưa đối chiếu được với tổng."
            )
        return "\n".join(answer_lines), citations, [*source_contexts, *derivation_contexts]

    def _collect_table_components(
        self,
        parsed_query: ParsedQuery,
        target: ReportSchemaNode,
        report_chunks: list[FinancialChunk],
        all_chunks: list[FinancialChunk],
        component_node_ids: list[str],
        value_year: int,
    ) -> list[_ComponentValue]:
        """Recover configured rows omitted by OCR-to-financial-item mapping."""
        document_ids = {
            chunk.document_id or chunk.source_file for chunk in report_chunks
        }
        report_companies = {
            (chunk.company_code or "").casefold() for chunk in report_chunks
        }
        report_types = {chunk.report_type for chunk in report_chunks if chunk.report_type}
        table_chunks = [
            chunk for chunk in all_chunks
            if chunk.chunk_type == ChunkType.TABLE
            and chunk.statement_type == target.statement_type
            and (chunk.document_id or chunk.source_file) in document_ids
            and (chunk.company_code or "").casefold() in report_companies
            and (not report_types or chunk.report_type in report_types)
            and self._chunk_matches_requested_year(parsed_query, chunk)
            and (parsed_query.fiscal_quarter is None or chunk.fiscal_quarter == parsed_query.fiscal_quarter)
        ]
        if not table_chunks:
            return []
        main_chunks = self._select_main_statement_chunks(target.statement_type, table_chunks)
        rows = self._read_statement_table_rows(main_chunks, value_year, parsed_query)
        allowed_ids = set(component_node_ids)
        found: dict[str, _ComponentValue] = {}
        normalized_fvtpl_prefix = "lai tu cac tstc ghi nhan theo fvtpl"
        normalized_htm_prefix = "lai tu cac khoan dau tu nam giu"
        for label, value, chunk in rows:
            normalized_label = self.catalog.normalize_label(label)
            node: Optional[ReportSchemaNode] = None
            if (
                "INCOME.SECURITIES.FVTPL_GAIN" in allowed_ids
                and normalized_label.startswith(normalized_fvtpl_prefix)
            ):
                node = self.catalog.nodes.get("INCOME.SECURITIES.FVTPL_GAIN")
            elif (
                "INCOME.SECURITIES.HTM_INCOME" in allowed_ids
                and normalized_label.startswith(normalized_htm_prefix)
            ):
                node = self.catalog.nodes.get("INCOME.SECURITIES.HTM_INCOME")
            else:
                node = self._resolve_component_table_node(label, target.statement_type, allowed_ids)
            if node is None or node.node_id in found:
                continue
            found[node.node_id] = _ComponentValue(
                node=node,
                chunk=chunk,
                value=value,
                number=self._decimal_value(value),
            )
        return list(found.values())

    def _resolve_component_table_node(
        self,
        label: str,
        statement_type: StatementType,
        allowed_node_ids: set[str],
    ) -> Optional[ReportSchemaNode]:
        """Resolve a table row within a configured group to avoid cross-sector aliases."""
        normalized_label = self.catalog.normalize_label(label)
        exact_or_prefix: dict[str, tuple[int, int]] = {}
        for node_id in allowed_node_ids:
            node = self.catalog.nodes.get(node_id)
            if node is None or node.statement_type != statement_type:
                continue
            aliases = [node.canonical_name, *node.aliases]
            normalized_aliases = [
                self.catalog.normalize_label(alias) for alias in aliases
            ]
            if any(
                normalized_alias
                and (
                    normalized_label == normalized_alias
                    or normalized_label.startswith(normalized_alias)
                    or normalized_alias in normalized_label
                )
                for normalized_alias in normalized_aliases
            ):
                quality = max(
                    (
                        (int(normalized_label == normalized_alias), len(normalized_alias))
                        for normalized_alias in normalized_aliases
                        if normalized_alias
                        and (
                            normalized_label == normalized_alias
                            or normalized_label.startswith(normalized_alias)
                            or normalized_alias in normalized_label
                        )
                    ),
                    default=(0, 0),
                )
                exact_or_prefix[node.node_id] = quality
        if exact_or_prefix:
            best_quality = max(exact_or_prefix.values())
            best_ids = [node_id for node_id, quality in exact_or_prefix.items() if quality == best_quality]
            return self.catalog.nodes[best_ids[0]] if len(best_ids) == 1 else None

        # OCR can alter one or two characters in a row label. Permit a fuzzy
        # match only inside the target group's schema subtree, with a clear
        # winner, so an unmapped sibling cannot leak into the answer.
        scores: dict[str, float] = {}
        for node_id in allowed_node_ids:
            node = self.catalog.nodes.get(node_id)
            if node is None or node.statement_type != statement_type:
                continue
            for alias in (node.canonical_name, *node.aliases):
                normalized_alias = self.catalog.normalize_label(alias)
                if len(normalized_alias) < 10 or len(normalized_label) < 10:
                    continue
                score = SequenceMatcher(None, normalized_label, normalized_alias).ratio()
                scores[node_id] = max(scores.get(node_id, 0.0), score)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not ranked or ranked[0][1] < 0.88:
            return None
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        if ranked[0][1] - second_score < 0.06:
            return None
        return self.catalog.nodes[ranked[0][0]]

    def _answer_aggregate_without_breakdown(
        self,
        parsed_query: ParsedQuery,
        target: ReportSchemaNode,
        document_groups: list[tuple[str, int, list[FinancialChunk]]],
    ) -> Optional[tuple[str, list[Citation], list[str]]]:
        """Return a verified aggregate when the report has no mapped children."""
        selected: list[_ComponentValue] = []
        for _, report_year, report_chunks in document_groups:
            target_chunks = [
                chunk for chunk in report_chunks
                if chunk.metadata.get("schema_node_id") == target.node_id
            ]
            value = self._unique_value(self._values_for_node(
                target, target_chunks, self._value_year(parsed_query, report_year)
            ))
            if value:
                selected.append(value)
        if not selected:
            return None
        citations, _ = CitationMapper.map_citations([
            RetrievalResult(chunk=value.chunk, rrf_score=0.0) for value in selected
        ])
        citation_by_chunk = {
            citation.chunk_id: citation for citation in citations if citation.chunk_id
        }
        lines: list[str] = []
        contexts = [value.chunk.content for value in selected]
        for value in selected:
            citation = citation_by_chunk.get(value.chunk.chunk_id)
            if citation is None:
                continue
            company_code = value.chunk.company_code or parsed_query.company_code or "doanh nghiệp"
            source_fiscal_year = value.chunk.fiscal_year or parsed_query.fiscal_year
            period_label = self._period_label(parsed_query, source_fiscal_year or 0)
            unit_suffix = f" {value.chunk.unit}" if value.chunk.unit else ""
            lines.append(
                f"{target.canonical_name} — {company_code} — {period_label}: "
                f"{value.value}{unit_suffix} [{citation.citation_id}]."
            )
            contexts.append(f"| {target.canonical_name} | {value.value}{unit_suffix} |")
        if not lines:
            return None
        lines.append(
            "Báo cáo có dòng tổng hợp nhưng chưa có các dòng thành phần chi tiết đã ánh xạ; "
            "vì vậy không thể lập cơ cấu hoặc tính tỷ trọng từ dữ liệu hiện có."
        )
        return "\n".join(lines), citations, contexts

    def _answer_statement_table(
        self,
        parsed_query: ParsedQuery,
        all_chunks: list[FinancialChunk],
    ) -> Optional[tuple[str, list[Citation], list[str]]]:
        """Read the requested full statement from its table chunks."""
        if parsed_query.statement_type is None:
            return None
        table_chunks = [
            chunk
            for chunk in all_chunks
            if chunk.chunk_type == ChunkType.TABLE
            and chunk.metadata.get("report_eligible") is not False
            and (parsed_query.fiscal_quarter is None or chunk.fiscal_quarter == parsed_query.fiscal_quarter)
            and (parsed_query.report_type is None or chunk.report_type == parsed_query.report_type)
            and self._chunk_matches_requested_year(parsed_query, chunk)
            and self._chunk_matches_requested_companies(parsed_query, chunk)
        ]
        if not table_chunks:
            return None

        document_groups = self._choose_documents(parsed_query, None, table_chunks)
        if not document_groups:
            return None

        table_facts: list[tuple[str, int, list[tuple[str, str, FinancialChunk]]]] = []
        for company_code, report_year, report_chunks in document_groups:
            value_year = self._value_year(parsed_query, report_year)
            main_chunks = self._select_main_statement_chunks(
                parsed_query.statement_type, report_chunks
            )
            rows = self._read_statement_table_rows(main_chunks, value_year, parsed_query)
            if not rows:
                # Older ingests can mislabel statement metadata. Retry with
                # the row vocabulary while keeping the same strict table scope.
                main_chunks = self._select_main_statement_chunks(
                    parsed_query.statement_type,
                    report_chunks,
                    content_only=True,
                )
                rows = self._read_statement_table_rows(main_chunks, value_year, parsed_query)
            if rows:
                table_facts.append((company_code, report_year, rows))
        if not table_facts:
            return None

        source_chunks = list({
            chunk.chunk_id: chunk
            for _, _, rows in table_facts
            for _, _, chunk in rows
        }.values())
        citation_chunks = [
            chunk.model_copy(update={
                "statement_type": parsed_query.statement_type,
                "section": parsed_query.statement_type.value,
            })
            for chunk in source_chunks
        ]
        citations, _ = CitationMapper.map_citations([
            RetrievalResult(chunk=chunk, rrf_score=0.0) for chunk in citation_chunks
        ])
        citation_by_chunk = {
            citation.chunk_id: citation for citation in citations if citation.chunk_id
        }
        title = self._statement_label(parsed_query)
        lines: list[str] = []
        for company_code, report_year, rows in table_facts:
            lines.append(
                f"{title} — {company_code or 'doanh nghiệp'} — "
                f"{self._period_label(parsed_query, report_year)}:"
            )
            unit = next((chunk.unit for _, _, chunk in rows if chunk.unit), None)
            if not unit:
                unit = next((
                    match.group(1).strip()
                    for _, _, chunk in rows
                    if (match := re.search(r"(?i)đơn vị\s*:\s*([^|\n]+)", chunk.content))
                ), None)
            lines.append(f"Đơn vị: {unit or 'theo báo cáo'}.")
            lines.extend([
                "| Khoản mục | Giá trị | Trang / nguồn |",
                "| --- | ---: | --- |",
            ])
            for label, value, chunk in rows:
                citation = citation_by_chunk.get(chunk.chunk_id)
                if citation is None:
                    continue
                page = f"trang {chunk.page}" if chunk.page else ""
                location = f"{page} [{citation.citation_id}]".strip()
                row_unit = chunk.unit or unit
                value_with_unit = f"{value} {row_unit}" if row_unit else value
                lines.append(f"| {label} | {value_with_unit} | {location} |")
        if not lines:
            return None
        lines.append("Các dòng được trích trực tiếp từ bảng báo cáo trong tài liệu đã nạp.")
        source_contexts = [chunk.content for chunk in source_chunks]
        effective_units: dict[str, str] = {}
        for _, _, rows in table_facts:
            report_unit = next((chunk.unit for _, _, chunk in rows if chunk.unit), None)
            if not report_unit:
                report_unit = next((
                    match.group(1).strip()
                    for _, _, chunk in rows
                    if (match := re.search(r"(?i)đơn vị\s*:\s*([^|\n]+)", chunk.content))
                ), None)
            for _, _, chunk in rows:
                effective_units[chunk.chunk_id] = chunk.unit or report_unit or ""
        source_contexts.extend(
            f"| {label} | {value} {effective_units.get(chunk.chunk_id, '')} |"
            for _, _, rows in table_facts
            for label, value, chunk in rows
        )
        return "\n".join(lines), citations, source_contexts

    def _answer_cash_flow_structure(
        self,
        parsed_query: ParsedQuery,
        all_chunks: list[FinancialChunk],
    ) -> Optional[tuple[str, list[Citation], list[str]]]:
        """Read cash-flow activity totals from the report table itself.

        Cash-flow totals are often absent from the normalized financial-item
        index (especially in OCR text), even though the statement table has
        the reported totals. Parse those rows directly and keep the answer
        tied to the selected company's consolidated report.
        """
        table_chunks = [
            chunk
            for chunk in all_chunks
            if chunk.chunk_type == ChunkType.TABLE
            and chunk.statement_type == StatementType.CASH_FLOW
            and chunk.metadata.get("report_eligible") is not False
            and (parsed_query.fiscal_quarter is None or chunk.fiscal_quarter == parsed_query.fiscal_quarter)
            and (parsed_query.report_type is None or chunk.report_type == parsed_query.report_type)
            and self._chunk_matches_requested_year(parsed_query, chunk)
            and self._chunk_matches_requested_companies(parsed_query, chunk)
        ]
        if not table_chunks:
            return None

        document_groups = self._choose_documents(parsed_query, None, table_chunks)
        if not document_groups:
            return None

        # Internal activity keys and their Vietnamese labels are stable even
        # when OCR introduces minor diacritic or spelling differences.
        activity_labels = {
            "operating": "Lưu chuyển tiền thuần từ hoạt động kinh doanh",
            "investing": "Lưu chuyển tiền thuần từ hoạt động đầu tư",
            "financing": "Lưu chuyển tiền thuần từ hoạt động tài chính",
        }
        facts: list[tuple[str, int, list[tuple[str, str, FinancialChunk]], Optional[tuple[str, str, FinancialChunk]]]] = []
        for company_code, report_year, report_chunks in document_groups:
            main_chunks = self._select_main_statement_chunks(
                StatementType.CASH_FLOW, report_chunks
            )
            value_year = self._value_year(parsed_query, report_year)
            parsed_rows = self._read_statement_table_rows(
                main_chunks, value_year, parsed_query
            )
            if not parsed_rows:
                # Metadata can be inconsistent in older ingests; retain the
                # strict report/document scope and retry with row vocabulary.
                main_chunks = self._select_main_statement_chunks(
                    StatementType.CASH_FLOW, report_chunks, content_only=True
                )
                parsed_rows = self._read_statement_table_rows(
                    main_chunks, value_year, parsed_query
                )

            grouped_values: dict[str, dict[str, tuple[str, str, FinancialChunk]]] = {
                key: {} for key in activity_labels
            }
            net_change_values: dict[str, tuple[str, str, FinancialChunk]] = {}
            for label, value, chunk in parsed_rows:
                normalized = self.catalog.normalize_label(label)
                if "luu chuyen" not in normalized or "tien" not in normalized or "thuan" not in normalized:
                    continue
                if "truoc nhung thay doi" in normalized or "truoc thay doi" in normalized:
                    continue
                if "trong nam" in normalized or "trong ky" in normalized:
                    net_change_values.setdefault(normalized, (label, value, chunk))
                    continue
                for key, activity_label in activity_labels.items():
                    normalized_activity = self.catalog.normalize_label(activity_label)
                    # The label may include an OCR variant of "thuần", but
                    # the activity phrase remains a reliable discriminator.
                    activity_phrase = normalized_activity.replace("luu chuyen tien thuan tu ", "")
                    if activity_phrase in normalized:
                        grouped_values[key].setdefault(normalized, (label, value, chunk))
                        break

            # Duplicate chunks created at table boundaries are harmless when
            # they report the same amount. Conflicting values are omitted so
            # we never choose between different figures silently.
            selected_activities: list[tuple[str, str, FinancialChunk]] = []
            for key, candidates in grouped_values.items():
                unique_amounts = {self._decimal_value(item[1]) for item in candidates.values()}
                if len(unique_amounts) == 1:
                    selected_activities.append(next(iter(candidates.values())))
            unique_net_changes = {self._decimal_value(item[1]) for item in net_change_values.values()}
            net_change = (
                next(iter(net_change_values.values()))
                if len(unique_net_changes) == 1
                else None
            )
            if selected_activities:
                facts.append((company_code, report_year, selected_activities, net_change))

        if not facts:
            return None

        source_chunks = list({
            chunk.chunk_id: chunk
            for _, _, activities, net_change in facts
            for _, _, chunk in [*activities, *([net_change] if net_change else [])]
        }.values())
        citations, _ = CitationMapper.map_citations([
            RetrievalResult(chunk=chunk, rrf_score=0.0) for chunk in source_chunks
        ])
        citation_by_chunk = {
            citation.chunk_id: citation for citation in citations if citation.chunk_id
        }
        lines: list[str] = []
        source_contexts = [chunk.content for chunk in source_chunks]
        for company_code, report_year, activities, net_change in facts:
            available = [
                item for item in activities
                if item[2].chunk_id in citation_by_chunk
            ]
            if not available:
                continue
            unit = next((item[2].unit for item in available if item[2].unit), None)
            if not unit:
                unit = next((
                    match.group(1).strip()
                    for _, _, chunk in available
                    if (match := re.search(r"(?i)đơn vị\s*:\s*([^|\n]+)", chunk.content))
                ), None)
            total_number = self._decimal_value(net_change[1]) if net_change else None
            component_numbers = [self._decimal_value(value) for _, value, _ in available]
            reconciles = bool(
                net_change
                and total_number is not None
                and all(number is not None for number in component_numbers)
                and sum((number for number in component_numbers if number is not None), Decimal(0)) == total_number
                and total_number != 0
            )
            lines.append(
                f"Cơ cấu dòng tiền — {company_code or 'doanh nghiệp'} — "
                f"{self._period_label(parsed_query, report_year)}:"
            )
            if net_change and net_change[2].chunk_id in citation_by_chunk:
                total_citation = citation_by_chunk[net_change[2].chunk_id]
                total_suffix = f" {unit}" if unit else ""
                lines.append(
                    f"Lưu chuyển tiền thuần trong năm/kỳ: {net_change[1]}{total_suffix} "
                    f"[{total_citation.citation_id}]."
                )
                source_contexts.append(
                    f"| Lưu chuyển tiền thuần trong năm/kỳ | {net_change[1]}{total_suffix} |"
                )
            lines.extend([
                "| Hoạt động | Lưu chuyển tiền thuần | Tỷ trọng trong dòng tiền thuần | Nguồn |",
                "| --- | ---: | ---: | --- |",
            ])
            for label, value, chunk in available:
                citation = citation_by_chunk.get(chunk.chunk_id)
                if citation is None:
                    continue
                value_number = self._decimal_value(value)
                ratio_cell = "—"
                if reconciles and net_change is not None and value_number is not None and total_number is not None:
                    ratio = (value_number / total_number * Decimal(100)).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )
                    ratio_text = format(ratio, "f").replace(".", ",")
                    ratio_cell = f"{ratio_text}%"
                    source_contexts.append(
                        f"Tính toán có kiểm soát: {value} / {net_change[1]} * 100 = {ratio_text}%"
                    )
                unit_suffix = f" {chunk.unit or unit}" if (chunk.unit or unit) else ""
                lines.append(
                    f"| {label} | {value}{unit_suffix} | {ratio_cell} | [{citation.citation_id}] |"
                )
                source_contexts.append(f"| {label} | {value}{unit_suffix} |")
            if not reconciles:
                lines.append(
                    "Tỷ trọng được để trống vì các dòng hoạt động hiện có không đối chiếu khớp "
                    "với dòng lưu chuyển tiền thuần được báo cáo."
                )
        if not lines:
            return None
        lines.append("Các khoản lưu chuyển được trích từ bảng báo cáo; không suy diễn dòng hoạt động không có trong nguồn.")
        return "\n".join(lines), citations, source_contexts

    def _select_main_statement_chunks(
        self,
        statement_type: Any,
        chunks: list[FinancialChunk],
        *,
        content_only: bool = False,
    ) -> list[FinancialChunk]:
        """Keep primary statement tables and reject similarly tagged note tables."""
        title_markers = {
            "balance_sheet": ("bang can doi ke toan", "bao cao tinh hinh tai chinh"),
            "income_statement": (
                "bao cao ket qua hoat dong kinh doanh", "bao cao ket qua kinh doanh",
                "bao cao ket qua hoat dong",
            ),
            "cash_flow": (
                "bao cao luu chuyen tien te", "luu chuyen tien te",
                "luu chuyen tien tu hoat dong",
            ),
        }.get(statement_type.value, ())
        positive_marker_map = {
            "balance_sheet": (
                "tai san ngan han", "tai san dai han", "tong cong tai san",
                "tong tai san", "tong no phai tra", "von chu so huu", "nguon von",
            ),
            "income_statement": (
                "doanh thu ban hang", "doanh thu thuan", "gia von hang ban",
                "loi nhuan gop", "loi nhuan thuan", "tong loi nhuan",
                "loi nhuan sau thue", "thu nhap lai thuan", "doanh thu hoat dong",
                "tong doanh thu hoat dong", "chi phi hoat dong",
                "tong chi phi hoat dong", "loi nhuan ke toan truoc thue",
            ),
            "cash_flow": (
                "luu chuyen tien tu hoat dong", "luu chuyen tien thuan",
                "tien va cac khoan tuong duong tien dau nam",
                "tien va cac khoan tuong duong tien cuoi nam",
            ),
        }
        negative_markers = {
            "balance_sheet": (
                "chi tieu ngoai bao cao", "ngoai bang can doi",
            ),
            "income_statement": (
                "luu chuyen tien", "thong tin bo sung",
                "chi phi san xuat va kinh doanh theo yeu to",
            ),
            "cash_flow": ("thong tin bo sung",),
        }.get(statement_type.value, ())

        grouped: dict[tuple[str, int], list[FinancialChunk]] = {}
        for chunk in chunks:
            raw_title = chunk.table_title or str(chunk.metadata.get("table_title") or "")
            normalized_title = self.catalog.normalize_label(raw_title)
            normalized_title = re.sub(r"\s+tiep theo\s*$", "", normalized_title)
            # Table titles often contain only a form code (for example,
            # "49/2014/TT-NHNN"), reused throughout the entire report. Keep
            # every title page-scoped so that same-code balance, income and
            # note tables cannot collapse into one group.
            group_key = (normalized_title, chunk.page or 0)
            grouped.setdefault(group_key, []).append(chunk)

        all_title_markers = {
            "balance_sheet": ("bang can doi ke toan", "bao cao tinh hinh tai chinh"),
            "income_statement": (
                "bao cao ket qua hoat dong kinh doanh", "bao cao ket qua kinh doanh",
                "bao cao ket qua hoat dong",
            ),
            "cash_flow": (
                "bao cao luu chuyen tien te", "luu chuyen tien te",
                "luu chuyen tien tu hoat dong",
            ),
        }
        metrics: dict[tuple[str, int], dict[str, Any]] = {}
        for group_key, group in grouped.items():
            normalized_title = group_key[0]
            raw_title = group[0].table_title or str(group[0].metadata.get("table_title") or "")
            excluded_title = any(marker in normalized_title for marker in (
                "chi tieu ngoai bao cao", "ngoai bang can doi", "thong tin bo sung",
                "chi tiet", "thuyet minh", "ghi nhan trong bao cao",
            ))
            numbered_note_title = bool(re.match(r"^\s*(?:\([a-z]\)\s*)?\d{1,3}[.)]\s+", raw_title, re.IGNORECASE))
            letter_note_title = bool(re.match(r"^\s*\([a-z]\)\s+", raw_title, re.IGNORECASE))
            if letter_note_title or (
                numbered_note_title
                and not any(marker in normalized_title for marker in title_markers)
            ):
                excluded_title = True
            if "su kien sau ngay" in normalized_title:
                excluded_title = True
            title_match = (
                not excluded_title
                and any(marker in normalized_title for marker in title_markers)
            )
            content = self.catalog.normalize_label("\n".join(chunk.content for chunk in group))
            # Ignore retained-earnings labels when classifying an income
            # statement; they occur on balance sheets but are not income rows.
            score_content = content.replace("loi nhuan sau thue chua phan phoi", "")
            scores = {
                statement: sum(1 for marker in markers if marker in score_content)
                for statement, markers in positive_marker_map.items()
            }
            positive_hits = scores.get(statement_type.value, 0)
            other_hits = max(
                (score for statement, score in scores.items() if statement != statement_type.value),
                default=0,
            )
            negative_hits = sum(1 for marker in negative_markers if marker in content)
            if excluded_title:
                negative_hits = max(1, negative_hits)
            metadata_match = any(
                chunk.statement_type == statement_type
                or (chunk.section or "").casefold() == statement_type.value
                for chunk in group
            )
            note_group = all(
                chunk.statement_type.value == "notes" or (chunk.section or "").casefold() == "notes"
                for chunk in group
            )
            other_title_match = any(
                marker in normalized_title
                for other_statement, markers in all_title_markers.items()
                if other_statement != statement_type.value
                for marker in markers
            )
            metrics[group_key] = {
                "group": group,
                "title_match": title_match,
                "excluded_title": excluded_title,
                "positive_hits": positive_hits,
                "other_hits": other_hits,
                "negative_hits": negative_hits,
                "metadata_match": metadata_match,
                "note_group": note_group,
                "other_title_match": other_title_match,
            }

        anchored = set() if content_only else {
            key
            for key, metric in metrics.items()
            if metric["title_match"] and not metric["negative_hits"]
        }
        if anchored:
            anchor_pages = sorted({key[1] for key in anchored})
            primary_pages = {anchor_pages[0]}
            last_page = anchor_pages[0]
            for page in anchor_pages[1:]:
                if page - last_page > 2:
                    break
                primary_pages.add(page)
                last_page = page
            anchored = {key for key in anchored if key[1] in primary_pages}
        selected_keys: set[tuple[str, int]] = set(anchored)
        if anchored:
            anchor_pages = [key[1] for key in anchored]
            for key, metric in metrics.items():
                if key in anchored or metric["excluded_title"] or metric["other_title_match"]:
                    continue
                if metric["negative_hits"] or metric["note_group"]:
                    continue
                content_supports_statement = (
                    metric["positive_hits"] > 0
                    and metric["positive_hits"] >= metric["other_hits"]
                )
                untitled_continuation = (
                    metric["metadata_match"]
                    and metric["positive_hits"] == 0
                    and metric["other_hits"] == 0
                )
                if (content_supports_statement or untitled_continuation) and any(
                    abs(key[1] - anchor_page) <= 1 for anchor_page in anchor_pages
                ):
                    selected_keys.add(key)
        else:
            # For title-less reports, compare this page's row vocabulary
            # against the other statement families. This prevents a balance
            # sheet page that mentions retained earnings from being treated
            # as an income statement, and separates cash-flow continuations.
            selected_keys = {
                key
                for key, metric in metrics.items()
                if not metric["excluded_title"]
                and not metric["other_title_match"]
                and not metric["negative_hits"]
                and not metric["note_group"]
                and metric["positive_hits"] > 0
                and metric["positive_hits"] >= metric["other_hits"]
            }
            candidate_pages = sorted({key[1] for key in selected_keys})
            if candidate_pages:
                primary_pages = {candidate_pages[0]}
                last_page = candidate_pages[0]
                for page in candidate_pages[1:]:
                    if page - last_page > 2:
                        break
                    primary_pages.add(page)
                    last_page = page
                selected_keys = {key for key in selected_keys if key[1] in primary_pages}
            selected_pages = [key[1] for key in selected_keys]
            for key, metric in metrics.items():
                if key in selected_keys or key[0] or metric["excluded_title"] or metric["other_title_match"]:
                    continue
                if metric["negative_hits"] or metric["note_group"]:
                    continue
                if not metric["metadata_match"] or metric["positive_hits"] or metric["other_hits"]:
                    continue
                if any(abs(key[1] - page) <= 1 for page in selected_pages):
                    selected_keys.add(key)

        selected = [
            chunk
            for key in selected_keys
            for chunk in metrics[key]["group"]
        ]

        return sorted(
            {chunk.chunk_id: chunk for chunk in selected}.values(),
            key=lambda chunk: (
                chunk.page or 0,
                int(chunk.metadata.get("row_start", 0) or 0),
                chunk.chunk_id,
            ),
        )

    def _read_statement_table_rows(
        self,
        table_chunks: list[FinancialChunk],
        value_year: int,
        parsed_query: ParsedQuery,
    ) -> list[tuple[str, str, FinancialChunk]]:
        rows: list[tuple[str, str, FinancialChunk]] = []
        seen: set[tuple[Optional[int], str, str]] = set()
        # Preserve the column layout across page and chunk boundaries. Some
        # extracted statements only print the year header on the first page.
        period_column: Optional[int] = None
        code_column: Optional[int] = None
        label_column = 0
        for chunk in table_chunks:
            for line in chunk.content.splitlines():
                if not line.lstrip().startswith("|"):
                    continue
                cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
                if not cells or all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells):
                    continue
                matching_year_columns = [
                    index for index, cell in enumerate(cells)
                    if re.search(rf"(?<!\d){value_year}(?!\d)", cell)
                ]
                normalized_headers = [self.catalog.normalize_label(cell) for cell in cells]
                code_header_present = any(
                    header in {"ma so", "ma chi tieu", "code"}
                    for header in normalized_headers
                )
                if matching_year_columns and (period_column is None or code_header_present):
                    if (
                        parsed_query.period_type == PeriodType.BEGINNING_OF_YEAR
                        and len(matching_year_columns) > 1
                    ):
                        period_column = matching_year_columns[-1]
                    else:
                        period_column = matching_year_columns[0]
                    code_column = next(
                        (
                            index for index, header in enumerate(normalized_headers)
                            if header in {"ma so", "ma chi tieu", "code"}
                        ),
                        None,
                    )
                    label_column = 0 if code_column != 0 else 1
                    continue
                if period_column is None:
                    period_column = self._relative_period_column(cells, value_year, chunk)
                    if period_column is not None:
                        code_column = next(
                            (
                                index for index, header in enumerate(normalized_headers)
                                if header in {"ma so", "ma chi tieu", "code"}
                            ),
                            None,
                        )
                        label_column = 0 if code_column != 0 else 1
                        continue
                if (
                    period_column is None
                    and parsed_query.statement_type is not None
                    and parsed_query.statement_type.value == "cash_flow"
                ):
                    # Some extracted cash-flow tables drop the repeated year
                    # header on continuation pages. Their numeric value columns
                    # are the final two cells; use report metadata to select the
                    # requested side while preserving the preceding label/code.
                    if value_year == chunk.fiscal_year:
                        inferred_column = len(cells) - 2
                    elif value_year == chunk.comparative_fiscal_year:
                        inferred_column = len(cells) - 1
                    else:
                        inferred_column = -1
                    if (
                        0 <= inferred_column < len(cells)
                        and self._decimal_value(cells[inferred_column]) is not None
                    ):
                        period_column = inferred_column
                        code_column = (
                            1
                            if len(cells) >= 4
                            and re.fullmatch(r"\d+[a-z]?", cells[1].strip(), re.IGNORECASE)
                            else None
                        )
                        label_column = 0
                if period_column is None or period_column >= len(cells):
                    continue
                value = self._first_table_number(cells[period_column])
                if value is None:
                    continue
                if code_column is not None:
                    text_cells = [
                        (index, cell)
                        for index, cell in enumerate(cells[:period_column])
                        if index != code_column
                        and cell
                        and cell not in {"-", "—", "*"}
                        and self._decimal_value(cell) is None
                        and not re.fullmatch(r"(?:\d+[a-z]?|[ivxlcdm]+|[a-z])", cell, re.IGNORECASE)
                    ]
                    if text_cells:
                        label = max(text_cells, key=lambda item: (len(item[1]), item[0]))[1]
                    else:
                        label = cells[label_column] if label_column < len(cells) else ""
                else:
                    first = cells[0] if cells else ""
                    second = cells[1] if len(cells) > 1 else ""
                    first_is_code = bool(
                        re.fullmatch(
                            r"(?:\d+[a-z]?|[ivxlcdm]+|[a-z])",
                            first.strip(),
                            re.IGNORECASE,
                        )
                    )
                    label = second if first_is_code and second else first or second
                label = re.sub(r"\s+", " ", label).strip()
                if not label:
                    continue
                signature = (chunk.page, self.catalog.normalize_label(label), value)
                if signature in seen:
                    continue
                seen.add(signature)
                rows.append((label, value, chunk))
        return rows

    def _relative_period_column(
        self,
        cells: list[str],
        value_year: int,
        chunk: FinancialChunk,
    ) -> Optional[int]:
        """Infer a year column from relative headings and report metadata."""
        if value_year == chunk.fiscal_year:
            wanted_role = "current"
        elif value_year == chunk.comparative_fiscal_year:
            wanted_role = "previous"
        else:
            return None

        current_markers = (
            "so cuoi nam", "cuoi nam", "nam nay", "ky nay", "cuoi ky",
            "current year", "current period", "ending balance",
        )
        previous_markers = (
            "so dau nam", "dau nam", "nam truoc", "ky truoc", "dau ky",
            "previous year", "previous period", "prior period", "opening balance",
            "comparative",
        )
        markers = current_markers if wanted_role == "current" else previous_markers
        return next(
            (
                index
                for index, cell in enumerate(cells)
                if any(marker in self.catalog.normalize_label(cell) for marker in markers)
            ),
            None,
        )

    @staticmethod
    def _chunk_matches_requested_year(parsed_query: ParsedQuery, chunk: FinancialChunk) -> bool:
        years = parsed_query.fiscal_years or (
            [parsed_query.fiscal_year] if parsed_query.fiscal_year is not None else []
        )
        if not years:
            return True
        return any(
            chunk.fiscal_year == year
            or chunk.comparative_fiscal_year == year
            or str(year) in (chunk.metadata.get("period_values") or {})
            for year in years
        )

    @staticmethod
    def _chunk_matches_requested_companies(parsed_query: ParsedQuery, chunk: FinancialChunk) -> bool:
        companies = parsed_query.company_codes or (
            [parsed_query.company_code] if parsed_query.company_code else []
        )
        return not companies or (chunk.company_code or "").casefold() in {
            company.casefold() for company in companies
        }

    @staticmethod
    def _statement_label(parsed_query: ParsedQuery) -> str:
        if parsed_query.statement_name_vi:
            return parsed_query.statement_name_vi
        if parsed_query.statement_type is None:
            return "Báo cáo tài chính"
        return parsed_query.statement_type.value.replace("_", " ").title()

    def _answer_asset_table(
        self,
        parsed_query: ParsedQuery,
        target: ReportSchemaNode,
        document_groups: list[tuple[str, int, list[FinancialChunk]]],
        all_chunks: list[FinancialChunk],
    ) -> Optional[tuple[str, list[Citation], list[str]]]:
        """Read high-level asset rows from the balance-sheet table itself.

        Financial-item aliases can also match note tables and nested detail rows.
        For a request for the asset table, use the source table's own section and
        row codes so collateral notes or breakdown rows cannot replace reported
        balance-sheet totals.
        """
        table_facts: list[
            tuple[str, int, list[_ComponentValue], Optional[_ComponentValue]]
        ] = []
        for company_code, report_year, report_chunks in document_groups:
            document_ids = {
                chunk.document_id or chunk.source_file for chunk in report_chunks
            }
            report_types = {
                chunk.report_type for chunk in report_chunks if chunk.report_type
            }
            if parsed_query.report_type:
                report_types = {parsed_query.report_type}
            elif "consolidated" in report_types:
                report_types = {"consolidated"}
            elif report_types:
                report_types = {sorted(report_types)[0]}

            table_chunks = [
                chunk
                for chunk in all_chunks
                if chunk.chunk_type == ChunkType.TABLE
                and (chunk.company_code or "").casefold() == (company_code or "").casefold()
                and (chunk.document_id or chunk.source_file) in document_ids
                and (
                    chunk.fiscal_year == report_year
                    or chunk.comparative_fiscal_year == report_year
                    or str(report_year) in (chunk.metadata.get("period_values") or {})
                )
                and (not report_types or chunk.report_type in report_types)
            ]
            if not table_chunks:
                continue

            value_year = self._value_year(parsed_query, report_year)
            rows, total = self._read_asset_table_rows(
                target, table_chunks, value_year
            )
            if rows and total:
                table_facts.append((company_code, report_year, rows, total))

        if not table_facts:
            return None

        source_chunks = list({
            item.chunk.chunk_id: item.chunk
            for _, _, rows, total in table_facts
            for item in [*rows, *([total] if total else [])]
        }.values())
        retrieval_results = [
            RetrievalResult(chunk=chunk, rrf_score=0.0) for chunk in source_chunks
        ]
        citations, _ = CitationMapper.map_citations(retrieval_results)
        citation_by_chunk = {
            citation.chunk_id: citation for citation in citations if citation.chunk_id
        }
        lines: list[str] = []
        derivation_contexts: list[str] = []

        for company_code, report_year, rows, total in table_facts:
            available_rows = [
                row for row in rows if row.chunk.chunk_id in citation_by_chunk
            ]
            available_total = (
                total if total and total.chunk.chunk_id in citation_by_chunk else None
            )
            if not available_rows:
                continue
            if parsed_query.structured_operation == "ranking":
                available_rows.sort(
                    key=lambda item: item.number if item.number is not None else Decimal("-Infinity"),
                    reverse=True,
                )

            period_label = self._period_label(parsed_query, report_year)
            lines.append(f"{target.canonical_name} — {company_code or 'doanh nghiệp'} — {period_label}:")
            if available_total:
                total_citation = citation_by_chunk[available_total.chunk.chunk_id]
                total_unit = available_total.chunk.unit or ""
                unit_suffix = f" {total_unit}" if total_unit else ""
                lines.append(
                    f"Tổng tài sản được báo cáo: {available_total.value}{unit_suffix} "
                    f"[{total_citation.citation_id}]."
                )
            lines.extend([
                "| Khoản mục | Giá trị | Tỷ trọng | Nguồn |",
                "| --- | ---: | ---: | --- |",
            ])
            for item in available_rows:
                citation = citation_by_chunk[item.chunk.chunk_id]
                unit = item.chunk.unit or ""
                unit_suffix = f" {unit}" if unit else ""
                ratio_cell = "—"
                if (
                    parsed_query.structured_operation == "composition"
                    and available_total
                    and item.number is not None
                    and available_total.number is not None
                    and item.chunk.unit == available_total.chunk.unit
                    and available_total.number != 0
                ):
                    ratio = (item.number / available_total.number * Decimal(100)).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )
                    ratio_text = format(ratio, "f").replace(".", ",")
                    total_citation = citation_by_chunk[available_total.chunk.chunk_id]
                    ratio_cell = f"{ratio_text}% tổng [{total_citation.citation_id}]"
                    derivation_contexts.append(
                        f"Tính toán có kiểm soát: {item.value} / "
                        f"{available_total.value} * 100 = {ratio_text}%"
                    )
                lines.append(
                    f"| {item.node.canonical_name} | {item.value}{unit_suffix} | "
                    f"{ratio_cell} | [{citation.citation_id}] |"
                )

        if not lines:
            return None
        lines.append("Các khoản mục được lấy từ các dòng cấp cao của bảng cân đối kế toán.")
        source_contexts = [chunk.content for chunk in source_chunks]
        for _, _, rows, total in table_facts:
            if total:
                total_unit = total.chunk.unit or ""
                source_contexts.append(
                    f"| Tổng tài sản | {total.value} {total_unit} |"
                )
            source_contexts.extend(
                f"| {item.node.canonical_name} | {item.value} {item.chunk.unit or ''} |"
                for item in rows
            )
        source_contexts.extend(derivation_contexts)
        return "\n".join(lines), citations, source_contexts

    def _read_asset_table_rows(
        self,
        target: ReportSchemaNode,
        table_chunks: list[FinancialChunk],
        value_year: int,
    ) -> tuple[list[_ComponentValue], Optional[_ComponentValue]]:
        # Keep parser state across consecutive pages: statutory balance sheets
        # commonly split current assets, long-term assets, and the total.
        by_document: dict[str, list[FinancialChunk]] = {}
        for chunk in table_chunks:
            by_document.setdefault(chunk.document_id or chunk.source_file, []).append(chunk)

        roman_code = re.compile(
            r"(?=[MDCLXVI])M{0,4}(?:CM|CD|D?C{0,3})"
            r"(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})$",
            re.IGNORECASE,
        )
        reconciled: list[tuple[list[_ComponentValue], _ComponentValue, int]] = []

        for document_chunks in by_document.values():
            document_chunks.sort(key=lambda chunk: (
                chunk.page or 0,
                int(chunk.metadata.get("row_start", 0) or 0),
                chunk.chunk_id,
            ))
            in_asset_section = False
            period_column: Optional[int] = None
            code_column: Optional[int] = None
            label_column = 0
            column_layout_known = False
            roman_rows: list[_ComponentValue] = []
            numeric_100_rows: list[_ComponentValue] = []
            numeric_10_rows: list[_ComponentValue] = []
            numeric_other_rows: list[_ComponentValue] = []
            uncoded_rows: list[_ComponentValue] = []
            has_any_code = False
            total: Optional[_ComponentValue] = None
            row_index = 0

            for chunk in document_chunks:
                if total is not None:
                    break
                for line in chunk.content.splitlines():
                    if not line.lstrip().startswith("|"):
                        continue
                    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
                    if not cells or all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells):
                        continue
                    matching_year_columns = [
                        index for index, cell in enumerate(cells)
                        if re.search(rf"(?<!\d){value_year}(?!\d)", cell)
                    ]
                    normalized_headers = [self.catalog.normalize_label(cell) for cell in cells]
                    if matching_year_columns:
                        period_column = matching_year_columns[0]
                        code_header = next((
                            index for index, header in enumerate(normalized_headers)
                            if header in {"ma so", "ma chi tieu", "code"}
                        ), None)
                        note_header = next((
                            index for index, header in enumerate(normalized_headers)
                            if header.startswith("thuyet minh")
                        ), None)
                        if code_header is not None:
                            code_column = code_header
                            label_column = self._asset_label_column(normalized_headers, code_header)
                            column_layout_known = True
                        elif note_header is not None:
                            code_column = 0 if note_header >= 2 else None
                            label_column = 1 if note_header >= 2 else 0
                            column_layout_known = True
                        continue
                    relative_period_column = self._relative_period_column(
                        cells, value_year, chunk
                    )
                    if relative_period_column is not None:
                        period_column = relative_period_column
                        code_header = next((
                            index for index, header in enumerate(normalized_headers)
                            if header in {"ma so", "ma chi tieu", "code"}
                        ), None)
                        note_header = next((
                            index for index, header in enumerate(normalized_headers)
                            if header.startswith("thuyet minh")
                        ), None)
                        if code_header is not None:
                            code_column = code_header
                            label_column = self._asset_label_column(normalized_headers, code_header)
                            column_layout_known = True
                        elif note_header is not None:
                            code_column = 0 if note_header >= 2 else None
                            label_column = 1 if note_header >= 2 else 0
                            column_layout_known = True
                        continue
                    if period_column is None or period_column >= len(cells):
                        continue

                    first_cell = cells[0]
                    second_cell = cells[1] if len(cells) > 1 else ""
                    if column_layout_known:
                        code = cells[code_column] if code_column is not None and code_column < len(cells) else ""
                        label = cells[label_column] if label_column < len(cells) else first_cell or second_cell
                        has_any_code = has_any_code or bool(code)
                    else:
                        first_code = re.sub(r"[.)\s]+$", "", first_cell).strip().upper()
                        first_is_code = bool(
                            re.fullmatch(r"\d+", first_code)
                            or (first_code and roman_code.fullmatch(first_code))
                        )
                        if first_is_code and second_cell:
                            code, label = first_cell, second_cell
                            has_any_code = True
                        else:
                            code, label = "", first_cell or second_cell

                    label = re.sub(r"^\s*[A-Z]\.\s*", "", label, flags=re.IGNORECASE)
                    normalized_label = self.catalog.normalize_label(label)
                    if not normalized_label:
                        continue
                    if normalized_label == "tai san" or normalized_label.startswith("tai san "):
                        in_asset_section = True
                        heading_value = cells[period_column].strip()
                        if not heading_value or self._decimal_value(heading_value) is None:
                            continue
                    if not in_asset_section:
                        continue
                    if normalized_label.startswith(("tong tai san", "tong cong tai san")):
                        value = cells[period_column]
                        if value and value not in {"-", "—"}:
                            total = self._table_component_value(
                                target, label, value, chunk, row_index, "TOTAL_ASSETS"
                            )
                        in_asset_section = False
                        break

                    value = cells[period_column]
                    if not value or value in {"-", "—"}:
                        continue
                    number = self._decimal_value(value)
                    if number is None:
                        continue
                    normalized_code = re.sub(r"[.)\s]+$", "", code).strip().upper()
                    item = self._table_component_value(
                        target, label, value, chunk, row_index, f"ROW_{row_index + 1}"
                    )
                    row_index += 1
                    if normalized_label.startswith("trong do"):
                        continue
                    if normalized_code and roman_code.fullmatch(normalized_code):
                        roman_rows.append(item)
                    elif normalized_code == "1" and not roman_rows:
                        # Some bank statements number their first asset row
                        # "1" and switch to Roman numerals for the remaining
                        # high-level categories.
                        roman_rows.append(item)
                    elif re.fullmatch(r"\d{3}", normalized_code):
                        numeric_code = int(normalized_code)
                        if numeric_code % 100 == 0:
                            numeric_100_rows.append(item)
                        elif numeric_code % 100 == 10:
                            numeric_10_rows.append(item)
                        elif numeric_code % 10 == 0:
                            numeric_other_rows.append(item)
                    elif not normalized_code:
                        uncoded_rows.append(item)

            if total is None or total.number is None:
                continue
            candidate_sets: list[list[_ComponentValue]] = [
                roman_rows, numeric_100_rows, numeric_10_rows, numeric_other_rows,
            ]
            if uncoded_rows and not has_any_code:
                candidate_sets.append(self._collapse_subtotal_rows(uncoded_rows))
            for candidate_rows in candidate_sets:
                if not candidate_rows:
                    continue
                component_sum = sum(
                    (row.number or Decimal(0) for row in candidate_rows), Decimal(0)
                )
                if abs(component_sum - total.number) <= Decimal(1):
                    reconciled.append((candidate_rows, total, len(candidate_rows)))
                    break

        if not reconciled:
            return [], None
        reconciled.sort(key=lambda candidate: candidate[2], reverse=True)
        rows, total, _ = reconciled[0]
        return rows, total

    @staticmethod
    def _asset_label_column(headers: list[str], code_column: int) -> int:
        """Find the label column even when a report adds sequence-number columns."""
        label_column = next((
            index for index, header in enumerate(headers)
            if header in {"chi tieu", "khoan muc", "noi dung", "dien giai", "item", "description"}
        ), None)
        if label_column is not None:
            return label_column
        if code_column == 0 and len(headers) > 1:
            return 1
        return max(0, code_column - 1)

    @staticmethod
    def _collapse_subtotal_rows(rows: list[_ComponentValue]) -> list[_ComponentValue]:
        """Collapse contiguous child rows when they reconcile to their subtotal.

        Some bank statements omit item codes and indenting. In those tables the
        reported subtotal itself is the only reliable hierarchy signal. Reduce
        the rows from bottom to top so nested subtotals are collapsed first.
        """
        active: list[_ComponentValue] = []
        for item in reversed(rows):
            value = item.number
            matched_count = 0
            if value is not None:
                subtotal = Decimal(0)
                for index, child in enumerate(active[:12], start=1):
                    if child.number is None:
                        break
                    subtotal += child.number
                    if abs(subtotal - value) <= Decimal(1):
                        matched_count = index
                        break
            if matched_count:
                active = [item, *active[matched_count:]]
            else:
                active.insert(0, item)
        return active

    def _table_component_value(
        self,
        target: ReportSchemaNode,
        label: str,
        value: str,
        chunk: FinancialChunk,
        row_index: int,
        node_suffix: str,
    ) -> _ComponentValue:
        node = ReportSchemaNode(
            node_id=f"{target.node_id}.TABLE.{node_suffix}.{row_index}",
            canonical_name=label,
            statement_type=target.statement_type,
            parent_id=target.node_id,
            children=(),
            kind="line",
            sector=target.sector,
            aliases=(label,),
            parent_contexts=(),
            schema_version=target.schema_version,
        )
        return _ComponentValue(node, chunk, value, self._decimal_value(value))

    def _answer_report_summary(
        self,
        parsed_query: ParsedQuery,
        chunks: Iterable[FinancialChunk],
    ) -> Optional[tuple[str, list[Citation], list[str]]]:
        scoped = self._scope_chunks(parsed_query, chunks)
        if not scoped:
            return None
        document_groups = self._choose_documents(parsed_query, None, scoped)
        if not document_groups:
            return None

        grouped_values: list[tuple[str, int, list[_ComponentValue]]] = []
        for company_code, report_year, report_chunks in document_groups:
            period_year = self._value_year(parsed_query, report_year)
            values: list[_ComponentValue] = []
            used_ids: set[str] = set()
            for statement_type, nodes in self.catalog.nodes_by_statement.items():
                if parsed_query.statement_type and statement_type != parsed_query.statement_type:
                    continue
                for node in nodes:
                    if not self._is_summary_node(node) or node.node_id in used_ids:
                        continue
                    candidates = [
                        chunk for chunk in report_chunks
                        if chunk.metadata.get("schema_node_id") == node.node_id
                    ]
                    node_values = self._values_for_node(node, candidates, period_year)
                    unique_value = self._unique_value(node_values)
                    if unique_value:
                        values.append(unique_value)
                        used_ids.add(node.node_id)
            if values:
                grouped_values.append((company_code, report_year, values))
        if not grouped_values:
            return None

        selected_chunks = list({
            item.chunk.chunk_id: item.chunk
            for _, _, items in grouped_values
            for item in items
        }.values())
        results = [RetrievalResult(chunk=chunk, rrf_score=0.0) for chunk in selected_chunks]
        citations, _ = CitationMapper.map_citations(results)
        source_contexts = [chunk.content for chunk in selected_chunks]
        citation_by_chunk = {citation.chunk_id: citation for citation in citations if citation.chunk_id}
        lines: list[str] = []
        for company_code, report_year, items in grouped_values:
            lines.append(f"Tổng hợp chỉ tiêu của {company_code or 'doanh nghiệp'} — năm {report_year}:")
            lines.extend([
                "| Chỉ tiêu | Giá trị | Nguồn |",
                "| --- | ---: | --- |",
            ])
            for item in items:
                citation = citation_by_chunk.get(item.chunk.chunk_id)
                if citation is None:
                    continue
                unit_suffix = f" {item.chunk.unit}" if item.chunk.unit else ""
                lines.append(
                    f"| {item.node.canonical_name} | {item.value}{unit_suffix} | "
                    f"[{citation.citation_id}] |"
                )
        if not any(
            line.startswith("| ") and not line.startswith(("| Chỉ tiêu |", "| ---"))
            for line in lines
        ):
            return None
        lines.append("Tổng hợp các chỉ tiêu đã được nhận diện trong schema và tìm thấy trong báo cáo.")
        return "\n".join(lines), citations, source_contexts

    def _resolve_component_group(
        self,
        group: dict[str, Any],
        parsed_query: ParsedQuery,
        chunks: list[FinancialChunk],
    ) -> Optional[dict[str, Any]]:
        variants = group.get("variants")
        if not isinstance(variants, dict):
            return group

        requested_ids = set(parsed_query.company_codes or (
            [parsed_query.company_code] if parsed_query.company_code else []
        ))
        variant_scores: list[tuple[int, dict[str, Any]]] = []
        for variant in variants.values():
            if not isinstance(variant, dict):
                continue
            expected_ids = set(str(value) for value in variant.get("component_node_ids", []))
            if variant.get("total_node_id"):
                expected_ids.add(str(variant["total_node_id"]))
            present_ids = {
                str(chunk.metadata.get("schema_node_id", ""))
                for chunk in chunks
                if chunk.chunk_type == ChunkType.FINANCIAL_ITEM
                and str(chunk.metadata.get("schema_node_id", "")) in expected_ids
                and (not requested_ids or (chunk.company_code or "").casefold() in {
                    company.casefold() for company in requested_ids
                })
                and (parsed_query.statement_type is None or chunk.statement_type == parsed_query.statement_type)
                and (parsed_query.report_type is None or chunk.report_type == parsed_query.report_type)
                and (parsed_query.fiscal_quarter is None or chunk.fiscal_quarter == parsed_query.fiscal_quarter)
                and self._chunk_matches_requested_year(parsed_query, chunk)
            }
            variant_scores.append((len(present_ids), variant))
        if not variant_scores:
            return None
        score, selected = max(variant_scores, key=lambda item: item[0])
        if score == 0:
            return None
        return {**group, **selected}

    def _has_company_sector_chunks(
        self,
        parsed_query: ParsedQuery,
        chunks: list[FinancialChunk],
        schema_prefix: str,
    ) -> bool:
        requested_companies = parsed_query.company_codes or (
            [parsed_query.company_code] if parsed_query.company_code else []
        )
        if not requested_companies:
            return False
        requested = {company.casefold() for company in requested_companies}
        group_node_id = (
            "INCOME.BANK.TOTAL_OPERATING_INCOME"
            if schema_prefix == "INCOME.BANK."
            else "INCOME.SECURITIES.TOTAL_OPERATING_REVENUE"
        )
        expected_nodes = set(self.catalog.component_groups.get(group_node_id, {}).get(
            "component_node_ids", []
        ))
        expected_nodes.add(group_node_id)
        matching_nodes = {
            str(chunk.metadata.get("schema_node_id", ""))
            for chunk in chunks
            if (
                chunk.chunk_type == ChunkType.FINANCIAL_ITEM
                and chunk.statement_type == StatementType.INCOME_STATEMENT
                and str(chunk.metadata.get("schema_node_id", "")).startswith(schema_prefix)
                and (chunk.company_code or "").casefold() in requested
                and (parsed_query.report_type is None or chunk.report_type == parsed_query.report_type)
                and (parsed_query.fiscal_quarter is None or chunk.fiscal_quarter == parsed_query.fiscal_quarter)
                and self._chunk_matches_requested_year(parsed_query, chunk)
            )
        }
        sector_revenue_nodes = matching_nodes.intersection(expected_nodes)
        # A single expense or tax row can be misclassified by a shared alias.
        # A sector's total-revenue node is conclusive; otherwise require at
        # least two rows from that sector's defined revenue component group.
        return (
            group_node_id in sector_revenue_nodes
            or len(sector_revenue_nodes) >= 2
        )

    def _scope_chunks(
        self,
        parsed_query: ParsedQuery,
        chunks: Iterable[FinancialChunk],
        allowed_node_ids: Optional[set[str]] = None,
    ) -> list[FinancialChunk]:
        allowed_node_ids = allowed_node_ids or set()
        requested_companies = {
            company.casefold()
            for company in (parsed_query.company_codes or ([parsed_query.company_code] if parsed_query.company_code else []))
        }
        return [
            chunk
            for chunk in chunks
            if chunk.chunk_type == ChunkType.FINANCIAL_ITEM
            and (
                parsed_query.statement_type is None
                or chunk.statement_type == parsed_query.statement_type
                or str(chunk.metadata.get("schema_node_id", "")) in allowed_node_ids
            )
            and (
                chunk.metadata.get("schema_node_id") in self.catalog.nodes
                or str(chunk.metadata.get("schema_node_id", "")) in allowed_node_ids
            )
            and chunk.metadata.get("report_eligible") is not False
            and (not requested_companies or (chunk.company_code or "").casefold() in requested_companies)
            and (parsed_query.fiscal_quarter is None or chunk.fiscal_quarter == parsed_query.fiscal_quarter)
            and (parsed_query.report_type is None or chunk.report_type == parsed_query.report_type)
        ]

    def _choose_documents(
        self,
        parsed_query: ParsedQuery,
        target: Optional[ReportSchemaNode],
        chunks: list[FinancialChunk],
    ) -> list[tuple[str, int, list[FinancialChunk]]]:
        grouped: dict[tuple[str, str], list[FinancialChunk]] = {}
        for chunk in chunks:
            key = (chunk.company_code or "", chunk.document_id or chunk.source_file)
            grouped.setdefault(key, []).append(chunk)

        requested_companies = parsed_query.company_codes or (
            [parsed_query.company_code] if parsed_query.company_code else []
        )
        companies = requested_companies or list(dict.fromkeys(company for company, _ in grouped))
        requested_years = parsed_query.fiscal_years or (
            [parsed_query.fiscal_year] if parsed_query.fiscal_year is not None else []
        )
        target_child_ids = set(
            self.catalog.children.get(target.node_id, target.children) if target else ()
        )
        selected: list[tuple[str, int, list[FinancialChunk]]] = []

        for company in companies:
            company_docs = [
                (key, values)
                for key, values in grouped.items()
                if key[0].casefold() == (company or "").casefold()
            ]
            if not company_docs:
                continue
            available_years = [
                chunk.fiscal_year
                for _, values in company_docs
                for chunk in values
                if chunk.fiscal_year
            ]
            available_years.extend(
                int(year)
                for _, values in company_docs
                for chunk in values
                for year in (chunk.metadata.get("period_values") or {})
                if str(year).isdigit()
            )
            years = requested_years or [max(available_years, default=0)]
            for year in years:
                exact = [
                    (key, values)
                    for key, values in company_docs
                    if any(chunk.fiscal_year == year for chunk in values)
                ]
                candidates = exact or [
                    (key, values)
                    for key, values in company_docs
                    if any(
                        chunk.comparative_fiscal_year == year
                        or str(year) in (chunk.metadata.get("period_values") or {})
                        for chunk in values
                    )
                ]
                if not candidates:
                    continue
                if target is None:
                    # A production report may store each statement in a
                    # separate source document. Keep one preferred report type
                    # per statement, then summarize across all selected parts.
                    docs_by_statement: dict[
                        StatementType | None, list[tuple[tuple[str, str], list[FinancialChunk]]]
                    ] = {}
                    for candidate in candidates:
                        statement_types = {chunk.statement_type for chunk in candidate[1]}
                        for statement_type in statement_types:
                            docs_by_statement.setdefault(statement_type, []).append(candidate)
                    selected_keys: set[tuple[str, str]] = set()
                    for statement_key, statement_docs in docs_by_statement.items():
                        consolidated_docs = [
                            item for item in statement_docs
                            if any(
                                chunk.statement_type == statement_key
                                and chunk.report_type == "consolidated"
                                for chunk in item[1]
                            )
                        ]
                        preferred_docs = consolidated_docs or statement_docs
                        selected_keys.update(item[0] for item in preferred_docs)
                    combined = {
                        chunk.chunk_id: chunk
                        for key, values in candidates
                        if key in selected_keys
                        for chunk in values
                    }
                    selected.append((company, year, list(combined.values())))
                    continue
                if parsed_query.report_type is None and target is not None:
                    consolidated = [
                        item for item in candidates
                        if any(chunk.report_type == "consolidated" for chunk in item[1])
                    ]
                    if consolidated:
                        candidates = consolidated
                candidates.sort(
                    key=lambda item: (
                        sum(
                            1 for chunk in item[1]
                            if (
                                target is not None
                                and (
                                    chunk.metadata.get("schema_node_id") == target.node_id
                                    or chunk.metadata.get("schema_node_id") in target_child_ids
                                )
                            ) or (
                                target is None
                                and self._is_summary_node(
                                    self.catalog.nodes[str(chunk.metadata.get("schema_node_id", ""))]
                                )
                            )
                        ),
                        any(chunk.report_type == "consolidated" for chunk in item[1]),
                        item[0][1],
                    ),
                    reverse=True,
                )
                _, best_chunks = candidates[0]
                selected.append((company, year, best_chunks))

        if not requested_companies and not requested_years:
            return selected[:5]
        return selected

    def _collect_components(
        self,
        target: ReportSchemaNode,
        chunks: list[FinancialChunk],
        value_year: int,
        component_node_ids: Optional[list[str]] = None,
    ) -> list[_ComponentValue]:
        by_node: dict[str, list[FinancialChunk]] = {}
        for chunk in chunks:
            node_id = str(chunk.metadata.get("schema_node_id", ""))
            by_node.setdefault(node_id, []).append(chunk)

        configured_ids = set(component_node_ids) if component_node_ids is not None else None
        child_ids = [
            child_id for child_id in (
                component_node_ids
                if component_node_ids is not None
                else self.catalog.children.get(target.node_id, target.children)
            )
            if child_id in self.catalog.nodes
            and (
                configured_ids is not None
                or not self._is_total(self.catalog.nodes[child_id])
            )
        ]
        selected: list[_ComponentValue] = []

        def collect_branch(node_id: str, visiting: set[str]) -> list[_ComponentValue]:
            node = self.catalog.nodes.get(node_id)
            if node is None or node_id in visiting:
                return []
            if (
                self._is_total(node)
                and (configured_ids is None or node_id not in configured_ids)
            ):
                return []

            direct = self._unique_value(
                self._values_for_node(node, by_node.get(node_id, []), value_year)
            )
            if direct:
                # Prefer a reported amount at this node; descending further
                # would count its note-level breakdown a second time.
                return [direct]

            next_visiting = {*visiting, node_id}
            descendants: list[_ComponentValue] = []
            for child_id in self.catalog.children.get(node_id, node.children):
                descendants.extend(collect_branch(child_id, next_visiting))
            return descendants

        for child_id in child_ids:
            selected.extend(collect_branch(child_id, set()))
        return selected

    def _collect_unmapped_components(
        self,
        target: ReportSchemaNode,
        chunks: list[FinancialChunk],
        value_year: int,
        component_node_ids: Optional[list[str]] = None,
    ) -> list[_ComponentValue]:
        """Recover unannotated source rows by matching only within this subtree."""
        roots = (
            list(component_node_ids)
            if component_node_ids is not None
            else [
                node_id
                for node_id in self.catalog.children.get(target.node_id, target.children)
                if node_id in self.catalog.nodes
                and not self._is_total(self.catalog.nodes[node_id])
            ]
        )
        allowed: set[str] = set()
        pending = list(roots)
        root_ids = set(roots)
        while pending:
            node_id = pending.pop()
            node = self.catalog.nodes.get(node_id)
            if node is None or node_id in allowed:
                continue
            if self._is_total(node) and node_id not in root_ids:
                continue
            allowed.add(node_id)
            pending.extend(self.catalog.children.get(node_id, node.children))

        chunks_by_node: dict[str, list[FinancialChunk]] = {}
        for chunk in chunks:
            if chunk.chunk_type != ChunkType.FINANCIAL_ITEM:
                continue
            source_node_value = chunk.metadata.get("schema_node_id")
            source_node_id = str(source_node_value) if source_node_value else ""
            if source_node_id in allowed or source_node_id in self.catalog.nodes:
                # A known schema assignment is authoritative. A row assigned
                # elsewhere must not be pulled into this group by a stale
                # section heading or a similar-looking alias.
                continue
            raw_label = source_item_label(chunk)
            normalized_label = self.catalog.normalize_label(raw_label)
            target_label_score = max(
                (
                    SequenceMatcher(
                        None,
                        normalized_label,
                        self.catalog.normalize_label(alias),
                    ).ratio()
                    for alias in (target.canonical_name, *target.aliases)
                    if len(self.catalog.normalize_label(alias)) >= 10
                ),
                default=0.0,
            )
            if target_label_score >= 0.88:
                # Keep an OCR-damaged parent aggregate available to
                # _find_total; do not misreport it as one of its own children.
                continue
            node = self._resolve_component_table_node(
                raw_label, target.statement_type, allowed
            ) if raw_label else None
            if node is None or not self._period_value(chunk, value_year):
                continue
            chunks_by_node.setdefault(node.node_id, []).append(chunk)

        recovered: list[_ComponentValue] = []
        for node_id, node_chunks in chunks_by_node.items():
            node = self.catalog.nodes[node_id]
            value = self._unique_value(self._values_for_node(node, node_chunks, value_year))
            if value:
                recovered.append(value)
        return recovered

    def _find_total(
        self,
        target: ReportSchemaNode,
        chunks: list[FinancialChunk],
        value_year: int,
        total_node_ids: Optional[list[str]] = None,
    ) -> Optional[_ComponentValue]:
        total_ids = [
            child_id for child_id in (
                total_node_ids
                if total_node_ids is not None
                else self.catalog.children.get(target.node_id, target.children)
            )
            if child_id in self.catalog.nodes and self._is_total(self.catalog.nodes[child_id])
        ]
        if not total_ids:
            total_ids = [target.node_id]
        for node_id in total_ids:
            node = self.catalog.nodes.get(node_id)
            if node is None:
                continue
            values = self._values_for_node(
                node,
                [chunk for chunk in chunks if chunk.metadata.get("schema_node_id") == node_id],
                value_year,
            )
            unique_value = self._unique_value(values)
            if unique_value:
                return unique_value

        # If an extracted parent-total row missed schema annotation because of
        # OCR, resolve it against the target label. Never reinterpret a known
        # row from another schema branch as the total.
        target_aliases = (target.canonical_name, *target.aliases)
        fuzzy_totals: list[tuple[float, _ComponentValue]] = []
        for chunk in chunks:
            if chunk.chunk_type != ChunkType.FINANCIAL_ITEM:
                continue
            node_value = chunk.metadata.get("schema_node_id")
            node_id = str(node_value) if node_value else ""
            if node_id and node_id != target.node_id:
                continue
            raw_label = source_item_label(chunk)
            normalized_label = self.catalog.normalize_label(raw_label)
            if len(normalized_label) < 10:
                continue
            score = max(
                (
                    SequenceMatcher(
                        None,
                        normalized_label,
                        self.catalog.normalize_label(alias),
                    ).ratio()
                    for alias in target_aliases
                    if len(self.catalog.normalize_label(alias)) >= 10
                ),
                default=0.0,
            )
            if score < 0.88:
                continue
            value = self._period_value(chunk, value_year)
            if value:
                fuzzy_totals.append((score, _ComponentValue(
                    target, chunk, value, self._decimal_value(value)
                )))
        if fuzzy_totals:
            best_score = max(score for score, _ in fuzzy_totals)
            best = [item for score, item in fuzzy_totals if score == best_score]
            if best_score - max(
                (score for score, _ in fuzzy_totals if score < best_score),
                default=0.0,
            ) >= 0.06:
                unique_best = self._unique_value([item for item in best])
                if unique_best:
                    return unique_best
        return None

    def _values_for_node(
        self,
        node: ReportSchemaNode,
        chunks: list[FinancialChunk],
        value_year: int,
    ) -> list[_ComponentValue]:
        values: list[_ComponentValue] = []
        seen: set[tuple[str, str]] = set()
        for chunk in chunks:
            value = self._period_value(chunk, value_year)
            if not value:
                continue
            key = (chunk.document_id or chunk.source_file, value)
            if key in seen:
                continue
            seen.add(key)
            values.append(_ComponentValue(node, chunk, value, self._decimal_value(value)))
        return values

    @staticmethod
    def _unique_value(values: list[_ComponentValue]) -> Optional[_ComponentValue]:
        if not values:
            return None
        distinct = {
            item.number if item.number is not None else item.value
            for item in values
        }
        return values[0] if len(distinct) == 1 else None

    @staticmethod
    def _period_value(chunk: FinancialChunk, year: int) -> Optional[str]:
        period_values = chunk.metadata.get("period_values") or {}
        value = period_values.get(str(year))
        if value:
            return str(value).strip()
        if chunk.fiscal_year == year:
            value = chunk.metadata.get("end_value")
        elif chunk.comparative_fiscal_year == year:
            value = chunk.metadata.get("start_value")
        else:
            value = None
        return str(value).strip() if value else None

    @staticmethod
    def _value_year(parsed_query: ParsedQuery, report_year: int) -> int:
        if parsed_query.period_type == PeriodType.BEGINNING_OF_YEAR:
            requested_year = parsed_query.fiscal_year or report_year
            return requested_year - 1
        return parsed_query.fiscal_year or report_year

    @staticmethod
    def _period_label(parsed_query: ParsedQuery, report_year: int) -> str:
        year = parsed_query.fiscal_year or report_year
        if parsed_query.period_type == PeriodType.BEGINNING_OF_YEAR:
            return f"đầu năm {year}"
        if parsed_query.period_type == PeriodType.END_OF_YEAR:
            return f"cuối năm {year}"
        return f"năm {year}"

    @staticmethod
    def _is_total(node: ReportSchemaNode) -> bool:
        node_id = node.node_id.upper()
        return (
            node.kind == "total"
            or node_id.startswith("TOTAL_")
            or node_id.endswith(".TOTAL")
            or ".TOTAL_" in node_id
        )

    @staticmethod
    def _is_summary_node(node: ReportSchemaNode) -> bool:
        node_id = node.node_id.upper()
        canonical = node.canonical_name.casefold()
        if node.statement_type.value == "balance_sheet":
            return node_id in {
                "TOTAL_ASSETS",
                "TOTAL_LIABILITIES",
                "TOTAL_EQUITY",
                "TOTAL_LIABILITIES_AND_EQUITY",
            }
        if node.statement_type.value == "income_statement":
            return any(token in node_id for token in (
                "REVENUE_NET",
                "NET_INTEREST_INCOME",
                "TOTAL_OPERATING_INCOME",
                "OPERATING_INCOME",
                "PROFIT_BEFORE_TAX",
                "PROFIT_AFTER_TAX",
                "NET_PROFIT",
            )) or any(token in canonical for token in ("doanh thu thuần", "thu nhập lãi thuần", "lợi nhuận trước thuế", "lợi nhuận sau thuế"))
        if node.statement_type.value == "cash_flow":
            return any(token in node_id for token in (
                "NET_OPERATING_CASH",
                "NET_CASH",
                "NET_CHANGE_IN_CASH",
                "CASH_ENDING",
            ))
        return False

    @staticmethod
    def _decimal_value(value: str) -> Optional[Decimal]:
        normalized = NumericValidator.normalize_number(value)
        if not normalized:
            return None
        try:
            return Decimal(normalized)
        except InvalidOperation:
            return None

    @classmethod
    def _first_table_number(cls, value: str) -> Optional[str]:
        """Read a table amount, including OCR cells that concatenate adjacent values."""
        clean_value = value.strip()
        if cls._decimal_value(clean_value) is not None:
            return clean_value
        # Some financial-statement OCR merges adjacent numeric cells without
        # spaces. A valid thousands-grouped amount ends where the next amount
        # begins without a separating dot, so the first complete token is the
        # reported aggregate for that row.
        match = re.search(
            r"(?:\(\s*)?[+-]?\s*\d{1,3}(?:\.\d{3}){2,}(?:,\d+)?\s*\)?",
            clean_value,
        )
        if match and cls._decimal_value(match.group(0)) is not None:
            return match.group(0).strip()
        return None
