import os
from pathlib import Path
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import re
import requests
from contextvars import ContextVar
from typing import Dict, List, Optional, Tuple

from ..schemas import (
    ChunkType,
    Citation,
    FinancialChunk,
    ParsedQuery,
    PeriodType,
    QueryIntent,
    RetrievalResult,
)
from ..analysis.contribution import CalculationEvidence, ContributionAnswer, build_contribution_answer
from ..analysis.evidence import calculation, source_input
from ..analysis.ratio import RatioAnswer, build_ratio_answer
from ..context.assembler import ContextAssembler
from ..context.budget import TokenBudget, estimate_tokens
from ..query.directives import normalize_directive_text
from ..validation.numeric import NumericValidator
from ..structure.source_rows import SourceCompositionRows
from ..structure.source_groups import (
    is_total_source_label,
    source_label_matches_query,
    source_item_label,
    source_phrase_tokens,
)


class FinancialGenerationError(RuntimeError):
    """Raised when the configured answer model cannot produce a response."""


generation_budget_trace: ContextVar[dict | None] = ContextVar("generation_budget_trace", default=None)

class FinancialChatEngine(SourceCompositionRows):
    """
    Sinh câu trả lời tài chính có trích dẫn từ các đoạn bằng chứng đã truy xuất.
    Vector/BM25 chọn nội dung nguồn; schema và metadata chỉ hỗ trợ diễn giải,
    không tự chọn giá trị thay cho kết quả retrieval.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model_name: str = "qwen3.5:4b",
        prompt_path: Optional[Path] = None
    ):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.timeout_seconds = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "90"))
        self.num_predict = max(32, int(os.getenv("OLLAMA_NUM_PREDICT", "256")))
        self.num_ctx = max(1024, int(os.getenv("OLLAMA_NUM_CTX", "4096")))
        self.context_assembler = ContextAssembler()

        if prompt_path is None:
            prompt_path = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "prompts" / "answer.md"

        self.prompt_template = ""
        if prompt_path.exists():
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.prompt_template = f.read()

    @staticmethod
    def _scope_prompt_instruction(parsed_query: ParsedQuery) -> str:
        """Tell the model which source metadata must match the user's scope."""
        fields: list[str] = []
        companies = parsed_query.company_codes or (
            [parsed_query.company_code] if parsed_query.company_code else []
        )
        years = parsed_query.fiscal_years or (
            [parsed_query.fiscal_year] if parsed_query.fiscal_year is not None else []
        )
        if companies:
            fields.append(f"company={','.join(companies)}")
        if years:
            fields.append(f"fiscal_year={','.join(str(year) for year in years)}")
        if parsed_query.fiscal_quarter is not None:
            fields.append(f"fiscal_quarter=Q{parsed_query.fiscal_quarter}")
        if parsed_query.report_type:
            fields.append(f"report_basis={parsed_query.report_type}")
        scope = "; ".join(fields) if fields else "none inferred; use only scope named in the question"
        return (
            "[QUY TẮC KHỚP PHẠM VI NGUỒN]\n"
            f"Phạm vi câu hỏi đã phân tích: {scope}.\n"
            "Chỉ các trường được nêu rõ trong phạm vi hoặc câu hỏi mới là điều kiện bắt buộc; không "
            "đòi hỏi trường câu hỏi không nêu và không từ chối chỉ vì trường đó vắng mặt. Bỏ qua fact "
            "xung đột với mã doanh nghiệp, kỳ hoặc cơ sở được hỏi. Kiểm tra toàn bộ nguồn; nếu có fact "
            "khớp thì dùng và trích dẫn fact đó. Chỉ nói thiếu dữ liệu khi không có bằng chứng khớp. "
            "Nhắc lại mã doanh nghiệp và kỳ nếu câu hỏi nêu chúng.\n"
        )

    @staticmethod
    def _source_item_label(chunk: FinancialChunk) -> str:
        return source_item_label(chunk)

    @staticmethod
    def _source_period_value(chunk: FinancialChunk, year: int) -> Optional[str]:
        from ..structure.semantics import period_supported
        if period_supported(chunk, year) is False:
            return None
        period_values = chunk.metadata.get("period_values")
        if isinstance(period_values, dict):
            value = period_values.get(str(year), period_values.get(year))
            if value not in (None, "", "N/A"):
                return str(value).strip()
        if chunk.fiscal_year == year:
            value = chunk.metadata.get("end_value")
        elif chunk.comparative_fiscal_year == year:
            value = chunk.metadata.get("start_value")
        else:
            value = None
        return str(value).strip() if value not in (None, "", "N/A") else None

    @staticmethod
    def _answer_source_subset(
        answer: str,
        citations: List[Citation],
        results: List[RetrievalResult],
        fallback_contexts: List[str],
    ) -> tuple[List[Citation], List[str]]:
        """Keep exact-answer provenance limited to citations actually used."""
        cited_ids = {
            citation_id.casefold()
            for citation_id in re.findall(r"\[(c\d+)\]", answer, flags=re.IGNORECASE)
        }
        selected = [
            citation for citation in citations
            if citation.citation_id.casefold() in cited_ids
        ]
        if not selected:
            return citations, fallback_contexts
        selected_chunk_ids = {citation.chunk_id for citation in selected if citation.chunk_id}
        contexts = [
            result.chunk.content for result in results
            if result.chunk.chunk_id in selected_chunk_ids
        ]
        return selected, contexts or fallback_contexts

    @staticmethod
    def _growth_year_pair(
        parsed_query: ParsedQuery,
        fallback_current_year: Optional[int],
        fallback_prior_year: Optional[int],
    ) -> tuple[Optional[int], Optional[int]]:
        """Resolve the direction of a two-period change from the user's wording."""
        years = list(dict.fromkeys(int(year) for year in parsed_query.fiscal_years))
        normalized_query = normalize_directive_text(parsed_query.original_query)
        if len(years) >= 2:
            from_to = re.search(
                r"\btu(?:\s+nam)?\s*(20\d{2})\s+(?:den|toi|sang|len)"
                r"(?:\s+nam)?\s*(20\d{2})\b",
                normalized_query,
            )
            if from_to:
                return int(from_to.group(2)), int(from_to.group(1))

            compared = re.search(
                r"\b(20\d{2})\b.{0,80}\bso\s+(?:sanh\s+)?voi\b"
                r".{0,30}\b(20\d{2})\b",
                normalized_query,
            )
            if compared:
                return int(compared.group(1)), int(compared.group(2))

            # If the wording does not define a direction, use the newer period
            # as current and the older period as the baseline.
            return max(years), min(years)

        current_year = fallback_current_year
        if current_year is None:
            current_year = years[-1] if years else None
        prior_year = fallback_prior_year
        if prior_year is None and current_year is not None:
            prior_year = current_year - 1
        return current_year, prior_year

    @classmethod
    def _source_exact_metric_answer(
        cls,
        parsed_query: ParsedQuery,
        results: List[RetrievalResult],
        citations: List[Citation],
    ) -> Optional[str]:
        """Answer one exact source-row lookup without regenerating its value or citation."""
        if (
            parsed_query.intent != QueryIntent.METRIC_LOOKUP
            or parsed_query.structured_operation is not None
        ):
            return None

        years = list(parsed_query.fiscal_years)
        if not years and parsed_query.fiscal_year is not None:
            years = [parsed_query.fiscal_year]
        if not years:
            years = list(dict.fromkeys(
                result.chunk.fiscal_year
                for result in results
                if result.chunk.fiscal_year is not None
            ))
        if len(years) != 1:
            return None
        year = years[0]
        company_codes = {
            str(code).casefold()
            for code in (
                parsed_query.company_codes
                or ([parsed_query.company_code] if parsed_query.company_code is not None else [])
            )
            if code
        }
        citation_by_chunk = {
            citation.chunk_id: citation
            for citation in citations
            if citation.chunk_id
        }

        matches: list[tuple[FinancialChunk, Citation, str, str]] = []
        seen_ids: set[str] = set()
        for result in results:
            chunk = result.chunk
            if chunk.chunk_type != ChunkType.FINANCIAL_ITEM or chunk.chunk_id in seen_ids:
                continue
            seen_ids.add(chunk.chunk_id)
            if company_codes and (chunk.company_code or "").casefold() not in company_codes:
                continue
            label = cls._source_item_label(chunk)
            if not source_label_matches_query(
                parsed_query.original_query,
                label,
                parsed_query.company_codes or ([parsed_query.company_code] if parsed_query.company_code else []),
            ):
                continue
            # "Đầu năm 2025" is the opening balance carried from the 2024
            # comparative column. The source row keeps that value under the
            # preceding fiscal year, even though the user names 2025 as the
            # requested period.
            evidence_year = (
                year - 1
                if parsed_query.period_type == PeriodType.BEGINNING_OF_YEAR
                else year
            )
            value = cls._source_period_value(chunk, evidence_year)
            citation = citation_by_chunk.get(chunk.chunk_id)
            if value is None or citation is None:
                continue
            unit = str(chunk.unit or chunk.metadata.get("unit") or citation.unit or "").strip()
            matches.append((chunk, citation, label, f"{value} {unit}".strip()))

        # The same source label is often repeated in notes (for example,
        # liquidity or segment tables). When the query identifies a statement,
        # prefer the matching statement's row before treating the result as
        # ambiguous. This is a source-context preference; it does not use a
        # schema alias to choose the amount.
        statement_matches = [
            match for match in matches
            if match[0].statement_type == parsed_query.statement_type
        ] if parsed_query.statement_type is not None else []
        if statement_matches:
            matches = statement_matches
        if len(matches) != 1:
            return None
        chunk, citation, label, amount = matches[0]
        company = parsed_query.company_code or chunk.company_code or "doanh nghiệp được hỏi"
        return f"{label} của {company} năm {year} là {amount} [{citation.citation_id}]."



    @classmethod
    def _source_composition_answer(
        cls,
        parsed_query: ParsedQuery,
        results: List[RetrievalResult],
        citations: List[Citation],
    ) -> Optional[str]:
        """Format vector-selected rows directly, without schema-based row selection."""
        if parsed_query.structured_operation not in {"composition", "components"}:
            return None
        years = list(parsed_query.fiscal_years)
        if not years and parsed_query.fiscal_year is not None:
            years = [parsed_query.fiscal_year]
        if not years:
            years = list(dict.fromkeys(
                result.chunk.fiscal_year
                for result in results
                if result.chunk.fiscal_year is not None
            ))
        years = list(dict.fromkeys(years))
        if not years:
            return None

        citation_by_chunk = {
            citation.chunk_id: citation
            for citation in citations
            if citation.chunk_id
        }
        rows: list[dict[str, object]] = []
        seen_chunk_ids: set[str] = set()
        for result in results:
            chunk = result.chunk
            if chunk.chunk_type != ChunkType.FINANCIAL_ITEM or chunk.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk.chunk_id)
            citation = citation_by_chunk.get(chunk.chunk_id)
            label = cls._source_item_label(chunk)
            if citation is None or not label:
                continue
            values = {year: cls._source_period_value(chunk, year) for year in years}
            if not any(values.values()):
                continue
            rows.append({
                "chunk": chunk,
                "label": label,
                "values": values,
                "citation": citation,
                "unit": str(chunk.unit or chunk.metadata.get("unit") or citation.unit or "").strip(),
            })
        if not rows:
            return None

        company_codes = parsed_query.company_codes or (
            [parsed_query.company_code] if parsed_query.company_code else []
        )
        query_tokens = set(source_phrase_tokens(parsed_query.original_query, company_codes))
        total_rows = [
            row for row in rows
            if is_total_source_label(str(row["label"]))
        ]
        total_row = max(
            total_rows,
            key=lambda row: len(
                query_tokens & set(source_phrase_tokens(str(row["label"]), company_codes))
            ),
            default=None,
        )
        denominator_kind = "tổng"
        if total_row is None:
            matching_group_rows = [
                row for row in rows
                if set(source_phrase_tokens(str(row["label"]), company_codes)) == query_tokens
            ]
            if matching_group_rows:
                total_row = matching_group_rows[0]
                denominator_kind = "nhóm"

        if total_row is not None and len(rows) == 1:
            total_values = total_row["values"]
            total_values = total_values if isinstance(total_values, dict) else {}
            value = total_values.get(years[0]) if years else None
            citation_value = total_row["citation"]
            unit = str(total_row["unit"])
            if value is not None and isinstance(citation_value, Citation):
                return (
                    f"Đã tìm thấy dòng {total_row['label']}: "
                    f"{value} {unit} [{citation_value.citation_id}], nhưng nguồn truy xuất "
                    "chưa có các dòng thành phần để lập cơ cấu."
                )

        display_rows = cls._non_overlapping_composition_rows(rows, total_row)
        component_reconciliations: dict[int, Optional[bool]] = {}
        if total_row is not None:
            total_values = total_row.get("values")
            total_chunk = total_row.get("chunk")
            total_unit = NumericValidator._normalize_unit(str(total_row.get("unit") or ""))
            if isinstance(total_values, dict) and isinstance(total_chunk, FinancialChunk):
                for year in years:
                    raw_total = NumericValidator.normalize_number(
                        str(total_values.get(year) or "")
                    )
                    if not raw_total or not total_unit:
                        component_reconciliations[year] = None
                        continue
                    amounts: list[Decimal] = []
                    complete = True
                    for row in display_rows:
                        row_chunk = row.get("chunk")
                        row_values = row.get("values")
                        raw_value = (
                            NumericValidator.normalize_number(str(row_values.get(year) or ""))
                            if isinstance(row_values, dict)
                            else None
                        )
                        if (
                            not isinstance(row_chunk, FinancialChunk)
                            or not isinstance(row_values, dict)
                            or not raw_value
                            or NumericValidator._normalize_unit(str(row.get("unit") or "")) != total_unit
                            or not cls._same_source_table(row_chunk, total_chunk)
                        ):
                            complete = False
                            break
                        try:
                            amounts.append(Decimal(raw_value))
                        except InvalidOperation:
                            complete = False
                            break
                    if not complete or not amounts:
                        component_reconciliations[year] = None
                        continue
                    try:
                        total_amount = Decimal(raw_total)
                    except InvalidOperation:
                        component_reconciliations[year] = None
                        continue
                    component_sum = sum(amounts, Decimal("0"))
                    component_reconciliations[year] = (
                        abs(component_sum - total_amount)
                        <= max(Decimal("1"), abs(total_amount) * Decimal("0.000001"))
                    )
        group_total_line: Optional[str] = None
        if denominator_kind == "nhóm" and total_row is not None and len(rows) > 1:
            # The queried group row is the denominator, not another component.
            # Keep its label, amount, and citation visible as a short total line
            # while listing only its actual child rows in the composition table.
            group_values = total_row["values"]
            group_values = group_values if isinstance(group_values, dict) else {}
            group_amount = group_values.get(years[0]) if years else None
            group_unit = str(total_row["unit"])
            group_citation = total_row["citation"]
            if group_amount is not None and isinstance(group_citation, Citation):
                group_total_line = (
                    f"Tổng nhóm {total_row['label']}: {group_amount} {group_unit} "
                    f"[{group_citation.citation_id}]."
                )
        elif denominator_kind == "tổng" and total_row is not None:
            total_values = total_row["values"]
            total_values = total_values if isinstance(total_values, dict) else {}
            total_amounts = [
                f"{year}: {total_values[year]} {total_row['unit']}"
                for year in years
                if total_values.get(year) is not None
            ]
            total_citation = total_row["citation"]
            if total_amounts and isinstance(total_citation, Citation):
                group_total_line = (
                    f"Tổng cơ sở {total_row['label']}: "
                    + "; ".join(total_amounts)
                    + f" [{total_citation.citation_id}]."
                )

        def share_for(row: dict[str, object], year: int) -> Optional[str]:
            if total_row is None:
                return None
            row_chunk = row["chunk"]
            total_chunk = total_row["chunk"]
            if not isinstance(row_chunk, FinancialChunk) or not isinstance(total_chunk, FinancialChunk):
                return None
            if not cls._same_source_table(row_chunk, total_chunk):
                return None
            row_unit = NumericValidator._normalize_unit(str(row["unit"]))
            total_unit = NumericValidator._normalize_unit(str(total_row["unit"]))
            if not row_unit or row_unit != total_unit:
                return None
            row_values = row["values"]
            total_values = total_row["values"]
            raw_value = row_values.get(year) if isinstance(row_values, dict) else None
            raw_total = total_values.get(year) if isinstance(total_values, dict) else None
            normalized_value = NumericValidator.normalize_number(str(raw_value or ""))
            normalized_total = NumericValidator.normalize_number(str(raw_total or ""))
            if not normalized_value or not normalized_total:
                return None
            total = Decimal(normalized_total)
            if total == 0:
                return None
            percentage = (
                Decimal(normalized_value) * Decimal("100") / total
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            if percentage == 0:
                percentage = Decimal("0.00")
            return format(percentage, ".2f").replace(".", ",") + "%"

        company = parsed_query.company_code or next(
            (result.chunk.company_code for result in results if result.chunk.company_code),
            "doanh nghiệp được hỏi",
        )
        heading_metric = parsed_query.primary_metric or "các khoản mục"
        year_text = ", ".join(str(year) for year in years)
        headers = ["Chỉ tiêu gốc trong BCTC"]
        headers.extend(f"Giá trị {year}" for year in years)
        headers.extend(f"Tỷ trọng / {denominator_kind} {year}" for year in years)
        headers.append("Nguồn")
        lines = [
            f"Cơ cấu {heading_metric} của {company} năm {year_text}:",
        ]
        if group_total_line:
            lines.extend(["", group_total_line])
        lines.extend([
            "",
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ])
        for row in display_rows:
            row_values = row["values"]
            row_values = row_values if isinstance(row_values, dict) else {}
            cells = [str(row["label"])]
            for year in years:
                value = row_values.get(year)
                unit = str(row["unit"])
                cells.append(f"{value} {unit}".strip() if value is not None else "—")
            cells.extend(share_for(row, year) or "—" for year in years)
            citation_value = row["citation"]
            if isinstance(citation_value, Citation):
                cells.append(f"[{citation_value.citation_id}]")
            else:
                cells.append("—")
            lines.append("| " + " | ".join(cells) + " |")

        if total_row is not None:
            if component_reconciliations:
                reconciliation_text = []
                for year, matched in component_reconciliations.items():
                    if matched is True:
                        reconciliation_text.append(
                            f"năm {year}, tổng các dòng thành phần khớp với tổng cơ sở"
                        )
                    elif matched is False:
                        reconciliation_text.append(
                            f"năm {year}, tổng các dòng thành phần chưa khớp với tổng cơ sở"
                        )
                if reconciliation_text:
                    lines.extend(["", "Đối chiếu: " + "; ".join(reconciliation_text) + "."])
                elif any(value is None for value in component_reconciliations.values()):
                    lines.extend([
                        "",
                        "Chưa đối chiếu được toàn bộ các dòng thành phần với tổng cơ sở do thiếu giá trị cùng kỳ, đơn vị hoặc nguồn.",
                    ])
            lines.extend([
                "",
                f"Tỷ trọng được tính so với {denominator_kind} cùng kỳ và cùng đơn vị. "
                "Dòng tổng hợp chỉ được lược khi các dòng con đối chiếu được với số nguồn; "
                "dòng chưa phân loại hoặc chưa đối chiếu được vẫn cần được xem xét riêng.",
            ])
        return "\n".join(lines)


    @classmethod
    def source_composition_calculations(
        cls,
        parsed_query: ParsedQuery,
        results: List[RetrievalResult],
        citations: List[Citation],
    ) -> List[CalculationEvidence]:
        """Return citation lineage for percentages shown in source-composition answers."""
        if parsed_query.structured_operation not in {"composition", "components"}:
            return []
        years = list(parsed_query.fiscal_years)
        if not years and parsed_query.fiscal_year is not None:
            years = [parsed_query.fiscal_year]
        if not years:
            years = list(dict.fromkeys(
                result.chunk.fiscal_year
                for result in results
                if result.chunk.fiscal_year is not None
            ))
        years = list(dict.fromkeys(years))
        if not years:
            return []

        citation_by_chunk = {
            citation.chunk_id: citation
            for citation in citations
            if citation.chunk_id
        }
        rows: list[dict[str, object]] = []
        seen_chunk_ids: set[str] = set()
        for result in results:
            chunk = result.chunk
            if chunk.chunk_type != ChunkType.FINANCIAL_ITEM or chunk.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk.chunk_id)
            citation = citation_by_chunk.get(chunk.chunk_id)
            label = cls._source_item_label(chunk)
            if citation is None or not label:
                continue
            values = {year: cls._source_period_value(chunk, year) for year in years}
            if not any(values.values()):
                continue
            rows.append({
                "chunk": chunk,
                "label": label,
                "values": values,
                "citation": citation,
                "unit": str(chunk.unit or chunk.metadata.get("unit") or citation.unit or "").strip(),
            })
        if len(rows) < 2:
            return []

        company_codes = parsed_query.company_codes or (
            [parsed_query.company_code] if parsed_query.company_code else []
        )
        query_tokens = set(source_phrase_tokens(parsed_query.original_query, company_codes))
        total_rows = [row for row in rows if is_total_source_label(str(row["label"]))]
        total_row = max(
            total_rows,
            key=lambda row: len(
                query_tokens & set(source_phrase_tokens(str(row["label"]), company_codes))
            ),
            default=None,
        )
        if total_row is None:
            group_rows = [
                row for row in rows
                if set(source_phrase_tokens(str(row["label"]), company_codes)) == query_tokens
            ]
            if group_rows:
                total_row = group_rows[0]
        if total_row is None:
            return []

        display_rows = cls._non_overlapping_composition_rows(rows, total_row)
        total_chunk = total_row["chunk"]
        total_values = total_row["values"]
        total_citation = total_row["citation"]
        if not isinstance(total_chunk, FinancialChunk) or not isinstance(total_values, dict) or not isinstance(total_citation, Citation):
            return []

        calculations: list[CalculationEvidence] = []
        for row in display_rows:
            row_chunk = row["chunk"]
            row_values = row["values"]
            row_citation = row["citation"]
            if (
                not isinstance(row_chunk, FinancialChunk)
                or not isinstance(row_values, dict)
                or not isinstance(row_citation, Citation)
                or not cls._same_source_table(row_chunk, total_chunk)
                or NumericValidator._normalize_unit(str(row["unit"]))
                != NumericValidator._normalize_unit(str(total_row["unit"]))
            ):
                continue
            for year in years:
                numerator_text = NumericValidator.normalize_number(str(row_values.get(year) or ""))
                denominator_text = NumericValidator.normalize_number(str(total_values.get(year) or ""))
                if not numerator_text or not denominator_text or Decimal(denominator_text) == 0:
                    continue
                percentage = (
                    Decimal(numerator_text) * Decimal("100") / Decimal(denominator_text)
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                calculations.append(calculation(percentage, "%", "share_of_total", (
                    source_input("numerator", row_chunk, row_citation, year, Decimal(numerator_text)),
                    source_input("denominator", total_chunk, total_citation, year, Decimal(denominator_text)),
                )))
        return calculations

    @staticmethod
    def _composition_validation_contexts(
        results: List[RetrievalResult],
        citations: List[Citation],
    ) -> List[str]:
        """Give the numeric validator the schema-backed parent of each component row."""
        cited_ids = {citation.chunk_id for citation in citations if citation.chunk_id}
        chunks = [
            result.chunk
            for result in results
            if result.chunk.chunk_id in cited_ids
            and result.chunk.chunk_type == ChunkType.FINANCIAL_ITEM
        ]
        labels_by_node = {
            str(chunk.metadata.get("schema_node_id") or ""): source_item_label(chunk)
            for chunk in chunks
            if chunk.metadata.get("schema_node_id") and source_item_label(chunk)
        }
        parent_labels: dict[str, str] = {}
        child_nodes_by_parent: dict[str, set[str]] = {}
        for chunk in chunks:
            node_id = str(chunk.metadata.get("schema_node_id") or "")
            label = source_item_label(chunk)
            if not node_id or not label:
                continue
            for child_id in chunk.metadata.get("schema_children", []):
                child_id = str(child_id)
                if child_id in labels_by_node:
                    parent_labels.setdefault(child_id, label)
                    child_nodes_by_parent.setdefault(node_id, set()).add(child_id)
            for candidate in chunks:
                if str(candidate.metadata.get("schema_parent_id") or "") == node_id:
                    child_id = str(candidate.metadata.get("schema_node_id") or "")
                    if child_id:
                        parent_labels.setdefault(child_id, label)
                        child_nodes_by_parent.setdefault(node_id, set()).add(child_id)

        contexts: list[str] = []
        for chunk in chunks:
            label = source_item_label(chunk)
            node_id = str(chunk.metadata.get("schema_node_id") or "")
            group_label = parent_labels.get(node_id, "")
            if not group_label and child_nodes_by_parent.get(node_id):
                group_label = label
            context = chunk.content
            if group_label:
                replacement = f"Phân nhóm: {group_label}"
                context, count = re.subn(
                    r"(?im)^\s*phân\s+nhóm\s*:[^\n]*",
                    replacement,
                    context,
                    count=1,
                )
                if count == 0:
                    context = f"{context.rstrip()}\n{replacement}"
            contexts.append(context)
        return contexts

    def generate_answer(
        self,
        user_query: str,
        parsed_query: ParsedQuery,
        results: List[RetrievalResult],
        chunks_by_id: Optional[Dict[str, FinancialChunk]] = None,
    ) -> Tuple[str, List[Citation], List[str]]:
        if not results:
            return "Không tìm thấy dữ liệu liên quan trong tài liệu báo cáo tài chính.", [], []

        # 1. Lắp ráp Context và trích xuất Citations
        structured_operation = parsed_query.structured_operation in {
            "table", "components", "composition", "ranking", "summary",
            "contribution_analysis", "ratio_analysis",
        }
        context_str, citations, source_contexts = self.context_assembler.assemble_with_sources(
            results,
            chunks_by_id,
            max_chars=18_000 if structured_operation else None,
        )
        if not context_str or not citations:
            return "Không có đủ ngữ cảnh nguồn để trả lời câu hỏi.", [], []

        source_composition_answer = self._source_composition_answer(
            parsed_query, results, citations
        )
        if source_composition_answer:
            cited_ids = {
                citation_id.casefold()
                for citation_id in re.findall(
                    r"\[(c\d+)\]", source_composition_answer, flags=re.IGNORECASE
                )
            }
            used_citations = [
                citation for citation in citations
                if citation.citation_id.casefold() in cited_ids
            ]
            return (
                source_composition_answer,
                used_citations,
                self._composition_validation_contexts(results, used_citations),
            )

        source_exact_metric_answer = self._source_exact_metric_answer(
            parsed_query, results, citations
        )
        if source_exact_metric_answer:
            exact_citations, exact_contexts = self._answer_source_subset(
                source_exact_metric_answer, citations, results, source_contexts
            )
            return source_exact_metric_answer, exact_citations, exact_contexts

        # Comparisons and period changes are arithmetic over source values,
        # so resolve them before calling the language model. This keeps the
        # result grounded in the selected rows and makes those queries work
        # when the optional Ollama service is unavailable. The formatter only
        # returns values present in the cited source chunks; unavailable years
        # become an explicit abstention.
        structured_answer = self._structured_financial_answer(
            parsed_query, results, citations
        )
        if structured_answer:
            exact_citations, exact_contexts = self._answer_source_subset(
                structured_answer, citations, results, source_contexts
            )
            return structured_answer, exact_citations, exact_contexts

        # Do not select a numeric row by the parser's canonical metric or
        # schema_node_id. Those are useful ingestion hints, but may be stale or
        # ambiguous; the vector-ranked source context must decide what answers
        # the user's actual wording.
        if structured_operation:
            prompt = (
                "Bạn là trợ lý phân tích báo cáo tài chính. Chỉ dùng nội dung nguồn bên dưới. "
                "Hãy đối chiếu nhãn dòng gốc với câu hỏi trước khi lấy số; không lấy dòng chỉ vì "
                "nó có alias hoặc nhãn chuẩn hóa gần giống. Với câu hỏi cơ cấu/thành phần, trả lời "
                "bằng bảng và liệt kê tất cả các dòng tài chính trong nhóm được hỏi có trong nguồn, "
                "kể cả dòng tổng và các dòng chi tiết; không dừng sau vài dòng. Giữ nguyên nhãn gốc, "
                "số liệu, đơn vị và citation cho từng dòng. Không trộn số từ các trang/bảng khác nhau. "
                "Chỉ tính tỷ trọng khi có tổng cùng bảng, cùng kỳ và cùng đơn vị. "
                "Nếu nguồn không đủ hoặc không xác định được dòng phù hợp, hãy nói rõ điều đó, không suy đoán.\n\n"
                f"[NGỮ CẢNH TÀI LIỆU]:\n{context_str}\n\n"
                f"[CÂU HỎI]: {user_query}\n\n[CÂU TRẢ LỜI]:"
            )
        elif self.prompt_template:
            prompt = self.prompt_template.format(context=context_str, query=user_query)
            prompt = (
                "Hãy khớp nhãn dòng gốc trong nguồn với chỉ tiêu người dùng hỏi. "
                "Không trả lời bằng một dòng khác chỉ vì có alias hoặc từ ngữ gần giống. "
                "Nếu không có dòng tương ứng, hãy nói chưa tìm thấy thay vì thay bằng chỉ tiêu khác.\n\n"
                + prompt
            )
        else:
            prompt = (
                f"Bạn là chuyên gia tài chính. Hãy đối chiếu nhãn dòng gốc với câu hỏi, không dùng alias gần giống thay thế. "
                f"Dựa DUY NHẤT vào dữ liệu sau để trả lời con số chính xác kèm trích dẫn.\n\n"
                f"[NGỮ CẢNH TÀI LIỆU]:\n{context_str}\n\n"
                f"[CÂU HỎI]: {user_query}\n\n"
                f"[CÂU TRẢ LỜI]:"
            )

        prompt = self._scope_prompt_instruction(parsed_query) + "\n" + prompt

        # 3. Gọi API chat, tắt thinking và giới hạn độ dài câu trả lời.
        output_limit = max(self.num_predict, 512) if structured_operation else self.num_predict
        budget = TokenBudget(context_window=self.num_ctx, output_reserve=output_limit,
                             instruction_tokens=estimate_tokens(prompt) - estimate_tokens(context_str))
        bounded_context, bounded_citations, bounded_sources = self.context_assembler.assemble_with_sources(
            results, chunks_by_id, max_chars=budget.available, max_tokens=budget.available)
        generation_budget_trace.set({"context_window": self.num_ctx, "output_reserve": output_limit,
            "budget_tokens": budget.available, "estimated_tokens": estimate_tokens(bounded_context),
            "estimator": "utf8_bytes_upper_estimate", "model_called": bool(bounded_context),
            "omitted_chunk_ids": [item.chunk.chunk_id for item in results
                                  if item.chunk.chunk_id not in {c.chunk_id for c in bounded_citations}]})
        if not bounded_context:
            return "Không có đủ ngân sách ngữ cảnh để giữ nguyên bằng chứng; hãy thu hẹp câu hỏi.", [], []
        prompt = prompt.replace(context_str, bounded_context, 1)
        citations, source_contexts = bounded_citations, bounded_sources
        request_timeout = max(self.timeout_seconds, 180) if structured_operation else self.timeout_seconds
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "think": False,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": output_limit,
                        "num_ctx": self.num_ctx,
                    }
                },
                timeout=request_timeout
            )
            if resp.status_code == 200:
                answer = resp.json().get("message", {}).get("content", "").strip()
                if not answer:
                    raise FinancialGenerationError("The answer model returned empty content")
                return answer, citations, source_contexts
            resp.raise_for_status()
            raise FinancialGenerationError("The answer model returned an unexpected response")
        except Exception as e:
            raise FinancialGenerationError(f"Could not generate a grounded answer: {e}") from e

    def generate_contribution_answer(
        self,
        parsed_query: ParsedQuery,
        results: List[RetrievalResult],
        chunks_by_id: Optional[Dict[str, FinancialChunk]] = None,
    ) -> ContributionAnswer:
        """Calculate first-level contributions only from cited, reconciled source rows."""
        if not results:
            return ContributionAnswer(
                answer="Không đủ bằng chứng nguồn để tính mức đóng góp.",
                citations=[],
                source_contexts=[],
                calculations=[],
            )
        context, citations, source_contexts = self.context_assembler.assemble_with_sources(
            results,
            chunks_by_id,
            max_chars=18_000,
        )
        if not context or not citations:
            return ContributionAnswer(
                answer="Không đủ bằng chứng nguồn để tính mức đóng góp.",
                citations=[],
                source_contexts=[],
                calculations=[],
            )
        return build_contribution_answer(
            parsed_query,
            results,
            citations,
            source_contexts,
            chunks_by_id=chunks_by_id,
        )

    def generate_ratio_answer(
        self,
        parsed_query: ParsedQuery,
        results: List[RetrievalResult],
        chunks_by_id: Optional[Dict[str, FinancialChunk]] = None,
    ) -> RatioAnswer:
        """Calculate registered balance-sheet ratios from cited source rows."""
        if not results:
            return RatioAnswer(
                answer="Không đủ bằng chứng nguồn để tính tỷ số.",
                citations=[],
                source_contexts=[],
                calculations=[],
            )
        context, citations, source_contexts = self.context_assembler.assemble_with_sources(
            results,
            chunks_by_id,
            max_chars=18_000,
        )
        if not context or not citations:
            return RatioAnswer(
                answer="Không đủ bằng chứng nguồn để tính tỷ số.",
                citations=[],
                source_contexts=[],
                calculations=[],
            )
        return build_ratio_answer(
            parsed_query,
            results,
            citations,
            source_contexts,
            chunks_by_id=chunks_by_id,
        )

    @staticmethod
    def _structured_financial_answer(
        parsed_query: ParsedQuery,
        results: List[RetrievalResult],
        citations: List[Citation],
    ) -> Optional[str]:
        """Answer simple metric lookups from parsed values without model paraphrasing."""
        if parsed_query.intent not in (
            QueryIntent.METRIC_LOOKUP,
            QueryIntent.COMPARISON,
            QueryIntent.GROWTH_RATE,
        ):
            return None
        metric = (parsed_query.canonical_metric or "").strip()
        if not metric:
            return None

        requested_companies = parsed_query.company_codes or (
            [parsed_query.company_code] if parsed_query.company_code else []
        )
        query_metric_names = {
            name.strip().casefold()
            for name in [metric, *parsed_query.metric_aliases]
            if name and name.strip()
        }
        requested_years = list(parsed_query.fiscal_years)
        if not requested_years and parsed_query.fiscal_year is not None:
            requested_years = [parsed_query.fiscal_year]
        if not requested_years:
            requested_years = list(dict.fromkeys(
                result.chunk.fiscal_year for result in results if result.chunk.fiscal_year is not None
            ))

        year_pairs = [
            (year, year - 1 if parsed_query.period_type == PeriodType.BEGINNING_OF_YEAR else year)
            for year in requested_years
        ]
        citation_by_chunk = {citation.chunk_id: citation for citation in citations if citation.chunk_id}
        evidence: Dict[tuple[str, int], tuple[str, str, Citation]] = {}
        growth_evidence: Dict[
            str,
            tuple[str, str, str, str, str, Citation, int | None, int | None],
        ] = {}
        has_matching_metric_item = False

        for result in results:
            chunk = result.chunk
            if chunk.chunk_type == ChunkType.TABLE:
                if parsed_query.schema_node_id:
                    table_title = str(
                        chunk.table_title or chunk.metadata.get("table_title") or ""
                    ).casefold()
                    if any(
                        marker in table_title
                        for marker in ("bộ phận", "segment", "thuyết minh", "note")
                    ):
                        continue
                table_values, table_unit = FinancialChatEngine._table_metric_values(
                    chunk, query_metric_names
                )
                if not table_values or (requested_companies and chunk.company_code not in requested_companies):
                    continue
                if parsed_query.report_type and chunk.report_type and chunk.report_type != parsed_query.report_type:
                    continue
                citation = citation_by_chunk.get(chunk.chunk_id)
                if citation is None or not chunk.company_code:
                    continue
                has_matching_metric_item = True
                current_year, prior_year = FinancialChatEngine._growth_year_pair(
                    parsed_query,
                    parsed_query.fiscal_year or chunk.fiscal_year,
                    chunk.comparative_fiscal_year,
                )
                current_value = table_values.get(current_year) if current_year is not None else None
                prior_value = table_values.get(prior_year) if prior_year is not None else None
                if parsed_query.intent == QueryIntent.GROWTH_RATE and current_value and prior_value:
                    current_decimal = FinancialChatEngine._decimal_value(current_value)
                    prior_decimal = FinancialChatEngine._decimal_value(prior_value)
                    if current_decimal is not None and prior_decimal is not None:
                        delta = current_decimal - prior_decimal
                        growth_evidence.setdefault(
                            chunk.company_code,
                            (
                                FinancialChatEngine._format_number(abs(delta)),
                                "Tăng" if delta > 0 else "Giảm" if delta < 0 else "Không đổi",
                                table_unit,
                                current_value,
                                prior_value,
                                citation,
                                prior_year,
                                current_year,
                            ),
                        )
                for display_year, evidence_year in year_pairs:
                    value = table_values.get(evidence_year)
                    if value is not None:
                        evidence.setdefault(
                            (chunk.company_code, display_year),
                            (value, table_unit, citation),
                        )
                continue

            if chunk.chunk_type != ChunkType.FINANCIAL_ITEM:
                continue
            if parsed_query.schema_node_id:
                source_node_id = str(chunk.metadata.get("schema_node_id") or "")
                if source_node_id:
                    if source_node_id != parsed_query.schema_node_id:
                        continue
                else:
                    source_label = source_item_label(chunk)
                    normalized_source_label = FinancialChatEngine._normalize_table_label(
                        source_label
                    )
                    expected_source_labels = {
                        FinancialChatEngine._normalize_table_label(alias)
                        for alias in parsed_query.metric_aliases
                        if alias
                    }
                    if (
                        not normalized_source_label
                        or normalized_source_label not in expected_source_labels
                    ):
                        continue
            chunk_metric = (
                chunk.canonical_metric
                or chunk.metadata.get("canonical_metric")
                or chunk.metadata.get("metric_name")
                or chunk.metadata.get("metric")
                or ""
            )
            if str(chunk_metric).strip().casefold() not in query_metric_names:
                continue
            if parsed_query.report_type and chunk.report_type and chunk.report_type != parsed_query.report_type:
                continue
            if requested_companies and chunk.company_code not in requested_companies:
                continue
            has_matching_metric_item = True

            citation = citation_by_chunk.get(chunk.chunk_id)
            if citation is None or not chunk.company_code:
                continue
            period_values = chunk.metadata.get("period_values", {})
            if not isinstance(period_values, dict):
                period_values = {}
            if parsed_query.intent == QueryIntent.GROWTH_RATE:
                difference = chunk.metadata.get("difference_formatted")
                trend = chunk.metadata.get("trend")
                current_year, prior_year = FinancialChatEngine._growth_year_pair(
                    parsed_query,
                    parsed_query.fiscal_year or chunk.fiscal_year,
                    chunk.comparative_fiscal_year,
                )
                current_value = period_values.get(str(current_year)) if current_year is not None else None
                prior_value = period_values.get(str(prior_year)) if prior_year is not None else None
                current_decimal = FinancialChatEngine._decimal_value(current_value)
                prior_decimal = FinancialChatEngine._decimal_value(prior_value)
                difference_decimal = FinancialChatEngine._decimal_value(difference)
                if current_decimal is not None and prior_decimal is not None:
                    delta = current_decimal - prior_decimal
                    difference_decimal = abs(delta)
                    trend = "Tăng" if delta > 0 else "Giảm" if delta < 0 else "Không đổi"
                elif not (
                    current_year == chunk.fiscal_year
                    and prior_year == chunk.comparative_fiscal_year
                    and difference_decimal is not None
                ):
                    difference_decimal = None
                if difference_decimal is not None:
                    growth_evidence.setdefault(
                        chunk.company_code,
                        (
                            FinancialChatEngine._format_number(difference_decimal),
                            str(trend or "thay đổi"),
                            chunk.unit or "",
                            str(current_value or ""),
                            str(prior_value or ""),
                            citation,
                            prior_year or chunk.comparative_fiscal_year,
                            current_year or chunk.fiscal_year,
                        ),
                    )
            if not year_pairs:
                year_pairs = [(chunk.fiscal_year, chunk.fiscal_year)] if chunk.fiscal_year is not None else []

            for display_year, evidence_year in year_pairs:
                if display_year is None or evidence_year is None:
                    continue
                value = period_values.get(str(evidence_year))
                if value is None:
                    if chunk.fiscal_year == evidence_year:
                        value = chunk.metadata.get("end_value")
                    elif chunk.comparative_fiscal_year == evidence_year:
                        value = chunk.metadata.get("start_value")
                if value is None:
                    continue
                key = (chunk.company_code, display_year)
                evidence.setdefault(key, (str(value), chunk.unit or "", citation))

        if not has_matching_metric_item:
            if parsed_query.schema_node_id:
                return (
                    f"Không tìm thấy đúng dòng dữ liệu {metric} cho "
                    f"{', '.join(requested_companies) or 'doanh nghiệp được hỏi'} "
                    "trong nguồn đã nạp."
                )
            return None

        companies = requested_companies or list(dict.fromkeys(company for company, _ in evidence))
        if not companies:
            companies = list(growth_evidence)
        if parsed_query.intent == QueryIntent.GROWTH_RATE:
            if not growth_evidence:
                return (
                    f"Chưa tìm thấy giá trị biến động có cấu trúc cho {metric} "
                    "trong nguồn đã nạp."
                )
            lines = []
            for company in companies:
                growth = growth_evidence.get(company)
                if growth is None:
                    lines.append(f"{company}: chưa tìm thấy giá trị biến động trong nguồn đã nạp")
                    continue
                difference, trend, unit, current_value, prior_value, citation, prior_year, current_year = growth
                page_suffix = f" (trang {citation.page})" if citation.page is not None else ""
                unit_suffix = f" {unit}" if unit else ""
                periods = (
                    f" (năm {current_year}: {current_value}{unit_suffix}; "
                    f"năm {prior_year}: {prior_value}{unit_suffix})"
                    if current_value and prior_value and current_year and prior_year
                    else ""
                )
                lines.append(
                    f"{company}: {trend.casefold()} {difference}{unit_suffix} so với kỳ trước{periods} "
                    f"[{citation.citation_id}]{page_suffix}"
                )
            return f"{metric}: " + "; ".join(lines) + "."
        display_year_pairs: list[tuple[Optional[int], Optional[int]]] = list(year_pairs)
        if not display_year_pairs:
            display_year_pairs.append((None, None))
        period_kind = parsed_query.period_type
        period_labels = {
            PeriodType.END_OF_YEAR: "cuối năm",
            PeriodType.BEGINNING_OF_YEAR: "đầu năm",
            PeriodType.FULL_YEAR: "năm",
            PeriodType.HALF_YEAR: "kỳ sáu tháng",
            PeriodType.QUARTERLY: "quý",
        }
        label = period_labels.get(period_kind, "năm")
        lines = []

        for display_label_year, _evidence_year in display_year_pairs:
            year_label = f" {display_label_year}" if display_label_year is not None else ""
            period_label = f"{label}{year_label}"
            period_rows = []
            for company in companies:
                row = evidence.get((company, display_label_year)) if display_label_year is not None else None
                if row is None and display_label_year is None:
                    available = next(
                        (item for (code, _year), item in evidence.items() if code == company),
                        None,
                    )
                    row = available
                if row is None:
                    period_rows.append(f"{company}: chưa tìm thấy số liệu cho {period_label} trong nguồn đã nạp")
                    continue
                value, unit, citation = row
                unit_suffix = f" {unit}" if unit else ""
                page_suffix = f" (trang {citation.page})" if citation.page is not None else ""
                period_rows.append(
                    f"{company}: {value}{unit_suffix} [{citation.citation_id}]{page_suffix}"
                )

            if len(companies) > 1 or len(display_year_pairs) > 1:
                lines.append(f"{metric} {period_label}: " + "; ".join(period_rows) + ".")
            else:
                lines.append(f"{metric} {period_label} của {companies[0]}: {period_rows[0].split(': ', 1)[1]}.")

        return "\n".join(lines)


    @staticmethod
    def _table_metric_values(
        chunk: FinancialChunk,
        query_metric_names: set[str],
    ) -> tuple[Dict[int, str], str]:
        """Read a matching metric row from a preserved Markdown table chunk."""
        lines = chunk.content.splitlines()
        normalized_query_names = {
            FinancialChatEngine._normalize_table_label(name)
            for name in query_metric_names
            if name
        }
        headers: list[str] | None = None
        current_index = None
        prior_index = None
        label_index = 0

        def clean_cell(cell: str) -> str:
            return re.sub(r"\s+", " ", cell.replace("**", " ").replace("__", " ")).strip()

        def is_value(cell: str) -> bool:
            value = clean_cell(cell)
            return bool(re.fullmatch(r"(?:\(\s*[+-]?\s*\d[\d.,]*\s*\)|[+-]?\s*\d[\d.,]*)", value))

        period_values = FinancialChatEngine._narrative_period_values(chunk, normalized_query_names)
        if period_values:
            return period_values, chunk.unit or ""

        for line in lines:
            if "|" not in line:
                continue
            cells = [clean_cell(cell) for cell in line.strip().strip("|").split("|")]
            normalized = [FinancialChatEngine._normalize_table_label(cell) for cell in cells]
            current_candidates = [
                index for index, header in enumerate(normalized)
                if any(token in header for token in ("nam nay", "so cuoi nam", "ky nay", "current year"))
            ]
            prior_candidates = [
                index for index, header in enumerate(normalized)
                if any(token in header for token in ("nam truoc", "so dau nam", "ky truoc", "prior year"))
            ]
            if current_candidates or prior_candidates:
                headers = cells
                current_index = current_candidates[0] if current_candidates else None
                prior_index = prior_candidates[0] if prior_candidates else None
                label_candidates = [
                    index for index, header in enumerate(normalized)
                    if any(token in header for token in ("chi tieu", "noi dung", "khoan muc", "description", "item"))
                ]
                label_index = label_candidates[0] if label_candidates else 0
                continue
            if headers is None or current_index is None:
                continue
            if re.fullmatch(r"\s*[-:| ]+\s*", line):
                continue
            if len(cells) <= max(label_index, current_index, prior_index or 0):
                continue

            label = FinancialChatEngine._normalize_table_label(cells[label_index])
            matches = [
                alias for alias in normalized_query_names
                if alias and (label == alias or (alias in label and len(alias) / max(len(label), 1) >= 0.55))
            ]
            if not matches:
                continue
            if not is_value(cells[current_index]):
                continue

            values: Dict[int, str] = {}
            if chunk.fiscal_year is not None:
                values[chunk.fiscal_year] = clean_cell(cells[current_index])
            if prior_index is not None and prior_index < len(cells) and is_value(cells[prior_index]):
                prior_year = chunk.comparative_fiscal_year
                if prior_year is None and chunk.fiscal_year is not None:
                    prior_year = chunk.fiscal_year - 1
                if prior_year is not None:
                    values[prior_year] = clean_cell(cells[prior_index])

            row_unit = chunk.unit or ""
            if "co phieu" in label:
                row_unit = "VNĐ/cổ phiếu"
            if values:
                return values, row_unit
        return {}, chunk.unit or ""

    @staticmethod
    def _narrative_period_values(
        chunk: FinancialChunk,
        normalized_query_names: set[str],
    ) -> Dict[int, str]:
        """Extract named current/prior values from preserved cash-flow item lines."""
        value_pattern = re.compile(
            r"(?P<value>\(\s*[+-]?\s*\d[\d.,]*\s*\)|[+-]?\s*\d[\d.,]*)\s*"
            r"(?P<unit>(?:(?:triệu|tỷ|nghìn|ngàn)\s+)?(?:đồng|vnd|vnđ))?\s*"
            r"\(\s*(?P<period>năm\s+nay|năm\s+trước|số\s+cuối\s+năm|số\s+đầu\s+năm)\s*\)",
            re.IGNORECASE,
        )
        for line in chunk.content.splitlines():
            if ":" not in line:
                continue
            label = FinancialChatEngine._normalize_table_label(line.split(":", 1)[0])
            if not any(
                alias and (label == alias or (alias in label and len(alias) / max(len(label), 1) >= 0.55))
                for alias in normalized_query_names
            ):
                continue
            values: Dict[int, str] = {}
            for match in value_pattern.finditer(line):
                period = match.group("period").casefold()
                if period in {"năm nay", "số cuối năm"} and chunk.fiscal_year is not None:
                    values[chunk.fiscal_year] = match.group("value").strip()
                elif period in {"năm trước", "số đầu năm"}:
                    prior_year = chunk.comparative_fiscal_year
                    if prior_year is None and chunk.fiscal_year is not None:
                        prior_year = chunk.fiscal_year - 1
                    if prior_year is not None:
                        values[prior_year] = match.group("value").strip()
            if values:
                return values
        return {}

    @staticmethod
    def _decimal_value(value: object) -> Optional[Decimal]:
        if value is None:
            return None
        normalized = NumericValidator.normalize_number(str(value))
        if not normalized:
            return None
        try:
            return Decimal(normalized)
        except InvalidOperation:
            return None

    @staticmethod
    def _format_number(value: Decimal) -> str:
        text = format(abs(value), "f")
        integer, separator, fraction = text.partition(".")
        grouped_integer = f"{int(integer):,}".replace(",", ".")
        return f"{grouped_integer}{separator}{fraction}" if separator else grouped_integer
