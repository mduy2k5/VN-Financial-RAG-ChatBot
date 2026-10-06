from typing import List, Dict, Optional
from ..schemas import RetrievalResult, FinancialChunk, Citation
from .citation_mapper import CitationMapper
from .budget import estimate_tokens

class ContextAssembler:
    """
    Module lắp ráp ngữ cảnh (Context Assembly) cho LLM:
    - Khử trùng lặp (Deduplication).
    - Khôi phục ngữ cảnh cha (Parent context restoration).
    - Gắn nhãn Citation ID [c1], [c2] cho từng khối.
    - Quản lý ngân sách ký tự / token.
    """

    def __init__(self, max_chars: int = 6000):
        self.max_chars = max_chars

    def assemble(
        self,
        results: List[RetrievalResult],
        chunks_by_id: Optional[Dict[str, FinancialChunk]] = None
    ) -> tuple[str, List[Citation]]:
        context, citations, _ = self.assemble_with_sources(results, chunks_by_id)
        return context, citations

    def assemble_with_sources(
        self,
        results: List[RetrievalResult],
        chunks_by_id: Optional[Dict[str, FinancialChunk]] = None,
        max_chars: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, List[Citation], List[str]]:
        if not results:
            return "", [], []

        scope_values: Dict[str, set[object]] = {
            "company": {str(res.chunk.company_code).casefold() for res in results if res.chunk.company_code},
            "report_basis": {str(res.chunk.report_type).casefold() for res in results if res.chunk.report_type},
            "filing_year": {res.chunk.fiscal_year for res in results if res.chunk.fiscal_year is not None},
            "fiscal_quarter": {res.chunk.fiscal_quarter for res in results if res.chunk.fiscal_quarter is not None},
            "comparative_year": {
                res.chunk.comparative_fiscal_year for res in results
                if res.chunk.comparative_fiscal_year is not None
            },
        }
        mixed_scope = any(len(values) > 1 for values in scope_values.values())

        # Chọn context trước rồi mới đánh số citation để citation luôn khớp
        # với các chunk thực sự được gửi cho LLM.
        selected_results: List[RetrievalResult] = []
        context_blocks: List[str] = []
        source_contexts: List[str] = []
        total_len = 0
        total_tokens = 0
        seen_chunk_ids: set[str] = set()
        context_limit = max_chars if max_chars is not None else self.max_chars

        for res in results:
            c = res.chunk
            if c.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(c.chunk_id)

            c_id = f"c{len(selected_results) + 1}"
            page_str = f"Trang {c.page}" if c.page else "N/A"
            stmt_str = c.statement_type.value.upper()
            block_header = f"[{c_id}] [{c.company_code or 'DN'} - {stmt_str} - {page_str}]"
            if mixed_scope:
                # Add machine-readable scope only when the retrieved evidence
                # mixes companies, filings, or periods. This disambiguates
                # distractors without inflating prompts for single-scope runs.
                scope_fields = [f"company={c.company_code or 'unknown'}"]
                if c.report_type:
                    scope_fields.append(f"report_basis={c.report_type}")
                if c.fiscal_year is not None:
                    scope_fields.append(f"filing_year={c.fiscal_year}")
                if c.fiscal_quarter is not None:
                    scope_fields.append(f"fiscal_quarter=Q{c.fiscal_quarter}")
                if c.comparative_fiscal_year is not None:
                    scope_fields.append(f"comparative_year={c.comparative_fiscal_year}")
                block_header += f" [source_scope: {'; '.join(scope_fields)}]"

            # Khôi phục thông tin từ parent chunk nếu có
            parent_note_info = ""
            parent_source_context = None
            if c.parent_chunk_id and chunks_by_id and c.parent_chunk_id in chunks_by_id:
                parent_chunk = chunks_by_id[c.parent_chunk_id]
                parent_source_context = parent_chunk.content if max_tokens is not None else parent_chunk.content[:150]
                parent_note_info = f"(Thuộc Thuyết minh cha: {parent_source_context})\n"

            block_text = f"{block_header}\n{parent_note_info}{c.content.strip()}\n"

            separator_len = 5 if context_blocks else 0
            block_tokens = estimate_tokens(block_text) + separator_len
            if max_tokens is not None and (total_tokens + block_tokens > max_tokens
                                          or total_len + separator_len + len(block_text) > context_limit):
                # Keep the child intact if an entire parent cannot fit. A partial
                # parent must never appear to be complete table evidence.
                block_text = f"{block_header}\n{c.content.strip()}\n"
                block_tokens = estimate_tokens(block_text) + separator_len
                parent_source_context = None
                if total_tokens + block_tokens > max_tokens:
                    continue
            if total_len + separator_len + len(block_text) > context_limit:
                continue

            selected_results.append(res)
            context_blocks.append(block_text)
            source_contexts.append(c.content)
            if parent_source_context:
                source_contexts.append(parent_source_context)
            total_len += separator_len + len(block_text)
            total_tokens += block_tokens

        full_context = "\n---\n".join(context_blocks)
        citations, _ = CitationMapper.map_citations(selected_results)
        return full_context, citations, source_contexts
