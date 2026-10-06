from ..schemas import FinancialChunk
from ..structure.source_groups import source_item_label

class TextBuilder:
    """
    Xây dựng chuỗi văn bản tối ưu hóa cho mô hình sinh embedding.
    Gắn ngữ cảnh nguồn (doanh nghiệp, năm, tên bảng, phần và nhãn gốc) lên đầu
    chuỗi. Không đưa nhãn chuẩn hóa suy đoán vào embedding passage.
    """

    @classmethod
    def build_chunk_text(cls, chunk: FinancialChunk) -> str:
        parts = []
        if chunk.company_code:
            parts.append(f"Doanh nghiệp: {chunk.company_code}")
        if chunk.fiscal_year:
            parts.append(f"Năm: {chunk.fiscal_year}")
        table_title = chunk.table_title or chunk.metadata.get("table_title")
        if table_title:
            parts.append(f"Bảng: {table_title}")
        section = chunk.section or chunk.metadata.get("section_group")
        if section:
            parts.append(f"Phần: {section}")
        schema_context = str(chunk.metadata.get("schema_context_name") or "").strip()
        if schema_context:
            # The schema's verified parent row adds hierarchy for retrieval
            # (for example, the investment subledger group). The leaf row label
            # and value still come from the source body below.
            parts.append(f"Nhóm cha theo schema: {schema_context}")
        if chunk.page:
            parts.append(f"Trang: {chunk.page}")

        # Use the label extracted from the source, not a canonical label
        # inferred by a registry. A wrong mapping must not become the strongest
        # signal in the vector representation of an otherwise correct chunk.
        raw_label = source_item_label(chunk)
        if raw_label:
            parts.append(f"Nhãn gốc: {raw_label}")

        header = " | ".join(parts)
        source_content = cls.build_source_content(chunk)
        return f"{header}\n{source_content}"

    @staticmethod
    def build_source_content(chunk: FinancialChunk) -> str:
        """Remove synthetic mapping lines while preserving extracted source data."""
        source_lines = [
            line for line in chunk.content.splitlines()
            if not line.strip().casefold().startswith((
                "chỉ tiêu chuẩn hóa:",
                "từ khóa liên quan (aliases):",
                "canonical metric:",
                "metric aliases:",
            ))
        ]
        return "\n".join(source_lines)
