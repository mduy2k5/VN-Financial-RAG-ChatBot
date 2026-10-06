from enum import Enum
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field

class StatementType(str, Enum):
    BALANCE_SHEET = "balance_sheet"         # Bảng cân đối kế toán / Báo cáo tình hình tài chính
    INCOME_STATEMENT = "income_statement"   # Báo cáo kết quả hoạt động kinh doanh
    CASH_FLOW = "cash_flow"                 # Báo cáo lưu chuyển tiền tệ
    NOTES = "notes"                         # Thuyết minh báo cáo tài chính
    AUDIT_OPINION = "audit_opinion"         # Báo cáo của đơn vị kiểm toán độc lập
    ACCOUNTING_POLICY = "accounting_policy" # Chính sách kế toán chủ yếu
    GENERAL = "general"                     # Khái quát chung / bìa / mục lục

class PeriodType(str, Enum):
    BEGINNING_OF_YEAR = "beginning_of_year" # Số đầu năm
    END_OF_YEAR = "end_of_year"             # Số cuối năm
    FULL_YEAR = "full_year"                 # Cả năm
    HALF_YEAR = "half_year"                 # Sáu tháng / Bán niên
    QUARTERLY = "quarterly"                 # Theo quý
    UNSPECIFIED = "unspecified"

class QueryIntent(str, Enum):
    METRIC_LOOKUP = "metric_lookup"         # Tra cứu một chỉ tiêu cụ thể
    COMPARISON = "comparison"               # So sánh giữa các kỳ hoặc giữa các công ty
    GROWTH_RATE = "growth_rate"             # Tỷ lệ tăng trưởng %
    EXPLANATION = "explanation"             # Giải thích nguyên nhân biến động / chính sách kế toán
    GENERAL = "general"

class ChunkType(str, Enum):
    NARRATIVE = "narrative"                 # Đoạn văn bản thuyết minh, chính sách, ý kiến kiểm toán
    TABLE = "table"                         # Bảng số liệu tài chính đầy đủ hàng/cột
    FINANCIAL_ITEM = "financial_item"       # Cặp chỉ tiêu - giá trị độc lập (key-value atomic)
    PARENT_CHILD = "parent_child"           # Cấu trúc cha - con (ví dụ Thuyết minh -> Các khoản mục con)

class FinancialChunk(BaseModel):
    chunk_id: str
    source_file: str
    document_id: Optional[str] = None
    source_pdf: Optional[str] = None
    company_code: Optional[str] = None
    fiscal_year: Optional[int] = None
    comparative_fiscal_year: Optional[int] = None
    fiscal_quarter: Optional[int] = None
    report_type: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    statement_type: StatementType = StatementType.GENERAL
    statement_name_vi: Optional[str] = None
    canonical_metric: Optional[str] = None
    metric_aliases: List[str] = Field(default_factory=list)
    chunk_type: ChunkType = ChunkType.NARRATIVE
    parent_chunk_id: Optional[str] = None
    is_table: bool = False
    table_title: Optional[str] = None
    unit: Optional[str] = None
    content: str
    primary_metrics: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ParsedQuery(BaseModel):
    original_query: str
    source_policy: Literal["source_report", "latest_restated"] = "source_report"
    company_code: Optional[str] = Field(None, description="Mã cổ phiếu hoặc tên viết tắt (VD: ABB, VNM, FPT)")
    company_codes: List[str] = Field(default_factory=list, description="Tất cả công ty được nêu trong câu hỏi")
    statement_type: Optional[StatementType] = Field(None, description="Loại báo cáo tài chính liên quan")
    statement_name_vi: Optional[str] = Field(None, description="Tên tiếng Việt của phần báo cáo tài chính")
    canonical_metric: Optional[str] = Field(None, description="Tên chỉ tiêu chuẩn hóa trong danh mục BCTC")
    metric_id: Optional[str] = Field(None, description="ID ổn định của chỉ tiêu trong financial metric registry")
    primary_metric: Optional[str] = Field(None, description="Chỉ tiêu kế toán chuẩn (VD: TỔNG NỢ PHẢI TRẢ, LỢI NHUẬN SAU THUẾ)")
    metric_aliases: List[str] = Field(default_factory=list, description="Tên gọi tương đương của chỉ tiêu để tìm bằng chứng")
    period_type: PeriodType = Field(PeriodType.UNSPECIFIED, description="Kỳ báo cáo (đầu/cuối năm, quý hoặc bán niên)")
    fiscal_year: Optional[int] = Field(None, description="Năm tài chính (VD: 2024, 2025)")
    fiscal_years: List[int] = Field(default_factory=list, description="Các năm được nhắc trong câu hỏi so sánh")
    fiscal_quarter: Optional[int] = Field(None, description="Quý (1, 2, 3, 4)")
    report_type: Optional[str] = Field(None, description="Loại báo cáo: consolidated hoặc separate")
    intent: QueryIntent = Field(QueryIntent.METRIC_LOOKUP, description="Ý định câu hỏi")
    structured_operation: Optional[str] = Field(
        None,
        description=(
            "Thao tác phân tích cấu trúc: components, composition, ranking, summary "
            "contribution_analysis hoặc ratio_analysis"
        ),
    )
    schema_node_id: Optional[str] = Field(None, description="Node chuẩn trong bộ schema báo cáo")
    schema_node_name: Optional[str] = Field(None, description="Tên node chuẩn trong bộ schema báo cáo")

    # Sinh tự động cho các bộ tìm kiếm
    rewritten_query_bm25: str = Field("", description="Câu truy vấn chuẩn hóa cho tìm kiếm từ khóa BM25")
    rewritten_query_vector: str = Field("", description="Câu truy vấn tối ưu hóa ngữ nghĩa cho Vector search")
    metadata_filter: Dict[str, Any] = Field(default_factory=dict, description="Bộ lọc metadata cho DB")


class RetrievalResult(BaseModel):
    chunk: FinancialChunk
    rrf_score: float
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None

class Citation(BaseModel):
    citation_id: str
    document_id: str
    chunk_id: Optional[str] = None
    file_name: str
    page: Optional[int] = None
    statement_type: Optional[str] = None
    section: Optional[str] = None
    unit: Optional[str] = None
    snippet: str
    notebook_source_id: Optional[str] = None
    version_id: Optional[str] = None
    projection_id: Optional[str] = None
    block_id: Optional[str] = None
    source_span: Optional[dict] = None
    source_sha256: Optional[str] = None

class ValidationResult(BaseModel):
    is_valid: bool = True
    extracted_number: Optional[str] = None
    source_number: Optional[str] = None
    unit_match: bool = True
    citation_valid: bool = True
    details: str = "Validation passed"

class CalculationInputResponse(BaseModel):
    role: str
    value: str
    unit: str
    year: int
    label: str
    chunk_id: str
    citation_id: str


class CalculationEvidenceResponse(BaseModel):
    value: str
    unit: str
    formula_id: str
    formula_version: str
    expression: str
    input_chunk_ids: List[str] = Field(default_factory=list)
    input_citation_ids: List[str] = Field(default_factory=list)
    inputs: List[CalculationInputResponse] = Field(default_factory=list)


class AnalysisCoverageResponse(BaseModel):
    missing_components: List[str] = Field(default_factory=list)
    residuals: Dict[int, str] = Field(default_factory=dict)
    missing_periods: List[int] = Field(default_factory=list)

class FinancialAnswerResponse(BaseModel):
    query: str
    answer: str
    parsed_query: ParsedQuery
    citations: List[Citation]
    validation: ValidationResult
    calculations: List[CalculationEvidenceResponse] = Field(default_factory=list)
    answer_status: Optional[Literal["complete", "partial", "needs_clarification", "insufficient_evidence"]] = None
    analysis_coverage: Optional[AnalysisCoverageResponse] = None
    source_snapshot: Dict[str, Any] = Field(default_factory=dict)
    claim_checks: List[Dict[str, Any]] = Field(default_factory=list)
