import re
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from ..schemas import ChunkType, FinancialChunk, ParsedQuery, StatementType, PeriodType, QueryIntent
from ..metadata.company_extractor import CompanyExtractor
from ..metadata.metric_mapper import FinancialMetricMapper, STATEMENT_NAMES_VI
from ..structure.report_schema_catalog import get_report_schema_catalog

class QueryUnderstandingEngine:

    """
    Lớp Query Understanding chuyên biệt cho Báo cáo Tài chính Việt Nam.
    Nhiệm vụ:
    1. Bóc tách Mã cổ phiếu / Công ty (VD: ABB, VCB, FPT).
    2. Nhận diện Chỉ tiêu tài chính chính thức (VD: TỔNG NỢ PHẢI TRẢ, LỢI NHUẬN SAU THUẾ).
    3. Nhận diện Báo cáo liên quan (Bảng cân đối kế toán vs Kết quả kinh doanh vs Thuyết minh).
    4. Xác định Kỳ báo cáo (Số đầu năm vs Số cuối năm).
    5. Viết lại câu truy vấn (Query Rewrite) để tối ưu cho BM25 và Vector Search.
    """

    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            # Tìm file financial_terms_vi.json trong thư mục configs
            current_dir = Path(__file__).resolve().parent
            config_path = current_dir.parent.parent.parent / "configs" / "financial_terms_vi.json"

        self.config_path = config_path
        self.terms_data = self._load_terms_config()
        self.company_extractor = CompanyExtractor()
        self.metric_mapper = FinancialMetricMapper(config_path)
        self.schema_catalog = get_report_schema_catalog()

    def _load_terms_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            return {}
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def parse(self, user_query: str) -> ParsedQuery:
        query_lower = user_query.lower().strip()

        # 1. Resolve ticker abbreviations and full company/bank names to codes.
        company_codes = self._extract_companies(user_query)
        company_code = company_codes[0] if company_codes else None

        # 2. Extract years after resolving the company so retrieval can scope
        #    reports to the requested company before choosing the fiscal year.
        fiscal_years = self._extract_years(query_lower)
        fiscal_year = fiscal_years[0] if fiscal_years else None

        # 3. Trích xuất Quý tài chính
        fiscal_quarter = self._extract_quarter(query_lower)
        report_type = self._extract_report_type(query_lower)

        # 4. Trích xuất Kỳ báo cáo (Đầu năm vs Cuối năm)
        period_type = self._extract_period_type(query_lower)

        # 5. Nhận diện Báo cáo & Chỉ tiêu tài chính bằng Bộ Ánh Xạ Chuẩn Hóa
        metric_def = self.metric_mapper.lookup_metric(user_query)
        # A broad registry alias such as "doanh thu" can hide a more specific
        # industry line. Let an explicitly named source row win when the schema
        # match is strictly more specific; retrieved row labels and values still
        # have to be confirmed from the source document later in the pipeline.
        schema_statement_hint = (
            metric_def.statement_type
            if metric_def
            else self.metric_mapper.detect_statement_from_query(user_query)
        )
        query_schema_node = self.schema_catalog.resolve_query_node(
            user_query, statement_type=schema_statement_hint
        )
        normalized_for_specificity = self.schema_catalog.normalize_label(user_query)

        def matched_alias_length(aliases: List[str]) -> int:
            return max((
                len(normalized_alias)
                for alias in aliases
                if (normalized_alias := self.schema_catalog.normalize_label(alias))
                and normalized_alias in normalized_for_specificity
            ), default=0)

        schema_preferred_node = None
        if query_schema_node and query_schema_node.kind in {"line", "total", "group"}:
            mapper_specificity = matched_alias_length(metric_def.aliases) if metric_def else 0
            schema_specificity = matched_alias_length([
                query_schema_node.canonical_name,
                *query_schema_node.aliases,
            ])
            if schema_specificity > mapper_specificity:
                schema_preferred_node = query_schema_node
                metric_def = None
        asset_table_request = self._is_asset_table_request(user_query)
        primary_metric: Optional[str] = None
        canonical_metric: Optional[str] = None
        statement_type: Optional[StatementType] = None
        statement_name_vi: Optional[str] = None
        metric_aliases: List[str] = []
        if metric_def:
            primary_metric = metric_def.canonical_name
            canonical_metric = metric_def.canonical_name
            statement_type = metric_def.statement_type
            statement_name_vi = metric_def.statement_name_vi
            metric_aliases = metric_def.aliases
        elif schema_preferred_node:
            statement_type = schema_preferred_node.statement_type
            statement_name_vi = STATEMENT_NAMES_VI.get(statement_type)
            primary_metric = schema_preferred_node.canonical_name
            canonical_metric = schema_preferred_node.canonical_name
            metric_aliases = list(schema_preferred_node.aliases)
        else:
            statement_type, primary_metric = self._extract_statement_and_metric(query_lower)
            canonical_metric = primary_metric
            if not statement_type:
                statement_type = self.metric_mapper.detect_statement_from_query(user_query)
            statement_name_vi = (
                STATEMENT_NAMES_VI.get(statement_type) if statement_type else None
            )
            metric_aliases = self._aliases_for_metric(primary_metric)

        # "Bảng tài sản" is a request for the asset side of the balance
        # sheet, not a request for the fixed-assets note or a free-form search.
        if asset_table_request and metric_def is None and schema_preferred_node is None:
            statement_type = StatementType.BALANCE_SHEET
            statement_name_vi = STATEMENT_NAMES_VI[StatementType.BALANCE_SHEET]
            primary_metric = "Tài sản"
            canonical_metric = None
            metric_aliases = ["Tài sản"]

        structured_operation = self._detect_structured_operation(user_query)
        if structured_operation == "ratio_analysis":
            statement_type = StatementType.BALANCE_SHEET
            statement_name_vi = STATEMENT_NAMES_VI[StatementType.BALANCE_SHEET]
        if asset_table_request and metric_def is None and structured_operation is None:
            structured_operation = "components"
        full_statement_table_request = (
            structured_operation is None
            and metric_def is None
            and not primary_metric
            and self._is_full_statement_table_request(user_query)
        )
        if full_statement_table_request:
            structured_operation = "table"

        normalized_query = self.schema_catalog.normalize_label(user_query)
        issuer_ranking = structured_operation == "ranking" and len(company_codes) > 1
        special_group_id: Optional[str] = None
        special_group_name: Optional[str] = None
        if structured_operation and "nguon von" in normalized_query and "chu so huu" not in normalized_query:
            special_group_id = "SPECIAL.FUNDING_STRUCTURE"
            special_group_name = "Nguồn vốn"
            statement_type = StatementType.BALANCE_SHEET
            statement_name_vi = STATEMENT_NAMES_VI[statement_type]
            primary_metric = special_group_name
            canonical_metric = None
            metric_aliases = [special_group_name]
        elif structured_operation and (
            "dong tien" in normalized_query or "luu chuyen tien" in normalized_query
        ) and "tien va cac khoan tuong duong tien" not in normalized_query:
            special_group_id = "SPECIAL.CASH_FLOW_STRUCTURE"
            special_group_name = "Lưu chuyển tiền theo hoạt động"
            statement_type = StatementType.CASH_FLOW
            statement_name_vi = STATEMENT_NAMES_VI[statement_type]
            primary_metric = special_group_name
            canonical_metric = None
            metric_aliases = [special_group_name]
        elif structured_operation and metric_def is None and "chi phi" in normalized_query:
            special_group_id = "SPECIAL.INCOME_EXPENSES"
            special_group_name = "Các khoản chi phí"
            statement_type = StatementType.INCOME_STATEMENT
            statement_name_vi = STATEMENT_NAMES_VI[statement_type]
            primary_metric = special_group_name
            canonical_metric = None
            metric_aliases = [special_group_name]

        if structured_operation == "table":
            schema_node = None
        elif special_group_id:
            schema_node = None
        elif asset_table_request and metric_def is None:
            schema_node = self.schema_catalog.nodes.get("ASSET")
        else:
            # Với metric lookup, resolve từ metric đã chuẩn hóa trước. Dò cả
            # câu hỏi có thể thất bại khi thêm động từ như “so sánh”, dù metric
            # đã nhận diện được (ví dụ “so sánh nợ phải trả”).
            schema_node = None
            if (not structured_operation or issuer_ranking) and primary_metric:
                schema_node = self.schema_catalog.resolve_query_node(
                    primary_metric,
                    statement_type=statement_type,
                )
            if schema_node is None:
                schema_node = self.schema_catalog.resolve_query_node(
                    user_query,
                    statement_type=statement_type,
                    prefer_container=bool(structured_operation) and not issuer_ranking,
                )
            if schema_node is None and structured_operation and primary_metric:
                # The metric mapper can recognize a short synonym such as
                # "giá vốn" even when that exact short phrase is not an alias
                # in the report schema. Resolve its canonical label so a
                # composition request still enters the schema-backed path.
                schema_node = self.schema_catalog.resolve_query_node(
                    primary_metric,
                    statement_type=statement_type,
                    prefer_container=True,
                )
        if (
            schema_node
            and not structured_operation
            and primary_metric
            and schema_node.kind in {"line", "total", "group"}
            and schema_node.aliases
        ):
            # Once a taxonomy node is resolved, use only that node's source
            # labels. Registry-wide aliases can be overly broad (for example,
            # "doanh thu" also matches a segment-note row when the target is
            # enterprise net revenue).
            metric_aliases = list(schema_node.aliases)
        if structured_operation and schema_node:
            # A question about the components of a reported total targets its
            # parent section, while a direct lookup continues to target the total.
            if schema_node.kind == "total" and schema_node.parent_id and not issuer_ranking:
                schema_node = self.schema_catalog.nodes.get(schema_node.parent_id) or schema_node
            statement_type = schema_node.statement_type
            statement_name_vi = STATEMENT_NAMES_VI.get(statement_type)
            primary_metric = schema_node.canonical_name
            is_container = (
                schema_node.kind in {"section", "group", "root", "note_section"}
                or bool(schema_node.children)
            )
            canonical_metric = None if is_container else schema_node.canonical_name
            metric_aliases = list(schema_node.aliases)
        elif (
            schema_node
            and schema_node.kind in {"line", "total", "group"}
            and not primary_metric
        ):
            # Tên dòng có trong taxonomy nhưng chưa có trong metric registry
            # vẫn phải được nhận diện như một metric trực tiếp. Ví dụ: “Đầu
            # tư nắm giữ đến ngày đáo hạn” trong BCTC doanh nghiệp.
            statement_type = schema_node.statement_type
            statement_name_vi = STATEMENT_NAMES_VI.get(statement_type)
            primary_metric = schema_node.canonical_name
            canonical_metric = schema_node.canonical_name
            metric_aliases = list(schema_node.aliases)

        # 6. Nhận diện Ý định (Intent)
        intent = self._extract_intent(query_lower)

        # 7. Sinh câu truy vấn chuẩn hóa cho BM25 (Sparse) & Vector (Dense)
        rewritten_bm25 = self._build_bm25_query(
            company_codes, primary_metric, statement_type, period_type, fiscal_years,
            fiscal_quarter, report_type,
        )
        rewritten_vector = self._build_vector_query(
            user_query, company_codes, primary_metric, statement_type, period_type, fiscal_years,
            fiscal_quarter, report_type,
        )

        # 8. Tạo Metadata Filter
        metadata_filter: Dict[str, Any] = {}
        if statement_type:
            metadata_filter["statement_type"] = statement_type.value
            if statement_name_vi:
                metadata_filter["statement_name_vi"] = statement_name_vi
        if canonical_metric:
            metadata_filter["canonical_metric"] = canonical_metric
        if len(company_codes) == 1:
            metadata_filter["company_code"] = company_code
        elif company_codes:
            metadata_filter["company_codes"] = company_codes
        if fiscal_year:
            metadata_filter["fiscal_year"] = fiscal_year
        if len(fiscal_years) > 1:
            metadata_filter["fiscal_years"] = fiscal_years
        if fiscal_quarter:
            metadata_filter["fiscal_quarter"] = fiscal_quarter
        if report_type:
            metadata_filter["report_type"] = report_type

        return ParsedQuery(
            original_query=user_query,
            company_code=company_code,
            company_codes=company_codes,
            statement_type=statement_type,
            statement_name_vi=statement_name_vi,
            canonical_metric=canonical_metric,
            metric_id=(
                special_group_id
                if special_group_id
                else schema_node.node_id
                if schema_node and ((structured_operation and not issuer_ranking) or schema_preferred_node)
                else metric_def.metric_id if metric_def else None
            ),
            primary_metric=primary_metric,
            metric_aliases=metric_aliases,
            period_type=period_type,
            fiscal_year=fiscal_year,
            fiscal_years=fiscal_years,
            fiscal_quarter=fiscal_quarter,
            report_type=report_type,
            intent=intent,
            structured_operation=structured_operation,
            schema_node_id=(
                special_group_id
                or (
                    schema_node.node_id
                    if schema_node
                    and (
                        structured_operation
                        or (
                            primary_metric
                            and schema_node.kind in {"line", "total", "group"}
                        )
                    )
                    else None
                )
            ),
            schema_node_name=(
                special_group_name
                or (
                    schema_node.canonical_name
                    if schema_node
                    and (
                        structured_operation
                        or (
                            primary_metric
                            and schema_node.kind in {"line", "total", "group"}
                        )
                    )
                    else None
                )
            ),
            rewritten_query_bm25=rewritten_bm25,
            rewritten_query_vector=rewritten_vector,
            metadata_filter=metadata_filter
        )

    def resolve_company_context(
        self,
        parsed_query: ParsedQuery,
        chunks: list[FinancialChunk],
    ) -> ParsedQuery:
        """Resolve broad revenue wording against the company's report schema.

        "Doanh thu" is not a single interchangeable line across listed
        companies. Banks report total operating income, while securities firms
        report total operating revenue. Use the indexed company's own
        schema-mapped income statement to choose that sector-level line; leave
        specific phrases such as "doanh thu thuần" untouched.
        """
        if (
            parsed_query.statement_type != StatementType.INCOME_STATEMENT
            or not parsed_query.company_codes
            or parsed_query.canonical_metric not in {
                "DOANH THU THUẦN",
                "DOANH THU BÁN HÀNG VÀ CUNG CẤP DỊCH VỤ",
            }
        ):
            return parsed_query

        normalized_query = self.schema_catalog.normalize_label(parsed_query.original_query)
        has_broad_revenue_term = any(
            re.search(rf"\b{term}\b", normalized_query)
            for term in ("doanh thu", "doanh so", "revenue")
        )
        specific_revenue_terms = (
            "doanh thu thuan",
            "doanh thu ban hang",
            "ban hang va cung cap dich vu",
            "doanh thu gop",
            "gross revenue",
            "net revenue",
            "doanh thu tai chinh",
            "doanh thu hoat dong tai chinh",
            "doanh thu moi gioi",
            "doanh thu bao hiem",
            "doanh thu phi bao hiem",
        )
        if not has_broad_revenue_term or any(
            re.search(rf"\b{re.escape(term)}\b", normalized_query)
            for term in specific_revenue_terms
        ):
            return parsed_query

        group_ids = (
            "INCOME.BANK.TOTAL_OPERATING_INCOME",
            "INCOME.SECURITIES.TOTAL_OPERATING_REVENUE",
        )
        sector_by_company: list[str] = []
        for company in parsed_query.company_codes:
            company_chunks = [
                chunk
                for chunk in chunks
                if (chunk.company_code or "").casefold() == company.casefold()
                and chunk.statement_type == StatementType.INCOME_STATEMENT
                and chunk.chunk_type == ChunkType.FINANCIAL_ITEM
                and (
                    parsed_query.report_type is None
                    or chunk.report_type == parsed_query.report_type
                )
            ]
            scores: list[tuple[int, str]] = []
            for group_id in group_ids:
                component_ids = set(
                    self.schema_catalog.component_groups.get(group_id, {}).get(
                        "component_node_ids", []
                    )
                )
                observed_ids = {
                    str(chunk.metadata.get("schema_node_id", ""))
                    for chunk in company_chunks
                }
                component_hits = len(component_ids.intersection(observed_ids))
                has_total = group_id in observed_ids
                score = (100 if has_total else 0) + component_hits
                if score:
                    scores.append((score, group_id))
            if not scores:
                return parsed_query
            scores.sort(reverse=True)
            if len(scores) > 1 and scores[0][0] == scores[1][0]:
                return parsed_query
            sector_by_company.append(scores[0][1])

        # Multi-company comparisons can only share one sector-level metric if
        # all requested companies resolve to the same reporting profile.
        if len(set(sector_by_company)) != 1:
            return parsed_query

        node = self.schema_catalog.nodes.get(sector_by_company[0])
        if node is None:
            return parsed_query
        metadata_filter = dict(parsed_query.metadata_filter)
        metadata_filter["canonical_metric"] = node.canonical_name
        metadata_filter["statement_type"] = node.statement_type.value
        metadata_filter["statement_name_vi"] = STATEMENT_NAMES_VI[node.statement_type]
        return parsed_query.model_copy(update={
            "canonical_metric": node.canonical_name,
            "metric_id": node.node_id,
            "primary_metric": node.canonical_name,
            "metric_aliases": list(node.aliases),
            "statement_type": node.statement_type,
            "statement_name_vi": STATEMENT_NAMES_VI[node.statement_type],
            "schema_node_id": node.node_id,
            "schema_node_name": node.canonical_name,
            "rewritten_query_bm25": self._build_bm25_query(
                parsed_query.company_codes,
                node.canonical_name,
                node.statement_type,
                parsed_query.period_type,
                parsed_query.fiscal_years,
                parsed_query.fiscal_quarter,
                parsed_query.report_type,
            ),
            "rewritten_query_vector": self._build_vector_query(
                parsed_query.original_query,
                parsed_query.company_codes,
                node.canonical_name,
                node.statement_type,
                parsed_query.period_type,
                parsed_query.fiscal_years,
                parsed_query.fiscal_quarter,
                parsed_query.report_type,
            ),
            "metadata_filter": metadata_filter,
        })

    def _detect_structured_operation(self, query: str) -> Optional[str]:
        normalized = self.schema_catalog.normalize_label(query)
        if "xep hang" in normalized:
            return "ranking"
        if any(term in normalized for term in (
            "bien loi nhuan gop",
            "ty suat loi nhuan gop",
            "gross margin",
            "bien loi nhuan hoat dong",
            "bien loi nhuan kinh doanh",
            "operating margin",
            "bien loi nhuan sau thue",
            "bien loi nhuan rong",
            "net profit margin",
        )):
            return "ratio_analysis"
        has_ratio_language = any(
            term in normalized
            for term in ("ty le", "ti le", "ty trong", "he so", " tren ")
        )
        has_asset_scope = "tai san" in normalized
        if has_ratio_language and (
            ("no phai tra" in normalized and has_asset_scope)
            or ("no tren" in normalized and has_asset_scope)
            or ("he so no" in normalized and has_asset_scope)
            or ("von chu so huu" in normalized and has_asset_scope)
        ):
            return "ratio_analysis"
        if any(term in normalized for term in (
            "dong gop",
            "gop phan",
            "khoan muc nao lam tang",
            "yeu to nao lam tang",
        )):
            return "contribution_analysis"
        component_terms = (
            "co cau",
            "cau thanh",
            "cau truc",
            "thanh phan",
            "cac bo phan",
            "gom nhung gi",
            "gom nhung",
            "gom cac",
            "gom gi",
            "bao gom",
            "cau phan",
            "cau tao",
            "tao nen",
            "hinh thanh",
            "cac khoan muc",
            "liet ke",
            "chi tiet cac",
            "chi tiet tung",
            "tong hop cac khoan",
            "tom tat cac khoan",
            "tong hop",
            "tom tat",
            "ty trong",
            "ty le",
            "phan tram",
            "phan bo",
            "phan tach",
        )
        if not any(term in normalized for term in component_terms):
            return None
        if any(term in normalized for term in ("lon nhat", "cao nhat", "nhieu nhat", "xep hang")):
            return "ranking"
        if any(term in normalized for term in (
            "co cau", "ty trong", "chiem bao nhieu", "phan tram", "ty le trong",
            "phan bo", "ty le", "ti le",
        )):
            return "composition"
        if "tong hop" in normalized or "tom tat" in normalized:
            return "summary"
        return "components"

    def _is_asset_table_request(self, query: str) -> bool:
        normalized = self.schema_catalog.normalize_label(query)
        return bool(re.search(r"\bbang tai san\b", normalized))

    def _is_full_statement_table_request(self, query: str) -> bool:
        """Identify requests for a complete statement, distinct from one section."""
        normalized = self.schema_catalog.normalize_label(query)
        if self._is_asset_table_request(query):
            return False
        if not any(
            marker in normalized
            for marker in (
                "bang can doi ke toan",
                "bao cao tinh hinh tai chinh",
                "bang ket qua kinh doanh",
                "bang bao cao ket qua",
                "bao cao ket qua hoat dong kinh doanh",
                "bang luu chuyen tien te",
                "bao cao luu chuyen tien te",
                "toan bo bang",
                "day du bang",
            )
        ):
            return False
        return self.metric_mapper.detect_statement_from_query(query) is not None


    def _extract_company(self, query: str) -> Optional[str]:
        # Share the document-side aliases so query parsing and ingestion resolve
        # company names to the same ticker.
        return self.company_extractor.extract(query)

    def _extract_companies(self, query: str) -> list[str]:
        return self.company_extractor.extract_all(query)

    def set_dynamic_company_aliases(self, aliases: Dict[str, List[str]]) -> None:
        self.company_extractor.set_dynamic_aliases(aliases)

    def _extract_year(self, query_lower: str) -> Optional[int]:
        years = self._extract_years(query_lower)
        return years[0] if years else None

    @staticmethod
    def _extract_years(query_lower: str) -> list[int]:
        return list(dict.fromkeys(int(year) for year in re.findall(r"\b(20\d{2}|19\d{2})\b", query_lower)))

    def _extract_quarter(self, query_lower: str) -> Optional[int]:
        match = re.search(r"(?:quý|q)\s*([1-4])\b", query_lower)
        if match:
            return int(match.group(1))
        roman_match = re.search(r"\bquý\s*(i{1,3}|iv)\b", query_lower)
        if roman_match:
            return {"i": 1, "ii": 2, "iii": 3, "iv": 4}[roman_match.group(1)]
        return None

    @staticmethod
    def _extract_report_type(query_lower: str) -> Optional[str]:
        if "hợp nhất" in query_lower:
            return "consolidated"
        if "báo cáo tài chính riêng" in query_lower or "báo cáo riêng" in query_lower:
            return "separate"
        return None

    def _extract_period_type(self, query_lower: str) -> PeriodType:
        if any(w in query_lower for w in ["sáu tháng", "6 tháng", "bán niên", "nửa đầu năm"]):
            return PeriodType.HALF_YEAR
        if any(w in query_lower for w in ["đầu năm", "đầu kỳ", "số đầu năm", "bắt đầu năm"]):
            return PeriodType.BEGINNING_OF_YEAR
        if any(w in query_lower for w in ["cuối năm", "cuối kỳ", "số cuối năm", "kết thúc năm"]):
            return PeriodType.END_OF_YEAR
        if any(w in query_lower for w in ["cả năm", "lũy kế", "toàn năm"]):
            return PeriodType.FULL_YEAR
        if self._extract_quarter(query_lower) is not None:
            return PeriodType.QUARTERLY
        return PeriodType.UNSPECIFIED

    def _extract_statement_and_metric(self, query_lower: str):
        st_data = self.terms_data.get("statement_types", {})

        # 1. Tìm chỉ tiêu cụ thể trước (ưu tiên cụm từ dài nhất)
        matched_metric = None
        matched_st_type = None
        longest_len = 0

        for st_key, st_info in st_data.items():
            for m in st_info.get("primary_metrics", []):
                for alias in m.get("aliases", []):
                    if alias in query_lower and len(alias) > longest_len:
                        matched_metric = m.get("canonical")
                        matched_st_type = StatementType(st_key)
                        longest_len = len(alias)

        if matched_metric:
            return matched_st_type, matched_metric

        # 2. Nếu không có chỉ tiêu, tìm xem có nhắc đến tên báo cáo không
        for st_key, st_info in st_data.items():
            for kw in st_info.get("keywords", []):
                if kw in query_lower:
                    return StatementType(st_key), None

        return None, None

    def _aliases_for_metric(self, primary_metric: Optional[str]) -> list[str]:
        if not primary_metric:
            return []
        for statement in self.terms_data.get("statement_types", {}).values():
            for metric in statement.get("primary_metrics", []):
                if metric.get("canonical", "").casefold() == primary_metric.casefold():
                    return list(dict.fromkeys([primary_metric, *metric.get("aliases", [])]))
        return [primary_metric]

    def _extract_intent(self, query_lower: str) -> QueryIntent:
        if any(w in query_lower for w in ["tăng bao nhiêu", "giảm bao nhiêu", "%", "tỷ lệ tăng", "tăng trưởng"]):
            return QueryIntent.GROWTH_RATE
        if any(w in query_lower for w in ["so sánh", "so voi", "so với", "biến động", "chênh lệch"]):
            return QueryIntent.COMPARISON
        if any(w in query_lower for w in ["tại sao", "nguyên nhân", "lý do", "chính sách"]):
            return QueryIntent.EXPLANATION
        return QueryIntent.METRIC_LOOKUP

    def _build_bm25_query(
        self,
        company_codes: list[str],
        primary_metric: Optional[str],
        statement_type: Optional[StatementType],
        period_type: PeriodType,
        fiscal_years: list[int],
        fiscal_quarter: Optional[int],
        report_type: Optional[str],
    ) -> str:
        parts = []
        if primary_metric:
            parts.append(f'"{primary_metric}"')

        if period_type == PeriodType.BEGINNING_OF_YEAR:
            parts.append('"Số đầu năm"')
        elif period_type == PeriodType.END_OF_YEAR:
            parts.append('"Số cuối năm"')
        elif period_type == PeriodType.FULL_YEAR:
            parts.append('"Cả năm"')
        elif period_type == PeriodType.HALF_YEAR:
            parts.append('"Sáu tháng"')

        if statement_type == StatementType.BALANCE_SHEET:
            parts.append('"BÁO CÁO TÌNH HÌNH TÀI CHÍNH" "BẢNG CÂN ĐỐI KẾ TOÁN"')
        elif statement_type == StatementType.INCOME_STATEMENT:
            parts.append('"KẾT QUẢ HOẠT ĐỘNG KINH DOANH"')
        elif statement_type == StatementType.CASH_FLOW:
            parts.append('"LƯU CHUYỂN TIỀN TỆ"')
        elif statement_type == StatementType.NOTES:
            parts.append('"THUYẾT MINH BÁO CÁO TÀI CHÍNH"')

        parts.extend(company_codes)
        parts.extend(str(year) for year in fiscal_years)
        if fiscal_quarter:
            parts.append(f'"Quý {fiscal_quarter}"')
        if report_type == "consolidated":
            parts.append('"Hợp nhất"')
        elif report_type == "separate":
            parts.append('"Riêng"')

        return " ".join(parts) if parts else ""

    def _build_vector_query(
        self,
        user_query: str,
        company_codes: list[str],
        primary_metric: Optional[str],
        statement_type: Optional[StatementType],
        period_type: PeriodType,
        fiscal_years: list[int],
        fiscal_quarter: Optional[int],
        report_type: Optional[str],
    ) -> str:
        context_parts = []
        if company_codes:
            context_parts.append(f"công ty {' và '.join(company_codes)}")
        if statement_type and statement_type in STATEMENT_NAMES_VI:
            context_parts.append(f"phân hệ {STATEMENT_NAMES_VI[statement_type]}")
        context_parts.extend(f"năm {year}" for year in fiscal_years)
        if period_type == PeriodType.BEGINNING_OF_YEAR:
            context_parts.append("thời điểm số đầu năm")

        elif period_type == PeriodType.END_OF_YEAR:
            context_parts.append("thời điểm số cuối năm")
        elif period_type == PeriodType.FULL_YEAR:
            context_parts.append("kết quả cả năm")
        elif period_type == PeriodType.HALF_YEAR:
            context_parts.append("kỳ sáu tháng")
        if fiscal_quarter:
            context_parts.append(f"quý {fiscal_quarter}")
        if report_type == "consolidated":
            context_parts.append("báo cáo hợp nhất")
        elif report_type == "separate":
            context_parts.append("báo cáo riêng")

        if primary_metric and statement_type:
            st_names = {
                StatementType.BALANCE_SHEET: "Bảng cân đối kế toán",
                StatementType.INCOME_STATEMENT: "Báo cáo kết quả kinh doanh",
                StatementType.CASH_FLOW: "Báo cáo lưu chuyển tiền tệ",
                StatementType.NOTES: "Thuyết minh báo cáo tài chính",
            }
            st_name = st_names.get(statement_type, "báo cáo tài chính")
            return f"Chỉ tiêu {primary_metric} trên {st_name} {' '.join(context_parts)}".strip()

        return f"{user_query} {' '.join(context_parts)}".strip() if context_parts else user_query
