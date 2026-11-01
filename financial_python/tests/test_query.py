from financial_rag.query.understanding import QueryUnderstandingEngine
from financial_rag.metadata.company_extractor import CompanyExtractor
from financial_rag.metadata.period_extractor import PeriodExtractor
from financial_rag.schemas import StatementType, PeriodType

def test_query_understanding():
    engine = QueryUnderstandingEngine()

    # Test case 1: Tra cứu nợ phải trả đầu năm
    q1 = "Tổng nợ phải trả của ABB đầu năm 2025 là bao nhiêu?"
    res1 = engine.parse(q1)
    assert res1.company_code == "ABB"
    assert res1.primary_metric == "TỔNG NỢ PHẢI TRẢ"
    assert res1.metric_id == "tong_no_phai_tra"
    assert res1.statement_type == StatementType.BALANCE_SHEET
    assert res1.period_type == PeriodType.BEGINNING_OF_YEAR
    assert "TỔNG NỢ PHẢI TRẢ" in res1.rewritten_query_bm25
    assert res1.metadata_filter.get("statement_type") == "balance_sheet"

    # Test case 2: Tra cứu doanh thu cuối năm
    q2 = "Doanh thu thuần của FPT năm 2024"
    res2 = engine.parse(q2)
    assert res2.company_code == "FPT"
    assert res2.primary_metric == "DOANH THU THUẦN"
    assert res2.metric_id == "doanh_thu_thuan"
    assert res2.statement_type == StatementType.INCOME_STATEMENT


def test_quarter_report_type_and_embedding_prefix_contract():
    result = QueryUnderstandingEngine().parse(
        "Doanh thu thuần của FPT quý IV năm 2024, báo cáo hợp nhất"
    )

    assert result.fiscal_quarter == 4
    assert result.period_type == PeriodType.QUARTERLY
    assert result.report_type == "consolidated"
    assert result.metadata_filter["fiscal_quarter"] == 4
    assert not result.rewritten_query_vector.startswith("query:")


def test_query_understanding_reuses_company_aliases_from_ingestion():
    query = QueryUnderstandingEngine().parse("Lợi nhuận của Techcombank năm 2025")

    assert query.company_code == "TCB"


def test_query_understanding_recognizes_acbs_without_confusing_it_with_acb():
    query = QueryUnderstandingEngine().parse("Doanh thu thuần của ACBS năm 2024")

    assert query.company_code == "ACBS"


def test_fvtpl_asset_query_with_securities_abbreviation_resolves_to_balance_sheet():
    query = QueryUnderstandingEngine().parse(
        "Các TSTC ghi nhận thông qua lãi/lỗ (FVTPL) của ACBS năm 2023 là bao nhiêu?"
    )

    assert query.company_code == "ACBS"
    assert query.statement_type == StatementType.BALANCE_SHEET
    assert query.schema_node_id == "ASSET.FIN_INV.TRADING_SECURITIES"


def test_fvtpl_loss_query_stays_on_income_statement():
    query = QueryUnderstandingEngine().parse(
        "Lỗ từ các tài sản tài chính FVTPL của ACBS năm 2023 là bao nhiêu?"
    )

    assert query.company_code == "ACBS"
    assert query.statement_type == StatementType.INCOME_STATEMENT
    assert query.schema_node_id == "INCOME.SECURITIES.FVTPL_LOSS"


def test_query_understanding_handles_vinafreight_and_six_month_periods():
    query = QueryUnderstandingEngine().parse(
        "Doanh thu thuần hợp nhất của VNF lũy kế sáu tháng đầu năm 2025 là bao nhiêu?"
    )

    assert query.company_code == "VNF"
    assert query.primary_metric == "DOANH THU THUẦN"
    assert query.period_type == PeriodType.HALF_YEAR
    assert '"Sáu tháng"' in query.rewritten_query_bm25


def test_query_understanding_recognizes_accounting_profit_before_tax_alias():
    query = QueryUnderstandingEngine().parse(
        "Lợi nhuận kế toán trước thuế hợp nhất của VNF lũy kế sáu tháng đầu năm 2025 là bao nhiêu?"
    )

    assert query.company_code == "VNF"
    assert query.metric_id == "tong_loi_nhuan_ke_toan_truoc_thue"
    assert query.primary_metric == "TỔNG LỢI NHUẬN KẾ TOÁN TRƯỚC THUẾ"
    assert query.period_type == PeriodType.HALF_YEAR


def test_query_understanding_matches_labeled_company_and_period_fields():
    examples = [
        {
            "id": "demo_assets_2025",
            "query": "Tổng tài sản của DEMO cuối năm 2025 là bao nhiêu?",
            "company_code": "DEMO",
            "fiscal_year": 2025,
        },
        {
            "id": "demo_revenue_comparison",
            "query": "So sánh doanh thu thuần của DEMO năm 2025 với năm 2024.",
            "company_code": "DEMO",
            "fiscal_year": 2025,
            "fiscal_years": [2025, 2024],
        },
        {
            "id": "demo_quarter_revenue",
            "query": "Doanh thu thuần của DEMO quý II năm 2025 là bao nhiêu?",
            "company_code": "DEMO",
            "fiscal_year": 2025,
            "fiscal_quarter": 2,
        },
    ]
    engine = QueryUnderstandingEngine()

    assert examples
    for example in examples:
        query = engine.parse(example["query"])
        if example.get("company_code"):
            assert query.company_code == example["company_code"], example["id"]
        if example.get("fiscal_year"):
            assert query.fiscal_year == example["fiscal_year"], example["id"]
        if example.get("fiscal_years"):
            assert set(query.fiscal_years) == set(example["fiscal_years"]), example["id"]
        if example.get("fiscal_quarter") is not None:
            assert query.fiscal_quarter == example["fiscal_quarter"], example["id"]
        assert query.metric_id is not None, example["id"]


def test_company_extractor_does_not_treat_common_vietnamese_words_as_tickers():
    extractor = CompanyExtractor()

    assert extractor.extract("Doanh thu thuần") is None
    assert extractor.extract("Lợi nhuận sau thuế") is None
    assert extractor.extract("DOANH THU THUẦN") is None
    assert extractor.extract("LỢI NHUẬN SAU THUẾ") is None
    assert extractor.extract_all("So sánh doanh thu ABB và ACB") == ["ABB", "ACB"]
    assert extractor.extract_all("HÀNG TỒN KHO của ABB") == ["ABB"]
    assert extractor.extract_all("CHO VAY KHÁCH HÀNG của ACB") == ["ACB"]
    assert extractor.extract_all("Báo cáo tài chính của ACB Securities") == ["ACBS"]


def test_period_extractor_recognizes_six_month_reporting_periods():
    assert PeriodExtractor.extract_period_type("Sáu tháng đầu năm 2025") == PeriodType.HALF_YEAR


def test_operating_income_metric_is_classified_as_income_statement():
    query = QueryUnderstandingEngine().parse("Tổng thu nhập hoạt động ABB năm 2024")

    assert query.primary_metric == "TỔNG THU NHẬP HOẠT ĐỘNG"
    assert query.statement_type == StatementType.INCOME_STATEMENT


def test_comparison_query_preserves_both_years_and_metric_aliases():
    query = QueryUnderstandingEngine().parse(
        "Doanh thu thuần FPT năm 2025 tăng bao nhiêu so với năm 2024?"
    )

    assert query.company_code == "FPT"
    assert query.primary_metric == "DOANH THU THUẦN"
    assert query.metric_aliases
    assert query.fiscal_year == 2025
    assert query.fiscal_years == [2025, 2024]
    assert query.metadata_filter["fiscal_years"] == [2025, 2024]


def test_multi_company_comparison_query_preserves_all_companies():
    query = QueryUnderstandingEngine().parse(
        "So sánh tổng tài sản cuối năm 2024 của ABB và ACB theo BCTC hợp nhất."
    )

    assert query.company_code == "ABB"
    assert query.company_codes == ["ABB", "ACB"]
    assert query.metadata_filter["company_codes"] == ["ABB", "ACB"]
    assert "ABB" in query.rewritten_query_bm25
    assert "ACB" in query.rewritten_query_bm25
    assert "ABB và ACB" in query.rewritten_query_vector


def test_query_understanding_4_statements_routing():
    engine = QueryUnderstandingEngine()

    # 1. Bảng cân đối kế toán
    q_bs = engine.parse("Nợ ngắn hạn của ACV cuối năm 2019 là bao nhiêu?")
    assert q_bs.statement_type == StatementType.BALANCE_SHEET
    assert q_bs.statement_name_vi == "Bảng cân đối kế toán"
    assert q_bs.canonical_metric == "NỢ NGẮN HẠN"
    assert "nợ ngắn hạn" in [a.lower() for a in q_bs.metric_aliases]

    # 2. Báo cáo kết quả kinh doanh
    q_is = engine.parse("Lợi nhuận ròng của FPT năm 2024")
    assert q_is.statement_type == StatementType.INCOME_STATEMENT
    assert q_is.statement_name_vi == "Báo cáo kết quả hoạt động kinh doanh"
    assert q_is.canonical_metric == "LỢI NHUẬN SAU THUẾ"

    # 3. Báo cáo lưu chuyển tiền tệ
    q_cf = engine.parse("Dòng tiền kinh doanh của ACV năm 2019")
    assert q_cf.statement_type == StatementType.CASH_FLOW
    assert q_cf.statement_name_vi == "Báo cáo lưu chuyển tiền tệ"
    assert q_cf.canonical_metric == "LƯU CHUYỂN TIỀN TỪ HOẠT ĐỘNG KINH DOANH"

    # 4. Thuyết minh báo cáo tài chính
    q_note = engine.parse("Xem thuyết minh hàng tồn kho của ABB")
    assert q_note.statement_type == StatementType.NOTES
    assert q_note.statement_name_vi == "Thuyết minh báo cáo tài chính"
    assert q_note.canonical_metric == "THUYẾT MINH HÀNG TỒN KHO"
