from financial_rag.validation.numeric import NumericValidator
from financial_rag.validation.citation import CitationValidator
from financial_rag.schemas import Citation

def test_numeric_validation():
    source_context = ["TỔNG NỢ PHẢI TRẢ: Số đầu năm: 620.000 triệu đồng. Số cuối năm: 750.000 triệu đồng."]

    # Ca đúng
    valid_ans = "Tổng nợ phải trả đầu năm của DEMO là 620.000 triệu đồng."
    v_res = NumericValidator.validate(valid_ans, source_context)
    assert v_res.is_valid is True
    assert "620.000" in v_res.extracted_number

    # Ca sai số liệu (ảo giác)
    hallucinated_ans = "Tổng nợ phải trả đầu năm của DEMO là 999.888.777 triệu đồng."
    f_res = NumericValidator.validate(hallucinated_ans, source_context)
    assert f_res.is_valid is False

    # Không được pass nếu câu chứa cả số đúng lẫn số bịa.
    mixed_ans = "Tổng nợ là 620.000 và 999.888.777 triệu đồng."
    assert NumericValidator.validate(mixed_ans, source_context).is_valid is False

    # Không được ghép chữ số từ hai giá trị khác nhau thành một số mới.
    assert NumericValidator.validate("Giá trị là 123.456", ["Giá trị 123 và 456"]).is_valid is False

    # Giá trị đúng nhưng sai đơn vị cũng phải bị từ chối.
    wrong_unit = "Tổng nợ là 620.000 tỷ đồng."
    wrong_unit_result = NumericValidator.validate(wrong_unit, source_context)
    assert wrong_unit_result.is_valid is False
    assert wrong_unit_result.unit_match is False


def test_numeric_validation_preserves_decimal_values_and_localized_grouping():
    vietnamese_decimal = NumericValidator.validate(
        "Tỷ suất tăng 1,5%.", ["Tỷ suất tăng 1.5%."]
    )
    assert vietnamese_decimal.is_valid is True

    wrong_decimal_scale = NumericValidator.validate(
        "Tỷ suất tăng 15%.", ["Tỷ suất tăng 1,5%."]
    )
    assert wrong_decimal_scale.is_valid is False

    equivalent_grouping = NumericValidator.validate(
        "Giá trị 123,456 triệu đồng.", ["Giá trị 123.456 triệu đồng."]
    )
    assert equivalent_grouping.is_valid is True

    equivalent_decimal_locales = NumericValidator.validate(
        "Giá trị 1,234.56 triệu đồng.", ["Giá trị 1.234,56 triệu đồng."]
    )
    assert equivalent_decimal_locales.is_valid is True

    equivalent_negative = NumericValidator.validate(
        "Lỗ lũy kế là -1.234 triệu đồng.", ["Lỗ lũy kế là (1.234) triệu đồng."]
    )
    assert equivalent_negative.is_valid is True


def test_numeric_validation_ignores_labeled_years_and_citation_pages():
    result = NumericValidator.validate(
        "Tổng tài sản cuối năm 2024: DEMO 2.400 Triệu VND [c1] (trang 7); "
        "MOCK 4.000 Triệu đồng [c2] (trang 7).",
        [
            "TỔNG CỘNG TÀI SẢN: Số cuối năm (2024): 2.400 Triệu VND",
            "TỔNG CỘNG TÀI SẢN: Số đầu năm (2024): 4.000 Triệu đồng",
        ],
    )

    assert result.is_valid is True
    assert result.unit_match is True


def test_numeric_validation_ignores_report_row_codes_and_line_numbers():
    cases = [
        (
            "Tổng cộng tài sản là 3.378.833.589.624 VNĐ [c1] (270 = 100 + 200).",
            "TỔNG CỘNG TÀI SẢN (270 = 100 + 200): 3.378.833.589.624 VNĐ",
        ),
        (
            "Nợ phải trả là 501.375.767.445 VNĐ [c1] (300 = 310 + 330).",
            "A. NỢ PHẢI TRẢ (300 = 310 + 330): 501.375.767.445 VNĐ",
        ),
        (
            "3. Đầu tư nắm giữ đến ngày đáo hạn là 1.760.000.000.000 VNĐ [c1].",
            "3. Đầu tư nắm giữ đến ngày đáo hạn: 1.760.000.000.000 VNĐ",
        ),
    ]

    for answer, source in cases:
        result = NumericValidator.validate(answer, [source])
        assert result.is_valid is True, result
        assert result.unit_match is True
        assert result.extracted_number == source.split(":")[-1].strip().split()[0], result


def test_numeric_validation_accepts_difference_derived_from_two_labeled_period_values():
    result = NumericValidator.validate(
        "Lợi nhuận tăng 32 Triệu đồng so với kỳ trước [c1].",
        [
            "Số cuối năm (2025): 96 Triệu đồng\n"
            "Số đầu năm (2024): 64 Triệu đồng"
        ],
    )

    assert result.is_valid is True
    assert "chênh lệch được tính" in result.details


def test_numeric_validation_resolves_units_and_differences_from_markdown_table_columns():
    source = [
        "(Đơn vị: Triệu đồng)\n"
        "**Thuyết minh** | **Năm nay (Triệu đồng)** | **Năm trước (Triệu đồng)**\n"
        "---|---|---\n"
        "**LỢI NHUẬN SAU THUẾ** | **96** | **64**\n"
        "**Lãi cơ bản trên cổ phiếu (đồng/cổ phiếu)** | 12 | 8"
    ]
    answer = (
        "LỢI NHUẬN SAU THUẾ: DEMO: tăng 32 Triệu đồng so với kỳ trước "
        "(năm 2025: 96 Triệu đồng; năm 2024: 64 Triệu đồng) [c1] (trang 11)."
    )

    result = NumericValidator.validate(answer, source)

    assert result.is_valid is True, result
    assert result.unit_match is True


def test_numeric_unit_does_not_leak_from_a_previous_markdown_table():
    source = (
        "| Chỉ tiêu tỷ trọng | Tỷ trọng (%) |\n"
        "| --- | ---: |\n"
        "| Một dòng tham chiếu | 12,5 |\n\n"
        "| Chỉ tiêu | Giá trị (Triệu đồng) |\n"
        "| --- | ---: |\n"
        "| TỔNG NỢ PHẢI TRẢ | 8.125 |"
    )
    number = "8.125"
    start = source.index(number)

    unit = NumericValidator._unit_for_number(source, start, start + len(number))

    assert unit == NumericValidator._normalize_unit("Triệu đồng")


def test_citation_validation():
    citations = [
        Citation(citation_id="c1", document_id="doc1", file_name="abb.pdf", page=9, snippet="test")
    ]
    assert CitationValidator.validate("Theo tài liệu [c1], nợ là...", citations) is True
    assert CitationValidator.validate("Theo tài liệu [c99], nợ là...", citations) is False
    assert CitationValidator.validate("Nợ phải trả là 620.000 triệu đồng.", citations) is False


def test_citation_validation_can_be_optional_for_empty_answers():
    assert CitationValidator.validate_ids("Không có trích dẫn", [], require_citation=False) is True
    assert CitationValidator.validate_ids("Số liệu", ["c1"], require_citation=True) is False
