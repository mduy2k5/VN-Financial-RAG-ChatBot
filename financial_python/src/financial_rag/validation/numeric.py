import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from itertools import combinations
from typing import Iterable, List, Optional, Tuple

from ..schemas import ValidationResult
from ..query.directives import normalize_directive_text


class NumericValidator:
    """Kiểm tra từng số và đơn vị được nêu trong câu trả lời với ngữ cảnh nguồn."""

    NON_FINANCIAL_IDENTIFIER_PATTERN = re.compile(
        r"\b\d{1,3}\s*/\s*\d{4}\s*/\s*TT\s*-\s*[A-ZĐ-]+\b|"
        r"\b\d{1,2}/\d{1,2}/(?:19|20)\d{2}\b|"
        r"\(\s*\d{1,4}\s*=\s*\d{1,4}(?:\s*[+-]\s*\d{1,4})+\s*\)|"
        r"^[ \t]*(?:\d{1,3}(?:\.\d+)*|[IVXLCDM]+|[A-Z])\s*[.)]\s+(?=\D)|"
        r"\b(?:19|20)\d{2}\b",
        re.IGNORECASE | re.MULTILINE,
    )

    NUMBER_PATTERN = re.compile(
        r"(?<![\w])(?P<number>(?:\(\s*)?[+-]?\s*\d(?:[\d.,]*\d)?\s*\)?)(?!\w)"
    )
    UNIT_PATTERN = re.compile(
        r"(?i)(?<!\w)(triệu\s+(?:đồng|vnd)|tỷ\s+(?:đồng|vnd)|"
        r"nghìn\s+đồng|ngàn\s+đồng|(?:vnđ|vnd|đồng)\s*/\s*(?:cổ\s+phiếu|cp|share)|"
        r"vnđ|vnd|đồng)(?!\w)|điểm\s+phần\s+trăm|%"
    )
    PERIOD_VALUE_PATTERN = re.compile(
        r"(?im)^\s*(?P<period>số\s+(?:cuối|đầu)\s+(?:năm|kỳ)|năm\s+(?:nay|trước)|kỳ\s+(?:này|trước))"
        r"(?:\s*\((?P<year>(?:19|20)\d{2}|[^\n)]*)\))?\s*[:=]\s*"
        r"(?P<number>\(\s*[+-]?\s*\d[\d.,]*\s*\)|[+-]?\s*\d[\d.,]*)"
    )

    @classmethod
    def normalize_number(cls, num_str: str) -> str:
        """Chuẩn hóa số Việt/Anh mà không làm mất phần thập phân."""
        value = num_str.strip().replace(" ", "")
        is_parenthesized_negative = value.startswith("(") and value.endswith(")")
        if is_parenthesized_negative:
            value = value[1:-1]

        sign = ""
        if value.startswith(("+", "-")):
            if value[0] == "-" or is_parenthesized_negative:
                sign = "-"
            value = value[1:]
        elif is_parenthesized_negative:
            sign = "-"

        if not value or re.fullmatch(r"\d+(?:[.,]\d+)*", value) is None:
            return ""

        separators = {separator for separator in value if separator in ".,"}
        decimal_separator: Optional[str] = None
        if len(separators) == 2:
            # With both punctuation marks present, the last one is the decimal
            # separator (1,234.56 or 1.234,56); the other groups thousands.
            decimal_separator = max((".", ","), key=value.rfind)
        elif separators:
            separator = next(iter(separators))
            groups = value.split(separator)
            if len(groups) == 2:
                # Vietnamese reports commonly use dot grouping; English reports
                # commonly use comma grouping. A three-digit tail is treated as
                # grouping, while other tail lengths preserve decimal meaning.
                if len(groups[1]) != 3 or len(groups[0]) > 3:
                    decimal_separator = separator
            elif not all(len(group) == 3 for group in groups[1:]):
                decimal_separator = separator

        if decimal_separator is None:
            canonical = re.sub(r"[.,]", "", value)
        else:
            integer_part, fractional_part = value.rsplit(decimal_separator, 1)
            integer_part = re.sub(r"[.,]", "", integer_part)
            canonical = f"{integer_part}.{fractional_part}"

        try:
            number = Decimal(f"{sign}{canonical}")
        except InvalidOperation:
            return ""
        if not number.is_finite():
            return ""
        if number == 0:
            return "0"
        canonical_number = format(number, "f")
        if "." in canonical_number:
            canonical_number = canonical_number.rstrip("0").rstrip(".")
        return canonical_number

    @classmethod
    def extract_numbers(cls, text: str) -> List[str]:
        return [match.group("number").strip() for match in cls.NUMBER_PATTERN.finditer(text)]

    @classmethod
    def _normalize_unit(cls, unit: Optional[str]) -> Optional[str]:
        if unit is None:
            return None
        normalized = " ".join(unit.casefold().split())
        aliases = {
            "vnđ": "đồng",
            "vnd": "đồng",
            "triệu vnd": "triệu đồng",
            "tỷ vnd": "tỷ đồng",
            "ngàn đồng": "nghìn đồng",
            "điểm phần trăm": "percentage_points",
            "vnđ/cổ phiếu": "vndshare",
            "vnd/cổ phiếu": "vndshare",
            "đồng/cổ phiếu": "vndshare",
            "đồng/cp": "vndshare",
            "vnd/share": "vndshare",
        }
        return aliases.get(normalized, normalized)

    @classmethod
    def _unit_for_number(cls, text: str, start: int, end: int) -> Optional[str]:
        line_start = text.rfind("\n", 0, start) + 1
        line_end = text.find("\n", end)
        if line_end < 0:
            line_end = len(text)
        row_text = text[line_start:line_end]
        if "|" in row_text:
            row_cells = [cell.strip() for cell in row_text.strip().strip("|").split("|")]
            value_cell_index = next(
                (
                    index for index, cell in enumerate(row_cells)
                    if re.search(r"(?<!\d)" + re.escape(text[start:end].strip()) + r"(?!\d)", cell)
                ),
                None,
            )
            if value_cell_index is not None:
                label = row_cells[0].casefold() if row_cells else ""
                if "cổ phiếu" in label or "đồng/cp" in label:
                    return "vndshare"
                previous_lines = text[:line_start].splitlines()
                for header_line in reversed(previous_lines):
                    if "|" not in header_line:
                        break
                    header_cells = [cell.strip() for cell in header_line.strip().strip("|").split("|")]
                    if header_cells and all(
                        re.fullmatch(r"[:\-\s]+", cell or "-")
                        for cell in header_cells
                    ):
                        continue
                    if value_cell_index < len(header_cells):
                        header_units = list(cls.UNIT_PATTERN.finditer(header_cells[value_cell_index]))
                        if header_units:
                            return cls._normalize_unit(header_units[0].group(0))
                    # Do not scan through a preceding table's header: another
                    # table may use a different unit (for example `%`). The
                    # nearest table header is the only relevant column header.
                    break

        # Ưu tiên đơn vị gần con số; nếu văn bản chỉ có một đơn vị ở tiêu đề bảng,
        # dùng đơn vị đó cho các ô trong bảng.
        window_start = max(0, start - 32)
        window_end = min(len(text), end + 48)
        window = text[window_start:window_end]
        matches = list(cls.UNIT_PATTERN.finditer(window))
        if not matches:
            matches = list(cls.UNIT_PATTERN.finditer(text))
            if len({cls._normalize_unit(match.group(0)) for match in matches}) != 1:
                return None
            return cls._normalize_unit(matches[0].group(0))

        nearest = min(
            matches,
            key=lambda match: min(
                abs(window_start + match.start() - start),
                abs(window_start + match.end() - end),
            ),
        )
        return cls._normalize_unit(nearest.group(0))

    @classmethod
    def _number_entries(cls, text: str) -> List[Tuple[str, Optional[str]]]:
        entries = []
        for match in cls.NUMBER_PATTERN.finditer(text):
            normalized = cls.normalize_number(match.group("number"))
            if normalized:
                entries.append((normalized, cls._unit_for_number(text, match.start(), match.end())))
        return entries

    @classmethod
    def _derived_period_differences(cls, source_contexts: List[str]) -> dict[str, set[str]]:
        """Allow a cited amount only when it is the difference of two labeled periods in one chunk."""
        derived: dict[str, set[str]] = {}
        for context in source_contexts:
            period_values = []
            for match in cls.PERIOD_VALUE_PATTERN.finditer(context):
                number = cls.normalize_number(match.group("number"))
                if not number:
                    continue
                unit = cls._unit_for_number(context, match.start("number"), match.end("number"))
                if unit:
                    period_values.append((Decimal(number), unit))
            for (left_value, left_unit), (right_value, right_unit) in combinations(period_values, 2):
                if left_unit != right_unit:
                    continue
                difference = abs(left_value - right_value)
                if difference == 0:
                    continue
                normalized = cls.normalize_number(format(difference, "f"))
                if normalized:
                    derived.setdefault(normalized, set()).add(left_unit)

            line_offset = 0
            current_index = None
            prior_index = None
            for line in context.splitlines(keepends=True):
                if "|" not in line:
                    line_offset += len(line)
                    continue
                raw_cells = line.strip().strip("|").split("|")
                current_candidates = [
                    index for index, cell in enumerate(raw_cells)
                    if any(marker in cell.casefold() for marker in ("năm nay", "số cuối năm", "kỳ này"))
                ]
                prior_candidates = [
                    index for index, cell in enumerate(raw_cells)
                    if any(marker in cell.casefold() for marker in ("năm trước", "số đầu năm", "kỳ trước"))
                ]
                if current_candidates or prior_candidates:
                    current_index = current_candidates[0] if current_candidates else None
                    prior_index = prior_candidates[0] if prior_candidates else None
                    line_offset += len(line)
                    continue
                if current_index is None or prior_index is None or max(current_index, prior_index) >= len(raw_cells):
                    line_offset += len(line)
                    continue

                def table_cell_value(cell: str) -> tuple[str, int, int] | None:
                    clean = cell.replace("**", " ").replace("__", " ").strip()
                    match = re.fullmatch(r"(?:\(\s*[+-]?\s*\d[\d.,]*\s*\)|[+-]?\s*\d[\d.,]*)", clean)
                    if not match:
                        return None
                    number = cls.normalize_number(clean)
                    if not number:
                        return None
                    position = line.find(cell, 0)
                    if position < 0:
                        return None
                    raw_number_match = re.search(
                        r"\(\s*[+-]?\s*\d[\d.,]*\s*\)|[+-]?\s*\d[\d.,]*",
                        cell,
                    )
                    if raw_number_match is None:
                        return None
                    absolute_start = line_offset + position + raw_number_match.start()
                    absolute_end = line_offset + position + raw_number_match.end()
                    return number, absolute_start, absolute_end

                current_cell = table_cell_value(raw_cells[current_index])
                prior_cell = table_cell_value(raw_cells[prior_index])
                if current_cell and prior_cell:
                    current_unit = cls._unit_for_number(context, current_cell[1], current_cell[2])
                    prior_unit = cls._unit_for_number(context, prior_cell[1], prior_cell[2])
                    if current_unit and current_unit == prior_unit:
                        difference = abs(Decimal(current_cell[0]) - Decimal(prior_cell[0]))
                        normalized = cls.normalize_number(format(difference, "f"))
                        if normalized and difference:
                            derived.setdefault(normalized, set()).add(current_unit)
                line_offset += len(line)
        return derived

    @classmethod
    def _derived_source_percentages(cls, source_contexts: List[str]) -> set[str]:
        """Return two-decimal shares grounded in a same-period, same-unit total row."""
        values_by_period: dict[tuple[str, str], list[Decimal]] = {}
        totals_by_period: dict[tuple[str, str], list[Decimal]] = {}

        parsed_contexts = []
        for context in source_contexts:
            label_match = re.search(
                r"(?im)^\s*chỉ\s+tiêu\s+gốc\s+trong\s+bctc\s*:\s*(?P<label>[^\n|]+)",
                context,
            )
            if label_match is None:
                continue
            label = label_match.group("label").strip()
            group_match = re.search(
                r"(?im)^\s*phân\s+nhóm\s*:\s*(?P<group>[^\n|]+)",
                context,
            )
            group = group_match.group("group").strip() if group_match else ""
            parsed_contexts.append((context, label, group))

        group_signatures = {
            normalize_directive_text(group)
            for _context, _label, group in parsed_contexts
            if group
        }
        for context, raw_label, _group in parsed_contexts:
            label = raw_label.casefold()
            normalized_label = normalize_directive_text(raw_label)
            is_total = bool(re.match(r"^(?:tổng|cộng|total)\b", label)) or (
                normalized_label in group_signatures
            )

            for match in cls.PERIOD_VALUE_PATTERN.finditer(context):
                normalized = cls.normalize_number(match.group("number"))
                if not normalized:
                    continue
                unit = cls._unit_for_number(
                    context, match.start("number"), match.end("number")
                )
                if not unit or unit == "%":
                    continue
                raw_year = (match.group("year") or "").strip()
                year = raw_year if re.fullmatch(r"(?:19|20)\d{2}", raw_year) else (
                    " ".join(match.group("period").casefold().split())
                )
                key = (year, unit)
                value = Decimal(normalized)
                values_by_period.setdefault(key, []).append(value)
                if is_total:
                    totals_by_period.setdefault(key, []).append(value)

        derived: set[str] = set()
        for key, values in values_by_period.items():
            totals = totals_by_period.get(key, [])
            for value in values:
                for total in totals:
                    if total == 0:
                        continue
                    percentage = (value * Decimal("100") / total).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )
                    normalized = cls.normalize_number(format(percentage, "f"))
                    if normalized:
                        derived.add(normalized)
        return derived

    @classmethod
    def validate(
        cls,
        answer: str,
        source_contexts: List[str],
        derived_values: Iterable[tuple[str, str]] = (),
    ) -> ValidationResult:
        answer_without_citations = re.sub(r"</?(?:ref|kb|web)\b[^>]*>", " ", answer, flags=re.IGNORECASE)
        answer_without_citations = re.sub(r"\[(?:c?\d+)\]", " ", answer_without_citations, flags=re.IGNORECASE)
        # Table identifiers, filing dates, and years are labels rather than
        # financial amounts; do not require them to carry a currency unit.
        answer_without_citations = cls.NON_FINANCIAL_IDENTIFIER_PATTERN.sub(
            " ", answer_without_citations
        )
        # Dates and citation page labels are identifiers, not financial values.
        # Leaving them in can bind a nearby currency unit to a year/page number.
        answer_without_citations = re.sub(
            r"(?i)\b(?:năm|year)\s+(?:19|20)\d{2}\b|\b(?:trang|page)\s+\d+\b",
            " ",
            answer_without_citations,
        )
        answer_number_strings = cls.extract_numbers(answer_without_citations)
        answer_entries = cls._number_entries(answer_without_citations)
        if not answer_entries:
            return ValidationResult(
                is_valid=True,
                details="Câu trả lời không chứa dữ liệu số cần xác thực.",
            )

        source_text = "\n".join(source_contexts)
        source_entries = cls._number_entries(source_text)
        source_numbers = {number for number, _ in source_entries}
        derived_differences = cls._derived_period_differences(source_contexts)
        derived_percentages = cls._derived_source_percentages(source_contexts)
        calculated_units: dict[str, set[str]] = {}
        for raw_value, raw_unit in derived_values:
            value = cls.normalize_number(str(raw_value))
            unit = cls._normalize_unit(raw_unit)
            if value and unit:
                calculated_units.setdefault(value, set()).add(unit)
        unmatched = [
            number for number, _ in answer_entries
            if number not in source_numbers
            and number not in derived_differences
            and number not in calculated_units
            and not (
                any(
                    answer_number == number and answer_unit == "%"
                    for answer_number, answer_unit in answer_entries
                )
                and number in derived_percentages
            )
        ]

        unit_mismatches = []
        for number, answer_unit in answer_entries:
            if answer_unit is None:
                continue
            matching_sources = [unit for source_number, unit in source_entries if source_number == number]
            if not matching_sources:
                matching_sources = list(derived_differences.get(number, set()))
            matching_sources.extend(calculated_units.get(number, set()))
            if answer_unit == "%" and number in derived_percentages:
                matching_sources.append("%")
            if answer_unit not in matching_sources:
                unit_mismatches.append(number)

        number_match = not unmatched
        unit_match = not unit_mismatches
        is_valid = number_match and unit_match
        answer_numbers = {number for number, _ in answer_entries}
        matched_numbers = []
        seen_source_numbers = set()
        for raw_number in cls.extract_numbers(source_text):
            normalized = cls.normalize_number(raw_number)
            if normalized in answer_numbers and normalized not in seen_source_numbers:
                matched_numbers.append(raw_number)
                seen_source_numbers.add(normalized)

        derived_matches = [
            number for number, _unit in answer_entries
            if number not in source_numbers
            and (
                number in derived_differences
                or number in calculated_units
                or (
                    number in derived_percentages
                    and any(
                        answer_number == number and answer_unit == "%"
                        for answer_number, answer_unit in answer_entries
                    )
                )
            )
        ]
        if is_valid and derived_matches:
            details = (
                "Các con số trong câu trả lời khớp với nguồn; chênh lệch được tính "
                "hoặc tỷ trọng được tính từ các giá trị cùng kỳ, cùng đơn vị trong nguồn."
            )
        elif is_valid:
            details = "Tất cả con số và đơn vị trong câu trả lời khớp với ngữ cảnh nguồn."
        else:
            problems = []
            if unmatched:
                problems.append(f"không tìm thấy các số {', '.join(unmatched)} trong nguồn")
            if unit_mismatches:
                problems.append(f"đơn vị không khớp với nguồn cho các số {', '.join(unit_mismatches)}")
            details = "Cảnh báo: " + "; ".join(problems) + "."

        return ValidationResult(
            is_valid=is_valid,
            extracted_number=", ".join(answer_number_strings),
            source_number=", ".join(matched_numbers) or None,
            unit_match=unit_match,
            details=details,
        )
