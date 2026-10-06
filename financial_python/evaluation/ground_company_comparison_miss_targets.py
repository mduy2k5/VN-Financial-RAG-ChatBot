"""Extract independent statement-row candidates for the 120 comparison misses.

The review sheet's target values came from the same parser being evaluated.
This script re-anchors those targets to statement totals (Vietnamese financial
statement row codes 411, 130, and 100 where applicable). It writes a separate
candidate artifact and never edits or verifies the original review sheet.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


FINANCIAL_PYTHON = Path(__file__).resolve().parents[1]
DATA_DIR = FINANCIAL_PYTHON.parent / "dataset" / "ocr_annual_financials"
DEFAULT_INPUT = FINANCIAL_PYTHON / "evaluation" / "review" / "company_comparison_miss_review_v2.csv"
DEFAULT_JSON = FINANCIAL_PYTHON / "evaluation" / "review" / "company_comparison_miss_grounded_targets_v1.json"
DEFAULT_CSV = FINANCIAL_PYTHON / "evaluation" / "review" / "company_comparison_miss_grounded_targets_v1.csv"

PAGE_RE = re.compile(r"(?im)^\s*=+\s*PAGE\s+(\d+)\s*=+\s*$")
ROW_RE = re.compile(r"<tr\b[^>]*>.*?</tr>", re.IGNORECASE | re.DOTALL)
CELL_RE = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]\s*>", re.IGNORECASE | re.DOTALL)
AMOUNT_RE = re.compile(r"(?:\(?-?\d[\d.,]*\)?|-)")

CSV_FIELDS = [
    "sample_id",
    "case_id",
    "company_code",
    "fiscal_year",
    "report_type",
    "expected_metric",
    "parser_expected_value",
    "grounded_candidate_value",
    "candidate_status",
    "candidate_metric_row_label",
    "candidate_account_code",
    "candidate_source_file",
    "candidate_source_page",
    "candidate_row_text",
    "candidate_source_sha256",
    "top5_has_grounded_value",
    "top5_grounded_value_match_count",
    "original_review_status",
]


def _normalize_label(value: str) -> str:
    value = (value or "").casefold().replace("đ", "d")
    value = re.sub(r"^\s*(?:\d+|[ivx]+|[a-z])\s*[.):-]\s*", "", value)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value)).strip()


def _cell_text(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def _parse_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    pages = list(PAGE_RE.finditer(text))
    result: list[dict[str, Any]] = []
    for page_index, marker in enumerate(pages):
        page_number = int(marker.group(1))
        page_end = pages[page_index + 1].start() if page_index + 1 < len(pages) else len(text)
        page_text = text[marker.end():page_end]
        for raw_row in ROW_RE.findall(page_text):
            cells = [_cell_text(cell) for cell in CELL_RE.findall(raw_row)]
            labels = [
                cell for cell in cells
                if re.search(r"[A-Za-zÀ-ỹ]{2,}", cell)
                and not re.fullmatch(r"[A-Za-z]{1,3}[\w.-]*", cell)
            ]
            codes = [cell.strip() for cell in cells if re.fullmatch(r"\d{2,3}", cell.strip())]
            numeric_cells = [
                cell for cell in cells
                if AMOUNT_RE.fullmatch(cell.strip()) and cell.strip() not in codes
            ]
            if not cells:
                continue
            result.append({
                "page": page_number,
                "cells": cells,
                "labels": labels,
                "normalized_labels": [_normalize_label(label) for label in labels],
                "codes": codes,
                "amounts": numeric_cells[-2:],
                "row_text": " | ".join(cells),
            })
    return result


def _document_year(source_file: str) -> int | None:
    match = re.search(r"(?<!\d)(20\d{2})(?!\d)", source_file)
    return int(match.group(1)) if match else None


def _row_matches_metric(row: dict[str, Any], metric: str) -> tuple[bool, int]:
    labels = row["normalized_labels"]
    codes = row["codes"]
    if metric == "VỐN ĐIỀU LỆ":
        acceptable = {
            "von dieu le",
            "von gop cua chu so huu",
            "von dau tu cua chu so huu",
            "von co phan",
            "von co phan da phat hanh",
        }
        direct = any(label in acceptable for label in labels)
        if any("lien doanh" in label for label in labels):
            direct = False
        return direct or ("411" in codes and not any("lien doanh" in label for label in labels)), 100 if "411" in codes and direct else 80 if direct else 0

    if metric == "CÁC KHOẢN PHẢI THU NGẮN HẠN":
        direct = any(label in {"cac khoan phai thu ngan han", "cac khoan phai thu"} for label in labels)
        # Code 130 is not stable across bank-specific forms; in some reports it
        # labels other current assets. Require the aggregate receivables label.
        return direct, 100 if direct and "130" in codes else 80 if direct else 0

    if metric == "CHI PHÍ THUẾ TNDN":
        excluded = ("hien hanh", "hoan lai", "uoc tinh", "da nop", "phai nop", "khong duoc tru")
        direct_labels = {
            "chi phi thue thu nhap doanh nghiep",
            "tong chi phi thue thu nhap doanh nghiep",
            "chi phi thue tndn",
            "tong chi phi thue tndn",
            "chi phi thue thu nhap doanh nghiep tndn",
        }
        direct = any(label in direct_labels for label in labels)
        code_100_total = "100" in codes and any(
            ("chi phi thue" in label or "chi phi thue" in label.replace("thue ", "thue "))
            and "doanh nghiep" in label
            and not any(term in label for term in excluded)
            for label in labels
        )
        return direct or code_100_total, 100 if code_100_total else 80 if direct else 0

    if metric == "CHI PHÍ LÃI VAY":
        direct = any(
            ("chi phi lai vay" in label or "chi phi lai va cac chi phi tuong tu" in label)
            and "khong bao gom" not in label
            for label in labels
        )
        return direct, 100 if direct and "23" in codes else 80 if direct else 0

    if metric == "LƯU CHUYỂN TIỀN TỪ HOẠT ĐỘNG KINH DOANH":
        direct = any(
            "luu chuyen" in label and "kinh doanh" in label
            and not any(term in label for term in (
                "truoc thay doi von luu dong",
                "truoc bien dong von luu dong",
                "truoc nhung thay doi",
                "truoc thay doi tai san",
            ))
            for label in labels
        )
        return direct, 100 if direct and "20" in codes else 80 if direct else 0

    if metric == "LƯU CHUYỂN TIỀN TỪ HOẠT ĐỘNG TÀI CHÍNH":
        direct = any("luu chuyen" in label and "tai chinh" in label for label in labels)
        return direct, 100 if direct and "40" in codes else 80 if direct else 0

    if metric == "LỢI NHUẬN CHƯA PHÂN PHỐI":
        direct = any(
            "loi nhuan sau thue chua phan phoi" in label
            and not any(term in label for term in ("den cuoi nam truoc", "den cuoi ky truoc", "nam nay", "ky nay", "luy ke"))
            for label in labels
        )
        return direct, 100 if direct and ("421" in codes or "417" in codes) else 80 if direct else 0

    if metric == "LỢI NHUẬN SAU THUẾ":
        direct = any(
            "loi nhuan sau thue" in label
            and not any(term in label for term in ("cong ty me", "co dong khong kiem soat", "co dong khong kiem soat"))
            for label in labels
        )
        bank_after_tax_total = "200" in codes and any(
            "loi nhuan ke toan sau" in label for label in labels
        )
        return direct or bank_after_tax_total, 100 if bank_after_tax_total or (direct and "60" in codes) else 80 if direct else 0

    if metric == "TIỀN VÀ TƯƠNG ĐƯƠNG TIỀN":
        direct = any(
            label in {"tien va tuong duong tien", "tien va cac khoan tuong duong tien"}
            for label in labels
        )
        return direct, 100 if direct and "110" in codes else 80 if direct else 0

    if metric in {"TIỀN VÀ TƯƠNG ĐƯƠNG TIỀN CUỐI KỲ", "TIỀN VÀ TƯƠNG ĐƯƠNG TIỀN ĐẦU KỲ"}:
        period_terms = ("cuoi ky", "cuoi nam", "cuoi ky bao cao") if metric.endswith("CUỐI KỲ") else ("dau ky", "dau nam", "dau ky bao cao")
        direct = any(
            "tien" in label and "tuong duong tien" in label and any(term in label for term in period_terms)
            for label in labels
        )
        expected_code = "70" if metric.endswith("CUỐI KỲ") else "60"
        return direct, 100 if direct and expected_code in codes else 80 if direct else 0

    if metric == "TỔNG LỢI NHUẬN KẾ TOÁN TRƯỚC THUẾ":
        direct = any(
            "loi nhuan" in label and "truoc thue" in label
            and ("ke toan" in label or "tong loi nhuan" in label)
            and "theo bo phan" not in label
            for label in labels
        )
        return direct, 100 if direct and ("50" in codes or "90" in codes) else 80 if direct else 0

    return False, 0


def _target_period_value(row: dict[str, Any], doc_year: int | None, fiscal_year: int) -> str | None:
    values = row["amounts"]
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    if doc_year == fiscal_year:
        return values[0]
    if doc_year == fiscal_year + 1:
        return values[1]
    return None


def _canonical_amount(value: str | None) -> tuple[int, str] | None:
    if not value:
        return None
    value = value.strip()
    negative = value.startswith("(") and value.endswith(")") or value.startswith("-")
    digits = re.sub(r"\D", "", value)
    if not digits:
        return None
    return (-1 if negative else 1, digits)


def _canonical_magnitude(value: str | None) -> str | None:
    canonical = _canonical_amount(value)
    return canonical[1] if canonical else None


def _as_signed_int(value: str | None) -> int | None:
    canonical = _canonical_amount(value)
    if not canonical:
        return None
    sign, digits = canonical
    return sign * int(digits)


def _format_signed_int(value: int) -> str:
    text = f"{abs(value):,}".replace(",", ".")
    return f"({text})" if value < 0 else text


def _tax_component_derivation(
    rows: list[dict[str, Any]], doc_year: int | None, fiscal_year: int
) -> dict[str, Any] | None:
    by_page: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        label = " ".join(row["normalized_labels"])
        code = next((code for code in row["codes"] if code in {"51", "52"}), None)
        if not code or "thue" not in label or not ("doanh nghiep" in label or "tndn" in label):
            continue
        if "tai san" in label or "phai tra" in label or "da nop" in label:
            continue
        if code == "51" and not ("chi phi" in label and ("hien hanh" in label or "tndn" in label)):
            continue
        if code == "52" and not any(term in label for term in ("hoan lai", "hoan lai", "tndn")):
            continue
        value = _target_period_value(row, doc_year, fiscal_year)
        if value is None:
            continue
        by_page.setdefault(row["page"], {})[code] = {
            "label": " / ".join(row["labels"]),
            "code": code,
            "value": value,
            "row_text": row["row_text"],
        }

    complete = [
        (page, items)
        for page, items in by_page.items()
        if "51" in items and "52" in items
    ]
    if not complete:
        return None
    page, items = complete[0]
    current = _as_signed_int(items["51"]["value"])
    deferred = _as_signed_int(items["52"]["value"])
    if current is None or deferred is None:
        return None
    net = current + deferred
    return {
        "page": page,
        "current_tax": items["51"],
        "deferred_tax": items["52"],
        "net_signed_statement_amount": _format_signed_int(net),
        "net_expense_magnitude": f"{abs(net):,}".replace(",", "."),
        "arithmetic": f"{items['51']['value']} + {items['52']['value']} = {_format_signed_int(net)}",
    }


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _side_candidates(
    target: dict[str, Any], source_file: str, parser_value: str, original_page: int | None,
    top5: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    doc_year = _document_year(source_file)
    candidates: list[dict[str, Any]] = []
    for row in rows:
        match, priority = _row_matches_metric(row, target["canonical_metric"])
        if not match:
            continue
        value = _target_period_value(row, doc_year, int(target["fiscal_year"]))
        if value is None:
            continue
        candidates.append({
            "priority": priority,
            "page_priority": 1 if row["page"] == original_page else 0,
            "page": row["page"],
            "label": " / ".join(row["labels"]),
            "code": ",".join(row["codes"]),
            "value": value,
            "row_text": row["row_text"],
            "is_parser_value": _canonical_amount(value) == _canonical_amount(parser_value),
        })

    if not candidates:
        status = "no_direct_total_row_found"
        selected = None
    else:
        highest = max((candidate["priority"], candidate["page_priority"]) for candidate in candidates)
        best = [candidate for candidate in candidates if (candidate["priority"], candidate["page_priority"]) == highest]
        distinct_values = {_canonical_amount(candidate["value"]) for candidate in best}
        if len(distinct_values) > 1:
            status = "conflicting_direct_rows_need_review"
            selected = None
        else:
            selected = best[0]
            if selected["is_parser_value"]:
                status = "direct_total_matches_parser_value"
            elif _canonical_magnitude(selected["value"]) == _canonical_magnitude(parser_value):
                status = "direct_total_matches_parser_magnitude_sign_diff"
            else:
                status = "direct_total_corrects_parser_value"

    derivation = (
        _tax_component_derivation(rows, doc_year, int(target["fiscal_year"]))
        if target["canonical_metric"] == "CHI PHÍ THUẾ TNDN"
        else None
    )
    if selected is None and derivation:
        selected = {
            "priority": 0,
            "page_priority": 1 if derivation["page"] == original_page else 0,
            "page": derivation["page"],
            "label": "Chi phí thuế hiện hành + thuế hoãn lại (tính từ dòng 51 + 52)",
            "code": "51+52",
            "value": derivation["net_signed_statement_amount"],
            "row_text": derivation["arithmetic"],
            "is_parser_value": _canonical_amount(derivation["net_signed_statement_amount"]) == _canonical_amount(parser_value),
        }
        if selected["is_parser_value"]:
            status = "derived_tax_total_matches_parser_signed_value"
        elif _canonical_magnitude(selected["value"]) == _canonical_magnitude(parser_value):
            status = "derived_tax_total_matches_parser_magnitude_sign_diff"
        else:
            status = "derived_tax_total_corrects_parser_value"

    grounded_value = selected["value"] if selected else None
    company = target.get("company_code")
    metric = target.get("canonical_metric")
    year = int(target["fiscal_year"])
    report_type = target.get("report_type")
    matching_top5_signed = [
        item for item in top5
        if item.get("company_code") == company
        and item.get("canonical_metric") == metric
        and item.get("report_type") == report_type
        and year in {item.get("fiscal_year"), item.get("comparative_fiscal_year")}
        and _canonical_amount(str((item.get("period_values") or {}).get(str(year), ""))) == _canonical_amount(grounded_value)
    ] if grounded_value else []
    matching_top5_magnitude = [
        item for item in top5
        if item.get("company_code") == company
        and item.get("canonical_metric") == metric
        and item.get("report_type") == report_type
        and year in {item.get("fiscal_year"), item.get("comparative_fiscal_year")}
        and _canonical_magnitude(str((item.get("period_values") or {}).get(str(year), ""))) == _canonical_magnitude(grounded_value)
    ] if grounded_value else []
    selected_page = selected.get("page") if selected else None
    matching_top5_source_page = [
        item for item in top5
        if item.get("company_code") == company
        and item.get("canonical_metric") == metric
        and item.get("report_type") == report_type
        and item.get("source_file") == source_file
        and item.get("page") == selected_page
        and year in {item.get("fiscal_year"), item.get("comparative_fiscal_year")}
        and _canonical_amount(str((item.get("period_values") or {}).get(str(year), ""))) == _canonical_amount(grounded_value)
    ] if grounded_value else []
    return {
        "company_code": company,
        "fiscal_year": year,
        "report_type": report_type,
        "expected_metric": metric,
        "parser_expected_value": parser_value,
        "grounded_candidate_value": grounded_value,
        "candidate_status": status,
        "candidate_rows_with_period_values": candidates,
        "selected_candidate": selected,
        "tax_component_derivation": derivation,
        "top5_grounded_value_match_count": len(matching_top5_signed),
        "top5_has_grounded_value": bool(matching_top5_signed),
        "top5_has_grounded_magnitude": bool(matching_top5_magnitude),
        "top5_grounded_magnitude_match_count": len(matching_top5_magnitude),
        "top5_grounded_source_page_match_count": len(matching_top5_source_page),
        "top5_has_grounded_source_page_value": bool(matching_top5_source_page),
        "review_status_in_original_sheet": "pending",
    }


def ground(input_csv: Path = DEFAULT_INPUT, output_json: Path = DEFAULT_JSON, output_csv: Path = DEFAULT_CSV) -> dict[str, Any]:
    cache: dict[str, list[dict[str, Any]]] = {}
    sha_cache: dict[str, str] = {}
    sides: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []

    with input_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            targets = json.loads(row["parser_expected_evidence"])
            source_candidates = json.loads(row["raw_source_evidence_candidates"])
            source_by_file = {item["source_file"]: item for item in source_candidates}
            top5 = json.loads(row["retrieved_top5_evidence"])
            case_sides: list[dict[str, Any]] = []
            for target in targets:
                source_file = target["source_file"]
                source_path = DATA_DIR / source_file
                cache.setdefault(source_file, _parse_rows(source_path))
                sha_cache.setdefault(source_file, _source_sha(source_path))
                source_side = _side_candidates(
                    target,
                    source_file,
                    str(target.get("value") or ""),
                    int(target["page"]) if target.get("page") is not None else None,
                    top5,
                    cache[source_file],
                )
                selected = source_side["selected_candidate"] or {}
                item = {
                    "sample_id": row["sample_id"],
                    "case_id": row["case_id"],
                    **source_side,
                    "candidate_source_file": source_file,
                    "candidate_source_page": selected.get("page"),
                    "candidate_source_sha256": sha_cache[source_file],
                    "original_source_page": source_by_file.get(source_file, {}).get("page"),
                }
                sides.append(item)
                case_sides.append(item)

            corrected_complete = len(case_sides) == 2 and all(item["grounded_candidate_value"] for item in case_sides)
            cases.append({
                "sample_id": row["sample_id"],
                "case_id": row["case_id"],
                "expected_metric": row["expected_metric"],
                "corrected_targets_available_both_sides": corrected_complete,
                "top5_correct_sides": sum(bool(item["top5_has_grounded_value"]) for item in case_sides),
                "top5_has_both_corrected_values": corrected_complete and all(item["top5_has_grounded_value"] for item in case_sides),
                "top5_has_both_corrected_magnitudes": corrected_complete and all(item["top5_has_grounded_magnitude"] for item in case_sides),
                "candidate_statuses": [item["candidate_status"] for item in case_sides],
                "original_review_status": row["review_status"],
            })

    status_counts = Counter(item["candidate_status"] for item in sides)
    summary = {
        "schema_version": 1,
        "artifact_type": "source_row_grounded_candidates_not_final_human_gold",
        "input_csv_sha256": hashlib.sha256(input_csv.read_bytes()).hexdigest(),
        "case_count": len(cases),
        "target_side_count": len(sides),
        "candidate_status_counts": dict(status_counts),
        "cases_with_source_total_candidates_both_sides": sum(item["corrected_targets_available_both_sides"] for item in cases),
        "cases_with_both_corrected_values_in_original_top5": sum(item["top5_has_both_corrected_values"] for item in cases),
        "cases_with_both_corrected_magnitudes_in_original_top5": sum(item["top5_has_both_corrected_magnitudes"] for item in cases),
        "review_status_counts_unchanged": dict(Counter(item["review_status_in_original_sheet"] for item in sides)),
        "method": (
            "Scan each original source TXT for direct financial statement totals: capital row labels/code 411, "
            "short-term receivable total/code 130, and income-tax expense total/code 100 or explicit total labels. "
            "For two-period rows, select current or comparative amount by matching the target fiscal year to the "
            "year in the source filename. Every selection retains source file, page, row text, and SHA-256."
        ),
        "interpretation_limit": (
            "These are source-row-based correction candidates, not independently signed-off labels. Rows with no "
            "clear aggregate line, conflicting rows, report-period ambiguity, or missing unit context still require review. "
            "The original review sheet remains pending and must not be replaced by this artifact."
        ),
        "cases": cases,
        "target_sides": sides,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with output_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for item in sides:
            selected = item.get("selected_candidate") or {}
            writer.writerow({
                "sample_id": item["sample_id"],
                "case_id": item["case_id"],
                "company_code": item["company_code"],
                "fiscal_year": item["fiscal_year"],
                "report_type": item["report_type"],
                "expected_metric": item["expected_metric"],
                "parser_expected_value": item["parser_expected_value"],
                "grounded_candidate_value": item["grounded_candidate_value"] or "",
                "candidate_status": item["candidate_status"],
                "candidate_metric_row_label": selected.get("label", ""),
                "candidate_account_code": selected.get("code", ""),
                "candidate_source_file": item["candidate_source_file"],
                "candidate_source_page": item["candidate_source_page"] or "",
                "candidate_row_text": selected.get("row_text", ""),
                "candidate_source_sha256": item["candidate_source_sha256"],
                "top5_has_grounded_value": item["top5_has_grounded_value"],
                "top5_grounded_value_match_count": item["top5_grounded_value_match_count"],
                "original_review_status": item["review_status_in_original_sheet"],
            })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--json", dest="output_json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv", dest="output_csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    result = ground(args.input, args.output_json, args.output_csv)
    print(json.dumps({key: value for key, value in result.items() if key not in {"cases", "target_sides"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
