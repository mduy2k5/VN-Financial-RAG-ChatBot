"""Check analytical questions against a running API and save reviewable answers.

The VPB expected values refer to the audited consolidated 2025 OCR report.
Cross-bank cases check scope and validated formulas, not an independent audit.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import requests


CASES = [
    {
        "id": "credit_quality", "company": "VPB",
        "query": "Đánh giá chất lượng tín dụng VPB năm 2025 theo báo cáo hợp nhất: tỷ lệ nợ xấu giảm có đi cùng giảm nợ có khả năng mất vốn không?",
        "formulas": {"nonperforming_loans_ratio_percent": {"4.22", "3.46"}},
        "text": ["chưa thể kết luận chất lượng tín dụng cải thiện đồng đều", "49,61%"],
    },
    {
        "id": "average_returns", "company": "VPB",
        "query": "Đánh giá khả năng sinh lời VPB năm 2025 theo báo cáo hợp nhất, tính ROA và ROE trên số dư bình quân đầu và cuối năm.",
        "formulas": {"return_on_average_assets_after_tax_percent": {"2.23"}, "return_on_average_equity_after_tax_percent": {"14.87"}},
    },
    {
        "id": "cost_efficiency", "company": "VPB",
        "query": "Đánh giá hiệu quả hoạt động VPB năm 2025 theo báo cáo hợp nhất, tính CIR.",
        "formulas": {"cost_to_income_percent": {"23.03", "24.96"}},
        "text": ["CIR tăng"], "absent_text": ["Giảm: TỔNG CHI PHÍ HOẠT ĐỘNG"],
    },
    {
        "id": "loan_deposit_growth", "company": "VPB",
        "query": "Đánh giá tăng trưởng tín dụng và tiền gửi khách hàng VPB năm 2025 theo báo cáo hợp nhất.",
        "text": ["35,27%", "29,32%", "tăng trưởng nhanh hơn tiền gửi khách hàng"],
        "absent_text": ["Không đưa vào phép tính"],
    },
    {
        "id": "liquidity_limits", "company": "VPB", "status": "partial",
        "query": "Đánh giá thanh khoản VPB năm 2025 theo báo cáo hợp nhất, có đủ cơ sở kết luận an toàn không?",
        "missing": ["đánh giá thanh khoản theo kỳ hạn"],
    },
    {
        "id": "capital_buffer", "company": "VPB", "status": "partial",
        "query": "Đánh giá an toàn vốn VPB năm 2025 theo báo cáo hợp nhất: tài sản tăng nhanh hơn vốn chủ sở hữu có làm giảm đệm vốn không?",
        "text": ["đệm vốn kế toán trên tài sản thấp hơn"], "missing": ["CAR"],
    },
    {
        "id": "original_health", "company": "VPB", "status": "partial",
        "query": "Đánh giá sức khỏe tài chính VPB năm 2025 theo báo cáo hợp nhất",
        "text": ["196.097.219"], "absent_text": ["DOANH THU THUẦN", "Chưa truy xuất được số liệu"],
        "formulas": {"nonperforming_loans_ratio_percent": {"4.22", "3.46"}},
    },
    *[{
        "id": f"{company.casefold()}_average_returns", "company": company,
        "status": "partial" if company == "ACB" else "complete",
        "query": f"Đánh giá khả năng sinh lời {company} năm 2025 theo báo cáo hợp nhất, tính ROA và ROE trên số dư bình quân đầu và cuối năm.",
        "formula_presence": ["return_on_average_assets_after_tax_percent", "return_on_average_equity_after_tax_percent"],
    } for company in ("ACB", "MBB")],
    {
        "id": "unavailable_year", "company": "VPB", "year": 2030, "status": "insufficient_evidence",
        "query": "Đánh giá sức khỏe tài chính VPB năm 2030 theo báo cáo hợp nhất",
        "no_evidence": True,
    },
]


def check_case(base_url: str, case: dict) -> dict:
    start = time.perf_counter()
    response = requests.post(base_url + "/api/v1/financial/answer", json={"query": case["query"]}, timeout=90)
    response.raise_for_status()
    data = response.json()
    errors = []
    validation = data["validation"]
    if not all(validation.get(key) for key in ("is_valid", "citation_valid", "unit_match")):
        errors.append("numeric, citation or unit validation failed")
    parsed = data["parsed_query"]
    if parsed["company_codes"] != [case["company"]] or parsed["fiscal_years"] != [case.get("year", 2025)] or parsed["report_type"] != "consolidated":
        errors.append("company, year or reporting basis changed")
    if any(case["company"] not in row["file_name"] or "Hopnhat" not in row["file_name"] for row in data["citations"]):
        errors.append("citations outside requested company/basis")
    for formula, values in case.get("formulas", {}).items():
        actual = {row["value"] for row in data["calculations"] if row["formula_id"] == formula}
        if actual != values:
            errors.append(f"{formula}: expected {sorted(values)}, got {sorted(actual)}")
    for formula in case.get("formula_presence", []):
        if not any(row["formula_id"] == formula for row in data["calculations"]):
            errors.append(f"missing {formula}")
    for text in case.get("text", []):
        if text not in data["answer"]:
            errors.append(f"missing expected text: {text}")
    for text in case.get("absent_text", []):
        if text in data["answer"]:
            errors.append(f"unexpected text: {text}")
    if data.get("answer_status") != case.get("status", "complete"):
        errors.append("unexpected completeness status")
    missing = (data.get("analysis_coverage") or {}).get("missing_components", [])
    for text in case.get("missing", []):
        if not any(text in item for item in missing):
            errors.append(f"missing coverage limitation: {text}")
    if case.get("no_evidence") and (data["citations"] or data["calculations"]):
        errors.append("unavailable year borrowed evidence from another year")
    return {"id": case["id"], "query": case["query"], "seconds": round(time.perf_counter() - start, 3), "passed": not errors, "errors": errors, "response": data}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8092")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results/bank_analysis_smoke_2026-10-01.json")
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda case: check_case(args.base_url.rstrip("/"), case), CASES))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "executed_at": datetime.now(timezone.utc).isoformat(), "base_url": args.base_url,
        "passed": sum(row["passed"] for row in results), "total": len(results), "cases": results,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in results:
        print(f"{'PASS' if row['passed'] else 'FAIL'} {row['id']} ({row['seconds']}s): {'; '.join(row['errors'])}")
    print(f"Saved {args.output}")
    return int(any(not row["passed"] for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
