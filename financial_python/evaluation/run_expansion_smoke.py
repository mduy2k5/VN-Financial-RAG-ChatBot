"""Live regression and bounded multi-report behavior; not a verified holdout."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import requests

from run_bank_analysis_smoke import CASES, check_case

SERIES_CASES = [
    {"id": "fpt_three_years", "companies": ["FPT"], "years": [2023, 2024, 2025],
     "basis": "consolidated", "status": "complete", "growth_count": 2,
     "query": "So sánh tổng tài sản FPT năm 2023, 2024 và 2025 theo báo cáo hợp nhất"},
    {"id": "fpt_three_years_separate", "companies": ["FPT"], "years": [2023, 2024, 2025],
     "basis": "separate", "status": "complete", "growth_count": 2,
     "query": "So sánh tổng tài sản FPT năm 2023, 2024 và 2025 theo báo cáo riêng lẻ"},
    {"id": "mbb_profit_three_years", "companies": ["MBB"], "years": [2023, 2024, 2025],
     "basis": "consolidated", "status": "complete", "growth_count": 2,
     "query": "So sánh lợi nhuận sau thuế MBB năm 2023, 2024 và 2025 theo báo cáo hợp nhất"},
    {"id": "three_banks", "companies": ["VPB", "ACB", "MBB"], "years": [2025],
     "basis": "consolidated", "status": "complete", "growth_count": 0,
     "query": "So sánh tổng tài sản VPB, ACB và MBB năm 2025 theo báo cáo hợp nhất"},
    {"id": "vpb_missing_year", "companies": ["VPB"], "years": [2023, 2024, 2025],
     "basis": "consolidated", "status": "partial", "growth_count": 1,
     "query": "So sánh tổng tài sản VPB năm 2023, 2024 và 2025 theo báo cáo hợp nhất"},
    {"id": "budget_rejected", "companies": ["FPT"], "years": list(range(2010, 2026)),
     "basis": "consolidated", "status": "needs_clarification", "growth_count": 0,
     "query": "So sánh tổng tài sản FPT các năm " + ", ".join(map(str, range(2010, 2026))) + " theo báo cáo hợp nhất"},
]


def check_series(base_url: str, case: dict) -> dict:
    start = time.perf_counter()
    response = requests.post(base_url + "/api/v1/financial/answer", json={"query": case["query"]}, timeout=90)
    response.raise_for_status()
    data = response.json()
    errors = []
    parsed = data["parsed_query"]
    if set(parsed["company_codes"]) != set(case["companies"]) or parsed["fiscal_years"] != case["years"] or parsed["report_type"] != case["basis"]:
        errors.append("requested scope changed")
    if data["answer_status"] != case["status"]:
        errors.append(f"expected {case['status']}, got {data['answer_status']}")
    if not data["validation"]["is_valid"]:
        errors.append("source/calculation validation failed")
    growth = [item for item in data["calculations"] if item["formula_id"] == "growth_rate"]
    if len(growth) != case["growth_count"]:
        errors.append(f"expected {case['growth_count']} growth pairs, got {len(growth)}")
    if case["status"] == "needs_clarification":
        if data["citations"] or data["calculations"]:
            errors.append("budget guard performed work")
    else:
        for company in case["companies"]:
            if not any(citation["file_name"].startswith(company + "_") for citation in data["citations"]):
                errors.append(f"no evidence for {company}")
        expected_basis = "Hopnhat" if case["basis"] == "consolidated" else "Congtyme"
        if any(expected_basis not in citation["file_name"] for citation in data["citations"]):
            errors.append("citation outside requested reporting basis")
        for item in data["calculations"]:
            if not set(fact["year"] for fact in item["inputs"]).issubset(set(case["years"])):
                errors.append("derived an unrequested period")
    return {"id": case["id"], "query": case["query"], "passed": not errors, "errors": errors,
            "seconds": round(time.perf_counter() - start, 3), "response": data}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8092")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for _ in range(45):
        try:
            if requests.get(args.base_url + "/health/ready", timeout=2).status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(1)
    rows = [check_case(args.base_url, case) for case in CASES]
    rows.extend(check_series(args.base_url, case) for case in SERIES_CASES)
    output = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "holdout_verified": False,
              "passed": sum(row["passed"] for row in rows), "total": len(rows), "cases": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": output["passed"], "total": output["total"],
                      "failures": [{"id": row["id"], "errors": row["errors"]} for row in rows if not row["passed"]]}, ensure_ascii=False))
    return 0 if output["passed"] == output["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
