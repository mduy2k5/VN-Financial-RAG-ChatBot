"""Create 120 dev + 60 held-out review drafts, never machine-promote them to gold."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from financial_rag.schemas import FinancialChunk, ChunkType
from financial_rag.structure.semantics import table_scope, period_supported
from financial_rag.structure.source_groups import source_item_label, source_root_metric_id
from financial_rag.validation.numeric import NumericValidator

DEV = ("FPT", "BMP", "ACB", "MBB", "VPB", "ACBS")
HOLDOUT = ("ABB", "VIC", "DTL", "SEA", "TNT", "YEG")


def build(database):
    cases, review_rows = [], []
    for split, companies, count in (("dev", DEV, 20), ("holdout", HOLDOUT, 10)):
        for company in companies:
            documents = list(database.financial_documents.find({"company_code": company,
                "fiscal_quarter": None}).sort([("fiscal_year", -1), ("report_type", 1)]))
            if not documents:
                raise ValueError(f"No annual source for {company}; quota must not be filled with invented data")
            preferred = [doc for doc in documents if doc.get("report_type") == "consolidated"]
            source = (preferred or documents)[0]
            year, basis = source["fiscal_year"], source.get("report_type")
            source_hash = hashlib.sha256(source["raw_content"].encode()).hexdigest()
            rows = [FinancialChunk.model_validate(item["payload"]) for item in
                    database.financial_chunks.find({"logical_key": source["_id"]}).sort("ordinal", 1)]
            candidates = [row for row in rows if row.chunk_type == ChunkType.FINANCIAL_ITEM and row.page
                          and table_scope(row) not in {"maturity", "risk_dimension", "segment"}
                          and period_supported(row, year) is not False and source_item_label(row)
                          and NumericValidator.normalize_number(str((row.metadata.get("period_values") or {}).get(str(year), "")))]
            roots = [row for row in candidates if source_root_metric_id(row)]
            candidates = roots + [row for row in candidates if row not in roots]
            candidates = list({source_item_label(row).casefold(): row for row in reversed(candidates)}.values())[::-1]
            if not candidates:
                raise ValueError(f"No numeric source rows for {company}")
            for index in range(count):
                row = candidates[index % len(candidates)]
                label = source_item_label(row)
                period_values = row.metadata.get("period_values") or {}
                periods = sorted(int(value) for value in period_values if str(value).isdigit()
                                 and NumericValidator.normalize_number(str(period_values[value])))
                previous = max((period for period in periods if period < year), default=None)
                mode = index % 10
                category = ("lookup", "lookup", "period_comparison", "growth", "composition",
                            "contribution", "ratio", "analysis", "causal", "missing_evidence")[mode]
                question = {
                    "lookup": f"{label} của {company} năm {year} là bao nhiêu? Cho biết đơn vị và nguồn.",
                    "period_comparison": f"So sánh {label} của {company} năm {previous or year - 1} và {year}.",
                    "growth": f"{label} của {company} thay đổi bao nhiêu từ năm {previous or year - 1} đến {year}?",
                    "composition": f"Cơ cấu {label} của {company} năm {year}; đối chiếu tổng với các thành phần.",
                    "contribution": f"Khoản nào đóng góp vào biến động {label} của {company} giữa năm {previous or year - 1} và {year}?",
                    "ratio": f"Tỷ lệ {'nợ phải trả' if index < 10 else 'vốn chủ sở hữu'} trên tổng tài sản {company} năm {year}.",
                    "analysis": f"Phân tích {label} của {company} năm {year}, nêu phần đủ và thiếu bằng chứng.",
                    "causal": f"Vì sao {label} của {company} biến động trong năm {year}? Chỉ dùng nguyên nhân báo cáo nêu trực tiếp.",
                    "missing_evidence": f"{label} của {company} năm {2099 - index} là bao nhiêu?",
                }[category]
                if basis:
                    question += " Theo báo cáo " + ("hợp nhất." if basis == "consolidated" else "riêng lẻ.")
                years = [2099 - index] if category == "missing_evidence" else (
                    [previous or year - 1, year] if category in {"period_comparison", "growth", "contribution"} else [year])
                binding = {"file_name": source["source_file"], "page": row.page, "source_sha256": source_hash}
                raw_value = str(period_values.get(str(year), ""))
                facts = [{"label": label, "year": year, "value": raw_value,
                          "unit": str(row.unit or ""), "sources": [binding]}]
                if previous and category in {"period_comparison", "growth", "contribution"}:
                    facts.insert(0, {**facts[0], "year": previous, "value": str(period_values[str(previous)])})
                case = {"id": f"expansion_{split}_{company.lower()}_{index:02d}", "split": split,
                        "category": category, "review_status": "draft", "author_id": "machine_draft",
                        "query": question, "answerable": category != "missing_evidence",
                        "expected_scope": {"company_codes": [company], "fiscal_years": years, "report_type": basis},
                        "expected_facts": facts if category != "missing_evidence" else [],
                        "expected_calculations": [],
                        "evidence_groups": [{"id": "candidate_source_row", "sources": [binding]}]
                                           if category != "missing_evidence" else [],
                        "reviewer_notes": "Machine draft only. Review scope, original row/column, values and applicability; replace incomplete draft labels before scoring. Composition/contribution/ratio/causal tasks need full operand/evidence labels, not just this candidate row."}
                if category == "missing_evidence":
                    case["abstention_markers"] = ["chưa có báo cáo", "không đủ bằng chứng", "chưa đủ bằng chứng", "không tìm thấy"]
                cases.append(case)
                review_rows.append({"case_id": case["id"], "source": binding, "chunk_id": row.chunk_id,
                                    "source_excerpt": row.content, "table_title": row.table_title,
                                    "raw_cells": row.metadata.get("source_cells"), "headers": row.metadata.get("table_headers"),
                                    "rubric": {"scope": None, "row_column": None, "arithmetic": None,
                                               "causal_support": None, "interpretation_1_to_5": None},
                                    "reviewer_id": None, "reviewed_at": None})
    assert len(cases) == 180
    return cases, {"case_count": len(cases), "verified_count": 0,
                   "production_accuracy_gate_passed": False,
                   "by_split": dict(Counter(case["split"] for case in cases)),
                   "by_category": dict(Counter(case["category"] for case in cases)),
                   "reasons": ["machine_drafts_need_independent_source_review",
                               "task_operand_labels_incomplete",
                               "corpus_has_only_one_securities_issuer_no_isolated_sector_holdout"],
                   "review_rows": review_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from pymongo import MongoClient
    with MongoClient(os.environ["MONGODB_URI"]) as client:
        cases, pack = build(client[os.getenv("FINANCIAL_MONGO_DATABASE", "financial_rag")])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "expansion_180_drafts.jsonl").write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n", encoding="utf-8")
    (args.output_dir / "expansion_180_review_pack.json").write_text(
        json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: pack[key] for key in ("case_count", "verified_count", "by_split", "reasons")}))


if __name__ == "__main__":
    main()
