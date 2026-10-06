"""Local service smoke: use existing pinned sources; never ingest or replace reports."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8092")
    parser.add_argument("--notebook-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.url.rstrip("/") + "/api/v1/notebooks/" + args.notebook_id
    session = requests.Session()
    notebook = session.get(base, timeout=30).json()
    assert notebook["sources"], "Attach a source before running the smoke"
    evidence = []

    def ask(query, expected_status=200, **values):
        started = time.perf_counter()
        response = session.post(base + "/answer", json={"query": query,
            "notebook_revision": notebook["revision"], **values}, timeout=180)
        payload = response.json()
        evidence.append({"query": query, "http_status": response.status_code,
                         "elapsed_seconds": round(time.perf_counter() - started, 3), "response": payload})
        assert response.status_code == expected_status, payload
        return payload

    first = ask("Tổng tài sản FPT năm 2025 theo báo cáo hợp nhất là bao nhiêu?")
    assert first["citations"] and first["answer_status"] == "complete"
    prior = ask("Còn năm trước?", conversation_id=first["conversation_id"],
                conversation_revision=first["conversation_revision"])
    assert "2024" in prior["resolved_query"] and prior["citations"]
    missing = ask("Tổng tài sản VPB năm 2025 theo báo cáo hợp nhất là bao nhiêu?")
    assert missing["answer_status"] == "insufficient_evidence" and not missing["citations"]
    ask("Đọc nguồn", expected_status=422, source_ids=[])
    ask("Đọc nguồn", expected_status=404, source_ids=["outside-notebook"])
    summary = ask("Tóm tắt tài liệu", mode="summary")
    assert summary["reader_mode"] == "source_extract" and summary["citations"]
    assert summary["context_budget"]["estimated_tokens"] <= summary["context_budget"]["budget_tokens"]
    coverage = summary["evidence_coverage"]
    if summary["context_budget"]["omitted"]:
        assert summary["answer_status"] == "partial"
    source = first["citations"][0]
    content = session.get(base + f"/sources/{source['notebook_source_id']}/content", params={
        "run_id": first["run_id"], "block_id": source["block_id"]}, timeout=30).json()
    assert source["version_id"] == content["source"]["active_version_id"]
    assert first["validation"]["source_number"] in content["text"]
    note_response = session.post(base + "/notes", json={"title": "Smoke: câu trả lời có bằng chứng",
                                                       "run_id": first["run_id"]}, timeout=30)
    assert note_response.status_code == 201
    output = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
        "notebook_id": notebook["id"], "production_documents_modified": False,
        "scope": "functional smoke; not human-verified answer quality benchmark",
        "summary_coverage": coverage, "checks": ["financial_lookup", "followup_previous_year",
            "missing_company_abstention", "empty_selection_rejected", "foreign_source_rejected",
            "extractive_summary_budget", "citation_version_and_span", "save_answer_note"],
        "cases": evidence}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "cases": len(evidence),
                      "elapsed_seconds": [item["elapsed_seconds"] for item in evidence]}))


if __name__ == "__main__":
    main()
