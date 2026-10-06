"""Verify owned synthetic TXT/Markdown documents through ingest, restart, update and delete.

Run `prepare`, restart the API, then run `resume` with the same artifact path.
These are operational fixtures, not financial ground truth or benchmark labels.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote
import uuid

import requests


def request(base: str, method: str, path: str, body=None):
    response = requests.request(method, base.rstrip("/") + path, json=body, timeout=120)
    response.raise_for_status()
    return response.json()


def source(company: str, extension: str, total: int) -> str:
    rows = [("Vay ngắn hạn", "311", 80, 60),
            ("Phải trả người bán", "312", total - 80, 40),
            ("TỔNG NỢ PHẢI TRẢ", "300", total, 100)]
    intro = (
        f"# Page 7\nBÁO CÁO TÀI CHÍNH HỢP NHẤT {company}\n"
        "Cho năm tài chính kết thúc ngày 31 tháng 12 năm 2029\n"
        "TÀI LIỆU THỬ NGHIỆM TỔNG HỢP, KHÔNG PHẢI SỐ LIỆU THẬT\n"
        "## BẢNG CÂN ĐỐI KẾ TOÁN\nĐơn vị: Triệu đồng\n"
    )
    if extension == "md":
        return intro + "| Chỉ tiêu | Mã số | 31/12/2029 | 01/01/2029 |\n| --- | --- | ---: | ---: |\n" + "\n".join(
            f"| {label} | {code} | {current} | {prior} |" for label, code, current, prior in rows
        )
    return intro + '<table><tr><th>Chỉ tiêu</th><th>Mã số</th><th>31/12/2029</th><th>01/01/2029</th></tr>' + "".join(
        f"<tr><td>{label}</td><td>{code}</td><td>{current}</td><td>{prior}</td></tr>"
        for label, code, current, prior in rows
    ) + "</table>"


def wait_dense(base: str):
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        health = request(base, "GET", "/health")
        if health["dense_retrieval"] == "ready":
            return health
        if health["dense_retrieval"] == "error":
            raise RuntimeError(health)
        time.sleep(1)
    raise TimeoutError("The existing indexing process did not become ready within the observation window")


def assert_answer(base: str, item: dict, total: int) -> dict:
    query = f"Khoản nào đóng góp vào biến động nợ phải trả {item['company']} năm 2028 và 2029"
    payload = request(base, "POST", "/api/v1/financial/answer", {"query": query})
    if not payload["validation"]["is_valid"] or payload.get("answer_status") != "complete":
        raise AssertionError(payload)
    totals = [calculation for calculation in payload["calculations"]
              if calculation["formula_id"] == "absolute_change"
              and {fact["label"] for fact in calculation["inputs"]} == {"TỔNG NỢ PHẢI TRẢ"}]
    if len(totals) != 1 or totals[0]["value"] != str(total - 100):
        raise AssertionError(payload)
    inputs = {fact["role"]: fact for fact in totals[0]["inputs"]}
    if inputs["prior_value"]["year"] != 2028 or inputs["current_value"]["year"] != 2029:
        raise AssertionError(inputs)
    if inputs["current_value"]["value"] != str(total):
        raise AssertionError(inputs)
    if not payload["citations"] or any(citation["file_name"] != item["source_file"] for citation in payload["citations"]):
        raise AssertionError(payload["citations"])
    return payload


def save(output: Path, artifact: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=["prepare", "resume"])
    parser.add_argument("--base-url", default="http://127.0.0.1:8092")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base, output = args.base_url, args.output
    if args.phase == "prepare":
        if output.exists():
            raise ValueError("Use a new artifact path; existing lifecycle state must not be overwritten")
        before = request(base, "GET", "/api/v1/financial/documents")
        nonce = uuid.uuid4().hex
        artifact = {
            "artifact_type": "synthetic_runtime_lifecycle", "phase": "preparing", "base_url": base,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "original_documents": before, "health_before": wait_dense(base), "documents": [], "checks": {},
        }
        for company, extension in (("FPT", "md"), ("VPB", "txt")):
            artifact["documents"].append({
                "company": company, "extension": extension,
                "source_file": f"{company}_MVP_PROBE_{nonce}_2029_Hopnhat.{extension}",
                "document_id": f"mvp-runtime-probe-{nonce}-{extension}",
            })
    else:
        artifact = json.loads(output.read_text(encoding="utf-8"))
        if artifact.get("phase") != "prepared" or artifact["base_url"] != base:
            raise ValueError("A prepared artifact for the same API is required")
    try:
        for item in artifact["documents"]:
            name = item["extension"]
            if args.phase == "prepare":
                if any(doc["source_file"] == item["source_file"] for doc in artifact["original_documents"]):
                    raise ValueError("Probe filename already belongs to an existing document")
                content = source(item["company"], name, 130)
                ingested = request(base, "POST", "/api/v1/financial/ingest", {
                    "content": content, "company_code": item["company"],
                    "source_file": item["source_file"], "document_id": item["document_id"],
                })
                item["logical_key"] = ingested["logical_key"]
                item["content_sha256"] = hashlib.sha256(content.encode()).hexdigest()
                save(output, artifact)
                artifact["checks"][f"{name}_immediate"] = assert_answer(base, item, 130)
                artifact["checks"][f"{name}_dense_health"] = wait_dense(base)
                artifact["checks"][f"{name}_dense_answer"] = assert_answer(base, item, 130)
            else:
                # Ownership must be checked before replacing or deleting anything.
                documents = request(base, "GET", "/api/v1/financial/documents")
                owned = next((doc for doc in documents if doc["logical_key"] == item["logical_key"]), None)
                if not owned or owned["document_id"] != item["document_id"] or owned["source_file"] != item["source_file"]:
                    raise ValueError("Probe ownership changed; refusing to mutate the document")
                artifact["checks"][f"{name}_after_restart"] = assert_answer(base, item, 130)
                replaced = request(base, "POST", "/api/v1/financial/ingest", {
                    "content": source(item["company"], name, 150), "company_code": item["company"],
                    "source_file": item["source_file"], "document_id": item["document_id"],
                })
                if replaced["logical_key"] != item["logical_key"]:
                    raise AssertionError(replaced)
                artifact["checks"][f"{name}_updated_immediate"] = assert_answer(base, item, 150)
                artifact["checks"][f"{name}_updated_dense_health"] = wait_dense(base)
                artifact["checks"][f"{name}_updated_dense_answer"] = assert_answer(base, item, 150)
            print(f"Verified {args.phase}: {name}", flush=True)
        if args.phase == "prepare":
            artifact["health_prepared"] = wait_dense(base)
            artifact["phase"] = "prepared"
            save(output, artifact)
            print("Prepared owned probes. Restart API, then resume with this artifact.", flush=True)
            return 0
    except Exception as exc:
        artifact["error"] = repr(exc)
        raise
    finally:
        if args.phase == "resume" or artifact.get("error"):
            for item in artifact["documents"]:
                if not item.get("logical_key"):
                    continue
                documents = request(base, "GET", "/api/v1/financial/documents")
                owned = next((doc for doc in documents if doc["logical_key"] == item["logical_key"]), None)
                if owned and owned["document_id"] == item["document_id"] and owned["source_file"] == item["source_file"]:
                    request(base, "DELETE", "/api/v1/financial/documents/" + quote(item["logical_key"], safe=""))
                elif owned:
                    artifact["cleanup_error"] = "Ownership mismatch; left document untouched"
            after = request(base, "GET", "/api/v1/financial/documents")
            def identity(docs):
                return {(d["logical_key"], d["document_id"], d["source_file"], d["chunk_count"]) for d in docs}
            artifact["original_documents_restored"] = identity(after) == identity(artifact["original_documents"])
            artifact["health_after_cleanup"] = wait_dense(base)
            artifact["phase"] = "failed" if artifact.get("error") else "cleaned"
            save(output, artifact)
    if not artifact["original_documents_restored"]:
        raise AssertionError("The original document inventory was not restored")
    for item in artifact["documents"]:
        results = request(base, "POST", "/api/v1/financial/retrieve", {"query": f"Nợ phải trả {item['company']} năm 2029"})
        if any(result.get("chunk", {}).get("source_file") == item["source_file"] for result in results):
            raise AssertionError("Deleted probe still exists in active retrieval")
        artifact["checks"][f"{item['extension']}_deleted_retrieval"] = results
    artifact["phase"] = "completed"
    save(output, artifact)
    print("Lifecycle verified; original document inventory restored.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
