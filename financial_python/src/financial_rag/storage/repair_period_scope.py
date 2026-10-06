"""Repair incidental-quarter classification without changing source amounts or row IDs."""
import argparse
import json
import os
from pathlib import Path
import re

from ..metadata.period_extractor import PeriodExtractor
from ..schemas import FinancialChunk
from ..structure.normalized import fingerprint
from .migrate_normalized import backfill_coordinates
from .mongo_store import MongoFinancialStore


def repair(store, *, apply=False):
    result = {"mode": "apply" if apply else "dry_run", "repairs": []}
    for record in store.documents.find({"fiscal_quarter": {"$in": [1, 2, 3, 4]}}):
        front = " ".join(record["raw_content"][:5000].split())
        annual = bool(re.search(r"(?:báo cáo tài chính|cho năm|năm tài chính).{0,100}?(?:năm tài chính|năm\s*20\d{2}\s*(?:đã\s*)?kiểm toán|kết thúc.{0,50}(?:31|ngày 31))", front, re.I))
        if not annual or PeriodExtractor.extract_document_quarter(record["raw_content"], record["source_file"]):
            continue
        document = {**record, "logical_key": record["_id"], "chunks": [FinancialChunk.model_validate(item["payload"])
                    for item in store.chunks.find({"logical_key": record["_id"]}).sort("ordinal", 1)]}
        preserved = [(chunk.chunk_id, chunk.content, chunk.metadata.get("period_values"), chunk.metadata.get("end_value"), chunk.metadata.get("start_value"))
                     for chunk in document["chunks"]]
        original = fingerprint([chunk.model_dump(mode="json") for chunk in document["chunks"]])
        document["chunks"], _ = backfill_coordinates(document)
        document["fiscal_quarter"] = None
        for chunk in document["chunks"]:
            chunk.fiscal_quarter = None
            if "fiscal_quarter" in chunk.metadata:
                chunk.metadata["fiscal_quarter"] = None
        assert preserved == [(chunk.chunk_id, chunk.content, chunk.metadata.get("period_values"), chunk.metadata.get("end_value"), chunk.metadata.get("start_value"))
                             for chunk in document["chunks"]]
        if apply:
            # Publication uses a transaction; detect source drift before entering.
            current = store.documents.find_one({"_id": record["_id"]})
            payloads = [item["payload"] for item in store.chunks.find({"logical_key": record["_id"]}).sort("ordinal", 1)]
            if current["active_projection_id"] != record["active_projection_id"] or fingerprint(payloads) != original:
                raise RuntimeError("Source changed during repair; rerun dry-run")
            store.replace_documents([document], expected_projections={record["_id"]: record["active_projection_id"]})
        result["repairs"].append({"source_file": record["source_file"], "old_quarter": record["fiscal_quarter"],
                                  "new_quarter": None, "amounts_labels_content_and_ids_preserved": True})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    store = MongoFinancialStore(os.environ["MONGODB_URI"], os.getenv("FINANCIAL_MONGO_DATABASE", "financial_rag"))
    try:
        result = repair(store, apply=args.apply)
    finally:
        store.close()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
