"""Idempotent shadow migration; preserves active chunks and answer behavior.

Run dry-run first, then --apply. Backfill coordinates only where a reparsed row
has the same page, original label, unit and all numerical period values, and is
unique. Source/chunk hashes are checked again inside the publishing transaction.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from ..chunking import UniversalFinancialChunker
from ..schemas import ChunkType, FinancialChunk
from ..structure.normalized import fingerprint, normalize_report
from ..validation.numeric import NumericValidator
from .mongo_store import MongoFinancialStore

COORDINATE_KEYS = ("table_id", "source_start", "source_end", "source_row_index",
                   "source_cells", "table_headers", "source_period_columns", "source_statement_type")


def row_signature(chunk: FinancialChunk) -> tuple:
    metadata = chunk.metadata
    values = dict(metadata.get("period_values") or {})
    for year, key in ((chunk.fiscal_year, "end_value"), (chunk.comparative_fiscal_year, "start_value")):
        if year is not None and str(year) not in values and metadata.get(key) is not None:
            values[str(year)] = metadata[key]
    normalized = tuple(sorted((str(year), NumericValidator.normalize_number(str(value)))
                              for year, value in values.items()
                              if NumericValidator.normalize_number(str(value))))
    label = " ".join(str(metadata.get("metric_name") or metadata.get("metric") or "").casefold().split())
    return chunk.page, label, NumericValidator._normalize_unit(chunk.unit or metadata.get("unit")), normalized


def backfill_coordinates(document: dict) -> tuple[list[FinancialChunk], dict[str, int]]:
    parsed = UniversalFinancialChunker().chunk_text(
        document["raw_content"], source_file=document["source_file"],
        company_code=document.get("company_code"), source_pdf=document.get("source_pdf"),
        document_id=document.get("document_id"),
    )
    candidates: dict[tuple, list[FinancialChunk]] = defaultdict(list)
    for chunk in parsed:
        if chunk.chunk_type == ChunkType.FINANCIAL_ITEM:
            candidates[row_signature(chunk)].append(chunk)
    counts: Counter[str] = Counter()
    result = []
    for chunk in document["chunks"]:
        copy = chunk.model_copy(deep=True)
        if chunk.chunk_type == ChunkType.FINANCIAL_ITEM:
            matches = candidates.get(row_signature(chunk), [])
            if len(matches) == 1:
                copy.metadata.update({key: matches[0].metadata[key] for key in COORDINATE_KEYS
                                      if key in matches[0].metadata})
                counts["matched_rows"] += 1
            else:
                counts["ambiguous_rows" if matches else "unmatched_rows"] += 1
        result.append(copy)
    return result, dict(counts)


def migrate(store: MongoFinancialStore, *, apply: bool, logical_key: str | None = None,
            backfill: bool = True) -> dict[str, Any]:
    output: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "apply" if apply else "dry_run",
        "active_chunks_changed": False,
        "documents": [],
    }
    query = {"_id": logical_key} if logical_key else {}
    for document in store.documents.find(query).sort("_id", 1):
        document["logical_key"] = document["_id"]
        original = [FinancialChunk.model_validate(row["payload"]) for row in
                    store.chunks.find({"logical_key": document["logical_key"]}).sort("ordinal", 1)]
        if len(original) != document["chunk_count"]:
            raise RuntimeError(f"Chunk count mismatch: {document['source_file']}")
        document["chunks"] = original
        source_chunk_hash = fingerprint([chunk.model_dump(mode="json") for chunk in original])
        counts: dict[str, int] = {}
        if backfill:
            document["chunks"], counts = backfill_coordinates(document)
        bundle = normalize_report(document)
        if apply:
            with store.client.start_session() as session:
                with session.start_transaction():
                    current = store.documents.find_one({"_id": document["logical_key"]}, session=session)
                    current_chunks = [row["payload"] for row in store.chunks.find(
                        {"logical_key": document["logical_key"]}, session=session).sort("ordinal", 1)]
                    if (not current or current["raw_content"] != document["raw_content"]
                            or fingerprint(current_chunks) != source_chunk_hash):
                        raise RuntimeError("Document changed during migration; rerun the dry-run")
                    # Touch the source record to establish a write conflict with concurrent ingest.
                    store.documents.update_one({"_id": document["logical_key"]}, {"$set": {
                        "active_version_id": bundle.version.version_id,
                        "active_projection_id": bundle.projection_id}}, session=session)
                    store._save_projection(document, bundle, session=session)
                    # Persist the actual backfilled projection as well as its facts.
                    # It can be pinned independently of untouched legacy retrieval rows.
                    store._archive_chunks(bundle.projection_id, document["chunks"], session=session)
        output["documents"].append({"logical_key": document["logical_key"],
                                    "source_file": document["source_file"], **bundle.manifest(),
                                    "coordinate_backfill": counts})
    output["summary"] = {
        "document_count": len(output["documents"]),
        **{field: sum(row[field] for row in output["documents"])
           for field in ("table_count", "row_count", "fact_count")},
        "warnings": dict(sum((Counter(row["warnings"]) for row in output["documents"]), Counter())),
        "coordinate_backfill": dict(sum((Counter(row["coordinate_backfill"])
                                           for row in output["documents"]), Counter())),
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--logical-key")
    parser.add_argument("--no-backfill", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    store = MongoFinancialStore(os.environ["MONGODB_URI"], os.getenv("FINANCIAL_MONGO_DATABASE", "financial_rag"))
    try:
        result = migrate(store, apply=args.apply, logical_key=args.logical_key,
                         backfill=not args.no_backfill)
    finally:
        store.client.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
