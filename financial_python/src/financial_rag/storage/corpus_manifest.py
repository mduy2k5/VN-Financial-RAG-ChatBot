"""Read-only Mongo corpus snapshot; no raw report text or credentials in output."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ..structure.normalized import fingerprint, parser_revision


def mongo_manifest(database) -> dict:
    documents = []
    for row in database.financial_documents.find({}).sort("_id", 1):
        encoded = row["raw_content"].encode()
        documents.append({
            **{key: row.get(key) for key in ("document_id", "source_file", "company_code",
                                           "fiscal_year", "fiscal_quarter", "report_type",
                                           "chunk_count")},
            "logical_key": row["_id"], "source_sha256": hashlib.sha256(encoded).hexdigest(),
            "source_bytes": len(encoded),
        })
    collections = {}
    for name in sorted(database.list_collection_names()):
        stats = database.command("collStats", name)
        collections[name] = {key: stats.get(key) for key in
                             ("count", "size", "storageSize", "totalIndexSize")}
    # Hash the payloads too: identical source hashes can conceal a changed parser.
    digest = hashlib.sha256()
    for row in database.financial_chunks.find({}, {"payload": 1}).sort("_id", 1):
        digest.update(fingerprint(row["payload"]).encode())
    return {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "mongodb", "source_snapshot_sha256": fingerprint(documents),
        "chunk_snapshot_sha256": digest.hexdigest(), "parser_revision": parser_revision(),
        "document_count": len(documents),
        "declared_chunk_count": sum(row["chunk_count"] for row in documents),
        "by_company": dict(Counter(row["company_code"] for row in documents)),
        "collections": collections, "documents": documents,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    from pymongo import MongoClient
    client: Any
    with MongoClient(os.environ["MONGODB_URI"]) as client:
        result = mongo_manifest(client[os.getenv("FINANCIAL_MONGO_DATABASE", "financial_rag")])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("document_count", "declared_chunk_count", "source_snapshot_sha256")}))


if __name__ == "__main__":
    main()
