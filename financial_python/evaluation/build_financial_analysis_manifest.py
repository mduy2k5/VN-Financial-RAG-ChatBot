"""Capture the durable corpus used by an analytics run without changing SQLite."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3


def build_manifest(database: Path, code_root: Path | None = None) -> dict:
    uri = database.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT logical_key, document_id, source_file, company_code, fiscal_year, "
            "fiscal_quarter, report_type, raw_content, chunk_count FROM documents "
            "ORDER BY source_file, logical_key"
        ).fetchall()
        stored_chunks = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    documents = []
    for row in rows:
        document = dict(row)
        raw = document.pop("raw_content")
        encoded = raw.encode("utf-8")
        document["bytes_utf8"] = len(encoded)
        document["content_sha256"] = hashlib.sha256(encoded).hexdigest()
        pages = set(re.findall(
            r"(?im)^\s*(?:=+\s*PAGE\s+(\d+)\s*=+|#{1,6}\s*(?:Page|Trang)\s+(\d+))", raw,
        ))
        document["page_count"] = len(pages)
        document["html_table_count"] = len(re.findall(r"<table\b", raw, re.IGNORECASE))
        document["metadata_gaps"] = [
            field for field in ("company_code", "fiscal_year", "report_type")
            if document.get(field) is None or document.get(field) == ""
        ]
        documents.append(document)
    stable_content = json.dumps(documents, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    manifest = {
        "schema_version": 1,
        "artifact_type": "financial_analytics_runtime_corpus_manifest",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "database": str(database),
        "snapshot_sha256": hashlib.sha256(stable_content.encode("utf-8")).hexdigest(),
        "summary": {
            "document_count": len(documents),
            "stored_chunk_count": stored_chunks,
            "declared_chunk_count": sum(doc["chunk_count"] for doc in documents),
            "by_company": dict(sorted(Counter(doc["company_code"] or "unknown" for doc in documents).items())),
            "by_report_type": dict(sorted(Counter(doc["report_type"] or "unknown" for doc in documents).items())),
            "documents_with_metadata_gaps": sum(bool(doc["metadata_gaps"]) for doc in documents),
            "empty_sources": sum(doc["bytes_utf8"] == 0 for doc in documents),
        },
        "documents": documents,
    }
    if code_root is not None:
        files = sorted(
            path for folder in ("src", "configs", "schemas")
            for path in (code_root / folder).rglob("*")
            if path.is_file() and path.suffix in {".py", ".json"}
        )
        code_files = {
            path.relative_to(code_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        }
        manifest["code_files"] = code_files
        manifest["code_snapshot_sha256"] = hashlib.sha256(
            json.dumps(code_files, sort_keys=True).encode("utf-8")
        ).hexdigest()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-root", type=Path)
    args = parser.parse_args()
    manifest = build_manifest(args.database, args.code_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": manifest["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
