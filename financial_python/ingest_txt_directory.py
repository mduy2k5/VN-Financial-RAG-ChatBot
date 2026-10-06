"""Upload top-level TXT financial reports to the standalone service in batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests


def _batches(paths: list[Path], max_documents: int, max_bytes: int):
    batch: list[Path] = []
    byte_count = 0
    for path in paths:
        size = path.stat().st_size
        if batch and (len(batch) >= max_documents or byte_count + size > max_bytes):
            yield batch
            batch = []
            byte_count = 0
        if size > max_bytes:
            raise ValueError(f"Single file exceeds batch byte limit: {path}")
        batch.append(path)
        byte_count += size
    if batch:
        yield batch


def upload_directory(
    data_dir: Path,
    endpoint: str,
    max_documents: int = 20,
    max_bytes: int = 40_000_000,
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    paths = sorted(path for path in data_dir.glob("*.txt") if path.is_file())
    if not paths:
        raise ValueError(f"No direct .txt files found in {data_dir}")

    batches = list(_batches(paths, max_documents=max_documents, max_bytes=max_bytes))
    results = []
    with requests.Session() as session:
        for batch_number, batch in enumerate(batches, start=1):
            documents = [
                {
                    "source_file": path.name,
                    "content": path.read_text(encoding="utf-8", errors="replace"),
                }
                for path in batch
            ]
            response = session.post(
                endpoint,
                json={"documents": documents},
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            result = response.json()
            results.append(result)
            print(
                json.dumps(
                    {
                        "batch": batch_number,
                        "batch_count": len(batches),
                        "files": [path.name for path in batch],
                        "documents_with_chunks": result.get("documents_with_chunks"),
                        "new_chunks_count": result.get("new_chunks_count"),
                        "total_chunks_indexed": result.get("total_chunks_indexed"),
                    },
                    ensure_ascii=False,
                )
            )
    return {
        "source_files": len(paths),
        "batch_count": len(batches),
        "documents_with_chunks": sum(item.get("documents_with_chunks", 0) for item in results),
        "new_chunks_count": sum(item.get("new_chunks_count", 0) for item in results),
        "last_total_chunks_indexed": results[-1].get("total_chunks_indexed", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Directory containing top-level TXT reports")
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8092/api/v1/financial/ingest/batch",
    )
    parser.add_argument("--max-documents", type=int, default=20)
    parser.add_argument("--max-bytes", type=int, default=40_000_000)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()
    print(
        json.dumps(
            upload_directory(
                args.data_dir,
                args.endpoint,
                max_documents=args.max_documents,
                max_bytes=args.max_bytes,
                timeout_seconds=args.timeout_seconds,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
