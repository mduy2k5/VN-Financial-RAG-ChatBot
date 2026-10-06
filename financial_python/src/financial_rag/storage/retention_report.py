"""Read-only storage/retention inventory. Source and search history is retained by default."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path


def report(database):
    active = {row["active_projection_id"] for row in database.financial_documents.find(
        {"active_projection_id": {"$exists": True}}, {"active_projection_id": 1})}
    collections = {}
    for name in database.list_collection_names():
        stats = database.command("collStats", name)
        collections[name] = {key: stats.get(key, 0) for key in ("count", "size", "storageSize", "totalIndexSize")}
    return {"created_at_utc": datetime.now(timezone.utc).isoformat(), "mode": "read_only",
            "collections": collections, "active_projections": len(active),
            "historical_projections": database.financial_report_projections.count_documents({"_id": {"$nin": list(active)}}),
            "policy": {"sources_and_projections": "retain_for_reproducibility_and_rollback",
                       "archived_chunks": "retain; created on replacement, not whole-corpus duplication at startup",
                       "qdrant_search_points": "retain; exact request point IDs exclude stale/orphan points",
                       "terminal_jobs": "TTL 30 days; successful payload removed atomically",
                       "sparse_indexes": "rebuildable; disk budget 64 MiB with eviction"},
            "garbage_collection_applied": False,
            "deletion_guard": "Historical vectors cannot be deleted safely without tracking every remote reader lease. No unsafe vector/history GC enabled."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from pymongo import MongoClient
    with MongoClient(os.environ["MONGODB_URI"]) as client:
        output = report(client[os.getenv("FINANCIAL_MONGO_DATABASE", "financial_rag")])
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: output[key] for key in ("active_projections", "historical_projections", "garbage_collection_applied")}))


if __name__ == "__main__":
    main()
