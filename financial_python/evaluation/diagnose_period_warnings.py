"""Explain contradicted period headers from active source chunks; read-only audit."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path

from pymongo import MongoClient

from financial_rag.schemas import ChunkType, FinancialChunk
from financial_rag.structure.normalized import normalize_report
from financial_rag.structure.semantics import period_supported, table_scope
from financial_rag.storage.migrate_normalized import backfill_coordinates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    counts = Counter()
    normalizer_warning_counts = Counter()
    examples = defaultdict(list)
    with MongoClient(os.environ["MONGODB_URI"]) as client:
        database = client[os.getenv("FINANCIAL_MONGO_DATABASE", "financial_rag")]
        for document in database.financial_documents.find({}, {"_id": 1, "source_file": 1,
                                                               "company_code": 1, "fiscal_year": 1,
                                                               "fiscal_quarter": 1, "report_type": 1,
                                                               "source_pdf": 1, "raw_content": 1}):
            chunks = database.financial_chunks.find({"logical_key": document["_id"]},
                                                     {"payload": 1}).sort("ordinal", 1)
            parsed_chunks = [FinancialChunk.model_validate(stored["payload"]) for stored in chunks]
            migration_document = {**document, "logical_key": document["_id"], "chunks": parsed_chunks}
            backfilled_chunks, _ = backfill_coordinates(migration_document)
            for chunk in backfilled_chunks:
                if chunk.chunk_type != ChunkType.FINANCIAL_ITEM:
                    continue
                values = dict(chunk.metadata.get("period_values") or {})
                for year, key in ((chunk.fiscal_year, "end_value"),
                                  (chunk.comparative_fiscal_year, "start_value")):
                    if year is not None and str(year) not in values and chunk.metadata.get(key) is not None:
                        values[str(year)] = chunk.metadata[key]
                roles = chunk.metadata.get("source_period_columns") or {}
                headers = chunk.metadata.get("table_headers") or []
                for year in values:
                    if not str(year).isdigit() or period_supported(chunk, int(year)) is not False:
                        continue
                    column = roles.get(str(year))
                    header = headers[column] if isinstance(column, int) and 0 <= column < len(headers) else None
                    key = (table_scope(chunk), str(header or "<missing header>").strip(),
                           str(chunk.metadata.get("source_statement_type") or "unknown"))
                    counts[key] += 1
                    if len(examples[key]) < 4:
                        examples[key].append({"company": document.get("company_code"),
                                              "source_file": document.get("source_file"),
                                              "report_year": document.get("fiscal_year"),
                                              "chunk_year": chunk.fiscal_year,
                                              "cell_year": int(year), "column": column,
                                              "header": header, "statement_type": chunk.statement_type.value,
                                              "metric": chunk.metadata.get("metric_name") or chunk.metadata.get("metric"),
                                              "table_title": chunk.table_title})
            migrated = normalize_report({**migration_document, "chunks": backfilled_chunks})
            normalizer_warning_counts.update(migrated.warnings)
        client.close()
    records = [{"scope": key[0], "header": key[1], "source_statement_type": key[2],
                "count": count, "examples": examples[key]}
               for key, count in counts.most_common()]
    result = {"mode": "read_only", "contradicted_cell_count": sum(counts.values()),
              "normalizer_warning_counts": dict(normalizer_warning_counts),
              "diagnostic_false_header_count": sum(counts.values()),
              "normalizer_warning_counts_match": (
                  sum(counts.values()) == normalizer_warning_counts.get("contradicted_period_header", 0)),
              "pattern_count": len(records), "patterns": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"contradicted_cell_count": result["contradicted_cell_count"],
                      "normalizer_warning_counts": result["normalizer_warning_counts"],
                      "normalizer_warning_counts_match": result["normalizer_warning_counts_match"],
                      "pattern_count": result["pattern_count"], "top_patterns": records[:20]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
