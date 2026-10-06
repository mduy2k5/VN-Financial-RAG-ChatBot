"""SQLite persistence for documents and financial chunks.

The retrieval index is rebuilt in memory from durable chunk records at startup.
This keeps the standalone deployment dependency-free while making ingestion
survive service restarts. Dense embeddings remain an optional rebuildable index.
"""

from __future__ import annotations

from pathlib import Path
import json
import sqlite3
from typing import Any, Iterable

from ..metadata.company_extractor import CompanyExtractor
from ..schemas import FinancialChunk
from ..structure.normalized import normalize_report, fingerprint


class SQLiteFinancialStore:
    def __init__(self, database_path: Path | str):
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    logical_key TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    source_file TEXT NOT NULL,
                    source_pdf TEXT,
                    company_code TEXT,
                    fiscal_year INTEGER,
                    fiscal_quarter INTEGER,
                    report_type TEXT,
                    raw_content TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    ingested_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    logical_key TEXT NOT NULL REFERENCES documents(logical_key) ON DELETE CASCADE,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_logical_key ON chunks(logical_key);
                CREATE INDEX IF NOT EXISTS idx_documents_source_file ON documents(source_file);
                CREATE TABLE IF NOT EXISTS report_versions (
                    version_id TEXT PRIMARY KEY, logical_key TEXT NOT NULL,
                    payload_json TEXT NOT NULL, raw_content TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_report_versions_key ON report_versions(logical_key);
                CREATE TABLE IF NOT EXISTS report_projections (
                    projection_id TEXT PRIMARY KEY, logical_key TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS active_projections (
                    logical_key TEXT PRIMARY KEY, version_id TEXT NOT NULL,
                    projection_id TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS projection_chunk_snapshots (
                    projection_id TEXT PRIMARY KEY, logical_key TEXT NOT NULL,
                    payload_json TEXT NOT NULL, sha256 TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS source_tables (
                    table_id TEXT PRIMARY KEY, projection_id TEXT NOT NULL, payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS source_rows (
                    row_id TEXT PRIMARY KEY, projection_id TEXT NOT NULL, payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS financial_cells (
                    fact_id TEXT PRIMARY KEY, projection_id TEXT NOT NULL,
                    row_id TEXT NOT NULL, period_year INTEGER NOT NULL, payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_cells_scope
                    ON financial_cells(projection_id, period_year);
                """
            )

    def replace_documents(self, documents: Iterable[dict[str, Any]]) -> int:
        """Atomically replace each logical document and its complete chunk set."""
        document_list = list(documents)
        if not document_list:
            return 0
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for document in document_list:
                chunks = list(document["chunks"])
                document = {**document, "chunks": chunks}
                previous = connection.execute("SELECT * FROM documents WHERE logical_key = ?",
                                              (document["logical_key"],)).fetchone()
                if previous:
                    previous = dict(previous)
                    previous["chunks"] = [FinancialChunk.model_validate_json(row[0]) for row in
                                          connection.execute(
                                              "SELECT payload_json FROM chunks WHERE logical_key = ? ORDER BY rowid",
                                              (document["logical_key"],))]
                    self._save_projection(connection, previous, activate=False)
                self._save_projection(connection, document)
                connection.execute(
                    "DELETE FROM documents WHERE logical_key = ?",
                    (document["logical_key"],),
                )
                connection.execute(
                    """INSERT INTO documents (
                        logical_key, document_id, source_file, source_pdf, company_code,
                        fiscal_year, fiscal_quarter, report_type, raw_content, chunk_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        document["logical_key"],
                        document["document_id"],
                        document["source_file"],
                        document.get("source_pdf"),
                        document.get("company_code"),
                        document.get("fiscal_year"),
                        document.get("fiscal_quarter"),
                        document.get("report_type"),
                        document["raw_content"],
                        len(chunks),
                    ),
                )
                connection.executemany(
                    "INSERT INTO chunks (chunk_id, logical_key, payload_json) VALUES (?, ?, ?)",
                    [
                        (chunk.chunk_id, document["logical_key"], chunk.model_dump_json())
                        for chunk in chunks
                    ],
                )
        return sum(len(document["chunks"]) for document in document_list)

    @staticmethod
    def _save_projection(connection, document: dict, *, activate: bool = True) -> None:
        bundle = normalize_report(document)
        key = document["logical_key"]
        connection.execute("INSERT OR IGNORE INTO report_versions VALUES (?, ?, ?, ?)",
                           (bundle.version.version_id, key, bundle.version.model_dump_json(),
                            document["raw_content"]))
        connection.execute("INSERT OR IGNORE INTO report_projections VALUES (?, ?, ?)",
                           (bundle.projection_id, key, json.dumps({**bundle.manifest(),
                                                                  "scope": bundle.version.model_dump()})))
        payloads = [chunk.model_dump(mode="json") for chunk in document["chunks"]]
        connection.execute("INSERT OR IGNORE INTO projection_chunk_snapshots VALUES (?, ?, ?, ?)",
                           (bundle.projection_id, key, json.dumps(payloads, ensure_ascii=False),
                            fingerprint(payloads)))
        for table, records, field in (("source_tables", bundle.tables, "table_id"),
                                      ("source_rows", bundle.rows, "row_id")):
            connection.executemany(f"INSERT OR IGNORE INTO {table} VALUES (?, ?, ?)",
                                   [(getattr(item, field), bundle.projection_id, item.model_dump_json())
                                    for item in records])
        connection.executemany("INSERT OR IGNORE INTO financial_cells VALUES (?, ?, ?, ?, ?)",
                               [(item.fact_id, item.projection_id, item.row_id, item.period_year,
                                 item.model_dump_json()) for item in bundle.facts])
        if activate:
            connection.execute("INSERT OR REPLACE INTO active_projections VALUES (?, ?, ?)",
                               (key, bundle.version.version_id, bundle.projection_id))

    def get_projection_document(self, logical_key: str, projection_id: str) -> dict:
        """Read the pinned source, never substitute a newer active projection."""
        import hashlib
        with self._connect() as connection:
            projection_row = connection.execute(
                "SELECT payload_json FROM report_projections WHERE logical_key = ? AND projection_id = ?",
                (logical_key, projection_id)).fetchone()
            if not projection_row:
                raise ValueError("Projection does not belong to the requested document")
            projection = json.loads(projection_row[0])
            row = connection.execute("SELECT payload_json, raw_content FROM report_versions "
                "WHERE version_id = ? AND logical_key = ?", (projection["version_id"], logical_key)).fetchone()
            if not row:
                raise ValueError("Pinned source version is unavailable")
            version, raw = json.loads(row[0]), row[1]
            snapshot = connection.execute("SELECT payload_json, sha256 FROM projection_chunk_snapshots "
                "WHERE projection_id = ? AND logical_key = ?", (projection_id, logical_key)).fetchone()
            if snapshot is None:
                # Upgrade pre-notebook active records only after checking their original hash.
                active = connection.execute("SELECT projection_id FROM active_projections WHERE logical_key = ?",
                                            (logical_key,)).fetchone()
                if not active or active[0] != projection_id:
                    raise ValueError("Historical retrieval snapshot is unavailable")
                payloads = [json.loads(item[0]) for item in connection.execute(
                    "SELECT payload_json FROM chunks WHERE logical_key = ? ORDER BY rowid", (logical_key,))]
                if fingerprint(payloads) != projection["chunk_sha256"]:
                    raise ValueError("Active projection checksum mismatch")
                snapshot = (json.dumps(payloads, ensure_ascii=False), fingerprint(payloads))
                connection.execute("INSERT OR IGNORE INTO projection_chunk_snapshots VALUES (?, ?, ?, ?)",
                                   (projection_id, logical_key, *snapshot))
            payloads = json.loads(snapshot[0])
            if (fingerprint(payloads) != snapshot[1] or fingerprint(payloads) != projection["chunk_sha256"]
                    or hashlib.sha256(raw.encode()).hexdigest() != version["source_sha256"]):
                raise ValueError("Pinned source checksum mismatch")
            return {**version, "fiscal_year": version.get("report_year"), "raw_content": raw,
                    "active_version_id": version["version_id"], "active_projection_id": projection_id,
                    "chunks": [FinancialChunk.model_validate(item) for item in payloads]}

    def list_versions(self, logical_key: str) -> list[dict]:
        with self._connect() as connection:
            result = [json.loads(row[0]) for row in connection.execute(
                "SELECT payload_json FROM report_versions WHERE logical_key = ? ORDER BY rowid",
                (logical_key,))]
            for version in result:
                version["projections"] = [json.loads(row[0]) for row in connection.execute(
                    "SELECT payload_json FROM report_projections WHERE "
                    "json_extract(payload_json, '$.version_id') = ? ORDER BY rowid",
                    (version["version_id"],))]
            return result

    def list_facts(self, logical_key: str, *, period_year: int | None = None,
                   projection_id: str | None = None, limit: int = 100, offset: int = 0) -> dict:
        with self._connect() as connection:
            row = connection.execute("SELECT projection_id FROM active_projections WHERE logical_key = ?",
                                     (logical_key,)).fetchone()
            active = row[0] if row else None
            if projection_id and not connection.execute(
                "SELECT 1 FROM report_projections WHERE projection_id = ? AND logical_key = ?",
                (projection_id, logical_key),
            ).fetchone():
                raise ValueError("Projection does not belong to the requested document")
            projection_id = projection_id or active
            if not projection_id:
                return {"projection_id": None, "total": 0, "facts": []}
            where = "f.projection_id = ?" + (" AND f.period_year = ?" if period_year is not None else "")
            parameters = [projection_id] + ([period_year] if period_year is not None else [])
            count = connection.execute(f"SELECT COUNT(*) FROM financial_cells f WHERE {where}",
                                       parameters).fetchone()[0]
            records = connection.execute(
                f"SELECT f.payload_json, r.payload_json FROM financial_cells f "
                f"JOIN source_rows r ON r.row_id = f.row_id WHERE {where} "
                "ORDER BY f.fact_id LIMIT ? OFFSET ?", [*parameters, limit, offset])
            facts = [{**json.loads(row[0]), "row": json.loads(row[1])} for row in records]
            table_ids = {item["row"]["table_id"] for item in facts if item["row"]["table_id"]}
            tables = [json.loads(row[0]) for table_id in table_ids for row in connection.execute(
                "SELECT payload_json FROM source_tables WHERE table_id = ?", (table_id,))]
            projection = json.loads(connection.execute(
                "SELECT payload_json FROM report_projections WHERE projection_id = ?",
                (projection_id,)).fetchone()[0])
            return {"projection_id": projection_id, "active": projection_id == active,
                    "projection": projection, "tables": tables, "total": count, "facts": facts}

    def load_chunks(self) -> list[FinancialChunk]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM chunks ORDER BY rowid"
            ).fetchall()
        return [FinancialChunk.model_validate_json(row["payload_json"]) for row in rows]

    def list_documents(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT d.*, a.version_id AS active_version_id,
                          a.projection_id AS active_projection_id
                   FROM (SELECT logical_key, document_id, source_file, source_pdf, company_code,
                          fiscal_year, fiscal_quarter, report_type, chunk_count, ingested_at
                         FROM documents) d
                   LEFT JOIN active_projections a USING (logical_key)
                   ORDER BY ingested_at, source_file"""
            ).fetchall()
        return [dict(row) for row in rows]

    def load_company_aliases(self) -> dict[str, list[str]]:
        """Build ticker-to-legal-name aliases from the stored report headers."""
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT company_code, substr(raw_content, 1, 20000) AS opening_text
                   FROM documents
                   WHERE company_code IS NOT NULL AND trim(company_code) != ''
                   ORDER BY ingested_at, source_file"""
            ).fetchall()

        aliases: dict[str, list[str]] = {}
        for row in rows:
            company_code = str(row["company_code"]).strip().upper()
            legal_names = CompanyExtractor.extract_legal_name_aliases(
                str(row["opening_text"] or "")
            )
            if legal_names:
                company_aliases = aliases.setdefault(company_code, [])
                seen = {alias.casefold() for alias in company_aliases}
                for legal_name in legal_names:
                    if legal_name.casefold() not in seen:
                        company_aliases.append(legal_name)
                        seen.add(legal_name.casefold())
        return aliases

    def delete_document(self, logical_key: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM documents WHERE logical_key = ?", (logical_key,)
            )
            connection.execute("DELETE FROM active_projections WHERE logical_key = ?", (logical_key,))
            return cursor.rowcount > 0

    def document_count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM documents").fetchone()
        return int(row["count"])
