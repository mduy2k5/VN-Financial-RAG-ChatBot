"""Additive Mongo/SQLite workspace records with optimistic concurrency."""
from __future__ import annotations

import json
from typing import Any


class NotebookError(Exception):
    def __init__(self, detail: str, status: int = 422):
        self.detail = detail
        self.status = status


class RecordRepository:
    def __init__(self, documents: Any):
        self.documents = documents
        self.collection = None
        if hasattr(documents, "client"):
            self.collection = documents.documents.database["financial_notebook_records"]
            self.collection.create_index([("kind", 1), ("notebook_id", 1)])
        elif hasattr(documents, "_connect"):
            with documents._connect() as connection:
                connection.execute("""CREATE TABLE IF NOT EXISTS notebook_records (
                    record_key TEXT PRIMARY KEY, kind TEXT NOT NULL, notebook_id TEXT NOT NULL,
                    revision INTEGER NOT NULL, payload_json TEXT NOT NULL)""")
                connection.execute("CREATE INDEX IF NOT EXISTS idx_notebook_records_scope "
                                   "ON notebook_records(kind, notebook_id)")
        else:
            raise NotebookError("Notebook cần kho MongoDB hoặc SQLite bền vững.", 503)

    def get(self, kind: str, record_id: str, notebook_id: str | None = None) -> dict | None:
        key = f"{kind}:{record_id}"
        if self.collection is not None:
            query: dict = {"_id": key}
            if notebook_id is not None:
                query["notebook_id"] = notebook_id
            row = self.collection.find_one(query)
            return row["payload"] if row else None
        with self.documents._connect() as connection:
            row = connection.execute("SELECT payload_json, notebook_id FROM notebook_records "
                                     "WHERE record_key = ?", (key,)).fetchone()
        if not row or (notebook_id is not None and row["notebook_id"] != notebook_id):
            return None
        return json.loads(row["payload_json"])

    def list(self, kind: str, notebook_id: str | None = None) -> list[dict]:
        if self.collection is not None:
            query: dict = {"kind": kind}
            if notebook_id is not None:
                query["notebook_id"] = notebook_id
            values = [row["payload"] for row in self.collection.find(query)]
        else:
            where = "kind = ?" + (" AND notebook_id = ?" if notebook_id is not None else "")
            params = [kind] + ([notebook_id] if notebook_id is not None else [])
            with self.documents._connect() as connection:
                values = [json.loads(row[0]) for row in connection.execute(
                    f"SELECT payload_json FROM notebook_records WHERE {where}", params)]
        return sorted(values, key=lambda value: value.get("updated_at", ""), reverse=True)

    def put(self, kind: str, value: dict, expected: int | None = None) -> dict:
        """None creates exclusively; updates require the exact prior revision."""
        value = {**value, "revision": 1 if expected is None else expected + 1}
        encoded = json.dumps(value, ensure_ascii=False)
        if len(encoded.encode()) > 8_000_000:
            raise NotebookError("Bản ghi vượt ngân sách lưu trữ.")
        key = f"{kind}:{value['id']}"
        notebook_id = value.get("notebook_id", value["id"])
        if self.collection is not None:
            record = {"_id": key, "kind": kind, "notebook_id": notebook_id,
                      "revision": value["revision"], "payload": value}
            if expected is None:
                from pymongo.errors import DuplicateKeyError
                try:
                    self.collection.insert_one(record)
                except DuplicateKeyError as exc:
                    raise NotebookError("Bản ghi đã tồn tại; hãy tải lại.", 409) from exc
            else:
                result = self.collection.replace_one({"_id": key, "revision": expected}, record)
                if not result.matched_count:
                    raise NotebookError("Bản ghi đã thay đổi; hãy tải lại.", 409)
        else:
            import sqlite3
            with self.documents._connect() as connection:
                if expected is None:
                    try:
                        connection.execute("INSERT INTO notebook_records VALUES (?, ?, ?, ?, ?)",
                                           (key, kind, notebook_id, value["revision"], encoded))
                    except sqlite3.IntegrityError as exc:
                        raise NotebookError("Bản ghi đã tồn tại; hãy tải lại.", 409) from exc
                else:
                    result = connection.execute("UPDATE notebook_records SET revision = ?, "
                        "payload_json = ? WHERE record_key = ? AND revision = ?",
                        (value["revision"], encoded, key, expected))
                    if not result.rowcount:
                        raise NotebookError("Bản ghi đã thay đổi; hãy tải lại.", 409)
        return value
