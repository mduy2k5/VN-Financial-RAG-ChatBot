"""MongoDB persistence for uploaded financial documents and chunks."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
import uuid
from typing import Any, Iterable

from ..metadata.company_extractor import CompanyExtractor
from ..schemas import FinancialChunk
from ..structure.normalized import NormalizedReport, normalize_report


class MongoFinancialStore:
    """Store source text and chunk payloads in separate MongoDB collections.

    A single-node replica set is required so replacing a document and its chunks
    remains one transaction, matching the previous SQLite store's semantics.
    """

    def __init__(self, uri: str, database_name: str = "financial_rag"):
        if not uri.strip():
            raise ValueError("MONGODB_URI must be set for MongoDB persistence")

        from pymongo import ASCENDING, DESCENDING, MongoClient

        self.client: Any = MongoClient(
            uri,
            appname="vn-financial-understanding",
            serverSelectionTimeoutMS=30_000,
            connectTimeoutMS=10_000,
        )
        hello = self.client.admin.command("hello")
        if not hello.get("setName") or not hello.get("isWritablePrimary"):
            self.client.close()
            raise RuntimeError(
                "MongoDB must be initialized as a writable replica set before the API starts"
            )

        database = self.client[database_name]
        self.documents = database["financial_documents"]
        self.chunks = database["financial_chunks"]
        self.versions = database["financial_report_versions"]
        self.projections = database["financial_report_projections"]
        self.tables = database["financial_source_tables"]
        self.rows = database["financial_source_rows"]
        self.facts = database["financial_cells"]
        self.jobs = database["financial_ingest_jobs"]
        self.jobs.create_index([("status", ASCENDING), ("lease_until", ASCENDING), ("created_at", ASCENDING)])
        self.jobs.create_index("expires_at", expireAfterSeconds=0)
        self.retrieval_snapshots = database["financial_retrieval_snapshots"]
        self.archived_chunks = database["financial_archived_chunks"]
        self.archived_chunks.create_index([("projection_id", ASCENDING), ("ordinal", ASCENDING)], unique=True)
        self.versions.create_index([("logical_key", ASCENDING)])
        for collection in (self.tables, self.rows, self.facts):
            collection.create_index([("projection_id", ASCENDING)])
        self.facts.create_index([("projection_id", ASCENDING), ("period_year", ASCENDING)])
        self.conversations = database["financial_conversations"]
        self.conversation_turns = database["financial_conversation_turns"]
        self.documents.create_index([("ingested_at", ASCENDING), ("source_file", ASCENDING)])
        self.documents.create_index([("company_code", ASCENDING), ("fiscal_year", ASCENDING)])
        self.chunks.create_index([("logical_key", ASCENDING), ("ordinal", ASCENDING)], unique=True)
        self.chunks.create_index([
            ("company_code", ASCENDING),
            ("fiscal_year", ASCENDING),
            ("report_type", ASCENDING),
        ])
        self.conversations.create_index([("updated_at", DESCENDING), ("_id", ASCENDING)])
        self.conversation_turns.create_index(
            [("conversation_id", ASCENDING), ("ordinal", ASCENDING)], unique=True
        )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")

    def replace_documents(self, documents: Iterable[dict[str, Any]], *, job=None, expected_projections=None) -> int:
        """Atomically replace documents and their full chunk sets."""
        document_list = list(documents)
        if not document_list:
            return 0

        total_chunks = 0
        with self.client.start_session() as session:
            with session.start_transaction():
                if job:
                    job_id, lease_token = job
                    # This write fences an expired/reclaimed worker in the same
                    # transaction as the source publication; a read alone cannot.
                    fenced = self.jobs.update_one(
                        {"_id": job_id, "status": "running", "lease_token": lease_token,
                         "lease_until": {"$gt": datetime.now(timezone.utc)}},
                        {"$set": {"status": "succeeded", "finished_at": self._now(),
                                  "expires_at": datetime.now(timezone.utc) + timedelta(days=30)},
                         "$unset": {"payload": "", "lease_token": ""}}, session=session)
                    if fenced.matched_count != 1:
                        raise RuntimeError("Ingest lease lost; publication rejected")
                for document in document_list:
                    chunks = list(document["chunks"])
                    document = {**document, "chunks": chunks}
                    logical_key = str(document["logical_key"])
                    previous = self.documents.find_one({"_id": logical_key}, session=session)
                    if (expected_projections and logical_key in expected_projections
                            and (previous or {}).get("active_projection_id") != expected_projections[logical_key]):
                        raise RuntimeError("Source projection changed; publication rejected")
                    if previous and previous.get("active_projection_id"):
                        self._archive_active(previous, session=session)
                    if previous and not previous.get("active_projection_id"):
                        previous["chunks"] = [FinancialChunk.model_validate(row["payload"])
                                              for row in self.chunks.find(
                                                  {"logical_key": logical_key}, session=session
                                              ).sort("ordinal", 1)]
                        self._save_projection(previous, normalize_report(previous), session=session)
                    bundle = normalize_report(document)
                    self._save_projection(document, bundle, session=session)
                    self._archive_chunks(bundle.projection_id, chunks, session=session)
                    self.chunks.delete_many({"logical_key": logical_key}, session=session)
                    record = {
                        "_id": logical_key,
                        "logical_key": logical_key,
                        "document_id": document["document_id"],
                        "source_file": document["source_file"],
                        "source_pdf": document.get("source_pdf"),
                        "company_code": document.get("company_code"),
                        "fiscal_year": document.get("fiscal_year"),
                        "fiscal_quarter": document.get("fiscal_quarter"),
                        "report_type": document.get("report_type"),
                        "raw_content": document["raw_content"],
                        "chunk_count": len(chunks),
                        "ingested_at": document.get("ingested_at") or self._now(),
                        "active_version_id": bundle.version.version_id,
                        "active_projection_id": bundle.projection_id,
                        "search_index": document.get("search_index", {"sparse": "lazy", "dense": "disabled"}),
                    }
                    self.documents.replace_one(
                        {"_id": logical_key}, record, upsert=True, session=session
                    )
                    if chunks:
                        self.chunks.insert_many(
                            [
                                {
                                    "_id": chunk.chunk_id,
                                    "logical_key": logical_key,
                                    "ordinal": ordinal,
                                    "company_code": chunk.company_code,
                                    "fiscal_year": chunk.fiscal_year,
                                    "report_type": chunk.report_type,
                                    "payload": chunk.model_dump(mode="json"),
                                }
                                for ordinal, chunk in enumerate(chunks)
                            ],
                            ordered=True,
                            session=session,
                        )
                    total_chunks += len(chunks)
                if job:
                    result_documents = []
                    for document in document_list:
                        active_document = self.documents.find_one(
                            {"_id": document["logical_key"]}, session=session
                        )
                        if active_document is None or not active_document.get("active_projection_id"):
                            raise RuntimeError("Ingested document has no active projection")
                        result_documents.append({
                            "logical_key": document["logical_key"],
                            "projection_id": active_document["active_projection_id"],
                        })
                    self.jobs.update_one({"_id": job[0], "status": "succeeded"}, {"$set": {
                        "result": {"documents": result_documents, "chunk_count": total_chunks}}},
                        session=session,
                    )
        return total_chunks

    def _archive_chunks(self, projection_id, chunks, *, session):
        from ..structure.normalized import fingerprint
        if self.retrieval_snapshots.find_one({"_id": projection_id}, session=session):
            return
        payloads = [chunk.model_dump(mode="json") for chunk in chunks]
        if payloads:
            self.archived_chunks.insert_many([
                {"_id": f"{projection_id}:{ordinal}", "projection_id": projection_id,
                 "ordinal": ordinal, "payload": payload}
                for ordinal, payload in enumerate(payloads)], session=session)
        self.retrieval_snapshots.insert_one({"_id": projection_id, "chunk_count": len(payloads),
                                             "sha256": fingerprint(payloads)}, session=session)

    def _archive_active(self, document, *, session):
        chunks = [FinancialChunk.model_validate(row["payload"]) for row in self.chunks.find(
            {"logical_key": document["_id"]}, session=session).sort("ordinal", 1)]
        from ..structure.normalized import fingerprint
        projection = self.projections.find_one({"_id": document["active_projection_id"]}, session=session)
        if projection and fingerprint([item.model_dump(mode="json") for item in chunks]) != projection["chunk_sha256"]:
            # Facts-only legacy migrations did not publish these retrieval rows.
            # Retain them under their own identity, never under the facts hash.
            source = {**document, "logical_key": document["_id"], "chunks": chunks}
            bundle = normalize_report(source, revision="notebook_persisted_retrieval_v1")
            self._save_projection(source, bundle, session=session)
            self._archive_chunks(bundle.projection_id, chunks, session=session)
        else:
            self._archive_chunks(document["active_projection_id"], chunks, session=session)

    def projection_chunks(self, logical_key: str, projection_id: str) -> list[FinancialChunk]:
        from ..structure.normalized import fingerprint
        projection = self.projections.find_one({"_id": projection_id, "scope.logical_key": logical_key})
        if not projection:
            raise ValueError("Projection does not belong to the requested document")
        manifest = self.retrieval_snapshots.find_one({"_id": projection_id})
        if not manifest:
            raise ValueError("Historical retrieval snapshot is unavailable; source remains archived")
        rows = list(self.archived_chunks.find({"projection_id": projection_id}).sort("ordinal", 1))
        payloads = [row["payload"] for row in rows]
        if len(rows) != manifest["chunk_count"] or fingerprint(payloads) != manifest["sha256"]:
            raise ValueError("Historical retrieval snapshot checksum mismatch")
        return [FinancialChunk.model_validate(payload) for payload in payloads]

    def get_projection_document(self, logical_key: str, projection_id: str) -> dict:
        import hashlib
        projection = self.projections.find_one({"_id": projection_id, "scope.logical_key": logical_key})
        if not projection:
            raise ValueError("Projection does not belong to the requested document")
        version = self.versions.find_one({"_id": projection["version_id"], "logical_key": logical_key})
        if not version:
            raise ValueError("Pinned source version is unavailable")
        if not self.retrieval_snapshots.find_one({"_id": projection_id}):
            # Upgrade a legacy active report in one consistent transaction. An
            # absent historical snapshot must never be rebuilt with today's parser.
            from ..structure.normalized import fingerprint
            with self.client.start_session() as session:
                with session.start_transaction():
                    active = self.documents.find_one({"_id": logical_key}, session=session)
                    if not active or active.get("active_projection_id") != projection_id:
                        raise ValueError("Historical retrieval snapshot is unavailable")
                    current = [FinancialChunk.model_validate(row["payload"]) for row in self.chunks.find(
                        {"logical_key": logical_key}, session=session).sort("ordinal", 1)]
                    if fingerprint([item.model_dump(mode="json") for item in current]) != projection["chunk_sha256"]:
                        raise ValueError("Active projection checksum mismatch")
                    self._archive_chunks(projection_id, current, session=session)
        chunks = self.projection_chunks(logical_key, projection_id)
        from ..structure.normalized import fingerprint
        if (fingerprint([item.model_dump(mode="json") for item in chunks]) != projection["chunk_sha256"]
                or hashlib.sha256(version["raw_content"].encode()).hexdigest() != version["source_sha256"]):
            raise ValueError("Pinned source checksum mismatch")
        return {**{key: value for key, value in version.items() if key != "_id"},
                "fiscal_year": version.get("report_year"), "active_version_id": version["version_id"],
                "active_projection_id": projection_id, "chunks": chunks}

    def pin_active_document(self, logical_key: str, expected_projection: str) -> dict:
        """Pin existing retrieval rows when a legacy facts-only migration diverged.

        Older normalized migrations intentionally left active retrieval rows
        unchanged while publishing backfilled fact coordinates. They are two
        different projections. Preserve both; do not pretend their hashes match.
        """
        try:
            return self.get_projection_document(logical_key, expected_projection)
        except ValueError as exc:
            if str(exc) not in {"Active projection checksum mismatch", "Pinned source checksum mismatch"}:
                raise
        import hashlib
        with self.client.start_session() as session:
            with session.start_transaction():
                record = self.documents.find_one({"_id": logical_key}, session=session)
                if not record or record.get("active_projection_id") != expected_projection:
                    raise ValueError("Source projection changed; reload before attaching")
                version = self.versions.find_one({"_id": record["active_version_id"]}, session=session)
                if (not version or hashlib.sha256(record["raw_content"].encode()).hexdigest() != version["source_sha256"]
                        or hashlib.sha256(version["raw_content"].encode()).hexdigest() != version["source_sha256"]):
                    raise ValueError("Active source checksum mismatch")
                chunks = [FinancialChunk.model_validate(row["payload"]) for row in self.chunks.find(
                    {"logical_key": logical_key}, session=session).sort("ordinal", 1)]
                if len(chunks) != record["chunk_count"]:
                    raise ValueError("Active retrieval chunk count mismatch")
                document = {**record, "logical_key": logical_key, "chunks": chunks}
                bundle = normalize_report(document, revision="notebook_persisted_retrieval_v1")
                self._save_projection(document, bundle, session=session)
                self._archive_chunks(bundle.projection_id, chunks, session=session)
        result = self.get_projection_document(logical_key, bundle.projection_id)
        result["catalog_projection_id"] = expected_projection
        result["projection_origin"] = "persisted_active_retrieval"
        return result

    def activate_projection(self, logical_key: str, projection_id: str) -> dict:
        chunks = self.projection_chunks(logical_key, projection_id)
        projection = self.projections.find_one({"_id": projection_id})
        if projection is None:
            raise KeyError(f"Unknown projection: {projection_id}")
        version = self.versions.find_one({"_id": projection["version_id"]})
        if version is None:
            raise KeyError(f"Projection {projection_id} has no source version")
        scope = projection.get("scope")
        if not isinstance(scope, dict):
            raise ValueError(f"Projection {projection_id} has invalid scope metadata")
        with self.client.start_session() as session:
            with session.start_transaction():
                previous = self.documents.find_one({"_id": logical_key}, session=session)
                if previous and previous.get("active_projection_id"):
                    self._archive_active(previous, session=session)
                self.chunks.delete_many({"logical_key": logical_key}, session=session)
                record = {"_id": logical_key, "logical_key": logical_key,
                          "document_id": chunks[0].document_id if chunks else logical_key,
                          "source_file": version["source_file"], "source_pdf": version.get("source_pdf"),
                          "company_code": scope.get("company_code"), "fiscal_year": scope.get("report_year"),
                          "fiscal_quarter": scope.get("fiscal_quarter"), "report_type": scope.get("report_type"),
                          "raw_content": version["raw_content"], "chunk_count": len(chunks),
                          "ingested_at": self._now(), "active_version_id": version["version_id"],
                          "active_projection_id": projection_id}
                self.documents.replace_one({"_id": logical_key}, record, upsert=True, session=session)
                if chunks:
                    self.chunks.insert_many([{"_id": chunk.chunk_id, "logical_key": logical_key,
                        "ordinal": ordinal, "company_code": chunk.company_code, "fiscal_year": chunk.fiscal_year,
                        "report_type": chunk.report_type, "payload": chunk.model_dump(mode="json")}
                        for ordinal, chunk in enumerate(chunks)], session=session)
        return {"logical_key": logical_key, "active_projection_id": projection_id, "chunk_count": len(chunks)}

    def enqueue_ingest(self, payload: dict) -> dict:
        from ..structure.normalized import fingerprint
        from pymongo.errors import DuplicateKeyError
        job_id = fingerprint(payload)
        now = datetime.now(timezone.utc)
        try:
            self.jobs.insert_one({"_id": job_id, "payload": payload, "status": "queued",
                                  "attempts": 0, "created_at": now, "lease_until": now})
        except DuplicateKeyError:
            previous = self.jobs.find_one({"_id": job_id})
            if previous and previous.get("status") == "succeeded":
                outputs = previous.get("result", {}).get("documents", [])
                active = bool(outputs) and all(self.documents.find_one(
                    {"_id": item["logical_key"], "active_projection_id": item["projection_id"]}, {"_id": 1})
                    for item in outputs)
                if not active:
                    self.jobs.update_one({"_id": job_id, "status": "succeeded"},
                        {"$set": {"status": "queued", "payload": payload, "attempts": 0, "lease_until": now},
                         "$inc": {"generation": 1}, "$unset": {"expires_at": "", "finished_at": "",
                                                                 "expected_projections": "", "stage_events": "",
                                                                 "start_projections": ""},
                         "$push": {"previous_runs": {"$each": [{"result": previous.get("result"),
                                     "finished_at": previous.get("finished_at")}], "$slice": -10}}})
        job = self.get_ingest_job(job_id)
        if job is None:
            raise RuntimeError("Ingest job was not persisted")
        return job

    def retry_ingest(self, job_id: str) -> dict | None:
        self.jobs.update_one({"_id": job_id, "status": "failed", "payload": {"$exists": True}},
            {"$set": {"status": "queued", "attempts": 0, "lease_until": datetime.now(timezone.utc)},
             "$inc": {"generation": 1}, "$unset": {"error": "", "expires_at": ""}})
        return self.get_ingest_job(job_id)

    def get_ingest_job(self, job_id: str) -> dict | None:
        record = self.jobs.find_one({"_id": job_id}, {"payload": 0, "lease_token": 0})
        return {"job_id": record.pop("_id"), **record} if record else None

    def claim_ingest(self, worker: str, lease_seconds: int = 120, max_attempts: int = 3):
        from pymongo import ReturnDocument
        now = datetime.now(timezone.utc)
        # Expired final attempts must reach a terminal state instead of hanging.
        self.jobs.update_many({"status": "running", "lease_until": {"$lte": now},
                               "attempts": {"$gte": max_attempts}},
                              {"$set": {"status": "failed", "error": "Worker lease expired",
                                        "expires_at": now + timedelta(days=30)}})
        return self.jobs.find_one_and_update(
            {"status": {"$in": ["queued", "running"]}, "lease_until": {"$lte": now},
             "attempts": {"$lt": max_attempts}},
            {"$set": {"status": "running", "worker": worker, "lease_token": str(uuid.uuid4()),
                      "lease_until": now + timedelta(seconds=lease_seconds)}, "$inc": {"attempts": 1}},
            sort=[("created_at", 1)], return_document=ReturnDocument.AFTER)

    def renew_ingest(self, job_id: str, token: str, lease_seconds: int = 120) -> bool:
        now = datetime.now(timezone.utc)
        return self.jobs.update_one({"_id": job_id, "status": "running", "lease_token": token,
                                     "lease_until": {"$gt": now}},
                                    {"$set": {"lease_until": now + timedelta(seconds=lease_seconds)}}).matched_count == 1

    def update_ingest_progress(self, job_id: str, token: str, event: dict):
        now = datetime.now(timezone.utc)
        result = self.jobs.update_one({"_id": job_id, "status": "running", "lease_token": token,
                                      "lease_until": {"$gt": now}},
            {"$set": {"stage": event["stage"], "stage_updated_at": now},
             "$push": {"stage_events": {"$each": [event], "$slice": -200}}})
        if result.matched_count != 1:
            raise RuntimeError("Ingest lease lost; progress write rejected")

    def set_ingest_expected_projections(self, job_id: str, token: str, expected: dict):
        """Keep the first attempt's CAS baseline across retries; old jobs cannot overwrite new uploads."""
        return self._set_ingest_baseline(job_id, token, "expected_projections", expected, check_keys=True)

    def set_ingest_start_projections(self, job_id: str, token: str, active: dict):
        """Snapshot pointers before parsing, including retries after an early parse failure."""
        return self._set_ingest_baseline(job_id, token, "start_projections", active)

    def _set_ingest_baseline(self, job_id, token, field, values, *, check_keys=False):
        from pymongo import ReturnDocument
        query = {"_id": job_id, "status": "running", "lease_token": token,
                 "lease_until": {"$gt": datetime.now(timezone.utc)}}
        record = self.jobs.find_one_and_update({**query, field: {"$exists": False}},
            {"$set": {field: values}}, return_document=ReturnDocument.AFTER)
        if record is None:
            record = self.jobs.find_one(query)
        if not record or field not in record:
            raise RuntimeError("Ingest lease lost; CAS baseline write rejected")
        if check_keys and set(record[field]) != set(values):
            raise RuntimeError("Ingest source identities changed; enqueue a new job")
        return record[field]

    def fail_ingest(self, job_id: str, token: str, error: str, *, max_attempts: int = 3):
        record = self.jobs.find_one({"_id": job_id, "status": "running", "lease_token": token})
        if not record:
            return
        now = datetime.now(timezone.utc)
        failed = record["attempts"] >= max_attempts
        self.jobs.update_one({"_id": job_id, "status": "running", "lease_token": token},
                            {"$set": {"status": "failed" if failed else "queued", "error": error[:2000],
                                      "lease_until": now + timedelta(seconds=2 ** record["attempts"]),
                                      "expires_at": now + timedelta(days=30)},
                             "$unset": {"lease_token": ""}})

    def _save_projection(self, document: dict, bundle: NormalizedReport, *, session) -> None:
        """Publish immutable source and projection in the caller's transaction."""
        self.versions.update_one(
            {"_id": bundle.version.version_id},
            {"$setOnInsert": {**bundle.version.model_dump(),
                              "raw_content": document["raw_content"], "created_at": self._now()}},
            upsert=True, session=session,
        )
        if self.projections.find_one({"_id": bundle.projection_id}, session=session):
            return
        for collection, records, id_field in (
            (self.tables, bundle.tables, "table_id"), (self.rows, bundle.rows, "row_id"),
            (self.facts, bundle.facts, "fact_id"),
        ):
            if records:
                collection.insert_many([{**item.model_dump(), "_id": getattr(item, id_field)}
                                        for item in records], session=session)
        self.projections.insert_one({"_id": bundle.projection_id, **bundle.manifest(),
                                     "scope": bundle.version.model_dump(),
                                     "created_at": self._now()}, session=session)

    def list_versions(self, logical_key: str) -> list[dict]:
        result = []
        for row in self.versions.find({"logical_key": logical_key}).sort("created_at", 1):
            version = {key: value for key, value in row.items() if key not in {"_id", "raw_content"}}
            version["projections"] = list(self.projections.find(
                {"version_id": version["version_id"]}, {"_id": 0, "scope": 0}).sort("created_at", 1))
            result.append(version)
        return result

    def list_facts(self, logical_key: str, *, period_year: int | None = None,
                   projection_id: str | None = None, limit: int = 100, offset: int = 0) -> dict:
        document = self.documents.find_one({"_id": logical_key}, {"active_projection_id": 1})
        active = document.get("active_projection_id") if document else None
        if projection_id:
            projection = self.projections.find_one({"_id": projection_id})
            if not projection or projection.get("scope", {}).get("logical_key") != logical_key:
                raise ValueError("Projection does not belong to the requested document")
        else:
            projection_id = active
        if not projection_id:
            return {"projection_id": None, "total": 0, "facts": []}
        projection = self.projections.find_one({"_id": projection_id}, {"_id": 0})
        query: dict = {"projection_id": projection_id}
        if period_year is not None:
            query["period_year"] = period_year
        records = list(self.facts.find(query, {"_id": 0}).sort("fact_id", 1).skip(offset).limit(limit))
        row_ids = {item["row_id"] for item in records}
        rows = {row["row_id"]: {key: value for key, value in row.items() if key != "_id"}
                for row in self.rows.find({"_id": {"$in": list(row_ids)}})}
        tables = list(self.tables.find(
            {"_id": {"$in": list({row["table_id"] for row in rows.values() if row["table_id"]})}},
            {"_id": 0}))
        return {"projection_id": projection_id, "active": projection_id == active,
                "projection": projection, "tables": tables,
                "total": self.facts.count_documents(query),
                "facts": [{**item, "row": rows[item["row_id"]]} for item in records]}

    def load_chunks(self) -> list[FinancialChunk]:
        documents = self.list_documents()
        document_order = {
            item["logical_key"]: position for position, item in enumerate(documents)
        }
        records = list(self.chunks.find(
            {}, {"logical_key": 1, "ordinal": 1, "payload": 1}
        ))
        records.sort(key=lambda row: (
            document_order.get(row["logical_key"], len(document_order)),
            int(row.get("ordinal", 0)),
        ))
        return [FinancialChunk.model_validate(row["payload"]) for row in records]

    def read_scope(self, companies: list[str], years: list[int], report_type: str | None,
                   *, source_policy: str = "source_report", cache_lookup=None,
                   max_chunks: int = 30000) -> dict:
        """Read catalog and rows at one Mongo snapshot, or reuse that exact cached version."""
        from pymongo.read_concern import ReadConcern
        if not companies:
            return {"documents": [], "chunks": [], "cached": None}
        query: dict = {"company_code": {"$in": [code.upper() for code in companies]}}
        if report_type:
            query["report_type"] = report_type
        if years:
            query["fiscal_year"] = ({"$gte": min(years)} if source_policy == "latest_restated"
                                    else {"$in": sorted(set(years) | {year + 1 for year in years})})
        with self.client.start_session() as session:
            with session.start_transaction(read_concern=ReadConcern("snapshot")):
                documents = [{"logical_key": row.pop("_id"), **row} for row in
                             self.documents.find(query, {"raw_content": 0}, session=session)
                             .sort([("ingested_at", 1), ("source_file", 1)])]
                cached = cache_lookup(documents) if cache_lookup else None
                if cached is not None:
                    return {"documents": documents, "chunks": [], "cached": cached}
                if sum(item.get("chunk_count", 0) for item in documents) > max_chunks:
                    raise ValueError("Scope exceeds chunk budget; select fewer companies or periods")
                keys = [item["logical_key"] for item in documents]
                records = self.chunks.find({"logical_key": {"$in": keys}}, session=session).sort(
                    [("logical_key", 1), ("ordinal", 1)])
                chunks = [FinancialChunk.model_validate(row["payload"]) for row in records]
        return {"documents": documents, "chunks": chunks, "cached": None}

    def list_documents(self) -> list[dict[str, Any]]:
        fields = {
            "_id": 1,
            "document_id": 1,
            "source_file": 1,
            "source_pdf": 1,
            "company_code": 1,
            "fiscal_year": 1,
            "fiscal_quarter": 1,
            "report_type": 1,
            "chunk_count": 1,
            "ingested_at": 1,
            "active_version_id": 1,
            "active_projection_id": 1,
            "search_index": 1,
        }
        records = self.documents.find({}, fields).sort([
            ("ingested_at", 1), ("source_file", 1)
        ])
        return [
            {"logical_key": record.pop("_id"), **record}
            for record in records
        ]

    def load_company_aliases(self) -> dict[str, list[str]]:
        aliases: dict[str, list[str]] = {}
        records = self.documents.find(
            {"company_code": {"$exists": True, "$nin": [None, ""]}},
            {"company_code": 1, "raw_content": {"$substrCP": [
                {"$ifNull": ["$raw_content", ""]}, 0, 20_000]}},
        ).sort([("ingested_at", 1), ("source_file", 1)]).batch_size(16)
        for record in records:
            company_code = str(record["company_code"]).strip().upper()
            legal_names = CompanyExtractor.extract_legal_name_aliases(
                str(record.get("raw_content", "") or "")[:20_000]
            )
            if not legal_names:
                continue
            company_aliases = aliases.setdefault(company_code, [])
            seen = {alias.casefold() for alias in company_aliases}
            for legal_name in legal_names:
                if legal_name.casefold() not in seen:
                    company_aliases.append(legal_name)
                    seen.add(legal_name.casefold())
        return aliases

    def delete_document(self, logical_key: str) -> bool:
        with self.client.start_session() as session:
            with session.start_transaction():
                previous = self.documents.find_one({"_id": logical_key}, session=session)
                if previous and previous.get("active_projection_id"):
                    self._archive_active(previous, session=session)
                result = self.documents.delete_one({"_id": logical_key}, session=session)
                if result.deleted_count == 0:
                    return False
                self.chunks.delete_many({"logical_key": logical_key}, session=session)
        return True

    def document_count(self) -> int:
        return int(self.documents.count_documents({}))

    def chunk_count(self) -> int:
        return int(self.chunks.count_documents({}))

    def list_conversations(self, limit: int = 24) -> list[dict[str, Any]]:
        from pymongo import ASCENDING, DESCENDING

        records = list(
            self.conversations.find({}, {"title": 1, "created_at": 1, "updated_at": 1})
            .sort([("updated_at", DESCENDING), ("_id", ASCENDING)])
            .limit(max(1, min(int(limit), 100)))
        )
        if not records:
            return []

        conversation_ids = [record["_id"] for record in records]
        turns_by_conversation: dict[str, list[dict[str, Any]]] = {
            conversation_id: [] for conversation_id in conversation_ids
        }
        turn_records = self.conversation_turns.find(
            {"conversation_id": {"$in": conversation_ids}},
            {"_id": 0, "conversation_id": 1, "turn_id": 1, "query": 1,
             "summary": 1, "bookmarked": 1, "response": 1, "error": 1},
        ).sort([("conversation_id", ASCENDING), ("ordinal", ASCENDING)])
        for turn in turn_records:
            turns_by_conversation[turn["conversation_id"]].append({
                "id": turn["turn_id"],
                "query": turn.get("query", ""),
                "summary": turn.get("summary", ""),
                "bookmarked": bool(turn.get("bookmarked", False)),
                "response": turn.get("response"),
                "error": turn.get("error", ""),
            })

        conversations = []
        for record in records:
            conversation_id = record["_id"]
            turns = turns_by_conversation[conversation_id]
            if not turns:
                continue
            conversations.append({
                "id": conversation_id,
                "title": record.get("title", "Cuộc trò chuyện"),
                "createdAt": record.get("created_at", ""),
                "updatedAt": record.get("updated_at", ""),
                "turns": turns,
            })
        return conversations

    def save_conversation(self, conversation: dict[str, Any]) -> None:
        """Atomically replace one conversation and its bounded turn history."""
        conversation_id = str(conversation["id"])
        turns = list(conversation["turns"])
        with self.client.start_session() as session:
            with session.start_transaction():
                self.conversations.replace_one(
                    {"_id": conversation_id},
                    {
                        "_id": conversation_id,
                        "title": conversation.get("title", "Cuộc trò chuyện"),
                        "created_at": conversation.get("createdAt", ""),
                        "updated_at": conversation.get("updatedAt", ""),
                    },
                    upsert=True,
                    session=session,
                )
                self.conversation_turns.delete_many(
                    {"conversation_id": conversation_id}, session=session
                )
                if turns:
                    self.conversation_turns.insert_many(
                        [
                            {
                                "_id": f"{conversation_id}:{turn['id']}",
                                "conversation_id": conversation_id,
                                "turn_id": str(turn["id"]),
                                "ordinal": ordinal,
                                "query": turn.get("query", ""),
                                "summary": turn.get("summary", ""),
                                "bookmarked": bool(turn.get("bookmarked", False)),
                                "response": turn.get("response"),
                                "error": turn.get("error", ""),
                            }
                            for ordinal, turn in enumerate(turns)
                        ],
                        ordered=True,
                        session=session,
                    )

                # Match the UI retention limit and remove turns with their parent.
                stale = list(
                    self.conversations.find({}, {"_id": 1}, session=session)
                    .sort([("updated_at", -1), ("_id", 1)])
                    .skip(24)
                )
                stale_ids = [record["_id"] for record in stale]
                if stale_ids:
                    self.conversation_turns.delete_many(
                        {"conversation_id": {"$in": stale_ids}}, session=session
                    )
                    self.conversations.delete_many(
                        {"_id": {"$in": stale_ids}}, session=session
                    )

    def delete_conversation(self, conversation_id: str) -> bool:
        with self.client.start_session() as session:
            with session.start_transaction():
                result = self.conversations.delete_one({"_id": conversation_id}, session=session)
                self.conversation_turns.delete_many(
                    {"conversation_id": conversation_id}, session=session
                )
        return result.deleted_count > 0

    def close(self) -> None:
        self.client.close()
