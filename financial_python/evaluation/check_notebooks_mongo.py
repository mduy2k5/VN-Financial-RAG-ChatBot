"""Notebook integration against a real replica set, using an isolated temporary DB."""
import importlib
import json
import os
from pathlib import Path
from uuid import uuid4

from financial_rag.notebooks.api import NotebookQuery
from financial_rag.notebooks.records import NotebookError
from financial_rag.notebooks.service import NotebookService
from financial_rag.schemas import ChunkType, FinancialChunk, StatementType
from financial_rag.storage.mongo_store import MongoFinancialStore
from financial_rag.structure.normalized import normalize_report, fingerprint


def report(value):
    return {"logical_key": "notebook_fixture", "document_id": "fixture", "source_file": "fixture.md",
        "company_code": "FPT", "fiscal_year": 2025, "report_type": "consolidated",
        "raw_content": f"# FPT 2025\n\n## Bảng cân đối\n\nĐơn vị: Triệu đồng\n\n"
                       f"| Chỉ tiêu | 2025 | 2024 |\n| --- | --- | --- |\n| Tổng tài sản | {value} | 80 |",
        "chunks": [FinancialChunk(chunk_id="fixture_row", document_id="fixture", source_file="fixture.md",
            company_code="FPT", fiscal_year=2025, report_type="consolidated", unit="Triệu đồng",
            comparative_fiscal_year=2024, chunk_type=ChunkType.FINANCIAL_ITEM,
            statement_type=StatementType.BALANCE_SHEET,
            content=f"Tổng tài sản\nNăm nay (2025): {value} Triệu đồng\nNăm trước (2024): 80 Triệu đồng",
            metadata={"metric_name": "Tổng tài sản", "end_value": value, "start_value": "80",
                      "period_values": {"2025": value, "2024": "80"}})]}


def main():
    runtime = importlib.import_module("financial_rag.api.app")
    database_name = "financial_notebook_test_" + uuid4().hex
    store = MongoFinancialStore(os.environ["MONGODB_URI"], database_name)
    os.environ["FINANCIAL_ENABLE_DENSE"] = "false"
    runtime.document_store = store
    checks = []
    try:
        service = NotebookService(store, runtime)
        store.replace_documents([report("100")])
        legacy_projection = store.list_documents()[0]["active_projection_id"]
        store.retrieval_snapshots.delete_one({"_id": legacy_projection})
        store.archived_chunks.delete_many({"projection_id": legacy_projection})
        notebook = service.create("Notebook integration")
        notebook = service.attach(notebook["id"], "notebook_fixture", notebook["revision"])
        assert store.retrieval_snapshots.find_one({"_id": legacy_projection})
        checks.append("verified_legacy_active_snapshot_upgrade")
        store.replace_documents([report("200")])
        response = service.answer(notebook["id"], NotebookQuery(query="Tổng tài sản FPT năm 2025 hợp nhất?",
                                                               notebook_revision=notebook["revision"]))
        assert "100" in response["answer"] and "200" not in response["answer"]
        checks.append("pinned_financial_values_after_replacement")
        source_id = notebook["sources"][0]["id"]
        citation = response["citations"][0]
        assert citation["block_id"]
        content = service.content(notebook["id"], source_id, run_id=response["run_id"], block_id=citation["block_id"])
        assert "100" in content["text"] and "Triệu đồng" in content["text"]
        checks.append("citation_opens_original_atomic_table_with_unit")
        try:
            service.attach(notebook["id"], "notebook_fixture", 1)
        except NotebookError as exc:
            assert exc.status == 409
        else:
            raise AssertionError("CAS was not enforced")
        checks.append("notebook_revision_conflict")
        service.save_note(notebook["id"], "Saved answer", "", response["run_id"])
        notebook = service.attach(notebook["id"], "notebook_fixture", notebook["revision"])
        assert service.notes(notebook["id"])[0]["stale"]
        checks.append("notes_marked_stale_after_explicit_version_adoption")
        assert store.delete_document("notebook_fixture")
        reopened = NotebookService(store, runtime)
        assert "100" in reopened.content(notebook["id"], source_id, run_id=response["run_id"])["text"]
        assert reopened.conversation(notebook["id"], response["conversation_id"])["turns"]
        checks.append("saved_evidence_survives_document_deletion_and_service_recreation")
        other = service.create("Other")
        try:
            service.require("run", response["run_id"], other["id"])
        except NotebookError as exc:
            assert exc.status == 404
        else:
            raise AssertionError("Foreign run was accepted")
        checks.append("cross_notebook_run_rejected")
        store.replace_documents([report("200")])
        # Reproduce the earlier facts-only migration without altering active rows.
        fact_document = report("200")
        fact_document["chunks"][0].metadata["source_row_index"] = 0
        fact_bundle = normalize_report(fact_document, revision="legacy_facts_only")
        with store.client.start_session() as session:
            with session.start_transaction():
                store._save_projection(fact_document, fact_bundle, session=session)
                store.documents.update_one({"_id": "notebook_fixture"}, {"$set": {
                    "active_projection_id": fact_bundle.projection_id}}, session=session)
                # Some old replacement paths also archived unchanged retrieval
                # rows under the facts-only projection. Its hash must not be reused.
                store._archive_chunks(fact_bundle.projection_id, report("200")["chunks"], session=session)
        before = fingerprint([row["payload"] for row in store.chunks.find({}).sort("ordinal", 1)])
        recovered = service.attach(other["id"], "notebook_fixture", other["revision"])
        attachment = recovered["sources"][0]
        assert attachment["projection_origin"] == "persisted_active_retrieval"
        assert attachment["catalog_projection_id"] == fact_bundle.projection_id
        assert store.documents.find_one({"_id": "notebook_fixture"})["active_projection_id"] == fact_bundle.projection_id
        assert fingerprint([row["payload"] for row in store.chunks.find({}).sort("ordinal", 1)]) == before
        assert service.source(other["id"], attachment["id"])[1]["chunks"][0].metadata["end_value"] == "200"
        checks.append("facts_only_legacy_projection_reconciled_without_changing_active_rows_or_pointer")
        store.replace_documents([report("300")])
        assert service.source(other["id"], attachment["id"])[1]["chunks"][0].metadata["end_value"] == "200"
        checks.append("legacy_replacement_keeps_separate_retrieval_projection")
        payload = {"passed": True, "backend": "MongoDB replica set", "checks": checks,
                   "test_database": database_name, "production_documents_modified": False}
        output = os.environ.get("NOTEBOOK_CHECK_OUTPUT")
        if output:
            Path(output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload))
    finally:
        assert database_name.startswith("financial_notebook_test_") and database_name != "financial_rag"
        store.client.drop_database(database_name)
        store.client.close()


if __name__ == "__main__":
    main()
