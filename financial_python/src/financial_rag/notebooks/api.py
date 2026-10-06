"""Notebook APIs. Source IDs are capabilities only within the requested notebook.

Authentication/tenant authorization remains the deployment gateway's concern,
as with the existing private financial endpoints.
"""
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, StringConstraints

from .records import NotebookError
from .service import NotebookService

Short = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
RecordID = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9_-]{1,128}$")]
Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4096)]


class CreateNotebook(BaseModel):
    title: Short
    instructions: str = Field(default="", max_length=4096)


class UpdateNotebook(CreateNotebook):
    expected_revision: int = Field(ge=1)


class AttachSource(BaseModel):
    logical_key: str = Field(min_length=1, max_length=1024)
    expected_revision: int = Field(ge=1)


class NotebookQuery(BaseModel):
    query: Question
    notebook_revision: int = Field(ge=1)
    source_ids: list[RecordID] | None = Field(default=None, max_length=100)
    conversation_id: RecordID | None = None
    conversation_revision: int | None = Field(default=None, ge=1)
    top_k: int = Field(default=5, ge=1, le=100)
    mode: Literal["auto", "reader", "summary", "financial"] = "auto"
    source_policy: Literal["source_report", "latest_restated"] = "source_report"


class CreateNote(BaseModel):
    title: Short
    content: str = Field(default="", max_length=100000)
    run_id: RecordID | None = None


def create_router(runtime):
    router = APIRouter(prefix="/api/v1/notebooks", tags=["notebooks"])
    state = {"store": None, "service": None}

    def service():
        if runtime.document_store is None:
            raise HTTPException(503, "Notebook cần kho MongoDB hoặc SQLite bền vững.")
        if state["store"] is not runtime.document_store:
            state["service"] = NotebookService(runtime.document_store, runtime)
            state["store"] = runtime.document_store
        return state["service"]

    def execute(call):
        try:
            return call(service())
        except NotebookError as exc:
            raise HTTPException(exc.status, exc.detail) from exc

    @router.get("")
    def list_notebooks():
        return execute(lambda svc: {"notebooks": svc.records.list("notebook")})

    @router.post("", status_code=201)
    def create(request: CreateNotebook):
        return execute(lambda svc: svc.create(request.title, request.instructions))

    @router.get("/{notebook_id}")
    def read(notebook_id: RecordID):
        return execute(lambda svc: svc.notebook(notebook_id))

    @router.patch("/{notebook_id}")
    def update(notebook_id: RecordID, request: UpdateNotebook):
        return execute(lambda svc: svc.update(notebook_id, request.expected_revision,
                                              title=request.title, instructions=request.instructions))

    @router.post("/{notebook_id}/sources")
    def attach(notebook_id: RecordID, request: AttachSource):
        return execute(lambda svc: svc.attach(notebook_id, request.logical_key, request.expected_revision))

    @router.delete("/{notebook_id}/sources/{source_id}")
    def remove(notebook_id: RecordID, source_id: RecordID, expected_revision: int = Query(ge=1)):
        return execute(lambda svc: svc.remove(notebook_id, source_id, expected_revision))

    @router.get("/{notebook_id}/sources/{source_id}/guide")
    def guide(notebook_id: RecordID, source_id: RecordID):
        return execute(lambda svc: svc.guide(notebook_id, source_id))

    @router.get("/{notebook_id}/sources/{source_id}/content")
    def content(notebook_id: RecordID, source_id: RecordID, run_id: RecordID | None = None,
                block_id: RecordID | None = None, start: int = Query(default=0, ge=0),
                limit: int = Query(default=16000, ge=1, le=100000)):
        return execute(lambda svc: svc.content(notebook_id, source_id, run_id=run_id,
                        block_id=block_id, start=start, limit=limit))

    @router.post("/{notebook_id}/answer")
    def answer(notebook_id: RecordID, request: NotebookQuery):
        return execute(lambda svc: svc.answer(notebook_id, request))

    @router.get("/{notebook_id}/conversations")
    def conversations(notebook_id: RecordID):
        def listing(svc):
            svc.notebook(notebook_id)
            return {"conversations": [{key: value for key, value in item.items() if key != "last_scope"}
                                      for item in svc.records.list("conversation", notebook_id)]}
        return execute(listing)

    @router.get("/{notebook_id}/conversations/{conversation_id}")
    def conversation(notebook_id: RecordID, conversation_id: RecordID):
        return execute(lambda svc: svc.conversation(notebook_id, conversation_id))

    @router.get("/{notebook_id}/runs/{run_id}")
    def run(notebook_id: RecordID, run_id: RecordID):
        return execute(lambda svc: svc.require("run", run_id, notebook_id))

    @router.get("/{notebook_id}/notes")
    def notes(notebook_id: RecordID):
        return execute(lambda svc: {"notes": svc.notes(notebook_id)})

    @router.post("/{notebook_id}/notes", status_code=201)
    def save_note(notebook_id: RecordID, request: CreateNote):
        return execute(lambda svc: svc.save_note(notebook_id, request.title, request.content, request.run_id))

    return router
