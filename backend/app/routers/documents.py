"""Documents router — upload, list, inspect, delete study documents."""

from __future__ import annotations

import logging

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
)

from app.config import get_settings
from app.dependencies import require_auth
from app.schemas.models import (
    DocumentDetail,
    DocumentListResponse,
    DocumentUploadResponse,
)
from app.services.document_pipeline import (
    MAX_FILE_SIZE,
    DocumentPipeline,
    DocumentProcessingError,
    resolve_document_kind,
)
from app.services.document_store import DocumentStore
from app.services.learning_store import LearningStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

_document_store: DocumentStore | None = None
_pipeline: DocumentPipeline | None = None
_learning_store: LearningStore | None = None


def init(
    document_store: DocumentStore,
    pipeline: DocumentPipeline,
    learning_store: LearningStore | None = None,
):
    global _document_store, _pipeline, _learning_store
    _document_store = document_store
    _pipeline = pipeline
    _learning_store = learning_store


def _require_rag_enabled() -> None:
    settings = get_settings()
    if not settings.enable_rag_v1:
        raise HTTPException(
            status_code=503,
            detail=(
                "Document RAG is not enabled "
                "(STUDY_ENABLE_RAG_V1=false)"
            ),
        )


def _require_services() -> tuple[DocumentStore, DocumentPipeline]:
    if _document_store is None or _pipeline is None:
        raise HTTPException(status_code=503, detail="Service not initialised")
    return _document_store, _pipeline


def _title_for(filename: str) -> str:
    name = (filename or "document").strip() or "document"
    if "." in name:
        name = name.rsplit(".", 1)[0]
    return (name.strip() or "document")[:200]


@router.post("", response_model=DocumentUploadResponse, status_code=202)
async def upload_document(
    file: UploadFile = File(...),
    user: dict = Depends(require_auth),
):
    """Upload a study document for chunking and embedding."""
    _require_rag_enabled()
    store, pipeline = _require_services()
    settings = get_settings()

    filename = (file.filename or "").strip() or "document"
    mime_type = (file.content_type or "").strip().lower()

    # Validate the file type against the allowlist.
    try:
        resolve_document_kind(filename, mime_type)
    except DocumentProcessingError as e:
        raise HTTPException(status_code=415, detail=str(e))

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_FILE_SIZE // (1024 * 1024)} MB limit",
        )

    # Per-user daily cap, enforced via the feature_events table.
    daily_cap = int(settings.documents_daily_cap)
    used_today = await store.run_async(
        store.count_uploads_today, user_id=user["user"]
    )
    if used_today >= daily_cap:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Daily document limit reached ({daily_cap} per day)"
            ),
        )

    document = await store.run_async(
        store.create_document,
        user_id=user["user"],
        filename=filename,
        mime_type=mime_type or "application/octet-stream",
        title=_title_for(filename),
    )
    await store.run_async(
        store.update_document_status,
        user_id=user["user"],
        document_id=document["document_id"],
        status="processing",
    )
    await store.run_async(
        store.record_upload_event,
        user_id=user["user"],
        document_id=document["document_id"],
    )
    pipeline.schedule(
        user_id=user["user"],
        document_id=document["document_id"],
        filename=filename,
        mime_type=mime_type or "application/octet-stream",
        data=data,
    )
    return DocumentUploadResponse(
        document_id=document["document_id"],
        status="processing",
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(user: dict = Depends(require_auth)):
    """List the user's documents with status and chunk counts."""
    _require_rag_enabled()
    store, _ = _require_services()
    documents = await store.run_async(
        store.list_documents, user_id=user["user"]
    )
    return DocumentListResponse(
        documents=[DocumentDetail(**doc) for doc in documents]
    )


@router.get("/{document_id}", response_model=DocumentDetail)
async def get_document(
    document_id: str,
    user: dict = Depends(require_auth),
):
    """Return document metadata and any processing error."""
    _require_rag_enabled()
    store, _ = _require_services()
    document = await store.run_async(
        store.get_document,
        user_id=user["user"],
        document_id=document_id,
    )
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentDetail(**document)


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: str,
    user: dict = Depends(require_auth),
):
    """Delete a document along with its chunks and vectors."""
    _require_rag_enabled()
    store, _ = _require_services()
    deleted = await store.run_async(
        store.delete_document,
        user_id=user["user"],
        document_id=document_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
