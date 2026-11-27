"""Generation jobs router (issue #16).

Endpoint contract (flag STUDY_ENABLE_GENERATION_JOBS_V1, default off):
    POST /api/generation-jobs        {idempotency_key, outline} -> 201 {created, job}
    GET  /api/generation-jobs        -> {jobs}
    GET  /api/generation-jobs/{id}   -> {job}
    POST /api/generation-jobs/{id}/cancel
    POST /api/generation-jobs/{id}/retry

Submission is idempotent per (user, idempotency_key): a retried POST
returns the original job with ``created=false`` and is never re-charged.
Accepted items are persisted before the next item is attempted, so a
crash, timeout or retry never loses or duplicates accepted work.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.config import get_settings
from app.dependencies import require_auth
from app.schemas.generation import (
    GenerationJobCreateRequest,
    GenerationJobCreateResponse,
    GenerationJobGetResponse,
    GenerationJobListResponse,
)
from app.services.generation_jobs import (
    GenerationJobNotFoundError,
    GenerationJobStateError,
    GenerationOrchestrator,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/generation-jobs", tags=["generation-jobs"])

_orchestrator: GenerationOrchestrator | None = None


def init(orchestrator: GenerationOrchestrator):
    global _orchestrator
    _orchestrator = orchestrator


def _ensure_enabled() -> None:
    if not get_settings().enable_generation_jobs_v1:
        raise HTTPException(status_code=404, detail="Generation jobs API disabled")


def _ensure_ready() -> GenerationOrchestrator:
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Generation orchestrator not initialised")
    return _orchestrator


def _map_error(exc: Exception) -> HTTPException:
    if isinstance(exc, GenerationJobNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, GenerationJobStateError):
        return HTTPException(status_code=409, detail=str(exc))
    return HTTPException(status_code=500, detail="generation job operation failed")


@router.post("", status_code=201, response_model=GenerationJobCreateResponse)
async def create_generation_job(
    body: GenerationJobCreateRequest, user: dict = Depends(require_auth)
):
    _ensure_enabled()
    orchestrator = _ensure_ready()
    user_id = str(user["user"])
    job, created = orchestrator.store.create_job(
        user_id=user_id,
        idempotency_key=body.idempotency_key,
        outline=[str(item) for item in body.outline],
    )
    if created and job["outline"]:
        job = await orchestrator.run_job(user_id=user_id, job_id=job["id"])
    return {"created": created, "job": job}


@router.get("", response_model=GenerationJobListResponse)
async def list_generation_jobs(user: dict = Depends(require_auth)):
    _ensure_enabled()
    orchestrator = _ensure_ready()
    return {"jobs": orchestrator.store.list_jobs(user_id=str(user["user"]))}


@router.get("/{job_id}", response_model=GenerationJobGetResponse)
async def get_generation_job(job_id: str, user: dict = Depends(require_auth)):
    _ensure_enabled()
    orchestrator = _ensure_ready()
    try:
        return {"job": orchestrator.store.get_job(user_id=str(user["user"]), job_id=job_id)}
    except Exception as exc:
        raise _map_error(exc) from exc


@router.post("/{job_id}/cancel", response_model=GenerationJobGetResponse)
async def cancel_generation_job(job_id: str, user: dict = Depends(require_auth)):
    _ensure_enabled()
    orchestrator = _ensure_ready()
    try:
        job = orchestrator.store.cancel(user_id=str(user["user"]), job_id=job_id)
        return {"job": job}
    except Exception as exc:
        raise _map_error(exc) from exc


@router.post("/{job_id}/retry", response_model=GenerationJobGetResponse)
async def retry_generation_job(job_id: str, user: dict = Depends(require_auth)):
    _ensure_enabled()
    orchestrator = _ensure_ready()
    try:
        job = await orchestrator.retry_job(user_id=str(user["user"]), job_id=job_id)
        return {"job": job}
    except Exception as exc:
        raise _map_error(exc) from exc
