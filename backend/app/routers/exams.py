"""Timed exam router (STUDY_ENABLE_EXAMS_V1).

Endpoints:

- ``POST   /api/exams``              create an exam (questions pre-selected)
- ``GET    /api/exams``              exam history
- ``GET    /api/exams/{exam_id}``    exam detail + graded report
- ``POST   /api/exams/{exam_id}/start``   stamp the server-side start
- ``POST   /api/exams/{exam_id}/submit``  grade answers (server timestamps)
- ``POST   /api/exams/{exam_id}/remark``  deterministic re-grade

Server-side ``started_at``/``completed_at`` are
authoritative; the client timer is never trusted.
Late submits are flagged ``time_exceeded`` but still
graded. Exam size is capped at 50 questions.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.config import get_settings
from app.dependencies import require_auth
from app.schemas.models import (
    ExamCreateRequest,
    ExamCreateResponse,
    ExamDetailResponse,
    ExamListResponse,
    ExamRemarkResponse,
    ExamStartResponse,
    ExamSubmitRequest,
    ExamSubmitResponse,
)
from app.services.exam_service import ExamError, ExamService
from app.services.exam_store import ExamStore

logger = logging.getLogger(__name__)

exams_router = APIRouter(prefix="/api/exams", tags=["exams"])

_exam_store: ExamStore | None = None
_exam_service: ExamService | None = None


def init(exam_store: ExamStore, exam_service: ExamService) -> None:
    global _exam_store, _exam_service
    _exam_store = exam_store
    _exam_service = exam_service


def _ensure_enabled() -> None:
    if not get_settings().enable_exams_v1:
        raise HTTPException(
            status_code=503,
            detail="Exams disabled (STUDY_ENABLE_EXAMS_V1=false)",
        )


def _require_service() -> ExamService:
    if _exam_service is None:
        raise HTTPException(status_code=503, detail="Exam service not initialised")
    return _exam_service


def _to_http_error(exc: ExamError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=exc.message)


@exams_router.post("", response_model=ExamCreateResponse, status_code=201)
async def create_exam(
    body: ExamCreateRequest,
    user: dict = Depends(require_auth),
):
    """Create a timed exam with pre-selected questions."""
    _ensure_enabled()
    service = _require_service()
    try:
        exam = await service.create_exam(
            user_id=user["user"],
            title=body.title,
            topic_ids=list(body.source.topic_ids),
            deck_ids=list(body.source.deck_ids),
            weak_areas=body.source.weak_areas,
            question_count=body.question_count,
            duration_minutes=body.duration_minutes,
            question_types=list(body.question_types),
        )
    except ExamError as exc:
        raise _to_http_error(exc) from exc
    return ExamCreateResponse(
        exam_id=str(exam["exam_id"]),
        title=str(exam["title"]),
        status=str(exam["status"]),
        question_count=len(exam["questions"]),
        duration_minutes=int(exam["duration_minutes"]),
        questions=exam["questions"],
        created_at=str(exam["created_at"]),
    )


@exams_router.get("", response_model=ExamListResponse)
async def list_exams(user: dict = Depends(require_auth)):
    """Exam history for the authenticated user."""
    _ensure_enabled()
    service = _require_service()
    exams = await service.list_exams(user_id=user["user"])
    return ExamListResponse(
        exams=[
            {
                "exam_id": str(exam["exam_id"]),
                "title": str(exam["title"]),
                "status": str(exam["status"]),
                "question_count": len(exam["questions"]),
                "duration_minutes": int(exam["duration_minutes"]),
                "score": exam["score"],
                "passed": exam["passed"],
                "time_exceeded": bool(exam["time_exceeded"]),
                "started_at": exam["started_at"],
                "completed_at": exam["completed_at"],
                "created_at": str(exam["created_at"]),
            }
            for exam in exams
        ],
        total=len(exams),
    )


@exams_router.get("/{exam_id}", response_model=ExamDetailResponse)
async def get_exam(exam_id: str, user: dict = Depends(require_auth)):
    """Exam detail with the graded report."""
    _ensure_enabled()
    service = _require_service()
    try:
        exam = await service.get_exam(user_id=user["user"], exam_id=exam_id)
    except ExamError as exc:
        raise _to_http_error(exc) from exc
    return ExamDetailResponse(
        exam_id=str(exam["exam_id"]),
        title=str(exam["title"]),
        status=str(exam["status"]),
        question_count=len(exam["questions"]),
        duration_minutes=int(exam["duration_minutes"]),
        score=exam["score"],
        passed=exam["passed"],
        time_exceeded=bool(exam["time_exceeded"]),
        started_at=exam["started_at"],
        completed_at=exam["completed_at"],
        created_at=str(exam["created_at"]),
        questions=exam["questions"],
        results=exam.get("results") or [],
    )


@exams_router.post("/{exam_id}/start", response_model=ExamStartResponse)
async def start_exam(exam_id: str, user: dict = Depends(require_auth)):
    """Start the exam; the server stamps ``started_at``."""
    _ensure_enabled()
    service = _require_service()
    try:
        exam = await service.start_exam(user_id=user["user"], exam_id=exam_id)
    except ExamError as exc:
        raise _to_http_error(exc) from exc
    return ExamStartResponse(
        exam_id=str(exam["exam_id"]),
        started_at=str(exam["started_at"]),
        duration_minutes=int(exam["duration_minutes"]),
        questions=exam["questions"],
    )


@exams_router.post("/{exam_id}/submit", response_model=ExamSubmitResponse)
async def submit_exam(
    exam_id: str,
    body: ExamSubmitRequest,
    user: dict = Depends(require_auth),
):
    """Submit answers; the server stamps ``completed_at`` and grades."""
    _ensure_enabled()
    service = _require_service()
    answers: list[dict[str, Any]] = [
        {
            "question_ref": answer.question_ref,
            "answer_text": answer.answer_text,
            "time_ms": answer.time_ms,
        }
        for answer in body.answers
    ]
    try:
        result = await service.submit_exam(
            user_id=user["user"],
            exam_id=exam_id,
            answers=answers,
            user_identity=user,
        )
    except ExamError as exc:
        raise _to_http_error(exc) from exc
    return ExamSubmitResponse(**result)


@exams_router.post("/{exam_id}/remark", response_model=ExamRemarkResponse)
async def remark_exam(exam_id: str, user: dict = Depends(require_auth)):
    """Re-grade with an identical rubric and a fixed seed."""
    _ensure_enabled()
    service = _require_service()
    try:
        result = await service.remark_exam(
            user_id=user["user"],
            exam_id=exam_id,
            user_identity=user,
        )
    except ExamError as exc:
        raise _to_http_error(exc) from exc
    return ExamRemarkResponse(**result)
