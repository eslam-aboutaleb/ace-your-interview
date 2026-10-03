"""Hints router — four-level hints for quiz and interview questions.

Endpoint contract:
    POST /api/questions/{question_id}/hint?level=1..4
    body (optional): {"question_text": "...", "answer_context": "..."}
    → 200 {"question_id", "level", "hint", "hints_used", "hints_remaining"}

Cost cap: 4 generated hints per ``(user_id, question_id)``
enforced via ``feature_events``; exceeding it returns 429.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from app.config import get_settings
from app.dependencies import require_auth
from app.schemas.models import HintRequest, HintResponse
from app.services.hint_service import HintCostExceededError, HintService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/questions", tags=["hints"])

_service: HintService | None = None


def init(service: HintService):
    global _service
    _service = service


def _ensure_enabled() -> None:
    if not get_settings().enable_interview_plus_v1:
        raise HTTPException(status_code=404, detail="Interview personalization disabled")


def _ensure_ready() -> HintService:
    if _service is None:
        raise HTTPException(status_code=503, detail="Hint service not initialised")
    return _service


@router.post("/{question_id}/hint", response_model=HintResponse)
async def get_hint(
    question_id: str,
    level: int = Query(default=1, ge=1, le=4),
    body: HintRequest | None = None,
    user: dict = Depends(require_auth),
):
    _ensure_enabled()
    service = _ensure_ready()

    question_text = (body.question_text if body else "") or ""
    answer_context = (body.answer_context if body else "") or ""

    try:
        payload = await service.generate_hint(
            user_id=user["user"],
            question_id=question_id,
            level=level,
            question_text=question_text,
            answer_context=answer_context,
        )
    except HintCostExceededError as exc:
        raise HTTPException(
            status_code=429,
            detail={
                "code": "hint_budget_exhausted",
                "message": str(exc),
                "question_id": exc.question_id,
                "hints_used": exc.used,
                "hints_remaining": 0,
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return HintResponse(**payload)
