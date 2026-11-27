"""Tutor router — question-scoped explanations (issue #15).

Endpoint contract:
    POST /api/questions/{question_id}/explain
    body: {"action": "simpler|deeper|why|compare|example|follow_up",
           "question_text": "...", "answer_context": "...",
           "source_section": "...", "source_quote": "...",
           "user_answer": "...", "follow_up_prompt": "...", "compare_to": "..."}
    → 200 TutorExplainResponse

Cost cap: ``TUTOR_COST_CAP_PER_QUESTION`` explanations per
``(user_id, question_id)``; exceeding it returns 429. The tutor never
alters the canonical answer or any completion state.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.config import get_settings
from app.dependencies import require_auth
from app.schemas.models import TutorExplainRequest, TutorExplainResponse
from app.services.tutor_service import TutorCostExceededError, TutorService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/questions", tags=["tutor"])

_service: TutorService | None = None


def init(service: TutorService):
    global _service
    _service = service


def _ensure_enabled() -> None:
    if not get_settings().enable_tutor_v1:
        raise HTTPException(status_code=404, detail="Tutor API disabled")


def _ensure_ready() -> TutorService:
    if _service is None:
        raise HTTPException(status_code=503, detail="Tutor service not initialised")
    return _service


@router.post("/{question_id}/explain", response_model=TutorExplainResponse)
async def explain_question(
    question_id: str,
    body: TutorExplainRequest,
    user: dict = Depends(require_auth),
):
    _ensure_enabled()
    service = _ensure_ready()
    try:
        return await service.explain(
            user_id=str(user["user"]),
            question_id=question_id,
            action=body.action.value,
            question_text=body.question_text,
            answer_context=body.answer_context,
            source_section=body.source_section,
            source_quote=body.source_quote,
            user_answer=body.user_answer,
            follow_up_prompt=body.follow_up_prompt,
            compare_to=body.compare_to,
            llm_config=body.llm_config,
            user_identity=user,
        )
    except TutorCostExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
