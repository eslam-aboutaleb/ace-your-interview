"""Placement router — adaptive onboarding that seeds the learner profile.

Endpoints:

- ``POST /api/learning/placement/start``   open a placement session
- ``POST /api/learning/placement/answer``  grade + step the level ladder
- ``POST /api/learning/placement/finish``  seed the learner profile
- ``GET  /api/learning/placement/status``  latest completed placement

Every answer is graded deterministically (selected label == correct
label); no model is asked to grade a placement answer. The gate mirrors
``routers/learning.py``: the adaptive-learning flag must be on.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.config import get_settings
from app.dependencies import require_auth
from app.schemas.models import (
    PlacementAnswerRequest,
    PlacementAnswerResponse,
    PlacementFinishRequest,
    PlacementFinishResponse,
    PlacementStartRequest,
    PlacementStartResponse,
    PlacementStatusResponse,
)
from app.services.placement_service import (
    PlacementAnswerError,
    PlacementService,
    PlacementStateError,
    PlacementUnavailableError,
)

router = APIRouter(prefix="/api/learning/placement", tags=["learning"])

logger = logging.getLogger(__name__)

_service: PlacementService | None = None


def init(
    planner,
    parser=None,
    question_generator=None,
    gamification_service=None,
    learning_store=None,
) -> None:
    global _service
    _service = PlacementService(
        planner=planner,
        parser=parser,
        question_generator=question_generator,
        gamification_service=gamification_service,
        learning_store=learning_store,
    )


def get_service() -> PlacementService | None:
    return _service


def _ensure_enabled() -> None:
    if not get_settings().enable_adaptive_learning:
        raise HTTPException(status_code=404, detail="Adaptive learning disabled")


def _require_service() -> PlacementService:
    if _service is None:
        raise HTTPException(status_code=503, detail="Placement service not initialised")
    return _service


@router.post("/start", response_model=PlacementStartResponse)
async def start_placement(
    body: PlacementStartRequest, user: dict = Depends(require_auth)
):
    """Open a placement session and return the first question (level ``mid``)."""
    _ensure_enabled()
    service = _require_service()
    try:
        return await service.start_session(
            user_id=user["user"],
            track=body.track.value,
            target_role=body.target_role,
            daily_goal=body.daily_goal,
            user_identity=user,
        )
    except PlacementUnavailableError as exc:
        logger.warning("placement start failed user=%s: %s", user["user"], exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/answer", response_model=PlacementAnswerResponse)
async def answer_placement(
    body: PlacementAnswerRequest, user: dict = Depends(require_auth)
):
    """Grade one answer deterministically and step the level ladder."""
    _ensure_enabled()
    service = _require_service()
    try:
        return await service.answer(
            user_id=user["user"],
            session_id=body.session_id,
            question_id=body.question_id,
            selected_answer=body.selected_answer,
            response_time_ms=body.response_time_ms,
            user_identity=user,
        )
    except PlacementAnswerError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PlacementStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/finish", response_model=PlacementFinishResponse)
async def finish_placement(
    body: PlacementFinishRequest, user: dict = Depends(require_auth)
):
    """Derive starting levels from the walked path and seed the profile."""
    _ensure_enabled()
    service = _require_service()
    try:
        return await service.finish(
            user_id=user["user"], session_id=body.session_id
        )
    except PlacementStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/status", response_model=PlacementStatusResponse)
async def placement_status(user: dict = Depends(require_auth)):
    """Latest completed placement, so the dashboard can hide the CTA."""
    _ensure_enabled()
    service = _require_service()
    return await service.run_async(
        service.get_completion, user_id=user["user"]
    )
