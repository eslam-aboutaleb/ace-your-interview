"""Adaptive learning router — attempts, review queue, weak areas."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.config import get_settings
from app.dependencies import require_auth
from app.schemas.models import (
    LearnerProfile,
    LearnerProfileUpdateRequest,
    LearningAttemptRequest,
    LearningAttemptResponse,
    ProfileDiagnosticResponse,
    RecommendationsResponse,
    ReviewQueueResponse,
    StudyPlanResponse,
    TopicMasteryResponse,
    WeakAreasResponse,
)
from app.services.learning_planner import LearningPlannerStore
from app.services.learning_store import LearningStore

router = APIRouter(prefix="/api/learning", tags=["learning"])

_store: LearningStore | None = None
_planner: LearningPlannerStore | None = None


def init(store: LearningStore, planner: LearningPlannerStore):
    global _store, _planner
    _store = store
    _planner = planner


def _ensure_enabled() -> None:
    if not get_settings().enable_adaptive_learning:
        raise HTTPException(status_code=404, detail="Adaptive learning disabled")


@router.post("/attempts", response_model=LearningAttemptResponse)
async def record_attempt(
    body: LearningAttemptRequest, user: dict = Depends(require_auth)
):
    _ensure_enabled()
    if _store is None:
        raise HTTPException(status_code=503, detail="Learning store not initialised")
    saved = await _store.run_async(
        _store.record_attempt,
        user_id=user["user"],
        question_id=body.question_id,
        topic_id=body.topic_id,
        user_answer=body.user_answer,
        is_correct=body.is_correct,
        confidence=body.confidence,
        response_time_ms=body.response_time_ms,
        mode=body.mode.value,
    )
    return LearningAttemptResponse(**saved)


@router.get("/review-queue", response_model=ReviewQueueResponse)
async def review_queue(
    limit: int = Query(default=50, ge=1, le=200),
    user: dict = Depends(require_auth),
):
    _ensure_enabled()
    if _store is None:
        raise HTTPException(status_code=503, detail="Learning store not initialised")
    data = await _store.run_async(_store.get_review_queue, user_id=user["user"], limit=limit)
    return ReviewQueueResponse(**data)


@router.get("/weak-areas", response_model=WeakAreasResponse)
async def weak_areas(
    limit: int = Query(default=10, ge=1, le=50),
    user: dict = Depends(require_auth),
):
    _ensure_enabled()
    if _store is None:
        raise HTTPException(status_code=503, detail="Learning store not initialised")
    data = await _store.run_async(_store.get_weak_areas, user_id=user["user"], limit=limit)
    return WeakAreasResponse(**data)


@router.get("/mastery", response_model=TopicMasteryResponse)
async def topic_mastery(
    limit: int = Query(default=500, ge=1, le=1000),
    user: dict = Depends(require_auth),
):
    _ensure_enabled()
    if _store is None:
        raise HTTPException(status_code=503, detail="Learning store not initialised")
    data = await _store.run_async(_store.get_topic_mastery, user_id=user["user"], limit=limit)
    return TopicMasteryResponse(**data)


@router.get("/profile", response_model=LearnerProfile)
async def get_profile(user: dict = Depends(require_auth)):
    _ensure_enabled()
    if _planner is None:
        raise HTTPException(status_code=503, detail="Learning planner not initialised")
    data = await _planner.run_async(_planner.get_profile, user_id=user["user"])
    return LearnerProfile(**data)


@router.put("/profile", response_model=LearnerProfile)
async def update_profile(
    body: LearnerProfileUpdateRequest,
    user: dict = Depends(require_auth),
):
    _ensure_enabled()
    if _planner is None:
        raise HTTPException(status_code=503, detail="Learning planner not initialised")
    data = await _planner.run_async(
        _planner.upsert_profile,
        user_id=user["user"],
        payload=body.model_dump(exclude_none=True),
    )
    return LearnerProfile(**data)


@router.post("/profile/diagnostic", response_model=ProfileDiagnosticResponse)
async def run_profile_diagnostic(user: dict = Depends(require_auth)):
    _ensure_enabled()
    if _planner is None:
        raise HTTPException(status_code=503, detail="Learning planner not initialised")
    data = await _planner.run_async(_planner.run_diagnostic, user_id=user["user"])
    return ProfileDiagnosticResponse(**data)


@router.get("/recommendations", response_model=RecommendationsResponse)
async def learning_recommendations(
    limit: int = Query(default=8, ge=1, le=50),
    user: dict = Depends(require_auth),
):
    _ensure_enabled()
    if _planner is None:
        raise HTTPException(status_code=503, detail="Learning planner not initialised")
    data = await _planner.run_async(
        _planner.build_recommendations,
        user_id=user["user"],
        limit=limit,
    )
    return RecommendationsResponse(**data)


@router.get("/study-plan", response_model=StudyPlanResponse)
async def study_plan(
    days: int = Query(default=7, ge=1, le=31),
    daily_items: int = Query(default=3, ge=1, le=10),
    user: dict = Depends(require_auth),
):
    _ensure_enabled()
    if _planner is None:
        raise HTTPException(status_code=503, detail="Learning planner not initialised")
    data = await _planner.run_async(
        _planner.build_study_plan,
        user_id=user["user"],
        days=days,
        daily_items=daily_items,
    )
    return StudyPlanResponse(**data)
