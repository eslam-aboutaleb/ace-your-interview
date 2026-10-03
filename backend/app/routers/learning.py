"""Adaptive learning router — attempts, review queue, weak areas."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from app.config import get_settings
from app.dependencies import require_auth
from app.schemas.models import (
    CalibrationResponse,
    ForecastResponse,
    LearnerProfile,
    LearnerProfileUpdateRequest,
    LearningAttemptRequest,
    LearningAttemptResponse,
    LearningReviewRequest,
    LearningReviewResponse,
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

logger = logging.getLogger(__name__)

_store: LearningStore | None = None
_planner: LearningPlannerStore | None = None


def init(store: LearningStore, planner: LearningPlannerStore):
    global _store, _planner
    _store = store
    _planner = planner
    _run_startup_fsrs_backfill(store)


def _fsrs_enabled() -> bool:
    return bool(getattr(get_settings(), "enable_fsrs_v1", False))


def _ensure_enabled() -> None:
    if not get_settings().enable_adaptive_learning:
        raise HTTPException(status_code=404, detail="Adaptive learning disabled")


def _ensure_fsrs_enabled() -> None:
    if not _fsrs_enabled():
        raise HTTPException(status_code=404, detail="FSRS scheduling disabled")


def _run_startup_fsrs_backfill(store: LearningStore) -> None:
    """One-time migration of legacy question_progress rows into fsrs_cards."""
    if not _fsrs_enabled():
        return
    try:
        result = store.backfill_fsrs_from_progress()
        if result.get("inserted"):
            logger.info(
                "FSRS backfill migrated %s of %s question_progress rows",
                result.get("inserted"),
                result.get("question_progress_rows"),
            )
    except Exception:  # pragma: no cover - defensive: never block startup
        logger.exception("FSRS backfill failed")


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
    if _fsrs_enabled():
        data = await _store.run_async(
            _store.get_fsrs_review_queue, user_id=user["user"], limit=limit
        )
    else:
        data = await _store.run_async(
            _store.get_review_queue, user_id=user["user"], limit=limit
        )
    return ReviewQueueResponse(**data)


@router.post("/review", status_code=201, response_model=LearningReviewResponse)
async def review_card(
    body: LearningReviewRequest, user: dict = Depends(require_auth)
):
    _ensure_enabled()
    _ensure_fsrs_enabled()
    if _store is None:
        raise HTTPException(status_code=503, detail="Learning store not initialised")
    data = await _store.run_async(
        _store.record_review,
        user_id=user["user"],
        card_id=body.card_id,
        topic_id=body.topic_id,
        rating=body.rating.value,
        response_time_ms=body.response_time_ms,
        source_type=body.source_type.value,
    )
    return LearningReviewResponse(**data)


@router.get("/forecast", response_model=ForecastResponse)
async def forecast(
    days: int = Query(default=7, ge=1, le=31),
    user: dict = Depends(require_auth),
):
    _ensure_enabled()
    _ensure_fsrs_enabled()
    if _store is None:
        raise HTTPException(status_code=503, detail="Learning store not initialised")
    data = await _store.run_async(
        _store.get_forecast, user_id=user["user"], days=days
    )
    return ForecastResponse(**data)


@router.get("/calibration", response_model=CalibrationResponse)
async def calibration(user: dict = Depends(require_auth)):
    _ensure_enabled()
    _ensure_fsrs_enabled()
    if _store is None:
        raise HTTPException(status_code=503, detail="Learning store not initialised")
    data = await _store.run_async(_store.get_calibration, user_id=user["user"])
    return CalibrationResponse(**data)


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
    if _fsrs_enabled():
        data = await _store.run_async(
            _store.get_fsrs_topic_mastery, user_id=user["user"], limit=limit
        )
    else:
        data = await _store.run_async(
            _store.get_topic_mastery, user_id=user["user"], limit=limit
        )
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
