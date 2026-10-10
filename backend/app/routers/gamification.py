"""Gamification router (STUDY_ENABLE_GAMIFICATION_V1).

Endpoints:

- ``GET  /api/gamification/summary``  XP, level, streak, badges
- ``PUT  /api/gamification/goal``     daily goal (1-50)

XP/streak/badge state is updated by the
GamificationService hooks wired into
``LearningStore.record_attempt``/``record_review``
and by the exam completion path, so this router
is read-mostly.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.config import get_settings
from app.dependencies import require_auth
from app.schemas.models import (
    GamificationGoalResponse,
    GamificationGoalUpdateRequest,
    GamificationSummaryResponse,
)
from app.services.gamification_service import GamificationService

gamification_router = APIRouter(
    prefix="/api/gamification", tags=["gamification"]
)

_gamification_service: GamificationService | None = None


def init(gamification_service: GamificationService) -> None:
    global _gamification_service
    _gamification_service = gamification_service


def _ensure_enabled() -> None:
    if not get_settings().enable_gamification_v1:
        raise HTTPException(
            status_code=503,
            detail="Gamification disabled "
            "(STUDY_ENABLE_GAMIFICATION_V1=false)",
        )


def _require_service() -> GamificationService:
    if _gamification_service is None:
        raise HTTPException(
            status_code=503, detail="Gamification service not initialised"
        )
    return _gamification_service


@gamification_router.get("/summary", response_model=GamificationSummaryResponse)
async def get_gamification_summary(user: dict = Depends(require_auth)):
    """XP, level, streak, daily-goal progress and badges."""
    _ensure_enabled()
    service = _require_service()
    summary = await service.get_summary(user_id=user["user"])
    return GamificationSummaryResponse(**summary)


@gamification_router.put("/goal", response_model=GamificationGoalResponse)
async def update_daily_goal(
    body: GamificationGoalUpdateRequest,
    user: dict = Depends(require_auth),
):
    """Set the daily XP-free goal (events per day)."""
    _ensure_enabled()
    service = _require_service()
    goal = await service.set_daily_goal(
        user_id=user["user"], daily_goal=body.daily_goal
    )
    return GamificationGoalResponse(daily_goal=goal)
