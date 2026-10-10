"""Study analytics router (read-only, unflagged).

Endpoints:

- ``GET /api/analytics/heatmap``   activity heatmap (weeks)
- ``GET /api/analytics/retention`` 1d/7d/30d recall buckets
- ``GET /api/analytics/mastery``   EMA mastery curve (overall + topics)
- ``GET /api/analytics/forecast``  due-card forecast (existing logic)

All data is aggregated from existing tables
(``learning_attempts``, ``fsrs_review_log``,
``exam_sessions``); nothing is written.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.dependencies import require_auth
from app.schemas.models import ForecastResponse
from app.services.analytics_service import AnalyticsService

analytics_router = APIRouter(
    prefix="/api/analytics", tags=["analytics"]
)

_analytics_service: AnalyticsService | None = None


def init(analytics_service: AnalyticsService) -> None:
    global _analytics_service
    _analytics_service = analytics_service


def _require_service() -> AnalyticsService:
    if _analytics_service is None:
        raise HTTPException(
            status_code=503, detail="Analytics service not initialised"
        )
    return _analytics_service


@analytics_router.get("/heatmap")
async def heatmap(
    weeks: int = Query(default=52, ge=1, le=104),
    user: dict = Depends(require_auth),
):
    """Activity heatmap over the last ``weeks`` weeks."""
    service = _require_service()
    return await service.run_async(
        service.heatmap, user_id=user["user"], weeks=weeks
    )


@analytics_router.get("/retention")
async def retention(user: dict = Depends(require_auth)):
    """Recall rate at 1d/7d/30d buckets."""
    service = _require_service()
    return await service.run_async(
        service.retention, user_id=user["user"]
    )


@analytics_router.get("/mastery")
async def mastery(user: dict = Depends(require_auth)):
    """EMA mastery curve, overall and per topic."""
    service = _require_service()
    return await service.run_async(
        service.mastery_curve, user_id=user["user"]
    )


@analytics_router.get("/forecast", response_model=ForecastResponse)
async def forecast(
    days: int = Query(default=7, ge=1, le=31),
    user: dict = Depends(require_auth),
):
    """Due-card forecast (same logic as /api/learning/forecast)."""
    service = _require_service()
    data = await service.due_forecast(
        user_id=user["user"], days=days
    )
    return ForecastResponse(**data)
