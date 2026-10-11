"""Adaptive learning router — attempts, review queue, weak areas."""

from __future__ import annotations

import logging
import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from app.config import get_settings
from app.dependencies import require_auth
from app.routers import hints as hints_router
from app.routers import interview_sessions as interview_sessions_router
from app.schemas.models import (
    CalibrationResponse,
    ErrorPatternListResponse,
    ForecastResponse,
    FSRSCardResponse,
    LeechItem,
    LeechResumeResponse,
    LeechesResponse,
    LearnerProfile,
    LearnerProfileUpdateRequest,
    LearningAttemptRequest,
    LearningAttemptResponse,
    LearningReviewRequest,
    LearningReviewResponse,
    NextObjectiveItem,
    NextObjectivesResponse,
    PlanTaskItem,
    PlanTaskListResponse,
    ProfileDiagnosticResponse,
    RecommendationsResponse,
    ReviewQueueResponse,
    RoleReadinessResponse,
    StudyPlanFromJDRequest,
    StudyPlanFromJDResponse,
    StudyPlanResponse,
    TopicMasteryResponse,
    WeakAreasResponse,
)
from app.services.doc_parser import DocParser
from app.services.learning_planner import LearningPlannerStore
from app.services.learning_store import (
    ERROR_PATTERN_CLASSIFIED_EVENT,
    ERROR_PATTERN_FEATURE_KEY,
    LearningStore,
)
from app.services.misconception_service import (
    MIN_USER_ANSWER_CHARS,
    MisconceptionService,
    derive_objective,
    resolve_topic_title,
)

router = APIRouter(prefix="/api/learning", tags=["learning"])

logger = logging.getLogger(__name__)

_store: LearningStore | None = None
_planner: LearningPlannerStore | None = None
_parser: DocParser | None = None
_misconception: MisconceptionService | None = None


def init(
    store: LearningStore,
    planner: LearningPlannerStore,
    parser: DocParser | None = None,
    llm_client=None,
):
    global _store, _planner, _parser, _misconception
    _store = store
    _planner = planner
    _parser = parser
    _misconception = MisconceptionService(llm_client)
    _run_startup_fsrs_backfill(store)


def _fsrs_enabled() -> bool:
    return bool(getattr(get_settings(), "enable_fsrs_v1", False))


def _ensure_enabled() -> None:
    if not get_settings().enable_adaptive_learning:
        raise HTTPException(status_code=404, detail="Adaptive learning disabled")


def _ensure_fsrs_enabled() -> None:
    if not _fsrs_enabled():
        raise HTTPException(status_code=404, detail="FSRS scheduling disabled")


def _count_hints_used(user_id: str, question_id: str) -> int:
    """Hints already generated for this user+question.

    Hint usage is read from the hint service's event log so an
    attempt that leaned on hints is scheduled as weaker recall
    (see ``fsrs_scheduler.derive_rating_with_hints``). Returns 0
    when the hint service is not initialised or the count fails:
    scheduling must never break on hint bookkeeping.
    """
    service = hints_router.get_service()
    if service is None:
        return 0
    try:
        return service.count_hints_used(user_id, question_id)
    except Exception:
        logger.exception(
            "Failed to count hints for question %s", question_id
        )
        return 0


def _normalise_topic_token(value: str) -> str:
    """Lowercase and strip non-alphanumerics for matching."""
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _skill_matches_topic(skill_key: str, title_key: str) -> bool:
    """High-confidence match between a skill and a topic title.

    The normalised skill must equal the normalised title, or
    the shorter of the two — when it is at least four
    characters — must be contained in the longer. The length
    floor keeps short substrings ("sql" inside "mysql") from
    producing false-positive matches.
    """
    if not skill_key or not title_key:
        return False
    if skill_key == title_key:
        return True
    shorter, longer = (
        (skill_key, title_key)
        if len(skill_key) <= len(title_key)
        else (title_key, skill_key)
    )
    return len(shorter) >= 4 and shorter in longer


def map_skills_to_topic_ids(
    skills: list[str],
    topics: list[Any],
) -> tuple[dict[str, str], list[str]]:
    """Map skill strings to curriculum topic IDs.

    A skill maps to a topic only on a high-confidence title
    match (see :func:`_skill_matches_topic`). Skills with no
    confident match are returned unmapped so they stay as
    plain focus-area strings — a topic ID is never invented.
    Returns ``(mapped, unmapped)`` where ``mapped`` maps the
    original skill text to its topic ID.
    """
    mapped: dict[str, str] = {}
    unmapped: list[str] = []
    seen: set[str] = set()
    for skill in skills:
        cleaned = str(skill or "").strip()
        if not cleaned:
            continue
        key = _normalise_topic_token(cleaned)
        if key in seen:
            continue
        seen.add(key)
        topic_id = ""
        for topic in topics:
            if _skill_matches_topic(
                key, _normalise_topic_token(getattr(topic, "title", ""))
            ):
                topic_id = getattr(topic, "id", "")
                break
        if topic_id:
            mapped[cleaned] = topic_id
        else:
            unmapped.append(cleaned)
    return mapped, unmapped


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
    hints_used = _count_hints_used(user["user"], body.question_id)
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
        learning_objective=body.learning_objective,
        hints_used=hints_used,
    )
    await _capture_error_pattern(user=user, body=body)
    return LearningAttemptResponse(**saved)


@router.get("/review-queue", response_model=ReviewQueueResponse)
async def review_queue(
    limit: int = Query(default=50, ge=1, le=200),
    retrievability_threshold: float | None = Query(
        default=None, ge=0.0, le=1.0
    ),
    weak_area_boost: bool = Query(default=False),
    user: dict = Depends(require_auth),
):
    _ensure_enabled()
    if _store is None:
        raise HTTPException(status_code=503, detail="Learning store not initialised")
    data = await _store.run_async(
        _store.get_fsrs_review_queue,
        user_id=user["user"],
        limit=limit,
        retrievability_threshold=retrievability_threshold,
        weak_area_boost=weak_area_boost,
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
    data = await _store.run_async(
        _store.get_fsrs_topic_mastery, user_id=user["user"], limit=limit
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


@router.get("/next-objectives", response_model=NextObjectivesResponse)
async def next_objectives(
    topic_ids: list[str] = Query(default=[]),
    limit: int = Query(default=5, ge=1, le=20),
    user: dict = Depends(require_auth),
):
    """Rank learning objectives for the next quiz by (1 - accuracy) * uncertainty."""
    _ensure_enabled()
    if _store is None:
        raise HTTPException(status_code=503, detail="Learning store not initialised")
    clean_ids = [tid.strip() for tid in topic_ids if tid.strip()]
    ranked = await _store.run_async(
        _store.select_next_objectives,
        user_id=user["user"],
        topic_ids=clean_ids,
        limit=limit,
    )
    objectives: list[NextObjectiveItem] = []
    for item in ranked:
        topic_id = str(item["topic_id"])
        title = ""
        if _parser is not None:
            topic = _parser.get_topic(topic_id)
            if topic is not None:
                title = topic.title
        objectives.append(
            NextObjectiveItem(
                topic_id=topic_id,
                topic_title=title,
                learning_objective=str(item["learning_objective"]),
                attempts=int(item["attempts"]),
                correct_attempts=int(item["correct_attempts"]),
                accuracy=float(item["accuracy"]),
                uncertainty=float(item["uncertainty"]),
                priority=float(item["priority"]),
            )
        )
    return NextObjectivesResponse(objectives=objectives)


@router.post("/plan-from-jd", response_model=StudyPlanFromJDResponse)
async def study_plan_from_jd(
    body: StudyPlanFromJDRequest,
    user: dict = Depends(require_auth),
):
    """Build a study plan from a job description.

    The JD is analysed against the stored resume profile;
    high-confidence gap/focus-area matches are mapped to
    curriculum topic IDs, persisted as the learner's
    ``focus_topic_ids``, and the study plan is built from
    the updated profile.
    """
    _ensure_enabled()
    if _planner is None:
        raise HTTPException(status_code=503, detail="Learning planner not initialised")
    if not get_settings().enable_interview_plus_v1:
        raise HTTPException(status_code=404, detail="Interview personalization disabled")

    services = interview_sessions_router.get_jd_analysis_services()
    if services is None:
        raise HTTPException(status_code=503, detail="JD analysis services not initialised")
    jd_analyzer, parser, resume_store = services

    profile = resume_store.get(user["user"])
    analysis = await jd_analyzer.analyze(
        jd_text=body.jd_text,
        profile=profile,
        llm_config=body.llm_config,
        user_identity=user,
    )

    skills = list(analysis.get("gaps") or [])
    for area in analysis.get("recommended_focus_areas") or []:
        if area not in skills:
            skills.append(area)

    topics = parser.list_topics(track=body.track.value if body.track else None)
    mapped, unmapped = map_skills_to_topic_ids(skills, topics)
    mapped_topic_ids = list(dict.fromkeys(mapped.values()))

    existing = await _planner.run_async(
        _planner.get_profile, user_id=user["user"]
    )
    merged_ids = list(
        dict.fromkeys(
            list(existing.get("focus_topic_ids") or []) + mapped_topic_ids
        )
    )[:12]
    await _planner.run_async(
        _planner.upsert_profile,
        user_id=user["user"],
        payload={"focus_topic_ids": merged_ids},
    )

    plan = await _planner.run_async(
        _planner.build_study_plan,
        user_id=user["user"],
        days=body.days,
        daily_items=body.daily_items,
    )
    return StudyPlanFromJDResponse(
        **plan,
        mapped_topic_ids=mapped_topic_ids,
        unmapped_focus_areas=unmapped,
        focus_topic_ids=merged_ids,
    )


@router.get("/leeches", response_model=LeechesResponse)
async def list_leeches(
    limit: int = Query(default=100, ge=1, le=500),
    user: dict = Depends(require_auth),
):
    """List auto-suspended leeches awaiting an explicit resume."""
    _ensure_enabled()
    _ensure_fsrs_enabled()
    if _store is None:
        raise HTTPException(status_code=503, detail="Learning store not initialised")
    cards = await _store.run_async(
        _store.get_leeches, user_id=user["user"], limit=limit
    )
    items = [LeechItem(**card) for card in cards]
    return LeechesResponse(items=items, total=len(items))


@router.post("/leeches/{card_id}/resume", response_model=LeechResumeResponse)
async def resume_leech(
    card_id: str, user: dict = Depends(require_auth)
):
    """Resume a suspended leech, returning it to the review queue."""
    _ensure_enabled()
    _ensure_fsrs_enabled()
    if _store is None:
        raise HTTPException(status_code=503, detail="Learning store not initialised")
    card = await _store.run_async(
        _store.resume_leeched_card,
        user_id=user["user"],
        card_id=card_id,
    )
    if card is None:
        raise HTTPException(status_code=404, detail="Leech not found")
    return LeechResumeResponse(card=FSRSCardResponse(**card))
# ── Error-pattern / misconception memory ─────────────────────────────────


async def _capture_error_pattern(*, user: dict, body: LearningAttemptRequest) -> None:
    """Classify a wrong answer and store the resulting error pattern.

    Runs after the attempt is persisted and never fails the request: the
    attempt is already recorded, so any error here is observability only.
    """
    try:
        settings = get_settings()
        if not settings.enable_error_patterns:
            return
        if body.is_correct or _store is None or _misconception is None:
            return
        if len(str(body.user_answer or "").strip()) < MIN_USER_ANSWER_CHARS:
            return
        user_id = str(user.get("user", "") or "").strip()
        if not user_id:
            return

        hourly_cap = int(getattr(settings, "error_pattern_hourly_cap", 30))
        if hourly_cap > 0:
            classified_recently = await _store.run_async(
                _store.count_error_pattern_classifications,
                user_id=user_id,
            )
            if classified_recently >= hourly_cap:
                logger.info(
                    "error_pattern_classification_capped user=%s window_hours=1 cap=%s",
                    user_id,
                    hourly_cap,
                )
                return

        classified = await _misconception.classify_misconception(
            question=body.question_text,
            correct_answer=body.correct_answer,
            user_answer=body.user_answer,
            level=body.level or "",
            topic_title=resolve_topic_title(body.topic_id),
            user_identity=user,
        )
        source = str(classified.get("source", "fallback"))
        if source != "rule":
            # Counts against the hourly cap only when a model may have been
            # called; deterministic labels cost nothing.
            await _store.run_async(
                _store.record_feature_event,
                user_id=user_id,
                feature_key=ERROR_PATTERN_FEATURE_KEY,
                event_name=ERROR_PATTERN_CLASSIFIED_EVENT,
                metadata={
                    "question_id": body.question_id,
                    "error_type": str(classified.get("error_type", "")),
                    "source": source,
                },
            )

        objective = (
            " ".join(str(body.objective or "").split())[:240]
            or derive_objective(body.question_text, fallback=body.question_id)
        )
        await _store.run_async(
            _store.record_error_pattern,
            user_id=user_id,
            topic_id=body.topic_id,
            objective=objective,
            error_type=str(classified.get("error_type", "")),
            explanation=str(classified.get("explanation", "")),
            evidence=str(body.user_answer or "")[:600],
        )
    except Exception:
        logger.warning(
            "Error-pattern capture failed for user=%s question=%s",
            user.get("user"),
            body.question_id,
            exc_info=True,
        )


@router.get("/error-patterns", response_model=ErrorPatternListResponse)
async def error_patterns(
    topic_id: str | None = Query(default=None, max_length=200),
    limit: int = Query(default=20, ge=1, le=200),
    user: dict = Depends(require_auth),
):
    _ensure_enabled()
    if _store is None:
        raise HTTPException(status_code=503, detail="Learning store not initialised")
    items = await _store.run_async(
        _store.list_error_patterns,
        user_id=user["user"],
        topic_id=topic_id,
        limit=limit,
    )
    return ErrorPatternListResponse(items=items, total=len(items))


# ── Plan commitment: completable study-plan tasks ───────────────────────────


def _require_task_date(value: str) -> str:
    task_date = LearningPlannerStore._sanitize_date(value)
    if not task_date:
        raise HTTPException(
            status_code=422, detail="task_date must be an ISO date (YYYY-MM-DD)"
        )
    return task_date


@router.get("/plan-tasks", response_model=PlanTaskListResponse)
async def list_plan_tasks(
    task_date: str = Query(..., description="ISO date (YYYY-MM-DD)"),
    user: dict = Depends(require_auth),
):
    """Persisted plan tasks for one date, with their completion state.

    The study-plan endpoint already embeds per-task completion; this
    endpoint reads the same ``plan_tasks`` rows on their own so any
    surface can poll a single day's commitment.
    """
    _ensure_enabled()
    if _planner is None:
        raise HTTPException(status_code=503, detail="Learning planner not initialised")
    clean_date = _require_task_date(task_date)
    items = await _planner.run_async(
        _planner.list_plan_tasks,
        user_id=user["user"],
        task_date=clean_date,
    )
    return PlanTaskListResponse(
        items=[PlanTaskItem(**item) for item in items], total=len(items)
    )


@router.post("/plan-tasks/{task_key}/complete", response_model=PlanTaskItem)
async def complete_plan_task(
    task_key: str,
    task_date: str = Query(..., description="ISO date (YYYY-MM-DD)"),
    user: dict = Depends(require_auth),
):
    """Mark a scheduled plan task complete (idempotent)."""
    _ensure_enabled()
    if _planner is None:
        raise HTTPException(status_code=503, detail="Learning planner not initialised")
    clean_date = _require_task_date(task_date)
    task = await _planner.run_async(
        _planner.complete_plan_task,
        user_id=user["user"],
        task_key=task_key,
        task_date=clean_date,
    )
    if task is None:
        raise HTTPException(status_code=404, detail="Plan task not found")
    return PlanTaskItem(**task)


@router.post("/plan-tasks/{task_key}/uncomplete", response_model=PlanTaskItem)
async def uncomplete_plan_task(
    task_key: str,
    task_date: str = Query(..., description="ISO date (YYYY-MM-DD)"),
    user: dict = Depends(require_auth),
):
    """Clear a completion (idempotent revert path for the UI)."""
    _ensure_enabled()
    if _planner is None:
        raise HTTPException(status_code=503, detail="Learning planner not initialised")
    clean_date = _require_task_date(task_date)
    task = await _planner.run_async(
        _planner.uncomplete_plan_task,
        user_id=user["user"],
        task_key=task_key,
        task_date=clean_date,
    )
    if task is None:
        raise HTTPException(status_code=404, detail="Plan task not found")
    return PlanTaskItem(**task)


# ── Readiness per role/company ────────────────────────────────────────────────


@router.get("/readiness", response_model=RoleReadinessResponse)
async def role_readiness(user: dict = Depends(require_auth)):
    """Role- and company-aware readiness over the learner's evidence.

    Weighted readiness for the profile's target role (with the
    per-track / per-competency breakdown that moves the number), the
    ranked best-fit roles across the built-in catalog, and a fit
    verdict per company in ``target_companies`` against the built-in
    requirements profiles. Fully deterministic and offline: the same
    stored evidence the diagnostic reads, re-weighted — no LLM call
    is made.
    """
    _ensure_enabled()
    if _planner is None:
        raise HTTPException(status_code=503, detail="Learning planner not initialised")
    data = await _planner.run_async(
        _planner.build_readiness, user_id=user["user"]
    )
    return RoleReadinessResponse(**data)
