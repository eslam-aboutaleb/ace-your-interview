"""Learning journey API (issue #4).

Exposes durable journeys: versioned plans (phases/modules/objectives),
immutable numbered question batches, and optimistic reader checkpoints.

The whole surface sits behind ``STUDY_ENABLE_JOURNEYS_V1`` while the
learning-journey stages land incrementally; when the flag is off every
endpoint 404s so clients cannot observe a half-built feature.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.config import get_settings
from app.dependencies import require_auth
from app.schemas.journeys import (
    AttemptCreateRequest,
    BatchCreateRequest,
    BatchCreateResponse,
    BatchListResponse,
    CheckpointResponse,
    CheckpointUpdateRequest,
    CompletionUpdateRequest,
    GenerateBatchRequest,
    GenerateBatchResponse,
    JourneyCreateRequest,
    JourneyCreateResponse,
    JourneyDetail,
    JourneyListResponse,
    JourneyProgressSummary,
    JourneyRefreshResponse,
    JourneySourceAttachRequest,
    JourneySourceIngestResponse,
    JourneySourceListResponse,
    PlanPreviewRequest,
    PlanPreviewResponse,
    QuestionProgressResponse,
    ReaderStateResponse,
)
from app.services.doc_parser import DocParser
from app.services.embedding_client import EmbeddingClient
from app.services.journey_source_ingest import (
    GitHubApiFetcher,
    SourceFetcher,
    SourceIngestError,
    ingest_snapshot,
    refresh_source,
)
from app.services.journey_generation import (
    JourneyGenerationError,
    JourneyGenerationService,
    build_question_generator_fn,
)
from app.services.journey_store import (
    JourneyNotFoundError,
    JourneyStore,
    JourneyValidationError,
    NoPendingObjectivesError,
    RevisionConflictError,
)
from app.services.learning_store import LearningStore
from app.services.plan_preview import (
    PlanValidationError,
    mark_review_obligations,
    plan_stats,
    shape_plan_for_depth,
    validate_edited_plan,
)


from app.services.llm_client import LLMClient
from app.services.question_generator import QuestionGenerator

router = APIRouter(prefix="/api/journeys", tags=["journeys"])

_store: JourneyStore | None = None
_question_generator: QuestionGenerator | None = None
_llm_client: LLMClient | None = None
_embedding_client: EmbeddingClient | None = None
_learning_store: LearningStore | None = None
_parser: DocParser | None = None
_source_fetcher: SourceFetcher | None = None

# Objective extraction is deterministic: each bullet in a curriculum section
# becomes one objective, capped so a pathological section cannot create an
# unbounded plan. The goal-driven LLM planner (issue #6) replaces this
# baseline for richer journeys.
MAX_OBJECTIVES_PER_SECTION = 8
MAX_PLAN_MODULES = 200


def init(
    store: JourneyStore,
    learning_store: LearningStore | None = None,
    parser: DocParser | None = None,
    source_fetcher: SourceFetcher | None = None,
    question_generator: QuestionGenerator | None = None,
    llm_client: LLMClient | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> None:
    global _store, _learning_store, _parser, _source_fetcher
    global _question_generator, _llm_client, _embedding_client
    _store = store
    _learning_store = learning_store
    _parser = parser
    _source_fetcher = source_fetcher
    _question_generator = question_generator
    _llm_client = llm_client
    _embedding_client = embedding_client


def _ensure_enabled() -> None:
    if not getattr(get_settings(), "enable_journeys_v1", False):
        raise HTTPException(status_code=404, detail="Journeys API disabled")


def _ensure_sources_enabled() -> None:
    _ensure_enabled()
    if not getattr(get_settings(), "enable_journey_sources_v1", False):
        raise HTTPException(status_code=404, detail="Journey sources API disabled")


def _ensure_fetcher() -> SourceFetcher:
    return _source_fetcher or GitHubApiFetcher()


def _ensure_store() -> JourneyStore:
    if _store is None:
        raise HTTPException(status_code=503, detail="Journey store unavailable")
    return _store


def _objectives_from_section(heading: str, content: str) -> list[dict[str, str]]:
    titles: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            item = stripped[2:].strip()
            title = item.split(":", 1)[0].strip() or item
            if title:
                titles.append(title[:120])
        if len(titles) >= MAX_OBJECTIVES_PER_SECTION:
            break
    if not titles:
        titles = [f"Explain {heading}"[:120]]
    return [
        {"title": title, "kind": "concept", "source_section": heading}
        for title in titles
    ]


def _plan_from_sections(
    *, topic_title: str, sections: list[dict[str, str]]
) -> dict[str, Any]:
    modules = []
    for section in sections[:MAX_PLAN_MODULES]:
        heading = str(section.get("heading", "")).strip()
        if not heading:
            continue
        modules.append(
            {
                "title": heading,
                "description": "",
                "objectives": _objectives_from_section(
                    heading, str(section.get("content", ""))
                ),
            }
        )
    if not modules:
        modules = [
            {
                "title": topic_title,
                "description": "",
                "objectives": [
                    {
                        "title": f"Explain {topic_title}"[:120],
                        "kind": "concept",
                        "source_section": "",
                    }
                ],
            }
        ]
    return {
        "phases": [
            {
                "title": f"{topic_title} curriculum",
                "description": "",
                "modules": modules,
            }
        ]
    }


def _resolve_source(
    *, user: dict, source_type: str, source_topic_id: str
) -> dict[str, Any]:
    """Load the journey's source topic and derive its baseline plan.

    Returns ``{title, track, level, plan, provenance}`` or raises 404 when the
    source does not exist (or is not owned by the caller, for custom topics).
    """
    if source_type == "static_topic":
        if _parser is None:
            raise HTTPException(status_code=503, detail="Curriculum unavailable")
        detail = _parser.get_topic(source_topic_id)
        if detail is None:
            raise HTTPException(
                status_code=404, detail=f"topic {source_topic_id!r} not found"
            )
        sections = [
            {"heading": s.get("heading", ""), "content": s.get("content", "")}
            for s in detail.sections
        ]
        return {
            "title": detail.title,
            "track": detail.track,
            "level": detail.levels[0] if detail.levels else "",
            "plan": _plan_from_sections(topic_title=detail.title, sections=sections),
            "provenance": {"source": "static_curriculum", "topic_id": detail.id},
        }
    if source_type == "custom_topic":
        if _learning_store is None:
            raise HTTPException(
                status_code=503, detail="Custom topic store unavailable"
            )
        custom = _learning_store.get_custom_topic(
            user_id=str(user["user"]), topic_id=source_topic_id
        )
        if custom is None:
            raise HTTPException(
                status_code=404, detail=f"custom topic {source_topic_id!r} not found"
            )
        sections = custom.get("sections") or []
        return {
            "title": str(custom.get("title") or source_topic_id),
            "track": str(custom.get("track") or ""),
            "level": "",
            "plan": _plan_from_sections(
                topic_title=str(custom.get("title") or source_topic_id),
                sections=[
                    {
                        "heading": str(s.get("heading", "")),
                        "content": str(s.get("content", "")),
                    }
                    for s in sections
                    if isinstance(s, dict)
                ],
            ),
            "provenance": {"source": "custom_topic", "topic_id": source_topic_id},
        }
    raise HTTPException(
        status_code=422, detail=f"unsupported source_type {source_type!r}"
    )


def _map_store_error(exc: Exception) -> HTTPException:
    if isinstance(exc, JourneyNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, NoPendingObjectivesError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, RevisionConflictError):
        return HTTPException(
            status_code=409,
            detail={"message": str(exc), "current": exc.current},
        )
    if isinstance(exc, JourneyValidationError):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=500, detail="journey operation failed")


@router.get("", response_model=JourneyListResponse)
async def list_journeys(user: dict = Depends(require_auth)):
    _ensure_enabled()
    store = _ensure_store()
    try:
        journeys = await store.run_async(store.list_journeys, user_id=str(user["user"]))
    except Exception as exc:  # pragma: no cover - defensive
        raise _map_store_error(exc) from exc
    return {"journeys": journeys}


@router.post("", status_code=201, response_model=JourneyCreateResponse)
async def create_journey(
    body: JourneyCreateRequest, user: dict = Depends(require_auth)
):
    _ensure_enabled()
    store = _ensure_store()
    resolved = _resolve_source(
        user=user,
        source_type=body.source_type.value,
        source_topic_id=body.source_topic_id,
    )
    level = body.level or resolved["level"]
    plan = resolved["plan"]
    provenance = dict(resolved["provenance"])
    if body.plan is not None:
        try:
            plan = validate_edited_plan(body.plan)
        except PlanValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        provenance["edited_plan"] = True
    if body.plan_provenance:
        provenance["inputs"] = {
            key: str(value)[:500]
            for key, value in body.plan_provenance.items()
            if isinstance(key, str) and isinstance(value, (str, int, float, bool))
        }
    try:
        detail = await store.run_async(
            store.create_journey,
            user_id=str(user["user"]),
            title=body.title,
            goal=body.goal,
            source_type=body.source_type.value,
            source_topic_id=body.source_topic_id,
            track=resolved["track"],
            level=level,
            plan=plan,
            provenance=provenance,
        )
    except Exception as exc:
        raise _map_store_error(exc) from exc
    return {"journey": detail}


@router.get("/{journey_id}", response_model=JourneyDetail)
async def get_journey(journey_id: str, user: dict = Depends(require_auth)):
    _ensure_enabled()
    store = _ensure_store()
    try:
        return await store.run_async(
            store.get_journey, user_id=str(user["user"]), journey_id=journey_id
        )
    except Exception as exc:
        raise _map_store_error(exc) from exc


@router.post("/{journey_id}/refresh", response_model=JourneyRefreshResponse)
async def refresh_journey_plan(journey_id: str, user: dict = Depends(require_auth)):
    """Replan from the current source content (curriculum upgrade path).

    Unchanged objectives keep their IDs and progress; removed ones are
    retired, never deleted, so no recorded progress is silently lost.
    """
    _ensure_enabled()
    store = _ensure_store()
    try:
        existing = await store.run_async(
            store.get_journey, user_id=str(user["user"]), journey_id=journey_id
        )
    except Exception as exc:
        raise _map_store_error(exc) from exc
    resolved = _resolve_source(
        user=user,
        source_type=str(existing["source_type"]),
        source_topic_id=str(existing["source_topic_id"]),
    )
    try:
        result = await store.run_async(
            store.refresh_plan,
            user_id=str(user["user"]),
            journey_id=journey_id,
            plan=resolved["plan"],
        )
    except Exception as exc:
        raise _map_store_error(exc) from exc
    return {
        "journey": result["journey"],
        "plan_changed": result["plan_changed"],
        "added_objectives": result["added_objectives"],
        "retired_objectives": result["retired_objectives"],
    }


@router.post("/{journey_id}/batches", response_model=BatchCreateResponse)
async def create_batch(
    journey_id: str, body: BatchCreateRequest, user: dict = Depends(require_auth)
):
    """Reserve the next immutable batch of question slots.

    Replaying the same ``idempotency_key`` returns the original batch with
    ``created=false`` instead of duplicating reservations.
    """
    _ensure_enabled()
    store = _ensure_store()
    try:
        batch, created = await store.run_async(
            store.create_batch,
            user_id=str(user["user"]),
            journey_id=journey_id,
            idempotency_key=body.idempotency_key,
            slot_count=body.slot_count,
            perspectives=body.perspectives,
        )
    except Exception as exc:
        raise _map_store_error(exc) from exc
    return {"batch": batch, "created": created}


@router.get("/{journey_id}/batches", response_model=BatchListResponse)
async def list_batches(journey_id: str, user: dict = Depends(require_auth)):
    _ensure_enabled()
    store = _ensure_store()
    try:
        batches = await store.run_async(
            store.list_batches, user_id=str(user["user"]), journey_id=journey_id
        )
    except Exception as exc:
        raise _map_store_error(exc) from exc
    return {"batches": batches}


@router.get("/{journey_id}/reader-state", response_model=ReaderStateResponse)
async def get_reader_state(journey_id: str, user: dict = Depends(require_auth)):
    _ensure_enabled()
    store = _ensure_store()
    try:
        return await store.run_async(
            store.get_reader_state, user_id=str(user["user"]), journey_id=journey_id
        )
    except Exception as exc:
        raise _map_store_error(exc) from exc


@router.patch("/{journey_id}/checkpoints", response_model=CheckpointResponse)
async def update_checkpoint(
    journey_id: str, body: CheckpointUpdateRequest, user: dict = Depends(require_auth)
):
    _ensure_enabled()
    store = _ensure_store()
    try:
        checkpoint = await store.run_async(
            store.update_checkpoint,
            user_id=str(user["user"]),
            journey_id=journey_id,
            expected_revision=body.expected_revision,
            mutation_id=body.mutation_id,
            last_batch_id=body.last_batch_id,
            last_question_id=body.last_question_id,
            last_section_id=body.last_section_id,
            scroll_offset=body.scroll_offset,
        )
    except Exception as exc:
        raise _map_store_error(exc) from exc
    return {"checkpoint": checkpoint}


@router.post(
    "/{journey_id}/sources", status_code=201, response_model=JourneySourceIngestResponse
)
async def attach_journey_source(
    journey_id: str,
    body: JourneySourceAttachRequest,
    user: dict = Depends(require_auth),
):
    """Attach and index a GitHub repo or uploaded doc as a journey source
    (issue #14). Idempotent per (journey, kind, locator)."""
    _ensure_sources_enabled()
    store = _ensure_store()
    try:
        result = await store.run_async(
            lambda: ingest_snapshot(
                store,
                user_id=str(user["user"]),
                journey_id=journey_id,
                kind=body.kind.value,
                locator=body.locator,
                fetcher=_ensure_fetcher(),
            )
        )
        return {
            "source": result["source"],
            "diff": result["diff"],
        }
    except SourceIngestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise _map_store_error(exc) from exc


@router.get("/{journey_id}/sources", response_model=JourneySourceListResponse)
async def list_journey_sources(journey_id: str, user: dict = Depends(require_auth)):
    _ensure_sources_enabled()
    store = _ensure_store()
    try:
        sources = await store.run_async(
            store.list_journey_sources, user_id=str(user["user"]), journey_id=journey_id
        )
        return {"sources": sources}
    except Exception as exc:
        raise _map_store_error(exc) from exc


@router.post(
    "/{journey_id}/sources/{source_id}/refresh",
    response_model=JourneySourceIngestResponse,
)
async def refresh_journey_source(
    journey_id: str, source_id: str, user: dict = Depends(require_auth)
):
    """Re-fetch the source and re-index only changed files (issue #14).
    Journey progress is never touched: content is hash-addressed."""
    _ensure_sources_enabled()
    store = _ensure_store()
    try:
        result = await store.run_async(
            lambda: refresh_source(
                store,
                user_id=str(user["user"]),
                journey_id=journey_id,
                source_id=source_id,
                fetcher=_ensure_fetcher(),
            )
        )
        return {"source": result["source"], "diff": result["diff"]}
    except SourceIngestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise _map_store_error(exc) from exc


@router.post("/plan-preview", response_model=PlanPreviewResponse)
async def plan_preview(body: PlanPreviewRequest, user: dict = Depends(require_auth)):
    """Editable syllabus preview before commitment (issue #6).

    Resolves the source, shapes the baseline plan by depth, marks review
    obligations from the topic's active FSRS cards, estimates pacing from
    weekly hours, and echoes the inputs as provenance for the eventual
    ``POST /api/journeys`` commit. Persists nothing.
    """
    _ensure_enabled()
    _ensure_store()
    resolved = _resolve_source(
        user=user,
        source_type=body.source_type.value,
        source_topic_id=body.source_topic_id,
    )
    try:
        plan = shape_plan_for_depth(resolved["plan"], body.depth.value)
    except PlanValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    has_active_cards = False
    if _learning_store is not None and body.source_type.value == "static_topic":
        try:
            has_active_cards = await _ensure_store().run_async(
                _learning_store.count_active_topic_cards,
                user_id=str(user["user"]),
                topic_id=body.source_topic_id,
            ) > 0
        except Exception:  # review hints are best-effort, never block a preview
            has_active_cards = False
    plan, review_count = mark_review_obligations(plan, has_active_cards=has_active_cards)

    stats = plan_stats(plan, available_hours_per_week=body.available_hours_per_week)
    stats["review_objective_count"] = review_count
    level = body.target_level or resolved["level"]
    provenance = {
        **resolved["provenance"],
        "inputs": {
            "goal": body.goal,
            "target_level": body.target_level,
            "available_hours_per_week": body.available_hours_per_week,
            "depth": body.depth.value,
        },
    }
    return {
        "title": resolved["title"],
        "track": resolved["track"],
        "level": level,
        "plan": plan,
        "provenance": provenance,
        "stats": stats,
    }


@router.post("/{journey_id}/batches/{batch_id}/generate", response_model=GenerateBatchResponse)
async def generate_batch(
    journey_id: str, batch_id: str, body: GenerateBatchRequest, user: dict = Depends(require_auth)
):
    """Fill a batch's reserved slots with novelty-checked questions (#2).

    Runs the exact-hash / near-duplicate / concept-equivalence /
    embedding-similarity funnel per candidate, regenerating only rejected
    slots (bounded rounds), and persists fingerprints plus a generation
    job carrying prompt/model/schema versions and accepted/rejected
    counts. Optional LLM adjudication of the embedding review band is
    behind ``STUDY_ENABLE_JOURNEY_GENERATION_ADJUDICATION`` (default off).
    """
    _ensure_enabled()
    store = _ensure_store()
    if _question_generator is None:
        raise HTTPException(status_code=503, detail="Question generator unavailable")
    settings = get_settings()

    generator_fn = build_question_generator_fn(
        _question_generator, _parser, _learning_store, user_identity=user
    )

    embed_fn = None
    if _embedding_client is not None:
        async def embed_fn(text: str):  # type: ignore[no-redef]
            vectors = await _embedding_client.embed([text], user_identity=user)
            return vectors[0] if vectors else None

    adjudicator_fn = None
    if settings.enable_journey_generation_adjudication and _llm_client is not None:
        async def adjudicator_fn(candidate: str, matched: str) -> bool:  # type: ignore[no-redef]
            result = await _llm_client.completion(
                (
                    "Two interview questions follow. Answer with exactly YES if they "
                    "ask for the same knowledge or skill (semantically equivalent), "
                    "otherwise exactly NO.\n\n"
                    f"Question A:\n{candidate}\n\nQuestion B:\n{matched}"
                ),
                user_identity=user,
                max_tokens_cap=8,
                flow="journey_generation_adjudication",
                prompt_name="journey_novelty_adjudication",
                prompt_version="v1",
            )
            answer = str(result.get("content") or "").strip().lower()
            return answer.startswith("yes")

    service = JourneyGenerationService(
        store, embed_fn=embed_fn, adjudicator_fn=adjudicator_fn
    )
    try:
        return await service.generate_batch(
            user_id=str(user["user"]),
            journey_id=journey_id,
            batch_id=batch_id,
            generator_fn=generator_fn,
            prompt_version=body.prompt_version,
            model_version=body.model_version,
            schema_version=body.schema_version,
        )
    except JourneyGenerationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise _map_store_error(exc) from exc


@router.get("/{journey_id}/progress", response_model=JourneyProgressSummary)
async def get_journey_progress(journey_id: str, user: dict = Depends(require_auth)):
    _ensure_enabled()
    store = _ensure_store()
    try:
        return await store.run_async(
            store.get_journey_progress, user_id=str(user["user"]), journey_id=journey_id
        )
    except Exception as exc:
        raise _map_store_error(exc) from exc


@router.put("/{journey_id}/questions/{slot_id}/completion", response_model=QuestionProgressResponse)
async def set_question_completion(
    journey_id: str, slot_id: str, body: CompletionUpdateRequest, user: dict = Depends(require_auth)
):
    """Explicit complete/uncomplete; idempotent per mutation_id (issue #5)."""
    _ensure_enabled()
    store = _ensure_store()
    try:
        return await store.run_async(
            store.set_question_completion,
            user_id=str(user["user"]),
            journey_id=journey_id,
            slot_id=slot_id,
            completed=body.completed,
            mutation_id=body.mutation_id,
        )
    except Exception as exc:
        raise _map_store_error(exc) from exc


@router.post("/{journey_id}/questions/{slot_id}/attempts", response_model=QuestionProgressResponse)
async def record_question_attempt(
    journey_id: str, slot_id: str, body: AttemptCreateRequest, user: dict = Depends(require_auth)
):
    """Record a practice attempt; idempotent per mutation_id (issue #5)."""
    _ensure_enabled()
    store = _ensure_store()
    try:
        return await store.run_async(
            store.record_question_attempt,
            user_id=str(user["user"]),
            journey_id=journey_id,
            slot_id=slot_id,
            outcome=body.outcome.value,
            confidence=body.confidence,
            response_time_ms=body.response_time_ms,
            mutation_id=body.mutation_id,
        )
    except Exception as exc:
        raise _map_store_error(exc) from exc
