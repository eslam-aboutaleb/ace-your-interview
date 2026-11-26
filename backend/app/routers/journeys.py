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
    BatchCreateRequest,
    BatchCreateResponse,
    BatchListResponse,
    CheckpointResponse,
    CheckpointUpdateRequest,
    JourneyCreateRequest,
    JourneyCreateResponse,
    JourneyDetail,
    JourneyListResponse,
    JourneyRefreshResponse,
    JourneySourceAttachRequest,
    JourneySourceIngestResponse,
    JourneySourceListResponse,
    ReaderStateResponse,
)
from app.services.doc_parser import DocParser
from app.services.journey_source_ingest import (
    GitHubApiFetcher,
    SourceFetcher,
    SourceIngestError,
    ingest_snapshot,
    refresh_source,
)
from app.services.journey_store import (
    JourneyNotFoundError,
    JourneyStore,
    JourneyValidationError,
    NoPendingObjectivesError,
    RevisionConflictError,
)
from app.services.learning_store import LearningStore

router = APIRouter(prefix="/api/journeys", tags=["journeys"])

_store: JourneyStore | None = None
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
) -> None:
    global _store, _learning_store, _parser, _source_fetcher
    _store = store
    _learning_store = learning_store
    _parser = parser
    _source_fetcher = source_fetcher


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
            plan=resolved["plan"],
            provenance=resolved["provenance"],
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
