"""Progress router — cumulative per-topic study progress and its AI summary."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from app.dependencies import require_auth
from app.schemas.models import (
    LLMConfigRequest,
    SaveProgressRequest,
    TopicProgressDocument,
    TopicProgressListResponse,
)
from app.services.llm_client import LLMClient
from app.services.progress_summarizer import ProgressSummarizer, build_attempt_stats
from app.services.progress_store import ProgressStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/progress", tags=["progress"])

_store: ProgressStore | None = None
_summarizer: ProgressSummarizer | None = None
_llm_client: LLMClient | None = None

# Background repair of summaries owed by a dropped ``pagehide`` beacon. Held in a
# module-level set so the event loop cannot garbage-collect a pending task, and
# guarded by an in-flight key set so concurrent generations for the same topic
# do not each start an LLM call.
_background_tasks: set[asyncio.Task] = set()
_summary_inflight: set[tuple[str, str]] = set()


def init(
    store: ProgressStore,
    summarizer: ProgressSummarizer,
    llm_client: LLMClient | None = None,
):
    global _store, _summarizer, _llm_client
    _store = store
    _summarizer = summarizer
    _llm_client = llm_client


def _ensure_store() -> ProgressStore:
    if _store is None:
        raise HTTPException(status_code=503, detail="Progress store not initialised")
    return _store


@router.get("/topics", response_model=TopicProgressListResponse)
async def list_topic_progress(
    limit: int = Query(default=50, ge=1, le=200),
    user: dict = Depends(require_auth),
):
    store = _ensure_store()
    documents = await store.run_async(store.list_progress, user_id=user["user"], limit=limit)
    return TopicProgressListResponse(topics=documents, total=len(documents))


@router.get("/{topic_id}", response_model=TopicProgressDocument)
async def get_topic_progress(topic_id: str, user: dict = Depends(require_auth)):
    store = _ensure_store()
    document = await store.run_async(
        store.get_progress, user_id=user["user"], topic_id=topic_id
    )
    if not document:
        raise HTTPException(status_code=404, detail=f"No progress for topic '{topic_id}'")
    return document


@router.post("/{topic_id}/save", response_model=TopicProgressDocument)
async def save_topic_progress(
    topic_id: str,
    body: SaveProgressRequest,
    user: dict = Depends(require_auth),
):
    """Explicit save from the Save progress button.

    Raw data is written first so it is durable even if the summary never lands.
    The summary is awaited because the caller wants the finished note.
    """
    store = _ensure_store()
    user_id = str(user["user"])
    question_payload = [item.model_dump() for item in body.questions]
    question_texts = [text for text in (item.get("question", "") for item in question_payload) if text]

    document = await store.run_async(
        store.upsert_topic_progress,
        user_id=user_id,
        provider=str(user.get("provider", "") or ""),
        topic_id=topic_id,
        topic_title=body.topic_title,
        questions=question_texts,
        sections=body.sections,
        attempt_stats=build_attempt_stats(question_payload),
        preferred_language=body.preferred_language,
    )
    document = await _refresh_summary(
        store=store,
        user_id=user_id,
        topic_id=topic_id,
        document=document,
        user_identity=user,
        llm_config=None,
        on_failure="failed",
    )
    return document


@router.post("/{topic_id}/autosave", response_model=TopicProgressDocument)
async def autosave_topic_progress(
    topic_id: str,
    body: SaveProgressRequest,
    user: dict = Depends(require_auth),
):
    """``pagehide`` beacon target.

    The browser never waits for this response, so the summary is attempted inside
    the request but a failure deliberately leaves the row ``pending``: the next
    generation for this topic repairs it (see questions.py). Marking it ``failed``
    here would make the note permanently look owed-with-an-error for what is
    really just a dropped request.
    """
    store = _ensure_store()
    user_id = str(user["user"])
    question_payload = [item.model_dump() for item in body.questions]
    question_texts = [text for text in (item.get("question", "") for item in question_payload) if text]

    document = await store.run_async(
        store.upsert_topic_progress,
        user_id=user_id,
        provider=str(user.get("provider", "") or ""),
        topic_id=topic_id,
        topic_title=body.topic_title,
        questions=question_texts,
        sections=body.sections,
        attempt_stats=build_attempt_stats(question_payload),
        preferred_language=body.preferred_language,
    )
    # Mark owed before the LLM call so a dropped connection is still repairable.
    document = await store.run_async(store.mark_summary_pending, user_id=user_id, topic_id=topic_id) or document
    document = await _refresh_summary(
        store=store,
        user_id=user_id,
        topic_id=topic_id,
        document=document,
        user_identity=user,
        llm_config=None,
        on_failure="pending",
    )
    return document


async def _refresh_summary(
    *,
    store: ProgressStore,
    user_id: str,
    topic_id: str,
    document: dict,
    user_identity: dict,
    llm_config: LLMConfigRequest | None,
    on_failure: str,
) -> dict:
    """Generate and store a summary. Never raises; degrades to the fallback note."""
    if _summarizer is None:
        return document
    try:
        result = await _summarizer.generate(
            user_id=user_id,
            topic_title=str(document.get("topic_title", "")),
            sections=list(document.get("sections", [])),
            questions=list(document.get("questions_asked", [])),
            attempt_stats=dict(document.get("attempt_stats", {}) or {}),
            previous_summary=str(document.get("summary_text", "")),
            user_identity=user_identity,
            llm_config=llm_config,
        )
        return await store.run_async(
            store.set_summary,
            user_id=user_id,
            topic_id=topic_id,
            text=str(result.get("text", "")),
            provider_used=str(result.get("provider_used", "")),
            model_used=str(result.get("model_used", "")),
            source=str(result.get("source", "ai")),
        ) or document
    except Exception as exc:
        logger.exception("Progress summary write failed for user=%s topic=%s", user_id, topic_id)
        message = str(exc).strip() or "summary_failed"
        if on_failure == "failed":
            return (
                await store.run_async(
                    store.mark_summary_failed,
                    user_id=user_id,
                    topic_id=topic_id,
                    error=message,
                )
                or document
            )
        return document


def schedule_pending_summary_refresh(
    *,
    user_id: str,
    topic_id: str,
    user_identity: dict,
    llm_config: LLMConfigRequest | None = None,
) -> bool:
    """Repair a summary left ``pending`` by a dropped autosave.

    Fire-and-forget: the caller is a request handler that must not wait on a
    second LLM call. Returns whether a repair was actually scheduled.
    """
    if _store is None or _summarizer is None:
        return False
    key = (str(user_id), str(topic_id))
    if key in _summary_inflight:
        return False
    _summary_inflight.add(key)

    async def _run() -> None:
        try:
            store = _store
            document = await store.run_async(
                store.get_progress, user_id=user_id, topic_id=topic_id
            )
            if document is None or document.get("summary_status") != "pending":
                return
            await _refresh_summary(
                store=store,
                user_id=user_id,
                topic_id=topic_id,
                document=document,
                user_identity=user_identity,
                llm_config=llm_config,
                on_failure="pending",
            )
        except Exception:
            logger.exception("Pending summary refresh failed for user=%s topic=%s", user_id, topic_id)
        finally:
            _summary_inflight.discard(key)

    try:
        task = asyncio.create_task(_run())
    except RuntimeError:
        # No running loop (e.g. a sync call site in tests); nothing to repair.
        _summary_inflight.discard(key)
        return False
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return True
