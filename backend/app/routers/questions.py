"""Questions router — generate interview Q&A and quizzes via LLM."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.dependencies import require_auth
from app.routers import progress as progress_router
from app.schemas.models import (
    GenerateQuestionsRequest,
    GenerateQuestionsResponse,
    GenerateQuestionsV2Response,
    GenerateQuizRequest,
    GenerateQuizResponse,
    GenerateQuizV2Response,
    TopicDetail,
)
from app.services.doc_parser import DocParser
from app.services.learning_store import LearningStore
from app.services.llm_client import LLMClient
from app.services.mcp_gateway import MCPGateway
from app.services.progress_store import ProgressStore
from app.services.question_generator import QuestionGenerator
from app.services.topic_catalog import (
    PROBLEM_SOLVING_DEFAULT_LANGUAGE,
    is_problem_solving_topic,
    problem_solving_base_topic,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/questions", tags=["questions"])

# ── Prompt bounds (Stage 2.2) ────────────────────────────────
# `GenerateQuestionsRequest.section_content` / `section_title` are unbounded in
# the Pydantic schema and are embedded verbatim in the generation prompt, so
# they are bounded here. These match the `max_length` values requested for
# `schemas/models.py` so both layers agree on the same ceiling.
MAX_SECTION_CONTENT_CHARS = 12000
MAX_SECTION_TITLE_CHARS = 200
MAX_EXISTING_QUESTION_CHARS = 400

#: Bound on how many topics are resolved at once (Stage 3.5). The per-topic
#: resolution steps are independent, so they run concurrently; the semaphore
#: keeps one wide multi-topic request from monopolising the event loop or the
#: single-worker SQLite executor.
TOPIC_RESOLVE_CONCURRENCY = 4
PROGRESS_FETCH_CONCURRENCY = 4

#: Response header that surfaces a topic whose resolution failed while the rest
#: of the request still succeeded.
PARTIAL_FAILURE_HEADER = "X-Topic-Resolution-Failures"

# Injected at startup from main.py
_llm_client: LLMClient | None = None
_parser: DocParser | None = None
_learning_store: LearningStore | None = None
_mcp_gateway: MCPGateway | None = None
_progress_store: ProgressStore | None = None


class _TopicResolutionError(Exception):
    """Deterministic per-topic rejection (unknown topic, roadmap missing)."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _bounded(value: str | None, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _bounded_section_request(body) -> tuple[str, str, list[str]]:
    """Clamp the free-text section inputs before they reach a prompt.

    The section content is the single largest client-controlled block in the
    questions flow, so it is truncated rather than rejected: an oversized
    section must still produce a usable answer from the remaining context.
    """
    return (
        _bounded(body.section_title, MAX_SECTION_TITLE_CHARS),
        _bounded(body.section_content, MAX_SECTION_CONTENT_CHARS),
        [_bounded(item, MAX_EXISTING_QUESTION_CHARS) for item in (body.existing_questions or [])],
    )


async def _gather_bounded(coros, *, limit: int) -> list:
    """Run independent coroutines concurrently under a bounded semaphore.

    ``asyncio.gather`` preserves input order, so the reassembled list matches
    the request order exactly regardless of completion order.
    ``return_exceptions=True`` keeps one failing sibling from cancelling the
    rest of the batch.
    """
    semaphore = asyncio.Semaphore(max(1, int(limit)))

    async def _run(coro):
        async with semaphore:
            return await coro

    return list(await asyncio.gather(*(_run(coro) for coro in coros), return_exceptions=True))


def init(
    llm_client: LLMClient,
    parser: DocParser,
    learning_store: LearningStore,
    mcp_gateway: MCPGateway | None = None,
    progress_store: ProgressStore | None = None,
):
    global _llm_client, _parser, _learning_store, _mcp_gateway, _progress_store
    _llm_client = llm_client
    _parser = parser
    _learning_store = learning_store
    _mcp_gateway = mcp_gateway
    _progress_store = progress_store


def _ensure_services() -> tuple[LLMClient, DocParser, LearningStore]:
    if _llm_client is None or _parser is None or _learning_store is None:
        raise HTTPException(status_code=503, detail="Service not initialised")
    return _llm_client, _parser, _learning_store


async def _resolve_topic_for_user(
    *,
    topic_id: str,
    user_id: str,
    preferred_language_hint: str | None = None,
) -> tuple[TopicDetail, str] | None:
    _, parser, store = _ensure_services()
    if is_problem_solving_topic(topic_id):
        language = (preferred_language_hint or "").strip().lower()
        if not language:
            resolved = await store.run_async(
                store.resolve_topic_ai_settings,
                user_id=user_id,
                topic_id=topic_id,
                topic_detail=problem_solving_base_topic(),
            )
            language = resolved.get("preferred_language", "") or PROBLEM_SOLVING_DEFAULT_LANGUAGE
        detail = await store.run_async(
            store.resolve_problem_solving_topic_detail,
            user_id=user_id,
            preferred_language=language,
        )
        dynamic_topic = TopicDetail(**detail)
        return dynamic_topic, dynamic_topic.raw_content

    static_topic = parser.get_topic(topic_id)
    if static_topic:
        return static_topic, parser.get_topic_content(topic_id)

    custom_topic = await store.run_async(
        store.get_custom_topic,
        user_id=user_id,
        topic_id=topic_id,
    )
    if custom_topic:
        detail = TopicDetail(**custom_topic)
        return detail, detail.raw_content
    return None


async def _resolve_ai_settings(
    *,
    store: LearningStore,
    user_id: str,
    topic: TopicDetail,
    response_detail_override: str | None,
    preferred_language_override: str | None,
) -> dict:
    resolved = await store.run_async(
        store.resolve_topic_ai_settings,
        user_id=user_id,
        topic_id=topic.id,
        topic_detail=topic.model_dump(mode="json"),
    )
    if response_detail_override is not None:
        resolved["response_detail"] = (
            "very_detailed" if str(response_detail_override) == "very_detailed" else "concise"
        )
    if preferred_language_override is not None:
        normalized = preferred_language_override.strip().lower()
        if resolved.get("requires_programming"):
            allowed = [str(v) for v in resolved.get("language_options", [])]
            resolved["preferred_language"] = normalized if (not normalized or normalized in allowed) else ""
        else:
            resolved["preferred_language"] = ""
    return resolved


# ── Prior-progress wiring ──────────────────────────────────────
#
# Two independent no-repeat mechanisms are applied here:
#   1. `additional_existing_questions` unions the server-stored history into the
#      existing `uniqueness_block`.
#   2. `prior_progress` injects the stored AI summary as a semantic block.
# Both are best-effort: progress is an optimisation, never a reason to fail a
# generation request.


async def _load_progress_context(
    *,
    user_id: str,
    topic_ids: list[str],
    user_identity: dict,
    llm_config=None,
) -> tuple[str, list[str]]:
    """Return ``(prior_progress, additional_existing_questions)`` for generation.

    Asked questions are interleaved round-robin across the requested topics so a
    multi-topic quiz does not fill the whole prompt budget from the first topic.

    Per-topic reads are independent, so they run concurrently (Stage 3.5) and
    are reassembled in request order.
    """
    store = _progress_store
    if store is None:
        return "", []

    async def _load_one(topic_id: str) -> tuple[str, list[str], bool] | None:
        document = await store.run_async(
            store.get_progress, user_id=user_id, topic_id=topic_id
        )
        if not document:
            return None
        asked = await store.run_async(
            store.get_asked_questions,
            user_id=user_id,
            topic_id=topic_id,
            limit=200,
        )
        title = str(document.get("topic_title") or topic_id)
        summary = str(document.get("summary_text") or "").strip()
        pending = str(document.get("summary_status") or "") == "pending"
        return (f"{title}: {summary}" if summary else ""), asked, pending

    results = await _gather_bounded(
        [_load_one(topic_id) for topic_id in topic_ids],
        limit=PROGRESS_FETCH_CONCURRENCY,
    )

    summaries: list[str] = []
    per_topic: list[list[str]] = []
    failed = False
    for topic_id, result in zip(topic_ids, results):
        if isinstance(result, BaseException):
            failed = True
            logger.error(
                "Failed to load prior progress topic_id=%s error=%r",
                topic_id,
                result,
                exc_info=result,
            )
            continue
        if result is None:
            continue
        summary_line, asked, pending = result
        if summary_line:
            summaries.append(summary_line)
        if pending:
            # A pagehide beacon was dropped before its summary landed; repair
            # it in the background while this generation proceeds.
            progress_router.schedule_pending_summary_refresh(
                user_id=user_id,
                topic_id=topic_id,
                user_identity=user_identity,
                llm_config=llm_config,
            )
        per_topic.append(asked)

    if failed:
        # All-or-nothing, as before: a half-read history would silently bias
        # the no-repeat block.
        return "", []

    asked: list[str] = []
    for index in range(max((len(items) for items in per_topic), default=0)):
        for items in per_topic:
            if index < len(items):
                asked.append(items[index])
    return "\n\n".join(summaries), asked


async def _record_generated_questions(
    *,
    user_id: str,
    topic_id: str,
    topic_title: str,
    question_texts: list[str],
    section_title: str = "",
    preferred_language: str = "",
    provider: str = "",
) -> None:
    """Server-side capture: the stored history becomes a superset of the client's."""
    store = _progress_store
    if store is None or not question_texts:
        return
    try:
        await store.run_async(
            store.record_asked_questions,
            user_id=user_id,
            topic_id=topic_id,
            topic_title=topic_title,
            questions=question_texts,
            section_title=section_title,
            preferred_language=preferred_language,
            provider=provider,
        )
    except Exception:
        logger.exception("Failed to record asked questions for topic=%s", topic_id)


def _group_questions_by_topic(question_dicts: list[dict]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for item in question_dicts:
        if not isinstance(item, dict):
            continue
        text = str(item.get("question", "") or "").strip()
        key = str(item.get("topic_id") or "").strip()
        if not text or not key:
            continue
        grouped.setdefault(key, []).append(text)
    return grouped


async def _capture_grouped_questions(
    *,
    user_id: str,
    topic_titles: dict[str, str],
    question_dicts: list[dict],
    section_title: str = "",
    preferred_language: str = "",
    provider: str = "",
) -> None:
    """Capture a batch, grouped per topic. Used by the multi-topic quiz paths."""
    grouped = _group_questions_by_topic(question_dicts)
    for topic_id, texts in grouped.items():
        await _record_generated_questions(
            user_id=user_id,
            topic_id=topic_id,
            topic_title=topic_titles.get(topic_id, topic_id),
            question_texts=texts,
            section_title=section_title,
            preferred_language=preferred_language,
            provider=provider,
        )


# ── Multi-topic quiz resolution (Stage 3.5) ────────────────────
#
# Every requested topic is resolved independently: its content comes from a
# separate parser/store read and its AI settings from a separate preference
# read. Running them one after another multiplies the request latency by the
# topic count for no reason, so they run through a bounded `asyncio.gather`
# and are reassembled in request order.


async def _resolve_quiz_topic(*, topic_id: str, user: dict, store: LearningStore, body) -> dict:
    resolved = await _resolve_topic_for_user(
        topic_id=topic_id,
        user_id=user["user"],
        preferred_language_hint=body.preferred_language,
    )
    if not resolved:
        raise _TopicResolutionError(404, f"Topic '{topic_id}' not found")
    topic, doc_content = resolved
    if is_problem_solving_topic(topic_id) and not topic.content_ready:
        raise _TopicResolutionError(
            409,
            "Problem Solving roadmap is not generated yet for this language.",
        )
    ai_settings = await _resolve_ai_settings(
        store=store,
        user_id=user["user"],
        topic=topic,
        response_detail_override=(body.response_detail.value if body.response_detail else None),
        preferred_language_override=body.preferred_language,
    )
    return {
        "id": topic_id,
        "title": topic.title,
        "content": doc_content,
        "response_detail": ai_settings.get("response_detail", "concise"),
        "preferred_language": ai_settings.get("preferred_language", ""),
        "requires_programming": bool(ai_settings.get("requires_programming")),
    }


async def _resolve_quiz_topics(
    *,
    topic_ids: list[str],
    user: dict,
    store: LearningStore,
    body,
) -> tuple[list[dict], list[str]]:
    """Resolve every requested topic concurrently, preserving request order.

    Returns ``(payloads, failed_topic_ids)``.

    A deterministic rejection (unknown topic, roadmap not generated yet) still
    fails the request: silently generating a quiz without a requested topic
    would misreport coverage. An *unexpected* failure is logged and skipped so
    one broken topic cannot fail an otherwise usable multi-topic request; the
    caller surfaces it via a response header / stream warning.
    """
    if not topic_ids:
        raise HTTPException(status_code=400, detail="At least one topic_id required")

    results = await _gather_bounded(
        [_resolve_quiz_topic(topic_id=tid, user=user, store=store, body=body) for tid in topic_ids],
        limit=TOPIC_RESOLVE_CONCURRENCY,
    )

    payloads: list[dict] = []
    failed: list[str] = []
    rejected: _TopicResolutionError | None = None
    for topic_id, result in zip(topic_ids, results):
        if isinstance(result, _TopicResolutionError):
            rejected = result if rejected is None else rejected
            continue
        if isinstance(result, BaseException):
            logger.error(
                "topic_resolution_failed topic_id=%s error=%r",
                topic_id,
                result,
                exc_info=result,
            )
            failed.append(topic_id)
            continue
        payloads.append(result)

    if rejected is not None:
        raise HTTPException(status_code=rejected.status_code, detail=rejected.detail)
    if not payloads:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "topic_resolution_failed",
                "message": "None of the requested topics could be resolved.",
            },
        )
    return payloads, failed


def _note_partial_failures(response: Response, failed: list[str]) -> None:
    """Surface a skipped topic to the client without failing the whole request."""
    if not failed:
        return
    response.headers[PARTIAL_FAILURE_HEADER] = ",".join(failed)
    logger.warning(
        "quiz_generation_partial_topic_failure skipped_topics=%s",
        ",".join(failed),
    )


@router.post("/generate", response_model=GenerateQuestionsResponse)
async def generate_questions(
    body: GenerateQuestionsRequest,
    user: dict = Depends(require_auth),
):
    """Generate interview questions for a topic using the configured LLM."""
    llm_client, _, _ = _ensure_services()

    resolved = await _resolve_topic_for_user(
        topic_id=body.topic_id,
        user_id=user["user"],
        preferred_language_hint=body.preferred_language,
    )
    if not resolved:
        raise HTTPException(status_code=404, detail=f"Topic '{body.topic_id}' not found")
    topic, doc_content = resolved
    if is_problem_solving_topic(body.topic_id) and not topic.content_ready:
        raise HTTPException(
            status_code=409,
            detail="Problem Solving roadmap is not generated yet for this language.",
        )
    _, _, store = _ensure_services()
    generator = QuestionGenerator(llm_client, mcp_gateway=_mcp_gateway)
    ai_settings = await _resolve_ai_settings(
        store=store,
        user_id=user["user"],
        topic=topic,
        response_detail_override=(body.response_detail.value if body.response_detail else None),
        preferred_language_override=body.preferred_language,
    )
    prior_progress, additional_existing = await _load_progress_context(
        user_id=user["user"],
        topic_ids=[body.topic_id],
        user_identity=user,
        llm_config=body.llm_config,
    )

    section_title, section_content, existing_questions = _bounded_section_request(body)
    response = await generator.generate(
        topic_id=body.topic_id,
        topic_title=topic.title,
        doc_content=doc_content,
        count=body.count,
        requested_total_count=body.requested_total_count,
        existing_questions=existing_questions,
        difficulty=body.difficulty,
        level=body.level,
        llm_config=body.llm_config,
        user_identity=user,
        section_title=section_title,
        section_content=section_content,
        response_detail=ai_settings.get("response_detail", "very_detailed"),
        preferred_language=ai_settings.get("preferred_language", ""),
        requires_programming=bool(ai_settings.get("requires_programming")),
        prior_progress=prior_progress,
        additional_existing_questions=additional_existing,
    )
    await _record_generated_questions(
        user_id=user["user"],
        topic_id=body.topic_id,
        topic_title=topic.title,
        question_texts=[q.question for q in response.questions],
        section_title=body.section_title or "",
        preferred_language=ai_settings.get("preferred_language", ""),
        provider=str(user.get("provider", "") or ""),
    )
    return response


@router.post("/generate-v2", response_model=GenerateQuestionsV2Response)
async def generate_questions_v2(
    body: GenerateQuestionsRequest,
    user: dict = Depends(require_auth),
):
    """Generate grounded interview questions (strict schema v2)."""
    llm_client, _, _ = _ensure_services()
    if not get_settings().enable_v2_generation:
        raise HTTPException(status_code=404, detail="v2 generation disabled")

    resolved = await _resolve_topic_for_user(
        topic_id=body.topic_id,
        user_id=user["user"],
        preferred_language_hint=body.preferred_language,
    )
    if not resolved:
        raise HTTPException(status_code=404, detail=f"Topic '{body.topic_id}' not found")
    topic, doc_content = resolved
    if is_problem_solving_topic(body.topic_id) and not topic.content_ready:
        raise HTTPException(
            status_code=409,
            detail="Problem Solving roadmap is not generated yet for this language.",
        )
    _, _, store = _ensure_services()
    generator = QuestionGenerator(llm_client, mcp_gateway=_mcp_gateway)
    ai_settings = await _resolve_ai_settings(
        store=store,
        user_id=user["user"],
        topic=topic,
        response_detail_override=(body.response_detail.value if body.response_detail else None),
        preferred_language_override=body.preferred_language,
    )
    prior_progress, additional_existing = await _load_progress_context(
        user_id=user["user"],
        topic_ids=[body.topic_id],
        user_identity=user,
        llm_config=body.llm_config,
    )
    section_title, section_content, existing_questions = _bounded_section_request(body)
    response = await generator.generate_v2(
        topic_id=body.topic_id,
        topic_title=topic.title,
        doc_content=doc_content,
        count=body.count,
        requested_total_count=body.requested_total_count,
        existing_questions=existing_questions,
        difficulty=body.difficulty,
        level=body.level,
        llm_config=body.llm_config,
        user_identity=user,
        section_title=section_title,
        section_content=section_content,
        response_detail=ai_settings.get("response_detail", "very_detailed"),
        preferred_language=ai_settings.get("preferred_language", ""),
        requires_programming=bool(ai_settings.get("requires_programming")),
        prior_progress=prior_progress,
        additional_existing_questions=additional_existing,
    )
    await _record_generated_questions(
        user_id=user["user"],
        topic_id=body.topic_id,
        topic_title=topic.title,
        question_texts=[q.question for q in response.questions],
        section_title=body.section_title or "",
        preferred_language=ai_settings.get("preferred_language", ""),
        provider=str(user.get("provider", "") or ""),
    )
    return response


@router.post("/generate-v2/stream")
async def generate_questions_v2_stream(
    body: GenerateQuestionsRequest,
    user: dict = Depends(require_auth),
):
    """Generate grounded interview questions with incremental NDJSON events."""
    llm_client, _, _ = _ensure_services()
    if not get_settings().enable_v2_generation:
        raise HTTPException(status_code=404, detail="v2 generation disabled")

    resolved = await _resolve_topic_for_user(
        topic_id=body.topic_id,
        user_id=user["user"],
        preferred_language_hint=body.preferred_language,
    )
    if not resolved:
        raise HTTPException(status_code=404, detail=f"Topic '{body.topic_id}' not found")
    topic, doc_content = resolved
    if is_problem_solving_topic(body.topic_id) and not topic.content_ready:
        raise HTTPException(
            status_code=409,
            detail="Problem Solving roadmap is not generated yet for this language.",
        )
    _, _, store = _ensure_services()
    generator = QuestionGenerator(llm_client, mcp_gateway=_mcp_gateway)
    ai_settings = await _resolve_ai_settings(
        store=store,
        user_id=user["user"],
        topic=topic,
        response_detail_override=(body.response_detail.value if body.response_detail else None),
        preferred_language_override=body.preferred_language,
    )

    prior_progress, additional_existing = await _load_progress_context(
        user_id=user["user"],
        topic_ids=[body.topic_id],
        user_identity=user,
        llm_config=body.llm_config,
    )

    async def _event_stream():
        captured: list[dict] = []
        section_title, section_content, existing_questions = _bounded_section_request(body)
        async for event in generator.generate_v2_stream(
            topic_id=body.topic_id,
            topic_title=topic.title,
            doc_content=doc_content,
            count=body.count,
            requested_total_count=body.requested_total_count,
            existing_questions=existing_questions,
            difficulty=body.difficulty,
            level=body.level,
            llm_config=body.llm_config,
            user_identity=user,
            section_title=section_title,
            section_content=section_content,
            response_detail=ai_settings.get("response_detail", "very_detailed"),
            preferred_language=ai_settings.get("preferred_language", ""),
            requires_programming=bool(ai_settings.get("requires_programming")),
            prior_progress=prior_progress,
            additional_existing_questions=additional_existing,
        ):
            if event.get("type") == "question":
                captured.append(event.get("question") or {})
            yield json.dumps(event) + "\n"
            await asyncio.sleep(0)
        # Capture runs after the stream so a slow write can never stall delivery.
        await _record_generated_questions(
            user_id=user["user"],
            topic_id=body.topic_id,
            topic_title=topic.title,
            question_texts=[
                str(item.get("question", "") or "")
                for item in captured
                if isinstance(item, dict)
            ],
            section_title=body.section_title or "",
            preferred_language=ai_settings.get("preferred_language", ""),
            provider=str(user.get("provider", "") or ""),
        )

    return StreamingResponse(
        _event_stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/quiz/generate", response_model=GenerateQuizResponse)
async def generate_quiz(
    body: GenerateQuizRequest,
    http_response: Response,
    user: dict = Depends(require_auth),
):
    """Generate MCQ and/or True/False quiz questions across one or more topics."""
    llm_client, _, _ = _ensure_services()

    _, _, store = _ensure_services()
    topics_content, failed_topics = await _resolve_quiz_topics(
        topic_ids=body.topic_ids,
        user=user,
        store=store,
        body=body,
    )
    _note_partial_failures(http_response, failed_topics)

    generator = QuestionGenerator(llm_client, mcp_gateway=_mcp_gateway)
    prior_progress, additional_existing = await _load_progress_context(
        user_id=user["user"],
        topic_ids=[tc["id"] for tc in topics_content],
        user_identity=user,
        llm_config=body.llm_config,
    )
    response = await generator.generate_quiz(
        topics_content=topics_content,
        count=body.count,
        question_types=[qt.value for qt in body.question_types],
        difficulty=body.difficulty,
        level=body.level,
        response_detail=(body.response_detail.value if body.response_detail else None),
        preferred_language=(body.preferred_language or ""),
        llm_config=body.llm_config,
        user_identity=user,
        prior_progress=prior_progress,
        additional_existing_questions=additional_existing,
    )
    await _capture_grouped_questions(
        user_id=user["user"],
        topic_titles={tc["id"]: tc["title"] for tc in topics_content},
        question_dicts=[q.model_dump() for q in response.questions],
        preferred_language=(body.preferred_language or ""),
        provider=str(user.get("provider", "") or ""),
    )
    return response


@router.post("/quiz/generate-v2", response_model=GenerateQuizV2Response)
async def generate_quiz_v2(
    body: GenerateQuizRequest,
    http_response: Response,
    user: dict = Depends(require_auth),
):
    """Generate strictly validated quiz questions with grounding fields."""
    llm_client, _, _ = _ensure_services()
    if not get_settings().enable_v2_generation:
        raise HTTPException(status_code=404, detail="v2 generation disabled")

    _, _, store = _ensure_services()
    topics_content, failed_topics = await _resolve_quiz_topics(
        topic_ids=body.topic_ids,
        user=user,
        store=store,
        body=body,
    )
    _note_partial_failures(http_response, failed_topics)

    generator = QuestionGenerator(llm_client, mcp_gateway=_mcp_gateway)
    prior_progress, additional_existing = await _load_progress_context(
        user_id=user["user"],
        topic_ids=[tc["id"] for tc in topics_content],
        user_identity=user,
        llm_config=body.llm_config,
    )
    response = await generator.generate_quiz_v2(
        topics_content=topics_content,
        count=body.count,
        question_types=[qt.value for qt in body.question_types],
        difficulty=body.difficulty,
        level=body.level,
        response_detail=(body.response_detail.value if body.response_detail else None),
        preferred_language=(body.preferred_language or ""),
        llm_config=body.llm_config,
        user_identity=user,
        prior_progress=prior_progress,
        additional_existing_questions=additional_existing,
    )
    await _capture_grouped_questions(
        user_id=user["user"],
        topic_titles={tc["id"]: tc["title"] for tc in topics_content},
        question_dicts=[q.model_dump() for q in response.questions],
        preferred_language=(body.preferred_language or ""),
        provider=str(user.get("provider", "") or ""),
    )
    return response


@router.post("/quiz/generate-v2/stream")
async def generate_quiz_v2_stream(
    body: GenerateQuizRequest,
    user: dict = Depends(require_auth),
):
    """Generate grounded quiz questions with incremental NDJSON events."""
    llm_client, _, _ = _ensure_services()
    if not get_settings().enable_v2_generation:
        raise HTTPException(status_code=404, detail="v2 generation disabled")

    _, _, store = _ensure_services()
    topics_content, failed_topics = await _resolve_quiz_topics(
        topic_ids=body.topic_ids,
        user=user,
        store=store,
        body=body,
    )
    generator = QuestionGenerator(llm_client, mcp_gateway=_mcp_gateway)
    prior_progress, additional_existing = await _load_progress_context(
        user_id=user["user"],
        topic_ids=[tc["id"] for tc in topics_content],
        user_identity=user,
        llm_config=body.llm_config,
    )
    topic_titles = {tc["id"]: tc["title"] for tc in topics_content}

    async def _event_stream():
        captured: list[dict] = []
        if failed_topics:
            # The response has already started, so a partial topic failure is
            # surfaced as an event rather than a status code.
            logger.warning(
                "quiz_generation_partial_topic_failure topics=%s", ",".join(failed_topics)
            )
            yield json.dumps(
                {
                    "type": "warning",
                    "code": "topic_resolution_failed",
                    "message": "Some requested topics could not be resolved and were skipped.",
                    "topics": list(failed_topics),
                    "topics_used": [tc["id"] for tc in topics_content],
                }
            ) + "\n"
        async for event in generator.generate_quiz_v2_stream(
            topics_content=topics_content,
            count=body.count,
            question_types=[qt.value for qt in body.question_types],
            difficulty=body.difficulty,
            level=body.level,
            response_detail=(body.response_detail.value if body.response_detail else None),
            preferred_language=(body.preferred_language or ""),
            llm_config=body.llm_config,
            user_identity=user,
            prior_progress=prior_progress,
            additional_existing_questions=additional_existing,
        ):
            if event.get("type") == "question":
                captured.append(event.get("question") or {})
            yield json.dumps(event) + "\n"
            await asyncio.sleep(0)
        # Capture runs after the stream so a slow write can never stall delivery.
        await _capture_grouped_questions(
            user_id=user["user"],
            topic_titles=topic_titles,
            question_dicts=captured,
            preferred_language=(body.preferred_language or ""),
            provider=str(user.get("provider", "") or ""),
        )

    return StreamingResponse(
        _event_stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
