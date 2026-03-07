"""Questions router — generate interview Q&A and quizzes via LLM."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.dependencies import require_auth
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
from app.services.question_generator import QuestionGenerator
from app.services.topic_catalog import (
    PROBLEM_SOLVING_DEFAULT_LANGUAGE,
    is_problem_solving_topic,
    problem_solving_base_topic,
)

router = APIRouter(prefix="/api/questions", tags=["questions"])

# Injected at startup from main.py
_llm_client: LLMClient | None = None
_parser: DocParser | None = None
_learning_store: LearningStore | None = None
_mcp_gateway: MCPGateway | None = None


def init(
    llm_client: LLMClient,
    parser: DocParser,
    learning_store: LearningStore,
    mcp_gateway: MCPGateway | None = None,
):
    global _llm_client, _parser, _learning_store, _mcp_gateway
    _llm_client = llm_client
    _parser = parser
    _learning_store = learning_store
    _mcp_gateway = mcp_gateway


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

    return await generator.generate(
        topic_id=body.topic_id,
        topic_title=topic.title,
        doc_content=doc_content,
        count=body.count,
        requested_total_count=body.requested_total_count,
        existing_questions=body.existing_questions,
        difficulty=body.difficulty,
        level=body.level,
        llm_config=body.llm_config,
        user_identity=user,
        section_title=body.section_title,
        section_content=body.section_content,
        response_detail=ai_settings.get("response_detail", "concise"),
        preferred_language=ai_settings.get("preferred_language", ""),
        requires_programming=bool(ai_settings.get("requires_programming")),
    )


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
    return await generator.generate_v2(
        topic_id=body.topic_id,
        topic_title=topic.title,
        doc_content=doc_content,
        count=body.count,
        requested_total_count=body.requested_total_count,
        existing_questions=body.existing_questions,
        difficulty=body.difficulty,
        level=body.level,
        llm_config=body.llm_config,
        user_identity=user,
        section_title=body.section_title,
        section_content=body.section_content,
        response_detail=ai_settings.get("response_detail", "concise"),
        preferred_language=ai_settings.get("preferred_language", ""),
        requires_programming=bool(ai_settings.get("requires_programming")),
    )


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

    async def _event_stream():
        async for event in generator.generate_v2_stream(
            topic_id=body.topic_id,
            topic_title=topic.title,
            doc_content=doc_content,
            count=body.count,
            requested_total_count=body.requested_total_count,
            existing_questions=body.existing_questions,
            difficulty=body.difficulty,
            level=body.level,
            llm_config=body.llm_config,
            user_identity=user,
            section_title=body.section_title,
            section_content=body.section_content,
            response_detail=ai_settings.get("response_detail", "concise"),
            preferred_language=ai_settings.get("preferred_language", ""),
            requires_programming=bool(ai_settings.get("requires_programming")),
        ):
            yield json.dumps(event) + "\n"
            await asyncio.sleep(0)

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
    user: dict = Depends(require_auth),
):
    """Generate MCQ and/or True/False quiz questions across one or more topics."""
    llm_client, _, _ = _ensure_services()

    _, _, store = _ensure_services()
    topics_content: list[dict] = []
    for tid in body.topic_ids:
        resolved = await _resolve_topic_for_user(
            topic_id=tid,
            user_id=user["user"],
            preferred_language_hint=body.preferred_language,
        )
        if not resolved:
            raise HTTPException(status_code=404, detail=f"Topic '{tid}' not found")
        topic, doc_content = resolved
        if is_problem_solving_topic(tid) and not topic.content_ready:
            raise HTTPException(
                status_code=409,
                detail="Problem Solving roadmap is not generated yet for this language.",
            )
        ai_settings = await _resolve_ai_settings(
            store=store,
            user_id=user["user"],
            topic=topic,
            response_detail_override=(body.response_detail.value if body.response_detail else None),
            preferred_language_override=body.preferred_language,
        )
        topics_content.append(
            {
                "id": tid,
                "title": topic.title,
                "content": doc_content,
                "response_detail": ai_settings.get("response_detail", "concise"),
                "preferred_language": ai_settings.get("preferred_language", ""),
                "requires_programming": bool(ai_settings.get("requires_programming")),
            }
        )

    if not topics_content:
        raise HTTPException(status_code=400, detail="At least one topic_id required")

    generator = QuestionGenerator(llm_client, mcp_gateway=_mcp_gateway)
    return await generator.generate_quiz(
        topics_content=topics_content,
        count=body.count,
        question_types=[qt.value for qt in body.question_types],
        difficulty=body.difficulty,
        level=body.level,
        response_detail=(body.response_detail.value if body.response_detail else None),
        preferred_language=(body.preferred_language or ""),
        llm_config=body.llm_config,
        user_identity=user,
    )


@router.post("/quiz/generate-v2", response_model=GenerateQuizV2Response)
async def generate_quiz_v2(
    body: GenerateQuizRequest,
    user: dict = Depends(require_auth),
):
    """Generate strictly validated quiz questions with grounding fields."""
    llm_client, _, _ = _ensure_services()
    if not get_settings().enable_v2_generation:
        raise HTTPException(status_code=404, detail="v2 generation disabled")

    _, _, store = _ensure_services()
    topics_content: list[dict] = []
    for tid in body.topic_ids:
        resolved = await _resolve_topic_for_user(
            topic_id=tid,
            user_id=user["user"],
            preferred_language_hint=body.preferred_language,
        )
        if not resolved:
            raise HTTPException(status_code=404, detail=f"Topic '{tid}' not found")
        topic, doc_content = resolved
        if is_problem_solving_topic(tid) and not topic.content_ready:
            raise HTTPException(
                status_code=409,
                detail="Problem Solving roadmap is not generated yet for this language.",
            )
        ai_settings = await _resolve_ai_settings(
            store=store,
            user_id=user["user"],
            topic=topic,
            response_detail_override=(body.response_detail.value if body.response_detail else None),
            preferred_language_override=body.preferred_language,
        )
        topics_content.append(
            {
                "id": tid,
                "title": topic.title,
                "content": doc_content,
                "response_detail": ai_settings.get("response_detail", "concise"),
                "preferred_language": ai_settings.get("preferred_language", ""),
                "requires_programming": bool(ai_settings.get("requires_programming")),
            }
        )

    if not topics_content:
        raise HTTPException(status_code=400, detail="At least one topic_id required")

    generator = QuestionGenerator(llm_client, mcp_gateway=_mcp_gateway)
    return await generator.generate_quiz_v2(
        topics_content=topics_content,
        count=body.count,
        question_types=[qt.value for qt in body.question_types],
        difficulty=body.difficulty,
        level=body.level,
        response_detail=(body.response_detail.value if body.response_detail else None),
        preferred_language=(body.preferred_language or ""),
        llm_config=body.llm_config,
        user_identity=user,
    )
