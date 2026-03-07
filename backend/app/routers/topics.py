"""Topics router — static handbook topics plus user-defined custom topics."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.dependencies import require_auth
from app.schemas.models import (
    CreateCustomTopicRequest,
    ResponseDetailEnum,
    TopicDetail,
    TopicPreferencesResponse,
    TopicPreferencesUpdateRequest,
    TopicSummary,
)
from app.services.custom_topic_generator import CustomTopicGenerator
from app.services.doc_parser import DocParser
from app.services.learning_store import LearningStore
from app.services.mcp_gateway import MCPGateway
from app.services.topic_language_advisor import TopicLanguageAdvisor
from app.services.llm_policy import (
    LLMServiceApprovalRequiredError,
    PersonalCredentialRequiredError,
    StudyAppLLMNotAssignedError,
    policy_error_detail,
)
from app.services.llm_client import LLMClient

router = APIRouter(prefix="/api/topics", tags=["topics"])

_parser: DocParser | None = None
_llm_client: LLMClient | None = None
_learning_store: LearningStore | None = None
_mcp_gateway: MCPGateway | None = None


def init(
    parser: DocParser,
    llm_client: LLMClient,
    learning_store: LearningStore,
    mcp_gateway: MCPGateway | None = None,
):
    global _parser, _llm_client, _learning_store, _mcp_gateway
    _parser = parser
    _llm_client = llm_client
    _learning_store = learning_store
    _mcp_gateway = mcp_gateway


def _ensure_services() -> tuple[DocParser, LLMClient, LearningStore]:
    if _parser is None or _llm_client is None or _learning_store is None:
        raise HTTPException(status_code=503, detail="Service not initialised")
    return _parser, _llm_client, _learning_store


def _profile_is_stale(updated_at: str) -> bool:
    try:
        updated = datetime.fromisoformat(str(updated_at))
    except ValueError:
        return True
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=UTC)
    return updated < (datetime.now(UTC) - timedelta(days=7))


def _resolve_topic_detail_for_user(
    *,
    parser: DocParser,
    store: LearningStore,
    user_id: str,
    topic_id: str,
) -> TopicDetail | None:
    topic = parser.get_topic(topic_id)
    if topic:
        return topic
    custom = store.get_custom_topic(user_id=user_id, topic_id=topic_id)
    if custom:
        return TopicDetail(**custom)
    return None


async def _ensure_topic_language_profile(
    *,
    llm_client: LLMClient,
    store: LearningStore,
    topic: TopicDetail,
    user_identity: dict,
) -> dict:
    profile = store.get_topic_language_profile(topic_id=topic.id)
    if profile and not _profile_is_stale(profile.get("updated_at", "")):
        return profile

    advisor = TopicLanguageAdvisor(llm_client)
    mcp_context = ""
    if _mcp_gateway:
        mcp_context = await _mcp_gateway.gather_context(
            flow="questions",
            query=f"{topic.title} recommended programming languages interview preparation",
            topic_id=topic.id,
            topic_title=topic.title,
        )
    advised = await advisor.advise_topic(
        topic=topic,
        user_identity=user_identity,
        mcp_context=mcp_context,
    )
    return store.upsert_topic_language_profile(
        topic_id=topic.id,
        requires_programming=bool(advised.get("requires_programming")),
        language_options=[str(v) for v in advised.get("language_options", [])],
        source=str(advised.get("source", "llm")),
    )


@router.get("", response_model=list[TopicSummary])
async def list_topics(
    track: str | None = Query(default=None, pattern="^(backend|frontend|system_design|ai_stack)$"),
    level: str | None = Query(default=None, pattern="^(junior|mid|senior)$"),
    q: str | None = Query(default=None, max_length=200),
    user: dict = Depends(require_auth),
):
    """Return static topics + user custom topics with shared filters."""
    parser, _, store = _ensure_services()
    static_topics = parser.list_topics(track=track, level=level, q=q)
    custom_topics = [
        TopicSummary(**item)
        for item in store.list_custom_topics(
            user_id=user["user"], track=track, level=level, q=q
        )
    ]
    merged = sorted([*static_topics, *custom_topics], key=lambda t: t.id)
    return merged


@router.get("/{topic_id}", response_model=TopicDetail)
async def get_topic(topic_id: str, user: dict = Depends(require_auth)):
    """Return static topic first; then user-scoped custom topic."""
    parser, llm_client, store = _ensure_services()
    topic = _resolve_topic_detail_for_user(
        parser=parser,
        store=store,
        user_id=user["user"],
        topic_id=topic_id,
    )
    if topic is None:
        raise HTTPException(status_code=404, detail=f"Topic '{topic_id}' not found")

    profile = await _ensure_topic_language_profile(
        llm_client=llm_client,
        store=store,
        topic=topic,
        user_identity=user,
    )
    resolved = store.resolve_topic_ai_settings(
        user_id=user["user"],
        topic_id=topic_id,
        topic_detail=topic.model_dump(mode="json"),
    )

    return topic.model_copy(
        update={
            "requires_programming": bool(profile.get("requires_programming")),
            "language_options": [str(v) for v in profile.get("language_options", [])],
            "selected_language": resolved.get("preferred_language", ""),
            "response_detail": (
                ResponseDetailEnum.VERY_DETAILED
                if resolved.get("response_detail") == "very_detailed"
                else ResponseDetailEnum.CONCISE
            ),
        }
    )


@router.get("/{topic_id}/preferences", response_model=TopicPreferencesResponse)
async def get_topic_preferences(topic_id: str, user: dict = Depends(require_auth)):
    parser, llm_client, store = _ensure_services()
    topic = _resolve_topic_detail_for_user(
        parser=parser,
        store=store,
        user_id=user["user"],
        topic_id=topic_id,
    )
    if topic is None:
        raise HTTPException(status_code=404, detail=f"Topic '{topic_id}' not found")

    profile = await _ensure_topic_language_profile(
        llm_client=llm_client,
        store=store,
        topic=topic,
        user_identity=user,
    )
    resolved = store.resolve_topic_ai_settings(
        user_id=user["user"],
        topic_id=topic_id,
        topic_detail=topic.model_dump(mode="json"),
    )
    return TopicPreferencesResponse(
        topic_id=topic_id,
        response_detail=(
            ResponseDetailEnum.VERY_DETAILED
            if resolved.get("response_detail") == "very_detailed"
            else ResponseDetailEnum.CONCISE
        ),
        preferred_language=resolved.get("preferred_language", ""),
        requires_programming=bool(profile.get("requires_programming")),
        language_options=[str(v) for v in profile.get("language_options", [])],
    )


@router.put("/{topic_id}/preferences", response_model=TopicPreferencesResponse)
async def update_topic_preferences(
    topic_id: str,
    body: TopicPreferencesUpdateRequest,
    user: dict = Depends(require_auth),
):
    parser, llm_client, store = _ensure_services()
    topic = _resolve_topic_detail_for_user(
        parser=parser,
        store=store,
        user_id=user["user"],
        topic_id=topic_id,
    )
    if topic is None:
        raise HTTPException(status_code=404, detail=f"Topic '{topic_id}' not found")

    profile = await _ensure_topic_language_profile(
        llm_client=llm_client,
        store=store,
        topic=topic,
        user_identity=user,
    )
    allowed_languages = [str(v) for v in profile.get("language_options", [])]
    if body.preferred_language is not None:
        normalized = body.preferred_language.strip().lower()
        if normalized and normalized not in allowed_languages:
            raise HTTPException(
                status_code=422,
                detail=f"preferred_language must be one of: {', '.join(allowed_languages)}",
            )

    store.upsert_topic_preferences(
        user_id=user["user"],
        topic_id=topic_id,
        response_detail=(body.response_detail.value if body.response_detail else None),
        preferred_language=body.preferred_language,
    )
    resolved = store.resolve_topic_ai_settings(
        user_id=user["user"],
        topic_id=topic_id,
        topic_detail=topic.model_dump(mode="json"),
    )
    return TopicPreferencesResponse(
        topic_id=topic_id,
        response_detail=(
            ResponseDetailEnum.VERY_DETAILED
            if resolved.get("response_detail") == "very_detailed"
            else ResponseDetailEnum.CONCISE
        ),
        preferred_language=resolved.get("preferred_language", ""),
        requires_programming=bool(profile.get("requires_programming")),
        language_options=allowed_languages,
    )


@router.post("/custom", response_model=TopicDetail)
async def create_custom_topic(
    body: CreateCustomTopicRequest,
    user: dict = Depends(require_auth),
):
    """Analyze a custom topic and persist a deep user-scoped syllabus."""
    _, llm_client, store = _ensure_services()
    generator = CustomTopicGenerator(llm_client, _mcp_gateway)
    generated = await generator.generate_topic(
        topic=body.topic,
        target_sections=body.target_sections,
        llm_config=body.llm_config,
        user_identity=user,
    )
    store.upsert_custom_topic(
        user_id=user["user"],
        topic_id=generated.id,
        source_topic=body.topic.strip(),
        title=generated.title,
        description=generated.description,
        track=generated.track,
        levels=generated.levels,
        sections=generated.sections,
        raw_content=generated.raw_content,
    )
    return generated


@router.post("/custom/stream")
async def create_custom_topic_stream(
    body: CreateCustomTopicRequest,
    user: dict = Depends(require_auth),
):
    """Analyze a custom topic and stream generation events as NDJSON."""
    _, llm_client, store = _ensure_services()
    generator = CustomTopicGenerator(llm_client, _mcp_gateway)

    async def _event_stream():
        topic_name = body.topic.strip()
        target = int(body.target_sections)
        yield json.dumps(
            {
                "type": "start",
                "topic": topic_name,
                "target_sections": target,
            }
        ) + "\n"

        try:
            generation_task = asyncio.create_task(
                generator.generate_topic(
                    topic=topic_name,
                    target_sections=target,
                    llm_config=body.llm_config,
                    user_identity=user,
                )
            )

            elapsed = 0
            while not generation_task.done():
                yield json.dumps(
                    {
                        "type": "progress",
                        "stage": "analyzing",
                        "message": "Analyzing custom topic and building roadmap...",
                        "elapsed_seconds": elapsed,
                    }
                ) + "\n"
                elapsed += 1
                await asyncio.sleep(1)

            generated = await generation_task
            store.upsert_custom_topic(
                user_id=user["user"],
                topic_id=generated.id,
                source_topic=topic_name,
                title=generated.title,
                description=generated.description,
                track=generated.track,
                levels=generated.levels,
                sections=generated.sections,
                raw_content=generated.raw_content,
            )

            total = len(generated.sections)
            for idx, sec in enumerate(generated.sections, start=1):
                yield json.dumps(
                    {
                        "type": "section",
                        "index": idx,
                        "total_sections": total,
                        "heading": sec.get("heading", ""),
                        "content": sec.get("content", ""),
                    }
                ) + "\n"
                if idx % 8 == 0:
                    await asyncio.sleep(0)

            yield json.dumps(
                {
                    "type": "done",
                    "topic": generated.model_dump(mode="json"),
                }
            ) + "\n"
        except (
            LLMServiceApprovalRequiredError,
            StudyAppLLMNotAssignedError,
            PersonalCredentialRequiredError,
        ) as exc:
            detail = policy_error_detail(exc)
            yield json.dumps(
                {
                    "type": "error",
                    "code": detail["code"] or "generation_failed",
                    "message": detail["message"] or "Custom topic generation blocked by policy.",
                }
            ) + "\n"
        except Exception:
            yield json.dumps(
                {
                    "type": "error",
                    "code": "generation_failed",
                    "message": "Custom topic generation failed.",
                }
            ) + "\n"

    return StreamingResponse(_event_stream(), media_type="application/x-ndjson")
