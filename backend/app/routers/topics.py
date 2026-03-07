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
    GenerateTopicContentRequest,
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
from app.services.problem_solving_generator import ProblemSolvingGenerator
from app.services.topic_catalog import (
    PROBLEM_SOLVING_DEFAULT_LANGUAGE,
    PROBLEM_SOLVING_LANGUAGE_OPTIONS,
    PROBLEM_SOLVING_LEVELS,
    PROBLEM_SOLVING_TARGET_SECTIONS,
    PROBLEM_SOLVING_TITLE,
    PROBLEM_SOLVING_TOPIC_ID,
    PROBLEM_SOLVING_TRACK,
    is_problem_solving_topic,
    problem_solving_base_topic,
)
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
    if is_problem_solving_topic(topic_id):
        resolved = store.resolve_topic_ai_settings(
            user_id=user_id,
            topic_id=topic_id,
            topic_detail=problem_solving_base_topic(),
        )
        selected_language = (
            resolved.get("preferred_language") or PROBLEM_SOLVING_DEFAULT_LANGUAGE
        )
        dynamic = store.resolve_problem_solving_topic_detail(
            user_id=user_id,
            preferred_language=selected_language,
        )
        return TopicDetail(**dynamic)
    topic = parser.get_topic(topic_id)
    if topic:
        return topic
    custom = store.get_custom_topic(user_id=user_id, topic_id=topic_id)
    if custom:
        return TopicDetail(**custom)
    return None


def _passes_topic_filters(
    *,
    title: str,
    description: str,
    track: str,
    levels: list[str],
    track_filter: str | None,
    level_filter: str | None,
    query_filter: str | None,
) -> bool:
    if track_filter and track != track_filter:
        return False
    if level_filter and level_filter not in levels:
        return False
    q = (query_filter or "").strip().lower()
    if q and q not in f"{title} {description}".lower():
        return False
    return True


async def _ensure_topic_language_profile(
    *,
    llm_client: LLMClient,
    store: LearningStore,
    topic: TopicDetail,
    user_identity: dict,
) -> dict:
    if is_problem_solving_topic(topic.id):
        return store.upsert_topic_language_profile(
            topic_id=topic.id,
            requires_programming=True,
            language_options=list(PROBLEM_SOLVING_LANGUAGE_OPTIONS),
            source="builtin",
        )

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
    dynamic_topics: list[TopicSummary] = []
    if _passes_topic_filters(
        title=PROBLEM_SOLVING_TITLE,
        description=problem_solving_base_topic()["description"],
        track=PROBLEM_SOLVING_TRACK,
        levels=list(PROBLEM_SOLVING_LEVELS),
        track_filter=track,
        level_filter=level,
        query_filter=q,
    ):
        resolved = store.resolve_topic_ai_settings(
            user_id=user["user"],
            topic_id=PROBLEM_SOLVING_TOPIC_ID,
            topic_detail=problem_solving_base_topic(),
        )
        selected_language = (
            resolved.get("preferred_language") or PROBLEM_SOLVING_DEFAULT_LANGUAGE
        )
        cached = store.get_dynamic_topic_curriculum(
            user_id=user["user"],
            topic_id=PROBLEM_SOLVING_TOPIC_ID,
            preferred_language=selected_language,
        )
        section_count = len(cached.get("sections", [])) if cached else 0
        dynamic_topics.append(
            TopicSummary(
                id=PROBLEM_SOLVING_TOPIC_ID,
                title=PROBLEM_SOLVING_TITLE,
                description=problem_solving_base_topic()["description"],
                track=PROBLEM_SOLVING_TRACK,
                levels=list(PROBLEM_SOLVING_LEVELS),
                section_count=section_count,
                estimated_questions=max(5, section_count * 2) if section_count else 240,
            )
        )

    merged = sorted([*static_topics, *custom_topics, *dynamic_topics], key=lambda t: t.id)
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
            "is_dynamic_topic": bool(topic.is_dynamic_topic),
            "content_ready": bool(topic.content_ready and len(topic.sections) > 0),
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


@router.post("/{topic_id}/content/generate/stream")
async def generate_topic_content_stream(
    topic_id: str,
    body: GenerateTopicContentRequest,
    user: dict = Depends(require_auth),
):
    parser, llm_client, store = _ensure_services()
    if not is_problem_solving_topic(topic_id):
        raise HTTPException(status_code=404, detail=f"Topic '{topic_id}' not found")
    _ = parser  # parser is intentionally unused for this dynamic built-in topic.

    language = body.preferred_language.strip().lower()
    if language not in PROBLEM_SOLVING_LANGUAGE_OPTIONS:
        raise HTTPException(
            status_code=422,
            detail=(
                "preferred_language must be one of: "
                + ", ".join(PROBLEM_SOLVING_LANGUAGE_OPTIONS)
            ),
        )
    target_sections = PROBLEM_SOLVING_TARGET_SECTIONS
    generator = ProblemSolvingGenerator(llm_client, _mcp_gateway)

    async def _event_stream():
        yield json.dumps(
            {
                "type": "start",
                "topic_id": topic_id,
                "preferred_language": language,
                "target_sections": target_sections,
            }
        ) + "\n"
        try:
            store.upsert_topic_preferences(
                user_id=user["user"],
                topic_id=topic_id,
                preferred_language=language,
            )
            cached = store.get_dynamic_topic_curriculum(
                user_id=user["user"],
                topic_id=topic_id,
                preferred_language=language,
            )
            if cached and not body.force_regenerate:
                detail = store.resolve_problem_solving_topic_detail(
                    user_id=user["user"],
                    preferred_language=language,
                )
                yield json.dumps(
                    {
                        "type": "progress",
                        "stage": "ready",
                        "message": "Using cached roadmap for selected language.",
                    }
                ) + "\n"
                yield json.dumps({"type": "done", "topic": TopicDetail(**detail).model_dump(mode="json")}) + "\n"
                return

            if body.force_regenerate:
                store.delete_dynamic_topic_curriculum(
                    user_id=user["user"],
                    topic_id=topic_id,
                    preferred_language=language,
                )

            generation_task = asyncio.create_task(
                generator.generate_topic(
                    preferred_language=language,
                    target_sections=target_sections,
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
                        "message": "Generating language-specific problem solving roadmap...",
                        "elapsed_seconds": elapsed,
                    }
                ) + "\n"
                elapsed += 1
                await asyncio.sleep(1)

            generated = await generation_task
            store.upsert_dynamic_topic_curriculum(
                user_id=user["user"],
                topic_id=topic_id,
                preferred_language=language,
                title=generated.title,
                description=generated.description,
                track=generated.track,
                levels=list(generated.levels),
                sections=list(generated.sections),
                raw_content=generated.raw_content,
                target_sections=target_sections,
                source="llm",
            )

            for idx, sec in enumerate(generated.sections, start=1):
                yield json.dumps(
                    {
                        "type": "section",
                        "index": idx,
                        "total_sections": len(generated.sections),
                        "heading": sec.get("heading", ""),
                        "content": sec.get("content", ""),
                    }
                ) + "\n"
                if idx % 8 == 0:
                    await asyncio.sleep(0)

            detail = store.resolve_problem_solving_topic_detail(
                user_id=user["user"],
                preferred_language=language,
            )
            resolved = store.resolve_topic_ai_settings(
                user_id=user["user"],
                topic_id=topic_id,
                topic_detail=detail,
            )
            final_topic = TopicDetail(**detail).model_copy(
                update={
                    "selected_language": resolved.get("preferred_language", language),
                    "response_detail": (
                        ResponseDetailEnum.VERY_DETAILED
                        if resolved.get("response_detail") == "very_detailed"
                        else ResponseDetailEnum.CONCISE
                    ),
                    "is_dynamic_topic": True,
                    "content_ready": True,
                }
            )
            yield json.dumps({"type": "done", "topic": final_topic.model_dump(mode="json")}) + "\n"
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
                    "message": detail["message"] or "Dynamic topic generation blocked by policy.",
                }
            ) + "\n"
        except Exception:
            yield json.dumps(
                {
                    "type": "error",
                    "code": "generation_failed",
                    "message": "Dynamic topic generation failed.",
                }
            ) + "\n"

    return StreamingResponse(_event_stream(), media_type="application/x-ndjson")


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
