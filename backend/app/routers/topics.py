"""Topics router — static handbook topics plus user-defined custom topics."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.dependencies import require_auth
from app.schemas.models import CreateCustomTopicRequest, TopicDetail, TopicSummary
from app.services.custom_topic_generator import CustomTopicGenerator
from app.services.doc_parser import DocParser
from app.services.learning_store import LearningStore
from app.services.llm_client import LLMClient

router = APIRouter(prefix="/api/topics", tags=["topics"])

_parser: DocParser | None = None
_llm_client: LLMClient | None = None
_learning_store: LearningStore | None = None


def init(
    parser: DocParser,
    llm_client: LLMClient,
    learning_store: LearningStore,
):
    global _parser, _llm_client, _learning_store
    _parser = parser
    _llm_client = llm_client
    _learning_store = learning_store


def _ensure_services() -> tuple[DocParser, LLMClient, LearningStore]:
    if _parser is None or _llm_client is None or _learning_store is None:
        raise HTTPException(status_code=503, detail="Service not initialised")
    return _parser, _llm_client, _learning_store


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
    parser, _, store = _ensure_services()
    topic = parser.get_topic(topic_id)
    if topic:
        return topic
    custom = store.get_custom_topic(user_id=user["user"], topic_id=topic_id)
    if custom:
        return TopicDetail(**custom)
    raise HTTPException(status_code=404, detail=f"Topic '{topic_id}' not found")


@router.post("/custom", response_model=TopicDetail)
async def create_custom_topic(
    body: CreateCustomTopicRequest,
    user: dict = Depends(require_auth),
):
    """Analyze a custom topic and persist a deep user-scoped syllabus."""
    _, llm_client, store = _ensure_services()
    generator = CustomTopicGenerator(llm_client)
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
