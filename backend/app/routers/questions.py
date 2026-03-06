"""Questions router — generate interview Q&A and quizzes via LLM."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

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
from app.services.question_generator import QuestionGenerator

router = APIRouter(prefix="/api/questions", tags=["questions"])

# Injected at startup from main.py
_llm_client: LLMClient | None = None
_parser: DocParser | None = None
_learning_store: LearningStore | None = None


def init(llm_client: LLMClient, parser: DocParser, learning_store: LearningStore):
    global _llm_client, _parser, _learning_store
    _llm_client = llm_client
    _parser = parser
    _learning_store = learning_store


def _ensure_services() -> tuple[LLMClient, DocParser, LearningStore]:
    if _llm_client is None or _parser is None or _learning_store is None:
        raise HTTPException(status_code=503, detail="Service not initialised")
    return _llm_client, _parser, _learning_store


def _resolve_topic_for_user(*, topic_id: str, user_id: str) -> tuple[TopicDetail, str] | None:
    _, parser, store = _ensure_services()
    static_topic = parser.get_topic(topic_id)
    if static_topic:
        return static_topic, parser.get_topic_content(topic_id)

    custom_topic = store.get_custom_topic(user_id=user_id, topic_id=topic_id)
    if custom_topic:
        detail = TopicDetail(**custom_topic)
        return detail, detail.raw_content
    return None


@router.post("/generate", response_model=GenerateQuestionsResponse)
async def generate_questions(
    body: GenerateQuestionsRequest,
    user: dict = Depends(require_auth),
):
    """Generate interview questions for a topic using the configured LLM."""
    llm_client, _, _ = _ensure_services()

    resolved = _resolve_topic_for_user(topic_id=body.topic_id, user_id=user["user"])
    if not resolved:
        raise HTTPException(status_code=404, detail=f"Topic '{body.topic_id}' not found")
    topic, doc_content = resolved
    generator = QuestionGenerator(llm_client)

    return await generator.generate(
        topic_id=body.topic_id,
        topic_title=topic.title,
        doc_content=doc_content,
        count=body.count,
        difficulty=body.difficulty,
        level=body.level,
        llm_config=body.llm_config,
        user_identity=user,
        section_title=body.section_title,
        section_content=body.section_content,
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

    resolved = _resolve_topic_for_user(topic_id=body.topic_id, user_id=user["user"])
    if not resolved:
        raise HTTPException(status_code=404, detail=f"Topic '{body.topic_id}' not found")
    topic, doc_content = resolved
    generator = QuestionGenerator(llm_client)
    return await generator.generate_v2(
        topic_id=body.topic_id,
        topic_title=topic.title,
        doc_content=doc_content,
        count=body.count,
        difficulty=body.difficulty,
        level=body.level,
        llm_config=body.llm_config,
        user_identity=user,
        section_title=body.section_title,
        section_content=body.section_content,
    )


@router.post("/quiz/generate", response_model=GenerateQuizResponse)
async def generate_quiz(
    body: GenerateQuizRequest,
    user: dict = Depends(require_auth),
):
    """Generate MCQ and/or True/False quiz questions across one or more topics."""
    llm_client, _, _ = _ensure_services()

    topics_content: list[dict] = []
    for tid in body.topic_ids:
        resolved = _resolve_topic_for_user(topic_id=tid, user_id=user["user"])
        if not resolved:
            raise HTTPException(status_code=404, detail=f"Topic '{tid}' not found")
        topic, doc_content = resolved
        topics_content.append(
            {
                "id": tid,
                "title": topic.title,
                "content": doc_content,
            }
        )

    if not topics_content:
        raise HTTPException(status_code=400, detail="At least one topic_id required")

    generator = QuestionGenerator(llm_client)
    return await generator.generate_quiz(
        topics_content=topics_content,
        count=body.count,
        question_types=[qt.value for qt in body.question_types],
        difficulty=body.difficulty,
        level=body.level,
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

    topics_content: list[dict] = []
    for tid in body.topic_ids:
        resolved = _resolve_topic_for_user(topic_id=tid, user_id=user["user"])
        if not resolved:
            raise HTTPException(status_code=404, detail=f"Topic '{tid}' not found")
        topic, doc_content = resolved
        topics_content.append(
            {
                "id": tid,
                "title": topic.title,
                "content": doc_content,
            }
        )

    if not topics_content:
        raise HTTPException(status_code=400, detail="At least one topic_id required")

    generator = QuestionGenerator(llm_client)
    return await generator.generate_quiz_v2(
        topics_content=topics_content,
        count=body.count,
        question_types=[qt.value for qt in body.question_types],
        difficulty=body.difficulty,
        level=body.level,
        llm_config=body.llm_config,
        user_identity=user,
    )
