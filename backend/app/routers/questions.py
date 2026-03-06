"""Questions router — generate interview Q&A and quizzes via LLM."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.schemas.models import (
    GenerateQuestionsRequest,
    GenerateQuestionsResponse,
    GenerateQuestionsV2Response,
    GenerateQuizRequest,
    GenerateQuizResponse,
    GenerateQuizV2Response,
)
from app.services.doc_parser import DocParser
from app.services.llm_client import LLMClient
from app.services.question_generator import QuestionGenerator

router = APIRouter(prefix="/api/questions", tags=["questions"])

# Injected at startup from main.py
_llm_client: LLMClient | None = None
_parser: DocParser | None = None


def init(llm_client: LLMClient, parser: DocParser):
    global _llm_client, _parser
    _llm_client = llm_client
    _parser = parser


@router.post("/generate", response_model=GenerateQuestionsResponse)
async def generate_questions(body: GenerateQuestionsRequest):
    """Generate interview questions for a topic using the configured LLM."""
    if _llm_client is None or _parser is None:
        raise HTTPException(status_code=503, detail="Service not initialised")

    topic = _parser.get_topic(body.topic_id)
    if not topic:
        raise HTTPException(
            status_code=404, detail=f"Topic '{body.topic_id}' not found"
        )

    doc_content = _parser.get_topic_content(body.topic_id)
    generator = QuestionGenerator(_llm_client)

    return await generator.generate(
        topic_id=body.topic_id,
        topic_title=topic.title,
        doc_content=doc_content,
        count=body.count,
        difficulty=body.difficulty,
        llm_config=body.llm_config,
        section_title=body.section_title,
        section_content=body.section_content,
    )


@router.post("/generate-v2", response_model=GenerateQuestionsV2Response)
async def generate_questions_v2(body: GenerateQuestionsRequest):
    """Generate grounded interview questions (strict schema v2)."""
    if _llm_client is None or _parser is None:
        raise HTTPException(status_code=503, detail="Service not initialised")
    if not get_settings().enable_v2_generation:
        raise HTTPException(status_code=404, detail="v2 generation disabled")

    topic = _parser.get_topic(body.topic_id)
    if not topic:
        raise HTTPException(
            status_code=404, detail=f"Topic '{body.topic_id}' not found"
        )

    doc_content = _parser.get_topic_content(body.topic_id)
    generator = QuestionGenerator(_llm_client)
    return await generator.generate_v2(
        topic_id=body.topic_id,
        topic_title=topic.title,
        doc_content=doc_content,
        count=body.count,
        difficulty=body.difficulty,
        llm_config=body.llm_config,
        section_title=body.section_title,
        section_content=body.section_content,
    )


@router.post("/quiz/generate", response_model=GenerateQuizResponse)
async def generate_quiz(body: GenerateQuizRequest):
    """Generate MCQ and/or True/False quiz questions across one or more topics."""
    if _llm_client is None or _parser is None:
        raise HTTPException(status_code=503, detail="Service not initialised")

    topics_content: list[dict] = []
    for tid in body.topic_ids:
        topic = _parser.get_topic(tid)
        if not topic:
            raise HTTPException(status_code=404, detail=f"Topic '{tid}' not found")
        topics_content.append({
            "id": tid,
            "title": topic.title,
            "content": _parser.get_topic_content(tid),
        })

    if not topics_content:
        raise HTTPException(status_code=400, detail="At least one topic_id required")

    generator = QuestionGenerator(_llm_client)
    return await generator.generate_quiz(
        topics_content=topics_content,
        count=body.count,
        question_types=[qt.value for qt in body.question_types],
        difficulty=body.difficulty,
        llm_config=body.llm_config,
    )


@router.post("/quiz/generate-v2", response_model=GenerateQuizV2Response)
async def generate_quiz_v2(body: GenerateQuizRequest):
    """Generate strictly validated quiz questions with grounding fields."""
    if _llm_client is None or _parser is None:
        raise HTTPException(status_code=503, detail="Service not initialised")
    if not get_settings().enable_v2_generation:
        raise HTTPException(status_code=404, detail="v2 generation disabled")

    topics_content: list[dict] = []
    for tid in body.topic_ids:
        topic = _parser.get_topic(tid)
        if not topic:
            raise HTTPException(status_code=404, detail=f"Topic '{tid}' not found")
        topics_content.append({
            "id": tid,
            "title": topic.title,
            "content": _parser.get_topic_content(tid),
        })

    if not topics_content:
        raise HTTPException(status_code=400, detail="At least one topic_id required")

    generator = QuestionGenerator(_llm_client)
    return await generator.generate_quiz_v2(
        topics_content=topics_content,
        count=body.count,
        question_types=[qt.value for qt in body.question_types],
        difficulty=body.difficulty,
        llm_config=body.llm_config,
    )
