"""Generate interview questions & answers via LLM."""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from app.schemas.models import (
    GenerateQuestionsResponse,
    GenerateQuizResponse,
    LLMConfigRequest,
    QuestionAnswer,
    QuizChoice,
    QuizQuestion,
    QuizQuestionType,
)
from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


def _build_prompt(
    topic_title: str,
    doc_content: str,
    count: int = 5,
    difficulty: Optional[str] = None,
    section_title: Optional[str] = None,
    section_content: Optional[str] = None,
) -> str:
    diff_clause = ""
    if difficulty:
        diff_clause = f' All questions should be "{difficulty}" difficulty.'

    if section_title and section_content:
        scope = f'the "{section_title}" section of "{topic_title}"'
        content = section_content[:30000]
    else:
        scope = f'"{topic_title}"'
        content = doc_content[:30000]

    return f"""You are an expert technical interviewer and educator.

Given the following documentation about {scope}, generate exactly {count} interview-style questions with detailed, educational answers.{diff_clause}

IMPORTANT: Return ONLY a valid JSON array with this exact structure (no markdown, no code fences, no extra text):
[
  {{
    "question": "The interview question here?",
    "answer": "A detailed, educational answer that thoroughly explains the concept. Include examples, reasoning, and practical implications where relevant.",
    "difficulty": "easy|medium|hard"
  }}
]

Rules:
- Questions should test deep understanding, not just memorisation
- Answers should be 3-8 sentences, thorough and educational
- Vary question types: conceptual, practical, scenario-based, comparison
- Questions MUST be specifically about the provided documentation content
- Each question must stand alone (no "as mentioned above")

Documentation:
{content}"""


def _build_quiz_prompt(
    topics_content: list[dict],
    count: int = 10,
    question_types: list[str] | None = None,
    difficulty: Optional[str] = None,
) -> str:
    types = question_types or ["mcq", "true_false"]
    diff_clause = ""
    if difficulty:
        diff_clause = f' All questions should be "{difficulty}" difficulty.'

    type_instructions = ""
    if "mcq" in types and "true_false" in types:
        type_instructions = "Mix of multiple-choice (4 options labeled A,B,C,D) and true/false questions."
    elif "mcq" in types:
        type_instructions = "All questions must be multiple-choice with exactly 4 options labeled A,B,C,D."
    else:
        type_instructions = "All questions must be true/false format with exactly 2 options: A (True) and B (False)."

    docs_text = ""
    for tc in topics_content:
        docs_text += f"\n\n--- Topic: {tc['title']} (id: {tc['id']}) ---\n{tc['content'][:8000]}"

    return f"""You are an expert quiz creator for technical education.

Create exactly {count} quiz questions based on the following documentation.{diff_clause}
{type_instructions}

IMPORTANT: Return ONLY a valid JSON array with this exact structure (no markdown, no code fences, no extra text):
[
  {{
    "question": "The question text?",
    "type": "mcq",
    "choices": [
      {{"label": "A", "text": "First option"}},
      {{"label": "B", "text": "Second option"}},
      {{"label": "C", "text": "Third option"}},
      {{"label": "D", "text": "Fourth option"}}
    ],
    "correct_answer": "A",
    "explanation": "Detailed explanation of why this is correct.",
    "difficulty": "easy|medium|hard",
    "topic_id": "the topic id this question is about"
  }}
]

For true_false type, use only two choices:
  "choices": [{{"label": "A", "text": "True"}}, {{"label": "B", "text": "False"}}]

Rules:
- correct_answer must be one of the choice labels (A, B, C, or D)
- Explanations should be 2-4 sentences
- Questions should test real understanding, not trivial details
- Distribute questions across the provided topics
- Each question must stand alone

Documentation:
{docs_text[:30000]}"""


def _parse_questions_json(raw: str) -> list[dict]:
    """Robustly extract JSON array from LLM response."""
    # Try direct parse first
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    # Try finding the array boundaries
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(raw[start : end + 1])
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    return []


class QuestionGenerator:
    """Uses LiteLLM to generate interview Q&A."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def generate(
        self,
        topic_id: str,
        topic_title: str,
        doc_content: str,
        count: int = 5,
        difficulty: Optional[str] = None,
        llm_config: Optional[LLMConfigRequest] = None,
        section_title: Optional[str] = None,
        section_content: Optional[str] = None,
    ) -> GenerateQuestionsResponse:
        prompt = _build_prompt(
            topic_title, doc_content, count, difficulty,
            section_title=section_title, section_content=section_content,
        )

        result = await self.llm.completion(prompt, llm_config)

        questions: list[QuestionAnswer] = []

        if result["success"] and result["analysis"]:
            parsed = _parse_questions_json(result["analysis"])
            if not parsed:
                # Retry with a stricter prompt
                retry_prompt = (
                    "Your previous response was not valid JSON. "
                    "Please return ONLY a valid JSON array. No extra text.\n\n"
                    + prompt
                )
                result = await self.llm.completion(retry_prompt, llm_config)
                if result["success"] and result["analysis"]:
                    parsed = _parse_questions_json(result["analysis"])

            for item in parsed:
                if isinstance(item, dict) and "question" in item and "answer" in item:
                    questions.append(
                        QuestionAnswer(
                            question=item["question"],
                            answer=item["answer"],
                            difficulty=item.get("difficulty", "medium"),
                        )
                    )

        if not questions and result.get("error"):
            logger.error(f"Failed to generate questions: {result['error']}")
            # Return a fallback question indicating the error
            questions.append(
                QuestionAnswer(
                    question="Unable to generate questions at this time.",
                    answer=f"The LLM service returned an error: {result['error']}. "
                    "Please check your LLM configuration and ensure the service is running.",
                    difficulty="easy",
                )
            )

        metadata = result.get("metadata", {})
        return GenerateQuestionsResponse(
            topic_id=topic_id,
            topic_title=topic_title,
            questions=questions,
            provider_used=metadata.get("provider", ""),
            model_used=metadata.get("model", ""),
        )

    async def generate_quiz(
        self,
        topics_content: list[dict],
        count: int = 10,
        question_types: list[str] | None = None,
        difficulty: Optional[str] = None,
        llm_config: Optional[LLMConfigRequest] = None,
    ) -> GenerateQuizResponse:
        """Generate quiz questions (MCQ and/or True/False) across topics."""
        prompt = _build_quiz_prompt(topics_content, count, question_types, difficulty)

        result = await self.llm.completion(prompt, llm_config)

        questions: list[QuizQuestion] = []

        if result["success"] and result["analysis"]:
            parsed = _parse_questions_json(result["analysis"])
            if not parsed:
                retry_prompt = (
                    "Your previous response was not valid JSON. "
                    "Please return ONLY a valid JSON array. No extra text.\n\n"
                    + prompt
                )
                result = await self.llm.completion(retry_prompt, llm_config)
                if result["success"] and result["analysis"]:
                    parsed = _parse_questions_json(result["analysis"])

            for item in parsed:
                if not isinstance(item, dict) or "question" not in item:
                    continue
                q_type = item.get("type", "mcq")
                choices_raw = item.get("choices", [])
                choices = []
                for c in choices_raw:
                    if isinstance(c, dict) and "label" in c and "text" in c:
                        choices.append(QuizChoice(label=c["label"], text=c["text"]))
                if not choices:
                    continue
                questions.append(
                    QuizQuestion(
                        question=item["question"],
                        type=QuizQuestionType(q_type) if q_type in ("mcq", "true_false") else QuizQuestionType.MCQ,
                        choices=choices,
                        correct_answer=item.get("correct_answer", "A"),
                        explanation=item.get("explanation", ""),
                        difficulty=item.get("difficulty", "medium"),
                        topic_id=item.get("topic_id", ""),
                    )
                )

        if not questions and result.get("error"):
            logger.error(f"Failed to generate quiz: {result['error']}")

        metadata = result.get("metadata", {})
        topic_ids = list({tc["id"] for tc in topics_content})
        return GenerateQuizResponse(
            questions=questions,
            topics_used=topic_ids,
            provider_used=metadata.get("provider", ""),
            model_used=metadata.get("model", ""),
        )
