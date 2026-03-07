"""Generate interview questions & answers via LLM with strict validation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections import Counter
from typing import Any, AsyncIterator, Callable, Optional

from app.schemas.models import (
    GenerateQuestionsResponse,
    GenerateQuestionsV2Response,
    GenerateQuizResponse,
    GenerateQuizV2Response,
    LLMConfigRequest,
    QuestionAnswer,
    QuestionAnswerV2,
    QuizChoice,
    QuizQuestion,
    QuizQuestionType,
    QuizQuestionV2,
)
from app.services.llm_client import LLMClient
from app.services.llm_policy import (
    APPROVAL_REQUIRED_CODE,
    APPROVAL_REQUIRED_MESSAGE,
    PERSONAL_CREDENTIAL_REQUIRED_CODE,
    PERSONAL_CREDENTIAL_REQUIRED_MESSAGE,
    STUDY_APP_NOT_ASSIGNED_CODE,
    STUDY_APP_NOT_ASSIGNED_MESSAGE,
    LLMServiceApprovalRequiredError,
    PersonalCredentialRequiredError,
    StudyAppLLMNotAssignedError,
    raise_if_policy_blocked_result,
)

logger = logging.getLogger(__name__)

_MAX_DOC_CONTEXT = 45000
_MAX_TOPIC_CONTEXT = 9000
_MAX_ATTEMPTS = 5
_VALID_DIFFICULTIES = {"easy", "medium", "hard"}
_VALID_LEVELS = {"junior", "mid", "senior"}


def _question_id(topic_id: str, question: str) -> str:
    digest = hashlib.sha1(f"{topic_id}:{question.strip().lower()}".encode("utf-8")).hexdigest()
    return f"{topic_id}:{digest[:14]}"


def _normalise_question(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _clamp_content(doc_content: str) -> str:
    return doc_content[:_MAX_DOC_CONTEXT]


def _normalise_level(level: Optional[str]) -> str:
    lv = (level or "mid").strip().lower()
    return lv if lv in _VALID_LEVELS else "mid"


def _build_prompt(
    topic_title: str,
    doc_content: str,
    count: int = 5,
    difficulty: Optional[str] = None,
    level: Optional[str] = None,
    section_title: Optional[str] = None,
    section_content: Optional[str] = None,
) -> str:
    target_level = _normalise_level(level)
    diff_clause = ""
    if difficulty:
        diff_clause = f' All questions should be "{difficulty}" difficulty.'

    if section_title and section_content:
        scope = f'the "{section_title}" section of "{topic_title}"'
        content = _clamp_content(section_content)
    else:
        scope = f'"{topic_title}"'
        content = _clamp_content(doc_content)

    return f"""You are an expert technical interviewer and educator.

Given documentation about {scope}, generate exactly {count} interview-style questions with detailed educational answers.{diff_clause}
Target candidate level: "{target_level}".

Return ONLY valid JSON in this shape:
[
  {{
    "question": "Question text",
    "answer": "3-8 sentence educational answer",
    "difficulty": "easy|medium|hard",
    "learning_objective": "what this question tests",
    "source_section": "exact section heading",
    "source_quote": "short direct quote from docs",
    "misconception_trap": "common mistake this question targets",
    "reasoning_summary": "1-2 sentence reasoning path to answer",
    "target_level": "junior|mid|senior"
  }}
]

Rules:
- Questions must be standalone and non-duplicative.
- Questions must cover conceptual + practical angles.
- Answers must be grounded in the provided documentation.
- Format answers as markdown, but keep structure adaptive:
  - default to clear prose paragraphs for normal explanations,
  - use bullets only when listing steps/checklists/categories,
  - use headings only when the answer naturally has sections,
  - use tables only for direct comparisons/category matrices.
- If you use a table, output valid GFM table syntax:
  - one row per line,
  - include a separator row (e.g. `| --- | --- |`).
- You may include fenced code blocks when code clarifies an implementation detail.
- You may include fenced Mermaid diagrams when architecture or flows are better shown visually.
- If you include fences, always use explicit language tags (for example: ```python, ```mermaid).
- Because output must be valid JSON, escape newlines, quotes, and backslashes correctly inside string values.
- source_quote must be factual text from provided docs.
- target_level must match "{target_level}" exactly.
- Complexity must match target_level:
  - junior: fundamentals, definitions, and straightforward tradeoffs.
  - mid: implementation details, constraints, and moderate tradeoffs.
  - senior: architecture, scaling, risk, and deep tradeoff decisions.
- Keep markdown compact and practical.

Documentation:
{content}"""


def _build_quiz_prompt(
    topics_content: list[dict],
    count: int = 10,
    question_types: list[str] | None = None,
    difficulty: Optional[str] = None,
    level: Optional[str] = None,
) -> str:
    target_level = _normalise_level(level)
    types = question_types or ["mcq", "true_false"]
    diff_clause = ""
    if difficulty:
        diff_clause = f' All questions should be "{difficulty}" difficulty.'

    if "mcq" in types and "true_false" in types:
        type_instructions = "Mix multiple-choice (exactly 4 options A/B/C/D) and true_false (exactly A=True, B=False)."
    elif "mcq" in types:
        type_instructions = "All questions must be MCQ with exactly 4 options A/B/C/D."
    else:
        type_instructions = "All questions must be true_false with exactly 2 options: A=True, B=False."

    docs_text = ""
    for tc in topics_content:
        docs_text += f"\n\n--- Topic: {tc['title']} (id: {tc['id']}) ---\n{tc['content'][:_MAX_TOPIC_CONTEXT]}"

    return f"""You are an expert technical quiz creator.

Create exactly {count} quiz questions from the documentation below.{diff_clause}
Target candidate level: "{target_level}".
{type_instructions}

Return ONLY valid JSON:
[
  {{
    "question": "Question text",
    "type": "mcq|true_false",
    "choices": [{{"label":"A","text":"..."}}, {{"label":"B","text":"..."}}, ...],
    "correct_answer": "A|B|C|D",
    "explanation": "2-4 sentence explanation",
    "difficulty": "easy|medium|hard",
    "topic_id": "one of provided topic ids",
    "source_quote": "short direct quote from docs",
    "reasoning_summary": "1-2 sentence why answer is correct",
    "target_level": "junior|mid|senior"
  }}
]

Rules:
- topic_id must be one of the provided ids.
- correct_answer must match one choice label.
- Avoid trick ambiguity; one clearly correct answer.
- Distribute questions across topics as evenly as possible.
- Format explanations as markdown, but keep structure adaptive:
  - default to clear prose paragraphs for normal explanations,
  - use bullets only when listing steps/checklists/categories,
  - use headings only when the explanation naturally has sections,
  - use tables only for direct comparisons/category matrices.
- If you use a table, output valid GFM table syntax:
  - one row per line,
  - include a separator row (e.g. `| --- | --- |`).
- You may include fenced code blocks when code clarifies an implementation detail.
- You may include fenced Mermaid diagrams when architecture or flows are better shown visually.
- If you include fences, always use explicit language tags (for example: ```yaml, ```mermaid).
- Because output must be valid JSON, escape newlines, quotes, and backslashes correctly inside string values.
- target_level must match "{target_level}" exactly.
- Complexity must match target_level:
  - junior: direct recall and simple application.
  - mid: applied reasoning with concrete constraints.
  - senior: architecture-level reasoning and tradeoff depth.
- Keep markdown compact and practical.

Documentation:
{docs_text[:_MAX_DOC_CONTEXT]}"""


def _build_retry_prompt(base_prompt: str, missing_count: int, issues: str, existing_questions: list[str]) -> str:
    existing_blob = "\n".join([f"- {q}" for q in existing_questions[:50]])
    return f"""Your previous output did not satisfy the schema or quality constraints.

Missing items needed: {missing_count}
Validation issues:
{issues}

Do not repeat any of these existing questions:
{existing_blob if existing_blob else "- (none)"}

Return ONLY a JSON array with exactly {missing_count} NEW valid items.

{base_prompt}
"""


def _parse_questions_json(raw: str) -> list[dict]:
    """Robustly extract JSON array from LLM response."""
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            payload = data.get("questions") or data.get("items")
            if isinstance(payload, list):
                return payload
    except json.JSONDecodeError:
        pass

    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                payload = data.get("questions") or data.get("items")
                if isinstance(payload, list):
                    return payload
        except json.JSONDecodeError:
            pass

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


def _validate_question_item(
    item: dict[str, Any],
    topic_id: str,
    difficulty: Optional[str],
    level: Optional[str],
) -> tuple[bool, str]:
    required = [
        "question",
        "answer",
        "learning_objective",
        "source_section",
        "source_quote",
        "misconception_trap",
        "reasoning_summary",
        "target_level",
    ]
    for key in required:
        if not isinstance(item.get(key), str) or not item.get(key).strip():
            return False, f"missing_or_empty_{key}"

    diff = str(item.get("difficulty", "medium")).lower()
    if diff not in _VALID_DIFFICULTIES:
        return False, "invalid_difficulty"
    if difficulty and diff != difficulty:
        return False, "difficulty_mismatch"
    if len(item["question"].strip()) < 12:
        return False, "question_too_short"
    if len(item["answer"].strip()) < 60:
        return False, "answer_too_short"
    target_level = str(item.get("target_level", "")).strip().lower()
    if target_level not in _VALID_LEVELS:
        return False, "invalid_target_level"
    if _normalise_level(level) != target_level:
        return False, "level_mismatch"
    item["topic_id"] = topic_id
    item["difficulty"] = diff
    return True, ""


def _validate_quiz_item(
    item: dict[str, Any],
    allowed_topics: set[str],
    allowed_types: set[str],
    difficulty: Optional[str],
    level: Optional[str],
) -> tuple[bool, str]:
    required = [
        "question",
        "type",
        "choices",
        "correct_answer",
        "explanation",
        "topic_id",
        "source_quote",
        "reasoning_summary",
        "target_level",
    ]
    for key in required:
        if key not in item:
            return False, f"missing_{key}"

    q_text = item.get("question")
    q_type = str(item.get("type", "")).lower()
    choices = item.get("choices")
    correct = str(item.get("correct_answer", "")).strip()
    explanation = item.get("explanation")
    topic_id = str(item.get("topic_id", "")).strip()
    source_quote = item.get("source_quote")
    reasoning_summary = item.get("reasoning_summary")
    target_level = str(item.get("target_level", "")).strip().lower()

    if not isinstance(q_text, str) or len(q_text.strip()) < 8:
        return False, "invalid_question"
    if q_type not in {"mcq", "true_false"}:
        return False, "invalid_type"
    if q_type not in allowed_types:
        return False, "disallowed_type"
    if topic_id not in allowed_topics:
        return False, "invalid_topic_id"
    if not isinstance(explanation, str) or len(explanation.strip()) < 15:
        return False, "invalid_explanation"
    if not isinstance(source_quote, str) or len(source_quote.strip()) < 8:
        return False, "invalid_source_quote"
    if not isinstance(reasoning_summary, str) or len(reasoning_summary.strip()) < 8:
        return False, "invalid_reasoning_summary"
    if target_level not in _VALID_LEVELS:
        return False, "invalid_target_level"
    if _normalise_level(level) != target_level:
        return False, "level_mismatch"

    if not isinstance(choices, list):
        return False, "invalid_choices_type"
    labels = []
    for c in choices:
        if not isinstance(c, dict):
            return False, "invalid_choice_item"
        label = str(c.get("label", "")).strip()
        text = c.get("text")
        if label not in {"A", "B", "C", "D"}:
            return False, "invalid_choice_label"
        if not isinstance(text, str) or not text.strip():
            return False, "invalid_choice_text"
        labels.append(label)

    if q_type == "mcq":
        if labels != ["A", "B", "C", "D"]:
            return False, "mcq_labels_must_be_abcd"
        if len(choices) != 4:
            return False, "mcq_must_have_4_choices"
        if correct not in {"A", "B", "C", "D"}:
            return False, "invalid_correct_answer_mcq"
    else:
        if len(choices) != 2:
            return False, "true_false_must_have_2_choices"
        if labels != ["A", "B"]:
            return False, "true_false_labels_must_be_ab"
        true_false_texts = [str(choices[0].get("text", "")).strip().lower(), str(choices[1].get("text", "")).strip().lower()]
        if true_false_texts != ["true", "false"]:
            return False, "true_false_text_must_be_true_false"
        if correct not in {"A", "B"}:
            return False, "invalid_correct_answer_true_false"

    diff = str(item.get("difficulty", "medium")).lower()
    if diff not in _VALID_DIFFICULTIES:
        return False, "invalid_difficulty"
    if difficulty and diff != difficulty:
        return False, "difficulty_mismatch"
    item["difficulty"] = diff
    item["type"] = q_type
    return True, ""


async def _collect_with_retries(
    *,
    llm: LLMClient,
    base_prompt: str,
    llm_config: Optional[LLMConfigRequest],
    user_identity: Optional[dict] = None,
    target_count: int,
    validator: Callable[[dict[str, Any]], tuple[bool, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dedup: set[str] = set()
    valid_items: list[dict[str, Any]] = []
    malformed_items = 0
    retries_used = 0
    issue_counter: Counter[str] = Counter()
    metadata: dict[str, Any] = {}
    prompt = base_prompt
    last_error = ""

    for attempt in range(_MAX_ATTEMPTS):
        result = await llm.completion(
            prompt,
            llm_config,
            user_identity=user_identity,
        )
        raise_if_policy_blocked_result(result)
        metadata = result.get("metadata", {})
        if not result.get("success"):
            last_error = result.get("error", "unknown_error")
            retries_used += 1
            prompt = _build_retry_prompt(
                base_prompt=base_prompt,
                missing_count=max(1, target_count - len(valid_items)),
                issues=f"transport_or_provider_error: {last_error}",
                existing_questions=[x.get("question", "") for x in valid_items],
            )
            continue

        parsed = _parse_questions_json(result.get("analysis", ""))
        if not parsed:
            retries_used += 1
            issue_counter["json_parse_failed"] += 1
            prompt = _build_retry_prompt(
                base_prompt=base_prompt,
                missing_count=max(1, target_count - len(valid_items)),
                issues="json_parse_failed",
                existing_questions=[x.get("question", "") for x in valid_items],
            )
            continue

        for item in parsed:
            if not isinstance(item, dict):
                malformed_items += 1
                issue_counter["non_dict_item"] += 1
                continue

            is_valid, issue = validator(item)
            if not is_valid:
                malformed_items += 1
                issue_counter[issue] += 1
                continue

            q_norm = _normalise_question(str(item.get("question", "")))
            if not q_norm or q_norm in dedup:
                malformed_items += 1
                issue_counter["duplicate_question"] += 1
                continue
            dedup.add(q_norm)
            valid_items.append(item)
            if len(valid_items) >= target_count:
                break

        if len(valid_items) >= target_count:
            break

        retries_used += 1
        top_issues = ", ".join([f"{k}:{v}" for k, v in issue_counter.most_common(5)]) or "insufficient_valid_items"
        prompt = _build_retry_prompt(
            base_prompt=base_prompt,
            missing_count=target_count - len(valid_items),
            issues=top_issues,
            existing_questions=[x.get("question", "") for x in valid_items],
        )

    if len(valid_items) < target_count:
        logger.warning(
            "Generation produced %d/%d valid items after retries=%d issues=%s last_error=%s",
            len(valid_items),
            target_count,
            retries_used,
            dict(issue_counter),
            last_error,
        )

    stats = {
        "metadata": metadata,
        "retries_used": retries_used,
        "malformed_items_dropped": malformed_items,
    }
    return valid_items[:target_count], stats


class QuestionGenerator:
    """Uses LiteLLM to generate interview Q&A and quizzes."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    @staticmethod
    def _to_question_answer_v2(topic_id: str, item: dict[str, Any]) -> QuestionAnswerV2:
        return QuestionAnswerV2(
            question_id=_question_id(topic_id, item["question"]),
            topic_id=topic_id,
            question=item["question"].strip(),
            answer=item["answer"].strip(),
            difficulty=item["difficulty"],
            learning_objective=item["learning_objective"].strip(),
            source_section=item["source_section"].strip(),
            source_quote=item["source_quote"].strip(),
            misconception_trap=item["misconception_trap"].strip(),
            reasoning_summary=item["reasoning_summary"].strip(),
        )

    @staticmethod
    def _policy_error_payload(exc: Exception) -> tuple[str, str]:
        if isinstance(exc, LLMServiceApprovalRequiredError):
            return APPROVAL_REQUIRED_CODE, APPROVAL_REQUIRED_MESSAGE
        if isinstance(exc, StudyAppLLMNotAssignedError):
            return STUDY_APP_NOT_ASSIGNED_CODE, STUDY_APP_NOT_ASSIGNED_MESSAGE
        if isinstance(exc, PersonalCredentialRequiredError):
            return PERSONAL_CREDENTIAL_REQUIRED_CODE, PERSONAL_CREDENTIAL_REQUIRED_MESSAGE
        return "generation_failed", str(exc).strip() or "Question generation failed"

    async def generate_v2_stream(
        self,
        topic_id: str,
        topic_title: str,
        doc_content: str,
        count: int = 5,
        difficulty: Optional[str] = None,
        level: Optional[str] = None,
        llm_config: Optional[LLMConfigRequest] = None,
        user_identity: Optional[dict] = None,
        section_title: Optional[str] = None,
        section_content: Optional[str] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        target_count = max(0, int(count))
        prompt_base = _build_prompt(
            topic_title=topic_title,
            doc_content=doc_content,
            count=1,
            difficulty=difficulty,
            level=level,
            section_title=section_title,
            section_content=section_content,
        )

        dedup_norm: set[str] = set()
        existing_questions: list[str] = []
        dedup_lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(max(1, min(3, target_count or 1)))

        async def _generate_one() -> dict[str, Any]:
            retries_used = 0
            malformed_dropped = 0
            metadata: dict[str, Any] = {}
            prompt = prompt_base
            issue_counter: Counter[str] = Counter()
            last_error = ""

            try:
                for _ in range(_MAX_ATTEMPTS):
                    async with semaphore:
                        result = await self.llm.completion(
                            prompt,
                            llm_config,
                            user_identity=user_identity,
                        )
                    raise_if_policy_blocked_result(result)
                    metadata = result.get("metadata", {})

                    if not result.get("success"):
                        retries_used += 1
                        last_error = str(result.get("error", "unknown_error"))
                        issue_counter["transport_or_provider_error"] += 1
                        async with dedup_lock:
                            existing_snapshot = list(existing_questions)
                        prompt = _build_retry_prompt(
                            base_prompt=prompt_base,
                            missing_count=1,
                            issues=f"transport_or_provider_error: {last_error}",
                            existing_questions=existing_snapshot,
                        )
                        continue

                    parsed = _parse_questions_json(result.get("analysis", ""))
                    if not parsed:
                        retries_used += 1
                        malformed_dropped += 1
                        issue_counter["json_parse_failed"] += 1
                        async with dedup_lock:
                            existing_snapshot = list(existing_questions)
                        prompt = _build_retry_prompt(
                            base_prompt=prompt_base,
                            missing_count=1,
                            issues="json_parse_failed",
                            existing_questions=existing_snapshot,
                        )
                        continue

                    selected_item: dict[str, Any] | None = None
                    for item in parsed:
                        if not isinstance(item, dict):
                            malformed_dropped += 1
                            issue_counter["non_dict_item"] += 1
                            continue

                        valid, issue = _validate_question_item(item, topic_id, difficulty, level)
                        if not valid:
                            malformed_dropped += 1
                            issue_counter[issue] += 1
                            continue

                        q_text = str(item.get("question", "")).strip()
                        q_norm = _normalise_question(q_text)
                        if not q_norm:
                            malformed_dropped += 1
                            issue_counter["invalid_question"] += 1
                            continue

                        async with dedup_lock:
                            if q_norm in dedup_norm:
                                duplicate = True
                            else:
                                dedup_norm.add(q_norm)
                                existing_questions.append(q_text)
                                duplicate = False
                        if duplicate:
                            malformed_dropped += 1
                            issue_counter["duplicate_question"] += 1
                            continue

                        selected_item = item
                        break

                    if selected_item is not None:
                        qa = self._to_question_answer_v2(topic_id, selected_item)
                        return {
                            "question": qa.model_dump(),
                            "metadata": metadata,
                            "retries_used": retries_used,
                            "malformed_items_dropped": malformed_dropped,
                        }

                    retries_used += 1
                    top_issues = (
                        ", ".join(f"{k}:{v}" for k, v in issue_counter.most_common(5))
                        or "insufficient_valid_items"
                    )
                    async with dedup_lock:
                        existing_snapshot = list(existing_questions)
                    prompt = _build_retry_prompt(
                        base_prompt=prompt_base,
                        missing_count=1,
                        issues=top_issues,
                        existing_questions=existing_snapshot,
                    )

                if issue_counter:
                    logger.warning(
                        "Streaming slot failed to produce valid question after retries=%d issues=%s last_error=%s",
                        retries_used,
                        dict(issue_counter),
                        last_error,
                    )
                return {
                    "question": None,
                    "metadata": metadata,
                    "retries_used": retries_used,
                    "malformed_items_dropped": malformed_dropped,
                }
            except (
                LLMServiceApprovalRequiredError,
                StudyAppLLMNotAssignedError,
                PersonalCredentialRequiredError,
            ) as exc:
                code, message = self._policy_error_payload(exc)
                return {
                    "error": {"code": code, "message": message},
                    "metadata": metadata,
                    "retries_used": retries_used,
                    "malformed_items_dropped": malformed_dropped,
                }
            except Exception as exc:
                logger.exception("Streaming question generation failed: %s", exc)
                return {
                    "error": {
                        "code": "generation_failed",
                        "message": str(exc).strip() or "Question generation failed",
                    },
                    "metadata": metadata,
                    "retries_used": retries_used,
                    "malformed_items_dropped": malformed_dropped,
                }

        yield {
            "type": "start",
            "topic_id": topic_id,
            "topic_title": topic_title,
            "target_count": target_count,
        }

        if target_count == 0:
            yield {
                "type": "done",
                "topic_id": topic_id,
                "topic_title": topic_title,
                "generated_count": 0,
                "provider_used": "",
                "model_used": "",
                "retries_used": 0,
                "malformed_items_dropped": 0,
            }
            return

        provider_used = ""
        model_used = ""
        retries_used = 0
        malformed_items_dropped = 0
        generated_count = 0

        tasks = [asyncio.create_task(_generate_one()) for _ in range(target_count)]
        for pending in asyncio.as_completed(tasks):
            result = await pending
            retries_used += int(result.get("retries_used", 0))
            malformed_items_dropped += int(result.get("malformed_items_dropped", 0))
            metadata = result.get("metadata", {}) or {}
            if metadata.get("provider"):
                provider_used = str(metadata.get("provider"))
            if metadata.get("model"):
                model_used = str(metadata.get("model"))

            error = result.get("error")
            if error:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                yield {
                    "type": "error",
                    "code": str(error.get("code", "generation_failed")),
                    "message": str(error.get("message", "Question generation failed")),
                }
                return

            question = result.get("question")
            if not isinstance(question, dict):
                continue

            generated_count += 1
            yield {
                "type": "question",
                "question": question,
            }

        yield {
            "type": "done",
            "topic_id": topic_id,
            "topic_title": topic_title,
            "generated_count": generated_count,
            "provider_used": provider_used,
            "model_used": model_used,
            "retries_used": retries_used,
            "malformed_items_dropped": malformed_items_dropped,
        }

    async def generate(
        self,
        topic_id: str,
        topic_title: str,
        doc_content: str,
        count: int = 5,
        difficulty: Optional[str] = None,
        level: Optional[str] = None,
        llm_config: Optional[LLMConfigRequest] = None,
        user_identity: Optional[dict] = None,
        section_title: Optional[str] = None,
        section_content: Optional[str] = None,
    ) -> GenerateQuestionsResponse:
        v2 = await self.generate_v2(
            topic_id=topic_id,
            topic_title=topic_title,
            doc_content=doc_content,
            count=count,
            difficulty=difficulty,
            level=level,
            llm_config=llm_config,
            user_identity=user_identity,
            section_title=section_title,
            section_content=section_content,
        )
        questions = [
            QuestionAnswer(question=q.question, answer=q.answer, difficulty=q.difficulty)
            for q in v2.questions
        ]
        if not questions:
            questions = [
                QuestionAnswer(
                    question="Unable to generate questions at this time.",
                    answer="The LLM did not return valid structured output. Please try again.",
                    difficulty="easy",
                )
            ]
        return GenerateQuestionsResponse(
            topic_id=topic_id,
            topic_title=topic_title,
            questions=questions,
            provider_used=v2.provider_used,
            model_used=v2.model_used,
        )

    async def generate_v2(
        self,
        topic_id: str,
        topic_title: str,
        doc_content: str,
        count: int = 5,
        difficulty: Optional[str] = None,
        level: Optional[str] = None,
        llm_config: Optional[LLMConfigRequest] = None,
        user_identity: Optional[dict] = None,
        section_title: Optional[str] = None,
        section_content: Optional[str] = None,
    ) -> GenerateQuestionsV2Response:
        prompt = _build_prompt(
            topic_title=topic_title,
            doc_content=doc_content,
            count=count,
            difficulty=difficulty,
            level=level,
            section_title=section_title,
            section_content=section_content,
        )

        validator = lambda item: _validate_question_item(item, topic_id, difficulty, level)
        raw_items, stats = await _collect_with_retries(
            llm=self.llm,
            base_prompt=prompt,
            llm_config=llm_config,
            user_identity=user_identity,
            target_count=count,
            validator=validator,
        )

        questions = [self._to_question_answer_v2(topic_id, item) for item in raw_items]

        metadata = stats.get("metadata", {})
        return GenerateQuestionsV2Response(
            topic_id=topic_id,
            topic_title=topic_title,
            questions=questions,
            provider_used=metadata.get("provider", ""),
            model_used=metadata.get("model", ""),
            retries_used=stats.get("retries_used", 0),
            malformed_items_dropped=stats.get("malformed_items_dropped", 0),
        )

    async def generate_quiz(
        self,
        topics_content: list[dict],
        count: int = 10,
        question_types: list[str] | None = None,
        difficulty: Optional[str] = None,
        level: Optional[str] = None,
        llm_config: Optional[LLMConfigRequest] = None,
        user_identity: Optional[dict] = None,
    ) -> GenerateQuizResponse:
        v2 = await self.generate_quiz_v2(
            topics_content=topics_content,
            count=count,
            question_types=question_types,
            difficulty=difficulty,
            level=level,
            llm_config=llm_config,
            user_identity=user_identity,
        )
        legacy_questions = [
            QuizQuestion(
                question=q.question,
                type=q.type,
                choices=q.choices,
                correct_answer=q.correct_answer,
                explanation=q.explanation,
                difficulty=q.difficulty,
                topic_id=q.topic_id,
            )
            for q in v2.questions
        ]
        return GenerateQuizResponse(
            questions=legacy_questions,
            topics_used=v2.topics_used,
            provider_used=v2.provider_used,
            model_used=v2.model_used,
        )

    async def generate_quiz_v2(
        self,
        topics_content: list[dict],
        count: int = 10,
        question_types: list[str] | None = None,
        difficulty: Optional[str] = None,
        level: Optional[str] = None,
        llm_config: Optional[LLMConfigRequest] = None,
        user_identity: Optional[dict] = None,
    ) -> GenerateQuizV2Response:
        prompt = _build_quiz_prompt(topics_content, count, question_types, difficulty, level)
        allowed_topics = {str(tc["id"]) for tc in topics_content}
        allowed_types = set(question_types or ["mcq", "true_false"])
        validator = lambda item: _validate_quiz_item(
            item,
            allowed_topics,
            allowed_types,
            difficulty,
            level,
        )

        raw_items, stats = await _collect_with_retries(
            llm=self.llm,
            base_prompt=prompt,
            llm_config=llm_config,
            user_identity=user_identity,
            target_count=count,
            validator=validator,
        )

        questions = []
        for item in raw_items:
            topic_id = str(item.get("topic_id", "")).strip()
            qid = _question_id(topic_id or "quiz", item["question"])
            choices = [QuizChoice(label=c["label"], text=c["text"]) for c in item["choices"]]
            questions.append(
                QuizQuestionV2(
                    question_id=qid,
                    question=item["question"].strip(),
                    type=QuizQuestionType(item["type"]),
                    choices=choices,
                    correct_answer=str(item["correct_answer"]).strip(),
                    explanation=item["explanation"].strip(),
                    difficulty=item["difficulty"],
                    topic_id=topic_id,
                    source_quote=item["source_quote"].strip(),
                    reasoning_summary=item["reasoning_summary"].strip(),
                )
            )

        metadata = stats.get("metadata", {})
        topic_ids = sorted({tc["id"] for tc in topics_content})
        return GenerateQuizV2Response(
            questions=questions,
            topics_used=topic_ids,
            provider_used=metadata.get("provider", ""),
            model_used=metadata.get("model", ""),
            retries_used=stats.get("retries_used", 0),
            malformed_items_dropped=stats.get("malformed_items_dropped", 0),
        )
