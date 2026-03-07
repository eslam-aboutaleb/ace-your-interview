"""Generate interview questions & answers via LLM with strict validation."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from difflib import SequenceMatcher
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
from app.services.markdown_formatter import format_markdown_readable
from app.services.mcp_gateway import MCPGateway
from app.services.prompt_blocks import optional_context_block, render_contract
from app.services.topic_catalog import (
    PROBLEM_SOLVING_DEFAULT_LANGUAGE,
    is_problem_solving_topic,
)

logger = logging.getLogger(__name__)

_MAX_DOC_CONTEXT = 45000
_MAX_TOPIC_CONTEXT = 9000
_BASE_MAX_ATTEMPTS = 5
_MAX_TOTAL_ATTEMPTS = 18
_MAX_RECOVERY_ATTEMPTS = 24
_VALID_DIFFICULTIES = {"easy", "medium", "hard"}
_VALID_LEVELS = {"junior", "mid", "senior"}
_QUESTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "why",
    "with",
    "would",
    "you",
}


def _question_id(topic_id: str, question: str) -> str:
    digest = hashlib.sha1(f"{topic_id}:{question.strip().lower()}".encode("utf-8")).hexdigest()
    return f"{topic_id}:{digest[:14]}"


def _normalise_question(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _question_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", _normalise_question(text))
    normalized: set[str] = set()
    for word in words:
        if len(word) <= 2 or word in _QUESTION_STOPWORDS:
            continue
        if word.endswith("ies") and len(word) > 4:
            word = f"{word[:-3]}y"
        elif word.endswith("s") and len(word) > 4:
            word = word[:-1]
        normalized.add(word)
    return normalized


def _is_near_duplicate_question(candidate: str, seen_questions: list[str]) -> bool:
    cand_norm = _normalise_question(candidate)
    if not cand_norm:
        return True
    cand_tokens = _question_tokens(candidate)
    for seen in seen_questions:
        seen_norm = _normalise_question(seen)
        if not seen_norm:
            continue
        if cand_norm == seen_norm:
            return True
        if (cand_norm in seen_norm or seen_norm in cand_norm) and min(len(cand_norm), len(seen_norm)) >= 24:
            return True
        if SequenceMatcher(None, cand_norm, seen_norm).ratio() >= 0.87:
            return True
        seen_tokens = _question_tokens(seen)
        if cand_tokens and seen_tokens:
            overlap = len(cand_tokens & seen_tokens) / max(1, len(cand_tokens | seen_tokens))
            if overlap >= 0.65:
                return True
    return False


def _clamp_content(doc_content: str) -> str:
    return doc_content[:_MAX_DOC_CONTEXT]


def _normalise_level(level: Optional[str]) -> str:
    lv = (level or "mid").strip().lower()
    return lv if lv in _VALID_LEVELS else "mid"


def _attempt_budget(target_count: int) -> int:
    target = max(1, int(target_count or 1))
    return min(_MAX_TOTAL_ATTEMPTS, max(_BASE_MAX_ATTEMPTS, target + 3))


def _recovery_budget(missing_count: int) -> int:
    missing = max(1, int(missing_count or 1))
    return min(_MAX_RECOVERY_ATTEMPTS, max(3, missing * 3))


def _normalise_existing_questions(existing_questions: list[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in existing_questions or []:
        text = str(raw or "").strip()
        if not text:
            continue
        norm = _normalise_question(text)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(text)
        if len(out) >= 80:
            break
    return out


def _build_prompt(
    topic_id: str,
    topic_title: str,
    doc_content: str,
    count: int = 5,
    difficulty: Optional[str] = None,
    level: Optional[str] = None,
    section_title: Optional[str] = None,
    section_content: Optional[str] = None,
    response_detail: str = "concise",
    preferred_language: str = "",
    requires_programming: bool = False,
    requested_total_count: Optional[int] = None,
    existing_questions: Optional[list[str]] = None,
    mcp_context: str = "",
) -> str:
    problem_solving_mode = is_problem_solving_topic(topic_id)
    target_level = _normalise_level(level)
    requested_total = max(1, int(requested_total_count or count or 1))
    existing_seed = _normalise_existing_questions(existing_questions)
    diff_clause = ""
    if difficulty:
        diff_clause = f' All questions should be "{difficulty}" difficulty.'

    if section_title and section_content:
        scope = f'the "{section_title}" section of "{topic_title}"'
        content = _clamp_content(section_content)
    else:
        scope = f'"{topic_title}"'
        content = _clamp_content(doc_content)

    detail_clause = (
        "Be concise and high-signal."
        if response_detail != "very_detailed"
        else "Be very detailed with layered explanation depth and concrete examples."
    )
    selected_language = (preferred_language or "").strip().lower()
    code_clause = ""
    if problem_solving_mode:
        code_language = selected_language or PROBLEM_SOLVING_DEFAULT_LANGUAGE
        code_clause = (
            f'When code materially helps, provide one fenced "{code_language}" example with inline comments '
            "that explain each key step."
        )
    elif requires_programming and selected_language:
        code_clause = (
            f'Include one practical fenced code example in "{selected_language}" when code clarifies the answer.'
        )
    elif not requires_programming:
        code_clause = "Avoid code blocks unless code is explicitly required by the question."
    mcp_block = optional_context_block(
        "External context (optional, use only if relevant and factual)",
        mcp_context,
        2500,
    )

    role_clause = (
        "You are an expert algorithm interview coach and problem-solving educator."
        if problem_solving_mode
        else "You are an expert technical interviewer and educator."
    )
    problem_scope_rules = (
        [
            "Questions must be true problem-solving prompts (algorithmic/coding style), not generic theory prompts.",
            "Answers should explain problem understanding and constraints.",
            "Answers should explain solution strategy and why it works.",
            "Answers should explain complexity analysis and tradeoffs.",
            "Code is optional; if included, keep it practical and commented.",
        ]
        if problem_solving_mode
        else ["Questions must cover conceptual + practical angles."]
    )
    uniqueness_block = ""
    if existing_seed:
        existing_blob = "\n".join([f"- {q}" for q in existing_seed[:60]])
        uniqueness_block = (
            "\nAlready generated questions for this checkpoint. Do NOT repeat or rephrase these:\n"
            f"{existing_blob}\n"
        )
    prompt_contract = render_contract(
        schema_label="Return ONLY valid JSON in this shape",
        schema_block=f"""[
  {{
    "question": "Question text",
    "answer": "3-8 sentence educational answer",
    "difficulty": "easy|medium|hard"
  }}
]""",
        rules=[
            f"Return exactly {count} items; never return fewer.",
            "Questions must be standalone and non-duplicative.",
            "Questions must be NEW relative to already generated checkpoint questions listed above.",
            *problem_scope_rules,
            "Answers must be grounded in the provided documentation.",
            detail_clause,
            "Format answers as markdown, but keep structure adaptive.",
            "long answers must be split into short readable paragraphs with blank lines.",
            "use bullets only when listing steps/checklists/categories.",
            "use headings only when the answer naturally has sections.",
            "use tables only for direct comparisons/category matrices.",
            "If you use a table, output valid GFM table syntax.",
            "one row per line.",
            "include a separator row (e.g. `| --- | --- |`).",
            "You may include fenced code blocks when code clarifies an implementation detail.",
            code_clause if code_clause else "Use code examples only when they materially improve clarity.",
            "You may include fenced Mermaid diagrams when architecture or flows are better shown visually.",
            "If you include fences, always use explicit language tags (for example: ```python, ```mermaid).",
            "Because output must be valid JSON, escape newlines, quotes, and backslashes correctly inside string values.",
            "source_quote must be factual text from provided docs.",
            f'target_level must match "{target_level}" exactly.',
            "Do not wrap JSON with prose; return raw JSON only.",
            "Complexity must match target_level.",
            "junior: fundamentals, definitions, and straightforward tradeoffs.",
            "mid: implementation details, constraints, and moderate tradeoffs.",
            "senior: architecture, scaling, risk, and deep tradeoff decisions.",
            "Keep markdown compact and practical.",
        ],
    )
    return f"""{role_clause}

Given documentation about {scope}, generate exactly {count} interview-style questions with detailed educational answers.{diff_clause}
Target candidate level: "{target_level}".
User requested total questions for this checkpoint: {requested_total}.
This call is generating {count} new questions to fill remaining slots.
{uniqueness_block}

{prompt_contract}

Optional fields (recommended when available): learning_objective, source_section, source_quote,
misconception_trap, reasoning_summary, target_level.

Documentation:
{content}{mcp_block}"""


def _build_quiz_prompt(
    topics_content: list[dict],
    count: int = 10,
    question_types: list[str] | None = None,
    difficulty: Optional[str] = None,
    level: Optional[str] = None,
    response_detail: str = "concise",
    preferred_language: str = "",
    mcp_context: str = "",
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

    detail_clause = (
        "Keep explanations concise and high-signal."
        if response_detail != "very_detailed"
        else "Provide very detailed explanations with practical depth."
    )
    docs_text = ""
    for tc in topics_content:
        topic_requires_programming = bool(tc.get("requires_programming"))
        topic_language = str(tc.get("preferred_language", "")).strip().lower()
        topic_detail = str(tc.get("response_detail", response_detail)).strip().lower()
        topic_notes = [f"detail={topic_detail or response_detail}"]
        if topic_requires_programming and topic_language:
            topic_notes.append(f"preferred_language={topic_language}")
        docs_text += (
            f"\n\n--- Topic: {tc['title']} (id: {tc['id']}) [{', '.join(topic_notes)}] ---\n"
            f"{tc['content'][:_MAX_TOPIC_CONTEXT]}"
        )
    mcp_block = optional_context_block(
        "External context (optional, use only if relevant and factual)",
        mcp_context,
        2500,
    )
    prompt_contract = render_contract(
        schema_block="""[
  {
    "question": "Question text",
    "type": "mcq|true_false",
    "choices": [{"label":"A","text":"..."}, {"label":"B","text":"..."}, ...],
    "correct_answer": "A|B|C|D",
    "explanation": "2-4 sentence explanation",
    "difficulty": "easy|medium|hard",
    "topic_id": "one of provided topic ids",
    "source_quote": "short direct quote from docs",
    "reasoning_summary": "1-2 sentence why answer is correct",
    "target_level": "junior|mid|senior"
  }
]""",
        rules=[
            f"Return exactly {count} items; never return fewer.",
            "topic_id must be one of the provided ids.",
            "correct_answer must match one choice label.",
            "Avoid trick ambiguity; one clearly correct answer.",
            "Distribute questions across topics as evenly as possible.",
            detail_clause,
            "Format explanations as markdown, but keep structure adaptive.",
            "long explanations must be split into short readable paragraphs with blank lines.",
            "use bullets only when listing steps/checklists/categories.",
            "use headings only when the explanation naturally has sections.",
            "use tables only for direct comparisons/category matrices.",
            "If you use a table, output valid GFM table syntax.",
            "one row per line.",
            "include a separator row (e.g. `| --- | --- |`).",
            "You may include fenced code blocks when code clarifies an implementation detail.",
            "Avoid unnecessary code blocks for non-programming topics.",
            (
                f'If code materially clarifies an explanation, prefer fenced "{preferred_language}" snippets.'
                if preferred_language
                else ""
            ),
            "You may include fenced Mermaid diagrams when architecture or flows are better shown visually.",
            "If you include fences, always use explicit language tags (for example: ```yaml, ```mermaid).",
            "Because output must be valid JSON, escape newlines, quotes, and backslashes correctly inside string values.",
            f'target_level must match "{target_level}" exactly.',
            "Complexity must match target_level.",
            "junior: direct recall and simple application.",
            "mid: applied reasoning with concrete constraints.",
            "senior: architecture-level reasoning and tradeoff depth.",
            "Keep markdown compact and practical.",
        ],
    )
    return f"""You are an expert technical quiz creator.

Create exactly {count} quiz questions from the documentation below.{diff_clause}
Target candidate level: "{target_level}".
{type_instructions}

{prompt_contract}

Documentation:
{docs_text[:_MAX_DOC_CONTEXT]}{mcp_block}"""


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
    text = (raw or "").strip()
    if not text:
        return []

    def _extract_items(payload: Any) -> list[dict]:
        if isinstance(payload, list):
            return [x for x in payload if isinstance(x, dict)]
        if isinstance(payload, dict):
            nested = payload.get("questions") or payload.get("items")
            if isinstance(nested, list):
                return [x for x in nested if isinstance(x, dict)]
            if payload.get("question") and payload.get("answer"):
                return [payload]
        return []

    def _try_load_json(candidate: str) -> list[dict]:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return []
        return _extract_items(parsed)

    def _strip_outer_code_fence(candidate: str) -> str:
        stripped = candidate.strip()
        if not stripped.startswith("```"):
            return stripped
        lines = stripped.splitlines()
        if len(lines) < 3 or not lines[-1].strip().startswith("```"):
            return stripped
        return "\n".join(lines[1:-1]).strip()

    def _parse_text_qa_pairs(candidate: str) -> list[dict]:
        cleaned = candidate.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = _strip_outer_code_fence(cleaned)

        q_pattern = re.compile(
            r"(?im)^\s*(?:\d+\s*[\).:\-]\s*)?(?:\*\*)?\s*(?:question|q)\s*\d*\s*[:\-]\s*"
        )
        a_pattern = re.compile(r"(?im)^\s*(?:\*\*)?\s*(?:answer|a)\s*\d*\s*[:\-]\s*")
        q_matches = list(q_pattern.finditer(cleaned))
        items: list[dict] = []

        if q_matches:
            for idx, q_match in enumerate(q_matches):
                block_end = q_matches[idx + 1].start() if idx + 1 < len(q_matches) else len(cleaned)
                block = cleaned[q_match.end() : block_end].strip()
                if not block:
                    continue
                answer_match = a_pattern.search(block)
                if answer_match:
                    question = block[: answer_match.start()].strip(" -*\t\n")
                    answer = block[answer_match.end() :].strip(" \t\n")
                else:
                    parts = [p.strip() for p in block.split("\n", 1)]
                    question = parts[0].strip(" -*\t")
                    answer = parts[1].strip() if len(parts) > 1 else ""
                if question and answer:
                    items.append({"question": question, "answer": answer})
            if items:
                return items

        numbered_pattern = re.compile(
            r"(?ims)^\s*(\d+)[\).:\-]\s*(.+?)(?:\n\s*(?:answer|a)\s*[:\-]\s*(.+?))(?=^\s*\d+[\).:\-]|\Z)"
        )
        for match in numbered_pattern.finditer(cleaned):
            question = match.group(2).strip(" -*\t\n")
            answer = match.group(3).strip()
            if question and answer:
                items.append({"question": question, "answer": answer})
        return items

    parsed = _try_load_json(text)
    if parsed:
        return parsed

    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        parsed = _try_load_json(m.group(1))
        if parsed:
            return parsed

    decoder = json.JSONDecoder()
    cursor = 0
    while cursor < len(text):
        start_match = re.search(r"[\[{]", text[cursor:])
        if not start_match:
            break
        start = cursor + start_match.start()
        try:
            decoded, consumed = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        parsed = _extract_items(decoded)
        if parsed:
            return parsed
        cursor = start + consumed

    parsed = _parse_text_qa_pairs(text)
    if parsed:
        return parsed

    return []


def _extract_content_fragments(content: str, *, limit: int = 48) -> list[str]:
    text = (content or "").replace("\r\n", "\n").replace("\r", "\n")
    if not text:
        return []

    fragments: list[str] = []
    seen: set[str] = set()
    lines = text.splitlines()
    for line in lines:
        chunk = line.strip()
        if not chunk:
            continue
        if chunk.startswith("```"):
            continue
        chunk = re.sub(r"^#{1,6}\s*", "", chunk)
        chunk = re.sub(r"^[\-\*\d\.\)\s]+", "", chunk).strip()
        chunk = re.sub(r"\s+", " ", chunk)
        if len(chunk) < 6:
            continue
        if len(chunk) > 140:
            chunk = chunk[:140].rsplit(" ", 1)[0].strip() or chunk[:140]
        norm = _normalise_question(chunk)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        fragments.append(chunk.rstrip(":;,. "))
        if len(fragments) >= limit:
            break

    if not fragments:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for p in paragraphs[:limit]:
            cleaned = re.sub(r"\s+", " ", p)
            if len(cleaned) < 6:
                continue
            fragments.append(cleaned[:140].rsplit(" ", 1)[0].strip() or cleaned[:140])
            if len(fragments) >= limit:
                break
    return fragments


def _build_fallback_question_items(
    *,
    topic_id: str,
    topic_title: str,
    doc_content: str,
    count: int,
    difficulty: Optional[str],
    level: Optional[str],
    section_title: Optional[str] = None,
    section_content: Optional[str] = None,
    existing_questions: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    remaining = max(0, int(count or 0))
    if remaining <= 0:
        return []

    source_scope = section_title or topic_title or "this topic"
    source_content = section_content if (section_title and section_content) else doc_content
    fragments = _extract_content_fragments(source_content, limit=64)
    if not fragments:
        fragments = [source_scope, topic_title or "core concepts", "practical implementation"]

    existing_seed = _normalise_existing_questions(existing_questions)
    dedup: set[str] = {_normalise_question(q) for q in existing_seed}
    seen_questions: list[str] = [*existing_seed]
    difficulty_value = (difficulty or "medium").strip().lower()
    if difficulty_value not in _VALID_DIFFICULTIES:
        difficulty_value = "medium"

    templates = [
        "What are the key design considerations for {focus} in {topic} from a {angle} perspective?",
        "How would you apply {focus} in a real implementation for {topic} while prioritizing {angle}?",
        "What tradeoffs should be evaluated when working with {focus} in {topic} regarding {angle}?",
        "How would you validate that {focus} is implemented correctly in {topic} with emphasis on {angle}?",
        "A production issue appears around {focus} in {topic}. How would you diagnose and fix it with focus on {angle}?",
        "How would you test and monitor {focus} in {topic} to maintain strong {angle} guarantees?",
        "If {focus} must scale in {topic}, what architecture changes would you make for better {angle}?",
        "When refactoring {focus} in {topic}, how would you reduce risk and preserve {angle} behavior?",
    ]
    angles = [
        "correctness",
        "reliability",
        "performance",
        "security",
        "maintainability",
        "scalability",
        "observability",
        "cost efficiency",
        "testing",
    ]

    items: list[dict[str, Any]] = []
    idx = 0
    max_rounds = max(remaining * 12, 24)
    while len(items) < remaining and idx < max_rounds:
        focus = fragments[idx % len(fragments)]
        angle = angles[(idx // max(1, len(fragments))) % len(angles)]
        template = templates[idx % len(templates)]
        question = template.format(
            focus=focus,
            topic=topic_title or "this topic",
            angle=angle,
        ).strip()
        q_norm = _normalise_question(question)
        idx += 1
        if not q_norm or q_norm in dedup or _is_near_duplicate_question(question, seen_questions):
            continue

        answer = (
            f"Start by defining clear requirements for **{focus}** in the context of {source_scope}.\n\n"
            "Then evaluate constraints, tradeoffs, and failure modes before choosing an approach.\n\n"
            "A strong answer should justify decisions with practical implementation and validation steps."
        )
        item: dict[str, Any] = {
            "question": question,
            "answer": answer,
            "difficulty": difficulty_value,
            "learning_objective": f"Assess practical understanding of {focus}.",
            "source_section": source_scope,
            "source_quote": focus,
            "misconception_trap": "Choosing an approach without validating constraints and tradeoffs.",
            "reasoning_summary": "Clarify requirements first, then compare options and justify a decision.",
            "target_level": _normalise_level(level),
            "topic_id": topic_id,
        }
        valid, _issue = _validate_question_item(item, topic_id, difficulty, level)
        if not valid:
            continue
        dedup.add(q_norm)
        seen_questions.append(question)
        items.append(item)

    return items[:remaining]


def _validate_question_item(
    item: dict[str, Any],
    topic_id: str,
    difficulty: Optional[str],
    level: Optional[str],
) -> tuple[bool, str]:
    question = str(item.get("question", "")).strip()
    answer = str(item.get("answer", "")).strip()
    if not question:
        return False, "missing_or_empty_question"
    if not answer:
        return False, "missing_or_empty_answer"

    diff = str(item.get("difficulty", "")).strip().lower()
    if difficulty:
        diff = str(difficulty).strip().lower()
    if diff not in _VALID_DIFFICULTIES:
        diff = "medium"

    target_level = _normalise_level(level)
    item["question"] = question
    item["answer"] = answer
    item["learning_objective"] = (
        str(item.get("learning_objective", "")).strip()
        or "Assess understanding of the current checkpoint and practical tradeoffs."
    )
    item["source_section"] = str(item.get("source_section", "")).strip() or "Topic section"
    item["source_quote"] = (
        str(item.get("source_quote", "")).strip()
        or "Generated from the current topic content."
    )
    item["misconception_trap"] = (
        str(item.get("misconception_trap", "")).strip()
        or "Overlooking requirements and constraints from the topic context."
    )
    item["reasoning_summary"] = (
        str(item.get("reasoning_summary", "")).strip()
        or "Start from requirements, then evaluate tradeoffs before finalizing an answer."
    )
    item["target_level"] = target_level
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
    existing_questions: Optional[list[str]] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    existing_seed = _normalise_existing_questions(existing_questions)
    dedup: set[str] = {_normalise_question(q) for q in existing_seed}
    seen_questions: list[str] = [*existing_seed]
    valid_items: list[dict[str, Any]] = []
    malformed_items = 0
    retries_used = 0
    issue_counter: Counter[str] = Counter()
    metadata: dict[str, Any] = {}
    prompt = base_prompt
    last_error = ""

    for _ in range(_attempt_budget(target_count)):
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
                existing_questions=[*existing_seed, *[x.get("question", "") for x in valid_items]],
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
                existing_questions=[*existing_seed, *[x.get("question", "") for x in valid_items]],
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
            question_text = str(item.get("question", "")).strip()
            if not q_norm or q_norm in dedup or _is_near_duplicate_question(question_text, seen_questions):
                malformed_items += 1
                issue_counter["duplicate_question"] += 1
                continue
            dedup.add(q_norm)
            seen_questions.append(question_text)
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
            existing_questions=[*existing_seed, *[x.get("question", "") for x in valid_items]],
        )

    if len(valid_items) < target_count:
        # Final pass: request one item at a time to reduce duplicate/schema drift.
        for _ in range(_recovery_budget(target_count - len(valid_items))):
            if len(valid_items) >= target_count:
                break

            single_prompt = _build_retry_prompt(
                base_prompt=base_prompt,
                missing_count=1,
                issues="final_recovery_fill_missing_items",
                existing_questions=[*existing_seed, *[x.get("question", "") for x in valid_items]],
            )
            result = await llm.completion(
                single_prompt,
                llm_config,
                user_identity=user_identity,
            )
            raise_if_policy_blocked_result(result)
            metadata = result.get("metadata", {})
            if not result.get("success"):
                retries_used += 1
                issue_counter["transport_or_provider_error"] += 1
                continue

            parsed = _parse_questions_json(result.get("analysis", ""))
            if not parsed:
                retries_used += 1
                malformed_items += 1
                issue_counter["json_parse_failed"] += 1
                continue

            added = False
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
                question_text = str(item.get("question", "")).strip()
                if not q_norm or q_norm in dedup or _is_near_duplicate_question(question_text, seen_questions):
                    malformed_items += 1
                    issue_counter["duplicate_question"] += 1
                    continue

                dedup.add(q_norm)
                seen_questions.append(question_text)
                valid_items.append(item)
                added = True
                break

            if not added:
                retries_used += 1

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

    def __init__(self, llm_client: LLMClient, mcp_gateway: MCPGateway | None = None):
        self.llm = llm_client
        self.mcp = mcp_gateway

    async def _mcp_context_for_flow(
        self,
        *,
        flow: str,
        query: str,
        topic_id: str = "",
        topic_title: str = "",
    ) -> str:
        if self.mcp is None:
            return ""
        return await self.mcp.gather_context(
            flow=flow,
            query=query,
            topic_id=topic_id,
            topic_title=topic_title,
        )

    @staticmethod
    def _to_question_answer_v2(topic_id: str, item: dict[str, Any]) -> QuestionAnswerV2:
        return QuestionAnswerV2(
            question_id=_question_id(topic_id, item["question"]),
            topic_id=topic_id,
            question=item["question"].strip(),
            answer=format_markdown_readable(item["answer"].strip()),
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
        requested_total_count: Optional[int] = None,
        existing_questions: Optional[list[str]] = None,
        difficulty: Optional[str] = None,
        level: Optional[str] = None,
        llm_config: Optional[LLMConfigRequest] = None,
        user_identity: Optional[dict] = None,
        section_title: Optional[str] = None,
        section_content: Optional[str] = None,
        response_detail: str = "concise",
        preferred_language: str = "",
        requires_programming: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        target_count = max(0, int(count))
        requested_total = max(1, int(requested_total_count or count or 1))
        seed_existing = _normalise_existing_questions(existing_questions)
        mcp_context = await self._mcp_context_for_flow(
            flow="questions",
            query=f"{topic_title} {section_title or ''} interview questions and practical examples",
            topic_id=topic_id,
            topic_title=topic_title,
        )
        dedup_norm: set[str] = {_normalise_question(q) for q in seed_existing}
        generated_question_texts: list[str] = [*seed_existing]

        async def _generate_one(*, include_section: bool, relaxed: bool = False) -> dict[str, Any]:
            retries_used = 0
            malformed_dropped = 0
            metadata: dict[str, Any] = {}
            issue_counter: Counter[str] = Counter()
            last_error = ""
            base_prompt = _build_prompt(
                topic_id=topic_id,
                topic_title=topic_title,
                doc_content=doc_content,
                count=1,
                difficulty=None if relaxed else difficulty,
                level=level,
                section_title=section_title if include_section else None,
                section_content=section_content if include_section else None,
                response_detail=response_detail,
                preferred_language="" if relaxed else preferred_language,
                requires_programming=False if relaxed else requires_programming,
                requested_total_count=requested_total,
                existing_questions=generated_question_texts,
                mcp_context=mcp_context,
            )
            prompt = base_prompt

            try:
                for _ in range(_attempt_budget(1)):
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
                        prompt = _build_retry_prompt(
                            base_prompt=base_prompt,
                            missing_count=1,
                            issues=f"transport_or_provider_error: {last_error}",
                            existing_questions=generated_question_texts,
                        )
                        continue

                    parsed = _parse_questions_json(result.get("analysis", ""))
                    if not parsed:
                        retries_used += 1
                        malformed_dropped += 1
                        issue_counter["json_parse_failed"] += 1
                        prompt = _build_retry_prompt(
                            base_prompt=base_prompt,
                            missing_count=1,
                            issues="json_parse_failed",
                            existing_questions=generated_question_texts,
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

                        if q_norm in dedup_norm or _is_near_duplicate_question(q_text, generated_question_texts):
                            malformed_dropped += 1
                            issue_counter["duplicate_question"] += 1
                            continue

                        dedup_norm.add(q_norm)
                        generated_question_texts.append(q_text)
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
                    prompt = _build_retry_prompt(
                        base_prompt=base_prompt,
                        missing_count=1,
                        issues=top_issues,
                        existing_questions=generated_question_texts,
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

        section_first = bool(section_title and section_content)
        phase_modes: list[tuple[bool, bool]] = (
            [(True, False), (False, False), (False, True)]
            if section_first
            else [(False, False), (False, True)]
        )
        for include_section, relaxed in phase_modes:
            while generated_count < target_count:
                result = await _generate_one(include_section=include_section, relaxed=relaxed)
                retries_used += int(result.get("retries_used", 0))
                malformed_items_dropped += int(result.get("malformed_items_dropped", 0))
                metadata = result.get("metadata", {}) or {}
                if metadata.get("provider"):
                    provider_used = str(metadata.get("provider"))
                if metadata.get("model"):
                    model_used = str(metadata.get("model"))

                error = result.get("error")
                if error:
                    yield {
                        "type": "error",
                        "code": str(error.get("code", "generation_failed")),
                        "message": str(error.get("message", "Question generation failed")),
                    }
                    return

                question = result.get("question")
                if not isinstance(question, dict):
                    break

                generated_count += 1
                yield {
                    "type": "question",
                    "question": question,
                }

        if generated_count < target_count:
            for _ in range(_recovery_budget(target_count - generated_count)):
                if generated_count >= target_count:
                    break

                result = await _generate_one(include_section=False, relaxed=True)
                retries_used += int(result.get("retries_used", 0))
                malformed_items_dropped += int(result.get("malformed_items_dropped", 0))
                metadata = result.get("metadata", {}) or {}
                if metadata.get("provider"):
                    provider_used = str(metadata.get("provider"))
                if metadata.get("model"):
                    model_used = str(metadata.get("model"))

                error = result.get("error")
                if error:
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

        if generated_count < target_count:
            fallback_items = _build_fallback_question_items(
                topic_id=topic_id,
                topic_title=topic_title,
                doc_content=doc_content,
                count=target_count - generated_count,
                difficulty=difficulty,
                level=level,
                section_title=None,
                section_content=None,
                existing_questions=generated_question_texts,
            )
            for item in fallback_items:
                q_text = str(item.get("question", "")).strip()
                q_norm = _normalise_question(q_text)
                if not q_norm or q_norm in dedup_norm or _is_near_duplicate_question(q_text, generated_question_texts):
                    continue
                dedup_norm.add(q_norm)
                generated_question_texts.append(q_text)
                qa = self._to_question_answer_v2(topic_id, item)
                generated_count += 1
                yield {
                    "type": "question",
                    "question": qa.model_dump(),
                }
                if generated_count >= target_count:
                    break

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
        requested_total_count: Optional[int] = None,
        existing_questions: Optional[list[str]] = None,
        difficulty: Optional[str] = None,
        level: Optional[str] = None,
        llm_config: Optional[LLMConfigRequest] = None,
        user_identity: Optional[dict] = None,
        section_title: Optional[str] = None,
        section_content: Optional[str] = None,
        response_detail: str = "concise",
        preferred_language: str = "",
        requires_programming: bool = False,
    ) -> GenerateQuestionsResponse:
        v2 = await self.generate_v2(
            topic_id=topic_id,
            topic_title=topic_title,
            doc_content=doc_content,
            count=count,
            requested_total_count=requested_total_count,
            existing_questions=existing_questions,
            difficulty=difficulty,
            level=level,
            llm_config=llm_config,
            user_identity=user_identity,
            section_title=section_title,
            section_content=section_content,
            response_detail=response_detail,
            preferred_language=preferred_language,
            requires_programming=requires_programming,
        )
        target_count = max(1, int(count or 1))
        questions = [
            QuestionAnswer(question=q.question, answer=q.answer, difficulty=q.difficulty)
            for q in v2.questions
        ]
        if len(questions) < target_count:
            fallback_items = _build_fallback_question_items(
                topic_id=topic_id,
                topic_title=topic_title,
                doc_content=doc_content,
                count=target_count - len(questions),
                difficulty=difficulty,
                level=level,
                section_title=section_title,
                section_content=section_content,
                existing_questions=[
                    *(_normalise_existing_questions(existing_questions)),
                    *[q.question for q in questions],
                ],
            )
            for item in fallback_items:
                questions.append(
                    QuestionAnswer(
                        question=str(item.get("question", "")).strip(),
                        answer=str(item.get("answer", "")).strip(),
                        difficulty=str(item.get("difficulty", "medium")).strip().lower() or "medium",
                    )
                )
                if len(questions) >= target_count:
                    break
        return GenerateQuestionsResponse(
            topic_id=topic_id,
            topic_title=topic_title,
            questions=questions[:target_count],
            provider_used=v2.provider_used,
            model_used=v2.model_used,
        )

    async def generate_v2(
        self,
        topic_id: str,
        topic_title: str,
        doc_content: str,
        count: int = 5,
        requested_total_count: Optional[int] = None,
        existing_questions: Optional[list[str]] = None,
        difficulty: Optional[str] = None,
        level: Optional[str] = None,
        llm_config: Optional[LLMConfigRequest] = None,
        user_identity: Optional[dict] = None,
        section_title: Optional[str] = None,
        section_content: Optional[str] = None,
        response_detail: str = "concise",
        preferred_language: str = "",
        requires_programming: bool = False,
    ) -> GenerateQuestionsV2Response:
        target_count = max(1, int(count or 1))
        mcp_context = await self._mcp_context_for_flow(
            flow="questions",
            query=f"{topic_title} {section_title or ''} interview questions and practical examples",
            topic_id=topic_id,
            topic_title=topic_title,
        )
        validator = lambda item: _validate_question_item(item, topic_id, difficulty, level)
        provider_used = ""
        model_used = ""
        retries_used = 0
        malformed_items_dropped = 0
        raw_items: list[dict[str, Any]] = []
        existing_seed = _normalise_existing_questions(existing_questions)
        dedup_norm: set[str] = {_normalise_question(q) for q in existing_seed}
        seen_questions: list[str] = [*existing_seed]

        pass_modes: list[dict[str, Any]] = [
            {
                "section_title": section_title,
                "section_content": section_content,
                "difficulty": difficulty,
                "preferred_language": preferred_language,
                "requires_programming": requires_programming,
            }
        ]
        if section_title and section_content:
            pass_modes.append(
                {
                    "section_title": None,
                    "section_content": None,
                    "difficulty": difficulty,
                    "preferred_language": preferred_language,
                    "requires_programming": requires_programming,
                }
            )
        pass_modes.append(
            {
                "section_title": None,
                "section_content": None,
                "difficulty": None,
                "preferred_language": "",
                "requires_programming": False,
            }
        )

        for mode in pass_modes:
            if len(raw_items) >= target_count:
                break
            remaining = target_count - len(raw_items)
            existing_for_pass = [
                *existing_seed,
                *[str(item.get("question", "")) for item in raw_items],
            ]
            prompt = _build_prompt(
                topic_id=topic_id,
                topic_title=topic_title,
                doc_content=doc_content,
                count=remaining,
                requested_total_count=requested_total_count,
                existing_questions=existing_for_pass,
                difficulty=mode["difficulty"],
                level=level,
                section_title=mode["section_title"],
                section_content=mode["section_content"],
                response_detail=response_detail,
                preferred_language=mode["preferred_language"],
                requires_programming=mode["requires_programming"],
                mcp_context=mcp_context,
            )
            pass_items, pass_stats = await _collect_with_retries(
                llm=self.llm,
                base_prompt=prompt,
                llm_config=llm_config,
                user_identity=user_identity,
                target_count=remaining,
                validator=validator,
                existing_questions=existing_for_pass,
            )
            retries_used += int(pass_stats.get("retries_used", 0))
            malformed_items_dropped += int(pass_stats.get("malformed_items_dropped", 0))
            pass_metadata = pass_stats.get("metadata", {}) or {}
            if pass_metadata.get("provider"):
                provider_used = str(pass_metadata.get("provider"))
            if pass_metadata.get("model"):
                model_used = str(pass_metadata.get("model"))

            for item in pass_items:
                question_text = str(item.get("question", "")).strip()
                q_norm = _normalise_question(question_text)
                if not q_norm or q_norm in dedup_norm or _is_near_duplicate_question(question_text, seen_questions):
                    continue
                dedup_norm.add(q_norm)
                seen_questions.append(question_text)
                raw_items.append(item)
                if len(raw_items) >= target_count:
                    break

        if len(raw_items) < target_count:
            fallback_items = _build_fallback_question_items(
                topic_id=topic_id,
                topic_title=topic_title,
                doc_content=doc_content,
                count=target_count - len(raw_items),
                difficulty=difficulty,
                level=level,
                section_title=None,
                section_content=None,
                existing_questions=[
                    *existing_seed,
                    *[str(item.get("question", "")) for item in raw_items],
                ],
            )
            for item in fallback_items:
                question_text = str(item.get("question", "")).strip()
                q_norm = _normalise_question(question_text)
                if not q_norm or q_norm in dedup_norm or _is_near_duplicate_question(question_text, seen_questions):
                    continue
                dedup_norm.add(q_norm)
                seen_questions.append(question_text)
                raw_items.append(item)
                if len(raw_items) >= target_count:
                    break

        questions = [self._to_question_answer_v2(topic_id, item) for item in raw_items[:target_count]]

        return GenerateQuestionsV2Response(
            topic_id=topic_id,
            topic_title=topic_title,
            questions=questions,
            provider_used=provider_used,
            model_used=model_used,
            retries_used=retries_used,
            malformed_items_dropped=malformed_items_dropped,
        )

    async def generate_quiz(
        self,
        topics_content: list[dict],
        count: int = 10,
        question_types: list[str] | None = None,
        difficulty: Optional[str] = None,
        level: Optional[str] = None,
        response_detail: str | None = None,
        preferred_language: str = "",
        llm_config: Optional[LLMConfigRequest] = None,
        user_identity: Optional[dict] = None,
    ) -> GenerateQuizResponse:
        v2 = await self.generate_quiz_v2(
            topics_content=topics_content,
            count=count,
            question_types=question_types,
            difficulty=difficulty,
            level=level,
            response_detail=response_detail,
            preferred_language=preferred_language,
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
        response_detail: str | None = None,
        preferred_language: str = "",
        llm_config: Optional[LLMConfigRequest] = None,
        user_identity: Optional[dict] = None,
    ) -> GenerateQuizV2Response:
        effective_detail = "very_detailed" if response_detail == "very_detailed" else "concise"
        mcp_context = await self._mcp_context_for_flow(
            flow="quiz",
            query=" ".join(
                [
                    " ".join(str(tc.get("title", "")).strip() for tc in topics_content[:4]),
                    "technical quiz generation",
                ]
            ).strip(),
        )
        prompt = _build_quiz_prompt(
            topics_content,
            count,
            question_types,
            difficulty,
            level,
            response_detail=effective_detail,
            preferred_language=(preferred_language or "").strip().lower(),
            mcp_context=mcp_context,
        )
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
                    explanation=format_markdown_readable(item["explanation"].strip()),
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
