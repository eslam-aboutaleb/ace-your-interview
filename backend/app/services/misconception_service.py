"""Classify *why* a learner's answer was wrong (misconception memory).

Wrong answers were stored but never read back. This service turns one wrong
answer into a tagged error pattern — an ``error_type`` plus a short
explanation — that the learning store can rank and later hand to the
learner-memory wave as an LLM-visible learner-state block.

Contract: :func:`classify_misconception` never raises. Every failure mode
(policy block, budget exhaustion, provider error, unparsable or invalid JSON
payload) degrades to the deterministic fallback, and callers treat the
fallback exactly like an LLM result.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from app.config import get_settings
from app.schemas.models import LLMConfigRequest
from app.services.llm_client import (
    TERMINAL_ERROR_CODES,
    parse_json_object,
)
from app.services.llm_policy import raise_if_policy_blocked_result
from app.services.prompt_blocks import (
    UNTRUSTED_CLAUSE,
    render_contract,
    untrusted_block,
)

logger = logging.getLogger(__name__)

# ── Bounds ───────────────────────────────────────────────────
MAX_ATTEMPTS = 2
MAX_TOKENS_CAP = 400
MAX_QUESTION_CHARS = 2000
MAX_ANSWER_CHARS = 2000
MAX_USER_ANSWER_CHARS = 2000
MAX_TOPIC_TITLE_CHARS = 200
MAX_EXPLANATION_CHARS = 400
MAX_OBJECTIVE_CHARS = 240
#: Below this length an answer is too thin to diagnose (MCQ letters, "it
#: depends" style one-liners); classifying them is not worth an LLM call.
MIN_USER_ANSWER_CHARS = 20

# ── Error-type vocabulary ────────────────────────────────────
ERROR_TYPE_NO_ANSWER = "no_answer"
ERROR_TYPE_PARTIAL_ANSWER = "partial_answer"
ERROR_TYPE_MISCONCEPTION = "misconception"
ERROR_TYPE_FACTUAL_ERROR = "factual_error"
ERROR_TYPE_MISSING_DEPTH = "missing_depth"
ERROR_TYPE_UNCLASSIFIED = "unclassified"

ERROR_TYPES: tuple[str, ...] = (
    ERROR_TYPE_NO_ANSWER,
    ERROR_TYPE_PARTIAL_ANSWER,
    ERROR_TYPE_MISCONCEPTION,
    ERROR_TYPE_FACTUAL_ERROR,
    ERROR_TYPE_MISSING_DEPTH,
    ERROR_TYPE_UNCLASSIFIED,
)

#: Deterministic fallback explanations, keyed by error type.
_FALLBACK_EXPLANATIONS: dict[str, str] = {
    ERROR_TYPE_NO_ANSWER: (
        "The learner left the answer blank or said they did not know it."
    ),
    ERROR_TYPE_PARTIAL_ANSWER: (
        "The answer shares almost no vocabulary with the expected answer, "
        "so it reads as a partial or off-target attempt."
    ),
    ERROR_TYPE_UNCLASSIFIED: (
        "The answer could not be diagnosed automatically; the learner needs "
        "a human or LLM pass to name the gap."
    ),
}

_SYSTEM_PROMPT = (
    "You are an interview coach who diagnoses why one learner answer was "
    "wrong. You compare the learner answer against the expected answer and "
    "name the specific gap. You always answer with a single JSON object and "
    "never with prose."
)

#: Phrases that mean "I am not attempting this answer" once normalised
#: (lowercased, punctuation stripped, whitespace collapsed).
_NO_ANSWER_PHRASES = frozenset(
    {
        "i do not know",
        "i dont know",
        "dont know",
        "do not know",
        "idk",
        "i have no idea",
        "no idea",
        "no clue",
        "not sure",
        "im not sure",
        "i m not sure",
        "unsure",
        "no answer",
        "n a",
        "na",
        "none",
        "null",
        "blank",
        "pass",
        "skip",
        "skipped",
        "unknown",
        "i cannot answer",
        "cant answer",
        "can t answer",
        "i m guessing",
        "im guessing",
        "no opinion",
    }
)

#: Phrases that open a non-answer when the whole reply stays short.
_NO_ANSWER_LEADING_PHRASES = frozenset(
    {
        "i do not know",
        "i dont know",
        "dont know",
        "i have no idea",
        "no idea",
        "not sure",
        "im not sure",
        "i m not sure",
        "unsure",
        "i cannot answer",
        "idk",
    }
)

#: A "I don't know ..." opener only counts inside a short reply.
_MAX_NO_ANSWER_WORDS = 14

# Tokens too generic to diagnose overlap with the expected answer.
_OVERLAP_STOPWORDS = frozenset(
    {
        "answer", "because", "before", "being", "could", "every", "first",
        "found", "given", "have", "into", "just", "know", "like", "more",
        "most", "much", "must", "need", "next", "only", "other", "over",
        "same", "should", "since", "some", "such", "sure", "than", "that",
        "their", "them", "then", "there", "these", "they", "this", "those",
        "true", "very", "want", "well", "were", "what", "when", "where",
        "which", "while", "will", "with", "without", "would", "your",
    }
)

#: Minimum share of expected-answer keywords an answer must cover to count as
#: more than a partial attempt.
_MIN_EXPECTED_OVERLAP = 0.3


def _normalise_level(level: str) -> str:
    candidate = str(level or "").strip().lower()
    return candidate if candidate in ("junior", "mid", "senior") else "mid"


def _collapse(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _looks_like_no_answer(user_answer: str) -> bool:
    """True when the answer is empty or is an explicit "I don't know".

    A short answer that *opens* with such a phrase counts as well, so
    "I don't know much about this" is not diagnosed as a misconception.
    """
    # Strip apostrophes first so "don't" and "dont" normalise alike.
    normalised = _collapse(user_answer).replace("'", "").replace("\u2019", "")
    collapsed = " ".join(re.sub(r"[^a-z0-9 ]", " ", normalised).split())
    if not collapsed:
        return True
    if collapsed in _NO_ANSWER_PHRASES:
        return True
    if len(collapsed.split()) > _MAX_NO_ANSWER_WORDS:
        return False
    return any(
        collapsed.startswith(f"{phrase} ")
        for phrase in _NO_ANSWER_LEADING_PHRASES
    )


def _keyword_set(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]{4,}", str(text or "").lower())
        if token not in _OVERLAP_STOPWORDS
    }


def deterministic_classification(
    *,
    question: str = "",
    correct_answer: str = "",
    user_answer: str = "",
) -> dict[str, str]:
    """Label a wrong answer without any LLM call.

    Empty / "I don't know" answers are :data:`ERROR_TYPE_NO_ANSWER`; an answer
    that barely overlaps the expected answer's keywords is
    :data:`ERROR_TYPE_PARTIAL_ANSWER`; anything else is
    :data:`ERROR_TYPE_UNCLASSIFIED`.
    """
    answer = str(user_answer or "")
    if _looks_like_no_answer(answer):
        return _result(ERROR_TYPE_NO_ANSWER, source="rule")

    expected_keywords = _keyword_set(correct_answer)
    if not expected_keywords:
        # No expected answer text was available, so overlap cannot be measured.
        return _result(ERROR_TYPE_UNCLASSIFIED, source="rule")

    answer_keywords = _keyword_set(answer)
    overlap = (
        len(answer_keywords & expected_keywords) / len(expected_keywords)
        if answer_keywords
        else 0.0
    )
    if overlap < _MIN_EXPECTED_OVERLAP:
        return _result(ERROR_TYPE_PARTIAL_ANSWER, source="rule")
    return _result(ERROR_TYPE_UNCLASSIFIED, source="rule")


def _result(error_type: str, *, explanation: str = "", source: str) -> dict[str, str]:
    text = " ".join(str(explanation or "").split())[:MAX_EXPLANATION_CHARS].strip()
    if not text:
        text = _FALLBACK_EXPLANATIONS.get(
            error_type, _FALLBACK_EXPLANATIONS[ERROR_TYPE_UNCLASSIFIED]
        )
    return {
        "error_type": error_type,
        "explanation": text,
        "source": source,
    }


def derive_objective(question: str, fallback: str = "") -> str:
    """Reduce a question to the stable objective it is testing.

    Two phrasings of the same question must collapse to the same objective so
    the store's dedupe key groups repeat mistakes together.
    """
    text = " ".join(str(question or "").split())
    if text:
        for separator in ("?", ".", "!", "\n"):
            text = text.split(separator)[0]
        text = text.strip(" -–—:\t")
    objective = _collapse(text)[:MAX_OBJECTIVE_CHARS].strip(" -–—:\t")
    if objective:
        return objective
    fallback_text = _collapse(fallback)[:MAX_OBJECTIVE_CHARS].strip(" -–—:\t")
    return fallback_text or "general"


@lru_cache(maxsize=8)
def _curriculum_topic_titles(curriculum_path: str) -> dict[str, str]:
    try:
        with open(curriculum_path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    topics = data.get("topics") if isinstance(data, dict) else None
    if not isinstance(topics, list):
        return {}
    titles: dict[str, str] = {}
    for item in topics:
        if not isinstance(item, dict):
            continue
        topic_id = str(item.get("id", "")).strip()
        title = str(item.get("title", "")).strip()
        if topic_id and title:
            titles[topic_id] = title
    return titles


def resolve_topic_title(topic_id: str) -> str:
    """Best-effort topic title for the static curriculum (never raises)."""
    configured = str(getattr(get_settings(), "curriculum_path", "") or "").strip()
    if not configured:
        return ""
    candidates = [Path(configured)]
    if not Path(configured).is_absolute():
        candidates.append(Path(__file__).resolve().parents[2] / configured)
    for candidate in candidates:
        title = _curriculum_topic_titles(str(candidate)).get(
            str(topic_id or "").strip(), ""
        )
        if title:
            return title
    return ""


def _supported_options(client: Any, options: dict[str, Any]) -> dict[str, Any]:
    """Drop call options ``client.completion`` does not accept.

    ``LLMClient`` takes the full option set, but the client is injected, so a
    narrower implementation may be in use.
    """
    try:
        params = inspect.signature(client.completion).parameters
    except (TypeError, ValueError):
        return dict(options)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return options
    return {key: value for key, value in options.items() if key in params}


def build_misconception_prompt(
    *,
    question: str,
    correct_answer: str,
    user_answer: str,
    level: str = "",
    topic_title: str = "",
) -> str:
    contract = render_contract(
        schema_label="Return ONLY valid JSON",
        schema_block=(
            "{\n"
            '  "error_type": "no_answer" | "partial_answer" | "misconception" '
            "| \"factual_error\" | \"missing_depth\",\n"
            '  "explanation": "the specific gap, at most 40 words"\n'
            "}"
        ),
        rules=[
            "Pick exactly one error_type from the vocabulary above.",
            "misconception = the learner states something false; factual_error = "
            "the concept is right but a detail is wrong; missing_depth = the "
            "direction is right but stops short of the expected reasoning; "
            "partial_answer = only a fragment of the expected answer is present.",
            "Base the diagnosis only on the learner answer compared with the "
            "expected answer, not on how confident or fluent it sounds.",
            "Describe the gap without restating the correct answer and without "
            "adding new facts.",
            "Use partial_answer or no_answer for the fallback cases, never "
            "misconception.",
            "Output JSON only.",
            UNTRUSTED_CLAUSE,
        ],
    )
    metadata = "\n".join(
        [
            f"Expected level: {_normalise_level(level)}",
            f"Learner answer length: {len(str(user_answer or ''))} characters",
        ]
    )
    untrusted = "\n\n".join(
        block
        for block in (
            untrusted_block("topic_title", topic_title, MAX_TOPIC_TITLE_CHARS),
            untrusted_block("question", question, MAX_QUESTION_CHARS),
            untrusted_block("expected_answer", correct_answer, MAX_ANSWER_CHARS),
            untrusted_block("learner_answer", user_answer, MAX_USER_ANSWER_CHARS),
        )
        if block
    )
    return f"{contract}\n\n{metadata}\n\n{untrusted}\n"


def _normalise_payload(payload: dict[str, Any]) -> dict[str, str] | None:
    if not isinstance(payload, dict):
        return None
    error_type = str(payload.get("error_type", "")).strip().lower()
    if error_type not in ERROR_TYPES:
        return None
    return _result(error_type, explanation=str(payload.get("explanation", "") or ""), source="llm")


async def _classify_with_llm(
    llm_client: Any,
    *,
    question: str,
    correct_answer: str,
    user_answer: str,
    level: str = "",
    topic_title: str = "",
    llm_config: Optional[LLMConfigRequest] = None,
    user_identity: Optional[dict] = None,
) -> dict[str, str]:
    """Ask the LLM for a label; return ``{}`` when nothing usable comes back."""
    prompt = build_misconception_prompt(
        question=question,
        correct_answer=correct_answer,
        user_answer=user_answer,
        level=level,
        topic_title=topic_title,
    )
    for _ in range(MAX_ATTEMPTS):
        result = await llm_client.completion(
            prompt,
            llm_config,
            user_identity=user_identity,
            **_supported_options(
                llm_client,
                {
                    "task": "final",
                    "system": _SYSTEM_PROMPT,
                    "structured": True,
                    "max_tokens_cap": MAX_TOKENS_CAP,
                },
            ),
        )
        raise_if_policy_blocked_result(result)
        if not result.get("success"):
            # A truncated response or an exhausted budget cannot be fixed by
            # re-sending the identical prompt.
            if str(result.get("error_code") or "") in TERMINAL_ERROR_CODES:
                return {}
            continue
        classified = _normalise_payload(parse_json_object(result.get("analysis", "")))
        if classified:
            return classified
    return {}


async def classify_misconception(
    llm_client: Any = None,
    *,
    question: str = "",
    correct_answer: str = "",
    user_answer: str = "",
    level: str = "",
    topic_title: str = "",
    llm_config: Optional[LLMConfigRequest] = None,
    user_identity: Optional[dict] = None,
) -> dict[str, str]:
    """Label one wrong answer. Never raises.

    Returns ``{"error_type", "explanation", "source"}`` where ``source`` is
    ``"llm"`` when a model produced the label, ``"fallback"`` when a model was
    asked but produced nothing usable, and ``"rule"`` when the answer was
    labelled deterministically (or no model is wired up at all).
    """
    answer = str(user_answer or "")
    rule_result = deterministic_classification(
        question=question,
        correct_answer=correct_answer,
        user_answer=answer,
    )
    # "I don't know" answers are settled by a rule: an LLM call cannot add
    # anything and would burn budget the hourly cap exists to protect.
    if rule_result["error_type"] == ERROR_TYPE_NO_ANSWER:
        return rule_result
    if len(answer.strip()) < MIN_USER_ANSWER_CHARS:
        return _result(
            ERROR_TYPE_PARTIAL_ANSWER,
            explanation=(
                "The answer is too short to diagnose beyond being a partial "
                "attempt."
            ),
            source="rule",
        )
    if llm_client is None:
        # No model is wired up (tests, or the router was initialised without a
        # client): the deterministic label is the whole answer.
        return rule_result

    try:
        classified = await _classify_with_llm(
            llm_client,
            question=question,
            correct_answer=correct_answer,
            user_answer=answer,
            level=level,
            topic_title=topic_title,
            llm_config=llm_config,
            user_identity=user_identity,
        )
    except Exception:
        logger.warning(
            "Misconception classification failed; using deterministic fallback",
            exc_info=True,
        )
        return {**rule_result, "source": "fallback"}
    if not classified:
        # The model was asked and produced nothing usable, so the failed call
        # has to count against the caller's hourly budget.
        return {**rule_result, "source": "fallback"}
    return classified


class MisconceptionService:
    """Client-scoped wrapper so routers can hold one reusable instance."""

    def __init__(self, llm_client: Any = None) -> None:
        # Only accept something that can actually complete: a router initialised
        # with the wrong object must degrade to the deterministic label rather
        # than spend attempts on an unusable client.
        self.llm_client = (
            llm_client if callable(getattr(llm_client, "completion", None)) else None
        )

    async def classify_misconception(
        self,
        *,
        question: str = "",
        correct_answer: str = "",
        user_answer: str = "",
        level: str = "",
        topic_title: str = "",
        llm_config: Optional[LLMConfigRequest] = None,
        user_identity: Optional[dict] = None,
    ) -> dict[str, str]:
        return await classify_misconception(
            self.llm_client,
            question=question,
            correct_answer=correct_answer,
            user_answer=user_answer,
            level=level,
            topic_title=topic_title,
            llm_config=llm_config,
            user_identity=user_identity,
        )
