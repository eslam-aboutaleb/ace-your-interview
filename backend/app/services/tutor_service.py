"""Contextual tutor: question-scoped explanations with misconception tracking.

Actions (issue #15):
    simpler   - re-explain the answer in plainer words
    deeper    - go one level below the canonical answer
    why       - justify why the canonical answer is correct
    compare   - contrast the concept with a caller-named alternative
    example   - produce a runnable or worked example
    follow_up - answer a free-form follow-up scoped to the question

Grounding contract:
    The explanation must be grounded in the question text and answer
    context the caller supplies. The response carries a deterministic
    citation (section + one verbatim quote) built from that context, or
    an explicit "cites none" line when the context has none. With no
    question text and no context at all the service refuses without an
    LLM call, so the tutor can never invent an answer from nothing.

Misconceptions:
    When the caller supplies ``user_answer``, the answer is classified
    (injectable classifier; deterministic rules by default) and the
    response carries the misconception plus a remediation offer
    (objective + flashcard suggestion). The canonical answer and any
    completion state are never altered: this service only appends to
    ``feature_events``.

Cost cap: at most :data:`TUTOR_COST_CAP_PER_QUESTION` explanations per
``(user_id, question_id)`` via the shared ``feature_events`` table.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import UTC, datetime
from typing import Any, Callable

from app.services.llm_client import LLMClient
from app.services.llm_policy import raise_if_policy_blocked_result
from app.services.misconception_service import (
    ERROR_TYPE_NO_ANSWER,
    derive_objective,
    deterministic_classification,
)
from app.services.prompt_blocks import UNTRUSTED_CLAUSE, untrusted_block

logger = logging.getLogger(__name__)

TUTOR_ACTIONS = ("simpler", "deeper", "why", "compare", "example", "follow_up")
TUTOR_COST_CAP_PER_QUESTION = 20
TUTOR_FEATURE_KEY = "tutor_explain"
TUTOR_EVENT_NAME = "explain_generated"

#: Bounds on externally sourced prompt fields.
MAX_QUESTION_CHARS = 4_000
MAX_CONTEXT_CHARS = 8_000
MAX_SECTION_CHARS = 200
MAX_QUOTE_CHARS = 400
MAX_FOLLOW_UP_CHARS = 2_000
MAX_COMPARE_TO_CHARS = 200
HARD_INPUT_GUARD_CHARS = 200_000

TUTOR_SYSTEM_PROMPT = (
    "You are a careful study tutor helping a learner with one technical "
    "interview question. Answer only what the supplied question and answer "
    "context support. If the context does not contain the answer, say so "
    "plainly instead of inventing one. Never change the canonical answer. "
    f"{UNTRUSTED_CLAUSE}"
)

_ACTION_INSTRUCTIONS: dict[str, str] = {
    "simpler": "Re-explain the answer in plainer words, as if to a beginner.",
    "deeper": "Go one level deeper than the canonical answer: mechanism, trade-offs, edge cases.",
    "why": "Justify why the canonical answer is correct and why the obvious alternatives fail.",
    "compare": "Contrast the concept with the alternative the learner named.",
    "example": "Give one concrete, worked example. Code must be short and runnable as written.",
    "follow_up": "Answer the learner's follow-up question, scoped strictly to this question.",
}

REFUSAL_MESSAGE = (
    "I don't have the question or its answer context, so I can't explain "
    "without guessing. Open the question and try again."
)

FAILURE_MESSAGE = (
    "The tutor could not produce an explanation right now. Please try again."
)

ClassifierFn = Callable[..., dict[str, str]]


class TutorCostExceededError(Exception):
    """The per-question tutor budget is exhausted."""

    def __init__(self, question_id: str, used: int, cap: int):
        super().__init__(
            f"Tutor budget exhausted for question {question_id}: "
            f"{used}/{cap} explanations used."
        )
        self.question_id = question_id
        self.used = used
        self.cap = cap


def _fence(label: str, value: str, max_chars: int) -> str:
    return untrusted_block(label, str(value or "").strip()[:HARD_INPUT_GUARD_CHARS], max_chars)


def _grounded_citation(
    source_section: str, source_quote: str
) -> tuple[str, str, str]:
    """Return ``(section, quote, citation_line)`` built from the context.

    Mirrors the chat highlight flow: when the context names no section and
    carries no quote, the line says so, so the answer never invents a
    reference.
    """
    section = str(source_section or "").strip()[:MAX_SECTION_CHARS]
    quote = " ".join(str(source_quote or "").split())[:MAX_QUOTE_CHARS]
    if not section and not quote:
        return "", "", "The supplied context names no source section, so this answer cites none."
    parts: list[str] = []
    if section:
        parts.append(f'section "{section}"')
    if quote:
        parts.append(f'quote "{quote}"')
    return section, quote, "Grounded in the question context: " + " and ".join(parts) + "."


class TutorService:
    """Question-scoped explanations with cost capping and misconception tracking."""

    def __init__(
        self,
        llm: LLMClient,
        db_path: str,
        classifier: ClassifierFn | None = None,
    ):
        self.llm = llm
        self.db_path = db_path
        self.classifier = classifier or deterministic_classification
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_table()

    def _ensure_table(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feature_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    feature_key TEXT NOT NULL,
                    event_name TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                )
                """
            )

    def count_explains_used(self, user_id: str, question_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT COUNT(*) AS total
                FROM feature_events
                WHERE user_id = ? AND feature_key = ? AND event_name = ?
                  AND json_extract(metadata_json, '$.question_id') = ?
                """,
                (user_id, TUTOR_FEATURE_KEY, TUTOR_EVENT_NAME, question_id),
            ).fetchone()
            return int(row["total"]) if row else 0

    def record_explain_event(self, user_id: str, question_id: str, action: str) -> None:
        payload = json.dumps(
            {"question_id": question_id, "action": action}, ensure_ascii=True
        )
        now = datetime.now(UTC).isoformat()
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO feature_events(
                    user_id, feature_key, event_name, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, TUTOR_FEATURE_KEY, TUTOR_EVENT_NAME, payload, now),
            )

    async def explain(
        self,
        *,
        user_id: str,
        question_id: str,
        action: str,
        question_text: str = "",
        answer_context: str = "",
        source_section: str = "",
        source_quote: str = "",
        user_answer: str = "",
        follow_up_prompt: str = "",
        compare_to: str = "",
        llm_config: Any = None,
        user_identity: dict | None = None,
    ) -> dict[str, Any]:
        if action not in TUTOR_ACTIONS:
            raise ValueError(f"action must be one of {TUTOR_ACTIONS}")

        # Hallucination refusal: with neither the question nor its context
        # there is nothing to ground on, so refuse before any LLM call.
        if not str(question_text or "").strip() and not str(answer_context or "").strip():
            return {
                "question_id": question_id,
                "action": action,
                "explanation": REFUSAL_MESSAGE,
                "refusal": True,
                "citations": [],
                "explains_used": self.count_explains_used(user_id, question_id),
                "explains_remaining": max(
                    0,
                    TUTOR_COST_CAP_PER_QUESTION
                    - self.count_explains_used(user_id, question_id),
                ),
            }

        used = self.count_explains_used(user_id, question_id)
        if used >= TUTOR_COST_CAP_PER_QUESTION:
            raise TutorCostExceededError(question_id, used, TUTOR_COST_CAP_PER_QUESTION)

        section, quote, citation_line = _grounded_citation(source_section, source_quote)
        prompt = self._build_prompt(
            action=action,
            question_text=question_text,
            answer_context=answer_context,
            section=section,
            quote=quote,
            follow_up_prompt=follow_up_prompt,
            compare_to=compare_to,
            user_answer=user_answer,
        )

        result = await self.llm.completion(prompt, llm_config, user_identity=user_identity)
        raise_if_policy_blocked_result(result)
        if result.get("success") and str(result.get("analysis") or "").strip():
            explanation = str(result["analysis"]).strip()
        else:
            logger.warning("Tutor explanation generation failed for action %s", action)
            explanation = FAILURE_MESSAGE

        self.record_explain_event(user_id, question_id, action)
        used += 1

        response: dict[str, Any] = {
            "question_id": question_id,
            "action": action,
            "explanation": explanation,
            "refusal": False,
            "citations": [{"section": section, "quote": quote, "line": citation_line}],
            "explains_used": used,
            "explains_remaining": max(0, TUTOR_COST_CAP_PER_QUESTION - used),
        }

        misconception = self._classify_user_answer(
            question_text=question_text,
            answer_context=answer_context,
            user_answer=user_answer,
        )
        if misconception is not None:
            response["misconception"] = misconception
            response["remediation"] = self._remediation(
                question_text=question_text, misconception=misconception
            )
        return response

    def _classify_user_answer(
        self, *, question_text: str, answer_context: str, user_answer: str
    ) -> dict[str, str] | None:
        if not str(user_answer or "").strip():
            return None
        try:
            result = self.classifier(
                question=str(question_text or ""),
                correct_answer=str(answer_context or ""),
                user_answer=str(user_answer or ""),
            )
        except Exception:
            logger.warning("Misconception classification failed", exc_info=True)
            return None
        if not result or result.get("error_type") in (None, ERROR_TYPE_NO_ANSWER):
            return result or None
        return {
            "error_type": str(result.get("error_type", "unclassified")),
            "explanation": str(result.get("explanation", "")),
            "source": str(result.get("source", "rule")),
        }

    def _remediation(
        self, *, question_text: str, misconception: dict[str, str]
    ) -> dict[str, Any]:
        objective = derive_objective(
            str(question_text or ""), fallback="Review the core concept"
        )
        return {
            "objectives": [f"Review: {objective}"],
            "flashcard_suggestion": (
                f"Create a flashcard that tests the misconception "
                f"({misconception['error_type']}): {objective}"
            ),
        }

    def _build_prompt(
        self,
        *,
        action: str,
        question_text: str,
        answer_context: str,
        section: str,
        quote: str,
        follow_up_prompt: str,
        compare_to: str,
        user_answer: str,
    ) -> str:
        parts = [
            TUTOR_SYSTEM_PROMPT,
            "",
            f"Action: {_ACTION_INSTRUCTIONS[action]}",
            "",
            _fence("Question", question_text, MAX_QUESTION_CHARS),
            _fence("Canonical answer context", answer_context, MAX_CONTEXT_CHARS),
        ]
        citation_lines: list[str] = []
        if section:
            citation_lines.append(f"Source section: {section}")
        if quote:
            citation_lines.append(f"Verbatim quote from the context: {quote}")
        if citation_lines:
            parts.append(_fence("Context source", "\n".join(citation_lines), MAX_SECTION_CHARS + MAX_QUOTE_CHARS))
        else:
            parts.append(
                _fence(
                    "Context source",
                    "This context carries no section or quote; say you cite none instead of inventing a reference.",
                    MAX_SECTION_CHARS + MAX_QUOTE_CHARS,
                )
            )
        if action == "follow_up" and str(follow_up_prompt or "").strip():
            parts.append(_fence("Learner follow-up", follow_up_prompt, MAX_FOLLOW_UP_CHARS))
        if action == "compare" and str(compare_to or "").strip():
            parts.append(_fence("Alternative to compare against", compare_to, MAX_COMPARE_TO_CHARS))
        if str(user_answer or "").strip():
            parts.append(_fence("Learner answer (diagnose, never copy)", user_answer, MAX_QUESTION_CHARS))
        parts.append("")
        parts.append(
            "Answer as one markdown response. Cite the context source above "
            "when you rely on it. Do not modify or restate the canonical "
            "answer as if it were yours to change."
        )
        return "\n".join(parts)
