"""Four-level hint service with per-question cost capping.

Levels:
    1. gentle nudge — one sentence, never reveals the answer
    2. direction — points at the right concept, no solution
    3. partial solution — key insight or first step
    4. full walkthrough — complete answer with reasoning

Cost cap: at most :data:`HINT_COST_CAP_PER_QUESTION` generated
hints per ``(user_id, question_id)``, enforced via the shared
``feature_events`` table (feature key ``interview_hints``).

In interviews, using a hint lowers the rubric's
``independent_reasoning`` sub-score by a level-dependent
penalty (see :func:`independent_reasoning_score`).
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from datetime import UTC, datetime
from typing import Any

from app.services.llm_client import LLMClient
from app.services.llm_policy import raise_if_policy_blocked_result

logger = logging.getLogger(__name__)

HINT_LEVELS = (1, 2, 3, 4)
HINT_COST_CAP_PER_QUESTION = 4
HINT_FEATURE_KEY = "interview_hints"
HINT_EVENT_NAME = "hint_generated"

#: Penalty applied to ``independent_reasoning`` per hint level.
HINT_PENALTIES: dict[int, int] = {1: 1, 2: 1, 3: 2, 4: 3}

#: Fallback nudges for levels 1-2. They are static and therefore
#: can never contain answer content.
_FALLBACK_HINTS: dict[int, str] = {
    1: (
        "Consider what the question is really asking for, and which "
        "single concept connects the given inputs to the expected output."
    ),
    2: (
        "Name the core concept that links the question's inputs to its "
        "outputs first; the right approach usually follows from that term."
    ),
}

#: Tokens too common to count as answer-specific evidence.
_COMMON_TOKENS = frozenset(
    {
        "about", "after", "answer", "because", "before", "could",
        "example", "following", "given", "great", "question", "should",
        "since", "their", "there", "these", "thing", "think", "using",
        "water", "which", "while", "would", "your", "yours",
    }
)


class HintCostExceededError(Exception):
    """The per-question hint budget is exhausted."""

    def __init__(self, question_id: str, used: int, cap: int):
        super().__init__(
            f"Hint budget exhausted for question {question_id}: "
            f"{used}/{cap} hints used."
        )
        self.question_id = question_id
        self.used = used
        self.cap = cap


def independent_reasoning_score(reasoning_depth: int, hint_level: int) -> int:
    """Score independent reasoning given a hint level.

    Without hints the candidate reasoned on their own, so the
    sub-score equals their reasoning depth. Each hint level
    applies a penalty; level 4 (full walkthrough) hurts most.
    """
    depth = max(0, min(5, int(reasoning_depth)))
    level = int(hint_level or 0)
    if level <= 0:
        return depth
    penalty = HINT_PENALTIES.get(level, 0)
    return max(0, depth - penalty)


def apply_hint_penalty(rubric: dict[str, Any], hint_level: int) -> dict[str, Any]:
    """Return a copy of ``rubric`` with ``independent_reasoning`` set."""
    out = dict(rubric)
    reasoning_depth = out.get("reasoning_depth")
    depth = int(reasoning_depth) if isinstance(reasoning_depth, (int, float)) else 3
    out["independent_reasoning"] = independent_reasoning_score(depth, hint_level)
    return out


def _distinctive_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]{5,}", str(text or "").lower())
    return {t for t in tokens if t not in _COMMON_TOKENS}


class HintService:
    """Generate cost-capped, level-aware hints for questions."""

    def __init__(self, llm: LLMClient, db_path: str):
        self.llm = llm
        self.db_path = db_path
        # Dedicated read-only-ish connection for counting hint
        # events. Writes go through the same table; SQLite
        # serialises them across connections.
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

    def count_hints_used(self, user_id: str, question_id: str) -> int:
        """Hints already generated for this user+question."""
        with self._lock:
            try:
                row = self._conn.execute(
                    """
                    SELECT COUNT(*) AS total
                    FROM feature_events
                    WHERE user_id = ? AND feature_key = ? AND event_name = ?
                      AND json_extract(metadata_json, '$.question_id') = ?
                    """,
                    (user_id, HINT_FEATURE_KEY, HINT_EVENT_NAME, question_id),
                ).fetchone()
                return int(row["total"]) if row else 0
            except sqlite3.Error:
                # json_extract unavailable or malformed metadata:
                # fall back to a LIKE scan.
                rows = self._conn.execute(
                    """
                    SELECT metadata_json FROM feature_events
                    WHERE user_id = ? AND feature_key = ? AND event_name = ?
                    """,
                    (user_id, HINT_FEATURE_KEY, HINT_EVENT_NAME),
                ).fetchall()
                needle = f'"question_id": "{question_id}"'
                return sum(
                    1
                    for r in rows
                    if needle in str(r["metadata_json"] or "")
                )

    def record_hint_event(
        self,
        user_id: str,
        question_id: str,
        level: int,
    ) -> None:
        payload = json.dumps(
            {"question_id": question_id, "level": int(level)},
            ensure_ascii=True,
        )
        now = datetime.now(UTC).isoformat()
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO feature_events(
                        user_id, feature_key, event_name, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (user_id, HINT_FEATURE_KEY, HINT_EVENT_NAME, payload, now),
                )

    async def generate_hint(
        self,
        *,
        user_id: str,
        question_id: str,
        level: int,
        question_text: str = "",
        answer_context: str = "",
        llm_config: Any = None,
        user_identity: dict | None = None,
    ) -> dict[str, Any]:
        level = int(level)
        if level not in HINT_LEVELS:
            raise ValueError(f"Hint level must be one of {HINT_LEVELS}")

        used = self.count_hints_used(user_id, question_id)
        if used >= HINT_COST_CAP_PER_QUESTION:
            raise HintCostExceededError(question_id, used, HINT_COST_CAP_PER_QUESTION)

        hint = await self._generate(
            level=level,
            question_text=question_text,
            answer_context=answer_context,
            llm_config=llm_config,
            user_identity=user_identity,
        )

        if level <= 2 and self._leakage_detected(
            hint,
            question_text=question_text,
            answer_context=answer_context,
        ):
            logger.warning(
                "Hint level %s leaked answer content; using fallback nudge",
                level,
            )
            hint = _FALLBACK_HINTS[level]

        self.record_hint_event(user_id, question_id, level)
        used += 1
        return {
            "question_id": question_id,
            "level": level,
            "hint": hint,
            "hints_used": used,
            "hints_remaining": max(0, HINT_COST_CAP_PER_QUESTION - used),
        }

    async def _generate(
        self,
        *,
        level: int,
        question_text: str,
        answer_context: str,
        llm_config: Any = None,
        user_identity: dict | None = None,
    ) -> str:
        prompt = self._build_prompt(
            level=level,
            question_text=question_text,
            answer_context=answer_context,
        )
        result = await self.llm.completion(
            prompt,
            llm_config,
            user_identity=user_identity,
        )
        # Policy blocks (approval required, personal credential
        # required, study-app not assigned) must propagate to the
        # app-level handlers; only generation failures fall back.
        raise_if_policy_blocked_result(result)
        if result.get("success"):
            hint = str(result.get("analysis") or "").strip()
            if hint:
                return hint[:600]
        return self._fallback_hint(level)

    @staticmethod
    def _fallback_hint(level: int) -> str:
        if level in _FALLBACK_HINTS:
            return _FALLBACK_HINTS[level]
        if level == 3:
            return (
                "Start by naming the key constraint or invariant in the "
                "question, then outline the first concrete step you would "
                "take and why it follows from that constraint."
            )
        return (
            "Walk through the problem step by step: restate the goal and "
            "constraints, name the core concept, apply it to the given "
            "inputs, and finish with the complete answer and why it holds."
        )

    @staticmethod
    def _build_prompt(*, level: int, question_text: str, answer_context: str) -> str:
        rules = {
            1: (
                "- Give ONE short sentence (max 25 words) that gently nudges "
                "the candidate toward the right direction.\n"
                "- NEVER reveal the answer, the solution, or any part of it. "
                "Do not name the correct option, the final result, or any "
                "concrete step of the solution.\n"
                "- No markdown, no preamble."
            ),
            2: (
                "- Point toward the right concept or approach (max 40 words) "
                "without giving the solution or any concrete steps.\n"
                "- NEVER reveal the answer, the solution, or any part of it. "
                "Do not name the correct option, the final result, or any "
                "concrete step of the solution.\n"
                "- No markdown, no preamble."
            ),
            3: (
                "- Give a PARTIAL solution: the key insight or the first "
                "concrete step (max 90 words). Do not give the complete "
                "answer.\n"
                "- No markdown, no preamble."
            ),
            4: (
                "- Give the COMPLETE walkthrough with the full answer "
                "(max 250 words), including the reasoning and the final "
                "result.\n"
                "- No markdown, no preamble."
            ),
        }[level]

        answer_block = ""
        if level >= 3 and answer_context:
            answer_block = (
                "\nReference answer material (use it to build the "
                f"walkthrough, do not quote it verbatim):\n{answer_context[:4000]}\n"
            )

        return f"""You are a study coach giving a level-{level} hint for a practice question.

Question:
{question_text[:8000]}
{answer_block}
Hint rules for level {level}:
{rules}
"""

    @staticmethod
    def _leakage_detected(
        hint: str,
        *,
        question_text: str,
        answer_context: str,
    ) -> bool:
        """Heuristic leak guard for levels 1-2.

        A hint leaks when it contains two or more distinctive
        tokens that appear in the answer material but not in the
        question — i.e. content only the answer could supply.
        """
        if not hint:
            return False
        answer_tokens = _distinctive_tokens(answer_context)
        question_tokens = _distinctive_tokens(question_text)
        answer_only = answer_tokens - question_tokens
        if not answer_only:
            return False
        hint_tokens = _distinctive_tokens(hint)
        return len(hint_tokens & answer_only) >= 2
