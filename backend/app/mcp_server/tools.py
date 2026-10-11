"""Read-only MCP tools over one learner's study data.

Every tool is scoped to the single user id configured for the server
(``mcp_server_user_id``) — this is a **single-user local bridge**, not a
multi-tenant surface, so no tool accepts a user id from the client and the
scoping cannot be widened from the wire.

All tools are read-only in the agent-facing sense: they only call existing
store readers (``get_fsrs_review_queue``, ``get_weak_areas``,
``build_recommendations``, ``get_topic_mastery``, ``build_study_plan``,
``get_stats``) and never record attempts, reviews, or sessions.
``build_recommendations`` persisting a recomputed competency snapshot is the
stores' own bookkeeping, identical to what the REST endpoint does on the same
call.

The review queue is the FSRS queue (``get_fsrs_review_queue``), the same
reader the REST ``/api/learning/review-queue`` endpoint uses by default, so
both surfaces always agree. No one-time FSRS backfill runs here: it is
store maintenance the REST lifespan performs once, not learner-authored
data, and a per-session MCP process should not silently migrate tables.

Empty / missing data is a normal answer, not an error: each tool returns its
store's empty shape (``items: []``, ``topics: []``, zero counters) and the
planner's cold-start fallback supplies starter objectives for a brand-new
learner.
"""

from __future__ import annotations

import logging
from typing import Any

from app.mcp_server.protocol import text_content
from app.services.interview_store import InterviewStore
from app.services.learning_planner import LearningPlannerStore
from app.services.learning_store import LearningStore

logger = logging.getLogger(__name__)

#: Matches the REST router's bounds: /api/learning/review-queue (1..200).
_LIMIT_MIN = 1
_LIMIT_MAX = 200
_LIMIT_DEFAULT = 50

#: Matches the stores' own clamps: study plans span 1..31 days.
_DAYS_MIN = 1
_DAYS_MAX = 31
_DAYS_DEFAULT = 7


class UnknownToolError(KeyError):
    """The client called a tool name that does not exist."""


class InvalidToolParams(ValueError):
    """The client sent malformed tool arguments."""


def _coerce_int(
    arguments: dict[str, Any],
    key: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    """Read an optional integer argument, clamped into the store's bounds."""
    value = arguments.get(key, default)
    if value is None:
        return default
    # JSON booleans are ints in Python; reject them explicitly.
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidToolParams(f"{key!r} must be an integer")
    return max(minimum, min(maximum, value))


class MCPToolkit:
    """The read-only tool catalog exposed to MCP clients."""

    def __init__(
        self,
        *,
        learning_store: LearningStore,
        learning_planner: LearningPlannerStore,
        interview_store: InterviewStore,
        user_id: str,
    ) -> None:
        self._learning_store = learning_store
        self._learning_planner = learning_planner
        self._interview_store = interview_store
        self._user_id = user_id

    # ── catalog ─────────────────────────────────────────────────

    def definitions(self) -> list[dict[str, Any]]:
        """MCP ``tools/list`` entries: name, description, JSON Schema."""
        return [
            {
                "name": "get_review_queue",
                "description": (
                    "The learner's due review queue: FSRS cards whose "
                    "next-review time has passed, ordered by due date."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "minimum": _LIMIT_MIN,
                            "maximum": _LIMIT_MAX,
                            "default": _LIMIT_DEFAULT,
                            "description": "Maximum number of due items to return.",
                        }
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "get_weak_areas",
                "description": (
                    "The learner's weakest topics ranked by mastery, accuracy "
                    "and confidence, each with attempts and a due count."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
            {
                "name": "get_next_objectives",
                "description": (
                    "The learner's next best objectives: review, study, quiz "
                    "and mock-interview actions ranked by priority, each with a "
                    "reason, estimated minutes and an in-app route. Falls back "
                    "to cold-start objectives for a new learner."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
            {
                "name": "get_mastery",
                "description": (
                    "Per-topic mastery scores (0..1) with attempt counts for "
                    "every topic the learner has practised."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
            {
                "name": "get_study_plan",
                "description": (
                    "A day-by-day study plan for the learner, built from their "
                    "next objectives and preferred session length."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "minimum": _DAYS_MIN,
                            "maximum": _DAYS_MAX,
                            "default": _DAYS_DEFAULT,
                            "description": "Number of days the plan should span.",
                        }
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "get_interview_stats",
                "description": (
                    "Mock-interview statistics: total and completed sessions "
                    "plus the average interview-readiness score."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        ]

    # ── dispatch ────────────────────────────────────────────────

    def execute(self, name: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        """Run one tool and return an MCP content array (JSON text payload)."""
        handlers = {
            "get_review_queue": self._get_review_queue,
            "get_weak_areas": self._get_weak_areas,
            "get_next_objectives": self._get_next_objectives,
            "get_mastery": self._get_mastery,
            "get_study_plan": self._get_study_plan,
            "get_interview_stats": self._get_interview_stats,
        }
        handler = handlers.get(name)
        if handler is None:
            raise UnknownToolError(name)
        logger.debug("MCP tool %s for user %s", name, self._user_id)
        payload = handler(arguments)
        return [text_content(payload)]

    # ── tool implementations ────────────────────────────────────

    def _get_review_queue(self, arguments: dict[str, Any]) -> dict[str, Any]:
        limit = _coerce_int(
            arguments,
            "limit",
            default=_LIMIT_DEFAULT,
            minimum=_LIMIT_MIN,
            maximum=_LIMIT_MAX,
        )
        return self._learning_store.get_fsrs_review_queue(
            user_id=self._user_id, limit=limit
        )

    def _get_weak_areas(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._learning_store.get_weak_areas(user_id=self._user_id)

    def _get_next_objectives(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._learning_planner.build_recommendations(user_id=self._user_id)

    def _get_mastery(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._learning_store.get_topic_mastery(user_id=self._user_id)

    def _get_study_plan(self, arguments: dict[str, Any]) -> dict[str, Any]:
        days = _coerce_int(
            arguments,
            "days",
            default=_DAYS_DEFAULT,
            minimum=_DAYS_MIN,
            maximum=_DAYS_MAX,
        )
        return self._learning_planner.build_study_plan(user_id=self._user_id, days=days)

    def _get_interview_stats(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._interview_store.get_stats(user_id=self._user_id)
