"""SQLite-backed adaptive learning persistence and scheduling."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import Any, Callable, Mapping, TypeVar
from uuid import uuid4

from app.services import fsrs_scheduler
from app.services.topic_catalog import (
    PROBLEM_SOLVING_DEFAULT_LANGUAGE,
    PROBLEM_SOLVING_LANGUAGE_OPTIONS,
    PROBLEM_SOLVING_LEVELS,
    PROBLEM_SOLVING_TARGET_SECTIONS,
    PROBLEM_SOLVING_TOPIC_ID,
    PROBLEM_SOLVING_TRACK,
    PROBLEM_SOLVING_TITLE,
    is_problem_solving_topic,
)

T = TypeVar("T")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _to_iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


def _from_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


# Beta-smoothed standard error of a Bernoulli accuracy estimate,
# normalised so the no-evidence prior (alpha=beta=1) yields exactly
# 1.0. Evidence shrinks the value toward 0; a 50/50 split keeps it
# highest for a given attempt count.
_UNCERTAINTY_PRIOR_ALPHA = 1.0
_UNCERTAINTY_PRIOR_BETA = 1.0
_UNCERTAINTY_COLD_START_VARIANCE = (
    _UNCERTAINTY_PRIOR_ALPHA
    * _UNCERTAINTY_PRIOR_BETA
    / (
        (_UNCERTAINTY_PRIOR_ALPHA + _UNCERTAINTY_PRIOR_BETA) ** 2
        * (_UNCERTAINTY_PRIOR_ALPHA + _UNCERTAINTY_PRIOR_BETA + 1)
    )
)


def accuracy_uncertainty(
    correct_attempts: int, attempts: int
) -> float:
    """Standard error of a Beta-smoothed accuracy estimate, in [0, 1]."""
    alpha = _UNCERTAINTY_PRIOR_ALPHA + max(0, int(correct_attempts))
    beta = _UNCERTAINTY_PRIOR_BETA + max(0, int(attempts) - int(correct_attempts))
    total = alpha + beta
    variance = (alpha * beta) / (total * total * (total + 1))
    return min(
        1.0,
        max(0.0, (variance / _UNCERTAINTY_COLD_START_VARIANCE) ** 0.5),
    )


# Legacy bucket-engine review intervals, retained only by
# backfill_fsrs_from_progress to carry the retired
# question_progress schedule onto migrated fsrs_cards rows.
_LEGACY_BUCKET_INTERVAL_DAYS = (0, 1, 3, 7, 14, 30)


logger = logging.getLogger(__name__)


async def record_generation_run_safe(
    run_store: Any,
    *,
    user_identity: dict | None,
    metadata: Mapping[str, Any],
    flow: str,
    prompt_name: str = "",
    prompt_version: str = "",
    attempts: int = 1,
    terminal_reason: str = "",
    finish_reason: str = "",
    error_code: str = "",
) -> None:
    """Persist one generation-run row, never failing the generation.

    Extracts the row fields from a completion result's metadata and
    the caller's retry outcome, then writes through the store's
    async executor when it has one. Telemetry is best-effort: any
    failure is logged and swallowed.
    """
    if run_store is None:
        return
    record = getattr(run_store, "record_generation_run", None)
    if record is None:
        return
    row = {
        "run_id": str(uuid4()),
        "user_id": str((user_identity or {}).get("user", "") or ""),
        "flow": str(flow or ""),
        "provider": str(metadata.get("provider", "") or ""),
        "model": str(metadata.get("model", "") or ""),
        "prompt_name": str(prompt_name or ""),
        "prompt_version": str(prompt_version or ""),
        "attempts": int(attempts or 1),
        "terminal_reason": str(terminal_reason or ""),
        "input_tokens": int(metadata.get("input_tokens", 0) or 0),
        "output_tokens": int(metadata.get("output_tokens", 0) or 0),
        "finish_reason": str(finish_reason or ""),
        "error_code": str(error_code or ""),
    }
    # A single SQLite insert is microseconds, so it is
    # written inline rather than on the store's async
    # executor. Awaiting an executor future would suspend
    # the generation task on a Future; a background task
    # created on a short-lived event loop (for example a
    # per-request test portal) is then orphaned when the
    # loop closes before the future resolves, and a
    # fire-and-forget thread can outlive a test's tempdir
    # cleanup. Writing inline keeps the generation task
    # in a single loop step and the row on disk before
    # the caller returns. Telemetry is best-effort: a
    # failed write is logged and swallowed.
    try:
        record(**row)
    except Exception:
        logger.warning(
            "generation_run persist failed flow=%s", flow, exc_info=True
        )


class LearningStore:
    """Persist learning attempts and maintain per-question review state."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="learning-store")
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    async def run_async(self, fn: Callable[..., T], /, *args, **kwargs) -> T:
        """Run synchronous SQLite work on a dedicated worker thread."""
        loop = asyncio.get_running_loop()
        if kwargs:
            return await loop.run_in_executor(
                self._executor,
                lambda: fn(*args, **kwargs),
            )
        return await loop.run_in_executor(self._executor, fn, *args)

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS learning_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    question_id TEXT NOT NULL,
                    topic_id TEXT NOT NULL,
                    user_answer TEXT NOT NULL,
                    is_correct INTEGER NOT NULL,
                    confidence INTEGER NOT NULL,
                    response_time_ms INTEGER NOT NULL,
                    mode TEXT NOT NULL,
                    learning_objective TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
                """
            )
            existing_columns = {
                str(row["name"])
                for row in self._conn.execute(
                    "PRAGMA table_info(learning_attempts)"
                ).fetchall()
            }
            if "learning_objective" not in existing_columns:
                self._conn.execute(
                    "ALTER TABLE learning_attempts "
                    "ADD COLUMN learning_objective TEXT NOT NULL DEFAULT ''"
                )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS question_progress (
                    user_id TEXT NOT NULL,
                    question_id TEXT NOT NULL,
                    topic_id TEXT NOT NULL,
                    mastery_score REAL NOT NULL,
                    due_at TEXT NOT NULL,
                    review_bucket INTEGER NOT NULL,
                    attempts INTEGER NOT NULL,
                    correct_attempts INTEGER NOT NULL,
                    avg_confidence REAL NOT NULL,
                    last_confidence INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, question_id, topic_id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_progress_due
                ON question_progress(user_id, due_at)
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_progress_topic
                ON question_progress(user_id, topic_id)
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS custom_topics (
                    user_id TEXT NOT NULL,
                    topic_id TEXT NOT NULL,
                    source_topic TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    track TEXT NOT NULL,
                    levels_json TEXT NOT NULL,
                    sections_json TEXT NOT NULL,
                    raw_content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, topic_id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_custom_topics_user_updated
                ON custom_topics(user_id, updated_at DESC)
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS topic_user_preferences (
                    user_id TEXT NOT NULL,
                    topic_id TEXT NOT NULL,
                    response_detail TEXT NOT NULL,
                    preferred_language TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, topic_id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS topic_language_profiles (
                    topic_id TEXT PRIMARY KEY,
                    requires_programming INTEGER NOT NULL,
                    language_options_json TEXT NOT NULL,
                    source TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_topic_language_profiles_updated
                ON topic_language_profiles(updated_at DESC)
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dynamic_topic_curricula (
                    user_id TEXT NOT NULL,
                    topic_id TEXT NOT NULL,
                    preferred_language TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    track TEXT NOT NULL,
                    levels_json TEXT NOT NULL,
                    sections_json TEXT NOT NULL,
                    raw_content TEXT NOT NULL,
                    target_sections INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, topic_id, preferred_language)
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_dynamic_topic_curricula_user_updated
                ON dynamic_topic_curricula(user_id, topic_id, updated_at DESC)
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS assistant_memory (
                    user_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    flow TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, conversation_id, flow)
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_assistant_memory_updated
                ON assistant_memory(user_id, flow, updated_at DESC)
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feature_runtime_state (
                    feature_key TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    disabled_until TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS topic_video_cache (
                    cache_key TEXT PRIMARY KEY,
                    topic_id TEXT NOT NULL,
                    section_heading TEXT NOT NULL,
                    preferred_language TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    fetched_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_topic_video_cache_topic
                ON topic_video_cache(topic_id, fetched_at DESC)
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feature_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    feature_key TEXT NOT NULL,
                    event_name TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_feature_events_feature_time
                ON feature_events(feature_key, event_name, created_at DESC)
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fsrs_cards (
                    user_id TEXT NOT NULL,
                    card_id TEXT NOT NULL,
                    topic_id TEXT NOT NULL,
                    source_type TEXT NOT NULL CHECK(source_type IN ('question','flashcard','interview','exam')),
                    state TEXT NOT NULL DEFAULT 'new' CHECK(state IN ('new','learning','review','relearning')),
                    stability REAL, difficulty REAL,
                    due_at TEXT NOT NULL,
                    last_review_at TEXT,
                    reps INTEGER NOT NULL DEFAULT 0,
                    lapses INTEGER NOT NULL DEFAULT 0,
                    scheduled_days INTEGER NOT NULL DEFAULT 0,
                    elapsed_days INTEGER NOT NULL DEFAULT 0,
                    suspended INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, card_id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_fsrs_due
                ON fsrs_cards(user_id, due_at) WHERE suspended = 0
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fsrs_review_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL, card_id TEXT NOT NULL,
                    rating TEXT NOT NULL CHECK(rating IN ('again','hard','good','easy')),
                    state TEXT NOT NULL, review_duration_ms INTEGER NOT NULL,
                    scheduled_days INTEGER, elapsed_days INTEGER,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_fsrs_review_log_user_time
                ON fsrs_review_log(user_id, created_at DESC)
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS decks (
                    user_id TEXT NOT NULL, deck_id TEXT NOT NULL,
                    name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, deck_id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cards (
                    user_id TEXT NOT NULL, card_id TEXT NOT NULL,
                    deck_id TEXT NOT NULL, front TEXT NOT NULL, back TEXT NOT NULL,
                    source_ref TEXT NOT NULL DEFAULT '',
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    fsrs_card_id TEXT NOT NULL,
                    suspended INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, card_id)
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cards_deck ON cards(user_id, deck_id)"
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS generation_runs (
                    run_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    flow TEXT NOT NULL,
                    provider TEXT NOT NULL DEFAULT '',
                    model TEXT NOT NULL DEFAULT '',
                    prompt_name TEXT NOT NULL DEFAULT '',
                    prompt_version TEXT NOT NULL DEFAULT '',
                    attempts INTEGER NOT NULL DEFAULT 1,
                    terminal_reason TEXT NOT NULL DEFAULT '',
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    finish_reason TEXT NOT NULL DEFAULT '',
                    error_code TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_generation_runs_user_time
                ON generation_runs(user_id, flow, created_at DESC)
                """
            )

    def record_generation_run(
        self,
        *,
        run_id: str,
        user_id: str,
        flow: str,
        provider: str = "",
        model: str = "",
        prompt_name: str = "",
        prompt_version: str = "",
        attempts: int = 1,
        terminal_reason: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        finish_reason: str = "",
        error_code: str = "",
        created_at: str = "",
    ) -> None:
        """Persist one generation-run telemetry row.

        One row per completed generation flow (a full
        ``_collect_with_retries`` cycle or an interview answer
        evaluation), carrying the provider/model/prompt identity,
        the retry outcome, and the token/finish-reason summary of
        the final call. Telemetry only: a failed write must never
        fail the generation itself.
        """
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO generation_runs(
                        run_id, user_id, flow, provider, model, prompt_name,
                        prompt_version, attempts, terminal_reason, input_tokens,
                        output_tokens, finish_reason, error_code, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(run_id),
                        str(user_id),
                        str(flow),
                        str(provider or ""),
                        str(model or ""),
                        str(prompt_name or ""),
                        str(prompt_version or ""),
                        max(1, int(attempts or 1)),
                        str(terminal_reason or ""),
                        max(0, int(input_tokens or 0)),
                        max(0, int(output_tokens or 0)),
                        str(finish_reason or ""),
                        str(error_code or ""),
                        str(created_at or _to_iso(_utc_now())),
                    ),
                )

    def get_generation_runs(
        self,
        *,
        user_id: str,
        flow: str = "",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return a user's generation-run rows, newest first."""
        query = (
            "SELECT run_id, user_id, flow, provider, model, prompt_name, "
            "prompt_version, attempts, terminal_reason, input_tokens, "
            "output_tokens, finish_reason, error_code, created_at "
            "FROM generation_runs WHERE user_id = ?"
        )
        params: list[Any] = [str(user_id)]
        if flow:
            query += " AND flow = ?"
            params.append(str(flow))
        query += " ORDER BY created_at DESC, run_id DESC LIMIT ?"
        params.append(max(1, int(limit or 100)))
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def record_attempt(
        self,
        *,
        user_id: str,
        question_id: str,
        topic_id: str,
        user_answer: str,
        is_correct: bool,
        confidence: int,
        response_time_ms: int,
        mode: str,
        learning_objective: str = "",
    ) -> dict:
        """Record a quiz attempt or interview turn.

        The attempt is logged to ``learning_attempts`` and
        routed onto the FSRS scheduler: the self-reported
        ``(is_correct, confidence)`` pair is mapped onto an
        FSRS rating (see :func:`fsrs_scheduler.derive_rating`)
        and the question's card is upserted, so quiz mistakes
        and interview results drive the same scheduler as
        flashcards and the manual review queue. The legacy
        ``question_progress`` bucket engine is retired; the
        table is kept read-only for a soak period.
        """
        now = _utc_now()
        now_iso = _to_iso(now)
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO learning_attempts(
                        user_id, question_id, topic_id, user_answer, is_correct,
                        confidence, response_time_ms, mode, learning_objective,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        question_id,
                        topic_id,
                        user_answer,
                        1 if is_correct else 0,
                        confidence,
                        response_time_ms,
                        mode,
                        learning_objective,
                        now_iso,
                    ),
                )
                attempt_id = int(
                    self._conn.execute(
                        "SELECT last_insert_rowid()"
                    ).fetchone()[0]
                )
                rating = fsrs_scheduler.derive_rating(
                    is_correct, confidence
                )
                new_card, _log_entry, _log_id = self._apply_fsrs_review(
                    user_id=user_id,
                    card_id=question_id,
                    topic_id=topic_id,
                    source_type=self._fsrs_source_type(
                        mode, question_id
                    ),
                    rating=rating,
                    now=now,
                    now_iso=now_iso,
                    response_time_ms=response_time_ms,
                )

        return {
            "attempt_id": attempt_id,
            "topic_id": topic_id,
            "question_id": question_id,
            "state": new_card["state"],
            "due_at": new_card["due_at"],
            "lapses": new_card["lapses"],
            "leech": fsrs_scheduler.is_leech(new_card),
        }

    def select_next_objectives(
        self,
        *,
        user_id: str,
        topic_ids: list[str],
        limit: int = 5,
    ) -> list[dict]:
        """Rank learning objectives for the next quiz.

        Candidate objectives are the ``(topic_id, learning_objective)``
        pairs observed in ``learning_attempts``. Each candidate's
        priority is ``(1 - accuracy) * uncertainty``: objectives the
        learner got wrong with little evidence outrank objectives they
        have mastered or barely touched. Ties break toward higher
        uncertainty, then fewer attempts (cold start first), then
        stable identifiers for deterministic ordering.
        """
        if not topic_ids:
            return []
        placeholders = ", ".join("?" for _ in topic_ids)
        rows = self._conn.execute(
            f"""
            SELECT topic_id, learning_objective,
                   COUNT(*) AS attempts,
                   SUM(is_correct) AS correct_attempts
            FROM learning_attempts
            WHERE user_id = ?
              AND topic_id IN ({placeholders})
              AND learning_objective != ''
            GROUP BY topic_id, learning_objective
            """,
            (user_id, *topic_ids),
        ).fetchall()
        candidates: list[dict] = []
        for row in rows:
            attempts = int(row["attempts"])
            correct = int(row["correct_attempts"] or 0)
            accuracy = correct / attempts if attempts else 0.0
            uncertainty = accuracy_uncertainty(correct, attempts)
            candidates.append(
                {
                    "topic_id": str(row["topic_id"]),
                    "learning_objective": str(row["learning_objective"]),
                    "attempts": attempts,
                    "correct_attempts": correct,
                    "accuracy": accuracy,
                    "uncertainty": uncertainty,
                    "priority": (1.0 - accuracy) * uncertainty,
                }
            )
        candidates.sort(
            key=lambda item: (
                -item["priority"],
                -item["uncertainty"],
                item["attempts"],
                item["topic_id"],
                item["learning_objective"],
            )
        )
        return candidates[: max(0, int(limit))]

    # ── FSRS spaced repetition (STUDY_ENABLE_FSRS_V1) ────────────

    _FSRS_CARD_COLUMNS = (
        "user_id, card_id, topic_id, source_type, state, stability, "
        "difficulty, due_at, last_review_at, reps, lapses, "
        "scheduled_days, elapsed_days, suspended, created_at, updated_at"
    )

    @staticmethod
    def _fsrs_row_to_dict(row: sqlite3.Row) -> dict:
        return {
            "card_id": str(row["card_id"]),
            "topic_id": str(row["topic_id"]),
            "source_type": str(row["source_type"]),
            "state": str(row["state"]),
            "stability": (
                float(row["stability"])
                if row["stability"] is not None
                else None
            ),
            "difficulty": (
                float(row["difficulty"])
                if row["difficulty"] is not None
                else None
            ),
            "due_at": str(row["due_at"]),
            "last_review_at": (
                str(row["last_review_at"])
                if row["last_review_at"] is not None
                else None
            ),
            "reps": int(row["reps"]),
            "lapses": int(row["lapses"]),
            "scheduled_days": int(row["scheduled_days"]),
            "elapsed_days": int(row["elapsed_days"]),
            "suspended": int(row["suspended"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    @staticmethod
    def _fsrs_source_type(mode: str, question_id: str) -> str:
        """Map a learning mode to an ``fsrs_cards.source_type``.

        The ``fsrs_cards.source_type`` CHECK allows
        ``question``/``flashcard``/``interview``/``exam``. The
        ``LearningMode`` values are ``study``/``quiz``, so both map to
        ``question``. Interview turns are recorded with
        ``interview:``-prefixed question ids and map to ``interview``.
        """
        if str(question_id).startswith("interview:"):
            return "interview"
        return "question"

    def _apply_fsrs_review(
        self,
        *,
        user_id: str,
        card_id: str,
        topic_id: str,
        source_type: str,
        rating: str,
        now: datetime,
        now_iso: str,
        response_time_ms: int,
    ) -> tuple[dict, dict, int]:
        """Apply an FSRS review and persist the card + log rows.

        Caller must hold ``self._lock`` and an open transaction. When the
        card already exists its ``topic_id``/``source_type`` are preserved
        (a card's topic and source are immutable once created). Returns
        ``(new_card, log_entry, log_id)``.
        """
        existing = self._conn.execute(
            f"""
            SELECT {self._FSRS_CARD_COLUMNS}
            FROM fsrs_cards
            WHERE user_id = ? AND card_id = ?
            """,
            (user_id, card_id),
        ).fetchone()
        if existing:
            card_data = self._fsrs_row_to_dict(existing)
            topic_id = str(existing["topic_id"])
            source_type = str(existing["source_type"])
            suspended = int(existing["suspended"])
            created_at = str(existing["created_at"])
        else:
            card_data = fsrs_scheduler.create_card(now)
            suspended = 0
            created_at = now_iso

        new_card, log_entry = fsrs_scheduler.review(
            card_data,
            rating,
            now,
            response_time_ms,
        )
        new_card["card_id"] = card_id
        new_card["topic_id"] = topic_id
        new_card["source_type"] = source_type
        new_card["suspended"] = suspended
        new_card["created_at"] = created_at
        new_card["updated_at"] = now_iso

        self._conn.execute(
            """
            INSERT INTO fsrs_cards(
                user_id, card_id, topic_id, source_type, state,
                stability, difficulty, due_at, last_review_at,
                reps, lapses, scheduled_days, elapsed_days,
                suspended, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, card_id) DO UPDATE SET
                topic_id = excluded.topic_id,
                source_type = excluded.source_type,
                state = excluded.state,
                stability = excluded.stability,
                difficulty = excluded.difficulty,
                due_at = excluded.due_at,
                last_review_at = excluded.last_review_at,
                reps = excluded.reps,
                lapses = excluded.lapses,
                scheduled_days = excluded.scheduled_days,
                elapsed_days = excluded.elapsed_days,
                suspended = excluded.suspended,
                updated_at = excluded.updated_at
            """,
            (
                user_id,
                card_id,
                new_card["topic_id"],
                new_card["source_type"],
                new_card["state"],
                new_card["stability"],
                new_card["difficulty"],
                new_card["due_at"],
                new_card["last_review_at"],
                new_card["reps"],
                new_card["lapses"],
                new_card["scheduled_days"],
                new_card["elapsed_days"],
                new_card["suspended"],
                new_card["created_at"],
                new_card["updated_at"],
            ),
        )
        self._conn.execute(
            """
            INSERT INTO fsrs_review_log(
                user_id, card_id, rating, state, review_duration_ms,
                scheduled_days, elapsed_days, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                card_id,
                log_entry["rating"],
                log_entry["state"],
                log_entry["review_duration_ms"],
                log_entry["scheduled_days"],
                log_entry["elapsed_days"],
                log_entry["created_at"],
            ),
        )
        log_id = int(
            self._conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        )
        return new_card, log_entry, log_id

    def record_review(
        self,
        *,
        user_id: str,
        card_id: str,
        topic_id: str,
        rating: str,
        response_time_ms: int,
        source_type: str = "question",
    ) -> dict:
        """Apply an FSRS review to a card and persist card + log row."""
        if rating not in fsrs_scheduler.RATINGS:
            raise ValueError(f"unknown FSRS rating: {rating!r}")
        now = _utc_now()
        now_iso = _to_iso(now)
        with self._lock:
            with self._conn:
                new_card, log_entry, log_id = self._apply_fsrs_review(
                    user_id=user_id,
                    card_id=card_id,
                    topic_id=topic_id,
                    source_type=source_type,
                    rating=rating,
                    now=now,
                    now_iso=now_iso,
                    response_time_ms=response_time_ms,
                )

        log_entry = dict(log_entry)
        log_entry["id"] = log_id
        log_entry["card_id"] = card_id
        return {
            "card": new_card,
            "log": log_entry,
            "leech": fsrs_scheduler.is_leech(new_card),
        }

    def get_due_cards(
        self, *, user_id: str, limit: int = 50
    ) -> list[dict]:
        now_iso = _to_iso(_utc_now())
        rows = self._conn.execute(
            f"""
            SELECT {self._FSRS_CARD_COLUMNS}
            FROM fsrs_cards
            WHERE user_id = ? AND suspended = 0 AND due_at <= ?
            ORDER BY due_at ASC, card_id ASC
            LIMIT ?
            """,
            (user_id, now_iso, limit),
        ).fetchall()
        return [self._fsrs_row_to_dict(row) for row in rows]

    def get_fsrs_review_queue(
        self, *, user_id: str, limit: int = 50
    ) -> dict:
        cards = self.get_due_cards(user_id=user_id, limit=limit)
        items = []
        for card in cards:
            items.append(
                {
                    "topic_id": card["topic_id"],
                    "question_id": card["card_id"],
                    "due_at": card["due_at"],
                    "mastery_score": round(
                        fsrs_scheduler.retention_estimate(card), 4
                    ),
                    "last_confidence": 0,
                    "review_bucket": 0,
                    "attempts": card["reps"],
                    "state": card["state"],
                    "lapses": card["lapses"],
                    "suspended": card["suspended"],
                    "leech": fsrs_scheduler.is_leech(card),
                }
            )
        return {"items": items, "total_due": len(items)}

    def get_forecast(self, *, user_id: str, days: int = 7) -> dict:
        horizon_days = max(1, min(int(days), 31))
        now = _utc_now()
        horizon_iso = _to_iso(now + timedelta(days=horizon_days))
        rows = self._conn.execute(
            """
            SELECT due_at
            FROM fsrs_cards
            WHERE user_id = ? AND suspended = 0 AND due_at <= ?
            ORDER BY due_at ASC
            """,
            (user_id, horizon_iso),
        ).fetchall()
        counts: dict[str, int] = {
            (now + timedelta(days=offset)).date().isoformat(): 0
            for offset in range(horizon_days)
        }
        total_due = 0
        today = now.date()
        for row in rows:
            due_date = _from_iso(str(row["due_at"])).date()
            bucket = max(due_date, today).isoformat()
            if bucket in counts:
                counts[bucket] += 1
                total_due += 1
        due_counts = [
            {"date": date, "count": counts[date]} for date in sorted(counts)
        ]
        return {"due_counts": due_counts, "total_due": total_due}

    def get_calibration(self, *, user_id: str) -> dict:
        rows = self._conn.execute(
            """
            SELECT rating
            FROM fsrs_review_log
            WHERE user_id = ? AND elapsed_days >= 1
            """,
            (user_id,),
        ).fetchall()
        eligible = len(rows)
        successful = 0
        good_count = 0
        for row in rows:
            rating = str(row["rating"])
            if rating in (fsrs_scheduler.RATING_GOOD, fsrs_scheduler.RATING_EASY):
                successful += 1
            if rating == fsrs_scheduler.RATING_GOOD:
                good_count += 1
        true_retention = (successful / eligible) if eligible else 0.0
        good_rating_pct = (good_count / eligible) if eligible else 0.0
        within_band = eligible > 0 and (
            0.8 <= true_retention <= 0.9
        )
        low_signal = eligible < 50 or good_rating_pct > 0.8
        return {
            "eligible_reviews": eligible,
            "successful_reviews": successful,
            "true_retention": round(true_retention, 4),
            "target_band_low": 0.8,
            "target_band_high": 0.9,
            "within_band": within_band,
            "good_rating_pct": round(good_rating_pct, 4),
            "low_signal": low_signal,
        }

    def get_fsrs_topic_mastery(
        self, *, user_id: str, limit: int = 500
    ) -> dict:
        now_iso = _to_iso(_utc_now())
        rows = self._conn.execute(
            """
            SELECT topic_id, card_id, stability, difficulty,
                   due_at, last_review_at, reps, lapses, suspended
            FROM fsrs_cards
            WHERE user_id = ?
            ORDER BY topic_id ASC, card_id ASC
            """,
            (user_id,),
        ).fetchall()
        # Per-topic review accuracy from the review log, joined to
        # cards so each logged review rolls up under its card's topic.
        accuracy_rows = self._conn.execute(
            """
            SELECT c.topic_id AS topic_id,
                   COUNT(*) AS total,
                   SUM(CASE WHEN l.rating IN ('good', 'easy')
                            THEN 1 ELSE 0 END) AS successful
            FROM fsrs_review_log l
            JOIN fsrs_cards c
              ON c.user_id = l.user_id AND c.card_id = l.card_id
            WHERE l.user_id = ?
            GROUP BY c.topic_id
            """,
            (user_id,),
        ).fetchall()
        accuracy_by_topic: dict[str, float] = {}
        for row in accuracy_rows:
            total = float(row["total"] or 0)
            successful = float(row["successful"] or 0)
            accuracy_by_topic[str(row["topic_id"])] = (
                round(successful / total, 4) if total else 0.0
            )
        totals: dict[str, list[float]] = {}
        attempts_by_topic: dict[str, int] = {}
        due_by_topic: dict[str, int] = {}
        for row in rows:
            card = {
                "stability": (
                    float(row["stability"])
                    if row["stability"] is not None
                    else None
                ),
                "last_review_at": (
                    str(row["last_review_at"])
                    if row["last_review_at"] is not None
                    else None
                ),
            }
            topic_id = str(row["topic_id"])
            totals.setdefault(topic_id, []).append(
                fsrs_scheduler.retention_estimate(card)
            )
            attempts_by_topic[topic_id] = (
                attempts_by_topic.get(topic_id, 0)
                + int(row["reps"] or 0)
            )
            if (
                int(row["suspended"] or 0) == 0
                and str(row["due_at"] or "") <= now_iso
            ):
                due_by_topic[topic_id] = due_by_topic.get(topic_id, 0) + 1
        topics = [
            {
                "topic_id": topic_id,
                "mastery_score": round(
                    sum(scores) / len(scores), 4
                ) if scores else 0.0,
                "attempts": attempts_by_topic.get(topic_id, 0),
                "accuracy": accuracy_by_topic.get(topic_id, 0.0),
                "due_count": due_by_topic.get(topic_id, 0),
            }
            for topic_id, scores in sorted(totals.items())
        ]
        return {"topics": topics[: max(0, int(limit))]}

    def backfill_fsrs_from_progress(
        self, *, batch_size: int = 500
    ) -> dict:
        """Migrate legacy ``question_progress`` rows into ``fsrs_cards``.

        Idempotent: every row is guarded by ``INSERT OR IGNORE`` on
        the ``(user_id, card_id)`` primary key, so already-migrated
        rows are skipped per row. There is deliberately no global
        "fsrs_cards is populated" short-circuit — a mixed-state
        database (some users migrated, others not) must still migrate
        the remaining users' rows.
        """
        batch = max(1, int(batch_size))
        total_rows = int(
            self._conn.execute(
                "SELECT COUNT(*) FROM question_progress"
            ).fetchone()[0]
        )
        if total_rows == 0:
            return {
                "skipped": False,
                "question_progress_rows": 0,
                "inserted": 0,
            }

        inserted = 0
        offset = 0
        while True:
            rows = self._conn.execute(
                """
                SELECT user_id, question_id, topic_id, mastery_score,
                       due_at, review_bucket, attempts, correct_attempts,
                       updated_at
                FROM question_progress
                ORDER BY user_id, question_id, topic_id
                LIMIT ? OFFSET ?
                """,
                (batch, offset),
            ).fetchall()
            if not rows:
                break
            with self._lock:
                with self._conn:
                    for row in rows:
                        mastery = max(
                            0.0, min(1.0, float(row["mastery_score"] or 0.0))
                        )
                        bucket = int(row["review_bucket"] or 0)
                        attempts = int(row["attempts"] or 0)
                        correct_attempts = int(row["correct_attempts"] or 0)
                        if bucket <= 0:
                            state = fsrs_scheduler.STATE_LEARNING
                        elif mastery < 0.3:
                            state = fsrs_scheduler.STATE_RELEARNING
                        else:
                            state = fsrs_scheduler.STATE_REVIEW
                        stability = round(max(0.2, mastery * 20.0), 4)
                        difficulty = round(
                            max(1.0, min(10.0, 10.0 - mastery * 5.0)), 4
                        )
                        interval_days = _LEGACY_BUCKET_INTERVAL_DAYS[
                            min(
                                bucket,
                                len(_LEGACY_BUCKET_INTERVAL_DAYS) - 1,
                            )
                        ]
                        updated_at = str(row["updated_at"] or "")
                        cursor = self._conn.execute(
                            """
                            INSERT OR IGNORE INTO fsrs_cards(
                                user_id, card_id, topic_id, source_type, state,
                                stability, difficulty, due_at, last_review_at,
                                reps, lapses, scheduled_days, elapsed_days,
                                suspended, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                str(row["user_id"]),
                                str(row["question_id"]),
                                str(row["topic_id"]),
                                "question",
                                state,
                                stability,
                                difficulty,
                                str(row["due_at"] or updated_at or _to_iso(_utc_now())),
                                updated_at or None,
                                attempts,
                                max(0, attempts - correct_attempts),
                                interval_days,
                                0,
                                0,
                                updated_at or _to_iso(_utc_now()),
                                updated_at or _to_iso(_utc_now()),
                            ),
                        )
                        if cursor.rowcount:
                            inserted += 1
            offset += batch
        return {
            "skipped": False,
            "question_progress_rows": total_rows,
            "inserted": inserted,
        }

    def get_weak_areas(self, *, user_id: str, limit: int = 10) -> dict:
        now_iso = _to_iso(_utc_now())
        rows = self._conn.execute(
            """
            SELECT topic_id, card_id, stability, difficulty,
                   due_at, last_review_at, reps, lapses, suspended
            FROM fsrs_cards
            WHERE user_id = ?
            ORDER BY topic_id ASC, card_id ASC
            """,
            (user_id,),
        ).fetchall()
        # Per-topic review accuracy from the review log, joined to
        # cards so each logged review rolls up under its card's topic.
        accuracy_rows = self._conn.execute(
            """
            SELECT c.topic_id AS topic_id,
                   COUNT(*) AS total,
                   SUM(CASE WHEN l.rating IN ('good', 'easy')
                            THEN 1 ELSE 0 END) AS successful
            FROM fsrs_review_log l
            JOIN fsrs_cards c
              ON c.user_id = l.user_id AND c.card_id = l.card_id
            WHERE l.user_id = ?
            GROUP BY c.topic_id
            """,
            (user_id,),
        ).fetchall()
        accuracy_by_topic: dict[str, float] = {}
        for row in accuracy_rows:
            total = float(row["total"] or 0)
            successful = float(row["successful"] or 0)
            accuracy_by_topic[str(row["topic_id"])] = (
                round(successful / total, 4) if total else 0.0
            )
        topics: dict[str, dict[str, Any]] = {}
        for row in rows:
            topic_id = str(row["topic_id"])
            entry = topics.setdefault(
                topic_id,
                {
                    "topic_id": topic_id,
                    "attempts": 0,
                    "correct_attempts": 0,
                    "retention": [],
                    "due_count": 0,
                },
            )
            reps = int(row["reps"] or 0)
            lapses = int(row["lapses"] or 0)
            entry["attempts"] += reps
            entry["correct_attempts"] += max(0, reps - lapses)
            entry["retention"].append(
                fsrs_scheduler.retention_estimate(
                    {
                        "stability": (
                            float(row["stability"])
                            if row["stability"] is not None
                            else None
                        ),
                        "last_review_at": (
                            str(row["last_review_at"])
                            if row["last_review_at"] is not None
                            else None
                        ),
                    }
                )
            )
            if (
                int(row["suspended"] or 0) == 0
                and str(row["due_at"] or "") <= now_iso
            ):
                entry["due_count"] += 1
        weak_areas = []
        for entry in topics.values():
            attempts = entry["attempts"]
            correct_attempts = entry["correct_attempts"]
            retention = entry["retention"]
            weak_areas.append(
                {
                    "topic_id": entry["topic_id"],
                    "attempts": attempts,
                    "accuracy": round(
                        correct_attempts / attempts, 4
                    ) if attempts else 0.0,
                    # FSRS does not track self-reported confidence.
                    "avg_confidence": 0.0,
                    "mastery_score": round(
                        sum(retention) / len(retention), 4
                    ) if retention else 0.0,
                    "due_count": entry["due_count"],
                }
            )
        weak_areas.sort(
            key=lambda item: (
                item["mastery_score"],
                item["accuracy"],
                -item["attempts"],
            )
        )
        return {"weak_areas": weak_areas[: max(0, int(limit))]}

    def get_topic_mastery(self, *, user_id: str, limit: int = 500) -> dict:
        rows = self._conn.execute(
            """
            SELECT topic_id, AVG(mastery_score) AS mastery_score, SUM(attempts) AS attempts
            FROM question_progress
            WHERE user_id = ?
            GROUP BY topic_id
            ORDER BY topic_id ASC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        return {
            "topics": [
                {
                    "topic_id": row["topic_id"],
                    "mastery_score": round(float(row["mastery_score"] or 0.0), 4),
                    "attempts": int(row["attempts"] or 0),
                }
                for row in rows
            ]
        }

    def build_study_plan(
        self,
        *,
        user_id: str,
        days: int = 7,
        daily_items: int = 3,
    ) -> dict[str, Any]:
        target_days = max(1, min(int(days), 31))
        target_daily_items = max(1, min(int(daily_items), 10))
        now = _utc_now()
        now_iso = _to_iso(now)
        today_iso = now_iso
        # Per-topic evidence is read from the consolidated
        # FSRS scheduler (fsrs_cards + retention estimates)
        # instead of the retired question_progress buckets.
        # ``attempts`` is the total review count (reps),
        # ``correct_attempts`` the successful reviews
        # (reps - lapses), and ``mastery_score`` the mean
        # retention estimate.
        cards = self._conn.execute(
            """
            SELECT topic_id, card_id, stability, difficulty,
                   due_at, last_review_at, reps, lapses, suspended
            FROM fsrs_cards
            WHERE user_id = ?
            ORDER BY topic_id ASC, card_id ASC
            """,
            (user_id,),
        ).fetchall()
        topics: dict[str, dict[str, Any]] = {}
        for card in cards:
            topic_id = str(card["topic_id"])
            entry = topics.setdefault(
                topic_id,
                {
                    "topic_id": topic_id,
                    "attempts": 0,
                    "correct_attempts": 0,
                    "retention": [],
                    "due_count": 0,
                },
            )
            reps = int(card["reps"] or 0)
            lapses = int(card["lapses"] or 0)
            entry["attempts"] += reps
            entry["correct_attempts"] += max(0, reps - lapses)
            entry["retention"].append(
                fsrs_scheduler.retention_estimate(
                    {
                        "stability": (
                            float(card["stability"])
                            if card["stability"] is not None
                            else None
                        ),
                        "last_review_at": (
                            str(card["last_review_at"])
                            if card["last_review_at"] is not None
                            else None
                        ),
                    }
                )
            )
            if (
                int(card["suspended"] or 0) == 0
                and str(card["due_at"] or "") <= today_iso
            ):
                entry["due_count"] += 1
        rows = []
        for entry in topics.values():
            retention = entry.pop("retention")
            rows.append(
                {
                    "topic_id": entry["topic_id"],
                    "attempts": entry["attempts"],
                    "correct_attempts": entry["correct_attempts"],
                    "mastery_score": (
                        sum(retention) / len(retention)
                        if retention
                        else 0.0
                    ),
                    "due_count": entry["due_count"],
                }
            )
        rows.sort(key=lambda row: row["topic_id"])

        days_plan: list[dict[str, Any]] = []
        if not rows:
            for idx in range(target_days):
                day_date = (now + timedelta(days=idx)).date().isoformat()
                label = "Today" if idx == 0 else f"Day {idx + 1}"
                days_plan.append(
                    {
                        "day_index": idx + 1,
                        "label": label,
                        "date": day_date,
                        "tasks": [],
                    }
                )
            return {
                "generated_at": now_iso,
                "days": target_days,
                "daily_items": target_daily_items,
                "total_tasks": 0,
                "days_plan": days_plan,
            }

        topic_rows: list[dict[str, Any]] = []
        max_due = 0
        for row in rows:
            attempts = int(row["attempts"] or 0)
            correct_attempts = int(row["correct_attempts"] or 0)
            mastery = max(0.0, min(1.0, float(row["mastery_score"] or 0.0)))
            due_count = max(0, int(row["due_count"] or 0))
            max_due = max(max_due, due_count)
            accuracy = (correct_attempts / attempts) if attempts > 0 else 0.0
            topic_rows.append(
                {
                    "topic_id": str(row["topic_id"]),
                    "attempts": attempts,
                    "accuracy": max(0.0, min(1.0, accuracy)),
                    "mastery": mastery,
                    "due_count": due_count,
                }
            )

        weak_topics: list[dict[str, Any]] = []
        for item in topic_rows:
            due_pressure = (item["due_count"] / max_due) if max_due > 0 else 0.0
            score = (
                0.5 * (1.0 - item["mastery"])
                + 0.3 * due_pressure
                + 0.2 * (1.0 - item["accuracy"])
            )
            weak_topics.append({**item, "score": score})

        weak_topics.sort(
            key=lambda x: (
                -x["score"],
                -x["due_count"],
                x["mastery"],
                x["topic_id"],
            )
        )
        due_topics = [x for x in weak_topics if x["due_count"] > 0]
        due_topics.sort(key=lambda x: (-x["due_count"], x["mastery"], x["topic_id"]))
        weak_cursor = 0

        def _build_task(task_type: str, topic: dict[str, Any]) -> dict[str, Any]:
            topic_id = topic["topic_id"]
            mastery_pct = round(topic["mastery"] * 100)
            due_count = int(topic["due_count"])
            if task_type == "review":
                return {
                    "task_type": "review",
                    "topic_id": topic_id,
                    "title": f"Review due questions for {topic_id}",
                    "reason": f"{due_count} item(s) due now; current mastery {mastery_pct}%.",
                    "estimated_minutes": 20,
                    "cta_route": f"/topics/{topic_id}",
                }
            if task_type == "topic_study":
                return {
                    "task_type": "topic_study",
                    "topic_id": topic_id,
                    "title": f"Targeted study on {topic_id}",
                    "reason": f"Weakness score priority; mastery {mastery_pct}%.",
                    "estimated_minutes": 30,
                    "cta_route": f"/topics/{topic_id}",
                }
            return {
                "task_type": "quiz",
                "topic_id": topic_id,
                "title": f"Reinforce {topic_id} with quiz",
                "reason": "Validate retention and confidence after focused review.",
                "estimated_minutes": 15,
                "cta_route": f"/quiz/{topic_id}",
            }

        for idx in range(target_days):
            tasks: list[dict[str, Any]] = []
            used_topics: set[str] = set()

            if due_topics:
                due_topic = due_topics[idx % len(due_topics)]
                tasks.append(_build_task("review", due_topic))
                used_topics.add(due_topic["topic_id"])

            guard = 0
            while len(tasks) < target_daily_items and weak_topics and guard < (len(weak_topics) * 3):
                guard += 1
                topic = weak_topics[weak_cursor % len(weak_topics)]
                weak_cursor += 1
                if (
                    topic["topic_id"] in used_topics
                    and len(used_topics) < len(weak_topics)
                ):
                    continue
                task_type = "topic_study" if ((idx + len(tasks)) % 2 == 0) else "quiz"
                tasks.append(_build_task(task_type, topic))
                used_topics.add(topic["topic_id"])

            day_date = (now + timedelta(days=idx)).date().isoformat()
            label = "Today" if idx == 0 else f"Day {idx + 1}"
            days_plan.append(
                {
                    "day_index": idx + 1,
                    "label": label,
                    "date": day_date,
                    "tasks": tasks,
                }
            )

        total_tasks = sum(len(day["tasks"]) for day in days_plan)
        return {
            "generated_at": now_iso,
            "days": target_days,
            "daily_items": target_daily_items,
            "total_tasks": total_tasks,
            "days_plan": days_plan,
        }

    def upsert_custom_topic(
        self,
        *,
        user_id: str,
        topic_id: str,
        source_topic: str,
        title: str,
        description: str,
        track: str,
        levels: list[str],
        sections: list[dict[str, str]],
        raw_content: str,
    ) -> dict:
        now_iso = _to_iso(_utc_now())
        levels_json = json.dumps(levels, ensure_ascii=True)
        sections_json = json.dumps(sections, ensure_ascii=True)
        with self._lock:
            with self._conn:
                existing = self._conn.execute(
                    """
                    SELECT created_at
                    FROM custom_topics
                    WHERE user_id = ? AND topic_id = ?
                    """,
                    (user_id, topic_id),
                ).fetchone()
                created_at = str(existing["created_at"]) if existing else now_iso
                self._conn.execute(
                    """
                    INSERT INTO custom_topics(
                        user_id, topic_id, source_topic, title, description, track,
                        levels_json, sections_json, raw_content, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, topic_id) DO UPDATE SET
                        source_topic = excluded.source_topic,
                        title = excluded.title,
                        description = excluded.description,
                        track = excluded.track,
                        levels_json = excluded.levels_json,
                        sections_json = excluded.sections_json,
                        raw_content = excluded.raw_content,
                        updated_at = excluded.updated_at
                    """,
                    (
                        user_id,
                        topic_id,
                        source_topic,
                        title,
                        description,
                        track,
                        levels_json,
                        sections_json,
                        raw_content,
                        created_at,
                        now_iso,
                    ),
                )
        return {
            "id": topic_id,
            "title": title,
            "description": description,
            "track": track,
            "levels": levels,
            "sections": sections,
            "raw_content": raw_content,
            "source_topic": source_topic,
            "created_at": created_at,
            "updated_at": now_iso,
        }

    @staticmethod
    def _loads_json_list(value: str) -> list:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []

    def _custom_topic_row_to_detail(self, row: sqlite3.Row) -> dict:
        levels = [str(v) for v in self._loads_json_list(str(row["levels_json"]))]
        sections_raw = self._loads_json_list(str(row["sections_json"]))
        sections: list[dict[str, str]] = []
        for item in sections_raw:
            if not isinstance(item, dict):
                continue
            heading = str(item.get("heading", "")).strip()
            content = str(item.get("content", "")).strip()
            if not heading:
                continue
            sections.append({"heading": heading, "content": content})
        return {
            "id": str(row["topic_id"]),
            "title": str(row["title"]),
            "description": str(row["description"]),
            "track": str(row["track"]),
            "levels": levels,
            "sections": sections,
            "raw_content": str(row["raw_content"]),
            "source_topic": str(row["source_topic"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    def list_custom_topics(
        self,
        *,
        user_id: str,
        track: str | None = None,
        level: str | None = None,
        q: str | None = None,
    ) -> list[dict]:
        query = (q or "").strip().lower()
        rows = self._conn.execute(
            """
            SELECT
                user_id, topic_id, source_topic, title, description, track,
                levels_json, sections_json, raw_content, created_at, updated_at
            FROM custom_topics
            WHERE user_id = ?
            ORDER BY updated_at DESC, topic_id ASC
            """,
            (user_id,),
        ).fetchall()

        out: list[dict] = []
        for row in rows:
            detail = self._custom_topic_row_to_detail(row)
            if track and detail["track"] != track:
                continue
            if level and level not in detail["levels"]:
                continue
            if query and query not in f"{detail['title']} {detail['description']}".lower():
                continue
            out.append(
                {
                    "id": detail["id"],
                    "title": detail["title"],
                    "description": detail["description"],
                    "track": detail["track"],
                    "levels": detail["levels"],
                    "section_count": len(detail["sections"]),
                    "estimated_questions": max(3, len(detail["sections"]) * 2),
                }
            )
        return out

    def get_custom_topic(self, *, user_id: str, topic_id: str) -> dict | None:
        row = self._conn.execute(
            """
            SELECT
                user_id, topic_id, source_topic, title, description, track,
                levels_json, sections_json, raw_content, created_at, updated_at
            FROM custom_topics
            WHERE user_id = ? AND topic_id = ?
            LIMIT 1
            """,
            (user_id, topic_id),
        ).fetchone()
        if not row:
            return None
        return self._custom_topic_row_to_detail(row)

    def get_dynamic_topic_curriculum(
        self,
        *,
        user_id: str,
        topic_id: str,
        preferred_language: str,
    ) -> dict | None:
        language = self._normalise_language(preferred_language)
        row = self._conn.execute(
            """
            SELECT
                user_id, topic_id, preferred_language, title, description, track,
                levels_json, sections_json, raw_content, target_sections, source,
                created_at, updated_at
            FROM dynamic_topic_curricula
            WHERE user_id = ? AND topic_id = ? AND preferred_language = ?
            LIMIT 1
            """,
            (user_id, topic_id, language),
        ).fetchone()
        if not row:
            return None
        sections_raw = self._loads_json_list(str(row["sections_json"]))
        sections: list[dict[str, str]] = []
        for item in sections_raw:
            if not isinstance(item, dict):
                continue
            heading = str(item.get("heading", "")).strip()
            content = str(item.get("content", "")).strip()
            if not heading:
                continue
            sections.append({"heading": heading, "content": content})
        return {
            "id": str(row["topic_id"]),
            "title": str(row["title"]),
            "description": str(row["description"]),
            "track": str(row["track"]),
            "levels": [str(v) for v in self._loads_json_list(str(row["levels_json"]))],
            "sections": sections,
            "raw_content": str(row["raw_content"]),
            "preferred_language": str(row["preferred_language"]),
            "target_sections": int(row["target_sections"] or 0),
            "source": str(row["source"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    def upsert_dynamic_topic_curriculum(
        self,
        *,
        user_id: str,
        topic_id: str,
        preferred_language: str,
        title: str,
        description: str,
        track: str,
        levels: list[str],
        sections: list[dict[str, str]],
        raw_content: str,
        target_sections: int,
        source: str,
    ) -> dict:
        now_iso = _to_iso(_utc_now())
        language = self._normalise_language(preferred_language)
        levels_json = json.dumps(levels, ensure_ascii=True)
        sections_json = json.dumps(sections, ensure_ascii=True)
        with self._lock:
            with self._conn:
                existing = self._conn.execute(
                    """
                    SELECT created_at
                    FROM dynamic_topic_curricula
                    WHERE user_id = ? AND topic_id = ? AND preferred_language = ?
                    """,
                    (user_id, topic_id, language),
                ).fetchone()
                created_at = str(existing["created_at"]) if existing else now_iso
                self._conn.execute(
                    """
                    INSERT INTO dynamic_topic_curricula(
                        user_id, topic_id, preferred_language, title, description, track,
                        levels_json, sections_json, raw_content, target_sections, source,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, topic_id, preferred_language) DO UPDATE SET
                        title = excluded.title,
                        description = excluded.description,
                        track = excluded.track,
                        levels_json = excluded.levels_json,
                        sections_json = excluded.sections_json,
                        raw_content = excluded.raw_content,
                        target_sections = excluded.target_sections,
                        source = excluded.source,
                        updated_at = excluded.updated_at
                    """,
                    (
                        user_id,
                        topic_id,
                        language,
                        title,
                        description,
                        track,
                        levels_json,
                        sections_json,
                        raw_content,
                        int(target_sections),
                        str(source or "").strip()[:40] or "llm",
                        created_at,
                        now_iso,
                    ),
                )
        return {
            "id": topic_id,
            "preferred_language": language,
            "title": title,
            "description": description,
            "track": track,
            "levels": levels,
            "sections": sections,
            "raw_content": raw_content,
            "target_sections": int(target_sections),
            "source": str(source or "").strip()[:40] or "llm",
            "created_at": created_at,
            "updated_at": now_iso,
        }

    def delete_dynamic_topic_curriculum(
        self,
        *,
        user_id: str,
        topic_id: str,
        preferred_language: str,
    ) -> None:
        language = self._normalise_language(preferred_language)
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    DELETE FROM dynamic_topic_curricula
                    WHERE user_id = ? AND topic_id = ? AND preferred_language = ?
                    """,
                    (user_id, topic_id, language),
                )

    def resolve_problem_solving_topic_detail(
        self,
        *,
        user_id: str,
        preferred_language: str,
    ) -> dict:
        language = self._normalise_language(preferred_language)
        if language not in PROBLEM_SOLVING_LANGUAGE_OPTIONS:
            language = PROBLEM_SOLVING_DEFAULT_LANGUAGE

        cached = self.get_dynamic_topic_curriculum(
            user_id=user_id,
            topic_id=PROBLEM_SOLVING_TOPIC_ID,
            preferred_language=language,
        )
        if cached:
            return {
                "id": PROBLEM_SOLVING_TOPIC_ID,
                "title": str(cached.get("title") or PROBLEM_SOLVING_TITLE),
                "description": str(cached.get("description") or ""),
                "track": str(cached.get("track") or PROBLEM_SOLVING_TRACK),
                "levels": list(cached.get("levels") or list(PROBLEM_SOLVING_LEVELS)),
                "sections": list(cached.get("sections") or []),
                "raw_content": str(cached.get("raw_content") or ""),
                "requires_programming": True,
                "language_options": list(PROBLEM_SOLVING_LANGUAGE_OPTIONS),
                "selected_language": language,
                "response_detail": "very_detailed",
                "is_dynamic_topic": True,
                "content_ready": True,
            }

        return {
            "id": PROBLEM_SOLVING_TOPIC_ID,
            "title": PROBLEM_SOLVING_TITLE,
            "description": (
                f"Pick a language and generate a {PROBLEM_SOLVING_TARGET_SECTIONS}-section "
                "problem-solving roadmap from beginner to senior."
            ),
            "track": PROBLEM_SOLVING_TRACK,
            "levels": list(PROBLEM_SOLVING_LEVELS),
            "sections": [],
            "raw_content": "",
            "requires_programming": True,
            "language_options": list(PROBLEM_SOLVING_LANGUAGE_OPTIONS),
            "selected_language": language,
            "response_detail": "very_detailed",
            "is_dynamic_topic": True,
            "content_ready": False,
        }

    @staticmethod
    def _normalise_response_detail(value: str | None) -> str:
        detail = str(value or "").strip().lower()
        if detail == "very_detailed":
            return "very_detailed"
        return "concise"

    @staticmethod
    def _normalise_language(value: str | None) -> str:
        return str(value or "").strip().lower()[:60]

    @staticmethod
    def _normalise_language_options(options: list[str] | None) -> list[str]:
        out: list[str] = []
        for raw in options or []:
            lang = str(raw or "").strip().lower()
            if not lang or lang in out:
                continue
            out.append(lang)
            if len(out) >= 8:
                break
        return out

    @staticmethod
    def _infer_requires_programming_from_topic(topic_detail: dict | None) -> bool:
        if not isinstance(topic_detail, dict):
            return False
        track = str(topic_detail.get("track", "")).strip().lower()
        text = (
            f"{topic_detail.get('id', '')} {topic_detail.get('title', '')} "
            f"{topic_detail.get('description', '')}"
        ).lower()
        if track in {"backend", "frontend"}:
            return True
        if track == "ai_stack":
            return any(
                key in text
                for key in (
                    "agent",
                    "rag",
                    "prompt",
                    "python",
                    "javascript",
                    "typescript",
                    "llm app",
                    "serving",
                )
            )
        if track == "system_design":
            return any(
                key in text
                for key in (
                    "algorithm",
                    "coding",
                    "implementation",
                    "api",
                    "database",
                    "java",
                    "python",
                    "typescript",
                )
            )
        return any(
            key in text
            for key in (
                "code",
                "coding",
                "programming",
                "implementation",
                "api",
                "service",
                "backend",
                "frontend",
            )
        )

    def get_topic_preferences(self, *, user_id: str, topic_id: str) -> dict | None:
        row = self._conn.execute(
            """
            SELECT user_id, topic_id, response_detail, preferred_language, created_at, updated_at
            FROM topic_user_preferences
            WHERE user_id = ? AND topic_id = ?
            LIMIT 1
            """,
            (user_id, topic_id),
        ).fetchone()
        if not row:
            return None
        return {
            "topic_id": str(row["topic_id"]),
            "response_detail": self._normalise_response_detail(str(row["response_detail"])),
            "preferred_language": self._normalise_language(str(row["preferred_language"])),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    def upsert_topic_preferences(
        self,
        *,
        user_id: str,
        topic_id: str,
        response_detail: str | None = None,
        preferred_language: str | None = None,
    ) -> dict:
        now_iso = _to_iso(_utc_now())
        existing = self.get_topic_preferences(user_id=user_id, topic_id=topic_id)
        next_response_detail = (
            self._normalise_response_detail(response_detail)
            if response_detail is not None
            else (existing or {}).get("response_detail", "very_detailed")
        )
        next_language = (
            self._normalise_language(preferred_language)
            if preferred_language is not None
            else (existing or {}).get("preferred_language", "")
        )
        created_at = (existing or {}).get("created_at", now_iso)

        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO topic_user_preferences(
                        user_id, topic_id, response_detail, preferred_language, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, topic_id) DO UPDATE SET
                        response_detail = excluded.response_detail,
                        preferred_language = excluded.preferred_language,
                        updated_at = excluded.updated_at
                    """,
                    (
                        user_id,
                        topic_id,
                        next_response_detail,
                        next_language,
                        created_at,
                        now_iso,
                    ),
                )
        return {
            "topic_id": topic_id,
            "response_detail": next_response_detail,
            "preferred_language": next_language,
            "created_at": created_at,
            "updated_at": now_iso,
        }

    def get_topic_language_profile(self, *, topic_id: str) -> dict | None:
        row = self._conn.execute(
            """
            SELECT topic_id, requires_programming, language_options_json, source, updated_at
            FROM topic_language_profiles
            WHERE topic_id = ?
            LIMIT 1
            """,
            (topic_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "topic_id": str(row["topic_id"]),
            "requires_programming": bool(int(row["requires_programming"])),
            "language_options": self._normalise_language_options(
                [str(v) for v in self._loads_json_list(str(row["language_options_json"]))]
            ),
            "source": str(row["source"]),
            "updated_at": str(row["updated_at"]),
        }

    def upsert_topic_language_profile(
        self,
        *,
        topic_id: str,
        requires_programming: bool,
        language_options: list[str],
        source: str,
    ) -> dict:
        now_iso = _to_iso(_utc_now())
        normalized_options = self._normalise_language_options(language_options)
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO topic_language_profiles(
                        topic_id, requires_programming, language_options_json, source, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(topic_id) DO UPDATE SET
                        requires_programming = excluded.requires_programming,
                        language_options_json = excluded.language_options_json,
                        source = excluded.source,
                        updated_at = excluded.updated_at
                    """,
                    (
                        topic_id,
                        1 if requires_programming else 0,
                        json.dumps(normalized_options, ensure_ascii=True),
                        str(source or "").strip()[:80] or "unknown",
                        now_iso,
                    ),
                )
        return {
            "topic_id": topic_id,
            "requires_programming": bool(requires_programming),
            "language_options": normalized_options,
            "source": str(source or "").strip()[:80] or "unknown",
            "updated_at": now_iso,
        }

    def resolve_topic_ai_settings(
        self,
        *,
        user_id: str,
        topic_id: str,
        topic_detail: dict | None,
    ) -> dict:
        profile = self.get_topic_language_profile(topic_id=topic_id)
        prefs = self.get_topic_preferences(user_id=user_id, topic_id=topic_id)
        if is_problem_solving_topic(topic_id):
            requires_programming = True
            language_options = list(PROBLEM_SOLVING_LANGUAGE_OPTIONS)
        else:
            requires_programming = (
                bool(profile["requires_programming"])
                if profile is not None
                else self._infer_requires_programming_from_topic(topic_detail)
            )
            language_options = (
                self._normalise_language_options(profile.get("language_options", []))
                if profile is not None
                else []
            )
        response_detail = self._normalise_response_detail(
            (prefs or {}).get("response_detail", "very_detailed")
        )
        preferred_language = self._normalise_language((prefs or {}).get("preferred_language", ""))

        if not requires_programming:
            preferred_language = ""
        elif preferred_language and language_options and preferred_language not in language_options:
            preferred_language = ""
        if requires_programming and not preferred_language and language_options:
            preferred_language = language_options[0]

        return {
            "topic_id": topic_id,
            "response_detail": response_detail,
            "preferred_language": preferred_language,
            "requires_programming": requires_programming,
            "language_options": language_options,
            "profile_source": (profile or {}).get("source", ""),
            "profile_updated_at": (profile or {}).get("updated_at", ""),
        }

    def get_assistant_memory(
        self,
        *,
        user_id: str,
        conversation_id: str,
        flow: str = "chat",
    ) -> dict | None:
        row = self._conn.execute(
            """
            SELECT summary_json, created_at, updated_at
            FROM assistant_memory
            WHERE user_id = ? AND conversation_id = ? AND flow = ?
            LIMIT 1
            """,
            (user_id, conversation_id, flow),
        ).fetchone()
        if not row:
            return None
        try:
            summary = json.loads(str(row["summary_json"]))
        except json.JSONDecodeError:
            summary = {}
        if not isinstance(summary, dict):
            summary = {}
        return {
            "conversation_id": conversation_id,
            "flow": flow,
            "summary": summary,
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    def upsert_assistant_memory(
        self,
        *,
        user_id: str,
        conversation_id: str,
        flow: str = "chat",
        summary: dict | None = None,
    ) -> dict:
        now_iso = _to_iso(_utc_now())
        summary_obj = summary if isinstance(summary, dict) else {}
        payload = json.dumps(summary_obj, ensure_ascii=True)
        existing = self.get_assistant_memory(
            user_id=user_id,
            conversation_id=conversation_id,
            flow=flow,
        )
        created_at = (existing or {}).get("created_at", now_iso)
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO assistant_memory(
                        user_id, conversation_id, flow, summary_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, conversation_id, flow) DO UPDATE SET
                        summary_json = excluded.summary_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        user_id,
                        conversation_id,
                        flow,
                        payload,
                        created_at,
                        now_iso,
                    ),
                )
        return {
            "conversation_id": conversation_id,
            "flow": flow,
            "summary": summary_obj,
            "created_at": created_at,
            "updated_at": now_iso,
        }

    def get_feature_state(self, *, feature_key: str) -> dict | None:
        row = self._conn.execute(
            """
            SELECT feature_key, status, disabled_until, reason, updated_at
            FROM feature_runtime_state
            WHERE feature_key = ?
            LIMIT 1
            """,
            (feature_key,),
        ).fetchone()
        if not row:
            return None
        return {
            "feature_key": str(row["feature_key"]),
            "status": str(row["status"]),
            "disabled_until": str(row["disabled_until"]),
            "reason": str(row["reason"]),
            "updated_at": str(row["updated_at"]),
        }

    def set_feature_state(
        self,
        *,
        feature_key: str,
        status: str,
        disabled_until: str = "",
        reason: str = "",
    ) -> dict:
        now_iso = _to_iso(_utc_now())
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO feature_runtime_state(
                        feature_key, status, disabled_until, reason, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(feature_key) DO UPDATE SET
                        status = excluded.status,
                        disabled_until = excluded.disabled_until,
                        reason = excluded.reason,
                        updated_at = excluded.updated_at
                    """,
                    (
                        feature_key,
                        str(status or "").strip()[:60] or "enabled",
                        str(disabled_until or "").strip()[:80],
                        str(reason or "").strip()[:200],
                        now_iso,
                    ),
                )
        return {
            "feature_key": feature_key,
            "status": str(status or "").strip()[:60] or "enabled",
            "disabled_until": str(disabled_until or "").strip()[:80],
            "reason": str(reason or "").strip()[:200],
            "updated_at": now_iso,
        }

    def get_topic_video_cache(
        self,
        *,
        cache_key: str,
        max_age_hours: int,
    ) -> dict | None:
        row = self._conn.execute(
            """
            SELECT cache_key, topic_id, section_heading, preferred_language, payload_json, fetched_at
            FROM topic_video_cache
            WHERE cache_key = ?
            LIMIT 1
            """,
            (cache_key,),
        ).fetchone()
        if not row:
            return None
        fetched_at = str(row["fetched_at"])
        try:
            fetched_dt = _from_iso(fetched_at)
        except ValueError:
            return None
        if fetched_dt.tzinfo is None:
            fetched_dt = fetched_dt.replace(tzinfo=UTC)
        age_seconds = (_utc_now() - fetched_dt.astimezone(UTC)).total_seconds()
        if age_seconds > max(0, int(max_age_hours)) * 3600:
            return None
        try:
            payload = json.loads(str(row["payload_json"]))
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        return {
            "cache_key": str(row["cache_key"]),
            "topic_id": str(row["topic_id"]),
            "section_heading": str(row["section_heading"]),
            "preferred_language": str(row["preferred_language"]),
            "payload": payload,
            "fetched_at": fetched_at,
        }

    def upsert_topic_video_cache(
        self,
        *,
        cache_key: str,
        topic_id: str,
        section_heading: str,
        preferred_language: str,
        payload: dict,
    ) -> dict:
        now_iso = _to_iso(_utc_now())
        payload_json = json.dumps(payload if isinstance(payload, dict) else {}, ensure_ascii=True)
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO topic_video_cache(
                        cache_key, topic_id, section_heading, preferred_language, payload_json, fetched_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        topic_id = excluded.topic_id,
                        section_heading = excluded.section_heading,
                        preferred_language = excluded.preferred_language,
                        payload_json = excluded.payload_json,
                        fetched_at = excluded.fetched_at
                    """,
                    (
                        cache_key,
                        topic_id,
                        section_heading[:240],
                        preferred_language[:60],
                        payload_json,
                        now_iso,
                    ),
                )
        return {
            "cache_key": cache_key,
            "topic_id": topic_id,
            "section_heading": section_heading[:240],
            "preferred_language": preferred_language[:60],
            "fetched_at": now_iso,
        }

    def record_feature_event(
        self,
        *,
        user_id: str,
        feature_key: str,
        event_name: str,
        metadata: dict | None = None,
    ) -> dict:
        now_iso = _to_iso(_utc_now())
        payload = json.dumps(metadata if isinstance(metadata, dict) else {}, ensure_ascii=True)
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO feature_events(
                        user_id, feature_key, event_name, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        str(user_id or "").strip()[:160] or "anonymous",
                        str(feature_key or "").strip()[:80],
                        str(event_name or "").strip()[:120],
                        payload,
                        now_iso,
                    ),
                )
        return {
            "user_id": str(user_id or "").strip()[:160] or "anonymous",
            "feature_key": str(feature_key or "").strip()[:80],
            "event_name": str(event_name or "").strip()[:120],
            "created_at": now_iso,
        }

    def get_feature_metrics(
        self,
        *,
        feature_key: str,
    ) -> dict:
        now = _utc_now()
        since_7 = now - timedelta(days=7)
        since_30 = now - timedelta(days=30)
        since_30_iso = _to_iso(since_30)
        state = self.get_feature_state(feature_key=feature_key) or {
            "feature_key": feature_key,
            "status": "disabled_config",
            "disabled_until": "",
            "reason": "",
            "updated_at": "",
        }

        rows_30 = self._conn.execute(
            """
            SELECT event_name, metadata_json, created_at
            FROM feature_events
            WHERE feature_key = ? AND created_at >= ?
            ORDER BY created_at ASC
            """,
            (feature_key, since_30_iso),
        ).fetchall()

        panel_views_30d = 0
        video_clicks_30d = 0
        reenabled_at = ""
        hidden_intervals: list[tuple[datetime, datetime]] = []
        weekly_hidden: dict[str, int] = {}

        for row in rows_30:
            event_name = str(row["event_name"])
            created_at_raw = str(row["created_at"])
            try:
                created_at = _from_iso(created_at_raw)
            except ValueError:
                continue
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=UTC)
            created_at_utc = created_at.astimezone(UTC)
            if event_name == "video_panel_viewed":
                panel_views_30d += 1
            elif event_name == "video_click":
                video_clicks_30d += 1
            elif event_name == "video_feature_reenabled":
                reenabled_at = created_at_raw
            elif event_name == "video_panel_hidden_quota":
                week_start = (created_at_utc - timedelta(days=created_at_utc.weekday())).date().isoformat()
                weekly_hidden[week_start] = weekly_hidden.get(week_start, 0) + 1
                try:
                    metadata = json.loads(str(row["metadata_json"]))
                except json.JSONDecodeError:
                    metadata = {}
                disabled_until_raw = str((metadata or {}).get("disabled_until", "")).strip()
                if disabled_until_raw:
                    try:
                        disabled_until = _from_iso(disabled_until_raw)
                        if disabled_until.tzinfo is None:
                            disabled_until = disabled_until.replace(tzinfo=UTC)
                        hidden_intervals.append((created_at_utc, disabled_until.astimezone(UTC)))
                    except ValueError:
                        continue

        status = str(state.get("status", "disabled_config")).strip() or "disabled_config"
        disabled_until_raw = str(state.get("disabled_until", "")).strip()
        disabled_until_dt: datetime | None = None
        if disabled_until_raw:
            try:
                parsed = _from_iso(disabled_until_raw)
                disabled_until_dt = parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
                disabled_until_dt = disabled_until_dt.astimezone(UTC)
            except ValueError:
                disabled_until_dt = None

        hidden_now = bool(
            status == "disabled_quota_exhausted"
            and disabled_until_dt is not None
            and disabled_until_dt > now
        )

        if hidden_now:
            try:
                start = _from_iso(str(state.get("updated_at", "")))
                if start.tzinfo is None:
                    start = start.replace(tzinfo=UTC)
                hidden_intervals.append((start.astimezone(UTC), disabled_until_dt))
            except ValueError:
                pass

        def _hidden_days(interval_start: datetime, interval_end: datetime, window_start: datetime) -> float:
            start = max(interval_start, window_start)
            end = min(interval_end, now)
            if end <= start:
                return 0.0
            return (end - start).total_seconds() / 86400.0

        hidden_days_7 = 0.0
        hidden_days_30 = 0.0
        for start, end in hidden_intervals:
            hidden_days_7 += _hidden_days(start, end, since_7)
            hidden_days_30 += _hidden_days(start, end, since_30)

        if reenabled_at:
            since_reenable_views = int(
                self._conn.execute(
                    """
                    SELECT COUNT(*) FROM feature_events
                    WHERE feature_key = ? AND event_name = 'video_panel_viewed' AND created_at >= ?
                    """,
                    (feature_key, reenabled_at),
                ).fetchone()[0]
            )
            since_reenable_clicks = int(
                self._conn.execute(
                    """
                    SELECT COUNT(*) FROM feature_events
                    WHERE feature_key = ? AND event_name = 'video_click' AND created_at >= ?
                    """,
                    (feature_key, reenabled_at),
                ).fetchone()[0]
            )
        else:
            since_reenable_views = 0
            since_reenable_clicks = 0

        weekly_list = [
            {"week_start": week_start, "hidden_events": count}
            for week_start, count in sorted(weekly_hidden.items(), reverse=True)[:8]
        ]

        return {
            "status": status,
            "disabled_until": disabled_until_raw,
            "reason": str(state.get("reason", "")),
            "hidden_now": hidden_now,
            "hidden_days_last_7": round(hidden_days_7, 2),
            "hidden_days_last_30": round(hidden_days_30, 2),
            "hidden_frequency_weekly": weekly_list,
            "panel_views_30d": panel_views_30d,
            "video_clicks_30d": video_clicks_30d,
            "ctr_30d": round((video_clicks_30d / panel_views_30d), 4) if panel_views_30d else 0.0,
            "reenabled_at": reenabled_at,
            "panel_views_since_reenable": since_reenable_views,
            "clicks_since_reenable": since_reenable_clicks,
            "ctr_since_reenable": (
                round((since_reenable_clicks / since_reenable_views), 4)
                if since_reenable_views
                else 0.0
            ),
        }
