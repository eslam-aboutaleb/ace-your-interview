"""SQLite-backed persistence for mock interview sessions, turns, and reports."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)

_DEFAULT_INTERVIEWER_STYLE = "neutral"
_DEFAULT_FEEDBACK_MODE = "concise"

#: Fallbacks for the interview-memory limits. The settings object is the source
#: of truth; these only keep a malformed/zero setting from disabling memory.
DEFAULT_MEMORY_SUMMARY_MAX_CHARS = 3000
DEFAULT_MEMORY_VERBATIM_TURNS = 4
DEFAULT_MEMORY_ANSWER_EXCERPT_CHARS = 320


def _positive_int(value: Any, fallback: int, *, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(minimum, parsed)


def memory_summary_limit() -> int:
    """The one digest ceiling for the interview memory (Stage 3.2).

    The router and this store both cap here, so a digest cannot be written at
    one length and rendered at another.
    """
    return _positive_int(
        getattr(get_settings(), "memory_summary_max_chars", DEFAULT_MEMORY_SUMMARY_MAX_CHARS),
        DEFAULT_MEMORY_SUMMARY_MAX_CHARS,
    )


def memory_verbatim_turn_count() -> int:
    """How many recent turns are kept verbatim alongside the digest."""
    return _positive_int(
        getattr(get_settings(), "memory_verbatim_turns", DEFAULT_MEMORY_VERBATIM_TURNS),
        DEFAULT_MEMORY_VERBATIM_TURNS,
        minimum=0,
    )


def memory_answer_excerpt_limit() -> int:
    """Per-turn answer excerpt length inside the verbatim window."""
    return _positive_int(
        getattr(get_settings(), "memory_answer_excerpt_chars", DEFAULT_MEMORY_ANSWER_EXCERPT_CHARS),
        DEFAULT_MEMORY_ANSWER_EXCERPT_CHARS,
    )


class TurnConflictError(Exception):
    """A turn with this ``(session_id, turn_index)`` already exists.

    Raised instead of a raw ``sqlite3.IntegrityError`` so the router can map a
    double submission of the same answer onto HTTP 409 rather than a 500.
    """

    code = "turn_conflict"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class InterviewStore:
    """Persist interview sessions and turn-by-turn rubric feedback."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        # Reentrant so read helpers can hold the lock and call each other. A
        # single sqlite connection is not safe for concurrent use, and two
        # concurrent answer submissions read the session while one writes.
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interview_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    track TEXT NOT NULL,
                    level TEXT NOT NULL,
                    interview_type TEXT NOT NULL,
                    turn_count INTEGER NOT NULL,
                    turns_completed INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    target_role TEXT NOT NULL,
                    interviewer_style TEXT NOT NULL DEFAULT 'neutral',
                    feedback_mode TEXT NOT NULL DEFAULT 'concise',
                    job_description_text TEXT NOT NULL,
                    resume_summary_text TEXT NOT NULL,
                    focus_areas_json TEXT NOT NULL,
                    asked_questions_json TEXT NOT NULL,
                    current_question TEXT NOT NULL,
                    memory_summary TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interview_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    question TEXT NOT NULL,
                    user_answer TEXT NOT NULL,
                    rubric_json TEXT NOT NULL,
                    strengths_json TEXT NOT NULL,
                    improvements_json TEXT NOT NULL,
                    follow_up_note TEXT NOT NULL,
                    response_time_ms INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interview_reports (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_interview_sessions_user
                ON interview_sessions(user_id, updated_at)
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_interview_turns_session
                ON interview_turns(session_id, turn_index)
                """
            )
            existing_columns = {
                str(row["name"]) for row in self._conn.execute("PRAGMA table_info(interview_sessions)").fetchall()
            }
            required_columns = {
                "interviewer_style": "TEXT NOT NULL DEFAULT 'neutral'",
                "feedback_mode": "TEXT NOT NULL DEFAULT 'concise'",
                "memory_summary": "TEXT NOT NULL DEFAULT ''",
                # Stage 3.2: `memory_digest` is the compacted digest that replaced
                # append-and-truncate; `memory_summary` is kept as its legacy
                # read alias so existing prompts and clients keep working.
                "memory_digest": "TEXT NOT NULL DEFAULT ''",
                "memory_window_json": "TEXT NOT NULL DEFAULT '[]'",
            }
            for column, ddl in required_columns.items():
                if column not in existing_columns:
                    self._conn.execute(f"ALTER TABLE interview_sessions ADD COLUMN {column} {ddl}")

            self._migrate_interview_memory()
            self._migrate_interview_turns()

    def _migrate_interview_memory(self) -> None:
        """Seed the new memory columns from what pre-migration rows already hold.

        Idempotent: a row already carrying a digest (or a window) is left alone,
        so re-opening a migrated database rewrites nothing. A legacy
        ``memory_summary`` becomes the digest seed, and a session that already
        has turns gets its verbatim window rebuilt from them.
        """
        limit = memory_summary_limit()
        window_size = memory_verbatim_turn_count()
        rows = self._conn.execute(
            """
            SELECT session_id, user_id,
                   COALESCE(memory_digest, '') AS digest,
                   COALESCE(memory_summary, '') AS legacy,
                   COALESCE(memory_window_json, '[]') AS window
            FROM interview_sessions
            """
        ).fetchall()

        pending: list[tuple[str, str, str]] = []
        for row in rows:
            digest = str(row["digest"] or "").strip()
            legacy = str(row["legacy"] or "").strip()
            if not digest and legacy:
                digest = legacy[:limit]
            window = self._loads_dict_list(row["window"])
            if not window:
                window = self._window_from_turns(
                    session_id=str(row["session_id"]),
                    user_id=str(row["user_id"]),
                    size=window_size,
                )
            current_window = str(row["window"] or "")
            if digest == str(row["digest"] or "").strip() and window == self._loads_dict_list(current_window):
                continue
            pending.append((digest, self._dumps(window), str(row["session_id"])))

        if pending:
            self._conn.executemany(
                """
                UPDATE interview_sessions
                SET memory_digest = ?, memory_window_json = ?
                WHERE session_id = ?
                """,
                pending,
            )
            logger.info("Seeded interview memory digest/window for %s session(s)", len(pending))

    def _window_from_turns(self, *, session_id: str, user_id: str, size: int) -> list[dict[str, Any]]:
        """Rebuild a verbatim window from the newest ``size`` stored turns."""
        if size <= 0:
            return []
        excerpt_limit = memory_answer_excerpt_limit()
        rows = self._conn.execute(
            """
            SELECT turn_index, question, user_answer
            FROM interview_turns
            WHERE user_id = ? AND session_id = ?
            ORDER BY turn_index DESC
            LIMIT ?
            """,
            (user_id, session_id, size),
        ).fetchall()
        return [
            {
                "turn_index": int(row["turn_index"]),
                "question": str(row["question"])[:220],
                "answer_excerpt": str(row["user_answer"])[:excerpt_limit],
            }
            for row in reversed(rows)
        ]

    def _migrate_interview_turns(self) -> None:
        """Add ``degraded`` and make ``(session_id, turn_index)`` unique.

        Both steps are idempotent so an already-migrated database is a no-op.
        The de-duplication must run before the unique index is created:
        databases written before the constraint can already hold two turns with
        the same index (concurrent submissions), and creating the index first
        would fail outright.
        """
        turn_columns = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(interview_turns)").fetchall()
        }
        if "degraded" not in turn_columns:
            self._conn.execute(
                "ALTER TABLE interview_turns ADD COLUMN degraded INTEGER NOT NULL DEFAULT 0"
            )

        existing_indexes = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA index_list(interview_turns)").fetchall()
        }
        if "idx_interview_turns_unique_turn" in existing_indexes:
            return

        removed = self._conn.execute(
            """
            DELETE FROM interview_turns
            WHERE id NOT IN (
                SELECT MIN(id) FROM interview_turns GROUP BY session_id, turn_index
            )
            """
        ).rowcount
        if removed and removed > 0:
            logger.warning(
                "Removed %s duplicate interview_turns row(s) while enforcing unique (session_id, turn_index)",
                removed,
            )

        self._conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_interview_turns_unique_turn
            ON interview_turns(session_id, turn_index)
            """
        )

    @staticmethod
    def _loads_list(raw: str) -> list[str]:
        try:
            data = json.loads(raw or "[]")
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        return [str(x).strip() for x in data if str(x).strip()]

    @staticmethod
    def _loads_dict(raw: str) -> dict[str, Any]:
        try:
            data = json.loads(raw or "{}")
        except json.JSONDecodeError:
            return {}
        if isinstance(data, dict):
            return data
        return {}

    @staticmethod
    def _loads_dict_list(raw: str) -> list[dict[str, Any]]:
        try:
            data = json.loads(raw or "[]")
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict)]

    @staticmethod
    def _memory_window_entry(entry: dict[str, Any]) -> dict[str, Any]:
        """Normalise one rolling-window entry, applying the excerpt ceiling."""
        try:
            turn_index = int(entry.get("turn_index") or 0)
        except (TypeError, ValueError):
            turn_index = 0
        excerpt = str(entry.get("answer_excerpt") or "").strip()
        return {
            "turn_index": turn_index,
            "question": str(entry.get("question") or "").strip()[:220],
            "answer_excerpt": excerpt[: memory_answer_excerpt_limit()],
            "strengths": [
                str(x).strip() for x in (entry.get("strengths") or []) if str(x).strip()
            ][:2],
            "improvements": [
                str(x).strip() for x in (entry.get("improvements") or []) if str(x).strip()
            ][:2],
        }

    @staticmethod
    def _dumps(data: Any) -> str:
        return json.dumps(data, ensure_ascii=True)

    def _session_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        keys = row.keys()
        # ``memory_summary`` stays the read alias of ``memory_digest``: prompts
        # and clients predate the digest column and must keep working.
        digest = (
            str(row["memory_digest"] or "").strip()
            if "memory_digest" in keys
            else ""
        )
        legacy_summary = (
            str(row["memory_summary"] or "").strip()
            if "memory_summary" in keys
            else ""
        )
        memory = digest or legacy_summary
        return {
            "session_id": row["session_id"],
            "track": row["track"],
            "level": row["level"],
            "interview_type": row["interview_type"],
            "turn_count": int(row["turn_count"]),
            "turns_completed": int(row["turns_completed"]),
            "status": row["status"],
            "target_role": row["target_role"],
            "interviewer_style": (
                row["interviewer_style"] if "interviewer_style" in keys else _DEFAULT_INTERVIEWER_STYLE
            ),
            "feedback_mode": row["feedback_mode"] if "feedback_mode" in keys else _DEFAULT_FEEDBACK_MODE,
            "focus_areas": self._loads_list(row["focus_areas_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "current_question": row["current_question"],
            "memory_summary": memory[: memory_summary_limit()],
            "memory_digest": memory[: memory_summary_limit()],
            "memory_window": (
                self._loads_dict_list(row["memory_window_json"])
                if "memory_window_json" in keys
                else []
            ),
            "report_ready": bool(int(row["report_ready"])) if "report_ready" in keys else False,
        }

    def create_session(
        self,
        *,
        user_id: str,
        track: str,
        level: str,
        interview_type: str,
        turn_count: int,
        target_role: str,
        job_description_text: str,
        resume_summary_text: str,
        focus_areas: list[str],
        interviewer_style: str = _DEFAULT_INTERVIEWER_STYLE,
        feedback_mode: str = _DEFAULT_FEEDBACK_MODE,
    ) -> dict[str, Any]:
        now = _utc_now_iso()
        session_id = f"is_{uuid.uuid4().hex[:16]}"
        focus = [x.strip() for x in focus_areas if x.strip()]

        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO interview_sessions(
                        session_id, user_id, track, level, interview_type,
                        turn_count, turns_completed, status,
                        target_role, interviewer_style, feedback_mode,
                        job_description_text, resume_summary_text,
                        focus_areas_json, asked_questions_json,
                        current_question, memory_summary, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        user_id,
                        track,
                        level,
                        interview_type,
                        turn_count,
                        "active",
                        target_role,
                        interviewer_style,
                        feedback_mode,
                        job_description_text,
                        resume_summary_text,
                        self._dumps(focus),
                        self._dumps([]),
                        "",
                        "",
                        now,
                        now,
                    ),
                )
        return {
            "session_id": session_id,
            "track": track,
            "level": level,
            "interview_type": interview_type,
            "turn_count": turn_count,
            "turns_completed": 0,
            "status": "active",
            "target_role": target_role,
            "interviewer_style": interviewer_style,
            "feedback_mode": feedback_mode,
            "focus_areas": focus,
            "created_at": now,
            "updated_at": now,
            "current_question": "",
            "memory_summary": "",
            "memory_digest": "",
            "memory_window": [],
            "report_ready": False,
        }

    def list_sessions(self, *, user_id: str, limit: int = 20, offset: int = 0) -> dict[str, Any]:
        rows = self._conn.execute(
            """
            SELECT
                s.*,
                EXISTS(SELECT 1 FROM interview_reports r WHERE r.session_id = s.session_id) AS report_ready
            FROM interview_sessions s
            WHERE s.user_id = ?
            ORDER BY s.updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (user_id, limit, offset),
        ).fetchall()
        total = int(
            self._conn.execute(
                "SELECT COUNT(*) FROM interview_sessions WHERE user_id = ?",
                (user_id,),
            ).fetchone()[0]
        )
        sessions = [self._session_row_to_dict(r) for r in rows]
        return {"sessions": sessions, "total": total}

    def get_session(self, *, user_id: str, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT
                    s.*,
                    EXISTS(SELECT 1 FROM interview_reports r WHERE r.session_id = s.session_id) AS report_ready
                FROM interview_sessions s
                WHERE s.user_id = ? AND s.session_id = ?
                """,
                (user_id, session_id),
            ).fetchone()
            if not row:
                return None
            return self._session_row_to_dict(row)

    def get_session_context(self, *, user_id: str, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT *
                FROM interview_sessions
                WHERE user_id = ? AND session_id = ?
                """,
                (user_id, session_id),
            ).fetchone()
            if not row:
                return None
            session = self._session_row_to_dict(row)
            # ``turn_index`` is allocated from this value, so it must be derived
            # from the same rows the unique constraint covers rather than from a
            # denormalised counter that can drift.
            session["turns_completed"] = self._count_turns(
                user_id=user_id,
                session_id=session_id,
            )
            return {
                **session,
                "job_description_text": row["job_description_text"],
                "resume_summary_text": row["resume_summary_text"],
                "asked_questions": self._loads_list(row["asked_questions_json"]),
            }

    def _count_turns(self, *, user_id: str, session_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT COUNT(*) AS total
                FROM interview_turns
                WHERE user_id = ? AND session_id = ?
                """,
                (user_id, session_id),
            ).fetchone()
        return int(row["total"]) if row else 0

    def set_current_question(
        self,
        *,
        user_id: str,
        session_id: str,
        question: str,
    ) -> dict[str, Any] | None:
        session = self.get_session_context(user_id=user_id, session_id=session_id)
        if not session:
            return None
        asked = list(session.get("asked_questions") or [])
        q = question.strip()
        if q:
            asked.append(q)
        now = _utc_now_iso()
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    UPDATE interview_sessions
                    SET current_question = ?, asked_questions_json = ?, updated_at = ?
                    WHERE user_id = ? AND session_id = ?
                    """,
                    (q, self._dumps(asked), now, user_id, session_id),
                )
        return self.get_session(user_id=user_id, session_id=session_id)

    def set_memory_digest(
        self,
        *,
        user_id: str,
        session_id: str,
        digest: str,
    ) -> dict[str, Any] | None:
        """Persist the compacted digest, capped at the one shared limit.

        The legacy ``memory_summary`` column is written alongside it so a
        database read by an older build still shows the note.
        """
        now = _utc_now_iso()
        text = str(digest or "").strip()[: memory_summary_limit()]
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    UPDATE interview_sessions
                    SET memory_digest = ?, memory_summary = ?, updated_at = ?
                    WHERE user_id = ? AND session_id = ?
                    """,
                    (text, text, now, user_id, session_id),
                )
        return self.get_session(user_id=user_id, session_id=session_id)

    def set_memory_summary(
        self,
        *,
        user_id: str,
        session_id: str,
        summary: str,
    ) -> dict[str, Any] | None:
        """Legacy spelling of :meth:`set_memory_digest`, kept for callers."""
        return self.set_memory_digest(
            user_id=user_id,
            session_id=session_id,
            digest=summary,
        )

    def get_memory_window(self, *, user_id: str, session_id: str) -> list[dict[str, Any]]:
        """The verbatim rolling window: the last ``memory_verbatim_turns`` turns."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT memory_window_json
                FROM interview_sessions
                WHERE user_id = ? AND session_id = ?
                """,
                (user_id, session_id),
            ).fetchone()
        if not row or "memory_window_json" not in row.keys():
            return []
        return self._loads_dict_list(row["memory_window_json"])

    def append_memory_turn(
        self,
        *,
        user_id: str,
        session_id: str,
        entry: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Append one turn to the rolling window and return the trimmed window.

        The oldest entry falls off once the window is full, which is what keeps
        the verbatim turns from growing without bound.
        """
        window = self.get_memory_window(user_id=user_id, session_id=session_id)
        normalised = self._memory_window_entry(entry)
        if normalised.get("turn_index"):
            window = [item for item in window if item.get("turn_index") != normalised["turn_index"]]
        window.append(normalised)
        size = memory_verbatim_turn_count()
        window = window[-size:] if size else []
        now = _utc_now_iso()
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    UPDATE interview_sessions
                    SET memory_window_json = ?, updated_at = ?
                    WHERE user_id = ? AND session_id = ?
                    """,
                    (self._dumps(window), now, user_id, session_id),
                )
        return window

    def turn_exists(self, *, user_id: str, session_id: str, turn_index: int) -> bool:
        """True when this turn index is already taken for the session."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT 1 FROM interview_turns
                WHERE user_id = ? AND session_id = ? AND turn_index = ?
                LIMIT 1
                """,
                (user_id, session_id, int(turn_index)),
            ).fetchone()
        return row is not None

    def record_turn(
        self,
        *,
        user_id: str,
        session_id: str,
        turn_index: int,
        question: str,
        user_answer: str,
        rubric: dict[str, Any],
        strengths: list[str],
        improvements: list[str],
        follow_up_note: str,
        response_time_ms: int,
        degraded: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        now = _utc_now_iso()
        try:
            with self._lock:
                with self._conn:
                    self._conn.execute(
                        """
                        INSERT INTO interview_turns(
                            session_id, user_id, turn_index, question, user_answer,
                            rubric_json, strengths_json, improvements_json,
                            follow_up_note, response_time_ms, degraded, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session_id,
                            user_id,
                            turn_index,
                            question,
                            user_answer,
                            self._dumps(rubric),
                            self._dumps(strengths),
                            self._dumps(improvements),
                            follow_up_note,
                            max(0, int(response_time_ms)),
                            1 if degraded else 0,
                            now,
                        ),
                    )

                    current = self._conn.execute(
                        """
                        SELECT turn_count, turns_completed
                        FROM interview_sessions
                        WHERE user_id = ? AND session_id = ?
                        """,
                        (user_id, session_id),
                    ).fetchone()
                    if current:
                        turns_completed = int(current["turns_completed"]) + 1
                        turn_count = int(current["turn_count"])
                        status = "completed" if turns_completed >= turn_count else "active"
                        self._conn.execute(
                            """
                            UPDATE interview_sessions
                            SET turns_completed = ?, status = ?, current_question = '', updated_at = ?
                            WHERE user_id = ? AND session_id = ?
                            """,
                            (turns_completed, status, now, user_id, session_id),
                        )
        except sqlite3.IntegrityError as exc:
            if "UNIQUE" not in str(exc).upper():
                raise
            # Two concurrent submissions both allocated the same turn_index; the
            # loser must not overwrite or double-count the winner's turn.
            logger.warning(
                "Turn conflict for session=%s turn_index=%s: %s",
                session_id,
                turn_index,
                exc,
            )
            raise TurnConflictError(
                f"turn {turn_index} already recorded for session {session_id}"
            ) from exc

        turn = {
            "session_id": session_id,
            "turn_index": int(turn_index),
            "question": question,
            "user_answer": user_answer,
            "rubric": rubric,
            "strengths": strengths,
            "improvements": improvements,
            "follow_up_note": follow_up_note,
            "response_time_ms": max(0, int(response_time_ms)),
            "degraded": bool(degraded),
            "created_at": now,
        }
        session = self.get_session(user_id=user_id, session_id=session_id)
        return turn, session

    def get_turns(self, *, user_id: str, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT session_id, turn_index, question, user_answer, rubric_json,
                       strengths_json, improvements_json, follow_up_note,
                       response_time_ms, degraded, created_at
                FROM interview_turns
                WHERE user_id = ? AND session_id = ?
                ORDER BY turn_index ASC
                """,
                (user_id, session_id),
            ).fetchall()

        turns: list[dict[str, Any]] = []
        for row in rows:
            turns.append(
                {
                    "session_id": row["session_id"],
                    "turn_index": int(row["turn_index"]),
                    "question": row["question"],
                    "user_answer": row["user_answer"],
                    "rubric": self._loads_dict(row["rubric_json"]),
                    "strengths": self._loads_list(row["strengths_json"]),
                    "improvements": self._loads_list(row["improvements_json"]),
                    "follow_up_note": row["follow_up_note"],
                    "response_time_ms": int(row["response_time_ms"]),
                    "degraded": bool(int(row["degraded"])) if "degraded" in row.keys() else False,
                    "created_at": row["created_at"],
                }
            )
        return turns

    def save_report(
        self,
        *,
        user_id: str,
        session_id: str,
        report: dict[str, Any],
    ) -> dict[str, Any]:
        now = _utc_now_iso()
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO interview_reports(session_id, user_id, report_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(session_id)
                    DO UPDATE SET report_json = excluded.report_json, updated_at = excluded.updated_at
                    """,
                    (
                        session_id,
                        user_id,
                        self._dumps(report),
                        now,
                        now,
                    ),
                )
        return {
            **report,
            "session_id": session_id,
            "created_at": report.get("created_at") or now,
            "updated_at": now,
        }

    def get_report(self, *, user_id: str, session_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            """
            SELECT report_json, created_at, updated_at
            FROM interview_reports
            WHERE user_id = ? AND session_id = ?
            """,
            (user_id, session_id),
        ).fetchone()
        if not row:
            return None
        parsed = self._loads_dict(row["report_json"])
        return {
            **parsed,
            "session_id": session_id,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def get_trends(self, *, user_id: str, limit: int = 50) -> dict[str, Any]:
        rows = self._conn.execute(
            """
            SELECT
                s.session_id,
                s.track,
                s.level,
                s.interview_type,
                s.updated_at AS completed_at,
                r.report_json
            FROM interview_reports r
            JOIN interview_sessions s ON s.session_id = r.session_id AND s.user_id = r.user_id
            WHERE r.user_id = ?
            ORDER BY s.updated_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

        points_desc: list[dict[str, Any]] = []
        for row in rows:
            report = self._loads_dict(row["report_json"])
            overall_score = float(report.get("overall_score") or 0.0)
            rubric_raw = report.get("rubric_averages") if isinstance(report.get("rubric_averages"), dict) else {}
            rubric_averages = {
                "technical_accuracy": float(rubric_raw.get("technical_accuracy") or 0.0),
                "reasoning_depth": float(rubric_raw.get("reasoning_depth") or 0.0),
                "communication_clarity": float(rubric_raw.get("communication_clarity") or 0.0),
                "completeness": float(rubric_raw.get("completeness") or 0.0),
                "confidence_signal": float(rubric_raw.get("confidence_signal") or 0.0),
                "overall": float(rubric_raw.get("overall") or overall_score),
            }
            points_desc.append(
                {
                    "session_id": str(row["session_id"]),
                    "completed_at": str(row["completed_at"]),
                    "overall_score": round(overall_score, 2),
                    "rubric_averages": rubric_averages,
                    "track": str(row["track"]),
                    "level": str(row["level"]),
                    "interview_type": str(row["interview_type"]),
                    "readiness_label": str(report.get("readiness_label") or ""),
                }
            )

        points = list(reversed(points_desc))
        latest_score = float(points[-1]["overall_score"]) if points else 0.0
        previous_score = float(points[-2]["overall_score"]) if len(points) > 1 else 0.0
        delta = round(latest_score - previous_score, 2) if len(points) > 1 else 0.0
        return {
            "points": points,
            "summary": {
                "latest_score": round(latest_score, 2),
                "previous_score": round(previous_score, 2),
                "delta": delta,
                "session_count": len(points),
            },
        }

    def get_stats(self, *, user_id: str) -> dict[str, Any]:
        total_sessions = int(
            self._conn.execute(
                "SELECT COUNT(*) FROM interview_sessions WHERE user_id = ?",
                (user_id,),
            ).fetchone()[0]
        )
        completed_sessions = int(
            self._conn.execute(
                "SELECT COUNT(*) FROM interview_sessions WHERE user_id = ? AND status = 'completed'",
                (user_id,),
            ).fetchone()[0]
        )

        report_rows = self._conn.execute(
            "SELECT report_json FROM interview_reports WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        scores: list[float] = []
        for row in report_rows:
            payload = self._loads_dict(row["report_json"])
            value = payload.get("overall_score")
            if isinstance(value, (int, float)):
                scores.append(float(value))
        readiness = round(sum(scores) / len(scores), 2) if scores else 0.0

        return {
            "total_sessions": total_sessions,
            "completed_sessions": completed_sessions,
            "interview_readiness_score": readiness,
        }
