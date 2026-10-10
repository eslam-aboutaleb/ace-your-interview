"""SQLite-backed timed exam persistence.

Follows the :class:`~app.services.card_store.CardStore`
pattern: a dedicated SQLite connection, a single-thread
executor, and an async ``run_async`` shim so FastAPI
handlers never block the event loop.

Exam sessions live in the same database file as the
learning stores (``settings.learning_db_path``) so the
analytics service can aggregate exam events alongside
``learning_attempts`` and ``fsrs_review_log``.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any, Callable, TypeVar
from uuid import uuid4

T = TypeVar("T")

STATUS_CREATED = "created"
STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETED = "completed"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _to_iso(dt: datetime | str) -> str:
    if isinstance(dt, str):
        dt = _from_iso(dt)
    return dt.astimezone(UTC).isoformat()


def _from_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


class ExamStore:
    """Persist timed exam sessions and their graded answers."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="exam-store"
        )
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

    # ── Schema ─────────────────────────────────────────
    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS exam_sessions (
                    user_id TEXT NOT NULL, exam_id TEXT NOT NULL,
                    title TEXT NOT NULL, question_refs_json TEXT NOT NULL,
                    duration_minutes INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('created','in_progress','completed')),
                    started_at TEXT, completed_at TEXT,
                    score REAL, passed INTEGER,
                    time_exceeded INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, exam_id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS exam_answers (
                    user_id TEXT NOT NULL, exam_id TEXT NOT NULL,
                    question_ref TEXT NOT NULL, answer_text TEXT NOT NULL,
                    is_correct INTEGER, score REAL, graded_by TEXT,
                    rubric_json TEXT, feedback TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, exam_id, question_ref)
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_exam_sessions_user_created "
                "ON exam_sessions(user_id, created_at DESC)"
            )

    # ── Row mapping ────────────────────────────────────
    @staticmethod
    def _session_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        questions: list[dict[str, Any]] = []
        try:
            raw = json.loads(str(row["question_refs_json"] or "[]"))
            if isinstance(raw, list):
                questions = [dict(item) for item in raw]
        except json.JSONDecodeError:
            questions = []
        score = row["score"]
        passed = row["passed"]
        return {
            "exam_id": str(row["exam_id"]),
            "title": str(row["title"]),
            "status": str(row["status"]),
            "duration_minutes": int(row["duration_minutes"]),
            "questions": questions,
            "question_count": len(questions),
            "started_at": (
                str(row["started_at"]) if row["started_at"] is not None else None
            ),
            "completed_at": (
                str(row["completed_at"])
                if row["completed_at"] is not None
                else None
            ),
            "score": float(score) if score is not None else None,
            "passed": bool(passed) if passed is not None else None,
            "time_exceeded": int(row["time_exceeded"] or 0),
            "created_at": str(row["created_at"]),
        }

    @staticmethod
    def _answer_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        rubric: dict[str, Any] | None = None
        raw = row["rubric_json"]
        if raw is not None:
            try:
                parsed = json.loads(str(raw))
                if isinstance(parsed, dict):
                    rubric = parsed
            except json.JSONDecodeError:
                rubric = None
        is_correct = row["is_correct"]
        return {
            "question_ref": str(row["question_ref"]),
            "answer_text": str(row["answer_text"]),
            "is_correct": (
                bool(is_correct) if is_correct is not None else None
            ),
            "score": float(row["score"]) if row["score"] is not None else 0.0,
            "graded_by": str(row["graded_by"] or ""),
            "rubric": rubric,
            "feedback": str(row["feedback"] or ""),
            "created_at": str(row["created_at"]),
        }

    # ── Sessions ───────────────────────────────────────
    def create_exam(
        self,
        *,
        user_id: str,
        title: str,
        questions: list[dict[str, Any]],
        duration_minutes: int,
    ) -> dict[str, Any]:
        exam_id = uuid4().hex
        now_iso = _to_iso(_utc_now())
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO exam_sessions(
                        user_id, exam_id, title, question_refs_json,
                        duration_minutes, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        exam_id,
                        str(title)[:200],
                        json.dumps(questions, ensure_ascii=True),
                        int(duration_minutes),
                        STATUS_CREATED,
                        now_iso,
                    ),
                )
        exam = self.get_exam(user_id=user_id, exam_id=exam_id)
        assert exam is not None
        return exam

    def get_exam(self, *, user_id: str, exam_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            """
            SELECT * FROM exam_sessions
            WHERE user_id = ? AND exam_id = ?
            """,
            (user_id, exam_id),
        ).fetchone()
        return self._session_row_to_dict(row) if row else None

    def list_exams(
        self, *, user_id: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT * FROM exam_sessions
            WHERE user_id = ?
            ORDER BY created_at DESC, exam_id ASC
            LIMIT ?
            """,
            (user_id, int(limit)),
        ).fetchall()
        return [self._session_row_to_dict(row) for row in rows]

    def start_exam(
        self, *, user_id: str, exam_id: str, started_at: datetime | None = None
    ) -> dict[str, Any] | None:
        """Stamp the server-side start time and flip to ``in_progress``."""
        moment = started_at or _utc_now()
        started_iso = _to_iso(moment)
        with self._lock:
            with self._conn:
                cur = self._conn.execute(
                    """
                    UPDATE exam_sessions
                    SET status = ?, started_at = ?
                    WHERE user_id = ? AND exam_id = ? AND status = ?
                    """,
                    (
                        STATUS_IN_PROGRESS,
                        started_iso,
                        user_id,
                        exam_id,
                        STATUS_CREATED,
                    ),
                )
        if cur.rowcount == 0:
            return None
        return self.get_exam(user_id=user_id, exam_id=exam_id)

    def record_answer(
        self,
        *,
        user_id: str,
        exam_id: str,
        question_ref: str,
        answer_text: str,
        is_correct: bool | None,
        score: float,
        graded_by: str,
        rubric: dict[str, Any] | None,
        feedback: str,
        created_at: datetime | None = None,
    ) -> None:
        moment = created_at or _utc_now()
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO exam_answers(
                        user_id, exam_id, question_ref, answer_text,
                        is_correct, score, graded_by, rubric_json,
                        feedback, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, exam_id, question_ref) DO UPDATE SET
                        answer_text = excluded.answer_text,
                        is_correct = excluded.is_correct,
                        score = excluded.score,
                        graded_by = excluded.graded_by,
                        rubric_json = excluded.rubric_json,
                        feedback = excluded.feedback
                    """,
                    (
                        user_id,
                        exam_id,
                        str(question_ref),
                        str(answer_text),
                        None if is_correct is None else (1 if is_correct else 0),
                        float(score),
                        str(graded_by),
                        json.dumps(rubric, ensure_ascii=True)
                        if rubric is not None
                        else None,
                        str(feedback),
                        _to_iso(moment),
                    ),
                )

    def get_answers(
        self, *, user_id: str, exam_id: str
    ) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT * FROM exam_answers
            WHERE user_id = ? AND exam_id = ?
            ORDER BY created_at ASC, question_ref ASC
            """,
            (user_id, exam_id),
        ).fetchall()
        return [self._answer_row_to_dict(row) for row in rows]

    def complete_exam(
        self,
        *,
        user_id: str,
        exam_id: str,
        completed_at: datetime | None = None,
        score: float,
        passed: bool,
        time_exceeded: bool,
    ) -> dict[str, Any] | None:
        moment = completed_at or _utc_now()
        completed_iso = _to_iso(moment)
        with self._lock:
            with self._conn:
                cur = self._conn.execute(
                    """
                    UPDATE exam_sessions
                    SET status = ?, completed_at = ?, score = ?,
                        passed = ?, time_exceeded = ?
                    WHERE user_id = ? AND exam_id = ? AND status = ?
                    """,
                    (
                        STATUS_COMPLETED,
                        completed_iso,
                        float(score),
                        1 if passed else 0,
                        1 if time_exceeded else 0,
                        user_id,
                        exam_id,
                        STATUS_IN_PROGRESS,
                    ),
                )
        if cur.rowcount == 0:
            return None
        return self.get_exam(user_id=user_id, exam_id=exam_id)

    def update_score(
        self, *, user_id: str, exam_id: str, score: float, passed: bool
    ) -> dict[str, Any] | None:
        """Persist a remark re-grade result (both scores stay visible)."""
        with self._lock:
            with self._conn:
                cur = self._conn.execute(
                    """
                    UPDATE exam_sessions
                    SET score = ?, passed = ?
                    WHERE user_id = ? AND exam_id = ? AND status = ?
                    """,
                    (
                        float(score),
                        1 if passed else 0,
                        user_id,
                        exam_id,
                        STATUS_COMPLETED,
                    ),
                )
        if cur.rowcount == 0:
            return None
        return self.get_exam(user_id=user_id, exam_id=exam_id)

    # ── Analytics support ──────────────────────────────
    def completed_exam_days(
        self, *, user_id: str
    ) -> list[dict[str, Any]]:
        """Completion dates of completed exams, for the activity heatmap."""
        rows = self._conn.execute(
            """
            SELECT completed_at, COUNT(*) AS count
            FROM exam_sessions
            WHERE user_id = ? AND status = ? AND completed_at IS NOT NULL
            GROUP BY completed_at
            """,
            (user_id, STATUS_COMPLETED),
        ).fetchall()
        return [
            {"completed_at": str(row["completed_at"]), "count": int(row["count"])}
            for row in rows
        ]
