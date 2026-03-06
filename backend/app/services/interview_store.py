"""SQLite-backed persistence for mock interview sessions, turns, and reports."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class InterviewStore:
    """Persist interview sessions and turn-by-turn rubric feedback."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._lock = threading.Lock()
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
                    job_description_text TEXT NOT NULL,
                    resume_summary_text TEXT NOT NULL,
                    focus_areas_json TEXT NOT NULL,
                    asked_questions_json TEXT NOT NULL,
                    current_question TEXT NOT NULL,
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
    def _dumps(data: Any) -> str:
        return json.dumps(data, ensure_ascii=True)

    def _session_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "session_id": row["session_id"],
            "track": row["track"],
            "level": row["level"],
            "interview_type": row["interview_type"],
            "turn_count": int(row["turn_count"]),
            "turns_completed": int(row["turns_completed"]),
            "status": row["status"],
            "target_role": row["target_role"],
            "focus_areas": self._loads_list(row["focus_areas_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "current_question": row["current_question"],
            "report_ready": bool(int(row["report_ready"])) if "report_ready" in row.keys() else False,
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
                        target_role, job_description_text, resume_summary_text,
                        focus_areas_json, asked_questions_json,
                        current_question, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        job_description_text,
                        resume_summary_text,
                        self._dumps(focus),
                        self._dumps([]),
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
            "focus_areas": focus,
            "created_at": now,
            "updated_at": now,
            "current_question": "",
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
        return {
            **self._session_row_to_dict(row),
            "job_description_text": row["job_description_text"],
            "resume_summary_text": row["resume_summary_text"],
            "asked_questions": self._loads_list(row["asked_questions_json"]),
        }

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
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        now = _utc_now_iso()
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO interview_turns(
                        session_id, user_id, turn_index, question, user_answer,
                        rubric_json, strengths_json, improvements_json,
                        follow_up_note, response_time_ms, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            "created_at": now,
        }
        session = self.get_session(user_id=user_id, session_id=session_id)
        return turn, session

    def get_turns(self, *, user_id: str, session_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT session_id, turn_index, question, user_answer, rubric_json,
                   strengths_json, improvements_json, follow_up_note,
                   response_time_ms, created_at
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
