"""SQLite-backed adaptive learning persistence and scheduling."""

from __future__ import annotations

import os
import sqlite3
import threading
from datetime import UTC, datetime, timedelta


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _to_iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


def _from_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


class LearningStore:
    """Persist learning attempts and maintain per-question review state."""

    _INTERVAL_DAYS = [0, 1, 3, 7, 14, 30]

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
                    created_at TEXT NOT NULL
                )
                """
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

    @staticmethod
    def _event_score(is_correct: bool, confidence: int) -> float:
        confidence_norm = max(1, min(confidence, 5)) / 5.0
        calibrated = confidence_norm if is_correct else (1.0 - confidence_norm)
        return (0.7 * (1.0 if is_correct else 0.0)) + (0.3 * calibrated)

    def _next_bucket(self, previous: int, is_correct: bool, confidence: int) -> int:
        if not is_correct:
            return 0
        bump = 2 if confidence >= 4 else 1
        return min(5, max(0, previous + bump))

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
    ) -> dict:
        now = _utc_now()
        now_iso = _to_iso(now)
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO learning_attempts(
                        user_id, question_id, topic_id, user_answer, is_correct,
                        confidence, response_time_ms, mode, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        now_iso,
                    ),
                )
                attempt_id = int(self._conn.execute("SELECT last_insert_rowid()").fetchone()[0])
                existing = self._conn.execute(
                    """
                    SELECT mastery_score, due_at, review_bucket, attempts, correct_attempts, avg_confidence
                    FROM question_progress
                    WHERE user_id = ? AND question_id = ? AND topic_id = ?
                    """,
                    (user_id, question_id, topic_id),
                ).fetchone()

                event_score = self._event_score(is_correct, confidence)
                if existing:
                    attempts = int(existing["attempts"]) + 1
                    correct_attempts = int(existing["correct_attempts"]) + (1 if is_correct else 0)
                    old_mastery = float(existing["mastery_score"])
                    mastery = max(0.0, min(1.0, (0.7 * old_mastery) + (0.3 * event_score)))
                    bucket = self._next_bucket(int(existing["review_bucket"]), is_correct, confidence)
                    old_avg_conf = float(existing["avg_confidence"])
                    avg_confidence = ((old_avg_conf * (attempts - 1)) + confidence) / attempts
                    self._conn.execute(
                        """
                        UPDATE question_progress
                        SET mastery_score = ?, due_at = ?, review_bucket = ?, attempts = ?,
                            correct_attempts = ?, avg_confidence = ?, last_confidence = ?, updated_at = ?
                        WHERE user_id = ? AND question_id = ? AND topic_id = ?
                        """,
                        (
                            mastery,
                            "",  # placeholder, updated below
                            bucket,
                            attempts,
                            correct_attempts,
                            avg_confidence,
                            confidence,
                            now_iso,
                            user_id,
                            question_id,
                            topic_id,
                        ),
                    )
                else:
                    attempts = 1
                    correct_attempts = 1 if is_correct else 0
                    mastery = max(0.0, min(1.0, (0.4 * 0.7) + (0.3 * event_score)))
                    bucket = self._next_bucket(0, is_correct, confidence)
                    avg_confidence = float(confidence)
                    self._conn.execute(
                        """
                        INSERT INTO question_progress(
                            user_id, question_id, topic_id, mastery_score, due_at, review_bucket,
                            attempts, correct_attempts, avg_confidence, last_confidence, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            user_id,
                            question_id,
                            topic_id,
                            mastery,
                            "",  # placeholder, updated below
                            bucket,
                            attempts,
                            correct_attempts,
                            avg_confidence,
                            confidence,
                            now_iso,
                        ),
                    )

                interval_days = self._INTERVAL_DAYS[bucket]
                due_at = now + timedelta(days=interval_days)
                due_at_iso = _to_iso(due_at)
                self._conn.execute(
                    """
                    UPDATE question_progress
                    SET due_at = ?
                    WHERE user_id = ? AND question_id = ? AND topic_id = ?
                    """,
                    (due_at_iso, user_id, question_id, topic_id),
                )

        return {
            "attempt_id": attempt_id,
            "topic_id": topic_id,
            "question_id": question_id,
            "mastery_score": mastery,
            "due_at": due_at_iso,
            "review_bucket": bucket,
        }

    def get_review_queue(self, *, user_id: str, limit: int = 50) -> dict:
        now_iso = _to_iso(_utc_now())
        rows = self._conn.execute(
            """
            SELECT topic_id, question_id, due_at, mastery_score, last_confidence, review_bucket, attempts
            FROM question_progress
            WHERE user_id = ? AND due_at <= ?
            ORDER BY mastery_score ASC, due_at ASC
            LIMIT ?
            """,
            (user_id, now_iso, limit),
        ).fetchall()
        items = [
            {
                "topic_id": row["topic_id"],
                "question_id": row["question_id"],
                "due_at": row["due_at"],
                "mastery_score": float(row["mastery_score"]),
                "last_confidence": int(row["last_confidence"]),
                "review_bucket": int(row["review_bucket"]),
                "attempts": int(row["attempts"]),
            }
            for row in rows
        ]
        return {"items": items, "total_due": len(items)}

    def get_weak_areas(self, *, user_id: str, limit: int = 10) -> dict:
        rows = self._conn.execute(
            """
            SELECT
                topic_id,
                SUM(attempts) AS attempts,
                CASE WHEN SUM(attempts) > 0 THEN (SUM(correct_attempts) * 1.0 / SUM(attempts)) ELSE 0 END AS accuracy,
                AVG(avg_confidence) AS avg_confidence,
                AVG(mastery_score) AS mastery_score,
                SUM(CASE WHEN due_at <= ? THEN 1 ELSE 0 END) AS due_count
            FROM question_progress
            WHERE user_id = ?
            GROUP BY topic_id
            ORDER BY mastery_score ASC, accuracy ASC, attempts DESC
            LIMIT ?
            """,
            (_to_iso(_utc_now()), user_id, limit),
        ).fetchall()
        weak_areas = [
            {
                "topic_id": row["topic_id"],
                "attempts": int(row["attempts"] or 0),
                "accuracy": round(float(row["accuracy"] or 0.0), 4),
                "avg_confidence": round(float(row["avg_confidence"] or 0.0), 2),
                "mastery_score": round(float(row["mastery_score"] or 0.0), 4),
                "due_count": int(row["due_count"] or 0),
            }
            for row in rows
        ]
        return {"weak_areas": weak_areas}

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

