"""SQLite-backed adaptive learning persistence and scheduling."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from typing import Any

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
        rows = self._conn.execute(
            """
            SELECT
                topic_id,
                SUM(attempts) AS attempts,
                SUM(correct_attempts) AS correct_attempts,
                AVG(mastery_score) AS mastery_score,
                SUM(CASE WHEN due_at <= ? THEN 1 ELSE 0 END) AS due_count
            FROM question_progress
            WHERE user_id = ?
            GROUP BY topic_id
            ORDER BY topic_id ASC
            """,
            (today_iso, user_id),
        ).fetchall()

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
                "reason": f"Validate retention and confidence after focused review.",
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
                "response_detail": "concise",
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
            "response_detail": "concise",
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
            else (existing or {}).get("response_detail", "concise")
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
            (prefs or {}).get("response_detail", "concise")
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
