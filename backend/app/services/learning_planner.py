"""Goal-aware learner profile, diagnostics, and recommendations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from typing import Any, Callable, TypeVar
from zoneinfo import ZoneInfo

from app.services import fsrs_scheduler
from app.services.gamification_store import (
    DEFAULT_DAILY_GOAL,
    add_days,
    today_in_timezone,
)
from app.services.learning_store import accuracy_uncertainty

T = TypeVar("T")

_TRACKS = ("backend", "frontend", "system_design", "ai_stack")
_LEVELS = ("junior", "mid", "senior")
_MODALITIES = ("study", "quiz", "interview", "voice")
_DEFAULT_MODALITIES = ["study", "quiz", "interview"]
_DEFAULT_CONFIDENCE = {track: 3 for track in _TRACKS}
_COLD_START_TOPICS = {
    "backend": [
        "01-backend-fundamentals-and-http",
        "02-backend-api-design-and-contracts",
        "03-backend-data-modeling-and-persistence",
    ],
    "frontend": [
        "06-frontend-core-architecture",
        "07-frontend-state-data-fetching",
        "08-frontend-performance-accessibility",
    ],
    "system_design": [
        "10-system-design-foundations",
        "11-system-design-scaling-and-reliability",
        "12-system-design-data-consistency-and-tradeoffs",
    ],
    "ai_stack": [
        "13-ai-stack-llm-and-prompting",
        "14-ai-stack-rag-and-evaluation",
        "15-ai-stack-agents-tools-and-guardrails",
    ],
}
_INTERVIEW_DIMENSIONS = (
    ("technical_accuracy", "Technical Accuracy"),
    ("reasoning_depth", "Reasoning Depth"),
    ("communication_clarity", "Communication Clarity"),
    ("completeness", "Completeness"),
    ("confidence_signal", "Confidence Signal"),
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _to_iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


def _loads_json_list(value: str) -> list[Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _loads_json_dict(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


# ── Plan commitment ─────────────────────────────────────────────
# The 7-day plan used to be rebuilt identically on every load with no
# record of what had actually been done, so it was not a commitment
# device. Scheduled tasks are now persisted in ``plan_tasks`` under a
# stable key, and the daily load is shaped by the learner's streak.

#: Learning-event types that count towards the daily goal. Mirrors
#: ``gamification_store.LEARNING_EVENT_TYPES`` without importing the
#: tuple, so a rename there cannot silently change this count.
_COMMITMENT_EVENT_TYPES = ("attempt", "review")


def plan_task_key(
    *, task_date: str, task_type: str, topic_id: str, occurrence: int = 1
) -> str:
    """Stable identity for one scheduled plan task.

    The plan is regenerated on every load, so a completion only
    survives regeneration when the key is derived from *what* the
    task is and *when* it is scheduled — never from volatile fields
    such as the recommendation title, its priority, or the cursor
    position inside the plan. The hash of
    ``task_date | task_type | topic_id | occurrence`` is stable
    across regenerations and unique per scheduled slot:
    ``occurrence`` counts how many tasks with the same
    (type, topic) already fill that day, which matters for
    topic-less recommendations (an interview rec can legitimately
    fill several slots in the same day).
    """
    raw = f"{task_date}|{task_type}|{str(topic_id or '')}|{int(occurrence)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _streak_adjusted_daily_items(daily_items: int, *, behind: bool) -> int:
    """Halve the daily item count (rounding up, minimum 1) when behind.

    A learner who has missed days gets a smaller, restart-sized
    commitment instead of the full load they already failed to
    clear, so the plan nudges the habit back instead of
    overwhelming it. ``ceil(n / 2)`` keeps 3 → 2 and 1 → 1.
    """
    if not behind:
        return daily_items
    return max(1, (int(daily_items) + 1) // 2)


class LearningPlannerStore:
    """Persist learner profile data and rank next-best actions."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="learning-planner")
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    async def run_async(self, fn: Callable[..., T], /, *args, **kwargs) -> T:
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
                CREATE TABLE IF NOT EXISTS learner_profiles (
                    user_id TEXT PRIMARY KEY,
                    target_role TEXT NOT NULL,
                    target_date TEXT NOT NULL,
                    weekly_minutes INTEGER NOT NULL,
                    preferred_session_minutes INTEGER NOT NULL,
                    current_level TEXT NOT NULL,
                    target_level TEXT NOT NULL,
                    primary_track TEXT NOT NULL,
                    focus_topic_ids_json TEXT NOT NULL,
                    target_companies_json TEXT NOT NULL,
                    preferred_modalities_json TEXT NOT NULL,
                    confidence_by_track_json TEXT NOT NULL,
                    timezone TEXT NOT NULL DEFAULT 'UTC',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    diagnostic_updated_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS learner_competency_scores (
                    user_id TEXT NOT NULL,
                    competency_id TEXT NOT NULL,
                    competency_type TEXT NOT NULL,
                    label TEXT NOT NULL,
                    track TEXT NOT NULL,
                    score REAL NOT NULL,
                    confidence_gap REAL NOT NULL,
                    uncertainty REAL NOT NULL DEFAULT 0.0,
                    evidence_count INTEGER NOT NULL,
                    priority REAL NOT NULL,
                    source TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, competency_id)
                )
                """
            )
            profile_columns = {
                str(row["name"])
                for row in self._conn.execute(
                    "PRAGMA table_info(learner_profiles)"
                ).fetchall()
            }
            if "timezone" not in profile_columns:
                self._conn.execute(
                    "ALTER TABLE learner_profiles "
                    "ADD COLUMN timezone TEXT NOT NULL DEFAULT 'UTC'"
                )
            existing_columns = {
                str(row["name"])
                for row in self._conn.execute(
                    "PRAGMA table_info(learner_competency_scores)"
                ).fetchall()
            }
            if "uncertainty" not in existing_columns:
                self._conn.execute(
                    "ALTER TABLE learner_competency_scores "
                    "ADD COLUMN uncertainty REAL NOT NULL DEFAULT 0.0"
                )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_learner_competencies_priority
                ON learner_competency_scores(user_id, priority DESC, updated_at DESC)
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS plan_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    task_date TEXT NOT NULL,
                    task_key TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    completed_at TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE (user_id, task_date, task_key)
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_plan_tasks_user_date
                ON plan_tasks(user_id, task_date)
                """
            )

    def _default_profile(self) -> dict[str, Any]:
        now_iso = _to_iso(_utc_now())
        return {
            "target_role": "",
            "target_date": "",
            "weekly_minutes": 240,
            "preferred_session_minutes": 30,
            "current_level": "mid",
            "target_level": "senior",
            "primary_track": "backend",
            "focus_topic_ids": [],
            "target_companies": [],
            "preferred_modalities": list(_DEFAULT_MODALITIES),
            "confidence_by_track": dict(_DEFAULT_CONFIDENCE),
            "timezone": "UTC",
            "created_at": now_iso,
            "updated_at": now_iso,
            "diagnostic_updated_at": "",
        }

    @staticmethod
    def _sanitize_date(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        try:
            return date.fromisoformat(text).isoformat()
        except ValueError:
            return ""

    @staticmethod
    def _sanitize_level(value: Any, fallback: str) -> str:
        level = str(value or "").strip().lower()
        return level if level in _LEVELS else fallback

    @staticmethod
    def _sanitize_track(value: Any, fallback: str) -> str:
        track = str(value or "").strip().lower()
        return track if track in _TRACKS else fallback

    @staticmethod
    def _sanitize_minutes(value: Any, *, fallback: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = fallback
        return max(minimum, min(maximum, parsed))

    @staticmethod
    def _sanitize_string_list(value: Any, *, limit: int, max_len: int) -> list[str]:
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if not text:
                continue
            trimmed = text[:max_len]
            if trimmed not in out:
                out.append(trimmed)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _sanitize_modalities(value: Any) -> list[str]:
        items = LearningPlannerStore._sanitize_string_list(value, limit=4, max_len=20)
        out = [item for item in items if item in _MODALITIES]
        return out or list(_DEFAULT_MODALITIES)

    @staticmethod
    def _sanitize_confidence_map(value: Any) -> dict[str, int]:
        base = dict(_DEFAULT_CONFIDENCE)
        if not isinstance(value, dict):
            return base
        for track in _TRACKS:
            raw = value.get(track)
            try:
                parsed = int(raw)
            except (TypeError, ValueError):
                parsed = base[track]
            base[track] = max(1, min(5, parsed))
        return base

    def _row_to_profile(self, row: sqlite3.Row | None) -> dict[str, Any]:
        if row is None:
            return self._default_profile()
        return {
            "target_role": str(row["target_role"]),
            "target_date": str(row["target_date"]),
            "weekly_minutes": int(row["weekly_minutes"]),
            "preferred_session_minutes": int(row["preferred_session_minutes"]),
            "current_level": str(row["current_level"]),
            "target_level": str(row["target_level"]),
            "primary_track": str(row["primary_track"]),
            "focus_topic_ids": [str(v) for v in _loads_json_list(str(row["focus_topic_ids_json"]))],
            "target_companies": [str(v) for v in _loads_json_list(str(row["target_companies_json"]))],
            "preferred_modalities": [str(v) for v in _loads_json_list(str(row["preferred_modalities_json"]))],
            "confidence_by_track": self._sanitize_confidence_map(
                _loads_json_dict(str(row["confidence_by_track_json"]))
            ),
            "timezone": str(row["timezone"] or "UTC"),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "diagnostic_updated_at": str(row["diagnostic_updated_at"]),
        }

    def get_profile(self, *, user_id: str) -> dict[str, Any]:
        row = self._conn.execute(
            """
            SELECT *
            FROM learner_profiles
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
        if row is not None:
            return self._row_to_profile(row)

        default = self._default_profile()
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO learner_profiles(
                        user_id, target_role, target_date, weekly_minutes,
                        preferred_session_minutes, current_level, target_level,
                        primary_track, focus_topic_ids_json, target_companies_json,
                        preferred_modalities_json, confidence_by_track_json,
                        timezone, created_at, updated_at, diagnostic_updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id) DO NOTHING
                    """,
                    (
                        user_id,
                        default["target_role"],
                        default["target_date"],
                        default["weekly_minutes"],
                        default["preferred_session_minutes"],
                        default["current_level"],
                        default["target_level"],
                        default["primary_track"],
                        json.dumps(default["focus_topic_ids"], ensure_ascii=True),
                        json.dumps(default["target_companies"], ensure_ascii=True),
                        json.dumps(default["preferred_modalities"], ensure_ascii=True),
                        json.dumps(default["confidence_by_track"], ensure_ascii=True),
                        default["timezone"],
                        default["created_at"],
                        default["updated_at"],
                        default["diagnostic_updated_at"],
                    ),
                )
        return default

    def upsert_profile(self, *, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        existing = self.get_profile(user_id=user_id)
        now_iso = _to_iso(_utc_now())
        target_role = str(payload.get("target_role", existing["target_role"]) or "").strip()[:200]
        target_date = self._sanitize_date(payload.get("target_date", existing["target_date"]))
        weekly_minutes = self._sanitize_minutes(
            payload.get("weekly_minutes", existing["weekly_minutes"]),
            fallback=int(existing["weekly_minutes"]),
            minimum=30,
            maximum=2400,
        )
        preferred_session_minutes = self._sanitize_minutes(
            payload.get("preferred_session_minutes", existing["preferred_session_minutes"]),
            fallback=int(existing["preferred_session_minutes"]),
            minimum=10,
            maximum=180,
        )
        current_level = self._sanitize_level(
            payload.get("current_level", existing["current_level"]),
            fallback=str(existing["current_level"]),
        )
        target_level = self._sanitize_level(
            payload.get("target_level", existing["target_level"]),
            fallback=str(existing["target_level"]),
        )
        primary_track = self._sanitize_track(
            payload.get("primary_track", existing["primary_track"]),
            fallback=str(existing["primary_track"]),
        )
        focus_topic_ids = self._sanitize_string_list(
            payload.get("focus_topic_ids", existing["focus_topic_ids"]),
            limit=12,
            max_len=120,
        )
        target_companies = self._sanitize_string_list(
            payload.get("target_companies", existing["target_companies"]),
            limit=12,
            max_len=120,
        )
        preferred_modalities = self._sanitize_modalities(
            payload.get("preferred_modalities", existing["preferred_modalities"])
        )
        confidence_by_track = self._sanitize_confidence_map(
            payload.get("confidence_by_track", existing["confidence_by_track"])
        )
        timezone = str(
            payload.get("timezone", existing.get("timezone") or "UTC") or "UTC"
        ).strip()[:64] or "UTC"

        created_at = existing["created_at"] or now_iso
        diagnostic_updated_at = existing["diagnostic_updated_at"]
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO learner_profiles(
                        user_id, target_role, target_date, weekly_minutes,
                        preferred_session_minutes, current_level, target_level,
                        primary_track, focus_topic_ids_json, target_companies_json,
                        preferred_modalities_json, confidence_by_track_json,
                        timezone, created_at, updated_at, diagnostic_updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        target_role = excluded.target_role,
                        target_date = excluded.target_date,
                        weekly_minutes = excluded.weekly_minutes,
                        preferred_session_minutes = excluded.preferred_session_minutes,
                        current_level = excluded.current_level,
                        target_level = excluded.target_level,
                        primary_track = excluded.primary_track,
                        focus_topic_ids_json = excluded.focus_topic_ids_json,
                        target_companies_json = excluded.target_companies_json,
                        preferred_modalities_json = excluded.preferred_modalities_json,
                        confidence_by_track_json = excluded.confidence_by_track_json,
                        timezone = excluded.timezone,
                        updated_at = excluded.updated_at
                    """,
                    (
                        user_id,
                        target_role,
                        target_date,
                        weekly_minutes,
                        preferred_session_minutes,
                        current_level,
                        target_level,
                        primary_track,
                        json.dumps(focus_topic_ids, ensure_ascii=True),
                        json.dumps(target_companies, ensure_ascii=True),
                        json.dumps(preferred_modalities, ensure_ascii=True),
                        json.dumps(confidence_by_track, ensure_ascii=True),
                        timezone,
                        created_at,
                        now_iso,
                        diagnostic_updated_at,
                    ),
                )
        return self.get_profile(user_id=user_id)

    def _table_exists(self, table_name: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
            (table_name,),
        ).fetchone()
        return row is not None

    @staticmethod
    def _topic_track(topic_id: str) -> str:
        topic = str(topic_id or "").strip().lower()
        if topic.startswith("custom-"):
            return "backend"
        if topic.startswith("00-"):
            return "backend"
        if topic.startswith(("01-", "02-", "03-", "04-", "05-")):
            return "backend"
        if topic.startswith(("06-", "07-", "08-", "09-")):
            return "frontend"
        if topic.startswith(("10-", "11-", "12-", "17-", "18-", "19-", "20-", "21-", "22-")):
            return "system_design"
        if topic.startswith(("13-", "14-", "15-", "16-")):
            return "ai_stack"
        return "backend"

    @staticmethod
    def _topic_label(topic_id: str) -> str:
        text = str(topic_id or "").strip()
        if not text:
            return "Topic"
        if ":" in text:
            text = text.split(":", 1)[0]
        parts = text.split("-")
        while parts and parts[0].isdigit():
            parts = parts[1:]
        if not parts:
            parts = text.split("-")
        return " ".join(parts).replace("_", " ").strip().title() or text

    @staticmethod
    def _deadline_days(target_date: str) -> int | None:
        if not target_date:
            return None
        try:
            target = date.fromisoformat(target_date)
        except ValueError:
            return None
        return (target - _utc_now().date()).days

    def _question_progress_rows(self, *, user_id: str) -> list[dict[str, Any]]:
        """Per-topic learning evidence, read from the FSRS tables.

        The planner consumes the consolidated FSRS scheduler
        (``fsrs_cards`` + ``fsrs_review_log``) instead of the
        legacy ``question_progress`` buckets. ``attempts`` is the
        total review count (``reps``), ``correct_attempts`` the
        successful reviews (``reps - lapses``), and
        ``mastery_score`` the mean FSRS retention estimate. FSRS
        does not track self-reported confidence, so a neutral 3
        keeps the confidence-gap term meaningful.
        """
        if not self._table_exists("fsrs_cards"):
            return []
        now_iso = _to_iso(_utc_now())
        # Interview-sourced cards are excluded: their pseudo-topic
        # (``interview-competency``) is a skill bucket, not curriculum, so a
        # plan must never be built around it.
        cards = self._conn.execute(
            """
            SELECT topic_id, card_id, stability, difficulty,
                   due_at, last_review_at, reps, lapses, suspended,
                   source_type
            FROM fsrs_cards
            WHERE user_id = ? AND source_type != 'interview'
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
                and str(card["due_at"] or "") <= now_iso
            ):
                entry["due_count"] += 1
        rows: list[dict[str, Any]] = []
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
                    "avg_confidence": 3.0,
                    "due_count": entry["due_count"],
                }
            )
        rows.sort(key=lambda row: row["topic_id"])
        return rows

    def _latest_interview_reports(self, *, user_id: str, limit: int = 5) -> list[dict[str, Any]]:
        if not self._table_exists("interview_reports"):
            return []
        rows = self._conn.execute(
            """
            SELECT report_json
            FROM interview_reports
            WHERE user_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (user_id, max(1, limit)),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            parsed = _loads_json_dict(str(row["report_json"]))
            if parsed:
                out.append(parsed)
        return out

    def _replace_competencies(self, *, user_id: str, competencies: list[dict[str, Any]], updated_at: str) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "DELETE FROM learner_competency_scores WHERE user_id = ?",
                    (user_id,),
                )
                for item in competencies:
                    self._conn.execute(
                        """
                        INSERT INTO learner_competency_scores(
                            user_id, competency_id, competency_type, label, track, score,
                            confidence_gap, uncertainty, evidence_count, priority, source,
                            metadata_json, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            user_id,
                            str(item["competency_id"]),
                            str(item["competency_type"]),
                            str(item["label"]),
                            str(item.get("track", "")),
                            float(item["score"]),
                            float(item.get("confidence_gap", 0.0)),
                            float(item.get("uncertainty", 0.0)),
                            int(item.get("evidence_count", 0)),
                            float(item.get("priority", 0.0)),
                            str(item.get("source", "diagnostic")),
                            json.dumps(item.get("metadata", {}), ensure_ascii=True),
                            updated_at,
                        ),
                    )
                self._conn.execute(
                    """
                    UPDATE learner_profiles
                    SET diagnostic_updated_at = ?, updated_at = ?
                    WHERE user_id = ?
                    """,
                    (updated_at, updated_at, user_id),
                )

    def list_competencies(self, *, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT *
            FROM learner_competency_scores
            WHERE user_id = ?
            ORDER BY priority DESC, score ASC, label ASC
            LIMIT ?
            """,
            (user_id, max(1, limit)),
        ).fetchall()
        return [
            {
                "competency_id": str(row["competency_id"]),
                "competency_type": str(row["competency_type"]),
                "label": str(row["label"]),
                "track": str(row["track"]),
                "score": round(float(row["score"]), 2),
                "confidence_gap": round(float(row["confidence_gap"]), 2),
                "uncertainty": round(float(row["uncertainty"]), 4),
                "evidence_count": int(row["evidence_count"]),
                "priority": round(float(row["priority"]), 2),
                "source": str(row["source"]),
                "metadata": _loads_json_dict(str(row["metadata_json"])),
                "updated_at": str(row["updated_at"]),
            }
            for row in rows
        ]

    def run_diagnostic(self, *, user_id: str) -> dict[str, Any]:
        profile = self.get_profile(user_id=user_id)
        now_iso = _to_iso(_utc_now())
        rows = self._question_progress_rows(user_id=user_id)
        max_due = max([int(row["due_count"] or 0) for row in rows] or [0])
        topic_entries: list[dict[str, Any]] = []

        for row in rows:
            topic_id = str(row["topic_id"])
            attempts = int(row["attempts"] or 0)
            correct_attempts = int(row["correct_attempts"] or 0)
            mastery = max(0.0, min(1.0, float(row["mastery_score"] or 0.0)))
            score = round(mastery * 100.0, 2)
            avg_confidence = float(row["avg_confidence"] or 0.0)
            confidence_pct = max(0.0, min(100.0, (avg_confidence / 5.0) * 100.0))
            confidence_gap = round(abs(confidence_pct - score), 2)
            due_count = max(0, int(row["due_count"] or 0))
            due_pressure = (due_count / max_due) if max_due > 0 else 0.0
            accuracy = (correct_attempts / attempts) if attempts > 0 else 0.0
            uncertainty = round(
                accuracy_uncertainty(correct_attempts, attempts), 4
            )
            priority = min(
                100.0,
                round(
                    ((100.0 - score) * 0.62)
                    + ((1.0 - accuracy) * 22.0)
                    + (due_pressure * 16.0)
                    + min(confidence_gap, 40.0) * 0.25,
                    2,
                ),
            )
            topic_entries.append(
                {
                    "competency_id": f"topic:{topic_id}",
                    "competency_type": "topic",
                    "label": self._topic_label(topic_id),
                    "track": self._topic_track(topic_id),
                    "score": score,
                    "confidence_gap": confidence_gap,
                    "uncertainty": uncertainty,
                    "evidence_count": attempts,
                    "priority": priority,
                    "source": "adaptive_learning",
                    "metadata": {
                        "topic_id": topic_id,
                        "due_count": due_count,
                        "accuracy": round(accuracy, 4),
                        "correct_attempts": correct_attempts,
                    },
                }
            )

        track_entries: list[dict[str, Any]] = []
        for track in _TRACKS:
            items = [entry for entry in topic_entries if entry["track"] == track]
            if not items:
                base_confidence = int(profile["confidence_by_track"].get(track, 3))
                score = float(base_confidence * 20)
                priority = 60.0 if track == profile["primary_track"] else 35.0
                evidence_count = 0
                confidence_gap = 0.0
            else:
                score = round(
                    sum(float(item["score"]) * max(1, int(item["evidence_count"])) for item in items)
                    / sum(max(1, int(item["evidence_count"])) for item in items),
                    2,
                )
                declared_confidence = int(profile["confidence_by_track"].get(track, 3)) * 20.0
                confidence_gap = round(abs(declared_confidence - score), 2)
                evidence_count = sum(int(item["evidence_count"]) for item in items)
                uncertainty = round(
                    sum(
                        float(item["uncertainty"]) * max(1, int(item["evidence_count"]))
                        for item in items
                    )
                    / sum(max(1, int(item["evidence_count"])) for item in items),
                    4,
                )
                primary_bonus = 8.0 if track == profile["primary_track"] else 0.0
                priority = min(100.0, round(((100.0 - score) * 0.75) + (confidence_gap * 0.25) + primary_bonus, 2))
            track_entries.append(
                {
                    "competency_id": f"track:{track}",
                    "competency_type": "track",
                    "label": track.replace("_", " ").title(),
                    "track": track,
                    "score": score,
                    "confidence_gap": confidence_gap,
                    "uncertainty": 1.0 if not items else uncertainty,
                    "evidence_count": evidence_count,
                    "priority": priority,
                    "source": "profile_plus_learning",
                    "metadata": {},
                }
            )

        interview_entries: list[dict[str, Any]] = []
        reports = self._latest_interview_reports(user_id=user_id)
        if reports:
            for key, label in _INTERVIEW_DIMENSIONS:
                values: list[float] = []
                for report in reports:
                    rubric = report.get("rubric_averages")
                    if not isinstance(rubric, dict):
                        continue
                    raw = float(rubric.get(key) or 0.0)
                    values.append(max(0.0, min(5.0, raw)) * 20.0)
                if not values:
                    continue
                score = round(sum(values) / len(values), 2)
                priority = round((100.0 - score) * 0.9, 2)
                interview_entries.append(
                    {
                        "competency_id": f"interview:{key}",
                        "competency_type": "interview_dimension",
                        "label": label,
                        "track": profile["primary_track"],
                        "score": score,
                        "confidence_gap": 0.0,
                        "uncertainty": round(
                            accuracy_uncertainty(len(values), len(values)), 4
                        ),
                        "evidence_count": len(values),
                        "priority": priority,
                        "source": "interview_reports",
                        "metadata": {},
                    }
                )

        competencies = topic_entries + track_entries + interview_entries
        competencies.sort(key=lambda item: (-float(item["priority"]), float(item["score"]), str(item["label"])))
        self._replace_competencies(user_id=user_id, competencies=competencies, updated_at=now_iso)
        profile = self.get_profile(user_id=user_id)

        readiness_inputs = [float(item["score"]) for item in track_entries if item["evidence_count"] > 0]
        if not readiness_inputs:
            readiness_inputs = [float(item["score"]) for item in track_entries]
        interview_overall = [
            float(report.get("overall_score") or 0.0)
            for report in reports
            if report.get("overall_score") is not None
        ]
        if interview_overall:
            readiness_inputs.append(sum(interview_overall) / len(interview_overall))
        readiness_score = round(sum(readiness_inputs) / len(readiness_inputs), 2) if readiness_inputs else 0.0
        strongest = [item["label"] for item in sorted(competencies, key=lambda item: (-float(item["score"]), float(item["priority"])))[:3]]
        urgent = [item["label"] for item in competencies[:5]]
        return {
            "generated_at": now_iso,
            "profile": profile,
            "readiness_score": readiness_score,
            "strongest_competencies": strongest,
            "urgent_competencies": urgent,
            "competencies": self.list_competencies(user_id=user_id, limit=50),
        }

    def _build_cold_start_items(self, *, profile: dict[str, Any]) -> list[dict[str, Any]]:
        topics = _COLD_START_TOPICS.get(profile["primary_track"], _COLD_START_TOPICS["backend"])
        items: list[dict[str, Any]] = []
        for idx, topic_id in enumerate(topics):
            items.append(
                {
                    "recommendation_id": f"study:{topic_id}",
                    "recommendation_type": "study",
                    "topic_id": topic_id,
                    "title": f"Build foundations in {self._topic_label(topic_id)}",
                    "reason": "You have limited learning evidence here, so start with a focused study pass.",
                    "estimated_minutes": int(profile["preferred_session_minutes"]),
                    "cta_route": f"/topics/{topic_id}",
                    "priority": round(78.0 - (idx * 5.0), 2),
                    "reason_codes": ["cold_start", "primary_track"],
                    "track": self._topic_track(topic_id),
                }
            )
        if profile["target_role"]:
            items.append(
                {
                    "recommendation_id": "interview:cold-start",
                    "recommendation_type": "interview",
                    "topic_id": "",
                    "title": f"Run a baseline mock interview for {profile['target_role']}",
                    "reason": "Create a benchmark report before you deepen topic practice.",
                    "estimated_minutes": 25,
                    "cta_route": "/interview",
                    "priority": 76.0,
                    "reason_codes": ["baseline_interview", "target_role"],
                    "track": profile["primary_track"],
                }
            )
        return items

    def build_recommendations(self, *, user_id: str, limit: int = 8) -> dict[str, Any]:
        profile = self.get_profile(user_id=user_id)
        diagnostic = self.run_diagnostic(user_id=user_id)
        competencies = diagnostic["competencies"]
        topic_entries = [item for item in competencies if item["competency_type"] == "topic"]
        interview_entries = [item for item in competencies if item["competency_type"] == "interview_dimension"]
        days_until_target = self._deadline_days(profile["target_date"])
        deadline_soon = days_until_target is not None and days_until_target <= 30
        deadline_urgent = days_until_target is not None and days_until_target <= 14
        focus_ids = {str(topic_id) for topic_id in profile["focus_topic_ids"]}
        modalities = set(profile["preferred_modalities"])

        items: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        def _push(item: dict[str, Any]) -> None:
            rec_id = str(item["recommendation_id"])
            if rec_id in seen_ids:
                return
            seen_ids.add(rec_id)
            items.append(item)

        for entry in topic_entries:
            meta = entry.get("metadata") or {}
            topic_id = str(meta.get("topic_id") or "")
            if not topic_id:
                continue
            score = float(entry["score"])
            due_count = int(meta.get("due_count", 0) or 0)
            priority = float(entry["priority"])
            primary_bonus = 8.0 if entry["track"] == profile["primary_track"] else 0.0
            focus_bonus = 10.0 if topic_id in focus_ids else 0.0
            deadline_bonus = 12.0 if deadline_urgent else (6.0 if deadline_soon else 0.0)
            total_priority = min(100.0, round(priority + primary_bonus + focus_bonus + deadline_bonus, 2))
            label = str(entry["label"])

            if due_count > 0:
                _push(
                    {
                        "recommendation_id": f"review:{topic_id}",
                        "recommendation_type": "review",
                        "topic_id": topic_id,
                        "title": f"Review due questions in {label}",
                        "reason": f"{due_count} item(s) are due and this topic is below your target mastery.",
                        "estimated_minutes": 20,
                        "cta_route": f"/topics/{topic_id}",
                        "priority": min(100.0, total_priority + 4.0),
                        "reason_codes": ["due_now", "mastery_gap"],
                        "track": entry["track"],
                    }
                )

            if score < 62.0 and "study" in modalities:
                _push(
                    {
                        "recommendation_id": f"study:{topic_id}",
                        "recommendation_type": "study",
                        "topic_id": topic_id,
                        "title": f"Deepen {label}",
                        "reason": "Your mastery is still low here, so a focused study pass will pay off fastest.",
                        "estimated_minutes": int(profile["preferred_session_minutes"]),
                        "cta_route": f"/topics/{topic_id}",
                        "priority": total_priority,
                        "reason_codes": ["low_mastery", "targeted_study"],
                        "track": entry["track"],
                    }
                )
            elif "quiz" in modalities:
                _push(
                    {
                        "recommendation_id": f"quiz:{topic_id}",
                        "recommendation_type": "quiz",
                        "topic_id": topic_id,
                        "title": f"Validate retention in {label}",
                        "reason": "Use a quiz pass to check whether the concept is actually sticking under pressure.",
                        "estimated_minutes": 15,
                        "cta_route": f"/quiz/{topic_id}",
                        "priority": max(1.0, total_priority - 6.0),
                        "reason_codes": ["retention_check"],
                        "track": entry["track"],
                    }
                )

        lowest_interview = sorted(interview_entries, key=lambda item: float(item["score"]))
        if "interview" in modalities and (profile["target_role"] or deadline_soon or not interview_entries):
            weakest = lowest_interview[0]["label"] if lowest_interview else "overall interview readiness"
            interview_priority = 72.0 + (10.0 if deadline_urgent else 5.0 if deadline_soon else 0.0)
            _push(
                {
                    "recommendation_id": "interview:target-role",
                    "recommendation_type": "interview",
                    "topic_id": "",
                    "title": (
                        f"Run a mock interview for {profile['target_role']}"
                        if profile["target_role"]
                        else "Run a focused mock interview"
                    ),
                    "reason": (
                        f"Use interview practice to improve {weakest} and pressure-test your current preparation."
                    ),
                    "estimated_minutes": 25,
                    "cta_route": "/interview",
                    "priority": interview_priority,
                    "reason_codes": ["interview_practice", "deadline_pressure" if deadline_soon else "baseline"],
                    "track": profile["primary_track"],
                }
            )

        if not items:
            items = self._build_cold_start_items(profile=profile)

        items.sort(key=lambda item: (-float(item["priority"]), int(item["estimated_minutes"]), str(item["title"])))
        return {
            "generated_at": _to_iso(_utc_now()),
            "profile": profile,
            "days_until_target": days_until_target,
            "items": items[: max(1, min(int(limit), 50))],
        }

    def build_study_plan(self, *, user_id: str, days: int = 7, daily_items: int = 3) -> dict[str, Any]:
        """Build the 7-day plan as a persistent commitment device.

        Every scheduled task is persisted in ``plan_tasks`` under a
        stable :func:`plan_task_key`, so re-generating the plan (which
        happens on every load) recognises previously completed work
        instead of resurrecting identical unchecked items. The per-day
        item count is shaped by the learner's streak via
        :meth:`_commitment_context`.
        """
        target_days = max(1, min(int(days), 31))
        requested_daily_items = max(1, min(int(daily_items), 10))
        profile = self.get_profile(user_id=user_id)
        commitment = self._commitment_context(user_id=user_id, profile=profile)
        target_daily_items = _streak_adjusted_daily_items(
            requested_daily_items, behind=commitment["behind"]
        )
        commitment["requested_daily_items"] = requested_daily_items
        commitment["effective_daily_items"] = target_daily_items
        recs_payload = self.build_recommendations(
            user_id=user_id,
            limit=max(target_days * target_daily_items * 2, 8),
        )
        recs = list(recs_payload["items"])
        now = _utc_now()
        days_plan: list[dict[str, Any]] = []

        if not recs:
            for idx in range(target_days):
                day_date = (now + timedelta(days=idx)).date().isoformat()
                days_plan.append(
                    {
                        "day_index": idx + 1,
                        "label": "Today" if idx == 0 else f"Day {idx + 1}",
                        "date": day_date,
                        "tasks": [],
                    }
                )
            self._annotate_plan_completion(
                user_id=user_id,
                days_plan=days_plan,
                commitment=commitment,
            )
            return {
                "generated_at": _to_iso(now),
                "days": target_days,
                "daily_items": target_daily_items,
                "total_tasks": 0,
                "days_plan": days_plan,
                "commitment": commitment,
            }

        cursor = 0
        for idx in range(target_days):
            tasks: list[dict[str, Any]] = []
            used_ids: set[str] = set()
            day_occurrences: dict[tuple[str, str], int] = {}
            guard = 0
            day_date = (now + timedelta(days=idx)).date().isoformat()
            while len(tasks) < target_daily_items and guard < len(recs) * 4:
                guard += 1
                rec = recs[cursor % len(recs)]
                cursor += 1
                topic_id = str(rec.get("topic_id") or "")
                if topic_id and topic_id in used_ids and len(used_ids) < len(recs):
                    continue
                rec_type = str(rec["recommendation_type"])
                task_type = "topic_study" if rec_type == "study" else rec_type
                occurrence = day_occurrences.get((task_type, topic_id), 0) + 1
                day_occurrences[(task_type, topic_id)] = occurrence
                tasks.append(
                    {
                        "task_type": task_type,
                        "topic_id": topic_id,
                        "title": str(rec["title"]),
                        "reason": str(rec["reason"]),
                        "estimated_minutes": int(rec["estimated_minutes"]),
                        "cta_route": str(rec["cta_route"]),
                        "recommendation_id": str(rec.get("recommendation_id") or ""),
                        "task_key": plan_task_key(
                            task_date=day_date,
                            task_type=task_type,
                            topic_id=topic_id,
                            occurrence=occurrence,
                        ),
                    }
                )
                if topic_id:
                    used_ids.add(topic_id)

            days_plan.append(
                {
                    "day_index": idx + 1,
                    "label": "Today" if idx == 0 else f"Day {idx + 1}",
                    "date": day_date,
                    "tasks": tasks,
                }
            )

        self._persist_plan_tasks(
            user_id=user_id, days_plan=days_plan, created_at=_to_iso(now)
        )
        self._annotate_plan_completion(
            user_id=user_id,
            days_plan=days_plan,
            commitment=commitment,
        )
        return {
            "generated_at": _to_iso(now),
            "days": target_days,
            "daily_items": target_daily_items,
            "total_tasks": sum(len(day["tasks"]) for day in days_plan),
            "days_plan": days_plan,
            "commitment": commitment,
        }

    # ── Plan-task persistence (commitment device) ────────────────
    def _commitment_context(
        self, *, user_id: str, profile: dict[str, Any]
    ) -> dict[str, Any]:
        """Streak/daily-goal inputs that shape today's plan load.

        The gamification tables (``gamification_state`` +
        ``xp_events``) share this database file, so the planner reads
        them directly instead of holding a service reference — the
        router and the MCP ``get_study_plan`` tool then get identical
        commitment shaping for free.

        Streak semantics mirror
        :meth:`GamificationService.get_summary`: the live streak is
        recomputed on read, so activity today (or yesterday, with the
        day not over yet) keeps the live streak, while any longer
        silence lapses it to zero without mutating history.

        Documented rule (deterministic; the only clock input is the
        user-timezone date):

        * ``streak_at_risk`` — last activity was *yesterday* and
          nothing is logged today: the streak survives only if the
          learner does something today.
        * ``streak_lapsed`` — at least one full day was missed while
          the learner had earlier activity (``longest_streak`` > 0 or
          any learning event); the live streak already reads zero.
        * ``behind`` = ``streak_at_risk or streak_lapsed`` — the
          learner has missed days, so
          :meth:`build_study_plan` halves the daily item count
          (``ceil(n / 2)``, minimum 1) to a restart-sized commitment.
        * ``catch_up`` = ``behind or today_progress < daily_goal`` —
          surfaced by the UI as a catch-up prompt. A learner with no
          history at all is not "behind" and keeps the full load.
        """
        timezone = str(profile.get("timezone") or "UTC")
        today = today_in_timezone(timezone)
        state = {
            "current_streak": 0,
            "longest_streak": 0,
            "daily_goal": DEFAULT_DAILY_GOAL,
            "last_activity_date": None,
        }
        if self._table_exists("gamification_state"):
            row = self._conn.execute(
                """
                SELECT current_streak, longest_streak, daily_goal,
                       last_activity_date
                FROM gamification_state
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
            if row is not None:
                state = {
                    "current_streak": int(row["current_streak"] or 0),
                    "longest_streak": int(row["longest_streak"] or 0),
                    "daily_goal": int(row["daily_goal"] or DEFAULT_DAILY_GOAL),
                    "last_activity_date": (
                        str(row["last_activity_date"])
                        if row["last_activity_date"] is not None
                        else None
                    ),
                }

        last_activity = state["last_activity_date"]
        has_history = state["longest_streak"] > 0 or self._count_commitment_events(
            user_id=user_id
        ) > 0
        try:
            streak_at_risk = bool(
                last_activity and add_days(last_activity, 1) == today
            )
        except ValueError:
            # Malformed anchor: treat it as no streak rather than failing
            # the whole plan build.
            streak_at_risk = False
        streak_lapsed = bool(
            last_activity and last_activity != today and not streak_at_risk
        ) and has_history
        today_progress = self._count_commitment_events(
            user_id=user_id, day=today, timezone=timezone
        )
        behind = bool(streak_at_risk or streak_lapsed)
        return {
            "daily_goal": max(1, state["daily_goal"]),
            "today_progress": today_progress,
            "current_streak": state["current_streak"],
            "longest_streak": state["longest_streak"],
            "streak_at_risk": streak_at_risk,
            "streak_lapsed": streak_lapsed,
            "behind": behind,
            "catch_up": bool(behind or today_progress < state["daily_goal"]),
            "completed_tasks": 0,
            "pending_tasks": 0,
        }

    def _count_commitment_events(
        self,
        *,
        user_id: str,
        day: str | None = None,
        timezone: str = "UTC",
    ) -> int:
        """Learning events (attempts + card reviews) for the daily goal.

        Mirrors :meth:`GamificationStore.count_events_on_date`: with
        no ``day`` it counts every learning event (used to decide
        whether the learner has any history at all); with one it
        counts the events whose ``created_at`` falls on that date in
        the user's timezone.
        """
        if not self._table_exists("xp_events"):
            return 0
        rows = self._conn.execute(
            """
            SELECT created_at FROM xp_events
            WHERE user_id = ?
              AND event_type IN (?, ?)
            """,
            (user_id, *_COMMITMENT_EVENT_TYPES),
        ).fetchall()
        if day is None:
            return len(rows)
        try:
            tz = ZoneInfo(str(timezone or "UTC"))
        except Exception:
            tz = ZoneInfo("UTC")
        count = 0
        for row in rows:
            try:
                moment = datetime.fromisoformat(str(row["created_at"]))
            except ValueError:
                continue
            if moment.astimezone(tz).date().isoformat() == day:
                count += 1
        return count

    @staticmethod
    def _plan_task_payload(task: dict[str, Any]) -> dict[str, Any]:
        """The persisted slice of a scheduled task.

        ``task_key`` rides along so a stored row stays self-describing
        (it is derivable from the key schedule, but the planner hands
        it over when it builds the plan).
        """
        return {
            "task_type": str(task.get("task_type") or ""),
            "topic_id": str(task.get("topic_id") or ""),
            "title": str(task.get("title") or ""),
            "reason": str(task.get("reason") or ""),
            "estimated_minutes": int(task.get("estimated_minutes") or 0),
            "cta_route": str(task.get("cta_route") or ""),
            "recommendation_id": str(task.get("recommendation_id") or ""),
            "task_key": str(task.get("task_key") or ""),
        }

    def _persist_plan_tasks(
        self,
        *,
        user_id: str,
        days_plan: list[dict[str, Any]],
        created_at: str,
    ) -> None:
        """Upsert every scheduled task, preserving completion state.

        ``completed_at`` is deliberately never touched here: a task
        the learner already completed stays completed when the plan is
        regenerated, which is what makes the plan a commitment device
        rather than a snapshot.
        """
        rows: list[tuple[Any, ...]] = []
        for day in days_plan:
            task_date = str(day["date"])
            for task in day["tasks"]:
                rows.append(
                    (
                        user_id,
                        task_date,
                        str(task["task_key"]),
                        json.dumps(
                            self._plan_task_payload(task), ensure_ascii=True
                        ),
                        created_at,
                    )
                )
        if not rows:
            return
        with self._lock:
            with self._conn:
                self._conn.executemany(
                    """
                    INSERT INTO plan_tasks(
                        user_id, task_date, task_key, payload_json, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, task_date, task_key) DO UPDATE SET
                        payload_json = excluded.payload_json
                    """,
                    rows,
                )

    @staticmethod
    def _plan_task_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        payload = _loads_json_dict(str(row["payload_json"]))
        completed_at = row["completed_at"]
        return {
            "task_key": str(row["task_key"]),
            "task_date": str(row["task_date"]),
            "task_type": str(payload.get("task_type") or ""),
            "topic_id": str(payload.get("topic_id") or ""),
            "title": str(payload.get("title") or ""),
            "reason": str(payload.get("reason") or ""),
            "estimated_minutes": int(payload.get("estimated_minutes") or 0),
            "cta_route": str(payload.get("cta_route") or ""),
            "completed": completed_at is not None,
            "completed_at": str(completed_at) if completed_at is not None else None,
        }

    def _annotate_plan_completion(
        self,
        *,
        user_id: str,
        days_plan: list[dict[str, Any]],
        commitment: dict[str, Any],
    ) -> None:
        """Attach ``task_key``/``completed`` to every scheduled task.

        Also fills the commitment counters so completed work is never
        re-counted as pending.
        """
        dates = [str(day["date"]) for day in days_plan]
        if dates:
            placeholders = ", ".join("?" for _ in dates)
            rows = self._conn.execute(
                f"""
                SELECT task_date, task_key, completed_at
                FROM plan_tasks
                WHERE user_id = ? AND task_date IN ({placeholders})
                """,
                (user_id, *dates),
            ).fetchall()
        else:
            rows = []
        completed_keys = {
            (str(row["task_date"]), str(row["task_key"]))
            for row in rows
            if row["completed_at"] is not None
        }
        completed = 0
        total = 0
        for day in days_plan:
            task_date = str(day["date"])
            for task in day["tasks"]:
                total += 1
                task_key = str(task.get("task_key") or "")
                task["task_key"] = task_key
                task["completed"] = (task_date, task_key) in completed_keys
                if task["completed"]:
                    completed += 1
        commitment["completed_tasks"] = completed
        commitment["pending_tasks"] = total - completed

    def list_plan_tasks(self, *, user_id: str, task_date: str) -> list[dict[str, Any]]:
        """Every persisted task scheduled on ``task_date`` (ISO date)."""
        rows = self._conn.execute(
            """
            SELECT * FROM plan_tasks
            WHERE user_id = ? AND task_date = ?
            ORDER BY id ASC
            """,
            (user_id, task_date),
        ).fetchall()
        return [self._plan_task_row_to_dict(row) for row in rows]

    def complete_plan_task(
        self, *, user_id: str, task_key: str, task_date: str
    ) -> dict[str, Any] | None:
        """Mark a persisted task complete; ``None`` when it is unknown.

        Idempotent: completing an already-completed task keeps the
        original ``completed_at``.
        """
        key = str(task_key or "").strip()
        if not key:
            return None
        now_iso = _to_iso(_utc_now())
        with self._lock:
            with self._conn:
                cursor = self._conn.execute(
                    """
                    UPDATE plan_tasks
                    SET completed_at = ?
                    WHERE user_id = ? AND task_key = ? AND task_date = ?
                      AND completed_at IS NULL
                    """,
                    (now_iso, user_id, key, task_date),
                )
                if cursor.rowcount == 0 and self._plan_task_row(
                    user_id=user_id, task_key=key, task_date=task_date
                ) is None:
                    return None
        row = self._plan_task_row(
            user_id=user_id, task_key=key, task_date=task_date
        )
        return self._plan_task_row_to_dict(row) if row is not None else None

    def uncomplete_plan_task(
        self, *, user_id: str, task_key: str, task_date: str
    ) -> dict[str, Any] | None:
        """Clear a completion; ``None`` when the task is unknown."""
        key = str(task_key or "").strip()
        if not key:
            return None
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    UPDATE plan_tasks
                    SET completed_at = NULL
                    WHERE user_id = ? AND task_key = ? AND task_date = ?
                    """,
                    (user_id, key, task_date),
                )
        row = self._plan_task_row(
            user_id=user_id, task_key=key, task_date=task_date
        )
        return self._plan_task_row_to_dict(row) if row is not None else None

    def _plan_task_row(
        self, *, user_id: str, task_key: str, task_date: str
    ) -> sqlite3.Row | None:
        return self._conn.execute(
            """
            SELECT * FROM plan_tasks
            WHERE user_id = ? AND task_key = ? AND task_date = ?
            """,
            (user_id, task_key, task_date),
        ).fetchone()
