"""Goal-aware learner profile, diagnostics, and recommendations."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from typing import Any, Callable, TypeVar

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
                    evidence_count INTEGER NOT NULL,
                    priority REAL NOT NULL,
                    source TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, competency_id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_learner_competencies_priority
                ON learner_competency_scores(user_id, priority DESC, updated_at DESC)
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
                        created_at, updated_at, diagnostic_updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        created_at, updated_at, diagnostic_updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    def _question_progress_rows(self, *, user_id: str) -> list[sqlite3.Row]:
        if not self._table_exists("question_progress"):
            return []
        return self._conn.execute(
            """
            SELECT
                topic_id,
                SUM(attempts) AS attempts,
                SUM(correct_attempts) AS correct_attempts,
                AVG(mastery_score) AS mastery_score,
                AVG(avg_confidence) AS avg_confidence,
                SUM(CASE WHEN due_at <= ? THEN 1 ELSE 0 END) AS due_count
            FROM question_progress
            WHERE user_id = ?
            GROUP BY topic_id
            ORDER BY topic_id ASC
            """,
            (_to_iso(_utc_now()), user_id),
        ).fetchall()

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
                            confidence_gap, evidence_count, priority, source,
                            metadata_json, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            user_id,
                            str(item["competency_id"]),
                            str(item["competency_type"]),
                            str(item["label"]),
                            str(item.get("track", "")),
                            float(item["score"]),
                            float(item.get("confidence_gap", 0.0)),
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
                    "evidence_count": attempts,
                    "priority": priority,
                    "source": "adaptive_learning",
                    "metadata": {
                        "topic_id": topic_id,
                        "due_count": due_count,
                        "accuracy": round(accuracy, 4),
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
        target_days = max(1, min(int(days), 31))
        target_daily_items = max(1, min(int(daily_items), 10))
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
            return {
                "generated_at": _to_iso(now),
                "days": target_days,
                "daily_items": target_daily_items,
                "total_tasks": 0,
                "days_plan": days_plan,
            }

        cursor = 0
        for idx in range(target_days):
            tasks: list[dict[str, Any]] = []
            used_ids: set[str] = set()
            guard = 0
            while len(tasks) < target_daily_items and guard < len(recs) * 4:
                guard += 1
                rec = recs[cursor % len(recs)]
                cursor += 1
                topic_id = str(rec.get("topic_id") or "")
                if topic_id and topic_id in used_ids and len(used_ids) < len(recs):
                    continue
                rec_type = str(rec["recommendation_type"])
                task_type = "topic_study" if rec_type == "study" else rec_type
                tasks.append(
                    {
                        "task_type": task_type,
                        "topic_id": topic_id,
                        "title": str(rec["title"]),
                        "reason": str(rec["reason"]),
                        "estimated_minutes": int(rec["estimated_minutes"]),
                        "cta_route": str(rec["cta_route"]),
                    }
                )
                if topic_id:
                    used_ids.add(topic_id)

            day_date = (now + timedelta(days=idx)).date().isoformat()
            days_plan.append(
                {
                    "day_index": idx + 1,
                    "label": "Today" if idx == 0 else f"Day {idx + 1}",
                    "date": day_date,
                    "tasks": tasks,
                }
            )

        return {
            "generated_at": _to_iso(now),
            "days": target_days,
            "daily_items": target_daily_items,
            "total_tasks": sum(len(day["tasks"]) for day in days_plan),
            "days_plan": days_plan,
        }
