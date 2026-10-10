"""SQLite-backed gamification persistence (XP events + streak state).

Follows the :class:`~app.services.card_store.CardStore`
pattern: a dedicated SQLite connection, a single-thread
executor, and an async ``run_async`` shim so FastAPI
handlers never block the event loop.

XP awarding is idempotent: ``xp_events.event_key`` carries a
UNIQUE constraint, so replaying the same learning event
(attempt, card review, exam completion) never double-counts.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Any, Callable, TypeVar

T = TypeVar("T")

DEFAULT_DAILY_GOAL = 3
DEFAULT_TIMEZONE = "UTC"

EVENT_ATTEMPT = "attempt"
EVENT_REVIEW = "review"
EVENT_EXAM = "exam"

LEARNING_EVENT_TYPES = (EVENT_ATTEMPT, EVENT_REVIEW)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _to_iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


def _from_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


def resolve_timezone(timezone_str: str | None) -> ZoneInfo:
    """Resolve a tz-database name; fall back to UTC for unknown values."""
    try:
        return ZoneInfo(str(timezone_str or "").strip() or DEFAULT_TIMEZONE)
    except Exception:
        return ZoneInfo(DEFAULT_TIMEZONE)


def today_in_timezone(timezone_str: str | None) -> str:
    """Today's date string in the user's timezone."""
    return datetime.now(resolve_timezone(timezone_str)).date().isoformat()


def add_days(date_str: str, days: int) -> str:
    """Shift an ISO date string by ``days``."""
    return (date.fromisoformat(date_str) + timedelta(days=days)).isoformat()


class GamificationStore:
    """Persist XP events and per-user gamification state."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="gamification-store"
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
                CREATE TABLE IF NOT EXISTS xp_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL, event_key TEXT NOT NULL UNIQUE,
                    event_type TEXT NOT NULL, xp INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS gamification_state (
                    user_id TEXT PRIMARY KEY, total_xp INTEGER NOT NULL DEFAULT 0,
                    current_streak INTEGER NOT NULL DEFAULT 0,
                    longest_streak INTEGER NOT NULL DEFAULT 0,
                    last_activity_date TEXT,
                    daily_goal INTEGER NOT NULL DEFAULT 3,
                    badges_json TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_xp_events_user_created "
                "ON xp_events(user_id, created_at DESC)"
            )

    # ── Row mapping ────────────────────────────────────
    @staticmethod
    def _state_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        try:
            badges = json.loads(str(row["badges_json"] or "[]"))
            if not isinstance(badges, list):
                badges = []
        except json.JSONDecodeError:
            badges = []
        last_activity = row["last_activity_date"]
        return {
            "user_id": str(row["user_id"]),
            "total_xp": int(row["total_xp"] or 0),
            "current_streak": int(row["current_streak"] or 0),
            "longest_streak": int(row["longest_streak"] or 0),
            "last_activity_date": (
                str(last_activity) if last_activity is not None else None
            ),
            "daily_goal": int(row["daily_goal"] or DEFAULT_DAILY_GOAL),
            "badges": [str(b) for b in badges],
            "updated_at": str(row["updated_at"]),
        }

    def _default_state(self, *, user_id: str) -> dict[str, Any]:
        return {
            "user_id": user_id,
            "total_xp": 0,
            "current_streak": 0,
            "longest_streak": 0,
            "last_activity_date": None,
            "daily_goal": DEFAULT_DAILY_GOAL,
            "badges": [],
            "updated_at": _to_iso(_utc_now()),
        }

    def get_state(self, *, user_id: str) -> dict[str, Any]:
        """Return the persisted state, or a default when no row exists."""
        row = self._conn.execute(
            "SELECT * FROM gamification_state WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return self._state_row_to_dict(row) if row else self._default_state(user_id=user_id)

    def _ensure_state(self, *, user_id: str) -> dict[str, Any]:
        """Return the state row, creating the default row when missing."""
        state = self.get_state(user_id=user_id)
        if state["updated_at"] and self._row_exists(user_id=user_id):
            return state
        now_iso = _to_iso(_utc_now())
        self._conn.execute(
            """
            INSERT INTO gamification_state(
                user_id, total_xp, current_streak, longest_streak,
                last_activity_date, daily_goal, badges_json, updated_at
            ) VALUES (?, 0, 0, 0, NULL, ?, '[]', ?)
            ON CONFLICT(user_id) DO NOTHING
            """,
            (user_id, DEFAULT_DAILY_GOAL, now_iso),
        )
        return self.get_state(user_id=user_id)

    def _row_exists(self, *, user_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM gamification_state WHERE user_id = ? LIMIT 1",
            (user_id,),
        ).fetchone()
        return row is not None

    # ── XP awarding ──────────────────────────────────
    def award_xp(
        self,
        *,
        user_id: str,
        event_key: str,
        event_type: str,
        xp: int,
        timezone: str | None = None,
        created_at: datetime | None = None,
    ) -> dict[str, Any]:
        """Insert an XP event and advance streak state.

        Idempotent: a replayed ``event_key`` hits the UNIQUE
        constraint and returns the current state unchanged.
        Caller must hold ``self._lock`` (or use the public
        synchronous wrappers, which take it).
        """
        moment = created_at or _utc_now()
        moment_iso = _to_iso(moment)
        with self._lock:
            with self._conn:
                try:
                    self._conn.execute(
                        """
                        INSERT INTO xp_events(
                            user_id, event_key, event_type, xp, created_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (user_id, str(event_key), str(event_type), int(xp), moment_iso),
                    )
                except sqlite3.IntegrityError:
                    # Duplicate event_key — idempotent no-op.
                    return self.get_state(user_id=user_id)

                state = self._ensure_state(user_id=user_id)
                today = today_in_timezone(timezone)
                last_activity = state["last_activity_date"]
                if not last_activity:
                    streak = 1
                elif last_activity == today:
                    streak = state["current_streak"]
                elif add_days(last_activity, 1) == today:
                    streak = state["current_streak"] + 1
                else:
                    streak = 1
                total_xp = state["total_xp"] + int(xp)
                longest_streak = max(state["longest_streak"], streak)
                self._conn.execute(
                    """
                    UPDATE gamification_state
                    SET total_xp = ?, current_streak = ?, longest_streak = ?,
                        last_activity_date = ?, updated_at = ?
                    WHERE user_id = ?
                    """,
                    (
                        total_xp,
                        streak,
                        longest_streak,
                        today,
                        moment_iso,
                        user_id,
                    ),
                )
        return self.get_state(user_id=user_id)

    # ── State mutations ──────────────────────────────
    def set_daily_goal(self, *, user_id: str, daily_goal: int) -> dict[str, Any]:
        goal = max(1, min(50, int(daily_goal)))
        now_iso = _to_iso(_utc_now())
        with self._lock:
            with self._conn:
                self._ensure_state(user_id=user_id)
                self._conn.execute(
                    """
                    UPDATE gamification_state
                    SET daily_goal = ?, updated_at = ?
                    WHERE user_id = ?
                    """,
                    (goal, now_iso, user_id),
                )
        return self.get_state(user_id=user_id)

    def set_badges(
        self, *, user_id: str, badges: list[str]
    ) -> dict[str, Any]:
        clean = sorted({str(b) for b in badges if str(b).strip()})
        now_iso = _to_iso(_utc_now())
        with self._lock:
            with self._conn:
                self._ensure_state(user_id=user_id)
                self._conn.execute(
                    """
                    UPDATE gamification_state
                    SET badges_json = ?, updated_at = ?
                    WHERE user_id = ?
                    """,
                    (json.dumps(clean, ensure_ascii=True), now_iso, user_id),
                )
        return self.get_state(user_id=user_id)

    # ── Reads ────────────────────────────────────────
    def count_events(
        self,
        *,
        user_id: str,
        event_types: tuple[str, ...] = (),
    ) -> int:
        """Total XP events (optionally filtered by event type)."""
        if event_types:
            placeholders = ", ".join("?" for _ in event_types)
            rows = self._conn.execute(
                f"""
                SELECT COUNT(*) AS count FROM xp_events
                WHERE user_id = ? AND event_type IN ({placeholders})
                """,
                (user_id, *event_types),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT COUNT(*) AS count FROM xp_events WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        return int(rows[0]["count"]) if rows else 0

    def count_events_on_date(
        self,
        *,
        user_id: str,
        date_str: str,
        timezone: str | None = None,
    ) -> int:
        """Learning events (attempt/review) whose created_at falls on
        ``date_str`` in the user's timezone."""
        tz = resolve_timezone(timezone)
        rows = self._conn.execute(
            """
            SELECT created_at FROM xp_events
            WHERE user_id = ? AND event_type IN ('attempt', 'review')
            """,
            (user_id,),
        ).fetchall()
        count = 0
        for row in rows:
            moment = _from_iso(str(row["created_at"]))
            if moment.astimezone(tz).date().isoformat() == date_str:
                count += 1
        return count

    def has_passed_exam(self, *, user_id: str) -> bool:
        """Whether the user has any completed, passed exam.

        ``exam_sessions`` is created by :class:`ExamStore` in the
        same database file; the table-exists guard keeps this safe
        when the exam store was never initialised.
        """
        if not self._table_exists("exam_sessions"):
            return False
        row = self._conn.execute(
            """
            SELECT 1 FROM exam_sessions
            WHERE user_id = ? AND status = 'completed' AND passed = 1
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()
        return row is not None

    def _table_exists(self, table_name: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
            (table_name,),
        ).fetchone()
        return row is not None
