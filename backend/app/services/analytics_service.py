"""Study analytics: activity heatmap, retention, mastery curve.

Plan: ``1790987112911-analytics-gamification-exam-mode.md``

All endpoints are read-only aggregations over existing
tables (``learning_attempts``, ``fsrs_review_log``,
``exam_sessions``), so the analytics router is
deliberately unflagged.

The service opens its own read-only connection to the
learning database (the same file every other store
uses) and follows the single-thread ``run_async``
pattern so FastAPI handlers never block the event loop.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import Any, Callable, TypeVar

from app.services.learning_store import LearningStore

T = TypeVar("T")

#: EMA smoothing factor for the mastery curve.
MASTERY_EMA_ALPHA = 0.3


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _from_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


class AnalyticsService:
    """Read-only study analytics over the learning database."""

    def __init__(self, db_path: str, learning_store: LearningStore):
        self.db_path = db_path
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._learning_store = learning_store
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="analytics-service"
        )
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    async def run_async(self, fn: Callable[..., T], /, *args, **kwargs) -> T:
        """Run synchronous SQLite work on a dedicated worker thread."""
        loop = asyncio.get_running_loop()
        if kwargs:
            return await loop.run_in_executor(
                self._executor,
                lambda: fn(*args, **kwargs),
            )
        return await loop.run_in_executor(self._executor, fn, *args)

    # ── Activity heatmap ───────────────────
    def heatmap(self, *, user_id: str, weeks: int = 52) -> dict[str, Any]:
        """GitHub-style activity heatmap over the last ``weeks`` weeks.

        Counts events per UTC day from three sources:
        ``learning_attempts`` (``attempt``),
        ``fsrs_review_log`` (``review``) and completed
        exams (``exam``).
        """
        horizon = max(1, min(int(weeks), 104))
        today = _utc_now().date()
        start = today - timedelta(days=horizon * 7 - 1)
        days: dict[str, dict[str, Any]] = {}
        cursor = start
        while cursor <= today:
            days[cursor.isoformat()] = {"count": 0, "types": {}}
            cursor += timedelta(days=1)

        def _bump(day_str: str, kind: str) -> None:
            entry = days.get(day_str)
            if entry is None:
                return
            entry["count"] += 1
            entry["types"][kind] = entry["types"].get(kind, 0) + 1

        with self._lock:
            for row in self._conn.execute(
                "SELECT created_at FROM learning_attempts WHERE user_id = ?",
                (user_id,),
            ):
                _bump(_from_iso(str(row["created_at"])).date().isoformat(), "attempt")
            for row in self._conn.execute(
                "SELECT created_at FROM fsrs_review_log WHERE user_id = ?",
                (user_id,),
            ):
                _bump(_from_iso(str(row["created_at"])).date().isoformat(), "review")
            if self._table_exists("exam_sessions"):
                for row in self._conn.execute(
                    """
                    SELECT completed_at FROM exam_sessions
                    WHERE user_id = ? AND status = 'completed'
                      AND completed_at IS NOT NULL
                    """,
                    (user_id,),
                ):
                    _bump(_from_iso(str(row["completed_at"])).date().isoformat(), "exam")

        return {
            "days": [
                {"date": day, "count": days[day]["count"], "types": days[day]["types"]}
                for day in sorted(days)
            ],
            "weeks": horizon,
        }

    # ── Retention buckets ──────────────────
    def retention(self, *, user_id: str) -> dict[str, Any]:
        """Recall rate at 1d/7d/30d from ``fsrs_review_log``.

        A review belongs to a bucket when its
        ``elapsed_days`` (days since the previous review)
        is at least the bucket threshold; the recall rate
        is the share of successful ratings (``good``/``easy``).
        """
        rows = self._conn.execute(
            """
            SELECT rating, elapsed_days
            FROM fsrs_review_log
            WHERE user_id = ? AND elapsed_days IS NOT NULL
            """,
            (user_id,),
        ).fetchall()
        reviews = [
            (str(row["rating"]), int(row["elapsed_days"] or 0))
            for row in rows
        ]
        buckets: list[dict[str, Any]] = []
        for name, threshold in (("1d", 1), ("7d", 7), ("30d", 30)):
            eligible = [
                rating for rating, elapsed in reviews if elapsed >= threshold
            ]
            successful = [
                rating
                for rating in eligible
                if rating in ("good", "easy")
            ]
            buckets.append(
                {
                    "bucket": name,
                    "reviews": len(eligible),
                    "successful": len(successful),
                    "recall_rate": (
                        round(len(successful) / len(eligible), 4)
                        if eligible
                        else 0.0
                    ),
                }
            )
        return {"buckets": buckets}

    # ── Mastery curve ──────────────────────
    def mastery_curve(self, *, user_id: str) -> dict[str, Any]:
        """Topic mastery over time from ``learning_attempts``.

        An exponential moving average (alpha=0.3) of
        ``is_correct`` is computed per topic and overall,
        in ``created_at`` order; each day contributes the
        EMA value at the end of that day.
        """
        rows = self._conn.execute(
            """
            SELECT topic_id, is_correct, created_at
            FROM learning_attempts
            WHERE user_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (user_id,),
        ).fetchall()

        overall_ema: float | None = None
        topic_ema: dict[str, float] = {}
        overall_points: dict[str, float] = {}
        topic_points: dict[str, dict[str, float]] = {}

        for row in rows:
            value = float(row["is_correct"] or 0)
            overall_ema = (
                value
                if overall_ema is None
                else MASTERY_EMA_ALPHA * value + (1 - MASTERY_EMA_ALPHA) * overall_ema
            )
            topic_id = str(row["topic_id"])
            topic_value = topic_ema.get(topic_id)
            topic_ema[topic_id] = (
                value
                if topic_value is None
                else MASTERY_EMA_ALPHA * value + (1 - MASTERY_EMA_ALPHA) * topic_value
            )
            day = _from_iso(str(row["created_at"])).date().isoformat()
            assert overall_ema is not None
            overall_points[day] = overall_ema
            topic_points.setdefault(topic_id, {})[day] = topic_ema[topic_id]

        return {
            "points": [
                {"date": day, "mastery": round(overall_points[day], 4)}
                for day in sorted(overall_points)
            ],
            "topics": [
                {
                    "topic_id": topic_id,
                    "points": [
                        {"date": day, "mastery": round(points[day], 4)}
                        for day in sorted(points)
                    ],
                }
                for topic_id, points in sorted(topic_points.items())
            ],
        }

    # ── Due forecast ───────────────────────
    async def due_forecast(
        self, *, user_id: str, days: int = 7
    ) -> dict[str, Any]:
        """Delegate to the existing FSRS forecast endpoint logic."""
        return await self._learning_store.run_async(
            self._learning_store.get_forecast, user_id=user_id, days=days
        )

    # ── Internals ──────────────────────────
    def _table_exists(self, table_name: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
            (table_name,),
        ).fetchone()
        return row is not None
