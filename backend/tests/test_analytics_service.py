"""Unit tests for the read-only analytics service."""

from __future__ import annotations

import os
import tempfile
import unittest

from app.services.analytics_service import AnalyticsService
from app.services.learning_store import LearningStore


class AnalyticsServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.db_path = os.path.join(self._td.name, "learning.db")
        self.learning = LearningStore(self.db_path)
        self.service = AnalyticsService(
            self.db_path, self.learning
        )

    async def _record_attempt(
        self, *, user_id="u1", topic_id="t1", is_correct=True, mode="quiz"
    ):
        return self.learning.record_attempt(
            user_id=user_id,
            question_id=f"q-{topic_id}-{is_correct}",
            topic_id=topic_id,
            user_answer="answer",
            is_correct=is_correct,
            confidence=4,
            response_time_ms=1200,
            mode=mode,
        )

    async def test_mastery_curve_excludes_interview_turns(self):
        # A quiz run and an interview turn both write learning_attempts rows;
        # the curve must track topic mastery only, so the interview row is
        # invisible to it while still being stored for competency analysis.
        await self._record_attempt(is_correct=True)
        await self._record_attempt(is_correct=False, mode="interview")
        curve = await self.service.run_async(
            self.service.mastery_curve, user_id="u1"
        )
        topics = {t["topic_id"] for t in curve["topics"]}
        self.assertIn("t1", topics)
        self.assertNotIn("interview-competency", topics)
        points = curve["points"]
        self.assertEqual(points[-1]["mastery"], 1.0)

    async def test_heatmap_empty(self):
        result = await self.service.run_async(
            self.service.heatmap, user_id="u1", weeks=2
        )
        self.assertEqual(result["weeks"], 2)
        self.assertEqual(len(result["days"]), 14)
        for day in result["days"]:
            self.assertEqual(day["count"], 0)
            self.assertEqual(day["types"], {})

    async def test_heatmap_counts_attempts(self):
        # A quiz attempt persists both a learning_attempts
        # row and an fsrs_review_log row, so the heatmap
        # counts one "attempt" and one "review" event each.
        await self._record_attempt()
        await self._record_attempt()
        await self._record_attempt(is_correct=False)
        result = await self.service.run_async(
            self.service.heatmap, user_id="u1", weeks=1
        )
        today = result["days"][-1]
        self.assertEqual(today["count"], 6)
        self.assertEqual(
            today["types"], {"attempt": 3, "review": 3}
        )
        # Other days are empty.
        self.assertEqual(
            sum(d["count"] for d in result["days"]), 6
        )

    async def test_heatmap_isolates_users(self):
        await self._record_attempt(user_id="u1")
        await self._record_attempt(user_id="u2")
        result = await self.service.run_async(
            self.service.heatmap, user_id="u1", weeks=1
        )
        # u1's single attempt = 1 attempt + 1 review event.
        self.assertEqual(
            sum(d["count"] for d in result["days"]), 2
        )

    async def test_heatmap_weeks_clamped(self):
        result = await self.service.run_async(
            self.service.heatmap, user_id="u1", weeks=999
        )
        self.assertEqual(result["weeks"], 104)
        self.assertEqual(len(result["days"]), 104 * 7)

    async def test_retention_empty(self):
        result = await self.service.run_async(
            self.service.retention, user_id="u1"
        )
        self.assertEqual(len(result["buckets"]), 3)
        for bucket in result["buckets"]:
            self.assertEqual(bucket["reviews"], 0)
            self.assertEqual(bucket["successful"], 0)
            self.assertEqual(bucket["recall_rate"], 0.0)
        self.assertEqual(
            [b["bucket"] for b in result["buckets"]],
            ["1d", "7d", "30d"],
        )

    async def test_retention_buckets(self):
        # Insert review-log rows with controlled
        # elapsed_days to exercise the bucket logic.
        with self.learning._conn:
            for elapsed, rating in (
                (0, "again"),
                (1, "good"),
                (2, "easy"),
                (7, "good"),
                (8, "hard"),
                (30, "easy"),
                (45, "good"),
            ):
                self.learning._conn.execute(
                    """
                    INSERT INTO fsrs_review_log(
                        user_id, card_id, rating, state,
                        review_duration_ms, scheduled_days,
                        elapsed_days, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "u1", "c1", rating, "review",
                        100, 1, elapsed,
                        "2026-01-01T00:00:00+00:00",
                    ),
                )
        result = await self.service.run_async(
            self.service.retention, user_id="u1"
        )
        by_bucket = {b["bucket"]: b for b in result["buckets"]}
        # 1d: elapsed >= 1 -> 6 reviews, 4 successful
        # (good/easy: 1,2,7,30,45 are good/easy; 8 is hard)
        self.assertEqual(by_bucket["1d"]["reviews"], 6)
        self.assertEqual(by_bucket["1d"]["successful"], 5)
        # 7d: elapsed >= 7 -> 4 reviews (7,8,30,45)
        self.assertEqual(by_bucket["7d"]["reviews"], 4)
        self.assertEqual(by_bucket["7d"]["successful"], 3)
        # 30d: elapsed >= 30 -> 2 reviews (30,45)
        self.assertEqual(by_bucket["30d"]["reviews"], 2)
        self.assertEqual(by_bucket["30d"]["successful"], 2)
        self.assertEqual(by_bucket["30d"]["recall_rate"], 1.0)

    async def test_mastery_curve_empty(self):
        result = await self.service.run_async(
            self.service.mastery_curve, user_id="u1"
        )
        self.assertEqual(result["points"], [])
        self.assertEqual(result["topics"], [])

    async def test_mastery_curve_ema(self):
        # All-correct attempts converge the EMA toward 1.
        for _ in range(5):
            await self._record_attempt(topic_id="t1", is_correct=True)
        result = await self.service.run_async(
            self.service.mastery_curve, user_id="u1"
        )
        self.assertEqual(len(result["points"]), 1)
        point = result["points"][0]
        self.assertGreater(point["mastery"], 0.9)
        self.assertLessEqual(point["mastery"], 1.0)
        self.assertEqual(len(result["topics"]), 1)
        self.assertEqual(result["topics"][0]["topic_id"], "t1")

    async def test_mastery_curve_all_incorrect(self):
        for _ in range(5):
            await self._record_attempt(topic_id="t1", is_correct=False)
        result = await self.service.run_async(
            self.service.mastery_curve, user_id="u1"
        )
        self.assertLess(result["points"][0]["mastery"], 0.1)

    async def test_mastery_curve_mixed_topics(self):
        await self._record_attempt(topic_id="t1", is_correct=True)
        await self._record_attempt(topic_id="t2", is_correct=False)
        result = await self.service.run_async(
            self.service.mastery_curve, user_id="u1"
        )
        self.assertEqual(len(result["topics"]), 2)
        topic_ids = {t["topic_id"] for t in result["topics"]}
        self.assertEqual(topic_ids, {"t1", "t2"})

    async def test_due_forecast_delegates(self):
        result = await self.service.due_forecast(
            user_id="u1", days=7
        )
        self.assertIn("due_counts", result)
        self.assertIn("total_due", result)


if __name__ == "__main__":
    unittest.main()
