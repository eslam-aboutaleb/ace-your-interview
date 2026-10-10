"""Unit tests for the gamification store and service."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import UTC, datetime, timedelta

from app.services.gamification_service import (
    CARDS_100_THRESHOLD,
    GamificationService,
    LEVEL_XP,
    XP_ATTEMPT,
    XP_ATTEMPT_CORRECT_BONUS,
    XP_EXAM_COMPLETE,
    XP_EXAM_PASS_BONUS,
    XP_REVIEW,
)
from app.services.gamification_store import (
    GamificationStore,
    add_days,
    today_in_timezone,
)
from app.services.learning_planner import LearningPlannerStore


class GamificationStoreTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.db_path = os.path.join(self._td.name, "learning.db")
        self.store = GamificationStore(self.db_path)

    def test_default_state(self):
        state = self.store.get_state(user_id="u1")
        self.assertEqual(state["total_xp"], 0)
        self.assertEqual(state["current_streak"], 0)
        self.assertEqual(state["daily_goal"], 3)
        self.assertEqual(state["badges"], [])

    def test_award_xp_accumulates(self):
        state = self.store.award_xp(
            user_id="u1",
            event_key="attempt:u1:1",
            event_type="attempt",
            xp=10,
        )
        self.assertEqual(state["total_xp"], 10)
        state = self.store.award_xp(
            user_id="u1",
            event_key="attempt:u1:2",
            event_type="attempt",
            xp=15,
        )
        self.assertEqual(state["total_xp"], 25)

    def test_award_xp_idempotent(self):
        self.store.award_xp(
            user_id="u1",
            event_key="attempt:u1:1",
            event_type="attempt",
            xp=10,
        )
        state = self.store.award_xp(
            user_id="u1",
            event_key="attempt:u1:1",
            event_type="attempt",
            xp=10,
        )
        self.assertEqual(state["total_xp"], 10)
        self.assertEqual(
            self.store.count_events(user_id="u1"), 1
        )

    def test_streak_advances_on_consecutive_days(self):
        # Awarding twice on the same day keeps the
        # streak at 1 (same-day activity does not
        # advance the streak).
        self.store.award_xp(
            user_id="u1",
            event_key="a:1",
            event_type="attempt",
            xp=10,
            timezone="UTC",
        )
        state = self.store.get_state(user_id="u1")
        self.assertEqual(state["current_streak"], 1)

        self.store.award_xp(
            user_id="u1",
            event_key="a:2",
            event_type="attempt",
            xp=10,
            timezone="UTC",
        )
        state = self.store.get_state(user_id="u1")
        self.assertEqual(state["current_streak"], 1)
        self.assertEqual(state["longest_streak"], 1)

    def test_streak_resets_after_gap(self):
        # Activity three days ago, then today: the gap
        # resets the streak to 1.
        moment = datetime.now(UTC) - timedelta(days=3)
        self.store.award_xp(
            user_id="u1",
            event_key="a:1",
            event_type="attempt",
            xp=10,
            timezone="UTC",
            created_at=moment,
        )
        self.store.award_xp(
            user_id="u1",
            event_key="a:2",
            event_type="attempt",
            xp=10,
            timezone="UTC",
        )
        state = self.store.get_state(user_id="u1")
        self.assertEqual(state["current_streak"], 1)

    def test_set_daily_goal_clamps(self):
        state = self.store.set_daily_goal(user_id="u1", daily_goal=99)
        self.assertEqual(state["daily_goal"], 50)
        state = self.store.set_daily_goal(user_id="u1", daily_goal=0)
        self.assertEqual(state["daily_goal"], 1)
        state = self.store.set_daily_goal(user_id="u1", daily_goal=7)
        self.assertEqual(state["daily_goal"], 7)

    def test_set_badges_dedupes(self):
        state = self.store.set_badges(
            user_id="u1", badges=["b", "a", "b"]
        )
        self.assertEqual(state["badges"], ["a", "b"])

    def test_count_events_filters_by_type(self):
        self.store.award_xp(
            user_id="u1", event_key="a:1",
            event_type="attempt", xp=10,
        )
        self.store.award_xp(
            user_id="u1", event_key="r:1",
            event_type="review", xp=5,
        )
        self.store.award_xp(
            user_id="u2", event_key="a:2",
            event_type="attempt", xp=10,
        )
        self.assertEqual(
            self.store.count_events(user_id="u1"), 2
        )
        self.assertEqual(
            self.store.count_events(
                user_id="u1", event_types=("attempt",)
            ),
            1,
        )
        self.assertEqual(
            self.store.count_events(
                user_id="u1", event_types=("exam",)
            ),
            0,
        )

    def test_count_events_on_date(self):
        self.store.award_xp(
            user_id="u1", event_key="a:1",
            event_type="attempt", xp=10,
        )
        today = today_in_timezone("UTC")
        self.assertEqual(
            self.store.count_events_on_date(
                user_id="u1", date_str=today, timezone="UTC"
            ),
            1,
        )
        self.assertEqual(
            self.store.count_events_on_date(
                user_id="u1",
                date_str=add_days(today, -1),
                timezone="UTC",
            ),
            0,
        )

    def test_has_passed_exam_without_table(self):
        # exam_sessions table does not exist yet.
        self.assertFalse(self.store.has_passed_exam(user_id="u1"))


class GamificationServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.db_path = os.path.join(self._td.name, "learning.db")
        self.store = GamificationStore(self.db_path)
        self.planner = LearningPlannerStore(self.db_path)
        self.service = GamificationService(
            store=self.store, planner=self.planner
        )

    async def test_record_learning_attempt_xp(self):
        state = self.service.record_learning_attempt(
            user_id="u1", attempt_id=1, is_correct=False
        )
        self.assertEqual(state["total_xp"], XP_ATTEMPT)
        state = self.service.record_learning_attempt(
            user_id="u1", attempt_id=2, is_correct=True
        )
        self.assertEqual(
            state["total_xp"],
            XP_ATTEMPT + XP_ATTEMPT + XP_ATTEMPT_CORRECT_BONUS,
        )

    async def test_record_card_review_xp(self):
        state = self.service.record_card_review(
            user_id="u1", card_id="c1", review_log_id=1
        )
        self.assertEqual(state["total_xp"], XP_REVIEW)

    async def test_record_exam_completion_xp(self):
        state = self.service.record_exam_completion(
            user_id="u1", exam_id="e1", passed=False
        )
        self.assertEqual(state["total_xp"], XP_EXAM_COMPLETE)
        state = self.service.record_exam_completion(
            user_id="u1", exam_id="e2", passed=True
        )
        self.assertEqual(
            state["total_xp"],
            XP_EXAM_COMPLETE
            + XP_EXAM_COMPLETE
            + XP_EXAM_PASS_BONUS,
        )

    async def test_first_review_badge(self):
        self.service.record_card_review(
            user_id="u1", card_id="c1", review_log_id=1
        )
        summary = await self.service.get_summary(user_id="u1")
        self.assertIn("first_review", summary["badges"])

    async def test_cards_100_badge(self):
        for i in range(CARDS_100_THRESHOLD):
            self.service.record_card_review(
                user_id="u1", card_id=f"c{i}", review_log_id=i
            )
        summary = await self.service.get_summary(user_id="u1")
        self.assertIn("cards_100", summary["badges"])

    async def test_first_exam_and_exam_passed_badges(self):
        await self.service.record_exam_completion_async(
            user_id="u1", exam_id="e1", passed=True
        )
        summary = await self.service.get_summary(user_id="u1")
        self.assertIn("first_exam", summary["badges"])
        self.assertIn("exam_passed", summary["badges"])

    async def test_get_summary_level_and_progress(self):
        for i in range(10):
            self.service.record_learning_attempt(
                user_id="u1", attempt_id=i, is_correct=True
            )
        summary = await self.service.get_summary(user_id="u1")
        expected_xp = 10 * (XP_ATTEMPT + XP_ATTEMPT_CORRECT_BONUS)
        self.assertEqual(summary["total_xp"], expected_xp)
        self.assertEqual(
            summary["level"], expected_xp // LEVEL_XP + 1
        )
        self.assertGreaterEqual(summary["xp_progress"], 0.0)
        self.assertLessEqual(summary["xp_progress"], 1.0)
        self.assertEqual(summary["daily_goal"], 3)
        self.assertEqual(summary["today_progress"], 10)

    async def test_get_summary_streak_lapses_on_read(self):
        # Activity three days ago: the streak has lapsed,
        # so the read path reports 0 even though the
        # persisted streak was 1. award_xp always stamps
        # last_activity_date with today, so backdate the
        # state row directly.
        self.store.award_xp(
            user_id="u1",
            event_key="a:1",
            event_type="attempt",
            xp=10,
            timezone="UTC",
        )
        lapsed_date = add_days(today_in_timezone("UTC"), -3)
        with self.store._conn:
            self.store._conn.execute(
                "UPDATE gamification_state "
                "SET last_activity_date = ? WHERE user_id = ?",
                (lapsed_date, "u1"),
            )
        summary = await self.service.get_summary(user_id="u1")
        self.assertEqual(summary["current_streak"], 0)

    async def test_set_daily_goal(self):
        result = await self.service.set_daily_goal(
            user_id="u1", daily_goal=5
        )
        self.assertEqual(result["daily_goal"], 5)
        summary = await self.service.get_summary(user_id="u1")
        self.assertEqual(summary["daily_goal"], 5)

    async def test_user_timezone_from_profile(self):
        self.planner.upsert_profile(
            user_id="u1", payload={"timezone": "Europe/Berlin"}
        )
        self.assertEqual(
            self.service.user_timezone(user_id="u1"),
            "Europe/Berlin",
        )
        # Unknown user falls back to UTC.
        self.assertEqual(
            self.service.user_timezone(user_id="nobody"), "UTC"
        )

    async def test_summary_isolates_users(self):
        self.service.record_learning_attempt(
            user_id="u1", attempt_id=1, is_correct=True
        )
        summary = await self.service.get_summary(user_id="u2")
        self.assertEqual(summary["total_xp"], 0)
        self.assertEqual(summary["badges"], [])


if __name__ == "__main__":
    unittest.main()
