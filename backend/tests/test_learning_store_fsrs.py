"""Tests for FSRS persistence in LearningStore."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import UTC, datetime, timedelta

from app.services.learning_store import LearningStore


def _now() -> datetime:
    return datetime.now(UTC)


def _to_iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


class RecordReviewTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.store = LearningStore(
            os.path.join(self._td.name, "learning.db")
        )

    def test_record_review_creates_card(self):
        result = self.store.record_review(
            user_id="alice",
            card_id="q1",
            topic_id="topic-a",
            rating="good",
            response_time_ms=1200,
        )
        card = result["card"]
        self.assertEqual(card["card_id"], "q1")
        self.assertEqual(card["topic_id"], "topic-a")
        self.assertEqual(card["source_type"], "question")
        self.assertEqual(card["state"], "learning")
        self.assertEqual(card["reps"], 1)
        self.assertEqual(card["lapses"], 0)
        self.assertIsNotNone(card["stability"])
        self.assertIsNotNone(card["difficulty"])
        self.assertIsNotNone(card["last_review_at"])
        self.assertFalse(result["leech"])
        self.assertEqual(result["log"]["card_id"], "q1")
        self.assertEqual(result["log"]["rating"], "good")
        self.assertEqual(result["log"]["review_duration_ms"], 1200)
        self.assertGreaterEqual(result["log"]["id"], 1)

    def test_record_review_updates_existing_card(self):
        first = self.store.record_review(
            user_id="alice",
            card_id="q1",
            topic_id="topic-a",
            rating="good",
            response_time_ms=100,
        )
        second = self.store.record_review(
            user_id="alice",
            card_id="q1",
            topic_id="topic-a",
            rating="good",
            response_time_ms=100,
        )
        self.assertEqual(second["card"]["reps"], 2)
        self.assertEqual(second["card"]["topic_id"], "topic-a")
        # due_at must move forward after a successful review
        self.assertGreater(
            second["card"]["due_at"], first["card"]["due_at"]
        )

    def test_again_increments_lapses(self):
        result = None
        for _ in range(3):
            result = self.store.record_review(
                user_id="alice",
                card_id="q1",
                topic_id="topic-a",
                rating="again",
                response_time_ms=100,
            )
        self.assertEqual(result["card"]["lapses"], 3)
        self.assertFalse(result["leech"])

    def test_leech_flag_at_five_lapses(self):
        result = None
        for _ in range(5):
            result = self.store.record_review(
                user_id="alice",
                card_id="q1",
                topic_id="topic-a",
                rating="again",
                response_time_ms=100,
            )
        self.assertEqual(result["card"]["lapses"], 5)
        self.assertTrue(result["leech"])

    def test_user_isolation(self):
        self.store.record_review(
            user_id="alice",
            card_id="q1",
            topic_id="topic-a",
            rating="good",
            response_time_ms=100,
        )
        due = self.store.get_due_cards(user_id="bob", limit=10)
        self.assertEqual(due, [])

    def test_invalid_rating_rejected(self):
        with self.assertRaises(ValueError):
            self.store.record_review(
                user_id="alice",
                card_id="q1",
                topic_id="topic-a",
                rating="bogus",
                response_time_ms=100,
            )


class GetDueCardsTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.store = LearningStore(
            os.path.join(self._td.name, "learning.db")
        )

    def _make_due(self, card_id: str, rating: str = "easy") -> None:
        self.store.record_review(
            user_id="alice",
            card_id=card_id,
            topic_id="topic-a",
            rating=rating,
            response_time_ms=100,
        )
        past = _to_iso(_now() - timedelta(hours=1))
        self.store._conn.execute(
            "UPDATE fsrs_cards SET due_at = ? "
            "WHERE user_id = ? AND card_id = ?",
            (past, "alice", card_id),
        )
        self.store._conn.commit()

    def test_due_cards_returns_due_only(self):
        self._make_due("q1")
        # future card
        self.store.record_review(
            user_id="alice",
            card_id="q2",
            topic_id="topic-a",
            rating="easy",
            response_time_ms=100,
        )
        due = self.store.get_due_cards(user_id="alice", limit=10)
        self.assertEqual([c["card_id"] for c in due], ["q1"])

    def test_due_cards_excludes_suspended(self):
        self._make_due("q1")
        self.store._conn.execute(
            "UPDATE fsrs_cards SET suspended = 1 "
            "WHERE user_id = ? AND card_id = ?",
            ("alice", "q1"),
        )
        self.store._conn.commit()
        due = self.store.get_due_cards(user_id="alice", limit=10)
        self.assertEqual(due, [])

    def test_review_queue_includes_fsrs_fields(self):
        self._make_due("q1")
        queue = self.store.get_fsrs_review_queue(
            user_id="alice", limit=10
        )
        self.assertEqual(queue["total_due"], 1)
        item = queue["items"][0]
        self.assertEqual(item["question_id"], "q1")
        self.assertIn("state", item)
        self.assertIn("lapses", item)
        self.assertIn("suspended", item)
        self.assertIn("leech", item)
        self.assertIn("due_at", item)
        self.assertEqual(item["leech"], False)


class ForecastTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.store = LearningStore(
            os.path.join(self._td.name, "learning.db")
        )

    def _schedule(self, card_id: str, due_at: datetime) -> None:
        self.store.record_review(
            user_id="alice",
            card_id=card_id,
            topic_id="topic-a",
            rating="easy",
            response_time_ms=100,
        )
        self.store._conn.execute(
            "UPDATE fsrs_cards SET due_at = ? "
            "WHERE user_id = ? AND card_id = ?",
            (_to_iso(due_at), "alice", card_id),
        )
        self.store._conn.commit()

    def test_forecast_buckets_by_day(self):
        now = _now()
        self._schedule("q1", now + timedelta(days=1))
        self._schedule("q2", now + timedelta(days=1, hours=2))
        self._schedule("q3", now + timedelta(days=3))
        self._schedule("q4", now + timedelta(days=10))  # beyond 7d
        forecast = self.store.get_forecast(user_id="alice", days=7)
        self.assertEqual(forecast["total_due"], 3)
        by_date = {d["date"]: d["count"] for d in forecast["due_counts"]}
        self.assertEqual(len(forecast["due_counts"]), 7)
        day1 = (now + timedelta(days=1)).date().isoformat()
        day3 = (now + timedelta(days=3)).date().isoformat()
        self.assertEqual(by_date[day1], 2)
        self.assertEqual(by_date[day3], 1)

    def test_forecast_buckets_overdue_into_today(self):
        now = _now()
        self._schedule("q1", now - timedelta(days=2))
        forecast = self.store.get_forecast(user_id="alice", days=7)
        today = now.date().isoformat()
        by_date = {d["date"]: d["count"] for d in forecast["due_counts"]}
        self.assertEqual(by_date[today], 1)
        self.assertEqual(forecast["total_due"], 1)

    def test_forecast_excludes_suspended(self):
        now = _now()
        self._schedule("q1", now + timedelta(days=1))
        self.store._conn.execute(
            "UPDATE fsrs_cards SET suspended = 1 "
            "WHERE user_id = ? AND card_id = ?",
            ("alice", "q1"),
        )
        self.store._conn.commit()
        forecast = self.store.get_forecast(user_id="alice", days=7)
        self.assertEqual(forecast["total_due"], 0)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.store = LearningStore(
            os.path.join(self._td.name, "learning.db")
        )

    def _log(self, rating: str, elapsed_days: int) -> None:
        self.store._conn.execute(
            """
            INSERT INTO fsrs_review_log(
                user_id, card_id, rating, state, review_duration_ms,
                scheduled_days, elapsed_days, created_at
            ) VALUES ('alice', 'q1', ?, 'review', 1000, 7, ?, ?)
            """,
            (rating, elapsed_days, _to_iso(_now())),
        )
        self.store._conn.commit()

    def test_only_elapsed_day_reviews_are_eligible(self):
        self._log("good", 0)
        cal = self.store.get_calibration(user_id="alice")
        self.assertEqual(cal["eligible_reviews"], 0)
        self.assertTrue(cal["low_signal"])

    def test_true_retention_computation(self):
        for _ in range(30):
            self._log("good", 2)
        for _ in range(10):
            self._log("easy", 3)
        for _ in range(10):
            self._log("again", 2)
        cal = self.store.get_calibration(user_id="alice")
        self.assertEqual(cal["eligible_reviews"], 50)
        self.assertEqual(cal["successful_reviews"], 40)
        self.assertEqual(cal["true_retention"], 0.8)
        self.assertTrue(cal["within_band"])
        self.assertFalse(cal["low_signal"])

    def test_low_signal_when_few_reviews(self):
        self._log("good", 2)
        cal = self.store.get_calibration(user_id="alice")
        self.assertEqual(cal["eligible_reviews"], 1)
        self.assertTrue(cal["low_signal"])

    def test_low_signal_when_mostly_good(self):
        for _ in range(60):
            self._log("good", 2)
        cal = self.store.get_calibration(user_id="alice")
        self.assertGreater(cal["good_rating_pct"], 0.8)
        self.assertTrue(cal["low_signal"])


class BackfillTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.store = LearningStore(
            os.path.join(self._td.name, "learning.db")
        )

    def _progress_row(
        self,
        user_id: str,
        question_id: str,
        topic_id: str,
        mastery: float,
        bucket: int,
        attempts: int = 3,
        correct: int = 2,
    ) -> None:
        now_iso = _to_iso(_now())
        self.store._conn.execute(
            """
            INSERT INTO question_progress(
                user_id, question_id, topic_id, mastery_score, due_at,
                review_bucket, attempts, correct_attempts, avg_confidence,
                last_confidence, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                question_id,
                topic_id,
                mastery,
                now_iso,
                bucket,
                attempts,
                correct,
                3.0,
                3,
                now_iso,
            ),
        )
        self.store._conn.commit()

    def test_backfill_migrates_rows(self):
        self._progress_row("alice", "q1", "topic-a", 0.9, 3)
        self._progress_row("alice", "q2", "topic-a", 0.1, 0)
        self._progress_row("bob", "q3", "topic-b", 0.5, 2)
        result = self.store.backfill_fsrs_from_progress()
        self.assertEqual(result["question_progress_rows"], 3)
        self.assertEqual(result["inserted"], 3)
        cards = self.store._conn.execute(
            "SELECT COUNT(*) FROM fsrs_cards"
        ).fetchone()[0]
        self.assertEqual(cards, 3)

    def test_backfill_derives_state(self):
        self._progress_row("alice", "q-new", "topic-a", 0.5, 0)
        self._progress_row("alice", "q-review", "topic-a", 0.9, 3)
        self._progress_row("alice", "q-relearn", "topic-a", 0.1, 4)
        self.store.backfill_fsrs_from_progress()
        rows = {
            row["card_id"]: row
            for row in self.store._conn.execute(
                "SELECT card_id, state, reps, lapses, scheduled_days "
                "FROM fsrs_cards WHERE user_id = 'alice'"
            ).fetchall()
        }
        self.assertEqual(rows["q-new"]["state"], "learning")
        self.assertEqual(rows["q-review"]["state"], "review")
        self.assertEqual(rows["q-relearn"]["state"], "relearning")
        self.assertEqual(rows["q-review"]["reps"], 3)
        self.assertEqual(rows["q-review"]["lapses"], 1)
        self.assertEqual(rows["q-review"]["scheduled_days"], 7)

    def test_backfill_is_idempotent(self):
        self._progress_row("alice", "q1", "topic-a", 0.9, 3)
        first = self.store.backfill_fsrs_from_progress()
        second = self.store.backfill_fsrs_from_progress()
        third = self.store.backfill_fsrs_from_progress()
        self.assertEqual(first["inserted"], 1)
        self.assertEqual(second["inserted"], 0)
        self.assertTrue(third["skipped"])
        cards = self.store._conn.execute(
            "SELECT COUNT(*) FROM fsrs_cards"
        ).fetchone()[0]
        self.assertEqual(cards, 1)

    def test_backfill_skips_when_fsrs_cards_populated(self):
        self.store.record_review(
            user_id="alice",
            card_id="q1",
            topic_id="topic-a",
            rating="good",
            response_time_ms=100,
        )
        self._progress_row("alice", "q2", "topic-a", 0.9, 3)
        result = self.store.backfill_fsrs_from_progress()
        self.assertTrue(result["skipped"])
        self.assertEqual(result["inserted"], 0)

    def test_backfilled_cards_are_reviewable(self):
        self._progress_row("alice", "q1", "topic-a", 0.9, 3)
        self.store.backfill_fsrs_from_progress()
        result = self.store.record_review(
            user_id="alice",
            card_id="q1",
            topic_id="topic-a",
            rating="good",
            response_time_ms=100,
        )
        self.assertEqual(result["card"]["reps"], 4)
        self.assertEqual(result["card"]["state"], "review")


class TopicMasteryTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.store = LearningStore(
            os.path.join(self._td.name, "learning.db")
        )

    def test_fsrs_topic_mastery_groups_by_topic(self):
        for card_id in ("q1", "q2"):
            self.store.record_review(
                user_id="alice",
                card_id=card_id,
                topic_id="topic-a",
                rating="easy",
                response_time_ms=100,
            )
        self.store.record_review(
            user_id="alice",
            card_id="q3",
            topic_id="topic-b",
            rating="easy",
            response_time_ms=100,
        )
        mastery = self.store.get_fsrs_topic_mastery(user_id="alice")
        topics = {t["topic_id"]: t for t in mastery["topics"]}
        self.assertEqual(topics["topic-a"]["attempts"], 2)
        self.assertEqual(topics["topic-b"]["attempts"], 1)
        for topic in mastery["topics"]:
            self.assertGreaterEqual(topic["mastery_score"], 0.0)
            self.assertLessEqual(topic["mastery_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
