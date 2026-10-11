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
        self.assertEqual(third["inserted"], 0)
        # The per-row INSERT OR IGNORE guard — not a global
        # "fsrs_cards is populated" skip — is what makes the
        # repeat runs no-ops.
        self.assertFalse(third["skipped"])
        cards = self.store._conn.execute(
            "SELECT COUNT(*) FROM fsrs_cards"
        ).fetchone()[0]
        self.assertEqual(cards, 1)

    def test_backfill_migrates_each_user_independently(self):
        # A card that already exists for one user must not stop
        # another user's legacy rows from migrating. Mixed-state
        # databases (some users migrated, others not) rely on the
        # per-row (user_id, card_id) guard, not a global skip.
        self.store.record_review(
            user_id="alice",
            card_id="q1",
            topic_id="topic-a",
            rating="good",
            response_time_ms=100,
        )
        self._progress_row("alice", "q2", "topic-a", 0.9, 3)
        self._progress_row("bob", "q3", "topic-b", 0.5, 2)
        result = self.store.backfill_fsrs_from_progress()
        self.assertFalse(result["skipped"])
        # alice's q1 already exists and is ignored; her q2 and
        # bob's q3 still migrate.
        self.assertEqual(result["inserted"], 2)
        cards = self.store._conn.execute(
            "SELECT user_id, card_id FROM fsrs_cards ORDER BY user_id, card_id"
        ).fetchall()
        self.assertEqual(
            [(row["user_id"], row["card_id"]) for row in cards],
            [("alice", "q1"), ("alice", "q2"), ("bob", "q3")],
        )
        # A second run migrates nothing new but never skips.
        again = self.store.backfill_fsrs_from_progress()
        self.assertEqual(again["inserted"], 0)
        self.assertFalse(again["skipped"])

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

    def test_fsrs_topic_mastery_reports_accuracy_and_due_count(self):
        # Two easy reviews (successful) and one again review
        # (a lapse) on topic-a; topic-b has a single good
        # review. Accuracy rolls up from the review log.
        for rating in ("easy", "easy", "again"):
            self.store.record_review(
                user_id="alice",
                card_id=f"q-{rating}",
                topic_id="topic-a",
                rating=rating,
                response_time_ms=100,
            )
        self.store.record_review(
            user_id="alice",
            card_id="q-good",
            topic_id="topic-b",
            rating="good",
            response_time_ms=100,
        )
        mastery = self.store.get_fsrs_topic_mastery(user_id="alice")
        topics = {t["topic_id"]: t for t in mastery["topics"]}
        # topic-a: 2 of 3 reviews were good/easy.
        self.assertEqual(topics["topic-a"]["accuracy"], round(2 / 3, 4))
        self.assertEqual(topics["topic-b"]["accuracy"], 1.0)
        # Nothing is due yet (fresh reviews schedule forward).
        self.assertEqual(topics["topic-a"]["due_count"], 0)
        self.assertEqual(topics["topic-b"]["due_count"], 0)


class AttemptDualWriteTests(unittest.TestCase):
    """Quiz attempts and interview turns dual-write into fsrs_cards."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.store = LearningStore(
            os.path.join(self._td.name, "learning.db")
        )

    def test_attempt_creates_fsrs_card_and_log(self):
        self.store.record_attempt(
            user_id="alice",
            question_id="q1",
            topic_id="topic-a",
            user_answer="wrong",
            is_correct=False,
            confidence=2,
            response_time_ms=900,
            mode="quiz",
        )
        card = self.store._conn.execute(
            "SELECT card_id, topic_id, source_type, state, lapses "
            "FROM fsrs_cards WHERE user_id = ? AND card_id = ?",
            ("alice", "q1"),
        ).fetchone()
        self.assertIsNotNone(card)
        self.assertEqual(card["topic_id"], "topic-a")
        # study/quiz modes map to the "question" source type.
        self.assertEqual(card["source_type"], "question")
        # Incorrect → again → learning state + a lapse.
        self.assertEqual(card["state"], "learning")
        self.assertEqual(card["lapses"], 1)
        log = self.store._conn.execute(
            "SELECT rating, review_duration_ms FROM fsrs_review_log "
            "WHERE user_id = ? AND card_id = ?",
            ("alice", "q1"),
        ).fetchone()
        self.assertEqual(log["rating"], "again")
        self.assertEqual(log["review_duration_ms"], 900)

    def test_attempt_upserts_existing_card(self):
        self.store.record_review(
            user_id="alice",
            card_id="q1",
            topic_id="topic-a",
            rating="good",
            response_time_ms=100,
        )
        self.store.record_attempt(
            user_id="alice",
            question_id="q1",
            topic_id="topic-a",
            user_answer="wrong",
            is_correct=False,
            confidence=2,
            response_time_ms=900,
            mode="quiz",
        )
        card = self.store._conn.execute(
            "SELECT reps, lapses FROM fsrs_cards WHERE user_id = ? AND card_id = ?",
            ("alice", "q1"),
        ).fetchone()
        # One manual review + one attempt = 2 reps, 1 lapse.
        self.assertEqual(card["reps"], 2)
        self.assertEqual(card["lapses"], 1)

    def test_attempt_derives_fsrs_rating_from_confidence(self):
        # The (is_correct, confidence) pair is mapped onto
        # the FSRS ratings; the resulting card state and
        # lapse counter reflect the derived rating. For a
        # brand-new card only "easy" graduates to the
        # review state — "hard"/"good" stay in learning.
        cases = (
            (False, 5, "learning", 1),   # incorrect → again
            (True, 1, "learning", 0),    # correct, low confidence → hard
            (True, 3, "learning", 0),    # correct, neutral → good
            (True, 5, "review", 0),      # correct, confident → easy
        )
        for idx, (is_correct, confidence, state, lapses) in enumerate(cases):
            with self.subTest(case=idx):
                question_id = f"q-derive-{idx}"
                self.store.record_attempt(
                    user_id="alice",
                    question_id=question_id,
                    topic_id="topic-a",
                    user_answer="a",
                    is_correct=is_correct,
                    confidence=confidence,
                    response_time_ms=900,
                    mode="quiz",
                )
                card = self.store._conn.execute(
                    "SELECT state, lapses FROM fsrs_cards "
                    "WHERE user_id = ? AND card_id = ?",
                    ("alice", question_id),
                ).fetchone()
                self.assertEqual(card["state"], state)
                self.assertEqual(card["lapses"], lapses)

    def test_interview_turn_creates_interview_source_card(self):
        # Interview turns are recorded as attempts with an
        # ``interview:``-prefixed question id; they land in
        # fsrs_cards as interview-sourced cards so the
        # planner sees interview evidence.
        self.store.record_attempt(
            user_id="alice",
            question_id="interview:session-1:0",
            topic_id="01-backend",
            user_answer="wrong",
            is_correct=False,
            confidence=2,
            response_time_ms=1500,
            mode="study",
        )
        card = self.store._conn.execute(
            "SELECT card_id, topic_id, source_type, state, lapses "
            "FROM fsrs_cards WHERE user_id = ? AND card_id = ?",
            ("alice", "interview:session-1:0"),
        ).fetchone()
        self.assertIsNotNone(card)
        self.assertEqual(card["topic_id"], "01-backend")
        self.assertEqual(card["source_type"], "interview")
        # Incorrect turn → again → learning state + a lapse.
        self.assertEqual(card["state"], "learning")
        self.assertEqual(card["lapses"], 1)
        log = self.store._conn.execute(
            "SELECT rating FROM fsrs_review_log "
            "WHERE user_id = ? AND card_id = ?",
            ("alice", "interview:session-1:0"),
        ).fetchone()
        self.assertEqual(log["rating"], "again")


class ReviewQueueRankingTests(unittest.TestCase):
    """Queue ranking: default due order, threshold, weak-area boost."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.store = LearningStore(
            os.path.join(self._td.name, "learning.db")
        )

    def _card(
        self,
        card_id: str,
        *,
        topic_id: str = "topic-a",
        rating: str = "easy",
        due_at: datetime | None = None,
        stability: float | None = None,
        last_review_at: datetime | None = None,
    ) -> None:
        """Create a reviewed card with explicit scheduling fields."""
        self.store.record_review(
            user_id="alice",
            card_id=card_id,
            topic_id=topic_id,
            rating=rating,
            response_time_ms=100,
        )
        updates: dict[str, object] = {}
        if due_at is not None:
            updates["due_at"] = _to_iso(due_at)
        if stability is not None:
            updates["stability"] = float(stability)
        if last_review_at is not None:
            updates["last_review_at"] = _to_iso(last_review_at)
        if updates:
            assignments = ", ".join(f"{key} = ?" for key in updates)
            self.store._conn.execute(
                f"UPDATE fsrs_cards SET {assignments} "
                "WHERE user_id = ? AND card_id = ?",
                (*updates.values(), "alice", card_id),
            )
            self.store._conn.commit()

    def test_default_queue_orders_oldest_due_first(self):
        now = _now()
        self._card("q-new", due_at=now - timedelta(hours=1))
        self._card("q-mid", due_at=now - timedelta(hours=2))
        self._card("q-old", due_at=now - timedelta(hours=3))
        queue = self.store.get_fsrs_review_queue(user_id="alice", limit=10)
        self.assertEqual(
            [item["question_id"] for item in queue["items"]],
            ["q-old", "q-mid", "q-new"],
        )
        self.assertEqual(queue["total_due"], 3)

    def test_explicit_defaults_match_the_plain_queue(self):
        now = _now()
        self._card("q1", due_at=now - timedelta(hours=2))
        self._card("q2", due_at=now - timedelta(hours=1))
        default = self.store.get_fsrs_review_queue(user_id="alice", limit=10)
        explicit = self.store.get_fsrs_review_queue(
            user_id="alice",
            limit=10,
            retrievability_threshold=None,
            weak_area_boost=False,
        )
        self.assertEqual(default, explicit)

    def test_threshold_pulls_low_retrievability_item_forward(self):
        now = _now()
        # Fading card: due tomorrow but long forgotten (retention ~0.82).
        self._card(
            "q-fading",
            due_at=now + timedelta(days=1),
            stability=5.0,
            last_review_at=now - timedelta(days=10),
        )
        # Merely-due card: reviewed an hour ago, retention ~1.0.
        self._card(
            "q-strong",
            due_at=now - timedelta(hours=2),
            stability=100.0,
            last_review_at=now - timedelta(hours=1),
        )
        default = self.store.get_fsrs_review_queue(user_id="alice", limit=10)
        self.assertEqual(
            [item["question_id"] for item in default["items"]], ["q-strong"]
        )
        boosted = self.store.get_fsrs_review_queue(
            user_id="alice", limit=10, retrievability_threshold=0.9
        )
        self.assertEqual(
            [item["question_id"] for item in boosted["items"]],
            ["q-fading", "q-strong"],
        )
        self.assertEqual(boosted["total_due"], 2)

    def test_threshold_leaves_healthy_queue_untouched(self):
        now = _now()
        self._card(
            "q-strong",
            due_at=now - timedelta(hours=1),
            stability=100.0,
            last_review_at=now - timedelta(hours=1),
        )
        queue = self.store.get_fsrs_review_queue(
            user_id="alice", limit=10, retrievability_threshold=0.9
        )
        self.assertEqual(
            [item["question_id"] for item in queue["items"]], ["q-strong"]
        )

    def test_weak_area_boost_orders_weakest_topic_first(self):
        now = _now()
        self._card(
            "q-secure",
            topic_id="topic-secure",
            due_at=now - timedelta(hours=2),
            stability=100.0,
            last_review_at=now - timedelta(hours=1),
        )
        self._card(
            "q-shaky",
            topic_id="topic-shaky",
            due_at=now - timedelta(hours=1),
            stability=5.0,
            last_review_at=now - timedelta(days=10),
        )
        default = self.store.get_fsrs_review_queue(user_id="alice", limit=10)
        self.assertEqual(
            [item["question_id"] for item in default["items"]],
            ["q-secure", "q-shaky"],
        )
        boosted = self.store.get_fsrs_review_queue(
            user_id="alice", limit=10, weak_area_boost=True
        )
        self.assertEqual(
            [item["question_id"] for item in boosted["items"]],
            ["q-shaky", "q-secure"],
        )


class LeechSuspensionTests(unittest.TestCase):
    """Leeches auto-suspend out of the queue until explicitly resumed."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.store = LearningStore(
            os.path.join(self._td.name, "learning.db")
        )

    def _leech(self, card_id: str = "q-leech") -> dict:
        result = None
        for _ in range(5):
            result = self.store.record_review(
                user_id="alice",
                card_id=card_id,
                topic_id="topic-a",
                rating="again",
                response_time_ms=100,
            )
        # Force the card due now so suspension is the only thing
        # that can exclude it from the queue.
        self.store._conn.execute(
            "UPDATE fsrs_cards SET due_at = ? "
            "WHERE user_id = ? AND card_id = ?",
            (_to_iso(_now() - timedelta(hours=1)), "alice", card_id),
        )
        self.store._conn.commit()
        assert result is not None
        return result

    def test_leech_is_auto_suspended_out_of_the_queue(self):
        result = self._leech()
        self.assertTrue(result["leech"])
        self.assertEqual(result["card"]["lapses"], 5)
        self.assertEqual(result["card"]["suspended"], 1)
        self.assertEqual(self.store.get_due_cards(user_id="alice", limit=10), [])
        queue = self.store.get_fsrs_review_queue(user_id="alice", limit=10)
        self.assertEqual(queue["items"], [])
        self.assertEqual(queue["total_due"], 0)

    def test_suspension_also_excludes_from_the_filtered_queue(self):
        self._leech("q-leech")
        self.store.record_review(
            user_id="alice",
            card_id="q1",
            topic_id="topic-a",
            rating="good",
            response_time_ms=100,
        )
        self.store._conn.execute(
            "UPDATE fsrs_cards SET due_at = ? "
            "WHERE user_id = ? AND card_id = ?",
            (_to_iso(_now() - timedelta(hours=1)), "alice", "q1"),
        )
        self.store._conn.commit()
        queue = self.store.get_fsrs_review_queue(
            user_id="alice", limit=10, retrievability_threshold=0.99
        )
        self.assertEqual(
            [item["question_id"] for item in queue["items"]], ["q1"]
        )

    def test_get_leeches_lists_suspended_leeches(self):
        self._leech("q-leech")
        self._leech("q-leech-2")
        leeches = self.store.get_leeches(user_id="alice")
        self.assertEqual(
            [card["card_id"] for card in leeches], ["q-leech", "q-leech-2"]
        )
        self.assertEqual(leeches[0]["suspended"], 1)
        self.assertGreaterEqual(leeches[0]["lapses"], 5)
        self.assertEqual(self.store.get_leeches(user_id="bob"), [])

    def test_non_leech_cards_are_not_listed_as_leeches(self):
        result = None
        for _ in range(4):
            result = self.store.record_review(
                user_id="alice",
                card_id="q1",
                topic_id="topic-a",
                rating="again",
                response_time_ms=100,
            )
        self.assertFalse(result["leech"])
        self.assertEqual(result["card"]["suspended"], 0)
        self.assertEqual(self.store.get_leeches(user_id="alice"), [])

    def test_resume_restores_the_card_to_the_queue(self):
        self._leech()
        self.assertEqual(
            self.store.get_due_cards(user_id="alice", limit=10), []
        )
        resumed = self.store.resume_leeched_card(
            user_id="alice", card_id="q-leech"
        )
        self.assertIsNotNone(resumed)
        self.assertEqual(resumed["suspended"], 0)
        due = self.store.get_due_cards(user_id="alice", limit=10)
        self.assertEqual([card["card_id"] for card in due], ["q-leech"])
        self.assertEqual(self.store.get_leeches(user_id="alice"), [])
        queue = self.store.get_fsrs_review_queue(user_id="alice", limit=10)
        self.assertEqual(
            [item["question_id"] for item in queue["items"]], ["q-leech"]
        )

    def test_resume_unknown_or_unsuspended_card_returns_none(self):
        self.store.record_review(
            user_id="alice",
            card_id="q1",
            topic_id="topic-a",
            rating="good",
            response_time_ms=100,
        )
        self.assertIsNone(
            self.store.resume_leeched_card(user_id="alice", card_id="q1")
        )
        self.assertIsNone(
            self.store.resume_leeched_card(user_id="alice", card_id="nope")
        )
        self._leech()
        self.assertIsNone(
            self.store.resume_leeched_card(user_id="bob", card_id="q-leech")
        )


class AttemptHintRatingTests(unittest.TestCase):
    """Hint usage downgrades the rating derived for an attempt."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.store = LearningStore(
            os.path.join(self._td.name, "learning.db")
        )

    def _log_rating(self, question_id: str) -> str:
        log = self.store._conn.execute(
            "SELECT rating FROM fsrs_review_log "
            "WHERE user_id = ? AND card_id = ?",
            ("alice", question_id),
        ).fetchone()
        assert log is not None
        return str(log["rating"])

    def test_hint_usage_downgrades_the_derived_rating(self):
        self.store.record_attempt(
            user_id="alice",
            question_id="q-hinted",
            topic_id="topic-a",
            user_answer="a",
            is_correct=True,
            confidence=5,
            response_time_ms=900,
            mode="quiz",
            hints_used=2,
        )
        # (correct, confident) is "easy"; two hints downgrade to "hard".
        self.assertEqual(self._log_rating("q-hinted"), "hard")

    def test_attempt_without_hints_keeps_the_plain_rating(self):
        self.store.record_attempt(
            user_id="alice",
            question_id="q-plain",
            topic_id="topic-a",
            user_answer="a",
            is_correct=True,
            confidence=5,
            response_time_ms=900,
            mode="quiz",
        )
        self.assertEqual(self._log_rating("q-plain"), "easy")

    def test_heavy_hint_usage_floors_the_rating_at_again(self):
        self.store.record_attempt(
            user_id="alice",
            question_id="q-scaffolded",
            topic_id="topic-a",
            user_answer="a",
            is_correct=True,
            confidence=5,
            response_time_ms=900,
            mode="quiz",
            hints_used=9,
        )
        self.assertEqual(self._log_rating("q-scaffolded"), "again")


class InterviewCompetencyCardTests(unittest.TestCase):
    """Interview turns schedule competency cards, and no topic mastery.

    The interview loop records ``interview-competency:<dimension>`` with
    ``mode="interview"`` so a behavioral/communication/system-design skill is
    rehearsed on its own schedule. Those rows must not reach topic mastery,
    weak areas or the analytics curve, but must stay readable for competency
    analysis.
    """

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.store = LearningStore(
            os.path.join(self._td.name, "learning.db")
        )

    def _interview_turn(self, dimension, *, is_correct, confidence):
        self.store.record_attempt(
            user_id="alice",
            question_id=f"interview-competency:{dimension}",
            topic_id="interview-competency",
            user_answer="answer",
            is_correct=is_correct,
            confidence=confidence,
            response_time_ms=900,
            mode="interview",
        )

    def _quiz_review(self, card_id, topic_id, rating="easy"):
        self.store.record_review(
            user_id="alice",
            card_id=card_id,
            topic_id=topic_id,
            rating=rating,
            response_time_ms=100,
        )

    def test_a_competency_attempt_lands_as_an_interview_card(self):
        self._interview_turn(
            "communication_clarity", is_correct=True, confidence=5
        )
        card = self.store._conn.execute(
            "SELECT card_id, topic_id, source_type FROM fsrs_cards "
            "WHERE user_id = ? AND card_id = ?",
            ("alice", "interview-competency:communication_clarity"),
        ).fetchone()
        self.assertIsNotNone(card)
        self.assertEqual(card["topic_id"], "interview-competency")
        self.assertEqual(card["source_type"], "interview")

    def test_competency_cards_are_excluded_from_topic_mastery(self):
        self._quiz_review("q1", "01-http")
        self._interview_turn(
            "technical_accuracy", is_correct=False, confidence=2
        )
        mastery = self.store.get_fsrs_topic_mastery(user_id="alice")
        topics = {t["topic_id"] for t in mastery["topics"]}
        self.assertIn("01-http", topics)
        self.assertNotIn("interview-competency", topics)
        by_topic = {t["topic_id"]: t for t in mastery["topics"]}
        self.assertEqual(by_topic["01-http"]["accuracy"], 1.0)

    def test_competency_cards_are_excluded_from_weak_areas(self):
        self._quiz_review("q1", "01-http", rating="easy")
        # A failing competency does not invent a weak topic.
        for dimension in ("communication_clarity", "confidence_signal"):
            self._interview_turn(
                dimension, is_correct=False, confidence=1
            )
        weak = self.store.get_weak_areas(user_id="alice")
        topic_ids = {a["topic_id"] for a in weak["weak_areas"]}
        self.assertNotIn("interview-competency", topic_ids)
        self.assertIn("01-http", topic_ids)

    def test_competency_cards_stay_readable_for_competency_analysis(self):
        self._interview_turn(
            "communication_clarity", is_correct=True, confidence=5
        )
        self._interview_turn(
            "technical_accuracy", is_correct=False, confidence=1
        )
        result = self.store.get_interview_competencies(user_id="alice")
        self.assertEqual(result["total"], 2)
        competencies = {
            item["competency"]: item for item in result["competencies"]
        }
        self.assertEqual(
            sorted(competencies),
            ["communication_clarity", "technical_accuracy"],
        )
        self.assertEqual(competencies["technical_accuracy"]["reps"], 1)
        self.assertEqual(competencies["technical_accuracy"]["lapses"], 1)
        # Weakest-first ordering, so the next round can focus on the gap.
        self.assertEqual(result["weakest"][0], "technical_accuracy")

    def test_another_users_competency_cards_do_not_leak(self):
        self._interview_turn("reasoning_depth", is_correct=True, confidence=4)
        self.store.record_attempt(
            user_id="bob",
            question_id="interview-competency:reasoning_depth",
            topic_id="interview-competency",
            user_answer="answer",
            is_correct=True,
            confidence=4,
            response_time_ms=900,
            mode="interview",
        )
        self.assertEqual(
            self.store.get_interview_competencies(user_id="bob")["total"], 1
        )


if __name__ == "__main__":
    unittest.main()
