"""Tests for the FSRS scheduler facade (py-fsrs containment layer)."""

from __future__ import annotations

import random
import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from fsrs import Scheduler as FSRSScheduler

from app.services import fsrs_scheduler


def _no_fuzz_scheduler() -> FSRSScheduler:
    return FSRSScheduler(enable_fuzzing=False)


class FSRSchedulerCreateCardTests(unittest.TestCase):
    def test_create_card_returns_new_card(self):
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        card = fsrs_scheduler.create_card(now)
        self.assertEqual(card["state"], "new")
        self.assertIsNone(card["stability"])
        self.assertIsNone(card["difficulty"])
        self.assertIsNone(card["last_review_at"])
        self.assertEqual(card["due_at"], now.isoformat())
        self.assertEqual(card["reps"], 0)
        self.assertEqual(card["lapses"], 0)

    def test_create_card_defaults_to_now(self):
        before = datetime.now(UTC)
        card = fsrs_scheduler.create_card()
        after = datetime.now(UTC)
        due = datetime.fromisoformat(card["due_at"])
        self.assertLessEqual(before, due)
        self.assertLessEqual(due, after)


class FSRSchedulerReviewVectorTests(unittest.TestCase):
    """Known FSRS-4.5 vectors (default weights, fuzz disabled)."""

    def setUp(self):
        self.now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        self._patcher = patch.object(
            fsrs_scheduler, "_scheduler", _no_fuzz_scheduler()
        )
        self._patcher.start()
        self.addCleanup(self._patcher.stop)

    def _review(self, rating: str):
        card = fsrs_scheduler.create_card(self.now)
        return fsrs_scheduler.review(card, rating, self.now, 1500)

    def test_again_vector(self):
        new_card, log = self._review("again")
        self.assertEqual(new_card["state"], "learning")
        self.assertEqual(
            new_card["due_at"],
            (self.now + timedelta(minutes=1)).isoformat(),
        )
        self.assertAlmostEqual(new_card["stability"], 0.40255, places=4)
        self.assertAlmostEqual(new_card["difficulty"], 7.1949, places=3)
        self.assertEqual(new_card["reps"], 1)
        self.assertEqual(new_card["lapses"], 1)
        self.assertEqual(new_card["scheduled_days"], 0)
        self.assertEqual(new_card["elapsed_days"], 0)
        self.assertEqual(log["rating"], "again")
        self.assertEqual(log["review_duration_ms"], 1500)

    def test_hard_vector(self):
        new_card, _log = self._review("hard")
        self.assertEqual(new_card["state"], "learning")
        self.assertEqual(
            new_card["due_at"],
            (self.now + timedelta(minutes=5, seconds=30)).isoformat(),
        )
        self.assertAlmostEqual(new_card["stability"], 1.18385, places=4)
        self.assertEqual(new_card["lapses"], 0)

    def test_good_vector(self):
        new_card, _log = self._review("good")
        self.assertEqual(new_card["state"], "learning")
        self.assertEqual(
            new_card["due_at"],
            (self.now + timedelta(minutes=10)).isoformat(),
        )
        self.assertAlmostEqual(new_card["stability"], 3.173, places=3)
        self.assertAlmostEqual(new_card["difficulty"], 5.2824, places=3)

    def test_easy_vector(self):
        new_card, _log = self._review("easy")
        self.assertEqual(new_card["state"], "review")
        self.assertEqual(
            new_card["due_at"],
            (self.now + timedelta(days=16)).isoformat(),
        )
        self.assertAlmostEqual(new_card["stability"], 15.69105, places=4)
        self.assertAlmostEqual(new_card["difficulty"], 3.2245, places=3)
        self.assertEqual(new_card["scheduled_days"], 16)

    def test_review_state_vectors(self):
        card = fsrs_scheduler.create_card(self.now)
        card, _log = fsrs_scheduler.review(card, "easy", self.now, 100)
        later = self.now + timedelta(days=3)
        dues = {}
        for rating in ("again", "hard", "good", "easy"):
            reviewed, _ = fsrs_scheduler.review(card, rating, later, 100)
            dues[rating] = datetime.fromisoformat(reviewed["due_at"])
        self.assertEqual(dues["again"], later + timedelta(minutes=10))
        self.assertEqual(dues["hard"], later + timedelta(days=18))
        self.assertEqual(dues["good"], later + timedelta(days=25))
        self.assertEqual(dues["easy"], later + timedelta(days=43))

    def test_review_accumulates_reps_and_elapsed_days(self):
        card = fsrs_scheduler.create_card(self.now)
        card, _ = fsrs_scheduler.review(card, "good", self.now, 100)
        later = self.now + timedelta(days=2, hours=3)
        card, _ = fsrs_scheduler.review(card, "good", later, 100)
        self.assertEqual(card["reps"], 2)
        self.assertEqual(card["elapsed_days"], 2)
        self.assertEqual(card["lapses"], 0)

    def test_invalid_rating_raises(self):
        card = fsrs_scheduler.create_card(self.now)
        with self.assertRaises(ValueError):
            fsrs_scheduler.review(card, "bogus", self.now, 100)


class FSRSchedulerLeechTests(unittest.TestCase):
    def test_is_leech_threshold(self):
        card = {"lapses": 4}
        self.assertFalse(fsrs_scheduler.is_leech(card))
        card = {"lapses": 5}
        self.assertTrue(fsrs_scheduler.is_leech(card))
        card = {"lapses": 9}
        self.assertTrue(fsrs_scheduler.is_leech(card))
        self.assertFalse(fsrs_scheduler.is_leech({}))


class FSRSchedulerRetentionTests(unittest.TestCase):
    def test_new_card_has_zero_retention(self):
        card = fsrs_scheduler.create_card()
        self.assertEqual(fsrs_scheduler.retention_estimate(card), 0.0)

    def test_just_reviewed_card_has_full_retention(self):
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        card = fsrs_scheduler.create_card(now)
        card, _ = fsrs_scheduler.review(card, "easy", now, 100)
        estimate = fsrs_scheduler.retention_estimate(card, now)
        self.assertAlmostEqual(estimate, 1.0, places=6)

    def test_retention_decays_with_elapsed_time(self):
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        card = fsrs_scheduler.create_card(now)
        card, _ = fsrs_scheduler.review(card, "easy", now, 100)
        later = now + timedelta(days=30)
        estimate = fsrs_scheduler.retention_estimate(card, later)
        self.assertGreater(estimate, 0.0)
        self.assertLess(estimate, 1.0)


class FSRSchedulerDelegationTests(unittest.TestCase):
    def test_due_cards_forecast_calibration_delegate_to_store(self):
        class _StubStore:
            def get_due_cards(self, *, user_id, limit=50):
                return [{"card_id": "c1"}]

            def get_forecast(self, *, user_id, days=7):
                return {"due_counts": [], "total_due": 0}

            def get_calibration(self, *, user_id):
                return {"eligible_reviews": 0}

        store = _StubStore()
        self.assertEqual(
            fsrs_scheduler.due_cards(store, "alice", limit=5),
            [{"card_id": "c1"}],
        )
        self.assertEqual(
            fsrs_scheduler.forecast(store, "alice", days=3),
            {"due_counts": [], "total_due": 0},
        )
        self.assertEqual(
            fsrs_scheduler.calibration(store, "alice"),
            {"eligible_reviews": 0},
        )


class FSRSchedulerPropertyTests(unittest.TestCase):
    def test_due_at_ordering_again_lt_hard_lt_good_lt_easy(self):
        """Property: for random rating sequences, the four ratings
        produce strictly increasing due dates from the same state."""
        rng = random.Random(20260101)
        with patch.object(
            fsrs_scheduler, "_scheduler", _no_fuzz_scheduler()
        ):
            for _ in range(100):
                now = datetime(2026, 1, 1, tzinfo=UTC)
                card = fsrs_scheduler.create_card(now)
                # Build up random state with a random rating sequence.
                for _step in range(rng.randint(0, 6)):
                    rating = rng.choice(fsrs_scheduler.RATINGS)
                    card, _log = fsrs_scheduler.review(
                        card, rating, now, rng.randint(100, 5000)
                    )
                    now = now + timedelta(
                        minutes=rng.randint(10, 60 * 24 * 4)
                    )
                dues = {}
                for rating in fsrs_scheduler.RATINGS:
                    reviewed, _log = fsrs_scheduler.review(
                        card, rating, now, 1000
                    )
                    dues[rating] = datetime.fromisoformat(
                        reviewed["due_at"]
                    )
                self.assertLess(dues["again"], dues["hard"])
                self.assertLess(dues["hard"], dues["good"])
                self.assertLess(dues["good"], dues["easy"])

    def test_default_facade_scheduler_uses_library_defaults(self):
        scheduler = fsrs_scheduler._get_scheduler()
        self.assertIsInstance(scheduler, FSRSScheduler)
        self.assertTrue(scheduler.enable_fuzzing)
        self.assertEqual(scheduler.desired_retention, 0.9)


if __name__ == "__main__":
    unittest.main()
