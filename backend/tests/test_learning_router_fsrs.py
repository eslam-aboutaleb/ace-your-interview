"""Contract tests for the FSRS learning endpoints."""

from __future__ import annotations

import os
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.dependencies import require_auth
from app.routers import learning
from app.services.learning_planner import LearningPlannerStore
from app.services.learning_store import LearningStore


class LearningRouterFSRSContractTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.db_path = os.path.join(self._td.name, "learning.db")
        self.learning_store = LearningStore(self.db_path)
        self.planner_store = LearningPlannerStore(self.db_path)
        learning.init(self.learning_store, self.planner_store)

        app = FastAPI()
        app.include_router(learning.router)
        app.dependency_overrides[require_auth] = lambda: {
            "user": "test-user",
            "provider": "local",
        }
        self.client = TestClient(app)

    def test_review_endpoint_contract(self):
        res = self.client.post(
            "/api/learning/review",
            json={
                "card_id": "q1",
                "topic_id": "01-backend-fundamentals-and-http",
                "rating": "good",
                "response_time_ms": 1500,
                "source_type": "question",
            },
        )
        self.assertEqual(res.status_code, 201)
        body = res.json()
        self.assertIn("card", body)
        self.assertIn("log", body)
        self.assertIn("leech", body)
        card = body["card"]
        self.assertEqual(card["card_id"], "q1")
        self.assertEqual(card["topic_id"], "01-backend-fundamentals-and-http")
        self.assertEqual(card["source_type"], "question")
        self.assertEqual(card["state"], "learning")
        self.assertEqual(card["reps"], 1)
        self.assertEqual(card["lapses"], 0)
        self.assertIsNotNone(card["stability"])
        self.assertIsNotNone(card["difficulty"])
        self.assertTrue(card["due_at"])
        log = body["log"]
        self.assertEqual(log["card_id"], "q1")
        self.assertEqual(log["rating"], "good")
        self.assertEqual(log["review_duration_ms"], 1500)
        self.assertEqual(log["state"], "learning")
        self.assertIsInstance(body["leech"], bool)
        self.assertFalse(body["leech"])

    def test_review_defaults_source_type_and_response_time(self):
        res = self.client.post(
            "/api/learning/review",
            json={
                "card_id": "q1",
                "topic_id": "topic-a",
                "rating": "again",
            },
        )
        self.assertEqual(res.status_code, 201)
        body = res.json()
        self.assertEqual(body["card"]["source_type"], "question")
        self.assertEqual(body["card"]["lapses"], 1)
        self.assertEqual(body["log"]["review_duration_ms"], 0)

    def test_review_rejects_invalid_rating(self):
        res = self.client.post(
            "/api/learning/review",
            json={
                "card_id": "q1",
                "topic_id": "topic-a",
                "rating": "bogus",
            },
        )
        self.assertEqual(res.status_code, 422)

    def test_review_rejects_missing_fields(self):
        res = self.client.post(
            "/api/learning/review",
            json={"card_id": "q1"},
        )
        self.assertEqual(res.status_code, 422)

    def test_review_accumulates_lapses_and_flags_leech(self):
        for _ in range(5):
            res = self.client.post(
                "/api/learning/review",
                json={
                    "card_id": "q-leech",
                    "topic_id": "topic-a",
                    "rating": "again",
                },
            )
            self.assertEqual(res.status_code, 201)
        body = res.json()
        self.assertEqual(body["card"]["lapses"], 5)
        self.assertTrue(body["leech"])

    def test_forecast_endpoint_contract(self):
        self.client.post(
            "/api/learning/review",
            json={
                "card_id": "q1",
                "topic_id": "topic-a",
                "rating": "good",
            },
        )
        res = self.client.get("/api/learning/forecast")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertIn("due_counts", body)
        self.assertIn("total_due", body)
        self.assertEqual(len(body["due_counts"]), 7)
        self.assertEqual(body["total_due"], 1)
        for day in body["due_counts"]:
            self.assertIn("date", day)
            self.assertIn("count", day)

    def test_forecast_respects_days_parameter(self):
        res = self.client.get("/api/learning/forecast?days=3")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()["due_counts"]), 3)

    def test_calibration_endpoint_contract(self):
        res = self.client.get("/api/learning/calibration")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        for field in (
            "eligible_reviews",
            "successful_reviews",
            "true_retention",
            "target_band_low",
            "target_band_high",
            "within_band",
            "good_rating_pct",
            "low_signal",
        ):
            self.assertIn(field, body)
        self.assertEqual(body["eligible_reviews"], 0)
        self.assertTrue(body["low_signal"])

    def test_review_queue_has_fsrs_fields(self):
        self.client.post(
            "/api/learning/review",
            json={
                "card_id": "q1",
                "topic_id": "topic-a",
                "rating": "again",
            },
        )
        # again on a fresh card is due in 1 minute; force it due now
        self.learning_store._conn.execute(
            "UPDATE fsrs_cards SET due_at = ? "
            "WHERE user_id = ? AND card_id = ?",
            ("2000-01-01T00:00:00+00:00", "test-user", "q1"),
        )
        self.learning_store._conn.commit()
        res = self.client.get("/api/learning/review-queue")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["total_due"], 1)
        item = body["items"][0]
        self.assertEqual(item["question_id"], "q1")
        for field in ("state", "lapses", "suspended", "leech", "due_at"):
            self.assertIn(field, item)
        self.assertEqual(item["lapses"], 1)
        self.assertEqual(item["state"], "learning")

    def test_mastery_endpoint_uses_fsrs_cards(self):
        self.client.post(
            "/api/learning/review",
            json={
                "card_id": "q1",
                "topic_id": "topic-a",
                "rating": "easy",
            },
        )
        res = self.client.get("/api/learning/mastery")
        self.assertEqual(res.status_code, 200)
        topics = res.json()["topics"]
        self.assertTrue(any(t["topic_id"] == "topic-a" for t in topics))

    def test_fsrs_endpoints_404_when_flag_disabled(self):
        class _DisabledSettings:
            enable_adaptive_learning = True
            enable_fsrs_v1 = False

        original = learning.get_settings
        learning.get_settings = lambda: _DisabledSettings()
        try:
            res = self.client.post(
                "/api/learning/review",
                json={
                    "card_id": "q1",
                    "topic_id": "topic-a",
                    "rating": "good",
                },
            )
            self.assertEqual(res.status_code, 404)
            for path in (
                "/api/learning/forecast",
                "/api/learning/calibration",
            ):
                res = self.client.get(path)
                self.assertEqual(res.status_code, 404, path)
        finally:
            learning.get_settings = original

    def test_legacy_attempts_endpoint_still_works(self):
        res = self.client.post(
            "/api/learning/attempts",
            json={
                "question_id": "q-legacy",
                "topic_id": "topic-a",
                "user_answer": "answer",
                "is_correct": True,
                "confidence": 4,
                "response_time_ms": 900,
                "mode": "quiz",
            },
        )
        self.assertEqual(res.status_code, 200)
        self.assertIn("review_bucket", res.json())


if __name__ == "__main__":
    unittest.main()
