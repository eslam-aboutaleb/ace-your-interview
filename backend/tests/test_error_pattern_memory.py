"""Error-pattern memory: store CRUD, the attempt hook, and the read endpoint."""

import json
import os
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import require_auth
from app.routers import learning
from app.services.learning_store import (
    ERROR_PATTERN_CLASSIFIED_EVENT,
    ERROR_PATTERN_FEATURE_KEY,
    LearningStore,
)


class RecordingLLM:
    """Stub that counts calls and optionally fails."""

    def __init__(self, payload=None, *, fails=False):
        self.payload = payload or {
            "error_type": "misconception",
            "explanation": "Confuses idempotency with caching.",
        }
        self.fails = fails
        self.calls = 0

    async def completion(self, prompt, llm_config=None, user_identity=None, **kwargs):
        self.calls += 1
        if self.fails:
            raise RuntimeError("provider exploded")
        return {
            "success": True,
            "analysis": json.dumps(self.payload),
            "metadata": {},
            "error": "",
            "error_code": "",
        }


_WRONG_ANSWER = (
    "Idempotency only matters for DELETE, and the client keeps a copy so the "
    "server never has to re-check the request."
)

_ATTEMPT = {
    "question_id": "01-backend-fundamentals-and-http:abc123",
    "topic_id": "01-backend-fundamentals-and-http",
    "user_answer": _WRONG_ANSWER,
    "is_correct": False,
    "confidence": 4,
    "response_time_ms": 900,
    "mode": "quiz",
    "question_text": "Explain what an idempotent HTTP method guarantees.",
    "correct_answer": (
        "An idempotent method can be repeated several times with the same "
        "effect as calling it once, so retries are safe for PUT and DELETE."
    ),
}


class ErrorPatternStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = LearningStore(os.path.join(self._tmp.name, "learning.db"))

    def tearDown(self):
        self._tmp.cleanup()

    def test_upsert_increments_count_and_refreshes_last_seen(self):
        first = self.store.record_error_pattern(
            user_id="alice",
            topic_id="topic-a",
            objective="explain idempotency",
            error_type="misconception",
            explanation="Confuses idempotency with caching.",
            evidence="the client keeps a copy",
        )
        self.assertEqual(first["count"], 1)
        self.assertEqual(first["first_seen"], first["last_seen"])

        second = self.store.record_error_pattern(
            user_id="alice",
            topic_id="topic-a",
            objective="explain idempotency",
            error_type="misconception",
            explanation="Still confuses idempotency with caching.",
            evidence="the server never re-checks",
        )
        self.assertEqual(second["count"], 2)
        self.assertEqual(second["first_seen"], first["first_seen"])
        self.assertEqual(second["explanation"], "Still confuses idempotency with caching.")
        self.assertEqual(second["evidence"], "the server never re-checks")
        rows = self.store.list_error_patterns(user_id="alice")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["count"], 2)

    def test_distinct_error_types_are_separate_rows(self):
        for error_type in ("misconception", "missing_depth"):
            self.store.record_error_pattern(
                user_id="alice",
                topic_id="topic-a",
                objective="explain idempotency",
                error_type=error_type,
                explanation="gap",
                evidence=error_type,
            )
        self.assertEqual(len(self.store.list_error_patterns(user_id="alice")), 2)

    def test_list_ranks_by_count_then_recency(self):
        for index, topic_id in enumerate(("topic-c", "topic-a", "topic-b")):
            for _ in range(index + 1):
                self.store.record_error_pattern(
                    user_id="alice",
                    topic_id=topic_id,
                    objective="objective",
                    error_type="misconception",
                    explanation="gap",
                    evidence="evidence",
                )
        ranked = self.store.list_error_patterns(user_id="alice")
        self.assertEqual(
            [item["topic_id"] for item in ranked],
            ["topic-b", "topic-a", "topic-c"],
        )
        self.assertEqual([item["count"] for item in ranked], [3, 2, 1])

    def test_list_filters_by_topic_min_count_and_limit(self):
        for topic_id in ("topic-a", "topic-b"):
            for _ in range(3):
                self.store.record_error_pattern(
                    user_id="alice",
                    topic_id=topic_id,
                    objective=f"objective-{topic_id}",
                    error_type="misconception",
                    explanation="gap",
                    evidence="evidence",
                )
        self.assertEqual(
            [
                (item["topic_id"], item["count"])
                for item in self.store.list_error_patterns(
                    user_id="alice", topic_id="topic-a"
                )
            ],
            [("topic-a", 3)],
        )
        self.assertEqual(
            len(self.store.list_error_patterns(user_id="alice", min_count=4)),
            0,
        )
        self.assertEqual(len(self.store.list_error_patterns(user_id="alice", limit=1)), 1)
        self.assertEqual(len(self.store.list_error_patterns(user_id="bob")), 0)

    def test_classification_cap_counts_only_the_recent_window(self):
        for _ in range(3):
            self.store.record_feature_event(
                user_id="alice",
                feature_key=ERROR_PATTERN_FEATURE_KEY,
                event_name=ERROR_PATTERN_CLASSIFIED_EVENT,
                metadata={"question_id": "q"},
            )
        self.store.record_feature_event(
            user_id="alice",
            feature_key="some_other_feature",
            event_name="other_event",
            metadata={},
        )
        self.assertEqual(
            self.store.count_error_pattern_classifications(user_id="alice"), 3
        )
        self.assertEqual(
            self.store.count_error_pattern_classifications(user_id="alice", within_hours=0),
            3,
        )
        self.assertEqual(
            self.store.count_error_pattern_classifications(user_id="bob"), 0
        )


class _LearningRouterHarness(unittest.TestCase):
    """Wires a fresh store + optional LLM stub into the learning router."""

    def build(self, *, llm=None, flag=None, cap=None):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = LearningStore(os.path.join(self._tmp.name, "learning.db"))
        learning.init(self.store, None, llm_client=llm)

        if flag is not None or cap is not None:
            settings = get_settings()
            self._flag_backup = settings.enable_error_patterns
            self._cap_backup = settings.error_pattern_hourly_cap
            if flag is not None:
                object.__setattr__(settings, "enable_error_patterns", flag)
            if cap is not None:
                object.__setattr__(settings, "error_pattern_hourly_cap", cap)

        app = FastAPI()
        app.include_router(learning.router)
        app.dependency_overrides[require_auth] = lambda: {
            "user": "test-user",
            "provider": "local",
        }
        self.client = TestClient(app)

    def tearDown(self):
        if hasattr(self, "_flag_backup"):
            settings = get_settings()
            object.__setattr__(settings, "enable_error_patterns", self._flag_backup)
            object.__setattr__(settings, "error_pattern_hourly_cap", self._cap_backup)
        self._tmp.cleanup()


class ErrorPatternRouterTests(_LearningRouterHarness):
    def _anonymous_client(self):
        app = FastAPI()
        app.include_router(learning.router)
        return TestClient(app)

    def test_wrong_answer_records_a_ranked_error_pattern(self):
        self.build(llm=RecordingLLM())
        response = self.client.post("/api/learning/attempts", json=_ATTEMPT)
        self.assertEqual(response.status_code, 200)

        listed = self.client.get("/api/learning/error-patterns")
        self.assertEqual(listed.status_code, 200)
        payload = listed.json()
        self.assertEqual(payload["total"], 1)
        item = payload["items"][0]
        self.assertEqual(item["error_type"], "misconception")
        self.assertEqual(item["topic_id"], _ATTEMPT["topic_id"])
        self.assertEqual(item["count"], 1)
        self.assertTrue(item["objective"])
        self.assertIn("DELETE", item["evidence"])

    def test_endpoint_requires_auth(self):
        self.build(llm=RecordingLLM())
        response = self._anonymous_client().get("/api/learning/error-patterns")
        self.assertEqual(response.status_code, 401)

    def test_correct_answer_records_no_error_pattern(self):
        self.build(llm=RecordingLLM())
        attempt = dict(_ATTEMPT, is_correct=True)
        self.client.post("/api/learning/attempts", json=attempt)
        listed = self.client.get("/api/learning/error-patterns").json()
        self.assertEqual(listed["total"], 0)

    def test_short_answer_records_no_error_pattern(self):
        self.build(llm=RecordingLLM())
        self.client.post(
            "/api/learning/attempts", json=dict(_ATTEMPT, user_answer="B")
        )
        self.assertEqual(self.client.get("/api/learning/error-patterns").json()["total"], 0)

    def test_hourly_cap_stops_classification(self):
        self.build(llm=RecordingLLM(), cap=1)
        self.client.post("/api/learning/attempts", json=_ATTEMPT)
        self.store.record_feature_event(
            user_id="test-user",
            feature_key=ERROR_PATTERN_FEATURE_KEY,
            event_name=ERROR_PATTERN_CLASSIFIED_EVENT,
            metadata={"question_id": "seed"},
        )
        self.client.post("/api/learning/attempts", json=_ATTEMPT)
        self.client.post("/api/learning/attempts", json=_ATTEMPT)
        self.assertEqual(
            self.client.get("/api/learning/error-patterns").json()["total"], 1
        )

    def test_flag_off_skips_classification(self):
        self.build(llm=RecordingLLM(), flag=False)
        self.client.post("/api/learning/attempts", json=_ATTEMPT)
        self.assertEqual(self.client.get("/api/learning/error-patterns").json()["total"], 0)

    def test_classification_failure_leaves_the_attempt_recorded(self):
        self.build(llm=RecordingLLM(fails=True))
        response = self.client.post("/api/learning/attempts", json=_ATTEMPT)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("attempt_id", body)

        # The deterministic fallback still stores a pattern, so a wrong answer
        # is never silently lost.
        listed = self.client.get("/api/learning/error-patterns").json()
        self.assertEqual(listed["total"], 1)
        self.assertIn(listed["items"][0]["error_type"], {
            "partial_answer",
            "unclassified",
            "misconception",
        })

    def test_no_llm_client_uses_the_deterministic_fallback(self):
        self.build(llm=None)
        response = self.client.post(
            "/api/learning/attempts",
            json=dict(
                _ATTEMPT,
                user_answer="I don't know the answer to this idempotency question at all.",
            ),
        )
        self.assertEqual(response.status_code, 200)
        listed = self.client.get("/api/learning/error-patterns").json()
        self.assertEqual(listed["total"], 1)
        self.assertEqual(listed["items"][0]["error_type"], "no_answer")

    def test_error_patterns_are_scoped_to_the_user(self):
        self.build(llm=RecordingLLM())
        self.client.post("/api/learning/attempts", json=_ATTEMPT)
        other = FastAPI()
        other.include_router(learning.router)
        other.dependency_overrides[require_auth] = lambda: {
            "user": "another-user",
            "provider": "local",
        }
        other_client = TestClient(other)
        self.assertEqual(other_client.get("/api/learning/error-patterns").json()["total"], 0)

    def test_topic_filter_is_applied(self):
        self.build(llm=RecordingLLM())
        self.client.post("/api/learning/attempts", json=_ATTEMPT)
        self.client.post(
            "/api/learning/attempts",
            json=dict(_ATTEMPT, topic_id="02-frontend-fundamentals"),
        )
        filtered = self.client.get(
            "/api/learning/error-patterns",
            params={"topic_id": "02-frontend-fundamentals"},
        ).json()
        self.assertEqual(filtered["total"], 1)
        self.assertEqual(
            filtered["items"][0]["topic_id"], "02-frontend-fundamentals"
        )
        self.assertEqual(self.client.get("/api/learning/error-patterns").json()["total"], 2)

class _Sha:  # pragma: no cover - readability helper kept out of the suite
    pass


if __name__ == "__main__":
    unittest.main()
