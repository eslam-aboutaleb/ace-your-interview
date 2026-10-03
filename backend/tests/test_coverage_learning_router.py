"""Coverage for ``app.routers.learning`` — feature gates, FSRS branch, 503 paths.

Real ``LearningStore``/``LearningPlannerStore`` instances on a temp database are
used so response models still validate; spies record which store method each
endpoint selected.
"""

import os
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import require_auth
from app.routers import learning
from app.services.learning_planner import LearningPlannerStore
from app.services.learning_store import LearningStore

ADAPTIVE_ENV = "STUDY_ENABLE_ADAPTIVE_LEARNING"
FSRS_ENV = "STUDY_ENABLE_FSRS_V1"


class SpyLearningStore(LearningStore):
    """Real store plus a log of which query each request resolved to."""

    def __init__(self, db_path: str):
        super().__init__(db_path)
        self.calls: list[str] = []

    def get_review_queue(self, **kwargs):
        self.calls.append("get_review_queue")
        return super().get_review_queue(**kwargs)

    def get_fsrs_review_queue(self, **kwargs):
        self.calls.append("get_fsrs_review_queue")
        return super().get_fsrs_review_queue(**kwargs)

    def get_topic_mastery(self, **kwargs):
        self.calls.append("get_topic_mastery")
        return super().get_topic_mastery(**kwargs)

    def get_fsrs_topic_mastery(self, **kwargs):
        self.calls.append("get_fsrs_topic_mastery")
        return super().get_fsrs_topic_mastery(**kwargs)


class _FakeBackfillStore:
    def __init__(self, result=None, raises=None):
        self.result = result or {}
        self.raises = raises
        self.backfill_calls = 0

    def backfill_fsrs_from_progress(self):
        self.backfill_calls += 1
        if self.raises is not None:
            raise self.raises
        return self.result


class LearningRouterCoverageTestBase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "learning.db")
        self.store = SpyLearningStore(self.db_path)
        self.planner = LearningPlannerStore(self.db_path)
        learning.init(self.store, self.planner)

        self._prev_globals = (learning._store, learning._planner)

        self.prev_env = {
            key: os.environ.get(key) for key in (ADAPTIVE_ENV, FSRS_ENV)
        }
        os.environ[ADAPTIVE_ENV] = "true"
        os.environ[FSRS_ENV] = "true"
        get_settings.cache_clear()

        self.app = FastAPI()
        self.app.include_router(learning.router)
        self.app.dependency_overrides[require_auth] = lambda: {
            "user": "learner",
            "provider": "google",
        }
        self.client = TestClient(self.app)

    def tearDown(self):
        self.app.dependency_overrides.clear()
        learning._store, learning._planner = self._prev_globals
        self.tmpdir.cleanup()
        for key, value in self.prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()

    def _set_flags(self, *, adaptive: bool | None = None, fsrs: bool | None = None):
        if adaptive is not None:
            os.environ[ADAPTIVE_ENV] = "true" if adaptive else "false"
        if fsrs is not None:
            os.environ[FSRS_ENV] = "true" if fsrs else "false"
        get_settings.cache_clear()


class AdaptiveLearningGateTests(LearningRouterCoverageTestBase):
    #: Every endpoint first calls ``_ensure_enabled``.
    GATED_PATHS = [
        ("post", "/api/learning/attempts", {"question_id": "q1", "topic_id": "t1",
                                            "user_answer": "a", "is_correct": True,
                                            "confidence": 3, "response_time_ms": 500,
                                            "mode": "study"}),
        ("get", "/api/learning/review-queue", None),
        ("post", "/api/learning/review", {"card_id": "c1", "topic_id": "t1",
                                          "rating": "good", "source_type": "question"}),
        ("get", "/api/learning/forecast", None),
        ("get", "/api/learning/calibration", None),
        ("get", "/api/learning/weak-areas", None),
        ("get", "/api/learning/mastery", None),
        ("get", "/api/learning/profile", None),
        ("put", "/api/learning/profile", {"target_role": "Backend"}),
        ("post", "/api/learning/profile/diagnostic", None),
        ("get", "/api/learning/recommendations", None),
        ("get", "/api/learning/study-plan", None),
    ]

    def test_disabling_adaptive_learning_hides_every_endpoint_with_404(self):
        self._set_flags(adaptive=False)
        for method, path, body in self.GATED_PATHS:
            with self.subTest(path=path):
                kwargs = {"json": body} if body is not None else {}
                response = getattr(self.client, method)(path, **kwargs)
                self.assertEqual(response.status_code, 404)
                self.assertEqual(response.json()["detail"], "Adaptive learning disabled")

    def test_store_backed_endpoints_report_503_without_a_store(self):
        learning._store = None
        paths = [
            ("post", "/api/learning/attempts", {"question_id": "q1", "topic_id": "t1",
                                                "user_answer": "a", "is_correct": True,
                                                "confidence": 3, "response_time_ms": 500,
                                                "mode": "study"}),
            ("get", "/api/learning/review-queue", None),
            ("post", "/api/learning/review", {"card_id": "c1", "topic_id": "t1",
                                              "rating": "good", "source_type": "question"}),
            ("get", "/api/learning/forecast", None),
            ("get", "/api/learning/calibration", None),
            ("get", "/api/learning/weak-areas", None),
            ("get", "/api/learning/mastery", None),
        ]
        for method, path, body in paths:
            with self.subTest(path=path):
                kwargs = {"json": body} if body is not None else {}
                response = getattr(self.client, method)(path, **kwargs)
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json()["detail"], "Learning store not initialised")

    def test_planner_backed_endpoints_report_503_without_a_planner(self):
        learning._planner = None
        for method, path, body in (
            ("get", "/api/learning/profile", None),
            ("put", "/api/learning/profile", {"target_role": "Backend"}),
            ("post", "/api/learning/profile/diagnostic", None),
            ("get", "/api/learning/recommendations", None),
            ("get", "/api/learning/study-plan", None),
        ):
            with self.subTest(path=path):
                kwargs = {"json": body} if body is not None else {}
                response = getattr(self.client, method)(path, **kwargs)
                self.assertEqual(response.status_code, 503)
                self.assertEqual(
                    response.json()["detail"], "Learning planner not initialised"
                )


class FsrsFeatureGateTests(LearningRouterCoverageTestBase):
    def test_fsrs_only_endpoints_are_hidden_when_the_flag_is_off(self):
        self._set_flags(fsrs=False)
        for method, path, body in (
            ("post", "/api/learning/review", {"card_id": "c1", "topic_id": "t1",
                                              "rating": "good", "source_type": "question"}),
            ("get", "/api/learning/forecast", None),
            ("get", "/api/learning/calibration", None),
        ):
            with self.subTest(path=path):
                kwargs = {"json": body} if body is not None else {}
                response = getattr(self.client, method)(path, **kwargs)
                self.assertEqual(response.status_code, 404)
                self.assertEqual(response.json()["detail"], "FSRS scheduling disabled")

    def test_review_queue_prefers_fsrs_when_enabled(self):
        self.store.calls.clear()
        response = self.client.get("/api/learning/review-queue")
        self.assertEqual(response.status_code, 200)
        self.assertIn("items", response.json())
        self.assertEqual(self.store.calls, ["get_fsrs_review_queue"])

    def test_review_queue_falls_back_to_the_legacy_queue(self):
        self._set_flags(fsrs=False)
        self.store.calls.clear()
        response = self.client.get("/api/learning/review-queue")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.store.calls, ["get_review_queue"])

    def test_topic_mastery_prefers_fsrs_when_enabled(self):
        self.store.calls.clear()
        response = self.client.get("/api/learning/mastery")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.store.calls, ["get_fsrs_topic_mastery"])

    def test_topic_mastery_falls_back_to_the_legacy_query(self):
        self._set_flags(fsrs=False)
        self.store.calls.clear()
        response = self.client.get("/api/learning/mastery")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.store.calls, ["get_topic_mastery"])


class WeakAreasAndBoundsTests(LearningRouterCoverageTestBase):
    def test_weak_areas_returns_the_store_payload(self):
        response = self.client.get("/api/learning/weak-areas?limit=5")
        self.assertEqual(response.status_code, 200)
        self.assertIn("weak_areas", response.json())

    def test_query_bounds_are_enforced_by_the_schema(self):
        for path, query in (
            ("/api/learning/weak-areas", "limit=0"),
            ("/api/learning/weak-areas", "limit=51"),
            ("/api/learning/review-queue", "limit=201"),
            ("/api/learning/mastery", "limit=1001"),
            ("/api/learning/forecast", "days=32"),
            ("/api/learning/recommendations", "limit=51"),
            ("/api/learning/study-plan", "daily_items=11"),
        ):
            with self.subTest(path=path, query=query):
                self.assertEqual(self.client.get(f"{path}?{query}").status_code, 422)


class StartupBackfillTests(unittest.TestCase):
    def setUp(self):
        self.prev_env = os.environ.get(FSRS_ENV)
        self._prev_globals = (learning._store, learning._planner)
        get_settings.cache_clear()

    def tearDown(self):
        learning._store, learning._planner = self._prev_globals
        if self.prev_env is None:
            os.environ.pop(FSRS_ENV, None)
        else:
            os.environ[FSRS_ENV] = self.prev_env
        get_settings.cache_clear()

    def test_backfill_runs_at_init_and_logs_when_rows_were_migrated(self):
        os.environ[FSRS_ENV] = "true"
        get_settings.cache_clear()
        store = _FakeBackfillStore(
            result={"inserted": 4, "question_progress_rows": 9}
        )
        with self.assertLogs("app.routers.learning", level="INFO") as logs:
            learning.init(store, None)  # type: ignore[arg-type]

        self.assertEqual(store.backfill_calls, 1)
        self.assertTrue(any("FSRS backfill migrated 4 of 9" in line for line in logs.output))

    def test_no_log_line_when_nothing_was_migrated(self):
        os.environ[FSRS_ENV] = "true"
        get_settings.cache_clear()
        store = _FakeBackfillStore(result={"inserted": 0, "question_progress_rows": 0})
        learning.init(store, None)  # type: ignore[arg-type]
        self.assertEqual(store.backfill_calls, 1)

    def test_backfill_is_skipped_entirely_when_fsrs_is_disabled(self):
        os.environ[FSRS_ENV] = "false"
        get_settings.cache_clear()
        store = _FakeBackfillStore()
        learning.init(store, None)  # type: ignore[arg-type]
        self.assertEqual(store.backfill_calls, 0)

    def test_a_failing_backfill_never_blocks_startup(self):
        os.environ[FSRS_ENV] = "true"
        get_settings.cache_clear()
        store = _FakeBackfillStore(raises=RuntimeError("sqlite locked"))
        with self.assertLogs("app.routers.learning", level="ERROR"):
            learning.init(store, None)  # type: ignore[arg-type]

        self.assertEqual(learning._store, store)
        self.assertEqual(store.backfill_calls, 1)


if __name__ == "__main__":
    unittest.main()