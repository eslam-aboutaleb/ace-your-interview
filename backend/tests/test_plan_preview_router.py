"""API tests for plan preview and edited-plan commitment (#6)."""

import os
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import require_auth
from app.routers import journeys
from app.services.doc_parser import DocParser
from app.services.journey_store import JourneyStore
from app.services.learning_store import LearningStore

JOURNEYS_ENV = "STUDY_ENABLE_JOURNEYS_V1"


class PreviewRouterTestBase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "learning.db")
        self.store = JourneyStore(self.db_path)
        self.learning_store = LearningStore(self.db_path)
        self.parser = DocParser()
        journeys.init(self.store, self.learning_store, self.parser)

        self.prev_env = os.environ.get(JOURNEYS_ENV)
        os.environ[JOURNEYS_ENV] = "true"
        get_settings.cache_clear()

        self.app = FastAPI()
        self.app.include_router(journeys.router)
        self.current_user = {"user": "learner", "provider": "google"}
        self.app.dependency_overrides[require_auth] = lambda: self.current_user
        self.client = TestClient(self.app)

        topics = self.parser.list_topics()
        self.assertTrue(topics)
        self.topic_id = topics[0].id

    def tearDown(self):
        self.app.dependency_overrides.clear()
        journeys._store = None
        journeys._learning_store = None
        journeys._parser = None
        journeys._question_generator = None
        if self.prev_env is None:
            os.environ.pop(JOURNEYS_ENV, None)
        else:
            os.environ[JOURNEYS_ENV] = self.prev_env
        get_settings.cache_clear()
        self.tmpdir.cleanup()

    def _preview(self, **overrides):
        body = {
            "goal": "Pass the backend interview at a senior level",
            "source_type": "static_topic",
            "source_topic_id": self.topic_id,
        }
        body.update(overrides)
        return self.client.post("/api/journeys/plan-preview", json=body)


class FeatureGateTests(PreviewRouterTestBase):
    def test_flag_off_hides_preview(self):
        os.environ[JOURNEYS_ENV] = "false"
        get_settings.cache_clear()
        response = self._preview()
        self.assertEqual(response.status_code, 404, response.text)


class PreviewFlowTests(PreviewRouterTestBase):
    def test_preview_returns_plan_stats_and_provenance_without_persisting(self):
        response = self._preview(target_level="senior", available_hours_per_week=4, depth="survey")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["level"], "senior")
        self.assertEqual(payload["provenance"]["inputs"]["goal"], "Pass the backend interview at a senior level")
        self.assertEqual(payload["provenance"]["inputs"]["depth"], "survey")
        self.assertEqual(payload["provenance"]["inputs"]["available_hours_per_week"], 4)
        stats = payload["stats"]
        self.assertGreater(stats["objective_count"], 0)
        self.assertGreaterEqual(stats["estimated_weeks"], 1)
        modules = payload["plan"]["phases"][0]["modules"]
        self.assertLessEqual(len(modules), 4, "survey caps modules")
        for module in modules:
            self.assertLessEqual(len(module["objectives"]), 3)
            for objective in module["objectives"]:
                self.assertEqual(objective["coverage"], "uncovered")
        # nothing persisted
        self.assertEqual(self.client.get("/api/journeys").json()["journeys"], [])

    def test_preview_marks_review_obligations_with_active_cards(self):
        self.learning_store.record_attempt(
            user_id="learner",
            question_id="q1",
            topic_id=self.topic_id,
            user_answer="wrong",
            is_correct=False,
            confidence=1,
            response_time_ms=1000,
            mode="quiz",
        )
        response = self._preview()
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertGreater(payload["stats"]["review_objective_count"], 0)
        objective = payload["plan"]["phases"][0]["modules"][0]["objectives"][0]
        self.assertEqual(objective["coverage"], "review_suggested")

    def test_review_hints_are_per_user(self):
        self.learning_store.record_attempt(
            user_id="someone-else",
            question_id="q1",
            topic_id=self.topic_id,
            user_answer="wrong",
            is_correct=False,
            confidence=1,
            response_time_ms=1000,
            mode="quiz",
        )
        response = self._preview()
        self.assertEqual(response.json()["stats"]["review_objective_count"], 0)

    def test_unknown_source_404(self):
        response = self._preview(source_topic_id="topic-nope")
        self.assertEqual(response.status_code, 404, response.text)

    def test_invalid_depth_422(self):
        response = self._preview(depth="exhaustive")
        self.assertEqual(response.status_code, 422, response.text)


class EditedPlanCommitTests(PreviewRouterTestBase):
    def test_create_with_edited_plan_and_provenance(self):
        preview = self._preview().json()
        plan = preview["plan"]
        # learner edits: drop every objective but the first two
        module = plan["phases"][0]["modules"][0]
        module["objectives"] = module["objectives"][:2]
        plan["phases"][0]["modules"] = [module]
        response = self.client.post(
            "/api/journeys",
            json={
                "title": "My edited journey",
                "goal": "Pass the backend interview",
                "source_type": "static_topic",
                "source_topic_id": self.topic_id,
                "plan": plan,
                "plan_provenance": preview["provenance"]["inputs"],
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        journey = response.json()["journey"]
        modules = journey["phases"][0]["modules"]
        self.assertEqual(len(modules), 1)
        self.assertEqual(len(modules[0]["objectives"]), 2)
        import json as _json
        row = self.store._conn.execute(
            "SELECT provenance_json FROM journeys WHERE id = ?", (journey["id"],)
        ).fetchone()
        provenance = _json.loads(row["provenance_json"])
        self.assertEqual(provenance["edited_plan"], True)
        self.assertEqual(provenance["inputs"]["goal"], "Pass the backend interview at a senior level")
        self.assertEqual(provenance["inputs"]["depth"], "standard")

    def test_edited_plan_objective_ids_restamped_and_progress_preservable(self):
        preview = self._preview().json()
        plan = preview["plan"]
        response = self.client.post(
            "/api/journeys",
            json={
                "title": "Committed",
                "source_type": "static_topic",
                "source_topic_id": self.topic_id,
                "plan": plan,
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        journey = response.json()["journey"]
        objective = journey["phases"][0]["modules"][0]["objectives"][0]
        self.assertTrue(objective["id"], "store must stamp stable content-hash IDs")

    def test_invalid_edited_plan_422(self):
        response = self.client.post(
            "/api/journeys",
            json={
                "title": "Broken",
                "source_type": "static_topic",
                "source_topic_id": self.topic_id,
                "plan": {"phases": []},
            },
        )
        self.assertEqual(response.status_code, 422, response.text)

    def test_create_without_plan_unchanged(self):
        response = self.client.post(
            "/api/journeys",
            json={
                "title": "Baseline",
                "source_type": "static_topic",
                "source_topic_id": self.topic_id,
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        journey = response.json()["journey"]
        import json as _json
        row = self.store._conn.execute(
            "SELECT provenance_json FROM journeys WHERE id = ?", (journey["id"],)
        ).fetchone()
        self.assertNotIn("edited_plan", _json.loads(row["provenance_json"]))
        self.assertGreater(journey["objective_count"], 0)


if __name__ == "__main__":
    unittest.main()
