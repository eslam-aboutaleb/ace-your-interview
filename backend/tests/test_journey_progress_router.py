"""API tests for the journey progress endpoints (issue #5).

Covers the feature gate, auth, the completion/attempt/progress HTTP flows,
validation errors, and mutation_id replay over the wire.
"""

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


class ProgressRouterTestBase(unittest.TestCase):
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
        self.assertTrue(topics, "expected the bundled curriculum to list topics")
        self.topic_id = topics[0].id
        self.journey = self._create_journey()
        self.slots = self._reserve_slots()

    def tearDown(self):
        self.app.dependency_overrides.clear()
        journeys._store = None
        journeys._learning_store = None
        journeys._parser = None
        if self.prev_env is None:
            os.environ.pop(JOURNEYS_ENV, None)
        else:
            os.environ[JOURNEYS_ENV] = self.prev_env
        get_settings.cache_clear()
        self.tmpdir.cleanup()

    def _create_journey(self):
        response = self.client.post(
            "/api/journeys",
            json={
                "title": "Backend journey",
                "goal": "ace the interview",
                "source_type": "static_topic",
                "source_topic_id": self.topic_id,
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["journey"]

    def _reserve_slots(self):
        response = self.client.post(
            f"/api/journeys/{self.journey['id']}/batches",
            json={"idempotency_key": "batch-1", "slot_count": 3},
        )
        self.assertIn(response.status_code, (200, 201), response.text)
        batch = response.json()["batch"]
        # no HTTP slots endpoint exists yet (the reader slice lands in a
        # later issue); read the reserved slots straight from the store
        slots = self.store.get_batch_slots(
            user_id="learner", journey_id=self.journey["id"], batch_id=batch["id"]
        )
        self.assertEqual(len(slots), 3)
        return slots

    def _complete_url(self, slot_id):
        return f"/api/journeys/{self.journey['id']}/questions/{slot_id}/completion"

    def _attempts_url(self, slot_id):
        return f"/api/journeys/{self.journey['id']}/questions/{slot_id}/attempts"


class FeatureGateTests(ProgressRouterTestBase):
    def test_flag_off_hides_progress_endpoints(self):
        os.environ[JOURNEYS_ENV] = "false"
        get_settings.cache_clear()
        slot = self.slots[0]["id"]
        checks = [
            ("get", f"/api/journeys/{self.journey['id']}/progress", None),
            ("put", self._complete_url(slot), {"completed": True, "mutation_id": "m1"}),
            ("post", self._attempts_url(slot), {"outcome": "correct", "mutation_id": "a1"}),
        ]
        for method, url, body in checks:
            response = getattr(self.client, method)(url, json=body) if body else getattr(self.client, method)(url)
            self.assertEqual(response.status_code, 404, f"{method} {url}: {response.text}")


class CompletionFlowTests(ProgressRouterTestBase):
    def test_complete_then_progress_summary(self):
        slot = self.slots[0]["id"]
        response = self.client.put(
            self._complete_url(slot), json={"completed": True, "mutation_id": "m1"}
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload["state"]["completed"])
        self.assertEqual(payload["summary"]["completed"], 1)
        self.assertEqual(payload["summary"]["total_questions"], 3)
        self.assertAlmostEqual(payload["summary"]["percent_complete"], 100.0 / 3.0, places=1)

        response = self.client.get(f"/api/journeys/{self.journey['id']}/progress")
        self.assertEqual(response.status_code, 200, response.text)
        progress = response.json()
        self.assertEqual(progress["completed"], 1)
        self.assertEqual(len(progress["states"]), 3)

    def test_completion_replay_returns_same_revision(self):
        slot = self.slots[0]["id"]
        first = self.client.put(self._complete_url(slot), json={"completed": True, "mutation_id": "m1"})
        replay = self.client.put(self._complete_url(slot), json={"completed": True, "mutation_id": "m1"})
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(replay.status_code, 200, replay.text)
        self.assertEqual(replay.json()["state"]["revision"], first.json()["state"]["revision"])

    def test_uncomplete_roundtrip(self):
        slot = self.slots[0]["id"]
        self.client.put(self._complete_url(slot), json={"completed": True, "mutation_id": "m1"})
        response = self.client.put(self._complete_url(slot), json={"completed": False, "mutation_id": "m2"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()["state"]["completed"])
        self.assertEqual(response.json()["summary"]["completed"], 0)

    def test_unknown_slot_and_unknown_journey(self):
        response = self.client.put(
            self._complete_url("slot_nope"), json={"completed": True, "mutation_id": "m1"}
        )
        self.assertEqual(response.status_code, 422, response.text)
        response = self.client.put(
            "/api/journeys/jrn_nope/questions/slot_x/completion",
            json={"completed": True, "mutation_id": "m1"},
        )
        self.assertEqual(response.status_code, 404, response.text)

    def test_missing_mutation_id_is_422(self):
        response = self.client.put(self._complete_url(self.slots[0]["id"]), json={"completed": True})
        self.assertEqual(response.status_code, 422, response.text)


class AttemptFlowTests(ProgressRouterTestBase):
    def test_attempt_then_mastery_over_http(self):
        slot = self.slots[0]["id"]
        for mid in ("a1", "a2"):
            response = self.client.post(
                self._attempts_url(slot),
                json={"outcome": "correct", "confidence": 3, "response_time_ms": 1200, "mutation_id": mid},
            )
            self.assertEqual(response.status_code, 200, response.text)
        response = self.client.put(self._complete_url(slot), json={"completed": True, "mutation_id": "m1"})
        self.assertTrue(response.json()["state"]["mastered"])
        response = self.client.post(self._attempts_url(slot), json={"outcome": "incorrect", "mutation_id": "a3"})
        self.assertFalse(response.json()["state"]["mastered"])

    def test_attempt_replay_is_idempotent(self):
        slot = self.slots[0]["id"]
        first = self.client.post(self._attempts_url(slot), json={"outcome": "correct", "mutation_id": "a1"})
        replay = self.client.post(self._attempts_url(slot), json={"outcome": "correct", "mutation_id": "a1"})
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(replay.json()["state"]["revision"], first.json()["state"]["revision"])

    def test_invalid_outcome_is_422(self):
        response = self.client.post(
            self._attempts_url(self.slots[0]["id"]), json={"outcome": "maybe", "mutation_id": "a1"}
        )
        self.assertEqual(response.status_code, 422, response.text)

    def test_per_user_isolation(self):
        slot = self.slots[0]["id"]
        self.client.put(self._complete_url(slot), json={"completed": True, "mutation_id": "m1"})
        self.current_user = {"user": "someone-else", "provider": "google"}
        response = self.client.get(f"/api/journeys/{self.journey['id']}/progress")
        self.assertEqual(response.status_code, 404, response.text)


if __name__ == "__main__":
    unittest.main()
