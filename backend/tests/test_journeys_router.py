"""API tests for app/routers/journeys.py (issue #4).

Covers the feature gate, the create/list/detail flows against the real
curriculum, custom-topic sources, batch idempotency, reader state, and the
checkpoint conflict/idempotency contract over HTTP.
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


class JourneysRouterTestBase(unittest.TestCase):
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

    def _create_journey(self, **overrides):
        body = {
            "title": "Backend journey",
            "goal": "ace the interview",
            "source_type": "static_topic",
            "source_topic_id": self.topic_id,
        }
        body.update(overrides)
        response = self.client.post("/api/journeys", json=body)
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["journey"]


class FeatureGateTests(JourneysRouterTestBase):
    def test_flag_off_hides_every_endpoint(self):
        os.environ[JOURNEYS_ENV] = "false"
        get_settings.cache_clear()
        jid = "jrn_whatever"
        checks = [
            ("get", "/api/journeys", None),
            (
                "post",
                "/api/journeys",
                {"title": "t", "source_type": "static_topic", "source_topic_id": "x"},
            ),
            ("get", f"/api/journeys/{jid}", None),
            ("post", f"/api/journeys/{jid}/refresh", None),
            ("post", f"/api/journeys/{jid}/batches", {"idempotency_key": "k"}),
            ("get", f"/api/journeys/{jid}/batches", None),
            ("get", f"/api/journeys/{jid}/reader-state", None),
            (
                "patch",
                f"/api/journeys/{jid}/checkpoints",
                {"expected_revision": 0, "mutation_id": "m"},
            ),
        ]
        for method, path, body in checks:
            with self.subTest(path=path):
                response = (
                    getattr(self.client, method)(path, json=body)
                    if body is not None
                    else getattr(self.client, method)(path)
                )
                self.assertEqual(response.status_code, 404, response.text)


class CreateAndDetailTests(JourneysRouterTestBase):
    def test_create_list_get_roundtrip(self):
        journey = self._create_journey()
        self.assertTrue(journey["id"].startswith("jrn_"))
        self.assertEqual(journey["source_type"], "static_topic")
        self.assertEqual(journey["source_topic_id"], self.topic_id)
        self.assertEqual(journey["plan_version"], 1)
        self.assertGreater(journey["objective_count"], 0)
        self.assertTrue(journey["phases"])

        listed = self.client.get("/api/journeys").json()["journeys"]
        self.assertEqual([j["id"] for j in listed], [journey["id"]])
        self.assertEqual(
            listed[0]["api_version"] if "api_version" in listed[0] else "v1", "v1"
        )

        detail = self.client.get(f"/api/journeys/{journey['id']}").json()
        self.assertEqual(detail["id"], journey["id"])

    def test_create_with_unknown_topic_404s(self):
        response = self.client.post(
            "/api/journeys",
            json={
                "title": "x",
                "source_type": "static_topic",
                "source_topic_id": "no-such-topic",
            },
        )
        self.assertEqual(response.status_code, 404, response.text)

    def test_create_from_custom_topic(self):
        self.learning_store.upsert_custom_topic(
            user_id="learner",
            topic_id="custom-1",
            source_topic="upload",
            title="My custom topic",
            description="",
            track="backend",
            levels=["mid"],
            sections=[
                {"heading": "Section A", "content": "- alpha: one\n- beta: two"},
                {"heading": "Section B", "content": "no bullets here"},
            ],
            raw_content="",
        )
        journey = self._create_journey(
            title="Custom journey",
            source_type="custom_topic",
            source_topic_id="custom-1",
        )
        self.assertEqual(journey["source_type"], "custom_topic")
        titles = [
            obj["title"]
            for phase in journey["phases"]
            for module in phase["modules"]
            for obj in module["objectives"]
        ]
        self.assertEqual(titles, ["alpha", "beta", "Explain Section B"])

    def test_custom_topic_of_another_user_404s(self):
        self.learning_store.upsert_custom_topic(
            user_id="someone-else",
            topic_id="custom-9",
            source_topic="upload",
            title="Private",
            description="",
            track="backend",
            levels=[],
            sections=[],
            raw_content="",
        )
        response = self.client.post(
            "/api/journeys",
            json={
                "title": "x",
                "source_type": "custom_topic",
                "source_topic_id": "custom-9",
            },
        )
        self.assertEqual(response.status_code, 404, response.text)

    def test_journeys_are_not_shared_between_users(self):
        journey = self._create_journey()
        self.current_user = {"user": "intruder", "provider": "google"}
        self.assertEqual(
            self.client.get(f"/api/journeys/{journey['id']}").status_code, 404
        )
        self.assertEqual(self.client.get("/api/journeys").json()["journeys"], [])
        self.assertEqual(
            self.client.post(
                f"/api/journeys/{journey['id']}/batches", json={"idempotency_key": "k"}
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client.get(f"/api/journeys/{journey['id']}/reader-state").status_code,
            404,
        )
        self.assertEqual(
            self.client.patch(
                f"/api/journeys/{journey['id']}/checkpoints",
                json={"expected_revision": 0, "mutation_id": "m"},
            ).status_code,
            404,
        )


class BatchFlowTests(JourneysRouterTestBase):
    def test_batch_creation_and_idempotent_replay(self):
        journey = self._create_journey()
        jid = journey["id"]

        response = self.client.post(
            f"/api/journeys/{jid}/batches",
            json={"idempotency_key": "k1", "slot_count": 5},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload["created"])
        self.assertEqual(payload["batch"]["batch_number"], 1)
        self.assertEqual(payload["batch"]["plan_version"], 1)

        replay = self.client.post(
            f"/api/journeys/{jid}/batches",
            json={"idempotency_key": "k1", "slot_count": 5},
        )
        self.assertEqual(replay.status_code, 200, replay.text)
        self.assertFalse(replay.json()["created"])
        self.assertEqual(replay.json()["batch"]["id"], payload["batch"]["id"])

        second = self.client.post(
            f"/api/journeys/{jid}/batches",
            json={"idempotency_key": "k2", "slot_count": 5},
        )
        self.assertEqual(second.json()["batch"]["batch_number"], 2)

        batches = self.client.get(f"/api/journeys/{jid}/batches").json()["batches"]
        self.assertEqual([b["batch_number"] for b in batches], [1, 2])

    def test_batch_exhaustion_conflicts(self):
        self.learning_store.upsert_custom_topic(
            user_id="learner",
            topic_id="tiny",
            source_topic="upload",
            title="Tiny",
            description="",
            track="backend",
            levels=[],
            sections=[{"heading": "One", "content": "no bullets"}],
            raw_content="",
        )
        journey = self._create_journey(
            title="Tiny journey", source_type="custom_topic", source_topic_id="tiny"
        )
        jid = journey["id"]
        first = self.client.post(
            f"/api/journeys/{jid}/batches",
            json={"idempotency_key": "k1", "slot_count": 10},
        )
        self.assertEqual(first.status_code, 200, first.text)
        second = self.client.post(
            f"/api/journeys/{jid}/batches", json={"idempotency_key": "k2"}
        )
        self.assertEqual(second.status_code, 409, second.text)

    def test_batch_body_validation(self):
        journey = self._create_journey()
        jid = journey["id"]
        response = self.client.post(
            f"/api/journeys/{jid}/batches", json={"idempotency_key": ""}
        )
        self.assertEqual(response.status_code, 422)
        response = self.client.post(
            f"/api/journeys/{jid}/batches",
            json={"idempotency_key": "k", "slot_count": 0},
        )
        self.assertEqual(response.status_code, 422)


class ReaderStateAndCheckpointTests(JourneysRouterTestBase):
    def test_reader_state_and_checkpoint_flow(self):
        journey = self._create_journey()
        jid = journey["id"]

        state = self.client.get(f"/api/journeys/{jid}/reader-state").json()
        self.assertIsNone(state["latest_batch"])
        self.assertEqual(state["checkpoint"]["revision"], 0)

        batch = self.client.post(
            f"/api/journeys/{jid}/batches",
            json={"idempotency_key": "k1", "slot_count": 3},
        ).json()["batch"]
        state = self.client.get(f"/api/journeys/{jid}/reader-state").json()
        self.assertEqual(state["latest_batch"]["id"], batch["id"])
        self.assertEqual(len(state["questions"]), 3)
        first_slot = state["questions"][0]
        self.assertTrue(first_slot["objective_title"])
        positions = [q["position"] for q in state["questions"]]
        self.assertEqual(positions, [0, 1, 2])

        saved = self.client.patch(
            f"/api/journeys/{jid}/checkpoints",
            json={
                "expected_revision": 0,
                "mutation_id": "m1",
                "last_question_id": first_slot["id"],
                "scroll_offset": 0.5,
            },
        )
        self.assertEqual(saved.status_code, 200, saved.text)
        checkpoint = saved.json()["checkpoint"]
        self.assertEqual(checkpoint["revision"], 1)
        self.assertEqual(checkpoint["last_question_id"], first_slot["id"])

        # Refresh keeps the checkpoint: resume works across requests.
        state = self.client.get(f"/api/journeys/{jid}/reader-state").json()
        self.assertEqual(state["checkpoint"]["revision"], 1)
        self.assertEqual(state["checkpoint"]["last_question_id"], first_slot["id"])

    def test_checkpoint_conflict_returns_current_server_state(self):
        journey = self._create_journey()
        jid = journey["id"]
        self.client.post(
            f"/api/journeys/{jid}/batches",
            json={"idempotency_key": "k1", "slot_count": 3},
        )
        slots = self.client.get(f"/api/journeys/{jid}/reader-state").json()["questions"]

        first = self.client.patch(
            f"/api/journeys/{jid}/checkpoints",
            json={
                "expected_revision": 0,
                "mutation_id": "tab-a",
                "last_question_id": slots[0]["id"],
            },
        )
        self.assertEqual(first.status_code, 200, first.text)

        # A second tab with a stale expected revision loses and learns the truth.
        conflict = self.client.patch(
            f"/api/journeys/{jid}/checkpoints",
            json={
                "expected_revision": 0,
                "mutation_id": "tab-b",
                "last_question_id": slots[1]["id"],
            },
        )
        self.assertEqual(conflict.status_code, 409, conflict.text)
        detail = conflict.json()["detail"]
        self.assertEqual(detail["current"]["revision"], 1)
        self.assertEqual(detail["current"]["last_question_id"], slots[0]["id"])

        # The losing tab's write did not clobber confirmed state.
        state = self.client.get(f"/api/journeys/{jid}/reader-state").json()
        self.assertEqual(state["checkpoint"]["last_question_id"], slots[0]["id"])

    def test_checkpoint_mutation_replay(self):
        journey = self._create_journey()
        jid = journey["id"]
        self.client.post(
            f"/api/journeys/{jid}/batches",
            json={"idempotency_key": "k1", "slot_count": 3},
        )
        slots = self.client.get(f"/api/journeys/{jid}/reader-state").json()["questions"]

        first = self.client.patch(
            f"/api/journeys/{jid}/checkpoints",
            json={
                "expected_revision": 0,
                "mutation_id": "retry-me",
                "last_question_id": slots[0]["id"],
            },
        ).json()["checkpoint"]
        replay = self.client.patch(
            f"/api/journeys/{jid}/checkpoints",
            json={
                "expected_revision": 0,
                "mutation_id": "retry-me",
                "last_question_id": slots[1]["id"],
            },
        ).json()["checkpoint"]
        self.assertEqual(replay["revision"], first["revision"])
        self.assertEqual(replay["last_question_id"], first["last_question_id"])

    def test_checkpoint_rejects_foreign_references(self):
        journey = self._create_journey()
        jid = journey["id"]
        response = self.client.patch(
            f"/api/journeys/{jid}/checkpoints",
            json={
                "expected_revision": 0,
                "mutation_id": "m1",
                "last_question_id": "slot_nope",
            },
        )
        self.assertEqual(response.status_code, 422, response.text)


class UnavailableServiceTests(JourneysRouterTestBase):
    def test_missing_store_returns_503(self):
        journeys._store = None
        response = self.client.get("/api/journeys")
        self.assertEqual(response.status_code, 503, response.text)

    def test_missing_parser_returns_503_for_static_source(self):
        journeys._parser = None
        response = self.client.post(
            "/api/journeys",
            json={"title": "x", "source_type": "static_topic", "source_topic_id": self.topic_id},
        )
        self.assertEqual(response.status_code, 503, response.text)

    def test_missing_learning_store_returns_503_for_custom_source(self):
        journeys._learning_store = None
        response = self.client.post(
            "/api/journeys",
            json={"title": "x", "source_type": "custom_topic", "source_topic_id": "custom-1"},
        )
        self.assertEqual(response.status_code, 503, response.text)


class RefreshEndpointTests(JourneysRouterTestBase):
    def test_refresh_reports_no_change_for_stable_curriculum(self):
        journey = self._create_journey()
        response = self.client.post(f"/api/journeys/{journey['id']}/refresh")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertFalse(payload["plan_changed"])
        self.assertEqual(payload["journey"]["plan_version"], 1)


if __name__ == "__main__":
    unittest.main()
