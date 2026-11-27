"""API tests for the journey batch generation endpoint (issue #2).

The LLM generator is a deterministic stub injected in place of
``QuestionGenerator``; everything else (store, parser, app) is real.
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

_STEMS = [
    "How does {t} handle its first production outage?",
    "Walk me through the consistency guarantees of {t}.",
    "What trade-offs does {t} make under heavy write load?",
    "Explain {t} to a junior engineer with a concrete analogy.",
]


class _StubItem:
    def __init__(self, question: str):
        self.question = question

    def model_dump(self):
        return {"question": self.question}


class _StubResponse:
    def __init__(self, question: str):
        self.questions = [_StubItem(question)]
        self.provider_used = "stub"
        self.model_used = "stub-model"
        self.generation_version = "stub@v1"


class _StubGenerator:
    """Deterministic stand-in for QuestionGenerator.generate_v2."""

    def __init__(self):
        self.calls: list[dict] = []

    async def generate_v2(self, **kwargs):
        seq = len(self.calls)
        self.calls.append(kwargs)
        section = kwargs.get("section_title") or "topic"
        stem = _STEMS[seq % len(_STEMS)]
        return _StubResponse(stem.format(t=f"{section} v{seq}"))


class GenerationRouterTestBase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "learning.db")
        self.store = JourneyStore(self.db_path)
        self.learning_store = LearningStore(self.db_path)
        self.parser = DocParser()
        self.generator = _StubGenerator()
        journeys.init(
            self.store, self.learning_store, self.parser,
            question_generator=self.generator,
        )

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
        journeys._question_generator = None
        journeys._llm_client = None
        journeys._embedding_client = None
        if self.prev_env is None:
            os.environ.pop(JOURNEYS_ENV, None)
        else:
            os.environ[JOURNEYS_ENV] = self.prev_env
        get_settings.cache_clear()
        self.tmpdir.cleanup()

    def _journey_with_batch(self, slot_count=3, **batch_overrides):
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
        journey = response.json()["journey"]
        body = {"idempotency_key": "batch-1", "slot_count": slot_count}
        body.update(batch_overrides)
        response = self.client.post(f"/api/journeys/{journey['id']}/batches", json=body)
        self.assertIn(response.status_code, (200, 201), response.text)
        batch = response.json()["batch"]
        return journey, batch


class FeatureGateTests(GenerationRouterTestBase):
    def test_flag_off_hides_generate(self):
        os.environ[JOURNEYS_ENV] = "false"
        get_settings.cache_clear()
        response = self.client.post(
            "/api/journeys/jrn_x/batches/bat_x/generate", json={}
        )
        self.assertEqual(response.status_code, 404, response.text)


class GenerateFlowTests(GenerationRouterTestBase):
    def test_generate_fills_batch_and_records_versions(self):
        journey, batch = self._journey_with_batch(slot_count=3)
        response = self.client.post(
            f"/api/journeys/{journey['id']}/batches/{batch['id']}/generate",
            json={"prompt_version": "questions@v3", "model_version": "m1", "schema_version": "v2"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["job"]["status"], "completed")
        self.assertEqual(payload["job"]["accepted_count"], 3)
        self.assertEqual(payload["job"]["rejected_count"], 0)
        self.assertEqual(payload["job"]["prompt_version"], "questions@v3")
        self.assertEqual(payload["job"]["model_version"], "m1")
        self.assertEqual(payload["job"]["schema_version"], "v2")
        self.assertEqual(payload["batch"]["accepted_count"], 3)
        self.assertEqual(len(payload["results"]), 3)
        for result in payload["results"]:
            self.assertEqual(result["outcome"], "accepted")
        # the generator was aimed at each slot's objective
        self.assertEqual(len(self.generator.calls), 3)
        for call in self.generator.calls:
            self.assertTrue(call["section_title"])
            self.assertEqual(call["count"], 1)

    def test_regenerate_finished_batch_is_409(self):
        journey, batch = self._journey_with_batch(slot_count=1)
        url = f"/api/journeys/{journey['id']}/batches/{batch['id']}/generate"
        first = self.client.post(url, json={})
        self.assertEqual(first.status_code, 200, first.text)
        second = self.client.post(url, json={})
        self.assertEqual(second.status_code, 409, second.text)

    def test_unknown_journey_and_foreign_user_404(self):
        journey, batch = self._journey_with_batch(slot_count=1)
        response = self.client.post("/api/journeys/jrn_nope/batches/bat_x/generate", json={})
        self.assertEqual(response.status_code, 404, response.text)
        self.current_user = {"user": "intruder", "provider": "google"}
        response = self.client.post(
            f"/api/journeys/{journey['id']}/batches/{batch['id']}/generate", json={}
        )
        self.assertEqual(response.status_code, 404, response.text)

    def test_perspectives_flow_into_generation(self):
        journey, batch = self._journey_with_batch(
            slot_count=2, perspectives=["conceptual", "applied"]
        )
        response = self.client.post(
            f"/api/journeys/{journey['id']}/batches/{batch['id']}/generate", json={}
        )
        self.assertEqual(response.status_code, 200, response.text)
        titles = [call["section_title"] for call in self.generator.calls]
        self.assertTrue(any("conceptual perspective" in t for t in titles))
        self.assertTrue(any("applied perspective" in t for t in titles))
        # fingerprints record the perspective for the dedup funnel
        fps = self.store.fingerprints_for_journey(user_id="learner", journey_id=journey["id"])
        self.assertEqual({f["perspective"] for f in fps}, {"conceptual", "applied"})

    def test_novelty_enforced_over_http(self):
        journey, batch = self._journey_with_batch(slot_count=2)

        class _DupGenerator(_StubGenerator):
            async def generate_v2(self, **kwargs):
                seq = len(self.calls)
                self.calls.append(kwargs)
                if seq == 0:
                    return _StubResponse("What is the HTTP request/response lifecycle?")
                if seq == 1:
                    return _StubResponse("What is the HTTP request/response lifecycle?")
                return _StubResponse("Walk me through TLS session resumption trade-offs.")

        journeys._question_generator = _DupGenerator()
        response = self.client.post(
            f"/api/journeys/{journey['id']}/batches/{batch['id']}/generate", json={}
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["job"]["accepted_count"], 2)
        self.assertEqual(payload["job"]["rejected_count"], 0)
        self.assertEqual(len(journeys._question_generator.calls), 3, "only the dup slot retried")
        fps = self.store.fingerprints_for_journey(user_id="learner", journey_id=journey["id"])
        texts = [f["question_text"] for f in fps]
        self.assertEqual(len(set(texts)), 2, "zero exact duplicates accepted")


if __name__ == "__main__":
    unittest.main()
