"""API tests for the tutor explain endpoint (issue #15).

Covers the feature flag (404 when off), the explain flow over HTTP, the
cost cap (429), and per-user isolation of the budget.
"""

import os
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import require_auth
from app.routers import tutor
from app.services.tutor_service import TUTOR_COST_CAP_PER_QUESTION, TutorService

TUTOR_ENV = "STUDY_ENABLE_TUTOR_V1"


class FakeLLM:
    def __init__(self):
        self.calls = 0

    async def completion(self, prompt, llm_config=None, user_identity=None, **kwargs):
        self.calls += 1
        return {
            "success": True,
            "analysis": "A grounded explanation.",
            "metadata": {},
            "error_code": "",
            "finish_reason": "stop",
            "usage": {},
        }


class TutorRouterTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.service = TutorService(FakeLLM(), os.path.join(self.tmpdir.name, "learning.db"))
        tutor.init(self.service)

        self.prev = os.environ.get(TUTOR_ENV)
        os.environ[TUTOR_ENV] = "true"
        get_settings.cache_clear()

        self.app = FastAPI()
        self.app.include_router(tutor.router)
        self.current_user = {"user": "learner", "provider": "google"}
        self.app.dependency_overrides[require_auth] = lambda: self.current_user
        self.client = TestClient(self.app)

    def tearDown(self):
        self.app.dependency_overrides.clear()
        tutor._service = None
        if self.prev is None:
            os.environ.pop(TUTOR_ENV, None)
        else:
            os.environ[TUTOR_ENV] = self.prev
        get_settings.cache_clear()
        self.tmpdir.cleanup()

    def _explain(self, **overrides):
        body = {
            "action": "simpler",
            "question_text": "What is binary search?",
            "answer_context": "Binary search halves the search space each step.",
            "source_section": "Searching",
        }
        body.update(overrides)
        return self.client.post("/api/questions/q-1/explain", json=body)

    def test_explain_flow(self):
        response = self._explain()
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["explanation"], "A grounded explanation.")
        self.assertEqual(body["citations"][0]["section"], "Searching")
        self.assertEqual(body["explains_used"], 1)

    def test_flag_off_returns_404(self):
        os.environ[TUTOR_ENV] = "false"
        get_settings.cache_clear()
        try:
            self.assertEqual(self._explain().status_code, 404)
        finally:
            os.environ[TUTOR_ENV] = "true"
            get_settings.cache_clear()

    def test_invalid_action_returns_422(self):
        response = self.client.post("/api/questions/q-1/explain", json={"action": "dance"})
        self.assertEqual(response.status_code, 422)

    def test_cost_cap_returns_429(self):
        for _ in range(TUTOR_COST_CAP_PER_QUESTION):
            self.assertEqual(self._explain().status_code, 200)
        self.assertEqual(self._explain().status_code, 429)

    def test_budget_isolated_per_user(self):
        for _ in range(TUTOR_COST_CAP_PER_QUESTION):
            self._explain()
        self.current_user = {"user": "someone-else", "provider": "google"}
        self.assertEqual(self._explain().status_code, 200)

    def test_refusal_is_200_with_refusal_flag(self):
        response = self._explain(question_text="", answer_context="")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["refusal"])


if __name__ == "__main__":
    unittest.main()
