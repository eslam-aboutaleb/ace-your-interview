"""API tests for the generation jobs endpoints (issue #16).

Covers the feature flag, idempotent submission over HTTP, status polling,
cancellation, retry, and per-user isolation.
"""

import os
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import require_auth
from app.routers import generation_jobs
from app.services.generation_jobs import GenerationJobStore, GenerationOrchestrator

JOBS_ENV = "STUDY_ENABLE_GENERATION_JOBS_V1"


async def _ok_generator(item):
    return {"text": f"generated: {item}"}


class GenerationJobsRouterTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.store = GenerationJobStore(os.path.join(self.tmpdir.name, "learning.db"))
        generation_jobs.init(GenerationOrchestrator(self.store, _ok_generator))

        self.prev = os.environ.get(JOBS_ENV)
        os.environ[JOBS_ENV] = "true"
        get_settings.cache_clear()

        self.app = FastAPI()
        self.app.include_router(generation_jobs.router)
        self.current_user = {"user": "learner", "provider": "google"}
        self.app.dependency_overrides[require_auth] = lambda: self.current_user
        self.client = TestClient(self.app)

    def tearDown(self):
        self.app.dependency_overrides.clear()
        generation_jobs._orchestrator = None
        if self.prev is None:
            os.environ.pop(JOBS_ENV, None)
        else:
            os.environ[JOBS_ENV] = self.prev
        get_settings.cache_clear()
        self.tmpdir.cleanup()

    def _create(self, key="key-1", outline=("a", "b")):
        return self.client.post(
            "/api/generation-jobs",
            json={"idempotency_key": key, "outline": list(outline)},
        )

    def test_create_runs_and_returns_job(self):
        response = self._create()
        self.assertEqual(response.status_code, 201, response.text)
        body = response.json()
        self.assertTrue(body["created"])
        self.assertEqual(body["job"]["status"], "completed")
        self.assertEqual(len(body["job"]["accepted"]), 2)

    def test_duplicate_submission_is_not_recharged(self):
        first = self._create()
        second = self._create()
        self.assertEqual(second.status_code, 201)
        self.assertFalse(second.json()["created"])
        self.assertEqual(second.json()["job"]["id"], first.json()["job"]["id"])
        self.assertEqual(second.json()["job"]["cost_units"], 2)

    def test_get_and_list(self):
        job_id = self._create().json()["job"]["id"]
        got = self.client.get(f"/api/generation-jobs/{job_id}")
        self.assertEqual(got.status_code, 200)
        self.assertEqual(got.json()["job"]["id"], job_id)
        listed = self.client.get("/api/generation-jobs")
        self.assertEqual(len(listed.json()["jobs"]), 1)

    def test_get_other_users_job_is_404(self):
        job_id = self._create().json()["job"]["id"]
        self.current_user = {"user": "intruder", "provider": "google"}
        self.assertEqual(self.client.get(f"/api/generation-jobs/{job_id}").status_code, 404)

    def test_cancel_queued_job(self):
        job, _ = self.store.create_job(
            user_id="learner", idempotency_key="manual", outline=["x"]
        )
        response = self.client.post(f"/api/generation-jobs/{job['id']}/cancel")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["job"]["status"], "cancelled")
        again = self.client.post(f"/api/generation-jobs/{job['id']}/cancel")
        self.assertEqual(again.status_code, 409)

    def test_retry_completed_job_is_409(self):
        job_id = self._create().json()["job"]["id"]
        response = self.client.post(f"/api/generation-jobs/{job_id}/retry")
        self.assertEqual(response.status_code, 409)

    def test_flag_off_returns_404(self):
        os.environ[JOBS_ENV] = "false"
        get_settings.cache_clear()
        try:
            self.assertEqual(self._create(key="off").status_code, 404)
            self.assertEqual(self.client.get("/api/generation-jobs").status_code, 404)
        finally:
            os.environ[JOBS_ENV] = "true"
            get_settings.cache_clear()


if __name__ == "__main__":
    unittest.main()
