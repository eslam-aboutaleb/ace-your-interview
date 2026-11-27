"""API tests for the journey sources endpoints (issue #14).

Covers the feature flag (404 when off), attach/list/refresh flows against an
injected fake fetcher, validation errors, and ownership isolation over HTTP.
"""

import os
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import require_auth
from app.routers import journeys
from app.services.journey_source_ingest import SourceFileSnapshot, SourceSnapshot
from app.services.journey_store import JourneyStore

JOURNEYS_ENV = "STUDY_ENABLE_JOURNEYS_V1"
SOURCES_ENV = "STUDY_ENABLE_JOURNEY_SOURCES_V1"

FILES = {
    "README.md": "# Sample project",
    "docs/design.md": "Design notes.",
    "src/sync.py": "def sync():\n    return 1\n",
}

PLAN = {
    "modules": [
        {
            "title": "HTTP",
            "description": "",
            "objectives": [
                {"title": "GET", "kind": "concept", "source_section": "HTTP"}
            ],
        }
    ]
}


class FakeFetcher:
    def __init__(self, files, revision="rev-1"):
        self.files = dict(files)
        self.revision = revision

    def fetch(self, locator):
        return SourceSnapshot(
            revision=self.revision,
            files=[SourceFileSnapshot(path=p, content=c) for p, c in self.files.items()],
        )


class JourneySourcesRouterTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.store = JourneyStore(os.path.join(self.tmpdir.name, "learning.db"))
        self.fetcher = FakeFetcher(FILES)
        journeys.init(self.store, source_fetcher=self.fetcher)

        self.prev = {k: os.environ.get(k) for k in (JOURNEYS_ENV, SOURCES_ENV)}
        os.environ[JOURNEYS_ENV] = "true"
        os.environ[SOURCES_ENV] = "true"
        get_settings.cache_clear()

        self.app = FastAPI()
        self.app.include_router(journeys.router)
        self.current_user = {"user": "learner", "provider": "google"}
        self.app.dependency_overrides[require_auth] = lambda: self.current_user
        self.client = TestClient(self.app)

        self.journey_id = self.store.create_journey(
            user_id="learner",
            title="Journey",
            goal="ace the interview",
            source_type="static_topic",
            source_topic_id="01-backend-fundamentals-and-http",
            track="backend",
            level="junior",
            plan=PLAN,
        )["id"]

    def tearDown(self):
        self.app.dependency_overrides.clear()
        journeys._store = None
        journeys._source_fetcher = None
        for key, value in self.prev.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()
        self.tmpdir.cleanup()

    def _attach(self):
        return self.client.post(
            f"/api/journeys/{self.journey_id}/sources",
            json={"kind": "github_repo", "locator": "https://github.com/o/r"},
        )

    def test_attach_list_and_refresh(self):
        created = self._attach()
        self.assertEqual(created.status_code, 201, created.text)
        body = created.json()
        self.assertEqual(body["source"]["revision_hash"], "rev-1")
        self.assertEqual(body["source"]["file_count"], 3)
        self.assertEqual(
            sorted(body["diff"]["added"]), ["README.md", "docs/design.md", "src/sync.py"]
        )

        listed = self.client.get(f"/api/journeys/{self.journey_id}/sources")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(len(listed.json()["sources"]), 1)

        updated = dict(FILES)
        updated["src/sync.py"] = "def sync():\n    return 2\n"
        self.fetcher.files = updated
        self.fetcher.revision = "rev-2"
        source_id = body["source"]["id"]
        refreshed = self.client.post(
            f"/api/journeys/{self.journey_id}/sources/{source_id}/refresh"
        )
        self.assertEqual(refreshed.status_code, 200, refreshed.text)
        self.assertEqual(refreshed.json()["diff"]["changed"], ["src/sync.py"])

    def test_flag_off_returns_404(self):
        os.environ[SOURCES_ENV] = "false"
        get_settings.cache_clear()
        try:
            response = self._attach()
            self.assertEqual(response.status_code, 404)
            response = self.client.get(f"/api/journeys/{self.journey_id}/sources")
            self.assertEqual(response.status_code, 404)
        finally:
            os.environ[SOURCES_ENV] = "true"
            get_settings.cache_clear()

    def test_invalid_locator_returns_422(self):
        response = self.client.post(
            f"/api/journeys/{self.journey_id}/sources",
            json={"kind": "github_repo", "locator": "https://gitlab.com/o/r"},
        )
        self.assertEqual(response.status_code, 422)

    def test_missing_journey_returns_404(self):
        response = self.client.post(
            "/api/journeys/jrn_missing/sources",
            json={"kind": "github_repo", "locator": "https://github.com/o/r"},
        )
        self.assertEqual(response.status_code, 404)

    def test_other_users_journey_is_not_visible(self):
        source_id = self._attach().json()["source"]["id"]
        self.current_user = {"user": "intruder", "provider": "google"}
        listed = self.client.get(f"/api/journeys/{self.journey_id}/sources")
        self.assertEqual(listed.status_code, 404)
        refreshed = self.client.post(
            f"/api/journeys/{self.journey_id}/sources/{source_id}/refresh"
        )
        self.assertEqual(refreshed.status_code, 404)


if __name__ == "__main__":
    unittest.main()
