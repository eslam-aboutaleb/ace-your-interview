import json
import os
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.dependencies import optional_auth, require_auth
from app.routers import topics
from app.services.doc_parser import DocParser
from app.services.learning_store import LearningStore


class FakeLLM:
    async def completion(self, prompt, llm_config=None, user_identity=None):
        if '"requires_programming"' in prompt:
            payload = {
                "requires_programming": True,
                "language_options": ["python", "go", "typescript"],
            }
        else:
            payload = {
                "title": "Custom Topic",
                "description": "Custom desc",
                "track": "backend",
                "levels": ["junior", "mid", "senior"],
                "sections": [
                    {"heading": f"Section {i}", "content": "Content for section."}
                    for i in range(1, 101)
                ],
            }
        return {
            "success": True,
            "analysis": json.dumps(payload),
            "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
            "error": "",
        }


class TopicsRouterPreferencesTests(unittest.TestCase):
    def setUp(self):
        self.prev_parser = topics._parser
        self.prev_llm = topics._llm_client
        self.prev_store = topics._learning_store
        self.prev_mcp = topics._mcp_gateway

        self.tempdir = tempfile.TemporaryDirectory()
        handbook = os.path.join(self.tempdir.name, "owner-handbook")
        os.makedirs(handbook, exist_ok=True)
        with open(os.path.join(handbook, "01-static.md"), "w", encoding="utf-8") as f:
            f.write(
                """---
track: backend
levels: [junior, mid, senior]
---
# Static Topic

Static description.

## APIs
API implementation details.
"""
            )

        parser = DocParser(docs_path=self.tempdir.name)
        store = LearningStore(os.path.join(self.tempdir.name, "learning.db"))
        topics.init(parser, FakeLLM(), store)

        self.app = FastAPI()
        self.app.dependency_overrides[require_auth] = (
            lambda: {"user": "alice", "provider": "local"}
        )
        self.app.dependency_overrides[optional_auth] = (
            lambda: {"user": "alice", "provider": "local"}
        )
        self.app.include_router(topics.router)
        self.client = TestClient(self.app)

    def tearDown(self):
        topics._parser = self.prev_parser
        topics._llm_client = self.prev_llm
        topics._learning_store = self.prev_store
        topics._mcp_gateway = self.prev_mcp
        self.tempdir.cleanup()

    def test_topic_detail_is_enriched_with_ai_metadata(self):
        res = self.client.get("/api/topics/01-static")
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertIn("requires_programming", payload)
        self.assertIn("language_options", payload)
        self.assertIn("selected_language", payload)
        self.assertIn("response_detail", payload)
        self.assertTrue(payload["requires_programming"])
        self.assertGreaterEqual(len(payload["language_options"]), 3)

    def test_get_and_update_topic_preferences(self):
        get1 = self.client.get("/api/topics/01-static/preferences")
        self.assertEqual(get1.status_code, 200)
        initial = get1.json()
        self.assertEqual(initial["response_detail"], "very_detailed")

        put = self.client.put(
            "/api/topics/01-static/preferences",
            json={"response_detail": "very_detailed", "preferred_language": "go"},
        )
        self.assertEqual(put.status_code, 200)
        updated = put.json()
        self.assertEqual(updated["response_detail"], "very_detailed")
        self.assertEqual(updated["preferred_language"], "go")

        get2 = self.client.get("/api/topics/01-static/preferences")
        self.assertEqual(get2.status_code, 200)
        final = get2.json()
        self.assertEqual(final["response_detail"], "very_detailed")
        self.assertEqual(final["preferred_language"], "go")

    def test_update_topic_preferences_rejects_invalid_language(self):
        res = self.client.put(
            "/api/topics/01-static/preferences",
            json={"preferred_language": "haskell"},
        )
        self.assertEqual(res.status_code, 422)


if __name__ == "__main__":
    unittest.main()
