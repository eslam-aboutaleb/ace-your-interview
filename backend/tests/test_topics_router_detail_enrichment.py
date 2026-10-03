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
    async def completion(self, prompt, llm_config=None, user_identity=None, task=None, **kwargs):
        payload = {"requires_programming": False, "language_options": []}
        return {
            "success": True,
            "analysis": json.dumps(payload),
            "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
            "error": "",
        }


class TopicsRouterDetailEnrichmentTests(unittest.TestCase):
    def test_detail_enrichment_fields_exist_for_static_topic(self):
        prev_parser = topics._parser
        prev_llm = topics._llm_client
        prev_store = topics._learning_store
        prev_mcp = topics._mcp_gateway

        with tempfile.TemporaryDirectory() as td:
            handbook = os.path.join(td, "owner-handbook")
            os.makedirs(handbook, exist_ok=True)
            with open(os.path.join(handbook, "01-static.md"), "w", encoding="utf-8") as f:
                f.write(
                    """---
track: system_design
levels: [junior, mid, senior]
---
# Static Topic

Static description.
"""
                )

            parser = DocParser(docs_path=td)
            store = LearningStore(os.path.join(td, "learning.db"))
            topics.init(parser, FakeLLM(), store)

            app = FastAPI()
            app.dependency_overrides[require_auth] = (
                lambda: {"user": "alice", "provider": "local"}
            )
            app.dependency_overrides[optional_auth] = (
                lambda: {"user": "alice", "provider": "local"}
            )
            app.include_router(topics.router)
            client = TestClient(app)
            try:
                res = client.get("/api/topics/01-static")
                self.assertEqual(res.status_code, 200)
                payload = res.json()
                self.assertIn("requires_programming", payload)
                self.assertIn("language_options", payload)
                self.assertIn("selected_language", payload)
                self.assertIn("response_detail", payload)
            finally:
                topics._parser = prev_parser
                topics._llm_client = prev_llm
                topics._learning_store = prev_store
                topics._mcp_gateway = prev_mcp


if __name__ == "__main__":
    unittest.main()
