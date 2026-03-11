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
        payload = {
            "title": "Java Interview Roadmap",
            "description": "Deep roadmap to become interview-ready in Java.",
            "track": "backend",
            "levels": ["junior", "mid", "senior"],
            "sections": [
                {
                    "heading": f"Java Section {i}",
                    "content": "Concise learning notes with practical tradeoffs.",
                }
                for i in range(1, 101)
            ],
        }
        return {
            "success": True,
            "analysis": json.dumps(payload),
            "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
            "error": "",
        }


class TopicsRouterCustomTests(unittest.TestCase):
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
"""
            )

        self.parser = DocParser(docs_path=self.tempdir.name)
        self.store = LearningStore(os.path.join(self.tempdir.name, "learning.db"))
        self.fake_llm = FakeLLM()
        topics.init(self.parser, self.fake_llm, self.store)

        self.current_user = {"user": "alice", "provider": "local"}
        self.app = FastAPI()
        self.app.dependency_overrides[require_auth] = lambda: self.current_user
        self.app.dependency_overrides[optional_auth] = lambda: self.current_user
        self.app.include_router(topics.router)
        self.client = TestClient(self.app)

    def tearDown(self):
        topics._parser = self.prev_parser
        topics._llm_client = self.prev_llm
        topics._learning_store = self.prev_store
        topics._mcp_gateway = self.prev_mcp
        self.tempdir.cleanup()

    def test_custom_topic_create_list_and_get(self):
        create_res = self.client.post(
            "/api/topics/custom",
            json={"topic": "Java", "target_sections": 100},
        )
        self.assertEqual(create_res.status_code, 200)
        created = create_res.json()
        self.assertEqual(created["id"], "custom-java")
        self.assertEqual(len(created["sections"]), 100)

        list_res = self.client.get("/api/topics")
        self.assertEqual(list_res.status_code, 200)
        ids = {item["id"] for item in list_res.json()}
        self.assertIn("01-static", ids)
        self.assertIn("custom-java", ids)

        get_res = self.client.get("/api/topics/custom-java")
        self.assertEqual(get_res.status_code, 200)
        self.assertEqual(get_res.json()["title"], "Java Interview Roadmap")

    def test_custom_topics_are_user_scoped(self):
        self.client.post("/api/topics/custom", json={"topic": "Java", "target_sections": 100})

        self.current_user = {"user": "bob", "provider": "local"}
        list_res = self.client.get("/api/topics")
        self.assertEqual(list_res.status_code, 200)
        ids = {item["id"] for item in list_res.json()}
        self.assertIn("01-static", ids)
        self.assertNotIn("custom-java", ids)

        get_res = self.client.get("/api/topics/custom-java")
        self.assertEqual(get_res.status_code, 404)

    def test_topics_list_supports_opt_in_limit_offset_pagination(self):
        self.client.post("/api/topics/custom", json={"topic": "Java", "target_sections": 100})
        self.client.post("/api/topics/custom", json={"topic": "Python", "target_sections": 100})

        full = self.client.get("/api/topics")
        self.assertEqual(full.status_code, 200)
        total = len(full.json())
        self.assertGreaterEqual(total, 2)

        paged = self.client.get("/api/topics?limit=1&offset=1")
        self.assertEqual(paged.status_code, 200)
        self.assertEqual(len(paged.json()), 1)
        self.assertEqual(paged.headers.get("X-Total-Count"), str(total))
        self.assertEqual(paged.headers.get("X-Offset"), "1")
        self.assertEqual(paged.headers.get("X-Limit"), "1")

    def test_custom_topic_stream_emits_sections_and_done(self):
        res = self.client.post(
            "/api/topics/custom/stream",
            json={"topic": "Java", "target_sections": 100},
        )
        self.assertEqual(res.status_code, 200)
        events = [json.loads(line) for line in res.text.splitlines() if line.strip()]
        event_types = [evt.get("type") for evt in events]

        self.assertIn("start", event_types)
        self.assertIn("progress", event_types)
        self.assertIn("section", event_types)
        self.assertIn("done", event_types)
        first_section_idx = next(i for i, event in enumerate(events) if event.get("type") == "section")
        done_idx = next(i for i, event in enumerate(events) if event.get("type") == "done")
        self.assertLess(first_section_idx, done_idx)

        progress_events = [event for event in events if event.get("type") == "progress"]
        self.assertTrue(any(event.get("stage") == "batching" for event in progress_events))
        batching = next(event for event in progress_events if event.get("stage") == "batching")
        self.assertIn("batch_index", batching)
        self.assertIn("batch_count", batching)
        self.assertIn("generated_sections", batching)
        self.assertIn("target_sections", batching)

        done_event = next(evt for evt in events if evt.get("type") == "done")
        self.assertEqual(done_event["topic"]["id"], "custom-java")
        self.assertEqual(len(done_event["topic"]["sections"]), 100)


if __name__ == "__main__":
    unittest.main()
