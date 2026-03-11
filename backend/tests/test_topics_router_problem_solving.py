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
            "title": "Problem Solving and Algorithms (java)",
            "description": "Java roadmap for coding interview problem solving.",
            "track": "backend",
            "levels": ["junior", "mid", "senior"],
            "sections": [
                {
                    "heading": f"Java Pattern {i}",
                    "content": (
                        "Read constraints, choose strategy, analyze complexity, and explain tradeoffs."
                    ),
                }
                for i in range(1, 121)
            ],
        }
        return {
            "success": True,
            "analysis": json.dumps(payload),
            "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
            "error": "",
        }


class TopicsRouterProblemSolvingTests(unittest.TestCase):
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

        parser = DocParser(docs_path=self.tempdir.name)
        store = LearningStore(os.path.join(self.tempdir.name, "learning.db"))
        topics.init(parser, FakeLLM(), store)

        app = FastAPI()
        app.dependency_overrides[require_auth] = (
            lambda: {"user": "alice", "provider": "local"}
        )
        app.dependency_overrides[optional_auth] = (
            lambda: {"user": "alice", "provider": "local"}
        )
        app.include_router(topics.router)
        self.client = TestClient(app)

    def tearDown(self):
        topics._parser = self.prev_parser
        topics._llm_client = self.prev_llm
        topics._learning_store = self.prev_store
        topics._mcp_gateway = self.prev_mcp
        self.tempdir.cleanup()

    def test_problem_solving_topic_in_list_and_detail_without_generated_content(self):
        list_res = self.client.get("/api/topics")
        self.assertEqual(list_res.status_code, 200)
        ids = {item["id"] for item in list_res.json()}
        self.assertIn("00-problem-solving-and-algorithms", ids)

        detail_res = self.client.get("/api/topics/00-problem-solving-and-algorithms")
        self.assertEqual(detail_res.status_code, 200)
        detail = detail_res.json()
        self.assertTrue(detail["is_dynamic_topic"])
        self.assertFalse(detail["content_ready"])
        self.assertEqual(detail["selected_language"], "python")
        self.assertEqual(detail["sections"], [])

    def test_problem_solving_stream_generates_and_persists_curriculum(self):
        stream_res = self.client.post(
            "/api/topics/00-problem-solving-and-algorithms/content/generate/stream",
            json={"preferred_language": "java"},
        )
        self.assertEqual(stream_res.status_code, 200)
        events = [json.loads(line) for line in stream_res.text.splitlines() if line.strip()]
        event_types = [event.get("type") for event in events]
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

        done = next(e for e in events if e.get("type") == "done")
        topic_payload = done["topic"]
        self.assertEqual(topic_payload["id"], "00-problem-solving-and-algorithms")
        self.assertEqual(topic_payload["selected_language"], "java")
        self.assertEqual(len(topic_payload["sections"]), 120)
        self.assertTrue(topic_payload["content_ready"])

        detail_res = self.client.get("/api/topics/00-problem-solving-and-algorithms")
        self.assertEqual(detail_res.status_code, 200)
        detail = detail_res.json()
        self.assertTrue(detail["content_ready"])
        self.assertEqual(detail["selected_language"], "java")
        self.assertEqual(len(detail["sections"]), 120)


if __name__ == "__main__":
    unittest.main()
