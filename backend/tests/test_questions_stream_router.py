import json
import os
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.dependencies import require_auth
from app.routers import questions
from app.services.doc_parser import DocParser
from app.services.learning_store import LearningStore


def _question_payload(question: str) -> list[dict]:
    return [
        {
            "question": question,
            "answer": (
                "This answer describes the concept, the implementation constraints, and practical "
                "tradeoffs expected in a strong interview response."
            ),
            "difficulty": "medium",
            "learning_objective": "Understand practical tradeoffs from the docs.",
            "source_section": "JVM Fundamentals",
            "source_quote": "JVM memory behavior shapes concurrency and performance choices.",
            "misconception_trap": "Assuming GC removes all concurrency considerations.",
            "reasoning_summary": "Map constraints first, then reason through options.",
            "target_level": "mid",
        }
    ]


class FakeLLM:
    def __init__(self):
        self.blocked = False
        self.calls = 0

    async def completion(self, prompt, llm_config=None, user_identity=None):
        if self.blocked:
            return {
                "success": False,
                "analysis": "",
                "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
                "error": "approval required",
                "error_code": "llm_service_approval_required",
            }

        self.calls += 1
        return {
            "success": True,
            "analysis": json.dumps(_question_payload(f"Question {self.calls}: explain JVM tradeoffs.")),
            "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
            "error": "",
            "error_code": "",
        }


class QuestionsStreamRouterTests(unittest.TestCase):
    def setUp(self):
        self.prev_llm = questions._llm_client
        self.prev_parser = questions._parser
        self.prev_store = questions._learning_store

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

Static topic body.
"""
            )

        self.parser = DocParser(docs_path=self.tempdir.name)
        self.store = LearningStore(os.path.join(self.tempdir.name, "learning.db"))
        self.store.upsert_custom_topic(
            user_id="alice",
            topic_id="custom-java",
            source_topic="Java",
            title="Java Interview Roadmap",
            description="Deep custom Java roadmap.",
            track="backend",
            levels=["junior", "mid", "senior"],
            sections=[
                {
                    "heading": "JVM Fundamentals",
                    "content": "Understand memory model, GC, and runtime behavior.",
                }
            ],
            raw_content=(
                "# Java Interview Roadmap\n\nDeep custom Java roadmap.\n\n## JVM Fundamentals\n\n"
                "Understand memory model, GC, and runtime behavior."
            ),
        )

        self.fake_llm = FakeLLM()
        questions.init(self.fake_llm, self.parser, self.store)
        self.app = FastAPI()
        self.app.dependency_overrides[require_auth] = (
            lambda: {"user": "alice", "provider": "local"}
        )
        self.app.include_router(questions.router)
        self.client = TestClient(self.app)

    def tearDown(self):
        questions._llm_client = self.prev_llm
        questions._parser = self.prev_parser
        questions._learning_store = self.prev_store
        self.tempdir.cleanup()

    def _stream_events(self, payload: dict) -> tuple[int, list[dict]]:
        with self.client.stream("POST", "/api/questions/generate-v2/stream", json=payload) as res:
            events = []
            for line in res.iter_lines():
                if not line:
                    continue
                text = line.decode("utf-8") if isinstance(line, bytes) else line
                events.append(json.loads(text))
            return res.status_code, events

    def test_generate_v2_stream_emits_start_question_done(self):
        status, events = self._stream_events(
            {"topic_id": "custom-java", "count": 2, "level": "mid"}
        )
        self.assertEqual(status, 200)
        self.assertGreaterEqual(len(events), 3)
        self.assertEqual(events[0]["type"], "start")
        self.assertEqual(events[0]["topic_id"], "custom-java")
        self.assertIn("question", [e["type"] for e in events])
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["generated_count"], 2)

    def test_generate_v2_stream_emits_terminal_policy_error_event(self):
        self.fake_llm.blocked = True
        status, events = self._stream_events(
            {"topic_id": "custom-java", "count": 1, "level": "mid"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(events[0]["type"], "start")
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["code"], "llm_service_approval_required")


if __name__ == "__main__":
    unittest.main()
