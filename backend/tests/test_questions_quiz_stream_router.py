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


class FakeLLM:
    def __init__(self):
        self.mode = "success"
        self.calls = 0

    async def completion(self, prompt, llm_config=None, user_identity=None, task=None, **kwargs):
        if self.mode == "blocked":
            return {
                "success": False,
                "analysis": "",
                "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
                "error": "approval required",
                "error_code": "llm_service_approval_required",
            }
        if self.mode == "invalid":
            return {
                "success": True,
                "analysis": "not-json",
                "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
                "error": "",
                "error_code": "",
            }

        self.calls += 1
        payload = [self._item(self.calls)]
        return {
            "success": True,
            "analysis": json.dumps(payload),
            "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
            "error": "",
            "error_code": "",
        }

    def _item(self, call: int) -> dict:
        # Distinct stems per call: near-duplicate questions are
        # dropped by the generator's dedup pass.
        if call % 2 == 0:
            return {
                "question": "Which structure preserves insertion order while offering fast lookup?",
                "type": "mcq",
                "choices": [
                    {"label": "A", "text": "LinkedHashMap"},
                    {"label": "B", "text": "HashSet"},
                    {"label": "C", "text": "TreeSet"},
                    {"label": "D", "text": "PriorityQueue"},
                ],
                "correct_answer": "A",
                "explanation": "LinkedHashMap preserves insertion order with hash-based lookup.",
                "difficulty": "medium",
                "topic_id": "custom-java",
                "source_quote": "Understand memory model, GC, and runtime behavior.",
                "reasoning_summary": "Insertion order with key lookups points to LinkedHashMap.",
                "target_level": "mid",
                "learning_objective": "Explain which collection preserves insertion order.",
                "expected_answer": "LinkedHashMap",
                "distractor_rationales": [
                    {"label": "B", "rationale": "HashSet does not preserve insertion order."},
                    {"label": "C", "rationale": "TreeSet sorts elements naturally, not by insertion."},
                    {"label": "D", "rationale": "PriorityQueue orders by priority, not insertion."},
                ],
            }
        return {
            "question": "Which collection guarantees logarithmic-time lookups via a balanced tree?",
            "type": "mcq",
            "choices": [
                {"label": "A", "text": "TreeMap"},
                {"label": "B", "text": "HashMap"},
                {"label": "C", "text": "LinkedHashMap"},
                {"label": "D", "text": "ArrayList"},
            ],
            "correct_answer": "A",
            "explanation": "TreeMap backs its entries with a red-black tree for logarithmic lookups.",
            "difficulty": "medium",
            "topic_id": "custom-java",
            "source_quote": "Understand memory model, GC, and runtime behavior.",
            "reasoning_summary": "Balanced-tree backing gives logarithmic lookup cost.",
            "target_level": "mid",
            "learning_objective": "Explain which collection provides logarithmic lookups.",
            "expected_answer": "TreeMap",
            "distractor_rationales": [
                {"label": "B", "rationale": "HashMap uses hashing, not a balanced tree."},
                {"label": "C", "rationale": "LinkedHashMap keeps insertion order, not tree order."},
                {"label": "D", "rationale": "ArrayList is an indexed array, not a tree."},
            ],
        }


class QuizStreamRouterTests(unittest.TestCase):
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
        with self.client.stream("POST", "/api/questions/quiz/generate-v2/stream", json=payload) as res:
            events = []
            for line in res.iter_lines():
                if not line:
                    continue
                text = line.decode("utf-8") if isinstance(line, bytes) else line
                events.append(json.loads(text))
            return res.status_code, events

    def test_quiz_stream_emits_start_progress_question_done(self):
        status, events = self._stream_events(
            {
                "topic_ids": ["custom-java"],
                "count": 2,
                "question_types": ["mcq"],
                "level": "mid",
            }
        )
        self.assertEqual(status, 200)
        self.assertEqual(events[0]["type"], "start")
        self.assertIn("progress", [event.get("type") for event in events])
        self.assertEqual(len([event for event in events if event.get("type") == "question"]), 2)
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["generated_count"], 2)

    def test_quiz_stream_reports_a_degraded_partial_on_parse_failure(self):
        self.fake_llm.mode = "invalid"
        status, events = self._stream_events(
            {
                "topic_ids": ["custom-java"],
                "count": 3,
                "question_types": ["mcq"],
                "level": "mid",
            }
        )
        self.assertEqual(status, 200)
        question_events = [event for event in events if event.get("type") == "question"]
        # Unparseable provider output is never substituted with
        # generic questions: the stream reports a degraded partial.
        self.assertEqual(len(question_events), 0)
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["generated_count"], 0)
        self.assertTrue(events[-1]["degraded"])
        self.assertEqual(events[-1]["requested_count"], 3)
        self.assertEqual(events[-1]["returned_count"], 0)

    def test_quiz_stream_emits_policy_error_event(self):
        self.fake_llm.mode = "blocked"
        status, events = self._stream_events(
            {
                "topic_ids": ["custom-java"],
                "count": 1,
                "question_types": ["mcq"],
                "level": "mid",
            }
        )
        self.assertEqual(status, 200)
        self.assertEqual(events[0]["type"], "start")
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["code"], "llm_service_approval_required")


if __name__ == "__main__":
    unittest.main()
