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
    async def completion(self, prompt, llm_config=None, user_identity=None):
        prompt_l = prompt.lower()
        if "expert technical quiz creator" in prompt_l:
            payload = [
                {
                    "question": "Which Java collection preserves insertion order and allows fast key lookups?",
                    "type": "mcq",
                    "choices": [
                        {"label": "A", "text": "LinkedHashMap"},
                        {"label": "B", "text": "HashSet"},
                        {"label": "C", "text": "TreeSet"},
                        {"label": "D", "text": "PriorityQueue"},
                    ],
                    "correct_answer": "A",
                    "explanation": "LinkedHashMap preserves insertion order while keeping hash-based lookup behavior.",
                    "difficulty": "medium",
                    "topic_id": "custom-java",
                    "source_quote": "Collections track insertion order behavior.",
                    "reasoning_summary": "Insertion order plus key lookup points to LinkedHashMap.",
                    "target_level": "mid",
                }
            ]
        else:
            payload = [
                {
                    "question": "How would you explain the JVM memory model tradeoffs in an interview?",
                    "answer": (
                        "Start with stack vs heap responsibilities, then explain GC-managed lifecycles, "
                        "visibility guarantees, and why synchronization primitives prevent race conditions. "
                        "Connect this to real-world latency and throughput tradeoffs in production systems."
                    ),
                    "difficulty": "medium",
                    "learning_objective": "Explain memory model fundamentals and interview tradeoffs.",
                    "source_section": "JVM Fundamentals",
                    "source_quote": "JVM memory behavior shapes concurrency and performance choices.",
                    "misconception_trap": "Assuming GC removes the need for concurrency control.",
                    "reasoning_summary": "Map memory areas to correctness, then to performance decisions.",
                    "target_level": "mid",
                }
            ]
        return {
            "success": True,
            "analysis": json.dumps(payload),
            "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
            "error": "",
        }


class QuestionsRouterCustomTopicsTests(unittest.TestCase):
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

## Static Section
More context for generation.
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

        questions.init(FakeLLM(), self.parser, self.store)
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

    def test_generate_v2_accepts_custom_topic(self):
        res = self.client.post(
            "/api/questions/generate-v2",
            json={"topic_id": "custom-java", "count": 1, "level": "mid"},
        )
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertEqual(payload["topic_id"], "custom-java")
        self.assertEqual(len(payload["questions"]), 1)

    def test_generate_quiz_v2_accepts_custom_topic(self):
        res = self.client.post(
            "/api/questions/quiz/generate-v2",
            json={
                "topic_ids": ["custom-java"],
                "count": 1,
                "question_types": ["mcq"],
                "level": "mid",
            },
        )
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertEqual(payload["topics_used"], ["custom-java"])
        self.assertEqual(payload["questions"][0]["topic_id"], "custom-java")

    def test_generate_v2_still_supports_static_topics(self):
        res = self.client.post(
            "/api/questions/generate-v2",
            json={"topic_id": "01-static", "count": 1, "level": "mid"},
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["topic_id"], "01-static")

    def test_problem_solving_topic_requires_generated_curriculum(self):
        res = self.client.post(
            "/api/questions/generate-v2",
            json={
                "topic_id": "00-problem-solving-and-algorithms",
                "count": 1,
                "level": "mid",
                "preferred_language": "python",
            },
        )
        self.assertEqual(res.status_code, 409)

    def test_problem_solving_topic_works_after_curriculum_exists(self):
        self.store.upsert_dynamic_topic_curriculum(
            user_id="alice",
            topic_id="00-problem-solving-and-algorithms",
            preferred_language="python",
            title="Problem Solving and Algorithms (python)",
            description="Python roadmap.",
            track="backend",
            levels=["junior", "mid", "senior"],
            sections=[
                {
                    "heading": "Junior: Two pointers",
                    "content": "Understand constraints and sliding window tradeoffs.",
                }
            ],
            raw_content=(
                "# Problem Solving and Algorithms (python)\n\n## Junior: Two pointers\n\n"
                "Understand constraints and sliding window tradeoffs."
            ),
            target_sections=120,
            source="llm",
        )
        res = self.client.post(
            "/api/questions/generate-v2",
            json={
                "topic_id": "00-problem-solving-and-algorithms",
                "count": 1,
                "level": "mid",
                "preferred_language": "python",
            },
        )
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertEqual(payload["topic_id"], "00-problem-solving-and-algorithms")
        self.assertEqual(len(payload["questions"]), 1)


if __name__ == "__main__":
    unittest.main()
