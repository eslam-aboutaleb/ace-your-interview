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


def _problem_solving_payload(question: str) -> list[dict]:
    return [
        {
            "question": question,
            "answer": """### Problem

Understand the input-output contract before choosing the data structure.

### Solution Walkthrough

Use a hash map so every number can check whether its complement has already appeared in O(1). That keeps the solution single-pass and avoids reusing the same element twice.

### Complexity

The algorithm runs in O(n) time with O(n) extra space for the hash map.

### Code

```python
def two_sum(nums, target):
    # Store each seen value so complement lookups stay O(1).
    seen = {}

    # Scan once and return immediately when the matching pair is found.
    for index, value in enumerate(nums):
        complement = target - value
        if complement in seen:
            return [seen[complement], index]

        # Save the current value after checking so the same element is not reused.
        seen[value] = index

    # Return an empty answer when the input does not contain a valid pair.
    return []
```""",
            "difficulty": "medium",
            "learning_objective": "Turn a complement invariant into a one-pass solution.",
            "source_section": "Junior: Hash maps and frequency counting",
            "source_quote": "Use a hash map when the current value needs a previously seen complement.",
            "misconception_trap": "Keeping nested loops even after a constant-time lookup structure is available.",
            "reasoning_summary": "Identify the invariant first, then choose the structure that preserves it in one pass.",
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
        prompt_l = prompt.lower()
        if "expert algorithm interview coach and problem-solving educator" in prompt_l:
            payload = _problem_solving_payload(
                f"Question {self.calls}: solve two sum in python."
            )
        else:
            payload = _question_payload(f"Question {self.calls}: explain JVM tradeoffs.")
        return {
            "success": True,
            "analysis": json.dumps(payload),
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
                    "heading": "Junior: Hash maps and frequency counting",
                    "content": "Use a hash map when the current value needs a previously seen complement.",
                }
            ],
            raw_content=(
                "# Problem Solving and Algorithms (python)\n\n"
                "## Junior: Hash maps and frequency counting\n\n"
                "Use a hash map when the current value needs a previously seen complement."
            ),
            target_sections=120,
            source="llm",
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

    def test_problem_solving_stream_returns_structured_answer_and_code(self):
        status, events = self._stream_events(
            {
                "topic_id": "00-problem-solving-and-algorithms",
                "count": 1,
                "level": "mid",
                "preferred_language": "python",
            }
        )
        self.assertEqual(status, 200)
        question_event = next(event for event in events if event["type"] == "question")
        answer = question_event["question"]["answer"]
        self.assertIn("### Problem", answer)
        self.assertIn("### Solution Walkthrough", answer)
        self.assertIn("### Complexity", answer)
        self.assertIn("### Code", answer)
        self.assertIn("```python", answer)
        self.assertEqual(events[-1]["type"], "done")


if __name__ == "__main__":
    unittest.main()
