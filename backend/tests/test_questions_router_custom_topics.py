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
        elif "expert algorithm interviewer and problem-solving educator" in prompt_l:
            payload = [
                {
                    "question": "Given an integer array and a target, how would you solve two sum in python?",
                    "answer": """### Problem

**What the problem is asking:** Return the two indices whose values add up to the target without reusing the same element or doing repeated work.

### Solution Walkthrough

**Direct answer:** Reject the O(n^2) brute-force scan and use a hash map to preserve the complement invariant in one pass.

**Detailed explanation:** Each value only needs one thing from earlier in the scan: whether the complement has already appeared. That makes a hash map the right structure because it stores seen values with their indices and answers that lookup in O(1).

**Why this works:** Every number only needs a previously seen complement. The hash map stores each seen value with its index, so every step can compute the complement, check for it in O(1), and then store the current value after the check.

**Common mistake:** A weak answer writes nested loops before proving why the invariant allows a one-pass lookup.

### Complexity

**Time and space:** The loop visits each element once, so the time complexity is O(n). The hash map stores up to n values, so the extra space complexity is O(n).

**Tradeoff / scaling caveat:** The extra memory is the cost of removing repeated work and keeping the reasoning clean.

### Code

```python
def two_sum(nums, target):
    # Store each seen value so complement lookups stay O(1).
    seen = {}

    # Scan once and return as soon as the matching pair is found.
    for index, value in enumerate(nums):
        complement = target - value
        if complement in seen:
            return [seen[complement], index]

        # Save the current value after checking so the same element is not reused.
        seen[value] = index

    # Return an empty answer when the input does not contain a valid pair.
    return []
```

**Invariant note:** Checking the complement before storing the current value preserves the rule that every match must come from a previously seen element.""",
                    "difficulty": "medium",
                    "learning_objective": "After this question, the learner should be able to explain how to turn a complement invariant into a one-pass hash map solution.",
                    "source_section": "Junior: Two pointers",
                    "source_quote": "Understand constraints and sliding window tradeoffs.",
                    "misconception_trap": "Writing nested loops before checking whether a faster lookup structure removes repeated work.",
                    "reasoning_summary": "Identify the invariant first: every value needs a previously seen complement. That immediately points to a one-pass hash map.",
                    "target_level": "mid",
                }
            ]
        else:
            payload = [
                {
                    "question": "How would you explain the JVM memory model tradeoffs in an interview?",
                    "answer": (
                        "**Answer:** The JVM memory model tradeoffs are easiest to explain by starting with stack vs heap responsibilities, then tying that to visibility, synchronization, and runtime cost.\n\n"
                        "**Detailed explanation:** In plain language, memory layout affects correctness first and performance second. If you explain where data lives, who can see it, and what coordination it needs, the tradeoffs become concrete.\n\n"
                        "**Common mistake:** A weak answer assumes garbage collection removes the need to reason about concurrency or visibility."
                    ),
                    "difficulty": "medium",
                    "learning_objective": "After this question, the learner should be able to explain memory model fundamentals and interview tradeoffs.",
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
        answer = payload["questions"][0]["answer"]
        self.assertIn("### Problem", answer)
        self.assertIn("### Solution Walkthrough", answer)
        self.assertIn("### Complexity", answer)
        self.assertIn("### Code", answer)
        self.assertIn("```python", answer)

    def test_problem_solving_topic_legacy_generate_preserves_structured_answer(self):
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
            "/api/questions/generate",
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
        answer = payload["questions"][0]["answer"]
        self.assertIn("### Problem", answer)
        self.assertIn("```python", answer)


if __name__ == "__main__":
    unittest.main()
