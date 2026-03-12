import asyncio
import json
import unittest

from app.services.question_generator import (
    QuestionGenerator,
    _problem_solving_pattern_signature,
)


def _question_payload(question: str) -> dict:
    return {
        "question": question,
        "answer": (
            f"**Answer:** {question} should be answered by starting with the real constraint and the direct takeaway.\n\n"
            "**Why it's right:** In plain language, the best answer connects the concept to the implementation detail that actually controls correctness or tradeoffs.\n\n"
            "**Interviewer-ready phrasing:** \"I would start with the constraint, explain the tradeoff it creates, and then tie that back to the implementation choice.\"\n\n"
            "**Common mistake:** A weak answer repeats definitions without showing why the constraint changes the decision.\n\n"
            "**Self-check:** If the main constraint changed, what part of the explanation would you revisit first?"
        ),
        "difficulty": "medium",
        "learning_objective": "After this question, the learner should be able to understand and apply the documented concept.",
        "source_section": "Core Concepts",
        "source_quote": "Core concepts define behavior and tradeoffs.",
        "misconception_trap": "Assuming a default without validating constraints.",
        "reasoning_summary": "Start from the controlling constraint, then explain the tradeoff it creates.",
        "target_level": "mid",
    }


def _problem_solving_payload(question: str, code: str) -> dict:
    return {
        "question": question,
        "answer": (
            "### Problem\n"
            "**What the interviewer is really testing:** Can you identify the invariant before you write code and use it to justify the data structure?\n\n"
            "### Solution Walkthrough\n"
            "**Short answer:** Use the invariant to choose the data structure that removes repeated work.\n\n"
            "**How to think about it:** Clarify inputs, outputs, and the state the next step needs.\n\n"
            "**Why it works:** The chosen structure preserves the invariant after every update, so the scan never has to restart.\n\n"
            "**Interviewer-ready phrasing:** \"I would state the invariant first, then explain how each update preserves it and why that gives the target complexity.\"\n\n"
            "**Common mistake:** A weak answer starts coding before proving why the invariant supports the approach.\n\n"
            "### Complexity\n"
            "**Time and space:** Time complexity is derived from the primary loop or operations, and space reflects the supporting data structures.\n\n"
            "**Tradeoff / scaling caveat:** The structure improves speed, but it may cost extra memory or depend on a stronger invariant.\n\n"
            f"### Code\n```python\n{code}\n```\n\n"
            "**Invariant note:** The update inside the loop is the part that keeps the invariant true for the next iteration."
        ),
        "difficulty": "medium",
        "learning_objective": "After this question, the learner should be able to practice interview-grade problem decomposition and implementation.",
        "source_section": "Problem Solving",
        "source_quote": "Use constraints and invariants to choose an efficient strategy.",
        "misconception_trap": "Jumping to code before validating constraints and edge cases.",
        "reasoning_summary": "State invariant first, then map it to efficient data structures.",
        "target_level": "mid",
    }


_TWO_SUM_CODE = """def two_sum(nums, target):
    # Store seen values to check complements in O(1).
    seen = {}
    # Scan once and return as soon as a valid pair is found.
    for index, value in enumerate(nums):
        complement = target - value
        if complement in seen:
            return [seen[complement], index]
        seen[value] = index
    return []
"""

_MERGE_INTERVALS_CODE = """def merge_intervals(intervals):
    # Sort by start so potential overlaps are adjacent.
    intervals.sort(key=lambda pair: pair[0])
    merged = []
    # Extend the latest merged interval when overlap exists.
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
            continue
        merged[-1][1] = max(merged[-1][1], end)
    return merged
"""


class FakeLLM:
    def __init__(self, analyses: list[str]):
        self._analyses = analyses
        self._idx = 0

    async def completion(self, prompt, llm_config=None, user_identity=None):
        if self._idx < len(self._analyses):
            analysis = self._analyses[self._idx]
        else:
            analysis = self._analyses[-1]
        self._idx += 1
        return {
            "success": True,
            "analysis": analysis,
            "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
            "error": "",
            "error_code": "",
        }


async def _collect_events(generator):
    events = []
    async for event in generator:
        events.append(event)
    return events


class QuestionGeneratorStreamingTests(unittest.TestCase):
    def test_generate_v2_stream_emits_incremental_questions_then_done(self):
        llm = FakeLLM(
            [
                json.dumps([_question_payload("What is idempotency in APIs?")]),
                json.dumps([_question_payload("How do you design retry-safe endpoints?")]),
                json.dumps([_question_payload("When should optimistic concurrency be used?")]),
            ]
        )
        generator = QuestionGenerator(llm)

        events = asyncio.run(
            _collect_events(
                generator.generate_v2_stream(
                    topic_id="topic-api",
                    topic_title="API Design",
                    doc_content="API docs content",
                    count=3,
                    level="mid",
                )
            )
        )

        self.assertEqual(events[0]["type"], "start")
        question_events = [e for e in events if e.get("type") == "question"]
        self.assertEqual(len(question_events), 3)
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["generated_count"], 3)

    def test_generate_v2_stream_filters_duplicates_and_recovers_full_count(self):
        duplicate = json.dumps([_question_payload("Explain consistency models in distributed systems.")])
        unique = json.dumps([_question_payload("When should quorum reads be preferred over leader-only reads?")])
        llm = FakeLLM([duplicate, duplicate, unique])
        generator = QuestionGenerator(llm)

        events = asyncio.run(
            _collect_events(
                generator.generate_v2_stream(
                    topic_id="topic-consistency",
                    topic_title="Distributed Systems",
                    doc_content="Distributed systems docs content",
                    count=2,
                    level="mid",
                )
            )
        )

        question_events = [e for e in events if e.get("type") == "question"]
        done_event = events[-1]
        self.assertEqual(done_event["type"], "done")
        self.assertEqual(len(question_events), 2)
        self.assertEqual(done_event["generated_count"], 2)
        self.assertGreater(done_event["retries_used"], 0)

    def test_generate_v2_fills_target_count_with_fallback_when_llm_unstructured(self):
        llm = FakeLLM(["unstructured output that is not json"])
        generator = QuestionGenerator(llm)

        result = asyncio.run(
            generator.generate_v2(
                topic_id="topic-backend",
                topic_title="Backend API Design",
                doc_content="Use clear API contracts, validation, and idempotency for robust backend APIs.",
                count=5,
                difficulty="medium",
                level="mid",
            )
        )

        self.assertEqual(len(result.questions), 5)
        self.assertTrue(all(q.question.strip() for q in result.questions))
        self.assertTrue(all(q.answer.strip() for q in result.questions))
        self.assertTrue(all("**Answer:**" in q.answer for q in result.questions))

    def test_generate_v2_stream_fills_target_count_with_fallback_when_llm_unstructured(self):
        llm = FakeLLM(["unstructured output that is not json"])
        generator = QuestionGenerator(llm)

        events = asyncio.run(
            _collect_events(
                generator.generate_v2_stream(
                    topic_id="topic-backend",
                    topic_title="Backend API Design",
                    doc_content="Use clear API contracts, validation, and idempotency for robust backend APIs.",
                    count=3,
                    level="mid",
                )
            )
        )

        question_events = [e for e in events if e.get("type") == "question"]
        done_event = events[-1]
        self.assertEqual(done_event["type"], "done")
        self.assertEqual(done_event["generated_count"], 3)
        self.assertEqual(len(question_events), 3)
        self.assertTrue(all("**Answer:**" in event["question"]["answer"] for event in question_events))

    def test_problem_solving_fallback_question_is_natural(self):
        llm = FakeLLM(["unstructured output that is not json"])
        generator = QuestionGenerator(llm)

        events = asyncio.run(
            _collect_events(
                generator.generate_v2_stream(
                    topic_id="00-problem-solving-and-algorithms",
                    topic_title="Problem Solving and Algorithms (python)",
                    doc_content="Two pointers, hash maps, and complexity analysis.",
                    count=1,
                    level="mid",
                    preferred_language="python",
                    requires_programming=True,
                )
            )
        )

        question_event = next(e for e in events if e.get("type") == "question")
        question = str(question_event["question"]["question"]).lower()
        answer = str(question_event["question"]["answer"])
        self.assertNotIn("walk through how you would solve this using", question)
        self.assertNotIn("reasoning under interview pressure", question)
        self.assertIn("**What the interviewer is really testing:**", answer)
        self.assertIn("**Invariant note:**", answer)

    def test_problem_solving_fallback_rotates_problem_patterns(self):
        llm = FakeLLM(["unstructured output that is not json"])
        generator = QuestionGenerator(llm)

        result = asyncio.run(
            generator.generate_v2(
                topic_id="00-problem-solving-and-algorithms",
                topic_title="Problem Solving and Algorithms (python)",
                doc_content="Sliding windows, intervals, heaps, graphs, and binary search.",
                count=5,
                difficulty="medium",
                level="mid",
                preferred_language="python",
                requires_programming=True,
            )
        )

        questions = [q.question.lower() for q in result.questions]
        unique_questions = set(questions)
        self.assertGreaterEqual(len(unique_questions), 4)
        self.assertTrue(any("substring" in q for q in questions))
        self.assertTrue(any("interval" in q for q in questions))
        self.assertTrue(any("rotated" in q or "island" in q or "frequent" in q for q in questions))

    def test_problem_solving_stream_rejects_repeated_two_sum_pattern(self):
        llm = FakeLLM(
            [
                json.dumps(
                    [
                        _problem_solving_payload(
                            "Given nums and target, return indices for the pair that sums to target.",
                            _TWO_SUM_CODE,
                        )
                    ]
                ),
                json.dumps(
                    [
                        _problem_solving_payload(
                            "Find two indices whose values add to target using a complement hash map.",
                            _TWO_SUM_CODE,
                        )
                    ]
                ),
                json.dumps(
                    [
                        _problem_solving_payload(
                            "Given intervals [start, end], merge overlapping intervals and return the result.",
                            _MERGE_INTERVALS_CODE,
                        )
                    ]
                ),
            ]
        )
        generator = QuestionGenerator(llm)

        events = asyncio.run(
            _collect_events(
                generator.generate_v2_stream(
                    topic_id="00-problem-solving-and-algorithms",
                    topic_title="Problem Solving and Algorithms (python)",
                    doc_content="Hash maps, sorting, and intervals.",
                    count=2,
                    level="mid",
                    preferred_language="python",
                    requires_programming=True,
                )
            )
        )

        question_events = [e for e in events if e.get("type") == "question"]
        self.assertEqual(len(question_events), 2)
        signatures = [
            _problem_solving_pattern_signature(
                str(event["question"]["question"]),
                str(event["question"]["answer"]),
            )
            for event in question_events
        ]
        self.assertEqual(len(set(signatures)), 2)
        self.assertEqual(events[-1]["type"], "done")
        self.assertGreater(events[-1]["retries_used"], 0)

    def test_problem_solving_generate_v2_filters_repeated_two_sum_pattern(self):
        llm = FakeLLM(
            [
                json.dumps(
                    [
                        _problem_solving_payload(
                            "Given nums and target, return indices for the pair that sums to target.",
                            _TWO_SUM_CODE,
                        ),
                        _problem_solving_payload(
                            "Find the two numbers that add to target and return their indices.",
                            _TWO_SUM_CODE,
                        ),
                        _problem_solving_payload(
                            "Return the index pair for nums that equals target using hash lookup.",
                            _TWO_SUM_CODE,
                        ),
                    ]
                )
            ]
        )
        generator = QuestionGenerator(llm)

        result = asyncio.run(
            generator.generate_v2(
                topic_id="00-problem-solving-and-algorithms",
                topic_title="Problem Solving and Algorithms (python)",
                doc_content="Hash maps, sliding windows, intervals, heaps, and graph traversal.",
                count=3,
                difficulty="medium",
                level="mid",
                preferred_language="python",
                requires_programming=True,
            )
        )

        signatures = [
            _problem_solving_pattern_signature(question.question, question.answer)
            for question in result.questions
        ]
        self.assertEqual(len(result.questions), 3)
        self.assertEqual(len(set(signatures)), 3)


if __name__ == "__main__":
    unittest.main()
