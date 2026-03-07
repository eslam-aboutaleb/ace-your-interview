import asyncio
import json
import unittest

from app.services.question_generator import (
    _build_prompt,
    _build_quiz_prompt,
    _collect_with_retries,
    _parse_questions_json,
    _validate_question_item,
    _validate_quiz_item,
)


class QuestionGeneratorValidationTests(unittest.TestCase):
    def test_question_item_requires_grounding_fields(self):
        item = {
            "question": "What does this service do?",
            "answer": "It handles user sessions and validates signed cookies for auth.",
            "difficulty": "medium",
            "learning_objective": "Understand auth session responsibilities.",
            "source_section": "Auth",
            "source_quote": "Validate the session cookie and return user payload.",
            "misconception_trap": "Assuming session validation is done in frontend.",
            "reasoning_summary": "Auth must be enforced server-side per request.",
            "target_level": "mid",
        }
        valid, issue = _validate_question_item(item, "topic-auth", "medium", "mid")
        self.assertTrue(valid, issue)
        self.assertEqual(item["topic_id"], "topic-auth")

    def test_quiz_item_strict_choice_rules(self):
        item = {
            "question": "Which option is correct?",
            "type": "mcq",
            "choices": [
                {"label": "A", "text": "One"},
                {"label": "B", "text": "Two"},
                {"label": "C", "text": "Three"},
                {"label": "D", "text": "Four"},
            ],
            "correct_answer": "C",
            "explanation": "Only one option matches the documented behavior exactly.",
            "difficulty": "hard",
            "topic_id": "topic-1",
            "source_quote": "Only one branch performs strict validation.",
            "reasoning_summary": "Read the decision path and eliminate incorrect branches.",
            "target_level": "senior",
        }
        valid, issue = _validate_quiz_item(
            item,
            allowed_topics={"topic-1"},
            allowed_types={"mcq", "true_false"},
            difficulty="hard",
            level="senior",
        )
        self.assertTrue(valid, issue)

        broken = dict(item)
        broken["choices"] = [
            {"label": "A", "text": "True"},
            {"label": "B", "text": "False"},
        ]
        broken["type"] = "mcq"
        valid2, issue2 = _validate_quiz_item(
            broken,
            allowed_topics={"topic-1"},
            allowed_types={"mcq", "true_false"},
            difficulty="hard",
            level="senior",
        )
        self.assertFalse(valid2)
        self.assertEqual(issue2, "mcq_labels_must_be_abcd")

    def test_question_level_and_difficulty_are_coerced_for_reliability(self):
        item = {
            "question": "How would you design retries for this API?",
            "answer": "Use bounded retries with backoff and jitter while keeping idempotency guarantees and observability metrics.",
            "difficulty": "hard",
            "target_level": "senior",
        }
        valid, issue = _validate_question_item(item, "topic-reliability", "medium", "junior")
        self.assertTrue(valid, issue)
        self.assertEqual(item["difficulty"], "medium")
        self.assertEqual(item["target_level"], "junior")
        self.assertTrue(item["learning_objective"])
        self.assertTrue(item["source_section"])
        self.assertTrue(item["source_quote"])
        self.assertTrue(item["misconception_trap"])
        self.assertTrue(item["reasoning_summary"])

    def test_collect_with_retries_respects_existing_questions_seed(self):
        class FakeLLM:
            def __init__(self):
                self.calls = 0

            async def completion(self, prompt, llm_config=None, user_identity=None):
                self.calls += 1
                if self.calls == 1:
                    payload = [
                        {
                            "question": "How would you design retries for this API?",
                            "answer": "Use bounded retries with backoff and jitter while keeping idempotency guarantees and observability metrics.",
                            "difficulty": "medium",
                            "learning_objective": "Evaluate reliability reasoning.",
                            "source_section": "Retries",
                            "source_quote": "Retries must be bounded to avoid overload.",
                            "misconception_trap": "Assuming retries are always safe.",
                            "reasoning_summary": "Retries need limits and idempotency.",
                            "target_level": "mid",
                        }
                    ]
                else:
                    payload = [
                        {
                            "question": "When should idempotency keys be required?",
                            "answer": "Require idempotency keys for externally retried write operations to prevent duplicate side effects and support safe retries under network uncertainty.",
                            "difficulty": "medium",
                            "learning_objective": "Identify safe retry boundaries.",
                            "source_section": "Reliability",
                            "source_quote": "Idempotency prevents duplicate mutation effects.",
                            "misconception_trap": "Assuming retries are harmless without idempotency.",
                            "reasoning_summary": "Map retries to mutation safety guarantees.",
                            "target_level": "mid",
                        }
                    ]
                return {
                    "success": True,
                    "analysis": json.dumps(payload),
                    "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
                    "error": "",
                }

        prompt = _build_prompt(
            topic_id="topic-reliability",
            topic_title="Reliability",
            doc_content="Retries and idempotency guidance.",
            count=1,
            level="mid",
            requested_total_count=5,
            existing_questions=["How would you design retries for this API?"],
        )
        validator = lambda item: _validate_question_item(
            item, "topic-reliability", "medium", "mid"
        )
        items, _stats = asyncio.run(
            _collect_with_retries(
                llm=FakeLLM(),
                base_prompt=prompt,
                target_count=1,
                llm_config=None,
                validator=validator,
                existing_questions=["How would you design retries for this API?"],
            )
        )
        self.assertEqual(len(items), 1)
        self.assertIn("idempotency", items[0]["question"].lower())

    def test_collect_with_retries_filters_near_duplicate_paraphrases(self):
        class FakeLLM:
            def __init__(self):
                self.calls = 0

            async def completion(self, prompt, llm_config=None, user_identity=None):
                self.calls += 1
                if self.calls == 1:
                    payload = [
                        {
                            "question": "How do you design retry-safe API endpoints?",
                            "answer": "Use idempotency and bounded retries with backoff.",
                            "difficulty": "medium",
                        },
                        {
                            "question": "How would you design API endpoints that are safe for retries?",
                            "answer": "Enforce idempotency keys and avoid duplicate side effects.",
                            "difficulty": "medium",
                        },
                    ]
                else:
                    payload = [
                        {
                            "question": "When should idempotency keys be mandatory for writes?",
                            "answer": "They should be required for externally retried mutations to avoid duplicate effects.",
                            "difficulty": "medium",
                        }
                    ]
                return {
                    "success": True,
                    "analysis": json.dumps(payload),
                    "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
                    "error": "",
                }

        prompt = _build_prompt(
            topic_id="topic-reliability",
            topic_title="Reliability",
            doc_content="Retries and idempotency guidance.",
            count=2,
            level="mid",
        )
        validator = lambda item: _validate_question_item(
            item, "topic-reliability", "medium", "mid"
        )
        items, _stats = asyncio.run(
            _collect_with_retries(
                llm=FakeLLM(),
                base_prompt=prompt,
                target_count=2,
                llm_config=None,
                validator=validator,
                existing_questions=[],
            )
        )
        self.assertEqual(len(items), 2)
        retry_like = [item for item in items if "retry" in item["question"].lower()]
        self.assertLessEqual(len(retry_like), 1)
        combined = " ".join(item["question"].lower() for item in items)
        self.assertIn("idempotency", combined)

    def test_parse_questions_json_accepts_single_object_shape(self):
        parsed = _parse_questions_json(
            json.dumps(
                {
                    "question": "What is an API contract?",
                    "answer": "An API contract defines request/response behavior and guarantees.",
                    "difficulty": "easy",
                }
            )
        )
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["question"], "What is an API contract?")

    def test_parse_questions_json_recovers_from_plaintext_question_answer(self):
        parsed = _parse_questions_json(
            """
Question 1: What is idempotency in backend APIs?
Answer 1: Idempotency means repeating the same request results in the same side effect profile,
which is essential for safe retries on network failures.

Question 2: Why use pagination for list endpoints?
Answer 2: Pagination controls payload size, improves latency, and avoids memory pressure for clients and servers.
"""
        )
        self.assertEqual(len(parsed), 2)
        self.assertIn("idempotency", parsed[0]["question"].lower())
        self.assertIn("pagination", parsed[1]["question"].lower())

    def test_question_prompt_requires_adaptive_markdown_answers(self):
        prompt = _build_prompt(
            topic_id="topic-api",
            topic_title="API Design",
            doc_content="Use pagination and stable ordering for list endpoints.",
            count=3,
            level="mid",
        )
        self.assertIn("short readable paragraphs", prompt)
        self.assertIn("use bullets only when listing steps/checklists/categories", prompt)
        self.assertIn("valid GFM table syntax", prompt)
        self.assertIn("fenced code blocks", prompt)
        self.assertIn("fenced Mermaid diagrams", prompt)
        self.assertIn("valid JSON, escape newlines", prompt)
        self.assertIn("Return exactly 3 items", prompt)

    def test_quiz_prompt_requires_adaptive_markdown_explanations(self):
        prompt = _build_quiz_prompt(
            topics_content=[{"id": "api", "title": "API Design", "content": "Use ETags and cache control."}],
            count=3,
            question_types=["mcq"],
            level="mid",
        )
        self.assertIn("short readable paragraphs", prompt)
        self.assertIn("use bullets only when listing steps/checklists/categories", prompt)
        self.assertIn("valid GFM table syntax", prompt)
        self.assertIn("fenced code blocks", prompt)
        self.assertIn("fenced Mermaid diagrams", prompt)
        self.assertIn("valid JSON, escape newlines", prompt)
        self.assertIn("Return exactly 3 items", prompt)


if __name__ == "__main__":
    unittest.main()
