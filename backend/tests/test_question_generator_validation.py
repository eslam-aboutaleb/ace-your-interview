import asyncio
import json
import unittest

from app.services.question_generator import (
    _build_fallback_question_items,
    _build_prompt,
    _build_quiz_prompt,
    _build_retry_prompt,
    _collect_with_retries,
    _parse_questions_json,
    _validate_question_item,
    _validate_quiz_item,
)


def _conceptual_answer(topic: str = "this service") -> str:
    return (
        f"**Answer:** {topic.title()} works by handling the core request path directly and enforcing the important constraints.\n\n"
        f"**Why it's right:** In plain language, {topic} stays reliable when the system validates inputs, preserves invariants, and checks the failure cases that matter.\n\n"
        f"**Interviewer-ready phrasing:** \"I would explain {topic} by starting with the request flow, then I would connect the implementation to the constraints and failure modes.\"\n\n"
        f"**Common mistake:** A weak answer about {topic} names the components but never explains why they fit the requirements.\n\n"
        f"**Self-check:** If the main constraint changed tomorrow, what part of your explanation would you revisit first?"
    )


def _problem_solving_answer(language: str = "python") -> str:
    return f"""### Problem

**What the interviewer is really testing:** Can you name the invariant before coding and use it to justify the data structure?

### Solution Walkthrough

**Short answer:** Use a hash map so each complement lookup stays O(1) during a single left-to-right pass.

**How to think about it:** Translate the prompt into a complement lookup problem and reject the nested-loop version once the invariant is clear.

**Why it works:** Each number only needs to know whether its complement has already appeared, so the map preserves exactly the state the next step needs.

**Interviewer-ready phrasing:** "I would state the complement invariant first, then use a hash map to preserve that invariant while scanning once from left to right."

**Common mistake:** A weak answer starts coding the O(n^2) scan before explaining why the invariant allows a one-pass lookup.

### Complexity

**Time and space:** The solution runs in O(n) time with O(n) extra space.

**Tradeoff / scaling caveat:** The extra memory is worth it because it removes repeated work and keeps the explanation interview-friendly.

### Code

```{language}
def two_sum(nums, target):
    # Store each seen value so complement lookups stay O(1).
    seen = {{}}

    # Scan once and return as soon as the matching pair is found.
    for index, value in enumerate(nums):
        complement = target - value
        if complement in seen:
            return [seen[complement], index]

        # Save the current value after checking so the same element is not reused.
        seen[value] = index

    # Return an empty answer when no pair exists.
    return []
```

**Invariant note:** Checking the complement before storing the current value preserves the rule that every match must come from a previously seen element.
"""


class QuestionGeneratorValidationTests(unittest.TestCase):
    def test_question_item_requires_grounding_fields(self):
        item = {
            "question": "What does this service do?",
            "answer": _conceptual_answer("the auth service"),
            "difficulty": "medium",
            "learning_objective": "After this question, the learner should be able to explain auth session responsibilities.",
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
            "answer": _conceptual_answer("retry-safe APIs"),
            "difficulty": "hard",
            "target_level": "senior",
        }
        valid, issue = _validate_question_item(item, "topic-reliability", "medium", "junior")
        self.assertTrue(valid, issue)
        self.assertEqual(item["difficulty"], "medium")
        self.assertEqual(item["target_level"], "junior")
        self.assertTrue(
            item["learning_objective"].startswith(
                "After this question, the learner should be able to",
            ),
        )
        self.assertTrue(item["source_section"])
        self.assertTrue(item["source_quote"])
        self.assertIn("weak answer", item["misconception_trap"].lower())
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
                            "answer": _conceptual_answer("retry-safe APIs"),
                            "difficulty": "medium",
                            "learning_objective": "After this question, the learner should be able to evaluate reliability reasoning.",
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
                            "answer": _conceptual_answer("idempotency keys"),
                            "difficulty": "medium",
                            "learning_objective": "After this question, the learner should be able to identify safe retry boundaries.",
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
                            "answer": _conceptual_answer("retry-safe API endpoints"),
                            "difficulty": "medium",
                        },
                        {
                            "question": "How would you design API endpoints that are safe for retries?",
                            "answer": _conceptual_answer("safe retry APIs"),
                            "difficulty": "medium",
                        },
                    ]
                else:
                    payload = [
                        {
                            "question": "When should idempotency keys be mandatory for writes?",
                            "answer": _conceptual_answer("mandatory idempotency keys"),
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
        self.assertIn("**Answer:**", prompt)
        self.assertIn("**Why it's right:**", prompt)
        self.assertIn("**Interviewer-ready phrasing:**", prompt)
        self.assertIn("**Common mistake:**", prompt)
        self.assertIn("**Self-check:**", prompt)
        self.assertIn("Do not use meta-guideline phrasing", prompt)
        self.assertIn(
            "After this question, the learner should be able to",
            prompt,
        )
        self.assertIn("short readable paragraphs", prompt)
        self.assertIn("use bullets only when listing steps/checklists/categories", prompt)
        self.assertIn("valid GFM table syntax", prompt)
        self.assertIn("fenced code blocks", prompt)
        self.assertIn("fenced Mermaid diagrams", prompt)
        self.assertIn("valid JSON, escape newlines", prompt)
        self.assertIn("Return exactly 3 items", prompt)

    def test_problem_solving_prompt_requires_structured_answer_and_selected_language_code(self):
        prompt = _build_prompt(
            topic_id="00-problem-solving-and-algorithms",
            topic_title="Problem Solving and Algorithms (python)",
            doc_content="Use two pointers when the invariant moves from both sides.",
            count=2,
            level="mid",
            preferred_language="python",
            requires_programming=True,
        )
        self.assertIn("### Problem", prompt)
        self.assertIn("### Solution Walkthrough", prompt)
        self.assertIn("### Complexity", prompt)
        self.assertIn("### Code", prompt)
        self.assertIn("**What the interviewer is really testing:**", prompt)
        self.assertIn("**Short answer:**", prompt)
        self.assertIn("**How to think about it:**", prompt)
        self.assertIn("**Why it works:**", prompt)
        self.assertIn("**Tradeoff / scaling caveat:**", prompt)
        self.assertIn("**Invariant note:**", prompt)
        self.assertIn('exactly one fenced "python"', prompt)
        self.assertNotIn("Two Sum", prompt)
        self.assertNotIn("complement-hash-map", prompt)

    def test_problem_solving_validation_rejects_missing_required_headings(self):
        item = {
            "question": "Solve two sum in python.",
            "answer": "Use a hash map and then return the answer.",
            "difficulty": "medium",
            "reasoning_summary": "Spot the complement invariant first.",
        }
        valid, issue = _validate_question_item(
            item,
            "00-problem-solving-and-algorithms",
            "medium",
            "mid",
            preferred_language="python",
            requires_programming=True,
        )
        self.assertFalse(valid)
        self.assertEqual(issue, "invalid_problem_solving_heading_sequence")

    def test_problem_solving_validation_rejects_missing_code_fence(self):
        item = {
            "question": "Solve two sum in python.",
            "answer": """### Problem

Read the prompt carefully.

### Solution Walkthrough

Use a hash map to store seen values.

### Complexity

This takes O(n) time and O(n) space.

### Code

Use a dictionary and a loop.""",
            "difficulty": "medium",
            "reasoning_summary": "Spot the complement invariant first.",
        }
        valid, issue = _validate_question_item(
            item,
            "00-problem-solving-and-algorithms",
            "medium",
            "mid",
            preferred_language="python",
            requires_programming=True,
        )
        self.assertFalse(valid)
        self.assertEqual(issue, "invalid_problem_solving_code_block_count")

    def test_problem_solving_validation_rejects_wrong_code_language(self):
        item = {
            "question": "Solve two sum in python.",
            "answer": _problem_solving_answer(language="javascript"),
            "difficulty": "medium",
            "reasoning_summary": "Spot the complement invariant first.",
        }
        valid, issue = _validate_question_item(
            item,
            "00-problem-solving-and-algorithms",
            "medium",
            "mid",
            preferred_language="python",
            requires_programming=True,
        )
        self.assertFalse(valid)
        self.assertEqual(issue, "problem_solving_code_language_mismatch")

    def test_problem_solving_fallback_items_include_structured_answer_and_code(self):
        items = _build_fallback_question_items(
            topic_id="00-problem-solving-and-algorithms",
            topic_title="Problem Solving and Algorithms (python)",
            doc_content="Use a hash map or two pointers when the constraint suggests a faster lookup.",
            count=1,
            difficulty="medium",
            level="mid",
            section_title="Junior: Hash maps and frequency counting",
            section_content="Use a hash map when the current value needs a previously seen complement.",
            preferred_language="python",
            requires_programming=True,
        )
        self.assertEqual(len(items), 1)
        self.assertIn("### Problem", items[0]["answer"])
        self.assertIn("### Solution Walkthrough", items[0]["answer"])
        self.assertIn("### Complexity", items[0]["answer"])
        self.assertIn("### Code", items[0]["answer"])
        self.assertIn("**What the interviewer is really testing:**", items[0]["answer"])
        self.assertIn("**Short answer:**", items[0]["answer"])
        self.assertIn("**How to think about it:**", items[0]["answer"])
        self.assertIn("**Why it works:**", items[0]["answer"])
        self.assertIn("**Tradeoff / scaling caveat:**", items[0]["answer"])
        self.assertIn("**Invariant note:**", items[0]["answer"])
        self.assertIn("```python", items[0]["answer"])
        self.assertTrue(
            items[0]["learning_objective"].startswith(
                "After this question, the learner should be able to",
            ),
        )
        self.assertTrue(items[0]["reasoning_summary"])

    def test_conceptual_fallback_items_include_coach_sections(self):
        items = _build_fallback_question_items(
            topic_id="topic-api",
            topic_title="API Design",
            doc_content="Use pagination and stable ordering for list endpoints.",
            count=1,
            difficulty="medium",
            level="mid",
        )
        self.assertEqual(len(items), 1)
        self.assertIn("**Answer:**", items[0]["answer"])
        self.assertIn("**Why it's right:**", items[0]["answer"])
        self.assertIn("**Interviewer-ready phrasing:**", items[0]["answer"])
        self.assertIn("**Common mistake:**", items[0]["answer"])
        self.assertIn("**Self-check:**", items[0]["answer"])
        self.assertTrue(
            items[0]["learning_objective"].startswith(
                "After this question, the learner should be able to",
            ),
        )
        self.assertIn("weak answer", items[0]["misconception_trap"].lower())

    def test_retry_prompt_reinforces_coach_style_recovery(self):
        conceptual_retry = _build_retry_prompt(
            base_prompt="Base prompt",
            missing_count=1,
            issues="json_parse_failed",
            existing_questions=[],
            recovery_guidance=(
                "- Use `**Answer:**`.\n"
                "- Use `**Why it's right:**`.\n"
                "- Use `**Interviewer-ready phrasing:**`."
            ),
        )
        self.assertIn("Recovery guidance:", conceptual_retry)
        self.assertIn("**Answer:**", conceptual_retry)
        self.assertIn("**Interviewer-ready phrasing:**", conceptual_retry)

        problem_retry = _build_retry_prompt(
            base_prompt="Base prompt",
            missing_count=1,
            issues="invalid_problem_solving_heading_sequence",
            existing_questions=[],
            hard_requirements=['Under `### Code`, include exactly one fenced `python` block.'],
            recovery_guidance=(
                "- Keep exactly these H3 headings in order: `### Problem`, `### Solution Walkthrough`, `### Complexity`, `### Code`.\n"
                "- Include one practice twist.\n"
                "- End with `**Invariant note:**`."
            ),
        )
        self.assertIn("### Problem", problem_retry)
        self.assertIn("practice twist", problem_retry)
        self.assertIn("**Invariant note:**", problem_retry)

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
