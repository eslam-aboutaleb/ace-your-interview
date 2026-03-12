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
        f"**Detailed explanation:** In plain language, {topic} stays reliable when the system validates inputs, preserves invariants, and checks the failure cases that matter.\n\n"
        f"**Common mistake:** A weak answer about {topic} names the components but never explains why they fit the requirements."
    )


def _problem_solving_answer(language: str = "python") -> str:
    return f"""### Problem

**What the problem is asking:** Return the two indices whose values add up to the target while avoiding repeated work and reusing the same element.

### Solution Walkthrough

**Direct answer:** Use a hash map so each complement lookup stays O(1) during a single left-to-right pass.

**Detailed explanation:** For each number, compute the complement needed to reach the target. If that complement has already been seen, return the saved index and the current index. Otherwise, store the current value and continue scanning.

**Why this works:** Each number only needs to know whether its complement has already appeared, so the map preserves exactly the state the next step needs.

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
        self.assertTrue(item["reasoning_summary"])

    def test_question_item_rejects_generic_question_when_not_grounded_to_topic(self):
        item = {
            "question": "How would you design retries in a distributed system?",
            "answer": _conceptual_answer("distributed retries"),
            "difficulty": "medium",
            "learning_objective": "After this question, the learner should be able to explain topic-specific engineering tradeoffs.",
            "source_section": "Caching",
            "source_quote": "Cache invalidation and TTL shape correctness.",
            "reasoning_summary": "Tie the answer to the actual topic mechanism, not a generic pattern.",
            "target_level": "mid",
        }
        valid, issue = _validate_question_item(
            item,
            "topic-caching",
            "medium",
            "mid",
            topic_title="Caching",
            grounding_anchors=[
                "Cache invalidation",
                "TTL expiration",
                "Read-through caching",
            ],
        )
        self.assertFalse(valid)
        self.assertEqual(issue, "question_not_grounded_to_topic")

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
        self.assertIn("**Detailed explanation:**", prompt)
        self.assertIn("**Common mistake:**", prompt)
        self.assertIn("Answer the actual question itself", prompt)
        self.assertIn(
            "After this question, the learner should be able to",
            prompt,
        )
        self.assertIn("common interview questions", prompt)
        self.assertIn("how it works, when to use it, tradeoffs", prompt)
        self.assertIn("keep the batch naturally progressive", prompt)
        self.assertIn("Grounding anchors from the topic content", prompt)
        self.assertIn("Each question must be explicitly grounded to the current topic", prompt)
        self.assertIn("Do not ask generic interview questions that could fit many unrelated topics", prompt)
        self.assertIn("current topic recognizable", prompt)
        self.assertIn("short readable paragraphs", prompt)
        self.assertIn("Do not compress complex topics into a few generic sentences", prompt)
        self.assertIn(
            "use numbered lists instead of dense prose",
            prompt,
        )
        self.assertIn(
            "Use bullet lists for components, pros/cons, bottlenecks, failure modes",
            prompt,
        )
        self.assertIn(
            "If the question is mainly asking for a comparison, prefer a compact GFM table",
            prompt,
        )
        self.assertIn("valid GFM table syntax", prompt)
        self.assertIn("relevant consistency/CAP implications", prompt)
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
        self.assertIn("**What the problem is asking:**", prompt)
        self.assertIn("**Direct answer:**", prompt)
        self.assertIn("**Detailed explanation:**", prompt)
        self.assertIn("**Why this works:**", prompt)
        self.assertIn("**Common mistake:**", prompt)
        self.assertIn("**Tradeoff / scaling caveat:**", prompt)
        self.assertIn("**Invariant note:**", prompt)
        self.assertIn("common coding interview questions", prompt)
        self.assertIn("keep the batch naturally progressive", prompt)
        self.assertIn("Prefer high-frequency interviewer prompts first", prompt)
        self.assertIn('exactly one fenced "python"', prompt)
        self.assertNotIn("Two Sum", prompt)
        self.assertNotIn("complement-hash-map", prompt)

    def test_conceptual_programming_prompt_defaults_code_examples_to_python(self):
        prompt = _build_prompt(
            topic_id="topic-rate-limiting",
            topic_title="Rate Limiting",
            doc_content="Token bucket and sliding window counters protect services from bursts.",
            count=1,
            level="mid",
            requires_programming=True,
        )
        self.assertIn('fenced "python" example', prompt)

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
        self.assertIn("**What the problem is asking:**", items[0]["answer"])
        self.assertIn("**Direct answer:**", items[0]["answer"])
        self.assertIn("**Detailed explanation:**", items[0]["answer"])
        self.assertIn("**Why this works:**", items[0]["answer"])
        self.assertIn("**Tradeoff / scaling caveat:**", items[0]["answer"])
        self.assertIn("**Invariant note:**", items[0]["answer"])
        self.assertIn("| Approach |", items[0]["answer"])
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
        self.assertTrue(
            items[0]["question"].startswith(("How would you", "When would you", "If ")),
        )
        self.assertIn("**Answer:**", items[0]["answer"])
        self.assertIn("**Detailed explanation:**", items[0]["answer"])
        self.assertIn("**Common mistake:**", items[0]["answer"])
        self.assertIn("| Concern | What matters for this topic |", items[0]["answer"])
        self.assertTrue(
            items[0]["learning_objective"].startswith(
                "After this question, the learner should be able to",
            ),
        )

    def test_retry_prompt_reinforces_coach_style_recovery(self):
        conceptual_retry = _build_retry_prompt(
            base_prompt="Base prompt",
            missing_count=1,
            issues="json_parse_failed",
            existing_questions=[],
            recovery_guidance=(
                "- Use `**Answer:**`.\n"
                "- Use `**Detailed explanation:**`.\n"
                "- Use `**Common mistake:**`."
            ),
        )
        self.assertIn("Recovery guidance:", conceptual_retry)
        self.assertIn("**Answer:**", conceptual_retry)
        self.assertIn("**Common mistake:**", conceptual_retry)

        problem_retry = _build_retry_prompt(
            base_prompt="Base prompt",
            missing_count=1,
            issues="invalid_problem_solving_heading_sequence",
            existing_questions=[],
            hard_requirements=['Under `### Code`, include exactly one fenced `python` block.'],
            recovery_guidance=(
                "- Keep exactly these H3 headings in order: `### Problem`, `### Solution Walkthrough`, `### Complexity`, `### Code`.\n"
                "- Start with `**What the problem is asking:**`.\n"
                "- End with `**Invariant note:**`."
            ),
        )
        self.assertIn("### Problem", problem_retry)
        self.assertIn("What the problem is asking", problem_retry)
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
