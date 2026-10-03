"""Hint service tests — level escalation, cost cap, no-answer-leak."""

import asyncio
import os
import tempfile
import unittest

from app.services.hint_service import (
    HINT_COST_CAP_PER_QUESTION,
    HintCostExceededError,
    HintService,
    apply_hint_penalty,
    independent_reasoning_score,
)


class LevelAwareLLM:
    """Return level-appropriate hint text by inspecting the prompt."""

    def __init__(self, analysis: str = ""):
        self.analysis = analysis
        self.calls = 0

    async def completion(self, prompt, llm_config=None, user_identity=None, **kwargs):
        self.calls += 1
        if self.analysis:
            return self._success(self.analysis)
        if "level-1" in prompt:
            return self._success("Consider what the question is really asking for.")
        if "level-2" in prompt:
            return self._success("Name the core concept that links the inputs to the outputs.")
        if "level-3" in prompt:
            return self._success(
                "Start by naming the key constraint, then outline the first concrete step."
            )
        return self._success(
            "Walk through the problem step by step: restate the goal and constraints, "
            "name the core concept, apply it to the given inputs, and finish with the "
            "complete answer and why it holds."
        )

    @staticmethod
    def _success(analysis: str) -> dict:
        return {
            "success": True,
            "analysis": analysis,
            "metadata": {},
            "error_code": "",
            "finish_reason": "stop",
            "usage": {},
        }


QUESTION_TEXT = "What is the time complexity of binary search?"
ANSWER_CONTEXT = (
    "Binary search runs in O(log n) time because it halves the "
    "search space each step."
)
ANSWER_SUBSTRING = "O(log n)"


class HintServiceTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tempdir.name, "learning.db")
        self.llm = LevelAwareLLM()
        self.service = HintService(self.llm, self.db_path)

    def tearDown(self):
        self.tempdir.cleanup()

    def _hint(self, level: int, question_id: str = "q1") -> dict:
        return asyncio.run(
            self.service.generate_hint(
                user_id="u1",
                question_id=question_id,
                level=level,
                question_text=QUESTION_TEXT,
                answer_context=ANSWER_CONTEXT,
            )
        )

    def test_level_escalation_produces_distinct_hints(self):
        hints = [self._hint(level)["hint"] for level in (1, 2, 3, 4)]
        self.assertEqual(len(set(hints)), 4)

    def test_level_four_contains_walkthrough(self):
        payload = self._hint(4)
        self.assertEqual(payload["level"], 4)
        self.assertIn("walk through", payload["hint"].lower())
        # The walkthrough includes the complete answer material.
        self.assertIn("complete answer", payload["hint"].lower())

    def test_levels_one_and_two_never_leak_the_answer(self):
        for level in (1, 2):
            payload = self._hint(level)
            self.assertNotIn(ANSWER_SUBSTRING, payload["hint"])
            self.assertNotIn("halves the search space", payload["hint"])

    def test_leaking_llm_output_is_replaced_by_fallback(self):
        # The LLM returns the full answer as a level-1 hint; the leak
        # guard must swap it for the static nudge.
        self.llm.analysis = (
            "Binary search runs in O(log n) time because it halves "
            "the search space each step."
        )
        payload = self._hint(1)
        self.assertNotIn(ANSWER_SUBSTRING, payload["hint"])
        self.assertNotIn("halves", payload["hint"])
        self.assertNotIn("O(log n)", payload["hint"])

    def test_cost_cap_enforced_per_question(self):
        for _ in range(HINT_COST_CAP_PER_QUESTION):
            self._hint(1)
        with self.assertRaises(HintCostExceededError) as ctx:
            self._hint(1)
        self.assertEqual(ctx.exception.used, HINT_COST_CAP_PER_QUESTION)
        self.assertEqual(ctx.exception.cap, HINT_COST_CAP_PER_QUESTION)

    def test_cost_cap_is_per_question(self):
        self._hint(1, question_id="q1")
        self._hint(1, question_id="q1")
        # A different question has its own budget.
        payload = self._hint(1, question_id="q2")
        self.assertEqual(payload["hints_used"], 1)

    def test_usage_counters_are_reported(self):
        payload = self._hint(2)
        self.assertEqual(payload["hints_used"], 1)
        self.assertEqual(payload["hints_remaining"], HINT_COST_CAP_PER_QUESTION - 1)

    def test_invalid_level_rejected(self):
        with self.assertRaises(ValueError):
            asyncio.run(
                self.service.generate_hint(
                    user_id="u1",
                    question_id="q1",
                    level=5,
                    question_text=QUESTION_TEXT,
                )
            )

    def test_count_hints_used_persists(self):
        self._hint(1)
        self._hint(2)
        self.assertEqual(self.service.count_hints_used("u1", "q1"), 2)
        self.assertEqual(self.service.count_hints_used("u1", "other"), 0)


class IndependentReasoningTests(unittest.TestCase):
    def test_no_hint_keeps_reasoning_depth(self):
        self.assertEqual(independent_reasoning_score(4, 0), 4)

    def test_level_penalties(self):
        self.assertEqual(independent_reasoning_score(4, 1), 3)
        self.assertEqual(independent_reasoning_score(4, 2), 3)
        self.assertEqual(independent_reasoning_score(4, 3), 2)
        self.assertEqual(independent_reasoning_score(4, 4), 1)

    def test_floor_at_zero(self):
        self.assertEqual(independent_reasoning_score(1, 4), 0)
        self.assertEqual(independent_reasoning_score(0, 4), 0)

    def test_clamps_to_five(self):
        self.assertEqual(independent_reasoning_score(9, 0), 5)

    def test_apply_hint_penalty_sets_subscore(self):
        rubric = {
            "technical_accuracy": 4,
            "reasoning_depth": 4,
            "communication_clarity": 3,
            "completeness": 4,
            "confidence_signal": 3,
            "overall": 80,
        }
        out = apply_hint_penalty(rubric, 3)
        self.assertEqual(out["independent_reasoning"], 2)
        # The original reasoning_depth is untouched.
        self.assertEqual(out["reasoning_depth"], 4)
        # The input rubric is not mutated.
        self.assertNotIn("independent_reasoning", rubric)

    def test_apply_hint_penalty_without_hint(self):
        out = apply_hint_penalty({"reasoning_depth": 5}, 0)
        self.assertEqual(out["independent_reasoning"], 5)


if __name__ == "__main__":
    unittest.main()
