import asyncio
import unittest

from app.services.progress_summarizer import (
    MAX_SUMMARY_CHARS,
    ProgressSummarizer,
    build_attempt_stats,
    build_fallback_summary,
)


class FakeLLM:
    def __init__(self, result=None, raises: bool = False):
        self.result = result or {}
        self.raises = raises
        self.calls: list[dict] = []

    async def completion(self, prompt, llm_config=None, user_identity=None, task=None, **kwargs):
        self.calls.append(
            {"prompt": prompt, "llm_config": llm_config, "user_identity": user_identity, "task": task}
        )
        if self.raises:
            raise RuntimeError("llm exploded")
        return self.result


class ProgressSummarizerTests(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_build_attempt_stats_counts_outcomes_and_difficulty(self):
        stats = build_attempt_stats(
            [
                {"difficulty": "easy", "revealed": True, "is_correct": True, "confidence": 4},
                {"difficulty": "easy", "revealed": True, "is_correct": False, "confidence": 2},
                {"difficulty": "hard", "revealed": False, "is_correct": None, "confidence": 0},
                {"difficulty": "weird", "revealed": False, "is_correct": None, "confidence": 0},
            ]
        )
        self.assertEqual(stats["total"], 4)
        self.assertEqual(stats["revealed"], 2)
        self.assertEqual(stats["submitted"], 2)
        self.assertEqual(stats["correct"], 1)
        self.assertEqual(stats["incorrect"], 1)
        self.assertEqual(stats["not_submitted"], 2)
        self.assertEqual(stats["avg_confidence"], 3.0)
        # Unknown difficulties are bucketed rather than dropped.
        self.assertEqual(stats["by_difficulty"]["medium"]["total"], 1)
        self.assertEqual(stats["by_difficulty"]["hard"]["total"], 1)

    def test_prompt_contains_topic_sections_questions_and_previous(self):
        summarizer = ProgressSummarizer(FakeLLM())
        prompt = summarizer.build_summary_prompt(
            topic_title="HTTP Caching",
            sections=["Cache-Control", "ETag revalidation"],
            questions=["How does stale-while-revalidate work?"],
            attempt_stats={"total": 1},
            previous_summary="Covered Cache-Control basics.",
        )
        self.assertIn("HTTP Caching", prompt)
        self.assertIn("Cache-Control", prompt)
        self.assertIn("How does stale-while-revalidate work?", prompt)
        self.assertIn("Covered Cache-Control basics.", prompt)

    def test_prompt_handles_empty_history(self):
        summarizer = ProgressSummarizer(FakeLLM())
        prompt = summarizer.build_summary_prompt(topic_title="Empty Topic")
        self.assertIn("(none recorded)", prompt)
        self.assertIn("(no previous note", prompt)

    def test_successful_generation_returns_ai_source(self):
        llm = FakeLLM(
            {
                "success": True,
                "analysis": "Learner covered cache-control basics and is ready for ETag revalidation.",
                "metadata": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
                "error": "",
            }
        )
        result = self._run(
            ProgressSummarizer(llm).generate(
                user_id="alice",
                topic_title="HTTP Caching",
                sections=["Cache-Control"],
                questions=["What is ETag revalidation?"],
                attempt_stats={"total": 1},
            )
        )
        self.assertEqual(result["source"], "ai")
        self.assertIn("ETag revalidation", result["text"])
        self.assertEqual(result["provider_used"], "groq")
        self.assertEqual(result["model_used"], "llama-3.3-70b-versatile")
        # The learner's configured model must win over an inferred task route.
        self.assertEqual(llm.calls[0]["task"], "final")

    def test_policy_block_falls_back(self):
        llm = FakeLLM(
            {
                "success": False,
                "analysis": "",
                "metadata": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
                "error": "Personal credential required",
                "error_code": "personal_credential_required",
            }
        )
        result = self._run(
            ProgressSummarizer(llm).generate(
                user_id="alice",
                topic_title="HTTP Caching",
                sections=["Cache-Control"],
                questions=["What is ETag revalidation?"],
                attempt_stats=build_attempt_stats(
                    [
                        {"difficulty": "easy", "revealed": True, "is_correct": False, "confidence": 2}
                    ]
                ),
            )
        )
        self.assertEqual(result["source"], "fallback")
        self.assertIn("HTTP Caching", result["text"])
        self.assertIn("Cache-Control", result["text"])
        self.assertIn("needing more practice", result["text"])

    def test_empty_analysis_falls_back(self):
        llm = FakeLLM({"success": True, "analysis": "   ", "metadata": {}, "error": ""})
        result = self._run(
            ProgressSummarizer(llm).generate(user_id="alice", topic_title="Empty Topic")
        )
        self.assertEqual(result["source"], "fallback")
        self.assertIn("Empty Topic", result["text"])

    def test_transport_failure_falls_back(self):
        llm = FakeLLM({"success": False, "analysis": "", "metadata": {}, "error": "timeout"})
        result = self._run(
            ProgressSummarizer(llm).generate(user_id="alice", topic_title="Timeout Topic")
        )
        self.assertEqual(result["source"], "fallback")
        self.assertIn("Timeout Topic", result["text"])

    def test_exception_never_escapes(self):
        llm = FakeLLM(raises=True)
        with self.assertLogs("app.services.progress_summarizer", level="ERROR"):
            result = self._run(
                ProgressSummarizer(llm).generate(user_id="alice", topic_title="Boom Topic")
            )
        self.assertEqual(result["source"], "fallback")
        self.assertIn("Boom Topic", result["text"])

    def test_model_output_is_stripped_and_capped(self):
        llm = FakeLLM(
            {
                "success": True,
                "analysis": "```markdown\n" + ("word " * 900) + "\n```",
                "metadata": {},
            }
        )
        result = self._run(
            ProgressSummarizer(llm).generate(user_id="alice", topic_title="Long Topic")
        )
        self.assertEqual(result["source"], "ai")
        self.assertFalse(result["text"].startswith("```"))
        self.assertLessEqual(len(result["text"]), MAX_SUMMARY_CHARS)

    def test_fallback_summary_reports_weakest_difficulty(self):
        text = build_fallback_summary(
            topic_title="Indexes",
            sections=["B-tree"],
            questions=["How does a composite index help?"],
            attempt_stats=build_attempt_stats(
                [
                    {"difficulty": "easy", "is_correct": True, "confidence": 5},
                    {"difficulty": "hard", "is_correct": False, "confidence": 1},
                ]
            ),
        )
        self.assertIn("Weakest area: hard", text)
        self.assertIn("B-tree", text)

    def test_fallback_summary_caps_length(self):
        text = build_fallback_summary(
            topic_title="Huge",
            sections=[f"Section {i}" for i in range(200)],
            questions=[f"Question {i}?" for i in range(200)],
            attempt_stats={"total": 200, "submitted": 200, "correct": 100, "incorrect": 100},
        )
        self.assertLessEqual(len(text), MAX_SUMMARY_CHARS)
