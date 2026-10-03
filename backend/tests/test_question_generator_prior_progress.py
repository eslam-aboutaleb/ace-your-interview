"""Prompt-level prior-progress injection for both the questions and quiz paths."""

import unittest

from app.services import question_generator as qg


class QuestionGeneratorPriorProgressTests(unittest.TestCase):
    def _topics(self) -> list[dict]:
        return [
            {
                "id": "01-http",
                "title": "HTTP Fundamentals",
                "content": "Keep-alive reuses a connection across requests.",
                "response_detail": "very_detailed",
                "preferred_language": "",
                "requires_programming": False,
            }
        ]

    # ── questions prompt ────────────────────────────────────────

    def test_build_prompt_includes_progress_block(self):
        prompt = qg._build_prompt(
            topic_id="01-http",
            topic_title="HTTP Fundamentals",
            doc_content="Keep-alive reuses a connection across requests.",
            count=1,
            prior_progress="Learner already covered connection reuse.",
        )
        self.assertIn("Learner progress on this topic so far", prompt)
        self.assertIn("Learner already covered connection reuse.", prompt)

    def test_build_prompt_omits_progress_block_when_absent(self):
        prompt = qg._build_prompt(
            topic_id="01-http",
            topic_title="HTTP Fundamentals",
            doc_content="Keep-alive reuses a connection across requests.",
            count=1,
        )
        self.assertNotIn("Learner progress on this topic so far", prompt)

    def test_progress_block_is_capped(self):
        block = qg._progress_block("x" * 9000)
        self.assertIn("Learner progress on this topic so far", block)
        # Longest contiguous run of injected content is exactly the cap.
        self.assertEqual(block.count("x" * 20), qg._MAX_PROGRESS_BLOCK_CHARS // 20)

    def test_build_prompt_embeds_the_capped_block(self):
        prompt = qg._build_prompt(
            topic_id="01-http",
            topic_title="HTTP Fundamentals",
            doc_content="Keep-alive reuses a connection across requests.",
            count=1,
            prior_progress="x" * 9000,
        )
        self.assertIn(qg._progress_block("x" * 9000), prompt)

    def test_progress_block_precedes_uniqueness_block(self):
        prompt = qg._build_prompt(
            topic_id="01-http",
            topic_title="HTTP Fundamentals",
            doc_content="Keep-alive reuses a connection across requests.",
            count=1,
            prior_progress="Learner already covered connection reuse.",
            additional_existing_questions=["Stored question about idle timeouts?"],
        )
        progress_at = prompt.index("Learner progress on this topic so far")
        uniqueness_at = prompt.index("Already generated questions for this checkpoint")
        self.assertLess(progress_at, uniqueness_at)

    def test_client_list_wins_the_budget_then_history_fills_it(self):
        client = [f"Client question {i}?" for i in range(5)]
        stored = [f"Stored question {i}?" for i in range(5)]
        merged = qg._merge_existing_questions(client, stored)
        self.assertEqual(merged[:5], client)
        self.assertEqual(merged[5:], stored)

    def test_merge_dedupes_across_lists(self):
        merged = qg._merge_existing_questions(
            ["Explain  the   timeout  semantics."], ["explain the timeout semantics."]
        )
        self.assertEqual(len(merged), 1)

    def test_merge_respects_the_eighty_item_cap(self):
        client = [f"Client question {i}?" for i in range(70)]
        stored = [f"Stored question {i}?" for i in range(70)]
        merged = qg._merge_existing_questions(client, stored)
        self.assertEqual(len(merged), 80)
        # Oldest history is dropped; the client list is never truncated first.
        self.assertEqual(merged[:70], client)
        self.assertEqual(merged[70], stored[0])

    def test_additional_history_reaches_the_uniqueness_block(self):
        prompt = qg._build_prompt(
            topic_id="01-http",
            topic_title="HTTP Fundamentals",
            doc_content="Keep-alive reuses a connection across requests.",
            count=1,
            additional_existing_questions=["Stored question about idle timeouts?"],
        )
        self.assertIn("Stored question about idle timeouts?", prompt)

    # ── quiz prompt ─────────────────────────────────────────────

    def test_build_quiz_prompt_includes_progress_and_history(self):
        prompt = qg._build_quiz_prompt(
            self._topics(),
            count=2,
            prior_progress="Learner already covered connection reuse.",
            additional_existing_questions=["Stored quiz question about pipelining?"],
        )
        self.assertIn("Learner progress on this topic so far", prompt)
        self.assertIn("Learner already covered connection reuse.", prompt)
        self.assertIn("Already asked for these topics", prompt)
        self.assertIn("Stored quiz question about pipelining?", prompt)

    def test_build_quiz_prompt_without_history_has_no_blocks(self):
        prompt = qg._build_quiz_prompt(self._topics(), count=2)
        self.assertNotIn("Learner progress on this topic so far", prompt)
        self.assertNotIn("Already asked for these topics", prompt)

    def test_quiz_prompt_progress_block_is_capped(self):
        prompt = qg._build_quiz_prompt(
            self._topics(),
            count=1,
            prior_progress="y" * 9000,
        )
        self.assertIn(qg._progress_block("y" * 9000), prompt)

    # ── quiz dedup state seeding ────────────────────────────────

    def test_quiz_stream_seeds_dedup_state_from_history(self):
        import asyncio
        import json

        prompts: list[str] = []

        class FakeLLM:
            async def completion(self, prompt, llm_config=None, user_identity=None):
                prompts.append(prompt)
                return {
                    "success": True,
                    "analysis": json.dumps(
                        [
                            {
                                "question": (
                                    "Which HTTP header lets a client send a validator for a cached response?"
                                ),
                                "type": "mcq",
                                "choices": [
                                    {"label": "A", "text": "ETag"},
                                    {"label": "B", "text": "Age"},
                                    {"label": "C", "text": "Vary"},
                                    {"label": "D", "text": "Expires"},
                                ],
                                "correct_answer": "A",
                                "explanation": "The ETag header is the response validator.",
                                "difficulty": "medium",
                                "topic_id": "01-http",
                                "source_quote": "Keep-alive reuses a connection across requests.",
                                "reasoning_summary": "Identify the validator header.",
                                "target_level": "mid",
                            }
                        ]
                    ),
                    "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
                }

        generator = qg.QuestionGenerator(FakeLLM())

        async def _exercise() -> list[dict]:
            events = []
            async for event in generator.generate_quiz_v2_stream(
                topics_content=self._topics(),
                count=1,
                additional_existing_questions=[
                    "Which HTTP header lets a client send a validator for a cached response?"
                ],
            ):
                events.append(event)
            return events

        events = asyncio.run(_exercise())
        emitted = [e["question"]["question"] for e in events if e.get("type") == "question"]
        stored_question = (
            "Which HTTP header lets a client send a validator for a cached response?"
        )
        # The seeded dedup state rejects the duplicate the LLM keeps returning,
        # so the stream falls back to a different question instead of repeating it.
        self.assertEqual(len(emitted), 1)
        self.assertNotIn(stored_question, emitted)
        self.assertIn("Already asked for these topics", prompts[0])
        self.assertIn(stored_question, prompts[0])

    def test_quiz_batch_passes_history_to_collect_with_retries(self):
        captured: dict = {}

        async def fake_collect(**kwargs):
            captured.update(kwargs)
            return [], {"metadata": {}, "retries_used": 0, "malformed_items_dropped": 0}

        original = qg._collect_with_retries
        qg._collect_with_retries = fake_collect
        try:
            import asyncio

            generator = qg.QuestionGenerator(object())
            asyncio.run(
                generator.generate_quiz_v2(
                    topics_content=self._topics(),
                    count=2,
                    additional_existing_questions=["Stored quiz question about pipelining?"],
                    prior_progress="Learner already covered connection reuse.",
                )
            )
        finally:
            qg._collect_with_retries = original

        self.assertEqual(
            captured["existing_questions"], ["Stored quiz question about pipelining?"]
        )
        self.assertIn("Learner already covered connection reuse", captured["base_prompt"])
