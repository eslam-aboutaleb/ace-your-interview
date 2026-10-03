"""Stage 3.5 (batch concurrency) + Stage 2.2 (bounds) for the questions router."""

import asyncio
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.dependencies import require_auth
from app.routers import questions
from app.schemas.models import GenerateQuestionsRequest, GenerateQuizRequest, TopicDetail
from app.services.doc_parser import DocParser
from app.services.learning_store import LearningStore


class FakeLLM:
    def __init__(self):
        self.prompts: list[str] = []

    async def completion(self, prompt, llm_config=None, user_identity=None, task=None, **kwargs):
        self.prompts.append(prompt)
        # Stage 2.1 put each persona in the system role, so route on `system`
        # first and fall back to the task-shaped prompt text.
        haystack = f"{kwargs.get('system') or ''}\n{prompt}".lower()
        wants_quiz = "expert technical quiz creator" in haystack or (
            "quiz questions from the documentation" in haystack
        )
        payload = [
            _quiz_item()
            if wants_quiz
            else {
                "question": "How would you explain the tradeoffs of this design in an interview?",
                "answer": (
                    "**Answer:** The design trades a little extra memory for a much simpler "
                    "concurrency story, which is usually the right call in an interview.\n\n"
                    "**Detailed explanation:** Start from the invariant that only one writer "
                    "mutates shared state, then show why the extra storage removes the need "
                    "for a lock on the read path.\n\n"
                    "**Common mistake:** A weak answer reaches for a distributed lock before "
                    "naming the single-writer assumption the design already provides."
                ),
                "difficulty": "medium",
                "topic_id": "custom-java",
                "source_quote": "Collections track insertion order behavior.",
                "reasoning_summary": "Single writer removes the need for a read-path lock.",
                "target_level": "mid",
            }
        ]
        return {
            "success": True,
            "analysis": json.dumps(payload),
            "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
            "error": "",
            "error_code": "",
        }


def _quiz_item() -> dict:
    return {
        "question": "Which structure preserves insertion order and fast key lookups?",
        "type": "mcq",
        "choices": [
            {"label": "A", "text": "LinkedHashMap"},
            {"label": "B", "text": "HashSet"},
            {"label": "C", "text": "TreeSet"},
            {"label": "D", "text": "PriorityQueue"},
        ],
        "correct_answer": "A",
        "explanation": "LinkedHashMap preserves insertion order with hash lookups.",
        "difficulty": "medium",
        "topic_id": "custom-java",
        "source_quote": "Collections track insertion order behavior.",
        "reasoning_summary": "Insertion order plus key lookup points to LinkedHashMap.",
        "target_level": "mid",
    }


class ConcurrencyProbe:
    """Records how many topic resolutions were in flight at the same time."""

    def __init__(self, delays: dict[str, float] | None = None, unknown: set[str] | None = None):
        self.delays = delays or {}
        self.unknown = unknown or set()
        self.in_flight = 0
        self.max_in_flight = 0
        self.order: list[str] = []

    async def resolve(self, *, topic_id, user_id, preferred_language_hint=None):
        if topic_id in self.unknown:
            self.order.append(topic_id)
            return None
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delays.get(topic_id, 0.01))
            self.order.append(topic_id)
            detail = TopicDetail(
                id=topic_id,
                title=f"Title for {topic_id}",
                description=f"Desc for {topic_id}",
                track="backend",
                levels=["junior", "mid", "senior"],
                sections=[{"heading": "S", "content": f"content {topic_id}"}],
                raw_content=f"# {topic_id}\n\ncontent {topic_id}",
                content_ready=True,
            )
            return detail, detail.raw_content
        finally:
            self.in_flight -= 1


class QuestionsRouterConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.prev_llm = questions._llm_client
        self.prev_parser = questions._parser
        self.prev_store = questions._learning_store
        self.prev_progress = questions._progress_store

        self.tempdir = tempfile.TemporaryDirectory()
        handbook = os.path.join(self.tempdir.name, "owner-handbook")
        os.makedirs(handbook, exist_ok=True)
        with open(os.path.join(handbook, "01-static.md"), "w", encoding="utf-8") as handle:
            handle.write(
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
        for topic_id, title in (("custom-java", "Java"), ("custom-rust", "Rust")):
            self.store.upsert_custom_topic(
                user_id="alice",
                topic_id=topic_id,
                source_topic=title,
                title=f"{title} Interview Roadmap",
                description=f"Deep {title} roadmap.",
                track="backend",
                levels=["junior", "mid", "senior"],
                sections=[{"heading": "Fundamentals", "content": f"{title} basics."}],
                raw_content=f"# {title} Interview Roadmap\n\n## Fundamentals\n\n{title} basics.",
            )

        self.fake_llm = FakeLLM()
        questions.init(self.fake_llm, self.parser, self.store)
        self.app = FastAPI()
        self.app.dependency_overrides[require_auth] = lambda: {"user": "alice", "provider": "local"}
        self.app.include_router(questions.router)
        self.client = TestClient(self.app)

    def tearDown(self):
        questions._llm_client = self.prev_llm
        questions._parser = self.prev_parser
        questions._learning_store = self.prev_store
        questions._progress_store = self.prev_progress
        self.tempdir.cleanup()

    # ── ordering + real concurrency ────────────────────────

    def test_multi_topic_resolution_is_concurrent_and_order_preserving(self):
        probe = ConcurrencyProbe(
            delays={"topic-a": 0.05, "topic-b": 0.01, "topic-c": 0.03, "topic-d": 0.0}
        )
        body = GenerateQuizRequest(topic_ids=["topic-a", "topic-b", "topic-c", "topic-d"])
        user = {"user": "alice", "provider": "local"}

        async def _run():
            with patch.object(questions, "_resolve_topic_for_user", probe.resolve):
                return await questions._resolve_quiz_topics(
                    topic_ids=body.topic_ids,
                    user=user,
                    store=self.store,
                    body=body,
                )

        payloads, failed = asyncio.run(_run())

        # Completion order differs from request order, yet output order matches.
        self.assertNotEqual(probe.order, body.topic_ids)
        self.assertEqual([item["id"] for item in payloads], body.topic_ids)
        self.assertEqual(failed, [])
        # Genuine concurrency, not interleaved sequential awaits.
        self.assertGreater(probe.max_in_flight, 1)

    def test_concurrency_is_bounded_by_the_semaphore(self):
        topic_ids = [f"topic-{index}" for index in range(8)]
        probe = ConcurrencyProbe(delays={tid: 0.01 for tid in topic_ids})
        body = GenerateQuizRequest(topic_ids=topic_ids)
        user = {"user": "alice", "provider": "local"}

        async def _run():
            with patch.object(questions, "_resolve_topic_for_user", probe.resolve):
                return await questions._resolve_quiz_topics(
                    topic_ids=topic_ids,
                    user=user,
                    store=self.store,
                    body=body,
                )

        payloads, _ = asyncio.run(_run())
        self.assertEqual([item["id"] for item in payloads], topic_ids)
        self.assertLessEqual(probe.max_in_flight, questions.TOPIC_RESOLVE_CONCURRENCY)

    # ── partial failure ────────────────────────────────────

    def test_partial_failure_is_skipped_and_reported(self):
        probe = ConcurrencyProbe()
        body = GenerateQuizRequest(topic_ids=["custom-java", "custom-rust", "custom-java"])
        user = {"user": "alice", "provider": "local"}
        original = questions._resolve_ai_settings

        async def flaky(*, store, user_id, topic, response_detail_override, preferred_language_override):
            if topic.id == "custom-rust":
                raise RuntimeError("preference db unavailable")
            return await original(
                store=store,
                user_id=user_id,
                topic=topic,
                response_detail_override=response_detail_override,
                preferred_language_override=preferred_language_override,
            )

        async def _run():
            with patch.object(questions, "_resolve_topic_for_user", probe.resolve):
                with patch.object(questions, "_resolve_ai_settings", flaky):
                    return await questions._resolve_quiz_topics(
                        topic_ids=body.topic_ids,
                        user=user,
                        store=self.store,
                        body=body,
                    )

        with self.assertLogs("app.routers.questions", level="ERROR"):
            payloads, failed = asyncio.run(_run())

        self.assertEqual([item["id"] for item in payloads], ["custom-java", "custom-java"])
        self.assertEqual(failed, ["custom-rust"])

    def test_unknown_topic_still_fails_the_request(self):
        body = GenerateQuizRequest(topic_ids=["custom-java", "missing-topic"])
        user = {"user": "alice", "provider": "local"}
        probe = ConcurrencyProbe(unknown={"missing-topic"})

        async def _run():
            with patch.object(questions, "_resolve_topic_for_user", probe.resolve):
                return await questions._resolve_quiz_topics(
                    topic_ids=body.topic_ids,
                    user=user,
                    store=self.store,
                    body=body,
                )

        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(_run())
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertIn("missing-topic", str(ctx.exception.detail))

    def test_empty_topic_list_still_rejected(self):
        body = GenerateQuizRequest(topic_ids=[])
        user = {"user": "alice", "provider": "local"}

        async def _run():
            return await questions._resolve_quiz_topics(
                topic_ids=body.topic_ids,
                user=user,
                store=self.store,
                body=body,
            )

        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(_run())
        self.assertEqual(ctx.exception.status_code, 400)

    def test_all_topics_failing_surfaces_a_distinct_error(self):
        body = GenerateQuizRequest(topic_ids=["boom-a", "boom-b"])
        user = {"user": "alice", "provider": "local"}

        async def _explode(**kwargs):
            raise RuntimeError("store offline")

        async def _run():
            with patch.object(questions, "_resolve_topic_for_user", _explode):
                return await questions._resolve_quiz_topics(
                    topic_ids=body.topic_ids,
                    user=user,
                    store=self.store,
                    body=body,
                )

        with self.assertLogs("app.routers.questions", level="ERROR"):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(_run())
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ctx.exception.detail["code"], "topic_resolution_failed")

    def test_http_quiz_reports_partial_failure_but_still_answers(self):
        original = questions._resolve_ai_settings

        async def flaky(*, store, user_id, topic, response_detail_override, preferred_language_override):
            if topic.id == "custom-rust":
                raise RuntimeError("preference db unavailable")
            return await original(
                store=store,
                user_id=user_id,
                topic=topic,
                response_detail_override=response_detail_override,
                preferred_language_override=preferred_language_override,
            )

        with patch.object(questions, "_resolve_ai_settings", flaky):
            with self.assertLogs("app.routers.questions", level="ERROR"):
                res = self.client.post(
                    "/api/questions/quiz/generate-v2",
                    json={
                        "topic_ids": ["custom-java", "custom-rust"],
                        "count": 1,
                        "question_types": ["mcq"],
                        "level": "mid",
                    },
                )

        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(
            res.headers.get(questions.PARTIAL_FAILURE_HEADER), "custom-rust"
        )
        self.assertEqual(res.json()["topics_used"], ["custom-java"])
        self.assertEqual(len(res.json()["questions"]), 1)

    def test_quiz_stream_emits_a_warning_event_on_partial_failure(self):
        original = questions._resolve_ai_settings

        async def flaky(*, store, user_id, topic, response_detail_override, preferred_language_override):
            if topic.id == "custom-rust":
                raise RuntimeError("preference db unavailable")
            return await original(
                store=store,
                user_id=user_id,
                topic=topic,
                response_detail_override=response_detail_override,
                preferred_language_override=preferred_language_override,
            )

        with patch.object(questions, "_resolve_ai_settings", flaky):
            with self.assertLogs("app.routers.questions", level="ERROR"):
                with self.client.stream(
                    "POST",
                    "/api/questions/quiz/generate-v2/stream",
                    json={
                        "topic_ids": ["custom-java", "custom-rust"],
                        "count": 1,
                        "question_types": ["mcq"],
                        "level": "mid",
                    },
                ) as res:
                    events = [
                        json.loads(line)
                        for line in res.iter_lines()
                        if line
                    ]
                    status = res.status_code

        self.assertEqual(status, 200)
        warnings = [event for event in events if event.get("type") == "warning"]
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]["code"], "topic_resolution_failed")
        self.assertEqual(warnings[0]["topics"], ["custom-rust"])
        self.assertEqual(warnings[0]["topics_used"], ["custom-java"])
        self.assertEqual(events[-1]["type"], "done")

    # ── 2.2 bounds on client-supplied prompt inputs ────────

    @staticmethod
    def _schema_max_length(model, field: str) -> int:
        """Read a field's declared `max_length` straight off the schema.

        Deriving the expected bound from `model_fields` keeps this test pinned to
        the single declaration instead of restating the same literal twice, so
        moving a bound in `schemas/models.py` cannot leave a stale copy here.
        """
        for meta in model.model_fields[field].metadata:
            value = getattr(meta, "max_length", None)
            if value is not None:
                return int(value)
        raise AssertionError(f"{model.__name__}.{field} declares no max_length")

    def test_oversized_section_content_is_rejected_at_the_schema_boundary(self):
        title_bound = self._schema_max_length(GenerateQuestionsRequest, "section_title")
        content_bound = self._schema_max_length(GenerateQuestionsRequest, "section_content")

        res = self.client.post(
            "/api/questions/generate-v2",
            json={
                "topic_id": "custom-java",
                "count": 1,
                "level": "mid",
                "section_title": "T" * (title_bound + 1),
                "section_content": "S" * (content_bound + 1),
            },
        )
        # Oversized free text is refused outright, so no LLM call is made.
        self.assertEqual(res.status_code, 422, res.text)
        errors = {
            tuple(error["loc"]): error for error in res.json()["detail"]
        }
        self.assertEqual(
            set(errors),
            {("body", "section_title"), ("body", "section_content")},
        )
        for field, bound in (("section_title", title_bound), ("section_content", content_bound)):
            with self.subTest(field=field):
                self.assertEqual(errors[("body", field)]["type"], "string_too_long")
                self.assertEqual(errors[("body", field)]["ctx"]["max_length"], bound)
        self.assertEqual(self.fake_llm.prompts, [])

        # The declared limits are the same ones the router helper clamps to.
        self.assertEqual(questions.MAX_SECTION_TITLE_CHARS, title_bound)
        self.assertEqual(questions.MAX_SECTION_CONTENT_CHARS, content_bound)

    def test_section_content_at_the_declared_bound_still_reaches_the_prompt(self):
        title_bound = self._schema_max_length(GenerateQuestionsRequest, "section_title")
        content_bound = self._schema_max_length(GenerateQuestionsRequest, "section_content")

        res = self.client.post(
            "/api/questions/generate-v2",
            json={
                "topic_id": "custom-java",
                "count": 1,
                "level": "mid",
                "section_title": "T" * title_bound,
                "section_content": "S" * content_bound,
            },
        )
        self.assertEqual(res.status_code, 200, res.text)

        base_prompt = self.fake_llm.prompts[0]
        # The section still reaches the prompt, in full at the bound.
        self.assertIn("S" * 500, base_prompt)
        self.assertIn("T" * 10, base_prompt)
        self.assertLess(
            len(base_prompt),
            content_bound + title_bound + 20000,
        )

    def test_bounded_section_request_helper_matches_declared_limits(self):
        title_bound = self._schema_max_length(GenerateQuestionsRequest, "section_title")
        content_bound = self._schema_max_length(GenerateQuestionsRequest, "section_content")

        # The router clamp and the schema bound are one declaration, not two.
        self.assertEqual(questions.MAX_SECTION_TITLE_CHARS, title_bound)
        self.assertEqual(questions.MAX_SECTION_CONTENT_CHARS, content_bound)

        body = GenerateQuestionsRequest(
            topic_id="custom-java",
            section_title="T" * title_bound,
            section_content="S" * content_bound,
            existing_questions=["E" * 2000],
        )
        title, content, existing = questions._bounded_section_request(body)
        # A request already inside the schema bound is passed through whole.
        self.assertEqual(len(title), title_bound)
        self.assertEqual(len(content), content_bound)
        # `existing_questions` items carry no per-item schema bound, so the
        # router-level clamp is still the mechanism that bounds them.
        self.assertEqual(len(existing), 1)
        self.assertEqual(len(existing[0]), questions.MAX_EXISTING_QUESTION_CHARS)

        # One character past a bound is a validation error, not a silent clamp.
        with self.assertRaises(ValidationError) as ctx:
            GenerateQuestionsRequest(
                topic_id="custom-java",
                section_title="T" * (title_bound + 1),
            )
        error = ctx.exception.errors()[0]
        self.assertEqual(error["type"], "string_too_long")
        self.assertEqual(error["loc"], ("section_title",))
        self.assertEqual(error["ctx"]["max_length"], title_bound)

    def test_existing_questions_are_bounded_per_item(self):
        res = self.client.post(
            "/api/questions/generate-v2",
            json={
                "topic_id": "custom-java",
                "count": 1,
                "level": "mid",
                "existing_questions": ["Q" * 5000],
            },
        )
        self.assertEqual(res.status_code, 200, res.text)
        self.assertNotIn("Q" * 500, self.fake_llm.prompts[0])


if __name__ == "__main__":
    unittest.main()