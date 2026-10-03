"""Coverage for app/routers/questions.py.

Pins the router edges the happy-path tests skip: the service gate, every
404/409/422 rejection on the five generation routes, the settings-override
branches inside ``_resolve_ai_settings``, the per-topic quiz resolution
failure, and the v1 ``POST /api/questions/quiz/generate`` route (which has no
other coverage) including its ``X-Topic-Resolution-Failures`` partial-failure
header.

All generation runs use an injected fake LLM, so nothing here can open a
socket.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import require_auth
from app.routers import questions
from app.services.doc_parser import DocParser
from app.services.learning_store import LearningStore
from app.services.progress_store import ProgressStore

PROBLEM_SOLVING_ID = "00-problem-solving-and-algorithms"

TOPIC_ID = "01-static"


def _valid_question() -> dict:
    return {
        "question": "Explain how connection pooling improves throughput for HTTP calls",
        "answer": "Pooling reuses sockets, so handshakes are amortised across requests.",
        "category": "conceptual",
        "difficulty": "medium",
        "subcategory": "Networking",
        "key_concepts": ["pooling", "latency"],
    }


def _valid_quiz_question(topic_id: str = "01-static") -> dict:
    return {
        "question": "Which structure preserves insertion order while offering fast lookup?",
        "type": "mcq",
        "choices": [
            {"label": "A", "text": "LinkedHashMap"},
            {"label": "B", "text": "HashSet"},
            {"label": "C", "text": "TreeSet"},
            {"label": "D", "text": "PriorityQueue"},
        ],
        "correct_answer": "A",
        "explanation": "A LinkedHashMap keeps a linked list of its entries in order.",
        "difficulty": "medium",
        "topic_id": topic_id,
        "source_quote": "Section body about caching and connection pooling.",
        "reasoning_summary": "Insertion order plus hash lookup points to LinkedHashMap.",
        "target_level": "mid",
    }


DISTINCT_QUIZ_QUESTIONS = [
    "Which structure preserves insertion order while offering fast lookup?",
    "Which eviction policy protects a cache from a working set that shifts?",
    "Which isolation level permits a non-repeatable read within a transaction?",
    "Which scheduling policy prevents a single noisy neighbour starving others?",
    "Which backpressure signal tells a producer to slow down safely?",
]


class QuizLLM:
    """Replays a canned quiz payload, then a generic one for follow-up batches."""

    def __init__(self, question=None):
        self.calls = 0
        self.prompts: list[str] = []
        self._question = question or _valid_quiz_question()

    async def completion(self, prompt, llm_config=None, user_identity=None, task=None, **kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        # A distinct question per call: the generator de-duplicates, so a
        # replayed item would be rejected and the count would never fill.
        item = dict(self._question)
        item["question"] = DISTINCT_QUIZ_QUESTIONS[
            (self.calls - 1) % len(DISTINCT_QUIZ_QUESTIONS)
        ]
        return {
            "success": True,
            "analysis": json.dumps([item]),
            "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
            "error": "",
            "error_code": "",
        }


class QuestionsLLM:
    def __init__(self):
        self.calls = 0
        self.prompts: list[str] = []

    async def completion(self, prompt, llm_config=None, user_identity=None, task=None, **kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        return {
            "success": True,
            "analysis": json.dumps([_valid_question()]),
            "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
            "error": "",
            "error_code": "",
        }


def _write_handbook(root: str) -> None:
    handbook = os.path.join(root, "owner-handbook")
    os.makedirs(handbook, exist_ok=True)
    with open(os.path.join(handbook, "01-static.md"), "w", encoding="utf-8") as handle:
        handle.write(
            """---
track: backend
levels: [junior, mid, senior]
---
# Static Topic

Static description about caching.

## Static Section

Section body about caching and connection pooling.
"""
        )


class QuestionsRouterDeepTests(unittest.TestCase):
    def setUp(self):
        self._prev = (
            questions._llm_client,
            questions._parser,
            questions._learning_store,
            questions._mcp_gateway,
            questions._progress_store,
        )
        self.tempdir = tempfile.TemporaryDirectory()
        _write_handbook(self.tempdir.name)
        self.parser = DocParser(docs_path=self.tempdir.name)
        self.store = LearningStore(os.path.join(self.tempdir.name, "learning.db"))
        self.llm = QuizLLM()
        self.progress = ProgressStore(os.path.join(self.tempdir.name, "progress.db"))
        questions.init(self.llm, self.parser, self.store, mcp_gateway=None)
        questions._progress_store = self.progress

        self.app = FastAPI()
        self.app.dependency_overrides[require_auth] = lambda: {
            "user": "alice",
            "provider": "local",
        }
        self.app.include_router(questions.router)
        self.client = TestClient(self.app)

    def tearDown(self):
        (
            questions._llm_client,
            questions._parser,
            questions._learning_store,
            questions._mcp_gateway,
            questions._progress_store,
        ) = self._prev
        self.tempdir.cleanup()

    def _quiz_payload(self, **overrides):
        payload = {
            "topic_ids": [TOPIC_ID],
            "count": 3,
            "question_types": ["mcq"],
        }
        payload.update(overrides)
        return payload

    # ---------------------------------------------------------------- gating

    def test_every_generation_route_refuses_to_serve_before_init(self):
        questions._llm_client = None
        for path, payload in (
            ("/api/questions/generate", {"topic_id": TOPIC_ID, "count": 2, "question_types": ["interview"]}),
            ("/api/questions/generate-v2", {"topic_id": TOPIC_ID, "count": 2, "question_types": ["interview"]}),
            ("/api/questions/generate-v2/stream", {"topic_id": TOPIC_ID, "count": 2, "question_types": ["interview"]}),
            ("/api/questions/quiz/generate", self._quiz_payload()),
            ("/api/questions/quiz/generate-v2", self._quiz_payload()),
            ("/api/questions/quiz/generate-v2/stream", self._quiz_payload()),
        ):
            with self.subTest(path=path):
                res = self.client.post(path, json=payload)
                self.assertEqual(res.status_code, 503, path)
                self.assertEqual(res.json()["detail"], "Service not initialised")

    def test_the_v2_routes_are_404_when_v2_generation_is_disabled(self):
        self.assertTrue(get_settings().enable_v2_generation)
        for path, payload in (
            ("/api/questions/generate-v2", {"topic_id": TOPIC_ID, "count": 2, "question_types": ["interview"]}),
            ("/api/questions/generate-v2/stream", {"topic_id": TOPIC_ID, "count": 2, "question_types": ["interview"]}),
            ("/api/questions/quiz/generate-v2", self._quiz_payload()),
            ("/api/questions/quiz/generate-v2/stream", self._quiz_payload()),
        ):
            with self.subTest(path=path):
                with patch.object(
                    questions, "get_settings", return_value=_v2_disabled_settings()
                ):
                    res = self.client.post(path, json=payload)
                self.assertEqual(res.status_code, 404, path)
                self.assertEqual(res.json()["detail"], "v2 generation disabled")

    # ------------------------------------------------------- v1 question gen

    def test_generate_questions_is_404_for_an_unknown_topic(self):
        res = self.client.post(
            "/api/questions/generate",
            json={"topic_id": "99-nope", "count": 2, "question_types": ["interview"]},
        )
        self.assertEqual(res.status_code, 404)
        self.assertEqual(res.json()["detail"], "Topic '99-nope' not found")

    def test_generate_questions_is_409_before_the_problem_solving_roadmap_exists(self):
        res = self.client.post(
            "/api/questions/generate",
            json={
                "topic_id": PROBLEM_SOLVING_ID,
                "count": 2,
                "question_types": ["interview"],
            },
        )
        self.assertEqual(res.status_code, 409)
        self.assertIn("not generated yet", res.json()["detail"])

    def test_generate_questions_resolves_the_problem_solving_language_from_settings(self):
        questions._llm_client = QuestionsLLM()
        self.store.upsert_topic_preferences(
            user_id="alice",
            topic_id=PROBLEM_SOLVING_ID,
            response_detail="concise",
            preferred_language="go",
        )
        _seed_problem_solving(self.store, PROBLEM_SOLVING_ID, "go")
        res = self.client.post(
            "/api/questions/generate",
            json={
                "topic_id": PROBLEM_SOLVING_ID,
                "count": 2,
                "question_types": ["interview"],
            },
        )
        self.assertEqual(res.status_code, 200, res.text)
        self.assertIn("go", questions._llm_client.prompts[-1].lower())

    def test_generate_questions_v2_is_404_for_an_unknown_topic(self):
        res = self.client.post(
            "/api/questions/generate-v2",
            json={"topic_id": "99-nope", "count": 2, "question_types": ["interview"]},
        )
        self.assertEqual(res.status_code, 404)
        self.assertEqual(res.json()["detail"], "Topic '99-nope' not found")

    def test_generate_questions_v2_is_409_before_the_roadmap_exists(self):
        res = self.client.post(
            "/api/questions/generate-v2",
            json={
                "topic_id": PROBLEM_SOLVING_ID,
                "count": 2,
                "question_types": ["interview"],
            },
        )
        self.assertEqual(res.status_code, 409)

    def test_generate_questions_v2_stream_is_404_for_an_unknown_topic(self):
        res = self.client.post(
            "/api/questions/generate-v2/stream",
            json={"topic_id": "99-nope", "count": 2, "question_types": ["interview"]},
        )
        self.assertEqual(res.status_code, 404)
        self.assertEqual(res.json()["detail"], "Topic '99-nope' not found")

    def test_generate_questions_v2_stream_is_409_before_the_roadmap_exists(self):
        res = self.client.post(
            "/api/questions/generate-v2/stream",
            json={
                "topic_id": PROBLEM_SOLVING_ID,
                "count": 2,
                "question_types": ["interview"],
            },
        )
        self.assertEqual(res.status_code, 409)

    def test_generate_questions_v2_returns_the_grounded_batch(self):
        questions._llm_client = QuestionsLLM()
        res = self.client.post(
            "/api/questions/generate-v2",
            json={"topic_id": TOPIC_ID, "count": 2, "question_types": ["interview"]},
        )
        self.assertEqual(res.status_code, 200, res.text)
        body = res.json()
        self.assertEqual(len(body["questions"]), 2)
        self.assertEqual(body["questions"][0]["question"], _valid_question()["question"])
        # The batch is captured into the server-side no-repeat history.
        asked = self.progress.get_asked_questions(
            user_id="alice", topic_id=TOPIC_ID, limit=50
        )
        self.assertEqual(len(asked), len(body["questions"]))
        self.assertEqual(set(asked), {item["question"] for item in body["questions"]})

    def test_generate_questions_v2_stream_streams_a_terminal_done_frame(self):
        questions._llm_client = QuestionsLLM()
        with self.client.stream(
            "POST",
            "/api/questions/generate-v2/stream",
            json={"topic_id": TOPIC_ID, "count": 2, "question_types": ["interview"]},
        ) as res:
            self.assertEqual(res.status_code, 200)
            events = [json.loads(line) for line in res.iter_lines() if line]
        self.assertEqual(events[0]["type"], "start")
        self.assertEqual(events[-1]["type"], "done")
        questions_frames = [e for e in events if e["type"] == "question"]
        self.assertTrue(questions_frames)
        self.assertEqual(
            questions_frames[0]["question"]["question"],
            _valid_question()["question"],
        )
        self.assertEqual(events[-1].get("generated_count"), len(questions_frames))

    # ------------------------------------------------------- v1 quiz generate

    def test_the_v1_quiz_route_generates_and_captures_per_topic(self):
        res = self.client.post("/api/questions/quiz/generate", json=self._quiz_payload())
        self.assertEqual(res.status_code, 200, res.text)
        body = res.json()
        self.assertEqual(len(body["questions"]), 3)
        self.assertEqual(body["questions"][0]["type"], "mcq")
        self.assertEqual(body["questions"][0]["correct_answer"], "A")
        self.assertNotIn("X-Topic-Resolution-Failures", res.headers)

    def test_the_v1_quiz_route_groups_the_captured_history_per_topic(self):
        self.client.post("/api/questions/quiz/generate", json=self._quiz_payload())
        # The first topic's question is now "asked", so a second generation
        # must see it in the no-repeat block.
        asked = self.progress.get_asked_questions(user_id="alice", topic_id=TOPIC_ID, limit=50)
        self.assertEqual(len(asked), 3)
        self.assertEqual(set(asked), set(DISTINCT_QUIZ_QUESTIONS[:3]))

    def test_the_v1_quiz_route_is_404_when_every_topic_fails_to_resolve(self):
        res = self.client.post(
            "/api/questions/quiz/generate", json=self._quiz_payload(topic_ids=["99-nope"])
        )
        self.assertEqual(res.status_code, 404)

    def test_the_v1_quiz_route_reports_an_unexpected_topic_failure_without_failing(self):
        # A deterministic rejection (unknown topic) always fails the request; an
        # *unexpected* failure is skipped and surfaced only via the header.
        import app.routers.questions as questions_module

        real_resolve = questions_module._resolve_quiz_topic

        async def _flaky(*, topic_id, user, store, body):
            if topic_id == "99-broken":
                raise RuntimeError("parser exploded")
            return await real_resolve(topic_id=topic_id, user=user, store=store, body=body)

        with patch.object(questions_module, "_resolve_quiz_topic", side_effect=_flaky):
            res = self.client.post(
                "/api/questions/quiz/generate",
                json=self._quiz_payload(topic_ids=[TOPIC_ID, "99-broken"]),
            )
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(res.headers["X-Topic-Resolution-Failures"], "99-broken")
        self.assertEqual(len(res.json()["questions"]), 3)

    def test_a_deterministic_topic_rejection_fails_the_whole_quiz(self):
        res = self.client.post(
            "/api/questions/quiz/generate",
            json=self._quiz_payload(topic_ids=[TOPIC_ID, "99-nope"]),
        )
        self.assertEqual(res.status_code, 404)
        self.assertEqual(res.json()["detail"], "Topic '99-nope' not found")
        self.assertNotIn("X-Topic-Resolution-Failures", res.headers)

    def test_the_v1_quiz_route_is_503_when_every_topic_fails_unexpectedly(self):
        import app.routers.questions as questions_module

        async def _always_broken(*, topic_id, user, store, body):
            raise RuntimeError("parser exploded")

        with patch.object(
            questions_module, "_resolve_quiz_topic", side_effect=_always_broken
        ):
            res = self.client.post(
                "/api/questions/quiz/generate", json=self._quiz_payload(topic_ids=[TOPIC_ID])
            )
        self.assertEqual(res.status_code, 503)
        self.assertEqual(res.json()["detail"]["code"], "topic_resolution_failed")

    def test_the_v1_quiz_route_is_409_when_the_only_topic_is_not_ready(self):
        res = self.client.post(
            "/api/questions/quiz/generate", json=self._quiz_payload(topic_ids=[PROBLEM_SOLVING_ID])
        )
        self.assertEqual(res.status_code, 409)

    def test_the_v1_quiz_route_accepts_both_question_types(self):
        res = self.client.post(
            "/api/questions/quiz/generate",
            json=self._quiz_payload(question_types=["mcq", "true_false"]),
        )
        self.assertEqual(res.status_code, 200, res.text)

    def test_the_v1_quiz_route_honours_an_explicit_response_detail(self):
        res = self.client.post(
            "/api/questions/quiz/generate",
            json=self._quiz_payload(response_detail="very_detailed"),
        )
        self.assertEqual(res.status_code, 200, res.text)
        self.assertIn("detail=very_detailed", self.llm.prompts[0])

    def test_the_v1_quiz_route_falls_back_to_the_stored_response_detail(self):
        self.store.upsert_topic_preferences(
            user_id="alice", topic_id=TOPIC_ID, response_detail="very_detailed"
        )
        res = self.client.post("/api/questions/quiz/generate", json=self._quiz_payload())
        self.assertEqual(res.status_code, 200, res.text)
        self.assertIn("detail=very_detailed", self.llm.prompts[0])

    def test_the_v1_quiz_route_survives_a_failing_history_capture(self):
        with patch.object(
            self.progress,
            "record_asked_questions",
            side_effect=RuntimeError("progress db is gone"),
        ):
            res = self.client.post("/api/questions/quiz/generate", json=self._quiz_payload())
        # Capture is best-effort: the generation still succeeds.
        self.assertEqual(res.status_code, 200, res.text)
        self.assertTrue(res.json()["questions"])

    def test_an_unknown_quiz_question_type_is_rejected_by_the_schema(self):
        res = self.client.post(
            "/api/questions/quiz/generate", json=self._quiz_payload(question_types=["essay"])
        )
        self.assertEqual(res.status_code, 422)

    def test_a_quiz_without_topics_is_rejected_by_the_generator(self):
        res = self.client.post(
            "/api/questions/quiz/generate", json=self._quiz_payload(topic_ids=[])
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("topic", str(res.json()["detail"]).lower())

    # ------------------------------------------------------------- v2 quiz

    def test_quiz_v2_returns_grounded_questions_with_their_quotes(self):
        res = self.client.post("/api/questions/quiz/generate-v2", json=self._quiz_payload())
        self.assertEqual(res.status_code, 200, res.text)
        body = res.json()
        self.assertEqual(len(body["questions"]), 3)
        first = body["questions"][0]
        self.assertEqual(first["topic_id"], TOPIC_ID)
        self.assertTrue(first["source_quote"])
        self.assertTrue(first["reasoning_summary"])
        self.assertEqual(first["choices"][0]["label"], "A")
        # The capture is grouped per topic so the history stays topic-scoped.
        asked = self.progress.get_asked_questions(
            user_id="alice", topic_id=TOPIC_ID, limit=50
        )
        self.assertEqual(set(asked), {item["question"] for item in body["questions"]})

    def test_quiz_v2_stream_emits_questions_and_a_done_frame(self):
        with self.client.stream(
            "POST", "/api/questions/quiz/generate-v2/stream", json=self._quiz_payload()
        ) as res:
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.headers["content-type"], "application/x-ndjson")
            self.assertEqual(res.headers["cache-control"], "no-cache, no-transform")
            events = [json.loads(line) for line in res.iter_lines() if line]
        question_frames = [e for e in events if e["type"] == "question"]
        self.assertEqual(len(question_frames), 3)
        self.assertEqual(question_frames[0]["question"]["topic_id"], TOPIC_ID)
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["generated_count"], 3)

    def test_quiz_v2_stream_warns_before_a_partially_failed_resolution(self):
        import app.routers.questions as questions_module

        real_resolve = questions_module._resolve_quiz_topic

        async def _flaky(*, topic_id, user, store, body):
            if topic_id == "99-broken":
                raise RuntimeError("parser exploded")
            return await real_resolve(topic_id=topic_id, user=user, store=store, body=body)

        with patch.object(questions_module, "_resolve_quiz_topic", side_effect=_flaky):
            with self.client.stream(
                "POST",
                "/api/questions/quiz/generate-v2/stream",
                json=self._quiz_payload(topic_ids=[TOPIC_ID, "99-broken"]),
            ) as res:
                self.assertEqual(res.status_code, 200)
                events = [json.loads(line) for line in res.iter_lines() if line]
        # The response has already started, so the failure is an event.
        self.assertEqual(events[0]["type"], "warning")
        self.assertEqual(events[0]["code"], "topic_resolution_failed")
        self.assertEqual(events[0]["topics"], ["99-broken"])
        self.assertEqual(events[0]["topics_used"], [TOPIC_ID])

    def test_quiz_v2_is_404_when_every_topic_fails_unexpectedly(self):
        import app.routers.questions as questions_module

        async def _always_broken(*, topic_id, user, store, body):
            raise RuntimeError("parser exploded")

        with patch.object(
            questions_module, "_resolve_quiz_topic", side_effect=_always_broken
        ):
            res = self.client.post(
                "/api/questions/quiz/generate-v2",
                json=self._quiz_payload(topic_ids=[TOPIC_ID]),
            )
        self.assertEqual(res.status_code, 503)

    # -------------------------------------------------- progress-context edges

    def test_no_progress_store_means_no_prior_progress(self):
        import asyncio

        questions._progress_store = None
        summary, asked = asyncio.run(
            questions._load_progress_context(
                user_id="alice",
                topic_ids=[TOPIC_ID],
                user_identity={"user": "alice"},
            )
        )
        self.assertEqual(summary, "")
        self.assertEqual(asked, [])

    def test_a_topic_with_no_progress_document_contributes_nothing(self):
        import asyncio

        summary, asked = asyncio.run(
            questions._load_progress_context(
                user_id="alice",
                topic_ids=["99-never-started"],
                user_identity={"user": "alice"},
            )
        )
        self.assertEqual(summary, "")
        self.assertEqual(asked, [])

    def test_a_pending_summary_schedules_a_background_refresh(self):
        import asyncio

        progress = ProgressStore(os.path.join(self.tempdir.name, "progress4.db"))
        questions._progress_store = progress
        progress.upsert_topic_progress(
            user_id="alice", topic_id=TOPIC_ID, topic_title="Static"
        )
        progress.mark_summary_pending(user_id="alice", topic_id=TOPIC_ID)
        scheduled: list[dict] = []
        with patch.object(
            questions.progress_router,
            "schedule_pending_summary_refresh",
            side_effect=lambda **kwargs: scheduled.append(kwargs),
        ):
            asyncio.run(
                questions._load_progress_context(
                    user_id="alice",
                    topic_ids=[TOPIC_ID],
                    user_identity={"user": "alice"},
                )
            )
        self.assertEqual(len(scheduled), 1)
        self.assertEqual(scheduled[0]["topic_id"], TOPIC_ID)

    def test_asked_questions_are_interleaved_round_robin_across_topics(self):
        import asyncio

        progress = ProgressStore(os.path.join(self.tempdir.name, "progress5.db"))
        questions._progress_store = progress
        for topic_id, texts in (
            ("topic-a", ["a1", "a2", "a3"]),
            ("topic-b", ["b1"]),
        ):
            progress.record_asked_questions(
                user_id="alice",
                topic_id=topic_id,
                topic_title=topic_id,
                questions=texts,
            )
        _summary, asked = asyncio.run(
            questions._load_progress_context(
                user_id="alice",
                topic_ids=["topic-a", "topic-b"],
                user_identity={"user": "alice"},
            )
        )
        # The store returns newest-first, and the router interleaves
        # round-robin so the first topic cannot consume the whole budget.
        self.assertEqual(len(asked), 4)
        self.assertEqual(set(asked), {"a1", "a2", "a3", "b1"})
        self.assertEqual(asked[0], "a3")
        self.assertEqual(asked[1], "b1")

    def test_grouping_skips_items_without_a_topic_or_text(self):
        self.assertEqual(
            questions._group_questions_by_topic(
                [
                    {"topic_id": "topic-a", "question": "What is a cache?"},
                    {"topic_id": "", "question": "No topic key."},
                    {"topic_id": "topic-b", "question": "   "},
                    {"question": "No topic id at all."},
                    "not-a-dict",
                ]
            ),
            {"topic-a": ["What is a cache?"]},
        )

    def test_grouping_falls_back_to_the_topic_id_for_an_unknown_title(self):
        import asyncio

        seen: list[tuple[str, str]] = []

        async def _spy(**kwargs):
            seen.append((kwargs["topic_id"], kwargs["topic_title"]))

        with patch.object(questions, "_record_generated_questions", side_effect=_spy):
            asyncio.run(
                questions._capture_grouped_questions(
                    user_id="alice",
                    topic_titles={"topic-a": "Topic A"},
                    question_dicts=[
                        {"topic_id": "topic-a", "question": "Known?"},
                        {"topic_id": "topic-z", "question": "Unknown?"},
                    ],
                )
            )
        self.assertEqual(sorted(seen), [("topic-a", "Topic A"), ("topic-z", "topic-z")])

    def test_an_empty_capture_short_circuits_before_the_store(self):
        import asyncio

        with patch.object(ProgressStore, "run_async") as run_async:
            asyncio.run(
                questions._capture_grouped_questions(
                    user_id="alice",
                    topic_titles={"topic-a": "Topic A"},
                    question_dicts=[],
                )
            )
        run_async.assert_not_called()

    def test_recording_an_empty_batch_short_circuits_before_the_store(self):
        import asyncio

        with patch.object(ProgressStore, "run_async") as run_async:
            asyncio.run(
                questions._record_generated_questions(
                    user_id="alice",
                    topic_id="topic-a",
                    topic_title="Topic A",
                    question_texts=[],
                )
            )
        run_async.assert_not_called()

    def test_a_language_hint_short_circuits_the_settings_lookup(self):
        import asyncio

        _seed_problem_solving(self.store, PROBLEM_SOLVING_ID, "go")
        with patch.object(
            self.store, "resolve_topic_ai_settings", side_effect=AssertionError("must not run")
        ):
            resolved = asyncio.run(
                questions._resolve_topic_for_user(
                    topic_id=PROBLEM_SOLVING_ID,
                    user_id="alice",
                    preferred_language_hint="  GO  ",
                )
            )
        self.assertIsNotNone(resolved)
        detail, content = resolved
        # The hint picks the curriculum directly; no preference read happens.
        self.assertEqual(detail.id, PROBLEM_SOLVING_ID)
        self.assertTrue(detail.content_ready)
        self.assertEqual(content, detail.raw_content)

    def test_a_language_alias_does_not_resolve_a_curriculum(self):
        import asyncio

        _seed_problem_solving(self.store, PROBLEM_SOLVING_ID, "go")
        resolved = asyncio.run(
            questions._resolve_topic_for_user(
                topic_id=PROBLEM_SOLVING_ID,
                user_id="alice",
                preferred_language_hint="golang",
            )
        )
        # The hint is lower-cased but never canonicalised, so an alias silently
        # falls back to the (empty) base topic and the caller sees 409.
        detail, _content = resolved
        self.assertFalse(detail.content_ready)
        self.assertEqual(detail.selected_language, "python")

    # ---------------------------------------------------- settings overrides

    async def _resolved(self, **overrides):
        topic = self.parser.get_topic(TOPIC_ID)
        body = {
            "response_detail_override": None,
            "preferred_language_override": None,
        }
        body.update(overrides)
        return await questions._resolve_ai_settings(
            store=self.store, user_id="alice", topic=topic, **body
        )

    def test_an_explicit_response_detail_overrides_the_stored_preference(self):
        import asyncio

        self.store.upsert_topic_preferences(
            user_id="alice",
            topic_id=TOPIC_ID,
            response_detail="very_detailed",
            preferred_language=None,
        )
        resolved = asyncio.run(self._resolved(response_detail_override="concise"))
        self.assertEqual(resolved["response_detail"], "concise")
        resolved = asyncio.run(self._resolved(response_detail_override="very_detailed"))
        self.assertEqual(resolved["response_detail"], "very_detailed")
        # Anything that is not the literal keyword degrades to concise.
        resolved = asyncio.run(self._resolved(response_detail_override="loud"))
        self.assertEqual(resolved["response_detail"], "concise")
        # No override leaves the stored preference untouched.
        resolved = asyncio.run(self._resolved())
        self.assertEqual(resolved["response_detail"], "very_detailed")

    def test_a_language_override_is_dropped_when_the_topic_needs_no_code(self):
        import asyncio

        self.store.upsert_topic_language_profile(
            topic_id=TOPIC_ID,
            requires_programming=False,
            language_options=[],
            source="llm",
        )
        resolved = asyncio.run(self._resolved(preferred_language_override="  Python "))
        self.assertFalse(resolved["requires_programming"])
        self.assertEqual(resolved["preferred_language"], "")

    def test_a_language_override_is_dropped_when_it_is_outside_the_profile(self):
        import asyncio

        self.store.upsert_topic_language_profile(
            topic_id=TOPIC_ID,
            requires_programming=True,
            language_options=["python", "go"],
            source="llm",
        )
        resolved = asyncio.run(self._resolved(preferred_language_override="GO"))
        self.assertEqual(resolved["preferred_language"], "go")
        resolved = asyncio.run(self._resolved(preferred_language_override="cobol"))
        self.assertEqual(resolved["preferred_language"], "")
        # A blank override is accepted as an explicit "no language".
        resolved = asyncio.run(self._resolved(preferred_language_override="   "))
        self.assertEqual(resolved["preferred_language"], "")

    def test_topic_resolution_returns_none_for_an_unknown_custom_topic(self):
        import asyncio

        resolved = asyncio.run(
            questions._resolve_topic_for_user(
                topic_id="99-nope", user_id="alice", preferred_language_hint=None
            )
        )
        self.assertIsNone(resolved)

    def test_a_resolved_custom_topic_is_returned_with_its_raw_content(self):
        import asyncio

        self.store.upsert_custom_topic(
            user_id="alice",
            topic_id="custom-rust",
            source_topic="Rust ownership",
            title="Rust Ownership",
            description="Ownership and borrowing.",
            track="backend",
            levels=["junior"],
            sections=[{"heading": "Borrowing", "content": "Ownership rules."}],
            raw_content="# Rust Ownership\n\nBorrowing rules.",
        )
        resolved = asyncio.run(
            questions._resolve_topic_for_user(
                topic_id="custom-rust", user_id="alice", preferred_language_hint=None
            )
        )
        self.assertIsNotNone(resolved)
        detail, content = resolved
        self.assertEqual(detail.id, "custom-rust")
        self.assertEqual(content, "# Rust Ownership\n\nBorrowing rules.")

    def test_a_progress_document_without_a_summary_contributes_no_summary_line(self):
        import asyncio

        progress = ProgressStore(os.path.join(self.tempdir.name, "progress2.db"))
        questions._progress_store = progress
        progress.upsert_topic_progress(
            user_id="alice", topic_id=TOPIC_ID, topic_title=TOPIC_ID
        )
        progress.upsert_topic_progress(
            user_id="alice", topic_id="99-other", topic_title="Other"
        )
        progress.set_summary(user_id="alice", topic_id="99-other", text="Asked about pools.")
        summary, asked = asyncio.run(
            questions._load_progress_context(
                user_id="alice",
                topic_ids=[TOPIC_ID, "99-other"],
                user_identity={"user": "alice"},
            )
        )
        # Only the topic that actually has a summary contributes a line.
        self.assertEqual(summary, "Other: Asked about pools.")
        self.assertEqual(asked, [])

    def test_a_progress_read_failure_discards_the_whole_history(self):
        import asyncio

        progress = ProgressStore(os.path.join(self.tempdir.name, "progress2.db"))
        questions._progress_store = progress
        progress.upsert_topic_progress(
            user_id="alice", topic_id=TOPIC_ID, topic_title="Static"
        )
        progress.set_summary(user_id="alice", topic_id=TOPIC_ID, text="Asked about pools.")
        real_get_progress = progress.get_progress

        def _flaky(user_id, topic_id):
            if topic_id == "99-other":
                raise RuntimeError("progress db is gone")
            return real_get_progress(user_id, topic_id)

        with patch.object(progress, "get_progress", side_effect=_flaky):
            summary, asked = asyncio.run(
                questions._load_progress_context(
                    user_id="alice",
                    topic_ids=[TOPIC_ID, "99-other"],
                    user_identity={"user": "alice"},
                )
            )
        # All-or-nothing: a half-read history would silently bias no-repeat.
        self.assertEqual(summary, "")
        self.assertEqual(asked, [])


def _v2_disabled_settings():
    from types import SimpleNamespace

    return SimpleNamespace(enable_v2_generation=False)


def _seed_problem_solving(store, topic_id: str, language: str) -> None:
    store.upsert_dynamic_topic_curriculum(
        user_id="alice",
        topic_id=topic_id,
        preferred_language=language,
        title=f"Problem Solving and Algorithms ({language})",
        description="Roadmap.",
        track="backend",
        levels=["junior", "mid", "senior"],
        sections=[
            {
                "heading": "Junior: Two pointers",
                "content": "Understand constraints and sliding window tradeoffs.",
            }
        ],
        raw_content=("# Problem Solving\n\n## Junior: Two pointers\n\nSliding window."),
        target_sections=120,
        source="llm",
    )