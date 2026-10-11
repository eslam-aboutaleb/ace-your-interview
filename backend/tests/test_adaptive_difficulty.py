"""Adaptive difficulty: history-derived level recommendation and resolution.

Covers the adaptive-difficulty surface: the store's level recommendation
over the learner's recent attempt history (hint usage and response time
included), the resolution rule that keeps an explicit client level
authoritative, and the generator/router wiring that surfaces the
auto-resolved level back to the client.
"""

import asyncio
import json
import os
import re
import tempfile
import unittest
from datetime import UTC, datetime, timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.dependencies import require_auth
from app.routers import questions
from app.schemas.models import (
    GenerateQuestionsRequest,
    GenerateQuizRequest,
    GenerateQuestionsV2Response,
)
from app.services import question_generator as qg
from app.services.doc_parser import DocParser
from app.services.hint_service import HintService
from app.services.learning_store import LearningStore

TOPIC_ID = "01-http"
TOPIC_TITLE = "HTTP Fundamentals"
TOPIC_CONTENT = "Keep-alive lets one TCP connection carry many requests."
# A question that grounds to the topic title tokens.
GROUNDED_QUESTION = "How does HTTP keep-alive caching change connection reuse?"


class _Store:
    """Temporary-file learning store, one per test."""

    def __enter__(self):
        self._td = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._td.name, "learning.db")
        self.learning = LearningStore(self.db_path)
        self.hints = HintService(None, self.db_path)
        return self

    def __exit__(self, *exc_info):
        self._td.cleanup()
        return False


def _record(store, **overrides):
    payload = dict(
        user_id="alice",
        question_id="q-keep-alive",
        topic_id=TOPIC_ID,
        user_answer="answer",
        is_correct=True,
        confidence=4,
        response_time_ms=1500,
        mode="quiz",
    )
    payload.update(overrides)
    return store.record_attempt(**payload)


def _age_older_attempts(store, *, older_than_days=1):
    """Push every existing attempt into the past so newer ones sort last."""
    now = datetime.now(UTC) - timedelta(days=older_than_days)
    with store._lock:
        with store._conn:
            store._conn.execute(
                "UPDATE learning_attempts SET created_at = ?", (now.isoformat(),)
            )


class LevelRecommendationTests(unittest.TestCase):
    def test_cold_start_recommends_nothing(self):
        with _Store() as stores:
            self.assertIsNone(
                stores.learning.get_level_recommendation("alice", TOPIC_ID)
            )

    def test_fluent_history_escalates_to_senior(self):
        with _Store() as stores:
            for index in range(10):
                _record(
                    stores.learning,
                    question_id=f"q-{index}",
                    is_correct=True,
                    confidence=4,
                    response_time_ms=1500,
                )
            self.assertEqual(
                stores.learning.get_level_recommendation("alice", TOPIC_ID),
                "senior",
            )

    def test_struggling_history_de_escalates_to_junior(self):
        with _Store() as stores:
            for index in range(10):
                _record(
                    stores.learning,
                    question_id=f"q-{index}",
                    is_correct=False,
                    confidence=2,
                    response_time_ms=90000,
                )
            self.assertEqual(
                stores.learning.get_level_recommendation("alice", TOPIC_ID),
                "junior",
            )

    def test_middling_history_stays_at_mid(self):
        with _Store() as stores:
            for index in range(10):
                _record(
                    stores.learning,
                    question_id=f"q-{index}",
                    is_correct=True,
                    confidence=3,
                    response_time_ms=30000,
                )
            self.assertEqual(
                stores.learning.get_level_recommendation("alice", TOPIC_ID),
                "mid",
            )

    def test_hint_usage_downgrades_the_recommendation(self):
        with _Store() as stores:
            for index in range(10):
                question_id = f"q-{index}"
                _record(
                    stores.learning,
                    question_id=question_id,
                    is_correct=True,
                    confidence=4,
                    response_time_ms=1500,
                )
                # The same fluent answers, but scaffolded: two hints per
                # question downgrade the rating (easy -> hard) and with it
                # the recommendation.
                stores.hints.record_hint_event("alice", question_id, 3)
                stores.hints.record_hint_event("alice", question_id, 4)
            self.assertEqual(
                stores.learning.get_level_recommendation("alice", TOPIC_ID),
                "mid",
            )

    def test_hint_events_for_other_questions_do_not_count(self):
        with _Store() as stores:
            for index in range(10):
                _record(
                    stores.learning,
                    question_id=f"q-{index}",
                    is_correct=True,
                    confidence=4,
                    response_time_ms=1500,
                )
            for index in range(10):
                stores.hints.record_hint_event("alice", f"other-{index}", 4)
            self.assertEqual(
                stores.learning.get_level_recommendation("alice", TOPIC_ID),
                "senior",
            )

    def test_response_time_moves_the_recommendation(self):
        with _Store() as stores:
            # Correct answers with the middle confidence: quick recall
            # escalates, an unhurried answer does not.
            for response_time in (1500, 30000):
                with self.subTest(response_time=response_time):
                    _age_older_attempts(stores.learning)
                    for index in range(10):
                        _record(
                            stores.learning,
                            question_id=f"q-{response_time}-{index}",
                            is_correct=True,
                            confidence=3,
                            response_time_ms=response_time,
                        )
                    self.assertEqual(
                        stores.learning.get_level_recommendation(
                            "alice", TOPIC_ID
                        ),
                        "senior" if response_time == 1500 else "mid",
                    )

    def test_slow_wrong_answers_drop_the_signal(self):
        with _Store() as stores:
            for index in range(5):
                _record(
                    stores.learning,
                    question_id=f"fast-{index}",
                    is_correct=True,
                    confidence=3,
                    response_time_ms=1500,
                )
            for index in range(5):
                _record(
                    stores.learning,
                    question_id=f"slow-{index}",
                    is_correct=False,
                    confidence=3,
                    response_time_ms=90000,
                )
            # 5 fluent answers against 5 struggled misses settles at the
            # middle level rather than escalating on the correct half.
            self.assertEqual(
                stores.learning.get_level_recommendation("alice", TOPIC_ID),
                "mid",
            )

    def test_history_is_scoped_to_user_and_topic(self):
        with _Store() as stores:
            for index in range(10):
                _record(
                    stores.learning,
                    question_id=f"q-{index}",
                    is_correct=False,
                    confidence=1,
                    response_time_ms=90000,
                )
            for index in range(10):
                _record(
                    stores.learning,
                    user_id="bob",
                    question_id=f"bob-{index}",
                    is_correct=True,
                    confidence=4,
                    response_time_ms=1500,
                )
                _record(
                    stores.learning,
                    question_id=f"other-topic-{index}",
                    topic_id="02-sql",
                    is_correct=True,
                    confidence=4,
                    response_time_ms=1500,
                )
            self.assertEqual(
                stores.learning.get_level_recommendation("alice", TOPIC_ID),
                "junior",
            )
            self.assertEqual(
                stores.learning.get_level_recommendation("bob", TOPIC_ID),
                "senior",
            )
            self.assertEqual(
                stores.learning.get_level_recommendation("alice", "02-sql"),
                "senior",
            )

    def test_only_the_most_recent_attempts_are_scored(self):
        with _Store() as stores:
            for index in range(10):
                _record(
                    stores.learning,
                    question_id=f"stale-{index}",
                    is_correct=False,
                    confidence=1,
                    response_time_ms=90000,
                )
            _age_older_attempts(stores.learning)
            for index in range(10):
                _record(
                    stores.learning,
                    question_id=f"fresh-{index}",
                    is_correct=True,
                    confidence=4,
                    response_time_ms=1500,
                )
            self.assertEqual(
                stores.learning.get_level_recommendation("alice", TOPIC_ID),
                "senior",
            )


class LevelSignalTests(unittest.TestCase):
    def test_fast_correct_answer_is_bumped(self):
        signal = LearningStore._level_signal(
            is_correct=True,
            confidence=4,
            hints_used=0,
            response_time_ms=1500,
        )
        self.assertGreater(signal, 3.0)

    def test_slow_wrong_answer_is_dropped(self):
        signal = LearningStore._level_signal(
            is_correct=False,
            confidence=4,
            hints_used=0,
            response_time_ms=90000,
        )
        self.assertLess(signal, 0.0)

    def test_mid_range_and_unknown_response_times_leave_the_band(self):
        band = LearningStore._level_signal(
            is_correct=True,
            confidence=3,
            hints_used=0,
            response_time_ms=30000,
        )
        unknown = LearningStore._level_signal(
            is_correct=True,
            confidence=3,
            hints_used=0,
            response_time_ms=0,
        )
        self.assertEqual(band, 2.0)
        self.assertEqual(unknown, 2.0)

    def test_each_hint_downgrades_the_band(self):
        unaided = LearningStore._level_signal(
            is_correct=True,
            confidence=5,
            hints_used=0,
            response_time_ms=0,
        )
        scaffolded = LearningStore._level_signal(
            is_correct=True,
            confidence=5,
            hints_used=2,
            response_time_ms=0,
        )
        self.assertEqual(unaided, 3.0)
        self.assertEqual(scaffolded, 1.0)


class ResolveLevelTests(unittest.TestCase):
    def test_explicit_levels_always_win(self):
        for level in ("junior", "mid", "senior"):
            self.assertEqual(qg.resolve_level(level, "senior"), level)

    def test_auto_and_missing_levels_defer_to_the_recommendation(self):
        self.assertEqual(qg.resolve_level(None, "junior"), "junior")
        self.assertEqual(qg.resolve_level("", "senior"), "senior")
        self.assertEqual(qg.resolve_level("auto", "mid"), "mid")
        self.assertEqual(qg.resolve_level("AUTO", "senior"), "senior")

    def test_missing_recommendation_falls_back_to_mid(self):
        self.assertEqual(qg.resolve_level(None, None), "mid")
        self.assertEqual(qg.resolve_level("auto", ""), "mid")
        self.assertEqual(qg.resolve_level("mid", None), "mid")

    def test_unknown_level_is_not_treated_as_a_recommendation(self):
        self.assertEqual(qg.resolve_level("expert", "senior"), "mid")

    def test_auto_detection(self):
        self.assertTrue(qg._level_is_auto(None))
        self.assertTrue(qg._level_is_auto(""))
        self.assertTrue(qg._level_is_auto("auto"))
        self.assertFalse(qg._level_is_auto("mid"))


class _FakeRunStore:
    """Run store exposing only the level-recommendation surface."""

    def __init__(self, recommendation=None, error=None):
        self.recommendation = recommendation
        self.error = error
        self.calls = []

    def get_level_recommendation(self, user_id, topic_id):
        self.calls.append((user_id, topic_id))
        if self.error is not None:
            raise self.error
        return self.recommendation


class _UnhelpfulRunStore:
    """Run store without any recommendation support at all."""


def _generator(run_store):
    return qg.QuestionGenerator(llm_client=None, run_store=run_store)


class GenerationLevelResolutionTests(unittest.TestCase):
    def _resolve(self, run_store, *, level=None, user=None, topic=TOPIC_ID):
        return _generator(run_store)._resolve_generation_level(
            level=level,
            user_identity=user,
            topic_id=topic,
        )

    def test_auto_level_uses_the_recommendation_and_reports_it(self):
        store = _FakeRunStore("senior")
        self.assertEqual(
            self._resolve(store, user={"user": "alice"}),
            ("senior", "senior"),
        )
        self.assertEqual(store.calls, [("alice", TOPIC_ID)])

    def test_explicit_level_wins_and_reports_nothing(self):
        store = _FakeRunStore("senior")
        self.assertEqual(
            self._resolve(store, level="mid", user={"user": "alice"}),
            ("mid", ""),
        )
        # An explicit level never consults the learner's history.
        self.assertEqual(store.calls, [])

    def test_no_user_context_means_no_recommendation(self):
        for user in (None, {}, {"user": ""}):
            with self.subTest(user=user):
                store = _FakeRunStore("senior")
                self.assertEqual(
                    self._resolve(store, user=user),
                    ("mid", "mid"),
                )
                self.assertEqual(store.calls, [])

    def test_no_topic_means_no_recommendation(self):
        store = _FakeRunStore("senior")
        self.assertEqual(
            self._resolve(store, user={"user": "alice"}, topic=""),
            ("mid", "mid"),
        )
        self.assertEqual(store.calls, [])

    def test_store_without_the_helper_is_ignored(self):
        self.assertEqual(
            self._resolve(_UnhelpfulRunStore(), user={"user": "alice"}),
            ("mid", "mid"),
        )

    def test_recommendation_failure_degrades_to_mid(self):
        store = _FakeRunStore(error=RuntimeError("learning db unavailable"))
        with self.assertLogs("app.services.question_generator", level="WARNING"):
            self.assertEqual(
                self._resolve(store, user={"user": "alice"}),
                ("mid", "mid"),
            )

    def test_off_contract_recommendation_is_ignored(self):
        self.assertEqual(
            self._resolve(_FakeRunStore("expert"), user={"user": "alice"}),
            ("mid", "mid"),
        )


class _EchoLevelLLM:
    """Returns one valid item whose level echoes the prompt's target level."""

    def __init__(self):
        self.prompts = []

    async def completion(self, prompt, llm_config=None, user_identity=None):
        self.prompts.append(prompt)
        level = re.search(r'Target candidate level: "(\w+)"', prompt)
        payload = [
            {
                "question": GROUNDED_QUESTION,
                "answer": (
                    "Keep-alive reuses one TCP connection across requests, "
                    "trading a held socket for lower latency."
                ),
                "difficulty": "medium",
                "source_quote": "Keep-alive lets one TCP connection carry many requests.",
                "target_level": level.group(1) if level else "mid",
            }
        ]
        return {
            "success": True,
            "analysis": json.dumps(payload),
            "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
        }


class _EchoLevelQuizLLM:
    """Returns one valid MCQ whose level echoes the prompt's target level."""

    def __init__(self):
        self.prompts = []

    async def completion(self, prompt, llm_config=None, user_identity=None):
        self.prompts.append(prompt)
        level = re.search(r'Target candidate level: "(\w+)"', prompt)
        payload = [
            {
                "question": GROUNDED_QUESTION,
                "type": "mcq",
                "choices": [
                    {"label": "A", "text": "Connection reuse"},
                    {"label": "B", "text": "Header compression"},
                    {"label": "C", "text": "TLS resumption"},
                    {"label": "D", "text": "Request pipelining"},
                ],
                "correct_answer": "A",
                "explanation": "Keep-alive reuses the connection for later requests.",
                "difficulty": "medium",
                "topic_id": TOPIC_ID,
                "source_quote": "Keep-alive lets one TCP connection carry many requests.",
                "reasoning_summary": "Reused connections avoid a fresh handshake.",
                "target_level": level.group(1) if level else "mid",
                "learning_objective": "Explain when keep-alive pays off.",
                "expected_answer": "A",
                "distractor_rationales": [
                    {"label": "A", "rationale": "Reuse is the documented benefit."},
                    {"label": "B", "rationale": "Compression is a separate HPACK concern."},
                    {"label": "C", "rationale": "Resumption happens at the TLS layer."},
                    {"label": "D", "rationale": "Pipelining is disabled in modern browsers."},
                ],
            }
        ]
        return {
            "success": True,
            "analysis": json.dumps(payload),
            "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
        }


def _fluent_history(store, topic_id=TOPIC_ID, count=10):
    for index in range(count):
        store.record_attempt(
            user_id="alice",
            question_id=f"q-{index}",
            topic_id=topic_id,
            user_answer="answer",
            is_correct=True,
            confidence=4,
            response_time_ms=1500,
            mode="quiz",
        )


class GeneratorResolvedLevelTests(unittest.TestCase):
    def test_generate_v2_resolves_and_reports_the_level(self):
        with _Store() as stores:
            _fluent_history(stores.learning)
            llm = _EchoLevelLLM()
            generator = qg.QuestionGenerator(llm, run_store=stores.learning)
            result = asyncio.run(
                generator.generate_v2(
                    topic_id=TOPIC_ID,
                    topic_title=TOPIC_TITLE,
                    doc_content=TOPIC_CONTENT,
                    count=1,
                    user_identity={"user": "alice"},
                )
            )
            self.assertIsInstance(result, GenerateQuestionsV2Response)
            self.assertEqual(result.resolved_level, "senior")
            self.assertIn('Target candidate level: "senior"', llm.prompts[0])

    def test_generate_v2_explicit_level_is_kept(self):
        with _Store() as stores:
            _fluent_history(stores.learning)
            llm = _EchoLevelLLM()
            generator = qg.QuestionGenerator(llm, run_store=stores.learning)
            result = asyncio.run(
                generator.generate_v2(
                    topic_id=TOPIC_ID,
                    topic_title=TOPIC_TITLE,
                    doc_content=TOPIC_CONTENT,
                    count=1,
                    level="junior",
                    user_identity={"user": "alice"},
                )
            )
            self.assertEqual(result.resolved_level, "")
            self.assertIn('Target candidate level: "junior"', llm.prompts[0])

    def test_generate_quiz_v2_uses_the_first_topics_history(self):
        with _Store() as stores:
            _fluent_history(stores.learning, topic_id="other-topic")
            _fluent_history(stores.learning, topic_id=TOPIC_ID)
            llm = _EchoLevelQuizLLM()
            generator = qg.QuestionGenerator(llm, run_store=stores.learning)
            result = asyncio.run(
                generator.generate_quiz_v2(
                    topics_content=[
                        {"id": TOPIC_ID, "title": TOPIC_TITLE, "content": TOPIC_CONTENT},
                        {"id": "other-topic", "title": "Other", "content": "Other content."},
                    ],
                    count=1,
                    user_identity={"user": "alice"},
                )
            )
            self.assertEqual(result.resolved_level, "senior")
            self.assertIn('Target candidate level: "senior"', llm.prompts[0])

    def test_generate_quiz_v2_without_history_defaults_to_mid(self):
        with _Store() as stores:
            llm = _EchoLevelQuizLLM()
            generator = qg.QuestionGenerator(llm, run_store=stores.learning)
            result = asyncio.run(
                generator.generate_quiz_v2(
                    topics_content=[
                        {"id": TOPIC_ID, "title": TOPIC_TITLE, "content": TOPIC_CONTENT}
                    ],
                    count=1,
                    user_identity={"user": "alice"},
                )
            )
            # Auto-resolved against an empty history: the mid default.
            self.assertEqual(result.resolved_level, "mid")
            self.assertIn('Target candidate level: "mid"', llm.prompts[0])


class QuestionsRouterResolvedLevelTests(unittest.TestCase):
    def setUp(self):
        self.prev = (
            questions._llm_client,
            questions._parser,
            questions._learning_store,
            questions._progress_store,
        )
        self.tempdir = tempfile.TemporaryDirectory()
        handbook = os.path.join(self.tempdir.name, "owner-handbook")
        os.makedirs(handbook, exist_ok=True)
        with open(os.path.join(handbook, f"{TOPIC_ID}.md"), "w", encoding="utf-8") as f:
            f.write(
                f"""---
track: backend
levels: [junior, mid, senior]
---
# {TOPIC_TITLE}

{TOPIC_CONTENT}
"""
            )
        self.learning_store = LearningStore(
            os.path.join(self.tempdir.name, "learning.db")
        )
        self.fake_llm = _EchoLevelLLM()
        questions.init(
            self.fake_llm,
            DocParser(docs_path=self.tempdir.name),
            self.learning_store,
            None,
            None,
        )
        self.app = FastAPI()
        self.app.dependency_overrides[require_auth] = (
            lambda: {"user": "alice", "provider": "local"}
        )
        self.app.include_router(questions.router)
        self.client = TestClient(self.app)

    def tearDown(self):
        (
            questions._llm_client,
            questions._parser,
            questions._learning_store,
            questions._progress_store,
        ) = self.prev
        self.tempdir.cleanup()

    def test_omitted_level_resolves_from_history(self):
        _fluent_history(self.learning_store)
        response = self.client.post(
            "/api/questions/generate-v2",
            json={"topic_id": TOPIC_ID, "count": 1},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["resolved_level"], "senior")
        self.assertIn('Target candidate level: "senior"', self.fake_llm.prompts[0])

    def test_explicit_level_is_not_resolved(self):
        _fluent_history(self.learning_store)
        response = self.client.post(
            "/api/questions/generate-v2",
            json={"topic_id": TOPIC_ID, "count": 1, "level": "mid"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["resolved_level"], "")
        self.assertIn('Target candidate level: "mid"', self.fake_llm.prompts[0])

    def test_streaming_done_event_carries_the_resolved_level(self):
        _fluent_history(self.learning_store)
        with self.client.stream(
            "POST",
            "/api/questions/generate-v2/stream",
            json={"topic_id": TOPIC_ID, "count": 1, "level": "auto"},
        ) as res:
            self.assertEqual(res.status_code, 200)
            events = [json.loads(line) for line in res.iter_lines() if line]
        done = [event for event in events if event.get("type") == "done"]
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0]["resolved_level"], "senior")


class RequestSchemaLevelTests(unittest.TestCase):
    def test_omitted_level_is_none(self):
        self.assertIsNone(GenerateQuestionsRequest(topic_id="x").level)
        self.assertIsNone(GenerateQuizRequest(topic_ids=["x"]).level)

    def test_auto_opt_in_is_accepted(self):
        self.assertEqual(
            GenerateQuestionsRequest(topic_id="x", level="auto").level, "auto"
        )
        self.assertEqual(
            GenerateQuizRequest(topic_ids=["x"], level="auto").level, "auto"
        )

    def test_explicit_levels_are_unchanged(self):
        for level in ("junior", "mid", "senior"):
            self.assertEqual(
                GenerateQuestionsRequest(topic_id="x", level=level).level, level
            )
