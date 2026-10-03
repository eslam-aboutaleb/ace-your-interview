"""Coverage for ``app.routers.interview_sessions`` — guards, error events, report.

A real ``InterviewStore`` on a temp database plus the real ``InterviewGenerator``
keep response models honest; policy/failure paths swap in a stub generator so
no LLM provider is contacted.
"""

import json
import os
import tempfile
import unittest
from typing import Any
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import require_auth
from app.routers import interview_sessions
from app.services.doc_parser import DocParser
from app.services.interview_store import TurnConflictError
from app.services.learning_store import LearningStore
from app.services.llm_policy import (
    LLMServiceApprovalRequiredError,
    PersonalCredentialRequiredError,
    StudyAppLLMNotAssignedError,
)

USER = {"user": "test-user", "provider": "local"}


class FakeLLM:
    """Enough of a provider double for the real ``InterviewGenerator``."""

    def __init__(self):
        self.n = 0
        self._bank = [
            "How would you design cursor-based pagination so ordering stays stable across writes?",
            "What strategy makes a retryable write idempotent across network failures?",
            "How would you detect and mitigate cache stampede on a high-traffic endpoint?",
            "How would you design rate limiting for bursty tenants while keeping fairness?",
            "Explain how you would roll back a bad database migration without downtime?",
        ]

    def _question_payload(self) -> dict[str, Any]:
        question = self._bank[(self.n - 1) % len(self._bank)]
        return {
            "question": question,
            "competency_focus": "tradeoff analysis",
            "expected_signals": ["constraints", "consistency", "client impact"],
        }

    async def completion(self, prompt, llm_config=None, user_identity=None, task=None, **kwargs):
        if "Generate the NEXT interview question" in prompt:
            self.n += 1
            return {
                "success": True,
                "analysis": json.dumps(self._question_payload()),
                "metadata": {},
                "error_code": "",
                "finish_reason": "stop",
            }
        return {
            "success": True,
            "analysis": '{"rubric":{"technical_accuracy":4,"reasoning_depth":4,'
            '"communication_clarity":4,"completeness":4,"confidence_signal":4,'
            '"overall":80},"strengths":["Clear reasoning"],'
            '"improvements":["Add metrics"],'
            '"follow_up_note":"Quantify the latency win."}',
            "metadata": {},
            "error_code": "",
            "finish_reason": "stop",
        }


class StubGenerator:
    """Generator double whose question/eval outcome is scripted per test."""

    def __init__(self, *, question_error=None, eval_error=None, report=None):
        self.question_error = question_error
        self.eval_error = eval_error
        self.report = report or {"overall_score": 77.0}
        self.calls: list[dict] = []

    async def generate_question(self, *, session, turns, llm_config, user_identity):
        self.calls.append({"kind": "question", "turns": len(turns)})
        if self.question_error is not None:
            raise self.question_error
        return {
            "question": "stub question",
            "competency_focus": "design",
            "expected_signals": ["clarity"],
        }

    async def evaluate_answer(self, *, session, question, user_answer, turn_index,
                              llm_config, user_identity, hint_level=0):
        self.calls.append({"kind": "evaluate", "turn_index": turn_index})
        if self.eval_error is not None:
            raise self.eval_error
        return {
            "rubric": {
                "technical_accuracy": 4,
                "reasoning_depth": 4,
                "communication_clarity": 4,
                "completeness": 4,
                "confidence_signal": 4,
                "overall": 80,
            },
            "strengths": ["Clear reasoning"],
            "improvements": ["Add metrics"],
            "follow_up_note": "Quantify the latency win.",
            "degraded": False,
        }

    def build_report(self, *, session, turns):
        return self.report


class InterviewRouterCoverageTestBase(unittest.TestCase):
    def setUp(self):
        self.prev_flag = os.environ.get("STUDY_ENABLE_MOCK_INTERVIEW_V1")
        os.environ["STUDY_ENABLE_MOCK_INTERVIEW_V1"] = "true"
        get_settings.cache_clear()

        self.tmpdir = tempfile.TemporaryDirectory()
        self.app = FastAPI()
        self.app.include_router(interview_sessions.router)
        self.app.dependency_overrides[require_auth] = lambda: dict(USER)
        self.client = TestClient(self.app)
        self._prev_globals = (
            interview_sessions._store,
            interview_sessions._generator,
            interview_sessions._parser,
            interview_sessions._learning_store,
        )

        self.parser = DocParser(docs_path=os.path.join(os.getcwd(), "docs"))
        self.learning_store = LearningStore(os.path.join(self.tmpdir.name, "learning.db"))
        interview_sessions.init(
            FakeLLM(),
            self.parser,
            self.learning_store,
            os.path.join(self.tmpdir.name, "interview.db"),
        )
        self.store = interview_sessions._store

    def tearDown(self):
        (
            interview_sessions._store,
            interview_sessions._generator,
            interview_sessions._parser,
            interview_sessions._learning_store,
        ) = self._prev_globals
        self.app.dependency_overrides.clear()
        self.tmpdir.cleanup()
        if self.prev_flag is None:
            os.environ.pop("STUDY_ENABLE_MOCK_INTERVIEW_V1", None)
        else:
            os.environ["STUDY_ENABLE_MOCK_INTERVIEW_V1"] = self.prev_flag
        get_settings.cache_clear()

    def _create(self, turn_count: int = 2) -> str:
        response = self.client.post(
            "/api/interview-sessions",
            json={
                "track": "backend",
                "level": "mid",
                "interview_type": "technical",
                "turn_count": turn_count,
                "target_role": "Backend Engineer",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["session"]["session_id"]

    def _stream_events(self, path: str, payload: dict) -> tuple[int, list[dict]]:
        with self.client.stream("POST", path, json=payload) as response:
            events = []
            for line in response.iter_lines():
                if not line:
                    continue
                text = line.decode("utf-8") if isinstance(line, bytes) else line
                events.append(json.loads(text))
            return response.status_code, events


class FeatureFlagAndWiringTests(InterviewRouterCoverageTestBase):
    def test_disabling_the_feature_hides_every_endpoint_with_404(self):
        os.environ["STUDY_ENABLE_MOCK_INTERVIEW_V1"] = "false"
        get_settings.cache_clear()
        for method, path, body in (
            ("post", "/api/interview-sessions", {"track": "backend"}),
            ("post", "/api/interview-sessions/stream", {"track": "backend"}),
            ("get", "/api/interview-sessions", None),
            ("get", "/api/interview-sessions/stats", None),
            ("get", "/api/interview-sessions/trends", None),
            ("get", "/api/interview-sessions/abc", None),
            ("get", "/api/interview-sessions/abc/report", None),
            ("post", "/api/interview-sessions/abc/answer", {"user_answer": "hi"}),
            ("post", "/api/interview-sessions/abc/next-question", {}),
        ):
            with self.subTest(path=path):
                kwargs = {"json": body} if body is not None else {}
                response = getattr(self.client, method)(path, **kwargs)
                self.assertEqual(response.status_code, 404)
                self.assertEqual(response.json()["detail"], "Mock interviews disabled")

    def test_missing_wiring_reports_services_not_initialised(self):
        interview_sessions._store = None
        response = self.client.get("/api/interview-sessions")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"], "Mock interview services not initialised"
        )

    def test_missing_parser_alone_also_blocks_requests(self):
        interview_sessions._parser = None
        response = self.client.get("/api/interview-sessions/stats")
        self.assertEqual(response.status_code, 503)

    def test_missing_generator_alone_also_blocks_requests(self):
        interview_sessions._generator = None
        response = self.client.get("/api/interview-sessions")
        self.assertEqual(response.status_code, 503)


class TopicTrackingTests(InterviewRouterCoverageTestBase):
    def test_no_parser_yields_an_empty_topic_id(self):
        interview_sessions._parser = None
        self.assertEqual(interview_sessions._track_topic_id("backend"), "")

    def test_an_empty_track_resolves_to_an_empty_topic_id(self):
        with patch.object(self.parser, "list_topics", return_value=[]):
            self.assertEqual(interview_sessions._track_topic_id("unknown-track"), "")

    def test_the_first_topic_of_the_track_is_used(self):
        with patch.object(
            self.parser,
            "list_topics",
            return_value=[type("T", (), {"id": "01-http"})()],
        ):
            self.assertEqual(
                interview_sessions._track_topic_id("backend"), "01-http"
            )


class SessionCreationTests(InterviewRouterCoverageTestBase):
    def test_create_persists_the_first_question(self):
        session_id = self._create()
        response = self.client.get(f"/api/interview-sessions/{session_id}")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["session"]["current_question"])

    def test_create_without_a_session_context_is_a_500(self):
        with patch.object(self.store, "get_session_context", return_value=None):
            response = self.client.post(
                "/api/interview-sessions", json={"track": "backend"}
            )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "Failed to create session context")

    def test_unknown_session_is_a_404(self):
        response = self.client.get("/api/interview-sessions/no-such-session")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Interview session not found")


class SessionStreamFailureTests(InterviewRouterCoverageTestBase):
    def test_stream_emits_generation_failed_when_the_context_is_missing(self):
        with patch.object(self.store, "get_session_context", return_value=None):
            status, events = self._stream_events(
                "/api/interview-sessions/stream", {"track": "backend"}
            )

        self.assertEqual(status, 200)
        self.assertEqual(events[0]["type"], "start")
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["code"], "generation_failed")
        self.assertIn("Failed to create session context", events[-1]["message"])

    def test_stream_maps_policy_blocks_to_their_error_code(self):
        interview_sessions._generator = StubGenerator(
            question_error=LLMServiceApprovalRequiredError("nope")
        )
        status, events = self._stream_events(
            "/api/interview-sessions/stream", {"track": "backend"}
        )

        self.assertEqual(status, 200)
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["code"], "llm_service_approval_required")
        self.assertIn("Admin approval required", events[-1]["message"])

    def test_stream_reports_a_bare_message_when_policy_detail_is_empty(self):
        interview_sessions._generator = StubGenerator(
            question_error=PersonalCredentialRequiredError("no credential")
        )
        _status, events = self._stream_events(
            "/api/interview-sessions/stream", {"track": "backend"}
        )
        self.assertEqual(events[-1]["code"], "personal_credential_required")

    def test_stream_falls_back_to_a_generic_message_for_an_empty_error(self):
        interview_sessions._generator = StubGenerator(
            question_error=StudyAppLLMNotAssignedError("")
        )
        _status, events = self._stream_events(
            "/api/interview-sessions/stream", {"track": "backend"}
        )
        self.assertEqual(events[-1]["code"], "study_app_llm_not_assigned")

    def test_stream_emits_generation_failed_for_an_unexpected_error(self):
        interview_sessions._generator = StubGenerator(
            question_error=ValueError("   ")
        )
        status, events = self._stream_events(
            "/api/interview-sessions/stream", {"track": "backend"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(events[-1]["code"], "generation_failed")
        # A blank error string still produces a usable message.
        self.assertEqual(events[-1]["message"], "Interview generation failed.")


class ListingAndStatsTests(InterviewRouterCoverageTestBase):
    def test_list_reflects_created_sessions(self):
        self._create()
        response = self.client.get("/api/interview-sessions?limit=5&offset=0")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total"], 1)
        self.assertEqual(payload["sessions"][0]["turns_completed"], 0)

    def test_list_pagination_bounds_are_validated(self):
        self.assertEqual(self.client.get("/api/interview-sessions?limit=0").status_code, 422)
        self.assertEqual(self.client.get("/api/interview-sessions?offset=-1").status_code, 422)

    def test_stats_start_at_zero_and_track_completions(self):
        empty = self.client.get("/api/interview-sessions/stats")
        self.assertEqual(empty.status_code, 200)
        self.assertEqual(empty.json()["total_sessions"], 0)

        session_id = self._create(turn_count=1)
        self.client.post(
            f"/api/interview-sessions/{session_id}/answer",
            json={"user_answer": "A first answer.", "response_time_ms": 1000},
        )

        stats = self.client.get("/api/interview-sessions/stats").json()
        self.assertEqual(stats["total_sessions"], 1)
        self.assertEqual(stats["completed_sessions"], 1)

    def test_trends_are_empty_until_a_session_completes(self):
        self._create(turn_count=1)
        trends = self.client.get("/api/interview-sessions/trends")
        self.assertEqual(trends.status_code, 200)
        self.assertEqual(trends.json()["points"], [])


class SubmitAnswerGuardTests(InterviewRouterCoverageTestBase):
    def test_unknown_session_is_a_404(self):
        response = self.client.post(
            "/api/interview-sessions/no-such/answer", json={"user_answer": "hi"}
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Interview session not found")

    def test_completed_session_is_a_400(self):
        session_id = self._create(turn_count=1)
        self.client.post(
            f"/api/interview-sessions/{session_id}/answer",
            json={"user_answer": "Answer.", "response_time_ms": 100},
        )
        response = self.client.post(
            f"/api/interview-sessions/{session_id}/answer",
            json={"user_answer": "Another answer."},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["detail"], "Interview session already completed"
        )

    def test_missing_current_question_is_a_400(self):
        session_id = self._create()
        original = self.store.get_session_context
        self.store.get_session_context = lambda **kwargs: {
            **original(**kwargs),
            "current_question": "   ",
        }
        try:
            response = self.client.post(
                f"/api/interview-sessions/{session_id}/answer",
                json={"user_answer": "Answer."},
            )
        finally:
            self.store.get_session_context = original

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["detail"],
            "No active question. Generate next question first.",
        )

    def test_already_recorded_turn_index_is_a_409(self):
        session_id = self._create(turn_count=3)
        original = self.store.turn_exists
        self.store.turn_exists = lambda **kwargs: True
        try:
            response = self.client.post(
                f"/api/interview-sessions/{session_id}/answer",
                json={"user_answer": "Answer."},
            )
        finally:
            self.store.turn_exists = original

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"]["code"], "turn_conflict")

    def test_record_turn_race_is_surfaced_as_409(self):
        session_id = self._create(turn_count=3)
        original = self.store.record_turn

        def _conflict(**kwargs):
            raise TurnConflictError("turn already recorded")

        self.store.record_turn = _conflict
        try:
            response = self.client.post(
                f"/api/interview-sessions/{session_id}/answer",
                json={"user_answer": "Answer."},
            )
        finally:
            self.store.record_turn = original

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"]["code"], "turn_conflict")

    def test_unreadable_session_after_write_is_a_500(self):
        session_id = self._create(turn_count=3)
        original = self.store.get_session
        self.store.get_session = lambda **kwargs: None
        try:
            response = self.client.post(
                f"/api/interview-sessions/{session_id}/answer",
                json={"user_answer": "Answer."},
            )
        finally:
            self.store.get_session = original

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "Session update failed")

    def test_a_successful_answer_updates_the_memory_summary(self):
        session_id = self._create(turn_count=3)
        self.client.post(
            f"/api/interview-sessions/{session_id}/answer",
            json={"user_answer": "Cursor pagination keeps ordering stable."},
        )
        context = self.store.get_session_context(
            user_id=USER["user"], session_id=session_id
        )
        self.assertIn("Cursor pagination", context["memory_summary"])
        self.assertIn("Turn 1 question", context["memory_summary"])


class UnmappedTrackTests(InterviewRouterCoverageTestBase):
    """A track with no curriculum topic must not reach the SRS store."""

    def setUp(self):
        super().setUp()
        self.attempted: list[dict] = []
        original = self.learning_store.record_attempt

        def _spy(**kwargs):
            self.attempted.append(kwargs)
            return original(**kwargs)

        self.learning_store.record_attempt = _spy  # type: ignore[method-assign]
        self.parser.list_topics = lambda **kwargs: []  # type: ignore[method-assign]

    def test_sync_answer_skips_record_attempt_when_the_track_has_no_topic(self):
        session_id = self._create(turn_count=3)
        response = self.client.post(
            f"/api/interview-sessions/{session_id}/answer",
            json={"user_answer": "An answer."},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["turn"]["turn_index"], 1)
        self.assertEqual(self.attempted, [])

    def test_stream_answer_skips_record_attempt_when_the_track_has_no_topic(self):
        session_id = self._create(turn_count=3)
        status, events = self._stream_events(
            f"/api/interview-sessions/{session_id}/answer/stream",
            {"user_answer": "An answer."},
        )

        self.assertEqual(status, 200)
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["turn"]["turn_index"], 1)
        self.assertEqual(self.attempted, [])


class SubmitAnswerStreamFailureTests(InterviewRouterCoverageTestBase):
    def test_unknown_session_is_a_404(self):
        response = self.client.post(
            "/api/interview-sessions/no-such/answer/stream", json={"user_answer": "hi"}
        )
        self.assertEqual(response.status_code, 404)

    def test_completed_session_is_a_400(self):
        session_id = self._create(turn_count=1)
        self.client.post(
            f"/api/interview-sessions/{session_id}/answer",
            json={"user_answer": "Answer."},
        )
        response = self.client.post(
            f"/api/interview-sessions/{session_id}/answer/stream",
            json={"user_answer": "Another."},
        )
        self.assertEqual(response.status_code, 400)

    def test_missing_current_question_is_a_400(self):
        session_id = self._create()
        original = self.store.get_session_context
        self.store.get_session_context = lambda **kwargs: {
            **original(**kwargs),
            "current_question": None,
        }
        try:
            response = self.client.post(
                f"/api/interview-sessions/{session_id}/answer/stream",
                json={"user_answer": "Answer."},
            )
        finally:
            self.store.get_session_context = original
        self.assertEqual(response.status_code, 400)

    def test_policy_block_is_reported_as_an_error_event(self):
        session_id = self._create(turn_count=3)
        interview_sessions._generator = StubGenerator(
            eval_error=LLMServiceApprovalRequiredError("nope")
        )
        status, events = self._stream_events(
            f"/api/interview-sessions/{session_id}/answer/stream",
            {"user_answer": "Answer."},
        )

        self.assertEqual(status, 200)
        self.assertEqual(events[0]["type"], "start")
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["code"], "llm_service_approval_required")

    def test_session_write_failure_becomes_a_generic_error_event(self):
        session_id = self._create(turn_count=3)
        original = self.store.get_session
        self.store.get_session = lambda **kwargs: None
        try:
            status, events = self._stream_events(
                f"/api/interview-sessions/{session_id}/answer/stream",
                {"user_answer": "Answer."},
            )
        finally:
            self.store.get_session = original

        self.assertEqual(status, 200)
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["code"], "generation_failed")
        self.assertEqual(events[-1]["message"], "Session update failed")

    def test_unexpected_evaluation_error_becomes_a_generic_error_event(self):
        session_id = self._create(turn_count=3)
        interview_sessions._generator = StubGenerator(
            eval_error=RuntimeError("provider exploded")
        )
        status, events = self._stream_events(
            f"/api/interview-sessions/{session_id}/answer/stream",
            {"user_answer": "Answer."},
        )

        self.assertEqual(status, 200)
        self.assertEqual(events[-1]["code"], "generation_failed")
        self.assertEqual(events[-1]["message"], "provider exploded")


class NextQuestionTests(InterviewRouterCoverageTestBase):
    def test_unknown_session_is_a_404(self):
        response = self.client.post(
            "/api/interview-sessions/no-such/next-question", json={}
        )
        self.assertEqual(response.status_code, 404)

    def test_completed_session_is_a_400(self):
        session_id = self._create(turn_count=1)
        self.client.post(
            f"/api/interview-sessions/{session_id}/answer",
            json={"user_answer": "Answer."},
        )
        response = self.client.post(
            f"/api/interview-sessions/{session_id}/next-question", json={}
        )
        self.assertEqual(response.status_code, 400)

    def test_unpersistable_question_is_a_500(self):
        session_id = self._create()
        original = self.store.set_current_question
        self.store.set_current_question = lambda **kwargs: None
        try:
            response = self.client.post(
                f"/api/interview-sessions/{session_id}/next-question", json={}
            )
        finally:
            self.store.set_current_question = original

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "Could not persist next question")

    def test_next_question_advances_the_turn_index(self):
        session_id = self._create(turn_count=5)
        response = self.client.post(
            f"/api/interview-sessions/{session_id}/next-question", json={}
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["session_id"], session_id)
        self.assertEqual(payload["turn_index"], 1)
        self.assertIn("competency_focus", payload)
        self.assertIn("expected_signals", payload)


class NextQuestionStreamFailureTests(InterviewRouterCoverageTestBase):
    def test_unknown_session_is_a_404(self):
        response = self.client.post(
            "/api/interview-sessions/no-such/next-question/stream", json={}
        )
        self.assertEqual(response.status_code, 404)

    def test_completed_session_is_a_400(self):
        session_id = self._create(turn_count=1)
        self.client.post(
            f"/api/interview-sessions/{session_id}/answer",
            json={"user_answer": "Answer."},
        )
        response = self.client.post(
            f"/api/interview-sessions/{session_id}/next-question/stream", json={}
        )
        self.assertEqual(response.status_code, 400)

    def test_unpersistable_question_becomes_a_generic_error_event(self):
        session_id = self._create()
        original = self.store.set_current_question
        self.store.set_current_question = lambda **kwargs: None
        try:
            status, events = self._stream_events(
                f"/api/interview-sessions/{session_id}/next-question/stream", {}
            )
        finally:
            self.store.set_current_question = original

        self.assertEqual(status, 200)
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["code"], "generation_failed")
        self.assertIn("Could not persist next question", events[-1]["message"])

    def test_policy_block_is_reported_as_an_error_event(self):
        session_id = self._create()
        interview_sessions._generator = StubGenerator(
            question_error=StudyAppLLMNotAssignedError("no assignment")
        )
        status, events = self._stream_events(
            f"/api/interview-sessions/{session_id}/next-question/stream", {}
        )

        self.assertEqual(status, 200)
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["code"], "study_app_llm_not_assigned")
        self.assertIn("no admin assignment exists yet", events[-1]["message"])

    def test_unexpected_error_becomes_a_generic_error_event(self):
        session_id = self._create()
        interview_sessions._generator = StubGenerator(
            question_error=RuntimeError("boom")
        )
        status, events = self._stream_events(
            f"/api/interview-sessions/{session_id}/next-question/stream", {}
        )

        self.assertEqual(status, 200)
        self.assertEqual(events[-1]["code"], "generation_failed")
        self.assertEqual(events[-1]["message"], "boom")


class ReportTests(InterviewRouterCoverageTestBase):
    def test_unknown_session_report_is_a_404(self):
        response = self.client.get("/api/interview-sessions/no-such/report")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Interview session not found")

    def test_report_without_any_turn_is_a_400(self):
        session_id = self._create()
        response = self.client.get(f"/api/interview-sessions/{session_id}/report")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "No completed turns for report")

    def test_report_is_built_and_persisted_from_recorded_turns(self):
        session_id = self._create(turn_count=5)
        self.client.post(
            f"/api/interview-sessions/{session_id}/answer",
            json={"user_answer": "Cursor pagination keeps ordering stable."},
        )
        self.assertIsNone(
            self.store.get_report(user_id=USER["user"], session_id=session_id)
        )

        response = self.client.get(f"/api/interview-sessions/{session_id}/report")

        self.assertEqual(response.status_code, 200)
        self.assertIn("report", response.json())
        stored = self.store.get_report(user_id=USER["user"], session_id=session_id)
        self.assertIsNotNone(stored)

    def test_saved_report_is_returned_without_rebuilding(self):
        session_id = self._create(turn_count=5)
        self.client.post(
            f"/api/interview-sessions/{session_id}/answer",
            json={"user_answer": "An answer."},
        )
        first = self.client.get(f"/api/interview-sessions/{session_id}/report")
        self.assertEqual(first.status_code, 200)
        stored = first.json()["report"]

        generator = interview_sessions._generator
        generator.build_report = lambda **kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
            AssertionError("report must not be rebuilt when one is already stored")
        )
        try:
            second = self.client.get(f"/api/interview-sessions/{session_id}/report")
        finally:
            del generator.build_report

        self.assertEqual(second.status_code, 200)
        replayed = second.json()["report"]
        # Only the storage timestamp differs; the substance is identical.
        self.assertEqual(replayed["overall_score"], stored["overall_score"])
        self.assertEqual(replayed["completed_turns"], stored["completed_turns"])
        self.assertEqual(replayed["rubric_averages"], stored["rubric_averages"])


if __name__ == "__main__":
    unittest.main()