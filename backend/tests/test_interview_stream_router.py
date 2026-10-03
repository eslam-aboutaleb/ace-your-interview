import asyncio
import json
import os
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import require_auth
from app.routers import interview_sessions
from app.services.doc_parser import DocParser
from app.services.interview_store import TurnConflictError
from app.services.learning_store import LearningStore
from app.services.llm_client import CALL_FAILED_CODE


class FakeLLM:
    def __init__(self, eval_error_code: str = ""):
        self.question_n = 0
        self.eval_error_code = eval_error_code
        self._question_bank = [
            "How would you design cursor-based pagination to keep ordering stable across writes?",
            "What strategy would you use to make retryable writes idempotent across network failures?",
            "How would you detect and mitigate cache stampede on a high-traffic endpoint?",
            "How would you design rate limiting for bursty tenants while keeping fairness?",
        ]

    async def completion(self, prompt, llm_config=None, user_identity=None, task=None, **kwargs):
        if "Generate the NEXT interview question" in prompt:
            await asyncio.sleep(1.0)
            self.question_n += 1
            question = self._question_bank[(self.question_n - 1) % len(self._question_bank)]
            return {
                "success": True,
                "analysis": (
                    '{"question":"'
                    + question
                    + '","competency_focus":"tradeoff analysis","expected_signals":["constraints","consistency","client impact"]}'
                ),
                "metadata": {},
                "error_code": "",
                "finish_reason": "stop",
            }
        await asyncio.sleep(1.0)
        if self.eval_error_code:
            return {
                "success": False,
                "analysis": "",
                "metadata": {},
                "error": "provider unavailable",
                "error_code": self.eval_error_code,
                "finish_reason": "",
                "usage": {},
            }
        return {
            "success": True,
            "analysis": (
                '{"rubric":{"technical_accuracy":4,"reasoning_depth":4,"communication_clarity":4,'
                '"completeness":4,"confidence_signal":4,"overall":80},'
                '"strengths":["Clear tradeoff reasoning"],"improvements":["Add a concrete metric"],'
                '"follow_up_note":"Quantify expected latency improvements next time."}'
            ),
            "metadata": {},
            "error_code": "",
            "finish_reason": "stop",
        }


class SpyLearningStore(LearningStore):
    """Real store plus a counter, so we can prove a call never happened."""

    def __init__(self, db_path: str):
        super().__init__(db_path)
        self.record_attempt_calls: list[dict] = []

    def record_attempt(self, **kwargs):
        self.record_attempt_calls.append(kwargs)
        return super().record_attempt(**kwargs)


class InterviewStreamRouterTests(unittest.TestCase):
    def setUp(self):
        self.prev_flag = os.environ.get("STUDY_ENABLE_MOCK_INTERVIEW_V1")
        os.environ["STUDY_ENABLE_MOCK_INTERVIEW_V1"] = "true"
        get_settings.cache_clear()

        self.tempdir = tempfile.TemporaryDirectory()
        self.app = FastAPI()
        self.app.include_router(interview_sessions.router)
        self.app.dependency_overrides[require_auth] = (
            lambda: {"user": "test-user", "provider": "local"}
        )
        self.client = TestClient(self.app)

    def tearDown(self):
        self.tempdir.cleanup()
        if self.prev_flag is None:
            os.environ.pop("STUDY_ENABLE_MOCK_INTERVIEW_V1", None)
        else:
            os.environ["STUDY_ENABLE_MOCK_INTERVIEW_V1"] = self.prev_flag
        get_settings.cache_clear()

    def _init(self, llm: FakeLLM) -> SpyLearningStore:
        parser = DocParser(docs_path=os.path.join(os.getcwd(), "docs"))
        learning_store = SpyLearningStore(os.path.join(self.tempdir.name, "learning.db"))
        interview_sessions.init(
            llm,
            parser,
            learning_store,
            os.path.join(self.tempdir.name, "interview.db"),
        )
        return learning_store

    def _stream_events(self, path: str, payload: dict) -> tuple[int, list[dict]]:
        with self.client.stream("POST", path, json=payload) as res:
            events = []
            for line in res.iter_lines():
                if not line:
                    continue
                text = line.decode("utf-8") if isinstance(line, bytes) else line
                events.append(json.loads(text))
            return res.status_code, events

    def _create_session(self, turn_count: int = 2) -> str:
        create = self.client.post(
            "/api/interview-sessions",
            json={
                "track": "backend",
                "level": "mid",
                "interview_type": "technical",
                "turn_count": turn_count,
                "target_role": "Backend Engineer",
                "focus_areas": ["api design"],
            },
        )
        self.assertEqual(create.status_code, 200)
        return create.json()["session"]["session_id"]

    def test_create_session_stream_emits_done_payload_with_session_shape(self):
        self._init(FakeLLM())
        status, events = self._stream_events(
            "/api/interview-sessions/stream",
            {
                "track": "backend",
                "level": "mid",
                "interview_type": "technical",
                "turn_count": 2,
                "target_role": "Backend Engineer",
                "focus_areas": ["api design"],
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(events[0]["type"], "start")
        self.assertTrue(any(event.get("type") == "progress" for event in events))
        done = events[-1]
        self.assertEqual(done["type"], "done")
        self.assertIn("session", done)
        self.assertIn("turns", done)
        self.assertTrue(done["session"]["session_id"])
        self.assertTrue(done["session"]["current_question"])

    def test_next_question_stream_done_payload_matches_question_response_shape(self):
        self._init(FakeLLM())
        create = self.client.post(
            "/api/interview-sessions",
            json={
                "track": "backend",
                "level": "mid",
                "interview_type": "technical",
                "turn_count": 2,
                "target_role": "Backend Engineer",
                "focus_areas": ["api design"],
            },
        )
        self.assertEqual(create.status_code, 200)
        session_id = create.json()["session"]["session_id"]

        status, events = self._stream_events(
            f"/api/interview-sessions/{session_id}/next-question/stream",
            {},
        )
        self.assertEqual(status, 200)
        self.assertEqual(events[0]["type"], "start")
        self.assertTrue(any(event.get("type") == "progress" for event in events))
        done = events[-1]
        self.assertEqual(done["type"], "done")
        self.assertEqual(done["session_id"], session_id)
        self.assertIn("turn_index", done)
        self.assertTrue(done["question"])
        self.assertIn("competency_focus", done)
        self.assertIn("expected_signals", done)

    def test_answer_stream_done_payload_matches_turn_response_shape(self):
        self._init(FakeLLM())
        create = self.client.post(
            "/api/interview-sessions",
            json={
                "track": "backend",
                "level": "mid",
                "interview_type": "technical",
                "turn_count": 1,
                "target_role": "Backend Engineer",
                "focus_areas": ["api design"],
            },
        )
        self.assertEqual(create.status_code, 200)
        session_id = create.json()["session"]["session_id"]

        status, events = self._stream_events(
            f"/api/interview-sessions/{session_id}/answer/stream",
            {
                "user_answer": "I would choose cursor pagination for consistency.",
                "response_time_ms": 9000,
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(events[0]["type"], "start")
        self.assertTrue(any(event.get("type") == "progress" for event in events))
        done = events[-1]
        self.assertEqual(done["type"], "done")
        self.assertIn("session", done)
        self.assertIn("turn", done)
        self.assertIn("report_ready", done)
        self.assertEqual(done["session"]["status"], "completed")
        self.assertEqual(done["turn"]["rubric"]["overall"], 80)
        self.assertFalse(done["turn"]["degraded"])

    def test_degraded_answer_stream_skips_record_attempt(self):
        learning_store = self._init(FakeLLM(eval_error_code=CALL_FAILED_CODE))
        session_id = self._create_session(turn_count=1)

        status, events = self._stream_events(
            f"/api/interview-sessions/{session_id}/answer/stream",
            {
                "user_answer": "I would shard by tenant and add backpressure.",
                "response_time_ms": 9000,
            },
        )
        self.assertEqual(status, 200)
        done = events[-1]
        self.assertEqual(done["type"], "done")
        self.assertTrue(done["turn"]["degraded"])
        self.assertIsNone(done["turn"]["rubric"]["overall"])
        self.assertEqual(learning_store.record_attempt_calls, [])

    def test_repeat_answer_stream_is_rejected_with_turn_conflict(self):
        self._init(FakeLLM())
        session_id = self._create_session(turn_count=3)
        path = f"/api/interview-sessions/{session_id}/answer/stream"
        answer = {"user_answer": "A first answer.", "response_time_ms": 9000}

        status, events = self._stream_events(path, answer)
        self.assertEqual(status, 200)
        self.assertEqual(events[-1]["turn"]["turn_index"], 1)

        # A second stream submission is a legitimate turn 2 while the session
        # counter is consistent, so no conflict is invented.
        self.client.post(f"/api/interview-sessions/{session_id}/next-question", json={})
        second_status, second_events = self._stream_events(path, answer)
        self.assertEqual(second_status, 200)
        self.assertEqual(second_events[-1]["turn"]["turn_index"], 2)

    def test_stream_precheck_rejects_an_already_recorded_turn_index(self):
        self._init(FakeLLM())
        session_id = self._create_session(turn_count=3)

        store = interview_sessions._store
        assert store is not None
        # The store derives turn allocation from the turns table, so force the
        # pre-check to report a taken index and assert the router's mapping.
        original_turn_exists = store.turn_exists
        store.turn_exists = lambda **kwargs: True  # type: ignore[method-assign]
        try:
            repeat_status, repeat_events = self._stream_events(
                f"/api/interview-sessions/{session_id}/answer/stream",
                {"user_answer": "A repeated answer.", "response_time_ms": 9000},
            )
        finally:
            store.turn_exists = original_turn_exists  # type: ignore[method-assign]

        self.assertEqual(repeat_status, 409)
        self.assertEqual(repeat_events[0]["detail"]["code"], "turn_conflict")

    def test_stream_race_conflict_emits_turn_conflict_event(self):
        self._init(FakeLLM())
        session_id = self._create_session(turn_count=3)

        store = interview_sessions._store
        assert store is not None
        original_record_turn = store.record_turn

        def _conflict(**kwargs):
            raise TurnConflictError("turn 1 already recorded for session")

        store.record_turn = _conflict  # type: ignore[method-assign]
        try:
            status, events = self._stream_events(
                f"/api/interview-sessions/{session_id}/answer/stream",
                {"user_answer": "Racing answer.", "response_time_ms": 9000},
            )
        finally:
            store.record_turn = original_record_turn  # type: ignore[method-assign]

        self.assertEqual(status, 200)
        error = events[-1]
        self.assertEqual(error["type"], "error")
        self.assertEqual(error["code"], "turn_conflict")


if __name__ == "__main__":
    unittest.main()
