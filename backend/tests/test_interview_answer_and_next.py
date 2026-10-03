"""Stage 3.3 — one combined answer + next-question round trip, sequenced."""

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
from app.schemas.models import SubmitInterviewAnswerRequest
from app.services.doc_parser import DocParser
from app.services.interview_store import TurnConflictError
from app.services.learning_store import LearningStore
from app.services.llm_client import CALL_FAILED_CODE

_QUESTION_MARKER = "Generate the NEXT interview question"
_DIGEST_MARKER = interview_sessions._MEMORY_DIGEST_MARKER
_RUBRIC = {
    "technical_accuracy": 4,
    "reasoning_depth": 4,
    "communication_clarity": 4,
    "completeness": 4,
    "confidence_signal": 4,
    "overall": 80,
}


class _FakeLLM:
    """Records prompt order so sequencing can be asserted, not assumed."""

    def __init__(
        self,
        *,
        fail_question_from: int | None = None,
        eval_error_code: str = "",
        eval_gate: "asyncio.Event | None" = None,
    ):
        self.fail_question_from = fail_question_from
        self.eval_error_code = eval_error_code
        self.eval_gate = eval_gate
        self.prompts: list[str] = []
        self.eval_cancelled = False
        self.question_n = 0
        self._bank = [
            "How would you design cursor-based pagination to keep ordering stable across writes?",
            "What strategy would you use to make retryable writes idempotent across network failures?",
            "How would you detect and mitigate cache stampede on a high-traffic endpoint?",
            "How would you design rate limiting for bursty tenants while keeping fairness?",
        ]

    async def completion(self, prompt, llm_config=None, user_identity=None, task=None, **kwargs):
        self.prompts.append(prompt)
        if _QUESTION_MARKER in prompt:
            # ``fail_question_from`` is a call ordinal, so a test can fail only
            # the post-turn question and leave session creation working.
            if self.fail_question_from is not None and self.question_n >= self.fail_question_from:
                self.question_n += 1
                raise RuntimeError("question generation exploded")
            self.question_n += 1
            question = self._bank[self.question_n % len(self._bank)]
            return {
                "success": True,
                "analysis": (
                    '{"question":"'
                    + question
                    + '","competency_focus":"tradeoff analysis",'
                    '"expected_signals":["constraints","consistency","client impact"]}'
                ),
                "metadata": {},
                "error_code": "",
                "finish_reason": "stop",
            }
        if _DIGEST_MARKER in prompt:
            return {
                "success": True,
                "analysis": "Compacted session memory.",
                "metadata": {},
                "error_code": "",
                "finish_reason": "stop",
            }
        if self.eval_gate is not None:
            try:
                await self.eval_gate.wait()
            except asyncio.CancelledError:
                self.eval_cancelled = True
                raise
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
            "analysis": json.dumps(
                {
                    "rubric": _RUBRIC,
                    "strengths": ["Clear tradeoff reasoning"],
                    "improvements": ["Add a concrete metric"],
                    "follow_up_note": "Quantify expected latency improvements next time.",
                }
            ),
            "metadata": {},
            "error_code": "",
            "finish_reason": "stop",
        }


class InterviewAnswerAndNextTests(unittest.TestCase):
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

    def _init(self, llm: _FakeLLM) -> _FakeLLM:
        parser = DocParser(docs_path=os.path.join(os.getcwd(), "docs"))
        interview_sessions.init(
            llm,
            parser,
            LearningStore(os.path.join(self.tempdir.name, "learning.db")),
            os.path.join(self.tempdir.name, "interview.db"),
        )
        return llm

    def _create_session(self, turn_count: int = 3) -> str:
        created = self.client.post(
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
        self.assertEqual(created.status_code, 200, created.text)
        return created.json()["session"]["session_id"]

    def _stream_events(self, path: str, payload: dict) -> tuple[int, list[dict]]:
        with self.client.stream("POST", path, json=payload) as res:
            events = []
            for line in res.iter_lines():
                if not line:
                    continue
                text = line.decode("utf-8") if isinstance(line, bytes) else line
                events.append(json.loads(text))
            return res.status_code, events

    def test_combined_endpoint_returns_the_turn_and_a_post_turn_question(self):
        llm = self._init(_FakeLLM())
        session_id = self._create_session(turn_count=3)
        first_question = self.client.get(
            f"/api/interview-sessions/{session_id}"
        ).json()["session"]["current_question"]

        res = self.client.post(
            f"/api/interview-sessions/{session_id}/answer-and-next",
            json={"user_answer": "I would use cursor pagination.", "response_time_ms": 4000},
        )
        self.assertEqual(res.status_code, 200, res.text)
        payload = res.json()
        self.assertEqual(payload["turn"]["turn_index"], 1)
        self.assertEqual(payload["turn"]["question"], first_question)
        self.assertEqual(payload["turn"]["rubric"]["overall"], 80)
        self.assertEqual(payload["next_question_status"], "generated")
        self.assertIsNotNone(payload["next_question"])
        self.assertEqual(payload["next_question"]["turn_index"], 2)
        self.assertTrue(payload["next_question"]["question"])
        # The persisted session carries the freshly generated question.
        self.assertEqual(
            payload["session"]["current_question"],
            payload["next_question"]["question"],
        )

        # Sequencing: the evaluation is awaited and recorded before the next
        # question is generated, so the question prompt sees the new turn.
        stages = [
            "question" if _QUESTION_MARKER in p else ("digest" if _DIGEST_MARKER in p else "eval")
            for p in llm.prompts
        ]
        self.assertEqual(stages[:2], ["question", "eval"])
        self.assertEqual(stages[-1], "question")
        question_prompts = [p for p in llm.prompts if _QUESTION_MARKER in p]
        self.assertIn("turns_completed: 1", question_prompts[-1])
        self.assertIn("Turn 1 Q:", question_prompts[-1])

        session = self.client.get(f"/api/interview-sessions/{session_id}").json()
        self.assertEqual(len(session["turns"]), 1)

    def test_next_question_failure_returns_the_recorded_turn(self):
        self._init(_FakeLLM(fail_question_from=1))
        session_id = self._create_session(turn_count=3)

        res = self.client.post(
            f"/api/interview-sessions/{session_id}/answer-and-next",
            json={"user_answer": "A recorded answer.", "response_time_ms": 4000},
        )
        # Partial success: the saved turn, not a failed request.
        self.assertEqual(res.status_code, 200, res.text)
        payload = res.json()
        self.assertIsNone(payload["next_question"])
        self.assertEqual(payload["next_question_status"], "generation_failed")
        self.assertEqual(payload["turn"]["turn_index"], 1)
        self.assertEqual(payload["turn"]["rubric"]["overall"], 80)

        session = self.client.get(f"/api/interview-sessions/{session_id}").json()
        self.assertEqual(len(session["turns"]), 1, "the turn must be persisted")
        self.assertEqual(session["session"]["turns_completed"], 1)
        self.assertEqual(session["session"]["current_question"], "")

    def test_final_turn_reports_no_next_question(self):
        self._init(_FakeLLM())
        session_id = self._create_session(turn_count=1)

        res = self.client.post(
            f"/api/interview-sessions/{session_id}/answer-and-next",
            json={"user_answer": "The only answer.", "response_time_ms": 1000},
        )
        self.assertEqual(res.status_code, 200, res.text)
        payload = res.json()
        self.assertIsNone(payload["next_question"])
        self.assertEqual(payload["next_question_status"], "session_completed")
        self.assertEqual(payload["session"]["status"], "completed")
        self.assertTrue(payload["report_ready"])

    def test_degraded_evaluation_still_returns_the_turn_and_next_question(self):
        llm = self._init(_FakeLLM(eval_error_code=CALL_FAILED_CODE))
        session_id = self._create_session(turn_count=3)

        res = self.client.post(
            f"/api/interview-sessions/{session_id}/answer-and-next",
            json={"user_answer": "A short answer.", "response_time_ms": 1000},
        )
        self.assertEqual(res.status_code, 200, res.text)
        payload = res.json()
        self.assertTrue(payload["degraded"])
        self.assertIsNone(payload["turn"]["rubric"]["overall"])
        self.assertEqual(payload["next_question_status"], "generated")
        self.assertTrue(payload["next_question"]["question"])
        # A degraded turn must not trigger another provider call for the digest.
        digest_prompts = [p for p in llm.prompts if _DIGEST_MARKER in p]
        self.assertEqual(digest_prompts, [])

    def test_sequential_pair_still_works_alongside_the_combined_endpoint(self):
        self._init(_FakeLLM())
        session_id = self._create_session(turn_count=3)

        first = self.client.post(
            f"/api/interview-sessions/{session_id}/answer",
            json={"user_answer": "First answer.", "response_time_ms": 1000},
        )
        self.assertEqual(first.status_code, 200, first.text)
        self.assertNotIn("next_question", first.json())

        next_q = self.client.post(
            f"/api/interview-sessions/{session_id}/next-question",
            json={},
        )
        self.assertEqual(next_q.status_code, 200, next_q.text)
        self.assertTrue(next_q.json()["question"])

        second = self.client.post(
            f"/api/interview-sessions/{session_id}/answer-and-next",
            json={"user_answer": "Second answer.", "response_time_ms": 1000},
        )
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(second.json()["turn"]["turn_index"], 2)

    def test_turn_conflict_returns_409(self):
        self._init(_FakeLLM())
        session_id = self._create_session(turn_count=3)

        store = interview_sessions._store
        assert store is not None
        original = store.turn_exists
        store.turn_exists = lambda **kwargs: True  # type: ignore[method-assign]
        try:
            res = self.client.post(
                f"/api/interview-sessions/{session_id}/answer-and-next",
                json={"user_answer": "A repeated answer.", "response_time_ms": 1000},
            )
        finally:
            store.turn_exists = original  # type: ignore[method-assign]

        self.assertEqual(res.status_code, 409)
        self.assertEqual(res.json()["detail"]["code"], "turn_conflict")
        session = self.client.get(f"/api/interview-sessions/{session_id}").json()
        self.assertEqual(len(session["turns"]), 0)

    def test_racing_record_turn_conflict_returns_409(self):
        self._init(_FakeLLM())
        session_id = self._create_session(turn_count=3)

        store = interview_sessions._store
        assert store is not None
        original = store.record_turn

        def _conflict(**kwargs):
            raise TurnConflictError("turn 1 already recorded for session")

        store.record_turn = _conflict  # type: ignore[method-assign]
        try:
            res = self.client.post(
                f"/api/interview-sessions/{session_id}/answer-and-next",
                json={"user_answer": "A racing answer.", "response_time_ms": 1000},
            )
        finally:
            store.record_turn = original  # type: ignore[method-assign]

        self.assertEqual(res.status_code, 409)
        self.assertEqual(res.json()["detail"]["code"], "turn_conflict")

    def test_stream_twin_emits_the_same_sequence(self):
        self._init(_FakeLLM())
        session_id = self._create_session(turn_count=3)

        status, events = self._stream_events(
            f"/api/interview-sessions/{session_id}/answer-and-next/stream",
            {"user_answer": "I would shard by tenant.", "response_time_ms": 2500},
        )
        self.assertEqual(status, 200)
        self.assertEqual(events[0]["type"], "start")
        done = events[-1]
        self.assertEqual(done["type"], "done")
        self.assertEqual(done["turn"]["turn_index"], 1)
        self.assertEqual(done["session"]["turns_completed"], 1)
        self.assertEqual(done["next_question_status"], "generated")
        self.assertTrue(done["next_question"]["question"])
        self.assertEqual(
            done["session"]["current_question"],
            done["next_question"]["question"],
        )

    def test_stream_twin_returns_partial_success_when_the_question_fails(self):
        self._init(_FakeLLM(fail_question_from=1))
        session_id = self._create_session(turn_count=3)

        status, events = self._stream_events(
            f"/api/interview-sessions/{session_id}/answer-and-next/stream",
            {"user_answer": "A recorded answer.", "response_time_ms": 2500},
        )
        self.assertEqual(status, 200)
        done = events[-1]
        self.assertEqual(done["type"], "done", events)
        self.assertIsNone(done["next_question"])
        self.assertEqual(done["next_question_status"], "generation_failed")
        self.assertEqual(done["turn"]["turn_index"], 1)

        session = self.client.get(f"/api/interview-sessions/{session_id}").json()
        self.assertEqual(len(session["turns"]), 1)

    def test_stream_twin_rejects_a_taken_turn_with_409(self):
        self._init(_FakeLLM())
        session_id = self._create_session(turn_count=3)

        store = interview_sessions._store
        assert store is not None
        original = store.turn_exists
        store.turn_exists = lambda **kwargs: True  # type: ignore[method-assign]
        try:
            status, events = self._stream_events(
                f"/api/interview-sessions/{session_id}/answer-and-next/stream",
                {"user_answer": "A repeated answer.", "response_time_ms": 1000},
            )
        finally:
            store.turn_exists = original  # type: ignore[method-assign]

        self.assertEqual(status, 409)
        self.assertEqual(events[0]["detail"]["code"], "turn_conflict")

    def test_client_disconnect_cancels_the_in_flight_provider_call(self):
        # The gate is never opened, so the evaluation is still in flight when
        # the client goes away.
        llm = self._init(_FakeLLM(eval_gate=asyncio.Event()))
        session_id = self._create_session(turn_count=3)

        async def _drive():
            response = await interview_sessions.answer_and_next_stream(
                session_id=session_id,
                body=SubmitInterviewAnswerRequest(
                    user_answer="An answer nobody will read.",
                    response_time_ms=1000,
                ),
                user={"user": "test-user", "provider": "local"},
            )
            events = response.body_iterator
            assert hasattr(events, "__anext__")
            first = await events.__anext__()  # type: ignore[union-attr]
            self.assertEqual(json.loads(first)["type"], "start")
            pending_before = {t for t in asyncio.all_tasks() if not t.done()}
            # Second read enters the progress loop, so the evaluation task exists.
            await events.__anext__()  # type: ignore[union-attr]
            await events.aclose()  # type: ignore[union-attr]
            await asyncio.sleep(0.05)
            leaked = {
                t for t in asyncio.all_tasks() if not t.done()
            } - pending_before - {asyncio.current_task()}
            return leaked

        leaked = asyncio.run(_drive())
        self.assertEqual(leaked, set(), "the abandoned provider call was not cleaned up")
        self.assertTrue(llm.eval_cancelled, "the provider coroutine was never cancelled")

        # The turn was never recorded: the disconnect happened mid-evaluation.
        session = self.client.get(f"/api/interview-sessions/{session_id}").json()
        self.assertEqual(len(session["turns"]), 0)

    def test_client_disconnect_cleans_up_the_plain_answer_stream(self):
        llm = self._init(_FakeLLM(eval_gate=asyncio.Event()))
        session_id = self._create_session(turn_count=3)

        async def _drive():
            response = await interview_sessions.submit_answer_stream(
                session_id=session_id,
                body=SubmitInterviewAnswerRequest(
                    user_answer="Another abandoned answer.",
                    response_time_ms=1000,
                ),
                user={"user": "test-user", "provider": "local"},
            )
            events = response.body_iterator
            assert hasattr(events, "__anext__")
            await events.__anext__()  # type: ignore[union-attr]
            await events.__anext__()  # type: ignore[union-attr]
            await events.aclose()  # type: ignore[union-attr]
            await asyncio.sleep(0.05)
            return {t for t in asyncio.all_tasks() if not t.done()} - {asyncio.current_task()}

        leaked = asyncio.run(_drive())
        self.assertEqual(leaked, set())
        self.assertTrue(llm.eval_cancelled)


if __name__ == "__main__":
    unittest.main()