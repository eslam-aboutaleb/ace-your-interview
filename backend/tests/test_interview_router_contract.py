import concurrent.futures
import os
import tempfile
import threading
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import require_auth
from app.routers import interview_sessions
from app.services.doc_parser import DocParser
from app.services.learning_store import LearningStore
from app.services.llm_client import CALL_FAILED_CODE


class FakeLLM:
    def __init__(self, *, eval_error_code: str = ""):
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
            question = self._question_bank[self.question_n % len(self._question_bank)]
            self.question_n += 1
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
            "analysis": '{"rubric":{"technical_accuracy":4,"reasoning_depth":4,"communication_clarity":4,"completeness":4,"confidence_signal":4,"overall":80},"strengths":["Clear tradeoff reasoning"],"improvements":["Add a concrete metric"],"follow_up_note":"Quantify expected latency improvements next time."}',
            "metadata": {},
            "error_code": "",
            "finish_reason": "stop",
        }


class _BarrierEvalLLM(FakeLLM):
    """Holds every evaluation until the expected number of requests arrive."""

    def __init__(self, barrier: threading.Barrier):
        super().__init__()
        self._barrier = barrier

    async def completion(self, prompt, llm_config=None, user_identity=None, task=None, **kwargs):
        if "Generate the NEXT interview question" not in prompt:
            self._barrier.wait()
        return await super().completion(
            prompt, llm_config, user_identity, task=task, **kwargs
        )


class SpyLearningStore(LearningStore):
    """Real store plus a counter, so we can prove a call never happened."""

    def __init__(self, db_path: str):
        super().__init__(db_path)
        self.record_attempt_calls: list[dict] = []

    def record_attempt(self, **kwargs):
        self.record_attempt_calls.append(kwargs)
        return super().record_attempt(**kwargs)

    def attempt_row_count(self, user_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS total FROM learning_attempts WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return int(row["total"])


class _InterviewRouterTestBase(unittest.TestCase):
    def setUp(self):
        self.prev_flag = os.environ.get("STUDY_ENABLE_MOCK_INTERVIEW_V1")
        os.environ["STUDY_ENABLE_MOCK_INTERVIEW_V1"] = "true"
        get_settings.cache_clear()
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tempdir.cleanup()
        if self.prev_flag is None:
            os.environ.pop("STUDY_ENABLE_MOCK_INTERVIEW_V1", None)
        else:
            os.environ["STUDY_ENABLE_MOCK_INTERVIEW_V1"] = self.prev_flag
        get_settings.cache_clear()

    def build_client(self, llm: FakeLLM) -> tuple[TestClient, SpyLearningStore]:
        parser = DocParser(docs_path=os.path.join(os.getcwd(), "docs"))
        self.learning_store = SpyLearningStore(os.path.join(self.tempdir.name, "learning.db"))
        interview_sessions.init(
            llm,
            parser,
            self.learning_store,
            os.path.join(self.tempdir.name, "interview.db"),
        )
        app = FastAPI()
        app.include_router(interview_sessions.router)
        app.dependency_overrides[require_auth] = (
            lambda: {"user": "test-user", "provider": "local"}
        )
        return TestClient(app), self.learning_store

    @staticmethod
    def create_session(client: TestClient, turn_count: int = 3) -> str:
        res = client.post(
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
        assert res.status_code == 200, res.text
        return res.json()["session"]["session_id"]


class InterviewRouterContractTests(_InterviewRouterTestBase):
    def test_create_submit_and_report_flow(self):
        client, learning_store = self.build_client(FakeLLM())

        create_res = client.post(
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
        self.assertEqual(create_res.status_code, 200)
        payload = create_res.json()
        session_id = payload["session"]["session_id"]
        self.assertTrue(payload["session"]["current_question"])
        self.assertEqual(payload["session"]["interviewer_style"], "neutral")
        self.assertEqual(payload["session"]["feedback_mode"], "concise")

        explicit_res = client.post(
            "/api/interview-sessions",
            json={
                "track": "backend",
                "level": "mid",
                "interview_type": "behavioral",
                "turn_count": 1,
                "target_role": "Backend Engineer",
                "interviewer_style": "challenging",
                "feedback_mode": "deep",
                "focus_areas": ["leadership"],
            },
        )
        self.assertEqual(explicit_res.status_code, 200)
        explicit_payload = explicit_res.json()
        self.assertEqual(explicit_payload["session"]["interviewer_style"], "challenging")
        self.assertEqual(explicit_payload["session"]["feedback_mode"], "deep")

        max_turns_res = client.post(
            "/api/interview-sessions",
            json={
                "track": "backend",
                "level": "mid",
                "interview_type": "technical",
                "turn_count": 100,
                "target_role": "Backend Engineer",
                "focus_areas": ["api design"],
            },
        )
        self.assertEqual(max_turns_res.status_code, 200)
        self.assertEqual(max_turns_res.json()["session"]["turn_count"], 100)

        answer_res = client.post(
            f"/api/interview-sessions/{session_id}/answer",
            json={
                "user_answer": "I would use cursor pagination for stable ordering.",
                "response_time_ms": 9000,
            },
        )
        self.assertEqual(answer_res.status_code, 200)
        answer_payload = answer_res.json()
        self.assertEqual(answer_payload["turn"]["rubric"]["overall"], 80)
        self.assertEqual(answer_payload["session"]["status"], "completed")
        # A scored turn must still feed spaced repetition.
        self.assertEqual(len(learning_store.record_attempt_calls), 1)

        report_res = client.get(f"/api/interview-sessions/{session_id}/report")
        self.assertEqual(report_res.status_code, 200)
        report_payload = report_res.json()
        self.assertGreaterEqual(report_payload["report"]["overall_score"], 0)

        trends_res = client.get("/api/interview-sessions/trends?limit=50")
        self.assertEqual(trends_res.status_code, 200)
        trends_payload = trends_res.json()
        self.assertIn("points", trends_payload)
        self.assertIn("summary", trends_payload)
        self.assertGreaterEqual(trends_payload["summary"]["session_count"], 1)

    def test_degraded_evaluation_is_flagged_and_skips_record_attempt(self):
        client, learning_store = self.build_client(
            FakeLLM(eval_error_code=CALL_FAILED_CODE)
        )
        session_id = self.create_session(client, turn_count=1)

        answer_res = client.post(
            f"/api/interview-sessions/{session_id}/answer",
            json={
                "user_answer": "I would shard by tenant and add a backpressure queue.",
                "response_time_ms": 9000,
            },
        )
        self.assertEqual(answer_res.status_code, 200)
        turn = answer_res.json()["turn"]
        self.assertTrue(turn["degraded"])
        self.assertTrue(answer_res.json()["degraded"])
        self.assertIsNone(turn["rubric"]["overall"])
        self.assertIsNone(turn["rubric"]["technical_accuracy"])

        # The corruption path: a fabricated 60/3s used to be written to the SRS store.
        self.assertEqual(learning_store.record_attempt_calls, [])
        self.assertEqual(learning_store.attempt_row_count("test-user"), 0)

    def test_degraded_turn_is_excluded_from_the_report(self):
        client, learning_store = self.build_client(
            FakeLLM(eval_error_code=CALL_FAILED_CODE)
        )
        session_id = self.create_session(client, turn_count=1)

        answer_res = client.post(
            f"/api/interview-sessions/{session_id}/answer",
            json={"user_answer": "A short answer.", "response_time_ms": 5000},
        )
        self.assertEqual(answer_res.status_code, 200)
        self.assertEqual(answer_res.json()["session"]["status"], "completed")

        report_res = client.get(f"/api/interview-sessions/{session_id}/report")
        self.assertEqual(report_res.status_code, 200)
        report = report_res.json()["report"]
        self.assertEqual(report["readiness_label"], "Not Started")
        self.assertEqual(report["overall_score"], 0.0)
        self.assertEqual(report["completed_turns"], 1)

        session_res = client.get(f"/api/interview-sessions/{session_id}")
        turns = session_res.json()["turns"]
        self.assertEqual(len(turns), 1)
        self.assertTrue(turns[0]["degraded"])

    def test_concurrent_answers_produce_exactly_one_turn(self):
        # Both requests must pass the session read before either writes, so the
        # evaluation is held at a barrier. Each TestClient request runs on its own
        # event loop, so the threads really do overlap.
        barrier = threading.Barrier(2, timeout=15)
        client, _ = self.build_client(_BarrierEvalLLM(barrier))
        session_id = self.create_session(client, turn_count=3)

        def submit(_i: int):
            return client.post(
                f"/api/interview-sessions/{session_id}/answer",
                json={
                    "user_answer": "I would use cursor pagination for stable ordering.",
                    "response_time_ms": 9000,
                },
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(submit, range(2)))

        statuses = sorted(res.status_code for res in responses)
        self.assertEqual(statuses, [200, 409])
        conflict = next(res for res in responses if res.status_code == 409)
        detail = conflict.json()["detail"]
        self.assertEqual(detail["code"], "turn_conflict")

        session_res = client.get(f"/api/interview-sessions/{session_id}")
        turns = session_res.json()["turns"]
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["turn_index"], 1)
        self.assertEqual(session_res.json()["session"]["turns_completed"], 1)

    def test_sequential_turns_are_not_treated_as_conflicts(self):
        client, learning_store = self.build_client(FakeLLM())
        session_id = self.create_session(client, turn_count=3)

        first = client.post(
            f"/api/interview-sessions/{session_id}/answer",
            json={"user_answer": "First answer.", "response_time_ms": 9000},
        )
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()["turn"]["turn_index"], 1)

        next_q = client.post(f"/api/interview-sessions/{session_id}/next-question", json={})
        self.assertEqual(next_q.status_code, 200)

        second = client.post(
            f"/api/interview-sessions/{session_id}/answer",
            json={"user_answer": "Second answer.", "response_time_ms": 9000},
        )
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json()["turn"]["turn_index"], 2)
        self.assertEqual(len(learning_store.record_attempt_calls), 2)

        session_res = client.get(f"/api/interview-sessions/{session_id}")
        self.assertEqual(len(session_res.json()["turns"]), 2)


if __name__ == "__main__":
    unittest.main()
