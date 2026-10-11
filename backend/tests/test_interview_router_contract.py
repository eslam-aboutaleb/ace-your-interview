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
        # A scored turn must still feed spaced repetition: one competency card
        # per rubric dimension, never the track's first topic.
        self.assertEqual(len(learning_store.record_attempt_calls), 5)
        for call in learning_store.record_attempt_calls:
            self.assertTrue(
                call["question_id"].startswith("interview-competency:"), call
            )
            self.assertEqual(call["mode"], "interview")

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
        # Two scored turns schedule ten competency cards (five dimensions each).
        self.assertEqual(len(learning_store.record_attempt_calls), 10)

        session_res = client.get(f"/api/interview-sessions/{session_id}")
        self.assertEqual(len(session_res.json()["turns"]), 2)

    def test_question_contract_persists_and_reaches_the_answer(self):
        client, _ = self.build_client(FakeLLM())

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

        # The session carries the full question contract, not just text.
        contract = payload["session"]["current_question_contract"]
        self.assertEqual(contract["competency_focus"], "tradeoff analysis")
        self.assertEqual(
            contract["expected_signals"],
            ["constraints", "consistency", "client impact"],
        )
        self.assertTrue(contract["question_id"].startswith("iq-"))

        answer_res = client.post(
            f"/api/interview-sessions/{session_id}/answer",
            json={
                "user_answer": "I would use cursor pagination for stable ordering.",
                "response_time_ms": 9000,
            },
        )
        self.assertEqual(answer_res.status_code, 200)
        turn = answer_res.json()["turn"]
        self.assertEqual(turn["competency_focus"], "tradeoff analysis")
        self.assertEqual(
            turn["expected_signals"],
            ["constraints", "consistency", "client impact"],
        )
        self.assertEqual(turn["question_contract"], contract)

        session_res = client.get(f"/api/interview-sessions/{session_id}")
        turns = session_res.json()["turns"]
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["competency_focus"], "tradeoff analysis")
        self.assertEqual(
            turns[0]["expected_signals"],
            ["constraints", "consistency", "client impact"],
        )
        self.assertEqual(turns[0]["question_contract"], contract)


class _CapturingFakeLLM(FakeLLM):
    """Records the prompts production actually sent, and drives the design flow."""

    def __init__(self, *, eval_payload: str):
        super().__init__()
        self._eval_payload = eval_payload
        self.question_prompts: list[str] = []
        self.eval_prompts: list[str] = []

    async def completion(self, prompt, llm_config=None, user_identity=None, task=None, **kwargs):
        # Matches both the general and the system-design question contracts,
        # which is the point of this test double.
        if "interview evaluator" in prompt:
            self.eval_prompts.append(prompt)
            if self._eval_payload:
                return {
                    "success": True,
                    "analysis": self._eval_payload,
                    "metadata": {},
                    "error_code": "",
                    "finish_reason": "stop",
                }
        elif "Generate the NEXT" in prompt:
            self.question_prompts.append(prompt)
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
        return await super().completion(prompt, llm_config, user_identity, task=task, **kwargs)


_DESIGN_EVAL_PAYLOAD = (
    '{"rubric":{"technical_accuracy":5,"reasoning_depth":5,'
    '"communication_clarity":3,"completeness":3,"confidence_signal":3,"overall":70},'
    '"system_design":{"architecture":5,"tradeoffs":5,"scalability":5,'
    '"failure_modes":5,"communication":5},'
    '"strengths":["Clear tradeoff"],"improvements":["Add failure mode"],'
    '"follow_up_note":"Cover failure modes next."}'
)

_DIAGRAM_ANSWER = """I would put a gateway in front and a queue behind it.

```mermaid
graph TD
  C[Client] --> G[Gateway]
  G --> Q[Queue]
  Q --> W[Worker]
  W --> DB[(Postgres)]
```

I chose the queue instead of a direct write so the write path can shed load."""


class InterviewPersonaContractTests(_InterviewRouterTestBase):
    def test_the_persona_round_trips_into_the_interview_prompt(self):
        llm = _CapturingFakeLLM(eval_payload="")
        client, _ = self.build_client(llm)
        create_res = client.post(
            "/api/interview-sessions",
            json={
                "track": "backend",
                "level": "mid",
                "interview_type": "mixed",
                "turn_count": 1,
                "persona": "bar-raiser",
                "target_role": "Backend Engineer",
            },
        )
        self.assertEqual(create_res.status_code, 200)
        self.assertEqual(
            create_res.json()["session"]["persona"], "bar-raiser"
        )

        next_res = client.post(
            f"/api/interview-sessions/{create_res.json()['session']['session_id']}/next-question",
            json={},
        )
        self.assertEqual(next_res.status_code, 200, next_res.text)
        self.assertTrue(llm.question_prompts)
        prompt = llm.question_prompts[-1]
        self.assertIn("- persona: bar-raiser", prompt)
        self.assertIn("bar-raiser: hold the top-of-band bar", prompt)

    def test_the_persona_defaults_to_neutral_when_absent(self):
        llm = _CapturingFakeLLM(eval_payload="")
        client, _ = self.build_client(llm)
        create_res = client.post(
            "/api/interview-sessions",
            json={
                "track": "backend",
                "level": "mid",
                "interview_type": "mixed",
                "turn_count": 1,
                "target_role": "Backend Engineer",
            },
        )
        self.assertEqual(create_res.status_code, 200)
        self.assertEqual(create_res.json()["session"]["persona"], "neutral")


class SystemDesignSessionContractTests(_InterviewRouterTestBase):
    def test_a_system_design_session_uses_its_own_prompt_and_rubric(self):
        llm = _CapturingFakeLLM(eval_payload=_DESIGN_EVAL_PAYLOAD)
        client, _ = self.build_client(llm)
        create_res = client.post(
            "/api/interview-sessions",
            json={
                "track": "system_design",
                "level": "senior",
                "interview_type": "system_design",
                "turn_count": 1,
                "persona": "bar-raiser",
                "target_role": "Staff Engineer",
            },
        )
        self.assertEqual(create_res.status_code, 200)
        session_id = create_res.json()["session"]["session_id"]

        next_res = client.post(
            f"/api/interview-sessions/{session_id}/next-question",
            json={},
        )
        self.assertEqual(next_res.status_code, 200)
        prompt = llm.question_prompts[-1]
        # A dedicated contract, not the general or coding one.
        self.assertIn("system-design interviewer", prompt)
        self.assertIn("mermaid diagram", prompt)
        self.assertNotIn("senior interviewer running a personalized mock", prompt)

        answer_res = client.post(
            f"/api/interview-sessions/{session_id}/answer",
            json={"user_answer": _DIAGRAM_ANSWER, "response_time_ms": 4000},
        )
        self.assertEqual(answer_res.status_code, 200)
        turn = answer_res.json()["turn"]
        # The model's string of 5s on architecture/communication is what the
        # evaluator returned; the deterministic diagram evidence is what the
        # evaluator was shown.
        self.assertEqual(
            sorted(turn["system_design_scores"]),
            sorted(
                [
                    "architecture",
                    "tradeoffs",
                    "scalability",
                    "failure_modes",
                    "communication",
                ]
            ),
        )
        self.assertIn(
            "Deterministic diagram evidence",
            llm.eval_prompts[-1],
        )
        self.assertIn("structure_score", llm.eval_prompts[-1])

    def test_a_non_design_session_carries_no_design_rubric(self):
        llm = _CapturingFakeLLM(eval_payload=_DESIGN_EVAL_PAYLOAD)
        client, _ = self.build_client(llm)
        session_id = self.create_session(client, turn_count=1)
        answer_res = client.post(
            f"/api/interview-sessions/{session_id}/answer",
            json={"user_answer": _DIAGRAM_ANSWER, "response_time_ms": 2000},
        )
        self.assertEqual(answer_res.status_code, 200)
        self.assertNotIn("Deterministic diagram evidence", llm.eval_prompts[-1])
        self.assertEqual(answer_res.json()["turn"]["system_design_scores"], {})


if __name__ == "__main__":
    unittest.main()


class _PromptRecordingLLM(FakeLLM):
    """Records every prompt the interview router sends."""

    def __init__(self):
        super().__init__()
        self.prompts: list[str] = []

    async def completion(self, prompt, llm_config=None, user_identity=None, task=None, **kwargs):
        self.prompts.append(prompt)
        return await super().completion(
            prompt, llm_config, user_identity, task=task, **kwargs
        )


class _LevelSpyStore(SpyLearningStore):
    """Real store plus a recommendation-call recorder."""

    def __init__(self, db_path: str):
        super().__init__(db_path)
        self.recommendation_calls: list[tuple[str, str]] = []

    def get_level_recommendation(self, *args, **kwargs):
        self.recommendation_calls.append((args[0], args[1]))
        return super().get_level_recommendation(*args, **kwargs)


class InterviewSessionLevelResolutionTests(_InterviewRouterTestBase):
    """A session created without a level resolves it from the track's history."""

    def _build(self, llm):
        self.learning_store = _LevelSpyStore(
            os.path.join(self.tempdir.name, "learning.db")
        )
        interview_sessions.init(
            llm,
            DocParser(docs_path=os.path.join(os.getcwd(), "docs")),
            self.learning_store,
            os.path.join(self.tempdir.name, "interview.db"),
        )
        app = FastAPI()
        app.include_router(interview_sessions.router)
        app.dependency_overrides[require_auth] = (
            lambda: {"user": "test-user", "provider": "local"}
        )
        return TestClient(app)

    def _fluent_history(self, count=10):
        for index in range(count):
            self.learning_store.record_attempt(
                user_id="test-user",
                question_id=f"q-{index}",
                topic_id="backend",
                user_answer="answer",
                is_correct=True,
                confidence=4,
                response_time_ms=1500,
                mode="quiz",
            )

    def _create(self, client: TestClient, **overrides):
        payload = {
            "track": "backend",
            "interview_type": "technical",
            "turn_count": 3,
            "target_role": "Backend Engineer",
            "focus_areas": ["api design"],
        }
        payload.update(overrides)
        res = client.post("/api/interview-sessions", json=payload)
        self.assertEqual(res.status_code, 200, res.text)
        return res

    def test_omitted_level_resolves_from_the_track_history(self):
        llm = _PromptRecordingLLM()
        client = self._build(llm)
        self._fluent_history()
        res = self._create(client)
        self.assertEqual(res.json()["session"]["level"], "senior")
        self.assertIn("- level: senior", llm.prompts[0])
        self.assertEqual(
            self.learning_store.recommendation_calls, [("test-user", "backend")]
        )

    def test_omitted_level_defaults_to_mid_without_history(self):
        client = self._build(_PromptRecordingLLM())
        res = self._create(client)
        self.assertEqual(res.json()["session"]["level"], "mid")

    def test_explicit_level_is_kept_and_never_queries_the_history(self):
        llm = _PromptRecordingLLM()
        client = self._build(llm)
        self._fluent_history()
        res = self._create(client, level="junior")
        self.assertEqual(res.json()["session"]["level"], "junior")
        self.assertIn("- level: junior", llm.prompts[0])
        self.assertEqual(self.learning_store.recommendation_calls, [])

    def test_struggling_history_de_escalates_the_level(self):
        client = self._build(_PromptRecordingLLM())
        for index in range(10):
            self.learning_store.record_attempt(
                user_id="test-user",
                question_id=f"q-{index}",
                topic_id="backend",
                user_answer="answer",
                is_correct=False,
                confidence=2,
                response_time_ms=90000,
                mode="quiz",
            )
        res = self._create(client)
        self.assertEqual(res.json()["session"]["level"], "junior")
