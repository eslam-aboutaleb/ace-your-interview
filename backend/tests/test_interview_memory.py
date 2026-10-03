"""Stage 3.2 — interview memory is a compacted digest plus a rolling window."""

import json
import os
import sqlite3
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import require_auth
from app.routers import interview_sessions
from app.services import interview_store
from app.services.doc_parser import DocParser
from app.services.interview_store import InterviewStore, memory_summary_limit
from app.services.learning_store import LearningStore
from app.services.llm_client import CALL_FAILED_CODE

_RUBRIC = {
    "technical_accuracy": 4,
    "reasoning_depth": 4,
    "communication_clarity": 4,
    "completeness": 4,
    "confidence_signal": 4,
    "overall": 80,
}

_QUESTION_MARKER = "Generate the NEXT interview question"
_DIGEST_MARKER = interview_sessions._MEMORY_DIGEST_MARKER


class _LegacyMemoryLLM:
    """Answers every prompt kind and records what the prompt builder received."""

    def __init__(self, *, digest: str = "", digest_fails: bool = False, eval_error_code: str = ""):
        self.prompts: list[str] = []
        self.digest = digest
        self.digest_fails = digest_fails
        self.eval_error_code = eval_error_code
        self.digest_calls = 0
        self.question_n = 0
        self._bank = [
            "How would you design cursor-based pagination to keep ordering stable across writes?",
            "What strategy would you use to make retryable writes idempotent across network failures?",
            "How would you detect and mitigate cache stampede on a high-traffic endpoint?",
            "How would you design rate limiting for bursty tenants while keeping fairness?",
            "How would you keep a schema migration safe while writes are in flight?",
            "How would you verify a rollout of a new retry policy before full traffic?",
            "How would you make a background job idempotent after a partial failure?",
        ]

    async def completion(self, prompt, llm_config=None, user_identity=None, task=None, **kwargs):
        self.prompts.append(prompt)
        if _QUESTION_MARKER in prompt:
            question = self._bank[self.question_n % len(self._bank)]
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
        if _DIGEST_MARKER in prompt:
            self.digest_calls += 1
            if self.digest_fails:
                return {
                    "success": False,
                    "analysis": "",
                    "metadata": {},
                    "error": "provider unavailable",
                    "error_code": CALL_FAILED_CODE,
                    "finish_reason": "",
                    "usage": {},
                }
            return {
                "success": True,
                "analysis": self.digest,
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


def _legacy_memory_db(db_path: str, *, summary: str, turns: int = 5) -> None:
    """A pre-3.2 database: one append-truncated summary, no digest columns."""
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute(
            """
            CREATE TABLE interview_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                track TEXT NOT NULL,
                level TEXT NOT NULL,
                interview_type TEXT NOT NULL,
                turn_count INTEGER NOT NULL,
                turns_completed INTEGER NOT NULL,
                status TEXT NOT NULL,
                target_role TEXT NOT NULL,
                job_description_text TEXT NOT NULL,
                resume_summary_text TEXT NOT NULL,
                focus_areas_json TEXT NOT NULL,
                asked_questions_json TEXT NOT NULL,
                current_question TEXT NOT NULL,
                memory_summary TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE interview_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                question TEXT NOT NULL,
                user_answer TEXT NOT NULL,
                rubric_json TEXT NOT NULL,
                strengths_json TEXT NOT NULL,
                improvements_json TEXT NOT NULL,
                follow_up_note TEXT NOT NULL,
                response_time_ms INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE interview_reports (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                report_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO interview_sessions(
                session_id, user_id, track, level, interview_type, turn_count,
                turns_completed, status, target_role, job_description_text,
                resume_summary_text, focus_areas_json, asked_questions_json,
                current_question, memory_summary, created_at, updated_at
            ) VALUES ('legacy-mem', 'alice', 'backend', 'mid', 'technical', ?,
                ?, 'active', 'Backend Engineer', '', '', '[]', '[]', '', ?, 't', 't')
            """,
            (turns, turns, summary),
        )
        for index in range(1, turns + 1):
            conn.execute(
                """
                INSERT INTO interview_turns(
                    session_id, user_id, turn_index, question, user_answer,
                    rubric_json, strengths_json, improvements_json,
                    follow_up_note, response_time_ms, created_at
                ) VALUES ('legacy-mem', 'alice', ?, ?, ?, '{}', '[]', '[]', '', 1000, 't')
                """,
                (index, f"legacy question {index}", f"legacy answer {index} " * 40),
            )
    conn.close()


class InterviewMemoryStoreTests(unittest.TestCase):
    def test_window_keeps_the_last_verbatim_turns_and_trims_answer_excerpts(self):
        with tempfile.TemporaryDirectory() as td:
            store = InterviewStore(os.path.join(td, "interview.db"))
            session = store.create_session(
                user_id="alice",
                track="backend",
                level="mid",
                interview_type="technical",
                turn_count=9,
                target_role="Backend Engineer",
                job_description_text="",
                resume_summary_text="",
                focus_areas=[],
            )
            self.assertEqual(session["memory_window"], [])

            window: list[dict] = []
            for index in range(1, 7):
                window = store.append_memory_turn(
                    user_id="alice",
                    session_id=session["session_id"],
                    entry={
                        "turn_index": index,
                        "question": f"question {index}",
                        "answer_excerpt": "a" * 5000,
                        "strengths": ["Clear", "Measured", "Dropped"],
                        "improvements": ["Measure"],
                    },
                )

            size = interview_store.memory_verbatim_turn_count()
            self.assertEqual([item["turn_index"] for item in window], [3, 4, 5, 6])
            self.assertLessEqual(len(window), size)
            self.assertEqual(
                len(window[-1]["answer_excerpt"]),
                interview_store.memory_answer_excerpt_limit(),
            )
            self.assertEqual(window[-1]["strengths"], ["Clear", "Measured"])

            stored = store.get_memory_window(
                user_id="alice",
                session_id=session["session_id"]
            )
            self.assertEqual([item["turn_index"] for item in stored], [3, 4, 5, 6])
            context = store.get_session_context(
                user_id="alice",
                session_id=session["session_id"]
            )
            assert context is not None
            self.assertEqual(
                [item["turn_index"] for item in context["memory_window"]],
                [3, 4, 5, 6],
            )

    def test_digest_is_capped_at_the_one_shared_limit(self):
        with tempfile.TemporaryDirectory() as td:
            store = InterviewStore(os.path.join(td, "interview.db"))
            session = store.create_session(
                user_id="alice",
                track="backend",
                level="mid",
                interview_type="technical",
                turn_count=5,
                target_role="Backend Engineer",
                job_description_text="",
                resume_summary_text="",
                focus_areas=[],
            )
            limit = memory_summary_limit()
            updated = store.set_memory_digest(
                user_id="alice",
                session_id=session["session_id"],
                digest="d" * (limit * 3),
            )
            assert updated is not None
            self.assertEqual(len(updated["memory_digest"]), limit)
            # The legacy alias still resolves for pre-digest readers.
            self.assertEqual(updated["memory_summary"], updated["memory_digest"])
            self.assertLessEqual(len(store.set_memory_summary(
                user_id="alice",
                session_id=session["session_id"],
                summary="s" * (limit * 3),
            )["memory_summary"]), limit)

    def test_memory_migration_adds_columns_and_backfills_existing_rows(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "legacy_memory.db")
            legacy_summary = "Turn 1 question: legacy one\n" + "x" * 2500
            _legacy_memory_db(db_path, summary=legacy_summary, turns=5)

            store = InterviewStore(db_path)
            columns = {
                row["name"]
                for row in store._conn.execute("PRAGMA table_info(interview_sessions)").fetchall()
            }
            self.assertIn("memory_digest", columns)
            self.assertIn("memory_window_json", columns)

            limit = memory_summary_limit()
            context = store.get_session_context(user_id="alice", session_id="legacy-mem")
            assert context is not None
            # Seeded from the existing summary text, capped at the shared limit.
            self.assertTrue(context["memory_digest"])
            self.assertTrue(context["memory_digest"].startswith("Turn 1 question: legacy one"))
            self.assertLessEqual(len(context["memory_digest"]), limit)
            # Verbatim window rebuilt from the stored turns, newest last.
            window = [item["turn_index"] for item in context["memory_window"]]
            self.assertEqual(window, [2, 3, 4, 5])
            self.assertEqual(
                context["memory_window"][-1]["question"],
                "legacy question 5",
            )
            store._conn.close()

            # Idempotent: reopening must not re-seed or drop the window.
            for _ in range(2):
                reopened = InterviewStore(db_path)
                again = reopened.get_session_context(user_id="alice", session_id="legacy-mem")
                assert again is not None
                self.assertEqual(again["memory_digest"], context["memory_digest"])
                self.assertEqual(
                    [item["turn_index"] for item in again["memory_window"]],
                    [2, 3, 4, 5],
                )
                reopened._conn.close()


class InterviewMemoryRouterTests(unittest.TestCase):
    def setUp(self):
        self.prev = {
            key: os.environ.get(key)
            for key in (
                "STUDY_ENABLE_MOCK_INTERVIEW_V1",
                "STUDY_MEMORY_SUMMARY_MAX_CHARS",
                "STUDY_MEMORY_VERBATIM_TURNS",
                "STUDY_MEMORY_ANSWER_EXCERPT_CHARS",
            )
        }
        os.environ["STUDY_ENABLE_MOCK_INTERVIEW_V1"] = "true"
        get_settings.cache_clear()
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tempdir.cleanup()
        for key, value in self.prev.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()

    def _client(self, llm: _LegacyMemoryLLM) -> TestClient:
        parser = DocParser(docs_path=os.path.join(os.getcwd(), "docs"))
        interview_sessions.init(
            llm,
            parser,
            LearningStore(os.path.join(self.tempdir.name, "learning.db")),
            os.path.join(self.tempdir.name, "interview.db"),
        )
        app = FastAPI()
        app.include_router(interview_sessions.router)
        app.dependency_overrides[require_auth] = (
            lambda: {"user": "alice", "provider": "local"}
        )
        return TestClient(app)

    def _run_turns(self, client: TestClient, session_id: str, turns: int) -> None:
        for index in range(1, turns + 1):
            answer = client.post(
                f"/api/interview-sessions/{session_id}/answer",
                json={"user_answer": f"answer {index} " * 40, "response_time_ms": 1000},
            )
            self.assertEqual(answer.status_code, 200, answer.text)
            if index < turns:
                next_q = client.post(
                    f"/api/interview-sessions/{session_id}/next-question",
                    json={},
                )
                self.assertEqual(next_q.status_code, 200, next_q.text)

    def test_router_and_store_share_one_memory_summary_limit(self):
        os.environ["STUDY_MEMORY_SUMMARY_MAX_CHARS"] = "512"
        get_settings.cache_clear()
        # One function, imported by both layers: the 2400/3000 divergence is gone.
        self.assertIs(interview_sessions.memory_summary_limit, interview_store.memory_summary_limit)
        self.assertEqual(memory_summary_limit(), 512)

        with tempfile.TemporaryDirectory() as td:
            store = InterviewStore(os.path.join(td, "limits.db"))
            session = store.create_session(
                user_id="alice",
                track="backend",
                level="mid",
                interview_type="technical",
                turn_count=3,
                target_role="Backend Engineer",
                job_description_text="",
                resume_summary_text="",
                focus_areas=[],
            )
            updated = store.set_memory_digest(
                user_id="alice",
                session_id=session["session_id"],
                digest="d" * 4000,
            )
            assert updated is not None
            self.assertEqual(len(updated["memory_digest"]), 512)

        session = {
            "memory_digest": "d" * 4000,
            "memory_window": [
                {"turn_index": index, "question": "q" * 400, "answer_excerpt": "a" * 400}
                for index in range(1, 8)
            ],
        }
        block = interview_sessions._memory_prompt_block(session)
        self.assertLessEqual(len(block), memory_summary_limit())
        self.assertEqual(len(block), 512)

    def test_prompt_receives_digest_plus_recent_turns_not_a_capped_append(self):
        llm = _LegacyMemoryLLM(digest="Compacted note: cursor pagination then rate limiting.")
        client = self._client(llm)
        created = client.post(
            "/api/interview-sessions",
            json={
                "track": "backend",
                "level": "mid",
                "interview_type": "technical",
                "turn_count": 4,
                "target_role": "Backend Engineer",
                "focus_areas": ["api design"],
            },
        )
        self.assertEqual(created.status_code, 200, created.text)
        session_id = created.json()["session"]["session_id"]
        self._run_turns(client, session_id, 3)
        # One more question, generated once three turns are on record, so the
        # last question prompt is built from post-turn memory.
        follow_up = client.post(
            f"/api/interview-sessions/{session_id}/next-question",
            json={},
        )
        self.assertEqual(follow_up.status_code, 200, follow_up.text)

        store = interview_sessions._store
        assert store is not None
        context = store.get_session_context(user_id="alice", session_id=session_id)
        assert context is not None
        # The digest is the revised note, not the previous note plus an append.
        self.assertEqual(context["memory_digest"], llm.digest)
        self.assertNotIn("Turn 1 question:", context["memory_digest"])
        self.assertEqual([item["turn_index"] for item in context["memory_window"]], [1, 2, 3])

        question_prompts = [p for p in llm.prompts if _QUESTION_MARKER in p]
        self.assertTrue(question_prompts)
        last_prompt = question_prompts[-1]
        # Digest plus the rolling window both reach the prompt.
        self.assertIn("Compacted note: cursor pagination then rate limiting.", last_prompt)
        self.assertIn("Turn 3 Q:", last_prompt)
        self.assertIn("Turn 2 Q:", last_prompt)

        # The evaluator sees the same memory block, otherwise the next turn is
        # scored without any history.
        eval_prompts = [
            p
            for p in llm.prompts
            if _QUESTION_MARKER not in p and _DIGEST_MARKER not in p
        ]
        self.assertTrue(eval_prompts)
        self.assertIn(
            "Compacted note: cursor pagination then rate limiting.",
            eval_prompts[-1],
        )
        self.assertIn("Turn 1 Q:", eval_prompts[-1])

    def test_small_digest_budget_folds_old_turns_without_losing_them_silently(self):
        os.environ["STUDY_MEMORY_SUMMARY_MAX_CHARS"] = "400"
        os.environ["STUDY_MEMORY_VERBATIM_TURNS"] = "4"
        get_settings.cache_clear()
        self.assertEqual(memory_summary_limit(), 400)

        llm = _LegacyMemoryLLM(digest_fails=True)
        client = self._client(llm)
        created = client.post(
            "/api/interview-sessions",
            json={
                "track": "backend",
                "level": "mid",
                "interview_type": "technical",
                "turn_count": 6,
                "target_role": "Backend Engineer",
                "focus_areas": ["api design"],
            },
        )
        session_id = created.json()["session"]["session_id"]
        self._run_turns(client, session_id, 6)
        # The provider is asked once per turn and every failure falls back to
        # deterministic compaction rather than dropping the turn.
        self.assertEqual(llm.digest_calls, 6)

        store = interview_sessions._store
        assert store is not None
        context = store.get_session_context(user_id="alice", session_id=session_id)
        assert context is not None
        digest = context["memory_digest"]
        self.assertLessEqual(len(digest), 400)
        self.assertIn("Turn 6", digest)
        self.assertIn("compacted", digest)
        # The dropped turns are accounted for rather than silently truncated.
        self.assertEqual([item["turn_index"] for item in context["memory_window"]], [3, 4, 5, 6])

    def test_degraded_evaluation_skips_the_digest_call_but_keeps_the_note(self):
        llm = _LegacyMemoryLLM(eval_error_code=CALL_FAILED_CODE)
        client = self._client(llm)
        created = client.post(
            "/api/interview-sessions",
            json={
                "track": "backend",
                "level": "mid",
                "interview_type": "technical",
                "turn_count": 3,
                "target_role": "Backend Engineer",
                "focus_areas": ["api design"],
            },
        )
        session_id = created.json()["session"]["session_id"]
        self._run_turns(client, session_id, 2)
        # The provider already failed for the evaluation; re-asking would only
        # repeat the failure and cost a second call.
        self.assertEqual(llm.digest_calls, 0)

        store = interview_sessions._store
        assert store is not None
        context = store.get_session_context(user_id="alice", session_id=session_id)
        assert context is not None
        self.assertIn("Turn 2", context["memory_digest"])
        self.assertEqual([item["turn_index"] for item in context["memory_window"]], [1, 2])

    def test_early_turns_reach_the_prompt_through_the_verbatim_window(self):
        os.environ["STUDY_MEMORY_VERBATIM_TURNS"] = "6"
        get_settings.cache_clear()

        llm = _LegacyMemoryLLM(digest="Everything so far: rate limiting, pagination, rollouts.")
        client = self._client(llm)
        created = client.post(
            "/api/interview-sessions",
            json={
                "track": "backend",
                "level": "mid",
                "interview_type": "technical",
                "turn_count": 6,
                "target_role": "Backend Engineer",
                "focus_areas": ["api design"],
            },
        )
        session_id = created.json()["session"]["session_id"]
        self._run_turns(client, session_id, 6)

        store = interview_sessions._store
        assert store is not None
        context = store.get_session_context(user_id="alice", session_id=session_id)
        assert context is not None
        self.assertEqual(
            [item["turn_index"] for item in context["memory_window"]],
            [1, 2, 3, 4, 5, 6],
        )
        block = interview_sessions._memory_prompt_block(context)
        self.assertLessEqual(len(block), memory_summary_limit())
        # Turn 1 is still in the window after six turns: the append-and-truncate
        # behaviour destroyed it around turn four.
        self.assertIn("Turn 1 Q:", block)
        self.assertIn("Turn 6 Q:", block)


if __name__ == "__main__":
    unittest.main()