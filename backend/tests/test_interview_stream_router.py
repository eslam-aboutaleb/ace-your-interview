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
from app.services.learning_store import LearningStore


class FakeLLM:
    def __init__(self):
        self.question_n = 0

    async def completion(self, prompt, llm_config=None, user_identity=None):
        if "Generate the NEXT interview question" in prompt:
            await asyncio.sleep(1.0)
            self.question_n += 1
            return {
                "success": True,
                "analysis": (
                    '{"question":"What tradeoff would you make in API pagination design #'
                    + str(self.question_n)
                    + '?","competency_focus":"tradeoff analysis","expected_signals":["constraints","consistency","client impact"]}'
                ),
                "metadata": {},
            }
        await asyncio.sleep(1.0)
        return {
            "success": True,
            "analysis": (
                '{"rubric":{"technical_accuracy":4,"reasoning_depth":4,"communication_clarity":4,'
                '"completeness":4,"confidence_signal":4,"overall":80},'
                '"strengths":["Clear tradeoff reasoning"],"improvements":["Add a concrete metric"],'
                '"follow_up_note":"Quantify expected latency improvements next time."}'
            ),
            "metadata": {},
        }


class InterviewStreamRouterTests(unittest.TestCase):
    def setUp(self):
        self.prev_flag = os.environ.get("STUDY_ENABLE_MOCK_INTERVIEW_V1")
        os.environ["STUDY_ENABLE_MOCK_INTERVIEW_V1"] = "true"
        get_settings.cache_clear()

        self.tempdir = tempfile.TemporaryDirectory()
        parser = DocParser(docs_path=os.path.join(os.getcwd(), "docs"))
        learning_store = LearningStore(os.path.join(self.tempdir.name, "learning.db"))
        interview_sessions.init(
            FakeLLM(),
            parser,
            learning_store,
            os.path.join(self.tempdir.name, "interview.db"),
        )

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

    def _stream_events(self, path: str, payload: dict) -> tuple[int, list[dict]]:
        with self.client.stream("POST", path, json=payload) as res:
            events = []
            for line in res.iter_lines():
                if not line:
                    continue
                text = line.decode("utf-8") if isinstance(line, bytes) else line
                events.append(json.loads(text))
            return res.status_code, events

    def test_create_session_stream_emits_done_payload_with_session_shape(self):
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


if __name__ == "__main__":
    unittest.main()
