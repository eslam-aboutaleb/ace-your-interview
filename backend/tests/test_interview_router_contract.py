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
            self.question_n += 1
            return {
                "success": True,
                "analysis": (
                    '{"question":"What tradeoff would you make in API pagination design #' + str(self.question_n) +
                    '?","competency_focus":"tradeoff analysis","expected_signals":["constraints","consistency","client impact"]}'
                ),
                "metadata": {},
            }
        return {
            "success": True,
            "analysis": '{"rubric":{"technical_accuracy":4,"reasoning_depth":4,"communication_clarity":4,"completeness":4,"confidence_signal":4,"overall":80},"strengths":["Clear tradeoff reasoning"],"improvements":["Add a concrete metric"],"follow_up_note":"Quantify expected latency improvements next time."}',
            "metadata": {},
        }


class InterviewRouterContractTests(unittest.TestCase):
    def test_create_submit_and_report_flow(self):
        prev_flag = os.environ.get("STUDY_ENABLE_MOCK_INTERVIEW_V1")
        try:
            with tempfile.TemporaryDirectory() as td:
                os.environ["STUDY_ENABLE_MOCK_INTERVIEW_V1"] = "true"
                get_settings.cache_clear()

                parser = DocParser(docs_path=os.path.join(os.getcwd(), "docs"))
                learning_store = LearningStore(os.path.join(td, "learning.db"))
                interview_sessions.init(FakeLLM(), parser, learning_store, os.path.join(td, "interview.db"))

                app = FastAPI()
                app.include_router(interview_sessions.router)
                app.dependency_overrides[require_auth] = lambda: {"user": "test-user", "provider": "local"}
                client = TestClient(app)

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
        finally:
            if prev_flag is None:
                os.environ.pop("STUDY_ENABLE_MOCK_INTERVIEW_V1", None)
            else:
                os.environ["STUDY_ENABLE_MOCK_INTERVIEW_V1"] = prev_flag
            get_settings.cache_clear()


if __name__ == "__main__":
    unittest.main()
