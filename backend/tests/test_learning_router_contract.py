import os
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.dependencies import require_auth
from app.routers import learning
from app.services.learning_planner import LearningPlannerStore
from app.services.learning_store import LearningStore


class LearningRouterContractTests(unittest.TestCase):
    def test_profile_diagnostic_recommendations_and_study_plan_flow(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "learning.db")
            learning_store = LearningStore(db_path)
            planner_store = LearningPlannerStore(db_path)
            learning.init(learning_store, planner_store)

            app = FastAPI()
            app.include_router(learning.router)
            app.dependency_overrides[require_auth] = lambda: {
                "user": "test-user",
                "provider": "local",
            }
            client = TestClient(app)

            profile_res = client.get("/api/learning/profile")
            self.assertEqual(profile_res.status_code, 200)
            self.assertEqual(profile_res.json()["primary_track"], "backend")

            update_res = client.put(
                "/api/learning/profile",
                json={
                    "target_role": "Senior Backend Engineer",
                    "target_date": "2026-03-20",
                    "primary_track": "backend",
                    "weekly_minutes": 360,
                    "preferred_modalities": ["study", "quiz", "interview"],
                    "confidence_by_track": {
                        "backend": 2,
                        "frontend": 3,
                        "system_design": 2,
                        "ai_stack": 3,
                    },
                },
            )
            self.assertEqual(update_res.status_code, 200)
            self.assertEqual(update_res.json()["target_role"], "Senior Backend Engineer")

            attempt_res = client.post(
                "/api/learning/attempts",
                json={
                    "question_id": "q1",
                    "topic_id": "01-backend-fundamentals-and-http",
                    "user_answer": "wrong",
                    "is_correct": False,
                    "confidence": 4,
                    "response_time_ms": 1200,
                    "mode": "quiz",
                },
            )
            self.assertEqual(attempt_res.status_code, 200)

            diagnostic_res = client.post("/api/learning/profile/diagnostic")
            self.assertEqual(diagnostic_res.status_code, 200)
            diagnostic = diagnostic_res.json()
            self.assertIn("competencies", diagnostic)
            self.assertGreaterEqual(len(diagnostic["competencies"]), 1)
            self.assertGreaterEqual(diagnostic["readiness_score"], 0.0)

            recommendations_res = client.get("/api/learning/recommendations?limit=4")
            self.assertEqual(recommendations_res.status_code, 200)
            recommendations = recommendations_res.json()
            self.assertLessEqual(len(recommendations["items"]), 4)
            self.assertTrue(recommendations["items"])
            self.assertIn(
                "interview",
                {item["recommendation_type"] for item in recommendations["items"]},
            )

            study_plan_res = client.get("/api/learning/study-plan?days=2&daily_items=2")
            self.assertEqual(study_plan_res.status_code, 200)
            study_plan = study_plan_res.json()
            self.assertEqual(study_plan["days"], 2)
            self.assertEqual(len(study_plan["days_plan"]), 2)
            self.assertLessEqual(study_plan["total_tasks"], 4)
            valid_types = {"review", "topic_study", "quiz", "interview"}
            for day in study_plan["days_plan"]:
                self.assertLessEqual(len(day["tasks"]), 2)
                for task in day["tasks"]:
                    self.assertIn(task["task_type"], valid_types)


if __name__ == "__main__":
    unittest.main()
