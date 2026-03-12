import os
import tempfile
import unittest

from app.services.learning_planner import LearningPlannerStore
from app.services.learning_store import LearningStore


class LearningPlannerStoreTests(unittest.TestCase):
    def test_profile_defaults_and_upsert(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "learning.db")
            planner = LearningPlannerStore(db_path)

            initial = planner.get_profile(user_id="alice")
            self.assertEqual(initial["primary_track"], "backend")
            self.assertEqual(initial["weekly_minutes"], 240)

            saved = planner.upsert_profile(
                user_id="alice",
                payload={
                    "target_role": "Senior Backend Engineer",
                    "target_date": "2026-06-01",
                    "weekly_minutes": 360,
                    "primary_track": "system_design",
                    "preferred_modalities": ["study", "interview"],
                },
            )
            self.assertEqual(saved["target_role"], "Senior Backend Engineer")
            self.assertEqual(saved["target_date"], "2026-06-01")
            self.assertEqual(saved["weekly_minutes"], 360)
            self.assertEqual(saved["primary_track"], "system_design")
            self.assertEqual(saved["preferred_modalities"], ["study", "interview"])

    def test_diagnostic_creates_topic_and_track_competencies(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "learning.db")
            learning = LearningStore(db_path)
            planner = LearningPlannerStore(db_path)

            learning.record_attempt(
                user_id="alice",
                question_id="q1",
                topic_id="01-backend-fundamentals-and-http",
                user_answer="wrong",
                is_correct=False,
                confidence=5,
                response_time_ms=1200,
                mode="study",
            )
            learning.record_attempt(
                user_id="alice",
                question_id="q2",
                topic_id="06-frontend-core-architecture",
                user_answer="right",
                is_correct=True,
                confidence=3,
                response_time_ms=900,
                mode="study",
            )

            diagnostic = planner.run_diagnostic(user_id="alice")
            competency_ids = {item["competency_id"] for item in diagnostic["competencies"]}
            self.assertIn("topic:01-backend-fundamentals-and-http", competency_ids)
            self.assertIn("track:backend", competency_ids)
            self.assertIn("track:frontend", competency_ids)
            self.assertGreaterEqual(diagnostic["readiness_score"], 0.0)
            self.assertTrue(diagnostic["urgent_competencies"])

    def test_recommendations_and_study_plan_use_profile_and_deadline(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "learning.db")
            learning = LearningStore(db_path)
            planner = LearningPlannerStore(db_path)

            planner.upsert_profile(
                user_id="alice",
                payload={
                    "target_role": "Senior Backend Engineer",
                    "target_date": "2026-03-20",
                    "primary_track": "backend",
                    "preferred_modalities": ["study", "quiz", "interview"],
                },
            )
            learning.record_attempt(
                user_id="alice",
                question_id="q1",
                topic_id="01-backend-fundamentals-and-http",
                user_answer="wrong",
                is_correct=False,
                confidence=4,
                response_time_ms=1000,
                mode="quiz",
            )

            recommendations = planner.build_recommendations(user_id="alice", limit=10)
            recommendation_types = {
                item["recommendation_type"] for item in recommendations["items"]
            }
            self.assertIn("interview", recommendation_types)
            self.assertTrue({"review", "study"} & recommendation_types)

            study_plan = planner.build_study_plan(user_id="alice", days=3, daily_items=2)
            self.assertEqual(study_plan["days"], 3)
            self.assertEqual(len(study_plan["days_plan"]), 3)
            self.assertGreaterEqual(study_plan["total_tasks"], 1)


if __name__ == "__main__":
    unittest.main()
