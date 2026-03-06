import os
import tempfile
import unittest

from app.services.interview_store import InterviewStore


class InterviewStoreTests(unittest.TestCase):
    def test_session_turn_report_lifecycle(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "interview.db")
            store = InterviewStore(db_path)

            session = store.create_session(
                user_id="alice",
                track="backend",
                level="mid",
                interview_type="technical",
                turn_count=2,
                target_role="Backend Engineer",
                job_description_text="Build scalable APIs",
                resume_summary_text="Python backend developer",
                focus_areas=["api design"],
            )
            self.assertEqual(session["status"], "active")

            store.set_current_question(
                user_id="alice",
                session_id=session["session_id"],
                question="How would you design pagination for large datasets?",
            )

            turn, updated = store.record_turn(
                user_id="alice",
                session_id=session["session_id"],
                turn_index=1,
                question="How would you design pagination for large datasets?",
                user_answer="Use cursor pagination with stable sort keys.",
                rubric={
                    "technical_accuracy": 4,
                    "reasoning_depth": 4,
                    "communication_clarity": 4,
                    "completeness": 4,
                    "confidence_signal": 4,
                    "overall": 82,
                },
                strengths=["Good tradeoff awareness"],
                improvements=["Mention consistency edge cases"],
                follow_up_note="Add failure handling details.",
                response_time_ms=12000,
            )
            self.assertEqual(turn["turn_index"], 1)
            self.assertIsNotNone(updated)
            self.assertEqual(updated["status"], "active")

            report = store.save_report(
                user_id="alice",
                session_id=session["session_id"],
                report={
                    "overall_score": 82,
                    "readiness_label": "On Track",
                },
            )
            self.assertEqual(report["overall_score"], 82)

            stats = store.get_stats(user_id="alice")
            self.assertEqual(stats["total_sessions"], 1)
            self.assertAlmostEqual(stats["interview_readiness_score"], 82.0)


if __name__ == "__main__":
    unittest.main()
