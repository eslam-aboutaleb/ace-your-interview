import os
import tempfile
import unittest

from app.services.interview_store import InterviewStore


class InterviewStoreTrendsTests(unittest.TestCase):
    def test_get_trends_returns_chronological_points_and_delta(self):
        with tempfile.TemporaryDirectory() as td:
            store = InterviewStore(os.path.join(td, "interview.db"))
            first = store.create_session(
                user_id="alice",
                track="backend",
                level="mid",
                interview_type="technical",
                turn_count=1,
                target_role="Backend Engineer",
                job_description_text="",
                resume_summary_text="",
                focus_areas=["api design"],
            )
            second = store.create_session(
                user_id="alice",
                track="backend",
                level="mid",
                interview_type="coding",
                turn_count=1,
                target_role="Backend Engineer",
                job_description_text="",
                resume_summary_text="",
                focus_areas=["algorithms"],
            )

            store.save_report(
                user_id="alice",
                session_id=first["session_id"],
                report={
                    "overall_score": 68,
                    "readiness_label": "Developing",
                    "rubric_averages": {
                        "technical_accuracy": 3.2,
                        "reasoning_depth": 3.1,
                        "communication_clarity": 3.0,
                        "completeness": 3.1,
                        "confidence_signal": 3.0,
                        "overall": 68,
                    },
                },
            )
            store.save_report(
                user_id="alice",
                session_id=second["session_id"],
                report={
                    "overall_score": 77,
                    "readiness_label": "On Track",
                    "rubric_averages": {
                        "technical_accuracy": 3.8,
                        "reasoning_depth": 3.6,
                        "communication_clarity": 3.4,
                        "completeness": 3.5,
                        "confidence_signal": 3.3,
                        "overall": 77,
                    },
                },
            )

            trends = store.get_trends(user_id="alice", limit=50)
            self.assertEqual(len(trends["points"]), 2)
            self.assertEqual(trends["points"][-1]["session_id"], second["session_id"])
            self.assertEqual(trends["summary"]["latest_score"], 77.0)
            self.assertEqual(trends["summary"]["previous_score"], 68.0)
            self.assertEqual(trends["summary"]["delta"], 9.0)


if __name__ == "__main__":
    unittest.main()
