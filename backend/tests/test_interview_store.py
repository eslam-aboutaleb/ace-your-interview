import os
import sqlite3
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
                interviewer_style="supportive",
                feedback_mode="deep",
                job_description_text="Build scalable APIs",
                resume_summary_text="Python backend developer",
                focus_areas=["api design"],
            )
            self.assertEqual(session["status"], "active")
            self.assertEqual(session["interviewer_style"], "supportive")
            self.assertEqual(session["feedback_mode"], "deep")
            self.assertEqual(session["memory_summary"], "")

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
            updated_with_memory = store.set_memory_summary(
                user_id="alice",
                session_id=session["session_id"],
                summary="Candidate needs stronger quantification.",
            )
            self.assertIsNotNone(updated_with_memory)
            assert updated_with_memory is not None
            self.assertIn("quantification", updated_with_memory["memory_summary"])

            stats = store.get_stats(user_id="alice")
            self.assertEqual(stats["total_sessions"], 1)
            self.assertAlmostEqual(stats["interview_readiness_score"], 82.0)

    def test_migrates_legacy_session_table_with_new_columns(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "legacy_interview.db")
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
            conn.close()

            store = InterviewStore(db_path)
            session_columns = {
                row["name"] for row in store._conn.execute("PRAGMA table_info(interview_sessions)").fetchall()
            }
            self.assertIn("interviewer_style", session_columns)
            self.assertIn("feedback_mode", session_columns)
            self.assertIn("memory_summary", session_columns)

            session = store.create_session(
                user_id="bob",
                track="backend",
                level="mid",
                interview_type="technical",
                turn_count=1,
                target_role="Backend Engineer",
                job_description_text="",
                resume_summary_text="",
                focus_areas=[],
            )
            self.assertEqual(session["interviewer_style"], "neutral")
            self.assertEqual(session["feedback_mode"], "concise")


if __name__ == "__main__":
    unittest.main()
