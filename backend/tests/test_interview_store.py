import os
import sqlite3
import tempfile
import unittest

from app.services.interview_store import InterviewStore, TurnConflictError


_LEGACY_TURNS_DDL = """
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

_RUBRIC = {
    "technical_accuracy": 4,
    "reasoning_depth": 4,
    "communication_clarity": 4,
    "completeness": 4,
    "confidence_signal": 4,
    "overall": 82,
}


def _create_legacy_db(db_path: str, duplicate_turns: bool = False) -> None:
    """Build a pre-migration database, optionally holding duplicate turn rows."""
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
        conn.execute(_LEGACY_TURNS_DDL)
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
                current_question, created_at, updated_at
            ) VALUES ('legacy-1', 'alice', 'backend', 'mid', 'technical', 3, 1,
                'active', 'Backend Engineer', '', '', '[]', '[]', '', 't', 't')
            """
        )
        # Two submissions of the SAME turn_index, as a pre-constraint race leaves behind.
        rows = [
            (1, "kept answer"),
            (1, "duplicate answer"),
        ] if duplicate_turns else [(1, "kept answer")]
        for turn_index, answer in rows:
            conn.execute(
                """
                INSERT INTO interview_turns(
                    session_id, user_id, turn_index, question, user_answer,
                    rubric_json, strengths_json, improvements_json,
                    follow_up_note, response_time_ms, created_at
                ) VALUES ('legacy-1', 'alice', ?, 'Q', ?, '{}', '[]', '[]', '', 1000, 't')
                """,
                (turn_index, answer),
            )
    conn.close()


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

    def test_degraded_column_migration_is_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "degraded_migration.db")
            _create_legacy_db(db_path)

            for _ in range(3):
                store = InterviewStore(db_path)
                columns = {
                    row["name"]
                    for row in store._conn.execute("PRAGMA table_info(interview_turns)").fetchall()
                }
                self.assertIn("degraded", columns)
                rows = store._conn.execute(
                    "SELECT degraded FROM interview_turns"
                ).fetchall()
                self.assertEqual([int(r["degraded"]) for r in rows], [0])
                store._conn.close()

    def test_degraded_flag_round_trips(self):
        with tempfile.TemporaryDirectory() as td:
            store = InterviewStore(os.path.join(td, "interview.db"))
            session = store.create_session(
                user_id="alice",
                track="backend",
                level="mid",
                interview_type="technical",
                turn_count=2,
                target_role="Backend Engineer",
                job_description_text="",
                resume_summary_text="",
                focus_areas=[],
            )

            store.record_turn(
                user_id="alice",
                session_id=session["session_id"],
                turn_index=1,
                question="Q1",
                user_answer="A1",
                rubric={key: None for key in _RUBRIC},
                strengths=["Attempted the prompt"],
                improvements=["Retry for scored feedback"],
                follow_up_note="note",
                response_time_ms=1000,
                degraded=True,
            )
            store.record_turn(
                user_id="alice",
                session_id=session["session_id"],
                turn_index=2,
                question="Q2",
                user_answer="A2",
                rubric=_RUBRIC,
                strengths=["Clear"],
                improvements=["Measure"],
                follow_up_note="note",
                response_time_ms=1000,
            )

            turns = store.get_turns(user_id="alice", session_id=session["session_id"])
            self.assertEqual([t["degraded"] for t in turns], [True, False])
            self.assertIsNone(turns[0]["rubric"]["overall"])

    def test_duplicate_turn_index_raises_typed_conflict(self):
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
            kwargs = dict(
                user_id="alice",
                session_id=session["session_id"],
                turn_index=1,
                question="Q1",
                user_answer="A1",
                rubric=_RUBRIC,
                strengths=["Clear"],
                improvements=["Measure"],
                follow_up_note="note",
                response_time_ms=1000,
            )
            store.record_turn(**kwargs)
            with self.assertRaises(TurnConflictError):
                store.record_turn(**{**kwargs, "user_answer": "A1 again"})

            turns = store.get_turns(user_id="alice", session_id=session["session_id"])
            self.assertEqual(len(turns), 1)
            self.assertEqual(turns[0]["user_answer"], "A1")
            self.assertEqual(TurnConflictError.code, "turn_conflict")

    def test_dedup_migration_removes_duplicates_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "dedup.db")
            _create_legacy_db(db_path, duplicate_turns=True)

            store = InterviewStore(db_path)
            rows = store._conn.execute(
                "SELECT id, turn_index, user_answer FROM interview_turns ORDER BY id"
            ).fetchall()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["user_answer"], "kept answer")
            store._conn.close()

            probe = sqlite3.connect(db_path)
            indexes = {row[1] for row in probe.execute("PRAGMA index_list(interview_turns)")}
            probe.close()
            self.assertIn("idx_interview_turns_unique_turn", indexes)

            # Re-opening an already-migrated database must not delete anything.
            reopened = InterviewStore(db_path)
            again = reopened._conn.execute(
                "SELECT id, turn_index FROM interview_turns ORDER BY id"
            ).fetchall()
            self.assertEqual([(r["id"], r["turn_index"]) for r in again], [(1, 1)])

    def test_unique_index_blocks_duplicate_inserts_after_migration(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "unique.db")
            _create_legacy_db(db_path, duplicate_turns=True)
            store = InterviewStore(db_path)
            with self.assertRaises(sqlite3.IntegrityError):
                store._conn.execute(
                    """
                    INSERT INTO interview_turns(
                        session_id, user_id, turn_index, question, user_answer,
                        rubric_json, strengths_json, improvements_json,
                        follow_up_note, response_time_ms, created_at
                    ) VALUES ('legacy-1', 'alice', 1, 'Q', 'A', '{}', '[]', '[]', '', 1, 't')
                    """
                )

    def test_session_context_turns_completed_matches_stored_turns(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "context.db")
            _create_legacy_db(db_path, duplicate_turns=True)
            store = InterviewStore(db_path)

            context = store.get_session_context(user_id="alice", session_id="legacy-1")
            assert context is not None
            # The legacy row claims 1 turn, and exactly one turn survives migration.
            self.assertEqual(context["turns_completed"], len(store.get_turns(user_id="alice", session_id="legacy-1")))
            self.assertTrue(
                store.turn_exists(user_id="alice", session_id="legacy-1", turn_index=1)
            )
            self.assertFalse(
                store.turn_exists(user_id="alice", session_id="legacy-1", turn_index=2)
            )


if __name__ == "__main__":
    unittest.main()
