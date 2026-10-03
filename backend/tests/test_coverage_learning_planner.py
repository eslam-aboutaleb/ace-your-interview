"""Coverage for app/services/learning_planner.py.

Complements ``tests/test_learning_planner.py`` (happy paths) with the input
sanitisers, the JSON fallbacks, the cold-start branch, the quiz/interview
modality branches and the degenerate study-plan paths.
"""

import os
import sqlite3
import tempfile
import unittest
from datetime import UTC, date, datetime, timedelta

from app.services import learning_planner as lp
from app.services.learning_planner import LearningPlannerStore
from app.services.learning_store import LearningStore


class _Store:
    """Temporary-file store pair, one per test."""

    def __enter__(self):
        self._td = tempfile.TemporaryDirectory()
        db_path = os.path.join(self._td.name, "learning.db")
        self.learning = LearningStore(db_path)
        self.planner = LearningPlannerStore(db_path)
        return self

    def __exit__(self, *exc_info):
        self._td.cleanup()
        return False


def _iso_in(days: int) -> str:
    return (datetime.now(UTC) + timedelta(days=days)).date().isoformat()


class JsonFallbackTests(unittest.TestCase):
    def test_loads_json_list_returns_empty_for_non_list_json(self):
        self.assertEqual(lp._loads_json_list("[1, 2]"), [1, 2])
        self.assertEqual(lp._loads_json_list('{"a": 1}'), [])
        self.assertEqual(lp._loads_json_list(""), [])

    def test_loads_json_dict_returns_empty_for_non_dict_json(self):
        self.assertEqual(lp._loads_json_dict('{"a": 1}'), {"a": 1})
        self.assertEqual(lp._loads_json_dict("[1]"), {})
        self.assertEqual(lp._loads_json_dict("nope"), {})


class SanitizerTests(unittest.TestCase):
    def test_sanitize_date_accepts_only_iso_days(self):
        self.assertEqual(
            LearningPlannerStore._sanitize_date(" 2026-06-01 "), "2026-06-01"
        )
        self.assertEqual(LearningPlannerStore._sanitize_date("2026-13-01"), "")
        self.assertEqual(LearningPlannerStore._sanitize_date(""), "")
        self.assertEqual(LearningPlannerStore._sanitize_date(None), "")

    def test_sanitize_level_and_track_fall_back_on_unknown_values(self):
        self.assertEqual(LearningPlannerStore._sanitize_level("Senior", "mid"), "senior")
        self.assertEqual(LearningPlannerStore._sanitize_level("expert", "mid"), "mid")
        self.assertEqual(LearningPlannerStore._sanitize_track("AI_Stack", "backend"), "ai_stack")
        self.assertEqual(LearningPlannerStore._sanitize_track("infra", "backend"), "backend")

    def test_sanitize_minutes_clamps_and_falls_back_on_non_numbers(self):
        self.assertEqual(
            LearningPlannerStore._sanitize_minutes(99999, fallback=30, minimum=10, maximum=180),
            180,
        )
        self.assertEqual(
            LearningPlannerStore._sanitize_minutes(1, fallback=30, minimum=10, maximum=180), 10
        )
        self.assertEqual(
            LearningPlannerStore._sanitize_minutes("not-a-number", fallback=42, minimum=10, maximum=180),
            42,
        )
        self.assertEqual(
            LearningPlannerStore._sanitize_minutes(None, fallback=42, minimum=10, maximum=180),
            42,
        )

    def test_sanitize_string_list_trims_dedupes_and_limits(self):
        # Blanks are dropped, a duplicate does not consume the limit, and the
        # loop stops the moment the limit is reached.
        self.assertEqual(
            LearningPlannerStore._sanitize_string_list(
                [" a ", "", "a", "b", "c"], limit=3, max_len=2
            ),
            ["a", "b", "c"],
        )
        # Truncation happens before the dedupe check, so "ab" and "abc" collide.
        self.assertEqual(
            LearningPlannerStore._sanitize_string_list(["abc", "abd"], limit=5, max_len=2),
            ["ab"],
        )
        self.assertEqual(
            LearningPlannerStore._sanitize_string_list("not-a-list", limit=5, max_len=10), []
        )

    def test_sanitize_modalities_filters_unknown_and_defaults_when_empty(self):
        self.assertEqual(
            LearningPlannerStore._sanitize_modalities(["quiz", "telepathy"]),
            ["quiz"],
        )
        self.assertEqual(
            LearningPlannerStore._sanitize_modalities(["telepathy"]),
            list(lp._DEFAULT_MODALITIES),
        )
        self.assertEqual(LearningPlannerStore._sanitize_modalities(None), list(lp._DEFAULT_MODALITIES))

    def test_sanitize_confidence_map_clamps_and_backfills(self):
        self.assertEqual(
            LearningPlannerStore._sanitize_confidence_map({"backend": 9, "frontend": 0}),
            {"backend": 5, "frontend": 1, "system_design": 3, "ai_stack": 3},
        )
        self.assertEqual(
            LearningPlannerStore._sanitize_confidence_map({"backend": "four"}),
            lp._DEFAULT_CONFIDENCE,
        )
        self.assertEqual(LearningPlannerStore._sanitize_confidence_map("nope"), lp._DEFAULT_CONFIDENCE)


class TopicClassifierTests(unittest.TestCase):
    def test_topic_track_maps_numbered_prefixes_to_tracks(self):
        cases = {
            "custom-anything": "backend",
            "00-problem-solving": "backend",
            "01-http": "backend",
            "05-http": "backend",
            "06-frontend": "frontend",
            "09-frontend": "frontend",
            "10-system-design": "system_design",
            "22-anything": "system_design",
            "13-ai-stack": "ai_stack",
            "16-ai-stack": "ai_stack",
            "unnumbered-topic": "backend",
            "": "backend",
        }
        for topic_id, expected in cases.items():
            with self.subTest(topic_id=topic_id):
                self.assertEqual(LearningPlannerStore._topic_track(topic_id), expected)

    def test_topic_label_strips_numbering_and_extensions(self):
        self.assertEqual(
            LearningPlannerStore._topic_label("01-backend-fundamentals-and-http"),
            "Backend Fundamentals And Http",
        )
        self.assertEqual(
            LearningPlannerStore._topic_label("06-frontend-core:advanced"), "Frontend Core"
        )
        self.assertEqual(LearningPlannerStore._topic_label(""), "Topic")
        # All-numeric ids keep their original text rather than going empty.
        self.assertEqual(LearningPlannerStore._topic_label("01-02"), "01 02")

    def test_deadline_days_handles_absent_unparseable_and_real_dates(self):
        self.assertIsNone(LearningPlannerStore._deadline_days(""))
        self.assertIsNone(LearningPlannerStore._deadline_days("soon"))
        self.assertEqual(
            LearningPlannerStore._deadline_days(_iso_in(10)),
            (date.fromisoformat(_iso_in(10)) - datetime.now(UTC).date()).days,
        )


class ProfilePersistenceTests(unittest.TestCase):
    def test_run_async_passes_kwargs_through(self):
        import asyncio

        with _Store() as stores:
            saved = asyncio.run(
                stores.planner.run_async(
                    stores.planner.upsert_profile,
                    user_id="kwargs-user",
                    payload={"target_role": "Staff Engineer"},
                )
            )
            self.assertEqual(saved["target_role"], "Staff Engineer")

    def test_run_async_supports_positional_arguments(self):
        import asyncio

        with _Store() as stores:
            label = asyncio.run(stores.planner.run_async(LearningPlannerStore._topic_label, "01-http"))
            self.assertEqual(label, "Http")

    def test_row_to_profile_tolerates_corrupt_json_columns(self):
        with _Store() as stores:
            stores.planner.get_profile(user_id="corrupt")
            stores.planner._conn.execute(
                """
                UPDATE learner_profiles
                SET focus_topic_ids_json = ?, confidence_by_track_json = ?
                WHERE user_id = ?
                """,
                ("{not json", "[1,2]", "corrupt"),
            )
            profile = stores.planner.get_profile(user_id="corrupt")
            self.assertEqual(profile["focus_topic_ids"], [])
            self.assertEqual(profile["confidence_by_track"], lp._DEFAULT_CONFIDENCE)

    def test_upsert_clones_garbage_into_safe_defaults(self):
        with _Store() as stores:
            saved = stores.planner.upsert_profile(
                user_id="alice",
                payload={
                    "target_role": "   ",
                    "target_date": "whenever",
                    "weekly_minutes": "abc",
                    "preferred_session_minutes": 100000,
                    "current_level": "wizard",
                    "target_level": None,
                    "primary_track": "kubernetes",
                    "focus_topic_ids": "not-a-list",
                    "target_companies": ["", "  ", "Stripe"],
                    "preferred_modalities": ["telepathy"],
                    "confidence_by_track": "not-a-dict",
                },
            )
            self.assertEqual(saved["target_role"], "")
            self.assertEqual(saved["target_date"], "")
            self.assertEqual(saved["weekly_minutes"], 240)
            self.assertEqual(saved["preferred_session_minutes"], 180)
            self.assertEqual(saved["current_level"], "mid")
            self.assertEqual(saved["target_level"], "senior")
            self.assertEqual(saved["primary_track"], "backend")
            self.assertEqual(saved["focus_topic_ids"], [])
            self.assertEqual(saved["target_companies"], ["Stripe"])
            self.assertEqual(saved["preferred_modalities"], list(lp._DEFAULT_MODALITIES))
            self.assertEqual(saved["confidence_by_track"], lp._DEFAULT_CONFIDENCE)

    def test_upsert_preserves_created_at_and_diagnostic_timestamp(self):
        with _Store() as stores:
            stores.planner.run_diagnostic(user_id="alice")
            after_diagnostic = stores.planner.get_profile(user_id="alice")
            self.assertTrue(after_diagnostic["diagnostic_updated_at"])

            stores.planner.upsert_profile(user_id="alice", payload={"target_role": "New Title"})
            after_upsert = stores.planner.get_profile(user_id="alice")
            self.assertEqual(
                after_upsert["diagnostic_updated_at"], after_diagnostic["diagnostic_updated_at"]
            )
            self.assertEqual(after_upsert["created_at"], after_diagnostic["created_at"])

    def test_profile_table_is_created_without_a_parent_directory(self):
        with tempfile.TemporaryDirectory() as td:
            planner = LearningPlannerStore(os.path.join(td, "learning.db"))
            self.assertEqual(planner.db_path, os.path.join(td, "learning.db"))
            self.assertEqual(planner.get_profile(user_id="x")["primary_track"], "backend")

    def test_table_exists_reports_created_and_absent_tables(self):
        with _Store() as stores:
            self.assertTrue(stores.planner._table_exists("learner_profiles"))
            self.assertTrue(stores.planner._table_exists("learner_competency_scores"))
            # LearningStore owns question_progress; the planner only reads it.
            self.assertTrue(stores.planner._table_exists("question_progress"))
            self.assertFalse(stores.planner._table_exists("interview_reports"))
            self.assertFalse(stores.planner._table_exists("no_such_table"))


class DiagnosticTests(unittest.TestCase):
    def test_diagnostic_without_any_history_scores_tracks_from_confidence(self):
        with _Store() as stores:
            diagnostic = stores.planner.run_diagnostic(user_id="cold")
            tracks = {
                item["competency_id"]: item
                for item in diagnostic["competencies"]
                if item["competency_type"] == "track"
            }
            self.assertEqual(len(tracks), 4)
            # Primary track has no evidence, so it scores from the 1-5 self-rating.
            self.assertEqual(tracks["track:backend"]["score"], 60.0)
            self.assertEqual(tracks["track:backend"]["evidence_count"], 0)
            self.assertEqual(tracks["track:backend"]["priority"], 60.0)
            self.assertEqual(tracks["track:frontend"]["priority"], 35.0)
            self.assertEqual(diagnostic["readiness_score"], 60.0)

    def test_diagnostic_reads_interview_report_rubrics_when_present(self):
        with _Store() as stores:
            stores.learning._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interview_reports (
                    user_id TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            now = datetime.now(UTC).isoformat()
            stores.learning._conn.execute(
                "INSERT INTO interview_reports(user_id, report_json, updated_at) VALUES (?, ?, ?)",
                (
                    "alice",
                    '{"rubric_averages": {"reasoning_depth": 4, "completeness": 2}, "overall_score": 3}',
                    now,
                ),
            )
            stores.learning._conn.execute(
                "INSERT INTO interview_reports(user_id, report_json, updated_at) VALUES (?, ?, ?)",
                ("alice", "not-json", now),
            )
            stores.learning._conn.execute(
                "INSERT INTO interview_reports(user_id, report_json, updated_at) VALUES (?, ?, ?)",
                ("alice", '{"overall_score": null}', now),
            )
            stores.learning._conn.commit()

            diagnostic = stores.planner.run_diagnostic(user_id="alice")
            dimensions = {
                item["competency_id"]: item
                for item in diagnostic["competencies"]
                if item["competency_type"] == "interview_dimension"
            }
            self.assertEqual(len(dimensions), 5)
            self.assertEqual(dimensions["interview:reasoning_depth"]["score"], 80.0)
            self.assertEqual(dimensions["interview:completeness"]["score"], 40.0)
            self.assertEqual(dimensions["interview:reasoning_depth"]["evidence_count"], 1)
            # Keys absent from rubric_averages score 0.0 rather than being
            # skipped, so they rank as the weakest dimension. See the bug report.
            self.assertEqual(dimensions["interview:technical_accuracy"]["score"], 0.0)
            self.assertEqual(dimensions["interview:technical_accuracy"]["evidence_count"], 1)
            # overall_score 3 feeds the readiness blend alongside track scores.
            self.assertGreater(diagnostic["readiness_score"], 0.0)

    def test_dimension_is_skipped_only_when_no_report_carries_a_rubric(self):
        with _Store() as stores:
            stores.learning._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interview_reports (
                    user_id TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            now = datetime.now(UTC).isoformat()
            stores.learning._conn.execute(
                "INSERT INTO interview_reports(user_id, report_json, updated_at) VALUES (?, ?, ?)",
                ("alice", '{"overall_score": 4}', now),
            )
            stores.learning._conn.execute(
                "INSERT INTO interview_reports(user_id, report_json, updated_at) VALUES (?, ?, ?)",
                ("alice", '{"rubric_averages": "not-a-dict"}', now),
            )
            stores.learning._conn.execute(
                "INSERT INTO interview_reports(user_id, report_json, updated_at) VALUES (?, ?, ?)",
                ("alice", "unparseable", now),
            )
            stores.learning._conn.commit()

            diagnostic = stores.planner.run_diagnostic(user_id="alice")
            dimensions = [
                item
                for item in diagnostic["competencies"]
                if item["competency_type"] == "interview_dimension"
            ]
            self.assertEqual(dimensions, [])
            # overall_score still counts toward readiness even with no rubric.
            self.assertGreater(diagnostic["readiness_score"], 0.0)

    def test_diagnostic_clamps_out_of_range_rubric_scores(self):
        with _Store() as stores:
            stores.learning._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interview_reports (
                    user_id TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            stores.learning._conn.execute(
                "INSERT INTO interview_reports(user_id, report_json, updated_at) VALUES (?, ?, ?)",
                (
                    "bob",
                    '{"rubric_averages": {"communication_clarity": 99, "confidence_signal": -5}}',
                    datetime.now(UTC).isoformat(),
                ),
            )
            stores.learning._conn.commit()
            diagnostic = stores.planner.run_diagnostic(user_id="bob")
            scores = {
                item["competency_id"]: item["score"]
                for item in diagnostic["competencies"]
                if item["competency_type"] == "interview_dimension"
            }
            self.assertEqual(scores["interview:communication_clarity"], 100.0)
            self.assertEqual(scores["interview:confidence_signal"], 0.0)

    def test_diagnostic_sorts_by_priority_desc_then_score_then_label(self):
        with _Store() as stores:
            for topic_id, correct in (
                ("01-backend", False),
                ("06-frontend", True),
                ("10-system-design", False),
            ):
                stores.learning.record_attempt(
                    user_id="alice",
                    question_id=f"q-{topic_id}",
                    topic_id=topic_id,
                    user_answer="a",
                    is_correct=correct,
                    confidence=3,
                    response_time_ms=800,
                    mode="quiz",
                )
            diagnostic = stores.planner.run_diagnostic(user_id="alice")
            priorities = [item["priority"] for item in diagnostic["competencies"]]
            self.assertEqual(priorities, sorted(priorities, reverse=True))
            self.assertLessEqual(len(diagnostic["strongest_competencies"]), 3)
            self.assertLessEqual(len(diagnostic["urgent_competencies"]), 5)

    def test_list_competencies_respects_the_limit_and_rounds(self):
        with _Store() as stores:
            stores.learning.record_attempt(
                user_id="alice",
                question_id="q1",
                topic_id="01-backend",
                user_answer="a",
                is_correct=False,
                confidence=5,
                response_time_ms=800,
                mode="quiz",
            )
            stores.planner.run_diagnostic(user_id="alice")
            limited = stores.planner.list_competencies(user_id="alice", limit=2)
            self.assertEqual(len(limited), 2)
            # LIMIT is floored at 1 rather than returning nothing.
            self.assertEqual(len(stores.planner.list_competencies(user_id="alice", limit=0)), 1)
            self.assertEqual(len(stores.planner.list_competencies(user_id="alice", limit=-5)), 1)
            entry = next(i for i in limited if i["competency_id"].startswith("track:"))
            self.assertEqual(entry["source"], "profile_plus_learning")

    def test_diagnostic_tolerates_null_aggregate_columns(self):
        with _Store() as stores:
            # The NOT NULL columns are satisfied with 0/NULL-free sentinels so
            # the aggregate expressions in the diagnostic are the only thing
            # under test; a zero row still exercises every COALESCE path.
            stores.learning._conn.execute(
                """
                INSERT INTO question_progress(
                    user_id, question_id, topic_id, mastery_score, due_at,
                    review_bucket, attempts, correct_attempts, avg_confidence,
                    last_confidence, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "alice",
                    "q-nulls",
                    "01-nulls",
                    0.0,
                    datetime.now(UTC).isoformat(),
                    0,
                    0,
                    0,
                    0.0,
                    0,
                    datetime.now(UTC).isoformat(),
                ),
            )
            stores.learning._conn.commit()
            diagnostic = stores.planner.run_diagnostic(user_id="alice")
            entry = next(
                i for i in diagnostic["competencies"] if i["competency_id"] == "topic:01-nulls"
            )
            self.assertEqual(entry["score"], 0.0)
            # A zero-attempt row with a due date in the past still counts as due,
            # which is what makes due_pressure reach its maximum.
            self.assertEqual(entry["metadata"]["due_count"], 1)
            self.assertEqual(entry["metadata"]["accuracy"], 0.0)
            self.assertEqual(entry["evidence_count"], 0)
            # A null avg_confidence must not divide by zero or blow the priority cap.
            self.assertLessEqual(entry["priority"], 100.0)


class RecommendationTests(unittest.TestCase):
    def test_cold_start_uses_the_primary_track_topic_list(self):
        with _Store() as stores:
            stores.planner.upsert_profile(
                user_id="alice",
                payload={
                    "primary_track": "ai_stack",
                    "target_role": "AI Engineer",
                    "preferred_modalities": ["voice"],
                },
            )
            recs = stores.planner.build_recommendations(user_id="alice", limit=20)
            ids = [item["recommendation_id"] for item in recs["items"]]
            self.assertIn("study:13-ai-stack-llm-and-prompting", ids)
            self.assertIn("interview:cold-start", ids)
            study = next(i for i in recs["items"] if i["recommendation_id"].startswith("study:"))
            self.assertIn("cold_start", study["reason_codes"])
            self.assertEqual(study["track"], "ai_stack")
            self.assertTrue(recs["items"] == sorted(
                recs["items"],
                key=lambda item: (-float(item["priority"]), int(item["estimated_minutes"]), str(item["title"])),
            ))

    def test_cold_start_omits_the_interview_item_without_a_target_role(self):
        with _Store() as stores:
            stores.planner.upsert_profile(
                user_id="alice",
                payload={"primary_track": "frontend", "preferred_modalities": ["voice"]},
            )
            recs = stores.planner.build_recommendations(user_id="alice", limit=20)
            ids = [item["recommendation_id"] for item in recs["items"]]
            self.assertNotIn("interview:cold-start", ids)
            self.assertTrue(all(i.startswith("study:") for i in ids))
            self.assertIsNone(recs["days_until_target"])

    def test_quiz_modality_replaces_study_for_a_strong_topic(self):
        with _Store() as stores:
            stores.planner.upsert_profile(
                user_id="alice",
                payload={
                    "primary_track": "backend",
                    "preferred_modalities": ["quiz"],
                    "target_role": "",
                },
            )
            stores.learning.record_attempt(
                user_id="alice",
                question_id="q1",
                topic_id="01-backend",
                user_answer="right",
                is_correct=True,
                confidence=5,
                response_time_ms=400,
                mode="quiz",
            )
            recs = stores.planner.build_recommendations(user_id="alice", limit=20)
            types = {item["recommendation_type"] for item in recs["items"]}
            self.assertIn("quiz", types)
            self.assertNotIn("study", types)

    def test_focus_topic_ids_and_a_near_deadline_add_priority(self):
        with _Store() as stores:
            stores.planner.upsert_profile(
                user_id="alice",
                payload={
                    "primary_track": "backend",
                    "focus_topic_ids": ["01-backend-fundamentals-and-http"],
                    "target_date": _iso_in(7),
                    "preferred_modalities": ["study", "quiz", "interview"],
                    "target_role": "Backend Engineer",
                },
            )
            stores.learning.record_attempt(
                user_id="alice",
                question_id="q1",
                topic_id="01-backend-fundamentals-and-http",
                user_answer="wrong",
                is_correct=False,
                confidence=5,
                response_time_ms=900,
                mode="study",
            )
            recs = stores.planner.build_recommendations(user_id="alice", limit=20)
            self.assertEqual(recs["days_until_target"], 7)
            review = next(i for i in recs["items"] if i["recommendation_type"] == "review")
            self.assertEqual(review["reason_codes"], ["due_now", "mastery_gap"])
            self.assertGreaterEqual(review["priority"], 100.0)
            interview = next(i for i in recs["items"] if i["recommendation_type"] == "interview")
            self.assertIn("deadline_pressure", interview["reason_codes"])
            # Inside 14 days the interview item gets the larger urgency bonus.
            self.assertEqual(interview["priority"], 82.0)

    def test_deadline_soon_but_not_urgent_uses_the_lower_bonus(self):
        with _Store() as stores:
            stores.planner.upsert_profile(
                user_id="alice",
                payload={
                    "primary_track": "backend",
                    "target_date": _iso_in(21),
                    "preferred_modalities": ["interview"],
                },
            )
            recs = stores.planner.build_recommendations(user_id="alice", limit=20)
            interview = recs["items"][0]
            self.assertEqual(interview["reason_codes"], ["interview_practice", "deadline_pressure"])
            self.assertEqual(interview["priority"], 77.0)
            self.assertEqual(interview["title"], "Run a focused mock interview")

    def test_interview_fallback_names_the_weakest_rubric_dimension(self):
        with _Store() as stores:
            stores.planner.upsert_profile(
                user_id="alice",
                payload={
                    "primary_track": "backend",
                    "target_role": "Backend Engineer",
                    "preferred_modalities": ["study", "interview"],
                },
            )
            stores.learning._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interview_reports (
                    user_id TEXT NOT NULL, report_json TEXT NOT NULL, updated_at TEXT NOT NULL
                )
                """
            )
            stores.learning._conn.execute(
                "INSERT INTO interview_reports(user_id, report_json, updated_at) VALUES (?, ?, ?)",
                (
                    "alice",
                    '{"rubric_averages": {"reasoning_depth": 1, "technical_accuracy": 5}}',
                    datetime.now(UTC).isoformat(),
                ),
            )
            stores.learning._conn.commit()
            recs = stores.planner.build_recommendations(user_id="alice", limit=20)
            interview = next(i for i in recs["items"] if i["recommendation_type"] == "interview")
            # The report scores every dimension, so the lowest of the five wins.
            weakest = min(
                (
                    item
                    for item in stores.planner.run_diagnostic(user_id="alice")["competencies"]
                    if item["competency_type"] == "interview_dimension"
                ),
                key=lambda item: float(item["score"]),
            )["label"]
            self.assertIn(weakest, interview["reason"])
            self.assertEqual(interview["priority"], 72.0)
            self.assertIn("baseline", interview["reason_codes"])

    def test_interview_is_skipped_when_modality_is_not_selected_and_evidence_exists(self):
        with _Store() as stores:
            stores.planner.upsert_profile(
                user_id="alice",
                payload={
                    "primary_track": "backend",
                    "target_role": "Backend Engineer",
                    "preferred_modalities": ["study"],
                },
            )
            stores.learning._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interview_reports (
                    user_id TEXT NOT NULL, report_json TEXT NOT NULL, updated_at TEXT NOT NULL
                )
                """
            )
            stores.learning._conn.execute(
                "INSERT INTO interview_reports(user_id, report_json, updated_at) VALUES (?, ?, ?)",
                (
                    "alice",
                    '{"rubric_averages": {"reasoning_depth": 2}}',
                    datetime.now(UTC).isoformat(),
                ),
            )
            stores.learning._conn.commit()
            stores.learning.record_attempt(
                user_id="alice",
                question_id="q1",
                topic_id="01-backend",
                user_answer="wrong",
                is_correct=False,
                confidence=1,
                response_time_ms=900,
                mode="study",
            )
            recs = stores.planner.build_recommendations(user_id="alice", limit=20)
            self.assertNotIn("interview", {i["recommendation_type"] for i in recs["items"]})

    def test_recommendations_are_clamped_by_the_limit(self):
        with _Store() as stores:
            stores.planner.upsert_profile(
                user_id="alice",
                payload={"primary_track": "backend", "preferred_modalities": ["study"]},
            )
            self.assertLessEqual(
                len(stores.planner.build_recommendations(user_id="alice", limit=0)["items"]), 1
            )
            self.assertLessEqual(
                len(stores.planner.build_recommendations(user_id="alice", limit=500)["items"]), 50
            )

    def test_topic_entry_without_a_topic_id_is_skipped(self):
        with _Store() as stores:
            stores.planner.upsert_profile(
                user_id="alice", payload={"primary_track": "backend", "preferred_modalities": ["study"]}
            )
            stores.planner._replace_competencies(
                user_id="alice",
                competencies=[
                    {
                        "competency_id": "topic:orphan",
                        "competency_type": "topic",
                        "label": "Orphan",
                        "track": "backend",
                        "score": 10.0,
                        "metadata": {},
                    }
                ],
                updated_at=datetime.now(UTC).isoformat(),
            )
            recs = stores.planner.build_recommendations(user_id="alice", limit=20)
            self.assertNotIn("topic:orphan", [i["recommendation_id"] for i in recs["items"]])

    def test_row_to_profile_of_none_is_the_default(self):
        with _Store() as stores:
            default = stores.planner._row_to_profile(None)
            self.assertEqual(default["primary_track"], "backend")
            self.assertEqual(default["weekly_minutes"], 240)
            self.assertEqual(default["confidence_by_track"], lp._DEFAULT_CONFIDENCE)
            self.assertEqual(default["diagnostic_updated_at"], "")
            self.assertTrue(default["created_at"])

    def test_question_progress_rows_are_empty_without_the_table(self):
        with _Store() as stores:
            stores.planner._conn.execute("DROP TABLE question_progress")
            stores.planner._conn.commit()
            self.assertEqual(stores.planner._question_progress_rows(user_id="alice"), [])

    def test_duplicate_topic_entries_collapse_to_one_recommendation(self):
        with _Store() as stores:
            stores.planner.upsert_profile(
                user_id="alice", payload={"primary_track": "backend", "preferred_modalities": ["study"]}
            )
            entry = {
                "competency_id": "topic:01-backend",
                "competency_type": "topic",
                "label": "Backend",
                "track": "backend",
                "score": 20.0,
                "confidence_gap": 0.0,
                "evidence_count": 3,
                "priority": 80.0,
                "source": "adaptive_learning",
                "metadata": {"topic_id": "01-backend", "due_count": 2, "accuracy": 0.1},
                "updated_at": "",
            }
            clone = dict(entry, competency_id="topic:01-backend-dup")
            stores.planner.run_diagnostic = lambda **kwargs: {
                "generated_at": "",
                "profile": stores.planner.get_profile(user_id="alice"),
                "readiness_score": 0.0,
                "strongest_competencies": [],
                "urgent_competencies": [],
                "competencies": [entry, clone],
            }
            recs = stores.planner.build_recommendations(user_id="alice", limit=20)
            ids = [item["recommendation_id"] for item in recs["items"]]
            # Both entries describe the same topic, so _push drops the second.
            self.assertEqual(ids.count("review:01-backend"), 1)
            self.assertEqual(ids.count("study:01-backend"), 1)

    def test_topic_entry_without_a_topic_id_is_skipped_by_recommendations(self):
        with _Store() as stores:
            stores.planner.upsert_profile(
                user_id="alice", payload={"primary_track": "backend", "preferred_modalities": ["study"]}
            )
            orphan = {
                "competency_id": "topic:orphan",
                "competency_type": "topic",
                "label": "Orphan",
                "track": "backend",
                "score": 10.0,
                "confidence_gap": 0.0,
                "evidence_count": 1,
                "priority": 99.0,
                "source": "adaptive_learning",
                "metadata": {},
                "updated_at": "",
            }
            stores.planner.run_diagnostic = lambda **kwargs: {
                "generated_at": "",
                "profile": stores.planner.get_profile(user_id="alice"),
                "readiness_score": 0.0,
                "strongest_competencies": [],
                "urgent_competencies": [],
                "competencies": [orphan],
            }
            recs = stores.planner.build_recommendations(user_id="alice", limit=20)
            # No topic_id means nothing is emitted for it, so cold start fills in.
            self.assertTrue(all(item["recommendation_type"] == "study" for item in recs["items"]))
            self.assertTrue(all("Orphan" not in item["title"] for item in recs["items"]))

    def test_quiz_branch_continues_to_the_next_topic(self):
        with _Store() as stores:
            stores.planner.upsert_profile(
                user_id="alice",
                payload={"primary_track": "backend", "preferred_modalities": ["quiz"]},
            )
            for question_id, topic_id, correct in (
                ("q1", "01-backend", True),
                ("q2", "06-frontend", True),
            ):
                stores.learning.record_attempt(
                    user_id="alice",
                    question_id=question_id,
                    topic_id=topic_id,
                    user_answer="right",
                    is_correct=correct,
                    confidence=5,
                    response_time_ms=300,
                    mode="quiz",
                )
            recs = stores.planner.build_recommendations(user_id="alice", limit=20)
            quiz_topics = {
                item["topic_id"] for item in recs["items"] if item["recommendation_type"] == "quiz"
            }
            # Both strong topics get a quiz item; the second is only reachable
            # because the loop keeps going after the first elif.
            self.assertEqual(quiz_topics, {"01-backend", "06-frontend"})

    def test_duplicate_recommendation_ids_are_emitted_once(self):
        with _Store() as stores:
            stores.planner.upsert_profile(
                user_id="alice",
                payload={
                    "primary_track": "backend",
                    "target_role": "Backend Engineer",
                    "preferred_modalities": ["study", "quiz", "interview"],
                },
            )
            stores.learning.record_attempt(
                user_id="alice",
                question_id="q1",
                topic_id="01-backend",
                user_answer="wrong",
                is_correct=False,
                confidence=5,
                response_time_ms=900,
                mode="study",
            )
            ids = [
                item["recommendation_id"]
                for item in stores.planner.build_recommendations(user_id="alice", limit=20)["items"]
            ]
            self.assertEqual(len(ids), len(set(ids)))


class StudyPlanTests(unittest.TestCase):
    def test_plan_reuses_recommendations_across_days(self):
        with _Store() as stores:
            stores.planner.upsert_profile(
                user_id="alice",
                payload={
                    "primary_track": "ai_stack",
                    "target_role": "AI Engineer",
                    "preferred_modalities": ["study"],
                },
            )
            plan = stores.planner.build_study_plan(user_id="alice", days=4, daily_items=2)
            self.assertEqual(plan["days"], 4)
            self.assertEqual(plan["daily_items"], 2)
            self.assertEqual(len(plan["days_plan"]), 4)
            self.assertEqual(plan["days_plan"][0]["label"], "Today")
            self.assertEqual(plan["days_plan"][1]["label"], "Day 2")
            self.assertEqual(
                [day["date"] for day in plan["days_plan"]],
                [
                    (datetime.now(UTC) + timedelta(days=i)).date().isoformat()
                    for i in range(4)
                ],
            )
            self.assertEqual(plan["total_tasks"], sum(len(d["tasks"]) for d in plan["days_plan"]))
            self.assertEqual(plan["total_tasks"], 8)
            for task in plan["days_plan"][0]["tasks"]:
                self.assertIn(task["task_type"], {"topic_study", "interview", "review", "quiz"})
                self.assertTrue(task["cta_route"])

    def test_plan_clamps_days_and_daily_items(self):
        with _Store() as stores:
            stores.planner.upsert_profile(
                user_id="alice", payload={"primary_track": "backend", "preferred_modalities": ["study"]}
            )
            self.assertEqual(
                stores.planner.build_study_plan(user_id="alice", days=999, daily_items=999)["days"], 31
            )
            self.assertEqual(
                stores.planner.build_study_plan(user_id="alice", days=0, daily_items=0)["days"], 1
            )
            plan = stores.planner.build_study_plan(user_id="alice", days=31, daily_items=99)
            self.assertEqual(plan["daily_items"], 10)

    def test_empty_plan_when_the_store_yields_no_recommendations(self):
        with _Store() as stores:
            # Force the degenerate path: build_recommendations hands back no items.
            stores.planner.build_recommendations = lambda **kwargs: {"items": []}
            plan = stores.planner.build_study_plan(user_id="alice", days=2, daily_items=3)
            self.assertEqual(plan["total_tasks"], 0)
            self.assertEqual(plan["days"], 2)
            self.assertEqual([day["tasks"] for day in plan["days_plan"]], [[], []])
            self.assertEqual(plan["days_plan"][0]["label"], "Today")

    def test_a_single_rec_is_reused_when_it_has_no_topic_id(self):
        with _Store() as stores:
            stores.planner.build_recommendations = lambda **kwargs: {
                "items": [
                    {
                        "recommendation_id": "interview:target-role",
                        "recommendation_type": "interview",
                        "topic_id": "",
                        "title": "Run a mock interview",
                        "reason": "benchmark",
                        "estimated_minutes": 25,
                        "cta_route": "/interview",
                        "priority": 72.0,
                        "reason_codes": ["baseline"],
                        "track": "backend",
                    }
                ]
            }
            plan = stores.planner.build_study_plan(user_id="alice", days=2, daily_items=3)
            # No topic_id means the dedupe guard never trips, so it fills the day.
            self.assertEqual(plan["total_tasks"], 6)
            self.assertEqual(
                [t["task_type"] for t in plan["days_plan"][0]["tasks"]],
                ["interview", "interview", "interview"],
            )

    def test_study_tasks_are_relabelled_topic_study(self):
        with _Store() as stores:
            stores.planner.upsert_profile(
                user_id="alice", payload={"primary_track": "backend", "preferred_modalities": ["study"]}
            )
            plan = stores.planner.build_study_plan(user_id="alice", days=1, daily_items=3)
            self.assertTrue(all(t["task_type"] == "topic_study" for t in plan["days_plan"][0]["tasks"]))


class CompetencyReplacementTests(unittest.TestCase):
    def test_replace_competencies_is_idempotent_and_drops_old_rows(self):
        with _Store() as stores:
            stores.planner.get_profile(user_id="alice")
            now = datetime.now(UTC).isoformat()
            stores.planner._replace_competencies(
                user_id="alice",
                competencies=[
                    {
                        "competency_id": "track:backend",
                        "competency_type": "track",
                        "label": "Backend",
                        "track": "backend",
                        "score": 40,
                        "priority": 90,
                    }
                ],
                updated_at=now,
            )
            self.assertEqual(len(stores.planner.list_competencies(user_id="alice")), 1)

            stores.planner._replace_competencies(
                user_id="alice",
                competencies=[
                    {
                        "competency_id": "track:frontend",
                        "competency_type": "track",
                        "label": "Frontend",
                        "track": "frontend",
                        "score": 70,
                        "priority": 10,
                        "metadata": {"note": "x"},
                        "source": "manual",
                        "confidence_gap": 1.5,
                        "evidence_count": 4,
                    }
                ],
                updated_at=now,
            )
            rows = stores.planner.list_competencies(user_id="alice")
            self.assertEqual([r["competency_id"] for r in rows], ["track:frontend"])
            self.assertEqual(rows[0]["metadata"], {"note": "x"})
            self.assertEqual(rows[0]["source"], "manual")
            self.assertEqual(rows[0]["confidence_gap"], 1.5)
            self.assertEqual(rows[0]["evidence_count"], 4)

    def test_a_failing_replacement_rolls_back_the_whole_batch(self):
        with _Store() as stores:
            stores.planner.get_profile(user_id="alice")
            now = datetime.now(UTC).isoformat()
            stores.planner._replace_competencies(
                user_id="alice",
                competencies=[
                    {
                        "competency_id": "track:backend",
                        "competency_type": "track",
                        "label": "Backend",
                        "score": 40,
                        "priority": 90,
                    }
                ],
                updated_at=now,
            )
            # The second row is missing "score", so the INSERT raises mid-batch.
            with self.assertRaises(KeyError):
                stores.planner._replace_competencies(
                    user_id="alice",
                    competencies=[
                        {
                            "competency_id": "track:frontend",
                            "competency_type": "track",
                            "label": "Frontend",
                            "priority": 50,
                        },
                        {
                            "competency_id": "track:ai_stack",
                            "competency_type": "track",
                            "label": "AI",
                            "priority": 30,
                        },
                    ],
                    updated_at=now,
                )
            rows = stores.planner.list_competencies(user_id="alice")
            self.assertEqual([r["competency_id"] for r in rows], ["track:backend"])
            # The profile timestamp set by the successful call is untouched.
            self.assertEqual(
                stores.planner.get_profile(user_id="alice")["diagnostic_updated_at"], now
            )


class ConnectionHygieneTests(unittest.TestCase):
    def test_store_uses_a_row_factory_and_a_single_writer_executor(self):
        with _Store() as stores:
            self.assertIsInstance(stores.planner._conn, sqlite3.Connection)
            self.assertIs(stores.planner._conn.row_factory, sqlite3.Row)
            self.assertEqual(stores.planner._executor._max_workers, 1)

    def test_get_profile_seeds_the_row_so_a_second_read_is_a_hit(self):
        with _Store() as stores:
            self.assertEqual(
                stores.planner._conn.execute(
                    "SELECT COUNT(*) FROM learner_profiles WHERE user_id = ?", ("alice",)
                ).fetchone()[0],
                0,
            )
            stores.planner.get_profile(user_id="alice")
            self.assertEqual(
                stores.planner._conn.execute(
                    "SELECT COUNT(*) FROM learner_profiles WHERE user_id = ?", ("alice",)
                ).fetchone()[0],
                1,
            )


if __name__ == "__main__":
    unittest.main()
