"""Coverage for app/services/learning_store.py.

Targets the branches the rest of the suite leaves open: the repeat-attempt
mastery update, the topic-mastery rollup, the malformed-JSON guards in the
custom-topic / curriculum / assistant-memory / video-cache readers, the
response-detail and language-option normalisers, the ``requires_programming``
inference, and the ``get_feature_metrics`` window arithmetic.

The metrics tests write rows straight into the store's own tables, because the
corrupt-timestamp and malformed-metadata paths they target cannot be reached
through the public writers.
"""

import json
import os
import tempfile
import unittest
from datetime import UTC, datetime, timedelta

from app.services.learning_store import LearningStore
from app.services.topic_catalog import PROBLEM_SOLVING_LANGUAGE_OPTIONS


class _Store:
    def __enter__(self):
        self._td = tempfile.TemporaryDirectory()
        self.store = LearningStore(os.path.join(self._td.name, "learning.db"))
        return self.store

    def __exit__(self, *exc_info):
        self._td.cleanup()
        return False


def _iso_in(**kwargs) -> str:
    return (datetime.now(UTC) + timedelta(**kwargs)).isoformat()


def _backdate_events(store, *, when_iso, event_name=None):
    """Rewrite feature_events timestamps so the metrics window sees the past."""

    if event_name is None:
        store._conn.execute("UPDATE feature_events SET created_at = ?", (when_iso,))
    else:
        store._conn.execute(
            "UPDATE feature_events SET created_at = ? WHERE event_name = ?",
            (when_iso, event_name),
        )
    store._conn.commit()


def _attempt(store, *, user_id="alice", question_id="q1", topic_id="01-backend", **kwargs):
    payload = {
        "user_id": user_id,
        "question_id": question_id,
        "topic_id": topic_id,
        "user_answer": "answer",
        "is_correct": False,
        "confidence": 3,
        "response_time_ms": 900,
        "mode": "quiz",
    }
    payload.update(kwargs)
    return store.record_attempt(**payload)


class RunAsyncTests(unittest.TestCase):
    def test_positional_arguments_take_the_direct_executor_path(self):
        import asyncio

        with _Store() as store:
            async def _exercise():
                # get_review_queue's parameters are keyword-only, so the
                # positional path is exercised through a small wrapper.
                queued = await store.run_async(
                    lambda user_id: store.get_review_queue(user_id=user_id, limit=5),
                    "alice",
                )
                return queued

            queued = asyncio.run(_exercise())
            self.assertEqual(queued["total_due"], 0)
            self.assertEqual(queued["items"], [])

    def test_kwargs_take_the_lambda_executor_path(self):
        import asyncio

        with _Store() as store:
            queued = asyncio.run(
                store.run_async(store.get_review_queue, user_id="alice", limit=5)
            )
            self.assertEqual(queued["items"], [])


class RepeatAttemptTests(unittest.TestCase):
    def test_second_attempt_updates_mastery_instead_of_inserting_a_row(self):
        with _Store() as store:
            first = _attempt(store, is_correct=False, confidence=5)
            second = _attempt(store, is_correct=True, confidence=4)

            self.assertEqual(store._conn.execute("SELECT COUNT(*) FROM learning_attempts").fetchone()[0], 2)
            progress = store._conn.execute(
                "SELECT mastery_score, attempts, correct_attempts, avg_confidence, review_bucket, due_at"
                " FROM question_progress WHERE question_id = 'q1'"
            ).fetchone()
            self.assertEqual(progress["attempts"], 2)
            self.assertEqual(progress["correct_attempts"], 1)
            # A confidence-5 miss is the strongest negative signal, so the
            # corrective answer must raise mastery above the first attempt.
            self.assertGreater(progress["mastery_score"], first["mastery_score"])
            self.assertAlmostEqual(
                progress["mastery_score"], second["mastery_score"], places=9
            )
            # The running average is (5 + 4) / 2.
            self.assertAlmostEqual(progress["avg_confidence"], 4.5, places=6)
            self.assertEqual(progress["review_bucket"], second["review_bucket"])
            self.assertEqual(progress["due_at"], second["due_at"])

    def test_a_wrong_repeat_answer_lowers_mastery_and_demotes_the_bucket(self):
        with _Store() as store:
            first = _attempt(store, is_correct=True, confidence=5)
            second = _attempt(store, is_correct=False, confidence=1)
            progress = store._conn.execute(
                "SELECT mastery_score, review_bucket FROM question_progress WHERE question_id = 'q1'"
            ).fetchone()
            self.assertLess(progress["mastery_score"], first["mastery_score"])
            self.assertLess(progress["review_bucket"], second["review_bucket"] + 1)

    def test_mastery_stays_inside_the_unit_interval_over_many_attempts(self):
        with _Store() as store:
            for index in range(30):
                _attempt(store, is_correct=index % 2 == 0, confidence=5, question_id=f"q{index}")
            row = store._conn.execute(
                "SELECT MIN(mastery_score), MAX(mastery_score) FROM question_progress"
            ).fetchone()
            self.assertGreaterEqual(row[0], 0.0)
            self.assertLessEqual(row[1], 1.0)


class TopicMasteryTests(unittest.TestCase):
    def test_topic_mastery_rolls_up_by_topic(self):
        with _Store() as store:
            _attempt(store, question_id="q1", topic_id="01-backend", is_correct=True, confidence=5)
            _attempt(store, question_id="q2", topic_id="01-backend", is_correct=False, confidence=1)
            _attempt(store, question_id="q3", topic_id="06-frontend", is_correct=True, confidence=4)

            out = store.get_topic_mastery(user_id="alice", limit=10)
            topics = {t["topic_id"]: t for t in out["topics"]}
            self.assertEqual(sorted(topics), ["01-backend", "06-frontend"])
            self.assertEqual(topics["01-backend"]["attempts"], 2)
            self.assertEqual(topics["06-frontend"]["attempts"], 1)
            for topic in out["topics"]:
                self.assertGreaterEqual(topic["mastery_score"], 0.0)
                self.assertLessEqual(topic["mastery_score"], 1.0)

    def test_topic_mastery_is_empty_for_an_unknown_user(self):
        with _Store() as store:
            self.assertEqual(store.get_topic_mastery(user_id="nobody"), {"topics": []})


class CustomTopicReaderTests(unittest.TestCase):
    def _seed(self, store, *, user_id="alice", topic_id="custom-1", levels=None, sections=None):
        return store.upsert_custom_topic(
            user_id=user_id,
            topic_id=topic_id,
            source_topic="",
            title="Caching Deep Dive",
            description="Redis and HTTP caching.",
            raw_content="# Caching Deep Dive\n\nRedis and HTTP caching.",
            track="backend",
            levels=levels if levels is not None else ["junior", "mid"],
            sections=(
                sections
                if sections is not None
                else [{"heading": "Intro", "content": "body"}, {"heading": "Deep", "content": "body"}]
            ),
        )

    def test_round_trip_and_filters(self):
        with _Store() as store:
            self._seed(store, topic_id="custom-1")
            self._seed(store, topic_id="custom-2", levels=["senior"])
            store._conn.execute(
                "UPDATE custom_topics SET description = ? WHERE topic_id = ?",
                ("", "custom-2"),
            )
            store._conn.commit()

            detail = store.get_custom_topic(user_id="alice", topic_id="custom-1")
            self.assertEqual(detail["title"], "Caching Deep Dive")
            self.assertEqual(len(detail["sections"]), 2)
            self.assertEqual(detail["source_topic"], "")
            self.assertIsNone(store.get_custom_topic(user_id="alice", topic_id="missing"))
            self.assertIsNone(store.get_custom_topic(user_id="bob", topic_id="custom-1"))

            everything = store.list_custom_topics(user_id="alice")
            self.assertEqual(len(everything), 2)
            self.assertEqual(everything[0]["section_count"], 2)
            self.assertEqual(everything[0]["estimated_questions"], 4)

            self.assertEqual(
                [t["id"] for t in store.list_custom_topics(user_id="alice", level="senior")],
                ["custom-2"],
            )
            self.assertEqual(
                [t["id"] for t in store.list_custom_topics(user_id="alice", track="ai_stack")], []
            )
            self.assertEqual(
                [t["id"] for t in store.list_custom_topics(user_id="alice", q="redis")], ["custom-1"]
            )
            self.assertEqual(store.list_custom_topics(user_id="bob"), [])

    def test_malformed_level_and_section_json_is_skipped_not_fatal(self):
        with _Store() as store:
            self._seed(store)
            store._conn.execute(
                "UPDATE custom_topics SET levels_json = ?, sections_json = ? WHERE topic_id = ?",
                (
                    "{not json",
                    json.dumps(["junk", {"content": "no heading"}, {"heading": "   ", "content": "x"}, {"heading": "Ok", "content": "y"}]),
                    "custom-1",
                ),
            )
            store._conn.commit()

            detail = store.get_custom_topic(user_id="alice", topic_id="custom-1")
            self.assertEqual(detail["levels"], [])
            self.assertEqual([s["heading"] for s in detail["sections"]], ["Ok"])
            # section_count is derived, so the summary reflects the repair.
            summary = store.list_custom_topics(user_id="alice")[0]
            self.assertEqual(summary["section_count"], 1)
            self.assertEqual(summary["levels"], [])

    def test_non_list_json_columns_collapse_to_empty(self):
        with _Store() as store:
            self._seed(store)
            store._conn.execute(
                "UPDATE custom_topics SET levels_json = ?, sections_json = ? WHERE topic_id = ?",
                ('{"a": 1}', '"a string"', "custom-1"),
            )
            store._conn.commit()
            detail = store.get_custom_topic(user_id="alice", topic_id="custom-1")
            self.assertEqual(detail["levels"], [])
            self.assertEqual(detail["sections"], [])


class DynamicCurriculumReaderTests(unittest.TestCase):
    def _seed(self, store, *, language="python", topic_id="00-problem-solving-and-algorithms"):
        return store.upsert_dynamic_topic_curriculum(
            user_id="alice",
            topic_id=topic_id,
            preferred_language=language,
            title="Problem Solving and Algorithms (python)",
            description="Roadmap.",
            raw_content="# Problem Solving\n\nRoadmap.",
            track="backend",
            levels=["junior", "mid", "senior"],
            sections=[
                {"heading": "Junior: Arrays", "content": "x"},
                {"heading": "Mid: Graphs", "content": "y"},
            ],
            target_sections=120,
            source="generator",
        )

    def test_round_trip_and_language_normalisation(self):
        with _Store() as store:
            self._seed(store, language=" PYTHON ")
            cached = store.get_dynamic_topic_curriculum(
                user_id="alice",
                topic_id="00-problem-solving-and-algorithms",
                preferred_language="Python",
            )
            self.assertIsNotNone(cached)
            self.assertEqual(cached["preferred_language"], "python")
            self.assertEqual(len(cached["sections"]), 2)
            self.assertEqual(cached["target_sections"], 120)
            self.assertEqual(cached["source"], "generator")
            self.assertIsNone(
                store.get_dynamic_topic_curriculum(
                    user_id="alice",
                    topic_id="00-problem-solving-and-algorithms",
                    preferred_language="java",
                )
            )

    def test_malformed_section_json_is_skipped(self):
        with _Store() as store:
            self._seed(store)
            store._conn.execute(
                "UPDATE dynamic_topic_curricula SET sections_json = ?, levels_json = ?",
                (
                    json.dumps(["junk", {"content": "no heading"}, {"heading": " Ok "}]),
                    "{broken",
                ),
            )
            store._conn.commit()
            cached = store.get_dynamic_topic_curriculum(
                user_id="alice",
                topic_id="00-problem-solving-and-algorithms",
                preferred_language="python",
            )
            self.assertEqual([s["heading"] for s in cached["sections"]], ["Ok"])
            self.assertEqual(cached["levels"], [])

    def test_delete_only_removes_the_named_language(self):
        with _Store() as store:
            self._seed(store, language="python")
            self._seed(store, language="java")
            store.delete_dynamic_topic_curriculum(
                user_id="alice",
                topic_id="00-problem-solving-and-algorithms",
                preferred_language="PYTHON",
            )
            self.assertIsNone(
                store.get_dynamic_topic_curriculum(
                    user_id="alice",
                    topic_id="00-problem-solving-and-algorithms",
                    preferred_language="python",
                )
            )
            self.assertIsNotNone(
                store.get_dynamic_topic_curriculum(
                    user_id="alice",
                    topic_id="00-problem-solving-and-algorithms",
                    preferred_language="java",
                )
            )
            # Deleting again is a no-op rather than an error.
            store.delete_dynamic_topic_curriculum(
                user_id="alice",
                topic_id="00-problem-solving-and-algorithms",
                preferred_language="python",
            )


class ProblemSolvingResolveTests(unittest.TestCase):
    def test_uncached_resolution_reports_content_not_ready(self):
        with _Store() as store:
            out = store.resolve_problem_solving_topic_detail(
                user_id="alice", preferred_language="  JAVA  "
            )
            self.assertEqual(out["selected_language"], "java")
            self.assertFalse(out["content_ready"])
            self.assertEqual(out["sections"], [])
            self.assertEqual(out["raw_content"], "")
            self.assertIn("120-section", out["description"])

    def test_unsupported_language_is_replaced_by_the_default(self):
        with _Store() as store:
            out = store.resolve_problem_solving_topic_detail(
                user_id="alice", preferred_language="cobol"
            )
            self.assertEqual(out["selected_language"], "python")

    def test_cached_curriculum_is_returned_with_concise_response_detail(self):
        with _Store() as store:
            store.upsert_dynamic_topic_curriculum(
                user_id="alice",
                topic_id="00-problem-solving-and-algorithms",
                preferred_language="java",
                title="PS (java)",
                description="Roadmap.",
                raw_content="# Problem Solving\n\nRoadmap.",
                track="backend",
                levels=["junior"],
                sections=[{"heading": "Junior: Arrays", "content": "x"}],
                target_sections=120,
                source="generator",
            )
            out = store.resolve_problem_solving_topic_detail(
                user_id="alice", preferred_language="java"
            )
            self.assertTrue(out["content_ready"])
            self.assertEqual(out["title"], "PS (java)")
            self.assertEqual(out["response_detail"], "very_detailed")
            self.assertEqual(len(out["sections"]), 1)
            self.assertEqual(out["levels"], ["junior"])


class NormaliserTests(unittest.TestCase):
    def test_response_detail_only_admits_very_detailed(self):
        self.assertEqual(LearningStore._normalise_response_detail("VERY_DETAILED"), "very_detailed")
        self.assertEqual(LearningStore._normalise_response_detail(" concise "), "concise")
        self.assertEqual(LearningStore._normalise_response_detail(None), "concise")
        self.assertEqual(LearningStore._normalise_response_detail(""), "concise")

    def test_language_is_trimmed_lowercased_and_capped(self):
        self.assertEqual(LearningStore._normalise_language("  Go  "), "go")
        self.assertEqual(LearningStore._normalise_language(None), "")
        self.assertEqual(len(LearningStore._normalise_language("x" * 200)), 60)

    def test_language_options_dedupe_skip_blanks_and_cap_at_eight(self):
        options = LearningStore._normalise_language_options(
            ["Go", " go ", "", "RUST", "Java", "C", "C++", "Zig", "Elixir", "Swift", "Kotlin"]
        )
        self.assertEqual(options, ["go", "rust", "java", "c", "c++", "zig", "elixir", "swift"])
        self.assertEqual(LearningStore._normalise_language_options(None), [])
        self.assertEqual(LearningStore._normalise_language_options([None, "  "]), [])


class RequiresProgrammingInferenceTests(unittest.TestCase):
    def _infer(self, detail):
        return LearningStore._infer_requires_programming_from_topic(detail)

    def test_non_dict_topic_is_never_programming(self):
        self.assertFalse(self._infer(None))
        self.assertFalse(self._infer("not-a-dict"))
        self.assertFalse(self._infer(42))

    def test_backend_and_frontend_tracks_always_require_programming(self):
        self.assertTrue(self._infer({"track": "backend", "title": "Anything"}))
        self.assertTrue(self._infer({"track": "Frontend", "title": "Anything"}))

    def test_ai_stack_needs_a_matching_keyword(self):
        self.assertTrue(self._infer({"track": "ai_stack", "title": "Building Agents"}))
        self.assertTrue(self._infer({"track": "ai_stack", "description": "RAG pipelines"}))
        self.assertTrue(self._infer({"track": "ai_stack", "id": "prompt-engineering"}))
        self.assertTrue(self._infer({"track": "ai_stack", "title": "Python for AI"}))
        self.assertTrue(self._infer({"track": "ai_stack", "title": "TypeScript SDK"}))
        self.assertTrue(self._infer({"track": "ai_stack", "title": "LLM app serving"}))
        self.assertFalse(self._infer({"track": "ai_stack", "title": "Model Governance"}))

    def test_system_design_needs_a_matching_keyword(self):
        self.assertTrue(self._infer({"track": "system_design", "title": "Algorithm Choices"}))
        self.assertTrue(self._infer({"track": "system_design", "title": "Coding Guidelines"}))
        self.assertTrue(self._infer({"track": "system_design", "description": "API versioning"}))
        self.assertTrue(self._infer({"track": "system_design", "title": "Database Sharding"}))
        self.assertTrue(self._infer({"track": "system_design", "title": "Java and Python clients"}))
        self.assertTrue(self._infer({"track": "system_design", "title": "Implementation notes"}))
        self.assertFalse(self._infer({"track": "system_design", "title": "Team Topologies"}))

    def test_unknown_track_falls_back_to_a_broad_keyword_scan(self):
        self.assertTrue(self._infer({"track": "devops", "title": "Writing code review rules"}))
        self.assertTrue(self._infer({"track": "devops", "title": "Programming standards"}))
        self.assertTrue(self._infer({"track": "devops", "title": "Implementation contracts"}))
        self.assertTrue(self._infer({"track": "devops", "title": "API versioning"}))
        self.assertTrue(self._infer({"track": "devops", "title": "Service ownership"}))
        self.assertTrue(self._infer({"track": "devops", "title": "Backend and frontend parity"}))
        self.assertFalse(self._infer({"track": "devops", "title": "Interview story telling"}))
        self.assertFalse(self._infer({"track": ""}))


class TopicAiSettingsTests(unittest.TestCase):
    def test_language_is_dropped_when_the_topic_needs_no_programming(self):
        with _Store() as store:
            store.upsert_topic_language_profile(
                topic_id="01-http",
                requires_programming=False,
                language_options=["go", "rust"],
                source="catalog",
            )
            store.upsert_topic_preferences(
                user_id="alice", topic_id="01-http", preferred_language="rust"
            )
            resolved = store.resolve_topic_ai_settings(
                user_id="alice", topic_id="01-http", topic_detail={"track": "backend"}
            )
            self.assertEqual(resolved["preferred_language"], "")
            self.assertFalse(resolved["requires_programming"])

    def test_language_outside_the_supported_options_falls_back_to_the_first(self):
        with _Store() as store:
            store.upsert_topic_language_profile(
                topic_id="01-http",
                requires_programming=True,
                language_options=["go", "rust"],
                source="catalog",
            )
            store.upsert_topic_preferences(
                user_id="alice", topic_id="01-http", preferred_language="cobol"
            )
            resolved = store.resolve_topic_ai_settings(
                user_id="alice", topic_id="01-http", topic_detail={"track": "backend"}
            )
            # "cobol" is discarded, then the first option is chosen instead.
            self.assertEqual(resolved["preferred_language"], "go")
            self.assertTrue(resolved["requires_programming"])

    def test_first_option_is_used_when_the_user_has_no_language(self):
        with _Store() as store:
            store.upsert_topic_language_profile(
                topic_id="01-http",
                requires_programming=True,
                language_options=["go", "rust"],
                source="catalog",
            )
            resolved = store.resolve_topic_ai_settings(
                user_id="alice", topic_id="01-http", topic_detail={"track": "backend"}
            )
            self.assertEqual(resolved["preferred_language"], "go")
            self.assertEqual(resolved["profile_source"], "catalog")

    def test_defaults_are_applied_without_any_profile_or_preferences(self):
        with _Store() as store:
            resolved = store.resolve_topic_ai_settings(
                user_id="alice", topic_id="01-http", topic_detail=None
            )
            self.assertEqual(resolved["preferred_language"], "")
            self.assertEqual(resolved["response_detail"], "very_detailed")
            self.assertEqual(resolved["language_options"], [])
            self.assertEqual(resolved["profile_source"], "")
            self.assertEqual(resolved["profile_updated_at"], "")
            self.assertFalse(resolved["requires_programming"])

    def test_problem_solving_topic_ignores_the_language_profile(self):
        with _Store() as store:
            store.upsert_topic_language_profile(
                topic_id="00-problem-solving-and-algorithms",
                requires_programming=False,
                language_options=["cobol"],
                source="catalog",
            )
            store.upsert_topic_preferences(
                user_id="alice",
                topic_id="00-problem-solving-and-algorithms",
                preferred_language="cobol",
            )
            resolved = store.resolve_topic_ai_settings(
                user_id="alice",
                topic_id="00-problem-solving-and-algorithms",
                topic_detail=None,
            )
            # The problem-solving topic is always programming and always offers
            # the full language list, whatever the profile says.
            self.assertTrue(resolved["requires_programming"])
            self.assertEqual(resolved["language_options"], list(PROBLEM_SOLVING_LANGUAGE_OPTIONS))
            # "cobol" is not an offered language, so the first option wins.
            self.assertNotIn("cobol", resolved["language_options"])
            self.assertEqual(resolved["preferred_language"], PROBLEM_SOLVING_LANGUAGE_OPTIONS[0])

    def test_topic_preferences_round_trip_and_preserve_created_at(self):
        with _Store() as store:
            self.assertIsNone(store.get_topic_preferences(user_id="alice", topic_id="01-http"))
            first = store.upsert_topic_preferences(
                user_id="alice",
                topic_id="01-http",
                response_detail="very_detailed",
                preferred_language=" Go ",
            )
            self.assertEqual(first["response_detail"], "very_detailed")
            self.assertEqual(first["preferred_language"], "go")

            # A partial update keeps the untouched column.
            second = store.upsert_topic_preferences(user_id="alice", topic_id="01-http")
            self.assertEqual(second["response_detail"], "very_detailed")
            self.assertEqual(second["preferred_language"], "go")
            self.assertEqual(second["created_at"], first["created_at"])

            third = store.upsert_topic_preferences(
                user_id="alice", topic_id="01-http", response_detail="nonsense"
            )
            self.assertEqual(third["response_detail"], "concise")
            self.assertEqual(store.get_topic_preferences(user_id="bob", topic_id="01-http"), None)

    def test_language_profile_round_trip_defaults_source(self):
        with _Store() as store:
            self.assertIsNone(store.get_topic_language_profile(topic_id="01-http"))
            saved = store.upsert_topic_language_profile(
                topic_id="01-http",
                requires_programming=1,
                language_options=["Go", "go", "RUST", "C", "C++", "Zig", "Elixir", "Swift", "Kotlin"],
                source="   ",
            )
            self.assertEqual(saved["source"], "unknown")
            self.assertEqual(len(saved["language_options"]), 8)
            loaded = store.get_topic_language_profile(topic_id="01-http")
            self.assertTrue(loaded["requires_programming"])
            self.assertEqual(loaded["language_options"], saved["language_options"])


class AssistantMemoryTests(unittest.TestCase):
    def test_round_trip(self):
        with _Store() as store:
            self.assertIsNone(store.get_assistant_memory(user_id="alice", conversation_id="c1"))
            store.upsert_assistant_memory(
                user_id="alice",
                conversation_id="c1",
                summary={"topics": ["http"], "turns": 3},
            )
            memory = store.get_assistant_memory(user_id="alice", conversation_id="c1")
            self.assertEqual(memory["summary"], {"topics": ["http"], "turns": 3})
            self.assertEqual(memory["flow"], "chat")

            # A second flow shares the conversation_id but not the row.
            store.upsert_assistant_memory(
                user_id="alice", conversation_id="c1", flow="topics", summary={"topics": ["grpc"]}
            )
            self.assertEqual(
                store.get_assistant_memory(user_id="alice", conversation_id="c1", flow="topics")[
                    "summary"
                ],
                {"topics": ["grpc"]},
            )
            self.assertEqual(
                store.get_assistant_memory(user_id="alice", conversation_id="c1")["summary"],
                {"topics": ["http"], "turns": 3},
            )

    def test_malformed_summary_json_degrades_to_an_empty_dict(self):
        with _Store() as store:
            store.upsert_assistant_memory(
                user_id="alice", conversation_id="c1", summary={"a": 1}
            )
            store._conn.execute(
                "UPDATE assistant_memory SET summary_json = ? WHERE conversation_id = ?",
                ("{not json", "c1"),
            )
            store._conn.commit()
            self.assertEqual(
                store.get_assistant_memory(user_id="alice", conversation_id="c1")["summary"], {}
            )

    def test_non_dict_summary_json_degrades_to_an_empty_dict(self):
        with _Store() as store:
            store.upsert_assistant_memory(
                user_id="alice", conversation_id="c1", summary={"a": 1}
            )
            store._conn.execute(
                "UPDATE assistant_memory SET summary_json = ? WHERE conversation_id = ?",
                ('["a", "list"]', "c1"),
            )
            store._conn.commit()
            self.assertEqual(
                store.get_assistant_memory(user_id="alice", conversation_id="c1")["summary"], {}
            )


class VideoCacheTests(unittest.TestCase):
    def _write(self, store, cache_key="topic_videos:abc"):
        return store.upsert_topic_video_cache(
            cache_key=cache_key,
            topic_id="01-http",
            section_heading="Basics",
            preferred_language="en",
            payload={"videos": [{"video_id": "v1"}], "source": "youtube"},
        )

    def test_miss_returns_none(self):
        with _Store() as store:
            self.assertIsNone(
                store.get_topic_video_cache(cache_key="nope", max_age_hours=168)
            )

    def test_hit_returns_the_payload_and_metadata(self):
        with _Store() as store:
            saved = self._write(store)
            hit = store.get_topic_video_cache(cache_key="topic_videos:abc", max_age_hours=168)
            self.assertEqual(hit["topic_id"], "01-http")
            self.assertEqual(hit["section_heading"], "Basics")
            self.assertEqual(hit["preferred_language"], "en")
            self.assertEqual(hit["payload"]["videos"][0]["video_id"], "v1")
            self.assertEqual(hit["fetched_at"], saved["fetched_at"])

    def test_non_positive_max_age_expires_the_entry_immediately(self):
        with _Store() as store:
            self._write(store)
            self.assertIsNone(store.get_topic_video_cache(cache_key="topic_videos:abc", max_age_hours=0))
            self.assertIsNone(store.get_topic_video_cache(cache_key="topic_videos:abc", max_age_hours=-5))

    def test_unparseable_fetched_at_is_treated_as_a_miss(self):
        with _Store() as store:
            self._write(store)
            store._conn.execute(
                "UPDATE topic_video_cache SET fetched_at = ? WHERE cache_key = ?",
                ("not-a-timestamp", "topic_videos:abc"),
            )
            store._conn.commit()
            self.assertIsNone(store.get_topic_video_cache(cache_key="topic_videos:abc", max_age_hours=168))

    def test_naive_fetched_at_is_assumed_utc(self):
        with _Store() as store:
            self._write(store)
            naive = datetime.now(UTC).replace(tzinfo=None).isoformat()
            store._conn.execute(
                "UPDATE topic_video_cache SET fetched_at = ? WHERE cache_key = ?",
                (naive, "topic_videos:abc"),
            )
            store._conn.commit()
            # Without the UTC assumption this would look like the future and
            # produce a negative age; it must still count as fresh.
            hit = store.get_topic_video_cache(cache_key="topic_videos:abc", max_age_hours=168)
            self.assertIsNotNone(hit)
            self.assertEqual(hit["fetched_at"], naive)

    def test_malformed_payload_json_is_treated_as_a_miss(self):
        with _Store() as store:
            self._write(store)
            store._conn.execute(
                "UPDATE topic_video_cache SET payload_json = ? WHERE cache_key = ?",
                ("{not json", "topic_videos:abc"),
            )
            store._conn.commit()
            self.assertIsNone(store.get_topic_video_cache(cache_key="topic_videos:abc", max_age_hours=168))

    def test_non_dict_payload_json_is_treated_as_a_miss(self):
        with _Store() as store:
            self._write(store)
            store._conn.execute(
                "UPDATE topic_video_cache SET payload_json = ? WHERE cache_key = ?",
                ("[1, 2, 3]", "topic_videos:abc"),
            )
            store._conn.commit()
            self.assertIsNone(store.get_topic_video_cache(cache_key="topic_videos:abc", max_age_hours=168))

    def test_a_non_dict_payload_is_normalised_to_an_empty_object(self):
        with _Store() as store:
            store.upsert_topic_video_cache(
                cache_key="topic_videos:list",
                topic_id="01-http",
                section_heading="Basics",
                preferred_language="",
                payload=["not", "a", "dict"],
            )
            hit = store.get_topic_video_cache(cache_key="topic_videos:list", max_age_hours=168)
            self.assertEqual(hit["payload"], {})


class FeatureStateTests(unittest.TestCase):
    def test_state_round_trip_and_defaults(self):
        with _Store() as store:
            self.assertIsNone(store.get_feature_state(feature_key="topic_videos"))
            saved = store.set_feature_state(feature_key="topic_videos", status="enabled")
            self.assertEqual(saved["status"], "enabled")
            self.assertEqual(saved["disabled_until"], "")
            self.assertEqual(saved["reason"], "")

            # A blank status falls back to "enabled" rather than being stored empty.
            blank = store.set_feature_state(feature_key="other", status="   ")
            self.assertEqual(blank["status"], "enabled")

            # Overwriting replaces rather than duplicating.
            store.set_feature_state(
                feature_key="topic_videos", status="disabled_quota_exhausted", reason="quota"
            )
            loaded = store.get_feature_state(feature_key="topic_videos")
            self.assertEqual(loaded["status"], "disabled_quota_exhausted")
            self.assertEqual(loaded["reason"], "quota")

    def test_feature_event_defaults_the_user_id(self):
        with _Store() as store:
            event = store.record_feature_event(
                user_id="  ", feature_key="topic_videos", event_name="video_panel_viewed"
            )
            self.assertEqual(event["user_id"], "anonymous")
            self.assertEqual(
                store._conn.execute(
                    "SELECT metadata_json FROM feature_events WHERE user_id = 'anonymous'"
                ).fetchone()[0],
                "{}",
            )

    def test_non_dict_event_metadata_is_normalised(self):
        with _Store() as store:
            store.record_feature_event(
                user_id="alice",
                feature_key="topic_videos",
                event_name="video_panel_viewed",
                metadata=["not", "a", "dict"],
            )
            self.assertEqual(
                store._conn.execute("SELECT metadata_json FROM feature_events").fetchone()[0], "{}"
            )


class FeatureMetricsTests(unittest.TestCase):
    def test_empty_state_reports_the_disabled_config_defaults(self):
        with _Store() as store:
            metrics = store.get_feature_metrics(feature_key="topic_videos")
            self.assertEqual(metrics["status"], "disabled_config")
            self.assertFalse(metrics["hidden_now"])
            self.assertEqual(metrics["hidden_days_last_7"], 0.0)
            self.assertEqual(metrics["hidden_days_last_30"], 0.0)
            self.assertEqual(metrics["hidden_frequency_weekly"], [])
            self.assertEqual(metrics["ctr_30d"], 0.0)
            self.assertEqual(metrics["reenabled_at"], "")
            self.assertEqual(metrics["panel_views_since_reenable"], 0)
            self.assertEqual(metrics["clicks_since_reenable"], 0)
            self.assertEqual(metrics["ctr_since_reenable"], 0.0)

    def test_views_and_clicks_produce_a_ctr(self):
        with _Store() as store:
            for _ in range(4):
                store.record_feature_event(
                    user_id="alice", feature_key="topic_videos", event_name="video_panel_viewed"
                )
            for _ in range(1):
                store.record_feature_event(
                    user_id="alice", feature_key="topic_videos", event_name="video_click"
                )
            metrics = store.get_feature_metrics(feature_key="topic_videos")
            self.assertEqual(metrics["panel_views_30d"], 4)
            self.assertEqual(metrics["video_clicks_30d"], 1)
            self.assertEqual(metrics["ctr_30d"], 0.25)

    def test_quota_hide_events_accumulate_hidden_days_and_weekly_counts(self):
        with _Store() as store:
            store.record_feature_event(
                user_id="system",
                feature_key="topic_videos",
                event_name="video_panel_hidden_quota",
                metadata={"disabled_until": _iso_in(hours=6)},
            )
            # The interval starts when the hide event fired, so nothing is
            # measurable until time has actually passed.
            _backdate_events(store, when_iso=_iso_in(days=-2), event_name="video_panel_hidden_quota")
            metrics = store.get_feature_metrics(feature_key="topic_videos")
            self.assertEqual(len(metrics["hidden_frequency_weekly"]), 1)
            self.assertGreater(metrics["hidden_days_last_7"], 0.0)
            self.assertAlmostEqual(metrics["hidden_days_last_7"], 2.0, places=1)
            self.assertGreaterEqual(metrics["hidden_days_last_7"], metrics["hidden_days_last_30"])

    def test_hidden_days_are_clamped_to_the_start_of_the_30_day_window(self):
        with _Store() as store:
            store.record_feature_event(
                user_id="system",
                feature_key="topic_videos",
                event_name="video_panel_hidden_quota",
                metadata={"disabled_until": _iso_in(hours=6)},
            )
            _backdate_events(store, when_iso=_iso_in(days=-45), event_name="video_panel_hidden_quota")
            metrics = store.get_feature_metrics(feature_key="topic_videos")
            self.assertEqual(metrics["hidden_days_last_7"], 0.0)
            # Outside the 30-day query window entirely.
            self.assertEqual(metrics["hidden_days_last_30"], 0.0)

    def test_a_hide_event_without_metadata_still_counts_weekly(self):
        with _Store() as store:
            store.record_feature_event(
                user_id="system", feature_key="topic_videos", event_name="video_panel_hidden_quota"
            )
            metrics = store.get_feature_metrics(feature_key="topic_videos")
            self.assertEqual(len(metrics["hidden_frequency_weekly"]), 1)
            # No disabled_until, so no interval is measurable.
            self.assertEqual(metrics["hidden_days_last_7"], 0.0)

    def test_events_with_corrupt_timestamps_are_skipped(self):
        with _Store() as store:
            store.record_feature_event(
                user_id="alice", feature_key="topic_videos", event_name="video_panel_viewed"
            )
            store._conn.execute(
                "UPDATE feature_events SET created_at = ? WHERE event_name = ?",
                ("not-a-timestamp", "video_panel_viewed"),
            )
            store._conn.commit()
            metrics = store.get_feature_metrics(feature_key="topic_videos")
            self.assertEqual(metrics["panel_views_30d"], 0)

    def test_naive_event_timestamps_are_assumed_utc(self):
        with _Store() as store:
            store.record_feature_event(
                user_id="alice", feature_key="topic_videos", event_name="video_panel_viewed"
            )
            store._conn.execute(
                "UPDATE feature_events SET created_at = ?",
                (datetime.now(UTC).replace(tzinfo=None).isoformat(),),
            )
            store._conn.commit()
            self.assertEqual(
                store.get_feature_metrics(feature_key="topic_videos")["panel_views_30d"], 1
            )

    def test_a_corrupt_disabled_until_inside_a_hide_event_is_ignored(self):
        with _Store() as store:
            store.record_feature_event(
                user_id="system",
                feature_key="topic_videos",
                event_name="video_panel_hidden_quota",
                metadata={"disabled_until": "not-a-timestamp"},
            )
            metrics = store.get_feature_metrics(feature_key="topic_videos")
            self.assertEqual(len(metrics["hidden_frequency_weekly"]), 1)
            self.assertEqual(metrics["hidden_days_last_7"], 0.0)
            self.assertEqual(metrics["hidden_days_last_30"], 0.0)

    def test_a_naive_disabled_until_inside_a_hide_event_is_assumed_utc(self):
        with _Store() as store:
            store.record_feature_event(
                user_id="system",
                feature_key="topic_videos",
                event_name="video_panel_hidden_quota",
                metadata={"disabled_until": _iso_in(hours=3).replace("+00:00", "")},
            )
            _backdate_events(store, when_iso=_iso_in(days=-1), event_name="video_panel_hidden_quota")
            metrics = store.get_feature_metrics(feature_key="topic_videos")
            self.assertGreater(metrics["hidden_days_last_7"], 0.0)

    def test_corrupt_metadata_json_still_counts_the_weekly_bucket(self):
        with _Store() as store:
            store.record_feature_event(
                user_id="system",
                feature_key="topic_videos",
                event_name="video_panel_hidden_quota",
                metadata={"disabled_until": _iso_in(hours=2)},
            )
            store._conn.execute(
                "UPDATE feature_events SET metadata_json = ?", ("{not json",)
            )
            store._conn.commit()
            _backdate_events(store, when_iso=_iso_in(days=-1), event_name="video_panel_hidden_quota")
            metrics = store.get_feature_metrics(feature_key="topic_videos")
            self.assertEqual(len(metrics["hidden_frequency_weekly"]), 1)
            # The interval is unreachable without valid metadata.
            self.assertEqual(metrics["hidden_days_last_7"], 0.0)

    def test_current_quota_disable_is_hidden_now(self):
        with _Store() as store:
            store.set_feature_state(
                feature_key="topic_videos",
                status="disabled_quota_exhausted",
                disabled_until=_iso_in(hours=5),
                reason="youtube_quota_exhausted",
            )
            store._conn.execute(
                "UPDATE feature_runtime_state SET updated_at = ? WHERE feature_key = 'topic_videos'",
                (_iso_in(days=-3),),
            )
            store._conn.commit()
            metrics = store.get_feature_metrics(feature_key="topic_videos")
            self.assertTrue(metrics["hidden_now"])
            self.assertEqual(metrics["reason"], "youtube_quota_exhausted")
            self.assertAlmostEqual(metrics["hidden_days_last_7"], 3.0, places=1)

    def test_a_disable_that_has_not_been_disabled_yet_measures_zero_days(self):
        with _Store() as store:
            store.set_feature_state(
                feature_key="topic_videos",
                status="disabled_quota_exhausted",
                disabled_until=_iso_in(hours=5),
            )
            metrics = store.get_feature_metrics(feature_key="topic_videos")
            self.assertTrue(metrics["hidden_now"])
            # updated_at is "now", so the interval has no length yet.
            self.assertEqual(metrics["hidden_days_last_7"], 0.0)

    def test_a_naive_state_disabled_until_is_assumed_utc(self):
        with _Store() as store:
            store.set_feature_state(
                feature_key="topic_videos",
                status="disabled_quota_exhausted",
                disabled_until=_iso_in(hours=5).replace("+00:00", ""),
            )
            # updated_at is naive too, so the interval start needs the same
            # UTC assumption before the days can be measured.
            store._conn.execute(
                "UPDATE feature_runtime_state SET updated_at = ? WHERE feature_key = 'topic_videos'",
                (_iso_in(days=-2).replace("+00:00", ""),),
            )
            store._conn.commit()
            metrics = store.get_feature_metrics(feature_key="topic_videos")
            self.assertTrue(metrics["hidden_now"])
            self.assertAlmostEqual(metrics["hidden_days_last_7"], 2.0, places=1)

    def test_a_corrupt_state_disabled_until_is_not_hidden_now(self):
        with _Store() as store:
            store.set_feature_state(
                feature_key="topic_videos",
                status="disabled_quota_exhausted",
                disabled_until="not-a-timestamp",
            )
            metrics = store.get_feature_metrics(feature_key="topic_videos")
            self.assertFalse(metrics["hidden_now"])
            self.assertEqual(metrics["disabled_until"], "not-a-timestamp")

    def test_a_blank_state_status_falls_back_to_disabled_config(self):
        with _Store() as store:
            store.set_feature_state(feature_key="topic_videos", status="probe")
            store._conn.execute(
                "UPDATE feature_runtime_state SET status = '   ' WHERE feature_key = 'topic_videos'"
            )
            store._conn.commit()
            self.assertEqual(
                store.get_feature_metrics(feature_key="topic_videos")["status"], "disabled_config"
            )

    def test_corrupt_state_updated_at_does_not_break_hidden_now(self):
        with _Store() as store:
            store.set_feature_state(
                feature_key="topic_videos",
                status="disabled_quota_exhausted",
                disabled_until=_iso_in(hours=5),
            )
            store._conn.execute(
                "UPDATE feature_runtime_state SET updated_at = ? WHERE feature_key = 'topic_videos'",
                ("not-a-timestamp",),
            )
            store._conn.commit()
            metrics = store.get_feature_metrics(feature_key="topic_videos")
            # hidden_now still reads from disabled_until; only the interval is lost.
            self.assertTrue(metrics["hidden_now"])
            self.assertEqual(metrics["hidden_days_last_7"], 0.0)

    def test_reenable_event_rebases_the_view_and_click_counts(self):
        with _Store() as store:
            store.record_feature_event(
                user_id="alice", feature_key="topic_videos", event_name="video_panel_viewed"
            )
            store.record_feature_event(
                user_id="system", feature_key="topic_videos", event_name="video_feature_reenabled"
            )
            for _ in range(3):
                store.record_feature_event(
                    user_id="alice", feature_key="topic_videos", event_name="video_panel_viewed"
                )
            store.record_feature_event(
                user_id="alice", feature_key="topic_videos", event_name="video_click"
            )
            metrics = store.get_feature_metrics(feature_key="topic_videos")
            self.assertTrue(metrics["reenabled_at"])
            self.assertEqual(metrics["panel_views_30d"], 4)
            self.assertEqual(metrics["panel_views_since_reenable"], 3)
            self.assertEqual(metrics["clicks_since_reenable"], 1)
            self.assertEqual(metrics["ctr_since_reenable"], round(1 / 3, 4))

    def test_hidden_days_ignore_intervals_that_end_before_the_window(self):
        with _Store() as store:
            # A hide window that closed 40 days ago is outside both windows.
            store.record_feature_event(
                user_id="system",
                feature_key="topic_videos",
                event_name="video_panel_hidden_quota",
                metadata={"disabled_until": _iso_in(days=-40)},
            )
            metrics = store.get_feature_metrics(feature_key="topic_videos")
            self.assertEqual(metrics["hidden_days_last_7"], 0.0)
            self.assertEqual(metrics["hidden_days_last_30"], 0.0)

    def test_weekly_buckets_are_capped_at_eight_but_the_window_binds_first(self):
        with _Store() as store:
            for week in range(20):
                created = datetime.now(UTC) - timedelta(days=7 * week + 1)
                store._conn.execute(
                    "INSERT INTO feature_events(user_id, feature_key, event_name, metadata_json, created_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (
                        "system",
                        "topic_videos",
                        "video_panel_hidden_quota",
                        "{}",
                        created.isoformat(),
                    ),
                )
            store._conn.commit()
            metrics = store.get_feature_metrics(feature_key="topic_videos")
            weeks = [entry["week_start"] for entry in metrics["hidden_frequency_weekly"]]
            # Only events inside the 30-day window are read, so at most five
            # distinct week buckets can exist before the [:8] cap would apply.
            self.assertEqual(len(weeks), 5)
            self.assertLessEqual(len(weeks), 8)
            self.assertEqual(weeks, sorted(weeks, reverse=True))
            self.assertTrue(all(entry["hidden_events"] == 1 for entry in metrics["hidden_frequency_weekly"]))

    def test_metrics_are_scoped_to_the_requested_feature_key(self):
        with _Store() as store:
            store.record_feature_event(
                user_id="alice", feature_key="topic_videos", event_name="video_panel_viewed"
            )
            store.record_feature_event(
                user_id="alice", feature_key="other_feature", event_name="video_panel_viewed"
            )
            self.assertEqual(
                store.get_feature_metrics(feature_key="topic_videos")["panel_views_30d"], 1
            )
            self.assertEqual(
                store.get_feature_metrics(feature_key="other_feature")["panel_views_30d"], 1
            )


if __name__ == "__main__":
    unittest.main()
