"""Coverage for app/services/progress_store.py.

Complements ``tests/test_progress_store.py`` (the write/read happy paths and the
summary state machine) with the corrupt-JSON readers, the additive-column
migration, the duplicate-section filter, and the direct ``touch_user`` path.
"""

import asyncio
import os
import sqlite3
import tempfile
import unittest

from app.services.progress_store import (
    MAX_QUESTIONS_ASKED,
    ProgressStore,
    _loads_dict,
    _loads_list,
)


class JsonReaderTests(unittest.TestCase):
    def test_valid_list_round_trips(self):
        self.assertEqual(_loads_list('["a", "b"]'), ["a", "b"])

    def test_missing_and_empty_input_become_an_empty_list(self):
        self.assertEqual(_loads_list(""), [])
        self.assertEqual(_loads_list(None), [])

    def test_corrupt_json_becomes_an_empty_list(self):
        self.assertEqual(_loads_list("{not json"), [])
        self.assertEqual(_loads_list("[[["), [])

    def test_a_json_object_is_not_a_list(self):
        self.assertEqual(_loads_list('{"a": 1}'), [])
        self.assertEqual(_loads_list('"a string"'), [])
        self.assertEqual(_loads_list("42"), [])

    def test_dict_reader_mirrors_the_list_reader(self):
        self.assertEqual(_loads_dict('{"a": 1}'), {"a": 1})
        self.assertEqual(_loads_dict(""), {})
        self.assertEqual(_loads_dict(None), {})
        self.assertEqual(_loads_dict("{not json"), {})
        self.assertEqual(_loads_dict('["a"]'), {})
        self.assertEqual(_loads_dict("42"), {})


class StoreFixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = ProgressStore(os.path.join(self._tmp.name, "progress.db"))

    def _write_raw(self, sql, params=()):
        with self.store._conn:
            self.store._conn.execute(sql, params)


class PathAndExecutorTests(StoreFixture):
    def test_a_bare_filename_needs_no_directory_creation(self):
        cwd = os.getcwd()
        os.chdir(self._tmp.name)
        try:
            store = ProgressStore("bare.db")
            names = {
                str(row["name"])
                for row in store._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        finally:
            os.chdir(cwd)
        self.assertIn("topic_progress", names)

    def test_run_async_accepts_positional_arguments(self):
        self.store.upsert_topic_progress(
            user_id="alice",
            topic_id="positional",
            topic_title="Positional",
            questions=["Q1?"],
            sections=[],
            attempt_stats={},
        )

        async def _exercise():
            # Purely positional: the bound method's own parameters.
            title = await self.store.run_async(
                os.path.basename, "/tmp/learning.db"
            )
            # Keyword form, which is how every call site uses it.
            by_id = await self.store.run_async(
                self.store.get_progress, user_id="alice", topic_id="positional"
            )
            # Mixed positional and keyword.
            captured = await self.store.run_async(
                lambda *args, **kwargs: (args, kwargs), 1, 2, key="v"
            )
            return title, by_id, captured

        title, by_id, captured = asyncio.run(_exercise())
        self.assertEqual(title, "learning.db")
        self.assertEqual(by_id["topic_id"], "positional")
        self.assertEqual(captured, ((1, 2), {"key": "v"}))

    def test_touch_user_creates_then_refreshes_the_row(self):
        self.store.touch_user(user_id="alice", provider="google", count_session=True)
        rows = {
            str(row["user_id"]): dict(row)
            for row in self.store._conn.execute("SELECT * FROM users").fetchall()
        }
        self.assertEqual(rows["alice"]["provider"], "google")
        self.assertEqual(rows["alice"]["session_count"], 1)
        self.assertTrue(rows["alice"]["first_seen_at"])

        # A second touch refreshes last_seen and bumps the session count.
        self.store.touch_user(user_id="alice", provider="", count_session=True)
        rows = {
            str(row["user_id"]): dict(row)
            for row in self.store._conn.execute("SELECT * FROM users").fetchall()
        }
        # A blank provider never overwrites a known one.
        self.assertEqual(rows["alice"]["provider"], "google")
        self.assertEqual(rows["alice"]["session_count"], 2)

    def test_touch_user_without_counting_a_session_keeps_the_counter(self):
        self.store.touch_user(user_id="bob", provider="github")
        self.store.touch_user(user_id="bob", provider="github")
        rows = {
            str(row["user_id"]): dict(row)
            for row in self.store._conn.execute("SELECT * FROM users").fetchall()
        }
        self.assertEqual(rows["bob"]["session_count"], 0)
        self.assertEqual(rows["bob"]["provider"], "github")

    def test_touch_user_normalises_a_whitespace_provider(self):
        self.store.touch_user(user_id="carol", provider="   google  ")
        rows = {
            str(row["user_id"]): dict(row)
            for row in self.store._conn.execute("SELECT * FROM users").fetchall()
        }
        self.assertEqual(rows["carol"]["provider"], "google")


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = os.path.join(self._tmp.name, "legacy.db")

    def _create_legacy_table(self):
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE topic_progress (
                        user_id TEXT NOT NULL,
                        topic_id TEXT NOT NULL,
                        topic_title TEXT NOT NULL,
                        summary_text TEXT NOT NULL DEFAULT '',
                        summary_status TEXT NOT NULL DEFAULT 'empty',
                        summary_error TEXT NOT NULL DEFAULT '',
                        questions_asked_json TEXT NOT NULL DEFAULT '[]',
                        sections_json TEXT NOT NULL DEFAULT '[]',
                        attempt_stats_json TEXT NOT NULL DEFAULT '{}',
                        preferred_language TEXT NOT NULL DEFAULT '',
                        question_count INTEGER NOT NULL DEFAULT 0,
                        revision INTEGER NOT NULL DEFAULT 0,
                        provider_used TEXT NOT NULL DEFAULT '',
                        model_used TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (user_id, topic_id)
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO topic_progress(
                        user_id, topic_id, topic_title, questions_asked_json,
                        sections_json, attempt_stats_json, created_at, updated_at
                    ) VALUES ('legacy', 'topic-1', 'Legacy Topic', '["Legacy Q?"]',
                              '["S1"]', '{"total": 3}', '2026-01-01T00:00:00+00:00',
                              '2026-01-01T00:00:00+00:00')
                    """
                )
        finally:
            conn.close()

    def test_missing_columns_are_added_and_existing_rows_survive(self):
        # generation_source is absent from the legacy table, so _init_db must
        # add it via ALTER TABLE without touching the stored row.
        self._create_legacy_table()
        store = ProgressStore(self.db_path)
        columns = {
            str(row["name"])
            for row in store._conn.execute(
                "PRAGMA table_info(topic_progress)"
            ).fetchall()
        }
        self.assertIn("generation_source", columns)
        document = store.get_progress(user_id="legacy", topic_id="topic-1")
        self.assertEqual(document["topic_title"], "Legacy Topic")
        self.assertEqual(document["questions_asked"], ["Legacy Q?"])
        self.assertEqual(document["sections"], ["S1"])
        self.assertEqual(document["attempt_stats"], {"total": 3})
        self.assertEqual(document["generation_source"], "")

    def test_reopening_an_up_to_date_table_changes_nothing(self):
        store = ProgressStore(self.db_path)
        store.upsert_topic_progress(
            user_id="alice",
            topic_id="topic-a",
            topic_title="Topic A",
            questions=["Q?"],
            sections=["S1"],
            attempt_stats={},
        )
        reopened = ProgressStore(self.db_path)
        document = reopened.get_progress(user_id="alice", topic_id="topic-a")
        self.assertEqual(document["questions_asked"], ["Q?"])
        self.assertEqual(document["revision"], 1)


class CorruptRowTests(StoreFixture):
    def test_corrupt_json_columns_degrade_to_empty_values(self):
        self._write_raw(
            "UPDATE topic_progress SET questions_asked_json = 'nope', "
            "sections_json = 'nope', attempt_stats_json = 'nope'"
        )
        self.store.upsert_topic_progress(
            user_id="alice",
            topic_id="topic-corrupt",
            topic_title="Corrupt",
            questions=["Q?"],
            sections=["S1"],
            attempt_stats={},
        )
        self._write_raw(
            "UPDATE topic_progress SET questions_asked_json = '{oops', "
            "sections_json = '{\"not\": \"a list\"}', "
            "attempt_stats_json = '[1, 2, 3]' WHERE user_id = 'alice'"
        )
        document = self.store.get_progress(
            user_id="alice", topic_id="topic-corrupt"
        )
        self.assertEqual(document["questions_asked"], [])
        self.assertEqual(document["sections"], [])
        self.assertEqual(document["attempt_stats"], {})

    def test_a_non_list_attempt_stats_column_is_ignored_on_write(self):
        document = self.store.upsert_topic_progress(
            user_id="alice",
            topic_id="topic-stats",
            topic_title="Stats",
            questions=["Q?"],
            sections=[],
            attempt_stats=["not", "a", "dict"],
        )
        self.assertEqual(document["attempt_stats"], {})


class SectionDedupeTests(StoreFixture):
    def test_sections_are_deduped_case_insensitively(self):
        document = self.store.upsert_topic_progress(
            user_id="alice",
            topic_id="topic-sections",
            topic_title="Sections",
            questions=["Q?"],
            sections=["Intro", "intro", "  INTRO ", "Depth", "", None, "   "],
            attempt_stats={},
        )
        self.assertEqual(document["sections"], ["Intro", "Depth"])

    def test_a_repeated_section_is_not_re_added_by_record_asked_questions(self):
        self.store.record_asked_questions(
            user_id="alice",
            topic_id="topic-repeat",
            topic_title="Repeat",
            questions=["Q1?"],
            section_title="Overview",
        )
        document = self.store.record_asked_questions(
            user_id="alice",
            topic_id="topic-repeat",
            topic_title="Repeat",
            questions=["Q2?"],
            section_title="Overview",
        )
        self.assertEqual(document["sections"], ["Overview"])
        self.assertEqual(document["questions_asked"], ["Q2?", "Q1?"])

    def test_duplicate_questions_are_dropped_across_writes(self):
        self.store.record_asked_questions(
            user_id="alice",
            topic_id="topic-dup",
            topic_title="Dup",
            questions=["What is a connection pool?", "What is a connection pool?"],
        )
        document = self.store.record_asked_questions(
            user_id="alice",
            topic_id="topic-dup",
            topic_title="Dup",
            questions=["What is a connection pool?", "  what is   A connection pool?  "],
        )
        # The newest text wins the head of the list; the duplicate collapses
        # against it via the shared normaliser.
        self.assertEqual(document["questions_asked"], ["what is   A connection pool?"])

    def test_blank_question_texts_are_ignored(self):
        document = self.store.record_asked_questions(
            user_id="alice",
            topic_id="topic-blank",
            topic_title="Blank",
            questions=["", "   ", None, "Real question?"],
        )
        self.assertEqual(document["questions_asked"], ["Real question?"])


class TitleAndLanguageDefaultsTests(StoreFixture):
    def test_a_blank_title_falls_back_to_the_topic_id(self):
        document = self.store.upsert_topic_progress(
            user_id="alice",
            topic_id="topic-blank-title",
            topic_title="   ",
            questions=["Q?"],
            sections=[],
            attempt_stats={},
        )
        self.assertEqual(document["topic_title"], "topic-blank-title")

    def test_a_blank_title_on_upsert_keeps_a_known_one(self):
        # ``SaveProgressRequest.topic_title`` defaults to "" (models.py), so a
        # client that saves progress without a title must not replace the
        # stored human-readable title with the opaque topic id. The SQL
        # `CASE WHEN excluded.topic_title != '' ...` guard is only reachable
        # because the raw title reaches the statement un-substituted.
        self.store.upsert_topic_progress(
            user_id="alice",
            topic_id="topic-keep-title",
            topic_title="Real Title",
            questions=["Q1?"],
            sections=[],
            attempt_stats={},
        )
        document = self.store.upsert_topic_progress(
            user_id="alice",
            topic_id="topic-keep-title",
            topic_title="",
            questions=["Q2?"],
            sections=[],
            attempt_stats={},
        )
        self.assertEqual(document["topic_title"], "Real Title")
        self.assertEqual(sorted(document["questions_asked"]), ["Q1?", "Q2?"])

    def test_a_real_title_still_replaces_a_stored_one_on_upsert(self):
        self.store.upsert_topic_progress(
            user_id="alice",
            topic_id="topic-retitle",
            topic_title="Old Title",
            questions=["Q1?"],
            sections=[],
            attempt_stats={},
        )
        document = self.store.upsert_topic_progress(
            user_id="alice",
            topic_id="topic-retitle",
            topic_title="New Title",
            questions=["Q2?"],
            sections=[],
            attempt_stats={},
        )
        self.assertEqual(document["topic_title"], "New Title")

    def test_a_blank_title_on_record_asked_questions_keeps_a_known_one(self):
        self.store.upsert_topic_progress(
            user_id="alice",
            topic_id="topic-keep-title",
            topic_title="Real Title",
            questions=["Q1?"],
            sections=[],
            attempt_stats={},
        )
        document = self.store.record_asked_questions(
            user_id="alice",
            topic_id="topic-keep-title",
            questions=["Q2?"],
        )
        self.assertEqual(document["topic_title"], "Real Title")

    def test_a_blank_language_does_not_overwrite_a_known_one(self):
        self.store.upsert_topic_progress(
            user_id="alice",
            topic_id="topic-lang",
            topic_title="Lang",
            questions=["Q1?"],
            sections=[],
            attempt_stats={},
            preferred_language="python",
        )
        document = self.store.record_asked_questions(
            user_id="alice",
            topic_id="topic-lang",
            topic_title="Lang",
            questions=["Q2?"],
            preferred_language="  ",
        )
        self.assertEqual(document["preferred_language"], "python")

    def test_record_asked_questions_bootstraps_the_row(self):
        document = self.store.record_asked_questions(
            user_id="alice",
            topic_id="topic-new",
            topic_title="",
            questions=["Q?"],
        )
        # No title was supplied, so the topic id stands in.
        self.assertEqual(document["topic_title"], "topic-new")
        self.assertEqual(document["revision"], 1)
        self.assertEqual(document["summary_status"], "empty")


class ReadBoundariesTests(StoreFixture):
    def _seed(self, count):
        for index in range(count):
            self.store.record_asked_questions(
                user_id="alice",
                topic_id=f"topic-{index}",
                topic_title=f"Topic {index}",
                questions=[f"Question {index}?"],
            )

    def test_list_progress_limit_is_clamped(self):
        self._seed(4)
        # A falsy limit means "use the default", so it is not clamped to 1.
        self.assertEqual(len(self.store.list_progress(user_id="alice", limit=0)), 4)
        self.assertEqual(len(self.store.list_progress(user_id="alice", limit=2)), 2)
        # An absurd limit is capped at the 200 hard ceiling.
        self.assertEqual(
            len(self.store.list_progress(user_id="alice", limit=10_000)), 4
        )
        self.assertEqual(
            len(self.store.list_progress(user_id="alice", limit=None)), 4
        )

    def test_get_asked_questions_limit_is_clamped(self):
        self.store.record_asked_questions(
            user_id="alice",
            topic_id="topic-cap",
            topic_title="Cap",
            questions=[f"Question {index}?" for index in range(MAX_QUESTIONS_ASKED)],
        )
        asked = self.store.get_asked_questions(
            user_id="alice", topic_id="topic-cap", limit=0
        )
        # A falsy limit means "use the default", so all stored questions come
        # back rather than a single one.
        self.assertEqual(len(asked), MAX_QUESTIONS_ASKED)
        self.assertEqual(
            len(
                self.store.get_asked_questions(
                    user_id="alice",
                    topic_id="topic-cap",
                    limit=MAX_QUESTIONS_ASKED + 500,
                )
            ),
            MAX_QUESTIONS_ASKED,
        )
        self.assertEqual(
            len(
                self.store.get_asked_questions(
                    user_id="alice", topic_id="topic-cap", limit=5
                )
            ),
            5,
        )

    def test_list_progress_for_an_unknown_user_is_empty(self):
        self.assertEqual(self.store.list_progress(user_id="ghost"), [])

    def test_set_summary_on_a_missing_row_returns_none(self):
        self.assertIsNone(
            self.store.mark_summary_failed(
                user_id="ghost", topic_id="nope", error="boom"
            )
        )

    def test_summary_error_is_capped_at_500_characters(self):
        self.store.upsert_topic_progress(
            user_id="alice",
            topic_id="topic-err",
            topic_title="Err",
            questions=["Q?"],
            sections=[],
            attempt_stats={},
        )
        document = self.store.mark_summary_failed(
            user_id="alice", topic_id="topic-err", error="e" * 900
        )
        self.assertEqual(len(document["summary_error"]), 500)

    def test_summary_provider_and_model_are_only_overwritten_when_given(self):
        self.store.upsert_topic_progress(
            user_id="alice",
            topic_id="topic-prov",
            topic_title="Prov",
            questions=["Q?"],
            sections=[],
            attempt_stats={},
        )
        self.store.set_summary(
            user_id="alice",
            topic_id="topic-prov",
            text="first",
            provider_used="groq",
            model_used="llama",
            source="ai",
        )
        # mark_summary_pending passes no provider/model/source at all.
        pending = self.store.mark_summary_pending(
            user_id="alice", topic_id="topic-prov"
        )
        self.assertEqual(pending["provider_used"], "groq")
        self.assertEqual(pending["model_used"], "llama")
        self.assertEqual(pending["generation_source"], "ai")
        self.assertEqual(pending["summary_text"], "first")
        self.assertEqual(pending["summary_error"], "")

    def test_a_blank_summary_text_is_stored_as_empty(self):
        self.store.upsert_topic_progress(
            user_id="alice",
            topic_id="topic-empty-sum",
            topic_title="Empty",
            questions=["Q?"],
            sections=[],
            attempt_stats={},
        )
        document = self.store.set_summary(
            user_id="alice", topic_id="topic-empty-sum", text=None, source="fallback"
        )
        self.assertEqual(document["summary_text"], "")
        self.assertEqual(document["summary_status"], "ready")
        self.assertEqual(document["generation_source"], "fallback")


if __name__ == "__main__":
    unittest.main()