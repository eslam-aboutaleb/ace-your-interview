import asyncio
import os
import sqlite3
import tempfile
import unittest

from app.services.progress_store import MAX_QUESTIONS_ASKED, ProgressStore


class ProgressStoreTests(unittest.TestCase):
    def _store(self, td: str) -> ProgressStore:
        return ProgressStore(os.path.join(td, "learning.db"))

    def _users(self, store: ProgressStore) -> dict[str, dict]:
        rows = store._conn.execute("SELECT * FROM users").fetchall()
        return {str(row["user_id"]): dict(row) for row in rows}

    def test_ddl_creates_users_and_topic_progress_tables(self):
        with tempfile.TemporaryDirectory() as td:
            store = self._store(td)
            rows = store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            names = {str(row["name"]) for row in rows}
            self.assertIn("users", names)
            self.assertIn("topic_progress", names)

    def test_run_async_executes_sqlite_work_off_main_loop(self):
        async def _exercise() -> None:
            with tempfile.TemporaryDirectory() as td:
                store = self._store(td)
                saved = await store.run_async(
                    store.record_asked_questions,
                    user_id="alice",
                    topic_id="topic-async",
                    topic_title="Async Topic",
                    questions=["How does a connection pool behave under load?"],
                )
                asked = await store.run_async(
                    store.get_asked_questions,
                    user_id="alice",
                    topic_id="topic-async",
                    limit=200,
                )
                self.assertEqual(saved["topic_id"], "topic-async")
                self.assertEqual(asked, ["How does a connection pool behave under load?"])

        asyncio.run(_exercise())

    def test_upsert_creates_row_and_bumps_revision(self):
        with tempfile.TemporaryDirectory() as td:
            store = self._store(td)
            first = store.upsert_topic_progress(
                user_id="alice",
                provider="google",
                topic_id="topic-a",
                topic_title="Topic A",
                questions=["Question one?"],
                sections=["Intro"],
                attempt_stats={"total": 1},
                preferred_language="python",
            )
            second = store.upsert_topic_progress(
                user_id="alice",
                provider="google",
                topic_id="topic-a",
                topic_title="Topic A",
                questions=["Question two?"],
                sections=["Intro", "Depth"],
                attempt_stats={"total": 2},
                preferred_language="python",
            )
            self.assertEqual(first["revision"], 1)
            self.assertEqual(second["revision"], 2)
            self.assertEqual(second["summary_status"], "empty")
            self.assertEqual(second["sections"], ["Intro", "Depth"])
            self.assertEqual(second["attempt_stats"], {"total": 2})
            self.assertEqual(second["preferred_language"], "python")

    def test_questions_asked_is_newest_first_and_capped(self):
        with tempfile.TemporaryDirectory() as td:
            store = self._store(td)
            store.record_asked_questions(
                user_id="alice",
                topic_id="topic-cap",
                topic_title="Topic Cap",
                questions=["Older one?", "Newer one?"],
            )
            stored = store.get_progress(user_id="alice", topic_id="topic-cap")
            self.assertEqual(stored["questions_asked"], ["Newer one?", "Older one?"])

            batches = [f"Question {i}?" for i in range(MAX_QUESTIONS_ASKED + 40)]
            store.record_asked_questions(
                user_id="alice",
                topic_id="topic-cap",
                topic_title="Topic Cap",
                questions=batches,
            )
            asked = store.get_asked_questions(
                user_id="alice", topic_id="topic-cap", limit=MAX_QUESTIONS_ASKED
            )
            self.assertEqual(len(asked), MAX_QUESTIONS_ASKED)
            # The newest question survives; the oldest dropped off the tail.
            self.assertEqual(asked[0], batches[-1])
            self.assertNotIn("Older one?", asked)
            self.assertNotIn(batches[0], asked)

    def test_cross_list_dedupe_uses_generator_normalizer(self):
        with tempfile.TemporaryDirectory() as td:
            store = self._store(td)
            store.record_asked_questions(
                user_id="alice",
                topic_id="topic-dedupe",
                topic_title="Topic Dedupe",
                questions=["Explain  the   idempotency  guarantee of PUT?"],
            )
            # Different case + collapsed whitespace must collapse to one entry
            # rather than adding a near-identical second question to the prompt.
            store.record_asked_questions(
                user_id="alice",
                topic_id="topic-dedupe",
                topic_title="Topic Dedupe",
                questions=["explain the idempotency guarantee of put?"],
            )
            asked = store.get_asked_questions(user_id="alice", topic_id="topic-dedupe", limit=200)
            self.assertEqual(len(asked), 1)

            # The client-supplied list de-dupes against the stored one too.
            document = store.upsert_topic_progress(
                user_id="alice",
                topic_id="topic-dedupe",
                topic_title="Topic Dedupe",
                questions=["EXPLAIN THE IDEMPOTENCY GUARANTEE OF PUT?", "Brand new?"],
                sections=[],
                attempt_stats={},
            )
            self.assertEqual(len(document["questions_asked"]), 2)

    def test_save_union_keeps_prior_history(self):
        with tempfile.TemporaryDirectory() as td:
            store = self._store(td)
            store.record_asked_questions(
                user_id="alice",
                topic_id="topic-union",
                topic_title="Topic Union",
                questions=["Stored question one?"],
            )
            document = store.upsert_topic_progress(
                user_id="alice",
                provider="google",
                topic_id="topic-union",
                topic_title="Topic Union",
                questions=["Client question two?"],
                sections=["S1"],
                attempt_stats={},
            )
            self.assertEqual(
                document["questions_asked"],
                ["Client question two?", "Stored question one?"],
            )

    def test_summary_status_transitions(self):
        with tempfile.TemporaryDirectory() as td:
            store = self._store(td)
            store.upsert_topic_progress(
                user_id="alice",
                topic_id="topic-sum",
                topic_title="Topic Sum",
                questions=["Q?"],
                sections=[],
                attempt_stats={},
            )
            pending = store.mark_summary_pending(user_id="alice", topic_id="topic-sum")
            self.assertEqual(pending["summary_status"], "pending")
            self.assertEqual(pending["summary_text"], "")

            ready = store.set_summary(
                user_id="alice",
                topic_id="topic-sum",
                text="Learner covered basics.",
                provider_used="groq",
                model_used="llama-3.3-70b-versatile",
                source="ai",
            )
            self.assertEqual(ready["summary_status"], "ready")
            self.assertEqual(ready["summary_text"], "Learner covered basics.")
            self.assertEqual(ready["provider_used"], "groq")
            self.assertEqual(ready["generation_source"], "ai")

            failed = store.mark_summary_failed(
                user_id="alice", topic_id="topic-sum", error="boom"
            )
            self.assertEqual(failed["summary_status"], "failed")
            self.assertEqual(failed["summary_error"], "boom")
            # A failed refresh must not discard the note already stored.
            self.assertEqual(failed["summary_text"], "Learner covered basics.")

    def test_summary_is_capped(self):
        with tempfile.TemporaryDirectory() as td:
            store = self._store(td)
            store.upsert_topic_progress(
                user_id="alice",
                topic_id="topic-cap2",
                topic_title="Topic Cap2",
                questions=["Q?"],
                sections=[],
                attempt_stats={},
            )
            document = store.set_summary(
                user_id="alice",
                topic_id="topic-cap2",
                text="x" * 5000,
                source="fallback",
            )
            self.assertEqual(len(document["summary_text"]), 2000)

    def test_summary_writes_are_noop_without_a_row(self):
        with tempfile.TemporaryDirectory() as td:
            store = self._store(td)
            self.assertIsNone(
                store.set_summary(user_id="ghost", topic_id="nope", text="x")
            )
            self.assertIsNone(store.mark_summary_pending(user_id="ghost", topic_id="nope"))

    def test_touch_user_creates_and_counts_sessions(self):
        with tempfile.TemporaryDirectory() as td:
            store = self._store(td)
            store.upsert_topic_progress(
                user_id="alice",
                provider="google",
                topic_id="topic-users",
                topic_title="Topic Users",
                questions=["Q?"],
                sections=[],
                attempt_stats={},
            )
            # Server-side capture touches the row but is not a study session.
            store.record_asked_questions(
                user_id="alice",
                topic_id="topic-users",
                topic_title="Topic Users",
                questions=["Q2?"],
            )
            store.upsert_topic_progress(
                user_id="alice",
                provider="google",
                topic_id="topic-users",
                topic_title="Topic Users",
                questions=["Q3?"],
                sections=[],
                attempt_stats={},
            )
            users = self._users(store)
            self.assertIn("alice", users)
            self.assertEqual(users["alice"]["provider"], "google")
            self.assertEqual(users["alice"]["session_count"], 2)

    def test_blank_provider_does_not_overwrite_known_provider(self):
        with tempfile.TemporaryDirectory() as td:
            store = self._store(td)
            store.upsert_topic_progress(
                user_id="alice",
                provider="github",
                topic_id="topic-provider",
                topic_title="Topic Provider",
                questions=["Q?"],
                sections=[],
                attempt_stats={},
            )
            store.record_asked_questions(
                user_id="alice",
                topic_id="topic-provider",
                topic_title="Topic Provider",
                questions=["Q2?"],
            )
            users = self._users(store)
            self.assertEqual(users["alice"]["provider"], "github")

    def test_list_progress_is_scoped_and_newest_first(self):
        with tempfile.TemporaryDirectory() as td:
            store = self._store(td)
            store.upsert_topic_progress(
                user_id="alice",
                topic_id="topic-one",
                topic_title="One",
                questions=["Q1?"],
                sections=[],
                attempt_stats={},
            )
            store.upsert_topic_progress(
                user_id="bob",
                topic_id="topic-two",
                topic_title="Two",
                questions=["Q2?"],
                sections=[],
                attempt_stats={},
            )
            store.upsert_topic_progress(
                user_id="alice",
                topic_id="topic-three",
                topic_title="Three",
                questions=["Q3?"],
                sections=[],
                attempt_stats={},
            )
            documents = store.list_progress(user_id="alice", limit=10)
            self.assertEqual(
                [doc["topic_id"] for doc in documents], ["topic-three", "topic-one"]
            )
            self.assertEqual(store.list_progress(user_id="bob", limit=10)[0]["topic_id"], "topic-two")

    def test_get_progress_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            store = self._store(td)
            self.assertIsNone(store.get_progress(user_id="alice", topic_id="absent"))
            self.assertEqual(store.get_asked_questions(user_id="alice", topic_id="absent"), [])

    def test_shares_database_file_with_other_stores(self):
        from app.services.learning_store import LearningStore

        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "shared.db")
            progress = ProgressStore(db_path)
            learning = LearningStore(db_path)
            learning.record_attempt(
                user_id="alice",
                question_id="q1",
                topic_id="topic-a",
                user_answer="answer",
                is_correct=True,
                confidence=4,
                response_time_ms=800,
                mode="study",
            )
            progress.upsert_topic_progress(
                user_id="alice",
                topic_id="topic-a",
                topic_title="Topic A",
                questions=["Q?"],
                sections=[],
                attempt_stats={},
            )
            conn = sqlite3.connect(db_path)
            try:
                names = {
                    str(row[0])
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
            finally:
                conn.close()
            self.assertIn("learning_attempts", names)
            self.assertIn("topic_progress", names)
