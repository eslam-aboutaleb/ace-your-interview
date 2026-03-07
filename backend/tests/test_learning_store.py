import os
import tempfile
import unittest

from app.services.learning_store import LearningStore


class LearningStoreTests(unittest.TestCase):
    def test_incorrect_high_confidence_is_prioritized(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "learning.db")
            store = LearningStore(db_path)
            first = store.record_attempt(
                user_id="alice",
                question_id="q1",
                topic_id="topic-a",
                user_answer="wrong",
                is_correct=False,
                confidence=5,
                response_time_ms=1200,
                mode="quiz",
            )
            second = store.record_attempt(
                user_id="alice",
                question_id="q2",
                topic_id="topic-a",
                user_answer="right",
                is_correct=True,
                confidence=3,
                response_time_ms=900,
                mode="quiz",
            )

            self.assertLess(first["mastery_score"], second["mastery_score"])

            queue = store.get_review_queue(user_id="alice", limit=10)
            self.assertGreaterEqual(queue["total_due"], 1)
            self.assertEqual(queue["items"][0]["question_id"], "q1")

    def test_weak_area_aggregation(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "learning.db")
            store = LearningStore(db_path)
            store.record_attempt(
                user_id="bob",
                question_id="q1",
                topic_id="topic-x",
                user_answer="a",
                is_correct=False,
                confidence=4,
                response_time_ms=1000,
                mode="study",
            )
            store.record_attempt(
                user_id="bob",
                question_id="q2",
                topic_id="topic-y",
                user_answer="b",
                is_correct=True,
                confidence=4,
                response_time_ms=1000,
                mode="study",
            )

            weak = store.get_weak_areas(user_id="bob", limit=5)
            self.assertEqual(len(weak["weak_areas"]), 2)
            self.assertEqual(weak["weak_areas"][0]["topic_id"], "topic-x")

    def test_custom_topics_upsert_and_user_isolation(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "learning.db")
            store = LearningStore(db_path)
            sections = [{"heading": "Core Java", "content": "Understand JVM basics."}]

            saved = store.upsert_custom_topic(
                user_id="alice",
                topic_id="custom-java",
                source_topic="Java",
                title="Java Interview Roadmap",
                description="Deep Java prep roadmap.",
                track="backend",
                levels=["junior", "mid", "senior"],
                sections=sections,
                raw_content="# Java Interview Roadmap",
            )
            self.assertEqual(saved["id"], "custom-java")

            detail = store.get_custom_topic(user_id="alice", topic_id="custom-java")
            self.assertIsNotNone(detail)
            assert detail is not None
            self.assertEqual(detail["title"], "Java Interview Roadmap")
            self.assertEqual(len(detail["sections"]), 1)

            # Replace the same custom topic for the same user.
            store.upsert_custom_topic(
                user_id="alice",
                topic_id="custom-java",
                source_topic="Java",
                title="Java Teacher Roadmap",
                description="Updated Java roadmap.",
                track="backend",
                levels=["junior", "mid", "senior"],
                sections=[
                    {"heading": "Java Collections", "content": "Collections framework tradeoffs."}
                ],
                raw_content="# Java Teacher Roadmap",
            )
            updated = store.get_custom_topic(user_id="alice", topic_id="custom-java")
            self.assertIsNotNone(updated)
            assert updated is not None
            self.assertEqual(updated["title"], "Java Teacher Roadmap")
            self.assertEqual(updated["sections"][0]["heading"], "Java Collections")

            # Same topic ID is isolated per user.
            self.assertIsNone(store.get_custom_topic(user_id="bob", topic_id="custom-java"))
            self.assertEqual(len(store.list_custom_topics(user_id="alice")), 1)
            self.assertEqual(len(store.list_custom_topics(user_id="bob")), 0)

    def test_topic_preferences_and_language_profiles(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "learning.db")
            store = LearningStore(db_path)

            pref = store.get_topic_preferences(user_id="alice", topic_id="01-static")
            self.assertIsNone(pref)

            saved = store.upsert_topic_preferences(
                user_id="alice",
                topic_id="01-static",
                response_detail="very_detailed",
                preferred_language="python",
            )
            self.assertEqual(saved["response_detail"], "very_detailed")
            self.assertEqual(saved["preferred_language"], "python")

            profile = store.upsert_topic_language_profile(
                topic_id="01-static",
                requires_programming=True,
                language_options=["python", "go", "python", "typescript"],
                source="llm",
            )
            self.assertTrue(profile["requires_programming"])
            self.assertEqual(profile["language_options"], ["python", "go", "typescript"])

            resolved = store.resolve_topic_ai_settings(
                user_id="alice",
                topic_id="01-static",
                topic_detail={
                    "id": "01-static",
                    "title": "Static",
                    "description": "Static description",
                    "track": "backend",
                },
            )
            self.assertEqual(resolved["response_detail"], "very_detailed")
            self.assertEqual(resolved["preferred_language"], "python")
            self.assertTrue(resolved["requires_programming"])

    def test_topic_preferences_are_user_isolated(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "learning.db")
            store = LearningStore(db_path)

            store.upsert_topic_preferences(
                user_id="alice",
                topic_id="topic-1",
                response_detail="very_detailed",
                preferred_language="typescript",
            )
            bob_pref = store.get_topic_preferences(user_id="bob", topic_id="topic-1")
            self.assertIsNone(bob_pref)

    def test_dynamic_topic_curriculum_crud_and_isolation(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "learning.db")
            store = LearningStore(db_path)

            sections = [
                {"heading": "Junior: Two Pointers", "content": "Use two indices to scan arrays."}
            ]
            saved = store.upsert_dynamic_topic_curriculum(
                user_id="alice",
                topic_id="00-problem-solving-and-algorithms",
                preferred_language="python",
                title="Problem Solving and Algorithms (python)",
                description="Python-focused roadmap.",
                track="backend",
                levels=["junior", "mid", "senior"],
                sections=sections,
                raw_content="# Problem Solving and Algorithms (python)",
                target_sections=120,
                source="llm",
            )
            self.assertEqual(saved["preferred_language"], "python")

            detail = store.get_dynamic_topic_curriculum(
                user_id="alice",
                topic_id="00-problem-solving-and-algorithms",
                preferred_language="python",
            )
            self.assertIsNotNone(detail)
            assert detail is not None
            self.assertEqual(detail["title"], "Problem Solving and Algorithms (python)")
            self.assertEqual(len(detail["sections"]), 1)

            self.assertIsNone(
                store.get_dynamic_topic_curriculum(
                    user_id="alice",
                    topic_id="00-problem-solving-and-algorithms",
                    preferred_language="java",
                )
            )
            self.assertIsNone(
                store.get_dynamic_topic_curriculum(
                    user_id="bob",
                    topic_id="00-problem-solving-and-algorithms",
                    preferred_language="python",
                )
            )

            store.delete_dynamic_topic_curriculum(
                user_id="alice",
                topic_id="00-problem-solving-and-algorithms",
                preferred_language="python",
            )
            self.assertIsNone(
                store.get_dynamic_topic_curriculum(
                    user_id="alice",
                    topic_id="00-problem-solving-and-algorithms",
                    preferred_language="python",
                )
            )

            resolved = store.resolve_topic_ai_settings(
                user_id="alice",
                topic_id="00-problem-solving-and-algorithms",
                topic_detail={"id": "00-problem-solving-and-algorithms", "track": "backend"},
            )
            self.assertTrue(resolved["requires_programming"])
            self.assertEqual(resolved["preferred_language"], "python")

    def test_assistant_memory_upsert_and_isolation(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "learning.db")
            store = LearningStore(db_path)

            saved = store.upsert_assistant_memory(
                user_id="alice",
                conversation_id="conv-1",
                flow="chat",
                summary={"summary": "Previous chat context", "topic_id": "backend"},
            )
            self.assertEqual(saved["conversation_id"], "conv-1")

            loaded = store.get_assistant_memory(
                user_id="alice",
                conversation_id="conv-1",
                flow="chat",
            )
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded["summary"]["topic_id"], "backend")
            self.assertIsNone(
                store.get_assistant_memory(
                    user_id="bob",
                    conversation_id="conv-1",
                    flow="chat",
                )
            )


if __name__ == "__main__":
    unittest.main()
