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


if __name__ == "__main__":
    unittest.main()

