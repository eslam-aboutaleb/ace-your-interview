import os
import tempfile
import unittest

from app.services.learning_store import LearningStore


class LearningStoreStudyPlanTests(unittest.TestCase):
    def test_empty_dataset_returns_empty_plan_days(self):
        with tempfile.TemporaryDirectory() as td:
            store = LearningStore(os.path.join(td, "learning.db"))
            plan = store.build_study_plan(user_id="alice", days=7, daily_items=3)
            self.assertEqual(plan["days"], 7)
            self.assertEqual(plan["daily_items"], 3)
            self.assertEqual(plan["total_tasks"], 0)
            self.assertEqual(len(plan["days_plan"]), 7)
            self.assertTrue(all(len(day["tasks"]) == 0 for day in plan["days_plan"]))

    def test_plan_is_deterministic_for_same_state(self):
        with tempfile.TemporaryDirectory() as td:
            store = LearningStore(os.path.join(td, "learning.db"))
            store.record_attempt(
                user_id="alice",
                question_id="q1",
                topic_id="topic-a",
                user_answer="wrong",
                is_correct=False,
                confidence=5,
                response_time_ms=1000,
                mode="quiz",
            )
            store.record_attempt(
                user_id="alice",
                question_id="q2",
                topic_id="topic-b",
                user_answer="right",
                is_correct=True,
                confidence=3,
                response_time_ms=1200,
                mode="study",
            )

            first = store.build_study_plan(user_id="alice", days=7, daily_items=3)
            second = store.build_study_plan(user_id="alice", days=7, daily_items=3)
            first_tasks = [[task["cta_route"] for task in day["tasks"]] for day in first["days_plan"]]
            second_tasks = [[task["cta_route"] for task in day["tasks"]] for day in second["days_plan"]]
            self.assertEqual(first_tasks, second_tasks)

    def test_plan_tasks_have_valid_type_and_routes(self):
        with tempfile.TemporaryDirectory() as td:
            store = LearningStore(os.path.join(td, "learning.db"))
            for idx in range(1, 5):
                store.record_attempt(
                    user_id="bob",
                    question_id=f"q{idx}",
                    topic_id=f"topic-{idx}",
                    user_answer="wrong" if idx % 2 else "right",
                    is_correct=idx % 2 == 0,
                    confidence=4 if idx % 2 else 3,
                    response_time_ms=800 + idx,
                    mode="study",
                )

            plan = store.build_study_plan(user_id="bob", days=7, daily_items=3)
            self.assertLessEqual(plan["total_tasks"], 21)
            valid_types = {"review", "topic_study", "quiz"}
            for day in plan["days_plan"]:
                self.assertLessEqual(len(day["tasks"]), 3)
                for task in day["tasks"]:
                    self.assertIn(task["task_type"], valid_types)
                    self.assertTrue(task["cta_route"].startswith("/topics/") or task["cta_route"].startswith("/quiz/"))


if __name__ == "__main__":
    unittest.main()
