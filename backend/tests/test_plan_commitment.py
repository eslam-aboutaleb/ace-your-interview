"""Plan commitment: persisted plan tasks and streak-aware plan load.

Covers the ``plan_tasks`` CRUD appended to
:class:`~app.services.learning_planner.LearningPlannerStore`, the
completion state embedded in ``build_study_plan``, the documented
streak-aware daily-item rule, and the ``/api/learning/plan-tasks``
endpoints.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import UTC, date, datetime, timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import require_auth
from app.routers import learning
from app.services.gamification_store import GamificationStore
from app.services.learning_planner import LearningPlannerStore, plan_task_key
from app.services.learning_store import LearningStore

ADAPTIVE_ENV = "STUDY_ENABLE_ADAPTIVE_LEARNING"
FSRS_ENV = "STUDY_ENABLE_FSRS_V1"


class _PlannerFixture:
    """Learning + planner + gamification stores on one temp database."""

    def __enter__(self) -> "_PlannerFixture":
        self._td = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._td.name, "learning.db")
        self.learning = LearningStore(self.db_path)
        self.planner = LearningPlannerStore(self.db_path)
        self.gamification = GamificationStore(self.db_path)
        return self

    def __exit__(self, *exc_info) -> bool:
        self._td.cleanup()
        return False

    def seed_learning_evidence(self, *, user_id: str = "alice", topics: int = 3) -> None:
        self.planner.upsert_profile(
            user_id=user_id,
            payload={
                "primary_track": "backend",
                "target_role": "Backend Engineer",
                "preferred_modalities": ["study", "quiz", "interview"],
            },
        )
        for idx in range(topics):
            self.learning.record_attempt(
                user_id=user_id,
                question_id=f"q{idx}",
                topic_id=f"01-backend-topic-{idx}",
                user_answer="wrong",
                is_correct=False,
                confidence=4,
                response_time_ms=900,
                mode="quiz",
            )

    def set_last_activity(self, *, user_id: str, day: date, streak: int = 3) -> None:
        """Force the persisted streak anchor without logging an event.

        ``GamificationStore.award_xp`` stamps ``last_activity_date``
        with *today* and writes a counted learning event, so the
        anchor is written straight into ``gamification_state`` to
        simulate a past (or eventless) day.
        """
        self.gamification._conn.execute(
            """
            INSERT INTO gamification_state(
                user_id, total_xp, current_streak, longest_streak,
                last_activity_date, daily_goal, badges_json, updated_at
            ) VALUES (?, 0, ?, ?, ?, 3, '[]', ?)
            ON CONFLICT(user_id) DO UPDATE SET
                current_streak = excluded.current_streak,
                longest_streak = excluded.longest_streak,
                last_activity_date = excluded.last_activity_date
            """,
            (
                user_id,
                streak,
                streak,
                day.isoformat(),
                datetime.now(UTC).isoformat(),
            ),
        )
        self.gamification._conn.commit()

    def today(self) -> date:
        return datetime.now(UTC).date()


class PlanTaskKeyTests(unittest.TestCase):
    def test_key_is_deterministic_for_the_same_slot(self):
        first = plan_task_key(
            task_date="2026-10-11", task_type="topic_study", topic_id="01-backend"
        )
        second = plan_task_key(
            task_date="2026-10-11", task_type="topic_study", topic_id="01-backend"
        )
        self.assertEqual(first, second)
        self.assertTrue(first)

    def test_key_changes_with_date_type_topic_and_occurrence(self):
        base = dict(task_date="2026-10-11", task_type="topic_study", topic_id="01-backend")
        variants = {
            plan_task_key(**base),
            plan_task_key(**{**base, "task_date": "2026-10-12"}),
            plan_task_key(**{**base, "task_type": "quiz"}),
            plan_task_key(**{**base, "topic_id": "02-backend"}),
            plan_task_key(**{**base, "occurrence": 2}),
        }
        self.assertEqual(len(variants), 5)


class PlanTaskPersistenceTests(unittest.TestCase):
    def test_build_study_plan_persists_every_scheduled_task(self):
        with _PlannerFixture() as fx:
            fx.seed_learning_evidence()
            plan = fx.planner.build_study_plan(user_id="alice", days=3, daily_items=3)
            today = plan["days_plan"][0]["date"]
            stored = fx.planner.list_plan_tasks(user_id="alice", task_date=today)
            self.assertEqual(len(stored), len(plan["days_plan"][0]["tasks"]))
            self.assertTrue(
                all(task["task_key"] for task in plan["days_plan"][0]["tasks"])
            )

    def test_completed_task_survives_regeneration(self):
        with _PlannerFixture() as fx:
            fx.seed_learning_evidence()
            plan = fx.planner.build_study_plan(user_id="alice", days=3, daily_items=3)
            today = plan["days_plan"][0]["date"]
            task = plan["days_plan"][0]["tasks"][0]

            completed = fx.planner.complete_plan_task(
                user_id="alice", task_key=task["task_key"], task_date=today
            )
            self.assertIsNotNone(completed)
            self.assertTrue(completed["completed"])
            self.assertTrue(completed["completed_at"])

            regenerated = fx.planner.build_study_plan(
                user_id="alice", days=3, daily_items=3
            )
            flags = [t["completed"] for t in regenerated["days_plan"][0]["tasks"]]
            self.assertEqual(flags[0], True)
            # Only the completed slot flips, and keys stay stable.
            self.assertEqual(
                [t["task_key"] for t in regenerated["days_plan"][0]["tasks"]],
                [t["task_key"] for t in plan["days_plan"][0]["tasks"]],
            )
            self.assertEqual(flags.count(True), 1)
            commitment = regenerated["commitment"]
            self.assertEqual(commitment["completed_tasks"], 1)
            self.assertEqual(
                commitment["completed_tasks"] + commitment["pending_tasks"],
                regenerated["total_tasks"],
            )

    def test_repeated_topic_less_tasks_get_distinct_keys(self):
        with _PlannerFixture() as fx:
            # A fresh planner user gets the topic-less interview
            # recommendation, which fills every slot of the day.
            plan = fx.planner.build_study_plan(user_id="alice", days=1, daily_items=3)
            keys = [task["task_key"] for task in plan["days_plan"][0]["tasks"]]
            self.assertEqual(len(keys), 3)
            self.assertEqual(len(set(keys)), 3)

    def test_complete_is_idempotent_and_uncomplete_reverts(self):
        with _PlannerFixture() as fx:
            fx.seed_learning_evidence()
            plan = fx.planner.build_study_plan(user_id="alice", days=1, daily_items=2)
            today = plan["days_plan"][0]["date"]
            key = plan["days_plan"][0]["tasks"][0]["task_key"]

            first = fx.planner.complete_plan_task(
                user_id="alice", task_key=key, task_date=today
            )
            second = fx.planner.complete_plan_task(
                user_id="alice", task_key=key, task_date=today
            )
            self.assertEqual(first["completed_at"], second["completed_at"])

            reverted = fx.planner.uncomplete_plan_task(
                user_id="alice", task_key=key, task_date=today
            )
            self.assertIsNotNone(reverted)
            self.assertFalse(reverted["completed"])
            self.assertIsNone(reverted["completed_at"])

    def test_unknown_task_key_returns_none(self):
        with _PlannerFixture() as fx:
            fx.seed_learning_evidence()
            fx.planner.build_study_plan(user_id="alice", days=1, daily_items=2)
            self.assertIsNone(
                fx.planner.complete_plan_task(
                    user_id="alice", task_key="missing", task_date="2026-10-11"
                )
            )
            self.assertIsNone(
                fx.planner.uncomplete_plan_task(
                    user_id="alice", task_key="missing", task_date="2026-10-11"
                )
            )

    def test_completion_is_scoped_to_user_and_date(self):
        with _PlannerFixture() as fx:
            fx.seed_learning_evidence()
            plan = fx.planner.build_study_plan(user_id="alice", days=2, daily_items=2)
            key = plan["days_plan"][0]["tasks"][0]["task_key"]
            fx.planner.complete_plan_task(
                user_id="alice", task_key=key, task_date=plan["days_plan"][0]["date"]
            )
            # A different user never sees alice's completion, and the same
            # task scheduled on another day is independent.
            self.assertIsNone(
                fx.planner.complete_plan_task(
                    user_id="bob", task_key=key, task_date=plan["days_plan"][0]["date"]
                )
            )
            self.assertIsNone(
                fx.planner.complete_plan_task(
                    user_id="alice", task_key=key, task_date=plan["days_plan"][1]["date"]
                )
            )

    def test_empty_plan_still_reports_commitment(self):
        with _PlannerFixture() as fx:
            fx.planner.build_recommendations = lambda **kwargs: {"items": []}
            plan = fx.planner.build_study_plan(user_id="alice", days=2, daily_items=3)
            self.assertEqual(plan["total_tasks"], 0)
            self.assertEqual([day["tasks"] for day in plan["days_plan"]], [[], []])
            self.assertEqual(plan["commitment"]["pending_tasks"], 0)
            self.assertIn("daily_goal", plan["commitment"])


class StreakAwarePlanLoadTests(unittest.TestCase):
    def test_new_user_keeps_the_full_daily_load(self):
        with _PlannerFixture() as fx:
            fx.seed_learning_evidence()
            plan = fx.planner.build_study_plan(user_id="alice", days=2, daily_items=3)
            commitment = plan["commitment"]
            self.assertFalse(commitment["behind"])
            self.assertFalse(commitment["streak_at_risk"])
            self.assertFalse(commitment["streak_lapsed"])
            self.assertEqual(plan["daily_items"], 3)
            self.assertEqual(commitment["effective_daily_items"], 3)
            self.assertEqual(commitment["requested_daily_items"], 3)
            self.assertTrue(commitment["catch_up"])

    def test_missing_today_halves_the_daily_load(self):
        with _PlannerFixture() as fx:
            fx.seed_learning_evidence()
            fx.set_last_activity(user_id="alice", day=fx.today() - timedelta(days=1))
            plan = fx.planner.build_study_plan(user_id="alice", days=2, daily_items=3)
            commitment = plan["commitment"]
            self.assertTrue(commitment["streak_at_risk"])
            self.assertTrue(commitment["behind"])
            self.assertEqual(commitment["effective_daily_items"], 2)
            self.assertEqual(plan["daily_items"], 2)
            self.assertEqual(commitment["current_streak"], 3)
            self.assertEqual(len(plan["days_plan"][0]["tasks"]), 2)

    def test_lapsed_streak_with_history_halves_the_daily_load(self):
        with _PlannerFixture() as fx:
            fx.seed_learning_evidence()
            fx.set_last_activity(user_id="alice", day=fx.today() - timedelta(days=4))
            plan = fx.planner.build_study_plan(user_id="alice", days=2, daily_items=4)
            commitment = plan["commitment"]
            self.assertTrue(commitment["streak_lapsed"])
            self.assertTrue(commitment["behind"])
            self.assertEqual(commitment["effective_daily_items"], 2)

    def test_activity_today_keeps_the_full_daily_load(self):
        with _PlannerFixture() as fx:
            fx.seed_learning_evidence()
            fx.set_last_activity(user_id="alice", day=fx.today(), streak=5)
            fx.gamification.award_xp(
                user_id="alice",
                event_key="today:1",
                event_type="attempt",
                xp=10,
                timezone="UTC",
            )
            fx.gamification.award_xp(
                user_id="alice",
                event_key="today:2",
                event_type="review",
                xp=5,
                timezone="UTC",
            )
            plan = fx.planner.build_study_plan(user_id="alice", days=2, daily_items=3)
            commitment = plan["commitment"]
            self.assertFalse(commitment["streak_at_risk"])
            self.assertFalse(commitment["behind"])
            self.assertEqual(commitment["today_progress"], 2)
            self.assertEqual(commitment["effective_daily_items"], 3)
            self.assertTrue(commitment["catch_up"])  # 2 of 3 goal events

    def test_meeting_the_daily_goal_clears_catch_up(self):
        with _PlannerFixture() as fx:
            fx.seed_learning_evidence()
            fx.set_last_activity(user_id="alice", day=fx.today(), streak=5)
            for idx in range(3):
                fx.gamification.award_xp(
                    user_id="alice",
                    event_key=f"goal:{idx}",
                    event_type="attempt",
                    xp=10,
                    timezone="UTC",
                )
            plan = fx.planner.build_study_plan(user_id="alice", days=2, daily_items=3)
            commitment = plan["commitment"]
            self.assertEqual(commitment["today_progress"], 3)
            self.assertEqual(commitment["daily_goal"], 3)
            self.assertFalse(commitment["catch_up"])

    def test_stale_anchor_without_history_is_not_behind(self):
        with _PlannerFixture() as fx:
            fx.seed_learning_evidence()
            # A streak anchor with no learning events behind it: the
            # learner has no history to protect, so the load is not cut.
            fx.gamification._conn.execute(
                """
                INSERT INTO gamification_state(
                    user_id, total_xp, current_streak, longest_streak,
                    last_activity_date, daily_goal, badges_json, updated_at
                ) VALUES ('alice', 0, 1, 0, ?, 3, '[]', ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    last_activity_date = excluded.last_activity_date,
                    current_streak = excluded.current_streak,
                    longest_streak = 0
                """,
                (
                    (fx.today() - timedelta(days=3)).isoformat(),
                    datetime.now(UTC).isoformat(),
                ),
            )
            fx.gamification._conn.commit()
            plan = fx.planner.build_study_plan(user_id="alice", days=2, daily_items=3)
            self.assertFalse(plan["commitment"]["streak_lapsed"])
            self.assertFalse(plan["commitment"]["behind"])
            self.assertEqual(plan["commitment"]["effective_daily_items"], 3)

    def test_halving_rounds_up_and_never_drops_below_one(self):
        with _PlannerFixture() as fx:
            fx.seed_learning_evidence()
            fx.set_last_activity(user_id="alice", day=fx.today() - timedelta(days=2))
            for requested, expected in ((1, 1), (2, 1), (3, 2), (10, 5)):
                plan = fx.planner.build_study_plan(
                    user_id="alice", days=2, daily_items=requested
                )
                self.assertEqual(
                    plan["commitment"]["effective_daily_items"],
                    expected,
                    msg=f"requested={requested}",
                )


class PlanTaskRouterTestBase(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._td.name, "learning.db")
        self.store = LearningStore(self.db_path)
        self.planner = LearningPlannerStore(self.db_path)
        self.gamification = GamificationStore(self.db_path)
        learning.init(self.store, self.planner)
        self._prev_globals = (learning._store, learning._planner)
        self.prev_env = {
            key: os.environ.get(key) for key in (ADAPTIVE_ENV, FSRS_ENV)
        }
        os.environ[ADAPTIVE_ENV] = "true"
        os.environ[FSRS_ENV] = "true"
        get_settings.cache_clear()

        self.app = FastAPI()
        self.app.include_router(learning.router)
        self.app.dependency_overrides[require_auth] = lambda: {
            "user": "learner",
            "provider": "google",
        }
        self.client = TestClient(self.app)

    def tearDown(self):
        self.app.dependency_overrides.clear()
        learning._store, learning._planner = self._prev_globals
        self._td.cleanup()
        for key, value in self.prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()

    def _set_adaptive(self, enabled: bool) -> None:
        os.environ[ADAPTIVE_ENV] = "true" if enabled else "false"
        get_settings.cache_clear()

    def _seed_plan(self) -> tuple[dict, str]:
        self.planner.upsert_profile(
            user_id="learner",
            payload={
                "primary_track": "backend",
                "target_role": "Backend Engineer",
                "preferred_modalities": ["study", "quiz", "interview"],
            },
        )
        for idx in range(3):
            self.store.record_attempt(
                user_id="learner",
                question_id=f"q{idx}",
                topic_id=f"01-backend-topic-{idx}",
                user_answer="wrong",
                is_correct=False,
                confidence=4,
                response_time_ms=900,
                mode="quiz",
            )
        plan = self.client.get("/api/learning/study-plan?days=3&daily_items=3").json()
        return plan, plan["days_plan"][0]["date"]


class PlanTaskRouterTests(PlanTaskRouterTestBase):
    def test_complete_and_uncomplete_round_trip(self):
        plan, today = self._seed_plan()
        task = plan["days_plan"][0]["tasks"][0]
        self.assertFalse(task["completed"])
        self.assertEqual(task["task_key"], plan["days_plan"][0]["tasks"][0]["task_key"])

        response = self.client.post(
            f"/api/learning/plan-tasks/{task['task_key']}/complete",
            params={"task_date": today},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["completed"])
        self.assertEqual(response.json()["task_key"], task["task_key"])

        listed = self.client.get(
            "/api/learning/plan-tasks", params={"task_date": today}
        )
        self.assertEqual(listed.status_code, 200)
        payload = listed.json()
        self.assertEqual(payload["total"], len(plan["days_plan"][0]["tasks"]))
        completed_keys = {i["task_key"] for i in payload["items"] if i["completed"]}
        self.assertEqual(completed_keys, {task["task_key"]})

        reverted = self.client.post(
            f"/api/learning/plan-tasks/{task['task_key']}/uncomplete",
            params={"task_date": today},
        )
        self.assertEqual(reverted.status_code, 200)
        self.assertFalse(reverted.json()["completed"])

    def test_study_plan_reflects_the_persisted_completion(self):
        plan, today = self._seed_plan()
        task = plan["days_plan"][0]["tasks"][0]
        self.client.post(
            f"/api/learning/plan-tasks/{task['task_key']}/complete",
            params={"task_date": today},
        )
        regenerated = self.client.get(
            "/api/learning/study-plan?days=3&daily_items=3"
        ).json()
        flags = [t["completed"] for t in regenerated["days_plan"][0]["tasks"]]
        self.assertEqual(flags.count(True), 1)
        self.assertEqual(regenerated["commitment"]["completed_tasks"], 1)

    def test_completing_an_unknown_task_is_404(self):
        _, today = self._seed_plan()
        response = self.client.post(
            "/api/learning/plan-tasks/does-not-exist/complete",
            params={"task_date": today},
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Plan task not found")

    def test_malformed_task_date_is_rejected(self):
        _, _ = self._seed_plan()
        listed = self.client.get(
            "/api/learning/plan-tasks", params={"task_date": "11/10/2026"}
        )
        self.assertEqual(listed.status_code, 422)
        completed = self.client.post(
            "/api/learning/plan-tasks/abc/complete",
            params={"task_date": "not-a-date"},
        )
        self.assertEqual(completed.status_code, 422)

    def test_endpoints_respect_the_adaptive_learning_flag(self):
        self._set_adaptive(False)
        for method, path in (
            ("get", "/api/learning/plan-tasks?task_date=2026-10-11"),
            ("post", "/api/learning/plan-tasks/abc/complete?task_date=2026-10-11"),
            ("post", "/api/learning/plan-tasks/abc/uncomplete?task_date=2026-10-11"),
        ):
            with self.subTest(path=path):
                response = getattr(self.client, method)(path)
                self.assertEqual(response.status_code, 404)
                self.assertEqual(
                    response.json()["detail"], "Adaptive learning disabled"
                )

    def test_planner_backed_endpoints_report_503_without_a_planner(self):
        learning._planner = None
        for method, path in (
            ("get", "/api/learning/plan-tasks?task_date=2026-10-11"),
            ("post", "/api/learning/plan-tasks/abc/complete?task_date=2026-10-11"),
            ("post", "/api/learning/plan-tasks/abc/uncomplete?task_date=2026-10-11"),
        ):
            with self.subTest(path=path):
                response = getattr(self.client, method)(path)
                self.assertEqual(response.status_code, 503)
                self.assertEqual(
                    response.json()["detail"], "Learning planner not initialised"
                )

    def test_plan_tasks_requires_authentication(self):
        plain_app = FastAPI()
        plain_app.include_router(learning.router)
        client = TestClient(plain_app)
        self.assertEqual(
            client.get("/api/learning/plan-tasks?task_date=2026-10-11").status_code,
            401,
        )
        self.assertEqual(
            client.post(
                "/api/learning/plan-tasks/abc/complete?task_date=2026-10-11"
            ).status_code,
            401,
        )
        self.assertEqual(
            client.post(
                "/api/learning/plan-tasks/abc/uncomplete?task_date=2026-10-11"
            ).status_code,
            401,
        )


if __name__ == "__main__":
    unittest.main()
