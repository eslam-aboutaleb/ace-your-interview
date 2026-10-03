"""Stage 2.2: bounds on the learner-supplied strings in the planner layer."""

import os
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.dependencies import require_auth
from app.routers import learning
from app.services.learning_planner import LearningPlannerStore
from app.services.learning_store import LearningStore

#: Declared planner-layer ceilings. `LearnerProfileUpdateRequest` mirrors these
#: with `max_length` so a schema rejection and a store truncation agree.
MAX_TARGET_ROLE_CHARS = 200
MAX_FOCUS_TOPIC_CHARS = 120
MAX_FOCUS_TOPICS = 12


class LearningPlannerInputBoundsTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self.tempdir.name, "learning.db")
        self.store = LearningStore(db_path)
        self.planner = LearningPlannerStore(db_path)
        self.prev_store = learning._store
        self.prev_planner = learning._planner
        learning.init(self.store, self.planner)

        self.app = FastAPI()
        self.app.include_router(learning.router)
        self.app.dependency_overrides[require_auth] = lambda: {
            "user": "bounded-user",
            "provider": "local",
        }
        self.client = TestClient(self.app)

    def tearDown(self):
        learning._store = self.prev_store
        learning._planner = self.prev_planner
        self.tempdir.cleanup()

    def test_target_role_is_bounded_at_the_store(self):
        profile = self.planner.upsert_profile(
            user_id="bounded-user",
            payload={"target_role": "R" * 5000},
        )
        self.assertLessEqual(len(profile["target_role"]), MAX_TARGET_ROLE_CHARS)

    def test_focus_areas_are_bounded_in_count_and_length(self):
        profile = self.planner.upsert_profile(
            user_id="bounded-user",
            payload={
                "focus_topic_ids": [f"{'t' * 400}-{index}" for index in range(50)],
                "target_companies": [f"{'c' * 400}-{index}" for index in range(50)],
            },
        )
        self.assertLessEqual(len(profile["focus_topic_ids"]), MAX_FOCUS_TOPICS)
        self.assertLessEqual(len(profile["target_companies"]), MAX_FOCUS_TOPICS)
        for topic_id in profile["focus_topic_ids"]:
            self.assertLessEqual(len(topic_id), MAX_FOCUS_TOPIC_CHARS)
        for company in profile["target_companies"]:
            self.assertLessEqual(len(company), MAX_FOCUS_TOPIC_CHARS)

    def test_profile_update_route_rejects_oversized_payloads(self):
        # `LearnerProfileUpdateRequest` already declares max_length=200 on
        # target_role and max_length=12 on focus_topic_ids, so an oversized
        # request is rejected rather than silently truncated.
        res = self.client.put(
            "/api/learning/profile",
            json={
                "target_role": "S" * 5000,
                "focus_topic_ids": [f"{'f' * 400}-{index}" for index in range(30)],
            },
        )
        self.assertEqual(res.status_code, 422)
        locations = {tuple(item["loc"]) for item in res.json()["detail"]}
        self.assertIn(("body", "target_role"), locations)
        self.assertIn(("body", "focus_topic_ids"), locations)

    def test_profile_update_route_truncates_oversized_items(self):
        res = self.client.put(
            "/api/learning/profile",
            json={
                "target_role": "S" * 200,
                "focus_topic_ids": [f"{'f' * 400}-{index}" for index in range(12)],
            },
        )
        self.assertEqual(res.status_code, 200, res.text[:400])
        payload = res.json()
        self.assertLessEqual(len(payload["target_role"]), MAX_TARGET_ROLE_CHARS)
        self.assertLessEqual(len(payload["focus_topic_ids"]), MAX_FOCUS_TOPICS)
        for topic_id in payload["focus_topic_ids"]:
            self.assertLessEqual(len(topic_id), MAX_FOCUS_TOPIC_CHARS)

    def test_recommendation_titles_derived_from_stored_profile_stay_bounded(self):
        self.planner.upsert_profile(
            user_id="bounded-user",
            payload={"target_role": "T" * 500, "preferred_modalities": ["interview"]},
        )
        recs = self.planner.build_recommendations(user_id="bounded-user", limit=8)
        titles = [str(item["title"]) for item in recs["items"]]
        reasons = [str(item["reason"]) for item in recs["items"]]
        self.assertTrue(titles)
        for text in [*titles, *reasons]:
            self.assertLessEqual(len(text), MAX_TARGET_ROLE_CHARS + 120)

    def test_invalid_target_date_is_rejected_without_prompting(self):
        profile = self.planner.upsert_profile(
            user_id="bounded-user",
            payload={"target_role": "Backend", "target_date": "not-a-date"},
        )
        self.assertEqual(profile["target_date"], "")


if __name__ == "__main__":
    unittest.main()