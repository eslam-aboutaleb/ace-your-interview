"""STAR story bank tests — CRUD and Jaccard suggestion ranking."""

import os
import tempfile
import unittest

from app.services.star_store import StarStore, StoryNotFoundError


class StarStoreTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tempdir.name, "learning.db")
        self.store = StarStore(self.db_path)

    def tearDown(self):
        self.tempdir.cleanup()

    @staticmethod
    def _story(**overrides) -> dict:
        base = {
            "title": "Scaled a caching layer",
            "situation": "Our API p99 latency doubled during peak traffic.",
            "task": "Reduce latency without adding servers.",
            "action": "I introduced a Redis cache and measured hit rates.",
            "result": "p99 dropped 40% and costs stayed flat.",
            "tags": ["redis", "performance"],
        }
        base.update(overrides)
        return base

    def test_create_and_get(self):
        saved = self.store.create_story("u1", **self._story())
        self.assertTrue(saved["story_id"].startswith("star_"))
        loaded = self.store.get_story("u1", saved["story_id"])
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["title"], "Scaled a caching layer")
        self.assertEqual(loaded["tags"], ["redis", "performance"])
        self.assertEqual(loaded["result"], "p99 dropped 40% and costs stayed flat.")

    def test_stories_are_user_scoped(self):
        saved = self.store.create_story("u1", **self._story())
        self.assertIsNone(self.store.get_story("u2", saved["story_id"]))
        self.assertEqual(self.store.list_stories("u2"), [])

    def test_list_orders_by_recent_update(self):
        first = self.store.create_story("u1", **self._story(title="First story"))
        self.store.create_story("u1", **self._story(title="Second story"))
        listed = self.store.list_stories("u1")
        self.assertEqual([s["title"] for s in listed], ["Second story", "First story"])

        # Touching the first story moves it to the front.
        self.store.update_story("u1", first["story_id"], title="First story (touched)")
        listed = self.store.list_stories("u1")
        self.assertEqual(listed[0]["title"], "First story (touched)")

    def test_update_is_partial(self):
        saved = self.store.create_story("u1", **self._story())
        updated = self.store.update_story(
            "u1",
            saved["story_id"],
            title="Scaled a CDN layer",
            tags=["cdn"],
        )
        self.assertEqual(updated["title"], "Scaled a CDN layer")
        self.assertEqual(updated["tags"], ["cdn"])
        # Untouched fields keep their values.
        self.assertEqual(updated["situation"], self._story()["situation"])

    def test_update_missing_story_raises(self):
        with self.assertRaises(StoryNotFoundError):
            self.store.update_story("u1", "star_missing", title="Nope")

    def test_delete(self):
        saved = self.store.create_story("u1", **self._story())
        self.assertTrue(self.store.delete_story("u1", saved["story_id"]))
        self.assertIsNone(self.store.get_story("u1", saved["story_id"]))
        self.assertFalse(self.store.delete_story("u1", saved["story_id"]))

    def test_suggest_ranks_relevant_stories_first(self):
        self.store.create_story(
            "u1",
            **self._story(
                title="Redis caching win",
                situation="Latency spiked on the API gateway.",
                task="Cut p99 latency.",
                action="Added a Redis cache in front of the database.",
                result="p99 dropped 40 percent.",
                tags=["redis", "caching", "performance"],
            )
        )
        self.store.create_story(
            "u1",
            **self._story(
                title="Led a migration",
                situation="The monolith needed splitting.",
                task="Plan the microservices migration.",
                action="I drew the service boundaries and sequenced the rollout.",
                result="Teams shipped independently after the split.",
                tags=["microservices", "leadership"],
            )
        )
        suggestions = self.store.suggest_stories(
            "u1",
            "How did you use Redis to improve performance and caching?",
        )
        self.assertEqual(len(suggestions), 2)
        self.assertEqual(suggestions[0]["title"], "Redis caching win")

    def test_suggest_limits_results(self):
        for index in range(5):
            self.store.create_story(
                "u1",
                **self._story(title=f"Story {index}"),
            )
        suggestions = self.store.suggest_stories("u1", "story", limit=3)
        self.assertLessEqual(len(suggestions), 3)

    def test_suggest_with_no_overlap_falls_back_to_recent(self):
        self.store.create_story("u1", **self._story(title="Unrelated story"))
        suggestions = self.store.suggest_stories(
            "u1",
            "completely different topic with no shared tokens",
        )
        # No lexical overlap: the most recently updated story is returned.
        self.assertEqual(len(suggestions), 1)
        self.assertEqual(suggestions[0]["title"], "Unrelated story")

    def test_suggest_for_user_without_stories(self):
        self.assertEqual(self.store.suggest_stories("u1", "anything"), [])


if __name__ == "__main__":
    unittest.main()
