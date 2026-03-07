import asyncio
import os
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from app.services.learning_store import LearningStore
from app.services.video_recommender import VideoRecommender


def _settings(**overrides):
    base = {
        "enable_topic_videos": True,
        "youtube_api_key": "yt-key",
        "topic_videos_default_limit": 3,
        "topic_videos_cache_ttl_hours": 168,
        "topic_videos_disable_on_quota": True,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class _StubRecommender(VideoRecommender):
    def __init__(self, store, settings, result=None, error=None):
        super().__init__(store, settings)
        self.result = result if result is not None else []
        self.error = error
        self.fetch_calls = 0

    async def _fetch_videos(self, **kwargs):
        self.fetch_calls += 1
        return self.result, self.error


class VideoRecommenderTests(unittest.TestCase):
    def test_quota_disable_sets_next_pacific_midnight(self):
        with tempfile.TemporaryDirectory() as td:
            store = LearningStore(os.path.join(td, "learning.db"))
            recommender = _StubRecommender(store, _settings(), result=[], error="quota_exhausted")

            now_before = datetime.now(UTC)
            out = asyncio.run(
                recommender.recommend(
                    topic_id="01-static",
                    topic_title="Static",
                    section_heading="HTTP Basics",
                    preferred_language="",
                    limit=3,
                    force_refresh=True,
                )
            )
            self.assertEqual(out["status"], "disabled_quota_exhausted")
            disabled_until = datetime.fromisoformat(out["disabled_until"]).astimezone(
                ZoneInfo("America/Los_Angeles")
            )
            self.assertEqual(disabled_until.hour, 0)
            self.assertEqual(disabled_until.minute, 0)
            self.assertEqual(disabled_until.second, 0)
            self.assertGreater(disabled_until, now_before.astimezone(ZoneInfo("America/Los_Angeles")))

    def test_quota_error_disables_feature(self):
        with tempfile.TemporaryDirectory() as td:
            store = LearningStore(os.path.join(td, "learning.db"))
            recommender = _StubRecommender(store, _settings(), result=[], error="quota_exhausted")

            out = asyncio.run(
                recommender.recommend(
                    topic_id="01-static",
                    topic_title="Static",
                    section_heading="HTTP Basics",
                    preferred_language="",
                    limit=3,
                    force_refresh=True,
                )
            )
            self.assertFalse(out["enabled"])
            self.assertEqual(out["status"], "disabled_quota_exhausted")
            self.assertTrue(out["disabled_until"])

            status = recommender.get_status()
            self.assertFalse(status["enabled"])
            self.assertEqual(status["status"], "disabled_quota_exhausted")

    def test_disabled_state_blocks_provider_calls(self):
        with tempfile.TemporaryDirectory() as td:
            store = LearningStore(os.path.join(td, "learning.db"))
            future = (datetime.now(UTC) + timedelta(hours=3)).isoformat()
            store.set_feature_state(
                feature_key="topic_videos",
                status="disabled_quota_exhausted",
                disabled_until=future,
                reason="youtube_quota_exhausted",
            )
            recommender = _StubRecommender(store, _settings(), result=[], error=None)

            out = asyncio.run(
                recommender.recommend(
                    topic_id="01-static",
                    topic_title="Static",
                    section_heading="HTTP Basics",
                    preferred_language="",
                    limit=3,
                    force_refresh=False,
                )
            )
            self.assertFalse(out["enabled"])
            self.assertEqual(recommender.fetch_calls, 0)

    def test_expired_disable_enters_probe_and_reenables_on_success(self):
        with tempfile.TemporaryDirectory() as td:
            store = LearningStore(os.path.join(td, "learning.db"))
            past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
            store.set_feature_state(
                feature_key="topic_videos",
                status="disabled_quota_exhausted",
                disabled_until=past,
                reason="youtube_quota_exhausted",
            )
            recommender = _StubRecommender(
                store,
                _settings(),
                result=[
                    {
                        "video_id": "abc123",
                        "title": "HTTP Retry Patterns",
                        "url": "https://www.youtube.com/watch?v=abc123",
                        "channel": "Tech Channel",
                        "duration_seconds": 600,
                        "thumbnail_url": "",
                        "published_at": "",
                        "source": "youtube",
                    }
                ],
                error=None,
            )

            status_before = recommender.get_status()
            self.assertTrue(status_before["enabled"])
            self.assertEqual(status_before["status"], "probe")

            out = asyncio.run(
                recommender.recommend(
                    topic_id="01-static",
                    topic_title="Static",
                    section_heading="HTTP Basics",
                    preferred_language="",
                    limit=3,
                    force_refresh=True,
                )
            )
            self.assertTrue(out["enabled"])
            self.assertEqual(recommender.fetch_calls, 1)

            status_after = recommender.get_status()
            self.assertTrue(status_after["enabled"])
            self.assertEqual(status_after["status"], "enabled")

    def test_expired_disable_probe_quota_keeps_disabled(self):
        with tempfile.TemporaryDirectory() as td:
            store = LearningStore(os.path.join(td, "learning.db"))
            past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
            store.set_feature_state(
                feature_key="topic_videos",
                status="disabled_quota_exhausted",
                disabled_until=past,
                reason="youtube_quota_exhausted",
            )
            recommender = _StubRecommender(store, _settings(), result=[], error="quota_exhausted")
            out = asyncio.run(
                recommender.recommend(
                    topic_id="01-static",
                    topic_title="Static",
                    section_heading="HTTP Basics",
                    preferred_language="",
                    limit=3,
                    force_refresh=True,
                )
            )
            self.assertFalse(out["enabled"])
            self.assertEqual(out["status"], "disabled_quota_exhausted")
            self.assertEqual(recommender.fetch_calls, 1)


if __name__ == "__main__":
    unittest.main()
