import os
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.dependencies import require_auth
from app.routers import topics
from app.services.doc_parser import DocParser
from app.services.learning_store import LearningStore


class FakeLLM:
    async def completion(self, prompt, llm_config=None, user_identity=None, task=None, **kwargs):
        return {
            "success": True,
            "analysis": "{}",
            "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
            "error": "",
        }


class FakeVideoRecommender:
    def __init__(self):
        self.status = {
            "enabled": True,
            "status": "enabled",
            "disabled_until": "",
            "reason": "",
        }
        self.response = {
            "enabled": True,
            "status": "enabled",
            "disabled_until": "",
            "reason": "",
            "videos": [
                {
                    "video_id": "abc123",
                    "title": "HTTP Idempotency Interview Prep",
                    "url": "https://www.youtube.com/watch?v=abc123",
                    "channel": "System Design Hub",
                    "duration_seconds": 720,
                    "thumbnail_url": "",
                    "published_at": "",
                    "source": "youtube",
                }
            ],
            "source": "youtube",
            "cached": False,
        }

    def get_status(self):
        return dict(self.status)

    async def get_status_async(self):
        return self.get_status()

    async def recommend(self, **kwargs):
        return dict(self.response)


class TopicsRouterTopicVideosTests(unittest.TestCase):
    def setUp(self):
        self.prev_parser = topics._parser
        self.prev_llm = topics._llm_client
        self.prev_store = topics._learning_store
        self.prev_mcp = topics._mcp_gateway
        self.prev_video = topics._video_recommender

        self.tempdir = tempfile.TemporaryDirectory()
        handbook = os.path.join(self.tempdir.name, "owner-handbook")
        os.makedirs(handbook, exist_ok=True)
        with open(os.path.join(handbook, "01-static.md"), "w", encoding="utf-8") as f:
            f.write(
                """---
track: backend
levels: [junior, mid, senior]
---
# Static Topic

Static description.

## HTTP Basics
Focus on contracts and idempotency.
"""
            )

        parser = DocParser(docs_path=self.tempdir.name)
        store = LearningStore(os.path.join(self.tempdir.name, "learning.db"))
        self.video = FakeVideoRecommender()
        topics.init(parser, FakeLLM(), store, video_recommender=self.video)

        app = FastAPI()
        app.dependency_overrides[require_auth] = (
            lambda: {"user": "alice", "provider": "local", "is_admin": True}
        )
        app.include_router(topics.router)
        app.include_router(topics.features_router)
        self.client = TestClient(app)

    def tearDown(self):
        topics._parser = self.prev_parser
        topics._llm_client = self.prev_llm
        topics._learning_store = self.prev_store
        topics._mcp_gateway = self.prev_mcp
        topics._video_recommender = self.prev_video
        self.tempdir.cleanup()

    def test_topic_video_status_endpoint(self):
        res = self.client.get("/api/features/topic-videos/status")
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertTrue(payload["enabled"])
        self.assertEqual(payload["status"], "enabled")

    def test_topic_section_videos_endpoint(self):
        res = self.client.get("/api/topics/01-static/videos?section_index=0")
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertTrue(payload["enabled"])
        self.assertEqual(payload["topic_id"], "01-static")
        self.assertEqual(payload["section_index"], 0)
        self.assertEqual(len(payload["videos"]), 1)
        self.assertEqual(payload["videos"][0]["video_id"], "abc123")

    def test_topic_section_videos_hidden_when_disabled(self):
        self.video.response = {
            "enabled": False,
            "status": "disabled_quota_exhausted",
            "disabled_until": "2026-03-08T08:00:00+00:00",
            "reason": "youtube_quota_exhausted",
            "videos": [],
            "source": "youtube",
            "cached": False,
        }
        res = self.client.get("/api/topics/01-static/videos?section_index=0")
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertFalse(payload["enabled"])
        self.assertEqual(payload["status"], "disabled_quota_exhausted")
        self.assertEqual(payload["videos"], [])

    def test_topic_section_videos_rejects_out_of_range_section(self):
        res = self.client.get("/api/topics/01-static/videos?section_index=50")
        self.assertEqual(res.status_code, 422)


if __name__ == "__main__":
    unittest.main()
