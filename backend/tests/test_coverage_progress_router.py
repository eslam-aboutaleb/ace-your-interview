"""Coverage for ``app.routers.progress`` — uninitialised store and repair scheduling.

A real ``ProgressStore`` on a temp database keeps response models honest; the
scheduling tests drive ``schedule_pending_summary_refresh`` directly because its
"no running loop" contract is only observable off the event loop.
"""

import asyncio
import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.dependencies import require_auth
from app.routers import progress
from app.services.progress_summarizer import ProgressSummarizer

PAYLOAD = {
    "topic_title": "HTTP Fundamentals",
    "questions": [
        {
            "question_id": "q1",
            "question": "Explain keep-alive timeouts.",
            "difficulty": "medium",
            "revealed": True,
            "is_correct": True,
            "confidence": 4,
        }
    ],
    "sections": ["HTTP Basics"],
}


class FakeLLM:
    def __init__(self, text: str = "Learner covered keep-alive behaviour."):
        self.text = text
        self.calls = 0

    async def completion(self, prompt, llm_config=None, user_identity=None, task=None, **kwargs):
        self.calls += 1
        return {
            "success": True,
            "analysis": self.text,
            "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
            "error": "",
        }


class CountingSummarizer:
    def __init__(self):
        self.calls: list[dict] = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "text": "A generated summary.",
            "provider_used": "openai",
            "model_used": "gpt-4o-mini",
            "source": "ai",
        }


class ProgressRouterCoverageTestBase(unittest.TestCase):
    def setUp(self):
        self.prev = (progress._store, progress._summarizer, progress._llm_client)
        progress._summary_inflight.clear()
        progress._background_tasks.clear()

        self.tmpdir = tempfile.TemporaryDirectory()
        self.store = self._make_store()
        self.summarizer = CountingSummarizer()
        self.llm = FakeLLM()
        progress.init(self.store, self.summarizer, self.llm)  # type: ignore[arg-type]

        self.app = FastAPI()
        self.app.include_router(progress.router)
        self.app.dependency_overrides[require_auth] = lambda: {
            "user": "alice",
            "provider": "google",
        }
        self.client = TestClient(self.app)

    def tearDown(self):
        self.app.dependency_overrides.clear()
        progress._store, progress._summarizer, progress._llm_client = self.prev
        progress._summary_inflight.clear()
        progress._background_tasks.clear()
        self.tmpdir.cleanup()

    def _make_store(self):
        from app.services.progress_store import ProgressStore

        return ProgressStore(os.path.join(self.tmpdir.name, "learning.db"))


class StoreGuardTests(ProgressRouterCoverageTestBase):
    def test_every_route_reports_503_without_a_store(self):
        progress._store = None
        cases = [
            ("get", "/api/progress/topics", None),
            ("get", "/api/progress/01-http", None),
            ("post", "/api/progress/01-http/save", PAYLOAD),
            ("post", "/api/progress/01-http/autosave", PAYLOAD),
        ]
        for method, path, body in cases:
            with self.subTest(path=path):
                kwargs = {"json": body} if body is not None else {}
                response = getattr(self.client, method)(path, **kwargs)
                self.assertEqual(response.status_code, 503)
                self.assertEqual(
                    response.json()["detail"], "Progress store not initialised"
                )


class SummaryRefreshTests(ProgressRouterCoverageTestBase):
    def test_save_without_a_summarizer_leaves_the_document_untouched(self):
        progress._summarizer = None
        response = self.client.post("/api/progress/01-http/save", json=PAYLOAD)

        self.assertEqual(response.status_code, 200)
        # No summarizer means nothing generated a note; the row stays owed-free
        # rather than being marked failed.
        self.assertEqual(response.json()["summary_status"], "empty")
        self.assertEqual(response.json()["summary_text"], "")

    def test_save_with_a_summarizer_persists_the_generated_note(self):
        response = self.client.post("/api/progress/01-http/save", json=PAYLOAD)

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["summary_status"], "ready")
        self.assertEqual(payload["summary_text"], "A generated summary.")
        self.assertEqual(len(self.summarizer.calls), 1)

    def test_real_summarizer_writes_the_llm_note(self):
        progress._summarizer = ProgressSummarizer(self.llm)  # type: ignore[arg-type]
        response = self.client.post("/api/progress/01-http/save", json=PAYLOAD)

        self.assertEqual(response.json()["summary_text"], self.llm.text)
        self.assertEqual(self.llm.calls, 1)


class SchedulePendingSummaryTests(ProgressRouterCoverageTestBase):
    def test_no_store_means_nothing_is_scheduled(self):
        progress._store = None
        self.assertFalse(
            progress.schedule_pending_summary_refresh(
                user_id="alice", topic_id="01-http", user_identity={"user": "alice"}
            )
        )
        self.assertEqual(progress._summary_inflight, set())

    def test_no_summarizer_means_nothing_is_scheduled(self):
        progress._summarizer = None
        self.assertFalse(
            progress.schedule_pending_summary_refresh(
                user_id="alice", topic_id="01-http", user_identity={"user": "alice"}
            )
        )
        self.assertEqual(progress._summary_inflight, set())

    def test_a_duplicate_in_flight_request_is_rejected_without_scheduling(self):
        progress._summary_inflight.add(("alice", "01-http"))
        try:
            self.assertFalse(
                progress.schedule_pending_summary_refresh(
                    user_id="alice", topic_id="01-http", user_identity={"user": "alice"}
                )
            )
        finally:
            progress._summary_inflight.clear()
        self.assertEqual(progress._background_tasks, set())

    def test_the_in_flight_key_is_released_when_nothing_is_pending(self):
        async def scenario():
            scheduled = progress.schedule_pending_summary_refresh(
                user_id="alice",
                topic_id="missing-topic",
                user_identity={"user": "alice", "provider": "google"},
            )
            await asyncio.gather(*list(progress._background_tasks))
            return scheduled

        self.assertTrue(asyncio.run(scenario()))
        # Nothing was pending for a topic that was never saved, yet the guard key
        # must still be released or every later repair would be blocked.
        self.assertNotIn(("alice", "missing-topic"), progress._summary_inflight)

    def test_no_running_loop_is_not_an_error(self):
        def _refuse(coro, *args, **kwargs):
            # The caller already built the coroutine, so close it rather than
            # leaving a "never awaited" warning behind.
            coro.close()
            raise RuntimeError("no loop")

        with patch("asyncio.create_task", side_effect=_refuse):
            scheduled = progress.schedule_pending_summary_refresh(
                user_id="alice", topic_id="01-http", user_identity={"user": "alice"}
            )

        self.assertFalse(scheduled)
        # The key must not be leaked when scheduling fails.
        self.assertNotIn(("alice", "01-http"), progress._summary_inflight)

    def test_a_failing_repair_is_logged_and_still_releases_its_key(self):
        original = self.store.get_progress

        def _explode(**kwargs):
            raise RuntimeError("progress read failed")

        self.store.get_progress = _explode  # type: ignore[method-assign]
        try:
            async def scenario():
                self.assertTrue(progress.schedule_pending_summary_refresh(
                    user_id="alice", topic_id="01-http", user_identity={"user": "alice"}
                ))
                await asyncio.gather(*list(progress._background_tasks))

            with self.assertLogs("app.routers.progress", level="ERROR"):
                asyncio.run(scenario())
        finally:
            del self.store.get_progress
            self.assertIsNotNone(original)

        # A crash inside the repair must not wedge the in-flight guard.
        self.assertNotIn(("alice", "01-http"), progress._summary_inflight)

    def test_a_repair_actually_runs_on_the_event_loop(self):
        async def scenario():
            self.client.post("/api/progress/01-http/save", json=PAYLOAD)
            # Simulate a dropped autosave leaving the note owed.
            self.store.mark_summary_pending(user_id="alice", topic_id="01-http")
            self.assertTrue(progress.schedule_pending_summary_refresh(
                user_id="alice", topic_id="01-http", user_identity={"user": "alice"}
            ))
            await asyncio.gather(*list(progress._background_tasks))

        asyncio.run(scenario())
        document = self.store.get_progress(user_id="alice", topic_id="01-http")
        self.assertEqual(document["summary_status"], "ready")
        self.assertEqual(document["summary_text"], "A generated summary.")


if __name__ == "__main__":
    unittest.main()