import asyncio
import os
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.dependencies import require_auth
from app.routers import progress
from app.services.progress_summarizer import ProgressSummarizer
from app.services.progress_store import ProgressStore


class FakeLLM:
    def __init__(self, text: str = "Learner covered the fundamentals."):
        self.text = text
        self.calls = 0

    async def completion(self, prompt, llm_config=None, user_identity=None, task=None):
        self.calls += 1
        return {
            "success": True,
            "analysis": self.text,
            "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
            "error": "",
        }


class RaisingSummarizer:
    async def generate(self, **kwargs):
        raise RuntimeError("summarizer exploded")


class ProgressRouterTests(unittest.TestCase):
    def setUp(self):
        self.prev_store = progress._store
        self.prev_summarizer = progress._summarizer
        self.prev_llm = progress._llm_client
        progress._summary_inflight.clear()
        progress._background_tasks.clear()

        self.tempdir = tempfile.TemporaryDirectory()
        self.store = ProgressStore(os.path.join(self.tempdir.name, "learning.db"))
        self.fake_llm = FakeLLM()
        progress.init(self.store, ProgressSummarizer(self.fake_llm), self.fake_llm)

        self.app = FastAPI()
        self.app.dependency_overrides[require_auth] = (
            lambda: {"user": "alice", "provider": "google"}
        )
        self.app.include_router(progress.router)
        self.client = TestClient(self.app)

        self.payload = {
            "topic_title": "Backend Fundamentals",
            "questions": [
                {
                    "question_id": "q1",
                    "question": "Explain keep-alive timeouts.",
                    "difficulty": "medium",
                    "revealed": True,
                    "is_correct": False,
                    "confidence": 2,
                },
                {
                    "question_id": "q2",
                    "question": "Explain backpressure in streams.",
                    "difficulty": "hard",
                    "revealed": False,
                    "is_correct": None,
                    "confidence": 0,
                },
            ],
            "sections": ["HTTP basics"],
            "preferred_language": "python",
        }

    def tearDown(self):
        progress._store = self.prev_store
        progress._summarizer = self.prev_summarizer
        progress._llm_client = self.prev_llm
        progress._summary_inflight.clear()
        progress._background_tasks.clear()
        self.tempdir.cleanup()

    def test_save_writes_raw_data_then_awaited_summary(self):
        response = self.client.post("/api/progress/topic-a/save", json=self.payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["topic_id"], "topic-a")
        self.assertEqual(body["topic_title"], "Backend Fundamentals")
        self.assertEqual(body["summary_status"], "ready")
        self.assertEqual(body["summary_text"], "Learner covered the fundamentals.")
        self.assertEqual(body["generation_source"], "ai")
        self.assertEqual(body["provider_used"], "openai")
        self.assertEqual(body["sections"], ["HTTP basics"])
        self.assertEqual(
            body["questions_asked"],
            ["Explain backpressure in streams.", "Explain keep-alive timeouts."],
        )
        self.assertEqual(body["attempt_stats"]["submitted"], 1)
        self.assertEqual(body["attempt_stats"]["correct"], 0)
        self.assertEqual(body["attempt_stats"]["incorrect"], 1)
        self.assertEqual(body["attempt_stats"]["total"], 2)
        self.assertEqual(body["revision"], 1)

    def test_autosave_marks_pending_then_ready(self):
        response = self.client.post("/api/progress/topic-a/autosave", json=self.payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["summary_status"], "ready")
        self.assertEqual(body["summary_text"], "Learner covered the fundamentals.")

    def test_second_save_unions_history_and_bumps_revision(self):
        self.client.post("/api/progress/topic-a/save", json=self.payload)
        second = self.client.post(
            "/api/progress/topic-a/save",
            json={
                "topic_title": "Backend Fundamentals",
                "questions": [
                    {
                        "question_id": "q3",
                        "question": "Explain circuit breakers.",
                        "difficulty": "hard",
                        "revealed": True,
                        "is_correct": True,
                        "confidence": 4,
                    }
                ],
                "sections": ["HTTP basics", "Resilience"],
            },
        ).json()
        self.assertEqual(second["revision"], 2)
        self.assertEqual(
            second["questions_asked"],
            [
                "Explain circuit breakers.",
                "Explain backpressure in streams.",
                "Explain keep-alive timeouts.",
            ],
        )
        self.assertEqual(second["sections"], ["HTTP basics", "Resilience"])

    def test_list_and_get_contracts(self):
        self.client.post("/api/progress/topic-a/save", json=self.payload)
        self.client.post(
            "/api/progress/topic-b/save",
            json={**self.payload, "topic_title": "API Design"},
        )

        listing = self.client.get("/api/progress/topics")
        self.assertEqual(listing.status_code, 200)
        body = listing.json()
        self.assertEqual(body["total"], 2)
        self.assertEqual(
            {doc["topic_id"] for doc in body["topics"]}, {"topic-a", "topic-b"}
        )

        single = self.client.get("/api/progress/topic-a")
        self.assertEqual(single.status_code, 200)
        self.assertEqual(single.json()["topic_title"], "Backend Fundamentals")

        self.assertEqual(self.client.get("/api/progress/absent").status_code, 404)

    def test_progress_is_scoped_per_user(self):
        self.client.post("/api/progress/topic-a/save", json=self.payload)
        self.app.dependency_overrides[require_auth] = (
            lambda: {"user": "bob", "provider": "github"}
        )
        self.assertEqual(self.client.get("/api/progress/topic-a").status_code, 404)
        self.assertEqual(self.client.get("/api/progress/topics").json()["total"], 0)

    def test_autosave_leaves_pending_when_summary_raises(self):
        progress._summarizer = RaisingSummarizer()
        with self.assertLogs("app.routers.progress", level="ERROR"):
            response = self.client.post("/api/progress/topic-a/autosave", json=self.payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        # Raw data is durable; the note stays owed so the next generation repairs it.
        self.assertEqual(body["summary_status"], "pending")
        self.assertEqual(body["questions_asked"], [
            "Explain backpressure in streams.",
            "Explain keep-alive timeouts.",
        ])

    def test_save_marks_failed_when_summary_raises(self):
        progress._summarizer = RaisingSummarizer()
        with self.assertLogs("app.routers.progress", level="ERROR"):
            response = self.client.post("/api/progress/topic-a/save", json=self.payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["summary_status"], "failed")
        self.assertIn("exploded", body["summary_error"])
        self.assertEqual(len(body["questions_asked"]), 2)

    def test_policy_blocked_llm_falls_back_but_still_saves(self):
        blocked = FakeLLM(text="")

        async def _blocked_completion(prompt, llm_config=None, user_identity=None, task=None):
            return {
                "success": False,
                "analysis": "",
                "metadata": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
                "error": "Personal credential required",
                "error_code": "personal_credential_required",
            }

        blocked.completion = _blocked_completion
        progress._summarizer = ProgressSummarizer(blocked)
        body = self.client.post("/api/progress/topic-a/save", json=self.payload).json()
        self.assertEqual(body["summary_status"], "ready")
        self.assertEqual(body["generation_source"], "fallback")
        self.assertIn("Backend Fundamentals", body["summary_text"])
        self.assertEqual(len(body["questions_asked"]), 2)

    def test_schedule_pending_summary_refresh_repairs_and_is_deduped(self):
        self.store.upsert_topic_progress(
            user_id="alice",
            topic_id="topic-a",
            topic_title="Backend Fundamentals",
            questions=["Explain keep-alive timeouts."],
            sections=[],
            attempt_stats={},
        )
        self.store.mark_summary_pending(user_id="alice", topic_id="topic-a")

        async def _exercise() -> None:
            scheduled = progress.schedule_pending_summary_refresh(
                user_id="alice",
                topic_id="topic-a",
                user_identity={"user": "alice", "provider": "google"},
            )
            self.assertTrue(scheduled)
            # A second request while the repair is in flight must not double-spend.
            self.assertFalse(
                progress.schedule_pending_summary_refresh(
                    user_id="alice",
                    topic_id="topic-a",
                    user_identity={"user": "alice", "provider": "google"},
                )
            )
            await asyncio.gather(*list(progress._background_tasks))

            document = self.store.get_progress(user_id="alice", topic_id="topic-a")
            self.assertEqual(document["summary_status"], "ready")
            self.assertEqual(document["summary_text"], "Learner covered the fundamentals.")
            self.assertNotIn(("alice", "topic-a"), progress._summary_inflight)

        asyncio.run(_exercise())

    def test_schedule_refresh_skips_rows_that_are_not_pending(self):
        self.store.upsert_topic_progress(
            user_id="alice",
            topic_id="topic-a",
            topic_title="Backend Fundamentals",
            questions=["Q?"],
            sections=[],
            attempt_stats={},
        )

        async def _exercise() -> None:
            progress.schedule_pending_summary_refresh(
                user_id="alice",
                topic_id="topic-a",
                user_identity={"user": "alice", "provider": "google"},
            )
            await asyncio.gather(*list(progress._background_tasks))
            self.assertEqual(self.fake_llm.calls, 0)

        asyncio.run(_exercise())

    def test_endpoints_require_auth_dependency(self):
        self.app.dependency_overrides.clear()
        self.assertEqual(self.client.get("/api/progress/topics").status_code, 401)
        self.assertEqual(
            self.client.post("/api/progress/topic-a/save", json=self.payload).status_code,
            401,
        )
