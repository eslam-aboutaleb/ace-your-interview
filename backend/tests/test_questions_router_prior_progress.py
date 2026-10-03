"""Router-level wiring for prior progress: prompt injection + server-side capture."""

import json
import os
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.dependencies import require_auth
from app.routers import questions
from app.services.doc_parser import DocParser
from app.services.learning_store import LearningStore
from app.services.progress_store import ProgressStore


class FakeLLM:
    def __init__(self):
        self.calls = 0
        self.prompts: list[str] = []

    async def completion(self, prompt, llm_config=None, user_identity=None, task=None, **kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        question = f"Stored history question number {self.calls} about HTTP keep-alive."
        payload = [
            {
                "question": question,
                "answer": (
                    f"**Answer:** {question}\n\n"
                    "**Detailed explanation:** Keep-alive reuses one TCP connection across requests, "
                    "which trades a held socket for lower latency.\n\n"
                    "**Common mistake:** Confusing idle timeout with total connection lifetime."
                ),
                "difficulty": "medium",
                "learning_objective": "Explain keep-alive behavior in HTTP.",
                "source_section": "HTTP Basics",
                "source_quote": "Keep-alive reuses a single connection for multiple requests.",
                "reasoning_summary": "Identify the timeout that governs reuse.",
                "target_level": "mid",
            }
        ]
        return {
            "success": True,
            "analysis": json.dumps(payload),
            "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
            "error": "",
        }


class FakeQuizLLM:
    def __init__(self):
        self.calls = 0
        self.prompts: list[str] = []

    async def completion(self, prompt, llm_config=None, user_identity=None, task=None, **kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        payload = [
            {
                "question": f"Quiz item {self.calls}: which header validates a cached response?",
                "type": "mcq",
                "choices": [
                    {"label": "A", "text": "ETag"},
                    {"label": "B", "text": "Age"},
                    {"label": "C", "text": "Vary"},
                    {"label": "D", "text": "Expires"},
                ],
                "correct_answer": "A",
                "explanation": "ETag is the response validator header.",
                "difficulty": "medium",
                "topic_id": "01-http",
                "source_quote": "Keep-alive reuses a connection across requests.",
                "reasoning_summary": "Validators let a client revalidate cheaply.",
                "target_level": "mid",
            }
        ]
        return {
            "success": True,
            "analysis": json.dumps(payload),
            "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
        }


class QuestionsRouterPriorProgressTests(unittest.TestCase):
    def setUp(self):
        self.prev_llm = questions._llm_client
        self.prev_parser = questions._parser
        self.prev_store = questions._learning_store
        self.prev_progress = questions._progress_store

        self.tempdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self.tempdir.name, "learning.db")
        handbook = os.path.join(self.tempdir.name, "owner-handbook")
        os.makedirs(handbook, exist_ok=True)
        with open(os.path.join(handbook, "01-http.md"), "w", encoding="utf-8") as f:
            f.write(
                """---
track: backend
levels: [junior, mid, senior]
---
# HTTP Fundamentals

Keep-alive lets one TCP connection carry many requests.
"""
            )

        self.parser = DocParser(docs_path=self.tempdir.name)
        self.learning_store = LearningStore(db_path)
        self.progress_store = ProgressStore(db_path)
        self.progress_store.upsert_topic_progress(
            user_id="alice",
            provider="local",
            topic_id="01-http",
            topic_title="HTTP Fundamentals",
            questions=["Prior stored question about connection reuse?"],
            sections=["HTTP Basics"],
            attempt_stats={"total": 1, "submitted": 1, "correct": 1},
        )
        self.progress_store.set_summary(
            user_id="alice",
            topic_id="01-http",
            text="Learner already covered connection reuse and needs deeper timeout questions.",
            provider_used="openai",
            model_used="gpt-4o-mini",
            source="ai",
        )

        self.fake_llm = FakeLLM()
        questions.init(
            self.fake_llm, self.parser, self.learning_store, None, self.progress_store
        )
        self.app = FastAPI()
        self.app.dependency_overrides[require_auth] = (
            lambda: {"user": "alice", "provider": "local"}
        )
        self.app.include_router(questions.router)
        self.client = TestClient(self.app)

    def tearDown(self):
        questions._llm_client = self.prev_llm
        questions._parser = self.prev_parser
        questions._learning_store = self.prev_store
        questions._progress_store = self.prev_progress
        self.tempdir.cleanup()

    def test_generate_v2_injects_progress_and_stored_questions(self):
        response = self.client.post(
            "/api/questions/generate-v2",
            json={"topic_id": "01-http", "count": 1, "level": "mid"},
        )
        self.assertEqual(response.status_code, 200)
        prompt = self.fake_llm.prompts[0]
        self.assertIn("Learner already covered connection reuse", prompt)
        self.assertIn("Learner progress on this topic so far", prompt)
        self.assertIn("Prior stored question about connection reuse?", prompt)

    def test_server_side_capture_records_generated_questions(self):
        response = self.client.post(
            "/api/questions/generate-v2",
            json={"topic_id": "01-http", "count": 1, "level": "mid"},
        ).json()
        generated_text = response["questions"][0]["question"]
        asked = self.progress_store.get_asked_questions(
            user_id="alice", topic_id="01-http", limit=200
        )
        self.assertIn(generated_text, asked)
        # Stored history is a superset: the pre-existing entry survives.
        self.assertIn("Prior stored question about connection reuse?", asked)
        self.assertEqual(asked[0], generated_text)

    def test_capture_does_not_disturb_the_summary_status(self):
        self.client.post(
            "/api/questions/generate-v2",
            json={"topic_id": "01-http", "count": 1, "level": "mid"},
        )
        document = self.progress_store.get_progress(
            user_id="alice", topic_id="01-http"
        )
        self.assertEqual(document["summary_status"], "ready")
        # Capture bumps revision but never invalidates an existing summary: the
        # save path owns the summary lifecycle, and flipping to `pending` on every
        # generation would re-run the summary LLM call per generation.
        self.assertEqual(document["revision"], 2)

    def test_streaming_endpoint_injects_progress_and_captures(self):
        with self.client.stream(
            "POST",
            "/api/questions/generate-v2/stream",
            json={"topic_id": "01-http", "count": 1, "level": "mid"},
        ) as res:
            self.assertEqual(res.status_code, 200)
            events = [
                json.loads(line)
                for line in res.iter_lines()
                if line
            ]
        questions_seen = [e for e in events if e.get("type") == "question"]
        self.assertEqual(len(questions_seen), 1)
        self.assertIn("Learner already covered connection reuse", self.fake_llm.prompts[0])
        asked = self.progress_store.get_asked_questions(
            user_id="alice", topic_id="01-http", limit=200
        )
        self.assertIn(questions_seen[0]["question"]["question"], asked)

    def test_pending_summary_triggers_a_non_blocking_repair(self):
        from app.routers import progress as progress_router
        from app.services.progress_summarizer import ProgressSummarizer

        self.progress_store.mark_summary_pending(
            user_id="alice", topic_id="01-http"
        )
        repair_llm = FakeLLM()
        prev_summarizer = progress_router._summarizer
        progress_router._summary_inflight.clear()
        progress_router._background_tasks.clear()
        progress_router.init(self.progress_store, ProgressSummarizer(repair_llm), repair_llm)
        try:
            response = self.client.post(
                "/api/questions/generate-v2",
                json={"topic_id": "01-http", "count": 1, "level": "mid"},
            )
            self.assertEqual(response.status_code, 200)
            import asyncio

            asyncio.run(
                asyncio.wait_for(
                    _drain(progress_router._background_tasks), timeout=5
                )
            )
            document = self.progress_store.get_progress(
                user_id="alice", topic_id="01-http"
            )
            self.assertEqual(document["summary_status"], "ready")
            self.assertIn("Stored history question number", document["summary_text"])
        finally:
            progress_router._summarizer = prev_summarizer
            progress_router._summary_inflight.clear()
            progress_router._background_tasks.clear()

    def test_progress_store_failure_does_not_break_generation(self):
        class BrokenStore(ProgressStore):
            def get_progress(self, **kwargs):
                raise RuntimeError("progress db unavailable")

            def get_asked_questions(self, **kwargs):
                raise RuntimeError("progress db unavailable")

        broken = BrokenStore(self.progress_store.db_path)
        questions._progress_store = broken
        with self.assertLogs("app.routers.questions", level="ERROR"):
            response = self.client.post(
                "/api/questions/generate-v2",
                json={"topic_id": "01-http", "count": 1, "level": "mid"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["questions"]), 1)

    def test_multi_topic_history_is_interleaved_newest_first(self):
        import asyncio

        self.progress_store.record_asked_questions(
            user_id="alice",
            topic_id="01-http",
            topic_title="HTTP Fundamentals",
            questions=["HTTP oldest?", "HTTP newest?"],
        )
        with open(os.path.join(self.tempdir.name, "owner-handbook", "02-sql.md"), "w") as f:
            f.write(
                """---
track: backend
levels: [junior, mid, senior]
---
# SQL Fundamentals

Indexes speed up reads at the cost of write amplification.
"""
            )
        self.progress_store.record_asked_questions(
            user_id="alice",
            topic_id="02-sql",
            topic_title="SQL Fundamentals",
            questions=["SQL oldest?", "SQL newest?"],
        )
        self.progress_store.set_summary(
            user_id="alice",
            topic_id="02-sql",
            text="Learner covered basic selects.",
            source="ai",
        )

        async def _exercise() -> tuple[str, list[str]]:
            return await questions._load_progress_context(
                user_id="alice",
                topic_ids=["01-http", "02-sql"],
                user_identity={"user": "alice", "provider": "local"},
            )

        summary, asked = asyncio.run(_exercise())
        self.assertIn("HTTP Fundamentals", summary)
        self.assertIn("SQL Fundamentals", summary)
        self.assertIn("Learner covered basic selects.", summary)
        # Round-robin so the prompt budget covers both topics, not just the first.
        self.assertEqual(
            asked,
            [
                "HTTP newest?",
                "SQL newest?",
                "HTTP oldest?",
                "SQL oldest?",
                "Prior stored question about connection reuse?",
            ],
        )

    def test_topic_without_progress_is_skipped(self):
        import asyncio

        async def _exercise() -> tuple[str, list[str]]:
            return await questions._load_progress_context(
                user_id="alice",
                topic_ids=["01-http", "never-studied"],
                user_identity={"user": "alice", "provider": "local"},
            )

        summary, asked = asyncio.run(_exercise())
        self.assertIn("HTTP Fundamentals", summary)
        self.assertNotIn("never-studied", summary)
        self.assertEqual(asked, ["Prior stored question about connection reuse?"])

    def test_grouped_capture_buckets_by_topic(self):
        grouped = questions._group_questions_by_topic(
            [
                {"topic_id": "01-http", "question": "Q1?"},
                {"topic_id": "02-sql", "question": "Q2?"},
                {"topic_id": "01-http", "question": "Q3?"},
                {"topic_id": "", "question": "orphan"},
                {"question": "no topic id"},
                {"topic_id": "02-sql", "question": "  "},
                "not a dict",
            ]
        )
        self.assertEqual(
            grouped,
            {"01-http": ["Q1?", "Q3?"], "02-sql": ["Q2?"]},
        )

    def test_quiz_generation_injects_progress_and_captures(self):
        quiz_llm = FakeQuizLLM()
        questions._llm_client = quiz_llm
        response = self.client.post(
            "/api/questions/quiz/generate-v2",
            json={
                "topic_ids": ["01-http"],
                "count": 1,
                "question_types": ["mcq"],
                "level": "mid",
            },
        )
        self.assertEqual(response.status_code, 200)
        prompt = quiz_llm.prompts[0]
        self.assertIn("Learner already covered connection reuse", prompt)
        self.assertIn("Already asked for these topics", prompt)
        self.assertIn("Prior stored question about connection reuse?", prompt)

        asked = self.progress_store.get_asked_questions(
            user_id="alice", topic_id="01-http", limit=200
        )
        self.assertIn(
            "Quiz item 1: which header validates a cached response?",
            asked,
        )
        self.assertIn("Prior stored question about connection reuse?", asked)

    def test_quiz_stream_injects_progress_and_captures(self):
        quiz_llm = FakeQuizLLM()
        questions._llm_client = quiz_llm
        with self.client.stream(
            "POST",
            "/api/questions/quiz/generate-v2/stream",
            json={
                "topic_ids": ["01-http"],
                "count": 1,
                "question_types": ["mcq"],
                "level": "mid",
            },
        ) as res:
            self.assertEqual(res.status_code, 200)
            events = [json.loads(line) for line in res.iter_lines() if line]

        self.assertIn("Learner already covered connection reuse", quiz_llm.prompts[0])
        emitted = [e["question"] for e in events if e.get("type") == "question"]
        self.assertEqual(len(emitted), 1)
        asked = self.progress_store.get_asked_questions(
            user_id="alice", topic_id="01-http", limit=200
        )
        self.assertIn(emitted[0]["question"], asked)


async def _drain(tasks: set) -> None:
    import asyncio

    pending = [t for t in tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
