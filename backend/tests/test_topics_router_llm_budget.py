"""Stage 3.1: LLM budget exhaustion must not surface as a generic failure."""

import json
import os
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.dependencies import require_auth
from app.routers import topics
from app.services.doc_parser import DocParser
from app.services.learning_store import LearningStore
from app.services.llm_client import BUDGET_EXCEEDED_CODE


class BudgetExceeded(RuntimeError):
    """Mirrors a generator raising on the terminal per-user budget code."""

    error_code = BUDGET_EXCEEDED_CODE


class FakeLLM:
    def __init__(self):
        self.calls = 0

    async def completion(self, prompt, llm_config=None, user_identity=None, task=None, **kwargs):
        self.calls += 1
        if '"requires_programming"' in prompt:
            payload = {
                "requires_programming": True,
                "language_options": ["python", "go", "typescript"],
            }
        else:
            payload = {
                "title": "Custom Topic",
                "description": "Custom desc",
                "track": "backend",
                "levels": ["junior", "mid", "senior"],
                "sections": [
                    {"heading": f"Section {i}", "content": "Content for section."}
                    for i in range(1, 101)
                ],
            }
        return {
            "success": True,
            "analysis": json.dumps(payload),
            "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
            "error": "",
            "error_code": "",
        }


class TopicsRouterLLMBudgetTests(unittest.TestCase):
    def setUp(self):
        self.prev_parser = topics._parser
        self.prev_llm = topics._llm_client
        self.prev_store = topics._learning_store
        self.prev_mcp = topics._mcp_gateway

        self.tempdir = tempfile.TemporaryDirectory()
        handbook = os.path.join(self.tempdir.name, "owner-handbook")
        os.makedirs(handbook, exist_ok=True)
        with open(os.path.join(handbook, "01-static.md"), "w", encoding="utf-8") as handle:
            handle.write(
                """---
track: backend
levels: [junior, mid, senior]
---
# Static Topic

Static description.

## Static Section

Section body.
"""
            )
        self.parser = DocParser(docs_path=self.tempdir.name)
        self.store = LearningStore(os.path.join(self.tempdir.name, "learning.db"))

        topics.init(self.parser, FakeLLM(), self.store, mcp_gateway=None)
        self.app = FastAPI()
        self.app.dependency_overrides[require_auth] = lambda: {"user": "alice", "provider": "local"}
        self.app.include_router(topics.router)
        self.client = TestClient(self.app)

    def tearDown(self):
        topics._parser = self.prev_parser
        topics._llm_client = self.prev_llm
        topics._learning_store = self.prev_store
        topics._mcp_gateway = self.prev_mcp
        self.tempdir.cleanup()

    def _stream_events(self, path: str, payload: dict) -> list[dict]:
        with self.client.stream("POST", path, json=payload) as res:
            self.assertEqual(res.status_code, 200)
            return [json.loads(line) for line in res.iter_lines() if line]

    def test_custom_topic_create_surfaces_budget_exhaustion_as_429(self):
        import app.services.custom_topic_generator as generator_module

        async def _budget_exhausted(self, **kwargs):
            raise BudgetExceeded("LLM call budget exceeded. Retry in 30s.")

        original = generator_module.CustomTopicGenerator.generate_topic
        generator_module.CustomTopicGenerator.generate_topic = _budget_exhausted
        try:
            res = self.client.post("/api/topics/custom", json={"topic": "Rust ownership"})
        finally:
            generator_module.CustomTopicGenerator.generate_topic = original

        self.assertEqual(res.status_code, 429)
        detail = res.json()["detail"]
        self.assertEqual(detail["code"], BUDGET_EXCEEDED_CODE)

    def test_custom_topic_create_still_raises_other_failures(self):
        import app.services.custom_topic_generator as generator_module

        async def _boom(self, **kwargs):
            raise RuntimeError("provider exploded")

        original = generator_module.CustomTopicGenerator.generate_topic
        generator_module.CustomTopicGenerator.generate_topic = _boom
        try:
            with self.assertRaises(RuntimeError):
                self.client.post("/api/topics/custom", json={"topic": "Rust ownership"})
        finally:
            generator_module.CustomTopicGenerator.generate_topic = original

    def test_custom_topic_stream_reports_budget_exhaustion_distinctly(self):
        import app.services.custom_topic_generator as generator_module

        async def _budget_exhausted(self, **kwargs):
            raise BudgetExceeded("LLM call budget exceeded. Retry in 30s.")
            yield {}  # pragma: no cover - marks this as an async generator

        original = generator_module.CustomTopicGenerator.generate_topic_stream
        generator_module.CustomTopicGenerator.generate_topic_stream = _budget_exhausted
        try:
            events = self._stream_events("/api/topics/custom/stream", {"topic": "Rust ownership"})
        finally:
            generator_module.CustomTopicGenerator.generate_topic_stream = original

        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["code"], BUDGET_EXCEEDED_CODE)
        self.assertNotEqual(events[-1]["code"], "generation_failed")

    def test_custom_topic_stream_keeps_generic_code_for_other_failures(self):
        import app.services.custom_topic_generator as generator_module

        async def _boom(self, **kwargs):
            raise RuntimeError("provider exploded")
            yield {}  # pragma: no cover - marks this as an async generator

        original = generator_module.CustomTopicGenerator.generate_topic_stream
        generator_module.CustomTopicGenerator.generate_topic_stream = _boom
        try:
            events = self._stream_events("/api/topics/custom/stream", {"topic": "Rust ownership"})
        finally:
            generator_module.CustomTopicGenerator.generate_topic_stream = original

        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["code"], "generation_failed")

    def test_problem_solving_stream_reports_budget_exhaustion_distinctly(self):
        import app.services.problem_solving_generator as generator_module

        async def _budget_exhausted(self, **kwargs):
            raise BudgetExceeded("LLM call budget exceeded. Retry in 30s.")
            yield {}  # pragma: no cover - marks this as an async generator

        original = generator_module.ProblemSolvingGenerator.generate_topic_stream
        generator_module.ProblemSolvingGenerator.generate_topic_stream = _budget_exhausted
        try:
            events = self._stream_events(
                "/api/topics/00-problem-solving-and-algorithms/content/generate/stream",
                {"preferred_language": "python"},
            )
        finally:
            generator_module.ProblemSolvingGenerator.generate_topic_stream = original

        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["code"], BUDGET_EXCEEDED_CODE)

    def test_topic_detail_degrades_to_deterministic_profile_on_budget_exhaustion(self):
        """A cold topic read must not 500 when the advisor cannot spend a call."""
        import app.services.topic_language_advisor as advisor_module

        async def _budget_exhausted(self, **kwargs):
            from app.services.llm_client import BUDGET_EXCEEDED_CODE as code

            return {
                "success": False,
                "analysis": "",
                "metadata": {},
                "error": "LLM call budget exceeded.",
                "error_code": code,
            }

        original = advisor_module.TopicLanguageAdvisor.advise_topic
        advisor_module.TopicLanguageAdvisor.advise_topic = _budget_exhausted
        try:
            res = self.client.get("/api/topics/01-static")
        finally:
            advisor_module.TopicLanguageAdvisor.advise_topic = original

        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(res.json()["id"], "01-static")


if __name__ == "__main__":
    unittest.main()