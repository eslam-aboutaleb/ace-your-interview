"""Plan E coverage: latency telemetry, JSON-schema enforcement,
the prompt registry, and the generation-run store.

Every test patches ``app.services.llm_client.litellm.acompletion``
and ``app.services.llm_client.get_settings``; nothing reaches the
network. The run-store tests use a throwaway SQLite file.
"""

import asyncio
import os
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from app.schemas.models import LLMConfigRequest, LLMProviderEnum
from app.services.learning_store import (
    LearningStore,
    record_generation_run_safe,
)
from app.services.llm_client import (
    JSON_MODE_CAPABLE_PROVIDERS,
    JSON_SCHEMA_CAPABLE_PROVIDERS,
    RESPONSE_SCHEMA_CAPABLE_PROVIDERS,
    LLMClient,
    supported_options,
)
from app.services.prompt_registry import (
    PromptRegistry,
    PromptTemplate,
    default_registry,
)

_ALL_KEYS = ["__all__"]

_SCHEMA = {
    "title": "QuestionBatch",
    "type": "object",
    "properties": {"items": {"type": "array"}},
    "required": ["items"],
}


class _Capture:
    """Stub LiteLLM transport that records the kwargs it was called with."""

    def __init__(self, *, content='{"items": []}', finish_reason="stop", model="mock-model"):
        self.kwargs: dict = {}
        self.calls = 0
        self._response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                    finish_reason=finish_reason,
                )
            ],
            usage=None,
            model=model,
        )

    async def __call__(self, **kwargs):
        self.kwargs = kwargs
        self.calls += 1
        return self._response


def _settings(**overrides):
    base = dict(
        default_temperature=0.7,
        default_provider="openai",
        default_model="gpt-4o-mini",
        llm_task_routing_enabled=False,
        llm_task_route_map_json="",
        llm_user_call_budget=0,
        llm_user_call_budget_window_seconds=60,
        llm_user_max_concurrency=4,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _run(client, coro, settings=None, capture=None):
    async def _go():
        return await coro

    capture = capture or _Capture()
    with patch(
        "app.services.llm_client.get_settings",
        return_value=settings or _settings(),
    ):
        with patch("app.services.llm_client.litellm.acompletion", new=capture):
            result = asyncio.run(_go())
    return result, capture


class _ClientTest(unittest.TestCase):
    """Isolates the process-wide per-user budget and semaphore registries."""

    def setUp(self):
        LLMClient._budget.reset()
        LLMClient._inflight.clear()
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        LLMClient._budget.reset()
        LLMClient._inflight.clear()


class LatencyMetadataTests(_ClientTest):
    def test_latency_ms_is_present_in_metadata(self):
        client = LLMClient()
        result, _ = _run(
            client,
            client.completion("ping", None),
        )
        self.assertIn("latency_ms", result["metadata"])
        self.assertGreaterEqual(result["metadata"]["latency_ms"], 0)

    def test_latency_ms_is_a_number(self):
        client = LLMClient()
        result, _ = _run(
            client,
            client.completion("ping", None),
        )
        self.assertIsInstance(result["metadata"]["latency_ms"], (int, float))

    def test_retry_outcome_metadata_is_echoed(self):
        client = LLMClient()
        result, _ = _run(
            client,
            client.completion(
                "ping",
                None,
                attempts=3,
                terminal_reason="max_attempts",
                prompt_name="quiz_generation",
                prompt_version="2",
            ),
        )
        metadata = result["metadata"]
        self.assertEqual(metadata["attempts"], 3)
        self.assertEqual(metadata["terminal_reason"], "max_attempts")
        self.assertEqual(metadata["prompt_name"], "quiz_generation")
        self.assertEqual(metadata["prompt_version"], "2")


class SchemaEnforcementTests(_ClientTest):
    def test_openai_receives_json_schema_response_format(self):
        client = LLMClient()
        config = LLMConfigRequest(
            provider=LLMProviderEnum.OPENAI, model="gpt-4o-mini"
        )
        _, capture = _run(
            client,
            client.completion(
                "ping", config, structured=True, response_json_schema=_SCHEMA
            ),
        )
        self.assertEqual(
            capture.kwargs["response_format"],
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "QuestionBatch",
                    "schema": _SCHEMA,
                },
            },
        )

    def test_google_receives_response_schema_keyword(self):
        client = LLMClient()
        config = LLMConfigRequest(
            provider=LLMProviderEnum.GOOGLE, model="gemini/gemini-1.5-pro"
        )
        _, capture = _run(
            client,
            client.completion(
                "ping", config, structured=True, response_json_schema=_SCHEMA
            ),
        )
        self.assertEqual(capture.kwargs["response_schema"], _SCHEMA)
        self.assertNotIn("response_format", capture.kwargs)

    def test_groq_degrades_to_json_object_mode(self):
        client = LLMClient()
        config = LLMConfigRequest(provider=LLMProviderEnum.GROQ, model="llama-3.3-70b")
        _, capture = _run(
            client,
            client.completion(
                "ping", config, structured=True, response_json_schema=_SCHEMA
            ),
        )
        self.assertEqual(capture.kwargs["response_format"], {"type": "json_object"})
        self.assertNotIn("response_schema", capture.kwargs)

    def test_anthropic_gets_no_response_format(self):
        client = LLMClient()
        config = LLMConfigRequest(
            provider=LLMProviderEnum.ANTHROPIC, model="claude-3-5-sonnet"
        )
        _, capture = _run(
            client,
            client.completion(
                "ping", config, structured=True, response_json_schema=_SCHEMA
            ),
        )
        self.assertNotIn("response_format", capture.kwargs)
        self.assertNotIn("response_schema", capture.kwargs)

    def test_tools_take_precedence_over_schema(self):
        client = LLMClient()
        config = LLMConfigRequest(
            provider=LLMProviderEnum.OPENAI, model="gpt-4o-mini"
        )
        tools = [{"type": "function", "function": {"name": "noop"}}]
        _, capture = _run(
            client,
            client.completion(
                "ping",
                config,
                structured=True,
                response_json_schema=_SCHEMA,
                tools=tools,
            ),
        )
        self.assertEqual(capture.kwargs["tools"], tools)
        self.assertNotIn("response_format", capture.kwargs)

    def test_capability_sets_are_disjoint_where_expected(self):
        # A provider must never be asked for both surfaces at once.
        self.assertFalse(JSON_SCHEMA_CAPABLE_PROVIDERS & RESPONSE_SCHEMA_CAPABLE_PROVIDERS)
        # Every schema-capable provider is also json-object capable.
        self.assertTrue(JSON_SCHEMA_CAPABLE_PROVIDERS <= JSON_MODE_CAPABLE_PROVIDERS)


class SupportedOptionsTests(unittest.TestCase):
    def test_options_are_kept_when_signature_accepts_them(self):
        def completion(prompt, llm_config=None, **kwargs):
            return {}

        options = {"task": "final", "attempts": 2, "prompt_name": "quiz"}
        self.assertEqual(supported_options(SimpleNamespace(completion=completion), options), options)

    def test_unknown_options_are_dropped(self):
        def completion(prompt, llm_config=None, task=""):
            return {}

        client = SimpleNamespace(completion=completion)
        options = {"task": "final", "attempts": 2, "prompt_name": "quiz"}
        self.assertEqual(supported_options(client, options), {"task": "final"})

    def test_var_keyword_signature_keeps_everything(self):
        def completion(*args, **kwargs):
            return {}

        options = {"task": "final", "attempts": 2}
        client = SimpleNamespace(completion=completion)
        self.assertEqual(supported_options(client, options), options)

    def test_introspection_failure_returns_a_copy(self):
        client = SimpleNamespace(completion="not-a-callable")
        options = {"task": "final"}
        self.assertEqual(supported_options(client, options), {"task": "final"})


class PromptRegistryTests(unittest.TestCase):
    def test_register_and_get_by_name(self):
        registry = PromptRegistry()
        template = PromptTemplate(name="quiz_generation", version="1", system_prompt="sys")
        registry.register(template)
        self.assertIs(registry.get("quiz_generation"), template)

    def test_get_by_explicit_version(self):
        registry = PromptRegistry()
        v1 = PromptTemplate(name="quiz_generation", version="1", system_prompt="one")
        v2 = PromptTemplate(name="quiz_generation", version="2", system_prompt="two")
        registry.register(v1)
        registry.register(v2)
        self.assertIs(registry.get("quiz_generation", "1"), v1)
        self.assertIs(registry.get("quiz_generation", "2"), v2)

    def test_latest_version_tracks_the_highest(self):
        registry = PromptRegistry()
        registry.register(PromptTemplate(name="quiz_generation", version="1"))
        registry.register(PromptTemplate(name="quiz_generation", version="10"))
        registry.register(PromptTemplate(name="quiz_generation", version="2"))
        self.assertEqual(registry.latest_version("quiz_generation"), "10")
        self.assertEqual(registry.get("quiz_generation").version, "10")

    def test_generation_version_property(self):
        template = PromptTemplate(name="quiz_generation", version="3")
        self.assertEqual(template.generation_version, "quiz_generation@3")

    def test_get_unknown_name_raises(self):
        registry = PromptRegistry()
        with self.assertRaises(KeyError):
            registry.get("nope")

    def test_default_registry_carries_the_migrated_prompts(self):
        expected = {
            "question_generation",
            "question_generation_problem_solving",
            "quiz_generation",
            "card_generation",
            "custom_topic_architect",
            "problem_solving_instructor",
            "topic_language_advisor",
        }
        for name in expected:
            with self.subTest(name=name):
                template = default_registry.get(name)
                self.assertEqual(template.version, "1")
                self.assertTrue(template.system_prompt)

    def test_rag_answer_is_registered_without_a_system_prompt(self):
        # The RAG flow folds its persona into the user turn, so the
        # template pins only the (name, version) identity.
        template = default_registry.get("rag_answer")
        self.assertEqual(template.version, "1")
        self.assertEqual(template.system_prompt, "")

    def test_interview_templates_are_registered(self):
        # The interview templates register themselves as a side effect
        # of importing the interview generator (their user-prompt
        # builders are staticmethods on InterviewGenerator).
        import app.services.interview_generator  # noqa: F401

        for name in ("interview_question", "interview_evaluation"):
            with self.subTest(name=name):
                template = default_registry.get(name)
                self.assertEqual(template.version, "1")
                self.assertIsNotNone(template.build_user_prompt)


class _FakeRunStore:
    """Minimal stand-in exposing the run-store surface."""

    def __init__(self, *, fail=False):
        self.rows: list[dict] = []
        self.fail = fail
        self.written = threading.Event()

    def record_generation_run(self, **row):
        if self.fail:
            raise RuntimeError("boom")
        self.rows.append(row)
        self.written.set()


class GenerationRunStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = os.path.join(self._tmp.name, "learning.db")
        self.store = LearningStore(self.db_path)

    def test_record_generation_run_persists_a_row(self):
        run_id = uuid4().hex
        self.store.record_generation_run(
            run_id=run_id,
            user_id="user-1",
            flow="question_generation",
            provider="openai",
            model="gpt-4o-mini",
            prompt_name="question_generation",
            prompt_version="1",
            attempts=2,
            terminal_reason="ok",
            input_tokens=100,
            output_tokens=50,
            finish_reason="stop",
        )
        rows = self.store.get_generation_runs(user_id="user-1")
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["run_id"], run_id)
        self.assertEqual(row["flow"], "question_generation")
        self.assertEqual(row["provider"], "openai")
        self.assertEqual(row["model"], "gpt-4o-mini")
        self.assertEqual(row["prompt_name"], "question_generation")
        self.assertEqual(row["prompt_version"], "1")
        self.assertEqual(row["attempts"], 2)
        self.assertEqual(row["terminal_reason"], "ok")
        self.assertEqual(row["input_tokens"], 100)
        self.assertEqual(row["output_tokens"], 50)
        self.assertEqual(row["finish_reason"], "stop")
        self.assertTrue(row["created_at"])

    def test_get_generation_runs_filters_by_flow(self):
        self.store.record_generation_run(
            run_id=uuid4().hex, user_id="user-1", flow="quiz_generation"
        )
        self.store.record_generation_run(
            run_id=uuid4().hex, user_id="user-1", flow="question_generation"
        )
        rows = self.store.get_generation_runs(
            user_id="user-1", flow="question_generation"
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["flow"], "question_generation")

    def test_get_generation_runs_is_scoped_to_the_user(self):
        self.store.record_generation_run(
            run_id=uuid4().hex, user_id="user-1", flow="quiz_generation"
        )
        self.store.record_generation_run(
            run_id=uuid4().hex, user_id="user-2", flow="quiz_generation"
        )
        self.assertEqual(len(self.store.get_generation_runs(user_id="user-1")), 1)
        self.assertEqual(len(self.store.get_generation_runs(user_id="user-2")), 1)

    def test_record_generation_run_safe_writes_the_row(self):
        fake = _FakeRunStore()
        metadata = {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "input_tokens": 7,
            "output_tokens": 3,
        }
        asyncio.run(
            record_generation_run_safe(
                fake,
                user_identity={"user": "user-1"},
                metadata=metadata,
                flow="question_generation",
                prompt_name="question_generation",
                prompt_version="1",
                attempts=1,
                terminal_reason="ok",
                finish_reason="stop",
            )
        )
        # The write is synchronous, so the row is
        # on the fake store as soon as the call
        # returns.
        self.assertTrue(fake.written.is_set())
        self.assertEqual(len(fake.rows), 1)
        row = fake.rows[0]
        self.assertEqual(row["user_id"], "user-1")
        self.assertEqual(row["flow"], "question_generation")
        self.assertEqual(row["prompt_name"], "question_generation")
        self.assertEqual(row["prompt_version"], "1")
        self.assertEqual(row["input_tokens"], 7)
        self.assertEqual(row["output_tokens"], 3)

    def test_record_generation_run_safe_swallows_write_failures(self):
        fake = _FakeRunStore(fail=True)
        # Must not raise even though the underlying write fails.
        asyncio.run(
            record_generation_run_safe(
                fake,
                user_identity={"user": "user-1"},
                metadata={},
                flow="question_generation",
            )
        )
        self.assertEqual(fake.rows, [])

    def test_record_generation_run_safe_is_a_noop_without_a_store(self):
        self.assertIsNone(
            asyncio.run(
                record_generation_run_safe(
                    None,
                    user_identity={"user": "user-1"},
                    metadata={},
                    flow="question_generation",
                )
            )
        )

    def test_record_generation_run_safe_is_a_noop_without_the_method(self):
        plain = SimpleNamespace()
        self.assertIsNone(
            asyncio.run(
                record_generation_run_safe(
                    plain,
                    user_identity={"user": "user-1"},
                    metadata={},
                    flow="question_generation",
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
