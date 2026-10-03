"""``EnrichmentToolRunner`` (formerly ``MCPGateway``).

Covers the preserved config surface, the opt-out path, the deterministic
single-pass mode, and the bounded model tool loop including the capability guard.
"""

import asyncio
import unittest
from types import SimpleNamespace

from app.services.mcp_gateway import (
    ENRICHMENT_SYSTEM_PROMPT,
    EnrichmentToolRunner,
    MCPGateway,
)
from app.services.tools.agentic import ModelTurn, ToolCall
from app.services.tools.enrichment import (
    FIRECRAWL_TOOL,
    GITHUB_TOOL,
    TAVILY_TOOL,
    build_enrichment_registry,
)
from app.services.tools.registry import ToolRegistry, ToolSpec


def _settings(**overrides):
    base = {
        "enable_mcp_gateway": True,
        "mcp_rollout_stage": 1,
        "mcp_timeout_seconds": 3.0,
        "mcp_max_context_chars": 4000,
        "mcp_enable_custom_topic": True,
        "mcp_enable_chat": True,
        "mcp_enable_questions": True,
        "mcp_enable_quiz": True,
        "mcp_enable_interview": True,
        "mcp_agentic_loop_enabled": False,
        "mcp_agentic_max_steps": 2,
        "mcp_tavily_enabled": False,
        "tavily_api_key": "",
        "mcp_firecrawl_enabled": False,
        "firecrawl_api_key": "",
        "mcp_github_enabled": False,
        "github_token": "",
        "default_provider": "groq",
        "default_model": "llama-3.3-70b-versatile",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _run(coro):
    return asyncio.run(coro)


class ExplodingTransport:
    """Fails the test if the loop ever reaches the model."""

    def __init__(self):
        self.calls = 0

    async def complete(self, **kwargs):
        self.calls += 1
        raise AssertionError("the model must not be called")


class ScriptedTransport:
    def __init__(self, turns):
        self.turns = list(turns)
        self.calls: list[dict] = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        if not self.turns:
            return ModelTurn(text="done")
        return self.turns.pop(0)


class FakeRequest:
    def __init__(self, *payloads, error: Exception | None = None):
        self.payloads = list(payloads)
        self.error = error
        self.calls: list[tuple] = []

    async def __call__(self, method, url, headers, payload):
        self.calls.append((method, url, dict(payload)))
        if self.error is not None:
            raise self.error
        if not self.payloads:
            return {}
        value = self.payloads.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def _tavily_registry(request, **settings_overrides):
    settings = _settings(
        mcp_tavily_enabled=True,
        tavily_api_key="tvly",
        mcp_firecrawl_enabled=True,
        firecrawl_api_key="fc",
        mcp_github_enabled=True,
        github_token="ghp",
        **settings_overrides,
    )
    return build_enrichment_registry(settings, request=request), settings


_TAVILY_ROWS = {
    "results": [
        {
            "title": "Idempotency keys",
            "content": "Deduplicate retried requests with a client-supplied key.",
            "url": "https://example.com/idem",
        }
    ]
}
_FIRECRAWL_BODY = {"data": {"markdown": "# Idempotency\nUse a unique key per request."}}
_GITHUB_ROWS = {
    "items": [
        {
            "full_name": "acme/idem",
            "description": "idempotency helpers",
            "html_url": "https://github.com/acme/idem",
        }
    ]
}


class BackwardsCompatibilityTests(unittest.TestCase):
    def test_old_name_is_an_alias_for_the_renamed_class(self):
        self.assertIs(MCPGateway, EnrichmentToolRunner)

    def test_alias_exposes_the_same_public_surface(self):
        for name in ("is_enabled_for_flow", "gather_context", "max_steps", "resolve_provider_model"):
            self.assertTrue(hasattr(MCPGateway, name), name)


class GatingTests(unittest.TestCase):
    def test_rollout_stage_gating_is_unchanged(self):
        runner = EnrichmentToolRunner(_settings(mcp_rollout_stage=1))
        self.assertTrue(runner.is_enabled_for_flow("custom_topic"))
        self.assertTrue(runner.is_enabled_for_flow("chat"))
        self.assertFalse(runner.is_enabled_for_flow("questions"))
        self.assertFalse(runner.is_enabled_for_flow("quiz"))
        self.assertFalse(runner.is_enabled_for_flow("interview"))

        runner3 = EnrichmentToolRunner(_settings(mcp_rollout_stage=3))
        self.assertTrue(runner3.is_enabled_for_flow("questions"))
        self.assertTrue(runner3.is_enabled_for_flow("quiz"))
        self.assertTrue(runner3.is_enabled_for_flow("interview"))

    def test_per_flow_flags_are_honoured(self):
        runner = EnrichmentToolRunner(
            _settings(mcp_rollout_stage=3, mcp_enable_chat=False)
        )
        self.assertFalse(runner.is_enabled_for_flow("chat"))
        self.assertTrue(runner.is_enabled_for_flow("interview"))

    def test_unknown_flow_is_never_enabled(self):
        self.assertFalse(EnrichmentToolRunner(_settings()).is_enabled_for_flow("nope"))


class OptOutParityTests(unittest.TestCase):
    def test_disabled_by_default_returns_empty(self):
        runner = EnrichmentToolRunner(_settings(enable_mcp_gateway=False))
        self.assertEqual(
            _run(runner.gather_context(flow="chat", query="java retry patterns")), ""
        )

    def test_disabled_builds_no_registry_and_makes_no_model_call(self):
        transport = ExplodingTransport()
        runner = EnrichmentToolRunner(_settings(enable_mcp_gateway=False), transport=transport)
        out = _run(
            runner.gather_context(
                flow="chat",
                query="retry patterns",
                topic_id="backend",
                topic_title="Retry Patterns",
            )
        )
        self.assertEqual(out, "")
        self.assertEqual(transport.calls, 0)

    def test_flow_flag_off_returns_empty_without_a_model_call(self):
        transport = ExplodingTransport()
        runner = EnrichmentToolRunner(
            _settings(mcp_enable_chat=False), transport=transport
        )
        self.assertEqual(_run(runner.gather_context(flow="chat", query="x")), "")
        self.assertEqual(transport.calls, 0)

    def test_blank_query_returns_empty(self):
        transport = ExplodingTransport()
        runner = EnrichmentToolRunner(_settings(), transport=transport)
        self.assertEqual(_run(runner.gather_context(flow="chat", query="   ")), "")
        self.assertEqual(transport.calls, 0)

    def test_no_enabled_providers_returns_empty(self):
        transport = ExplodingTransport()
        runner = EnrichmentToolRunner(_settings(), transport=transport)
        self.assertEqual(
            _run(runner.gather_context(flow="chat", query="retry patterns")), ""
        )
        self.assertEqual(transport.calls, 0)


class SinglePassTests(unittest.TestCase):
    def test_providers_are_consulted_in_order(self):
        request = FakeRequest(_TAVILY_ROWS, _FIRECRAWL_BODY, _GITHUB_ROWS)
        registry, _ = _tavily_registry(request)
        runner = EnrichmentToolRunner(_settings(), registry=registry)
        out = _run(
            runner.gather_context(
                flow="chat",
                query="retry safety",
                topic_id="backend",
                topic_title="Retry Patterns",
            )
        )
        self.assertIn("Tavily web context", out)
        self.assertIn("Firecrawl extracted context", out)
        self.assertIn("GitHub context", out)
        self.assertEqual(len(request.calls), 3)
        # The scrape URL comes from the search result, not the raw query.
        self.assertEqual(request.calls[1][2]["url"], "https://example.com/idem")

    def test_scraper_is_skipped_when_search_returned_nothing(self):
        request = FakeRequest({"results": []}, _GITHUB_ROWS)
        registry, _ = _tavily_registry(request)
        runner = EnrichmentToolRunner(_settings(), registry=registry)
        out = _run(runner.gather_context(flow="chat", query="nothing matches"))
        self.assertNotIn("Firecrawl", out)
        self.assertIn("GitHub context", out)
        self.assertEqual(len(request.calls), 2)

    def test_disabled_providers_are_not_called(self):
        request = FakeRequest(_TAVILY_ROWS)
        settings = _settings(
            mcp_tavily_enabled=True,
            tavily_api_key="tvly",
            mcp_firecrawl_enabled=False,
            mcp_github_enabled=False,
        )
        registry = build_enrichment_registry(settings, request=request)
        runner = EnrichmentToolRunner(_settings(), registry=registry)
        out = _run(runner.gather_context(flow="chat", query="retry"))
        self.assertIn("Tavily web context", out)
        self.assertEqual(len(request.calls), 1)

    def test_provider_failure_degrades_to_empty(self):
        request = FakeRequest(error=RuntimeError("502"))
        registry, _ = _tavily_registry(request)
        runner = EnrichmentToolRunner(_settings(), registry=registry)
        self.assertEqual(_run(runner.gather_context(flow="chat", query="retry")), "")

    def test_context_is_capped_by_max_context_chars(self):
        request = FakeRequest(
            {"results": [{"title": "T", "content": "x" * 4000, "url": "https://e.dev"}]}
        )
        registry, _ = _tavily_registry(request)
        runner = EnrichmentToolRunner(_settings(mcp_max_context_chars=120), registry=registry)
        out = _run(runner.gather_context(flow="chat", query="retry"))
        self.assertLessEqual(len(out), 120)

    def test_agentic_flag_off_never_calls_the_model(self):
        transport = ExplodingTransport()
        request = FakeRequest(_TAVILY_ROWS)
        registry, settings = _tavily_registry(request)
        runner = EnrichmentToolRunner(
            settings, registry=registry, transport=transport
        )
        out = _run(runner.gather_context(flow="chat", query="retry"))
        self.assertIn("Tavily web context", out)
        self.assertEqual(transport.calls, 0)


class ToolLoopTests(unittest.TestCase):
    def _runner(self, turns, *, provider="groq", model="llama-3.3-70b-versatile", **extra):
        request = FakeRequest(_TAVILY_ROWS, _FIRECRAWL_BODY)
        registry, settings = _tavily_registry(
            request, mcp_agentic_loop_enabled=True, mcp_agentic_max_steps=3
        )
        transport = ScriptedTransport(turns)
        runner = EnrichmentToolRunner(
            settings,
            registry=registry,
            transport=transport,
            provider=provider,
            model=model,
            **extra,
        )
        return runner, transport

    @staticmethod
    def _tavily_call(call_id="c1", query="idempotency keys"):
        import json

        return ToolCall(
            id=call_id,
            name=TAVILY_TOOL,
            arguments=json.dumps({"query": query}),
        )

    def test_model_selects_a_tool_and_the_result_feeds_back(self):
        runner, transport = self._runner(
            [
                ModelTurn(text="", tool_calls=(self._tavily_call(),)),
                ModelTurn(text="## Briefing\nIdempotency keys dedupe retries."),
            ]
        )
        out = _run(
            runner.gather_context(
                flow="chat",
                query="how do idempotency keys work",
                topic_id="backend",
                topic_title="API Design",
            )
        )
        self.assertIn("Tavily web context", out)
        self.assertIn("## Briefing", out)

        # The second model call carries the assistant tool_call and the
        # role="tool" result.
        second = list(transport.calls[1]["messages"])
        self.assertEqual(
            [m["role"] for m in second], ["system", "user", "assistant", "tool"]
        )
        self.assertEqual(second[2]["tool_calls"][0]["function"]["name"], TAVILY_TOOL)
        self.assertIn("Idempotency keys", second[3]["content"])
        self.assertEqual(second[3]["tool_call_id"], "c1")

    def test_system_prompt_is_sent(self):
        runner, transport = self._runner([ModelTurn(text="brief")])
        _run(runner.gather_context(flow="chat", query="q"))
        messages = list(transport.calls[0]["messages"])
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], ENRICHMENT_SYSTEM_PROMPT)
        self.assertIn("Research brief: q", messages[1]["content"])

    def test_only_enabled_tools_are_advertised(self):
        runner, transport = self._runner([ModelTurn(text="brief")])
        _run(runner.gather_context(flow="chat", query="q"))
        names = {t["function"]["name"] for t in transport.calls[0]["tools"]}
        self.assertEqual(names, {TAVILY_TOOL, FIRECRAWL_TOOL, GITHUB_TOOL})

    def test_loop_stops_at_max_steps(self):
        import json

        call = ToolCall(
            id="cx",
            name=TAVILY_TOOL,
            arguments=json.dumps({"query": "loop"}),
        )
        runner, transport = self._runner(
            [ModelTurn(text=f"step{i}", tool_calls=(call,)) for i in range(6)]
        )
        settings = runner.settings
        settings.mcp_agentic_max_steps = 2
        out = _run(runner.gather_context(flow="chat", query="q"))
        self.assertEqual(len(transport.calls), 2)
        self.assertIn("Tavily web context", out)

    def test_model_that_calls_nothing_terminates_immediately(self):
        runner, transport = self._runner([ModelTurn(text="I know this already.")])
        out = _run(runner.gather_context(flow="chat", query="q"))
        self.assertEqual(len(transport.calls), 1)
        self.assertIn("I know this already.", out)

    def test_unknown_tool_does_not_loop_forever(self):
        import json

        call = ToolCall(id="g1", name="does_not_exist", arguments=json.dumps({}))
        runner, transport = self._runner([ModelTurn(text="", tool_calls=(call,))])
        out = _run(runner.gather_context(flow="chat", query="q"))
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(out, "")

    def test_failing_tool_does_not_raise(self):
        request = FakeRequest(RuntimeError("502"))
        registry, settings = _tavily_registry(
            request, mcp_agentic_loop_enabled=True, mcp_agentic_max_steps=3
        )
        transport = ScriptedTransport([ModelTurn(text="nothing usable"), ModelTurn(text="brief")])
        runner = EnrichmentToolRunner(settings, registry=registry, transport=transport)
        out = _run(runner.gather_context(flow="chat", query="q"))
        self.assertIn("brief", out)

    def test_max_steps_of_one_falls_back_to_the_single_pass(self):
        request = FakeRequest(_TAVILY_ROWS)
        registry, settings = _tavily_registry(
            request, mcp_agentic_loop_enabled=True, mcp_agentic_max_steps=1
        )
        transport = ExplodingTransport()
        runner = EnrichmentToolRunner(settings, registry=registry, transport=transport)
        out = _run(runner.gather_context(flow="chat", query="retry"))
        self.assertIn("Tavily web context", out)
        self.assertEqual(transport.calls, 0)

    def test_transport_failure_degrades_to_empty(self):
        class Boom:
            async def complete(self, **kwargs):
                raise RuntimeError("provider down")

        request = FakeRequest(_TAVILY_ROWS)
        registry, settings = _tavily_registry(
            request, mcp_agentic_loop_enabled=True, mcp_agentic_max_steps=3
        )
        runner = EnrichmentToolRunner(settings, registry=registry, transport=Boom())
        self.assertEqual(_run(runner.gather_context(flow="chat", query="q")), "")

    def test_injected_registry_replaces_the_enrichment_registry(self):
        registry = ToolRegistry(
            [
                ToolSpec(
                    name="custom",
                    description="d",
                    parameters={"type": "object", "properties": {}},
                    handler=lambda a: "ok",
                )
            ]
        )
        transport = ScriptedTransport([ModelTurn(text="brief")])
        runner = EnrichmentToolRunner(
            _settings(mcp_agentic_loop_enabled=True), registry=registry, transport=transport
        )
        out = _run(runner.gather_context(flow="chat", query="q"))
        names = {t["function"]["name"] for t in transport.calls[0]["tools"]}
        self.assertEqual(names, {"custom"})
        self.assertIn("brief", out)

    def test_injected_registry_with_nothing_enabled_short_circuits(self):
        registry = ToolRegistry(
            [
                ToolSpec(
                    name="custom",
                    description="d",
                    parameters={"type": "object", "properties": {}},
                    handler=lambda a: "ok",
                    enabled=False,
                )
            ]
        )
        transport = ExplodingTransport()
        runner = EnrichmentToolRunner(
            _settings(mcp_agentic_loop_enabled=True), registry=registry, transport=transport
        )
        out = _run(runner.gather_context(flow="chat", query="q"))
        self.assertEqual(transport.calls, 0)
        self.assertEqual(out, "")


class CapabilityGuardTests(unittest.TestCase):
    def _loop_settings(self, **overrides):
        return _settings(mcp_agentic_loop_enabled=True, mcp_agentic_max_steps=3, **overrides)

    def _gather_with_provider(self, provider, model):
        request = FakeRequest(_TAVILY_ROWS)
        registry = build_enrichment_registry(
            self._loop_settings(mcp_tavily_enabled=True, tavily_api_key="tvly"),
            request=request,
        )
        transport = ExplodingTransport()
        runner = EnrichmentToolRunner(
            self._loop_settings(), registry=registry, transport=transport, provider=provider, model=model
        )
        return _run(runner.gather_context(flow="chat", query="retry")), transport, request

    def test_anthropic_skips_enrichment_entirely(self):
        out, transport, request = self._gather_with_provider(
            "anthropic", "claude-3-5-sonnet-20241022"
        )
        self.assertEqual(out, "")
        self.assertEqual(transport.calls, 0, "no model call may be made")
        self.assertEqual(request.calls, [], "no provider call either")

    def test_ollama_skips_enrichment_entirely(self):
        out, transport, request = self._gather_with_provider("ollama", "llama3.2")
        self.assertEqual(out, "")
        self.assertEqual(transport.calls, 0)

    def test_incapable_model_on_a_capable_provider_is_skipped(self):
        out, transport, request = self._gather_with_provider("groq", "gemma2-9b-it")
        self.assertEqual(out, "")
        self.assertEqual(transport.calls, 0)

    def test_capable_provider_runs_the_loop(self):
        out, transport, _ = self._gather_with_provider("groq", "llama-3.3-70b-versatile")
        # The scripted transport raises if reached, so reaching this point proves
        # the guard let it through; the exception is contained as an empty result.
        self.assertEqual(out, "")
        self.assertEqual(transport.calls, 1)

    def test_guard_defaults_to_the_settings_model(self):
        runner = EnrichmentToolRunner(
            _settings(default_provider="anthropic", default_model="claude-3-5-sonnet")
        )
        self.assertEqual(runner.resolve_provider_model(), ("anthropic", "claude-3-5-sonnet"))


class ModelResolutionTests(unittest.TestCase):
    def test_defaults_to_the_backend_default_model(self):
        runner = EnrichmentToolRunner(_settings())
        self.assertEqual(
            runner.resolve_provider_model(), ("groq", "llama-3.3-70b-versatile")
        )

    def test_explicit_override_wins(self):
        runner = EnrichmentToolRunner(_settings(), provider="openai", model="gpt-4o-mini")
        self.assertEqual(runner.resolve_provider_model(), ("openai", "gpt-4o-mini"))

    def test_transport_factory_is_used(self):
        seen: list[tuple] = []
        sentinel = ScriptedTransport([ModelTurn(text="built")])

        def factory(provider, model):
            seen.append((provider, model))
            return sentinel

        runner = EnrichmentToolRunner(
            _settings(mcp_agentic_loop_enabled=True),
            transport_factory=factory,
            provider="openai",
            model="gpt-4o-mini",
        )
        runner._build_transport("openai", "gpt-4o-mini")
        self.assertEqual(seen, [("openai", "gpt-4o-mini")])


if __name__ == "__main__":
    unittest.main()
