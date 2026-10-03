"""Core ``LLMClient`` request-shaping and result-mapping coverage.

Every test here patches ``app.services.llm_client.litellm.acompletion`` and
``app.services.llm_client.get_settings``; nothing reaches the network or the
real process configuration.
"""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.schemas.models import LLMConfigRequest, LLMProviderEnum
from app.services.llm_client import (
    CALL_FAILED_CODE,
    DEFAULT_MAX_TOKENS,
    FLOW_MAX_TOKENS_CAPS,
    JSON_MODE_CAPABLE_PROVIDERS,
    STRUCTURED_TEMPERATURE,
    TERMINAL_ERROR_CODES,
    TRUNCATED_CODE,
    LLMClient,
    _extract_usage,
    _load_task_route_map,
    _provider_supports_json_mode,
    _resolve_model,
    parse_json_object,
)

_ALL_KEYS = ["__all__"]


class _Capture:
    """Stub LiteLLM transport that records the kwargs it was called with."""

    def __init__(self, *, content="ok", finish_reason="stop", usage=None, model="mock-model"):
        self.kwargs: dict = {}
        self.calls = 0
        self._response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                    finish_reason=finish_reason,
                )
            ],
            usage=usage,
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


class ParseJsonObjectTests(unittest.TestCase):
    def test_direct_object_is_parsed(self):
        self.assertEqual(parse_json_object('{"a": 1}'), {"a": 1})

    def test_fenced_block_is_extracted(self):
        self.assertEqual(parse_json_object('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(parse_json_object('```\n{"a": 2}\n```'), {"a": 2})

    def test_prose_around_a_fenced_block_is_ignored(self):
        raw = 'Here you go:\n```json\n{"score": 7}\n```\nHope that helps!'
        self.assertEqual(parse_json_object(raw), {"score": 7})

    def test_brace_slice_recovers_a_bare_object(self):
        raw = 'Sure! {"score": 7, "verdict": "good"} -- let me know.'
        self.assertEqual(parse_json_object(raw), {"score": 7, "verdict": "good"})

    def test_empty_input_returns_empty_dict(self):
        for raw in ("", "   ", None):
            with self.subTest(raw=raw):
                self.assertEqual(parse_json_object(raw), {})

    def test_non_dict_json_returns_empty_dict(self):
        self.assertEqual(parse_json_object("[1, 2, 3]"), {})
        self.assertEqual(parse_json_object('"just a string"'), {})
        self.assertEqual(parse_json_object("```json\n[1, 2]\n```"), {})
        self.assertEqual(parse_json_object("prefix [1, 2] suffix"), {})

    def test_unparsable_brace_slice_returns_empty_dict(self):
        self.assertEqual(parse_json_object("prefix {not json} suffix"), {})

    def test_reversed_braces_are_not_treated_as_an_object(self):
        self.assertEqual(parse_json_object("}reversed{"), {})

    def test_total_garbage_returns_empty_dict(self):
        for raw in ("not json at all", "{", "}", "{unbalanced", "}{"):
            with self.subTest(raw=raw):
                self.assertEqual(parse_json_object(raw), {})

    def test_broken_fenced_block_falls_through_to_the_brace_slice(self):
        raw = '```json\n{"a": 1\n```'
        self.assertEqual(parse_json_object(raw), {})


class TaskRouteMapTests(unittest.TestCase):
    def test_blank_input_yields_an_empty_map(self):
        for raw in ("", "   ", None):
            with self.subTest(raw=raw):
                self.assertEqual(_load_task_route_map(raw), {})

    def test_invalid_json_is_ignored(self):
        self.assertEqual(_load_task_route_map("{not json"), {})
        self.assertEqual(_load_task_route_map("[1, 2]"), {})

    def test_non_dict_json_is_ignored(self):
        for raw in ("[1, 2]", '"a string"', "42"):
            with self.subTest(raw=raw):
                self.assertEqual(_load_task_route_map(raw), {})

    def test_valid_map_keeps_only_known_task_keys_with_values(self):
        raw = json.dumps(
            {
                "  PLANNER ": "gpt-4o",
                "retrieval": "  gpt-4o-mini  ",
                "eval": "",
                "final": None,
                "unknown_task": "gpt-4o",
            }
        )
        self.assertEqual(
            _load_task_route_map(raw),
            {"planner": "gpt-4o", "retrieval": "gpt-4o-mini"},
        )

    def test_keys_collapsing_to_the_same_task_keep_the_last_value(self):
        raw = json.dumps({"  PLANNER ": "first", "Planner": "last"})
        self.assertEqual(_load_task_route_map(raw), {"planner": "last"})

    def test_empty_object_yields_an_empty_map(self):
        self.assertEqual(_load_task_route_map("{}"), {})


class ResolveModelTests(unittest.TestCase):
    def test_known_provider_prefix_is_applied_once(self):
        with patch("app.services.llm_client.get_settings", return_value=_settings()):
            self.assertEqual(_resolve_model("anthropic", "claude-3-5-sonnet-20241022"), "anthropic/claude-3-5-sonnet-20241022")
            self.assertEqual(_resolve_model("anthropic", "anthropic/claude-3-5-sonnet-20241022"), "anthropic/claude-3-5-sonnet-20241022")
            self.assertEqual(_resolve_model("openai", "gpt-4o"), "gpt-4o")
            self.assertEqual(_resolve_model("GOOGLE", "gemini-2.5-pro"), "gemini/gemini-2.5-pro")

    def test_unknown_provider_has_no_prefix(self):
        with patch("app.services.llm_client.get_settings", return_value=_settings()):
            self.assertEqual(_resolve_model("ollama", "llama3.2"), "ollama/llama3.2")
            self.assertEqual(_resolve_model("mystery", "some-model"), "some-model")

    def test_default_provider_uses_the_configured_defaults(self):
        settings = _settings(default_provider="groq", default_model="llama-3.3-70b-versatile")
        with patch("app.services.llm_client.get_settings", return_value=settings):
            self.assertEqual(_resolve_model("default", ""), "groq/llama-3.3-70b-versatile")
            self.assertEqual(_resolve_model("", ""), "groq/llama-3.3-70b-versatile")

    def test_explicit_model_beats_the_configured_default_model(self):
        settings = _settings(default_provider="groq", default_model="llama-3.3-70b-versatile")
        with patch("app.services.llm_client.get_settings", return_value=settings):
            self.assertEqual(_resolve_model("default", "mixtral-8x7b-32768"), "groq/mixtral-8x7b-32768")

    def test_missing_model_falls_back_to_the_provider_catalogue(self):
        with patch("app.services.llm_client.get_settings", return_value=_settings()):
            self.assertEqual(_resolve_model("openai", ""), "gpt-4o")
            self.assertEqual(_resolve_model("mystery", ""), "gpt-4o-mini")


class ProviderCapabilityTests(unittest.TestCase):
    def test_json_mode_capable_providers(self):
        for provider in JSON_MODE_CAPABLE_PROVIDERS:
            with self.subTest(provider=provider):
                self.assertTrue(_provider_supports_json_mode(provider))
                self.assertTrue(_provider_supports_json_mode(provider.upper()))

    def test_anthropic_and_unknown_providers_are_not_json_capable(self):
        for provider in ("anthropic", "", "   ", "mystery", None):
            with self.subTest(provider=provider):
                self.assertFalse(_provider_supports_json_mode(provider))


class ExtractUsageTests(unittest.TestCase):
    def test_object_usage_is_read_from_attributes(self):
        response = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=11, output_tokens=7, total_tokens=18),
        )
        self.assertEqual(
            _extract_usage(response),
            {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
        )

    def test_dict_shaped_response_usage_is_read(self):
        response = {"usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}}
        self.assertEqual(
            _extract_usage(response),
            {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
        )

    def test_dict_usage_on_an_object_response_is_read(self):
        response = SimpleNamespace(usage={"input_tokens": 1, "total_tokens": 5})
        self.assertEqual(_extract_usage(response), {"input_tokens": 1, "total_tokens": 5})

    def test_missing_keys_are_omitted(self):
        response = SimpleNamespace(usage=SimpleNamespace(input_tokens=5))
        self.assertEqual(_extract_usage(response), {"input_tokens": 5})

    def test_absent_usage_yields_an_empty_dict(self):
        self.assertEqual(_extract_usage(SimpleNamespace(usage=None)), {})
        self.assertEqual(_extract_usage({}), {})
        self.assertEqual(_extract_usage({"usage": None}), {})
        self.assertEqual(_extract_usage(SimpleNamespace()), {})

    def test_non_numeric_values_degrade_to_zero(self):
        response = SimpleNamespace(
            usage=SimpleNamespace(input_tokens="12", output_tokens=None, total_tokens="oops"),
        )
        self.assertEqual(
            _extract_usage(response),
            {"input_tokens": 12, "output_tokens": 0, "total_tokens": 0},
        )

    def test_numeric_strings_in_dict_usage_are_coerced(self):
        self.assertEqual(
            _extract_usage({"usage": {"input_tokens": "1", "output_tokens": "2", "total_tokens": "3"}}),
            {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        )


class MessageConstructionTests(_ClientTest):
    def test_system_prompt_becomes_a_real_system_message(self):
        capture = _Capture()
        result, _ = _run(
            LLMClient(),
            LLMClient().completion("user turn", system="you are terse"),
            capture=capture,
        )
        self.assertTrue(result["success"])
        self.assertEqual(
            capture.kwargs["messages"],
            [
                {"role": "system", "content": "you are terse"},
                {"role": "user", "content": "user turn"},
            ],
        )

    def test_absent_or_blank_system_prompt_produces_only_the_user_turn(self):
        for system in ("", "   \n "):
            with self.subTest(system=system):
                capture = _Capture()
                _run(LLMClient(), LLMClient().completion("user turn", system=system), capture=capture)
                self.assertEqual(capture.kwargs["messages"], [{"role": "user", "content": "user turn"}])


class StructuredModeTests(_ClientTest):
    def test_structured_mode_is_requested_for_a_json_capable_provider(self):
        capture = _Capture()
        _run(
            LLMClient(),
            LLMClient().completion(
                "prompt",
                LLMConfigRequest(provider=LLMProviderEnum.OPENAI, model="gpt-4o"),
                structured=True,
            ),
            capture=capture,
        )
        self.assertEqual(capture.kwargs["response_format"], {"type": "json_object"})

    def test_structured_mode_is_omitted_for_anthropic(self):
        capture = _Capture()
        _run(
            LLMClient(),
            LLMClient().completion(
                "prompt",
                LLMConfigRequest(provider=LLMProviderEnum.ANTHROPIC, model="claude-3-5-sonnet-20241022"),
                structured=True,
            ),
            capture=capture,
        )
        self.assertNotIn("response_format", capture.kwargs)

    def test_structured_mode_uses_the_structured_temperature(self):
        capture = _Capture()
        _run(LLMClient(), LLMClient().completion("prompt", structured=True), capture=capture)
        self.assertEqual(capture.kwargs["temperature"], STRUCTURED_TEMPERATURE)

    def test_free_form_mode_uses_the_default_temperature(self):
        capture = _Capture()
        _run(LLMClient(), LLMClient().completion("prompt"), capture=capture)
        self.assertEqual(capture.kwargs["temperature"], 0.7)

    def test_tools_win_over_structured_mode(self):
        capture = _Capture()
        schemas = [{"type": "function", "function": {"name": "t"}}]
        with self.assertLogs("app.services.llm_client", level="WARNING") as logs:
            result, _ = _run(
                LLMClient(),
                LLMClient().completion("prompt", structured=True, tools=schemas),
                capture=capture,
            )
        self.assertTrue(result["success"])
        self.assertEqual(capture.kwargs["tools"], schemas)
        self.assertNotIn("response_format", capture.kwargs)
        self.assertTrue(any("dropping response_format" in line for line in logs.output))
        # Dropping response_format also drops the structured temperature floor.
        self.assertEqual(capture.kwargs["temperature"], 0.7)

    def test_tool_choice_is_only_sent_with_tools(self):
        capture = _Capture()
        schemas = [{"type": "function", "function": {"name": "t"}}]
        _run(LLMClient(), LLMClient().completion("prompt", tool_choice={"any": 1}), capture=capture)
        self.assertNotIn("tool_choice", capture.kwargs)

        capture = _Capture()
        _run(
            LLMClient(),
            LLMClient().completion("prompt", tools=schemas, tool_choice={"any": 1}),
            capture=capture,
        )
        self.assertEqual(capture.kwargs["tool_choice"], {"any": 1})


class TemperatureResolutionTests(_ClientTest):
    def _temperature(self, **config_kwargs):
        capture = _Capture()
        config = LLMConfigRequest(**config_kwargs) if config_kwargs else None
        _run(LLMClient(), LLMClient().completion("prompt", config), capture=capture)
        return capture.kwargs["temperature"]

    def test_explicit_temperature_is_used(self):
        self.assertEqual(self._temperature(provider=LLMProviderEnum.OPENAI, temperature=0.42), 0.42)

    def test_absent_temperature_falls_back_to_the_flow_default(self):
        self.assertEqual(self._temperature(provider=LLMProviderEnum.OPENAI), 0.7)

    def test_zero_temperature_means_unset_and_falls_back(self):
        # The field default cannot express "explicitly deterministic", so 0.0 is
        # deliberately treated as unset.
        self.assertEqual(self._temperature(provider=LLMProviderEnum.OPENAI, temperature=0.0), 0.7)
        self.assertEqual(
            self._temperature(provider=LLMProviderEnum.OPENAI, temperature=0.0, structured=None),
            0.7,
        )

    def test_configured_default_temperature_is_used_when_available(self):
        capture = _Capture()
        _run(
            LLMClient(),
            LLMClient().completion("prompt"),
            settings=_settings(default_temperature=0.55),
            capture=capture,
        )
        self.assertEqual(capture.kwargs["temperature"], 0.55)


class MaxTokensTests(_ClientTest):
    def _max_tokens(self, *, task=None, cap=None, flow=None, config=None, settings=None):
        capture = _Capture()
        _run(
            LLMClient(),
            LLMClient().completion(
                "prompt", config, task=task, max_tokens_cap=cap, flow=flow or ""
            ),
            settings=settings,
            capture=capture,
        )
        return capture.kwargs["max_tokens"]

    def test_default_is_the_module_default(self):
        self.assertEqual(self._max_tokens(), DEFAULT_MAX_TOKENS)

    def test_flow_selects_its_cap_from_the_table(self):
        for flow, expected in FLOW_MAX_TOKENS_CAPS.items():
            with self.subTest(flow=flow):
                self.assertEqual(self._max_tokens(flow=flow), expected)

    def test_flow_name_is_case_and_whitespace_insensitive(self):
        self.assertEqual(self._max_tokens(flow="  Voice_Turn "), FLOW_MAX_TOKENS_CAPS["voice_turn"])

    def test_unknown_flow_falls_back_to_the_module_default(self):
        self.assertEqual(self._max_tokens(flow="no_such_flow"), DEFAULT_MAX_TOKENS)

    def test_task_does_not_select_a_cap(self):
        # `task` is normalised against the routing taxonomy
        # {"planner","retrieval","eval","final"}, which is disjoint from the
        # flow names, so it never picks up a flow ceiling.
        self.assertEqual(set(FLOW_MAX_TOKENS_CAPS) & {"planner", "retrieval", "eval", "final"}, set())
        for task in ("planner", "retrieval", "eval", "final"):
            with self.subTest(task=task):
                self.assertEqual(self._max_tokens(task=task), DEFAULT_MAX_TOKENS)

    def test_task_and_flow_are_independent(self):
        # The flow ceiling applies regardless of which task routes the model.
        self.assertEqual(self._max_tokens(task="eval", flow="interview_eval"), 2000)
        self.assertEqual(self._max_tokens(task="final", flow="interview_eval"), 2000)

    def test_caller_cap_clamps_the_default_down(self):
        self.assertEqual(self._max_tokens(cap=600), 600)

    def test_task_without_a_flow_cap_keeps_the_requested_value(self):
        config = LLMConfigRequest(provider=LLMProviderEnum.OPENAI, max_tokens=2000)
        self.assertEqual(self._max_tokens(task="final", config=config), 2000)

    def test_zero_requested_max_tokens_is_ignored(self):
        config = LLMConfigRequest(provider=LLMProviderEnum.OPENAI, max_tokens=0)
        self.assertEqual(self._max_tokens(config=config), DEFAULT_MAX_TOKENS)


class TaskRoutingTests(_ClientTest):
    def test_unknown_task_falls_back_to_final(self):
        capture = _Capture()
        result, _ = _run(
            LLMClient(),
            LLMClient().completion("prompt", task="  NOT_A_TASK  "),
            settings=_settings(
                llm_task_routing_enabled=True,
                llm_task_route_map_json=json.dumps({"final": "gpt-4o"}),
            ),
            capture=capture,
        )
        self.assertEqual(result["metadata"]["task"], "final")
        self.assertEqual(capture.kwargs["model"], "gpt-4o")

    def test_empty_task_is_reported_as_final(self):
        capture = _Capture()
        result, _ = _run(LLMClient(), LLMClient().completion("prompt", task=None), capture=capture)
        self.assertEqual(result["metadata"]["task"], "final")

    def test_route_map_without_an_entry_for_the_task_keeps_the_resolved_model(self):
        capture = _Capture()
        result, _ = _run(
            LLMClient(),
            LLMClient().completion("prompt", task="eval"),
            settings=_settings(
                llm_task_routing_enabled=True,
                llm_task_route_map_json=json.dumps({"final": "gpt-4o"}),
            ),
            capture=capture,
        )
        self.assertEqual(result["metadata"]["model"], "gpt-4o-mini")

    def test_invalid_route_map_json_keeps_the_resolved_model(self):
        capture = _Capture()
        result, _ = _run(
            LLMClient(),
            LLMClient().completion("prompt", task="eval"),
            settings=_settings(
                llm_task_routing_enabled=True,
                llm_task_route_map_json="{not json",
            ),
            capture=capture,
        )
        self.assertEqual(result["metadata"]["model"], "gpt-4o-mini")

    def test_task_route_map_reaches_each_documented_task(self):
        route_map = {key: "gpt-4o-mini" for key in ("planner", "retrieval", "eval", "final")}
        for task in route_map:
            with self.subTest(task=task):
                capture = _Capture()
                result, _ = _run(
                    LLMClient(),
                    LLMClient().completion("prompt", task=task),
                    settings=_settings(
                        llm_task_routing_enabled=True,
                        llm_task_route_map_json=json.dumps(route_map),
                    ),
                    capture=capture,
                )
                self.assertEqual(result["metadata"]["task"], task)
                self.assertEqual(capture.kwargs["model"], "gpt-4o-mini")

    def test_prompt_text_never_influences_routing(self):
        # Routing keys off the explicit `task` argument only; a prompt that
        # mentions a route map must not change the model.
        capture = _Capture()
        result, _ = _run(
            LLMClient(),
            LLMClient().completion('{"eval": "gpt-9-imaginary"}'),
            settings=_settings(
                llm_task_routing_enabled=True,
                llm_task_route_map_json=json.dumps({"eval": "gpt-4o"}),
            ),
            capture=capture,
        )
        self.assertEqual(result["metadata"]["model"], "gpt-4o-mini")


class ResultMappingTests(_ClientTest):
    def test_successful_result_shape(self):
        result, _ = _run(LLMClient(), LLMClient().completion("prompt"))
        self.assertEqual(result["success"], True)
        self.assertEqual(result["analysis"], "ok")
        self.assertEqual(result["error"], "")
        self.assertEqual(result["error_code"], "")
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["usage"], {})
        self.assertEqual(
            result["metadata"],
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "task": "final",
                "credential_source": "backend",
                "usage": {},
            },
        )
        self.assertEqual(sorted(_ALL_KEYS), sorted(["__all__"]))

    def test_null_content_is_treated_as_empty_text(self):
        result, _ = _run(LLMClient(), LLMClient().completion("prompt"), capture=_Capture(content=None))
        self.assertEqual(result["analysis"], "")
        self.assertTrue(result["success"])

    def test_truncated_response_is_a_distinct_terminal_outcome(self):
        result, _ = _run(
            LLMClient(),
            LLMClient().completion("prompt", max_tokens_cap=600),
            capture=_Capture(content="half an ans", finish_reason="LENGTH"),
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], TRUNCATED_CODE)
        self.assertIn(TRUNCATED_CODE, TERMINAL_ERROR_CODES)
        # The partial text is still returned so the caller can decide.
        self.assertEqual(result["analysis"], "half an ans")
        self.assertEqual(result["finish_reason"], "LENGTH")
        self.assertIn("truncated at max_tokens=600", result["error"])

    def test_usage_is_mapped_into_the_result_and_metadata(self):
        usage = SimpleNamespace(input_tokens=11, output_tokens=7, total_tokens=18)
        result, _ = _run(
            LLMClient(),
            LLMClient().completion("prompt"),
            capture=_Capture(usage=usage),
        )
        expected = {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}
        self.assertEqual(result["usage"], expected)
        self.assertEqual(result["metadata"]["usage"], expected)
        self.assertEqual(result["metadata"]["input_tokens"], 11)
        self.assertEqual(result["metadata"]["output_tokens"], 7)
        self.assertEqual(result["metadata"]["total_tokens"], 18)

    def test_dict_shaped_usage_is_mapped_too(self):
        class _DictUsage(dict):
            """Dict usage on a normal response object."""

        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
            usage=_DictUsage(input_tokens=1, output_tokens=2, total_tokens=3),
            model="mock-model",
        )

        async def _transport(**_kwargs):
            return response

        result, _ = _run(LLMClient(), LLMClient().completion("prompt"), capture=_transport)
        self.assertEqual(result["usage"], {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3})
        self.assertEqual(result["metadata"]["total_tokens"], 3)

    def test_missing_finish_reason_defaults_to_empty(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=None,
        )

        async def _transport(**_kwargs):
            return response

        result, _ = _run(LLMClient(), LLMClient().completion("prompt"), capture=_transport)
        self.assertEqual(result["finish_reason"], "")
        self.assertTrue(result["success"])

    def test_provider_exception_becomes_a_call_failure(self):
        async def _boom(**_kwargs):
            raise RuntimeError("upstream 503")

        result, _ = _run(LLMClient(), LLMClient().completion("prompt"), capture=_boom)
        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], CALL_FAILED_CODE)
        self.assertNotIn(CALL_FAILED_CODE, TERMINAL_ERROR_CODES)
        self.assertEqual(result["error"], "upstream 503")
        self.assertEqual(result["analysis"], "")
        self.assertEqual(result["usage"], {})
        self.assertEqual(result["metadata"]["model"], "gpt-4o-mini")

    def test_background_context_reports_the_backend_credential_source(self):
        result, _ = _run(LLMClient(), LLMClient().completion("prompt"))
        self.assertEqual(result["metadata"]["credential_source"], "backend")

    def test_identity_without_a_store_still_uses_backend_credentials(self):
        result, _ = _run(
            LLMClient(user_settings_store=None),
            LLMClient().completion("prompt", user_identity={"provider": "google", "user": "a@x.com"}),
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["metadata"]["credential_source"], "backend")


class HealthCheckTests(_ClientTest):
    def test_healthy_provider_reports_the_resolved_model(self):
        capture = _Capture(model="gpt-4o-2024")
        result, _ = _run(LLMClient(), LLMClient().health_check("openai"), capture=capture)
        self.assertEqual(result, {"healthy": True, "service_name": "openai", "version": "gpt-4o-2024"})
        self.assertEqual(capture.kwargs["model"], "gpt-4o")
        self.assertEqual(capture.kwargs["messages"], [{"role": "user", "content": "ping"}])
        self.assertEqual(capture.kwargs["max_tokens"], 5)
        self.assertEqual(capture.kwargs["timeout"], 10)

    def test_health_check_uses_the_first_suggestion_for_the_provider(self):
        capture = _Capture()
        _run(LLMClient(), LLMClient().health_check("anthropic"), capture=capture)
        self.assertEqual(capture.kwargs["model"], "anthropic/claude-3-5-sonnet-20241022")

    def test_unknown_provider_health_check_falls_back_to_a_default_model(self):
        capture = _Capture()
        result, _ = _run(LLMClient(), LLMClient().health_check("mystery-provider"), capture=capture)
        self.assertEqual(capture.kwargs["model"], "gpt-4o-mini")
        self.assertTrue(result["healthy"])

    def test_version_falls_back_to_the_requested_model_when_not_reported(self):
        async def _transport(**_kwargs):
            return SimpleNamespace(choices=[])

        result, _ = _run(LLMClient(), LLMClient().health_check("openai"), capture=_transport)
        self.assertEqual(result["version"], "gpt-4o")
        self.assertTrue(result["healthy"])

    def test_failing_provider_is_reported_as_unhealthy(self):
        async def _boom(**_kwargs):
            raise RuntimeError("connection refused")

        result, _ = _run(LLMClient(), LLMClient().health_check("groq"), capture=_boom)
        self.assertEqual(result, {"healthy": False, "service_name": "groq", "version": ""})

    def test_default_provider_is_used_when_none_is_given(self):
        capture = _Capture()
        result, _ = _run(LLMClient(), LLMClient().health_check(), capture=capture)
        self.assertEqual(result["service_name"], "openai")


if __name__ == "__main__":
    unittest.main()