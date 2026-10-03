"""Coverage for app/services/tools/agentic.py.

Complements ``tests/test_tools_agentic_loop.py`` (loop termination paths and the
capability guard) with the payload/response *parsing* branches -- object-shaped
vs dict-shaped provider payloads -- plus the ``succeeded`` / ``usage`` surfaces
of the loop result.
"""

import asyncio
import json
import unittest

from app.services.tools.agentic import (
    STOP_MAX_STEPS,
    STOP_MODEL_ERROR,
    STOP_NO_TOOL_CALLS,
    ModelTurn,
    ToolCall,
    ToolLoopResult,
    bare_model_name,
    provider_supports_tools,
    run_tool_loop,
)
from app.services.tools.registry import ToolRegistry, ToolResult, ToolSpec

_PARAMS = {"type": "object", "properties": {"value": {"type": "string"}}}


def _run(coro):
    return asyncio.run(coro)


class _Function:
    def __init__(self, name="", arguments=None):
        self.name = name
        self.arguments = arguments


class _Payload:
    """An object-shaped ``tool_calls`` entry with a top-level ``name``."""

    def __init__(self, call_id="", name="", function=None):
        self.id = call_id
        self.name = name
        self.function = function


class ToolCallFromPayloadTests(unittest.TestCase):
    def test_none_payload_is_rejected(self):
        self.assertIsNone(ToolCall.from_payload(None, index=3))

    def test_none_entry_inside_a_tool_calls_list_is_skipped(self):
        turn = ModelTurn.from_response(
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {"content": "", "tool_calls": [None, {"name": "n"}]},
                    }
                ]
            }
        )
        self.assertEqual([call.name for call in turn.tool_calls], ["n"])

    def test_dict_id_falls_back_to_tool_call_id_then_index(self):
        by_alias = ToolCall.from_payload({"tool_call_id": "alias", "name": "n"})
        self.assertEqual(by_alias.id, "alias")
        generated = ToolCall.from_payload({"name": "n"}, index=7)
        self.assertEqual(generated.id, "call_7")
        self.assertEqual(generated.index, 7)

    def test_dict_arguments_win_over_the_nested_function_arguments(self):
        call = ToolCall.from_payload(
            {
                "id": "c1",
                "name": "top",
                "arguments": '{"from": "top"}',
                "function": {"name": "nested", "arguments": '{"from": "nested"}'},
            }
        )
        self.assertEqual(call.name, "nested")
        self.assertEqual(json.loads(call.arguments), {"from": "top"})

    def test_dict_arguments_default_to_the_nested_function_arguments(self):
        call = ToolCall.from_payload(
            {"id": "c1", "name": "n", "function": {"arguments": '{"a": 1}'}}
        )
        self.assertEqual(json.loads(call.arguments), {"a": 1})

    def test_absent_arguments_become_the_empty_string(self):
        self.assertEqual(ToolCall.from_payload({"id": "c1", "name": "n"}).arguments, "")

    def test_dict_entry_without_a_name_is_rejected(self):
        self.assertIsNone(ToolCall.from_payload({"id": "c1"}))
        self.assertIsNone(
            ToolCall.from_payload({"id": "c1", "function": {"arguments": "{}"}})
        )

    def test_object_payload_reads_id_name_and_function(self):
        call = ToolCall.from_payload(
            _Payload(
                call_id="obj1",
                name="lookup",
                function=_Function(arguments='{"v": 2}'),
            )
        )
        self.assertEqual(call.id, "obj1")
        self.assertEqual(call.name, "lookup")
        self.assertEqual(json.loads(call.arguments), {"v": 2})

    def test_object_payload_without_an_id_falls_back_to_the_index(self):
        call = ToolCall.from_payload(
            _Payload(name="lookup", function=_Function(arguments="{}")), index=2
        )
        self.assertEqual(call.id, "call_2")
        self.assertEqual(call.index, 2)

    def test_object_payload_without_a_function_has_no_arguments(self):
        call = ToolCall.from_payload(_Payload(call_id="obj2", name="n"))
        self.assertEqual(call.arguments, "")
        self.assertEqual(call.name, "n")

    def test_object_payload_without_a_name_is_rejected(self):
        self.assertIsNone(ToolCall.from_payload(_Payload(call_id="obj3")))

    def test_non_string_object_arguments_are_json_encoded(self):
        call = ToolCall.from_payload(
            _Payload(
                call_id="c", name="n", function=_Function(arguments={"a": 1})
            )
        )
        self.assertEqual(json.loads(call.arguments), {"a": 1})

    def test_unserialisable_object_arguments_fall_back_to_str(self):
        cycle = {}
        cycle["self"] = cycle
        call = ToolCall.from_payload(
            _Payload(call_id="c", name="n", function=_Function(arguments=cycle))
        )
        self.assertEqual(call.arguments, str(cycle))

    def test_message_fragment_is_openai_shaped(self):
        call = ToolCall(id="c1", name="lookup", arguments='{"v": 1}')
        self.assertEqual(
            call.to_message_fragment(),
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"v": 1}'},
            },
        )


class ModelTurnFromResponseTests(unittest.TestCase):
    def test_empty_choices_list_is_no_choices(self):
        self.assertEqual(ModelTurn.from_response({"choices": []}).error, "no_choices")

    def test_choice_without_a_message_is_reported(self):
        turn = ModelTurn.from_response({"choices": [{"finish_reason": "stop"}]})
        self.assertFalse(turn.success)
        self.assertEqual(turn.error, "no_message")

    def test_dict_message_without_tool_calls_is_a_text_turn(self):
        turn = ModelTurn.from_response(
            {"choices": [{"message": {"content": "text only"}, "finish_reason": "stop"}]}
        )
        self.assertTrue(turn.success)
        self.assertEqual(turn.text, "text only")
        self.assertEqual(turn.tool_calls, ())
        self.assertEqual(turn.finish_reason, "stop")

    def test_missing_content_reads_as_the_empty_string(self):
        turn = ModelTurn.from_response({"choices": [{"message": {}}]})
        self.assertEqual(turn.text, "")
        self.assertEqual(turn.finish_reason, "")

    def test_object_shaped_response_with_tool_calls(self):
        class _Message:
            content = "thinking"
            tool_calls = [
                _Payload(
                    call_id="c1", name="lookup", function=_Function(arguments='{"v": 1}')
                )
            ]

        class _Choice:
            message = _Message()
            finish_reason = "tool_calls"

        class _Response:
            choices = [_Choice()]

        turn = ModelTurn.from_response(_Response())
        self.assertEqual(turn.text, "thinking")
        self.assertEqual(turn.finish_reason, "tool_calls")
        self.assertEqual(turn.tool_calls[0].name, "lookup")

    def test_null_tool_calls_is_treated_as_no_tool_calls(self):
        turn = ModelTurn.from_response(
            {"choices": [{"message": {"content": "x", "tool_calls": None}}]}
        )
        self.assertEqual(turn.tool_calls, ())

    def test_dict_entries_mixing_shapes_are_all_read(self):
        turn = ModelTurn.from_response(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "a",
                                    "function": {
                                        "name": "one",
                                        "arguments": '{"i": 1}',
                                    },
                                },
                                {"tool_call_id": "b", "name": "two"},
                            ]
                        }
                    }
                ]
            }
        )
        self.assertEqual([(c.id, c.name) for c in turn.tool_calls], [("a", "one"), ("b", "two")])


class BareModelNameTests(unittest.TestCase):
    def test_only_the_last_path_segment_counts(self):
        self.assertEqual(bare_model_name("openrouter/vendor/gemma2-9b"), "gemma2-9b")

    def test_whitespace_and_case_are_normalised(self):
        self.assertEqual(bare_model_name("  GROQ/Gemma2-9B  "), "gemma2-9b")

    def test_none_is_treated_as_empty(self):
        self.assertEqual(bare_model_name(None), "")


class CapabilityGuardTests(unittest.TestCase):
    def test_every_denylisted_pattern_blocks_its_model(self):
        blocked = [
            ("groq", "gemma2-9b-it"),
            ("groq", "mixtral-8x7b"),
            ("openai", "deepseek-r1"),
            ("openai", "o1-preview"),
            ("openai", "o1"),
            ("openai", "text-embedding-3-small"),
        ]
        for provider, model in blocked:
            self.assertFalse(provider_supports_tools(provider, model), model)

    def test_an_ordinary_model_on_a_capable_provider_passes(self):
        self.assertTrue(provider_supports_tools("openai", "gpt-4o-mini"))
        self.assertTrue(provider_supports_tools("google", "gemini-2.0-flash"))

    def test_provider_matching_is_case_and_space_insensitive(self):
        self.assertTrue(provider_supports_tools("  OpenAI  ", "gpt-4o-mini"))

    def test_blank_model_falls_back_to_the_provider_verdict(self):
        # An unpinned model cannot be checked against the denylist, so the
        # provider's own verdict stands.
        self.assertTrue(provider_supports_tools("groq"))
        self.assertTrue(provider_supports_tools("groq", "   "))
        self.assertTrue(provider_supports_tools("groq", None))
        self.assertFalse(provider_supports_tools("ollama", "   "))


class LoopResultTests(unittest.TestCase):
    def test_succeeded_requires_a_stop_reason_and_text(self):
        ok = ToolResult(tool_name="t", tool_call_id="c", ok=True, content="body")
        cases = [
            ("no_tool_calls", "answer", True),
            ("max_steps", "partial", True),
            ("no_tool_calls", "", False),
            ("model_error", "answer", False),
            ("unknown_tool", "answer", False),
            ("all_tool_calls_failed", "answer", False),
        ]
        for reason, text, expected in cases:
            result = ToolLoopResult(
                text=text, steps=1, stopped_reason=reason, tool_results=(ok,)
            )
            self.assertEqual(result.succeeded, expected, reason)

    def test_tool_context_joins_only_successful_non_empty_results(self):
        results = (
            ToolResult(tool_name="a", tool_call_id="1", ok=True, content="alpha"),
            ToolResult(
                tool_name="b", tool_call_id="2", ok=False, content="", error="tool_failed"
            ),
            ToolResult(tool_name="c", tool_call_id="3", ok=True, content=""),
            ToolResult(tool_name="d", tool_call_id="4", ok=True, content="delta"),
        )
        result = ToolLoopResult(
            text="", steps=1, stopped_reason=STOP_NO_TOOL_CALLS, tool_results=results
        )
        self.assertEqual(result.tool_context, "alpha\n\ndelta")


class _ScriptedTransport:
    def __init__(self, turns):
        self.turns = list(turns)
        self.calls: list[dict] = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        if not self.turns:
            return ModelTurn(text="exhausted")
        return self.turns.pop(0)


def _registry(**overrides):
    async def lookup(arguments):
        return f"looked-up:{arguments.get('value', '')}"

    return ToolRegistry(
        [
            ToolSpec(
                name="lookup",
                description="look something up",
                parameters=_PARAMS,
                handler=lookup,
                **overrides,
            )
        ]
    )


def _call(name="lookup", value="v", call_id="c1"):
    return ToolCall(id=call_id, name=name, arguments=json.dumps({"value": value}))


class LiteLLMObjectShapeTests(unittest.TestCase):
    """How the real LiteLLM tool-call object is parsed.

    KNOWN BUG (app/services/tools/agentic.py:123, the non-dict branch of
    ``ToolCall.from_payload``): the object branch reads the tool name only from
    the top-level ``payload.name`` attribute and never falls back to
    ``payload.function.name`` the way the dict branch does (lines 117-120).
    ``LiteLLMToolTransport`` -- the production transport -- returns real
    LiteLLM response objects, and ``litellm.types.utils.ChatCompletionMessageToolCall``
    carries no top-level ``name``. The consequence is that every tool call the
    model emits is parsed as nameless and dropped, so the loop exits
    ``no_tool_calls`` on its first step and no tool ever runs in production.

    This test pins the current behaviour with the real class so the fix is a
    visible, deliberate change. It is NOT an endorsement of the behaviour.
    """

    def _real_tool_call(self):
        from litellm.types.utils import (
            ChatCompletionMessageToolCall,
            Function,
        )

        return ChatCompletionMessageToolCall(
            id="call_1",
            function=Function(name="lookup", arguments='{"value": "X"}'),
        )

    def test_real_litellm_tool_call_carries_no_top_level_name(self):
        payload = self._real_tool_call()
        self.assertFalse(hasattr(payload, "name"))
        self.assertEqual(payload.function.name, "lookup")

    def test_a_real_litellm_tool_call_parses_its_name_from_the_function(self):
        # `ChatCompletionMessageToolCall` has no top-level `.name`; the name
        # lives only on `.function.name`. Reading just `payload.name` would
        # discard every real tool call and silently exit the loop, so the
        # non-dict branch must mirror the dict branch's fallback.
        self.assertFalse(hasattr(self._real_tool_call(), "name"))
        call = ToolCall.from_payload(self._real_tool_call())
        self.assertIsNotNone(call)
        self.assertEqual(call.name, "lookup")
        self.assertEqual(call.id, "call_1")
        self.assertEqual(call.arguments, '{"value": "X"}')

    def test_the_loop_runs_the_tool_named_by_a_real_litellm_call(self):
        seen: list[dict] = []

        async def lookup(arguments):
            seen.append(arguments)
            return {"ok": True}

        registry = ToolRegistry(
            [
                ToolSpec(
                    name="lookup",
                    description="d",
                    parameters=_PARAMS,
                    handler=lookup,
                )
            ]
        )
        real_call = self._real_tool_call()

        class _Message:
            content = "thinking"
            tool_calls = [real_call]

        class _Choice:
            message = _Message()
            finish_reason = "tool_calls"

        class _Response:
            choices = [_Choice()]

        transport = _ScriptedTransport(
            [ModelTurn.from_response(_Response()), ModelTurn(text="done")]
        )
        result = _run(
            run_tool_loop(
                transport=transport, registry=registry, prompt="p", max_steps=3
            )
        )
        self.assertEqual(seen, [{"value": "X"}])
        self.assertEqual(len(result.tool_results), 1)
        self.assertTrue(result.tool_results[0].ok)
        self.assertEqual(result.stopped_reason, STOP_NO_TOOL_CALLS)
        self.assertEqual(len(transport.calls), 2)


class LoopAccountingTests(unittest.TestCase):
    def test_provider_usage_is_reported_on_the_result(self):
        transport = _ScriptedTransport(
            [
                ModelTurn(
                    text="",
                    tool_calls=(_call(),),
                    usage={"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
                ),
                ModelTurn(
                    text="done",
                    usage={"input_tokens": 7, "output_tokens": 2, "total_tokens": 9},
                ),
            ]
        )
        result = _run(
            run_tool_loop(
                transport=transport,
                registry=_registry(),
                prompt="p",
                max_steps=3,
            )
        )
        self.assertEqual(result.stopped_reason, STOP_NO_TOOL_CALLS)
        # The last reported usage wins.
        self.assertEqual(result.usage["total_tokens"], 9)

    def test_usage_is_preserved_when_a_later_step_omits_it(self):
        transport = _ScriptedTransport(
            [
                ModelTurn(text="", tool_calls=(_call(),), usage={"total_tokens": 6}),
                ModelTurn(text="done"),
            ]
        )
        result = _run(
            run_tool_loop(
                transport=transport, registry=_registry(), prompt="p", max_steps=3
            )
        )
        self.assertEqual(result.usage["total_tokens"], 6)

    def test_usage_is_carried_on_a_max_steps_exit(self):
        transport = _ScriptedTransport(
            [
                ModelTurn(text="s1", tool_calls=(_call(),), usage={"total_tokens": 3})
            ]
        )
        result = _run(
            run_tool_loop(
                transport=transport, registry=_registry(), prompt="p", max_steps=1
            )
        )
        self.assertEqual(result.stopped_reason, STOP_MAX_STEPS)
        self.assertEqual(result.usage["total_tokens"], 3)

    def test_none_tool_choice_is_forwarded_unchanged(self):
        transport = _ScriptedTransport([ModelTurn(text="done")])
        _run(
            run_tool_loop(
                transport=transport,
                registry=_registry(),
                prompt="p",
                tool_choice=None,
                max_steps=1,
            )
        )
        self.assertIsNone(transport.calls[0]["tool_choice"])

    def test_per_step_call_cap_is_floored_at_one(self):
        transport = _ScriptedTransport(
            [ModelTurn(text="", tool_calls=(_call(call_id="a"), _call(call_id="b")))]
        )
        result = _run(
            run_tool_loop(
                transport=transport,
                registry=_registry(),
                prompt="p",
                max_steps=1,
                max_tool_calls_per_step=0,
            )
        )
        self.assertEqual(len(result.tool_results), 1)

    def test_blank_prompt_is_still_sent_as_a_user_message(self):
        transport = _ScriptedTransport([ModelTurn(text="done")])
        _run(
            run_tool_loop(
                transport=transport, registry=_registry(), prompt="", max_steps=1
            )
        )
        self.assertEqual(
            [m for m in transport.calls[0]["messages"]],
            [{"role": "user", "content": ""}],
        )

    def test_blank_system_prompt_is_omitted(self):
        transport = _ScriptedTransport([ModelTurn(text="done")])
        _run(
            run_tool_loop(
                transport=transport,
                registry=_registry(),
                prompt="p",
                system="   ",
                max_steps=1,
            )
        )
        self.assertEqual(
            [m["role"] for m in transport.calls[0]["messages"]], ["user"]
        )

    def test_no_enabled_tools_still_returns_the_prompt_transcript(self):
        registry = ToolRegistry(
            [
                ToolSpec(
                    name="off",
                    description="d",
                    parameters=_PARAMS,
                    handler=lambda a: None,
                    enabled=False,
                )
            ]
        )
        transport = _ScriptedTransport([ModelTurn(text="unused")])
        result = _run(
            run_tool_loop(
                transport=transport,
                registry=registry,
                prompt="p",
                system="sys",
                max_steps=3,
            )
        )
        self.assertEqual(result.steps, 0)
        self.assertEqual(result.tool_results, ())
        self.assertEqual(
            list(result.messages),
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "p"},
            ],
        )

    def test_mixed_unknown_and_disabled_calls_stop_the_loop(self):
        registry = ToolRegistry(
            [
                ToolSpec(
                    name="other",
                    description="d",
                    parameters=_PARAMS,
                    handler=lambda a: "ok",
                ),
                ToolSpec(
                    name="lookup",
                    description="d",
                    parameters=_PARAMS,
                    handler=lambda a: "should-not-run",
                    enabled=False,
                ),
            ]
        )
        transport = _ScriptedTransport(
            [
                ModelTurn(
                    text="",
                    tool_calls=(_call(name="ghost", call_id="g"), _call(call_id="d")),
                )
            ]
        )
        result = _run(
            run_tool_loop(
                transport=transport, registry=registry, prompt="p", max_steps=3
            )
        )
        # Not every call was *unknown*, so the stop reason is the generic
        # "all calls failed" rather than "unknown_tool".
        self.assertEqual(result.stopped_reason, "all_tool_calls_failed")
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(
            [r.error.split(" ")[0] for r in result.tool_results],
            ["unknown_tool", "tool_disabled"],
        )

    def test_a_failed_tool_alongside_a_successful_one_keeps_the_loop_running(self):
        async def boom(arguments):
            raise RuntimeError("nope")

        registry = ToolRegistry(
            [
                ToolSpec(
                    name="lookup",
                    description="d",
                    parameters=_PARAMS,
                    handler=lambda a: "ok",
                ),
                ToolSpec(
                    name="boom",
                    description="d",
                    parameters=_PARAMS,
                    handler=boom,
                ),
            ]
        )
        transport = _ScriptedTransport(
            [
                ModelTurn(
                    text="", tool_calls=(_call(call_id="a"), _call(name="boom", call_id="b"))
                ),
                ModelTurn(text="recovered"),
            ]
        )
        result = _run(
            run_tool_loop(
                transport=transport, registry=registry, prompt="p", max_steps=3
            )
        )
        self.assertEqual(result.stopped_reason, STOP_NO_TOOL_CALLS)
        self.assertEqual(result.text, "recovered")
        self.assertEqual(len(result.tool_results), 2)
        self.assertTrue(result.tool_results[0].ok)
        self.assertFalse(result.tool_results[1].ok)

    def test_a_failed_step_keeps_the_transcript_and_partial_results(self):
        class _FailingTransport:
            def __init__(self):
                self.calls = 0

            async def complete(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return ModelTurn(
                        text="partial", tool_calls=(_call(),), usage={"total_tokens": 2}
                    )
                raise RuntimeError("provider died")

        transport = _FailingTransport()
        result = _run(
            run_tool_loop(
                transport=transport, registry=_registry(), prompt="p", max_steps=4
            )
        )
        self.assertEqual(result.stopped_reason, STOP_MODEL_ERROR)
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.text, "partial")
        self.assertEqual(len(result.tool_results), 1)
        self.assertEqual(result.usage["total_tokens"], 2)
        self.assertEqual(
            [m["role"] for m in result.messages], ["user", "assistant", "tool"]
        )


if __name__ == "__main__":
    unittest.main()