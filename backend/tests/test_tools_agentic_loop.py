"""The bounded tool loop: tool selection, feedback, and every termination path."""

import asyncio
import json
import unittest
from typing import Any

from app.services.tools.agentic import (
    STOP_ALL_CALLS_FAILED,
    STOP_MAX_STEPS,
    STOP_MODEL_ERROR,
    STOP_NO_TOOL_CALLS,
    STOP_UNKNOWN_TOOL,
    TOOL_CAPABLE_PROVIDERS,
    ModelTurn,
    ToolCall,
    bare_model_name,
    provider_supports_tools,
    run_tool_loop,
)
from app.services.tools.registry import ToolRegistry, ToolSpec

_PARAMS = {
    "type": "object",
    "properties": {"value": {"type": "string"}},
    "required": ["value"],
}


def _call(name="lookup", value="v", call_id="c1"):
    return ToolCall(id=call_id, name=name, arguments=json.dumps({"value": value}))


class ScriptedTransport:
    """Replays a fixed list of turns and records every request it received."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls: list[dict[str, Any]] = []
        self.raise_on: int | None = None

    async def complete(self, **kwargs):
        index = len(self.calls)
        self.calls.append(kwargs)
        if self.raise_on is not None and index == self.raise_on:
            raise RuntimeError("transport down")
        if index >= len(self.turns):
            return ModelTurn(text="exhausted", success=True)
        return self.turns[index]


def _registry(**spec_kwargs):
    async def lookup(arguments):
        return f"looked-up:{arguments.get('value', '')}"

    async def boom(arguments):
        raise RuntimeError("tool exploded")

    specs = {
        "lookup": ToolSpec(
            name="lookup",
            description="look something up",
            parameters=_PARAMS,
            handler=lookup,
            enabled=spec_kwargs.get("lookup_enabled", True),
        ),
        "boom": ToolSpec(
            name="boom",
            description="always fails",
            parameters=_PARAMS,
            handler=boom,
        ),
    }
    return ToolRegistry(list(specs.values()))


def _run(coro):
    return asyncio.run(coro)


class CapabilityGuardTests(unittest.TestCase):
    def test_anthropic_cannot_call_tools(self):
        self.assertFalse(provider_supports_tools("anthropic", "claude-3-5-sonnet-20241022"))

    def test_ollama_is_excluded_despite_json_mode(self):
        # JSON mode is not tool support: local models are version dependent.
        self.assertFalse(provider_supports_tools("ollama", "llama3.2"))

    def test_documented_providers_are_supported(self):
        for provider in ("openai", "groq", "google", "github"):
            self.assertTrue(provider_supports_tools(provider), provider)

    def test_capability_set_is_a_strict_subset_of_json_mode(self):
        from app.services.llm_client import JSON_MODE_CAPABLE_PROVIDERS

        self.assertTrue(TOOL_CAPABLE_PROVIDERS < JSON_MODE_CAPABLE_PROVIDERS)

    def test_model_level_denylist(self):
        self.assertFalse(provider_supports_tools("groq", "gemma2-9b-it"))
        self.assertFalse(provider_supports_tools("openai", "deepseek-r1-distill"))

    def test_model_prefix_is_ignored_for_the_denylist(self):
        self.assertFalse(provider_supports_tools("groq", "groq/gemma2-9b-it"))
        self.assertTrue(provider_supports_tools("groq", "groq/llama-3.3-70b-versatile"))

    def test_unknown_provider_is_rejected(self):
        self.assertFalse(provider_supports_tools("mystery", "some-model"))

    def test_bare_model_name(self):
        self.assertEqual(bare_model_name("openai/gpt-4o-mini"), "gpt-4o-mini")
        self.assertEqual(bare_model_name("gpt-4o-mini"), "gpt-4o-mini")
        self.assertEqual(bare_model_name(""), "")


class LoopTerminationTests(unittest.TestCase):
    def test_tool_result_feeds_back_as_tool_message(self):
        transport = ScriptedTransport(
            [
                ModelTurn(text="", tool_calls=(_call(value="alpha"),)),
                ModelTurn(text="final briefing", tool_calls=()),
            ]
        )
        result = _run(
            run_tool_loop(
                transport=transport,
                registry=_registry(),
                prompt="research X",
                max_steps=3,
            )
        )
        self.assertEqual(result.stopped_reason, STOP_NO_TOOL_CALLS)
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.text, "final briefing")
        self.assertEqual(len(transport.calls), 2)

        # Step 2's transcript must carry the assistant tool_call and the
        # role="tool" result the model asked for.
        second = list(transport.calls[1]["messages"])
        roles = [m["role"] for m in second]
        self.assertEqual(roles, ["user", "assistant", "tool"])
        self.assertEqual(second[1]["tool_calls"][0]["function"]["name"], "lookup")
        self.assertEqual(second[2]["tool_call_id"], "c1")
        self.assertEqual(second[2]["content"], "looked-up:alpha")

    def test_system_message_is_emitted_when_supplied(self):
        transport = ScriptedTransport([ModelTurn(text="done")])
        _run(
            run_tool_loop(
                transport=transport,
                registry=_registry(),
                prompt="p",
                system="be terse",
                max_steps=1,
            )
        )
        messages = list(transport.calls[0]["messages"])
        self.assertEqual(messages[0], {"role": "system", "content": "be terse"})

    def test_model_that_never_calls_a_tool_terminates_immediately(self):
        transport = ScriptedTransport([ModelTurn(text="I already know this.")])
        result = _run(
            run_tool_loop(
                transport=transport,
                registry=_registry(),
                prompt="p",
                max_steps=5,
            )
        )
        self.assertEqual(result.steps, 1)
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(result.stopped_reason, STOP_NO_TOOL_CALLS)
        self.assertEqual(result.text, "I already know this.")

    def test_loop_terminates_at_max_steps(self):
        # A model that calls the tool on every single step must be cut off.
        transport = ScriptedTransport(
            [ModelTurn(text=f"s{i}", tool_calls=(_call(value=f"v{i}", call_id=f"c{i}"),)) for i in range(10)]
        )
        result = _run(
            run_tool_loop(
                transport=transport,
                registry=_registry(),
                prompt="p",
                max_steps=3,
            )
        )
        self.assertEqual(result.steps, 3)
        self.assertEqual(len(transport.calls), 3)
        self.assertEqual(result.stopped_reason, STOP_MAX_STEPS)
        self.assertEqual(len(result.tool_results), 3)

    def test_unknown_tool_terminates_gracefully(self):
        transport = ScriptedTransport(
            [ModelTurn(text="", tool_calls=(_call(name="ghost", call_id="g1"),))]
        )
        result = _run(
            run_tool_loop(
                transport=transport,
                registry=_registry(),
                prompt="p",
                max_steps=5,
            )
        )
        self.assertEqual(result.stopped_reason, STOP_UNKNOWN_TOOL)
        self.assertEqual(len(transport.calls), 1, "must not re-prompt after an unknown tool")
        self.assertFalse(result.tool_results[0].ok)
        last = list(result.messages)[-1]
        self.assertEqual(last["role"], "tool")
        self.assertIn("unknown_tool", last["content"])

    def test_disabled_tool_stops_the_loop(self):
        registry = ToolRegistry(
            [
                ToolSpec(
                    name="other",
                    description="still enabled",
                    parameters=_PARAMS,
                    handler=lambda a: "other-ok",
                ),
                ToolSpec(
                    name="lookup",
                    description="disabled",
                    parameters=_PARAMS,
                    handler=lambda a: "should-not-run",
                    enabled=False,
                ),
                ToolSpec(
                    name="boom",
                    description="disabled",
                    parameters=_PARAMS,
                    handler=lambda a: "should-not-run",
                    enabled=False,
                ),
            ]
        )
        transport = ScriptedTransport(
            [ModelTurn(text="", tool_calls=(_call(name="lookup"), _call(name="boom", call_id="c2")))]
        )
        result = _run(
            run_tool_loop(
                transport=transport,
                registry=registry,
                prompt="p",
                max_steps=4,
            )
        )
        self.assertEqual(result.stopped_reason, STOP_ALL_CALLS_FAILED)
        self.assertEqual(len(transport.calls), 1)

    def test_failing_handler_does_not_stop_the_loop(self):
        # A handler that raises is a recoverable tool error, not a dispatch
        # failure: the model sees it and may continue.
        transport = ScriptedTransport(
            [
                ModelTurn(text="", tool_calls=(_call(name="boom"),)),
                ModelTurn(text="recovered", tool_calls=()),
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
        self.assertEqual(result.text, "recovered")
        self.assertEqual(len(transport.calls), 2)
        tool_message = list(transport.calls[1]["messages"])[-1]
        self.assertTrue(tool_message["content"].startswith("ERROR: tool_failed"))

    def test_transport_exception_is_contained(self):
        transport = ScriptedTransport([ModelTurn(text="never used")])
        transport.raise_on = 0
        result = _run(
            run_tool_loop(
                transport=transport,
                registry=_registry(),
                prompt="p",
                max_steps=2,
            )
        )
        self.assertEqual(result.stopped_reason, STOP_MODEL_ERROR)
        self.assertEqual(result.steps, 1)

    def test_no_enabled_tools_skips_the_model_entirely(self):
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
        transport = ScriptedTransport([ModelTurn(text="unused")])
        result = _run(
            run_tool_loop(transport=transport, registry=registry, prompt="p", max_steps=3)
        )
        self.assertEqual(transport.calls, [])
        self.assertEqual(result.steps, 0)
        self.assertEqual(result.stopped_reason, STOP_NO_TOOL_CALLS)

    def test_per_step_call_cap_is_enforced(self):
        calls = tuple(_call(value=f"v{i}", call_id=f"c{i}") for i in range(9))
        transport = ScriptedTransport([ModelTurn(text="", tool_calls=calls)])
        result = _run(
            run_tool_loop(
                transport=transport,
                registry=_registry(),
                prompt="p",
                max_steps=2,
                max_tool_calls_per_step=3,
            )
        )
        self.assertEqual(len(result.tool_results), 3)
        # Unhonoured calls are dropped rather than silently executed.
        self.assertEqual(
            len([m for m in result.messages if m["role"] == "tool"]), 3
        )

    def test_allow_list_narrows_the_advertised_tools(self):
        registry = _registry()
        transport = ScriptedTransport([ModelTurn(text="done")])
        _run(
            run_tool_loop(
                transport=transport,
                registry=registry,
                prompt="p",
                tool_names=["lookup"],
                max_steps=1,
            )
        )
        names = [t["function"]["name"] for t in transport.calls[0]["tools"]]
        self.assertEqual(names, ["lookup"])

    def test_tool_context_joins_successful_results_only(self):
        transport = ScriptedTransport(
            [
                ModelTurn(
                    text="briefing",
                    tool_calls=(_call(value="ok"), _call(name="boom", call_id="c2")),
                )
            ]
        )
        result = _run(
            run_tool_loop(
                transport=transport,
                registry=_registry(),
                prompt="p",
                max_steps=1,
            )
        )
        self.assertIn("looked-up:ok", result.tool_context)
        self.assertNotIn("tool exploded", result.tool_context)
        self.assertIn("briefing", result.text)

    def test_max_steps_is_floored_at_one(self):
        transport = ScriptedTransport([ModelTurn(text="once")])
        result = _run(
            run_tool_loop(
                transport=transport,
                registry=_registry(),
                prompt="p",
                max_steps=0,
            )
        )
        self.assertEqual(result.steps, 1)
        self.assertEqual(len(transport.calls), 1)

    def test_model_and_tool_choice_are_forwarded(self):
        transport = ScriptedTransport([ModelTurn(text="ok")])
        _run(
            run_tool_loop(
                transport=transport,
                registry=_registry(),
                prompt="p",
                model="llama-3.3-70b-versatile",
                tool_choice={"type": "function", "function": {"name": "lookup"}},
                max_steps=1,
                max_tokens=321,
                temperature=0.11,
            )
        )
        call = transport.calls[0]
        self.assertEqual(call["model"], "llama-3.3-70b-versatile")
        self.assertEqual(call["tool_choice"]["function"]["name"], "lookup")
        self.assertEqual(call["max_tokens"], 321)
        self.assertAlmostEqual(call["temperature"], 0.11)


class ModelTurnParsingTests(unittest.TestCase):
    def test_reads_tool_calls_from_object_response(self):
        response = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "thinking",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": '{"value": "z"}',
                                },
                            }
                        ],
                    },
                }
            ]
        }
        turn = ModelTurn.from_response(response)
        self.assertEqual(turn.text, "thinking")
        self.assertEqual(turn.finish_reason, "tool_calls")
        self.assertEqual(len(turn.tool_calls), 1)
        self.assertEqual(turn.tool_calls[0].name, "lookup")
        self.assertEqual(json.loads(turn.tool_calls[0].arguments), {"value": "z"})

    def test_skips_malformed_tool_call_entries(self):
        response = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [{"id": "x", "function": {}}],
                    }
                }
            ]
        }
        turn = ModelTurn.from_response(response)
        self.assertEqual(turn.tool_calls, ())

    def test_non_json_arguments_are_stringified(self):
        call = ToolCall.from_payload({"id": "c", "name": "n", "arguments": {"a": 1}})
        self.assertEqual(json.loads(call.arguments), {"a": 1})

    def test_missing_choices_is_an_error_turn(self):
        self.assertFalse(ModelTurn.from_response({}).success)
        self.assertEqual(ModelTurn.from_response({}).error, "no_choices")


if __name__ == "__main__":
    unittest.main()
