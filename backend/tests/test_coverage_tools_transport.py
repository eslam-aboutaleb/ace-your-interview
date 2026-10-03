"""Coverage for app/services/tools/transport.py.

``LiteLLMToolTransport`` is the only place in the tool framework that talks to
a provider, so it is exercised with ``app.services.tools.transport.litellm``
patched. The transport's job is to turn the loop's transcript into a faithful
LiteLLM request and a provider response back into a :class:`ModelTurn`, so
these tests assert the *request it builds* and the *turn it returns*.
"""

import asyncio
import unittest
from unittest.mock import patch

from app.services.tools import transport as transport_module
from app.services.tools.agentic import (
    STOP_MAX_STEPS,
    STOP_MODEL_ERROR,
    STOP_NO_TOOL_CALLS,
    ModelTurn,
    run_tool_loop,
)
from app.services.tools.registry import ToolRegistry, ToolSpec
from app.services.tools.transport import (
    LiteLLMToolTransport,
    _usage_to_dict,
)

_PARAMS = {"type": "object", "properties": {"value": {"type": "string"}}}


def _run(coro):
    return asyncio.run(coro)


def _usage(**values):
    class _Usage:
        def __init__(self):
            for key, value in values.items():
                setattr(self, key, value)

    return _Usage()


def _response(*, text="", tool_calls=(), finish_reason="stop", usage=None):
    """A LiteLLM-shaped response object (attributes, not a dict)."""

    class _Message:
        def __init__(self):
            self.role = "assistant"
            self.content = text
            self.tool_calls = list(tool_calls) or None

    class _Choice:
        def __init__(self):
            self.index = 0
            self.message = _Message()
            self.finish_reason = finish_reason

    class _Response:
        def __init__(self):
            self.choices = [_Choice()]
            if usage is not None:
                self.usage = usage

    return _Response()


def _tool_call_payload(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class RecordingLiteLLM:
    """Captures every ``acompletion`` kwargs dict and replays queued responses."""

    def __init__(self, *responses, error: Exception | None = None):
        self.responses = list(responses)
        self.error = error
        self.calls: list[dict] = []

    async def acompletion(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if not self.responses:
            return _response(text="exhausted")
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class ResolvedModelTests(unittest.TestCase):
    def test_explicit_model_wins_and_is_stripped(self):
        instance = LiteLLMToolTransport(provider="openai", model="  gpt-4o-mini  ")
        self.assertEqual(instance.resolved_model(), "gpt-4o-mini")

    def test_blank_model_falls_back_to_the_provider_default(self):
        instance = LiteLLMToolTransport(provider="groq", model="   ")
        # _resolve_model is the shared (provider, model) -> LiteLLM string
        # helper, so an unset model must produce a non-empty prefixed model.
        with patch.object(
            transport_module,
            "_resolve_model",
            return_value="groq/llama-3.3-70b-versatile",
        ) as resolver:
            self.assertEqual(
                instance.resolved_model(), "groq/llama-3.3-70b-versatile"
            )
        resolver.assert_called_once_with("groq", "")

    def test_blank_provider_is_normalised_to_default(self):
        instance = LiteLLMToolTransport(model="", provider="")
        with patch.object(
            transport_module, "_resolve_model", return_value="gpt-4o-mini"
        ) as resolver:
            instance.resolved_model()
        resolver.assert_called_once_with("default", "")

    def test_dataclass_defaults_match_the_documented_ceiling(self):
        instance = LiteLLMToolTransport()
        self.assertEqual(instance.max_tokens, transport_module.DEFAULT_TOOL_LOOP_MAX_TOKENS)
        self.assertEqual(instance.temperature, transport_module.DEFAULT_TOOL_LOOP_TEMPERATURE)
        self.assertEqual(
            instance.timeout_seconds,
            transport_module.DEFAULT_TOOL_LOOP_TIMEOUT_SECONDS,
        )
        self.assertEqual(instance.extra_kwargs, {})


class CompleteRequestTests(unittest.TestCase):
    def _complete(self, transport, fake, **overrides):
        kwargs = {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "tool_choice": "auto",
        }
        kwargs.update(overrides)
        with patch.object(
            transport_module.litellm, "acompletion", new=fake.acompletion
        ):
            return _run(transport.complete(**kwargs))

    def test_request_carries_the_whole_transcript_verbatim(self):
        fake = RecordingLiteLLM(_response(text="done"))
        transport = LiteLLMToolTransport(
            provider="openai", model="gpt-4o-mini", timeout_seconds=12.5
        )
        messages = [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "find X"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"value": "X"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "lookup",
                "content": "found: X",
            },
        ]
        turn = self._complete(transport, fake, messages=messages)

        self.assertTrue(turn.success)
        self.assertEqual(turn.text, "done")
        request = fake.calls[0]
        # The transcript is forwarded in order, including the assistant
        # tool_calls and the role="tool" result -- the reason this transport
        # does not go through LLMClient.completion.
        self.assertEqual(request["messages"], messages)
        self.assertEqual(request["messages"][3]["role"], "tool")
        self.assertEqual(request["messages"][3]["tool_call_id"], "call_1")
        self.assertEqual(
            request["messages"][2]["tool_calls"][0]["function"]["name"], "lookup"
        )
        self.assertEqual(request["model"], "gpt-4o-mini")
        self.assertEqual(request["tool_choice"], "auto")
        self.assertEqual(request["timeout"], 12.5)
        # Per-call overrides win over the instance defaults.
        self.assertEqual(
            request["max_tokens"], transport_module.DEFAULT_TOOL_LOOP_MAX_TOKENS
        )
        self.assertAlmostEqual(
            request["temperature"], transport_module.DEFAULT_TOOL_LOOP_TEMPERATURE
        )

    def test_model_max_tokens_and_temperature_are_forwarded(self):
        fake = RecordingLiteLLM(_response(text="ok"))
        transport = LiteLLMToolTransport(
            provider="openai", model="instance-model", max_tokens=10, temperature=0.9
        )
        self._complete(
            transport,
            fake,
            model="gpt-4o-mini",
            max_tokens=321,
            temperature=0.11,
        )
        request = fake.calls[0]
        self.assertEqual(request["model"], "gpt-4o-mini")
        self.assertEqual(request["max_tokens"], 321)
        self.assertAlmostEqual(request["temperature"], 0.11)

    def test_zero_and_blank_overrides_defer_to_instance_defaults(self):
        fake = RecordingLiteLLM(_response(text="ok"))
        transport = LiteLLMToolTransport(
            provider="openai",
            model="instance-model",
            max_tokens=64,
            temperature=0.5,
        )
        self._complete(
            transport,
            fake,
            model="   ",
            max_tokens=0,
            temperature=0.0,
        )
        request = fake.calls[0]
        self.assertEqual(request["model"], "instance-model")
        self.assertEqual(request["max_tokens"], 64)
        self.assertAlmostEqual(request["temperature"], 0.5)

    def test_forced_tool_choice_is_forwarded_verbatim(self):
        fake = RecordingLiteLLM(_response(text="ok"))
        transport = LiteLLMToolTransport(provider="openai", model="gpt-4o-mini")
        forced = {"type": "function", "function": {"name": "lookup"}}
        self._complete(transport, fake, tool_choice=forced)
        self.assertEqual(fake.calls[0]["tool_choice"], forced)

    def test_none_tool_choice_is_omitted(self):
        fake = RecordingLiteLLM(_response(text="ok"))
        transport = LiteLLMToolTransport(provider="openai", model="gpt-4o-mini")
        self._complete(transport, fake, tool_choice=None)
        self.assertNotIn("tool_choice", fake.calls[0])

    def test_extra_kwargs_are_merged_last(self):
        fake = RecordingLiteLLM(_response(text="ok"))
        transport = LiteLLMToolTransport(
            provider="openai",
            model="gpt-4o-mini",
            extra_kwargs={"api_key": "sk-test", "api_base": "https://example.test"},
        )
        self._complete(transport, fake)
        request = fake.calls[0]
        self.assertEqual(request["api_key"], "sk-test")
        self.assertEqual(request["api_base"], "https://example.test")
        self.assertEqual(request["model"], "gpt-4o-mini")

    def test_tool_calls_and_usage_are_read_from_the_response(self):
        fake = RecordingLiteLLM(
            _response(
                text="thinking",
                tool_calls=[
                    _tool_call_payload("call_1", "lookup", '{"value": "X"}')
                ],
                finish_reason="tool_calls",
                usage=_usage(input_tokens=11, output_tokens=7, total_tokens=18),
            )
        )
        transport = LiteLLMToolTransport(provider="openai", model="gpt-4o-mini")
        turn = self._complete(transport, fake)
        self.assertTrue(turn.success)
        self.assertEqual(turn.text, "thinking")
        self.assertEqual(turn.finish_reason, "tool_calls")
        self.assertEqual(len(turn.tool_calls), 1)
        self.assertEqual(turn.tool_calls[0].name, "lookup")
        self.assertEqual(turn.tool_calls[0].id, "call_1")
        self.assertEqual(
            turn.usage,
            {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
        )

    def test_usage_absent_leaves_usage_empty(self):
        fake = RecordingLiteLLM(_response(text="no usage"))
        transport = LiteLLMToolTransport(provider="openai", model="gpt-4o-mini")
        turn = self._complete(transport, fake)
        self.assertEqual(turn.usage, {})

    def test_dict_shaped_response_is_accepted(self):
        fake = RecordingLiteLLM(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "dict shaped"},
                    }
                ]
            }
        )
        transport = LiteLLMToolTransport(provider="openai", model="gpt-4o-mini")
        turn = self._complete(transport, fake)
        self.assertTrue(turn.success)
        self.assertEqual(turn.text, "dict shaped")

    def test_response_without_tool_calls_yields_no_tool_calls(self):
        fake = RecordingLiteLLM(_response(text="plain answer"))
        transport = LiteLLMToolTransport(provider="openai", model="gpt-4o-mini")
        turn = self._complete(transport, fake)
        self.assertEqual(turn.tool_calls, ())
        self.assertEqual(turn.text, "plain answer")


class TransportInsideTheLoopTests(unittest.TestCase):
    """The loop must terminate on a real transport failure, not raise."""

    def _registry(self):
        async def lookup(arguments):
            return f"looked-up:{arguments.get('value', '')}"

        return ToolRegistry(
            [
                ToolSpec(
                    name="lookup",
                    description="look something up",
                    parameters=_PARAMS,
                    handler=lookup,
                )
            ]
        )

    def test_litellm_exception_becomes_a_model_error_stop(self):
        fake = RecordingLiteLLM(error=RuntimeError("litellm exploded"))
        transport = LiteLLMToolTransport(provider="openai", model="gpt-4o-mini")
        with patch.object(
            transport_module.litellm, "acompletion", new=fake.acompletion
        ):
            result = _run(
                run_tool_loop(
                    transport=transport,
                    registry=self._registry(),
                    prompt="research X",
                    max_steps=2,
                )
            )
        self.assertEqual(result.stopped_reason, STOP_MODEL_ERROR)
        self.assertEqual(result.steps, 1)
        self.assertEqual(result.text, "")
        self.assertEqual(len(fake.calls), 1)

    def test_no_tool_calls_response_ends_the_loop(self):
        fake = RecordingLiteLLM(
            _response(
                text="here is the answer",
                usage=_usage(input_tokens=3, output_tokens=4, total_tokens=7),
            )
        )
        transport = LiteLLMToolTransport(provider="openai", model="gpt-4o-mini")
        with patch.object(
            transport_module.litellm, "acompletion", new=fake.acompletion
        ):
            result = _run(
                run_tool_loop(
                    transport=transport,
                    registry=self._registry(),
                    prompt="research X",
                    max_steps=3,
                    system="be terse",
                )
            )
        self.assertEqual(result.stopped_reason, STOP_NO_TOOL_CALLS)
        self.assertEqual(result.text, "here is the answer")
        self.assertEqual(result.steps, 1)
        # Provider-reported usage is surfaced on the loop result.
        self.assertEqual(result.usage["total_tokens"], 7)
        # The advertised tools reached the provider as an OpenAI schema.
        request = fake.calls[0]
        self.assertEqual(
            [t["function"]["name"] for t in request["tools"]], ["lookup"]
        )
        self.assertEqual(
            [m["role"] for m in request["messages"]], ["system", "user"]
        )

    def test_tool_result_is_fed_back_on_the_second_step(self):
        fake = RecordingLiteLLM(
            _response(
                text="",
                tool_calls=[
                    _tool_call_payload("call_1", "lookup", '{"value": "X"}')
                ],
                finish_reason="tool_calls",
            ),
            _response(text="X is a letter."),
        )
        transport = LiteLLMToolTransport(provider="openai", model="gpt-4o-mini")
        with patch.object(
            transport_module.litellm, "acompletion", new=fake.acompletion
        ):
            result = _run(
                run_tool_loop(
                    transport=transport,
                    registry=self._registry(),
                    prompt="what is X?",
                    max_steps=3,
                )
            )
        self.assertEqual(result.stopped_reason, STOP_NO_TOOL_CALLS)
        self.assertEqual(result.text, "X is a letter.")
        self.assertIn("looked-up:X", result.tool_context)
        # The second provider request carries the assistant tool_call and the
        # role="tool" result.
        second = fake.calls[1]["messages"]
        self.assertEqual(
            [m["role"] for m in second], ["user", "assistant", "tool"]
        )
        self.assertEqual(second[2]["content"], "looked-up:X")

    def test_a_model_that_always_calls_a_tool_hits_max_steps(self):
        fake = RecordingLiteLLM(
            *[
                _response(
                    text=f"step {i}",
                    tool_calls=[
                        _tool_call_payload(
                            f"call_{i}", "lookup", '{"value": "X"}'
                        )
                    ],
                    finish_reason="tool_calls",
                )
                for i in range(4)
            ]
        )
        transport = LiteLLMToolTransport(provider="openai", model="gpt-4o-mini")
        with patch.object(
            transport_module.litellm, "acompletion", new=fake.acompletion
        ):
            result = _run(
                run_tool_loop(
                    transport=transport,
                    registry=self._registry(),
                    prompt="what is X?",
                    max_steps=2,
                )
            )
        self.assertEqual(result.stopped_reason, STOP_MAX_STEPS)
        self.assertEqual(result.steps, 2)
        self.assertEqual(len(result.tool_results), 2)
        self.assertEqual(len(fake.calls), 2)


class UsageToDictTests(unittest.TestCase):
    def test_reads_attributes(self):
        self.assertEqual(
            _usage_to_dict(
                _usage(input_tokens=1, output_tokens=2, total_tokens=3)
            ),
            {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        )

    def test_reads_dict_shaped_usage(self):
        self.assertEqual(
            _usage_to_dict({"input_tokens": 5, "total_tokens": 9}),
            {"input_tokens": 5, "total_tokens": 9},
        )

    def test_absent_fields_are_skipped(self):
        self.assertEqual(_usage_to_dict(_usage()), {})
        self.assertEqual(_usage_to_dict(object()), {})

    def test_uncastable_values_become_zero(self):
        # A provider that returns a non-numeric counter must not raise here;
        # usage is telemetry, not a reason to fail the step.
        self.assertEqual(
            _usage_to_dict(_usage(input_tokens="n/a", total_tokens="7")),
            {"input_tokens": 0, "total_tokens": 7},
        )
        self.assertEqual(
            _usage_to_dict({"input_tokens": None, "output_tokens": 4}),
            {"output_tokens": 4},
        )

    def test_none_value_is_skipped_even_for_dicts(self):
        self.assertEqual(_usage_to_dict({"input_tokens": None}), {})


class ModelTurnContractTests(unittest.TestCase):
    def test_no_choices_is_an_error_turn(self):
        turn = ModelTurn.from_response(object())
        self.assertFalse(turn.success)
        self.assertEqual(turn.error, "no_choices")
        self.assertEqual(turn.tool_calls, ())


if __name__ == "__main__":
    unittest.main()