"""How ``LLMClient.completion`` handles ``tools`` / ``tool_choice``.

The registry and the loop depend on two guarantees from the client: tool
definitions reach LiteLLM unchanged, and ``response_format`` is dropped (with a
warning) when tools are present, because the two are mutually exclusive on a
single call. ``litellm.acompletion`` is stubbed, so nothing here touches the
network.
"""

import asyncio
import logging
import unittest
from types import SimpleNamespace

from app.schemas.models import LLMConfigRequest
from app.services.llm_client import (
    CALL_FAILED_CODE,
    JSON_MODE_CAPABLE_PROVIDERS,
    LLMClient,
)
from app.services.tools import TOOL_CAPABLE_PROVIDERS
from app.services.tools.registry import ToolRegistry, ToolSpec
from app.services.tools.submit_rubric import (
    SUBMIT_RUBRIC_TOOL_CHOICE,
    submit_rubric_tool,
)

_PARAMS = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}


def _tool(name="tavily_search"):
    async def handler(arguments):
        return "ok"

    return ToolSpec(name=name, description="d", parameters=_PARAMS, handler=handler)


class _StubLitellm:
    """Captures the kwargs LiteLLM would have been called with."""

    def __init__(self, content="hello"):
        self.kwargs: dict | None = None
        self.content = content

    async def acompletion(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )


def _run_with_stub(coro_factory, content="hello"):
    stub = _StubLitellm(content)
    original = _swap_stub(stub)
    try:
        return asyncio.run(coro_factory()), stub
    finally:
        _swap_restore(original)


def _swap_stub(stub):
    from app.services import llm_client as module

    original = module.litellm.acompletion
    module.litellm.acompletion = stub.acompletion
    return original


def _swap_restore(original):
    from app.services import llm_client as module

    module.litellm.acompletion = original


def _completion(**kwargs):
    async def go():
        return await LLMClient().completion("prompt", **kwargs)

    return go


class _LlmClientContractTest(unittest.TestCase):
    """Resets the process-wide per-user call budget so ordering cannot matter."""

    def setUp(self):
        LLMClient._budget.reset()

    def tearDown(self):
        LLMClient._budget.reset()


class ToolPassThroughTests(_LlmClientContractTest):
    def test_tools_reach_the_provider_unchanged(self):
        schemas = ToolRegistry([_tool()]).to_openai_schema()
        result, stub = _run_with_stub(_completion(tools=schemas))
        self.assertTrue(result["success"])
        self.assertEqual(stub.kwargs["tools"], schemas)
        self.assertNotIn("response_format", stub.kwargs)
        self.assertNotIn("tool_choice", stub.kwargs)

    def test_tool_choice_is_forwarded_when_supplied(self):
        schemas = ToolRegistry([_tool()]).to_openai_schema()
        _, stub = _run_with_stub(
            _completion(tools=schemas, tool_choice=SUBMIT_RUBRIC_TOOL_CHOICE)
        )
        self.assertEqual(stub.kwargs["tool_choice"], SUBMIT_RUBRIC_TOOL_CHOICE)

    def test_forced_submit_rubric_call_sends_no_response_format(self):
        schemas = ToolRegistry([submit_rubric_tool()]).to_openai_schema()
        _, stub = _run_with_stub(
            _completion(tools=schemas, tool_choice=SUBMIT_RUBRIC_TOOL_CHOICE)
        )
        self.assertEqual(stub.kwargs["tool_choice"]["function"]["name"], "submit_rubric")
        self.assertNotIn("response_format", stub.kwargs)

    def test_structured_alone_still_sends_response_format(self):
        _, stub = _run_with_stub(_completion(structured=True))
        self.assertEqual(stub.kwargs["response_format"], {"type": "json_object"})

    def test_structured_on_a_provider_without_json_mode_omits_it(self):
        _, stub = _run_with_stub(
            _completion(structured=True, llm_config=LLMConfigRequest(provider="anthropic"))
        )
        self.assertNotIn("response_format", stub.kwargs)
        self.assertNotIn("tools", stub.kwargs)


class MutualExclusionTests(_LlmClientContractTest):
    def test_tools_win_and_response_format_is_dropped_with_a_warning(self):
        schemas = ToolRegistry([_tool()]).to_openai_schema()
        client = LLMClient()
        stub = _StubLitellm()
        original = _swap_stub(stub)

        async def go():
            return await client.completion(
                "prompt",
                structured=True,
                tools=schemas,
                tool_choice=SUBMIT_RUBRIC_TOOL_CHOICE,
            )

        logger = logging.getLogger("app.services.llm_client")
        records: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = _Capture()
        logger.addHandler(handler)
        try:
            result = asyncio.run(go())
        finally:
            logger.removeHandler(handler)
            _swap_restore(original)

        self.assertTrue(result["success"])
        self.assertEqual(stub.kwargs["tools"], schemas)
        self.assertNotIn("response_format", stub.kwargs)
        warnings = [r for r in records if r.levelno == logging.WARNING]
        self.assertTrue(warnings, "dropping response_format must be logged")
        self.assertIn("response_format", warnings[0].getMessage())

    def test_empty_tools_list_does_not_count_as_tool_mode(self):
        _, stub = _run_with_stub(_completion(structured=True, tools=[]))
        self.assertEqual(stub.kwargs["response_format"], {"type": "json_object"})


class CapabilitySetTests(_LlmClientContractTest):
    def test_tool_capable_set_excludes_anthropic(self):
        self.assertNotIn("anthropic", TOOL_CAPABLE_PROVIDERS)
        self.assertNotIn("anthropic", JSON_MODE_CAPABLE_PROVIDERS)

    def test_tool_capable_set_is_strictly_narrower_than_json_mode(self):
        self.assertTrue(TOOL_CAPABLE_PROVIDERS.issubset(JSON_MODE_CAPABLE_PROVIDERS))
        self.assertLess(TOOL_CAPABLE_PROVIDERS, JSON_MODE_CAPABLE_PROVIDERS)

    def test_ollama_supports_json_mode_but_not_tools(self):
        self.assertIn("ollama", JSON_MODE_CAPABLE_PROVIDERS)
        self.assertNotIn("ollama", TOOL_CAPABLE_PROVIDERS)


class OutcomeTests(_LlmClientContractTest):
    def test_tool_mode_failure_is_reported_as_a_call_failure(self):
        from app.services import llm_client as module

        async def boom(**kwargs):
            raise RuntimeError("provider refused tools")

        original = module.litellm.acompletion
        module.litellm.acompletion = boom
        try:
            result = asyncio.run(
                _completion(tools=ToolRegistry([_tool()]).to_openai_schema())()
            )
        finally:
            module.litellm.acompletion = original
        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], CALL_FAILED_CODE)


if __name__ == "__main__":
    unittest.main()
