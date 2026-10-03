"""Coverage for app/services/tools/registry.py.

Complements ``tests/test_tools_registry.py`` (the happy paths and the four
named failure modes) with the schema-normalisation defaults, the result
rendering fallbacks, cancellation propagation, and the timeout-free async
handler path.
"""

import asyncio
import json
import unittest

from app.services.tools.registry import (
    ToolError,
    ToolRegistry,
    ToolResult,
    ToolSchemaError,
    ToolSpec,
    validate_json_schema,
)

_ECHO_PARAMS = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
}


def _spec(name="echo", handler=None, **overrides):
    async def _default(arguments):
        return arguments.get("text", "")

    return ToolSpec(
        name=name,
        description=overrides.pop("description", f"{name} tool"),
        parameters=overrides.pop("parameters", None) or _ECHO_PARAMS,
        handler=handler or _default,
        **overrides,
    )


def _run(coro):
    return asyncio.run(coro)


class SchemaNormalisationTests(unittest.TestCase):
    def test_null_properties_become_an_empty_object(self):
        out = validate_json_schema({"type": "object", "properties": None})
        self.assertEqual(out["properties"], {})
        # A schema with no properties at all is equally usable.
        self.assertEqual(
            validate_json_schema({"type": "object"})["properties"], {}
        )

    def test_null_required_becomes_an_empty_array(self):
        out = validate_json_schema(
            {"type": "object", "properties": {}, "required": None}
        )
        self.assertEqual(out["required"], [])

    def test_non_object_properties_is_rejected(self):
        with self.assertRaises(ToolSchemaError) as ctx:
            validate_json_schema({"type": "object", "properties": ["a", "b"]})
        self.assertIn("properties must be an object", str(ctx.exception))

    def test_non_schema_property_is_rejected(self):
        with self.assertRaises(ToolSchemaError) as ctx:
            validate_json_schema(
                {"type": "object", "properties": {"text": "a bare string"}}
            )
        self.assertIn("must be a schema object", str(ctx.exception))

    def test_non_array_required_is_rejected(self):
        with self.assertRaises(ToolSchemaError) as ctx:
            validate_json_schema(
                {"type": "object", "properties": {"text": {"type": "string"}}, "required": "text"}
            )
        self.assertIn("required must be an array", str(ctx.exception))

    def test_tuple_required_is_accepted_and_normalised_to_a_list(self):
        out = validate_json_schema(
            {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ("text",),
            }
        )
        self.assertEqual(out["required"], ["text"])

    def test_integer_property_names_are_stringified(self):
        out = validate_json_schema(
            {"type": "object", "properties": {1: {"type": "string"}}, "required": [1]}
        )
        self.assertEqual(list(out["properties"]), ["1"])
        self.assertEqual(out["required"], ["1"])

    def test_missing_type_is_reported_with_the_offending_name(self):
        with self.assertRaises(ToolSchemaError) as ctx:
            validate_json_schema(
                {"type": "object", "properties": {"q": {"description": "no type"}}}
            )
        self.assertIn("'q'", str(ctx.exception))
        self.assertIn("has no 'type'", str(ctx.exception))

    def test_tool_spec_normalises_its_own_name(self):
        spec = _spec(name="  spaced  ")
        self.assertEqual(spec.name, "spaced")
        self.assertEqual(spec.to_openai_schema()["function"]["name"], "spaced")


class RenderResultTests(unittest.TestCase):
    def _content(self, handler, **kwargs):
        registry = ToolRegistry([_spec("t", handler=handler, **kwargs)])
        result = _run(registry.invoke("t", {}, tool_call_id="c1"))
        self.assertTrue(result.ok, result.error)
        return result

    def test_none_result_renders_as_the_empty_string(self):
        result = self._content(lambda arguments: None)
        self.assertEqual(result.content, "")
        self.assertFalse(result.truncated)

    def test_string_result_is_returned_verbatim(self):
        result = self._content(lambda arguments: "plain text")
        self.assertEqual(result.content, "plain text")

    def test_tuple_result_is_json_encoded_as_an_array(self):
        result = self._content(lambda arguments: (1, 2))
        self.assertEqual(json.loads(result.content), [1, 2])

    def test_non_container_result_is_stringified(self):
        result = self._content(lambda arguments: 42)
        self.assertEqual(result.content, "42")

    def test_unserialisable_container_falls_back_to_str(self):
        # A handler returning a self-referential structure must not crash the
        # tool result; the raw repr is degraded but returned.
        cycle = {}
        cycle["self"] = cycle
        result = self._content(lambda arguments: cycle)
        self.assertEqual(result.content, str(cycle))
        self.assertIn("self", result.content)

    def test_zero_length_limit_yields_empty_content_and_marks_truncated(self):
        registry = ToolRegistry(
            [_spec("t", handler=lambda a: "0123456789", max_result_chars=100)]
        )
        result = _run(registry.invoke("t", {}, tool_call_id="c1", max_result_chars=0))
        self.assertTrue(result.ok)
        self.assertEqual(result.content, "")
        self.assertTrue(result.truncated)

    def test_negative_length_limit_is_clamped_to_zero(self):
        registry = ToolRegistry(
            [_spec("t", handler=lambda a: "0123456789", max_result_chars=100)]
        )
        result = _run(
            registry.invoke("t", {}, tool_call_id="c1", max_result_chars=-5)
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.content, "")
        self.assertTrue(result.truncated)


class HandlerAwaitableTests(unittest.TestCase):
    def test_async_handler_without_a_timeout_is_awaited(self):
        async def handler(arguments):
            await asyncio.sleep(0)
            return "awaited"

        registry = ToolRegistry([_spec("t", handler=handler, timeout_seconds=None)])
        result = _run(registry.invoke("t", {}, tool_call_id="c1"))
        self.assertTrue(result.ok)
        self.assertEqual(result.content, "awaited")

    def test_cancellation_is_propagated_not_swallowed(self):
        async def handler(arguments):
            raise asyncio.CancelledError()

        registry = ToolRegistry([_spec("t", handler=handler)])

        async def drive():
            try:
                await registry.invoke("t", {}, tool_call_id="c1")
            except asyncio.CancelledError:
                return "propagated"
            return "swallowed"

        # A cancelled request must abort, not be reported as a tool failure.
        self.assertEqual(_run(drive()), "propagated")


class ResultContractTests(unittest.TestCase):
    def test_tool_execution_error_from_a_handler_is_labelled(self):
        # ToolExecutionError is the registry's own "the tool failed" signal.
        from app.services.tools.registry import ToolExecutionError

        def handler(arguments):
            raise ToolExecutionError("upstream refused")

        registry = ToolRegistry([_spec("t", handler=handler)])
        result = _run(registry.invoke("t", {}, tool_call_id="c1"))
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "tool_failed: upstream refused")
        # A handler that ran is not a dispatch failure.
        self.assertFalse(result.is_dispatch_failure)

    def test_call_id_defaults_to_the_tool_name(self):
        registry = ToolRegistry([_spec("t", handler=lambda a: "x")])
        result = _run(registry.invoke("t", {}, tool_call_id=""))
        self.assertEqual(result.tool_call_id, "t")

    def test_unknown_tool_without_a_call_id_falls_back_to_the_name(self):
        registry = ToolRegistry()
        result = _run(registry.invoke("ghost", {}, tool_call_id=""))
        self.assertEqual(result.tool_call_id, "ghost")
        self.assertEqual(result.tool_name, "ghost")

    def test_result_with_no_error_is_not_a_dispatch_failure(self):
        self.assertFalse(
            ToolResult(tool_name="t", tool_call_id="c", ok=False, content="").is_dispatch_failure
        )
        self.assertTrue(
            ToolResult(
                tool_name="t",
                tool_call_id="c",
                ok=False,
                content="",
                error="unknown_tool 'x' is not available",
            ).is_dispatch_failure
        )
        self.assertTrue(
            ToolResult(
                tool_name="t",
                tool_call_id="c",
                ok=False,
                content="",
                error="tool_disabled 'x' is not enabled",
            ).is_dispatch_failure
        )

    def test_successful_result_message_uses_the_content(self):
        message = ToolResult(
            tool_name="t", tool_call_id="c", ok=True, content="body"
        ).to_message()
        self.assertEqual(message["content"], "body")


class RegistryMutationTests(unittest.TestCase):
    def test_unregistering_an_unknown_name_is_a_noop(self):
        registry = ToolRegistry([_spec("a")])
        registry.unregister("nope")
        registry.unregister("")
        self.assertEqual(registry.names(), ["a"])

    def test_lookup_strips_surrounding_whitespace(self):
        registry = ToolRegistry([_spec("alpha")])
        self.assertIsNotNone(registry.get("  alpha  "))
        self.assertIsNone(registry.get("   "))
        self.assertIsNone(registry.get(None))

    def test_empty_allow_list_advertises_nothing(self):
        registry = ToolRegistry([_spec("a"), _spec("b")])
        self.assertEqual(registry.to_openai_schema(only=[]), [])

    def test_register_all_replaces_by_default(self):
        registry = ToolRegistry([_spec("a", description="v1")])
        registry.register_all([_spec("a", description="v2")])
        self.assertEqual(registry.get("a").description, "v2")

    def test_register_all_can_refuse_to_replace(self):
        registry = ToolRegistry([_spec("a", description="v1")])
        with self.assertRaises(ToolError):
            registry.register_all([_spec("a", description="v2")], replace_existing=False)

    def test_set_enabled_round_trips_without_mutating_the_original_spec(self):
        original = _spec("a")
        registry = ToolRegistry([original])
        off = registry.set_enabled("a", False)
        self.assertFalse(off.enabled)
        self.assertTrue(original.enabled)
        self.assertTrue(registry.set_enabled("a", True).enabled)


if __name__ == "__main__":
    unittest.main()