"""Tool registry behaviour: registration, enable flags, schemas, containment."""

import asyncio
import unittest

from app.services.tools.registry import (
    TOOL_REGISTRY,
    ToolDisabledError,
    ToolError,
    ToolRegistry,
    ToolResult,
    ToolSchemaError,
    ToolSpec,
    UnknownToolError,
    coerce_arguments,
    get_registry,
    validate_json_schema,
)

_ECHO_PARAMS = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}


def _spec(name="echo", handler=None, enabled=True, **overrides):
    async def _default(arguments):
        return arguments.get("text", "")

    params = overrides.pop("parameters", None) or _ECHO_PARAMS
    return ToolSpec(
        name=name,
        description=overrides.pop("description", f"{name} tool"),
        parameters=params,
        handler=handler or _default,
        enabled=enabled,
        **overrides,
    )


def _run(coro):
    return asyncio.run(coro)


class CoerceArgumentsTests(unittest.TestCase):
    def test_accepts_dict_and_json_string(self):
        self.assertEqual(coerce_arguments({"a": 1}), {"a": 1})
        self.assertEqual(coerce_arguments('{"a": 1}'), {"a": 1})

    def test_empty_becomes_empty_dict(self):
        self.assertEqual(coerce_arguments(None), {})
        self.assertEqual(coerce_arguments(""), {})

    def test_rejects_non_object_json(self):
        with self.assertRaises(ToolSchemaError):
            coerce_arguments("[1, 2]")

    def test_rejects_unparseable_json(self):
        with self.assertRaises(ToolSchemaError):
            coerce_arguments("{not json")

    def test_rejects_scalar_arguments(self):
        with self.assertRaises(ToolSchemaError):
            coerce_arguments(7)


class ValidateJsonSchemaTests(unittest.TestCase):
    def test_accepts_object_schema(self):
        out = validate_json_schema(_ECHO_PARAMS)
        self.assertEqual(out["type"], "object")
        self.assertEqual(out["required"], ["text"])

    def test_rejects_non_object_type(self):
        with self.assertRaises(ToolSchemaError):
            validate_json_schema({"type": "array", "properties": {}})

    def test_rejects_property_without_type(self):
        with self.assertRaises(ToolSchemaError):
            validate_json_schema({"type": "object", "properties": {"x": {}}})

    def test_rejects_required_not_in_properties(self):
        with self.assertRaises(ToolSchemaError):
            validate_json_schema(
                {"type": "object", "properties": {}, "required": ["missing"]}
            )

    def test_rejects_non_mapping_schema(self):
        with self.assertRaises(ToolSchemaError):
            validate_json_schema("object")


class ToolSpecTests(unittest.TestCase):
    def test_rejects_empty_name(self):
        with self.assertRaises(ToolSchemaError):
            _spec(name="   ")

    def test_rejects_non_callable_handler(self):
        with self.assertRaises(ToolSchemaError):
            ToolSpec(
                name="bad",
                description="d",
                parameters=_ECHO_PARAMS,
                handler="not-callable",
            )

    def test_invalid_schema_fails_at_construction(self):
        with self.assertRaises(ToolSchemaError):
            _spec(parameters={"type": "string"})

    def test_to_openai_schema_shape(self):
        schema = _spec(name="echo").to_openai_schema()
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "echo")
        self.assertIn("description", schema["function"])
        self.assertEqual(schema["function"]["parameters"]["type"], "object")
        self.assertEqual(schema["function"]["parameters"]["required"], ["text"])

    def test_with_enabled_returns_new_spec(self):
        spec = _spec(enabled=True)
        off = spec.with_enabled(False)
        self.assertTrue(spec.enabled)
        self.assertFalse(off.enabled)


class RegistryLookupTests(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry([_spec("alpha"), _spec("beta", enabled=False)])

    def test_register_and_get(self):
        self.assertIsNotNone(self.registry.get("alpha"))
        self.assertIsNone(self.registry.get("missing"))

    def test_require_raises_for_unknown(self):
        with self.assertRaises(UnknownToolError):
            self.registry.require("missing")

    def test_duplicate_registration_rejected(self):
        with self.assertRaises(ToolError):
            self.registry.register(_spec("alpha"))

    def test_duplicate_allowed_with_replace(self):
        replacement = _spec("alpha", description="v2")
        self.registry.register(replacement, replace_existing=True)
        self.assertEqual(self.registry.get("alpha").description, "v2")

    def test_register_rejects_non_spec(self):
        with self.assertRaises(ToolSchemaError):
            self.registry.register({"name": "alpha"})

    def test_list_all_is_sorted(self):
        self.assertEqual(self.registry.names(), ["alpha", "beta"])

    def test_list_enabled_filters_disabled(self):
        self.assertEqual([s.name for s in self.registry.list_enabled()], ["alpha"])

    def test_list_enabled_with_allow_list_never_reenables(self):
        # An allow-list narrows; it must not override the operator's flag.
        self.assertEqual(self.registry.list_enabled(only=["alpha", "beta"]), [
            s for s in self.registry.list_all() if s.name == "alpha"
        ])

    def test_list_enabled_with_allow_list_raises_on_unknown(self):
        with self.assertRaises(UnknownToolError):
            self.registry.list_enabled(only=["nope"])

    def test_set_enabled_toggles(self):
        self.registry.set_enabled("alpha", False)
        self.assertEqual(self.registry.list_enabled(), [])
        self.registry.set_enabled("alpha", True)
        self.assertEqual([s.name for s in self.registry.list_enabled()], ["alpha"])

    def test_set_enabled_unknown_raises(self):
        with self.assertRaises(UnknownToolError):
            self.registry.set_enabled("nope", True)

    def test_unregister(self):
        self.registry.unregister("alpha")
        self.assertIsNone(self.registry.get("alpha"))

    def test_to_openai_schema_only_enabled(self):
        schemas = self.registry.to_openai_schema()
        self.assertEqual([s["function"]["name"] for s in schemas], ["alpha"])


class RegistryInvokeTests(unittest.TestCase):
    def test_unknown_tool_returns_result_not_exception(self):
        registry = ToolRegistry()
        result = _run(registry.invoke("ghost", {}, tool_call_id="c1"))
        self.assertIsInstance(result, ToolResult)
        self.assertFalse(result.ok)
        self.assertTrue(result.error.startswith("unknown_tool"))
        self.assertTrue(result.is_dispatch_failure)
        self.assertEqual(result.tool_call_id, "c1")

    def test_disabled_tool_is_not_run(self):
        calls: list[dict] = []

        async def handler(arguments):
            calls.append(arguments)
            return "ran"

        registry = ToolRegistry([_spec("off", handler=handler, enabled=False)])
        result = _run(registry.invoke("off", {"text": "x"}, tool_call_id="c1"))
        self.assertFalse(result.ok)
        self.assertTrue(result.error.startswith("tool_disabled"))
        self.assertEqual(calls, [])

    def test_handler_exception_is_contained(self):
        async def boom(arguments):
            raise RuntimeError("kaboom")

        registry = ToolRegistry([_spec("boom", handler=boom)])
        result = _run(registry.invoke("boom", {}, tool_call_id="c1"))
        self.assertFalse(result.ok)
        self.assertIn("tool_failed", result.error)
        self.assertIn("kaboom", result.error)

    def test_handler_timeout_is_contained(self):
        import asyncio as _asyncio

        async def slow(arguments):
            await _asyncio.sleep(5)

        registry = ToolRegistry([_spec("slow", handler=slow, timeout_seconds=0.01)])
        result = _run(registry.invoke("slow", {}, tool_call_id="c1"))
        self.assertFalse(result.ok)
        self.assertIn("timed out", result.error)

    def test_invalid_arguments_are_contained(self):
        async def handler(arguments):  # pragma: no cover - must not run
            raise AssertionError("handler should not be called")

        registry = ToolRegistry([_spec("echo", handler=handler)])
        result = _run(registry.invoke("echo", "not json", tool_call_id="c1"))
        self.assertFalse(result.ok)
        self.assertIn("invalid_arguments", result.error)

    def test_sync_handler_supported(self):
        registry = ToolRegistry([_spec("sync", handler=lambda arguments: "sync-ok")])
        result = _run(registry.invoke("sync", {}, tool_call_id="c1"))
        self.assertTrue(result.ok)
        self.assertEqual(result.content, "sync-ok")

    def test_result_is_truncated_to_limit(self):
        registry = ToolRegistry([_spec("big", max_result_chars=10)])
        result = _run(registry.invoke("big", {"text": "y" * 100}, tool_call_id="c1"))
        self.assertTrue(result.ok)
        self.assertTrue(result.truncated)
        self.assertEqual(len(result.content), 10)

    def test_dict_result_is_json_encoded(self):
        registry = ToolRegistry([_spec("obj", handler=lambda a: {"k": [1, 2]})])
        result = _run(registry.invoke("obj", {}, tool_call_id="c1"))
        self.assertEqual(result.content, '{"k": [1, 2]}')

    def test_to_message_shape(self):
        registry = ToolRegistry([_spec("echo")])
        ok = _run(registry.invoke("echo", {"text": "hi"}, tool_call_id="c9"))
        message = ok.to_message()
        self.assertEqual(message["role"], "tool")
        self.assertEqual(message["tool_call_id"], "c9")
        self.assertEqual(message["name"], "echo")
        self.assertEqual(message["content"], "hi")

        missing = _run(registry.invoke("ghost", {}, tool_call_id="c10"))
        self.assertTrue(missing.to_message()["content"].startswith("ERROR:"))

    def test_dispatch_failure_detection_is_narrow(self):
        # A handler that fails is not a dispatch failure: the model can retry it.
        async def boom(arguments):
            raise ValueError("nope")

        registry = ToolRegistry([_spec("boom", handler=boom)])
        result = _run(registry.invoke("boom", {}, tool_call_id="c1"))
        self.assertFalse(result.is_dispatch_failure)


class ModuleRegistryTests(unittest.TestCase):
    def test_get_registry_returns_shared_instance(self):
        self.assertIs(get_registry(), TOOL_REGISTRY)

    def test_module_registry_starts_empty(self):
        # Nothing is auto-registered at import time: flows build their own
        # registry from settings so an enable flag is the only source of truth.
        self.assertEqual(TOOL_REGISTRY.names(), [])


class DisabledErrorTests(unittest.TestCase):
    def test_tool_disabled_error_is_a_tool_error(self):
        self.assertTrue(issubclass(ToolDisabledError, ToolError))
        self.assertTrue(issubclass(UnknownToolError, ToolError))


if __name__ == "__main__":
    unittest.main()
