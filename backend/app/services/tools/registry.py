"""Tool registry: tool name → JSON schema + handler, with per-tool enable flags.

A tool is a :class:`ToolSpec`: a stable name, a description the model reads,
a JSON Schema for the tool arguments, and a handler coroutine. The registry is
the single place that knows which tools exist and which of them an operator has
enabled, so a flow never has to branch on tool names itself.

Design notes
------------
* **Handlers are injectable.** A :class:`ToolSpec` takes any awaitable-returning
  callable, so tests exercise the loop and the registry with plain functions
  and never touch the network.
* **A failing tool must never crash a caller.** :meth:`ToolRegistry.invoke`
  never raises for handler failures: it returns a :class:`ToolResult` with
  ``ok=False`` and a short error string, which the tool loop feeds back to the
  model as the tool result. A model that sees an error string can recover; a
  raised exception unwinds the whole request.
* **Unknown and disabled are different failures.** ``UnknownToolError`` means
  the name was never registered (a model hallucination, or a stale transcript
  from a different registry). ``ToolDisabledError`` means the tool exists but an
  operator turned it off. Both surface as ``ok=False`` results so the loop can
  terminate gracefully instead of spinning.
* **Argument coercion is centralised.** Providers emit tool arguments as a JSON
  *string*; tests and internal callers use dicts. :meth:`ToolRegistry.invoke`
  accepts both and rejects anything else.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

logger = logging.getLogger(__name__)

#: Hard ceiling on a single tool result handed back to the model. A tool that
#: returns a 200 KB scraped page would otherwise dominate the next turn's
#: context (and the next turn's cost) with text the model will not read.
DEFAULT_MAX_RESULT_CHARS = 4000

#: Upper bound on tool calls honoured per model turn. A model that emits 40
#: parallel calls in one step is a failure mode, not a feature.
MAX_TOOL_CALLS_PER_STEP = 6

#: Per-tool-call wall-clock ceiling. ``None`` disables the ceiling, which is
#: only appropriate for handlers that are pure computation.
DEFAULT_TOOL_TIMEOUT_SECONDS: float | None = 20.0


class ToolError(Exception):
    """Base class for tool-registry errors."""


class UnknownToolError(ToolError):
    """Raised when a tool name is not present in the registry."""


class ToolDisabledError(ToolError):
    """Raised when a tool exists but is disabled."""


class ToolSchemaError(ToolError):
    """Raised when a tool definition is not a usable JSON Schema."""


class ToolExecutionError(ToolError):
    """Raised when a tool handler fails. Contained by ``invoke``."""


ToolHandler = Callable[[dict[str, Any]], Awaitable[Any] | Any]


def coerce_arguments(arguments: Any) -> dict[str, Any]:
    """Normalise tool arguments from a provider payload into a dict.

    Providers send the arguments as a JSON string; internal callers and tests
    pass a dict. Anything else (a list, a number, unparseable JSON) is a
    caller error and raises :class:`ToolSchemaError`.
    """
    if arguments is None or arguments == "":
        return {}
    if isinstance(arguments, Mapping):
        return {str(k): v for k, v in arguments.items()}
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ToolSchemaError(f"tool arguments are not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ToolSchemaError(
                f"tool arguments must decode to an object, got {type(parsed).__name__}"
            )
        return {str(k): v for k, v in parsed.items()}
    raise ToolSchemaError(
        f"tool arguments must be an object or JSON string, got {type(arguments).__name__}"
    )


def validate_json_schema(schema: Any, *, context: str = "tool") -> dict[str, Any]:
    """Structurally validate a tool parameter schema.

    Deliberately not a full JSON Schema meta-schema check: it verifies the
    invariants this codebase depends on (an object schema with typed
    properties), so a malformed tool definition fails at registration time
    rather than as a confusing provider error on the first live call.
    """
    if not isinstance(schema, Mapping):
        raise ToolSchemaError(f"{context}: parameters must be an object, got {type(schema).__name__}")
    schema = dict(schema)
    schema_type = schema.get("type")
    if schema_type != "object":
        raise ToolSchemaError(f"{context}: parameters.type must be 'object', got {schema_type!r}")
    properties = schema.get("properties", {})
    if properties is None:
        properties = {}
    if not isinstance(properties, Mapping):
        raise ToolSchemaError(
            f"{context}: parameters.properties must be an object, got {type(properties).__name__}"
        )
    for name, prop in properties.items():
        if not isinstance(prop, Mapping):
            raise ToolSchemaError(f"{context}: property {name!r} must be a schema object")
        if "type" not in prop:
            raise ToolSchemaError(f"{context}: property {name!r} has no 'type'")
    schema["properties"] = {str(k): dict(v) for k, v in properties.items()}
    required = schema.get("required", [])
    if required is None:
        required = []
    if not isinstance(required, (list, tuple)):
        raise ToolSchemaError(f"{context}: parameters.required must be an array")
    unknown = [key for key in required if str(key) not in schema["properties"]]
    if unknown:
        raise ToolSchemaError(f"{context}: required names not in properties: {unknown}")
    schema["required"] = [str(key) for key in required]
    return schema


@dataclass(frozen=True)
class ToolSpec:
    """One tool: what the model sees, and what runs when it calls it."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    enabled: bool = True
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS
    timeout_seconds: float | None = DEFAULT_TOOL_TIMEOUT_SECONDS
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        if not name:
            raise ToolSchemaError("tool name must be a non-empty string")
        object.__setattr__(self, "name", name)
        if not callable(self.handler):
            raise ToolSchemaError(f"{name}: handler must be callable")
        object.__setattr__(
            self,
            "parameters",
            validate_json_schema(self.parameters, context=name),
        )

    def to_openai_schema(self) -> dict[str, Any]:
        """Render as an OpenAI-style ``tools`` entry."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": str(self.description or ""),
                "parameters": dict(self.parameters),
            },
        }

    def with_enabled(self, enabled: bool) -> ToolSpec:
        return replace(self, enabled=bool(enabled))


@dataclass(frozen=True)
class ToolResult:
    """Outcome of one tool invocation, shaped for a ``role="tool"`` message."""

    tool_name: str
    tool_call_id: str
    ok: bool
    content: str
    error: str = ""
    truncated: bool = False

    def to_message(self) -> dict[str, Any]:
        """Render as the chat message appended after an assistant tool call."""
        body = self.content if self.ok else f"ERROR: {self.error}"
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "name": self.tool_name,
            "content": body,
        }

    @property
    def is_dispatch_failure(self) -> bool:
        """True when the tool was never run (unknown or disabled name)."""
        return not self.ok and self.error.startswith(("unknown_tool", "tool_disabled"))


class ToolRegistry:
    """Name → :class:`ToolSpec`, with enable filtering and safe invocation."""

    def __init__(self, specs: Iterable[ToolSpec] = ()) -> None:
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs:
            self.register(spec)

    # ── registration ────────────────────────────────────────

    def register(self, spec: ToolSpec, *, replace_existing: bool = False) -> ToolSpec:
        if not isinstance(spec, ToolSpec):
            raise ToolSchemaError(f"expected ToolSpec, got {type(spec).__name__}")
        if spec.name in self._specs and not replace_existing:
            raise ToolError(f"tool {spec.name!r} is already registered")
        self._specs[spec.name] = spec
        logger.debug("tool_registered name=%s enabled=%s", spec.name, spec.enabled)
        return spec

    def register_all(self, specs: Iterable[ToolSpec], *, replace_existing: bool = True) -> None:
        for spec in specs:
            self.register(spec, replace_existing=replace_existing)

    def unregister(self, name: str) -> None:
        self._specs.pop(str(name or "").strip(), None)

    def set_enabled(self, name: str, enabled: bool) -> ToolSpec:
        spec = self.require(name)
        updated = spec.with_enabled(enabled)
        self._specs[spec.name] = updated
        return updated

    # ── lookup ───────────────────────────────────────────────

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(str(name or "").strip())

    def require(self, name: str) -> ToolSpec:
        spec = self.get(name)
        if spec is None:
            raise UnknownToolError(f"unknown tool {name!r}")
        return spec

    def list_all(self) -> list[ToolSpec]:
        return sorted(self._specs.values(), key=lambda s: s.name)

    def list_enabled(self, only: Iterable[str] | None = None) -> list[ToolSpec]:
        """Enabled specs, optionally restricted to an explicit allow-list.

        An allow-list never re-enables a disabled tool: the operator's flag
        still wins, so a flow cannot widen the enabled surface by naming tools.
        """
        specs = self.list_all() if only is None else [self.require(n) for n in only]
        return [spec for spec in specs if spec.enabled]

    def names(self) -> list[str]:
        return [spec.name for spec in self.list_all()]

    def to_openai_schema(self, only: Iterable[str] | None = None) -> list[dict[str, Any]]:
        """Tool definitions for ``LLMClient.completion(tools=...)``."""
        return [spec.to_openai_schema() for spec in self.list_enabled(only)]

    # ── invocation ───────────────────────────────────────────

    async def invoke(
        self,
        name: str,
        arguments: Any = None,
        *,
        tool_call_id: str = "",
        max_result_chars: int | None = None,
    ) -> ToolResult:
        """Run one tool and always return a result; never raises.

        Every failure mode (unknown name, disabled name, malformed arguments,
        handler exception, handler timeout) becomes a ``ToolResult`` with
        ``ok=False``. Callers append it to the transcript and let the model
        decide what to do next.
        """
        call_id = str(tool_call_id or name or "")
        spec = self.get(name)
        if spec is None:
            logger.info("tool_unknown name=%s", name)
            return ToolResult(
                tool_name=str(name or ""),
                tool_call_id=call_id,
                ok=False,
                content="",
                error=f"unknown_tool {name!r} is not available",
            )
        if not spec.enabled:
            logger.info("tool_disabled name=%s", spec.name)
            return ToolResult(
                tool_name=spec.name,
                tool_call_id=call_id,
                ok=False,
                content="",
                error=f"tool_disabled {spec.name!r} is not enabled",
            )

        try:
            payload = coerce_arguments(arguments)
        except ToolSchemaError as exc:
            return ToolResult(
                tool_name=spec.name,
                tool_call_id=call_id,
                ok=False,
                content="",
                error=f"invalid_arguments: {exc}",
            )

        try:
            raw = await self._call_handler(spec, payload)
        except asyncio.CancelledError:
            raise
        except ToolExecutionError as exc:
            return ToolResult(
                tool_name=spec.name,
                tool_call_id=call_id,
                ok=False,
                content="",
                error=f"tool_failed: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 - a tool must not kill the loop
            logger.warning("tool_handler_failed name=%s err=%s", spec.name, exc)
            return ToolResult(
                tool_name=spec.name,
                tool_call_id=call_id,
                ok=False,
                content="",
                error=f"tool_failed: {type(exc).__name__}: {exc}",
            )

        content = self._render_result(raw)
        limit = int(max_result_chars if max_result_chars is not None else spec.max_result_chars)
        limit = max(0, limit)
        truncated = len(content) > limit
        if truncated:
            content = content[:limit]
        return ToolResult(
            tool_name=spec.name,
            tool_call_id=call_id,
            ok=True,
            content=content,
            truncated=truncated,
        )

    async def _call_handler(self, spec: ToolSpec, payload: dict[str, Any]) -> Any:
        result = spec.handler(payload)
        if inspect.isawaitable(result):
            if spec.timeout_seconds is None:
                return await result
            try:
                return await asyncio.wait_for(result, timeout=spec.timeout_seconds)
            except asyncio.TimeoutError as exc:
                raise ToolExecutionError(
                    f"{spec.name} timed out after {spec.timeout_seconds}s"
                ) from exc
        return result

    @staticmethod
    def _render_result(raw: Any) -> str:
        if isinstance(raw, str):
            return raw
        if raw is None:
            return ""
        if isinstance(raw, (dict, list, tuple)):
            try:
                return json.dumps(raw, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                return str(raw)
        return str(raw)


#: Process-wide registry. Flows read from it; tests build their own instance
#: and inject it, so no test shares mutable registry state.
TOOL_REGISTRY = ToolRegistry()


def get_registry() -> ToolRegistry:
    return TOOL_REGISTRY
