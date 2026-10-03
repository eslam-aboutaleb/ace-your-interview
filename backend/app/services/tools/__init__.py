"""Tool registry and the bounded model tool loop (Stage 4.2).

Public surface:

* :class:`~app.services.tools.registry.ToolSpec` — name, description, JSON
  Schema, handler coroutine, enable flag.
* :class:`~app.services.tools.registry.ToolRegistry` — ``register`` / ``get`` /
  ``require`` / ``list_all`` / ``list_enabled`` / ``to_openai_schema`` / ``invoke``.
* :func:`~app.services.tools.agentic.run_tool_loop` — the bounded loop.
* :func:`~app.services.tools.agentic.provider_supports_tools` — the capability guard.
* :mod:`app.services.tools.enrichment` — Tavily / Firecrawl / GitHub as tools.
* :mod:`app.services.tools.submit_rubric` — the forced ``submit_rubric`` tool.
"""

from app.services.tools.agentic import (
    STOP_ALL_CALLS_FAILED,
    STOP_MAX_STEPS,
    STOP_MODEL_ERROR,
    STOP_NO_TOOL_CALLS,
    STOP_UNKNOWN_TOOL,
    TOOL_CAPABLE_PROVIDERS,
    TOOL_INCAPABLE_MODEL_PATTERNS,
    ModelTurn,
    ToolCall,
    ToolLoopResult,
    ToolModelTransport,
    bare_model_name,
    provider_supports_tools,
    run_tool_loop,
)
from app.services.tools.registry import (
    DEFAULT_MAX_RESULT_CHARS,
    TOOL_REGISTRY,
    ToolDisabledError,
    ToolError,
    ToolExecutionError,
    ToolRegistry,
    ToolResult,
    ToolSchemaError,
    ToolSpec,
    UnknownToolError,
    coerce_arguments,
    get_registry,
    validate_json_schema,
)

__all__ = [
    "DEFAULT_MAX_RESULT_CHARS",
    "ModelTurn",
    "STOP_ALL_CALLS_FAILED",
    "STOP_MAX_STEPS",
    "STOP_MODEL_ERROR",
    "STOP_NO_TOOL_CALLS",
    "STOP_UNKNOWN_TOOL",
    "TOOL_CAPABLE_PROVIDERS",
    "TOOL_INCAPABLE_MODEL_PATTERNS",
    "TOOL_REGISTRY",
    "ToolCall",
    "ToolDisabledError",
    "ToolError",
    "ToolExecutionError",
    "ToolLoopResult",
    "ToolModelTransport",
    "ToolRegistry",
    "ToolResult",
    "ToolSchemaError",
    "ToolSpec",
    "UnknownToolError",
    "bare_model_name",
    "coerce_arguments",
    "get_registry",
    "provider_supports_tools",
    "run_tool_loop",
    "validate_json_schema",
]
