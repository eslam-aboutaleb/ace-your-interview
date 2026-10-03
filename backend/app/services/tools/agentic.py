"""Model-transport seam and the bounded tool loop.

The loop is deliberately transport-agnostic: it takes any object with a
``complete(...)`` coroutine, so a test can drive a scripted model with no
network and no provider client, while production uses
:class:`LiteLLMToolTransport`.

Bounded by construction
-----------------------
``max_steps`` caps the number of *model* calls, and therefore caps the number
of tool-execution rounds. Every exit path is explicit and returns a
``stopped_reason``: ``no_tool_calls``, ``max_steps``, ``all_tool_calls_failed``,
``unknown_tool`` or ``model_error``. There is no unbounded recursion and no
loop that can be re-entered by a model that keeps asking for the same tool.

Capability guard
----------------
Tool calling is a *narrower* capability than JSON mode, so
:data:`TOOL_CAPABLE_PROVIDERS` is deliberately a strict subset of
``llm_client.JSON_MODE_CAPABLE_PROVIDERS``. See that constant's docstring for
the reasoning; the short version:

* ``anthropic`` — excluded. No OpenAI-style ``tools`` on the native surface, and
  ``tool_choice`` forcing is not equivalent. Anthropic is also absent from the
  JSON-mode set, so it is excluded from both.
* ``ollama`` — excluded. Local models are arbitrary and version-dependent; the
  OpenAI-compatible ``/v1/chat/completions`` surface only gained tool support in
  newer builds and forcing ``tool_choice`` is unreliable. A user pinning an
  old Ollama tag must not break the request.
* ``openai`` / ``groq`` / ``google`` / ``github`` — included. All four expose
  documented ``tools`` + ``tool_choice`` on the OpenAI-compatible surface
  litellm uses (Google via the OpenAI-compatible endpoint rewrite in
  ``llm_client``).

The set is checked *before* the first model call, so an unsupported model skips
enrichment entirely rather than failing mid-loop or silently returning nothing
after burning a call.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.services.tools.registry import (
    MAX_TOOL_CALLS_PER_STEP,
    ToolRegistry,
    ToolResult,
)

logger = logging.getLogger(__name__)

#: Providers whose OpenAI-compatible surface is known to accept ``tools`` and a
#: forced ``tool_choice``. A strict subset of
#: ``app.services.llm_client.JSON_MODE_CAPABLE_PROVIDERS``: JSON mode does not
#: imply tool support, so membership here must be verified independently.
TOOL_CAPABLE_PROVIDERS: frozenset[str] = frozenset(
    {"openai", "groq", "google", "github"}
)

#: Model-name patterns that are known not to support tool calling even on a
#: tool-capable provider. Matched case-insensitively against the bare model
#: name (any provider prefix stripped).
TOOL_INCAPABLE_MODEL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^gemma"),          # Groq-hosted gemma: no tool support
    re.compile(r"^mixtral"),        # legacy Mixtral endpoints: no tool support
    re.compile(r"^deepseek-r1"),    # reasoning models reject tools/tool_choice
    re.compile(r"^o1(?:-|$)"),      # o1 family has no tool_choice on some tiers
    re.compile(r"^text-embedding"), # never a chat model
)


def provider_supports_tools(provider: str, model: str = "") -> bool:
    """True when ``provider`` (optionally narrowed by ``model``) can call tools."""
    normalized = str(provider or "").strip().lower()
    if normalized not in TOOL_CAPABLE_PROVIDERS:
        return False
    bare = bare_model_name(model)
    if not bare:
        return True
    return not any(pattern.search(bare) for pattern in TOOL_INCAPABLE_MODEL_PATTERNS)


def bare_model_name(model: str) -> str:
    """Strip a LiteLLM ``provider/`` prefix from a model string."""
    text = str(model or "").strip()
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    return text.strip().lower()


# ── Model turn types ────────────────────────────────────────


@dataclass(frozen=True)
class ToolCall:
    """One tool call emitted by the model."""

    id: str
    name: str
    arguments: str = ""
    index: int = 0

    @classmethod
    def from_payload(cls, payload: Any, *, index: int = 0) -> ToolCall | None:
        """Build from an OpenAI-style ``tool_calls`` entry (or a lite dict)."""
        if payload is None:
            return None
        if isinstance(payload, dict):
            call_id = str(payload.get("id") or payload.get("tool_call_id") or f"call_{index}")
            name = str(payload.get("name") or "")
            function = payload.get("function")
            arguments: Any = payload.get("arguments")
            if isinstance(function, dict):
                name = str(function.get("name") or name)
                if arguments is None:
                    arguments = function.get("arguments")
        else:
            call_id = str(getattr(payload, "id", "") or f"call_{index}")
            function = getattr(payload, "function", None)
            # LiteLLM returns `ChatCompletionMessageToolCall` objects, which
            # carry the name ONLY on `.function.name` — there is no top-level
            # `.name` attribute. Reading only `payload.name` here would discard
            # every real tool call and silently exit the loop with
            # `no_tool_calls`, so mirror the dict branch's fallback.
            name = str(getattr(payload, "name", "") or "")
            if not name and function is not None:
                name = str(getattr(function, "name", "") or "")
            arguments = getattr(function, "arguments", None) if function is not None else None
        if not name:
            return None
        if arguments is None:
            arguments = ""
        if not isinstance(arguments, str):
            try:
                arguments = json.dumps(arguments, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                arguments = str(arguments)
        return cls(id=str(call_id), name=name, arguments=arguments, index=index)

    def to_message_fragment(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass(frozen=True)
class ModelTurn:
    """One model response: optional text plus optional tool calls."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    success: bool = True
    error: str = ""

    @classmethod
    def from_response(cls, response: Any) -> ModelTurn:
        """Read a LiteLLM/OpenAI-shaped response object."""
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices")
        if not choices:
            return cls(success=False, error="no_choices")
        first = choices[0]
        message = getattr(first, "message", None)
        if message is None and isinstance(first, dict):
            message = first.get("message")
        if message is None:
            return cls(success=False, error="no_message")
        text = getattr(message, "content", None)
        if text is None and isinstance(message, dict):
            text = message.get("content")
        raw_calls = getattr(message, "tool_calls", None)
        if raw_calls is None and isinstance(message, dict):
            raw_calls = message.get("tool_calls")
        calls: list[ToolCall] = []
        for index, payload in enumerate(raw_calls or []):
            call = ToolCall.from_payload(payload, index=index)
            if call is not None:
                calls.append(call)
        finish_reason = getattr(first, "finish_reason", None)
        if finish_reason is None and isinstance(first, dict):
            finish_reason = first.get("finish_reason")
        return cls(
            text=str(text or ""),
            tool_calls=tuple(calls),
            finish_reason=str(finish_reason or ""),
        )


class ToolModelTransport(Protocol):
    """Anything that can run one tool-aware model turn."""

    async def complete(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        tool_choice: Any,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> ModelTurn: ...


# ── The bounded loop ────────────────────────────────────────


@dataclass(frozen=True)
class ToolLoopResult:
    """Outcome of :func:`run_tool_loop`."""

    text: str
    steps: int
    stopped_reason: str
    tool_results: tuple[ToolResult, ...] = ()
    messages: tuple[dict[str, Any], ...] = ()
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def tool_context(self) -> str:
        """Concatenated successful tool output, for prompt injection."""
        return "\n\n".join(r.content for r in self.tool_results if r.ok and r.content)

    @property
    def succeeded(self) -> bool:
        return self.stopped_reason in ("no_tool_calls", "max_steps") and bool(self.text)


STOP_NO_TOOL_CALLS = "no_tool_calls"
STOP_MAX_STEPS = "max_steps"
STOP_ALL_CALLS_FAILED = "all_tool_calls_failed"
STOP_UNKNOWN_TOOL = "unknown_tool"
STOP_MODEL_ERROR = "model_error"


async def run_tool_loop(
    *,
    transport: ToolModelTransport,
    registry: ToolRegistry,
    prompt: str,
    system: str = "",
    tool_names: Iterable[str] | None = None,
    tool_choice: Any = "auto",
    max_steps: int = 2,
    max_tool_calls_per_step: int = MAX_TOOL_CALLS_PER_STEP,
    model: str = "",
    max_tokens: int = 1200,
    temperature: float = 0.2,
) -> ToolLoopResult:
    """Run a bounded tool loop: the model picks tools, results feed back.

    ``max_steps`` is the number of model calls, so the loop is bounded even if
    the model requests a tool on every single turn.
    """
    schemas = registry.to_openai_schema(tool_names)
    messages: list[dict[str, Any]] = []
    if str(system or "").strip():
        messages.append({"role": "system", "content": str(system)})
    messages.append({"role": "user", "content": str(prompt or "")})

    steps = 0
    results: list[ToolResult] = []
    usage: dict[str, int] = {}
    last_text = ""

    if not schemas:
        logger.info("tool_loop_skipped reason=no_enabled_tools")
        return ToolLoopResult(
            text="",
            steps=0,
            stopped_reason=STOP_NO_TOOL_CALLS,
            messages=tuple(messages),
        )

    total_steps = max(1, int(max_steps))
    per_step_cap = max(1, int(max_tool_calls_per_step))

    for step in range(1, total_steps + 1):
        steps = step
        try:
            turn = await transport.complete(
                messages=tuple(messages),
                tools=tuple(schemas),
                tool_choice=tool_choice,
                model=model,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
            )
        except Exception as exc:  # noqa: BLE001 - transport failure must not escape
            logger.warning("tool_loop_transport_failed step=%s err=%s", step, exc)
            return ToolLoopResult(
                text=last_text,
                steps=steps,
                stopped_reason=STOP_MODEL_ERROR,
                tool_results=tuple(results),
                messages=tuple(messages),
                usage=usage,
            )

        if turn.usage:
            usage = dict(turn.usage)
        if turn.text:
            last_text = turn.text
        if not turn.tool_calls:
            return ToolLoopResult(
                text=turn.text or last_text,
                steps=steps,
                stopped_reason=STOP_NO_TOOL_CALLS,
                tool_results=tuple(results),
                messages=tuple(messages),
                usage=usage,
            )

        messages.append(
            {
                "role": "assistant",
                "content": turn.text or "",
                "tool_calls": [call.to_message_fragment() for call in turn.tool_calls],
            }
        )

        step_results: list[ToolResult] = []
        for call in turn.tool_calls[:per_step_cap]:
            result = await registry.invoke(
                call.name,
                call.arguments,
                tool_call_id=call.id,
            )
            results.append(result)
            step_results.append(result)
            messages.append(result.to_message())

        if len(turn.tool_calls) > per_step_cap:
            logger.info(
                "tool_loop_truncated_calls step=%s requested=%s honoured=%s",
                step,
                len(turn.tool_calls),
                per_step_cap,
            )

        if step_results and all(r.is_dispatch_failure for r in step_results):
            # Every requested tool was unknown or disabled. Feeding this back
            # invites the model to ask again, so stop with the transcript
            # complete rather than burning the remaining step budget.
            reason = (
                STOP_UNKNOWN_TOOL
                if all(r.error.startswith("unknown_tool") for r in step_results)
                else STOP_ALL_CALLS_FAILED
            )
            return ToolLoopResult(
                text=last_text,
                steps=steps,
                stopped_reason=reason,
                tool_results=tuple(results),
                messages=tuple(messages),
                usage=usage,
            )

    return ToolLoopResult(
        text=last_text,
        steps=steps,
        stopped_reason=STOP_MAX_STEPS,
        tool_results=tuple(results),
        messages=tuple(messages),
        usage=usage,
    )
