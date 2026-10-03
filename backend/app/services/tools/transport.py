"""Default tool-loop transport: a thin, injectable wrapper over LiteLLM.

Why this does not call ``LLMClient.completion``
-----------------------------------------------
``LLMClient.completion`` takes a *string* prompt and returns only the text
content of the response, so it cannot (a) carry a multi-turn transcript that
includes ``role="tool"`` messages, or (b) return the ``tool_calls`` the loop
needs to read. Wiring that through would require a ``messages`` parameter and a
``tool_calls`` field on the result, which live in a file this change set does not
own. That follow-up is recorded in the plan; until it lands, the loop issues the
provider call itself.

The two-tier credential policy is not bypassed by that choice. The enrichment
path is invoked as ``gather_context(flow=..., query=..., ...)`` and receives no
``llm_config`` and no ``user_identity`` from any of its six call sites, so no
per-user provider/model pin participates: the only model in play is the backend
default from ``Settings.default_provider`` / ``Settings.default_model``, and the
only credentials are the backend ones LiteLLM already reads from the
environment. ``Settings`` is injected explicitly, so this cannot silently pick
up a user preference.

Every field is injected so tests never construct a live transport.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import litellm

from app.services.llm_client import _resolve_model
from app.services.tools.agentic import ModelTurn

logger = logging.getLogger(__name__)

DEFAULT_TOOL_LOOP_MAX_TOKENS = 1200
DEFAULT_TOOL_LOOP_TEMPERATURE = 0.2
DEFAULT_TOOL_LOOP_TIMEOUT_SECONDS = 45.0


@dataclass
class LiteLLMToolTransport:
    """Issues one tool-aware completion per :func:`run_tool_loop` step."""

    provider: str = ""
    model: str = ""
    max_tokens: int = DEFAULT_TOOL_LOOP_MAX_TOKENS
    temperature: float = DEFAULT_TOOL_LOOP_TEMPERATURE
    timeout_seconds: float = DEFAULT_TOOL_LOOP_TIMEOUT_SECONDS
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def resolved_model(self) -> str:
        if str(self.model or "").strip():
            return str(self.model).strip()
        return _resolve_model(str(self.provider or "default"), "")

    async def complete(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        tool_choice: Any,
        model: str = "",
        max_tokens: int = 0,
        temperature: float = 0.0,
    ) -> ModelTurn:
        target = str(model or "").strip() or self.resolved_model()
        kwargs: dict[str, Any] = dict(
            model=target,
            messages=list(messages),
            tools=list(tools),
            temperature=float(temperature or self.temperature),
            max_tokens=int(max_tokens or self.max_tokens),
            timeout=float(self.timeout_seconds),
        )
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        kwargs.update(self.extra_kwargs)
        response = await litellm.acompletion(**kwargs)
        turn = ModelTurn.from_response(response)
        usage = getattr(response, "usage", None)
        if usage is not None:
            turn = ModelTurn(
                text=turn.text,
                tool_calls=turn.tool_calls,
                finish_reason=turn.finish_reason,
                usage=_usage_to_dict(usage),
                success=turn.success,
                error=turn.error,
            )
        logger.info(
            "tool_loop_turn model=%s tool_calls=%s finish_reason=%s",
            target,
            len(turn.tool_calls),
            turn.finish_reason,
        )
        return turn


def _usage_to_dict(usage: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = getattr(usage, key, None)
        if value is None and isinstance(usage, dict):
            value = usage.get(key)
        if value is None:
            continue
        try:
            out[key] = int(value)
        except (TypeError, ValueError):
            out[key] = 0
    return out
