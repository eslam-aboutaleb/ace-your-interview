"""Unified LLM client using LiteLLM with per-user credential resolution."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import deque
from typing import Any, Optional

import litellm

from app.config import get_settings
from app.schemas.models import LLMConfigRequest
from app.services.user_settings_store import UserSettingsStore, identity_key_for_user
from app.services.llm_policy import (
    APPROVAL_REQUIRED_CODE,
    APPROVAL_REQUIRED_MESSAGE,
    PERSONAL_CREDENTIAL_REQUIRED_CODE,
    PERSONAL_CREDENTIAL_REQUIRED_MESSAGE,
    STUDY_APP_NOT_ASSIGNED_CODE,
    STUDY_APP_NOT_ASSIGNED_MESSAGE,
)

logger = logging.getLogger(__name__)

# Suppress verbose litellm logs unless debugging
litellm.suppress_debug_info = True

# ── Provider → LiteLLM model-prefix mapping ─────────────────
_PROVIDER_PREFIX: dict[str, str] = {
    "openai": "",              # e.g. "gpt-4o-mini"
    "anthropic": "anthropic/", # e.g. "anthropic/claude-3-5-sonnet-20241022"
    "google": "gemini/",       # e.g. "gemini/gemini-1.5-pro"
    "groq": "groq/",           # e.g. "groq/llama-3.3-70b-versatile"
    "ollama": "ollama/",       # e.g. "ollama/llama3.2"
    "github": "github/",       # e.g. "github/gpt-4o-mini"
}

_TASK_ROUTE_KEYS = {"planner", "retrieval", "eval", "final"}

# ── Capability map (Stage 2.3) ──────────────────────────────
# Providers whose OpenAI-compatible surface accepts
# ``response_format={"type": "json_object"}``. Anthropic has no
# equivalent parameter and must fall back to the shared salvage parser.
JSON_MODE_CAPABLE_PROVIDERS: frozenset[str] = frozenset(
    {"openai", "groq", "ollama", "google", "github"}
)

# Flow-level output ceilings. A caller-supplied ``max_tokens`` (or a saved user
# preference) may lower these but never raise them.
DEFAULT_MAX_TOKENS = 4096
FLOW_MAX_TOKENS_CAPS: dict[str, int] = {
    "interview_question": 600,
    "interview_eval": 2000,
    "custom_topic_batch": 4000,
    "progress_summary": 800,
    "voice_turn": 600,
}

STRUCTURED_TEMPERATURE = 0.1
FREE_FORM_TEMPERATURE_FLOOR = 0.7

# ── Terminal / retryable outcome codes ──────────────────────
BUDGET_EXCEEDED_CODE = "llm_budget_exceeded"
TRUNCATED_CODE = "llm_truncated"
CALL_FAILED_CODE = "llm_call_failed"
#: Codes a retry loop must not spin on: retrying cannot change the outcome.
TERMINAL_ERROR_CODES: frozenset[str] = frozenset({BUDGET_EXCEEDED_CODE, TRUNCATED_CODE})

_FENCED_JSON = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", flags=re.DOTALL)


def parse_json_object(raw: str) -> dict[str, Any]:
    """Best-effort extraction of a single JSON object from an LLM response.

    Single shared implementation for every flow: direct parse, then fenced
    block, then first-brace/last-brace slice. Returns ``{}`` on failure.
    """
    text = str(raw or "").strip()
    if not text:
        return {}

    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass

    fenced = _FENCED_JSON.search(text)
    if fenced:
        try:
            data = json.loads(fenced.group(1))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            pass

    return {}


def _resolve_model(provider: str, model: str) -> str:
    """Convert (provider, model) into a LiteLLM-compatible model string."""
    from app.schemas.models import MODEL_SUGGESTIONS

    provider = provider.lower()
    if provider in ("default", ""):
        settings = get_settings()
        provider = settings.default_provider
        model = model or settings.default_model

    if not model:
        suggestions = MODEL_SUGGESTIONS.get(provider, [])
        model = suggestions[0] if suggestions else "gpt-4o-mini"

    prefix = _PROVIDER_PREFIX.get(provider, "")
    if prefix and not model.startswith(prefix):
        return f"{prefix}{model}"
    return model


def _load_task_route_map(raw: str) -> dict[str, str]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Invalid STUDY_LLM_TASK_ROUTE_MAP_JSON, ignoring")
        return {}
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in parsed.items():
        task = str(key or "").strip().lower()
        model = str(value or "").strip()
        if task in _TASK_ROUTE_KEYS and model:
            out[task] = model
    return out


def _provider_supports_json_mode(provider: str) -> bool:
    return str(provider or "").strip().lower() in JSON_MODE_CAPABLE_PROVIDERS


def _extract_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return {}

    def _int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    out: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        if hasattr(usage, key):
            out[key] = _int(getattr(usage, key))
        elif isinstance(usage, dict) and key in usage:
            out[key] = _int(usage[key])
    return out


class _PerUserCallBudget:
    """Sliding-window per-user limiter for provider calls.

    Bounds how many LLM calls one user identity can issue inside
    ``window_seconds``. Bounded memory: at most ``max_calls`` timestamps are
    retained per user, and expired identities are pruned.
    """

    def __init__(self, max_calls: int, window_seconds: float):
        self._max_calls = max(0, int(max_calls))
        self._window = max(0.001, float(window_seconds))
        self._events: dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, identity_key: str) -> tuple[bool, float]:
        """Return ``(allowed, retry_after_seconds)``."""
        if self._max_calls <= 0:
            return True, 0.0
        key = identity_key or "__anonymous__"
        now = time.monotonic()
        async with self._lock:
            bucket = self._events.setdefault(key, deque())
            cutoff = now - self._window
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self._max_calls:
                retry_after = max(0.0, bucket[0] + self._window - now)
                return False, round(retry_after, 3)
            bucket.append(now)
            if len(self._events) > 10_000:
                self._prune_locked(cutoff)
        return True, 0.0

    def _prune_locked(self, cutoff: float) -> None:
        stale = [
            key
            for key, bucket in self._events.items()
            if not bucket or bucket[-1] <= cutoff
        ]
        for key in stale:
            self._events.pop(key, None)

    def reset(self) -> None:
        self._events.clear()


class LLMClient:
    """Thin async wrapper around litellm.acompletion."""

    #: Populated per-process from settings; reset alongside config changes.
    _budget: _PerUserCallBudget = _PerUserCallBudget(0, 60.0)
    _budget_signature: tuple[int, float] = (0, 60.0)
    _inflight: dict[str, asyncio.Semaphore] = {}
    _inflight_lock = asyncio.Lock()

    def __init__(self, user_settings_store: Optional[UserSettingsStore] = None):
        self._user_settings_store = user_settings_store

    # ── limits ──────────────────────────────────────────────

    @classmethod
    async def _budget_for(cls, settings: Any) -> _PerUserCallBudget:
        max_calls = int(getattr(settings, "llm_user_call_budget", 0) or 0)
        window = float(getattr(settings, "llm_user_call_budget_window_seconds", 60.0) or 60.0)
        signature = (max_calls, window)
        if cls._budget_signature != signature:
            cls._budget = _PerUserCallBudget(max_calls, window)
            cls._budget_signature = signature
        return cls._budget

    @classmethod
    async def _semaphore_for(cls, identity_key: str) -> asyncio.Semaphore:
        limit = 4
        settings = get_settings()
        configured = int(getattr(settings, "llm_user_max_concurrency", 0) or 0)
        if configured > 0:
            limit = configured
        key = identity_key or "__anonymous__"
        async with cls._inflight_lock:
            sem = cls._inflight.get(key)
            if sem is None:
                sem = asyncio.Semaphore(limit)
                cls._inflight[key] = sem
        return sem

    # ── main entry point ────────────────────────────────────

    async def completion(
        self,
        prompt: str,
        llm_config: Optional[LLMConfigRequest] = None,
        user_identity: Optional[dict] = None,
        task: Optional[str] = None,
        *,
        system: str = "",
        structured: bool = False,
        max_tokens_cap: Optional[int] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
    ) -> dict:
        """Send a prompt and return a standardised result dict.

        Args:
            prompt: user-turn content.
            system: optional system-role content, emitted as a real ``system``
                message rather than flattened into the user turn.
            structured: request JSON-mode output where the provider supports it.
            max_tokens_cap: per-flow ceiling. A caller/config/preference value
                may lower the cap but is clamped so it can never raise it.
            tools: OpenAI-style tool definitions. Mutually exclusive with
                ``structured`` (``response_format``).
        """
        settings = get_settings()
        structured = bool(structured)
        tools = list(tools or [])
        if structured and tools:
            logger.warning(
                "completion() received both structured and tools; honouring tools and dropping response_format"
            )
            structured = False

        provider_str = "default"
        model_str = ""
        model_locked = False
        temperature = (
            STRUCTURED_TEMPERATURE
            if structured
            else float(getattr(settings, "default_temperature", FREE_FORM_TEMPERATURE_FLOOR))
        )
        max_tokens = DEFAULT_MAX_TOKENS

        explicit_temperature: Optional[float] = None
        explicit_max_tokens: Optional[int] = None
        if llm_config:
            provider_str = llm_config.provider.value
            model_str = llm_config.model or ""
            if model_str:
                model_locked = True
            # 0.0 means "unset" (Stage 2.6); the field default cannot express
            # "explicitly deterministic" without collapsing to the default.
            if llm_config.temperature and llm_config.temperature > 0:
                explicit_temperature = float(llm_config.temperature)
                temperature = explicit_temperature
            if llm_config.max_tokens and llm_config.max_tokens > 0:
                explicit_max_tokens = int(llm_config.max_tokens)
                max_tokens = explicit_max_tokens

        runtime_credential: Optional[str | dict[str, str]] = None
        credential_source = "backend"
        policy_error_code = ""
        policy_error_message = ""

        identity_key = identity_key_for_user(user_identity)
        if self._user_settings_store and identity_key:
            prefs, _has_saved = self._user_settings_store.get_user_state(identity_key)
            llm_source = str(prefs.get("llm_source", "personal")).strip().lower()

            if explicit_temperature is None:
                try:
                    pref_temperature = float(prefs.get("temperature", temperature))
                except (TypeError, ValueError):
                    pass
                else:
                    if pref_temperature > 0:
                        temperature = pref_temperature
            if explicit_max_tokens is None:
                try:
                    pref_max_tokens = int(prefs.get("max_tokens", 0))
                    if pref_max_tokens > 0:
                        max_tokens = pref_max_tokens
                except (TypeError, ValueError):
                    pass

            if llm_source == "study_app":
                assigned_provider, assigned_model, policy_error_code = (
                    self._user_settings_store.resolve_study_app_provider_model(
                        identity_key=identity_key,
                    )
                )
                if assigned_provider:
                    provider_str = assigned_provider
                if assigned_model:
                    model_str = assigned_model
                    model_locked = True
                credential_source = "study_app_backend"
                if policy_error_code == APPROVAL_REQUIRED_CODE:
                    policy_error_message = APPROVAL_REQUIRED_MESSAGE
                elif policy_error_code == STUDY_APP_NOT_ASSIGNED_CODE:
                    policy_error_message = STUDY_APP_NOT_ASSIGNED_MESSAGE
            else:
                if provider_str in ("default", ""):
                    provider_str = str(prefs.get("provider", "default"))
                if not model_str:
                    model_str = str(prefs.get("model", ""))
                    if model_str:
                        model_locked = True
                effective_provider = (
                    provider_str if provider_str not in ("default", "") else settings.default_provider
                )
                auth_mode = str(prefs.get("auth_mode", "api_key"))
                (
                    runtime_credential,
                    credential_source,
                    policy_error_code,
                ) = await self._user_settings_store.resolve_personal_runtime_credential(
                    identity_key=identity_key,
                    provider=effective_provider,
                    auth_mode=auth_mode,
                )
                if policy_error_code == PERSONAL_CREDENTIAL_REQUIRED_CODE:
                    policy_error_message = PERSONAL_CREDENTIAL_REQUIRED_MESSAGE
        else:
            # Non-authenticated/background contexts continue using backend settings.
            credential_source = "backend"

        # Explicit task only (Stage 2.4). Prompt text is never inspected.
        resolved_task = str(task or "").strip().lower() or "final"
        if resolved_task not in _TASK_ROUTE_KEYS:
            resolved_task = "final"
        if settings.llm_task_routing_enabled and not model_locked:
            route_map = _load_task_route_map(settings.llm_task_route_map_json)
            routed_model = route_map.get(resolved_task, "")
            if routed_model:
                model_str = routed_model

        resolved_model = _resolve_model(provider_str, model_str)
        actual_provider = (
            provider_str if provider_str not in ("default", "") else settings.default_provider
        )
        actual_provider = str(actual_provider or "").strip().lower()

        cap = max_tokens_cap if max_tokens_cap and max_tokens_cap > 0 else None
        if cap is None:
            cap = FLOW_MAX_TOKENS_CAPS.get(resolved_task)
        if cap:
            max_tokens = max(1, min(max_tokens, int(cap)))

        base_metadata = {
            "provider": actual_provider,
            "model": resolved_model,
            "task": resolved_task,
            "credential_source": (
                "blocked_unapproved" if policy_error_code == APPROVAL_REQUIRED_CODE else credential_source
            ),
        }

        if policy_error_code:
            return {
                "success": False,
                "analysis": "",
                "metadata": dict(base_metadata),
                "error": policy_error_message,
                "error_code": policy_error_code,
                "finish_reason": "",
                "usage": {},
            }

        # ── per-user call budget (Stage 3.1) ──────────────────
        budget = await self._budget_for(settings)
        allowed, retry_after = await budget.acquire(identity_key)
        if not allowed:
            logger.warning(
                "llm_user_call_budget_exhausted identity=%s retry_after_s=%s",
                identity_key or "anonymous",
                retry_after,
            )
            return {
                "success": False,
                "analysis": "",
                "metadata": dict(base_metadata, budget_limited=True),
                "error": (
                    "LLM call budget exceeded. Retry in "
                    f"{retry_after:.0f}s."
                ),
                "error_code": BUDGET_EXCEEDED_CODE,
                "finish_reason": "",
                "usage": {},
            }

        messages: list[dict[str, str]] = []
        if str(system or "").strip():
            messages.append({"role": "system", "content": str(system)})
        messages.append({"role": "user", "content": prompt})

        completion_kwargs: dict[str, Any] = dict(
            model=resolved_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=120,
        )
        if tools:
            completion_kwargs["tools"] = tools
            if tool_choice is not None:
                completion_kwargs["tool_choice"] = tool_choice
        elif structured and _provider_supports_json_mode(actual_provider):
            completion_kwargs["response_format"] = {"type": "json_object"}

        if runtime_credential is not None:
            if isinstance(runtime_credential, dict):
                # Google OAuth Bearer token – litellm's gemini/ provider embeds
                # the api_key in the URL as ?key=, which breaks with dicts/tokens.
                # Route through Google's OpenAI-compatible endpoint instead so
                # the token is sent as a standard Authorization header.
                auth_value = runtime_credential.get("Authorization", "")
                token = auth_value.removeprefix("Bearer ").strip()
                completion_kwargs["api_key"] = token
                completion_kwargs["api_base"] = (
                    "https://generativelanguage.googleapis.com/v1beta/openai"
                )
                # Switch from gemini/ to openai/ so litellm uses the OpenAI path
                cur_model = completion_kwargs["model"]
                if cur_model.startswith("gemini/"):
                    completion_kwargs["model"] = "openai/" + cur_model[len("gemini/"):]
            else:
                completion_kwargs["api_key"] = runtime_credential

        semaphore = await self._semaphore_for(identity_key)
        async with semaphore:
            try:
                response = await litellm.acompletion(**completion_kwargs)
            except Exception as e:
                logger.error("LiteLLM completion error (%s): %s", resolved_model, e)
                return {
                    "success": False,
                    "analysis": "",
                    "metadata": dict(base_metadata),
                    "error": str(e),
                    "error_code": CALL_FAILED_CODE,
                    "finish_reason": "",
                    "usage": {},
                }

        text = response.choices[0].message.content or ""
        finish_reason = str(getattr(response.choices[0], "finish_reason", "") or "")
        usage = _extract_usage(response)
        if usage:
            logger.info(
                "llm_usage provider=%s model=%s task=%s input=%s output=%s total=%s",
                actual_provider,
                resolved_model,
                resolved_task,
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                usage.get("total_tokens", 0),
            )

        metadata = dict(base_metadata)
        metadata["usage"] = usage
        if usage:
            metadata["input_tokens"] = usage.get("input_tokens", 0)
            metadata["output_tokens"] = usage.get("output_tokens", 0)
            metadata["total_tokens"] = usage.get("total_tokens", 0)

        # A truncated response is not a parse failure: re-sending the identical
        # prompt at the identical cap cannot help, so surface it distinctly.
        truncated = finish_reason.lower() == "length"
        result = {
            "success": not truncated,
            "analysis": text,
            "metadata": metadata,
            "error": (
                f"Response truncated at max_tokens={max_tokens}." if truncated else ""
            ),
            "error_code": TRUNCATED_CODE if truncated else "",
            "finish_reason": finish_reason,
            "usage": usage,
        }
        return result

    async def health_check(self, provider: str = "openai") -> dict:
        """Lightweight connectivity check for a provider."""
        try:
            from app.schemas.models import MODEL_SUGGESTIONS

            suggestions = MODEL_SUGGESTIONS.get(provider, ["gpt-4o-mini"])
            model = _resolve_model(provider, suggestions[0])

            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
                timeout=10,
            )
            return {
                "healthy": True,
                "service_name": provider,
                "version": getattr(response, "model", model),
            }
        except Exception as e:
            logger.warning("Health check failed for %s: %s", provider, e)
            return {
                "healthy": False,
                "service_name": provider,
                "version": "",
            }