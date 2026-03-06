"""Unified LLM client using LiteLLM with per-user credential resolution."""

from __future__ import annotations

import logging
from typing import Optional

import litellm

from app.config import get_settings
from app.schemas.models import LLMConfigRequest
from app.services.user_settings_store import UserSettingsStore, identity_key_for_user

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


class LLMClient:
    """Thin async wrapper around litellm.acompletion."""

    def __init__(self, user_settings_store: Optional[UserSettingsStore] = None):
        self._user_settings_store = user_settings_store

    async def completion(
        self,
        prompt: str,
        llm_config: Optional[LLMConfigRequest] = None,
        user_identity: Optional[dict] = None,
    ) -> dict:
        """Send a prompt and return a standardised result dict."""
        settings = get_settings()

        provider_str = "default"
        model_str = ""
        temperature = settings.default_temperature
        max_tokens = 4096

        if llm_config:
            provider_str = llm_config.provider.value
            model_str = llm_config.model or ""
            if llm_config.temperature > 0:
                temperature = llm_config.temperature
            if llm_config.max_tokens > 0:
                max_tokens = llm_config.max_tokens

        runtime_credential: Optional[str | dict[str, str]] = None
        credential_source = "backend"
        policy_error_code = ""
        policy_error_message = ""

        identity_key = identity_key_for_user(user_identity)
        if self._user_settings_store and identity_key:
            prefs, _has_saved = self._user_settings_store.get_user_state(identity_key)

            if provider_str in ("default", ""):
                provider_str = str(prefs.get("provider", "default"))
            if not model_str:
                model_str = str(prefs.get("model", ""))
            if llm_config is None or llm_config.temperature <= 0:
                try:
                    temperature = float(prefs.get("temperature", temperature))
                except (TypeError, ValueError):
                    pass
            if llm_config is None or llm_config.max_tokens <= 0:
                try:
                    pref_max_tokens = int(prefs.get("max_tokens", 0))
                    if pref_max_tokens > 0:
                        max_tokens = pref_max_tokens
                except (TypeError, ValueError):
                    pass

            effective_provider = (
                provider_str if provider_str not in ("default", "") else settings.default_provider
            )
            auth_mode = str(prefs.get("auth_mode", "api_key"))
            runtime_credential, credential_source, policy_error_code = await self._user_settings_store.resolve_runtime_credential(
                identity_key=identity_key,
                provider=effective_provider,
                auth_mode=auth_mode,
            )
            if policy_error_code:
                policy_error_message = (
                    "Admin approval required for backend LLM service. "
                    "Add your own LLM credential in User Settings or ask admin approval."
                )

        resolved_model = _resolve_model(provider_str, model_str)
        actual_provider = (
            provider_str if provider_str not in ("default", "") else settings.default_provider
        )

        if credential_source == "blocked_unapproved":
            return {
                "success": False,
                "analysis": "",
                "metadata": {
                    "provider": actual_provider,
                    "model": resolved_model,
                    "credential_source": credential_source,
                },
                "error": policy_error_message,
                "error_code": policy_error_code,
            }

        completion_kwargs = dict(
            model=resolved_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=120,
        )
        if runtime_credential is not None:
            completion_kwargs["api_key"] = runtime_credential

        try:
            response = await litellm.acompletion(**completion_kwargs)

            text = response.choices[0].message.content or ""
            return {
                "success": True,
                "analysis": text,
                "metadata": {
                    "provider": actual_provider,
                    "model": resolved_model,
                    "credential_source": credential_source,
                },
                "error": "",
            }
        except Exception as e:
            logger.error("LiteLLM completion error (%s): %s", resolved_model, e)
            return {
                "success": False,
                "analysis": "",
                "metadata": {
                    "provider": actual_provider,
                    "model": resolved_model,
                    "credential_source": credential_source,
                },
                "error": str(e),
                "error_code": "",
            }

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
