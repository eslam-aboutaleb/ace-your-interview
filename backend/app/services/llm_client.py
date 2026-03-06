"""Unified LLM client using LiteLLM — supports OpenAI, Anthropic, Google, Groq, Ollama, GitHub Models."""

from __future__ import annotations

import logging
from typing import Optional

import litellm

from app.config import get_settings
from app.schemas.models import LLMConfigRequest

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
        # Pick the first suggested model for this provider
        suggestions = MODEL_SUGGESTIONS.get(provider, [])
        model = suggestions[0] if suggestions else "gpt-4o-mini"

    prefix = _PROVIDER_PREFIX.get(provider, "")

    # Don't double-prefix (e.g. if user already passed "anthropic/claude-...")
    if prefix and not model.startswith(prefix):
        return f"{prefix}{model}"
    return model


class LLMClient:
    """Thin async wrapper around litellm.acompletion."""

    async def completion(
        self,
        prompt: str,
        llm_config: Optional[LLMConfigRequest] = None,
    ) -> dict:
        """Send a prompt and return a standardised result dict.

        Returns:
            {
                "success": bool,
                "analysis": str,          # The LLM's text response
                "metadata": {"provider": str, "model": str},
                "error": str,
            }
        """
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

        resolved_model = _resolve_model(provider_str, model_str)

        # Resolve the actual provider name for metadata
        actual_provider = provider_str if provider_str not in ("default", "") else settings.default_provider

        try:
            response = await litellm.acompletion(
                model=resolved_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=120,
            )

            text = response.choices[0].message.content or ""
            return {
                "success": True,
                "analysis": text,
                "metadata": {"provider": actual_provider, "model": resolved_model},
                "error": "",
            }
        except Exception as e:
            logger.error("LiteLLM completion error (%s): %s", resolved_model, e)
            return {
                "success": False,
                "analysis": "",
                "metadata": {"provider": actual_provider, "model": resolved_model},
                "error": str(e),
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
