"""LLM settings router — provider listing, health checks, Ollama integration."""

from __future__ import annotations

import logging
import os

import httpx
from fastapi import APIRouter

from app.schemas.models import (
    MODEL_SUGGESTIONS,
    HealthStatus,
    LLMProvidersResponse,
    OllamaModelInfo,
    OllamaModelsResponse,
    OllamaTestResponse,
    ProviderStatus,
)
from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/llm", tags=["llm"])

_llm_client: LLMClient | None = None

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def init(llm_client: LLMClient):
    global _llm_client
    _llm_client = llm_client


# ── Provider env-var checks (no network call needed) ────────
_PROVIDER_ENV_KEYS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "github": "GITHUB_TOKEN",
}


def _provider_configured(name: str) -> bool:
    """True if the provider has its API key set (or is local/Ollama)."""
    if name == "ollama":
        return True  # Local, no key needed
    env_key = _PROVIDER_ENV_KEYS.get(name, "")
    return bool(os.getenv(env_key))


@router.get("/providers", response_model=LLMProvidersResponse)
async def list_providers():
    """Return available LLM providers with model suggestions."""
    providers: list[ProviderStatus] = []

    for provider_name, models in MODEL_SUGGESTIONS.items():
        providers.append(
            ProviderStatus(
                name=provider_name,
                available=_provider_configured(provider_name),
                models=models,
                backend="litellm",
            )
        )

    return LLMProvidersResponse(providers=providers)


@router.get("/health", response_model=HealthStatus)
async def health_check():
    """Check connectivity — now reports LiteLLM availability."""
    # With LiteLLM we no longer have separate gRPC backends.
    # Report overall health based on whether a default provider key is set.
    from app.config import get_settings
    settings = get_settings()
    default_ok = _provider_configured(settings.default_provider)

    return HealthStatus(
        llm_chain=default_ok,
        cli_agent=default_ok,
        llm_chain_version="litellm",
        cli_agent_version="litellm",
    )


@router.get("/ollama/test", response_model=OllamaTestResponse)
async def test_ollama():
    """Test connectivity to the local Ollama server."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/version")
            if resp.status_code == 200:
                data = resp.json()
                return OllamaTestResponse(
                    connected=True,
                    version=data.get("version", "unknown"),
                )
    except Exception as e:
        logger.warning(f"Ollama connection test failed: {e}")
    return OllamaTestResponse(connected=False)


@router.get("/ollama/models", response_model=OllamaModelsResponse)
async def list_ollama_models():
    """List downloaded Ollama models and cloud suggestions."""
    cloud_models = MODEL_SUGGESTIONS.get("ollama", [])
    downloaded: list[OllamaModelInfo] = []
    connected = False

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if resp.status_code == 200:
                connected = True
                data = resp.json()
                for model in data.get("models", []):
                    name = model.get("name", "")
                    size_bytes = model.get("size", 0)
                    if size_bytes > 0:
                        size_gb = size_bytes / (1024 ** 3)
                        size_str = f"{size_gb:.1f} GB" if size_gb >= 1 else f"{size_bytes / (1024 ** 2):.0f} MB"
                    else:
                        size_str = ""
                    downloaded.append(
                        OllamaModelInfo(
                            name=name,
                            size=size_str,
                            modified_at=model.get("modified_at", ""),
                        )
                    )
    except Exception as e:
        logger.warning(f"Failed to list Ollama models: {e}")

    return OllamaModelsResponse(
        connected=connected,
        downloaded_models=downloaded,
        cloud_models=cloud_models,
    )
