"""Provider-configurable embeddings for document retrieval.

Reuses the ``LLMClient`` credential policy: a user's personal API key or
the backend-funded study-app assignment. OpenAI ``text-embedding-3-small``
is the default provider model; Google ``gemini-embedding`` is second.
"""

from __future__ import annotations

import logging
from typing import Optional

import litellm

from app.config import get_settings
from app.services.user_settings_store import (
    UserSettingsStore,
    identity_key_for_user,
)

logger = logging.getLogger(__name__)

EMBEDDING_BATCH_SIZE = 32

# Provider → default embedding model (plan decision).
PROVIDER_EMBEDDING_MODEL: dict[str, str] = {
    "openai": "text-embedding-3-small",
    "google": "gemini-embedding",
}

# Model → vector dimension. The dimension is stored per-provider in the
# vec_meta table; the vec index is rebuilt when it changes.
EMBEDDING_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "gemini-embedding": 768,
}

DEFAULT_EMBEDDING_DIMENSION = 1536


def embedding_dimension_for(model: str) -> int:
    return EMBEDDING_DIMENSIONS.get(model, DEFAULT_EMBEDDING_DIMENSION)


class EmbeddingError(RuntimeError):
    """Raised when embedding generation fails."""


class EmbeddingClient:
    """Thin async wrapper around ``litellm.aembedding``."""

    def __init__(
        self,
        user_settings_store: Optional[UserSettingsStore] = None,
        provider: str = "",
        model: str = "",
    ):
        self._user_settings_store = user_settings_store
        self._provider_override = (provider or "").strip().lower()
        self._model_override = (model or "").strip()

    def resolved_provider(self) -> str:
        if self._provider_override:
            return self._provider_override
        settings = get_settings()
        return (settings.embedding_provider or "openai").strip().lower()

    def resolved_model(self) -> str:
        if self._model_override:
            return self._model_override
        settings = get_settings()
        if settings.embedding_model:
            return settings.embedding_model
        return PROVIDER_EMBEDDING_MODEL.get(
            self.resolved_provider(), "text-embedding-3-small"
        )

    @property
    def dimension(self) -> int:
        return embedding_dimension_for(self.resolved_model())

    async def embed(
        self,
        texts: list[str],
        user_identity: Optional[dict] = None,
    ) -> list[list[float]]:
        """Embed ``texts`` in batches, returning one vector per input."""
        if not texts:
            return []
        model = self._resolve_litellm_model()
        api_key, _credential_source = await self._resolve_credential(
            user_identity=user_identity,
        )
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[start : start + EMBEDDING_BATCH_SIZE]
            vectors.extend(await self._embed_batch(model, batch, api_key))
        return vectors

    def _resolve_litellm_model(self) -> str:
        provider = self.resolved_provider()
        model = self.resolved_model()
        prefix = {"google": "gemini/"}.get(provider, "")
        if prefix and not model.startswith(prefix):
            return f"{prefix}{model}"
        return model

    async def _embed_batch(
        self,
        model: str,
        texts: list[str],
        api_key: Optional[str],
    ) -> list[list[float]]:
        kwargs: dict = {"model": model, "input": texts}
        if api_key:
            kwargs["api_key"] = api_key
        try:
            response = await litellm.aembedding(**kwargs)
        except Exception as e:
            logger.error("Embedding failed for model %s: %s", model, e)
            raise EmbeddingError(str(e)) from e
        data = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in data]

    async def _resolve_credential(
        self,
        user_identity: Optional[dict],
    ) -> tuple[Optional[str], str]:
        """Resolve the embedding API key via the LLMClient credential policy."""
        if not self._user_settings_store or not user_identity:
            return None, "backend"
        identity_key = identity_key_for_user(user_identity)
        if not identity_key:
            return None, "backend"
        prefs, _has_saved = self._user_settings_store.get_user_state(identity_key)
        llm_source = str(prefs.get("llm_source", "personal")).strip().lower()
        if llm_source == "study_app":
            _provider, _model, error_code = (
                self._user_settings_store.resolve_study_app_provider_model(
                    identity_key=identity_key,
                )
            )
            if error_code:
                # Not approved / not assigned — fall back to backend keys.
                logger.warning(
                    "Study-app embedding credential unavailable (%s); "
                    "using backend keys",
                    error_code,
                )
                return None, "backend"
            return None, "study_app_backend"
        provider = self.resolved_provider()
        runtime_credential, source, error_code = (
            await self._user_settings_store.resolve_personal_runtime_credential(
                identity_key=identity_key,
                provider=provider,
                auth_mode="api_key",
            )
        )
        if error_code or runtime_credential is None:
            return None, "backend"
        if isinstance(runtime_credential, dict):
            # Google OAuth bearer token — send as a plain API key.
            token = str(runtime_credential.get("Authorization", "")).removeprefix(
                "Bearer "
            ).strip()
            return token or None, source
        return str(runtime_credential), source
