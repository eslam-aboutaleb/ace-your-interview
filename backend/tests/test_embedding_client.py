"""Tests for the provider-configurable embedding client."""

import asyncio
import unittest
from unittest.mock import patch

from app.services.embedding_client import (
    EMBEDDING_BATCH_SIZE,
    EmbeddingClient,
    EmbeddingError,
    embedding_dimension_for,
)


class _FakeUserSettingsStore:
    def __init__(
        self,
        llm_source="personal",
        api_key=None,
        assigned_provider=None,
        assigned_model=None,
        error_code=None,
    ):
        self._llm_source = llm_source
        self._api_key = api_key
        self._assigned_provider = assigned_provider
        self._assigned_model = assigned_model
        self._error_code = error_code
        self.credential_calls = []

    def get_user_state(self, identity_key):
        return {"llm_source": self._llm_source}, True

    def resolve_study_app_provider_model(self, *, identity_key):
        return (
            self._assigned_provider,
            self._assigned_model,
            self._error_code,
        )

    async def resolve_personal_runtime_credential(
        self, *, identity_key, provider, auth_mode
    ):
        self.credential_calls.append((provider, auth_mode))
        if self._error_code:
            return None, "missing_personal", self._error_code
        if self._api_key:
            return self._api_key, "user_api_key", None
        return None, "missing_personal", "personal_credential_required"


class EmbeddingClientTests(unittest.TestCase):
    def test_dimension_per_provider(self):
        self.assertEqual(
            embedding_dimension_for("text-embedding-3-small"),
            1536,
        )
        self.assertEqual(
            embedding_dimension_for("gemini-embedding"),
            768,
        )
        self.assertEqual(
            embedding_dimension_for("unknown-model"),
            1536,
        )

    def test_default_provider_and_model(self):
        client = EmbeddingClient()
        self.assertEqual(client.resolved_provider(), "openai")
        self.assertEqual(
            client.resolved_model(), "text-embedding-3-small"
        )
        self.assertEqual(client.dimension, 1536)

    def test_overrides(self):
        client = EmbeddingClient(
            provider="google", model="gemini-embedding"
        )
        self.assertEqual(client.resolved_provider(), "google")
        self.assertEqual(client.resolved_model(), "gemini-embedding")
        self.assertEqual(client.dimension, 768)

    def test_litellm_model_prefix_for_google(self):
        client = EmbeddingClient(
            provider="google", model="gemini-embedding"
        )
        self.assertEqual(
            client._resolve_litellm_model(),
            "gemini/gemini-embedding",
        )
        client = EmbeddingClient()
        self.assertEqual(
            client._resolve_litellm_model(),
            "text-embedding-3-small",
        )

    def test_embed_batches_at_32(self):
        client = EmbeddingClient()
        calls = []

        async def fake_aembedding(**kwargs):
            calls.append(kwargs)

            class _Item:
                def __init__(self, index, vector):
                    self.index = index
                    self.embedding = vector

            class _Response:
                pass

            response = _Response()
            response.data = [
                _Item(i, [float(i), 1.0, 0.0])
                for i in range(len(kwargs["input"]))
            ]
            return response

        with patch(
            "app.services.embedding_client.litellm.aembedding",
            new=fake_aembedding,
        ):
            vectors = asyncio.run(client.embed(["text"] * 70))

        self.assertEqual(len(vectors), 70)
        self.assertEqual(len(calls), 3)
        self.assertEqual(
            [len(call["input"]) for call in calls],
            [EMBEDDING_BATCH_SIZE, EMBEDDING_BATCH_SIZE, 6],
        )
        # Vectors preserve input order within each batch.
        self.assertEqual(vectors[0], [0.0, 1.0, 0.0])
        self.assertEqual(vectors[31], [31.0, 1.0, 0.0])
        self.assertEqual(vectors[69], [5.0, 1.0, 0.0])

    def test_embed_empty_list(self):
        client = EmbeddingClient()
        self.assertEqual(asyncio.run(client.embed([])), [])

    def test_embed_uses_personal_api_key(self):
        store = _FakeUserSettingsStore(api_key="sk-personal")
        client = EmbeddingClient(user_settings_store=store)
        captured = {}

        async def fake_aembedding(**kwargs):
            captured.update(kwargs)

            class _Item:
                index = 0
                embedding = [0.1, 0.2]

            class _Response:
                pass

            response = _Response()
            response.data = [_Item()]
            return response

        with patch(
            "app.services.embedding_client.litellm.aembedding",
            new=fake_aembedding,
        ):
            asyncio.run(
                client.embed(
                    ["hello"],
                    user_identity={"user": "u1", "provider": "google"},
                )
            )

        self.assertEqual(captured.get("api_key"), "sk-personal")
        self.assertEqual(
            store.credential_calls, [("openai", "api_key")]
        )

    def test_embed_study_app_uses_backend_keys(self):
        store = _FakeUserSettingsStore(
            llm_source="study_app",
            assigned_provider="openai",
            assigned_model="gpt-4o-mini",
        )
        client = EmbeddingClient(user_settings_store=store)
        captured = {}

        async def fake_aembedding(**kwargs):
            captured.update(kwargs)

            class _Item:
                index = 0
                embedding = [0.1, 0.2]

            class _Response:
                pass

            response = _Response()
            response.data = [_Item()]
            return response

        with patch(
            "app.services.embedding_client.litellm.aembedding",
            new=fake_aembedding,
        ):
            asyncio.run(
                client.embed(
                    ["hello"],
                    user_identity={"user": "u1", "provider": "google"},
                )
            )

        self.assertNotIn("api_key", captured)
        self.assertEqual(store.credential_calls, [])

    def test_embed_failure_raises_embedding_error(self):
        client = EmbeddingClient()

        async def failing_aembedding(**kwargs):
            raise RuntimeError("provider down")

        with patch(
            "app.services.embedding_client.litellm.aembedding",
            new=failing_aembedding,
        ):
            with self.assertRaises(EmbeddingError):
                asyncio.run(client.embed(["hello"]))


if __name__ == "__main__":
    unittest.main()
