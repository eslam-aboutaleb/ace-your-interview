"""Coverage for app/services/embedding_client.py.

Complements ``tests/test_embedding_client.py`` (batching, personal key,
study-app backend key, provider failure) with the credential fallback matrix
and the Google-OAuth bearer path.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.services import embedding_client as embedding_module
from app.services.embedding_client import (
    DEFAULT_EMBEDDING_DIMENSION,
    EmbeddingClient,
    EmbeddingError,
    embedding_dimension_for,
)


def _run(coro):
    return asyncio.run(coro)


def _settings(**overrides):
    base = {
        "embedding_provider": "openai",
        "embedding_model": "",
        "default_provider": "openai",
        "default_model": "gpt-4o-mini",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _patched_settings(**overrides):
    patcher = patch.object(
        embedding_module, "get_settings", lambda: _settings(**overrides)
    )
    patcher.start()
    return patcher


class _FakeUserSettingsStore:
    def __init__(
        self,
        llm_source="personal",
        credential=None,
        error_code=None,
        study_app_error=None,
        state=None,
    ):
        self._llm_source = llm_source
        self._credential = credential
        self._error_code = error_code
        self._study_app_error = study_app_error
        self._state = state
        self.credential_calls: list[tuple] = []
        self.study_app_calls: list[str] = []

    def get_user_state(self, identity_key):
        if self._state is not None:
            return self._state, True
        return {"llm_source": self._llm_source}, True

    def resolve_study_app_provider_model(self, *, identity_key):
        self.study_app_calls.append(identity_key)
        return "openai", "gpt-4o-mini", self._study_app_error

    async def resolve_personal_runtime_credential(
        self, *, identity_key, provider, auth_mode
    ):
        self.credential_calls.append((provider, auth_mode))
        if self._error_code:
            return None, "missing_personal", self._error_code
        return self._credential, "user_credential", None


class _EmbeddingItem:
    def __init__(self, index, embedding):
        self.index = index
        self.embedding = embedding


class _EmbeddingResponse:
    def __init__(self, vectors):
        self.data = [
            _EmbeddingItem(i, vector) for i, vector in enumerate(vectors)
        ]


class _CapturingLiteLLM:
    def __init__(self, vectors_per_call=None, error: Exception | None = None):
        self.calls: list[dict] = []
        self._vectors_per_call = vectors_per_call
        self._error = error

    async def aembedding(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        if self._vectors_per_call is not None:
            return _EmbeddingResponse(self._vectors_per_call)
        count = len(kwargs["input"])
        return _EmbeddingResponse([[0.1, 0.2] for _ in range(count)])


def _embed(client, texts, user_identity=None, fake=None):
    fake = fake or _CapturingLiteLLM()
    with patch.object(embedding_module.litellm, "aembedding", new=fake.aembedding):
        return _run(client.embed(texts, user_identity=user_identity)), fake


class SettingsResolutionTests(unittest.TestCase):
    def test_provider_comes_from_settings_when_not_overridden(self):
        patcher = _patched_settings(embedding_provider="GOOGLE")
        self.addCleanup(patcher.stop)
        client = EmbeddingClient()
        self.assertEqual(client.resolved_provider(), "google")
        # No model pinned, so the provider default is used.
        self.assertEqual(client.resolved_model(), "gemini-embedding")
        self.assertEqual(client.dimension, 768)

    def test_a_whitespace_provider_setting_is_not_normalised_to_openai(self):
        # The "openai" fallback only guards a falsy setting; a whitespace-only
        # value strips to "". That still lands on the OpenAI default *model*,
        # so the behaviour is benign, but resolved_provider() is not "openai".
        patcher = _patched_settings(embedding_provider="   ")
        self.addCleanup(patcher.stop)
        client = EmbeddingClient()
        self.assertEqual(client.resolved_provider(), "")
        self.assertEqual(client.resolved_model(), "text-embedding-3-small")
        self.assertEqual(client._resolve_litellm_model(), "text-embedding-3-small")

    def test_a_missing_provider_setting_falls_back_to_openai(self):
        patcher = _patched_settings(embedding_provider="")
        self.addCleanup(patcher.stop)
        self.assertEqual(EmbeddingClient().resolved_provider(), "openai")

    def test_an_explicit_embedding_model_setting_wins_over_the_provider_default(self):
        patcher = _patched_settings(
            embedding_provider="openai", embedding_model="gemini-embedding"
        )
        self.addCleanup(patcher.stop)
        client = EmbeddingClient()
        self.assertEqual(client.resolved_model(), "gemini-embedding")
        # The dimension follows the *model*, not the provider.
        self.assertEqual(client.dimension, 768)

    def test_an_explicit_override_wins_over_settings(self):
        patcher = _patched_settings(
            embedding_provider="openai", embedding_model="text-embedding-3-small"
        )
        self.addCleanup(patcher.stop)
        client = EmbeddingClient(provider="google", model="gemini-embedding")
        self.assertEqual(client.resolved_model(), "gemini-embedding")
        self.assertEqual(client._resolve_litellm_model(), "gemini/gemini-embedding")

    def test_a_google_model_that_already_has_the_prefix_is_not_doubled(self):
        client = EmbeddingClient(
            provider="google", model="gemini/gemini-embedding"
        )
        self.assertEqual(
            client._resolve_litellm_model(), "gemini/gemini-embedding"
        )

    def test_google_prefix_comes_from_the_provider_setting(self):
        patcher = _patched_settings(embedding_provider="google")
        self.addCleanup(patcher.stop)
        self.assertEqual(
            EmbeddingClient()._resolve_litellm_model(), "gemini/gemini-embedding"
        )

    def test_an_unknown_embedding_model_uses_the_default_dimension(self):
        self.assertEqual(
            embedding_dimension_for("some-future-model"), DEFAULT_EMBEDDING_DIMENSION
        )
        self.assertEqual(embedding_dimension_for(""), DEFAULT_EMBEDDING_DIMENSION)


class CredentialResolutionTests(unittest.TestCase):
    def test_no_store_means_backend_credentials(self):
        client = EmbeddingClient()
        api_key, source = _run(
            client._resolve_credential(
                user_identity={"user": "u1", "provider": "google"}
            )
        )
        self.assertIsNone(api_key)
        self.assertEqual(source, "backend")

    def test_no_identity_means_backend_credentials(self):
        store = _FakeUserSettingsStore(credential="sk-x")
        client = EmbeddingClient(user_settings_store=store)
        api_key, source = _run(client._resolve_credential(user_identity=None))
        self.assertIsNone(api_key)
        self.assertEqual(source, "backend")
        self.assertEqual(store.credential_calls, [])

    def test_an_identity_without_a_user_meaning_backend_credentials(self):
        store = _FakeUserSettingsStore(credential="sk-x")
        client = EmbeddingClient(user_settings_store=store)
        api_key, source = _run(
            client._resolve_credential(user_identity={"provider": "google"})
        )
        self.assertIsNone(api_key)
        self.assertEqual(source, "backend")
        self.assertEqual(store.credential_calls, [])

    def test_a_study_app_source_with_an_error_code_falls_back_to_backend(self):
        store = _FakeUserSettingsStore(
            llm_source="study_app", study_app_error="study_app_not_assigned"
        )
        client = EmbeddingClient(user_settings_store=store)
        api_key, source = _run(
            client._resolve_credential(
                user_identity={"user": "u1", "provider": "google"}
            )
        )
        self.assertIsNone(api_key)
        self.assertEqual(source, "backend")
        self.assertEqual(store.study_app_calls, ["google:u1"])
        self.assertEqual(store.credential_calls, [])

    def test_an_approved_study_app_source_uses_backend_keys(self):
        store = _FakeUserSettingsStore(llm_source="study_app")
        client = EmbeddingClient(user_settings_store=store)
        api_key, source = _run(
            client._resolve_credential(
                user_identity={"user": "u1", "provider": "google"}
            )
        )
        self.assertIsNone(api_key)
        self.assertEqual(source, "study_app_backend")

    def test_the_llm_source_is_matched_case_insensitively(self):
        store = _FakeUserSettingsStore(llm_source="  STUDY_APP ")
        client = EmbeddingClient(user_settings_store=store)
        _api_key, source = _run(
            client._resolve_credential(
                user_identity={"user": "u1", "provider": "google"}
            )
        )
        self.assertEqual(source, "study_app_backend")

    def test_a_personal_source_with_an_error_code_falls_back_to_backend(self):
        store = _FakeUserSettingsStore(
            credential=None, error_code="personal_credential_required"
        )
        client = EmbeddingClient(user_settings_store=store)
        api_key, source = _run(
            client._resolve_credential(
                user_identity={"user": "u1", "provider": "google"}
            )
        )
        self.assertIsNone(api_key)
        self.assertEqual(source, "backend")

    def test_a_personal_source_with_no_credential_falls_back_to_backend(self):
        store = _FakeUserSettingsStore(credential=None)
        client = EmbeddingClient(user_settings_store=store)
        api_key, source = _run(
            client._resolve_credential(
                user_identity={"user": "u1", "provider": "google"}
            )
        )
        self.assertIsNone(api_key)
        self.assertEqual(source, "backend")

    def test_a_google_oauth_bearer_is_sent_as_a_plain_api_key(self):
        store = _FakeUserSettingsStore(
            credential={"Authorization": "Bearer ya29.token"}
        )
        client = EmbeddingClient(user_settings_store=store)
        api_key, source = _run(
            client._resolve_credential(
                user_identity={"user": "u1", "provider": "google"}
            )
        )
        self.assertEqual(api_key, "ya29.token")
        self.assertEqual(source, "user_credential")

    def test_a_google_oauth_bearer_without_the_prefix_is_passed_through(self):
        store = _FakeUserSettingsStore(credential={"Authorization": "  raw-token "})
        client = EmbeddingClient(user_settings_store=store)
        api_key, _source = _run(
            client._resolve_credential(
                user_identity={"user": "u1", "provider": "google"}
            )
        )
        self.assertEqual(api_key, "raw-token")

    def test_an_empty_google_bearer_degrades_to_backend_keys(self):
        store = _FakeUserSettingsStore(credential={"Authorization": "Bearer "})
        client = EmbeddingClient(user_settings_store=store)
        api_key, source = _run(
            client._resolve_credential(
                user_identity={"user": "u1", "provider": "google"}
            )
        )
        self.assertIsNone(api_key)
        self.assertEqual(source, "user_credential")

    def test_an_empty_credential_dict_degrades_to_backend_keys(self):
        store = _FakeUserSettingsStore(credential={})
        client = EmbeddingClient(user_settings_store=store)
        api_key, source = _run(
            client._resolve_credential(
                user_identity={"user": "u1", "provider": "google"}
            )
        )
        self.assertIsNone(api_key)
        self.assertEqual(source, "user_credential")

    def test_a_plain_string_credential_is_used_verbatim(self):
        store = _FakeUserSettingsStore(credential="sk-personal")
        client = EmbeddingClient(user_settings_store=store)
        api_key, source = _run(
            client._resolve_credential(
                user_identity={"user": "u1", "provider": "google"}
            )
        )
        self.assertEqual(api_key, "sk-personal")
        self.assertEqual(source, "user_credential")


class EmbedTests(unittest.TestCase):
    def test_backend_credentials_omit_the_api_key_kwarg(self):
        client = EmbeddingClient()
        vectors, fake = _embed(client, ["hello"])
        self.assertEqual(vectors, [[0.1, 0.2]])
        self.assertNotIn("api_key", fake.calls[0])

    def test_a_user_key_is_forwarded_to_the_provider(self):
        store = _FakeUserSettingsStore(credential="sk-personal")
        client = EmbeddingClient(user_settings_store=store)
        _vectors, fake = _embed(
            client,
            ["hello"],
            user_identity={"user": "u1", "provider": "google"},
        )
        self.assertEqual(fake.calls[0]["api_key"], "sk-personal")

    def test_the_resolved_litellm_model_is_used(self):
        client = EmbeddingClient(provider="google", model="gemini-embedding")
        _vectors, fake = _embed(client, ["hello"])
        self.assertEqual(fake.calls[0]["model"], "gemini/gemini-embedding")

    def test_empty_input_makes_no_provider_call(self):
        client = EmbeddingClient()
        fake = _CapturingLiteLLM()
        with patch.object(
            embedding_module.litellm, "aembedding", new=fake.aembedding
        ):
            self.assertEqual(_run(client.embed([])), [])
        self.assertEqual(fake.calls, [])

    def test_vectors_are_returned_in_index_order_not_response_order(self):
        # A provider that returns items out of order must still produce
        # vectors aligned with the input order.
        class _ShuffledLiteLLM(_CapturingLiteLLM):
            async def aembedding(self, **kwargs):
                self.calls.append(kwargs)
                response = _EmbeddingResponse([])
                response.data = [
                    _EmbeddingItem(2, [2.0]),
                    _EmbeddingItem(0, [0.0]),
                    _EmbeddingItem(1, [1.0]),
                ]
                return response

        client = EmbeddingClient()
        fake = _ShuffledLiteLLM()
        with patch.object(
            embedding_module.litellm, "aembedding", new=fake.aembedding
        ):
            vectors = _run(client.embed(["a", "b", "c"]))
        self.assertEqual(vectors, [[0.0], [1.0], [2.0]])

    def test_a_provider_failure_becomes_an_embedding_error(self):
        client = EmbeddingClient()
        fake = _CapturingLiteLLM(error=RuntimeError("rate limited"))
        with patch.object(
            embedding_module.litellm, "aembedding", new=fake.aembedding
        ):
            with self.assertRaises(EmbeddingError) as ctx:
                _run(client.embed(["hello"]))
        self.assertIn("rate limited", str(ctx.exception))

    def test_a_credential_failure_never_reaches_the_provider(self):
        store = _FakeUserSettingsStore(error_code="personal_credential_required")
        client = EmbeddingClient(user_settings_store=store)
        _vectors, fake = _embed(
            client,
            ["hello"],
            user_identity={"user": "u1", "provider": "google"},
        )
        self.assertNotIn("api_key", fake.calls[0])

    def test_a_study_app_credential_error_never_reaches_the_provider(self):
        store = _FakeUserSettingsStore(
            llm_source="study_app", study_app_error="approval_required"
        )
        client = EmbeddingClient(user_settings_store=store)
        _vectors, fake = _embed(
            client,
            ["hello"],
            user_identity={"user": "u1", "provider": "google"},
        )
        self.assertNotIn("api_key", fake.calls[0])
        self.assertEqual(store.credential_calls, [])

    def test_the_user_settings_store_may_report_no_saved_state(self):
        store = _FakeUserSettingsStore(state={})
        client = EmbeddingClient(user_settings_store=store)
        # An empty state means llm_source defaults to "personal", so the
        # personal credential resolver is consulted.
        _vectors, fake = _embed(
            client,
            ["hello"],
            user_identity={"user": "u1", "provider": "google"},
        )
        self.assertEqual(store.credential_calls, [("openai", "api_key")])
        self.assertNotIn("api_key", fake.calls[0])


if __name__ == "__main__":
    unittest.main()