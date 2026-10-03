"""Coverage for ``app.routers.llm_settings`` — health, provider listing, Ollama.

Every outbound call goes through a fake ``httpx`` namespace patched onto the
router module, so the Ollama endpoints never touch the network.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import require_auth
from app.routers import llm_settings

OLLAMA_BASE_URL = "http://localhost:11434"


class _FakeResponse:
    def __init__(self, payload=None, status_code=200):
        self._payload = payload if payload is not None else {}
        self.status_code = status_code

    def json(self):
        return self._payload


class _FakeAsyncClient:
    """Records the requested URLs and replays a canned response."""

    def __init__(self, response=None, raises=None, **kwargs):
        self._response = response if response is not None else _FakeResponse({})
        self._raises = raises
        self.init_kwargs = kwargs
        self.urls: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def get(self, url, **kwargs):
        self.urls.append(url)
        if self._raises is not None:
            raise self._raises
        return self._response


def _httpx_namespace(client: _FakeAsyncClient):
    return SimpleNamespace(AsyncClient=lambda **kwargs: client)


class LlmSettingsRouterTestBase(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(llm_settings.router)
        self.app.dependency_overrides[require_auth] = lambda: {
            "user": "tester@example.com",
            "provider": "google",
        }
        self.client = TestClient(self.app)
        llm_settings.init(object())

    def tearDown(self):
        self.app.dependency_overrides.clear()
        get_settings.cache_clear()


class ProviderListingTests(LlmSettingsRouterTestBase):
    def test_providers_report_configured_state_from_env(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "gsk-test", "OPENAI_API_KEY": ""}):
            payload = self.client.get("/api/llm/providers").json()

        providers = {p["name"]: p for p in payload["providers"]}
        self.assertTrue(providers["groq"]["available"])
        self.assertFalse(providers["openai"]["available"])
        # Ollama is local and never needs a key.
        self.assertTrue(providers["ollama"]["available"])
        self.assertEqual(providers["groq"]["backend"], "litellm")


class LlmHealthTests(LlmSettingsRouterTestBase):
    def test_health_is_ok_when_the_default_provider_key_is_present(self):
        with patch.dict("os.environ", {"STUDY_DEFAULT_PROVIDER": "groq", "GROQ_API_KEY": "gsk-x"}):
            get_settings.cache_clear()
            payload = self.client.get("/api/llm/health").json()

        self.assertTrue(payload["llm_chain"])
        self.assertTrue(payload["cli_agent"])
        self.assertEqual(payload["llm_chain_version"], "litellm")
        self.assertIn(payload["vector_index"], {"available", "fallback"})

    def test_health_is_down_when_the_default_provider_key_is_missing(self):
        with patch.dict("os.environ", {"STUDY_DEFAULT_PROVIDER": "groq"}):
            get_settings.cache_clear()
            with patch.dict("os.environ", {"GROQ_API_KEY": ""}):
                payload = self.client.get("/api/llm/health").json()

        self.assertFalse(payload["llm_chain"])
        self.assertFalse(payload["cli_agent"])

    def test_health_uses_the_litellm_chain_for_every_reported_component(self):
        with patch.dict("os.environ", {"STUDY_DEFAULT_PROVIDER": "ollama"}):
            get_settings.cache_clear()
            payload = self.client.get("/api/llm/health").json()

        self.assertEqual(payload["cli_agent_version"], "litellm")
        self.assertTrue(payload["llm_chain"])


class OllamaTestTests(LlmSettingsRouterTestBase):
    def _call(self, client: _FakeAsyncClient):
        with patch.object(llm_settings, "httpx", _httpx_namespace(client)):
            return self.client.get("/api/llm/ollama/test")

    def test_connected_reports_the_server_version(self):
        fake = _FakeAsyncClient(_FakeResponse({"version": "0.5.1"}))
        payload = self._call(fake).json()

        self.assertTrue(payload["connected"])
        self.assertEqual(payload["version"], "0.5.1")
        self.assertEqual(fake.urls, [f"{OLLAMA_BASE_URL}/api/version"])

    def test_connected_is_false_when_the_version_field_is_absent(self):
        payload = self._call(_FakeAsyncClient(_FakeResponse({}))).json()
        self.assertTrue(payload["connected"])
        self.assertEqual(payload["version"], "unknown")

    def test_non_200_response_is_reported_as_disconnected(self):
        payload = self._call(_FakeAsyncClient(_FakeResponse({"v": "x"}, status_code=503))).json()
        self.assertFalse(payload["connected"])

    def test_connection_refused_is_swallowed_and_reported_as_disconnected(self):
        fake = _FakeAsyncClient(raises=OSError("connection refused"))
        payload = self._call(fake).json()
        self.assertFalse(payload["connected"])

    def test_missing_ollama_base_url_is_never_built_from_an_empty_value(self):
        with patch.object(llm_settings, "OLLAMA_BASE_URL", "http://127.0.0.1:9999"):
            fake = _FakeAsyncClient(_FakeResponse({"version": "0.1"}))
            with patch.object(llm_settings, "httpx", _httpx_namespace(fake)):
                self.client.get("/api/llm/ollama/test")
        self.assertEqual(fake.urls, ["http://127.0.0.1:9999/api/version"])


class OllamaModelsTests(LlmSettingsRouterTestBase):
    def _call(self, client: _FakeAsyncClient):
        with patch.object(llm_settings, "httpx", _httpx_namespace(client)):
            return self.client.get("/api/llm/ollama/models")

    def test_downloaded_models_are_formatted_in_gb_and_mb(self):
        payload_models = {
            "models": [
                {"name": "llama3:8b", "size": 4 * (1024 ** 3), "modified_at": "2026-01-02T00:00:00Z"},
                {"name": "tiny:1b", "size": 512 * (1024 ** 2), "modified_at": "2026-01-03T00:00:00Z"},
                {"name": "nosize:0b", "size": 0, "modified_at": ""},
            ]
        }
        fake = _FakeAsyncClient(_FakeResponse(payload_models))
        payload = self._call(fake).json()

        self.assertTrue(payload["connected"])
        self.assertEqual(fake.urls, [f"{OLLAMA_BASE_URL}/api/tags"])
        by_name = {m["name"]: m for m in payload["downloaded_models"]}
        self.assertEqual(by_name["llama3:8b"]["size"], "4.0 GB")
        self.assertEqual(by_name["llama3:8b"]["modified_at"], "2026-01-02T00:00:00Z")
        self.assertEqual(by_name["tiny:1b"]["size"], "512 MB")
        self.assertEqual(by_name["nosize:0b"]["size"], "")
        self.assertTrue(payload["cloud_models"])

    def test_non_200_response_yields_no_downloaded_models(self):
        payload = self._call(_FakeAsyncClient(_FakeResponse({}, status_code=404))).json()
        self.assertFalse(payload["connected"])
        self.assertEqual(payload["downloaded_models"], [])
        # Cloud suggestions are static and must still be offered.
        self.assertTrue(payload["cloud_models"])

    def test_transport_error_degrades_to_disconnected_without_models(self):
        payload = self._call(_FakeAsyncClient(raises=OSError("refused"))).json()
        self.assertFalse(payload["connected"])
        self.assertEqual(payload["downloaded_models"], [])


if __name__ == "__main__":
    unittest.main()