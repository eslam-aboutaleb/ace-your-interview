"""Tests for the LiteLLM gateway provider (user steering, Nov 2026).

Provider "litellm" targets an OpenAI-compatible gateway (the owner's
llm-gateway, same convention as ragnor/polymarket): model aliases resolve
through the ``openai/`` path, and api_base/api_key come from the
STUDY_-prefixed settings with LITELLM_API_BASE / LITELLM_API_KEY as the
cross-repo fallback. Switching providers is configuration, not code.
"""

import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.services.llm_client import (
    LLMClient,
    _litellm_gateway_endpoint,
    _resolve_model,
)


def _settings(**overrides):
    base = dict(
        default_temperature=0.7,
        default_provider="litellm",
        default_model="fast-alias",
        llm_task_routing_enabled=False,
        llm_task_route_map_json="",
        llm_user_call_budget=0,
        llm_user_call_budget_window_seconds=60,
        llm_user_max_concurrency=4,
        litellm_api_base="",
        litellm_api_key="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _response(content="ok"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            )
        ],
        usage=None,
        model="mock-model",
    )


class _Capture:
    def __init__(self):
        self.kwargs: dict = {}

    async def __call__(self, **kwargs):
        self.kwargs = kwargs
        return _response()


class ResolveTests(unittest.TestCase):
    def test_alias_resolves_through_openai_path(self):
        with patch(
            "app.services.llm_client.get_settings", return_value=_settings()
        ):
            self.assertEqual(
                _resolve_model("litellm", "fast-alias"), "openai/fast-alias"
            )

    def test_blank_model_uses_configured_default(self):
        with patch(
            "app.services.llm_client.get_settings", return_value=_settings()
        ):
            self.assertEqual(_resolve_model("default", ""), "openai/fast-alias")


class EndpointTests(unittest.TestCase):
    def test_settings_win_over_env(self):
        env = {"LITELLM_API_BASE": "https://env.example/v1", "LITELLM_API_KEY": "env-key"}
        with (
            patch(
                "app.services.llm_client.get_settings",
                return_value=_settings(
                    litellm_api_base="https://settings.example/v1",
                    litellm_api_key="settings-key",
                ),
            ),
            patch.dict(os.environ, env, clear=False),
        ):
            base, key = _litellm_gateway_endpoint()
        self.assertEqual(base, "https://settings.example/v1")
        self.assertEqual(key, "settings-key")

    def test_cross_repo_env_names_are_the_fallback(self):
        env = {"LITELLM_API_BASE": "https://env.example/v1", "LITELLM_API_KEY": "env-key"}
        with (
            patch("app.services.llm_client.get_settings", return_value=_settings()),
            patch.dict(os.environ, env, clear=False),
        ):
            base, key = _litellm_gateway_endpoint()
        self.assertEqual(base, "https://env.example/v1")
        self.assertEqual(key, "env-key")


class CompletionInjectionTests(unittest.TestCase):
    def setUp(self):
        LLMClient._budget.reset()
        LLMClient._inflight.clear()
        self.addCleanup(lambda: (LLMClient._budget.reset(), LLMClient._inflight.clear()))

    def test_gateway_endpoint_reaches_the_provider_call(self):
        capture = _Capture()
        env = {"LITELLM_API_BASE": "https://gw.example/v1", "LITELLM_API_KEY": "gw-key"}
        client = LLMClient()
        with (
            patch("app.services.llm_client.get_settings", return_value=_settings()),
            patch("app.services.llm_client.litellm.acompletion", new=capture),
            patch.dict(os.environ, env, clear=False),
        ):
            result = asyncio.run(client.completion("hi"))
        self.assertTrue(result["success"])
        self.assertEqual(capture.kwargs["model"], "openai/fast-alias")
        self.assertEqual(capture.kwargs["api_base"], "https://gw.example/v1")
        self.assertEqual(capture.kwargs["api_key"], "gw-key")

    def test_other_providers_get_no_gateway_kwargs(self):
        capture = _Capture()
        env = {"LITELLM_API_BASE": "https://gw.example/v1", "LITELLM_API_KEY": "gw-key"}
        client = LLMClient()
        with (
            patch(
                "app.services.llm_client.get_settings",
                return_value=_settings(default_provider="openai", default_model="gpt-4o-mini"),
            ),
            patch("app.services.llm_client.litellm.acompletion", new=capture),
            patch.dict(os.environ, env, clear=False),
        ):
            asyncio.run(client.completion("hi"))
        self.assertNotIn("api_base", capture.kwargs)
        self.assertNotIn("api_key", capture.kwargs)


class WhitelistTests(unittest.TestCase):
    def test_litellm_is_a_selectable_provider(self):
        from app.schemas.models import (
            MODEL_SUGGESTIONS,
            USER_SETTINGS_PROVIDER_WHITELIST,
            LLMProviderEnum,
        )

        self.assertIn("litellm", USER_SETTINGS_PROVIDER_WHITELIST)
        self.assertIn("litellm", MODEL_SUGGESTIONS)
        self.assertEqual(LLMProviderEnum.LITELLM.value, "litellm")


if __name__ == "__main__":
    unittest.main()
