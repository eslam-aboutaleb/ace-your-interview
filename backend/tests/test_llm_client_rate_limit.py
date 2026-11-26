"""Tests for provider rate-limit handling in LLMClient (eval-live fix).

A 429 is transient: the client honours the provider's retry-after hint and
retries inside the call instead of letting every flow's retry loop storm
the provider. gpt-oss models get reasoning_effort="low" so their hidden
reasoning does not consume the per-flow output caps.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.services.llm_client import (
    RATE_LIMIT_MAX_ATTEMPTS,
    LLMClient,
    _is_rate_limit_error,
    _rate_limit_wait_seconds,
)


def _settings(**overrides):
    base = dict(
        default_temperature=0.7,
        default_provider="groq",
        default_model="openai/gpt-oss-120b",
        llm_task_routing_enabled=False,
        llm_task_route_map_json="",
        llm_user_call_budget=0,
        llm_user_call_budget_window_seconds=60,
        llm_user_max_concurrency=4,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _response(content="[]"):
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


class _ScriptedTransport:
    """Raise scripted exceptions, then answer."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.kwargs: dict = {}

    async def __call__(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _ClientTest(unittest.TestCase):
    def setUp(self):
        LLMClient._budget.reset()
        LLMClient._inflight.clear()
        self.addCleanup(lambda: (LLMClient._budget.reset(), LLMClient._inflight.clear()))


class RateLimitHelpersTests(unittest.TestCase):
    def test_parses_provider_retry_hint(self):
        exc = Exception("Rate limit reached ... Please try again in 3.637s.")
        self.assertAlmostEqual(_rate_limit_wait_seconds(exc), 3.637 + 0.5, places=3)

    def test_caps_the_wait(self):
        exc = Exception("Please try again in 600s.")
        self.assertEqual(_rate_limit_wait_seconds(exc), 30.0)

    def test_default_wait_without_hint(self):
        self.assertEqual(_rate_limit_wait_seconds(Exception("rate_limit_exceeded")), 5.0)

    def test_detection_by_message(self):
        self.assertTrue(_is_rate_limit_error(Exception("Rate limit reached for model")))
        self.assertTrue(_is_rate_limit_error(Exception("code: rate_limit_exceeded")))
        self.assertFalse(_is_rate_limit_error(Exception("boom")))


class RateLimitRetryTests(_ClientTest):
    def _run(self, transport, sleeps):
        async def fake_sleep(seconds):
            sleeps.append(seconds)

        client = LLMClient()
        with (
            patch("app.services.llm_client.get_settings", return_value=_settings()),
            patch("app.services.llm_client.litellm.acompletion", new=transport),
            patch("app.services.llm_client.asyncio.sleep", new=fake_sleep),
        ):
            return asyncio.run(client.completion("give me json"))

    def test_rate_limit_is_retried_after_the_provider_hint(self):
        transport = _ScriptedTransport(
            [Exception("Rate limit reached. Please try again in 0.01s."), _response()]
        )
        sleeps: list[float] = []
        result = self._run(transport, sleeps)
        self.assertEqual(transport.calls, 2)
        self.assertEqual(len(sleeps), 1)
        self.assertGreater(sleeps[0], 0.01)
        self.assertTrue(result["success"])

    def test_non_rate_limit_error_is_not_retried(self):
        transport = _ScriptedTransport([ValueError("some other failure")])
        result = self._run(transport, [])
        self.assertEqual(transport.calls, 1)
        self.assertFalse(result["success"])

    def test_persistent_rate_limit_gives_up_after_cap(self):
        transport = _ScriptedTransport(
            [Exception("Rate limit reached. Please try again in 0.001s.")]
            * RATE_LIMIT_MAX_ATTEMPTS
        )
        result = self._run(transport, [])
        self.assertEqual(transport.calls, RATE_LIMIT_MAX_ATTEMPTS)
        self.assertFalse(result["success"])


class GptOssKwargsTests(_ClientTest):
    def test_gpt_oss_gets_low_reasoning_effort(self):
        transport = _ScriptedTransport([_response()])
        client = LLMClient()
        with (
            patch("app.services.llm_client.get_settings", return_value=_settings()),
            patch("app.services.llm_client.litellm.acompletion", new=transport),
        ):
            asyncio.run(client.completion("hi"))
        self.assertEqual(transport.kwargs.get("reasoning_effort"), "low")

    def test_non_gpt_oss_model_gets_no_reasoning_effort(self):
        transport = _ScriptedTransport([_response()])
        client = LLMClient()
        with (
            patch(
                "app.services.llm_client.get_settings",
                return_value=_settings(default_provider="openai", default_model="gpt-4o-mini"),
            ),
            patch("app.services.llm_client.litellm.acompletion", new=transport),
        ):
            asyncio.run(client.completion("hi"))
        self.assertNotIn("reasoning_effort", transport.kwargs)


class JsonValidateFailedTests(_ClientTest):
    """Groq rejects an in-JSON-mode truncated generation with a 400
    ``json_validate_failed`` instead of finish_reason="length". That must
    surface as the terminal truncation code, not a retryable transport
    failure, or every caller's retry loop storms the provider (one eval
    case burned 57 calls before this mapping)."""

    def test_json_validate_failed_maps_to_terminal_truncation(self):
        transport = _ScriptedTransport(
            [
                Exception(
                    'litellm.BadRequestError: GroqException - {"error":'
                    '{"message":"Failed to generate JSON. Please adjust your'
                    ' prompt.","code":"json_validate_failed"}}'
                )
            ]
        )
        client = LLMClient()
        with (
            patch("app.services.llm_client.get_settings", return_value=_settings()),
            patch("app.services.llm_client.litellm.acompletion", new=transport),
        ):
            result = asyncio.run(client.completion("hi"))
        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], "llm_truncated")
        self.assertEqual(result["finish_reason"], "length")
        self.assertEqual(transport.calls, 1)

    def test_other_provider_errors_stay_retryable_call_failed(self):
        transport = _ScriptedTransport([Exception("500 internal server error")])
        client = LLMClient()
        with (
            patch("app.services.llm_client.get_settings", return_value=_settings()),
            patch("app.services.llm_client.litellm.acompletion", new=transport),
        ):
            result = asyncio.run(client.completion("hi"))
        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], "llm_call_failed")


if __name__ == "__main__":
    unittest.main()


class ConfigurableMaxWaitTests(_ClientTest):
    def test_settings_raise_the_wait_cap(self):
        # A 55s provider hint exceeds the 30s default cap; the eval raises it
        # past Groq's 60s TPM window so the retry actually lands.
        transport = _ScriptedTransport(
            [Exception("Rate limit reached. Please try again in 55s."), _response()]
        )
        sleeps: list[float] = []

        async def fake_sleep(seconds):
            sleeps.append(seconds)

        client = LLMClient()
        with (
            patch(
                "app.services.llm_client.get_settings",
                return_value=_settings(llm_rate_limit_max_wait_seconds=65.0),
            ),
            patch("app.services.llm_client.litellm.acompletion", new=transport),
            patch("app.services.llm_client.asyncio.sleep", new=fake_sleep),
        ):
            result = asyncio.run(client.completion("give me json"))
        self.assertTrue(result["success"])
        self.assertEqual(len(sleeps), 1)
        self.assertAlmostEqual(sleeps[0], 55.5, places=3)

    def test_default_cap_still_applies(self):
        transport = _ScriptedTransport(
            [Exception("Rate limit reached. Please try again in 55s."), _response()]
        )
        sleeps: list[float] = []

        async def fake_sleep(seconds):
            sleeps.append(seconds)

        client = LLMClient()
        with (
            patch("app.services.llm_client.get_settings", return_value=_settings()),
            patch("app.services.llm_client.litellm.acompletion", new=transport),
            patch("app.services.llm_client.asyncio.sleep", new=fake_sleep),
        ):
            asyncio.run(client.completion("give me json"))
        self.assertAlmostEqual(sleeps[0], 30.0, places=3)
