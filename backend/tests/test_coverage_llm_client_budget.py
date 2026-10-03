"""Coverage for the ``LLMClient`` per-user call budget and concurrency gate.

``_PerUserCallBudget`` and ``LLMClient._inflight`` are *process-level* state, so
every test here resets both in ``setUp``/``addCleanup``: without that, a test
that exhausts the budget would leak into any later test sharing the identity.
"""

import asyncio
import json
import time
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

from app.schemas.models import MODEL_SUGGESTIONS, LLMConfigRequest, LLMProviderEnum
from app.services.llm_client import (
    BUDGET_EXCEEDED_CODE,
    DEFAULT_MAX_TOKENS,
    TERMINAL_ERROR_CODES,
    LLMClient,
    _PerUserCallBudget,
)


def _settings(**overrides):
    base = dict(
        default_temperature=0.7,
        default_provider="openai",
        default_model="gpt-4o-mini",
        llm_task_routing_enabled=False,
        llm_task_route_map_json="",
        llm_user_call_budget=0,
        llm_user_call_budget_window_seconds=60,
        llm_user_max_concurrency=4,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class _Capture:
    def __init__(self, *, content="ok"):
        self.calls = 0
        self.kwargs: dict = {}
        self._content = content

    async def __call__(self, **kwargs):
        self.kwargs = kwargs
        self.calls += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content), finish_reason="stop")],
            usage=None,
            model="mock-model",
        )


class _ConcurrencyProbe:
    """Records how many transport calls are in flight at the same moment."""

    def __init__(self, delay=0.01):
        self.active = 0
        self.peak = 0
        self.calls = 0
        self._delay = delay

    async def __call__(self, **kwargs):
        self.calls += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self._delay)
        finally:
            self.active -= 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
            usage=None,
            model="mock-model",
        )


class _GlobalStateIsolation(unittest.TestCase):
    """Clears the shared budget/semaphore registries around every test."""

    def setUp(self):
        LLMClient._budget.reset()
        LLMClient._inflight.clear()
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        LLMClient._budget.reset()
        LLMClient._inflight.clear()


def _acquire(budget, key):
    async def _go():
        return await budget.acquire(key)

    return asyncio.run(_go())


class PerUserCallBudgetUnitTests(unittest.TestCase):
    def test_calls_below_the_limit_are_allowed(self):
        budget = _PerUserCallBudget(3, 60.0)
        for i in range(3):
            with self.subTest(call=i):
                self.assertEqual(_acquire(budget, "google:a@x.com"), (True, 0.0))

    def test_call_at_the_limit_is_denied_with_a_retry_after(self):
        budget = _PerUserCallBudget(2, 60.0)
        self.assertEqual(_acquire(budget, "google:a@x.com"), (True, 0.0))
        self.assertEqual(_acquire(budget, "google:a@x.com"), (True, 0.0))

        allowed, retry_after = _acquire(budget, "google:a@x.com")
        self.assertFalse(allowed)
        self.assertGreater(retry_after, 0.0)
        self.assertLessEqual(retry_after, 60.0)
        self.assertIn(BUDGET_EXCEEDED_CODE, TERMINAL_ERROR_CODES)

    def test_retry_after_shrinks_as_the_window_drains(self):
        budget = _PerUserCallBudget(1, 0.4)
        self.assertEqual(_acquire(budget, "k"), (True, 0.0))
        time.sleep(0.2)
        first = _acquire(budget, "k")[1]
        time.sleep(0.1)
        second = _acquire(budget, "k")[1]
        self.assertLess(second, first)

    def test_window_expiry_lets_calls_through_again(self):
        budget = _PerUserCallBudget(2, 0.05)
        self.assertEqual(_acquire(budget, "k"), (True, 0.0))
        self.assertEqual(_acquire(budget, "k"), (True, 0.0))
        self.assertFalse(_acquire(budget, "k")[0])

        time.sleep(0.1)
        self.assertEqual(_acquire(budget, "k"), (True, 0.0))
        self.assertEqual(_acquire(budget, "k"), (True, 0.0))
        self.assertFalse(_acquire(budget, "k")[0])

    def test_budget_is_scoped_per_identity(self):
        budget = _PerUserCallBudget(1, 60.0)
        self.assertEqual(_acquire(budget, "google:a@x.com"), (True, 0.0))
        self.assertFalse(_acquire(budget, "google:a@x.com")[0])
        self.assertEqual(_acquire(budget, "google:b@x.com"), (True, 0.0))

    def test_empty_identity_shares_the_anonymous_bucket(self):
        budget = _PerUserCallBudget(1, 60.0)
        self.assertEqual(_acquire(budget, ""), (True, 0.0))
        self.assertFalse(_acquire(budget, "")[0])
        self.assertIn("__anonymous__", budget._events)

    def test_max_calls_of_zero_disables_the_limit(self):
        budget = _PerUserCallBudget(0, 60.0)
        for _ in range(50):
            self.assertEqual(_acquire(budget, "k"), (True, 0.0))
        self.assertEqual(budget._events, {})

    def test_negative_max_calls_is_clamped_to_disabled(self):
        budget = _PerUserCallBudget(-5, 60.0)
        self.assertEqual(_acquire(budget, "k"), (True, 0.0))

    def test_zero_window_is_clamped_to_a_positive_floor(self):
        budget = _PerUserCallBudget(1, 0)
        self.assertEqual(_acquire(budget, "k"), (True, 0.0))
        self.assertGreater(budget._window, 0.0)

    def test_reset_clears_all_recorded_events(self):
        budget = _PerUserCallBudget(1, 60.0)
        _acquire(budget, "k")
        self.assertTrue(budget._events)
        budget.reset()
        self.assertEqual(budget._events, {})
        self.assertEqual(_acquire(budget, "k"), (True, 0.0))

    def test_prune_drops_expired_and_empty_buckets(self):
        budget = _PerUserCallBudget(5, 60.0)
        now = time.monotonic()
        live = f"live-{now}"
        budget._events[live] = deque([now])
        budget._events["expired"] = deque([now - 3600])
        budget._events["empty"] = deque()

        budget._prune_locked(now - 60.0)

        self.assertIn(live, budget._events)
        self.assertNotIn("expired", budget._events)
        self.assertNotIn("empty", budget._events)

    def test_acquire_bounds_memory_by_pruning_past_ten_thousand_identities(self):
        budget = _PerUserCallBudget(2, 600.0)
        now = time.monotonic()
        for i in range(10_000):
            budget._events[f"live:{i}"] = deque([now])
        budget._events["long-dead"] = deque([now - 3600])
        self.assertGreater(len(budget._events), 10_000)

        allowed, retry_after = _acquire(budget, "google:new@example.com")

        self.assertTrue(allowed)
        self.assertEqual(retry_after, 0.0)
        self.assertNotIn("long-dead", budget._events)
        self.assertIn("google:new@example.com", budget._events)
        self.assertEqual(len(budget._events), 10_001)


class BudgetForTests(_GlobalStateIsolation):
    def _budget_for(self, **overrides):
        async def _go():
            return await LLMClient._budget_for(_settings(**overrides))

        return asyncio.run(_go())

    def test_signature_change_rebuilds_the_budget(self):
        first = self._budget_for(llm_user_call_budget=2, llm_user_call_budget_window_seconds=30)
        self.assertIs(self._budget_for(llm_user_call_budget=2, llm_user_call_budget_window_seconds=30), first)

        second = self._budget_for(llm_user_call_budget=3, llm_user_call_budget_window_seconds=30)
        self.assertIsNot(second, first)

    def test_absent_settings_fall_back_to_the_documented_defaults(self):
        settings = SimpleNamespace()
        budget = asyncio.run(LLMClient._budget_for(settings))
        self.assertEqual(budget._max_calls, 0)
        self.assertEqual(budget._window, 60.0)

    def test_budget_state_does_not_survive_a_signature_change(self):
        small = self._budget_for(llm_user_call_budget=1, llm_user_call_budget_window_seconds=30)
        self.assertEqual(_acquire(small, "k"), (True, 0.0))
        self._budget_for(llm_user_call_budget=5, llm_user_call_budget_window_seconds=30)
        rebuilt = LLMClient._budget
        self.assertEqual(_acquire(rebuilt, "k"), (True, 0.0))


class BudgetEnforcementTests(_GlobalStateIsolation):
    def _completion(self, capture, **overrides):
        async def _go():
            return await LLMClient().completion("prompt")

        with patch("app.services.llm_client.get_settings", return_value=_settings(**overrides)):
            with patch("app.services.llm_client.litellm.acompletion", new=capture):
                return asyncio.run(_go())

    def test_calls_within_the_budget_reach_the_provider(self):
        capture = _Capture()
        for _ in range(3):
            result = self._completion(
                capture,
                llm_user_call_budget=3,
                llm_user_call_budget_window_seconds=60,
            )
            self.assertTrue(result["success"])
        self.assertEqual(capture.calls, 3)

    def test_call_beyond_the_budget_is_blocked_before_the_provider(self):
        capture = _Capture()
        settings = {"llm_user_call_budget": 2, "llm_user_call_budget_window_seconds": 60}
        self._completion(capture, **settings)
        self._completion(capture, **settings)

        with self.assertLogs("app.services.llm_client", level="WARNING") as logs:
            result = self._completion(capture, **settings)

        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], BUDGET_EXCEEDED_CODE)
        self.assertTrue(result["metadata"]["budget_limited"])
        self.assertEqual(result["analysis"], "")
        self.assertEqual(result["usage"], {})
        self.assertEqual(result["finish_reason"], "")
        self.assertIn("Retry in", result["error"])
        self.assertTrue(any("budget_exhausted" in line for line in logs.output))
        # The provider was never called for the denied call.
        self.assertEqual(capture.calls, 2)

    def test_zero_budget_disables_the_limit(self):
        capture = _Capture()
        for _ in range(10):
            result = self._completion(capture, llm_user_call_budget=0, llm_user_call_budget_window_seconds=60)
            self.assertTrue(result["success"])
        self.assertEqual(capture.calls, 10)

    def test_budget_is_scoped_per_identity(self):
        capture = _Capture()
        client = LLMClient()

        async def _go(identity):
            return await client.completion("prompt", user_identity=identity)

        settings = _settings(llm_user_call_budget=1, llm_user_call_budget_window_seconds=60)
        with patch("app.services.llm_client.get_settings", return_value=settings):
            with patch("app.services.llm_client.litellm.acompletion", new=capture):
                self.assertTrue(asyncio.run(_go({"provider": "google", "user": "a@x.com"}))["success"])
                self.assertFalse(asyncio.run(_go({"provider": "google", "user": "a@x.com"}))["success"])
                self.assertTrue(asyncio.run(_go({"provider": "google", "user": "b@x.com"}))["success"])


class SemaphoreTests(_GlobalStateIsolation):
    def test_configured_limit_creates_a_semaphore_of_that_size(self):
        async def _go():
            return await LLMClient._semaphore_for("google:a@x.com")

        with patch("app.services.llm_client.get_settings", return_value=_settings(llm_user_max_concurrency=3)):
            sem = asyncio.run(_go())
        self.assertIsInstance(sem, asyncio.Semaphore)
        self.assertEqual(sem._value, 3)

    def test_same_identity_reuses_the_same_semaphore(self):
        async def _go():
            first = await LLMClient._semaphore_for("google:a@x.com")
            second = await LLMClient._semaphore_for("google:a@x.com")
            other = await LLMClient._semaphore_for("google:b@x.com")
            return first, second, other

        with patch("app.services.llm_client.get_settings", return_value=_settings(llm_user_max_concurrency=2)):
            first, second, other = asyncio.run(_go())
        self.assertIs(first, second)
        self.assertIsNot(first, other)

    def test_empty_identity_uses_the_anonymous_key(self):
        async def _go():
            return await LLMClient._semaphore_for("")

        with patch("app.services.llm_client.get_settings", return_value=_settings(llm_user_max_concurrency=2)):
            asyncio.run(_go())
        self.assertIn("__anonymous__", LLMClient._inflight)

    def test_non_positive_config_falls_back_to_the_default_of_four(self):
        async def _go():
            return await LLMClient._semaphore_for("google:a@x.com")

        for configured in (0, -1):
            with self.subTest(configured=configured):
                LLMClient._inflight.clear()
                with patch(
                    "app.services.llm_client.get_settings",
                    return_value=_settings(llm_user_max_concurrency=configured),
                ):
                    sem = asyncio.run(_go())
                self.assertEqual(sem._value, 4)

    def _peak_concurrency(self, *, concurrency, calls, identity=None):
        probe = _ConcurrencyProbe()
        client = LLMClient()

        async def _go():
            user = {"provider": "google", "user": identity} if identity else None
            return await asyncio.gather(
                *(client.completion("prompt", user_identity=user) for _ in range(calls)),
            )

        settings = _settings(llm_user_max_concurrency=concurrency)
        with patch("app.services.llm_client.get_settings", return_value=settings):
            with patch("app.services.llm_client.litellm.acompletion", new=probe):
                results = asyncio.run(_go())

        self.assertTrue(all(r["success"] for r in results))
        self.assertEqual(probe.calls, calls)
        return probe.peak

    def test_concurrent_calls_for_one_identity_are_bounded(self):
        self.assertEqual(self._peak_concurrency(concurrency=2, calls=8, identity="a@x.com"), 2)

    def test_default_fallback_bound_is_four(self):
        LLMClient._inflight.clear()
        self.assertEqual(self._peak_concurrency(concurrency=0, calls=8, identity="a@x.com"), 4)

    def test_a_single_permitted_call_is_not_throttled(self):
        self.assertEqual(self._peak_concurrency(concurrency=2, calls=2, identity="a@x.com"), 2)

    def test_distinct_identities_do_not_share_a_gate(self):
        probe = _ConcurrencyProbe(delay=0.02)
        client = LLMClient()

        async def _go():
            return await asyncio.gather(
                client.completion("p", user_identity={"provider": "google", "user": "a@x.com"}),
                client.completion("p", user_identity={"provider": "google", "user": "b@x.com"}),
            )

        settings = _settings(llm_user_max_concurrency=1)
        with patch("app.services.llm_client.get_settings", return_value=settings):
            with patch("app.services.llm_client.litellm.acompletion", new=probe):
                asyncio.run(_go())
        self.assertEqual(probe.peak, 2)


class GoogleOAuthCredentialTests(_GlobalStateIsolation):
    """The dict-credential branch of ``completion`` (Google account mode)."""

    def _completion(self, capture, *, model, resolve_result):
        store = _StubStore(
            {"llm_source": "personal", "provider": "google", "auth_mode": "account"},
            credential=resolve_result,
        )

        async def _go():
            return await LLMClient(user_settings_store=store).completion(
                "prompt",
                LLMConfigRequest(provider=LLMProviderEnum.GOOGLE, model=model),
                user_identity={"provider": "google", "user": "a@x.com"},
            )

        with patch("app.services.llm_client.get_settings", return_value=_settings()):
            with patch("app.services.llm_client.litellm.acompletion", new=capture):
                return asyncio.run(_go())

    def test_gemini_model_is_rewritten_for_the_openai_compatible_endpoint(self):
        capture = _Capture()
        result = self._completion(
            capture,
            model="gemini-2.5-pro",
            resolve_result=({"Authorization": "Bearer tok"}, "user_account", None),
        )
        self.assertTrue(result["success"])
        self.assertEqual(capture.kwargs["model"], "openai/gemini-2.5-pro")
        self.assertEqual(capture.kwargs["api_key"], "tok")
        self.assertEqual(
            capture.kwargs["api_base"],
            "https://generativelanguage.googleapis.com/v1beta/openai",
        )
        self.assertEqual(result["metadata"]["credential_source"], "user_account")

    def test_bearer_prefix_is_stripped_and_whitespace_trimmed(self):
        capture = _Capture()
        self._completion(
            capture,
            model="gemini-2.5-pro",
            resolve_result=({"Authorization": "Bearer  spaced-token  "}, "user_account", None),
        )
        self.assertEqual(capture.kwargs["api_key"], "spaced-token")

    def test_credential_without_a_bearer_prefix_is_used_verbatim(self):
        capture = _Capture()
        self._completion(
            capture,
            model="gemini-2.5-pro",
            resolve_result=({"Authorization": "raw-token"}, "user_account", None),
        )
        self.assertEqual(capture.kwargs["api_key"], "raw-token")

    def test_credential_with_no_authorization_header_yields_an_empty_key(self):
        capture = _Capture()
        self._completion(
            capture,
            model="gemini-2.5-pro",
            resolve_result=({"other": "x"}, "user_account", None),
        )
        self.assertEqual(capture.kwargs["api_key"], "")

    def test_model_outside_the_gemini_namespace_is_left_untouched(self):
        # The gemini/ -> openai/ rewrite only applies to the gemini namespace.
        capture = _Capture()
        with patch(
            "app.services.llm_client._resolve_model",
            return_value="openai/custom-deployment",
        ):
            self._completion(
                capture,
                model="gemini-2.5-pro",
                resolve_result=({"Authorization": "Bearer tok"}, "user_account", None),
            )
        self.assertEqual(capture.kwargs["model"], "openai/custom-deployment")
        self.assertEqual(capture.kwargs["api_key"], "tok")
        self.assertIn("generativelanguage.googleapis.com", capture.kwargs["api_base"])

    def test_string_credential_is_passed_straight_through(self):
        capture = _Capture()
        self._completion(
            capture,
            model="gemini-2.5-pro",
            resolve_result=("sk-google", "user_api_key", None),
        )
        self.assertEqual(capture.kwargs["api_key"], "sk-google")
        self.assertNotIn("api_base", capture.kwargs)
        # A plain key keeps the native gemini/ provider prefix.
        self.assertEqual(capture.kwargs["model"], "gemini/gemini-2.5-pro")

    def test_no_credential_means_no_api_key_is_sent(self):
        capture = _Capture()
        result = self._completion(capture, model="gemini-2.5-pro", resolve_result=(None, "backend", None))
        self.assertTrue(result["success"])
        self.assertNotIn("api_key", capture.kwargs)



class _StubStore:
    """Minimal ``UserSettingsStore`` stand-in: only the two calls the client makes."""

    def __init__(self, prefs, credential=("sk-stub", "user_api_key", None)):
        self.prefs = prefs
        self.credential = credential
        self.resolve_calls: list[dict] = []

    def get_user_state(self, identity_key):
        self.identity_key = identity_key
        return dict(self.prefs), True

    async def resolve_personal_runtime_credential(self, *, identity_key, provider, auth_mode):
        self.resolve_calls.append(
            {"identity_key": identity_key, "provider": provider, "auth_mode": auth_mode}
        )
        return self.credential


class PreferenceResolutionTests(_GlobalStateIsolation):
    """Saved preferences layer under the per-call config, and never raise it."""

    def _completion(self, prefs, *, config=None, settings=None, capture=None, credential=None):
        capture = capture or _Capture()
        store = _StubStore(prefs) if credential is None else _StubStore(prefs, credential)

        async def _go():
            return await LLMClient(user_settings_store=store).completion(
                "prompt",
                config,
                user_identity={"provider": "google", "user": "a@x.com"},
            )

        with patch("app.services.llm_client.get_settings", return_value=settings or _settings()):
            with patch("app.services.llm_client.litellm.acompletion", new=capture):
                result = asyncio.run(_go())
        return result, capture, store

    def test_saved_temperature_overrides_the_flow_default(self):
        prefs = {"llm_source": "personal", "temperature": 0.33, "auth_mode": "api_key"}
        _, capture, _ = self._completion(prefs)
        self.assertEqual(capture.kwargs["temperature"], 0.33)

    def test_explicit_config_temperature_wins_over_the_saved_preference(self):
        prefs = {"llm_source": "personal", "temperature": 0.33, "auth_mode": "api_key"}
        _, capture, _ = self._completion(
            prefs,
            config=LLMConfigRequest(provider=LLMProviderEnum.OPENAI, temperature=0.9),
        )
        self.assertEqual(capture.kwargs["temperature"], 0.9)

    def test_non_positive_saved_temperature_is_ignored(self):
        for value in (0, 0.0, -1):
            with self.subTest(value=value):
                prefs = {"llm_source": "personal", "temperature": value, "auth_mode": "api_key"}
                _, capture, _ = self._completion(prefs)
                self.assertEqual(capture.kwargs["temperature"], 0.7)

    def test_non_numeric_saved_temperature_is_ignored(self):
        for value in ("hot", None, [1]):
            with self.subTest(value=value):
                prefs = {"llm_source": "personal", "temperature": value, "auth_mode": "api_key"}
                _, capture, _ = self._completion(prefs)
                self.assertEqual(capture.kwargs["temperature"], 0.7)

    def test_saved_max_tokens_overrides_the_default(self):
        prefs = {"llm_source": "personal", "max_tokens": 1234, "auth_mode": "api_key"}
        _, capture, _ = self._completion(prefs)
        self.assertEqual(capture.kwargs["max_tokens"], 1234)

    def test_explicit_config_max_tokens_wins_over_the_saved_preference(self):
        prefs = {"llm_source": "personal", "max_tokens": 1234, "auth_mode": "api_key"}
        _, capture, _ = self._completion(
            prefs,
            config=LLMConfigRequest(provider=LLMProviderEnum.OPENAI, max_tokens=2000),
        )
        self.assertEqual(capture.kwargs["max_tokens"], 2000)

    def test_non_positive_saved_max_tokens_is_ignored(self):
        prefs = {"llm_source": "personal", "max_tokens": 0, "auth_mode": "api_key"}
        _, capture, _ = self._completion(prefs)
        self.assertEqual(capture.kwargs["max_tokens"], DEFAULT_MAX_TOKENS)

    def test_non_numeric_saved_max_tokens_is_ignored(self):
        for value in ("lots", None, [1]):
            with self.subTest(value=value):
                prefs = {"llm_source": "personal", "max_tokens": value, "auth_mode": "api_key"}
                _, capture, _ = self._completion(prefs)
                self.assertEqual(capture.kwargs["max_tokens"], DEFAULT_MAX_TOKENS)

    def test_saved_provider_and_model_are_adopted_when_the_config_omits_them(self):
        prefs = {
            "llm_source": "personal",
            "provider": "anthropic",
            "model": "claude-3-5-sonnet-20241022",
            "auth_mode": "api_key",
        }
        _, capture, store = self._completion(prefs, config=LLMConfigRequest())
        self.assertEqual(capture.kwargs["model"], "anthropic/claude-3-5-sonnet-20241022")
        self.assertEqual(store.resolve_calls[-1]["provider"], "anthropic")
        self.assertEqual(store.resolve_calls[-1]["identity_key"], "google:a@x.com")

    def test_config_provider_is_not_overridden_by_the_saved_preference(self):
        prefs = {
            "llm_source": "personal",
            "provider": "anthropic",
            "model": "claude-3-5-sonnet-20241022",
            "auth_mode": "api_key",
        }
        _, capture, store = self._completion(
            prefs,
            config=LLMConfigRequest(provider=LLMProviderEnum.OPENAI),
        )
        self.assertEqual(store.resolve_calls[-1]["provider"], "openai")
        # The credential is resolved for the *config* provider, and the saved
        # Anthropic model is deliberately NOT adopted: sending
        # `claude-3-5-sonnet-20241022` to provider `openai` is a guaranteed
        # provider-side error. The saved model is scoped to its own provider, so
        # this falls back to the default model for the requested provider.
        self.assertNotIn("claude", capture.kwargs["model"])
        self.assertEqual(capture.kwargs["model"], MODEL_SUGGESTIONS["openai"][0])

    def test_saved_model_is_adopted_when_the_provider_matches(self):
        prefs = {
            "llm_source": "personal",
            "provider": "openai",
            "model": "gpt-4o",
            "auth_mode": "api_key",
        }
        _, capture, _ = self._completion(prefs, config=LLMConfigRequest())
        self.assertEqual(capture.kwargs["model"], "gpt-4o")

    def test_an_adopted_saved_model_locks_out_task_routing(self):
        prefs = {
            "llm_source": "personal",
            "provider": "openai",
            "model": "gpt-4o",
            "auth_mode": "api_key",
        }
        _, capture, _ = self._completion(
            prefs,
            config=LLMConfigRequest(),
            settings=_settings(
                llm_task_routing_enabled=True,
                llm_task_route_map_json=json.dumps({"final": "gpt-3.5-turbo"}),
            ),
        )
        self.assertEqual(capture.kwargs["model"], "gpt-4o")

    def test_saved_auth_mode_is_forwarded_to_credential_resolution(self):
        prefs = {"llm_source": "personal", "provider": "google", "auth_mode": "api_key"}
        self._completion(prefs, config=LLMConfigRequest())
        self.assertEqual(self._completion(prefs, config=LLMConfigRequest())[2].resolve_calls[-1]["auth_mode"], "api_key")

    def test_credential_source_is_reported_from_the_resolver(self):
        prefs = {"llm_source": "personal", "provider": "openai", "auth_mode": "api_key"}
        result, _, _ = self._completion(prefs, config=LLMConfigRequest())
        self.assertEqual(result["metadata"]["credential_source"], "user_api_key")

    def test_policy_error_from_the_resolver_blocks_the_call(self):
        prefs = {"llm_source": "personal", "provider": "openai", "auth_mode": "api_key"}
        capture = _Capture()
        result, _, _ = self._completion(
            prefs,
            config=LLMConfigRequest(),
            capture=capture,
            credential=(None, "missing_personal", "personal_credential_required"),
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], "personal_credential_required")
        self.assertEqual(capture.calls, 0)


if __name__ == "__main__":
    unittest.main()
