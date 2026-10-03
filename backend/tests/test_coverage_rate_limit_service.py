"""Coverage for ``app.services.rate_limit`` — limiter edges and scope mapping.

Redis is never contacted: a fake ``redis`` module is injected into
``sys.modules`` when the real constructor path is exercised.
"""

import os
import sys
import types
import unittest
from unittest.mock import patch

from fastapi import Request

from app.config import Settings, get_settings
from app.services.rate_limit import (
    InMemoryRateLimiter,
    RedisBackend,
    _classify_topic_detail_scope,
    classify_rate_limit_scope,
    create_rate_limiter,
    request_ip,
)


def _build_request(path: str, headers: dict, client_host: str) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": (client_host, 54321),
        "server": ("testserver", 80),
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, receive)


class _FakeRedisClient:
    def __init__(self):
        self.sets: dict[str, dict[str, float]] = {}
        self.pings = 0

    @classmethod
    def from_url(cls, url, decode_responses=False):
        instance = cls()
        instance.url = url
        instance.decode_responses = decode_responses
        return instance

    def ping(self):
        self.pings += 1
        return True

    def zremrangebyscore(self, key, low, high):
        return 0

    def zcard(self, key):
        return len(self.sets.get(key, {}))

    def zrange(self, key, start, stop, withscores=False):
        return []

    def zadd(self, key, mapping):
        self.sets.setdefault(key, {}).update(mapping)
        return len(mapping)

    def expire(self, key, seconds):
        return True


class LimiterShortCircuitTests(unittest.TestCase):
    def test_in_memory_limiter_allows_when_the_limit_is_zero(self):
        limiter = InMemoryRateLimiter()
        for _ in range(5):
            decision = limiter.check(scope="llm", key="k", limit=0, window_seconds=60)
            self.assertTrue(decision.allowed)
            self.assertEqual(decision.retry_after_seconds, 0)

    def test_in_memory_limiter_allows_when_the_window_is_zero(self):
        decision = InMemoryRateLimiter().check(
            scope="llm", key="k", limit=5, window_seconds=0
        )
        self.assertTrue(decision.allowed)

    def test_redis_limiter_allows_when_the_limit_is_zero(self):
        backend = RedisBackend(client=_FakeRedisClient())
        decision = backend.check(scope="llm", key="k", limit=0, window_seconds=60)
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.retry_after_seconds, 0)

    def test_redis_limiter_allows_when_the_window_is_zero(self):
        backend = RedisBackend(client=_FakeRedisClient())
        self.assertTrue(
            backend.check(scope="llm", key="k", limit=5, window_seconds=0).allowed
        )


class RedisConstructionTests(unittest.TestCase):
    def setUp(self):
        self.prev_redis = sys.modules.get("redis")

    def tearDown(self):
        if self.prev_redis is None:
            sys.modules.pop("redis", None)
        else:
            sys.modules["redis"] = self.prev_redis

    def test_missing_redis_package_is_reported_with_install_instructions(self):
        # ``None`` in sys.modules makes ``import redis`` raise ImportError.
        sys.modules["redis"] = None
        with self.assertRaises(RuntimeError) as ctx:
            RedisBackend(redis_url="redis://localhost:6379/0")

        message = str(ctx.exception)
        self.assertIn("requires the 'redis' package", message)
        self.assertIn("uv add redis", message)

    def test_redis_url_is_resolved_from_the_injected_module(self):
        sys.modules["redis"] = types.SimpleNamespace(Redis=_FakeRedisClient)
        backend = RedisBackend(redis_url="redis://cache.internal:6379/2")
        backend.ping()

        self.assertEqual(backend._client.url, "redis://cache.internal:6379/2")
        # decode_responses keeps the backend's zrange parsing str-based.
        self.assertTrue(backend._client.decode_responses)
        self.assertEqual(backend._client.pings, 1)

    def test_url_falls_back_to_the_redis_url_environment_variable(self):
        sys.modules["redis"] = types.SimpleNamespace(Redis=_FakeRedisClient)
        with patch.dict("os.environ", {"REDIS_URL": "redis://from-env:6379/0"}):
            backend = RedisBackend()
        self.assertEqual(backend._client.url, "redis://from-env:6379/0")

    def test_study_redis_url_env_var_is_also_accepted(self):
        sys.modules["redis"] = types.SimpleNamespace(Redis=_FakeRedisClient)
        settings = Settings(
            rate_limit_backend="redis", redis_url="redis://from-settings:6379/0"
        )
        backend = create_rate_limiter(settings)
        self.assertIsInstance(backend, RedisBackend)
        self.assertEqual(backend._client.url, "redis://from-settings:6379/0")

    def test_ping_propagates_an_unreachable_server(self):
        class Unreachable:
            def ping(self):
                raise ConnectionError("redis is down")

        backend = RedisBackend(client=Unreachable())
        with self.assertRaises(ConnectionError):
            backend.ping()


class TrustedProxyForwardedForTests(unittest.TestCase):
    def setUp(self):
        self.prev_env = os.environ.get("STUDY_TRUSTED_PROXY_ENABLED")
        os.environ["STUDY_TRUSTED_PROXY_ENABLED"] = "true"
        get_settings.cache_clear()

    def tearDown(self):
        if self.prev_env is None:
            os.environ.pop("STUDY_TRUSTED_PROXY_ENABLED", None)
        else:
            os.environ["STUDY_TRUSTED_PROXY_ENABLED"] = self.prev_env
        get_settings.cache_clear()

    def test_blank_forwarded_for_falls_through_to_x_real_ip(self):
        request = _build_request(
            "/api/llm/providers",
            {"x-forwarded-for": ",  ", "x-real-ip": "203.0.113.9"},
            "10.0.0.1",
        )
        self.assertEqual(request_ip(request), "203.0.113.9")

    def test_blank_forwarded_for_without_x_real_ip_falls_back_to_the_peer(self):
        request = _build_request(
            "/api/llm/providers", {"x-forwarded-for": ""}, "10.0.0.1"
        )
        self.assertEqual(request_ip(request), "10.0.0.1")

    def test_missing_peer_address_reports_unknown(self):
        scope = {
            "type": "http",
            "method": "GET",
            "scheme": "http",
            "path": "/api/llm/providers",
            "raw_path": b"/api/llm/providers",
            "query_string": b"",
            "root_path": "",
            "headers": [],
            "client": None,
            "server": ("testserver", 80),
        }

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        self.assertEqual(request_ip(Request(scope, receive)), "unknown")


class ScopeClassificationTests(unittest.TestCase):
    def test_llm_backed_generation_routes_are_classified(self):
        self.assertEqual(
            classify_rate_limit_scope("/api/questions/generate", "POST"), "llm"
        )
        self.assertEqual(
            classify_rate_limit_scope("/api/chat/follow-up", "POST"), "llm"
        )
        self.assertEqual(
            classify_rate_limit_scope("/api/interview-sessions", "POST"), "llm"
        )
        self.assertEqual(
            classify_rate_limit_scope(
                "/api/interview-sessions/abc/answer/stream", "POST"
            ),
            "llm",
        )

    def test_llm_scopes_are_method_sensitive(self):
        self.assertIsNone(classify_rate_limit_scope("/api/questions/generate", "GET"))
        self.assertIsNone(classify_rate_limit_scope("/api/chat/follow-up", "GET"))
        self.assertIsNone(classify_rate_limit_scope("/api/interview-sessions", "GET"))

    def test_paths_are_normalised_before_classification(self):
        self.assertEqual(
            classify_rate_limit_scope("  /API/CHAT/FOLLOW-UP  ", " post "), "llm"
        )

    def test_oauth_scope_covers_both_providers_and_the_connect_routes(self):
        for path in (
            "/api/auth/github",
            "/api/auth/github/callback",
            "/api/auth/google",
            "/api/auth/google/callback",
            "/api/user-settings/google/connect",
            "/api/user-settings/google/callback",
        ):
            with self.subTest(path=path):
                self.assertEqual(classify_rate_limit_scope(path, "GET"), "oauth")
        self.assertIsNone(classify_rate_limit_scope("/api/auth/github", "POST"))

    def test_topic_detail_helper_covers_its_decision_table(self):
        self.assertEqual(_classify_topic_detail_scope("/api/topics/01-http", "GET"), "llm")
        self.assertIsNone(_classify_topic_detail_scope("/api/topics/01-http", "PUT"))
        self.assertEqual(
            _classify_topic_detail_scope("/api/topics/01-http/preferences", "PUT"), "llm"
        )
        self.assertIsNone(
            _classify_topic_detail_scope("/api/topics/01-http/videos", "GET")
        )
        self.assertIsNone(_classify_topic_detail_scope("/api/questions/1", "GET"))
        self.assertIsNone(_classify_topic_detail_scope("/api/topics/", "GET"))

    def test_empty_path_and_method_are_handled(self):
        self.assertIsNone(classify_rate_limit_scope("", "GET"))
        # The /api/llm/ prefix is deliberately method-agnostic: every route
        # under it is a GET.
        self.assertEqual(classify_rate_limit_scope("/api/llm/providers", ""), "llm")


if __name__ == "__main__":
    unittest.main()