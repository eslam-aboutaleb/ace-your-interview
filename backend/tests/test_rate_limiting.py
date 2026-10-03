import os
import tempfile
import unittest

from fastapi import Request
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import _init_rate_limiter, _rate_limiter, create_app
from app.services.auth import create_jwt_token
from app.services.rate_limit import (
    InMemoryRateLimiter,
    RedisBackend,
    classify_rate_limit_scope,
    create_rate_limiter,
    request_ip,
)


def _build_request(path: str, headers: dict, client_host: str):
    """Minimal ASGI request stand-in for `request_ip` (scope-derived only)."""
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


class RateLimitingTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.prev_env = {
            "STUDY_AUTH_SECRET_KEY": os.environ.get("STUDY_AUTH_SECRET_KEY"),
            "STUDY_CREDENTIALS_ENCRYPTION_KEY": os.environ.get("STUDY_CREDENTIALS_ENCRYPTION_KEY"),
            "STUDY_USER_SETTINGS_FILE": os.environ.get("STUDY_USER_SETTINGS_FILE"),
            "STUDY_LLM_SERVICE_USERS_FILE": os.environ.get("STUDY_LLM_SERVICE_USERS_FILE"),
            "STUDY_LLM_ASSIGNMENTS_FILE": os.environ.get("STUDY_LLM_ASSIGNMENTS_FILE"),
            "STUDY_LEARNING_DB_PATH": os.environ.get("STUDY_LEARNING_DB_PATH"),
            "STUDY_ENABLE_RATE_LIMITING": os.environ.get("STUDY_ENABLE_RATE_LIMITING"),
            "STUDY_OAUTH_RATE_LIMIT_REQUESTS": os.environ.get("STUDY_OAUTH_RATE_LIMIT_REQUESTS"),
            "STUDY_OAUTH_RATE_LIMIT_WINDOW_SECONDS": os.environ.get("STUDY_OAUTH_RATE_LIMIT_WINDOW_SECONDS"),
            "STUDY_SESSION_RATE_LIMIT_REQUESTS": os.environ.get("STUDY_SESSION_RATE_LIMIT_REQUESTS"),
            "STUDY_SESSION_RATE_LIMIT_WINDOW_SECONDS": os.environ.get("STUDY_SESSION_RATE_LIMIT_WINDOW_SECONDS"),
            "STUDY_LLM_RATE_LIMIT_REQUESTS": os.environ.get("STUDY_LLM_RATE_LIMIT_REQUESTS"),
            "STUDY_LLM_RATE_LIMIT_WINDOW_SECONDS": os.environ.get("STUDY_LLM_RATE_LIMIT_WINDOW_SECONDS"),
            "STUDY_TRUSTED_PROXY_ENABLED": os.environ.get("STUDY_TRUSTED_PROXY_ENABLED"),
        }

        os.environ["STUDY_AUTH_SECRET_KEY"] = "a" * 64
        os.environ["STUDY_CREDENTIALS_ENCRYPTION_KEY"] = "rate-limit-encryption-secret"
        os.environ["STUDY_USER_SETTINGS_FILE"] = os.path.join(self.tmpdir.name, "user_settings.json")
        os.environ["STUDY_LLM_SERVICE_USERS_FILE"] = os.path.join(self.tmpdir.name, "llm_service_users.json")
        os.environ["STUDY_LLM_ASSIGNMENTS_FILE"] = os.path.join(self.tmpdir.name, "llm_assignments.json")
        os.environ["STUDY_LEARNING_DB_PATH"] = os.path.join(self.tmpdir.name, "learning.db")
        os.environ["STUDY_ENABLE_RATE_LIMITING"] = "true"
        os.environ["STUDY_OAUTH_RATE_LIMIT_REQUESTS"] = "1"
        os.environ["STUDY_OAUTH_RATE_LIMIT_WINDOW_SECONDS"] = "120"
        os.environ["STUDY_SESSION_RATE_LIMIT_REQUESTS"] = "1"
        os.environ["STUDY_SESSION_RATE_LIMIT_WINDOW_SECONDS"] = "120"
        os.environ["STUDY_LLM_RATE_LIMIT_REQUESTS"] = "1"
        os.environ["STUDY_LLM_RATE_LIMIT_WINDOW_SECONDS"] = "120"

        get_settings.cache_clear()
        _rate_limiter.clear()

    def tearDown(self):
        self.tmpdir.cleanup()
        for key, value in self.prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()
        _rate_limiter.clear()

    def test_oauth_session_and_llm_endpoints_are_limited(self):
        app = create_app()
        with TestClient(app) as client:
            oauth_first = client.get("/api/auth/github", follow_redirects=False)
            self.assertIn(oauth_first.status_code, (302, 307))
            oauth_second = client.get("/api/auth/github", follow_redirects=False)
            self.assertEqual(oauth_second.status_code, 429)
            self.assertEqual(oauth_second.json()["detail"]["scope"], "oauth")

            session_token = create_jwt_token("rate-user@example.com", "google")
            client.cookies.set("session", session_token)

            session_first = client.get("/api/auth/me")
            self.assertEqual(session_first.status_code, 200)
            session_second = client.get("/api/auth/me")
            self.assertEqual(session_second.status_code, 429)
            self.assertEqual(session_second.json()["detail"]["scope"], "session")

            llm_first = client.get("/api/llm/providers")
            self.assertEqual(llm_first.status_code, 200)
            llm_second = client.get("/api/llm/providers")
            self.assertEqual(llm_second.status_code, 429)
            self.assertEqual(llm_second.json()["detail"]["scope"], "llm")
            self.assertIn("Retry-After", llm_second.headers)

    def test_explicit_progress_save_is_limited_but_autosave_is_not(self):
        app = create_app()
        with TestClient(app) as client:
            session_token = create_jwt_token("rate-user@example.com", "google")
            client.cookies.set("session", session_token)

            payload = {
                "topic_title": "HTTP Fundamentals",
                "questions": [
                    {
                        "question_id": "q1",
                        "question": "Explain keep-alive timeouts.",
                        "difficulty": "medium",
                        "revealed": True,
                        "is_correct": True,
                        "confidence": 4,
                    }
                ],
                "sections": ["HTTP Basics"],
            }

            save_first = client.post("/api/progress/01-http/save", json=payload)
            self.assertEqual(save_first.status_code, 200)
            save_second = client.post("/api/progress/01-http/save", json=payload)
            self.assertEqual(save_second.status_code, 429)
            self.assertEqual(save_second.json()["detail"]["scope"], "llm")

    def test_progress_scope_classification(self):
        self.assertEqual(
            classify_rate_limit_scope("/api/progress/01-http/save", "POST"), "llm"
        )
        # The beacon must stay unlimited or a 429 would drop the durable write.
        self.assertIsNone(
            classify_rate_limit_scope("/api/progress/01-http/autosave", "POST")
        )
        self.assertIsNone(classify_rate_limit_scope("/api/progress/topics", "GET"))

    # ── Stage 3.1: LLM-backed topic GET/PUT scopes ─────────

    def test_topic_detail_and_preferences_are_llm_scoped(self):
        # All three resolve the language profile, which runs the LLM advisor.
        self.assertEqual(classify_rate_limit_scope("/api/topics/01-http", "GET"), "llm")
        self.assertEqual(
            classify_rate_limit_scope("/api/topics/01-http/preferences", "GET"), "llm"
        )
        self.assertEqual(
            classify_rate_limit_scope("/api/topics/01-http/preferences", "PUT"), "llm"
        )
        # A user-scoped custom topic id takes the same route.
        self.assertEqual(classify_rate_limit_scope("/api/topics/custom-abc123", "GET"), "llm")

    def test_topic_list_and_non_llm_topic_routes_stay_unlimited(self):
        self.assertIsNone(classify_rate_limit_scope("/api/topics", "GET"))
        self.assertIsNone(classify_rate_limit_scope("/api/topics/", "GET"))
        self.assertIsNone(classify_rate_limit_scope("/api/topics/01-http/videos", "GET"))
        self.assertIsNone(classify_rate_limit_scope("/api/topics/01-http", "DELETE"))
        # The POST-scoped generation routes keep their own classification.
        self.assertEqual(classify_rate_limit_scope("/api/topics/custom", "POST"), "llm")
        self.assertEqual(
            classify_rate_limit_scope("/api/topics/01-http/content/generate/stream", "POST"),
            "llm",
        )

    def test_topic_preferences_scope_is_enforced_end_to_end(self):
        app = create_app()
        with TestClient(app) as client:
            session_token = create_jwt_token("topic-user@example.com", "google")
            client.cookies.set("session", session_token)

            first = client.get("/api/topics/does-not-exist")
            self.assertEqual(first.status_code, 404)
            second = client.get("/api/topics/does-not-exist")
            self.assertEqual(second.status_code, 429)
            self.assertEqual(second.json()["detail"]["scope"], "llm")
            self.assertEqual(second.json()["detail"]["code"], "rate_limit_exceeded")

    def test_voice_routes_are_not_http_rate_limited(self):
        # The HTTP voice routes make no LLM call, and the WebSocket is throttled
        # per connection inside routers/voice.py — never by the HTTP middleware.
        self.assertIsNone(classify_rate_limit_scope("/api/voice/stream", "GET"))
        self.assertIsNone(classify_rate_limit_scope("/api/voice/config", "GET"))
        self.assertIsNone(classify_rate_limit_scope("/api/voice/settings", "PUT"))
        app = create_app()
        with TestClient(app) as client:
            session_token = create_jwt_token("voice-user@example.com", "google")
            client.cookies.set("session", session_token)
            for _ in range(3):
                res = client.get("/api/voice/config")
                self.assertNotEqual(res.status_code, 429, res.text)

    # ── Stage 3.1: trusted-proxy gating ────────────────────

    def test_forwarded_for_is_ignored_without_trusted_proxy(self):
        os.environ["STUDY_TRUSTED_PROXY_ENABLED"] = "false"
        get_settings.cache_clear()
        request = _build_request(
            "/api/questions/generate",
            {"x-forwarded-for": "203.0.113.7, 10.0.0.1"},
            "10.1.2.3",
        )
        self.assertEqual(request_ip(request), "10.1.2.3")

        spoofed = _build_request(
            "/api/questions/generate",
            {"x-forwarded-for": "203.0.113.7"},
            "unknown-host",
        )
        self.assertEqual(request_ip(spoofed), "unknown-host")

        real_ip = _build_request("/api/questions/generate", {"x-real-ip": "203.0.113.9"}, "10.1.2.3")
        self.assertEqual(request_ip(real_ip), "10.1.2.3")

    def test_forwarded_for_is_honoured_with_trusted_proxy(self):
        os.environ["STUDY_TRUSTED_PROXY_ENABLED"] = "true"
        get_settings.cache_clear()
        request = _build_request(
            "/api/questions/generate",
            {"x-forwarded-for": "203.0.113.7, 10.0.0.1"},
            "10.1.2.3",
        )
        self.assertEqual(request_ip(request), "203.0.113.7")

        real_ip = _build_request("/api/questions/generate", {"x-real-ip": "203.0.113.9"}, "10.1.2.3")
        self.assertEqual(request_ip(real_ip), "203.0.113.9")

        # No proxy headers at all: fall back to the transport peer.
        plain = _build_request("/api/questions/generate", {}, "10.1.2.3")
        self.assertEqual(request_ip(plain), "10.1.2.3")

    def test_trusted_proxy_setting_changes_the_effective_limiter_key(self):
        os.environ["STUDY_TRUSTED_PROXY_ENABLED"] = "false"
        get_settings.cache_clear()
        request = _build_request("/api/questions/generate", {"x-forwarded-for": "203.0.113.7"}, "10.1.2.3")
        self.assertEqual(request_ip(request), "10.1.2.3")

        os.environ["STUDY_TRUSTED_PROXY_ENABLED"] = "true"
        get_settings.cache_clear()
        self.assertEqual(request_ip(request), "203.0.113.7")


# ── Pluggable backend: parity, factory, startup wiring ──────


class _FakeClock:
    """Deterministic clock shared by both backends under test."""

    def __init__(self, start: float = 1_700_000_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeRedis:
    """Minimal sorted-set client covering the commands RedisBackend uses."""

    def __init__(self):
        self.sets: dict[str, dict[str, float]] = {}
        self.expirations: dict[str, int] = {}

    @staticmethod
    def _as_score(value) -> float:
        if isinstance(value, str):
            if value == "-inf":
                return float("-inf")
            if value == "+inf":
                return float("inf")
            return float(value)
        return float(value)

    def zremrangebyscore(self, key, min_score, max_score):
        low = self._as_score(min_score)
        high = self._as_score(max_score)
        bucket = self.sets.setdefault(key, {})
        stale = [
            member for member, score in bucket.items() if low <= score <= high
        ]
        for member in stale:
            del bucket[member]
        return len(stale)

    def zcard(self, key):
        return len(self.sets.get(key, {}))

    def zrange(self, key, start, stop, withscores=False):
        items = sorted(
            self.sets.get(key, {}).items(),
            key=lambda item: (item[1], item[0]),
        )
        selected = items[start : stop + 1]
        return selected if withscores else [member for member, _ in selected]

    def zadd(self, key, mapping):
        self.sets.setdefault(key, {}).update(mapping)
        return len(mapping)

    def expire(self, key, seconds):
        self.expirations[key] = seconds
        return True

    def ping(self):
        return True


class RateLimiterBackendParityTests(unittest.TestCase):
    """The same request sequence yields the same decisions in both backends."""

    _SCOPE = "llm"
    _KEY = "user:parity"

    def _check(self, backend, *, limit=3, window_seconds=60):
        return backend.check(
            scope=self._SCOPE,
            key=self._KEY,
            limit=limit,
            window_seconds=window_seconds,
        )

    def test_backends_agree_on_allow_deny_and_retry(self):
        clock = _FakeClock()
        memory = InMemoryRateLimiter(clock=clock)
        redis = RedisBackend(client=FakeRedis(), clock=clock)

        memory_decisions = [self._check(memory) for _ in range(5)]
        redis_decisions = [self._check(redis) for _ in range(5)]

        self.assertEqual(
            [(d.allowed, d.retry_after_seconds) for d in memory_decisions],
            [(d.allowed, d.retry_after_seconds) for d in redis_decisions],
        )
        for decision in memory_decisions[:3]:
            self.assertTrue(decision.allowed)
            self.assertEqual(decision.retry_after_seconds, 0)
        for decision in memory_decisions[3:]:
            self.assertFalse(decision.allowed)
            self.assertEqual(decision.retry_after_seconds, 60)

    def test_backends_agree_after_partial_window_expiry(self):
        clock = _FakeClock()
        memory = InMemoryRateLimiter(clock=clock)
        redis = RedisBackend(client=FakeRedis(), clock=clock)

        for _ in range(3):
            self._check(memory)
            self._check(redis)

        clock.advance(30)
        denied_memory = self._check(memory)
        denied_redis = self._check(redis)
        self.assertFalse(denied_memory.allowed)
        self.assertFalse(denied_redis.allowed)
        self.assertEqual(denied_memory.retry_after_seconds, 30)
        self.assertEqual(denied_redis.retry_after_seconds, 30)

        # Once the window fully elapses, both backends allow again.
        clock.advance(31)
        self.assertTrue(self._check(memory).allowed)
        self.assertTrue(self._check(redis).allowed)

    def test_backends_agree_per_scope_and_key(self):
        clock = _FakeClock()
        memory = InMemoryRateLimiter(clock=clock)
        redis = RedisBackend(client=FakeRedis(), clock=clock)

        for backend in (memory, redis):
            # Exhaust the llm scope for one key.
            for _ in range(2):
                backend.check(
                    scope="llm", key="user:a", limit=2, window_seconds=60
                )
            # A different key in the same scope is unaffected.
            self.assertTrue(
                backend.check(
                    scope="llm", key="user:b", limit=2, window_seconds=60
                ).allowed
            )
            # A different scope is unaffected.
            self.assertTrue(
                backend.check(
                    scope="session", key="user:a", limit=2, window_seconds=60
                ).allowed
            )
            # The exhausted bucket denies.
            self.assertFalse(
                backend.check(
                    scope="llm", key="user:a", limit=2, window_seconds=60
                ).allowed
            )

    def test_redis_backend_uses_sorted_set_commands(self):
        fake = FakeRedis()
        backend = RedisBackend(client=fake, clock=_FakeClock())

        decision = backend.check(
            scope="oauth", key="ip:10.0.0.1", limit=5, window_seconds=120
        )

        self.assertTrue(decision.allowed)
        bucket = "ratelimit:oauth:ip:10.0.0.1"
        self.assertIn(bucket, fake.sets)
        self.assertEqual(fake.zcard(bucket), 1)
        self.assertEqual(fake.expirations.get(bucket), 120)

    def test_redis_backend_clear_is_a_safe_no_op(self):
        backend = RedisBackend(client=FakeRedis(), clock=_FakeClock())
        backend.clear()  # must not raise


class RateLimiterFactoryTests(unittest.TestCase):
    def test_default_backend_is_in_memory(self):
        self.assertIsInstance(create_rate_limiter(Settings()), InMemoryRateLimiter)

    def test_explicit_memory_backend_is_in_memory(self):
        settings = Settings(rate_limit_backend="memory")
        self.assertIsInstance(create_rate_limiter(settings), InMemoryRateLimiter)

    def test_unknown_backend_name_is_rejected(self):
        settings = Settings(rate_limit_backend="memcached")
        with self.assertRaises(ValueError):
            create_rate_limiter(settings)

    def test_redis_backend_requires_a_url(self):
        settings = Settings(rate_limit_backend="redis", redis_url="")
        with self.assertRaises(RuntimeError):
            create_rate_limiter(settings)


class RateLimiterInitTests(unittest.TestCase):
    """main._init_rate_limiter wires the configured backend at startup."""

    def test_memory_backend_keeps_the_shared_instance(self):
        self.assertIs(_init_rate_limiter(Settings()), _rate_limiter)

    def test_redis_strict_mode_fails_fast(self):
        settings = Settings(rate_limit_backend="redis", rate_limit_strict=True)
        with self.assertRaises(RuntimeError):
            _init_rate_limiter(settings)

    def test_redis_non_strict_falls_back_to_in_memory(self):
        settings = Settings(rate_limit_backend="redis", rate_limit_strict=False)
        self.assertIs(_init_rate_limiter(settings), _rate_limiter)


if __name__ == "__main__":
    unittest.main()
