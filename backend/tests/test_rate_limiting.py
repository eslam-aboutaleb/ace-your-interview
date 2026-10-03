import os
import tempfile
import unittest

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import _rate_limiter, create_app
from app.services.auth import create_jwt_token
from app.services.rate_limit import classify_rate_limit_scope, request_ip


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


if __name__ == "__main__":
    unittest.main()
