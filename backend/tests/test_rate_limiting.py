import os
import tempfile
import unittest

from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import _rate_limiter, create_app
from app.services.auth import create_jwt_token


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


if __name__ == "__main__":
    unittest.main()
