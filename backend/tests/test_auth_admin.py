import os
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import require_auth
from app.routers import auth
from app.services.oauth_state import create_oauth_state, verify_oauth_state


class AuthAdminTests(unittest.TestCase):
    def setUp(self):
        self.prev_admin_users = os.environ.get("STUDY_ADMIN_USERS")
        self.prev_llm_service_users_file = os.environ.get("STUDY_LLM_SERVICE_USERS_FILE")
        self.prev_llm_assignments_file = os.environ.get("STUDY_LLM_ASSIGNMENTS_FILE")
        self.prev_auth_secret = os.environ.get("STUDY_AUTH_SECRET_KEY")
        self.prev_frontend_url = os.environ.get("STUDY_FRONTEND_URL")
        self.prev_github_client_id = os.environ.get("STUDY_GITHUB_CLIENT_ID")
        self.prev_google_client_id = os.environ.get("STUDY_GOOGLE_CLIENT_ID")
        os.environ["STUDY_ADMIN_USERS"] = "google:admin@example.com,ops@example.com"
        os.environ["STUDY_AUTH_SECRET_KEY"] = "a" * 64
        os.environ["STUDY_FRONTEND_URL"] = "http://localhost:5174"
        os.environ["STUDY_GITHUB_CLIENT_ID"] = "github-client-id"
        os.environ["STUDY_GOOGLE_CLIENT_ID"] = "google-client-id"

        self.tmpdir = tempfile.TemporaryDirectory()
        self.allowed_file = Path(self.tmpdir.name) / "allowed_users.json"
        self.llm_service_file = Path(self.tmpdir.name) / "llm_service_users.json"
        self.llm_assignments_file = Path(self.tmpdir.name) / "llm_assignments.json"
        os.environ["STUDY_LLM_SERVICE_USERS_FILE"] = str(self.llm_service_file)
        os.environ["STUDY_LLM_ASSIGNMENTS_FILE"] = str(self.llm_assignments_file)
        get_settings.cache_clear()

        self.allowed_patch = patch("app.services.auth.ALLOWED_USERS_FILE", self.allowed_file)
        self.allowed_patch.start()
        auth._LLM_SERVICE_ACCESS = None
        auth._LLM_ASSIGNMENTS = None

        self.app = FastAPI()
        self.app.include_router(auth.router)
        self.client = TestClient(self.app)

    def tearDown(self):
        self.allowed_patch.stop()
        self.tmpdir.cleanup()
        self.app.dependency_overrides.clear()
        if self.prev_admin_users is None:
            os.environ.pop("STUDY_ADMIN_USERS", None)
        else:
            os.environ["STUDY_ADMIN_USERS"] = self.prev_admin_users
        if self.prev_llm_service_users_file is None:
            os.environ.pop("STUDY_LLM_SERVICE_USERS_FILE", None)
        else:
            os.environ["STUDY_LLM_SERVICE_USERS_FILE"] = self.prev_llm_service_users_file
        if self.prev_llm_assignments_file is None:
            os.environ.pop("STUDY_LLM_ASSIGNMENTS_FILE", None)
        else:
            os.environ["STUDY_LLM_ASSIGNMENTS_FILE"] = self.prev_llm_assignments_file
        if self.prev_auth_secret is None:
            os.environ.pop("STUDY_AUTH_SECRET_KEY", None)
        else:
            os.environ["STUDY_AUTH_SECRET_KEY"] = self.prev_auth_secret
        if self.prev_frontend_url is None:
            os.environ.pop("STUDY_FRONTEND_URL", None)
        else:
            os.environ["STUDY_FRONTEND_URL"] = self.prev_frontend_url
        if self.prev_github_client_id is None:
            os.environ.pop("STUDY_GITHUB_CLIENT_ID", None)
        else:
            os.environ["STUDY_GITHUB_CLIENT_ID"] = self.prev_github_client_id
        if self.prev_google_client_id is None:
            os.environ.pop("STUDY_GOOGLE_CLIENT_ID", None)
        else:
            os.environ["STUDY_GOOGLE_CLIENT_ID"] = self.prev_google_client_id
        auth._LLM_SERVICE_ACCESS = None
        auth._LLM_ASSIGNMENTS = None
        get_settings.cache_clear()

    def test_auth_me_includes_is_admin(self):
        self.app.dependency_overrides[require_auth] = lambda: {
            "user": "admin@example.com",
            "provider": "google",
        }
        res = self.client.get("/api/auth/me")
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertTrue(payload["is_admin"])

        self.app.dependency_overrides[require_auth] = lambda: {
            "user": "random@example.com",
            "provider": "google",
        }
        res2 = self.client.get("/api/auth/me")
        self.assertEqual(res2.status_code, 200)
        self.assertFalse(res2.json()["is_admin"])

    def test_allowed_users_endpoints_require_admin(self):
        self.app.dependency_overrides[require_auth] = lambda: {
            "user": "member@example.com",
            "provider": "google",
        }
        forbidden = self.client.get("/api/auth/allowed-users")
        self.assertEqual(forbidden.status_code, 403)

        self.app.dependency_overrides[require_auth] = lambda: {
            "user": "admin@example.com",
            "provider": "google",
        }
        ok = self.client.get("/api/auth/allowed-users")
        self.assertEqual(ok.status_code, 200)

    def test_llm_service_users_endpoints_require_admin(self):
        self.app.dependency_overrides[require_auth] = lambda: {
            "user": "member@example.com",
            "provider": "google",
        }
        forbidden = self.client.get("/api/auth/llm-service-users")
        self.assertEqual(forbidden.status_code, 403)

        self.app.dependency_overrides[require_auth] = lambda: {
            "user": "admin@example.com",
            "provider": "google",
        }
        ok = self.client.get("/api/auth/llm-service-users")
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(
            sorted(ok.json()["users"]),
            ["google:admin@example.com", "ops@example.com"],
        )

    def test_admin_can_add_remove_llm_service_user(self):
        self.app.dependency_overrides[require_auth] = lambda: {
            "user": "admin@example.com",
            "provider": "google",
        }

        add = self.client.post(
            "/api/auth/llm-service-users",
            json={"provider": "github", "identifier": "new-user"},
        )
        self.assertEqual(add.status_code, 200)
        self.assertIn("github:new-user", add.json()["users"])

        remove = self.client.delete("/api/auth/llm-service-users/github/new-user")
        self.assertEqual(remove.status_code, 200)
        self.assertNotIn("github:new-user", remove.json()["users"])

    def test_llm_assignments_endpoints_require_admin(self):
        self.app.dependency_overrides[require_auth] = lambda: {
            "user": "member@example.com",
            "provider": "google",
        }
        forbidden = self.client.get("/api/auth/llm-assignments/users")
        self.assertEqual(forbidden.status_code, 403)

    def test_admin_can_set_and_remove_llm_assignment_for_allowed_user(self):
        self.app.dependency_overrides[require_auth] = lambda: {
            "user": "admin@example.com",
            "provider": "google",
        }
        self.client.post(
            "/api/auth/allowed-users",
            json={"provider": "github", "identifier": "candidate-user"},
        )

        upsert = self.client.put(
            "/api/auth/llm-assignments/github/candidate-user",
            json={"provider": "openai", "model": "gpt-4o-mini"},
        )
        self.assertEqual(upsert.status_code, 200)
        upsert_payload = upsert.json()
        self.assertEqual(upsert_payload["identity_key"], "github:candidate-user")
        self.assertEqual(upsert_payload["assignment"]["provider"], "openai")
        self.assertEqual(upsert_payload["assignment"]["model"], "gpt-4o-mini")

        listing = self.client.get("/api/auth/llm-assignments/users")
        self.assertEqual(listing.status_code, 200)
        users = listing.json()["users"]
        self.assertTrue(any(u["identity_key"] == "github:candidate-user" for u in users))

        remove = self.client.delete("/api/auth/llm-assignments/github/candidate-user")
        self.assertEqual(remove.status_code, 200)
        self.assertIsNone(remove.json()["assignment"])

    def test_assignment_target_must_be_allowed_login_user(self):
        self.app.dependency_overrides[require_auth] = lambda: {
            "user": "admin@example.com",
            "provider": "google",
        }
        upsert = self.client.put(
            "/api/auth/llm-assignments/github/not-allowed",
            json={"provider": "openai", "model": "gpt-4o-mini"},
        )
        self.assertEqual(upsert.status_code, 404)

    def test_admin_can_set_my_assignment(self):
        self.app.dependency_overrides[require_auth] = lambda: {
            "user": "admin@example.com",
            "provider": "google",
        }

        me = self.client.put(
            "/api/auth/llm-assignments/me",
            json={"provider": "anthropic", "model": "claude-3-5-sonnet-20241022"},
        )
        self.assertEqual(me.status_code, 200)
        payload = me.json()
        self.assertEqual(payload["identity_key"], "google:admin@example.com")
        self.assertEqual(payload["assignment"]["provider"], "anthropic")

    def test_oauth_login_uses_signed_state_without_oauth_cookie(self):
        response = self.client.get("/api/auth/github", follow_redirects=False)
        self.assertIn(response.status_code, (302, 307))
        location = response.headers.get("location", "")
        state = parse_qs(urlparse(location).query).get("state", [""])[0]
        payload = verify_oauth_state(state, expected_purpose="auth_github")
        self.assertEqual(payload.get("purpose"), "auth_github")
        self.assertNotIn("oauth_state", response.headers.get("set-cookie", ""))

    def test_oauth_callback_sets_strict_session_cookie(self):
        async def _fake_exchange_google_code(code: str, redirect_uri: str) -> str:
            self.assertEqual(code, "code-123")
            self.assertTrue(redirect_uri.endswith("/api/auth/google/callback"))
            return "admin@example.com"

        state = create_oauth_state({"purpose": "auth_google"})
        with patch("app.routers.auth.auth_service.exchange_google_code", _fake_exchange_google_code):
            with patch("app.routers.auth.auth_service.is_allowed", return_value=True):
                response = self.client.get(
                    "/api/auth/google/callback",
                    params={"code": "code-123", "state": state},
                    follow_redirects=False,
                )

        self.assertIn(response.status_code, (302, 307))
        self.assertIn("session=", response.headers.get("set-cookie", ""))
        self.assertIn("samesite=strict", response.headers.get("set-cookie", "").lower())

    def test_oauth_callback_rejects_invalid_state(self):
        response = self.client.get(
            "/api/auth/github/callback",
            params={"code": "bad", "state": "tampered"},
            follow_redirects=False,
        )
        self.assertIn(response.status_code, (302, 307))
        self.assertIn("error=invalid_state", response.headers.get("location", ""))


if __name__ == "__main__":
    unittest.main()
