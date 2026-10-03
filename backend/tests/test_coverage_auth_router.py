"""Coverage for ``app.routers.auth`` — OAuth redirects, session, admin CRUD.

OAuth exchanges are stubbed at ``auth_service`` level so no token endpoint is
contacted. The admin identity comes from ``dependency_overrides[require_auth]``.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import require_auth
from app.routers import auth
from app.services.oauth_state import create_oauth_state, verify_oauth_state

STRONG_SECRET = "a" * 64
ADMIN = {"user": "admin@example.com", "provider": "google"}
MEMBER = {"user": "member@example.com", "provider": "google"}


class AuthRouterTestBase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.allowed_file = Path(self.tmpdir.name) / "allowed_users.json"
        self.env_keys = [
            "STUDY_ADMIN_USERS",
            "STUDY_AUTH_SECRET_KEY",
            "STUDY_FRONTEND_URL",
            "STUDY_GITHUB_CLIENT_ID",
            "STUDY_GOOGLE_CLIENT_ID",
            "STUDY_LLM_SERVICE_USERS_FILE",
            "STUDY_LLM_ASSIGNMENTS_FILE",
        ]
        self.prev_env = {key: os.environ.get(key) for key in self.env_keys}
        os.environ["STUDY_ADMIN_USERS"] = "google:admin@example.com"
        os.environ["STUDY_AUTH_SECRET_KEY"] = STRONG_SECRET
        os.environ["STUDY_FRONTEND_URL"] = "http://localhost:5174"
        os.environ["STUDY_GITHUB_CLIENT_ID"] = "github-client-id"
        os.environ["STUDY_GOOGLE_CLIENT_ID"] = "google-client-id"
        os.environ["STUDY_LLM_SERVICE_USERS_FILE"] = str(
            Path(self.tmpdir.name) / "llm_service_users.json"
        )
        os.environ["STUDY_LLM_ASSIGNMENTS_FILE"] = str(
            Path(self.tmpdir.name) / "llm_assignments.json"
        )
        get_settings.cache_clear()

        self.allowed_patch = patch(
            "app.services.auth.get_allowed_users_file",
            return_value=self.allowed_file,
        )
        self.allowed_patch.start()
        auth._LLM_SERVICE_ACCESS = None
        auth._LLM_ASSIGNMENTS = None

        self.app = FastAPI()
        self.app.include_router(auth.router)
        self.client = TestClient(self.app)
        self.app.dependency_overrides[require_auth] = lambda: dict(ADMIN)

    def tearDown(self):
        self.allowed_patch.stop()
        self.tmpdir.cleanup()
        self.app.dependency_overrides.clear()
        auth._LLM_SERVICE_ACCESS = None
        auth._LLM_ASSIGNMENTS = None
        for key, value in self.prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()

    def _write_allowed_users(self, github_users=None, google_emails=None):
        with open(self.allowed_file, "w") as f:
            json.dump(
                {
                    "github_users": github_users or [],
                    "google_emails": google_emails or [],
                },
                f,
            )

    def _location(self, response) -> str:
        return response.headers.get("location", "")


class GoogleLoginRedirectTests(AuthRouterTestBase):
    def test_google_login_redirect_carries_a_scoped_signed_state(self):
        response = self.client.get("/api/auth/google", follow_redirects=False)
        self.assertIn(response.status_code, (302, 307))

        parsed = urlparse(response.headers["location"])
        self.assertEqual(parsed.scheme + "://" + parsed.netloc + parsed.path,
                         "https://accounts.google.com/o/oauth2/v2/auth")
        params = parse_qs(parsed.query)
        self.assertEqual(params["client_id"], ["google-client-id"])
        self.assertEqual(params["response_type"], ["code"])
        self.assertEqual(params["access_type"], ["offline"])
        self.assertEqual(
            params["redirect_uri"],
            ["http://localhost:5174/api/auth/google/callback"],
        )
        payload = verify_oauth_state(params["state"][0], expected_purpose="auth_google")
        self.assertEqual(payload["purpose"], "auth_google")


class GithubCallbackTests(AuthRouterTestBase):
    def _state(self) -> str:
        return create_oauth_state({"purpose": "auth_github"})

    def test_successful_callback_sets_a_session_cookie_for_the_app_root(self):
        async def _exchange(code: str) -> str:
            self.assertEqual(code, "code-1")
            return "OctoCat"

        with patch("app.routers.auth.auth_service.exchange_github_code", _exchange):
            with patch("app.routers.auth.auth_service.is_allowed", return_value=True):
                response = self.client.get(
                    "/api/auth/github/callback",
                    params={"code": "code-1", "state": self._state()},
                    follow_redirects=False,
                )

        self.assertIn(response.status_code, (302, 307))
        self.assertEqual(self._location(response), "http://localhost:5174/")
        cookie = response.headers.get("set-cookie", "")
        self.assertIn("session=", cookie)
        self.assertIn("samesite=strict", cookie.lower())
        self.assertIn("httponly", cookie.lower())

    def test_exchange_failure_redirects_without_setting_a_session(self):
        async def _explode(code: str) -> str:
            raise RuntimeError("github down")

        with patch("app.routers.auth.auth_service.exchange_github_code", _explode):
            response = self.client.get(
                "/api/auth/github/callback",
                params={"code": "code-1", "state": self._state()},
                follow_redirects=False,
            )

        self.assertIn(response.status_code, (302, 307))
        self.assertIn("error=exchange_failed", self._location(response))
        self.assertNotIn("session=", response.headers.get("set-cookie", ""))

    def test_user_outside_the_allowlist_is_denied(self):
        async def _exchange(code: str) -> str:
            return "stranger"

        with patch("app.routers.auth.auth_service.exchange_github_code", _exchange):
            with patch("app.routers.auth.auth_service.is_allowed", return_value=False):
                response = self.client.get(
                    "/api/auth/github/callback",
                    params={"code": "code-1", "state": self._state()},
                    follow_redirects=False,
                )

        self.assertIn(response.status_code, (302, 307))
        self.assertIn("error=access_denied", self._location(response))
        self.assertNotIn("session=", response.headers.get("set-cookie", ""))

    def test_invalid_state_is_rejected_before_the_exchange_runs(self):
        async def _should_not_run(code: str) -> str:  # pragma: no cover - guard
            raise AssertionError("exchange must not run for an invalid state")

        with patch("app.routers.auth.auth_service.exchange_github_code", _should_not_run):
            response = self.client.get(
                "/api/auth/github/callback",
                params={"code": "code-1", "state": "tampered"},
                follow_redirects=False,
            )

        self.assertIn(response.status_code, (302, 307))
        self.assertIn("error=invalid_state", self._location(response))


class GoogleCallbackTests(AuthRouterTestBase):
    def _state(self) -> str:
        return create_oauth_state({"purpose": "auth_google"})

    def test_successful_callback_sets_the_session_cookie(self):
        async def _exchange(code: str, redirect_uri: str) -> str:
            self.assertEqual(code, "code-1")
            self.assertTrue(redirect_uri.endswith("/api/auth/google/callback"))
            return "Learner@Example.com"

        with patch("app.routers.auth.auth_service.exchange_google_code", _exchange):
            with patch("app.routers.auth.auth_service.is_allowed", return_value=True):
                response = self.client.get(
                    "/api/auth/google/callback",
                    params={"code": "code-1", "state": self._state()},
                    follow_redirects=False,
                )

        self.assertIn(response.status_code, (302, 307))
        self.assertEqual(self._location(response), "http://localhost:5174/")
        self.assertIn("session=", response.headers.get("set-cookie", ""))

    def test_exchange_failure_redirects_to_the_login_error(self):
        async def _explode(code: str, redirect_uri: str) -> str:
            raise RuntimeError("google down")

        with patch("app.routers.auth.auth_service.exchange_google_code", _explode):
            response = self.client.get(
                "/api/auth/google/callback",
                params={"code": "code-1", "state": self._state()},
                follow_redirects=False,
            )

        self.assertIn("error=exchange_failed", self._location(response))

    def test_invalid_state_is_rejected(self):
        response = self.client.get(
            "/api/auth/google/callback",
            params={"code": "code-1", "state": "tampered"},
            follow_redirects=False,
        )
        self.assertIn(response.status_code, (302, 307))
        self.assertIn("error=invalid_state", self._location(response))

    def test_email_outside_the_allowlist_is_denied(self):
        async def _exchange(code: str, redirect_uri: str) -> str:
            return "stranger@example.com"

        with patch("app.routers.auth.auth_service.exchange_google_code", _exchange):
            with patch("app.routers.auth.auth_service.is_allowed", return_value=False):
                response = self.client.get(
                    "/api/auth/google/callback",
                    params={"code": "code-1", "state": self._state()},
                    follow_redirects=False,
                )

        self.assertIn("error=access_denied", self._location(response))
        self.assertNotIn("session=", response.headers.get("set-cookie", ""))


class SessionEndpointTests(AuthRouterTestBase):
    def test_logout_clears_the_session_cookie(self):
        response = self.client.post("/api/auth/logout")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True})

        cookie = response.headers.get("set-cookie", "")
        self.assertIn("session=", cookie)
        self.assertIn('Max-Age=0', cookie)
        self.assertIn('Path=/', cookie)


class AllowedUsersCrudTests(AuthRouterTestBase):
    def test_admin_can_add_list_and_delete_an_allowed_user(self):
        added = self.client.post(
            "/api/auth/allowed-users",
            json={"provider": "github", "identifier": "candidate"},
        )
        self.assertEqual(added.status_code, 200)
        self.assertIn("candidate", added.json()["github_users"])

        listed = self.client.get("/api/auth/allowed-users")
        self.assertEqual(listed.json()["github_users"], ["candidate"])

        removed = self.client.delete("/api/auth/allowed-users/github/candidate")
        self.assertEqual(removed.status_code, 200)
        self.assertNotIn("candidate", removed.json()["github_users"])

    def test_delete_allowed_user_rejects_a_non_admin(self):
        self.app.dependency_overrides[require_auth] = lambda: dict(MEMBER)
        response = self.client.delete("/api/auth/allowed-users/github/candidate")
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"], "Admin access required")

    def test_add_allowed_user_validates_the_body(self):
        response = self.client.post("/api/auth/allowed-users", json={"provider": "github"})
        self.assertEqual(response.status_code, 422)


class LoginIdentityHelperTests(AuthRouterTestBase):
    def test_invalid_provider_yields_an_empty_identity(self):
        self.assertEqual(auth._normalise_login_identity("gitlab", "alice"), ("", "", ""))
        self.assertEqual(auth._normalise_login_identity("github", "  "), ("", "", ""))
        self.assertEqual(
            auth._normalise_login_identity(" GitHub ", " Alice "),
            ("github", "alice", "github:alice"),
        )

    def test_blank_and_duplicate_identities_are_skipped_once(self):
        self._write_allowed_users(
            github_users=["", "Alice", "alice", "bob"],
            google_emails=["ALICE@example.com", "alice@example.com"],
        )

        identities = auth._allowed_login_identities()
        keys = [identity_key for _, _, identity_key in identities]
        self.assertEqual(
            keys,
            ["github:alice", "github:bob", "google:alice@example.com"],
        )
        self.assertEqual(len(keys), len(set(keys)))

    def test_empty_identity_key_is_never_allowed(self):
        self._write_allowed_users(github_users=["alice"])
        self.assertFalse(auth._is_allowed_login_identity(""))
        self.assertFalse(auth._is_allowed_login_identity("   "))
        self.assertTrue(auth._is_allowed_login_identity("github:alice"))


class LlmAssignmentRouteTests(AuthRouterTestBase):
    def test_upsert_for_an_unsupported_provider_is_a_400(self):
        response = self.client.put(
            "/api/auth/llm-assignments/gitlab/alice",
            json={"provider": "openai", "model": "gpt-4o-mini"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Invalid user identity")

    def test_delete_for_an_unsupported_provider_is_a_400(self):
        response = self.client.delete("/api/auth/llm-assignments/gitlab/alice")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Invalid user identity")

    def test_upsert_with_an_unsupported_model_surfaces_the_store_error(self):
        self._write_allowed_users(github_users=["alice"])
        response = self.client.put(
            "/api/auth/llm-assignments/github/alice",
            json={"provider": "openai", "model": "not-a-real-model"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported model", response.json()["detail"])

    def test_get_my_assignment_reports_the_caller_identity(self):
        res = self.client.get("/api/auth/llm-assignments/me")
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertEqual(payload["login_provider"], "google")
        self.assertEqual(payload["identifier"], "admin@example.com")
        self.assertEqual(payload["identity_key"], "google:admin@example.com")
        self.assertIsNone(payload["assignment"])
        self.assertIn("openai", payload["provider_models"])

    def test_upsert_my_assignment_with_an_unsupported_model_is_a_400(self):
        response = self.client.put(
            "/api/auth/llm-assignments/me",
            json={"provider": "openai", "model": "not-a-real-model"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported model", response.json()["detail"])

    def test_get_my_assignment_requires_admin(self):
        self.app.dependency_overrides[require_auth] = lambda: dict(MEMBER)
        response = self.client.get("/api/auth/llm-assignments/me")
        self.assertEqual(response.status_code, 403)


if __name__ == "__main__":
    unittest.main()