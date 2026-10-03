"""Coverage for ``app.routers.user_settings`` — approval gate, Google OAuth, disconnect.

The Google token endpoint is replaced with a fake client so no network call is
made; the real ``UserSettingsStore`` is used on a temp file.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import require_auth
from app.routers import user_settings
from app.services.oauth_state import create_oauth_state
from app.services.user_settings_store import UserSettingsStore, identity_key_for_user

IDENTITY = {"user": "learner@example.com", "provider": "google"}
IDENTITY_KEY = "google:learner@example.com"


class _FakeResponse:
    def __init__(self, payload=None, raises=None):
        self._payload = payload if payload is not None else {}
        self.raises = raises

    def raise_for_status(self):
        if self.raises is not None:
            raise self.raises

    def json(self):
        return self._payload


class _FakeOAuthClient:
    def __init__(self, payload=None, raises=None):
        self.payload = payload or {
            "access_token": "ya29.token",
            "refresh_token": "1//refresh",
            "expires_in": 3600,
        }
        self.raises = raises
        self.posts: list[dict] = []

    async def post(self, url, **kwargs):
        self.posts.append({"url": url, **kwargs})
        return _FakeResponse(self.payload, self.raises)


class UserSettingsRouterCoverageTestBase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.env_keys = [
            "STUDY_USER_SETTINGS_FILE",
            "STUDY_LLM_SERVICE_USERS_FILE",
            "STUDY_LLM_ASSIGNMENTS_FILE",
            "STUDY_CREDENTIALS_ENCRYPTION_KEY",
            "STUDY_GOOGLE_CLIENT_ID",
            "STUDY_GOOGLE_CLIENT_SECRET",
            "STUDY_FRONTEND_URL",
            "STUDY_AUTH_SECRET_KEY",
        ]
        self.prev_env = {key: os.environ.get(key) for key in self.env_keys}
        os.environ["STUDY_USER_SETTINGS_FILE"] = os.path.join(
            self.tmpdir.name, "user_settings.json"
        )
        os.environ["STUDY_LLM_SERVICE_USERS_FILE"] = os.path.join(
            self.tmpdir.name, "llm_service_users.json"
        )
        os.environ["STUDY_LLM_ASSIGNMENTS_FILE"] = os.path.join(
            self.tmpdir.name, "llm_assignments.json"
        )
        os.environ["STUDY_CREDENTIALS_ENCRYPTION_KEY"] = "coverage-encryption-secret"
        os.environ["STUDY_GOOGLE_CLIENT_ID"] = "google-client-id"
        os.environ["STUDY_GOOGLE_CLIENT_SECRET"] = "google-client-secret"
        os.environ["STUDY_FRONTEND_URL"] = "http://localhost:5174"
        os.environ["STUDY_AUTH_SECRET_KEY"] = "u" * 64
        get_settings.cache_clear()

        self.store = UserSettingsStore()
        self._prev_store = user_settings._STORE
        user_settings.init(self.store)

        self.app = FastAPI()
        self.app.include_router(user_settings.router)
        self.app.dependency_overrides[require_auth] = lambda: dict(IDENTITY)
        self.client = TestClient(self.app)

    def tearDown(self):
        self.app.dependency_overrides.clear()
        user_settings._STORE = self._prev_store
        self.tmpdir.cleanup()
        for key, value in self.prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()

    def _state(self, identity_key: str = IDENTITY_KEY) -> str:
        return create_oauth_state(
            {
                "purpose": "user_settings_google_connect",
                "identity_key": identity_key,
            }
        )


class StoreGuardTests(UserSettingsRouterCoverageTestBase):
    def test_every_endpoint_reports_503_when_the_store_is_missing(self):
        user_settings._STORE = None
        cases = [
            ("get", "/api/user-settings", None),
            ("put", "/api/user-settings/preferences", {"provider": "openai",
                                                       "model": "gpt-4o-mini",
                                                       "auth_mode": "api_key"}),
            ("put", "/api/user-settings/api-key/openai", {"api_key": "sk-x"}),
            ("delete", "/api/user-settings/api-key/openai", None),
            ("post", "/api/user-settings/google/disconnect", None),
        ]
        for method, path, body in cases:
            with self.subTest(path=path):
                kwargs = {"json": body} if body is not None else {}
                response = getattr(self.client, method)(path, **kwargs)
                self.assertEqual(response.status_code, 503)
                self.assertEqual(
                    response.json()["detail"], "User settings store not initialised"
                )

    def test_google_connect_still_redirects_without_the_store(self):
        user_settings._STORE = None
        response = self.client.get(
            "/api/user-settings/google/connect", follow_redirects=False
        )
        self.assertIn(response.status_code, (302, 307))
        self.assertIn("accounts.google.com", response.headers["location"])


class PreferencePolicyTests(UserSettingsRouterCoverageTestBase):
    def test_study_app_source_requires_backend_approval(self):
        response = self.client.put(
            "/api/user-settings/preferences",
            json={"provider": "openai", "model": "gpt-4o-mini",
                  "auth_mode": "api_key", "llm_source": "study_app"},
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"]["code"], "llm_service_approval_required")
        self.assertIn("Admin approval required", response.json()["detail"]["message"])

    def test_approved_user_may_select_study_app_and_auth_mode_is_forced(self):
        self.store._llm_service_access.add_user("google", "learner@example.com")
        response = self.client.put(
            "/api/user-settings/preferences",
            json={"provider": "openai", "model": "gpt-4o-mini",
                  "auth_mode": "account", "llm_source": "study_app"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["llm_source"], "study_app")
        self.assertEqual(response.json()["auth_mode"], "api_key")

    def test_account_auth_mode_is_rejected_for_a_non_google_provider(self):
        response = self.client.put(
            "/api/user-settings/preferences",
            json={"provider": "openai", "model": "gpt-4o-mini", "auth_mode": "account"},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("only supported for provider=google", response.json()["detail"])

    def test_account_auth_mode_is_accepted_for_google(self):
        response = self.client.put(
            "/api/user-settings/preferences",
            json={"provider": "google", "model": "gemini-1.5-pro",
                  "auth_mode": "account"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["auth_mode"], "account")


class ApiKeyValidationTests(UserSettingsRouterCoverageTestBase):
    def test_unsupported_provider_is_reported_as_a_400(self):
        response = self.client.put(
            "/api/user-settings/api-key/cohere", json={"api_key": "co-x"}
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Unsupported provider")

    def test_blank_api_key_is_rejected_by_the_schema(self):
        response = self.client.put(
            "/api/user-settings/api-key/openai", json={"api_key": ""}
        )
        self.assertEqual(response.status_code, 422)


class GoogleConnectRedirectTests(UserSettingsRouterCoverageTestBase):
    def test_connect_requires_a_configured_google_client(self):
        os.environ["STUDY_GOOGLE_CLIENT_ID"] = ""
        get_settings.cache_clear()
        response = self.client.get("/api/user-settings/google/connect")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Google OAuth is not configured")

    def test_connect_redirect_requests_consent_and_carries_the_identity(self):
        response = self.client.get(
            "/api/user-settings/google/connect", follow_redirects=False
        )
        self.assertIn(response.status_code, (302, 307))
        self.assertIn("prompt=consent", response.headers["location"])
        self.assertIn("state=", response.headers["location"])


class GoogleCallbackErrorTests(UserSettingsRouterCoverageTestBase):
    def _callback(self, state: str, client: _FakeOAuthClient):
        with patch(
            "app.routers.user_settings.get_oauth_http_client", return_value=client
        ):
            return self.client.get(
                "/api/user-settings/google/callback",
                params={"code": "code-1", "state": state},
                follow_redirects=False,
            )

    def test_tampered_state_is_rejected_before_any_token_request(self):
        client = _FakeOAuthClient()
        response = self._callback("tampered", client)
        self.assertIn(response.status_code, (302, 307))
        self.assertIn("error=invalid_state", response.headers["location"])
        self.assertEqual(client.posts, [])

    def test_state_without_a_usable_identity_is_rejected(self):
        client = _FakeOAuthClient()
        response = self._callback(self._state(identity_key="no-colon-here"), client)
        self.assertIn("error=invalid_state", response.headers["location"])
        self.assertEqual(client.posts, [])

    def test_token_endpoint_failure_redirects_to_exchange_failed(self):
        client = _FakeOAuthClient(raises=RuntimeError("400 invalid_grant"))
        response = self._callback(self._state(), client)
        self.assertIn("error=exchange_failed", response.headers["location"])

    def test_response_without_an_access_token_redirects_to_token_missing(self):
        client = _FakeOAuthClient(payload={"id_token": "only-an-id-token"})
        response = self._callback(self._state(), client)
        self.assertIn("error=token_missing", response.headers["location"])
        self.assertFalse(self.store.get_google_oauth_status(IDENTITY_KEY))

    def test_successful_callback_sends_the_configured_client_credentials(self):
        client = _FakeOAuthClient()
        response = self._callback(self._state(), client)

        self.assertIn(response.status_code, (302, 307))
        self.assertIn("google_connected=1", response.headers["location"])
        post = client.posts[0]
        self.assertEqual(post["url"], "https://oauth2.googleapis.com/token")
        self.assertEqual(post["data"]["client_id"], "google-client-id")
        self.assertEqual(post["data"]["grant_type"], "authorization_code")
        self.assertEqual(
            post["data"]["redirect_uri"],
            "http://localhost:5174/api/user-settings/google/callback",
        )
        self.assertTrue(self.store.get_google_oauth_status(IDENTITY_KEY))

    def test_callback_promotes_auth_mode_to_account(self):
        self._callback(self._state(), _FakeOAuthClient())
        prefs, _ = self.store.get_user_state(IDENTITY_KEY)
        self.assertEqual(prefs["provider"], "google")
        self.assertEqual(prefs["auth_mode"], "account")

    def test_callback_replaces_a_model_the_google_provider_does_not_offer(self):
        self.store.save_preferences(
            IDENTITY_KEY,
            {
                "provider": "google",
                "model": "a-model-google-does-not-offer",
                "temperature": 0.5,
                "max_tokens": 128,
                "auth_mode": "api_key",
            },
        )

        self._callback(self._state(), _FakeOAuthClient())

        prefs, _ = self.store.get_user_state(IDENTITY_KEY)
        google_models = self.store.provider_models().get("google", [])
        self.assertTrue(google_models)
        self.assertEqual(prefs["model"], google_models[0])

    def test_callback_keeps_a_model_the_google_provider_does_offer(self):
        google_models = self.store.provider_models().get("google", [])
        self.assertTrue(google_models)
        self.store.save_preferences(
            IDENTITY_KEY,
            {
                "provider": "google",
                "model": google_models[0],
                "temperature": 0.5,
                "max_tokens": 128,
                "auth_mode": "api_key",
            },
        )

        self._callback(self._state(), _FakeOAuthClient())

        prefs, _ = self.store.get_user_state(IDENTITY_KEY)
        self.assertEqual(prefs["model"], google_models[0])


class GoogleDisconnectTests(UserSettingsRouterCoverageTestBase):
    def test_disconnect_clears_tokens_and_downgrades_account_auth(self):
        self.store.set_google_oauth_tokens(
            IDENTITY_KEY, access_token="ya29.token", refresh_token="1//r", expires_in=3600
        )
        self.store.save_preferences(
            IDENTITY_KEY,
            {
                "provider": "google",
                "model": "gemini-1.5-pro",
                "temperature": 0.5,
                "max_tokens": 128,
                "auth_mode": "account",
            },
        )

        response = self.client.post("/api/user-settings/google/disconnect")

        self.assertEqual(response.status_code, 200)
        self.assertFalse(self.store.get_google_oauth_status(IDENTITY_KEY))
        prefs, _ = self.store.get_user_state(IDENTITY_KEY)
        self.assertEqual(prefs["auth_mode"], "api_key")
        providers = {p["provider"]: p for p in response.json()["providers"]}
        self.assertFalse(providers["google"]["account_connected"])

    def test_disconnect_leaves_api_key_preferences_untouched(self):
        self.store.save_preferences(
            IDENTITY_KEY,
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.5,
                "max_tokens": 128,
                "auth_mode": "api_key",
            },
        )

        response = self.client.post("/api/user-settings/google/disconnect")

        self.assertEqual(response.status_code, 200)
        prefs, _ = self.store.get_user_state(IDENTITY_KEY)
        self.assertEqual(prefs["provider"], "openai")
        self.assertEqual(prefs["auth_mode"], "api_key")


class IdentityKeyTests(unittest.TestCase):
    def test_identity_key_is_derived_from_provider_and_user(self):
        self.assertEqual(
            identity_key_for_user({"user": " Alice ", "provider": "GitHub"}),
            "github:alice",
        )


if __name__ == "__main__":
    unittest.main()