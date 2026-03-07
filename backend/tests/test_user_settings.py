import json
import os
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import require_auth
from app.routers import user_settings
from app.services.user_settings_store import UserSettingsStore, identity_key_for_user


class _FakeHTTPResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeAsyncClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        return _FakeHTTPResponse(
            {
                "access_token": "ya29.access-token",
                "refresh_token": "1//refresh-token",
                "expires_in": 3600,
            }
        )


class UserSettingsRouterTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.prev_env = {
            "STUDY_USER_SETTINGS_FILE": os.environ.get("STUDY_USER_SETTINGS_FILE"),
            "STUDY_LLM_SERVICE_USERS_FILE": os.environ.get("STUDY_LLM_SERVICE_USERS_FILE"),
            "STUDY_LLM_ASSIGNMENTS_FILE": os.environ.get("STUDY_LLM_ASSIGNMENTS_FILE"),
            "STUDY_CREDENTIALS_ENCRYPTION_KEY": os.environ.get("STUDY_CREDENTIALS_ENCRYPTION_KEY"),
            "STUDY_GOOGLE_CLIENT_ID": os.environ.get("STUDY_GOOGLE_CLIENT_ID"),
            "STUDY_GOOGLE_CLIENT_SECRET": os.environ.get("STUDY_GOOGLE_CLIENT_SECRET"),
            "STUDY_FRONTEND_URL": os.environ.get("STUDY_FRONTEND_URL"),
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        }
        os.environ["STUDY_USER_SETTINGS_FILE"] = os.path.join(self.tmpdir.name, "user_settings.json")
        os.environ["STUDY_LLM_SERVICE_USERS_FILE"] = os.path.join(self.tmpdir.name, "llm_service_users.json")
        os.environ["STUDY_LLM_ASSIGNMENTS_FILE"] = os.path.join(self.tmpdir.name, "llm_assignments.json")
        os.environ["STUDY_CREDENTIALS_ENCRYPTION_KEY"] = "test-encryption-secret"
        os.environ["STUDY_GOOGLE_CLIENT_ID"] = "test-google-client-id"
        os.environ["STUDY_GOOGLE_CLIENT_SECRET"] = "test-google-client-secret"
        os.environ["STUDY_FRONTEND_URL"] = "http://localhost:5174"
        get_settings.cache_clear()

        self.store = UserSettingsStore()
        user_settings.init(self.store)

        self.app = FastAPI()
        self.app.include_router(user_settings.router)
        self.app.dependency_overrides[require_auth] = lambda: {
            "user": "learner@example.com",
            "provider": "google",
        }
        self.client = TestClient(self.app)

    def tearDown(self):
        self.app.dependency_overrides.clear()
        self.tmpdir.cleanup()
        for key, value in self.prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()

    def test_user_settings_preferences_and_api_key_crud(self):
        get_res = self.client.get("/api/user-settings")
        self.assertEqual(get_res.status_code, 200)
        payload = get_res.json()
        self.assertFalse(payload["has_saved_preferences"])
        self.assertTrue(payload["preferences"]["require_answer_reveal"])

        pref_res = self.client.put(
            "/api/user-settings/preferences",
            json={
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.6,
                "max_tokens": 256,
                "auth_mode": "api_key",
                "require_answer_reveal": False,
            },
        )
        self.assertEqual(pref_res.status_code, 200)
        self.assertFalse(pref_res.json()["require_answer_reveal"])

        save_key_res = self.client.put(
            "/api/user-settings/api-key/openai",
            json={"api_key": "sk-live-user-key"},
        )
        self.assertEqual(save_key_res.status_code, 200)
        providers = {p["provider"]: p for p in save_key_res.json()["providers"]}
        self.assertTrue(providers["openai"]["api_key_connected"])

        identity_key = identity_key_for_user({"user": "learner@example.com", "provider": "google"})
        self.assertEqual(self.store.get_api_key(identity_key, "openai"), "sk-live-user-key")

        with open(os.environ["STUDY_USER_SETTINGS_FILE"]) as f:
            raw = f.read()
        self.assertNotIn("sk-live-user-key", raw)

        delete_key_res = self.client.delete("/api/user-settings/api-key/openai")
        self.assertEqual(delete_key_res.status_code, 200)
        providers2 = {p["provider"]: p for p in delete_key_res.json()["providers"]}
        self.assertFalse(providers2["openai"]["api_key_connected"])

        after = self.client.get("/api/user-settings")
        self.assertEqual(after.status_code, 200)
        self.assertFalse(after.json()["preferences"]["require_answer_reveal"])

    def test_google_callback_stores_tokens(self):
        from unittest.mock import patch

        self.client.cookies.set("oauth_user_settings_state", "state-123")
        self.client.cookies.set(
            "oauth_user_settings_identity",
            "google:learner@example.com",
        )

        with patch("app.routers.user_settings.httpx.AsyncClient", _FakeAsyncClient):
            cb = self.client.get(
                "/api/user-settings/google/callback?code=code-123&state=state-123",
                follow_redirects=False,
            )

        self.assertIn(cb.status_code, (302, 307))
        self.assertIn("/user-settings?google_connected=1", cb.headers.get("location", ""))

        after = self.client.get("/api/user-settings")
        self.assertEqual(after.status_code, 200)
        payload = after.json()
        providers = {p["provider"]: p for p in payload["providers"]}
        self.assertTrue(providers["google"]["account_connected"])
        self.assertEqual(payload["preferences"]["provider"], "google")
        self.assertEqual(payload["preferences"]["auth_mode"], "account")

    def test_using_backend_fallback_flag_false_when_not_approved(self):
        os.environ["OPENAI_API_KEY"] = "sk-backend-openai"

        initial = self.client.get("/api/user-settings")
        self.assertEqual(initial.status_code, 200)
        providers_initial = {p["provider"]: p for p in initial.json()["providers"]}
        self.assertFalse(providers_initial["openai"]["backend_fallback_eligible"])
        self.assertFalse(providers_initial["openai"]["using_backend_fallback"])

        self.store._llm_service_access.add_user("google", "learner@example.com")
        identity_key = identity_key_for_user({"user": "learner@example.com", "provider": "google"})
        self.store._llm_assignments_store.set_assignment(
            identity_key=identity_key,
            provider="openai",
            model="gpt-4o-mini",
        )
        self.store.save_preferences(
            identity_key,
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "max_tokens": 0,
                "auth_mode": "api_key",
                "llm_source": "study_app",
            },
        )

        after = self.client.get("/api/user-settings")
        self.assertEqual(after.status_code, 200)
        self.assertTrue(after.json()["study_app_available"])
        self.assertEqual(after.json()["study_app_assignment"]["provider"], "openai")
        providers_after = {p["provider"]: p for p in after.json()["providers"]}
        self.assertTrue(providers_after["openai"]["backend_fallback_eligible"])
        self.assertTrue(providers_after["openai"]["using_backend_fallback"])


if __name__ == "__main__":
    unittest.main()
