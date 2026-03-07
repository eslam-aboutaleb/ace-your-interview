import asyncio
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.config import get_settings
from app.schemas.models import LLMConfigRequest, LLMProviderEnum
from app.services.llm_client import LLMClient
from app.services.llm_service_access import LLMServiceAccess
from app.services.user_settings_store import UserSettingsStore, identity_key_for_user


def _fake_litellm_response(text: str = "ok"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        model="mock-model",
    )


class LLMClientUserCredentialTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.prev_env = {
            "STUDY_USER_SETTINGS_FILE": os.environ.get("STUDY_USER_SETTINGS_FILE"),
            "STUDY_LLM_SERVICE_USERS_FILE": os.environ.get("STUDY_LLM_SERVICE_USERS_FILE"),
            "STUDY_LLM_ASSIGNMENTS_FILE": os.environ.get("STUDY_LLM_ASSIGNMENTS_FILE"),
            "STUDY_CREDENTIALS_ENCRYPTION_KEY": os.environ.get("STUDY_CREDENTIALS_ENCRYPTION_KEY"),
            "STUDY_GOOGLE_CLIENT_ID": os.environ.get("STUDY_GOOGLE_CLIENT_ID"),
            "STUDY_GOOGLE_CLIENT_SECRET": os.environ.get("STUDY_GOOGLE_CLIENT_SECRET"),
            "STUDY_ADMIN_USERS": os.environ.get("STUDY_ADMIN_USERS"),
        }
        os.environ["STUDY_USER_SETTINGS_FILE"] = os.path.join(self.tmpdir.name, "user_settings.json")
        os.environ["STUDY_LLM_SERVICE_USERS_FILE"] = os.path.join(self.tmpdir.name, "llm_service_users.json")
        os.environ["STUDY_LLM_ASSIGNMENTS_FILE"] = os.path.join(self.tmpdir.name, "llm_assignments.json")
        os.environ["STUDY_CREDENTIALS_ENCRYPTION_KEY"] = "test-encryption-secret"
        os.environ["STUDY_GOOGLE_CLIENT_ID"] = "cid"
        os.environ["STUDY_GOOGLE_CLIENT_SECRET"] = "csecret"
        os.environ["STUDY_ADMIN_USERS"] = ""
        get_settings.cache_clear()

        self.llm_service_access = LLMServiceAccess()
        self.store = UserSettingsStore(llm_service_access=self.llm_service_access)
        self.client = LLMClient(user_settings_store=self.store)
        self.user = {"user": "llm-user@example.com", "provider": "google"}
        self.identity_key = identity_key_for_user(self.user)

    def tearDown(self):
        self.tmpdir.cleanup()
        for key, value in self.prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()

    def test_llm_client_uses_user_api_key_when_present(self):
        self.store.save_preferences(
            self.identity_key,
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "max_tokens": 0,
                "auth_mode": "api_key",
                "llm_source": "personal",
            },
        )
        self.store.set_api_key(self.identity_key, "openai", "sk-user-openai")

        captured: dict = {}

        async def fake_acompletion(**kwargs):
            captured.update(kwargs)
            return _fake_litellm_response("done")

        with patch("app.services.llm_client.litellm.acompletion", side_effect=fake_acompletion):
            result = asyncio.run(
                self.client.completion(
                    "hello",
                    LLMConfigRequest(provider=LLMProviderEnum.OPENAI, model="gpt-4o-mini"),
                    user_identity=self.user,
                )
            )

        self.assertTrue(result["success"])
        self.assertEqual(captured.get("api_key"), "sk-user-openai")
        self.assertEqual(result["metadata"].get("credential_source"), "user_api_key")

    def test_non_approved_user_study_app_mode_is_blocked(self):
        self.store.save_preferences(
            self.identity_key,
            {
                "provider": "anthropic",
                "model": "claude-3-5-sonnet-20241022",
                "temperature": 0.7,
                "max_tokens": 0,
                "auth_mode": "api_key",  # ignored in study_app mode
                "llm_source": "study_app",
            },
        )

        with patch("app.services.llm_client.litellm.acompletion") as mock_completion:
            result = asyncio.run(
                self.client.completion(
                    "hello",
                    LLMConfigRequest(
                        provider=LLMProviderEnum.ANTHROPIC,
                        model="claude-3-5-sonnet-20241022",
                    ),
                    user_identity=self.user,
                )
            )

        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], "llm_service_approval_required")
        self.assertEqual(result["metadata"].get("credential_source"), "blocked_unapproved")
        mock_completion.assert_not_called()

    def test_study_app_mode_requires_assignment(self):
        self.llm_service_access.add_user("google", "llm-user@example.com")
        self.store.save_preferences(
            self.identity_key,
            {
                "provider": "anthropic",
                "model": "claude-3-5-sonnet-20241022",
                "temperature": 0.7,
                "max_tokens": 0,
                "auth_mode": "api_key",
                "llm_source": "study_app",
            },
        )

        with patch("app.services.llm_client.litellm.acompletion") as mock_completion:
            result = asyncio.run(
                self.client.completion(
                    "hello",
                    LLMConfigRequest(
                        provider=LLMProviderEnum.ANTHROPIC,
                        model="claude-3-5-sonnet-20241022",
                    ),
                    user_identity=self.user,
                )
            )

        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], "study_app_llm_not_assigned")
        mock_completion.assert_not_called()

    def test_approved_user_without_personal_credential_uses_backend_fallback(self):
        self.llm_service_access.add_user("google", "llm-user@example.com")
        self.store._llm_assignments_store.set_assignment(
            identity_key=self.identity_key,
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
        )
        self.store.save_preferences(
            self.identity_key,
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "max_tokens": 0,
                "auth_mode": "api_key",
                "llm_source": "study_app",
            },
        )

        captured: dict = {}

        async def fake_acompletion(**kwargs):
            captured.update(kwargs)
            return _fake_litellm_response("done")

        with patch("app.services.llm_client.litellm.acompletion", side_effect=fake_acompletion):
            result = asyncio.run(
                self.client.completion(
                    "hello",
                    LLMConfigRequest(
                        provider=LLMProviderEnum.OPENAI,
                        model="gpt-4o-mini",
                    ),
                    user_identity=self.user,
                )
            )

        self.assertTrue(result["success"])
        self.assertEqual(captured.get("model"), "anthropic/claude-3-5-sonnet-20241022")
        self.assertNotIn("api_key", captured)
        self.assertEqual(result["metadata"].get("credential_source"), "study_app_backend")

    def test_personal_mode_missing_personal_credential_is_blocked(self):
        self.store.save_preferences(
            self.identity_key,
            {
                "provider": "anthropic",
                "model": "claude-3-5-sonnet-20241022",
                "temperature": 0.7,
                "max_tokens": 0,
                "auth_mode": "api_key",
                "llm_source": "personal",
            },
        )

        with patch("app.services.llm_client.litellm.acompletion") as mock_completion:
            result = asyncio.run(
                self.client.completion(
                    "hello",
                    LLMConfigRequest(
                        provider=LLMProviderEnum.ANTHROPIC,
                        model="claude-3-5-sonnet-20241022",
                    ),
                    user_identity=self.user,
                )
            )
        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], "personal_credential_required")
        self.assertEqual(result["metadata"].get("credential_source"), "missing_personal")
        mock_completion.assert_not_called()

    def test_non_approved_user_with_personal_api_key_is_allowed(self):
        self.store.save_preferences(
            self.identity_key,
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "max_tokens": 0,
                "auth_mode": "api_key",
                "llm_source": "personal",
            },
        )
        self.store.set_api_key(self.identity_key, "openai", "sk-user-openai")

        async def fake_acompletion(**kwargs):
            return _fake_litellm_response("done")

        with patch("app.services.llm_client.litellm.acompletion", side_effect=fake_acompletion):
            result = asyncio.run(
                self.client.completion(
                    "hello",
                    LLMConfigRequest(provider=LLMProviderEnum.OPENAI, model="gpt-4o-mini"),
                    user_identity=self.user,
                )
            )
        self.assertTrue(result["success"])
        self.assertEqual(result["metadata"].get("credential_source"), "user_api_key")

    def test_google_account_mode_uses_oauth_path(self):
        self.store.save_preferences(
            self.identity_key,
            {
                "provider": "google",
                "model": "gemini-1.5-pro",
                "temperature": 0.7,
                "max_tokens": 0,
                "auth_mode": "account",
                "llm_source": "personal",
            },
        )
        self.store.set_google_oauth_tokens(
            self.identity_key,
            access_token="google-access-token",
            refresh_token="google-refresh-token",
            expires_in=3600,
        )

        captured: dict = {}

        async def fake_acompletion(**kwargs):
            captured.update(kwargs)
            return _fake_litellm_response("done")

        with patch("app.services.llm_client.litellm.acompletion", side_effect=fake_acompletion):
            result = asyncio.run(
                self.client.completion(
                    "hello",
                    LLMConfigRequest(provider=LLMProviderEnum.GOOGLE, model="gemini-1.5-pro"),
                    user_identity=self.user,
                )
            )

        self.assertTrue(result["success"])
        self.assertEqual(
            captured.get("api_key"),
            {"Authorization": "Bearer google-access-token"},
        )
        self.assertEqual(result["metadata"].get("credential_source"), "user_account")


if __name__ == "__main__":
    unittest.main()
