import os
import tempfile
import unittest

from app.config import get_settings
from app.services.user_settings_store import UserSettingsStore


class UserSettingsStoreEncryptionTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.prev_env = {
            "STUDY_ENVIRONMENT": os.environ.get("STUDY_ENVIRONMENT"),
            "STUDY_USER_SETTINGS_FILE": os.environ.get("STUDY_USER_SETTINGS_FILE"),
            "STUDY_CREDENTIALS_ENCRYPTION_KEY": os.environ.get("STUDY_CREDENTIALS_ENCRYPTION_KEY"),
            "STUDY_ALLOW_INSECURE_DEV_ENCRYPTION_FALLBACK": os.environ.get(
                "STUDY_ALLOW_INSECURE_DEV_ENCRYPTION_FALLBACK"
            ),
            "STUDY_INSECURE_DEV_CREDENTIALS_ENCRYPTION_SECRET": os.environ.get(
                "STUDY_INSECURE_DEV_CREDENTIALS_ENCRYPTION_SECRET"
            ),
        }
        os.environ["STUDY_ENVIRONMENT"] = "development"
        os.environ["STUDY_USER_SETTINGS_FILE"] = os.path.join(self.tmpdir.name, "user_settings.json")
        os.environ.pop("STUDY_CREDENTIALS_ENCRYPTION_KEY", None)
        os.environ.pop("STUDY_ALLOW_INSECURE_DEV_ENCRYPTION_FALLBACK", None)
        os.environ.pop("STUDY_INSECURE_DEV_CREDENTIALS_ENCRYPTION_SECRET", None)
        get_settings.cache_clear()

    def tearDown(self):
        self.tmpdir.cleanup()
        for key, value in self.prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()

    def test_missing_encryption_key_is_rejected_by_default(self):
        with self.assertRaises(RuntimeError):
            UserSettingsStore()

    def test_explicit_dev_fallback_requires_secret_value(self):
        os.environ["STUDY_ALLOW_INSECURE_DEV_ENCRYPTION_FALLBACK"] = "true"
        get_settings.cache_clear()
        with self.assertRaises(RuntimeError):
            UserSettingsStore()

    def test_explicit_dev_fallback_with_secret_is_allowed(self):
        os.environ["STUDY_ALLOW_INSECURE_DEV_ENCRYPTION_FALLBACK"] = "true"
        os.environ["STUDY_INSECURE_DEV_CREDENTIALS_ENCRYPTION_SECRET"] = "local-admin-approved-secret"
        get_settings.cache_clear()

        store = UserSettingsStore()
        store.set_api_key("google:learner@example.com", provider="openai", api_key="sk-local-user")
        self.assertEqual(
            store.get_api_key("google:learner@example.com", "openai"),
            "sk-local-user",
        )


if __name__ == "__main__":
    unittest.main()
