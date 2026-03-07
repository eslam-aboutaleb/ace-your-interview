import json
import os
import tempfile
import unittest
from unittest.mock import patch

from app.config import get_settings
from app.services.user_settings_store import UserSettingsStore


class UserSettingsStoreCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.user_settings_file = os.path.join(self.tmpdir.name, "user_settings.json")
        self.prev_env = {
            "STUDY_USER_SETTINGS_FILE": os.environ.get("STUDY_USER_SETTINGS_FILE"),
            "STUDY_CREDENTIALS_ENCRYPTION_KEY": os.environ.get("STUDY_CREDENTIALS_ENCRYPTION_KEY"),
        }
        os.environ["STUDY_USER_SETTINGS_FILE"] = self.user_settings_file
        os.environ["STUDY_CREDENTIALS_ENCRYPTION_KEY"] = "cache-test-encryption-secret"
        get_settings.cache_clear()

    def tearDown(self):
        self.tmpdir.cleanup()
        for key, value in self.prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()

    def test_store_reuses_cached_data_until_file_changes(self):
        store = UserSettingsStore()
        identity_key = "google:learner@example.com"
        store.set_api_key(identity_key, provider="openai", api_key="sk-user")

        # Simulate a fresh process cache state to verify first read hits disk once.
        store._cache_data = None
        store._cache_mtime_ns = None
        store._cache_size = None

        with patch("app.services.user_settings_store.json.load", wraps=json.load) as mocked_load:
            self.assertEqual(store.get_api_key(identity_key, "openai"), "sk-user")
            self.assertEqual(store.get_api_key(identity_key, "openai"), "sk-user")
            self.assertEqual(mocked_load.call_count, 1)

            with open(self.user_settings_file, "r", encoding="utf-8") as f:
                data = json.loads(f.read())
            data.setdefault("users", {})["google:second@example.com"] = {
                "preferences": store.default_preferences(),
                "credentials": {},
                "google_oauth": {},
            }
            with open(self.user_settings_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)

            prefs, has_saved = store.get_user_state("google:second@example.com")
            self.assertTrue(has_saved)
            self.assertEqual(prefs["provider"], store.default_preferences()["provider"])
            self.assertEqual(mocked_load.call_count, 2)


if __name__ == "__main__":
    unittest.main()
