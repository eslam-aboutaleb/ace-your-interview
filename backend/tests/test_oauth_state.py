import os
import unittest

from app.config import get_settings
from app.services.oauth_state import create_oauth_state, verify_oauth_state


class OAuthStateTests(unittest.TestCase):
    def setUp(self):
        self.prev_secret = os.environ.get("STUDY_AUTH_SECRET_KEY")
        os.environ["STUDY_AUTH_SECRET_KEY"] = "c" * 64
        get_settings.cache_clear()

    def tearDown(self):
        if self.prev_secret is None:
            os.environ.pop("STUDY_AUTH_SECRET_KEY", None)
        else:
            os.environ["STUDY_AUTH_SECRET_KEY"] = self.prev_secret
        get_settings.cache_clear()

    def test_create_and_verify_oauth_state_round_trip(self):
        token = create_oauth_state(
            {"purpose": "auth_google", "identity_key": "google:learner@example.com"}
        )
        payload = verify_oauth_state(token, expected_purpose="auth_google")
        self.assertEqual(payload["purpose"], "auth_google")
        self.assertEqual(payload["identity_key"], "google:learner@example.com")
        self.assertTrue(payload["nonce"])

    def test_verify_rejects_wrong_purpose(self):
        token = create_oauth_state({"purpose": "auth_github"})
        with self.assertRaises(ValueError):
            verify_oauth_state(token, expected_purpose="auth_google")


if __name__ == "__main__":
    unittest.main()
