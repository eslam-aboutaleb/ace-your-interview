import os
import unittest

from app.config import get_settings, resolve_auth_secret_key
from app.services.auth import create_jwt_token, decode_jwt_token


class AuthSecretValidationTests(unittest.TestCase):
    def setUp(self):
        self.prev_secret = os.environ.get("STUDY_AUTH_SECRET_KEY")
        get_settings.cache_clear()

    def tearDown(self):
        if self.prev_secret is None:
            os.environ.pop("STUDY_AUTH_SECRET_KEY", None)
        else:
            os.environ["STUDY_AUTH_SECRET_KEY"] = self.prev_secret
        get_settings.cache_clear()

    def test_missing_auth_secret_is_rejected(self):
        os.environ["STUDY_AUTH_SECRET_KEY"] = ""
        get_settings.cache_clear()
        with self.assertRaises(RuntimeError):
            resolve_auth_secret_key(get_settings())

    def test_short_auth_secret_is_rejected(self):
        os.environ["STUDY_AUTH_SECRET_KEY"] = "short-secret"
        get_settings.cache_clear()
        with self.assertRaises(RuntimeError):
            resolve_auth_secret_key(get_settings())

    def test_jwt_helpers_require_valid_secret(self):
        os.environ["STUDY_AUTH_SECRET_KEY"] = "b" * 64
        get_settings.cache_clear()
        token = create_jwt_token("learner@example.com", "google")
        payload = decode_jwt_token(token)
        self.assertEqual(payload["sub"], "learner@example.com")
        self.assertEqual(payload["provider"], "google")


if __name__ == "__main__":
    unittest.main()
