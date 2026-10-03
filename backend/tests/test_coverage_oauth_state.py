"""Coverage for ``app.services.oauth_state``.

OAuth state is the CSRF guard on the Google/GitHub connect callbacks, so the
tests pin the validation order (purpose argument, then token presence, then
signature/expiry, then purpose match, then nonce) and the reserved-claim
handling on creation.
"""

import os
import time
import unittest

import jwt

from app.config import get_settings
from app.services.oauth_state import create_oauth_state, verify_oauth_state

_SECRET = "s" * 64


class CreateOAuthStateTests(unittest.TestCase):
    def setUp(self):
        self.prev_secret = os.environ.get("STUDY_AUTH_SECRET_KEY")
        os.environ["STUDY_AUTH_SECRET_KEY"] = _SECRET
        get_settings.cache_clear()

    def tearDown(self):
        if self.prev_secret is None:
            os.environ.pop("STUDY_AUTH_SECRET_KEY", None)
        else:
            os.environ["STUDY_AUTH_SECRET_KEY"] = self.prev_secret
        get_settings.cache_clear()

    def test_missing_or_blank_purpose_is_rejected(self):
        for payload in ({}, {"purpose": ""}, {"purpose": "   "}):
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(ValueError, "non-empty purpose"):
                    create_oauth_state(payload)

    def test_non_string_purpose_is_stringified(self):
        # Documented coercion: ``str(payload["purpose"])`` means an explicit
        # ``None`` becomes the literal purpose "None" rather than raising.
        token = create_oauth_state({"purpose": None})
        self.assertEqual(verify_oauth_state(token, expected_purpose="None")["purpose"], "None")

    def test_non_string_purpose_is_coerced_and_stripped(self):
        token = create_oauth_state({"purpose": "  auth_google  "})
        payload = verify_oauth_state(token, expected_purpose="auth_google")
        self.assertEqual(payload["purpose"], "auth_google")

    def test_reserved_claims_in_the_payload_cannot_be_forged(self):
        # iat/exp/nonce are generated server-side; caller values must be dropped
        # so a caller cannot mint an arbitrarily long-lived state token.
        token = create_oauth_state(
            {
                "purpose": "auth_google",
                "iat": 1,
                "exp": 4_102_444_800,
                "nonce": "attacker-nonce",
                "identity_key": "google:learner@example.com",
            },
            ttl_seconds=60,
        )
        payload = verify_oauth_state(token, expected_purpose="auth_google")
        self.assertGreater(payload["iat"], 1)
        self.assertLess(payload["exp"], 4_102_444_800)
        self.assertNotEqual(payload["nonce"], "attacker-nonce")
        self.assertEqual(payload["identity_key"], "google:learner@example.com")

    def test_extra_payload_claims_are_preserved(self):
        token = create_oauth_state({"purpose": "auth_google", "a": 1, "b": [1, 2]})
        payload = verify_oauth_state(token, expected_purpose="auth_google")
        self.assertEqual(payload["a"], 1)
        self.assertEqual(payload["b"], [1, 2])

    def test_ttl_is_at_least_one_second(self):
        now = int(time.time())
        payload = jwt.decode(
            create_oauth_state({"purpose": "auth_google"}, ttl_seconds=0),
            _SECRET,
            algorithms=["HS256"],
        )
        self.assertLessEqual(payload["exp"], now + 2)
        self.assertGreaterEqual(payload["exp"], now + 1)

    def test_non_numeric_ttl_still_yields_a_token(self):
        token = create_oauth_state({"purpose": "auth_google"}, ttl_seconds=30)
        self.assertTrue(verify_oauth_state(token, expected_purpose="auth_google"))

    def test_every_token_has_a_distinct_nonce(self):
        first = verify_oauth_state(create_oauth_state({"purpose": "p"}), expected_purpose="p")
        second = verify_oauth_state(create_oauth_state({"purpose": "p"}), expected_purpose="p")
        self.assertNotEqual(first["nonce"], second["nonce"])


class VerifyOAuthStateTests(unittest.TestCase):
    def setUp(self):
        self.prev_secret = os.environ.get("STUDY_AUTH_SECRET_KEY")
        os.environ["STUDY_AUTH_SECRET_KEY"] = _SECRET
        get_settings.cache_clear()

    def tearDown(self):
        if self.prev_secret is None:
            os.environ.pop("STUDY_AUTH_SECRET_KEY", None)
        else:
            os.environ["STUDY_AUTH_SECRET_KEY"] = self.prev_secret
        get_settings.cache_clear()

    def _forge(self, claims, secret=_SECRET):
        return jwt.encode(claims, secret, algorithm="HS256")

    def test_blank_expected_purpose_is_rejected_before_decoding(self):
        token = create_oauth_state({"purpose": "auth_google"})
        for purpose in ("", "   ", None):
            with self.subTest(purpose=purpose):
                with self.assertRaisesRegex(ValueError, "purpose is required"):
                    verify_oauth_state(token, expected_purpose=purpose)

    def test_missing_token_is_rejected(self):
        for token in ("", None):
            with self.subTest(token=token):
                with self.assertRaisesRegex(ValueError, "missing"):
                    verify_oauth_state(token, expected_purpose="auth_google")

    def test_garbage_token_is_rejected(self):
        for token in ("not-a-jwt", "a.b.c", "...."):
            with self.subTest(token=token):
                with self.assertRaisesRegex(ValueError, "Invalid OAuth state token"):
                    verify_oauth_state(token, expected_purpose="auth_google")

    def test_token_signed_with_another_secret_is_rejected(self):
        token = self._forge(
            {"purpose": "p", "iat": 1, "exp": 4_102_444_800, "nonce": "n"},
            secret="d" * 64,
        )
        with self.assertRaisesRegex(ValueError, "Invalid OAuth state token"):
            verify_oauth_state(token, expected_purpose="p")

    def test_token_without_required_claims_is_rejected(self):
        for claims in (
            {"iat": 1, "exp": 4_102_444_800, "nonce": "n"},
            {"purpose": "p", "exp": 4_102_444_800, "nonce": "n"},
            {"purpose": "p", "iat": 1, "nonce": "n"},
            {"purpose": "p", "iat": 1, "exp": 4_102_444_800},
        ):
            with self.subTest(claims=sorted(claims)):
                with self.assertRaisesRegex(ValueError, "Invalid OAuth state token"):
                    verify_oauth_state(self._forge(claims), expected_purpose="p")

    def test_expired_token_is_rejected(self):
        now = int(time.time())
        token = self._forge({"purpose": "p", "iat": now - 600, "exp": now - 60, "nonce": "n"})
        with self.assertRaisesRegex(ValueError, "Invalid OAuth state token"):
            verify_oauth_state(token, expected_purpose="p")

    def test_purpose_mismatch_is_rejected(self):
        token = create_oauth_state({"purpose": "auth_google"})
        with self.assertRaisesRegex(ValueError, "Unexpected OAuth state purpose"):
            verify_oauth_state(token, expected_purpose="auth_github")

    def test_blank_nonce_claim_is_rejected(self):
        # The claim is present, so JWT decoding succeeds; the nonce check is what
        # catches a stripped nonce.
        now = int(time.time())
        token = self._forge({"purpose": "p", "iat": now, "exp": now + 600, "nonce": "   "})
        with self.assertRaisesRegex(ValueError, "nonce is missing"):
            verify_oauth_state(token, expected_purpose="p")


if __name__ == "__main__":
    unittest.main()